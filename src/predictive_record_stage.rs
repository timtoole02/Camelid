//! Bounded, source-agnostic predictive record staging.
//!
//! A stage owns an ordered, de-duplicated prefix of candidate `(layer, expert)`
//! keys. One coordinator reads those candidates serially in the background.
//! Authoritative demand uses [`PredictiveRecordStage::take_or_demand`], whose
//! state transition is deliberately exact:
//!
//! - `Queued -> DemandOwned`: demand wins the race and performs the only read;
//! - `Reading`: demand waits for that read instead of starting a duplicate;
//! - `Ready -> Taken`: demand takes the fully published staged bytes; and
//! - absent or previously failed keys are read synchronously by demand.
//!
//! The module has no model, Metal, Ghost, or file-format dependency. Callers
//! supply an immutable, thread-safe loader closure. A process-global permit
//! bounds live predictive stages to one, including the interval after a stage
//! owner is dropped while its final blocking read is still returning.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Exact identity of one independently loadable expert record.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredictiveRecordKey {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
}

impl PredictiveRecordKey {
    pub(crate) const fn new(layer: usize, expert: usize) -> Self {
        Self { layer, expert }
    }
}

/// Error text is owned because a failure can outlive the loader invocation.
pub(crate) type PredictiveRecordLoadResult = Result<Box<[u8]>, String>;

/// The source adapter used by both predictive reads and authoritative demand.
pub(crate) type PredictiveRecordLoader =
    Arc<dyn Fn(PredictiveRecordKey) -> PredictiveRecordLoadResult + Send + Sync + 'static>;

/// Observable state without exposing a staged byte buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PredictiveRecordStatus {
    Queued,
    Reading,
    Ready,
    Failed,
    DemandOwned,
    Taken,
}

#[derive(Debug)]
enum PredictiveRecordState {
    Queued,
    Reading,
    Ready(Box<[u8]>),
    Failed(String),
    DemandOwned,
    Taken,
}

impl PredictiveRecordState {
    fn status(&self) -> PredictiveRecordStatus {
        match self {
            Self::Queued => PredictiveRecordStatus::Queued,
            Self::Reading => PredictiveRecordStatus::Reading,
            Self::Ready(_) => PredictiveRecordStatus::Ready,
            Self::Failed(_) => PredictiveRecordStatus::Failed,
            Self::DemandOwned => PredictiveRecordStatus::DemandOwned,
            Self::Taken => PredictiveRecordStatus::Taken,
        }
    }
}

struct PredictiveRecordEntry {
    state: Mutex<PredictiveRecordState>,
    changed: Condvar,
}

impl PredictiveRecordEntry {
    fn queued() -> Self {
        Self {
            state: Mutex::new(PredictiveRecordState::Queued),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, PredictiveRecordState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait<'a>(
        &self,
        state: MutexGuard<'a, PredictiveRecordState>,
    ) -> MutexGuard<'a, PredictiveRecordState> {
        self.changed
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Whether returned bytes came from the coordinator or the demand path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PredictiveRecordSource {
    Staged,
    Demand,
}

/// One exact record, returned with ownership so no staged slice can escape.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PredictiveRecord {
    pub(crate) key: PredictiveRecordKey,
    pub(crate) source: PredictiveRecordSource,
    pub(crate) bytes: Box<[u8]>,
}

/// Result of atomically reconciling one predictive entry with the exact
/// consumer. `Demand` means the caller owns the authoritative fallback read;
/// the coordinator will not issue a duplicate read for that key.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PredictiveRecordClaim {
    Staged(PredictiveRecord),
    Demand,
}

/// Failure to start a bounded stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PredictiveStageStartError {
    ZeroCapacity,
    NoCandidates,
    StageAlreadyActive,
    CoordinatorSpawn(String),
}

impl fmt::Display for PredictiveStageStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "predictive record stage capacity must be non-zero"),
            Self::NoCandidates => write!(f, "predictive record stage has no candidates"),
            Self::StageAlreadyActive => {
                write!(f, "another predictive record stage is still active")
            }
            Self::CoordinatorSpawn(message) => {
                write!(
                    f,
                    "failed to spawn predictive record coordinator: {message}"
                )
            }
        }
    }
}

impl std::error::Error for PredictiveStageStartError {}

/// Failure of an exact [`PredictiveRecordStage::take_or_demand`] operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PredictiveRecordTakeError {
    Cancelled,
    AlreadyTaken(PredictiveRecordKey),
    LoadFailed {
        key: PredictiveRecordKey,
        message: String,
    },
}

impl fmt::Display for PredictiveRecordTakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "predictive record stage was cancelled"),
            Self::AlreadyTaken(key) => write!(
                f,
                "predictive record ({}, {}) was already taken",
                key.layer, key.expert
            ),
            Self::LoadFailed { key, message } => write!(
                f,
                "predictive record ({}, {}) load failed: {message}",
                key.layer, key.expert
            ),
        }
    }
}

impl std::error::Error for PredictiveRecordTakeError {}

/// A cheap diagnostic snapshot. Counts always sum to `total`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PredictiveStageSnapshot {
    pub(crate) total: usize,
    pub(crate) queued: usize,
    pub(crate) reading: usize,
    pub(crate) ready: usize,
    pub(crate) failed: usize,
    pub(crate) demand_owned: usize,
    pub(crate) taken: usize,
    pub(crate) cancelled: bool,
    pub(crate) coordinator_done: bool,
}

static PREDICTIVE_STAGE_ACTIVE: AtomicBool = AtomicBool::new(false);

struct GlobalStagePermit;

impl GlobalStagePermit {
    fn try_acquire() -> Option<Self> {
        PREDICTIVE_STAGE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for GlobalStagePermit {
    fn drop(&mut self) {
        PREDICTIVE_STAGE_ACTIVE.store(false, Ordering::Release);
    }
}

struct PredictiveStageInner {
    ordered_keys: Box<[PredictiveRecordKey]>,
    entries: BTreeMap<PredictiveRecordKey, Arc<PredictiveRecordEntry>>,
    loader: PredictiveRecordLoader,
    cancelled: AtomicBool,
    coordinator_done: AtomicBool,
    // This lives in the shared inner, rather than the public owner or worker
    // alone. Therefore the permit survives either side until both are gone.
    _permit: GlobalStagePermit,
}

/// Unique owner of one bounded predictive stage.
///
/// The type intentionally does not implement `Clone`. Dropping it cancels
/// queued work. The coordinator retains only the shared inner long enough for
/// a currently blocking loader call to return and then releases the global
/// permit.
pub(crate) struct PredictiveRecordStage {
    inner: Arc<PredictiveStageInner>,
}

impl PredictiveRecordStage {
    /// Start one serial background coordinator.
    ///
    /// Candidate order is preserved, duplicates keep their first position,
    /// and the unique sequence is truncated at `max_records`. This makes the
    /// caller-supplied cap a hard upper bound even if an upstream planner is
    /// malformed or accidentally duplicates candidates.
    pub(crate) fn start<I>(
        candidates: I,
        max_records: usize,
        loader: PredictiveRecordLoader,
    ) -> Result<Self, PredictiveStageStartError>
    where
        I: IntoIterator<Item = PredictiveRecordKey>,
    {
        if max_records == 0 {
            return Err(PredictiveStageStartError::ZeroCapacity);
        }

        let mut seen = BTreeSet::new();
        let mut ordered_keys = Vec::with_capacity(max_records);
        for key in candidates {
            if seen.insert(key) {
                ordered_keys.push(key);
                if ordered_keys.len() == max_records {
                    break;
                }
            }
        }
        if ordered_keys.is_empty() {
            return Err(PredictiveStageStartError::NoCandidates);
        }

        let permit = GlobalStagePermit::try_acquire()
            .ok_or(PredictiveStageStartError::StageAlreadyActive)?;
        let entries = ordered_keys
            .iter()
            .copied()
            .map(|key| (key, Arc::new(PredictiveRecordEntry::queued())))
            .collect();
        let inner = Arc::new(PredictiveStageInner {
            ordered_keys: ordered_keys.into_boxed_slice(),
            entries,
            loader,
            cancelled: AtomicBool::new(false),
            coordinator_done: AtomicBool::new(false),
            _permit: permit,
        });
        let coordinator_inner = Arc::clone(&inner);
        std::thread::Builder::new()
            .name("camelid-predictive-record-stage".into())
            .spawn(move || run_coordinator(coordinator_inner))
            .map_err(|error| PredictiveStageStartError::CoordinatorSpawn(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Ordered, unique candidates actually admitted under the hard cap.
    pub(crate) fn candidate_keys(&self) -> &[PredictiveRecordKey] {
        &self.inner.ordered_keys
    }

    /// Return the current state of an admitted candidate.
    pub(crate) fn status(&self, key: PredictiveRecordKey) -> Option<PredictiveRecordStatus> {
        self.inner
            .entries
            .get(&key)
            .map(|entry| entry.lock().status())
    }

    /// Last loader failure for an admitted key, if it is still failed.
    pub(crate) fn failure_message(&self, key: PredictiveRecordKey) -> Option<String> {
        let entry = self.inner.entries.get(&key)?;
        let state = entry.lock();
        match &*state {
            PredictiveRecordState::Failed(message) => Some(message.clone()),
            _ => None,
        }
    }

    /// Take a staged record or perform its exact authoritative demand load.
    ///
    /// A key outside the predictive set goes straight to demand and does not
    /// enlarge the stage. A queued key is atomically claimed by demand. A key
    /// already being read by the coordinator waits on that entry's condition
    /// variable and consumes the published result, so the race never issues a
    /// duplicate load. Background failures are retried once by this demand
    /// call because predictive failure cannot replace authoritative I/O.
    pub(crate) fn take_or_demand(
        &self,
        key: PredictiveRecordKey,
    ) -> Result<PredictiveRecord, PredictiveRecordTakeError> {
        if self.inner.cancelled.load(Ordering::Acquire) {
            return Err(PredictiveRecordTakeError::Cancelled);
        }
        let Some(entry) = self.inner.entries.get(&key) else {
            return load_for_demand(&self.inner.loader, key);
        };

        let mut state = entry.lock();
        loop {
            if self.inner.cancelled.load(Ordering::Acquire) {
                return Err(PredictiveRecordTakeError::Cancelled);
            }
            match &*state {
                PredictiveRecordState::Queued | PredictiveRecordState::Failed(_) => {
                    *state = PredictiveRecordState::DemandOwned;
                    entry.changed.notify_all();
                    drop(state);

                    let loaded = invoke_loader(&self.inner.loader, key);
                    let mut completed_state = entry.lock();
                    match loaded {
                        Ok(bytes) => {
                            *completed_state = PredictiveRecordState::Taken;
                            entry.changed.notify_all();
                            return Ok(PredictiveRecord {
                                key,
                                source: PredictiveRecordSource::Demand,
                                bytes,
                            });
                        }
                        Err(message) => {
                            *completed_state = PredictiveRecordState::Failed(message.clone());
                            entry.changed.notify_all();
                            return Err(PredictiveRecordTakeError::LoadFailed { key, message });
                        }
                    }
                }
                PredictiveRecordState::Reading | PredictiveRecordState::DemandOwned => {
                    state = entry.wait(state);
                }
                PredictiveRecordState::Ready(_) => {
                    let ready = std::mem::replace(&mut *state, PredictiveRecordState::Taken);
                    entry.changed.notify_all();
                    let PredictiveRecordState::Ready(bytes) = ready else {
                        unreachable!("state was matched as Ready")
                    };
                    return Ok(PredictiveRecord {
                        key,
                        source: PredictiveRecordSource::Staged,
                        bytes,
                    });
                }
                PredictiveRecordState::Taken => {
                    return Err(PredictiveRecordTakeError::AlreadyTaken(key));
                }
            }
        }
    }

    /// Take a completed predictive record, share a predictive read that is
    /// already in flight, or atomically hand an unstarted/failed key to the
    /// caller's authoritative batch reader.
    ///
    /// Unlike [`Self::take_or_demand`], this method never invokes the loader on
    /// the caller thread. That lets a model runtime retain its established
    /// batched demand path and direct-to-destination copy behavior.
    pub(crate) fn claim_ready_or_demand(
        &self,
        key: PredictiveRecordKey,
    ) -> Result<PredictiveRecordClaim, PredictiveRecordTakeError> {
        if self.inner.cancelled.load(Ordering::Acquire) {
            return Err(PredictiveRecordTakeError::Cancelled);
        }
        let Some(entry) = self.inner.entries.get(&key) else {
            return Ok(PredictiveRecordClaim::Demand);
        };

        let mut state = entry.lock();
        loop {
            if self.inner.cancelled.load(Ordering::Acquire) {
                return Err(PredictiveRecordTakeError::Cancelled);
            }
            match &*state {
                PredictiveRecordState::Queued | PredictiveRecordState::Failed(_) => {
                    *state = PredictiveRecordState::DemandOwned;
                    entry.changed.notify_all();
                    return Ok(PredictiveRecordClaim::Demand);
                }
                PredictiveRecordState::Reading => {
                    state = entry.wait(state);
                }
                PredictiveRecordState::Ready(_) => {
                    let ready = std::mem::replace(&mut *state, PredictiveRecordState::Taken);
                    entry.changed.notify_all();
                    let PredictiveRecordState::Ready(bytes) = ready else {
                        unreachable!("state was matched as Ready")
                    };
                    return Ok(PredictiveRecordClaim::Staged(PredictiveRecord {
                        key,
                        source: PredictiveRecordSource::Staged,
                        bytes,
                    }));
                }
                PredictiveRecordState::DemandOwned => {
                    state = entry.wait(state);
                }
                PredictiveRecordState::Taken => {
                    return Err(PredictiveRecordTakeError::AlreadyTaken(key));
                }
            }
        }
    }

    /// Publish completion of a demand claim made by
    /// [`Self::claim_ready_or_demand`]. Keys outside the predictive set need no
    /// publication. A failed batch can be retried by a later exact consumer.
    pub(crate) fn finish_demand(&self, key: PredictiveRecordKey, succeeded: bool) {
        let Some(entry) = self.inner.entries.get(&key) else {
            return;
        };
        let mut state = entry.lock();
        if matches!(*state, PredictiveRecordState::DemandOwned) {
            *state = if succeeded {
                PredictiveRecordState::Taken
            } else {
                PredictiveRecordState::Failed("authoritative demand failed".into())
            };
            entry.changed.notify_all();
        }
    }

    /// Stop predictive work and discard any ready-but-untaken buffers.
    ///
    /// A `Reading` entry is left in that state until its loader call returns;
    /// changing it early would permit an exact demand to start a duplicate
    /// read. An authoritative `DemandOwned` operation is likewise allowed to
    /// finish. All waiters are notified immediately and observe cancellation.
    pub(crate) fn cancel(&self) {
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }
        for entry in self.inner.entries.values() {
            let mut state = entry.lock();
            if matches!(
                &*state,
                PredictiveRecordState::Queued | PredictiveRecordState::Ready(_)
            ) {
                *state = PredictiveRecordState::Failed("stage cancelled".into());
            }
            entry.changed.notify_all();
        }
    }

    pub(crate) fn snapshot(&self) -> PredictiveStageSnapshot {
        let mut snapshot = PredictiveStageSnapshot {
            total: self.inner.entries.len(),
            cancelled: self.inner.cancelled.load(Ordering::Acquire),
            coordinator_done: self.inner.coordinator_done.load(Ordering::Acquire),
            ..PredictiveStageSnapshot::default()
        };
        for entry in self.inner.entries.values() {
            match entry.lock().status() {
                PredictiveRecordStatus::Queued => snapshot.queued += 1,
                PredictiveRecordStatus::Reading => snapshot.reading += 1,
                PredictiveRecordStatus::Ready => snapshot.ready += 1,
                PredictiveRecordStatus::Failed => snapshot.failed += 1,
                PredictiveRecordStatus::DemandOwned => snapshot.demand_owned += 1,
                PredictiveRecordStatus::Taken => snapshot.taken += 1,
            }
        }
        snapshot
    }
}

impl Drop for PredictiveRecordStage {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn run_coordinator(inner: Arc<PredictiveStageInner>) {
    for key in inner.ordered_keys.iter().copied() {
        if inner.cancelled.load(Ordering::Acquire) {
            break;
        }
        let entry = Arc::clone(
            inner
                .entries
                .get(&key)
                .expect("ordered predictive key must have an entry"),
        );
        {
            let mut state = entry.lock();
            if !matches!(*state, PredictiveRecordState::Queued) {
                continue;
            }
            if inner.cancelled.load(Ordering::Acquire) {
                *state = PredictiveRecordState::Failed("stage cancelled".into());
                entry.changed.notify_all();
                break;
            }
            *state = PredictiveRecordState::Reading;
            entry.changed.notify_all();
        }

        let loaded = invoke_loader(&inner.loader, key);
        let mut state = entry.lock();
        if matches!(*state, PredictiveRecordState::Reading) {
            *state = if inner.cancelled.load(Ordering::Acquire) {
                PredictiveRecordState::Failed("stage cancelled".into())
            } else {
                match loaded {
                    Ok(bytes) => PredictiveRecordState::Ready(bytes),
                    Err(message) => PredictiveRecordState::Failed(message),
                }
            };
        }
        entry.changed.notify_all();
    }

    // Publish terminal cancellation for candidates the coordinator never
    // reached. This is diagnostic state only; take_or_demand already checks
    // the stage-wide cancellation flag before consulting an entry.
    if inner.cancelled.load(Ordering::Acquire) {
        for entry in inner.entries.values() {
            let mut state = entry.lock();
            if matches!(*state, PredictiveRecordState::Queued) {
                *state = PredictiveRecordState::Failed("stage cancelled".into());
            }
            entry.changed.notify_all();
        }
    }
    inner.coordinator_done.store(true, Ordering::Release);
    for entry in inner.entries.values() {
        entry.changed.notify_all();
    }
}

fn load_for_demand(
    loader: &PredictiveRecordLoader,
    key: PredictiveRecordKey,
) -> Result<PredictiveRecord, PredictiveRecordTakeError> {
    invoke_loader(loader, key)
        .map(|bytes| PredictiveRecord {
            key,
            source: PredictiveRecordSource::Demand,
            bytes,
        })
        .map_err(|message| PredictiveRecordTakeError::LoadFailed { key, message })
}

fn invoke_loader(
    loader: &PredictiveRecordLoader,
    key: PredictiveRecordKey,
) -> PredictiveRecordLoadResult {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader(key))).unwrap_or_else(|panic| {
        Err(format!(
            "loader panicked: {}",
            panic_payload_message(panic.as_ref())
        ))
    })
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    // The production permit is intentionally process-global, so these tests
    // must not contend with one another when the Rust test harness is parallel.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    struct TestGuard {
        _serial: MutexGuard<'static, ()>,
    }

    impl Drop for TestGuard {
        fn drop(&mut self) {
            // The public owner drops before this guard because it is declared
            // later in each test. Wait for any just-cancelled coordinator to
            // release the process-global permit before another test begins.
            let deadline = Instant::now() + Duration::from_secs(2);
            while PREDICTIVE_STAGE_ACTIVE.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "test left a predictive stage coordinator stalled"
                );
                std::thread::yield_now();
            }
        }
    }

    fn test_guard() -> TestGuard {
        let serial = TEST_SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        TestGuard { _serial: serial }
    }

    fn key(layer: usize, expert: usize) -> PredictiveRecordKey {
        PredictiveRecordKey::new(layer, expert)
    }

    fn bytes(value: u8) -> Box<[u8]> {
        vec![value; 4].into_boxed_slice()
    }

    fn wait_for_status(
        stage: &PredictiveRecordStage,
        key: PredictiveRecordKey,
        wanted: PredictiveRecordStatus,
    ) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if stage.status(key) == Some(wanted) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {wanted:?}"
            );
            std::thread::yield_now();
        }
    }

    fn start_until_permit_released(
        candidates: Vec<PredictiveRecordKey>,
        loader: PredictiveRecordLoader,
    ) -> PredictiveRecordStage {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match PredictiveRecordStage::start(
                candidates.clone(),
                candidates.len(),
                Arc::clone(&loader),
            ) {
                Ok(stage) => return stage,
                Err(PredictiveStageStartError::StageAlreadyActive) => {
                    assert!(
                        Instant::now() < deadline,
                        "global stage permit was not released"
                    );
                    std::thread::yield_now();
                }
                Err(error) => panic!("unexpected stage start failure: {error}"),
            }
        }
    }

    #[test]
    fn ordered_unique_candidates_obey_the_hard_cap() {
        let _serial = test_guard();
        let loader: PredictiveRecordLoader = Arc::new(|key| Ok(bytes(key.expert as u8)));
        let stage =
            PredictiveRecordStage::start([key(2, 7), key(1, 4), key(2, 7), key(0, 9)], 2, loader)
                .unwrap();

        assert_eq!(stage.candidate_keys(), &[key(2, 7), key(1, 4)]);
        assert_eq!(stage.snapshot().total, 2);
        assert_eq!(stage.status(key(0, 9)), None);
    }

    #[test]
    fn invalid_capacity_and_empty_input_do_not_take_the_global_permit() {
        let _serial = test_guard();
        let loader: PredictiveRecordLoader = Arc::new(|_| Ok(bytes(1)));
        assert!(matches!(
            PredictiveRecordStage::start([key(0, 0)], 0, Arc::clone(&loader)),
            Err(PredictiveStageStartError::ZeroCapacity)
        ));
        assert!(matches!(
            PredictiveRecordStage::start([], 1, Arc::clone(&loader)),
            Err(PredictiveStageStartError::NoCandidates)
        ));
        let stage = PredictiveRecordStage::start([key(0, 0)], 1, loader).unwrap();
        assert_eq!(stage.candidate_keys(), &[key(0, 0)]);
    }

    #[test]
    fn demand_waits_for_reading_and_consumes_the_single_staged_load() {
        let _serial = test_guard();
        let wanted = key(3, 11);
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let calls_for_loader = Arc::clone(&calls);
        let release_rx = Mutex::new(release_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            assert_eq!(loaded_key, wanted);
            calls_for_loader.fetch_add(1, Ordering::SeqCst);
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(42))
        });
        let stage = PredictiveRecordStage::start([wanted], 1, loader).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(stage.status(wanted), Some(PredictiveRecordStatus::Reading));

        std::thread::scope(|scope| {
            let demand_started = Arc::new(AtomicBool::new(false));
            let demand_started_in_thread = Arc::clone(&demand_started);
            let (demand_tx, demand_rx) = mpsc::channel();
            let stage_ref = &stage;
            scope.spawn(move || {
                demand_started_in_thread.store(true, Ordering::Release);
                demand_tx.send(stage_ref.take_or_demand(wanted)).unwrap();
            });
            while !demand_started.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            assert!(matches!(
                demand_rx.recv_timeout(Duration::from_millis(20)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            release_tx.send(()).unwrap();
            let record = demand_rx
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap();
            assert_eq!(record.source, PredictiveRecordSource::Staged);
            assert_eq!(&*record.bytes, &[42; 4]);
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(stage.status(wanted), Some(PredictiveRecordStatus::Taken));
        assert_eq!(
            stage.take_or_demand(wanted),
            Err(PredictiveRecordTakeError::AlreadyTaken(wanted))
        );
    }

    #[test]
    fn demand_claims_a_queued_key_and_coordinator_skips_it() {
        let _serial = test_guard();
        let first = key(0, 1);
        let second = key(1, 2);
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let first_calls_for_loader = Arc::clone(&first_calls);
        let second_calls_for_loader = Arc::clone(&second_calls);
        let release_first_rx = Mutex::new(release_first_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            if loaded_key == first {
                first_calls_for_loader.fetch_add(1, Ordering::SeqCst);
                first_started_tx.send(()).unwrap();
                release_first_rx.lock().unwrap().recv().unwrap();
                Ok(bytes(1))
            } else {
                assert_eq!(loaded_key, second);
                second_calls_for_loader.fetch_add(1, Ordering::SeqCst);
                Ok(bytes(2))
            }
        });
        let stage = PredictiveRecordStage::start([first, second], 2, loader).unwrap();
        first_started_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        assert_eq!(stage.status(second), Some(PredictiveRecordStatus::Queued));

        let record = stage.take_or_demand(second).unwrap();
        assert_eq!(record.source, PredictiveRecordSource::Demand);
        assert_eq!(&*record.bytes, &[2; 4]);
        release_first_tx.send(()).unwrap();
        wait_for_status(&stage, first, PredictiveRecordStatus::Ready);

        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stage.status(second), Some(PredictiveRecordStatus::Taken));
    }

    #[test]
    fn concurrent_demand_owner_is_visible_and_never_duplicates_its_load() {
        let _serial = test_guard();
        let first = key(0, 1);
        let second = key(1, 2);
        let second_calls = Arc::new(AtomicUsize::new(0));
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (release_second_tx, release_second_rx) = mpsc::channel();
        let first_release = Mutex::new(release_first_rx);
        let second_release = Mutex::new(release_second_rx);
        let second_calls_for_loader = Arc::clone(&second_calls);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            if loaded_key == first {
                first_started_tx.send(()).unwrap();
                first_release.lock().unwrap().recv().unwrap();
                Ok(bytes(1))
            } else {
                assert_eq!(loaded_key, second);
                second_calls_for_loader.fetch_add(1, Ordering::SeqCst);
                second_started_tx.send(()).unwrap();
                second_release.lock().unwrap().recv().unwrap();
                Ok(bytes(2))
            }
        });
        let stage = PredictiveRecordStage::start([first, second], 2, loader).unwrap();
        first_started_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        std::thread::scope(|scope| {
            let (owner_tx, owner_rx) = mpsc::channel();
            let stage_ref = &stage;
            scope.spawn(move || owner_tx.send(stage_ref.take_or_demand(second)).unwrap());
            second_started_rx
                .recv_timeout(Duration::from_secs(2))
                .unwrap();
            assert_eq!(
                stage.status(second),
                Some(PredictiveRecordStatus::DemandOwned)
            );

            let (waiter_tx, waiter_rx) = mpsc::channel();
            let stage_ref = &stage;
            scope.spawn(move || waiter_tx.send(stage_ref.take_or_demand(second)).unwrap());
            assert!(matches!(
                waiter_rx.recv_timeout(Duration::from_millis(20)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            assert_eq!(second_calls.load(Ordering::SeqCst), 1);

            release_second_tx.send(()).unwrap();
            let owner = owner_rx
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap();
            assert_eq!(owner.source, PredictiveRecordSource::Demand);
            assert_eq!(&*owner.bytes, &[2; 4]);
            assert_eq!(
                waiter_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
                Err(PredictiveRecordTakeError::AlreadyTaken(second))
            );
        });
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        release_first_tx.send(()).unwrap();
        wait_for_status(&stage, first, PredictiveRecordStatus::Ready);
    }

    #[test]
    fn predictive_failure_is_retried_by_authoritative_demand() {
        let _serial = test_guard();
        let wanted = key(4, 8);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_loader = Arc::clone(&calls);
        let loader: PredictiveRecordLoader = Arc::new(move |_| {
            let call = calls_for_loader.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                Err("transient staged read".into())
            } else {
                Ok(bytes(8))
            }
        });
        let stage = PredictiveRecordStage::start([wanted], 1, loader).unwrap();
        wait_for_status(&stage, wanted, PredictiveRecordStatus::Failed);
        assert_eq!(
            stage.failure_message(wanted).as_deref(),
            Some("transient staged read")
        );

        let record = stage.take_or_demand(wanted).unwrap();
        assert_eq!(record.source, PredictiveRecordSource::Demand);
        assert_eq!(&*record.bytes, &[8; 4]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn absent_key_demands_without_expanding_the_bounded_stage() {
        let _serial = test_guard();
        let predicted = key(0, 0);
        let absent = key(9, 9);
        let loader: PredictiveRecordLoader =
            Arc::new(move |loaded_key| Ok(bytes((loaded_key.layer + loaded_key.expert) as u8)));
        let stage = PredictiveRecordStage::start([predicted], 1, loader).unwrap();
        let record = stage.take_or_demand(absent).unwrap();

        assert_eq!(record.source, PredictiveRecordSource::Demand);
        assert_eq!(record.key, absent);
        assert_eq!(&*record.bytes, &[18; 4]);
        assert_eq!(stage.snapshot().total, 1);
        assert_eq!(stage.status(absent), None);
    }

    #[test]
    fn loader_panic_becomes_failed_state_and_does_not_strand_waiters() {
        let _serial = test_guard();
        let wanted = key(5, 5);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_loader = Arc::clone(&calls);
        let loader: PredictiveRecordLoader = Arc::new(move |_| {
            if calls_for_loader.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("synthetic loader panic");
            }
            Ok(bytes(5))
        });
        let stage = PredictiveRecordStage::start([wanted], 1, loader).unwrap();
        wait_for_status(&stage, wanted, PredictiveRecordStatus::Failed);

        let record = stage.take_or_demand(wanted).unwrap();
        assert_eq!(record.source, PredictiveRecordSource::Demand);
        assert_eq!(&*record.bytes, &[5; 4]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn cancellation_discards_ready_bytes_and_rejects_new_demand() {
        let _serial = test_guard();
        let ready = key(0, 0);
        let reading = key(0, 1);
        let queued = key(0, 2);
        let (reading_started_tx, reading_started_rx) = mpsc::channel();
        let (release_reading_tx, release_reading_rx) = mpsc::channel();
        let release_reading_rx = Mutex::new(release_reading_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            if loaded_key == reading {
                reading_started_tx.send(()).unwrap();
                release_reading_rx.lock().unwrap().recv().unwrap();
            }
            Ok(bytes(7))
        });
        let stage = PredictiveRecordStage::start([ready, reading, queued], 3, loader).unwrap();
        reading_started_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        assert_eq!(stage.status(ready), Some(PredictiveRecordStatus::Ready));
        assert_eq!(stage.status(reading), Some(PredictiveRecordStatus::Reading));
        assert_eq!(stage.status(queued), Some(PredictiveRecordStatus::Queued));

        stage.cancel();
        assert_eq!(
            stage.take_or_demand(queued),
            Err(PredictiveRecordTakeError::Cancelled)
        );
        assert_eq!(stage.status(ready), Some(PredictiveRecordStatus::Failed));
        assert_eq!(stage.status(reading), Some(PredictiveRecordStatus::Reading));
        assert_eq!(stage.status(queued), Some(PredictiveRecordStatus::Failed));
        release_reading_tx.send(()).unwrap();
        wait_for_status(&stage, reading, PredictiveRecordStatus::Failed);
        let snapshot = stage.snapshot();
        assert!(snapshot.cancelled);
        assert_eq!(snapshot.failed, 3);
    }

    #[test]
    fn global_permit_survives_owner_drop_until_stalled_read_finishes() {
        let _serial = test_guard();
        let first = key(0, 0);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader_one: PredictiveRecordLoader = Arc::new(move |_| {
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(1))
        });
        let stage_one = PredictiveRecordStage::start([first], 1, loader_one).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let loader_two: PredictiveRecordLoader = Arc::new(|_| Ok(bytes(2)));
        assert!(matches!(
            PredictiveRecordStage::start([key(1, 1)], 1, Arc::clone(&loader_two)),
            Err(PredictiveStageStartError::StageAlreadyActive)
        ));
        drop(stage_one);
        assert!(matches!(
            PredictiveRecordStage::start([key(1, 1)], 1, Arc::clone(&loader_two)),
            Err(PredictiveStageStartError::StageAlreadyActive)
        ));

        release_tx.send(()).unwrap();
        let stage_two = start_until_permit_released(vec![key(1, 1)], loader_two);
        assert_eq!(stage_two.candidate_keys(), &[key(1, 1)]);
    }

    #[test]
    fn snapshot_counts_each_explicit_state_once() {
        let _serial = test_guard();
        let first = key(0, 0);
        let second = key(0, 1);
        let third = key(0, 2);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            if loaded_key == first {
                started_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
            Ok(bytes(loaded_key.expert as u8))
        });
        let stage = PredictiveRecordStage::start([first, second, third], 3, loader).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let _taken = stage.take_or_demand(second).unwrap();

        let snapshot = stage.snapshot();
        assert_eq!(snapshot.total, 3);
        assert_eq!(snapshot.reading, 1);
        assert_eq!(snapshot.taken, 1);
        assert_eq!(snapshot.queued, 1);
        assert_eq!(
            snapshot.queued
                + snapshot.reading
                + snapshot.ready
                + snapshot.failed
                + snapshot.demand_owned
                + snapshot.taken,
            snapshot.total
        );
        release_tx.send(()).unwrap();
    }
}
