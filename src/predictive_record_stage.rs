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
//! supply an immutable, thread-safe loader closure. Process-global lane
//! permits bound live fixed and rolling stages to one of each, including the
//! interval after a stage owner is dropped while its final blocking read is
//! still returning.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
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
    DemandInFlight(PredictiveRecordKey),
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
            Self::DemandInFlight(key) => write!(
                f,
                "predictive record ({}, {}) already has an authoritative demand owner",
                key.layer, key.expert
            ),
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

const FIXED_STAGE_PERMIT: u8 = 1 << 0;
const ROLLING_STAGE_PERMIT: u8 = 1 << 1;
static PREDICTIVE_STAGE_ACTIVE: AtomicU8 = AtomicU8::new(0);

struct GlobalStagePermit {
    lane: u8,
}

impl GlobalStagePermit {
    fn try_acquire(lane: u8) -> Option<Self> {
        debug_assert!(lane == FIXED_STAGE_PERMIT || lane == ROLLING_STAGE_PERMIT);
        PREDICTIVE_STAGE_ACTIVE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active & lane == 0).then_some(active | lane)
            })
            .ok()
            .map(|_| Self { lane })
    }
}

impl Drop for GlobalStagePermit {
    fn drop(&mut self) {
        PREDICTIVE_STAGE_ACTIVE.fetch_and(!self.lane, Ordering::AcqRel);
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

        let permit = GlobalStagePermit::try_acquire(FIXED_STAGE_PERMIT)
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

    /// Take a completed predictive record or atomically claim immediate
    /// authoritative fallback without waiting for speculative I/O.
    ///
    /// A queued, reading, or failed candidate moves to `DemandOwned` before
    /// this method returns [`PredictiveRecordClaim::Demand`]. If a speculative
    /// read was already running, its private result is discarded when the
    /// coordinator observes that the entry is no longer `Reading`. A second
    /// caller can therefore never receive another demand claim for the same
    /// key. The demand owner must publish completion with
    /// [`Self::finish_demand`].
    ///
    /// This method never invokes the loader or waits on a condition variable.
    pub(crate) fn try_claim_ready_or_demand(
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
        if self.inner.cancelled.load(Ordering::Acquire) {
            return Err(PredictiveRecordTakeError::Cancelled);
        }
        match &*state {
            PredictiveRecordState::Queued
            | PredictiveRecordState::Reading
            | PredictiveRecordState::Failed(_) => {
                *state = PredictiveRecordState::DemandOwned;
                entry.changed.notify_all();
                Ok(PredictiveRecordClaim::Demand)
            }
            PredictiveRecordState::Ready(_) => {
                let ready = std::mem::replace(&mut *state, PredictiveRecordState::Taken);
                entry.changed.notify_all();
                let PredictiveRecordState::Ready(bytes) = ready else {
                    unreachable!("state was matched as Ready")
                };
                Ok(PredictiveRecordClaim::Staged(PredictiveRecord {
                    key,
                    source: PredictiveRecordSource::Staged,
                    bytes,
                }))
            }
            PredictiveRecordState::DemandOwned => {
                Err(PredictiveRecordTakeError::DemandInFlight(key))
            }
            PredictiveRecordState::Taken => Err(PredictiveRecordTakeError::AlreadyTaken(key)),
        }
    }

    /// Publish completion of a demand claim made by
    /// [`Self::claim_ready_or_demand`] or
    /// [`Self::try_claim_ready_or_demand`]. Keys outside the predictive set
    /// need no publication. A failed batch can be retried by a later exact
    /// consumer.
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

/// Failure to create a persistent rolling predictive-record worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RollingPredictiveStageStartError {
    ZeroCapacity,
    InvalidWorkerCount {
        requested: usize,
        max_allowed: usize,
    },
    StageAlreadyActive,
    WorkerSpawn(String),
}

impl fmt::Display for RollingPredictiveStageStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(
                f,
                "rolling predictive record stage capacity must be non-zero"
            ),
            Self::InvalidWorkerCount {
                requested,
                max_allowed,
            } => write!(
                f,
                "rolling predictive record worker count {requested} is outside 1..={max_allowed}"
            ),
            Self::StageAlreadyActive => {
                write!(f, "another predictive record stage is still active")
            }
            Self::WorkerSpawn(message) => {
                write!(
                    f,
                    "failed to spawn rolling predictive record worker: {message}"
                )
            }
        }
    }
}

impl std::error::Error for RollingPredictiveStageStartError {}

/// Failure to replace the rolling worker's current generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RollingPredictiveLaunchError {
    Cancelled,
    LayerMismatch {
        target_layer: usize,
        key: PredictiveRecordKey,
    },
    GenerationExhausted,
}

impl fmt::Display for RollingPredictiveLaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "rolling predictive record stage was cancelled"),
            Self::LayerMismatch { target_layer, key } => write!(
                f,
                "rolling predictive target layer {target_layer} does not match candidate layer {}",
                key.layer
            ),
            Self::GenerationExhausted => {
                write!(f, "rolling predictive generation counter was exhausted")
            }
        }
    }
}

impl std::error::Error for RollingPredictiveLaunchError {}

/// Lifetime telemetry for one rolling stage.
///
/// `reads_succeeded` and `reads_failed` describe loader outcomes, including
/// outcomes that arrive after their generation became stale. A successful
/// stale result also increments `late_discarded`. `unused_ready` counts
/// already-published buffers discarded by replacement or cancellation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RollingPredictiveStageSnapshot {
    pub(crate) launches: u64,
    pub(crate) candidates: u64,
    pub(crate) reads_started: u64,
    pub(crate) reads_succeeded: u64,
    pub(crate) reads_failed: u64,
    pub(crate) ready_returned: u64,
    pub(crate) unused_ready: u64,
    pub(crate) late_discarded: u64,
    pub(crate) workers_started: u64,
    pub(crate) workers_done: u64,
    /// Compatibility summary: true only after every started worker exits.
    pub(crate) worker_done: bool,
    pub(crate) cancelled: bool,
}

#[derive(Debug)]
enum RollingPredictiveEntryState {
    Queued,
    Reading,
    Ready(Box<[u8]>),
    Failed,
}

#[derive(Debug)]
struct RollingPredictiveEntry {
    key: PredictiveRecordKey,
    state: RollingPredictiveEntryState,
}

#[derive(Debug)]
struct RollingPredictiveGeneration {
    id: u64,
    target_layer: usize,
    entries: Vec<RollingPredictiveEntry>,
}

#[derive(Debug)]
struct RollingPredictiveShared {
    next_generation: u64,
    active: Option<RollingPredictiveGeneration>,
    telemetry: RollingPredictiveStageSnapshot,
}

struct RollingPredictiveInner {
    max_records: usize,
    loader: PredictiveRecordLoader,
    shared: Mutex<RollingPredictiveShared>,
    work_available: Condvar,
    // Retain the process-global stage permit in shared ownership so a blocked
    // loader cannot be followed by an unbounded succession of new workers.
    // The rolling-lane permit is released only after both the public owner and
    // every worker drop their final `Arc`, including the late-read
    // cancellation interval. A fixed pre-assistant stage has its own bit.
    _permit: GlobalStagePermit,
}

/// A persistent, bounded, generation-aware predictive record stage.
///
/// One to four workers share the generation queue. Each worker calls the
/// supplied loader serially, so [`Self::start`] preserves the original
/// one-worker behavior while [`Self::start_with_workers`] permits bounded
/// parallel reads. Each [`Self::launch`] replaces the previous generation
/// without waiting for old reads. Loader output remains in a worker-private
/// `Box<[u8]>` until that worker can prove that the same generation is still
/// active; otherwise the box is dropped without publication.
///
/// [`Self::seal_layer`] never invokes the loader, waits on a condition
/// variable, or joins the worker. It only takes the short-lived metadata mutex,
/// removes the matching generation, and transfers buffers that were already
/// fully published. Queued, reading, failed, absent, and stale records remain
/// authoritative demand work for the caller.
pub(crate) struct RollingPredictiveRecordStage {
    inner: Arc<RollingPredictiveInner>,
}

impl RollingPredictiveRecordStage {
    /// Start one private serial worker that can serve many rolling launches.
    pub(crate) fn start(
        max_records: usize,
        loader: PredictiveRecordLoader,
    ) -> Result<Self, RollingPredictiveStageStartError> {
        Self::start_with_workers(max_records, 1, loader)
    }

    /// Start a bounded set of persistent workers sharing one generation
    /// queue. The worker count must fit both the record cap and the fixed
    /// four-worker safety ceiling.
    pub(crate) fn start_with_workers(
        max_records: usize,
        worker_count: usize,
        loader: PredictiveRecordLoader,
    ) -> Result<Self, RollingPredictiveStageStartError> {
        if max_records == 0 {
            return Err(RollingPredictiveStageStartError::ZeroCapacity);
        }
        let max_workers = max_records.min(4);
        if !(1..=max_workers).contains(&worker_count) {
            return Err(RollingPredictiveStageStartError::InvalidWorkerCount {
                requested: worker_count,
                max_allowed: max_workers,
            });
        }

        let permit = GlobalStagePermit::try_acquire(ROLLING_STAGE_PERMIT)
            .ok_or(RollingPredictiveStageStartError::StageAlreadyActive)?;

        let inner = Arc::new(RollingPredictiveInner {
            max_records,
            loader,
            shared: Mutex::new(RollingPredictiveShared {
                next_generation: 1,
                active: None,
                telemetry: RollingPredictiveStageSnapshot::default(),
            }),
            work_available: Condvar::new(),
            _permit: permit,
        });

        for worker_index in 0..worker_count {
            let worker_inner = Arc::clone(&inner);
            let spawn_result = std::thread::Builder::new()
                .name(format!(
                    "camelid-rolling-predictive-record-stage-{worker_index}"
                ))
                .spawn(move || run_rolling_predictive_worker(worker_inner));
            match spawn_result {
                Ok(handle) => {
                    // Detach deliberately: cancellation must never join a
                    // worker whose loader is stalled in an OS read.
                    drop(handle);
                    lock_rolling_shared(&inner).telemetry.workers_started += 1;
                }
                Err(error) => {
                    // Fail closed. Already-started workers see cancellation,
                    // discard any private result, and retain the global permit
                    // until every stalled loader has actually returned.
                    lock_rolling_shared(&inner).telemetry.cancelled = true;
                    inner.work_available.notify_all();
                    return Err(RollingPredictiveStageStartError::WorkerSpawn(
                        error.to_string(),
                    ));
                }
            }
        }
        Ok(Self { inner })
    }

    /// Replace pending work with an ordered, unique, hard-capped generation.
    ///
    /// All admitted keys must carry `target_layer`; a mismatch rejects the
    /// launch before it can replace current work. Empty launches are valid and
    /// still invalidate the previous generation.
    pub(crate) fn launch<I>(
        &self,
        target_layer: usize,
        candidates: I,
    ) -> Result<u64, RollingPredictiveLaunchError>
    where
        I: IntoIterator<Item = PredictiveRecordKey>,
    {
        let mut seen = BTreeSet::new();
        let mut entries = Vec::with_capacity(self.inner.max_records);
        for key in candidates {
            if key.layer != target_layer {
                return Err(RollingPredictiveLaunchError::LayerMismatch { target_layer, key });
            }
            if seen.insert(key) {
                entries.push(RollingPredictiveEntry {
                    key,
                    state: RollingPredictiveEntryState::Queued,
                });
                if entries.len() == self.inner.max_records {
                    break;
                }
            }
        }

        let candidate_count = entries.len() as u64;
        let mut shared = lock_rolling_shared(&self.inner);
        if shared.telemetry.cancelled {
            return Err(RollingPredictiveLaunchError::Cancelled);
        }
        let generation = shared.next_generation;
        shared.next_generation = generation
            .checked_add(1)
            .ok_or(RollingPredictiveLaunchError::GenerationExhausted)?;

        if let Some(stale) = shared.active.take() {
            shared.telemetry.unused_ready += rolling_ready_count(&stale);
        }
        shared.active = Some(RollingPredictiveGeneration {
            id: generation,
            target_layer,
            entries,
        });
        shared.telemetry.launches += 1;
        shared.telemetry.candidates += candidate_count;
        drop(shared);
        self.inner.work_available.notify_all();
        Ok(generation)
    }

    /// Seal one exact target layer and take only buffers already fully ready.
    ///
    /// This operation never waits for an outstanding read. Removing the
    /// generation before returning is the publication barrier: an in-flight
    /// result for this generation will subsequently fail the identity check
    /// in the worker and be discarded. Calling with a layer other than the
    /// active target leaves the active generation untouched and returns empty.
    pub(crate) fn seal_layer(&self, target_layer: usize) -> Vec<PredictiveRecord> {
        let mut shared = lock_rolling_shared(&self.inner);
        let Some(active) = shared.active.as_ref() else {
            return Vec::new();
        };
        if active.target_layer != target_layer {
            return Vec::new();
        }
        let active = shared
            .active
            .take()
            .expect("matching rolling generation must still be active");
        let mut ready = Vec::with_capacity(active.entries.len());
        for entry in active.entries {
            if let RollingPredictiveEntryState::Ready(bytes) = entry.state {
                debug_assert_eq!(entry.key.layer, target_layer);
                ready.push(PredictiveRecord {
                    key: entry.key,
                    source: PredictiveRecordSource::Staged,
                    bytes,
                });
            }
        }
        shared.telemetry.ready_returned += ready.len() as u64;
        ready
    }

    /// Cancel queued work and invalidate any generation without joining the
    /// worker. A loader call already in progress is allowed to return into its
    /// private buffer, which the worker then discards.
    pub(crate) fn cancel(&self) {
        let mut shared = lock_rolling_shared(&self.inner);
        if shared.telemetry.cancelled {
            return;
        }
        shared.telemetry.cancelled = true;
        if let Some(stale) = shared.active.take() {
            shared.telemetry.unused_ready += rolling_ready_count(&stale);
        }
        drop(shared);
        self.inner.work_available.notify_all();
    }

    pub(crate) fn snapshot(&self) -> RollingPredictiveStageSnapshot {
        lock_rolling_shared(&self.inner).telemetry
    }
}

impl Drop for RollingPredictiveRecordStage {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn lock_rolling_shared(inner: &RollingPredictiveInner) -> MutexGuard<'_, RollingPredictiveShared> {
    inner
        .shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn rolling_ready_count(generation: &RollingPredictiveGeneration) -> u64 {
    generation
        .entries
        .iter()
        .filter(|entry| matches!(entry.state, RollingPredictiveEntryState::Ready(_)))
        .count() as u64
}

fn run_rolling_predictive_worker(inner: Arc<RollingPredictiveInner>) {
    set_rolling_predictive_worker_utility_qos();
    loop {
        let (generation, target_layer, key) = {
            let mut shared = lock_rolling_shared(&inner);
            loop {
                if shared.telemetry.cancelled {
                    shared.telemetry.workers_done += 1;
                    shared.telemetry.worker_done =
                        shared.telemetry.workers_done == shared.telemetry.workers_started;
                    return;
                }

                let next = shared.active.as_mut().and_then(|active| {
                    active
                        .entries
                        .iter_mut()
                        .find(|entry| matches!(entry.state, RollingPredictiveEntryState::Queued))
                        .map(|entry| {
                            entry.state = RollingPredictiveEntryState::Reading;
                            (active.id, active.target_layer, entry.key)
                        })
                });
                if let Some(next) = next {
                    shared.telemetry.reads_started += 1;
                    break next;
                }
                shared = inner
                    .work_available
                    .wait(shared)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };

        // The worker owns this result privately. No reference to its bytes is
        // installed in shared state until the generation identity is checked.
        let loaded = invoke_loader(&inner.loader, key);
        let mut shared = lock_rolling_shared(&inner);
        match &loaded {
            Ok(_) => shared.telemetry.reads_succeeded += 1,
            Err(_) => shared.telemetry.reads_failed += 1,
        }

        let publish_entry = if shared.telemetry.cancelled {
            None
        } else {
            shared.active.as_mut().and_then(|active| {
                if active.id != generation || active.target_layer != target_layer {
                    return None;
                }
                active.entries.iter_mut().find(|entry| {
                    entry.key == key && matches!(entry.state, RollingPredictiveEntryState::Reading)
                })
            })
        };
        if let Some(entry) = publish_entry {
            entry.state = match loaded {
                Ok(bytes) => RollingPredictiveEntryState::Ready(bytes),
                Err(_) => RollingPredictiveEntryState::Failed,
            };
        } else if loaded.is_ok() {
            shared.telemetry.late_discarded += 1;
        }
    }
}

#[cfg(target_os = "macos")]
fn set_rolling_predictive_worker_utility_qos() {
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }

    // Best effort only: failure keeps the worker correct and merely leaves the
    // platform's inherited scheduling policy in place.
    const QOS_CLASS_UTILITY: u32 = 0x11;
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_rolling_predictive_worker_utility_qos() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    // The production lane permits are intentionally process-global, so these
    // tests must not contend with one another when the Rust harness is parallel.
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
            while PREDICTIVE_STAGE_ACTIVE.load(Ordering::Acquire) != 0 {
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
    fn try_claim_is_nonblocking_for_reading_and_cancel_preserves_demand_ownership() {
        let _serial = test_guard();
        let wanted = key(3, 12);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            assert_eq!(loaded_key, wanted);
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(12))
        });
        let stage = PredictiveRecordStage::start([wanted], 1, loader).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let claim_started = Instant::now();
        assert_eq!(
            stage.try_claim_ready_or_demand(wanted),
            Ok(PredictiveRecordClaim::Demand)
        );
        assert!(
            claim_started.elapsed() < Duration::from_millis(100),
            "try-claim waited for a stalled speculative loader"
        );
        assert_eq!(
            stage.status(wanted),
            Some(PredictiveRecordStatus::DemandOwned)
        );
        assert_eq!(
            stage.try_claim_ready_or_demand(wanted),
            Err(PredictiveRecordTakeError::DemandInFlight(wanted))
        );

        stage.cancel();
        assert_eq!(
            stage.status(wanted),
            Some(PredictiveRecordStatus::DemandOwned)
        );
        stage.finish_demand(wanted, true);
        assert_eq!(stage.status(wanted), Some(PredictiveRecordStatus::Taken));

        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !stage.snapshot().coordinator_done {
            assert!(
                Instant::now() < deadline,
                "speculative coordinator did not discard its late private result"
            );
            std::thread::yield_now();
        }
        assert_eq!(stage.status(wanted), Some(PredictiveRecordStatus::Taken));
    }

    #[test]
    fn try_claim_owns_queued_fallback_and_takes_ready_bytes() {
        let _serial = test_guard();
        let first = key(4, 1);
        let second = key(4, 2);
        let second_speculative_calls = Arc::new(AtomicUsize::new(0));
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let release_first_rx = Mutex::new(release_first_rx);
        let second_calls_for_loader = Arc::clone(&second_speculative_calls);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            if loaded_key == first {
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

        assert_eq!(
            stage.try_claim_ready_or_demand(second),
            Ok(PredictiveRecordClaim::Demand)
        );
        stage.finish_demand(second, true);
        release_first_tx.send(()).unwrap();
        wait_for_status(&stage, first, PredictiveRecordStatus::Ready);

        let first_record = match stage.try_claim_ready_or_demand(first).unwrap() {
            PredictiveRecordClaim::Staged(record) => record,
            PredictiveRecordClaim::Demand => panic!("ready record fell through to demand"),
        };
        assert_eq!(first_record.source, PredictiveRecordSource::Staged);
        assert_eq!(&*first_record.bytes, &[1; 4]);
        assert_eq!(second_speculative_calls.load(Ordering::SeqCst), 0);
        assert_eq!(stage.snapshot().taken, 2);
    }

    #[test]
    fn try_claim_retries_failed_and_leaves_absent_keys_untracked() {
        let _serial = test_guard();
        let failed = key(5, 7);
        let absent = key(6, 8);
        let loader: PredictiveRecordLoader = Arc::new(|_| Err("staged failure".into()));
        let stage = PredictiveRecordStage::start([failed], 1, loader).unwrap();
        wait_for_status(&stage, failed, PredictiveRecordStatus::Failed);

        assert_eq!(
            stage.try_claim_ready_or_demand(failed),
            Ok(PredictiveRecordClaim::Demand)
        );
        stage.finish_demand(failed, false);
        assert_eq!(stage.status(failed), Some(PredictiveRecordStatus::Failed));
        assert_eq!(
            stage.try_claim_ready_or_demand(failed),
            Ok(PredictiveRecordClaim::Demand)
        );
        stage.finish_demand(failed, true);
        assert_eq!(stage.status(failed), Some(PredictiveRecordStatus::Taken));

        assert_eq!(
            stage.try_claim_ready_or_demand(absent),
            Ok(PredictiveRecordClaim::Demand)
        );
        stage.finish_demand(absent, true);
        assert_eq!(stage.status(absent), None);
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
    fn fixed_and_rolling_permit_lanes_coexist_and_release_independently() {
        let _serial = test_guard();
        let fixed_key = key(20, 1);
        let (fixed_started_tx, fixed_started_rx) = mpsc::channel();
        let (release_fixed_tx, release_fixed_rx) = mpsc::channel();
        let release_fixed_rx = Mutex::new(release_fixed_rx);
        let fixed_loader_one: PredictiveRecordLoader = Arc::new(move |_| {
            fixed_started_tx.send(()).unwrap();
            release_fixed_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(1))
        });
        let fixed_one = PredictiveRecordStage::start([fixed_key], 1, fixed_loader_one).unwrap();
        fixed_started_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        let rolling_loader: PredictiveRecordLoader = Arc::new(|key| Ok(bytes(key.expert as u8)));
        let rolling_one =
            RollingPredictiveRecordStage::start(2, Arc::clone(&rolling_loader)).unwrap();
        let fixed_loader_two: PredictiveRecordLoader = Arc::new(|_| Ok(bytes(2)));
        assert!(matches!(
            PredictiveRecordStage::start([key(21, 2)], 1, Arc::clone(&fixed_loader_two)),
            Err(PredictiveStageStartError::StageAlreadyActive)
        ));
        assert!(matches!(
            RollingPredictiveRecordStage::start(2, Arc::clone(&rolling_loader)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));

        drop(fixed_one);
        assert!(matches!(
            PredictiveRecordStage::start([key(21, 2)], 1, Arc::clone(&fixed_loader_two)),
            Err(PredictiveStageStartError::StageAlreadyActive)
        ));
        release_fixed_tx.send(()).unwrap();
        let fixed_two =
            start_until_permit_released(vec![key(21, 2)], Arc::clone(&fixed_loader_two));

        // Releasing the fixed bit must not release the still-owned rolling bit.
        assert!(matches!(
            RollingPredictiveRecordStage::start(2, Arc::clone(&rolling_loader)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));
        stop_rolling(&rolling_one);
        drop(rolling_one);

        let rolling_two = RollingPredictiveRecordStage::start(2, rolling_loader).unwrap();
        // Releasing the rolling bit must not release the still-owned fixed bit.
        assert!(matches!(
            PredictiveRecordStage::start([key(22, 3)], 1, fixed_loader_two),
            Err(PredictiveStageStartError::StageAlreadyActive)
        ));
        stop_rolling(&rolling_two);
        drop(fixed_two);
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

    fn wait_for_rolling_snapshot(
        stage: &RollingPredictiveRecordStage,
        predicate: impl Fn(RollingPredictiveStageSnapshot) -> bool,
    ) -> RollingPredictiveStageSnapshot {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = stage.snapshot();
            if predicate(snapshot) {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for rolling snapshot; last={snapshot:?}"
            );
            std::thread::yield_now();
        }
    }

    fn stop_rolling(stage: &RollingPredictiveRecordStage) {
        stage.cancel();
        wait_for_rolling_snapshot(stage, |snapshot| snapshot.worker_done);
    }

    #[test]
    fn rolling_worker_count_is_validated_before_taking_the_global_permit() {
        let _serial = test_guard();
        let loader: PredictiveRecordLoader = Arc::new(|_| Ok(bytes(1)));

        assert!(matches!(
            RollingPredictiveRecordStage::start_with_workers(0, 1, Arc::clone(&loader)),
            Err(RollingPredictiveStageStartError::ZeroCapacity)
        ));
        assert!(matches!(
            RollingPredictiveRecordStage::start_with_workers(4, 0, Arc::clone(&loader)),
            Err(RollingPredictiveStageStartError::InvalidWorkerCount {
                requested: 0,
                max_allowed: 4,
            })
        ));
        assert!(matches!(
            RollingPredictiveRecordStage::start_with_workers(2, 3, Arc::clone(&loader)),
            Err(RollingPredictiveStageStartError::InvalidWorkerCount {
                requested: 3,
                max_allowed: 2,
            })
        ));
        assert!(matches!(
            RollingPredictiveRecordStage::start_with_workers(8, 5, Arc::clone(&loader)),
            Err(RollingPredictiveStageStartError::InvalidWorkerCount {
                requested: 5,
                max_allowed: 4,
            })
        ));

        // Invalid attempts must not consume the process-global permit, and
        // the legacy constructor must remain exactly one-worker behavior.
        let stage = RollingPredictiveRecordStage::start(4, loader).unwrap();
        assert_eq!(stage.snapshot().workers_started, 1);
        stop_rolling(&stage);
        let snapshot = stage.snapshot();
        assert_eq!(snapshot.workers_done, 1);
        assert!(snapshot.worker_done);
    }

    #[test]
    fn rolling_two_workers_are_concurrent_and_seal_discards_both_late_results() {
        let _serial = test_guard();
        let first = key(13, 1);
        let second = key(13, 2);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let release_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (started_tx, started_rx) = mpsc::channel();

        let in_flight_for_loader = Arc::clone(&in_flight);
        let max_in_flight_for_loader = Arc::clone(&max_in_flight);
        let release_gate_for_loader = Arc::clone(&release_gate);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            let concurrent = in_flight_for_loader.fetch_add(1, Ordering::SeqCst) + 1;
            max_in_flight_for_loader.fetch_max(concurrent, Ordering::SeqCst);
            started_tx.send(loaded_key).unwrap();

            let (released, changed) = &*release_gate_for_loader;
            let mut released = released.lock().unwrap();
            while !*released {
                released = changed.wait(released).unwrap();
            }
            in_flight_for_loader.fetch_sub(1, Ordering::SeqCst);
            Ok(bytes(loaded_key.expert as u8))
        });

        let stage = RollingPredictiveRecordStage::start_with_workers(2, 2, loader).unwrap();
        assert_eq!(stage.snapshot().workers_started, 2);
        stage.launch(13, [first, second]).unwrap();

        let mut started = BTreeSet::new();
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(started, BTreeSet::from([first, second]));
        assert_eq!(max_in_flight.load(Ordering::SeqCst), 2);

        let seal_started = Instant::now();
        let ready = stage.seal_layer(13);
        assert!(
            seal_started.elapsed() < Duration::from_millis(100),
            "seal waited for concurrently stalled predictive loaders"
        );
        assert!(ready.is_empty());

        let (released, changed) = &*release_gate;
        *released.lock().unwrap() = true;
        changed.notify_all();
        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.late_discarded == 2);
        assert_eq!(snapshot.reads_started, 2);
        assert_eq!(snapshot.reads_succeeded, 2);
        assert_eq!(snapshot.ready_returned, 0);

        stop_rolling(&stage);
        let snapshot = stage.snapshot();
        assert_eq!(snapshot.workers_started, 2);
        assert_eq!(snapshot.workers_done, 2);
        assert!(snapshot.worker_done);
    }

    #[test]
    fn rolling_cap16_returns_rank_nine_through_sixteen_in_candidate_order() {
        let _serial = test_guard();
        let loader: PredictiveRecordLoader =
            Arc::new(|loaded_key| Ok(bytes(loaded_key.expert as u8)));
        let stage = RollingPredictiveRecordStage::start_with_workers(16, 2, loader).unwrap();
        let candidates = (0..18).map(|expert| key(20, expert)).collect::<Vec<_>>();
        stage.launch(20, candidates).unwrap();

        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.reads_succeeded == 16);
        assert_eq!(snapshot.candidates, 16);
        assert_eq!(snapshot.reads_started, 16);
        let ready = stage.seal_layer(20);
        assert_eq!(ready.len(), 16);
        for (expert, record) in ready.iter().enumerate() {
            assert_eq!(record.key, key(20, expert));
            assert_eq!(record.source, PredictiveRecordSource::Staged);
            assert_eq!(record.bytes.as_ref(), &[expert as u8; 4]);
        }
        let snapshot = stage.snapshot();
        assert_eq!(snapshot.ready_returned, 16);
        stop_rolling(&stage);
    }

    #[test]
    fn rolling_cap16_dual_reader_stays_private_bounded_and_nonblocking() {
        let _serial = test_guard();
        let release_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let release_gate_for_loader = Arc::clone(&release_gate);
        let (started_tx, started_rx) = mpsc::channel();
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            started_tx.send(loaded_key).unwrap();
            let (released, changed) = &*release_gate_for_loader;
            let mut released = released.lock().unwrap();
            while !*released {
                released = changed.wait(released).unwrap();
            }
            Ok(bytes(loaded_key.expert as u8))
        });

        let stage = RollingPredictiveRecordStage::start_with_workers(16, 2, loader).unwrap();
        let candidates = (0..18).map(|expert| key(19, expert)).collect::<Vec<_>>();
        stage.launch(19, candidates).unwrap();

        let mut started = BTreeSet::new();
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(started, BTreeSet::from([key(19, 0), key(19, 1)]));
        let snapshot = stage.snapshot();
        assert_eq!(snapshot.candidates, 16);
        assert_eq!(snapshot.reads_started, 2);

        // A wrong exact layer cannot steal ownership or cancel the generation.
        assert!(stage.seal_layer(18).is_empty());
        let seal_started = Instant::now();
        assert!(stage.seal_layer(19).is_empty());
        assert!(
            seal_started.elapsed() < Duration::from_millis(100),
            "cap16 seal waited for private predictive reads"
        );

        let (released, changed) = &*release_gate;
        *released.lock().unwrap() = true;
        changed.notify_all();
        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.late_discarded == 2);
        assert_eq!(snapshot.reads_succeeded, 2);
        assert_eq!(snapshot.ready_returned, 0);
        stop_rolling(&stage);
        let snapshot = stage.snapshot();
        assert!(snapshot.cancelled);
        assert_eq!(snapshot.workers_done, 2);
    }

    #[test]
    fn rolling_launch_preserves_first_unique_order_under_hard_cap() {
        let _serial = test_guard();
        let loaded = Arc::new(Mutex::new(Vec::new()));
        let loaded_for_worker = Arc::clone(&loaded);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            loaded_for_worker.lock().unwrap().push(loaded_key);
            Ok(bytes(loaded_key.expert as u8))
        });
        let stage = RollingPredictiveRecordStage::start(3, loader).unwrap();
        let generation = stage
            .launch(6, [key(6, 4), key(6, 1), key(6, 4), key(6, 9), key(6, 2)])
            .unwrap();
        assert_eq!(generation, 1);
        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.reads_succeeded == 3);
        assert_eq!(snapshot.launches, 1);
        assert_eq!(snapshot.candidates, 3);
        assert_eq!(snapshot.reads_started, 3);
        assert_eq!(
            *loaded.lock().unwrap(),
            vec![key(6, 4), key(6, 1), key(6, 9)]
        );

        let ready = stage.seal_layer(6);
        assert_eq!(
            ready.iter().map(|record| record.key).collect::<Vec<_>>(),
            vec![key(6, 4), key(6, 1), key(6, 9)]
        );
        stop_rolling(&stage);
    }

    #[test]
    fn rolling_seal_is_nonblocking_during_a_stalled_private_read() {
        let _serial = test_guard();
        let wanted = key(7, 3);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            assert_eq!(loaded_key, wanted);
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(3))
        });
        let stage = RollingPredictiveRecordStage::start(8, loader).unwrap();
        stage.launch(7, [wanted]).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        std::thread::scope(|scope| {
            let (sealed_tx, sealed_rx) = mpsc::channel();
            let stage_ref = &stage;
            scope.spawn(move || sealed_tx.send(stage_ref.seal_layer(7)).unwrap());
            let sealed = sealed_rx
                .recv_timeout(Duration::from_millis(100))
                .expect("seal waited for the stalled predictive loader");
            assert!(sealed.is_empty());
            release_tx.send(()).unwrap();
        });

        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.late_discarded == 1);
        assert_eq!(snapshot.reads_succeeded, 1);
        assert_eq!(snapshot.ready_returned, 0);
        stop_rolling(&stage);
    }

    #[test]
    fn rolling_new_generation_replaces_stale_work_and_drops_late_bytes() {
        let _serial = test_guard();
        let old = key(2, 1);
        let new = key(3, 5);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_old_tx, release_old_rx) = mpsc::channel();
        let release_old_rx = Mutex::new(release_old_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            started_tx.send(loaded_key).unwrap();
            if loaded_key == old {
                release_old_rx.lock().unwrap().recv().unwrap();
            }
            Ok(bytes(loaded_key.expert as u8))
        });
        let stage = RollingPredictiveRecordStage::start(8, loader).unwrap();
        assert_eq!(stage.launch(2, [old]).unwrap(), 1);
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            old
        );

        // Replacement is a metadata operation and cannot wait for `old`.
        assert_eq!(stage.launch(3, [new]).unwrap(), 2);
        assert_eq!(stage.snapshot().launches, 2);
        release_old_tx.send(()).unwrap();
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            new
        );
        let snapshot = wait_for_rolling_snapshot(&stage, |snapshot| snapshot.reads_succeeded == 2);
        assert_eq!(snapshot.late_discarded, 1);

        let ready = stage.seal_layer(3);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].key, new);
        assert_eq!(&*ready[0].bytes, &[5; 4]);
        assert!(stage.seal_layer(2).is_empty());
        stop_rolling(&stage);
    }

    #[test]
    fn rolling_ready_records_keep_layer_identity_and_unused_ready_is_discarded() {
        let _serial = test_guard();
        let loader: PredictiveRecordLoader =
            Arc::new(move |loaded_key| Ok(bytes(loaded_key.expert as u8)));
        let stage = RollingPredictiveRecordStage::start(2, loader).unwrap();

        assert_eq!(
            stage.launch(8, [key(9, 1)]),
            Err(RollingPredictiveLaunchError::LayerMismatch {
                target_layer: 8,
                key: key(9, 1),
            })
        );
        assert_eq!(stage.snapshot().launches, 0);

        stage.launch(8, [key(8, 2), key(8, 7)]).unwrap();
        wait_for_rolling_snapshot(&stage, |snapshot| snapshot.reads_succeeded == 2);
        assert!(stage.seal_layer(7).is_empty());
        let ready = stage.seal_layer(8);
        assert_eq!(ready.len(), 2);
        assert!(ready.iter().all(|record| {
            record.key.layer == 8 && record.source == PredictiveRecordSource::Staged
        }));
        assert_eq!(&*ready[0].bytes, &[2; 4]);
        assert_eq!(&*ready[1].bytes, &[7; 4]);
        assert_eq!(stage.snapshot().ready_returned, 2);

        stage.launch(10, [key(10, 4)]).unwrap();
        wait_for_rolling_snapshot(&stage, |snapshot| snapshot.reads_succeeded == 3);
        stage.launch(11, []).unwrap();
        let snapshot = stage.snapshot();
        assert_eq!(snapshot.unused_ready, 1);
        assert_eq!(snapshot.launches, 3);
        assert_eq!(snapshot.candidates, 3);
        stop_rolling(&stage);
    }

    #[test]
    fn rolling_global_permit_survives_owner_drop_until_stalled_read_finishes() {
        let _serial = test_guard();
        let wanted = key(11, 3);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader_one: PredictiveRecordLoader = Arc::new(move |_| {
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(3))
        });
        let stage_one = RollingPredictiveRecordStage::start(8, loader_one).unwrap();
        stage_one.launch(11, [wanted]).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let loader_two: PredictiveRecordLoader = Arc::new(|key| Ok(bytes(key.expert as u8)));
        assert!(matches!(
            RollingPredictiveRecordStage::start(8, Arc::clone(&loader_two)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));
        drop(stage_one);
        assert!(matches!(
            RollingPredictiveRecordStage::start(8, Arc::clone(&loader_two)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));

        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let stage_two = loop {
            match RollingPredictiveRecordStage::start(8, Arc::clone(&loader_two)) {
                Ok(stage) => break stage,
                Err(RollingPredictiveStageStartError::StageAlreadyActive) => {
                    assert!(
                        Instant::now() < deadline,
                        "rolling global stage permit was not released"
                    );
                    std::thread::yield_now();
                }
                Err(error) => panic!("unexpected rolling stage start failure: {error}"),
            }
        };
        stage_two.launch(12, [key(12, 4)]).unwrap();
        wait_for_rolling_snapshot(&stage_two, |snapshot| snapshot.reads_succeeded == 1);
        stop_rolling(&stage_two);
    }

    #[test]
    fn rolling_multiworker_permit_survives_until_every_stalled_read_finishes() {
        let _serial = test_guard();
        let first = key(14, 1);
        let second = key(14, 2);
        let released_keys = Arc::new((Mutex::new(BTreeSet::new()), Condvar::new()));
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let released_keys_for_loader = Arc::clone(&released_keys);
        let loader_one: PredictiveRecordLoader = Arc::new(move |loaded_key| {
            started_tx.send(loaded_key).unwrap();
            let (released, changed) = &*released_keys_for_loader;
            let mut released = released.lock().unwrap();
            while !released.contains(&loaded_key) {
                released = changed.wait(released).unwrap();
            }
            finished_tx.send(loaded_key).unwrap();
            Ok(bytes(loaded_key.expert as u8))
        });
        let stage_one = RollingPredictiveRecordStage::start_with_workers(2, 2, loader_one).unwrap();
        stage_one.launch(14, [first, second]).unwrap();

        let mut started = BTreeSet::new();
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        started.insert(started_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(started, BTreeSet::from([first, second]));

        let loader_two: PredictiveRecordLoader = Arc::new(|key| Ok(bytes(key.expert as u8)));
        assert!(matches!(
            RollingPredictiveRecordStage::start(2, Arc::clone(&loader_two)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));
        drop(stage_one);
        assert!(matches!(
            RollingPredictiveRecordStage::start(2, Arc::clone(&loader_two)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));

        let (released, changed) = &*released_keys;
        released.lock().unwrap().insert(first);
        changed.notify_all();
        assert_eq!(
            finished_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            first
        );
        // One completed worker cannot release the permit while its sibling is
        // still blocked in a loader call.
        assert!(matches!(
            RollingPredictiveRecordStage::start(2, Arc::clone(&loader_two)),
            Err(RollingPredictiveStageStartError::StageAlreadyActive)
        ));

        released.lock().unwrap().insert(second);
        changed.notify_all();
        assert_eq!(
            finished_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            second
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        let stage_two = loop {
            match RollingPredictiveRecordStage::start(2, Arc::clone(&loader_two)) {
                Ok(stage) => break stage,
                Err(RollingPredictiveStageStartError::StageAlreadyActive) => {
                    assert!(
                        Instant::now() < deadline,
                        "multiworker global stage permit was not released"
                    );
                    std::thread::yield_now();
                }
                Err(error) => panic!("unexpected rolling stage start failure: {error}"),
            }
        };
        stage_two.launch(15, [key(15, 3)]).unwrap();
        wait_for_rolling_snapshot(&stage_two, |snapshot| snapshot.reads_succeeded == 1);
        stop_rolling(&stage_two);
    }

    #[test]
    fn rolling_drop_cancels_without_joining_a_stalled_read() {
        let _serial = test_guard();
        let wanted = key(12, 6);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let loader: PredictiveRecordLoader = Arc::new(move |_| {
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(bytes(6))
        });
        let stage = RollingPredictiveRecordStage::start(8, loader).unwrap();
        let inner = Arc::clone(&stage.inner);
        stage.launch(12, [wanted]).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let (dropped_tx, dropped_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(stage);
            dropped_tx.send(()).unwrap();
        });
        dropped_rx
            .recv_timeout(Duration::from_millis(100))
            .expect("drop joined the stalled rolling worker");
        assert!(lock_rolling_shared(&inner).telemetry.cancelled);
        assert!(!lock_rolling_shared(&inner).telemetry.worker_done);

        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = lock_rolling_shared(&inner).telemetry;
            if snapshot.worker_done {
                assert_eq!(snapshot.late_discarded, 1);
                assert_eq!(snapshot.reads_succeeded, 1);
                break;
            }
            assert!(Instant::now() < deadline, "rolling worker did not cancel");
            std::thread::yield_now();
        }
    }
}
