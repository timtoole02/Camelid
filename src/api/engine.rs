//! The engine worker: one dedicated OS thread that is the single place decode
//! compute and resident-GPU-state mutations execute.
//!
//! Ownership contract (mirrors llama.cpp's `server_queue` + single consumer
//! thread, see docs/recon/ENGINE_INVERSION_CONDUCTOR.md): HTTP handlers
//! validate and prepare OUTSIDE any serialization, then post a job on a
//! bounded queue and await its result. The engine thread executes at most one
//! compute step at a time; opt-in streaming jobs may yield between tokens and
//! rotate round-robin, but never execute concurrently. Anything that touches
//! engine-owned state (decode loops, the GPU-runnable parity probe,
//! resident-cache resets) must run as an engine job, never inline in a
//! handler.
//!
//! Cancellation stays cooperative: a posted job cannot be aborted, but every
//! decode loop observes its request's `GenerationCancel` once per step, so a
//! dropped handler (client disconnect) stops the running job within one step
//! and queued jobs from dropped handlers return immediately when they run.

use std::{
    any::Any,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use super::continuous_batch::ContinuousBatch;
pub(crate) use super::continuous_batch::StepOutcome;

/// Bounded queue depth (queued jobs, not counting the one running).
/// Overridable for hardening runs; the default keeps a small, honest queue —
/// beyond it the server answers 503 rather than parking unbounded waiters.
pub(crate) const QUEUE_DEPTH_ENV: &str = crate::runtime_config::ENGINE_QUEUE_DEPTH_ENV;

type ExclusiveJob = Box<dyn FnOnce() + Send + 'static>;

/// Scheduler state visible to one cooperative token step. A single active stream may
/// retain Metal encode-ahead; contention disables it before another session reaches
/// the shared command queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CooperativeStepContext {
    pub(crate) active_slots: usize,
}

type ScalarCooperativeJob = Box<dyn FnMut(CooperativeStepContext) -> StepOutcome + Send + 'static>;

/// Opaque grouping identity supplied by a cooperative job. Equality selects a
/// candidate pair; the job still owns exact, mutation-free compatibility
/// preflight before shared dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CooperativeBatchKey(u64, usize, CooperativeBatchPhase);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CooperativeBatchPhase {
    Decode,
    Prefill,
}

impl CooperativeBatchKey {
    pub(crate) fn new(group: u64, instance: usize) -> Self {
        Self(group, instance, CooperativeBatchPhase::Decode)
    }

    pub(crate) fn prefill(group: u64, instance: usize) -> Self {
        Self(group, instance, CooperativeBatchPhase::Prefill)
    }
}

/// Engine-thread-owned stream state that can optionally execute one token for
/// two compatible rows in one model forward.
pub(crate) trait BatchableCooperativeJob: Any + Send {
    fn compatibility_key(&mut self) -> Option<CooperativeBatchKey>;

    fn step(&mut self, context: CooperativeStepContext) -> StepOutcome;

    /// Return `None` only when no shared execution was dispatched and scalar
    /// fallback is safe for both rows.
    fn step_pair(
        &mut self,
        other: &mut dyn BatchableCooperativeJob,
        context: CooperativeStepContext,
    ) -> Option<[StepOutcome; 2]>;

    fn step_group(
        &mut self,
        others: &mut [&mut dyn BatchableCooperativeJob],
        context: CooperativeStepContext,
    ) -> Option<Vec<StepOutcome>> {
        if others.len() != 1 {
            return None;
        }
        self.step_pair(others[0], context).map(Vec::from)
    }

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn completed_units(&self) -> Option<u64> {
        None
    }

    fn prefill_progress(&self) -> Option<(u64, u64)> {
        None
    }

    fn is_prefill_pending(&self) -> bool {
        false
    }
}

enum CooperativeJobKind {
    Scalar(ScalarCooperativeJob),
    Batchable(Box<dyn BatchableCooperativeJob>),
}

/// A unit of engine work. Exclusive jobs run to completion; cooperative jobs
/// yield at token boundaries and rotate on the same engine thread.
pub(crate) enum EngineTask {
    /// A serialized blocking job: a decode loop, the GPU-runnable parity
    /// probe, a resident-cache reset, a prompt-cache mutation. The closure
    /// owns everything it needs and reports back through a channel it
    /// captured (typically `tokio::sync::oneshot`).
    Exclusive(ExclusiveJob),
    /// One-token-at-a-time streaming decode. The worker rotates active jobs
    /// round-robin; returning `Complete` releases the slot.
    #[cfg_attr(not(test), allow(dead_code))]
    Cooperative(ScalarCooperativeJob),
    /// A cooperative stream that may share one model forward with another
    /// compatible ready stream. Unsupported rounds retain scalar semantics.
    BatchableCooperative(Box<dyn BatchableCooperativeJob>),
}

enum QueuedEngineTask {
    Exclusive(ExclusiveJob),
    Cooperative {
        task_id: u64,
        kind: CooperativeJobKind,
    },
}

/// Why a post failed. `QueueFull` maps to the typed 503
/// (`engine_queue_full`); `Unavailable` means the engine thread is gone
/// (process shutdown) and maps to a 503 as well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnginePostError {
    QueueFull,
    Unavailable,
}

/// Cloneable handle to the engine worker. Lives in `AppState`; dropping every
/// clone closes the queue and the engine thread exits after finishing the
/// jobs already accepted.
#[derive(Clone)]
pub(crate) struct EngineHandle {
    tx: tokio::sync::mpsc::Sender<QueuedEngineTask>,
    /// Jobs accepted but not yet finished (queued + running). Surfaced in
    /// `/v1/health` and `/v1/slots` so backpressure is observable.
    depth: Arc<AtomicUsize>,
    /// Real task currently executing on the single-owner engine. Zero means
    /// idle; public snapshots convert the sentinel to `None`.
    active_task_id: Arc<AtomicU64>,
    active_started_epoch_millis: Arc<AtomicU64>,
    active_last_progress_epoch_millis: Arc<AtomicU64>,
    active_completed_units: Arc<AtomicU64>,
    next_task_id: Arc<AtomicU64>,
    continuous_batch_slots: usize,
    admitted_slots: usize,
    /// Cooperative slots the worker currently holds. Published by the worker
    /// itself, which already computes it for `CooperativeStepContext`.
    occupied_slots: Arc<AtomicUsize>,
    /// 1 while an exclusive job owns the engine. It is not a slot, but it does
    /// block every slot, so capacity questions must count it.
    exclusive_active: Arc<AtomicUsize>,
    cooperative_slot_states: Arc<std::sync::Mutex<Vec<CooperativePublicSlotState>>>,
}

fn queue_depth_from_env() -> usize {
    crate::runtime_config::engine_queue_depth()
}

fn effective_slot_capacity(
    configured: usize,
    cuda_resident_active: bool,
    cuda_sequence_capacity: usize,
) -> usize {
    if !cuda_resident_active {
        configured.max(1)
    } else {
        configured.clamp(1, cuda_sequence_capacity.clamp(1, 8))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineSlotSnapshot {
    pub(crate) active_task_id: Option<u64>,
    pub(crate) queued_tasks: usize,
    pub(crate) completed_units: u64,
    pub(crate) active_elapsed_seconds: u64,
    pub(crate) stalled_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineCooperativeSlotSnapshot {
    pub(crate) id: usize,
    pub(crate) active_task_id: Option<u64>,
    pub(crate) completed_units: u64,
    pub(crate) prefill_completed_units: u64,
    pub(crate) prefill_total_units: u64,
    pub(crate) active_elapsed_seconds: u64,
    pub(crate) stalled_seconds: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CooperativePublicSlotState {
    task_id: u64,
    started_epoch_millis: u64,
    last_progress_epoch_millis: u64,
    completed_units: u64,
    prefill_completed_units: u64,
    prefill_total_units: u64,
}

impl EngineSlotSnapshot {
    pub(crate) fn is_processing(self) -> bool {
        self.active_task_id.is_some()
    }
}

struct ActiveTaskGuard {
    active_task_id: Arc<AtomicU64>,
    active_started_epoch_millis: Arc<AtomicU64>,
    active_last_progress_epoch_millis: Arc<AtomicU64>,
    active_completed_units: Arc<AtomicU64>,
}

/// One cooperative job's slice of the shared engine-slot state. Held across
/// yields so the published `started`/`completed_units` describe the job, not
/// the current token step.
struct CooperativeSlotState {
    started: u64,
    last_progress: u64,
    completed_units: u64,
    prefill_completed_units: u64,
    prefill_total_units: u64,
    /// Held, never read: dropping it clears the shared slot atomics when the
    /// job completes or panics.
    #[allow(dead_code)]
    guard: ActiveTaskGuard,
}

struct ScheduledCooperativeJob {
    kind: CooperativeJobKind,
    task_id: u64,
    active_task_id: Arc<AtomicU64>,
    active_started_epoch_millis: Arc<AtomicU64>,
    active_last_progress_epoch_millis: Arc<AtomicU64>,
    active_completed_units: Arc<AtomicU64>,
    slot: Option<CooperativeSlotState>,
    public_slots: Arc<std::sync::Mutex<Vec<CooperativePublicSlotState>>>,
    public_slot_index: Option<usize>,
}

impl Drop for ActiveTaskGuard {
    fn drop(&mut self) {
        self.active_task_id.store(0, Ordering::SeqCst);
        self.active_started_epoch_millis.store(0, Ordering::SeqCst);
        self.active_last_progress_epoch_millis
            .store(0, Ordering::SeqCst);
        self.active_completed_units.store(0, Ordering::SeqCst);
    }
}

fn epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

impl ScheduledCooperativeJob {
    fn compatibility_key(&mut self) -> Option<CooperativeBatchKey> {
        match &mut self.kind {
            CooperativeJobKind::Scalar(_) => None,
            CooperativeJobKind::Batchable(job) => job.compatibility_key(),
        }
    }

    fn is_prefill_pending(&self) -> bool {
        match &self.kind {
            CooperativeJobKind::Scalar(_) => false,
            CooperativeJobKind::Batchable(job) => job.is_prefill_pending(),
        }
    }

    fn begin_step(&mut self) {
        let now = epoch_millis();
        let state = self.slot.get_or_insert_with(|| CooperativeSlotState {
            started: now,
            last_progress: now,
            completed_units: 0,
            prefill_completed_units: 0,
            prefill_total_units: 0,
            guard: ActiveTaskGuard {
                active_task_id: Arc::clone(&self.active_task_id),
                active_started_epoch_millis: Arc::clone(&self.active_started_epoch_millis),
                active_last_progress_epoch_millis: Arc::clone(
                    &self.active_last_progress_epoch_millis,
                ),
                active_completed_units: Arc::clone(&self.active_completed_units),
            },
        });
        self.active_started_epoch_millis
            .store(state.started, Ordering::SeqCst);
        self.active_last_progress_epoch_millis
            .store(state.last_progress, Ordering::SeqCst);
        self.active_completed_units
            .store(state.completed_units, Ordering::SeqCst);
        self.active_task_id.store(self.task_id, Ordering::SeqCst);

        let mut slots = self
            .public_slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let index = match self.public_slot_index {
            Some(index) => index,
            None => {
                let Some(index) = slots.iter().position(|slot| slot.task_id == 0) else {
                    return;
                };
                self.public_slot_index = Some(index);
                index
            }
        };
        slots[index] = CooperativePublicSlotState {
            task_id: self.task_id,
            started_epoch_millis: state.started,
            last_progress_epoch_millis: state.last_progress,
            completed_units: state.completed_units,
            prefill_completed_units: state.prefill_completed_units,
            prefill_total_units: state.prefill_total_units,
        };
    }

    fn finish_step(&mut self, outcome: StepOutcome) {
        let reported_completed_units = match &self.kind {
            CooperativeJobKind::Scalar(_) => None,
            CooperativeJobKind::Batchable(job) => job.completed_units(),
        };
        let reported_prefill_progress = match &self.kind {
            CooperativeJobKind::Scalar(_) => None,
            CooperativeJobKind::Batchable(job) => job.prefill_progress(),
        };
        if let Some(state) = self.slot.as_mut() {
            let completed_units = reported_completed_units
                .unwrap_or_else(|| self.active_completed_units.load(Ordering::SeqCst));
            let (prefill_completed_units, prefill_total_units) =
                reported_prefill_progress.unwrap_or((0, 0));
            if completed_units > state.completed_units
                || prefill_completed_units > state.prefill_completed_units
            {
                state.last_progress = epoch_millis();
            } else if reported_completed_units.is_none() {
                state.last_progress = self
                    .active_last_progress_epoch_millis
                    .load(Ordering::SeqCst);
            }
            state.completed_units = completed_units;
            state.prefill_completed_units = prefill_completed_units;
            state.prefill_total_units = prefill_total_units;
            if let Some(index) = self.public_slot_index {
                let mut slots = self
                    .public_slots
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                slots[index] = CooperativePublicSlotState {
                    task_id: self.task_id,
                    started_epoch_millis: state.started,
                    last_progress_epoch_millis: state.last_progress,
                    completed_units: state.completed_units,
                    prefill_completed_units: state.prefill_completed_units,
                    prefill_total_units: state.prefill_total_units,
                };
            }
        }
        if outcome == StepOutcome::Complete {
            self.slot = None;
            self.clear_public_slot();
        }
    }

    fn publish_aggregate(&self) {
        let Some(state) = self.slot.as_ref() else {
            return;
        };
        self.active_started_epoch_millis
            .store(state.started, Ordering::SeqCst);
        self.active_last_progress_epoch_millis
            .store(state.last_progress, Ordering::SeqCst);
        self.active_completed_units
            .store(state.completed_units, Ordering::SeqCst);
        self.active_task_id.store(self.task_id, Ordering::SeqCst);
    }

    fn clear_public_slot(&mut self) {
        let Some(index) = self.public_slot_index.take() else {
            return;
        };
        let mut slots = self
            .public_slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slots[index].task_id == self.task_id {
            slots[index] = CooperativePublicSlotState::default();
        }
    }

    fn step(&mut self, context: CooperativeStepContext) -> StepOutcome {
        self.begin_step();
        let outcome = match &mut self.kind {
            CooperativeJobKind::Scalar(job) => job(context),
            CooperativeJobKind::Batchable(job) => job.step(context),
        };
        self.finish_step(outcome);
        outcome
    }

    fn step_group(jobs: &mut [Self], context: CooperativeStepContext) -> Option<Vec<StepOutcome>> {
        if jobs.len() < 2 {
            return None;
        }
        for job in &mut *jobs {
            job.begin_step();
        }
        jobs[0]
            .active_task_id
            .store(jobs[0].task_id, Ordering::SeqCst);
        let outcomes = {
            let (first, rest) = jobs.split_first_mut()?;
            let CooperativeJobKind::Batchable(first_job) = &mut first.kind else {
                return None;
            };
            let mut others = Vec::<&mut dyn BatchableCooperativeJob>::with_capacity(rest.len());
            for job in rest {
                let CooperativeJobKind::Batchable(other) = &mut job.kind else {
                    return None;
                };
                others.push(other.as_mut());
            }
            first_job.step_group(&mut others, context)
        }?;
        let outcomes = if outcomes.len() == jobs.len() {
            outcomes
        } else {
            vec![StepOutcome::Complete; jobs.len()]
        };
        for (job, outcome) in jobs.iter_mut().zip(outcomes.iter().copied()) {
            job.finish_step(outcome);
        }
        if let Some(active) = jobs.iter().find(|job| job.slot.is_some()) {
            active.publish_aggregate();
        }
        Some(outcomes)
    }
}

impl Drop for ScheduledCooperativeJob {
    fn drop(&mut self) {
        self.clear_public_slot();
    }
}

impl EngineHandle {
    /// Spawn the engine worker thread and return the posting handle.
    pub(crate) fn spawn() -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<QueuedEngineTask>(queue_depth_from_env());
        // Resolve configuration on the spawning thread. Reading it lazily
        // inside the worker races tests and embedders that scope environment
        // overrides around `spawn()`.
        let continuous_batch_slots = crate::runtime_config::continuous_batch_slots();
        let admitted_slots = effective_slot_capacity(
            continuous_batch_slots,
            crate::inference::resident_decode_cuda_active(),
            crate::inference::resident_cuda_sequence_capacity(),
        );
        let depth = Arc::new(AtomicUsize::new(0));
        let worker_depth = Arc::clone(&depth);
        let active_task_id = Arc::new(AtomicU64::new(0));
        let active_started_epoch_millis = Arc::new(AtomicU64::new(0));
        let active_last_progress_epoch_millis = Arc::new(AtomicU64::new(0));
        let active_completed_units = Arc::new(AtomicU64::new(0));
        let occupied_slots = Arc::new(AtomicUsize::new(0));
        let exclusive_active = Arc::new(AtomicUsize::new(0));
        let cooperative_slot_states = Arc::new(std::sync::Mutex::new(vec![
            CooperativePublicSlotState::default();
            admitted_slots
        ]));
        let worker_active_task_id = Arc::clone(&active_task_id);
        let worker_active_started_epoch_millis = Arc::clone(&active_started_epoch_millis);
        let worker_active_last_progress_epoch_millis =
            Arc::clone(&active_last_progress_epoch_millis);
        let worker_active_completed_units = Arc::clone(&active_completed_units);
        let worker_occupied = Arc::clone(&occupied_slots);
        let worker_exclusive = Arc::clone(&exclusive_active);
        let worker_cooperative_slot_states = Arc::clone(&cooperative_slot_states);
        std::thread::Builder::new()
            .name("camelid-engine".to_string())
            .spawn(move || {
                let mut batch = ContinuousBatch::<ScheduledCooperativeJob>::new(admitted_slots);
                // At most ONE task is ever held outside the channel: a
                // cooperative job that arrived with every slot busy. Draining
                // the channel into an unbounded local queue instead would make
                // `try_send` never report `Full`, and the typed `QueueFull` ->
                // 503 backpressure would silently stop existing for as long as
                // any stream was running.
                let mut pending: Option<ScheduledCooperativeJob> = None;
                let mut disconnected = false;
                loop {
                    // A held-back stream takes the first freed slot, ahead of
                    // anything still in the channel.
                    if let Some(job) = pending.take() {
                        if batch.has_free_slot() {
                            batch.admit(job);
                        } else {
                            pending = Some(job);
                        }
                    }
                    // Admit until a stream arrives with no slot for it. Every
                    // slot is filled before the round starts, so two streams
                    // posted back to back alternate from their first token.
                    while pending.is_none() {
                        let task = if batch.is_empty() {
                            if disconnected {
                                break;
                            }
                            match rx.blocking_recv() {
                                Some(task) => task,
                                None => {
                                    disconnected = true;
                                    break;
                                }
                            }
                        } else {
                            match rx.try_recv() {
                                Ok(task) => task,
                                Err(_) => break,
                            }
                        };
                        match task {
                            // Exclusive work runs as soon as it is picked up.
                            // Making it wait for `batch.is_empty()` lets
                            // overlapping streams starve model load/unload,
                            // non-streaming completions, the parity probe and
                            // resident-cache resets indefinitely.
                            QueuedEngineTask::Exclusive(job) => {
                                worker_exclusive.store(1, Ordering::Relaxed);
                                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                                worker_exclusive.store(0, Ordering::Relaxed);
                                worker_depth.fetch_sub(1, Ordering::SeqCst);
                            }
                            QueuedEngineTask::Cooperative { task_id, kind } => {
                                let job = ScheduledCooperativeJob {
                                    kind,
                                    task_id,
                                    active_task_id: Arc::clone(&worker_active_task_id),
                                    active_started_epoch_millis: Arc::clone(
                                        &worker_active_started_epoch_millis,
                                    ),
                                    active_last_progress_epoch_millis: Arc::clone(
                                        &worker_active_last_progress_epoch_millis,
                                    ),
                                    active_completed_units: Arc::clone(
                                        &worker_active_completed_units,
                                    ),
                                    slot: None,
                                    public_slots: Arc::clone(&worker_cooperative_slot_states),
                                    public_slot_index: None,
                                };
                                if batch.has_free_slot() {
                                    batch.admit(job);
                                } else {
                                    pending = Some(job);
                                }
                            }
                        }
                    }
                    if batch.is_empty() {
                        if disconnected && pending.is_none() {
                            break;
                        }
                        continue;
                    }
                    let step_context = CooperativeStepContext {
                        active_slots: batch.scheduled_len(),
                    };
                    worker_occupied.store(step_context.active_slots, Ordering::Relaxed);
                    let completed = batch.run_prioritized_grouped_round_n(
                        |_, job| !job.is_prefill_pending(),
                        |_, job| job.compatibility_key(),
                        |_, job| {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                job.step(step_context)
                            }))
                            .unwrap_or(StepOutcome::Complete)
                        },
                        |_, jobs| {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                ScheduledCooperativeJob::step_group(jobs, step_context)
                            }))
                            .unwrap_or_else(|_| Some(vec![StepOutcome::Complete; jobs.len()]))
                        },
                    );
                    // Republish after the round so a finished stream frees its
                    // slot immediately rather than at the next admit.
                    worker_occupied.store(batch.active_len(), Ordering::Relaxed);
                    if !completed.is_empty() {
                        worker_depth.fetch_sub(completed.len(), Ordering::SeqCst);
                    }
                }
            })
            .expect("spawn camelid-engine worker thread");
        Self {
            tx,
            depth,
            active_task_id,
            active_started_epoch_millis,
            active_last_progress_epoch_millis,
            active_completed_units,
            next_task_id: Arc::new(AtomicU64::new(1)),
            continuous_batch_slots,
            admitted_slots,
            occupied_slots,
            exclusive_active,
            cooperative_slot_states,
        }
    }

    /// Jobs accepted and not yet finished.
    pub(crate) fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Configured cooperative streaming capacity captured when the worker starts.
    pub(crate) fn continuous_batch_slots(&self) -> usize {
        self.continuous_batch_slots
    }

    /// Streaming slots this engine will actually admit right now.
    ///
    /// Not the same as [`continuous_batch_slots`](Self::continuous_batch_slots):
    /// `stream_completion` only builds a cooperative job when the CUDA resident
    /// engine is NOT driving decode, because that engine is a process-global slot
    /// keyed by model id. On such a deployment every stream runs exclusive, so
    /// advertising two slots would invite a client to dispatch against capacity
    /// that does not exist.
    pub(crate) fn total_slots(&self) -> usize {
        self.admitted_slots
    }

    /// Slots that cannot accept a new stream right now, as a usable
    /// `busy / total` pair against [`total_slots`](Self::total_slots).
    ///
    /// An exclusive job saturates: it owns the entire engine while it runs, so
    /// no slot can start a token until it finishes. Counting it as ONE busy slot
    /// would tell a capacity-aware client to dispatch into a slot that cannot
    /// run — the same false-capacity failure this pair exists to prevent.
    pub(crate) fn busy_slots(&self) -> usize {
        let total = self.total_slots();
        if self.exclusive_active.load(Ordering::Relaxed) > 0 {
            return total;
        }
        self.occupied_slots.load(Ordering::Relaxed).min(total)
    }

    /// Privacy-safe, read-only state for the production engine's real slot.
    /// Queue depth remains separate because queued jobs do not own a slot.
    pub(crate) fn slot_snapshot(&self) -> EngineSlotSnapshot {
        let active = self.active_task_id.load(Ordering::SeqCst);
        let depth = self.depth();
        let active_jobs = if self.exclusive_active.load(Ordering::Relaxed) > 0 {
            1
        } else {
            self.occupied_slots.load(Ordering::Relaxed)
        };
        let now = epoch_millis();
        let started = self.active_started_epoch_millis.load(Ordering::SeqCst);
        let last_progress = self
            .active_last_progress_epoch_millis
            .load(Ordering::SeqCst);
        EngineSlotSnapshot {
            active_task_id: (active != 0).then_some(active),
            queued_tasks: depth.saturating_sub(active_jobs),
            completed_units: self.active_completed_units.load(Ordering::SeqCst),
            active_elapsed_seconds: if active == 0 {
                0
            } else {
                now.saturating_sub(started) / 1_000
            },
            stalled_seconds: if active == 0 {
                0
            } else {
                now.saturating_sub(last_progress.max(started)) / 1_000
            },
        }
    }

    pub(crate) fn cooperative_slot_snapshots(&self) -> Vec<EngineCooperativeSlotSnapshot> {
        let now = epoch_millis();
        self.cooperative_slot_states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .enumerate()
            .map(|(id, slot)| {
                let active_task_id = (slot.task_id != 0).then_some(slot.task_id);
                EngineCooperativeSlotSnapshot {
                    id,
                    active_task_id,
                    completed_units: slot.completed_units,
                    prefill_completed_units: slot.prefill_completed_units,
                    prefill_total_units: slot.prefill_total_units,
                    active_elapsed_seconds: active_task_id
                        .map(|_| now.saturating_sub(slot.started_epoch_millis) / 1_000)
                        .unwrap_or(0),
                    stalled_seconds: active_task_id
                        .map(|_| {
                            now.saturating_sub(
                                slot.last_progress_epoch_millis
                                    .max(slot.started_epoch_millis),
                            ) / 1_000
                        })
                        .unwrap_or(0),
                }
            })
            .collect()
    }

    pub(crate) fn cooperative_prefill_progress(&self) -> (u64, u64) {
        self.cooperative_slot_states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .fold((0u64, 0u64), |(completed, total), slot| {
                (
                    completed.saturating_add(slot.prefill_completed_units),
                    total.saturating_add(slot.prefill_total_units),
                )
            })
    }

    /// Report monotonic unit progress from the currently executing engine job.
    /// Generation uses decoded tokens as units. This is deliberately a handful
    /// of relaxed atomics outside the numerical path, so watchdog observation
    /// cannot change model output.
    pub(crate) fn record_progress(&self, completed_units: usize) {
        if self.active_task_id.load(Ordering::Relaxed) == 0 {
            return;
        }
        self.active_completed_units.store(
            completed_units.try_into().unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.active_last_progress_epoch_millis
            .store(epoch_millis(), Ordering::Relaxed);
    }

    /// Post a job without waiting for queue room: full queue is an explicit,
    /// typed condition (503 + Retry-After at the HTTP layer), never an
    /// invisible pile of waiters.
    pub(crate) fn post(&self, task: EngineTask) -> Result<(), EnginePostError> {
        let task_id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        let active_task_id = Arc::clone(&self.active_task_id);
        let active_started_epoch_millis = Arc::clone(&self.active_started_epoch_millis);
        let active_last_progress_epoch_millis = Arc::clone(&self.active_last_progress_epoch_millis);
        let active_completed_units = Arc::clone(&self.active_completed_units);
        let task = match task {
            EngineTask::Exclusive(job) => QueuedEngineTask::Exclusive(Box::new(move || {
                let started = epoch_millis();
                active_started_epoch_millis.store(started, Ordering::SeqCst);
                active_last_progress_epoch_millis.store(started, Ordering::SeqCst);
                active_completed_units.store(0, Ordering::SeqCst);
                // Publish the active id last so readers never observe an
                // active job paired with uninitialized timestamps.
                active_task_id.store(task_id, Ordering::SeqCst);
                let _guard = ActiveTaskGuard {
                    active_task_id,
                    active_started_epoch_millis,
                    active_last_progress_epoch_millis,
                    active_completed_units,
                };
                job();
            })),
            EngineTask::Cooperative(job) => QueuedEngineTask::Cooperative {
                task_id,
                kind: CooperativeJobKind::Scalar(job),
            },
            EngineTask::BatchableCooperative(job) => QueuedEngineTask::Cooperative {
                task_id,
                kind: CooperativeJobKind::Batchable(job),
            },
        };
        self.depth.fetch_add(1, Ordering::SeqCst);
        match self.tx.try_send(task) {
            Ok(()) => Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(EnginePostError::QueueFull)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(EnginePostError::Unavailable)
            }
        }
    }

    /// Run a blocking job on the engine thread and await its typed result.
    ///
    /// If the calling frame is dropped while waiting (client disconnect), the
    /// job still runs to completion on the engine thread — cancellation is
    /// signalled separately via `GenerationCancel`/`CancelOnDrop`, which the
    /// decode loops observe per step. The job's result is then discarded with
    /// the closed oneshot.
    pub(crate) async fn run_exclusive<T, F>(&self, job: F) -> Result<T, EnginePostError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        self.post(EngineTask::Exclusive(Box::new(move || {
            let _ = result_tx.send(job());
        })))?;
        result_rx.await.map_err(|_| EnginePostError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBatchableJob {
        key: CooperativeBatchKey,
        label: char,
        accept_pair: bool,
        prefill_pending: bool,
        calls: Arc<std::sync::Mutex<Vec<String>>>,
    }

    struct TestGroupJob {
        key: CooperativeBatchKey,
        label: char,
        calls: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl BatchableCooperativeJob for TestGroupJob {
        fn compatibility_key(&mut self) -> Option<CooperativeBatchKey> {
            Some(self.key)
        }

        fn step(&mut self, _context: CooperativeStepContext) -> StepOutcome {
            self.calls.lock().unwrap().push(self.label.to_string());
            StepOutcome::Complete
        }

        fn step_pair(
            &mut self,
            _other: &mut dyn BatchableCooperativeJob,
            _context: CooperativeStepContext,
        ) -> Option<[StepOutcome; 2]> {
            None
        }

        fn step_group(
            &mut self,
            others: &mut [&mut dyn BatchableCooperativeJob],
            _context: CooperativeStepContext,
        ) -> Option<Vec<StepOutcome>> {
            let mut labels = String::from(self.label);
            for other in &mut *others {
                labels.push(other.as_any_mut().downcast_mut::<Self>()?.label);
            }
            self.calls.lock().unwrap().push(labels);
            Some(vec![StepOutcome::Complete; others.len() + 1])
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl BatchableCooperativeJob for TestBatchableJob {
        fn compatibility_key(&mut self) -> Option<CooperativeBatchKey> {
            Some(self.key)
        }

        fn step(&mut self, _context: CooperativeStepContext) -> StepOutcome {
            self.calls.lock().unwrap().push(self.label.to_string());
            StepOutcome::Complete
        }

        fn step_pair(
            &mut self,
            other: &mut dyn BatchableCooperativeJob,
            _context: CooperativeStepContext,
        ) -> Option<[StepOutcome; 2]> {
            let other = other.as_any_mut().downcast_mut::<Self>()?;
            if !self.accept_pair || !other.accept_pair {
                return None;
            }
            self.calls
                .lock()
                .unwrap()
                .push(format!("{}{}", self.label, other.label));
            Some([StepOutcome::Complete, StepOutcome::Complete])
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn is_prefill_pending(&self) -> bool {
            self.prefill_pending
        }
    }

    async fn run_two_batchable_jobs(accept_pair: bool) -> Vec<String> {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        for label in ['a', 'b'] {
            engine
                .post(EngineTask::BatchableCooperative(Box::new(
                    TestBatchableJob {
                        key: CooperativeBatchKey::new(7, 11),
                        label,
                        accept_pair,
                        prefill_pending: false,
                        calls: Arc::clone(&calls),
                    },
                )))
                .unwrap();
        }
        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        Arc::try_unwrap(calls).unwrap().into_inner().unwrap()
    }

    #[tokio::test]
    async fn compatible_batchable_jobs_share_one_dispatch() {
        assert_eq!(run_two_batchable_jobs(true).await, vec!["ab"]);
    }

    #[tokio::test]
    async fn declined_batchable_pair_falls_back_in_fifo_order() {
        assert_eq!(run_two_batchable_jobs(false).await, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn phase7_four_compatible_jobs_receive_one_group_dispatch() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "4");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        for label in ['a', 'b', 'c', 'd'] {
            engine
                .post(EngineTask::BatchableCooperative(Box::new(TestGroupJob {
                    key: CooperativeBatchKey::new(7, 11),
                    label,
                    calls: Arc::clone(&calls),
                })))
                .unwrap();
        }
        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(*calls.lock().unwrap(), vec!["abcd"]);
    }

    /// The real invariant D5's lock test could not prove: the ENGINE executes
    /// at most one job at a time by construction, measured on the compute
    /// itself rather than on guard lifetimes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn engine_executes_at_most_one_job_at_a_time() {
        let engine = EngineHandle::spawn();
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        for _ in 0..24 {
            // Post with retry: the bounded queue is allowed to be full — the
            // invariant under test is serialization, not capacity.
            loop {
                let active = Arc::clone(&active);
                let max_seen = Arc::clone(&max_seen);
                match engine
                    .run_exclusive(move || {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_seen.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        active.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await
                {
                    Ok(()) => break,
                    Err(EnginePostError::QueueFull) => {
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    }
                    Err(EnginePostError::Unavailable) => panic!("engine gone"),
                }
            }
        }

        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "engine must never run two jobs concurrently",
        );
    }

    #[tokio::test]
    async fn cooperative_jobs_interleave_one_step_per_round() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        // Hold the worker so both cooperative jobs are queued before the first round.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        for label in ['a', 'b'] {
            let order = Arc::clone(&order);
            let mut steps = 0usize;
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    order.lock().unwrap().push(label);
                    steps += 1;
                    if steps == 3 {
                        StepOutcome::Complete
                    } else {
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }
        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(*order.lock().unwrap(), vec!['a', 'b', 'a', 'b', 'a', 'b']);
    }

    #[tokio::test]
    async fn panicking_cooperative_job_drops_owned_state_once() {
        struct DropProbe(Arc<AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let engine = EngineHandle::spawn();
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = DropProbe(Arc::clone(&drops));
        engine
            .post(EngineTask::Cooperative(Box::new(move |_| {
                let _keep_owned_until_drop = &probe;
                panic!("intentional cooperative panic containment probe")
            })))
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cooperative_context_returns_to_single_stream_fast_path() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        // Hold the worker until both jobs are waiting so the first round is contended.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        {
            let seen = Arc::clone(&seen);
            let mut steps = 0usize;
            engine
                .post(EngineTask::Cooperative(Box::new(move |context| {
                    seen.lock().unwrap().push(('a', context.active_slots));
                    steps += 1;
                    if steps == 2 {
                        StepOutcome::Complete
                    } else {
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }
        {
            let seen = Arc::clone(&seen);
            engine
                .post(EngineTask::Cooperative(Box::new(move |context| {
                    seen.lock().unwrap().push(('b', context.active_slots));
                    StepOutcome::Complete
                })))
                .unwrap();
        }
        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        assert_eq!(
            *seen.lock().unwrap(),
            vec![('a', 2), ('b', 2), ('a', 1)],
            "after contention clears, the remaining stream regains encode-ahead eligibility"
        );
    }

    #[tokio::test]
    async fn cooperative_slots_publish_distinct_task_progress() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let stop = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let stop = Arc::clone(&stop);
            let progress = engine.clone();
            let mut steps = 0usize;
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    steps += 1;
                    progress.record_progress(steps);
                    if stop.load(Ordering::SeqCst) != 0 {
                        StepOutcome::Complete
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let live = loop {
            let slots = engine.cooperative_slot_snapshots();
            if slots
                .iter()
                .filter(|slot| slot.active_task_id.is_some())
                .count()
                == 2
                && slots.iter().all(|slot| slot.completed_units > 0)
            {
                break slots;
            }
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        };
        let ids = live
            .iter()
            .map(|slot| slot.active_task_id.expect("both slots are active"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 2, "each live stream has its own task identity");

        stop.store(1, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert!(
            engine
                .cooperative_slot_snapshots()
                .iter()
                .all(|slot| slot.active_task_id.is_none()),
            "completed streams release their public slots"
        );
    }

    struct TestPrefillProgressJob {
        chunks_done: u64,
        chunks_total: u64,
        second_step_started: std::sync::mpsc::Sender<()>,
        release_second_step: std::sync::mpsc::Receiver<()>,
    }

    impl BatchableCooperativeJob for TestPrefillProgressJob {
        fn compatibility_key(&mut self) -> Option<CooperativeBatchKey> {
            None
        }

        fn step(&mut self, _context: CooperativeStepContext) -> StepOutcome {
            self.chunks_done += 1;
            if self.chunks_done == 2 {
                self.second_step_started.send(()).unwrap();
                self.release_second_step.recv().unwrap();
                StepOutcome::Complete
            } else {
                StepOutcome::Continue
            }
        }

        fn step_pair(
            &mut self,
            _other: &mut dyn BatchableCooperativeJob,
            _context: CooperativeStepContext,
        ) -> Option<[StepOutcome; 2]> {
            None
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn prefill_progress(&self) -> Option<(u64, u64)> {
            Some((self.chunks_done, self.chunks_total))
        }

        fn is_prefill_pending(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn cooperative_slot_reports_prefill_progress_without_generated_tokens() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "1");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::BatchableCooperative(Box::new(
                TestPrefillProgressJob {
                    chunks_done: 0,
                    chunks_total: 3,
                    second_step_started: started_tx,
                    release_second_step: release_rx,
                },
            )))
            .unwrap();
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("second prefill step starts");

        let slot = engine.cooperative_slot_snapshots()[0];
        assert!(slot.active_task_id.is_some());
        assert_eq!(slot.completed_units, 0, "prefill is not generated output");
        assert_eq!(slot.prefill_completed_units, 1);
        assert_eq!(slot.prefill_total_units, 3);
        assert_eq!(engine.cooperative_prefill_progress(), (1, 3));

        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        let slot = engine.cooperative_slot_snapshots()[0];
        assert_eq!(slot.active_task_id, None);
        assert_eq!(slot.prefill_completed_units, 0);
        assert_eq!(slot.prefill_total_units, 0);
        assert_eq!(engine.cooperative_prefill_progress(), (0, 0));
    }

    struct TestPrefillThenDecodeProgressJob {
        step: usize,
        second_decode_started: std::sync::mpsc::Sender<()>,
        release_second_decode: std::sync::mpsc::Receiver<()>,
    }

    impl BatchableCooperativeJob for TestPrefillThenDecodeProgressJob {
        fn compatibility_key(&mut self) -> Option<CooperativeBatchKey> {
            None
        }

        fn step(&mut self, _context: CooperativeStepContext) -> StepOutcome {
            self.step += 1;
            if self.step == 3 {
                self.second_decode_started.send(()).unwrap();
                self.release_second_decode.recv().unwrap();
                StepOutcome::Complete
            } else {
                StepOutcome::Continue
            }
        }

        fn step_pair(
            &mut self,
            _other: &mut dyn BatchableCooperativeJob,
            _context: CooperativeStepContext,
        ) -> Option<[StepOutcome; 2]> {
            None
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn completed_units(&self) -> Option<u64> {
            Some(self.step.saturating_sub(1) as u64)
        }

        fn prefill_progress(&self) -> Option<(u64, u64)> {
            (self.step < 2).then_some((4, 4))
        }
    }

    #[tokio::test]
    async fn finalized_prefill_progress_clears_when_decode_completes() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "1");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let (decode_tx, decode_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        engine
            .post(EngineTask::BatchableCooperative(Box::new(
                TestPrefillThenDecodeProgressJob {
                    step: 0,
                    second_decode_started: decode_tx,
                    release_second_decode: release_rx,
                },
            )))
            .unwrap();
        decode_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("second decode step starts after one generated token");
        let slot = engine.cooperative_slot_snapshots()[0];
        assert!(slot.active_task_id.is_some());
        assert_eq!(slot.completed_units, 1);
        assert_eq!(slot.prefill_completed_units, 0);
        assert_eq!(slot.prefill_total_units, 0);
        assert_eq!(engine.cooperative_prefill_progress(), (0, 0));

        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(engine.cooperative_prefill_progress(), (0, 0));
    }

    /// A cooperative job owns its slot for its whole life, not for one token.
    /// Re-stamping the start time (or zeroing the unit counter) at every step
    /// pins `active_elapsed_seconds`/`stalled_seconds` at 0 and blinds the
    /// stall watchdog for every streaming request.
    #[tokio::test]
    async fn cooperative_slot_state_survives_token_boundaries() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        {
            let observed = Arc::clone(&observed);
            let progress = engine.clone();
            let mut steps = 0u64;
            let release_rx = std::sync::Mutex::new(release_rx);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    steps += 1;
                    progress.record_progress(steps as usize);
                    // Observe the slot the way /health does, from inside the
                    // step but after progress was reported.
                    let slot = progress.slot_snapshot();
                    observed
                        .lock()
                        .unwrap()
                        .push((slot.active_task_id, slot.completed_units));
                    if steps == 3 {
                        // Hold the last step open long enough that the wall
                        // clock crosses a whole second, so elapsed is provable.
                        release_rx.lock().unwrap().recv().ok();
                        StepOutcome::Complete
                    } else {
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }
        // Let the job reach its third step, then check the slot from OUTSIDE.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while observed.lock().unwrap().len() < 3 {
            assert!(
                std::time::Instant::now() < deadline,
                "job never reached step 3"
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        let mid = engine.slot_snapshot();
        assert!(mid.is_processing(), "a yielding stream still owns its slot");
        assert_eq!(mid.completed_units, 3, "token progress survives the yield");
        assert!(
            mid.active_elapsed_seconds >= 1,
            "elapsed must accumulate across token steps, got {}",
            mid.active_elapsed_seconds
        );
        release_tx.send(()).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        let observed = observed.lock().unwrap();
        assert_eq!(
            observed.iter().map(|(_, units)| *units).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "completed units are cumulative, not reset per step"
        );
        let ids: Vec<_> = observed.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.iter().all(|id| *id == ids[0] && id.is_some()),
            "the active task id is stable across yields: {ids:?}"
        );
        assert_eq!(engine.slot_snapshot().active_task_id, None, "slot released");
    }

    /// `/slots`, `/props.total_slots` and `fail_on_no_slot` all arbitrate against
    /// `busy_slots` / `total_slots`, so a second stream must be admissible while
    /// the first is mid-generation, and an exclusive job must count as busy
    /// because it owns the whole engine while it runs.
    #[tokio::test]
    async fn slot_occupancy_tracks_cooperative_and_exclusive_work() {
        let _env_guard = crate::test_support::env_lock();
        // This test asserts the COOPERATIVE capacity contract. On a box with a
        // usable CUDA device, `total_slots()` truthfully reports 1 (every stream
        // runs exclusive on the GPU-resident lane), so pin the CPU lane the test
        // was written for; CI runners have no GPU and are unaffected.
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

        assert_eq!(engine.total_slots(), 2);
        assert_eq!(engine.busy_slots(), 0, "idle engine has every slot free");

        // An exclusive job ALONE must saturate: it owns the whole engine, so
        // reporting one free slot would invite a dispatch that cannot run.
        // (Checked before any stream exists, or `1 cooperative + 1 exclusive`
        // would reach `total` by arithmetic accident.)
        let (solo_tx, solo_rx) = std::sync::mpsc::channel::<()>();
        let (solo_release_tx, solo_release_rx) = std::sync::mpsc::channel::<()>();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                solo_tx.send(()).unwrap();
                solo_release_rx.recv().ok();
            })))
            .unwrap();
        solo_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("exclusive job starts");
        assert_eq!(
            engine.busy_slots(),
            engine.total_slots(),
            "a lone exclusive job blocks every slot"
        );
        solo_release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.busy_slots() != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "an exclusive job must release capacity when it returns"
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let stop_rx = Arc::new(std::sync::Mutex::new(stop_rx));
        let running = Arc::new(AtomicUsize::new(0));
        {
            let stop_rx = Arc::clone(&stop_rx);
            let running = Arc::clone(&running);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    running.fetch_add(1, Ordering::SeqCst);
                    if stop_rx.lock().unwrap().try_recv().is_ok() {
                        StepOutcome::Complete
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while running.load(Ordering::SeqCst) < 2 {
            assert!(std::time::Instant::now() < deadline, "stream never started");
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(engine.busy_slots(), 1, "one stream occupies one slot");
        assert!(
            engine.busy_slots() < engine.total_slots(),
            "a second stream is still admissible"
        );

        // An exclusive job owns the engine, so capacity must read as saturated.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx.recv().ok();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("exclusive job runs even with a stream active");
        assert_eq!(
            engine.busy_slots(),
            engine.total_slots(),
            "an exclusive job blocks every slot while it runs"
        );
        release_tx.send(()).unwrap();

        let _ = stop_tx.send(());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.busy_slots() != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "slots must free when work finishes"
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");
    }

    #[test]
    fn phase7_cuda_slot_capacity_respects_both_configured_and_safe_limits() {
        assert_eq!(effective_slot_capacity(8, false, 1), 8);
        assert_eq!(effective_slot_capacity(8, true, 1), 1);
        assert_eq!(effective_slot_capacity(1, true, 8), 1);
        assert_eq!(effective_slot_capacity(2, true, 4), 2);
        assert_eq!(effective_slot_capacity(8, true, 4), 4);
        assert_eq!(effective_slot_capacity(8, true, 8), 8);
    }

    /// The bounded channel is the backpressure device. A worker that drains it
    /// into an unbounded local queue makes `QueueFull` unreachable for as long
    /// as any stream is running, which is exactly when the server most needs to
    /// answer 503 instead of parking waiters.
    #[tokio::test]
    async fn queue_full_still_fires_while_a_cooperative_job_runs() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "1");
        std::env::set_var(QUEUE_DEPTH_ENV, "1");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var(QUEUE_DEPTH_ENV);

        // One never-ending cooperative job occupies the single slot.
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let running = Arc::new(AtomicUsize::new(0));
        {
            let running = Arc::clone(&running);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    running.fetch_add(1, Ordering::SeqCst);
                    if stop_rx.try_recv().is_ok() {
                        StepOutcome::Complete
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        StepOutcome::Continue
                    }
                })))
                .expect("first post fits an idle engine");
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while running.load(Ordering::SeqCst) < 2 {
            assert!(std::time::Instant::now() < deadline, "job never started");
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        // Depth-1 channel: one more post fits, the next must be refused rather
        // than silently absorbed into a local queue.
        engine
            .post(EngineTask::Exclusive(Box::new(|| {})))
            .expect("one queued job fits the depth-1 channel");
        let mut refusals = 0;
        for _ in 0..8 {
            if matches!(
                engine.post(EngineTask::Exclusive(Box::new(|| {}))),
                Err(EnginePostError::QueueFull)
            ) {
                refusals += 1;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert!(
            refusals > 0,
            "the bounded queue must still refuse posts while a stream is active"
        );
        let _ = stop_tx.send(());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }

    /// Exclusive work (model load/unload, non-streaming completions, the parity
    /// probe, resident-cache resets) must not wait for the streaming batch to
    /// drain: with overlapping streams that moment may never come.
    #[tokio::test]
    async fn exclusive_work_runs_while_cooperative_streams_are_active() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let stop_rx = Arc::new(std::sync::Mutex::new(stop_rx));
        for _ in 0..2 {
            let stop_rx = Arc::clone(&stop_rx);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    if stop_rx.lock().unwrap().try_recv().is_ok() {
                        StepOutcome::Complete
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }

        let (ran_tx, ran_rx) = std::sync::mpsc::channel::<()>();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                let _ = ran_tx.send(());
            })))
            .unwrap();
        ran_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("exclusive job must run without waiting for the streams to finish");

        let _ = stop_tx.send(());
        let _ = stop_tx.send(());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }

    #[tokio::test]
    async fn full_queue_is_a_typed_error_and_depth_recovers() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(QUEUE_DEPTH_ENV, "1");
        let engine = EngineHandle::spawn();
        std::env::remove_var(QUEUE_DEPTH_ENV);

        // Occupy the worker and wait until the job is RUNNING, so the queue
        // itself (capacity 1) is empty again.
        let entered = Arc::new(AtomicUsize::new(0));
        let (block_tx, block_rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        {
            let entered = Arc::clone(&entered);
            engine
                .post(EngineTask::Exclusive(Box::new(move || {
                    entered.store(1, Ordering::SeqCst);
                    block_rx.recv().ok();
                    let _ = done_tx.send(());
                })))
                .expect("first post fits an idle engine");
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while entered.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "worker never started the blocking job",
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        // One queued job fits (capacity 1); the next must be typed QueueFull.
        assert!(engine.post(EngineTask::Exclusive(Box::new(|| {}))).is_ok());
        assert_eq!(
            engine.post(EngineTask::Exclusive(Box::new(|| {}))),
            Err(EnginePostError::QueueFull),
        );

        block_tx.send(()).unwrap();
        done_rx.await.expect("blocking job completes");
        // Drain: depth returns to zero once the accepted jobs finish.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline, "depth never drained");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn slot_snapshot_tracks_real_active_task() {
        let engine = EngineHandle::spawn();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let progress = engine.clone();
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
                progress.record_progress(7);
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })))
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("engine starts task");

        let busy = engine.slot_snapshot();
        assert!(busy.is_processing());
        assert!(busy.active_task_id.is_some());
        assert_eq!(busy.queued_tasks, 0);
        assert_eq!(busy.completed_units, 7);

        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(
            engine.slot_snapshot(),
            EngineSlotSnapshot {
                active_task_id: None,
                queued_tasks: 0,
                completed_units: 0,
                active_elapsed_seconds: 0,
                stalled_seconds: 0,
            }
        );
    }
}
