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

use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
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

type CooperativeJob = Box<dyn FnMut(CooperativeStepContext) -> StepOutcome + Send + 'static>;

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
    Cooperative(CooperativeJob),
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
    tx: tokio::sync::mpsc::Sender<EngineTask>,
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
    /// Monotonic low-memory admission mode. Once enabled, retained
    /// cooperative sessions and exclusive jobs are drained one at a time so
    /// only one task can own populated KV state.
    single_kv_owner_mode: Arc<AtomicBool>,
    /// Cooperative slots the worker currently holds. Published by the worker
    /// itself, which already computes it for `CooperativeStepContext`.
    occupied_slots: Arc<AtomicUsize>,
    /// 1 while an exclusive job owns the engine. It is not a slot, but it does
    /// block every slot, so capacity questions must count it.
    exclusive_active: Arc<AtomicUsize>,
}

fn queue_depth_from_env() -> usize {
    crate::runtime_config::engine_queue_depth()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineSlotSnapshot {
    pub(crate) active_task_id: Option<u64>,
    pub(crate) queued_tasks: usize,
    pub(crate) completed_units: u64,
    pub(crate) active_elapsed_seconds: u64,
    pub(crate) stalled_seconds: u64,
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
    /// Held, never read: dropping it clears the shared slot atomics when the
    /// job completes or panics.
    #[allow(dead_code)]
    guard: ActiveTaskGuard,
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

fn run_exclusive_job(job: ExclusiveJob, exclusive_active: &AtomicUsize, depth: &AtomicUsize) {
    exclusive_active.store(1, Ordering::Relaxed);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    exclusive_active.store(0, Ordering::Relaxed);
    depth.fetch_sub(1, Ordering::SeqCst);
}

impl EngineHandle {
    /// Spawn the engine worker thread and return the posting handle.
    pub(crate) fn spawn() -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineTask>(queue_depth_from_env());
        // Resolve configuration on the spawning thread. Reading it lazily
        // inside the worker races tests and embedders that scope environment
        // overrides around `spawn()`.
        let continuous_batch_slots = crate::runtime_config::continuous_batch_slots();
        let depth = Arc::new(AtomicUsize::new(0));
        let worker_depth = Arc::clone(&depth);
        let active_task_id = Arc::new(AtomicU64::new(0));
        let active_started_epoch_millis = Arc::new(AtomicU64::new(0));
        let active_last_progress_epoch_millis = Arc::new(AtomicU64::new(0));
        let active_completed_units = Arc::new(AtomicU64::new(0));
        let occupied_slots = Arc::new(AtomicUsize::new(0));
        let exclusive_active = Arc::new(AtomicUsize::new(0));
        let single_kv_owner_mode = Arc::new(AtomicBool::new(false));
        let worker_occupied = Arc::clone(&occupied_slots);
        let worker_exclusive = Arc::clone(&exclusive_active);
        let worker_single_kv_owner = Arc::clone(&single_kv_owner_mode);
        std::thread::Builder::new()
            .name("camelid-engine".to_string())
            .spawn(move || {
                let mut batch = ContinuousBatch::<CooperativeJob>::new(continuous_batch_slots);
                // At most ONE task is ever held outside the channel: work that
                // cannot run against the KV owners currently retained by the
                // batch. Draining
                // the channel into an unbounded local queue instead would make
                // `try_send` never report `Full`, and the typed `QueueFull` ->
                // 503 backpressure would silently stop existing for as long as
                // any stream was running.
                let mut pending: Option<EngineTask> = None;
                let mut disconnected = false;
                loop {
                    // Held-back work takes the first safe opening, ahead of
                    // anything still in the channel. In low-memory mode an
                    // exclusive job must also wait for retained cooperative KV
                    // to drain: serialized compute alone does not release that
                    // memory between token steps.
                    if let Some(task) = pending.take() {
                        let single_owner = worker_single_kv_owner.load(Ordering::Acquire);
                        match task {
                            EngineTask::Exclusive(job) if !single_owner || batch.is_empty() => {
                                run_exclusive_job(
                                    job,
                                    worker_exclusive.as_ref(),
                                    worker_depth.as_ref(),
                                );
                            }
                            EngineTask::Exclusive(job) => {
                                pending = Some(EngineTask::Exclusive(job));
                            }
                            EngineTask::Cooperative(job)
                                if if single_owner {
                                    batch.is_empty()
                                } else {
                                    batch.has_free_slot()
                                } =>
                            {
                                batch.admit(job);
                            }
                            EngineTask::Cooperative(job) => {
                                pending = Some(EngineTask::Cooperative(job));
                            }
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
                        let single_owner = worker_single_kv_owner.load(Ordering::Acquire);
                        match task {
                            // Normally exclusive work runs as soon as it is
                            // picked up, avoiding starvation behind overlapping
                            // streams. Low-memory mode deliberately trades that
                            // concurrency for a single populated KV owner.
                            EngineTask::Exclusive(job) if !single_owner || batch.is_empty() => {
                                run_exclusive_job(
                                    job,
                                    worker_exclusive.as_ref(),
                                    worker_depth.as_ref(),
                                );
                            }
                            EngineTask::Exclusive(job) => {
                                pending = Some(EngineTask::Exclusive(job));
                            }
                            EngineTask::Cooperative(job) => {
                                if if single_owner {
                                    batch.is_empty()
                                } else {
                                    batch.has_free_slot()
                                } {
                                    batch.admit(job);
                                } else {
                                    pending = Some(EngineTask::Cooperative(job));
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
                    let completed = batch.run_round(|_, job| {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(step_context)))
                            .unwrap_or(StepOutcome::Complete)
                    });
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
            single_kv_owner_mode,
            occupied_slots,
            exclusive_active,
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

    /// Permanently serialize KV-owning engine work for this process. The
    /// transition is intentionally one-way: callers may have already prepared
    /// sessions under the stricter mode, so widening again without a global
    /// drain would make the aggregate-memory guarantee racy.
    pub(crate) fn enable_single_kv_owner_mode(&self) {
        self.single_kv_owner_mode.store(true, Ordering::Release);
    }

    pub(crate) fn single_kv_owner_mode(&self) -> bool {
        self.single_kv_owner_mode.load(Ordering::Acquire)
    }

    /// Streaming slots this engine will actually admit right now.
    ///
    /// Not the same as [`continuous_batch_slots`](Self::continuous_batch_slots):
    /// `stream_completion` only builds a cooperative job when the CUDA resident
    /// engine is NOT driving decode, because that engine is a process-global slot
    /// keyed by model id. On such a deployment every stream runs exclusive, so
    /// advertising two slots would invite a client to dispatch against capacity
    /// that does not exist. Low-memory mode keeps the cooperative task shape but
    /// admits only one such owner, so it reports one slot here as well.
    pub(crate) fn total_slots(&self) -> usize {
        if crate::inference::resident_decode_cuda_active() || self.single_kv_owner_mode() {
            1
        } else {
            self.continuous_batch_slots
        }
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
        let now = epoch_millis();
        let started = self.active_started_epoch_millis.load(Ordering::SeqCst);
        let last_progress = self
            .active_last_progress_epoch_millis
            .load(Ordering::SeqCst);
        EngineSlotSnapshot {
            active_task_id: (active != 0).then_some(active),
            queued_tasks: depth.saturating_sub(usize::from(active != 0)),
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
            EngineTask::Exclusive(job) => EngineTask::Exclusive(Box::new(move || {
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
            EngineTask::Cooperative(mut job) => {
                // A cooperative job spans many steps, so its slot state must
                // span them too: stamping `started` (and zeroing the unit
                // counter) on every token would pin `active_elapsed_seconds`
                // and `stalled_seconds` at 0 for the whole generation and blind
                // the stall watchdog. The guard is created once, on the first
                // step, and lives inside this closure — the scheduler drops the
                // closure when the job completes OR panics, so the RAII clear
                // still happens on every exit path.
                //
                // The four atomics describe "the job running right now": with
                // more than one slot the jobs take turns owning them, so each
                // step republishes this job's own start time and last observed
                // progress before handing control to the decode step.
                let mut slot: Option<CooperativeSlotState> = None;
                EngineTask::Cooperative(Box::new(move |context| {
                    let now = epoch_millis();
                    let state = slot.get_or_insert_with(|| CooperativeSlotState {
                        started: now,
                        last_progress: now,
                        completed_units: 0,
                        guard: ActiveTaskGuard {
                            active_task_id: Arc::clone(&active_task_id),
                            active_started_epoch_millis: Arc::clone(&active_started_epoch_millis),
                            active_last_progress_epoch_millis: Arc::clone(
                                &active_last_progress_epoch_millis,
                            ),
                            active_completed_units: Arc::clone(&active_completed_units),
                        },
                    });
                    active_started_epoch_millis.store(state.started, Ordering::SeqCst);
                    active_last_progress_epoch_millis.store(state.last_progress, Ordering::SeqCst);
                    active_completed_units.store(state.completed_units, Ordering::SeqCst);
                    // Published last so a reader never sees an active id paired
                    // with another job's timestamps.
                    active_task_id.store(task_id, Ordering::SeqCst);
                    let outcome = job(context);
                    // Carry this job's progress across the yield: the next slot
                    // to run overwrites the shared atomics.
                    state.completed_units = active_completed_units.load(Ordering::SeqCst);
                    state.last_progress = active_last_progress_epoch_millis.load(Ordering::SeqCst);
                    if outcome == StepOutcome::Complete {
                        slot = None;
                    }
                    outcome
                }))
            }
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
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

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
    async fn chunked_prefill_does_not_starve_a_short_stream() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

        // Queue both streams before scheduling starts. `p` models one long
        // prompt exposing four prefill chunks; `s` models a short request that
        // can produce its first token immediately.
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
        {
            let order = Arc::clone(&order);
            let mut chunks = 0usize;
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    chunks += 1;
                    order.lock().unwrap().push('p');
                    if chunks == 4 {
                        StepOutcome::Complete
                    } else {
                        StepOutcome::Continue
                    }
                })))
                .unwrap();
        }
        {
            let order = Arc::clone(&order);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    order.lock().unwrap().push('s');
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
            *order.lock().unwrap(),
            vec!['p', 's', 'p', 'p', 'p'],
            "the short stream must run after one prefill chunk, not after the long prompt"
        );
    }

    #[tokio::test]
    async fn single_kv_owner_mode_drains_tasks_without_retained_overlap() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

        // Hold the worker until every relevant task is queued. The mode switch
        // must affect already-accepted work, not only jobs posted afterward.
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
        {
            let order = Arc::clone(&order);
            let mut steps = 0usize;
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    order.lock().unwrap().push('a');
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
            let order = Arc::clone(&order);
            engine
                .post(EngineTask::Exclusive(Box::new(move || {
                    order.lock().unwrap().push('x');
                })))
                .unwrap();
        }
        {
            let order = Arc::clone(&order);
            engine
                .post(EngineTask::Cooperative(Box::new(move |_| {
                    order.lock().unwrap().push('b');
                    StepOutcome::Complete
                })))
                .unwrap();
        }

        engine.enable_single_kv_owner_mode();
        assert!(engine.single_kv_owner_mode());
        assert_eq!(engine.total_slots(), 1);
        release_tx.send(()).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while engine.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(
            *order.lock().unwrap(),
            vec!['a', 'a', 'x', 'b'],
            "the exclusive and second stream must wait until the retained first session drains"
        );
    }

    #[tokio::test]
    async fn cooperative_context_returns_to_single_stream_fast_path() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

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
        std::env::set_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV, "2");
        let engine = EngineHandle::spawn();
        std::env::remove_var(crate::runtime_config::CONTINUOUS_BATCH_SLOTS_ENV);

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
