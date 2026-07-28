//! The engine worker: one dedicated OS thread that is the single place decode
//! compute and resident-GPU-state mutations execute.
//!
//! Ownership contract (mirrors llama.cpp's `server_queue` + single consumer
//! thread, see docs/recon/ENGINE_INVERSION_CONDUCTOR.md): HTTP handlers
//! validate and prepare OUTSIDE any serialization, then post a job on a
//! bounded queue and await its result. The engine thread executes jobs one at
//! a time, so "at most one decode in flight" holds by construction — there is
//! no lock whose misuse can corrupt shared decode state. Anything that touches
//! engine-owned state (decode loops, the GPU-runnable parity probe,
//! resident-cache resets) must run as an engine job, never inline in a
//! handler.
//!
//! Cancellation stays cooperative: a posted job cannot be aborted, but every
//! decode loop observes its request's `GenerationCancel` once per step, so a
//! dropped handler (client disconnect) stops the running job within one step
//! and queued jobs from dropped handlers return immediately when they run.

use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

/// Bounded queue depth (queued jobs, not counting the one running).
/// Overridable for hardening runs; the default keeps a small, honest queue —
/// beyond it the server answers 503 rather than parking unbounded waiters.
pub(crate) const QUEUE_DEPTH_ENV: &str = crate::runtime_config::ENGINE_QUEUE_DEPTH_ENV;

type ExclusiveJob = Box<dyn FnOnce() + Send + 'static>;

/// A unit of engine work. Every variant runs to completion on the engine
/// thread before the next is picked up.
pub(crate) enum EngineTask {
    /// A serialized blocking job: a decode loop, the GPU-runnable parity
    /// probe, a resident-cache reset, a prompt-cache mutation. The closure
    /// owns everything it needs and reports back through a channel it
    /// captured (typically `tokio::sync::oneshot`).
    Exclusive(ExclusiveJob),
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
    next_task_id: Arc<AtomicU64>,
}

fn queue_depth_from_env() -> usize {
    crate::runtime_config::engine_queue_depth()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineSlotSnapshot {
    pub(crate) active_task_id: Option<u64>,
    pub(crate) queued_tasks: usize,
}

impl EngineSlotSnapshot {
    pub(crate) fn is_processing(self) -> bool {
        self.active_task_id.is_some()
    }
}

struct ActiveTaskGuard {
    active_task_id: Arc<AtomicU64>,
}

impl Drop for ActiveTaskGuard {
    fn drop(&mut self) {
        self.active_task_id.store(0, Ordering::SeqCst);
    }
}

impl EngineHandle {
    /// Spawn the engine worker thread and return the posting handle.
    pub(crate) fn spawn() -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineTask>(queue_depth_from_env());
        let depth = Arc::new(AtomicUsize::new(0));
        let worker_depth = Arc::clone(&depth);
        let active_task_id = Arc::new(AtomicU64::new(0));
        std::thread::Builder::new()
            .name("camelid-engine".to_string())
            .spawn(move || {
                while let Some(task) = rx.blocking_recv() {
                    match task {
                        EngineTask::Exclusive(job) => {
                            // The engine thread must survive a panicking job
                            // (uncurated models can panic deep in engine
                            // builds). The job's oneshot is dropped by the
                            // unwind, so the caller sees `Unavailable`; jobs
                            // that need to fail closed on panic wrap their own
                            // body in catch_unwind and return a verdict.
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        }
                    }
                    worker_depth.fetch_sub(1, Ordering::SeqCst);
                }
            })
            .expect("spawn camelid-engine worker thread");
        Self {
            tx,
            depth,
            active_task_id,
            next_task_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Jobs accepted and not yet finished.
    pub(crate) fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Privacy-safe, read-only state for the production engine's real slot.
    /// Queue depth remains separate because queued jobs do not own a slot.
    pub(crate) fn slot_snapshot(&self) -> EngineSlotSnapshot {
        let active = self.active_task_id.load(Ordering::SeqCst);
        let depth = self.depth();
        EngineSlotSnapshot {
            active_task_id: (active != 0).then_some(active),
            queued_tasks: depth.saturating_sub(usize::from(active != 0)),
        }
    }

    /// Post a job without waiting for queue room: full queue is an explicit,
    /// typed condition (503 + Retry-After at the HTTP layer), never an
    /// invisible pile of waiters.
    pub(crate) fn post(&self, task: EngineTask) -> Result<(), EnginePostError> {
        let task_id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        let active_task_id = Arc::clone(&self.active_task_id);
        let task = match task {
            EngineTask::Exclusive(job) => EngineTask::Exclusive(Box::new(move || {
                active_task_id.store(task_id, Ordering::SeqCst);
                let _guard = ActiveTaskGuard { active_task_id };
                job();
            })),
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
        engine
            .post(EngineTask::Exclusive(Box::new(move || {
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
            }
        );
    }
}
