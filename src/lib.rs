pub mod api;
pub mod cluster;
pub mod distributed;
pub mod error;
pub mod execution_plan;
pub mod gguf;
pub mod ghost;
pub mod inference;
pub mod metal;
pub mod model;
pub mod model_source;
pub mod tensor;
pub mod tokenizer;

pub use error::{BackendError, Result};

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Condvar, Mutex, MutexGuard};
    use std::thread::ThreadId;

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static Q8_FILE_TEST_LOCK: ReentrantTestLock = ReentrantTestLock::new();

    pub(crate) struct TestEnvGuard {
        _env_guard: MutexGuard<'static, ()>,
    }

    #[derive(Default)]
    struct ReentrantTestLockState {
        owner: Option<ThreadId>,
        depth: usize,
    }

    struct ReentrantTestLock {
        state: Mutex<ReentrantTestLockState>,
        cv: Condvar,
    }

    impl ReentrantTestLock {
        const fn new() -> Self {
            Self {
                state: Mutex::new(ReentrantTestLockState {
                    owner: None,
                    depth: 0,
                }),
                cv: Condvar::new(),
            }
        }

        fn lock(&'static self) -> ReentrantTestLockGuard {
            let current = std::thread::current().id();
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            loop {
                match state.owner {
                    Some(owner) if owner != current => {
                        state = self
                            .cv
                            .wait(state)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    Some(_) => {
                        state.depth += 1;
                        return ReentrantTestLockGuard { lock: self };
                    }
                    None => {
                        state.owner = Some(current);
                        state.depth = 1;
                        return ReentrantTestLockGuard { lock: self };
                    }
                }
            }
        }
    }

    pub(crate) struct ReentrantTestLockGuard {
        lock: &'static ReentrantTestLock,
    }

    impl Drop for ReentrantTestLockGuard {
        fn drop(&mut self) {
            let current = std::thread::current().id();
            let mut state = self
                .lock
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            debug_assert_eq!(state.owner, Some(current));
            state.depth -= 1;
            if state.depth == 0 {
                state.owner = None;
                self.lock.cv.notify_all();
            }
        }
    }

    pub(crate) fn q8_file_state_lock() -> ReentrantTestLockGuard {
        Q8_FILE_TEST_LOCK.lock()
    }

    pub(crate) fn env_lock() -> TestEnvGuard {
        let env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        TestEnvGuard {
            _env_guard: env_guard,
        }
    }
}
