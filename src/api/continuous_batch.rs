//! Cooperative round-robin scheduler foundation for continuous batching.
//!
//! The current production engine still runs one exclusive closure at a time.
//! This scheduler captures the lifecycle and fairness rules needed when decode
//! jobs are converted to one-token `step` tasks.  Keeping it independent makes
//! those rules testable before the hot inference path is switched over.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BatchTaskId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepOutcome {
    Continue,
    Complete,
}

#[derive(Debug)]
struct BatchTask<T> {
    id: BatchTaskId,
    state: T,
}

#[derive(Debug)]
pub(crate) struct ContinuousBatch<T> {
    max_slots: usize,
    next_id: u64,
    waiting: VecDeque<BatchTask<T>>,
    active: VecDeque<BatchTask<T>>,
}

impl<T> ContinuousBatch<T> {
    pub(crate) fn from_env() -> Self {
        Self::new(crate::runtime_config::continuous_batch_slots())
    }

    pub(crate) fn new(max_slots: usize) -> Self {
        Self {
            max_slots: max_slots.max(1),
            next_id: 1,
            waiting: VecDeque::new(),
            active: VecDeque::new(),
        }
    }

    pub(crate) fn admit(&mut self, state: T) -> BatchTaskId {
        let id = BatchTaskId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.waiting.push_back(BatchTask { id, state });
        id
    }

    pub(crate) fn cancel(&mut self, id: BatchTaskId) -> Option<T> {
        remove_task(&mut self.waiting, id).or_else(|| remove_task(&mut self.active, id))
    }

    pub(crate) fn active_len(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }

    /// Run at most one token-step for every active slot. Tasks that need more
    /// work rotate to the tail; completed tasks release their slot before the
    /// next round. Returns completed IDs in completion order.
    pub(crate) fn run_round(
        &mut self,
        mut step: impl FnMut(BatchTaskId, &mut T) -> StepOutcome,
    ) -> Vec<BatchTaskId> {
        self.fill_slots();
        let round_len = self.active.len();
        let mut completed = Vec::new();
        for _ in 0..round_len {
            let mut task = self.active.pop_front().expect("round length is exact");
            match step(task.id, &mut task.state) {
                StepOutcome::Continue => self.active.push_back(task),
                StepOutcome::Complete => completed.push(task.id),
            }
        }
        self.fill_slots();
        completed
    }

    fn fill_slots(&mut self) {
        while self.active.len() < self.max_slots {
            let Some(task) = self.waiting.pop_front() else {
                break;
            };
            self.active.push_back(task);
        }
    }
}

fn remove_task<T>(queue: &mut VecDeque<BatchTask<T>>, id: BatchTaskId) -> Option<T> {
    let index = queue.iter().position(|task| task.id == id)?;
    queue.remove(index).map(|task| task.state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_is_fair_and_slot_bounded() {
        let mut scheduler = ContinuousBatch::new(2);
        let ids = [
            scheduler.admit(0usize),
            scheduler.admit(0usize),
            scheduler.admit(0usize),
        ];
        let mut order = Vec::new();
        for _ in 0..3 {
            scheduler.run_round(|id, steps| {
                order.push(id);
                *steps += 1;
                if *steps == 2 {
                    StepOutcome::Complete
                } else {
                    StepOutcome::Continue
                }
            });
            assert!(scheduler.active_len() <= 2);
        }
        assert_eq!(order, vec![ids[0], ids[1], ids[0], ids[1], ids[2]]);
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(scheduler.waiting_len(), 0);
    }

    #[test]
    fn cancellation_releases_waiting_and_active_tasks() {
        let mut scheduler = ContinuousBatch::new(1);
        let first = scheduler.admit("first");
        let second = scheduler.admit("second");
        scheduler.run_round(|_, _| StepOutcome::Continue);
        assert_eq!(scheduler.cancel(second), Some("second"));
        assert_eq!(scheduler.cancel(first), Some("first"));
        assert!(scheduler.is_empty());
    }
}
