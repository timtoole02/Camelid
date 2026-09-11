//! Cooperative round-robin scheduler for opt-in continuous batching.
//!
//! Streaming decode jobs retain their own session state and yield after one
//! token. The engine remains the sole compute owner; this scheduler only
//! interleaves those token steps and bounds the number of active sessions.

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

    /// True while another session can be admitted without exceeding `max_slots`.
    /// The worker uses this to leave surplus work in the bounded engine channel
    /// rather than in an unbounded local queue, so queue-full backpressure keeps
    /// working while streams are running.
    pub(crate) fn has_free_slot(&self) -> bool {
        self.active.len() + self.waiting.len() < self.max_slots
    }

    /// Number of active slots the next round will have after waiting work is admitted.
    pub(crate) fn scheduled_len(&self) -> usize {
        self.max_slots
            .min(self.active.len().saturating_add(self.waiting.len()))
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

    /// Run one scheduling boundary, pairing the oldest compatible ready tasks.
    /// A pair callback may decline without mutation; both rows then take their
    /// ordinary scalar step in FIFO order. Continuing tasks return to the active
    /// queue in their original order, so batching cannot reorder fairness.
    pub(crate) fn run_grouped_round<K: Eq>(
        &mut self,
        mut key: impl FnMut(BatchTaskId, &mut T) -> Option<K>,
        mut step_one: impl FnMut(BatchTaskId, &mut T) -> StepOutcome,
        mut step_pair: impl FnMut(BatchTaskId, &mut T, BatchTaskId, &mut T) -> Option<[StepOutcome; 2]>,
    ) -> Vec<BatchTaskId> {
        self.run_prioritized_grouped_round(
            |_, _| false,
            |id, state| key(id, state),
            |id, state| step_one(id, state),
            |left_id, left, right_id, right| step_pair(left_id, left, right_id, right),
        )
    }

    /// Run one grouped round with a stable, bounded within-round preference.
    /// Preferred tasks execute first, but every task in the round still executes
    /// at most once and continuing tasks return in their original queue order.
    /// A decode row can therefore get ahead of a prefill chunk without starving
    /// that chunk or changing FIFO admission across scheduling boundaries.
    pub(crate) fn run_prioritized_grouped_round<K: Eq>(
        &mut self,
        mut preferred: impl FnMut(BatchTaskId, &T) -> bool,
        mut key: impl FnMut(BatchTaskId, &mut T) -> Option<K>,
        mut step_one: impl FnMut(BatchTaskId, &mut T) -> StepOutcome,
        mut step_pair: impl FnMut(BatchTaskId, &mut T, BatchTaskId, &mut T) -> Option<[StepOutcome; 2]>,
    ) -> Vec<BatchTaskId> {
        self.fill_slots();
        let round_len = self.active.len();
        let mut tasks = self.active.drain(..round_len).collect::<Vec<_>>();
        let keys = tasks
            .iter_mut()
            .map(|task| key(task.id, &mut task.state))
            .collect::<Vec<_>>();
        let preferred = tasks
            .iter()
            .map(|task| preferred(task.id, &task.state))
            .collect::<Vec<_>>();
        let mut outcomes = vec![None; round_len];
        let order = (0..round_len)
            .filter(|index| preferred[*index])
            .chain((0..round_len).filter(|index| !preferred[*index]))
            .collect::<Vec<_>>();

        for (order_index, left_index) in order.iter().copied().enumerate() {
            if outcomes[left_index].is_some() {
                continue;
            }
            let partner = keys[left_index].as_ref().and_then(|left_key| {
                order[(order_index + 1)..]
                    .iter()
                    .copied()
                    .find(|right_index| {
                        outcomes[*right_index].is_none()
                            && keys[*right_index].as_ref() == Some(left_key)
                    })
            });
            let Some(right_index) = partner else {
                outcomes[left_index] =
                    Some(step_one(tasks[left_index].id, &mut tasks[left_index].state));
                continue;
            };

            let (left, right) = tasks.split_at_mut(right_index);
            let left_task = &mut left[left_index];
            let right_task = &mut right[0];
            if let Some(pair_outcomes) = step_pair(
                left_task.id,
                &mut left_task.state,
                right_task.id,
                &mut right_task.state,
            ) {
                outcomes[left_index] = Some(pair_outcomes[0]);
                outcomes[right_index] = Some(pair_outcomes[1]);
            } else {
                outcomes[left_index] = Some(step_one(left_task.id, &mut left_task.state));
                outcomes[right_index] = Some(step_one(right_task.id, &mut right_task.state));
            }
        }

        let mut completed = Vec::new();
        for (task, outcome) in tasks.into_iter().zip(outcomes) {
            match outcome.expect("every task in the round receives one outcome") {
                StepOutcome::Continue => self.active.push_back(task),
                StepOutcome::Complete => completed.push(task.id),
            }
        }
        self.fill_slots();
        completed
    }

    /// Run one scheduling boundary, dispatching every ready task with the oldest
    /// compatible key as one group. Group members execute in preferred-then-FIFO
    /// order, while continuing tasks return to their original queue positions.
    /// `None` means no shared work was dispatched and permits scalar fallback.
    /// A mismatched result count fails the group closed instead of replaying rows
    /// whose state may already have advanced.
    pub(crate) fn run_prioritized_grouped_round_n<K: Eq>(
        &mut self,
        mut preferred: impl FnMut(BatchTaskId, &T) -> bool,
        mut key: impl FnMut(BatchTaskId, &mut T) -> Option<K>,
        mut step_one: impl FnMut(BatchTaskId, &mut T) -> StepOutcome,
        mut step_group: impl FnMut(&[BatchTaskId], &mut [T]) -> Option<Vec<StepOutcome>>,
    ) -> Vec<BatchTaskId> {
        self.fill_slots();
        let round_len = self.active.len();
        let mut drained = self.active.drain(..round_len).collect::<Vec<_>>();
        let keys = drained
            .iter_mut()
            .map(|task| key(task.id, &mut task.state))
            .collect::<Vec<_>>();
        let preferred = drained
            .iter()
            .map(|task| preferred(task.id, &task.state))
            .collect::<Vec<_>>();
        let order = (0..round_len)
            .filter(|index| preferred[*index])
            .chain((0..round_len).filter(|index| !preferred[*index]))
            .collect::<Vec<_>>();
        let mut tasks = drained.into_iter().map(Some).collect::<Vec<_>>();
        let mut results = std::iter::repeat_with(|| None)
            .take(round_len)
            .collect::<Vec<Option<(BatchTask<T>, StepOutcome)>>>();

        for (order_index, left_index) in order.iter().copied().enumerate() {
            if results[left_index].is_some() {
                continue;
            }
            let group_indices = match keys[left_index].as_ref() {
                Some(left_key) => order[order_index..]
                    .iter()
                    .copied()
                    .filter(|index| {
                        results[*index].is_none() && keys[*index].as_ref() == Some(left_key)
                    })
                    .collect::<Vec<_>>(),
                None => vec![left_index],
            };
            let mut ids = Vec::with_capacity(group_indices.len());
            let mut states = Vec::with_capacity(group_indices.len());
            for index in &group_indices {
                let task = tasks[*index]
                    .take()
                    .expect("an unprocessed round task remains available");
                ids.push(task.id);
                states.push(task.state);
            }

            let outcomes = if states.len() == 1 {
                vec![step_one(ids[0], &mut states[0])]
            } else {
                match step_group(&ids, &mut states) {
                    Some(outcomes) if outcomes.len() == states.len() => outcomes,
                    Some(_) => vec![StepOutcome::Complete; states.len()],
                    None => ids
                        .iter()
                        .copied()
                        .zip(states.iter_mut())
                        .map(|(id, state)| step_one(id, state))
                        .collect(),
                }
            };
            for (((index, id), state), outcome) in
                group_indices.into_iter().zip(ids).zip(states).zip(outcomes)
            {
                results[index] = Some((BatchTask { id, state }, outcome));
            }
        }

        debug_assert!(tasks.iter().all(Option::is_none));
        let mut completed = Vec::new();
        for result in results {
            let (task, outcome) = result.expect("every task in the round receives one outcome");
            match outcome {
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

    #[test]
    fn grouped_round_pairs_oldest_compatible_tasks_without_reordering() {
        let mut scheduler = ContinuousBatch::new(4);
        let ids = [
            scheduler.admit((1u8, 0usize)),
            scheduler.admit((2u8, 0usize)),
            scheduler.admit((1u8, 0usize)),
            scheduler.admit((2u8, 0usize)),
        ];
        let mut pairs = Vec::new();
        scheduler.run_grouped_round(
            |_, state| Some(state.0),
            |_, state| {
                state.1 += 1;
                StepOutcome::Continue
            },
            |left_id, left, right_id, right| {
                pairs.push((left_id, right_id));
                left.1 += 10;
                right.1 += 10;
                Some([StepOutcome::Continue, StepOutcome::Continue])
            },
        );

        assert_eq!(pairs, vec![(ids[0], ids[2]), (ids[1], ids[3])]);
        let mut order = Vec::new();
        scheduler.run_round(|id, state| {
            order.push((id, state.1));
            StepOutcome::Complete
        });
        assert_eq!(
            order,
            ids.into_iter().map(|id| (id, 10)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn grouped_round_transitions_one_two_one_without_losing_work() {
        let mut scheduler = ContinuousBatch::new(2);
        let first = scheduler.admit(0usize);
        let mut shapes = Vec::new();
        let mut scalar = Vec::new();
        let mut paired = Vec::new();

        scheduler.run_grouped_round(
            |_, _| Some(1u8),
            |id, steps| {
                scalar.push(id);
                *steps += 1;
                StepOutcome::Continue
            },
            |left, _, right, _| {
                paired.push((left, right));
                Some([StepOutcome::Continue, StepOutcome::Continue])
            },
        );
        shapes.push(scheduler.active_len());
        let second = scheduler.admit(0usize);
        scheduler.run_grouped_round(
            |_, _| Some(1u8),
            |id, steps| {
                scalar.push(id);
                *steps += 1;
                StepOutcome::Continue
            },
            |left, left_steps, right, right_steps| {
                paired.push((left, right));
                *left_steps += 1;
                *right_steps += 1;
                Some([StepOutcome::Complete, StepOutcome::Continue])
            },
        );
        shapes.push(scheduler.active_len());
        scheduler.run_grouped_round(
            |_, _| Some(1u8),
            |id, steps| {
                scalar.push(id);
                *steps += 1;
                StepOutcome::Complete
            },
            |_, _, _, _| unreachable!(),
        );
        shapes.push(scheduler.active_len());

        assert_eq!(scalar, vec![first, second]);
        assert_eq!(paired, vec![(first, second)]);
        assert_eq!(shapes, vec![1, 1, 0]);
    }

    #[test]
    fn declined_pair_falls_back_to_two_scalar_steps() {
        let mut scheduler = ContinuousBatch::new(2);
        let ids = [scheduler.admit(0usize), scheduler.admit(0usize)];
        let mut scalar = Vec::new();
        scheduler.run_grouped_round(
            |_, _| Some(7u8),
            |id, steps| {
                scalar.push(id);
                *steps += 1;
                StepOutcome::Complete
            },
            |_, _, _, _| None,
        );
        assert_eq!(scalar, ids);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn phase7_grouped_round_dispatches_four_rows_once_and_scatters_outcomes() {
        let mut scheduler = ContinuousBatch::new(4);
        let ids = [
            scheduler.admit(0usize),
            scheduler.admit(0usize),
            scheduler.admit(0usize),
            scheduler.admit(0usize),
        ];
        let mut groups = Vec::new();
        scheduler.run_prioritized_grouped_round_n(
            |_, _| false,
            |_, _| Some(7u8),
            |_, _| unreachable!("all four rows are compatible"),
            |group_ids, states| {
                groups.push(group_ids.to_vec());
                for state in states {
                    *state += 10;
                }
                Some(vec![
                    StepOutcome::Continue,
                    StepOutcome::Complete,
                    StepOutcome::Continue,
                    StepOutcome::Complete,
                ])
            },
        );
        assert_eq!(groups, vec![ids.to_vec()]);
        assert_eq!(scheduler.active_len(), 2);

        let mut survivors = Vec::new();
        scheduler.run_round(|id, state| {
            survivors.push((id, *state));
            StepOutcome::Complete
        });
        assert_eq!(survivors, vec![(ids[0], 10), (ids[2], 10)]);
    }

    #[test]
    fn phase7_key_initialization_precedes_preference_ordering() {
        let mut scheduler = ContinuousBatch::new(2);
        let first = scheduler.admit(false);
        let second = scheduler.admit(true);
        let mut order = Vec::new();
        scheduler.run_prioritized_grouped_round_n(
            |_, initialized| *initialized,
            |id, initialized| {
                if id == first {
                    *initialized = true;
                }
                None::<u8>
            },
            |id, _| {
                order.push(id);
                StepOutcome::Complete
            },
            |_, _| unreachable!(),
        );
        assert_eq!(order, vec![first, second]);
    }

    #[test]
    fn phase7_decode_group_runs_before_prefill_group_without_starving_it() {
        let mut scheduler = ContinuousBatch::new(4);
        scheduler.admit((false, 1u8));
        scheduler.admit((true, 2u8));
        scheduler.admit((false, 1u8));
        scheduler.admit((true, 2u8));
        let mut groups = Vec::new();
        scheduler.run_prioritized_grouped_round_n(
            |_, state| state.0,
            |_, state| Some(state.1),
            |_, _| unreachable!("both phases have a partner"),
            |_, states| {
                groups.push(states[0].1);
                Some(vec![StepOutcome::Continue; states.len()])
            },
        );
        assert_eq!(groups, vec![2, 1]);
        assert_eq!(scheduler.active_len(), 4);
    }

    #[test]
    fn decode_preference_is_bounded_and_preserves_round_order() {
        let mut scheduler = ContinuousBatch::new(2);
        let prefill = scheduler.admit((false, 0usize));
        let decode = scheduler.admit((true, 0usize));
        let mut order = Vec::new();

        for _ in 0..2 {
            scheduler.run_prioritized_grouped_round(
                |_, state| state.0,
                |_, _| None::<u8>,
                |id, state| {
                    order.push(id);
                    state.1 += 1;
                    if state.1 == 2 {
                        StepOutcome::Complete
                    } else {
                        StepOutcome::Continue
                    }
                },
                |_, _, _, _| unreachable!(),
            );
        }

        assert_eq!(order, vec![decode, prefill, decode, prefill]);
        assert!(scheduler.is_empty());
    }
}
