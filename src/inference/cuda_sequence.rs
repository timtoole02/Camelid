use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SEQUENCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct CudaSequenceId(u64);

impl CudaSequenceId {
    pub(super) fn next() -> Self {
        let value = NEXT_SEQUENCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.wrapping_add(1).max(1))
            })
            .unwrap_or_else(|current| current);
        Self(value.max(1))
    }

    pub(super) const fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    const fn for_test(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaKvSlotId(usize);

impl CudaKvSlotId {
    pub(super) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaSequenceLease {
    sequence_id: CudaSequenceId,
    slot_id: CudaKvSlotId,
    generation: u64,
}

impl CudaSequenceLease {
    #[allow(dead_code)] // Phase 3 executor validation; Phase 4 adds the production caller.
    pub(super) const fn sequence_id(self) -> CudaSequenceId {
        self.sequence_id
    }

    pub(super) const fn slot_id(self) -> CudaKvSlotId {
        self.slot_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaSlotAcquisition {
    Existing(CudaSequenceLease),
    Fresh(CudaSequenceLease),
}

impl CudaSlotAcquisition {
    pub(super) const fn lease(self) -> CudaSequenceLease {
        match self {
            Self::Existing(lease) | Self::Fresh(lease) => lease,
        }
    }

    #[cfg(test)]
    pub(super) const fn is_fresh(self) -> bool {
        matches!(self, Self::Fresh(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaSlotTeardown {
    slot_id: CudaKvSlotId,
    generation: u64,
}

impl CudaSlotTeardown {
    #[cfg(test)]
    pub(super) const fn slot_id(self) -> CudaKvSlotId {
        self.slot_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaSlotState {
    Vacant,
    Occupied(CudaSequenceId),
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaSlotEntry {
    generation: u64,
    state: CudaSlotState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaSequenceSlotSnapshot {
    pub(super) capacity: usize,
    pub(super) occupied: usize,
    pub(super) poisoned: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(super) enum CudaSequenceSlotError {
    #[error("CUDA sequence slot capacity must be between one and eight, got {0}")]
    InvalidCapacity(usize),
    #[error("all {capacity} CUDA sequence slots are occupied")]
    CapacityExhausted { capacity: usize },
    #[error("CUDA sequence lease is stale")]
    StaleLease,
    #[cfg(test)]
    #[error("CUDA sequence slot teardown token is stale")]
    StaleTeardown,
}

#[derive(Debug)]
pub(super) struct FixedCudaSequenceSlots {
    slots: Vec<CudaSlotEntry>,
}

impl FixedCudaSequenceSlots {
    pub(super) fn new(capacity: usize) -> Result<Self, CudaSequenceSlotError> {
        if !(1..=8).contains(&capacity) {
            return Err(CudaSequenceSlotError::InvalidCapacity(capacity));
        }
        Ok(Self {
            slots: vec![
                CudaSlotEntry {
                    generation: 0,
                    state: CudaSlotState::Vacant,
                };
                capacity
            ],
        })
    }

    pub(super) fn acquire(
        &mut self,
        sequence_id: CudaSequenceId,
    ) -> Result<CudaSlotAcquisition, CudaSequenceSlotError> {
        if let Some((index, entry)) = self
            .slots
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.state == CudaSlotState::Occupied(sequence_id))
        {
            return Ok(CudaSlotAcquisition::Existing(CudaSequenceLease {
                sequence_id,
                slot_id: CudaKvSlotId(index),
                generation: entry.generation,
            }));
        }

        let Some((index, entry)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.state == CudaSlotState::Vacant)
        else {
            return Err(CudaSequenceSlotError::CapacityExhausted {
                capacity: self.slots.len(),
            });
        };
        entry.generation = entry.generation.wrapping_add(1).max(1);
        entry.state = CudaSlotState::Occupied(sequence_id);
        Ok(CudaSlotAcquisition::Fresh(CudaSequenceLease {
            sequence_id,
            slot_id: CudaKvSlotId(index),
            generation: entry.generation,
        }))
    }

    pub(super) fn validate(&self, lease: CudaSequenceLease) -> bool {
        self.slots.get(lease.slot_id.index()).is_some_and(|entry| {
            entry.generation == lease.generation
                && entry.state == CudaSlotState::Occupied(lease.sequence_id)
        })
    }

    pub(super) fn release(
        &mut self,
        lease: CudaSequenceLease,
    ) -> Result<CudaKvSlotId, CudaSequenceSlotError> {
        if !self.validate(lease) {
            return Err(CudaSequenceSlotError::StaleLease);
        }
        self.slots[lease.slot_id.index()].state = CudaSlotState::Vacant;
        Ok(lease.slot_id)
    }

    pub(super) fn poison(
        &mut self,
        lease: CudaSequenceLease,
    ) -> Result<CudaSlotTeardown, CudaSequenceSlotError> {
        if !self.validate(lease) {
            return Err(CudaSequenceSlotError::StaleLease);
        }
        self.slots[lease.slot_id.index()].state = CudaSlotState::Poisoned;
        Ok(CudaSlotTeardown {
            slot_id: lease.slot_id,
            generation: lease.generation,
        })
    }

    #[cfg(test)]
    pub(super) fn complete_teardown(
        &mut self,
        teardown: CudaSlotTeardown,
    ) -> Result<(), CudaSequenceSlotError> {
        let Some(entry) = self.slots.get_mut(teardown.slot_id.index()) else {
            return Err(CudaSequenceSlotError::StaleTeardown);
        };
        if entry.generation != teardown.generation || entry.state != CudaSlotState::Poisoned {
            return Err(CudaSequenceSlotError::StaleTeardown);
        }
        entry.state = CudaSlotState::Vacant;
        Ok(())
    }

    pub(super) fn snapshot(&self) -> CudaSequenceSlotSnapshot {
        CudaSequenceSlotSnapshot {
            capacity: self.slots.len(),
            occupied: self
                .slots
                .iter()
                .filter(|entry| matches!(entry.state, CudaSlotState::Occupied(_)))
                .count(),
            poisoned: self
                .slots
                .iter()
                .filter(|entry| entry.state == CudaSlotState::Poisoned)
                .count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_distinct_sequences_get_distinct_stable_slots() {
        let mut slots = FixedCudaSequenceSlots::new(2).unwrap();
        let first = slots.acquire(CudaSequenceId::for_test(10)).unwrap();
        let second = slots.acquire(CudaSequenceId::for_test(20)).unwrap();

        assert_ne!(first.lease().slot_id(), second.lease().slot_id());
        assert_eq!(
            slots.acquire(CudaSequenceId::for_test(10)).unwrap(),
            CudaSlotAcquisition::Existing(first.lease())
        );
        assert_eq!(
            slots.snapshot(),
            CudaSequenceSlotSnapshot {
                capacity: 2,
                occupied: 2,
                poisoned: 0,
            }
        );
    }

    #[test]
    fn failed_third_admission_is_atomic() {
        let mut slots = FixedCudaSequenceSlots::new(2).unwrap();
        let first = slots.acquire(CudaSequenceId::for_test(1)).unwrap();
        let second = slots.acquire(CudaSequenceId::for_test(2)).unwrap();
        let before = slots.snapshot();

        assert_eq!(
            slots.acquire(CudaSequenceId::for_test(3)),
            Err(CudaSequenceSlotError::CapacityExhausted { capacity: 2 })
        );
        assert_eq!(slots.snapshot(), before);
        assert!(slots.validate(first.lease()));
        assert!(slots.validate(second.lease()));
    }

    #[test]
    fn phase7_eight_sequences_are_isolated_and_reuse_rejects_stale_leases() {
        let mut slots = FixedCudaSequenceSlots::new(8).unwrap();
        let leases = (1..=8)
            .map(|sequence| {
                slots
                    .acquire(CudaSequenceId::for_test(sequence))
                    .unwrap()
                    .lease()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            leases
                .iter()
                .map(|lease| lease.slot_id().index())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            8
        );
        let full = slots.snapshot();
        assert_eq!(full.capacity, 8);
        assert_eq!(full.occupied, 8);
        assert_eq!(
            slots.acquire(CudaSequenceId::for_test(9)),
            Err(CudaSequenceSlotError::CapacityExhausted { capacity: 8 })
        );
        assert_eq!(slots.snapshot(), full);

        let released = leases[3];
        assert_eq!(slots.release(released), Ok(released.slot_id()));
        let replacement = slots.acquire(CudaSequenceId::for_test(9)).unwrap().lease();
        assert_eq!(replacement.slot_id(), released.slot_id());
        assert_ne!(replacement.generation, released.generation);
        assert!(!slots.validate(released));
        assert_eq!(
            slots.release(released),
            Err(CudaSequenceSlotError::StaleLease)
        );
        assert!(slots.validate(replacement));
        assert!(leases
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != 3)
            .all(|(_, lease)| slots.validate(*lease)));
    }

    #[test]
    fn release_reuse_changes_generation_and_rejects_stale_lease() {
        let mut slots = FixedCudaSequenceSlots::new(1).unwrap();
        let first = slots.acquire(CudaSequenceId::for_test(1)).unwrap().lease();
        assert_eq!(slots.release(first), Ok(first.slot_id()));

        let second = slots.acquire(CudaSequenceId::for_test(2)).unwrap().lease();
        assert_eq!(second.slot_id(), first.slot_id());
        assert_ne!(second.generation, first.generation);
        assert!(!slots.validate(first));
        assert_eq!(slots.release(first), Err(CudaSequenceSlotError::StaleLease));
        assert!(slots.validate(second));
    }

    #[test]
    fn poisoned_slot_is_unavailable_until_teardown_completes() {
        let mut slots = FixedCudaSequenceSlots::new(1).unwrap();
        let lease = slots.acquire(CudaSequenceId::for_test(1)).unwrap().lease();
        let teardown = slots.poison(lease).unwrap();

        assert_eq!(teardown.slot_id(), lease.slot_id());
        assert_eq!(slots.snapshot().poisoned, 1);
        assert_eq!(
            slots.acquire(CudaSequenceId::for_test(2)),
            Err(CudaSequenceSlotError::CapacityExhausted { capacity: 1 })
        );
        slots.complete_teardown(teardown).unwrap();
        assert!(slots
            .acquire(CudaSequenceId::for_test(2))
            .unwrap()
            .is_fresh());
        assert_eq!(
            slots.complete_teardown(teardown),
            Err(CudaSequenceSlotError::StaleTeardown)
        );
    }

    #[test]
    fn invalid_capacity_is_refused() {
        assert!(matches!(
            FixedCudaSequenceSlots::new(0),
            Err(CudaSequenceSlotError::InvalidCapacity(0))
        ));
        assert!(matches!(
            FixedCudaSequenceSlots::new(9),
            Err(CudaSequenceSlotError::InvalidCapacity(9))
        ));
    }
}
