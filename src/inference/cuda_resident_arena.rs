#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentArenaDisposition<K> {
    Reused,
    Inserted { evicted: Option<K> },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ResidentArenaError {
    #[error("resident model arena is full and every model is active")]
    AllModelsActive,
}

#[derive(Debug)]
struct ResidentArenaEntry<K, V> {
    key: K,
    value: V,
    last_used: u64,
    active: usize,
}

#[derive(Debug)]
pub(crate) struct ResidentModelArena<K, V> {
    capacity: usize,
    clock: u64,
    entries: Vec<ResidentArenaEntry<K, V>>,
    evictions: u64,
    admission_failures: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentArenaSnapshot {
    pub(crate) capacity: usize,
    pub(crate) resident: usize,
    pub(crate) active: usize,
    pub(crate) evictions: u64,
    pub(crate) admission_failures: u64,
}

impl<K: Copy + Eq, V: Clone> ResidentModelArena<K, V> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            clock: 0,
            entries: Vec::new(),
            evictions: 0,
            admission_failures: 0,
        }
    }

    pub(crate) fn acquire(
        &mut self,
        key: K,
        create: impl FnOnce() -> V,
        is_idle: impl Fn(&V) -> bool,
    ) -> Result<(V, ResidentArenaDisposition<K>), ResidentArenaError> {
        self.clock = self.clock.wrapping_add(1).max(1);
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.last_used = self.clock;
            return Ok((entry.value.clone(), ResidentArenaDisposition::Reused));
        }

        let evicted = if self.entries.len() >= self.capacity {
            let index = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.active == 0 && is_idle(&entry.value))
                .min_by_key(|(index, entry)| (entry.last_used, *index))
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    self.admission_failures = self.admission_failures.saturating_add(1);
                    ResidentArenaError::AllModelsActive
                })?;
            self.evictions = self.evictions.saturating_add(1);
            Some(self.entries.remove(index).key)
        } else {
            None
        };

        let value = create();
        self.entries.push(ResidentArenaEntry {
            key,
            value: value.clone(),
            last_used: self.clock,
            active: 0,
        });
        Ok((value, ResidentArenaDisposition::Inserted { evicted }))
    }

    pub(crate) fn peek_matching(&self, predicate: impl Fn(&K) -> bool) -> Option<V> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| predicate(&entry.key))
            .max_by_key(|(index, entry)| (entry.last_used, *index))
            .map(|(_, entry)| entry.value.clone())
    }

    pub(crate) fn pin(&mut self, key: K) -> Option<V> {
        self.clock = self.clock.wrapping_add(1).max(1);
        let entry = self.entries.iter_mut().find(|entry| entry.key == key)?;
        entry.last_used = self.clock;
        entry.active = entry.active.saturating_add(1);
        Some(entry.value.clone())
    }

    pub(crate) fn unpin(&mut self, key: K) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) else {
            return false;
        };
        let Some(active) = entry.active.checked_sub(1) else {
            return false;
        };
        entry.active = active;
        true
    }

    pub(crate) fn active_matching(&self, predicate: impl Fn(&K) -> bool) -> usize {
        self.entries
            .iter()
            .filter(|entry| predicate(&entry.key) && entry.active > 0)
            .count()
    }

    pub(crate) fn matching_values(&self, predicate: impl Fn(&K) -> bool) -> Vec<V> {
        self.entries
            .iter()
            .filter(|entry| predicate(&entry.key))
            .map(|entry| entry.value.clone())
            .collect()
    }

    pub(crate) fn remove(&mut self, key: K) -> Option<V> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.key == key && entry.active == 0)?;
        Some(self.entries.remove(index).value)
    }

    pub(crate) fn remove_matching(&mut self, predicate: impl Fn(&K) -> bool) -> Vec<V> {
        let mut removed = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            if predicate(&self.entries[index].key) && self.entries[index].active == 0 {
                removed.push(self.entries.remove(index).value);
            } else {
                index += 1;
            }
        }
        removed
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    pub(crate) fn snapshot(&self) -> ResidentArenaSnapshot {
        ResidentArenaSnapshot {
            capacity: self.capacity,
            resident: self.entries.len(),
            active: self.entries.iter().filter(|entry| entry.active > 0).count(),
            evictions: self.evictions,
            admission_failures: self.admission_failures,
        }
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<K> {
        self.entries.iter().map(|entry| entry.key).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{ResidentArenaDisposition, ResidentArenaError, ResidentModelArena};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct FakeModel {
        name: &'static str,
        active: bool,
    }

    #[test]
    fn phase8_idle_lru_is_deterministic_and_reuse_refreshes_recency() {
        let mut arena = ResidentModelArena::new(2);
        let idle = |model: &FakeModel| !model.active;
        arena
            .acquire(
                1,
                || FakeModel {
                    name: "one",
                    active: false,
                },
                idle,
            )
            .unwrap();
        arena
            .acquire(
                2,
                || FakeModel {
                    name: "two",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert!(matches!(
            arena.acquire(1, || unreachable!(), idle).unwrap().1,
            ResidentArenaDisposition::Reused
        ));
        let (_, disposition) = arena
            .acquire(
                3,
                || FakeModel {
                    name: "three",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert_eq!(
            disposition,
            ResidentArenaDisposition::Inserted { evicted: Some(2) }
        );
        assert_eq!(arena.keys(), vec![1, 3]);
    }

    #[test]
    fn phase8_active_models_are_never_evicted_and_full_active_arena_is_unchanged() {
        let mut arena = ResidentModelArena::new(2);
        let idle = |model: &FakeModel| !model.active;
        arena
            .acquire(
                1,
                || FakeModel {
                    name: "one",
                    active: true,
                },
                idle,
            )
            .unwrap();
        arena
            .acquire(
                2,
                || FakeModel {
                    name: "two",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert_eq!(
            arena.acquire(
                3,
                || FakeModel {
                    name: "three",
                    active: false,
                },
                idle,
            ),
            Ok((
                FakeModel {
                    name: "three",
                    active: false,
                },
                ResidentArenaDisposition::Inserted { evicted: Some(2) }
            ))
        );
        assert_eq!(arena.keys(), vec![1, 3]);

        let mut full = ResidentModelArena::new(2);
        for key in [1, 2] {
            full.acquire(
                key,
                || FakeModel {
                    name: "active",
                    active: true,
                },
                idle,
            )
            .unwrap();
        }
        assert_eq!(
            full.acquire(
                3,
                || FakeModel {
                    name: "three",
                    active: false,
                },
                idle,
            ),
            Err(ResidentArenaError::AllModelsActive)
        );
        assert_eq!(full.keys(), vec![1, 2]);

        let mut pinned = ResidentModelArena::new(2);
        pinned
            .acquire(
                1,
                || FakeModel {
                    name: "pinned",
                    active: false,
                },
                idle,
            )
            .unwrap();
        pinned
            .acquire(
                2,
                || FakeModel {
                    name: "idle",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert_eq!(pinned.pin(1).unwrap().name, "pinned");
        assert_eq!(pinned.snapshot().active, 1);
        assert_eq!(pinned.active_matching(|key| *key == 1), 1);
        assert_eq!(pinned.active_matching(|key| *key == 2), 0);
        assert!(matches!(
            pinned
                .acquire(
                    3,
                    || FakeModel {
                        name: "replacement",
                        active: false,
                    },
                    idle,
                )
                .unwrap()
                .1,
            ResidentArenaDisposition::Inserted { evicted: Some(2) }
        ));
        assert_eq!(pinned.keys(), vec![1, 3]);
        assert!(pinned.unpin(1));
        assert_eq!(pinned.snapshot().active, 0);
        assert_eq!(pinned.active_matching(|key| *key == 1), 0);
    }

    #[test]
    fn phase8_failed_second_reservation_can_rollback_without_touching_first() {
        let mut arena = ResidentModelArena::new(2);
        let idle = |model: &FakeModel| !model.active;
        arena
            .acquire(
                1,
                || FakeModel {
                    name: "first",
                    active: false,
                },
                idle,
            )
            .unwrap();
        arena
            .acquire(
                2,
                || FakeModel {
                    name: "candidate",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert_eq!(arena.remove(2).unwrap().name, "candidate");
        assert_eq!(arena.keys(), vec![1]);
        assert_eq!(arena.peek_matching(|key| *key == 1).unwrap().name, "first");
    }

    #[test]
    fn phase8_observation_does_not_change_lru_order() {
        let mut arena = ResidentModelArena::new(2);
        let idle = |model: &FakeModel| !model.active;
        for key in [1, 2] {
            arena
                .acquire(
                    key,
                    || FakeModel {
                        name: "idle",
                        active: false,
                    },
                    idle,
                )
                .unwrap();
        }
        assert!(arena.peek_matching(|key| *key == 1).is_some());
        let (_, disposition) = arena
            .acquire(
                3,
                || FakeModel {
                    name: "new",
                    active: false,
                },
                idle,
            )
            .unwrap();
        assert_eq!(
            disposition,
            ResidentArenaDisposition::Inserted { evicted: Some(1) }
        );
    }
}
