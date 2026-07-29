//! Budgeted multi-sequence ownership for Llama K/V caches.
//!
//! `LlamaKvCache` remains the storage implementation. This layer adds stable
//! sequence identity, lifecycle state, a global byte ceiling, idle-only LRU
//! eviction, and sequence-local checkpoints. It is intentionally not wired
//! into the default serve loop yet; doing that safely requires the cooperative
//! token-step scheduler to own session lifetimes end-to-end.

use std::collections::HashMap;

use super::LlamaKvCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvSequenceId(u64);

impl KvSequenceId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSequenceState {
    Idle,
    Active,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSequenceCheckpoint {
    pub sequence_id: KvSequenceId,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolSnapshot {
    pub sequence_count: usize,
    pub active_sequences: usize,
    pub allocated_bytes: u64,
    pub budget_bytes: u64,
    pub evictions: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvPoolError {
    #[error("unknown KV sequence {0}")]
    UnknownSequence(u64),
    #[error("KV sequence {0} is already active")]
    AlreadyActive(u64),
    #[error("KV sequence {0} is not active")]
    NotActive(u64),
    #[error(
        "unified KV pool needs {needed_bytes} bytes but its global budget is {budget_bytes} bytes"
    )]
    BudgetExceeded {
        needed_bytes: u64,
        budget_bytes: u64,
    },
    #[error("KV sequence {0} is active and cannot be removed")]
    ActiveSequence(u64),
    #[error("KV cache operation failed: {0}")]
    Cache(String),
}

struct PoolEntry {
    cache: LlamaKvCache,
    state: KvSequenceState,
    last_used: u64,
}

pub struct UnifiedKvCachePool {
    entries: HashMap<KvSequenceId, PoolEntry>,
    budget_bytes: u64,
    next_id: u64,
    clock: u64,
    evictions: u64,
}

impl UnifiedKvCachePool {
    pub fn from_env() -> Self {
        Self::new(crate::runtime_config::kv_pool_budget_bytes())
    }

    pub fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            budget_bytes: budget_bytes.max(1),
            next_id: 1,
            clock: 0,
            evictions: 0,
        }
    }

    pub fn insert(&mut self, cache: LlamaKvCache) -> Result<KvSequenceId, KvPoolError> {
        let needed = cache.allocated_bytes();
        self.make_room(needed, None)?;
        let id = KvSequenceId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let last_used = self.tick();
        self.entries.insert(
            id,
            PoolEntry {
                cache,
                state: KvSequenceState::Idle,
                last_used,
            },
        );
        Ok(id)
    }

    pub fn activate(&mut self, id: KvSequenceId) -> Result<(), KvPoolError> {
        let last_used = self.tick();
        let entry = self.entry_mut(id)?;
        if entry.state == KvSequenceState::Active {
            return Err(KvPoolError::AlreadyActive(id.0));
        }
        entry.state = KvSequenceState::Active;
        entry.last_used = last_used;
        Ok(())
    }

    pub fn deactivate(&mut self, id: KvSequenceId) -> Result<(), KvPoolError> {
        let last_used = self.tick();
        let entry = self.entry_mut(id)?;
        if entry.state != KvSequenceState::Active {
            return Err(KvPoolError::NotActive(id.0));
        }
        entry.state = KvSequenceState::Idle;
        entry.last_used = last_used;
        Ok(())
    }

    /// Reserve one sequence's cache under the pool-wide budget. Idle LRU
    /// sequences may be evicted, but the target and every active sequence are
    /// protected.
    pub fn reserve(
        &mut self,
        id: KvSequenceId,
        required_sequence_length: usize,
    ) -> Result<(), KvPoolError> {
        let current = self
            .entries
            .get(&id)
            .ok_or(KvPoolError::UnknownSequence(id.0))?
            .cache
            .allocated_bytes();
        let projected = self
            .entries
            .get(&id)
            .expect("checked above")
            .cache
            .projected_allocated_bytes(required_sequence_length)
            .map_err(|error| KvPoolError::Cache(error.to_string()))?;
        let additional = projected.saturating_sub(current);
        self.make_room(additional, Some(id))?;
        let last_used = self.tick();
        let entry = self.entry_mut(id)?;
        entry
            .cache
            .ensure_position_capacity(required_sequence_length)
            .map_err(|error| KvPoolError::Cache(error.to_string()))?;
        entry.last_used = last_used;
        Ok(())
    }

    pub fn checkpoint(&mut self, id: KvSequenceId) -> Result<KvSequenceCheckpoint, KvPoolError> {
        let last_used = self.tick();
        let entry = self.entry_mut(id)?;
        entry.last_used = last_used;
        Ok(KvSequenceCheckpoint {
            sequence_id: id,
            position: entry.cache.position,
        })
    }

    pub fn rollback(&mut self, checkpoint: KvSequenceCheckpoint) -> Result<(), KvPoolError> {
        let last_used = self.tick();
        let entry = self.entry_mut(checkpoint.sequence_id)?;
        entry
            .cache
            .rollback_to_position(checkpoint.position)
            .map_err(|error| KvPoolError::Cache(error.to_string()))?;
        entry.last_used = last_used;
        Ok(())
    }

    pub fn cache(&self, id: KvSequenceId) -> Result<&LlamaKvCache, KvPoolError> {
        self.entries
            .get(&id)
            .map(|entry| &entry.cache)
            .ok_or(KvPoolError::UnknownSequence(id.0))
    }

    pub fn cache_mut(&mut self, id: KvSequenceId) -> Result<&mut LlamaKvCache, KvPoolError> {
        let last_used = self.tick();
        let entry = self.entry_mut(id)?;
        entry.last_used = last_used;
        Ok(&mut entry.cache)
    }

    pub fn remove(&mut self, id: KvSequenceId) -> Result<LlamaKvCache, KvPoolError> {
        let entry = self
            .entries
            .get(&id)
            .ok_or(KvPoolError::UnknownSequence(id.0))?;
        if entry.state == KvSequenceState::Active {
            return Err(KvPoolError::ActiveSequence(id.0));
        }
        Ok(self.entries.remove(&id).expect("checked above").cache)
    }

    pub fn snapshot(&self) -> KvPoolSnapshot {
        KvPoolSnapshot {
            sequence_count: self.entries.len(),
            active_sequences: self
                .entries
                .values()
                .filter(|entry| entry.state == KvSequenceState::Active)
                .count(),
            allocated_bytes: self.allocated_bytes(),
            budget_bytes: self.budget_bytes,
            evictions: self.evictions,
        }
    }

    fn entry_mut(&mut self, id: KvSequenceId) -> Result<&mut PoolEntry, KvPoolError> {
        self.entries
            .get_mut(&id)
            .ok_or(KvPoolError::UnknownSequence(id.0))
    }

    fn allocated_bytes(&self) -> u64 {
        self.entries
            .values()
            .map(|entry| entry.cache.allocated_bytes())
            .fold(0u64, u64::saturating_add)
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn make_room(
        &mut self,
        additional_bytes: u64,
        protected: Option<KvSequenceId>,
    ) -> Result<(), KvPoolError> {
        loop {
            let needed = self.allocated_bytes().saturating_add(additional_bytes);
            if needed <= self.budget_bytes {
                return Ok(());
            }
            let victim = self
                .entries
                .iter()
                .filter(|(id, entry)| {
                    Some(**id) != protected && entry.state == KvSequenceState::Idle
                })
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| *id);
            let Some(victim) = victim else {
                return Err(KvPoolError::BudgetExceeded {
                    needed_bytes: needed,
                    budget_bytes: self.budget_bytes,
                });
            };
            self.entries.remove(&victim);
            self.evictions = self.evictions.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{KvDtype, KvLayout, LlamaKvCachePlan};

    fn cache() -> LlamaKvCache {
        let shape = vec![1, 32, 1, 4];
        LlamaKvCache::new_with_layout_and_dtype(
            LlamaKvCachePlan {
                max_sequence_length: 32,
                layer_count: 1,
                kv_head_count: 1,
                head_dim: 4,
                k_head_dim: 4,
                v_head_dim: 4,
                key_shape: shape.clone(),
                value_shape: shape,
            },
            KvLayout::PositionMajor,
            KvDtype::F32,
        )
        .unwrap()
    }

    #[test]
    fn pool_enforces_global_budget_and_evicts_only_idle_sequences() {
        // 1 layer * 1 head * 4 values * K+V * 4 bytes = 32 bytes/token.
        let mut pool = UnifiedKvCachePool::new(64);
        let first = pool.insert(cache()).unwrap();
        let second = pool.insert(cache()).unwrap();
        pool.reserve(first, 1).unwrap();
        pool.reserve(second, 1).unwrap();
        pool.activate(first).unwrap();

        let third = pool.insert(cache()).unwrap();
        pool.reserve(third, 1).unwrap();
        assert!(pool.cache(first).is_ok(), "active cache must survive");
        assert!(
            pool.cache(second).is_err(),
            "oldest idle cache should be evicted"
        );
        assert!(pool.cache(third).is_ok());
        assert_eq!(pool.snapshot().allocated_bytes, 64);

        let fourth = pool.insert(cache()).unwrap();
        pool.activate(third).unwrap();
        assert!(matches!(
            pool.reserve(fourth, 1),
            Err(KvPoolError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn checkpoint_rollback_is_sequence_local() {
        let mut pool = UnifiedKvCachePool::new(4096);
        let first = pool.insert(cache()).unwrap();
        let second = pool.insert(cache()).unwrap();
        pool.reserve(first, 16).unwrap();
        pool.reserve(second, 16).unwrap();
        pool.cache_mut(first).unwrap().position = 8;
        pool.cache_mut(first).unwrap().materialized_through = 8;
        pool.cache_mut(second).unwrap().position = 7;
        let checkpoint = pool.checkpoint(first).unwrap();
        pool.cache_mut(first).unwrap().position = 12;
        pool.cache_mut(first).unwrap().materialized_through = 12;

        pool.rollback(checkpoint).unwrap();
        assert_eq!(pool.cache(first).unwrap().position, 8);
        assert_eq!(pool.cache(first).unwrap().materialized_through, 8);
        assert_eq!(pool.cache(second).unwrap().position, 7);
    }
}
