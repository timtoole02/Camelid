//! Typed configuration for the runtime-foundation work.
//!
//! Camelid historically accumulated one-off environment reads near each hot
//! path.  New runtime features resolve through this module so defaults,
//! bounds, and aliases are explicit and independently testable.  Invalid
//! values fail closed to the documented default.

use std::env;

pub const ENGINE_QUEUE_DEPTH_ENV: &str = "CAMELID_QUEUE_DEPTH";
pub const NGRAM_INDEX_MAX_ENTRIES_ENV: &str = "CAMELID_NGRAM_INDEX_MAX_ENTRIES";
pub const KV_POOL_BUDGET_BYTES_ENV: &str = "CAMELID_KV_POOL_BUDGET_BYTES";
pub const CONTINUOUS_BATCH_SLOTS_ENV: &str = "CAMELID_CONTINUOUS_BATCH_SLOTS";
pub const KQUANT_PREFILL_OWNER_ENV: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER";

pub const DEFAULT_ENGINE_QUEUE_DEPTH: usize = 8;
pub const DEFAULT_NGRAM_INDEX_MAX_ENTRIES: usize = 131_072;
pub const DEFAULT_KV_POOL_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const DEFAULT_CONTINUOUS_BATCH_SLOTS: usize = 1;

const MAX_ENGINE_QUEUE_DEPTH: usize = 65_536;
const MAX_NGRAM_INDEX_ENTRIES: usize = 4_194_304;
const MAX_CONTINUOUS_BATCH_SLOTS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub engine_queue_depth: usize,
    pub ngram_index_max_entries: usize,
    pub kv_pool_budget_bytes: u64,
    pub continuous_batch_slots: usize,
    pub kquant_prefill_owner: bool,
}

impl RuntimeConfig {
    pub fn from_env() -> Self {
        Self {
            engine_queue_depth: bounded_usize(
                ENGINE_QUEUE_DEPTH_ENV,
                DEFAULT_ENGINE_QUEUE_DEPTH,
                1,
                MAX_ENGINE_QUEUE_DEPTH,
            ),
            ngram_index_max_entries: bounded_usize(
                NGRAM_INDEX_MAX_ENTRIES_ENV,
                DEFAULT_NGRAM_INDEX_MAX_ENTRIES,
                1,
                MAX_NGRAM_INDEX_ENTRIES,
            ),
            kv_pool_budget_bytes: bounded_u64(
                KV_POOL_BUDGET_BYTES_ENV,
                DEFAULT_KV_POOL_BUDGET_BYTES,
                1,
                u64::MAX,
            ),
            continuous_batch_slots: bounded_usize(
                CONTINUOUS_BATCH_SLOTS_ENV,
                DEFAULT_CONTINUOUS_BATCH_SLOTS,
                1,
                MAX_CONTINUOUS_BATCH_SLOTS,
            ),
            kquant_prefill_owner: env_flag_default_off(KQUANT_PREFILL_OWNER_ENV),
        }
    }
}

pub fn engine_queue_depth() -> usize {
    RuntimeConfig::from_env().engine_queue_depth
}

pub fn ngram_index_max_entries() -> usize {
    RuntimeConfig::from_env().ngram_index_max_entries
}

pub fn kv_pool_budget_bytes() -> u64 {
    RuntimeConfig::from_env().kv_pool_budget_bytes
}

pub fn continuous_batch_slots() -> usize {
    RuntimeConfig::from_env().continuous_batch_slots
}

pub fn kquant_prefill_owner_enabled() -> bool {
    RuntimeConfig::from_env().kquant_prefill_owner
}

fn bounded_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| (min..=max).contains(value))
        .unwrap_or(default)
}

fn bounded_u64(name: &str, default: u64, min: u64, max: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| (min..=max).contains(value))
        .unwrap_or(default)
}

pub fn env_flag_default_off(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_and_out_of_range_values_fail_closed() {
        let _guard = crate::test_support::env_lock();
        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
        ] {
            env::remove_var(key);
        }
        env::set_var(ENGINE_QUEUE_DEPTH_ENV, "0");
        env::set_var(NGRAM_INDEX_MAX_ENTRIES_ENV, "not-a-number");
        env::set_var(KV_POOL_BUDGET_BYTES_ENV, "0");
        env::set_var(CONTINUOUS_BATCH_SLOTS_ENV, "999");
        env::set_var(KQUANT_PREFILL_OWNER_ENV, "maybe");

        assert_eq!(
            RuntimeConfig::from_env(),
            RuntimeConfig {
                engine_queue_depth: DEFAULT_ENGINE_QUEUE_DEPTH,
                ngram_index_max_entries: DEFAULT_NGRAM_INDEX_MAX_ENTRIES,
                kv_pool_budget_bytes: DEFAULT_KV_POOL_BUDGET_BYTES,
                continuous_batch_slots: DEFAULT_CONTINUOUS_BATCH_SLOTS,
                kquant_prefill_owner: false,
            }
        );

        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
        ] {
            env::remove_var(key);
        }
    }

    #[test]
    fn valid_values_resolve_to_typed_fields() {
        let _guard = crate::test_support::env_lock();
        env::set_var(ENGINE_QUEUE_DEPTH_ENV, "17");
        env::set_var(NGRAM_INDEX_MAX_ENTRIES_ENV, "2048");
        env::set_var(KV_POOL_BUDGET_BYTES_ENV, "4096");
        env::set_var(CONTINUOUS_BATCH_SLOTS_ENV, "4");
        env::set_var(KQUANT_PREFILL_OWNER_ENV, "yes");

        assert_eq!(
            RuntimeConfig::from_env(),
            RuntimeConfig {
                engine_queue_depth: 17,
                ngram_index_max_entries: 2048,
                kv_pool_budget_bytes: 4096,
                continuous_batch_slots: 4,
                kquant_prefill_owner: true,
            }
        );

        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
        ] {
            env::remove_var(key);
        }
    }
}
