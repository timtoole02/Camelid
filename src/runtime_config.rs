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
pub const CUDA_SEQUENCE_SLOTS_ENV: &str = "CAMELID_CUDA_SEQUENCE_SLOTS";
pub const CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV: &str = "CAMELID_CUDA_BATCHED_PREFILL_ROUND_TOKENS";
pub const KQUANT_PREFILL_OWNER_ENV: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER";
pub const MOE_EXPERT_STORAGE_ENV: &str = "CAMELID_MOE_EXPERT_STORAGE";
pub const MIXTRAL_LONG_GENERATION_ENV: &str = "CAMELID_MIXTRAL_LONG_GENERATION";

pub const DEFAULT_ENGINE_QUEUE_DEPTH: usize = 8;
pub const DEFAULT_NGRAM_INDEX_MAX_ENTRIES: usize = 131_072;
pub const DEFAULT_KV_POOL_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Two streaming sessions may take turns out of the box. The engine still owns one
/// compute thread and runs one token step at a time; a lone stream keeps the ordinary
/// single-request fast path. Set CAMELID_CONTINUOUS_BATCH_SLOTS=1 to force legacy
/// run-to-completion scheduling.
pub const DEFAULT_CONTINUOUS_BATCH_SLOTS: usize = 2;

const MAX_ENGINE_QUEUE_DEPTH: usize = 65_536;
const MAX_NGRAM_INDEX_ENTRIES: usize = 4_194_304;
const MAX_CONTINUOUS_BATCH_SLOTS: usize = 256;

/// Storage policy for the very large rank-3 expert tensors used by MoE models.
///
/// File-backed experts preserve Camelid's low-memory default. Resident Q8 keeps
/// the quantized blocks in RAM to remove per-token disk reads, but is admitted
/// only after the loader's live-memory preflight succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpertStorage {
    FileBacked,
    ResidentQ8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub engine_queue_depth: usize,
    pub ngram_index_max_entries: usize,
    pub kv_pool_budget_bytes: u64,
    pub continuous_batch_slots: usize,
    pub kquant_prefill_owner: bool,
    pub moe_expert_storage: MoeExpertStorage,
    pub mixtral_long_generation: bool,
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
            kquant_prefill_owner: kquant_prefill_owner_default(),
            moe_expert_storage: moe_expert_storage_from_env(),
            mixtral_long_generation: env_flag_default_off(MIXTRAL_LONG_GENERATION_ENV),
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

pub fn cuda_sequence_slots_override() -> Option<usize> {
    env::var(CUDA_SEQUENCE_SLOTS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| matches!(value, 1 | 2 | 4 | 8))
}

pub fn cuda_sequence_capacity(legacy_two_slots: bool, paged_kv: bool) -> usize {
    let requested = cuda_sequence_slots_override().unwrap_or(if legacy_two_slots { 2 } else { 1 });
    if requested > 2 && !paged_kv {
        1
    } else {
        requested
    }
}

pub fn cuda_batched_prefill_round_tokens() -> usize {
    env::var(CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| matches!(value, 1 | 4 | 16))
        .unwrap_or(16)
}

pub fn kquant_prefill_owner_enabled() -> bool {
    RuntimeConfig::from_env().kquant_prefill_owner
}

pub fn moe_expert_storage() -> MoeExpertStorage {
    RuntimeConfig::from_env().moe_expert_storage
}

pub fn mixtral_long_generation_enabled() -> bool {
    RuntimeConfig::from_env().mixtral_long_generation
}

fn moe_expert_storage_from_env() -> MoeExpertStorage {
    match env::var(MOE_EXPERT_STORAGE_ENV)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("resident") | Some("resident_q8") | Some("wire") => MoeExpertStorage::ResidentQ8,
        Some("file") | Some("file_backed") | Some("lazy") | None => MoeExpertStorage::FileBacked,
        Some(_) => MoeExpertStorage::FileBacked,
    }
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

/// A shipped default that only an explicit, recognised value can switch off, so
/// a typo cannot silently disable something that was promoted on measurement.
pub fn env_flag_default_on(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let value = value.trim();
            !(value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("no")
                || value.eq_ignore_ascii_case("disabled"))
        })
        .unwrap_or(true)
}

/// Default-on only for `win-x86_64`, the platform the promotion receipt
/// measured. Every other target keeps the owner opt-in until one is measured.
fn kquant_prefill_owner_default() -> bool {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        env_flag_default_on(KQUANT_PREFILL_OWNER_ENV)
    } else {
        env_flag_default_off(KQUANT_PREFILL_OWNER_ENV)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase7_cuda_sequence_slots_accept_only_promoted_capacities() {
        let _guard = crate::test_support::env_lock();
        env::remove_var(CUDA_SEQUENCE_SLOTS_ENV);
        assert_eq!(cuda_sequence_slots_override(), None);

        for capacity in [1usize, 2, 4, 8] {
            env::set_var(CUDA_SEQUENCE_SLOTS_ENV, capacity.to_string());
            assert_eq!(cuda_sequence_slots_override(), Some(capacity));
        }
        for invalid in ["0", "3", "5", "9", "256", "not-a-number", ""] {
            env::set_var(CUDA_SEQUENCE_SLOTS_ENV, invalid);
            assert_eq!(
                cuda_sequence_slots_override(),
                None,
                "{invalid:?} must fail closed"
            );
        }
        env::remove_var(CUDA_SEQUENCE_SLOTS_ENV);
    }

    #[test]
    fn phase7_cuda_sequence_capacity_preserves_legacy_and_requires_paging_above_two() {
        let _guard = crate::test_support::env_lock();
        env::remove_var(CUDA_SEQUENCE_SLOTS_ENV);
        assert_eq!(cuda_sequence_capacity(false, false), 1);
        assert_eq!(cuda_sequence_capacity(true, false), 2);

        env::set_var(CUDA_SEQUENCE_SLOTS_ENV, "1");
        assert_eq!(cuda_sequence_capacity(true, true), 1);
        env::set_var(CUDA_SEQUENCE_SLOTS_ENV, "2");
        assert_eq!(cuda_sequence_capacity(false, false), 2);
        env::set_var(CUDA_SEQUENCE_SLOTS_ENV, "4");
        assert_eq!(cuda_sequence_capacity(false, false), 1);
        assert_eq!(cuda_sequence_capacity(false, true), 4);
        env::set_var(CUDA_SEQUENCE_SLOTS_ENV, "8");
        assert_eq!(cuda_sequence_capacity(false, true), 8);

        env::set_var(CUDA_SEQUENCE_SLOTS_ENV, "invalid");
        assert_eq!(cuda_sequence_capacity(true, true), 2);
        assert_eq!(cuda_sequence_capacity(false, true), 1);
        env::remove_var(CUDA_SEQUENCE_SLOTS_ENV);
    }

    #[test]
    fn phase7_batched_prefill_quantum_accepts_only_preregistered_values() {
        let _guard = crate::test_support::env_lock();
        env::remove_var(CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV);
        assert_eq!(cuda_batched_prefill_round_tokens(), 16);
        for candidate in [1usize, 4, 16] {
            env::set_var(CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV, candidate.to_string());
            assert_eq!(cuda_batched_prefill_round_tokens(), candidate);
        }
        for invalid in ["0", "2", "8", "17", "invalid"] {
            env::set_var(CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV, invalid);
            assert_eq!(cuda_batched_prefill_round_tokens(), 16);
        }
        env::remove_var(CUDA_BATCHED_PREFILL_ROUND_TOKENS_ENV);
    }

    #[test]
    fn invalid_and_out_of_range_values_fail_closed() {
        let _guard = crate::test_support::env_lock();
        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
            MOE_EXPERT_STORAGE_ENV,
            MIXTRAL_LONG_GENERATION_ENV,
        ] {
            env::remove_var(key);
        }
        env::set_var(ENGINE_QUEUE_DEPTH_ENV, "0");
        env::set_var(NGRAM_INDEX_MAX_ENTRIES_ENV, "not-a-number");
        env::set_var(KV_POOL_BUDGET_BYTES_ENV, "0");
        env::set_var(CONTINUOUS_BATCH_SLOTS_ENV, "999");
        env::set_var(KQUANT_PREFILL_OWNER_ENV, "maybe");
        env::set_var(MOE_EXPERT_STORAGE_ENV, "surprise-me");
        env::set_var(MIXTRAL_LONG_GENERATION_ENV, "maybe");

        assert_eq!(
            RuntimeConfig::from_env(),
            RuntimeConfig {
                engine_queue_depth: DEFAULT_ENGINE_QUEUE_DEPTH,
                ngram_index_max_entries: DEFAULT_NGRAM_INDEX_MAX_ENTRIES,
                kv_pool_budget_bytes: DEFAULT_KV_POOL_BUDGET_BYTES,
                continuous_batch_slots: DEFAULT_CONTINUOUS_BATCH_SLOTS,
                // Unrecognised is not "off" for this one: it is default-on where the
                // promotion was measured, and a typo must not disable that.
                kquant_prefill_owner: cfg!(all(target_os = "windows", target_arch = "x86_64")),
                moe_expert_storage: MoeExpertStorage::FileBacked,
                mixtral_long_generation: false,
            }
        );

        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
            MOE_EXPERT_STORAGE_ENV,
            MIXTRAL_LONG_GENERATION_ENV,
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
        env::set_var(MOE_EXPERT_STORAGE_ENV, "resident_q8");
        env::set_var(MIXTRAL_LONG_GENERATION_ENV, "true");

        assert_eq!(
            RuntimeConfig::from_env(),
            RuntimeConfig {
                engine_queue_depth: 17,
                ngram_index_max_entries: 2048,
                kv_pool_budget_bytes: 4096,
                continuous_batch_slots: 4,
                kquant_prefill_owner: true,
                moe_expert_storage: MoeExpertStorage::ResidentQ8,
                mixtral_long_generation: true,
            }
        );

        for key in [
            ENGINE_QUEUE_DEPTH_ENV,
            NGRAM_INDEX_MAX_ENTRIES_ENV,
            KV_POOL_BUDGET_BYTES_ENV,
            CONTINUOUS_BATCH_SLOTS_ENV,
            KQUANT_PREFILL_OWNER_ENV,
            MOE_EXPERT_STORAGE_ENV,
            MIXTRAL_LONG_GENERATION_ENV,
        ] {
            env::remove_var(key);
        }
    }

    /// The owner ships on for win-x86_64, so the operator needs a switch that
    /// actually restores the previous path -- and it has to be an explicit word,
    /// not any stray value.
    #[test]
    fn kquant_owner_default_is_scoped_and_can_be_switched_off() {
        let _guard = crate::test_support::env_lock();
        env::remove_var(KQUANT_PREFILL_OWNER_ENV);
        let promoted = cfg!(all(target_os = "windows", target_arch = "x86_64"));
        assert_eq!(RuntimeConfig::from_env().kquant_prefill_owner, promoted);

        for off in ["0", "false", "off", "no", "disabled", "OFF", " off "] {
            env::set_var(KQUANT_PREFILL_OWNER_ENV, off);
            assert!(
                !RuntimeConfig::from_env().kquant_prefill_owner,
                "{off:?} must switch the owner off on every target"
            );
        }
        for on in ["1", "true", "on", "yes", "enabled"] {
            env::set_var(KQUANT_PREFILL_OWNER_ENV, on);
            assert!(
                RuntimeConfig::from_env().kquant_prefill_owner,
                "{on:?} must switch the owner on on every target"
            );
        }
        env::remove_var(KQUANT_PREFILL_OWNER_ENV);
    }
}
