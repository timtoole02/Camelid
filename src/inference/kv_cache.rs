use std::env;

use serde::Serialize;

use crate::{
    model::{DenseLlamaDims, LlamaModelConfig},
    tensor::kv_quant::{
        dequantize_row_q4_0, dequantize_row_q8_0, quantize_row_q4_0, quantize_row_q8_0, BlockQ4_0,
        BlockQ8_0, KV_QUANT_BLOCK_VALUES,
    },
    BackendError, Result,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaKvCachePlan {
    pub max_sequence_length: usize,
    pub layer_count: usize,
    pub kv_head_count: usize,
    /// Backward-compatible alias for the standard attention head width.
    ///
    /// For MLA plans this is the compressed key width; callers that need
    /// asymmetric key/value widths must use `k_head_dim` and `v_head_dim`.
    pub head_dim: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
}

impl LlamaKvCachePlan {
    pub fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let dims = DenseLlamaDims::from_config(config)?;
        let max_sequence_length = config.context_length as usize;
        let (kv_head_count, k_head_dim, v_head_dim, key_shape, value_shape) =
            if let Some(mla) = &config.mla {
                // For DeepSeek MLA, we store the compressed latent c_KV + k_pe in the keys buffer.
                // We treat it as a single "KV head" of width (kv_lora_rank + rope_head_dim).
                // The values buffer is left empty (width 0), but the logical output v_head_dim is kv_lora_rank.
                let k_head_dim = (mla.kv_lora_rank + mla.rope_head_dim) as usize;
                let v_head_dim = mla.kv_lora_rank as usize;
                let key_shape = vec![dims.block_count, max_sequence_length, 1, k_head_dim];
                let value_shape = vec![dims.block_count, max_sequence_length, 1, 0];
                (1, k_head_dim, v_head_dim, key_shape, value_shape)
            } else {
                let k_head_dim = dims.head_dim;
                let v_head_dim = dims.head_dim;
                let key_shape = vec![
                    dims.block_count,
                    max_sequence_length,
                    dims.attention_head_count_kv,
                    k_head_dim,
                ];
                let value_shape = key_shape.clone();
                (
                    dims.attention_head_count_kv,
                    k_head_dim,
                    v_head_dim,
                    key_shape,
                    value_shape,
                )
            };

        Ok(Self {
            max_sequence_length,
            layer_count: dims.block_count,
            kv_head_count,
            head_dim: k_head_dim,
            k_head_dim,
            v_head_dim,
            key_shape,
            value_shape,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaKvCacheTrace {
    pub layer_index: usize,
    pub position_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub key_value_width: usize,
    pub key_checksum: f64,
    pub value_checksum: f64,
    pub key_rms: f32,
    pub value_rms: f32,
    pub key_max_abs: f32,
    pub key_max_abs_position: usize,
    pub key_max_abs_index: usize,
    pub value_max_abs: f32,
    pub value_max_abs_position: usize,
    pub value_max_abs_index: usize,
    pub sampled_positions: Vec<LlamaKvCachePositionTrace>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaKvCachePositionTrace {
    pub position: usize,
    pub key_checksum: f64,
    pub value_checksum: f64,
    pub key_rms: f32,
    pub value_rms: f32,
    pub key_max_abs: f32,
    pub value_max_abs: f32,
    pub key_first_values: Vec<f32>,
    pub value_first_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LlamaKvCache {
    pub plan: LlamaKvCachePlan,
    /// Physical K/V arrangement; see [`KvLayout`]. Fixed at construction.
    pub layout: KvLayout,
    /// Element storage type; see [`KvDtype`]. Fixed at construction. In
    /// `F32` mode `keys`/`values` are live and the `_f16` buffers stay
    /// empty; in `F16` mode the reverse. The stored VALUES are identical
    /// either way — the write path has always rounded through f16 — so the
    /// dtype choice is bytes, not bits.
    pub dtype: KvDtype,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    /// f16 storage (bit patterns), live only when `dtype == KvDtype::F16`.
    pub keys_f16: Vec<u16>,
    pub values_f16: Vec<u16>,
    pub keys_q8_0: Vec<BlockQ8_0>,
    pub values_q8_0: Vec<BlockQ8_0>,
    pub keys_q4_0: Vec<BlockQ4_0>,
    pub values_q4_0: Vec<BlockQ4_0>,
    pub allocated_sequence_length: usize,
    pub position: usize,
    /// Materialized-through watermark: positions `[0, materialized_through)` have had
    /// their K/V actually WRITTEN by a CPU forward, a prefill, or a GPU→host mirror —
    /// as opposed to merely being addressable. `allocated_sequence_length` answers "are
    /// these bytes safe to index"; this answers "do these bytes hold this sequence's
    /// history". They diverge exactly when a GPU-resident engine produced positions the
    /// CPU never wrote: `ensure_position_capacity` then grows the buffers zero-filled and
    /// every bounds probe starts passing over a history that is all zeros.
    ///
    /// Maintained by [`store_kv_head_row`](Self::store_kv_head_row) (the canonical store
    /// every writer routes through) and clamped down by
    /// [`rollback_to_position`](Self::rollback_to_position) — a rollback must not leave the
    /// watermark vouching for rows a rejected speculative branch wrote, which are stale
    /// rather than zero and therefore even harder to spot.
    ///
    /// GRANULARITY: bumped per written row, so DURING a layer loop it leads by the layers
    /// of the current token not yet written. Every consumer reads it BETWEEN forward passes,
    /// where it is exactly token-complete.
    pub materialized_through: usize,
    /// Max f32 K+V bytes this session may materialize before the predict-and-abort
    /// guard in `ensure_position_capacity` refuses. Host-derived operational config
    /// (env / available RAM), NOT cache state — excluded from `PartialEq`. `pub(super)`
    /// so sibling session code can carry it onto a hollow placeholder cache without
    /// re-resolving (which would re-query host RAM every step).
    pub(super) kv_budget_bytes: u64,
}

impl PartialEq for LlamaKvCache {
    fn eq(&self, other: &Self) -> bool {
        // Cache STATE only. `kv_budget_bytes` is host-derived operational config and
        // must not affect equality (the session PartialEq compares caches across runs).
        self.plan == other.plan
            && self.layout == other.layout
            && self.dtype == other.dtype
            && self.keys == other.keys
            && self.values == other.values
            && self.keys_f16 == other.keys_f16
            && self.values_f16 == other.values_f16
            && self.keys_q8_0 == other.keys_q8_0
            && self.values_q8_0 == other.values_q8_0
            && self.keys_q4_0 == other.keys_q4_0
            && self.values_q4_0 == other.values_q4_0
            && self.allocated_sequence_length == other.allocated_sequence_length
            && self.position == other.position
            && self.materialized_through == other.materialized_through
    }
}

/// Physical arrangement of the K/V buffers. Chosen ONCE at cache
/// construction (`CAMELID_KV_LAYOUT_HEAD_MAJOR`, default off) and
/// carried on the cache; every element address goes through the layout-aware
/// accessors below, so the choice never appears in hot loops as more than a
/// resolved stride. Values and arithmetic are identical in both layouts —
/// only addresses differ — so the head-major lane carries a bitwise-identity
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// `[position, layer, kv_head, head_dim]` — the historical layout; one
    /// token's row is contiguous, one head's positions sit a full token
    /// stride apart.
    PositionMajor,
    /// `[layer, kv_head, position, head_dim]` — each head's K/V is one
    /// contiguous stream over positions (stride = `allocated_sequence_length
    /// * head_dim`, re-laid-out on growth); decode reads become sequential.
    HeadMajor,
}

/// Env gate for the head-major layout, read once per cache construction.
/// Windows-first per the standing directive; the mechanism is arch-agnostic
/// and lifting the gate is a one-line decision.
///
/// DEFAULT ON (Windows x86_64 promotion): the lane carries a bitwise-identity
/// contract (Item-3 Lane-A matrix incl. rollback/growth/CUDA-mirror, zero
/// divergent bits), so the flip cannot change any output byte. Explicit
/// rollback: `CAMELID_KV_LAYOUT_HEAD_MAJOR=0`.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn kv_layout_head_major_enabled() -> bool {
    super::q8_runtime::q8_0_env_flag_enabled_default_on_fail_closed("CAMELID_KV_LAYOUT_HEAD_MAJOR")
}

/// Element storage for the K/V buffers. Chosen ONCE at cache construction
/// (`CAMELID_KV_F16`, default off). f16 storage holds exactly the
/// values the write path has always produced (it rounds through f16
/// unconditionally), so both dtypes carry the bitwise-identity contract —
/// f16 just stops paying 2x the bytes for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F32,
    F16,
    Q8_0,
    Q4_0,
}

/// Env gate for f16 storage, read once per cache construction. Requires the
/// Item-1 blocked-dot lane (the fused f16 kernels realize its canonical
/// order; the legacy serial dot has no f16 variant by design) — requested
/// without it, the flag is inert and logged once.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn kv_f16_enabled() -> bool {
    let requested = super::q8_runtime::q8_0_env_flag_enabled_default_off("CAMELID_KV_F16");
    if requested && !super::attention_f32_blocked_dot_enabled() {
        static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!(
                "[kv-f16] CAMELID_KV_F16 requested without \
                 CAMELID_ATTENTION_F32_BLOCKED_DOT; the f16 lane is inert \
                 (the fused f16 kernels require the blocked-dot lane)"
            );
        });
        return false;
    }
    requested
}

impl LlamaKvCache {
    pub fn new(plan: LlamaKvCachePlan, quant: crate::model::KvCacheQuantization) -> Result<Self> {
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        let (layout, dtype) = (
            if kv_layout_head_major_enabled() {
                KvLayout::HeadMajor
            } else {
                KvLayout::PositionMajor
            },
            match quant {
                crate::model::KvCacheQuantization::Q8_0 => KvDtype::Q8_0,
                crate::model::KvCacheQuantization::Q4_0 => KvDtype::Q4_0,
                crate::model::KvCacheQuantization::F16 => {
                    if kv_f16_enabled() {
                        KvDtype::F16
                    } else {
                        KvDtype::F32
                    }
                }
            },
        );
        #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
        let (layout, dtype) = (
            KvLayout::PositionMajor,
            match quant {
                crate::model::KvCacheQuantization::Q8_0 => KvDtype::Q8_0,
                crate::model::KvCacheQuantization::Q4_0 => KvDtype::Q4_0,
                _ => KvDtype::F32,
            },
        );
        Self::new_with_layout_and_dtype(plan, layout, dtype)
    }

    /// Explicit-layout constructor so the bitwise-identity tests can build
    /// both layouts without touching process env.
    pub fn new_with_layout(plan: LlamaKvCachePlan, layout: KvLayout) -> Result<Self> {
        Self::new_with_layout_and_dtype(plan, layout, KvDtype::F32)
    }

    /// Explicit-everything constructor for the bitwise-identity tests.
    pub fn new_with_layout_and_dtype(
        plan: LlamaKvCachePlan,
        layout: KvLayout,
        dtype: KvDtype,
    ) -> Result<Self> {
        Ok(Self {
            plan,
            layout,
            dtype,
            keys: Vec::new(),
            values: Vec::new(),
            keys_f16: Vec::new(),
            values_f16: Vec::new(),
            keys_q8_0: Vec::new(),
            values_q8_0: Vec::new(),
            keys_q4_0: Vec::new(),
            values_q4_0: Vec::new(),
            allocated_sequence_length: 0,
            position: 0,
            materialized_through: 0,
            kv_budget_bytes: resolve_kv_cache_budget_bytes(),
        })
    }

    pub fn can_append(&self) -> bool {
        self.position < self.plan.max_sequence_length
    }

    /// Roll the cache back to an earlier `position`, discarding newer
    /// entries. In both layouts `position` alone bounds what attention
    /// reads, so entries past the rollback point are dead until overwritten
    /// by later appends — no buffer work is needed. Used by speculative
    /// decoding to drop rejected draft tokens.
    pub fn rollback_to_position(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV rollback target {position} is beyond current position {}",
                self.position
            )));
        }
        self.position = position;
        // Clamp the watermark with the position. Rows past the rollback point were written
        // by a branch that has been rejected: they are STALE, not zero, so leaving the
        // watermark above `position` would let a later GPU reseed copy a dead speculative
        // branch's K/V into the cache and call it history.
        self.materialized_through = self.materialized_through.min(position);
        Ok(())
    }

    pub(super) fn ensure_position_capacity(
        &mut self,
        required_sequence_length: usize,
    ) -> Result<()> {
        if required_sequence_length > self.plan.max_sequence_length {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV cache position {required_sequence_length} exceeds context length {}",
                self.plan.max_sequence_length
            )));
        }
        if required_sequence_length <= self.allocated_sequence_length {
            return Ok(());
        }
        let target_sequence_length = self.grow_sequence_length(required_sequence_length);
        // Predict-and-abort on host memory (conductor §9): the f32 K+V cache is the
        // dominant, otherwise-uncapped allocation. Project the bytes of the ACTUAL
        // (post-rounding) growth and refuse BEFORE the `resize` below — the host ceiling
        // must never be discovered by OOMing mid-generation. Budget is
        // `CAMELID_MAX_KV_CACHE_BYTES`, else max(80% of available, 25% of total) physical
        // RAM (Windows); unbounded where neither is known.
        let projected_bytes =
            (target_sequence_length as u64).saturating_mul(self.kv_bytes_per_token());
        if projected_bytes > self.kv_budget_bytes {
            return Err(BackendError::KvCacheBudgetExceeded {
                positions: target_sequence_length,
                needed_bytes: projected_bytes,
                budget_bytes: self.kv_budget_bytes,
            });
        }
        let row_count = target_sequence_length
            .checked_mul(self.plan.layer_count)
            .and_then(|value| value.checked_mul(self.plan.kv_head_count))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("KV cache row count overflow".to_string())
            })?;
        let k_elements = row_count
            .checked_mul(self.plan.key_shape[3])
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "KV cache key element count overflow".to_string(),
                )
            })?;
        let v_elements = row_count
            .checked_mul(self.plan.value_shape[3])
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "KV cache value element count overflow".to_string(),
                )
            })?;

        #[allow(clippy::too_many_arguments)]
        fn grow_buffers<T: Copy + Default>(
            keys: &mut Vec<T>,
            vals: &mut Vec<T>,
            layout: KvLayout,
            k_elements: usize,
            v_elements: usize,
            old_alloc: usize,
            new_alloc: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            streams: usize,
        ) {
            match layout {
                KvLayout::PositionMajor => {
                    // Positions are outermost, so growth is a pure append.
                    keys.resize(k_elements, T::default());
                    vals.resize(v_elements, T::default());
                }
                KvLayout::HeadMajor => {
                    // Each head's stream stride is the allocated capacity, so
                    // growth re-lays the buffer: copy every (layer, head)
                    // block to its new base. Pure permutation of identical
                    // bits — bitwise-neutral — amortized over the chunking.
                    let k_old_len = old_alloc * k_head_dim;
                    let k_new_len = new_alloc * k_head_dim;
                    let v_old_len = old_alloc * v_head_dim;
                    let v_new_len = new_alloc * v_head_dim;

                    let mut new_keys = vec![T::default(); k_elements];
                    let mut new_vals = vec![T::default(); v_elements];

                    for stream in 0..streams {
                        if k_old_len > 0 {
                            let old_base = stream * k_old_len;
                            let new_base = stream * k_new_len;
                            new_keys[new_base..new_base + k_old_len]
                                .copy_from_slice(&keys[old_base..old_base + k_old_len]);
                        }
                        if v_old_len > 0 {
                            let old_base = stream * v_old_len;
                            let new_base = stream * v_new_len;
                            new_vals[new_base..new_base + v_old_len]
                                .copy_from_slice(&vals[old_base..old_base + v_old_len]);
                        }
                    }
                    *keys = new_keys;
                    *vals = new_vals;
                }
            }
        }

        let k_head_dim = self.plan.key_shape[3];
        let v_head_dim = self.plan.value_shape[3];
        let streams = self.plan.layer_count * self.plan.kv_head_count;
        match self.dtype {
            KvDtype::F32 => grow_buffers(
                &mut self.keys,
                &mut self.values,
                self.layout,
                k_elements,
                v_elements,
                self.allocated_sequence_length,
                target_sequence_length,
                k_head_dim,
                v_head_dim,
                streams,
            ),
            KvDtype::F16 => grow_buffers(
                &mut self.keys_f16,
                &mut self.values_f16,
                self.layout,
                k_elements,
                v_elements,
                self.allocated_sequence_length,
                target_sequence_length,
                k_head_dim,
                v_head_dim,
                streams,
            ),
            KvDtype::Q8_0 => grow_buffers(
                &mut self.keys_q8_0,
                &mut self.values_q8_0,
                self.layout,
                row_count * k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                row_count * v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                self.allocated_sequence_length,
                target_sequence_length,
                k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                streams,
            ),
            KvDtype::Q4_0 => grow_buffers(
                &mut self.keys_q4_0,
                &mut self.values_q4_0,
                self.layout,
                row_count * k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                row_count * v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                self.allocated_sequence_length,
                target_sequence_length,
                k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES),
                streams,
            ),
        }
        self.allocated_sequence_length = target_sequence_length;
        Ok(())
    }

    fn grow_sequence_length(&self, required_sequence_length: usize) -> usize {
        let grow_tokens = kv_cache_grow_tokens(self.plan.max_sequence_length);
        if grow_tokens <= 1 {
            return required_sequence_length;
        }
        required_sequence_length
            .div_ceil(grow_tokens)
            .saturating_mul(grow_tokens)
            .min(self.plan.max_sequence_length)
    }

    pub fn allocated_elements(&self) -> usize {
        self.allocated_sequence_length
            .saturating_mul(self.position_stride() + self.value_position_stride())
    }

    pub fn allocated_bytes(&self) -> u64 {
        match self.dtype {
            KvDtype::F32 => {
                ((self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()) as u64
            }
            KvDtype::F16 => {
                ((self.keys_f16.len() + self.values_f16.len()) * std::mem::size_of::<u16>()) as u64
            }
            KvDtype::Q8_0 => {
                ((self.keys_q8_0.len() + self.values_q8_0.len()) * std::mem::size_of::<BlockQ8_0>())
                    as u64
            }
            KvDtype::Q4_0 => {
                ((self.keys_q4_0.len() + self.values_q4_0.len()) * std::mem::size_of::<BlockQ4_0>())
                    as u64
            }
        }
    }

    /// Whether the f32 `keys`/`values` buffers are ADDRESSABLE over `[0, position)` for every
    /// layer through `last_layer` — i.e. whether a reader may index them with
    /// [`offset`](Self::offset) over that range without going out of bounds.
    ///
    /// `position` alone does NOT imply the buffers exist: only `ensure_position_capacity`
    /// grows them, so a session whose positions were all produced by a GPU-resident engine
    /// carries a high `position` over empty buffers, and in `F16` dtype the entries live in
    /// `keys_f16`/`values_f16` instead. Both layouts put the maximum offset at the last
    /// layer's last kv head at `position - 1`, so one probe bounds the whole range.
    ///
    /// SCOPE — read this before relying on it. This is a BOUNDS check, not a validity check.
    /// It says the bytes are safe to index; it does NOT say they hold the sequence's real
    /// K/V. Once `ensure_position_capacity` has grown the buffers for ANY position, this
    /// returns true for every position `<= allocated_sequence_length` regardless of what was
    /// actually written — so a range this accepts may still be zero-filled for positions the
    /// GPU produced and the CPU never wrote.
    ///
    /// This remains as a regression-test probe for hollow resident-session buffers.
    #[cfg(test)]
    pub(super) fn f32_history_addressable(&self, last_layer: usize, position: usize) -> bool {
        if position == 0 {
            return true;
        }
        if self.dtype != KvDtype::F32
            || position > self.allocated_sequence_length
            || last_layer >= self.plan.layer_count
            || self.plan.kv_head_count == 0
        {
            return false;
        }
        let end = self.offset(last_layer, position - 1, self.plan.kv_head_count - 1)
            + self.plan.k_head_dim;
        self.keys.len() >= end && self.values.len() >= end
    }

    /// Has `[0, position)` actually been written?
    ///
    /// GPU seed paths read through [`copy_key_row_into`](Self::copy_key_row_into) and
    /// [`copy_value_row_into`](Self::copy_value_row_into), which serve every cache dtype.
    /// The watermark is therefore the correct predicate, and it only advances from
    /// [`store_kv_head_row`](Self::store_kv_head_row), which cannot write without capacity.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) fn history_materialized(&self, position: usize) -> bool {
        self.materialized_through >= position
    }

    /// Mirror ONE layer's GPU-resident K/V for positions `[0, position_count)` back into this
    /// cache. `keys`/`values` arrive in the resident engines' readback order —
    /// `[kv_head][position][head_dim]`, position stride `position_count` — which is what both
    /// `metal::ResidentDecodeState::read_kv_layer` and
    /// `cuda_resident::CudaResidentDecode::read_kv_layer` return.
    ///
    /// `layer_idx` is the ABSOLUTE layer id in this cache. Resident engines index their own
    /// slots relative to the session's owned layer range, so a sharded caller translates
    /// before calling (as the CUDA seed path already does in the other direction).
    ///
    /// Rows go through [`store_kv_head_row`](Self::store_kv_head_row), so the f16 rounding and
    /// the materialized-through watermark are handled exactly as for a CPU write — the GPU KV
    /// is f16 already, so the re-rounding is idempotent.
    ///
    /// Shared by the Metal and CUDA recovery paths so the only backend-specific code left is
    /// the device readback itself.
    pub(super) fn store_mirrored_layer_kv(
        &mut self,
        layer_idx: usize,
        position_count: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        let k_head_dim = self.plan.k_head_dim;
        let v_head_dim = self.plan.v_head_dim;
        let n_kv = self.plan.kv_head_count;
        let expected_k = n_kv
            .checked_mul(position_count)
            .and_then(|v| v.checked_mul(k_head_dim))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "resident KV readback element count overflow for keys".to_string(),
                )
            })?;
        let expected_v = n_kv
            .checked_mul(position_count)
            .and_then(|v| v.checked_mul(v_head_dim))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "resident KV readback element count overflow for values".to_string(),
                )
            })?;
        // If v_head_dim is 0, backend might still return an empty slice, which is expected_v = 0
        if keys.len() != expected_k || values.len() != expected_v {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "resident KV readback for layer {layer_idx} returned {} key / {} value elements, \
                 expected {expected_k} keys / {expected_v} values ({n_kv} kv heads x {position_count} positions x {k_head_dim}/{v_head_dim})",
                keys.len(),
                values.len()
            )));
        }
        self.ensure_position_capacity(position_count)?;
        for p in 0..position_count {
            for h in 0..n_kv {
                let k_src = (h * position_count + p) * k_head_dim;
                let v_src = (h * position_count + p) * v_head_dim;
                let value_slice = if v_head_dim > 0 {
                    &values[v_src..v_src + v_head_dim]
                } else {
                    &[]
                };
                self.store_kv_head_row(
                    layer_idx,
                    p,
                    h,
                    &keys[k_src..k_src + k_head_dim],
                    value_slice,
                );
            }
        }
        Ok(())
    }

    pub(super) fn offset(&self, layer_idx: usize, position: usize, kv_head: usize) -> usize {
        match self.layout {
            KvLayout::PositionMajor => {
                (((position * self.plan.layer_count) + layer_idx) * self.plan.kv_head_count
                    + kv_head)
                    * self.plan.k_head_dim
            }
            KvLayout::HeadMajor => {
                (((layer_idx * self.plan.kv_head_count) + kv_head) * self.allocated_sequence_length
                    + position)
                    * self.plan.k_head_dim
            }
        }
    }

    /// Block offset for a quantized row. Quantized rows are padded independently
    /// to a block boundary, so deriving this from the logical element offset is
    /// incorrect when a head dimension is not a multiple of 32.
    pub(super) fn quantized_offset(
        &self,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
        blocks_per_row: usize,
    ) -> usize {
        match self.layout {
            KvLayout::PositionMajor => {
                (((position * self.plan.layer_count) + layer_idx) * self.plan.kv_head_count
                    + kv_head)
                    * blocks_per_row
            }
            KvLayout::HeadMajor => {
                (((layer_idx * self.plan.kv_head_count) + kv_head) * self.allocated_sequence_length
                    + position)
                    * blocks_per_row
            }
        }
    }

    pub(super) fn head_base_offset(&self, layer_idx: usize, kv_head: usize) -> usize {
        self.offset(layer_idx, 0, kv_head)
    }

    pub(super) fn quantized_head_base_offset(
        &self,
        layer_idx: usize,
        kv_head: usize,
        blocks_per_row: usize,
    ) -> usize {
        self.quantized_offset(layer_idx, 0, kv_head, blocks_per_row)
    }

    /// Element step between one head's consecutive positions — the stride the
    /// per-head attention walks take. A full token stride in position-major,
    /// `head_dim` (contiguous) in head-major.
    pub(super) fn head_position_stride(&self) -> usize {
        match self.layout {
            KvLayout::PositionMajor => self.position_stride(),
            KvLayout::HeadMajor => self.plan.k_head_dim,
        }
    }

    pub(super) fn quantized_head_position_stride(&self, blocks_per_row: usize) -> usize {
        match self.layout {
            KvLayout::PositionMajor => {
                self.plan.layer_count * self.plan.kv_head_count * blocks_per_row
            }
            KvLayout::HeadMajor => blocks_per_row,
        }
    }

    /// Elements ONE TOKEN occupies per buffer across all layers/heads —
    /// layout-independent count used for capacity and budget math, NOT an
    /// address stride (use [`Self::head_position_stride`] for walks).
    pub(super) fn position_stride(&self) -> usize {
        self.plan.layer_count * self.plan.kv_head_count * self.plan.key_shape[3]
    }

    pub(super) fn value_position_stride(&self) -> usize {
        self.plan.layer_count * self.plan.kv_head_count * self.plan.value_shape[3]
    }

    /// Bytes one token's KV occupies across all layers/heads at the active
    /// dtype, counting both the K and V buffers — the per-token cost the
    /// predict-and-abort guard projects.
    fn kv_bytes_per_token(&self) -> u64 {
        let total_elements = (self.position_stride() + self.value_position_stride()) as u64;
        match self.dtype {
            KvDtype::F32 => total_elements * 4,
            KvDtype::F16 => total_elements * 2,
            KvDtype::Q8_0 => {
                let blocks_per_stream = self.plan.k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES)
                    + self.plan.v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                (self.plan.layer_count
                    * self.plan.kv_head_count
                    * blocks_per_stream
                    * std::mem::size_of::<BlockQ8_0>()) as u64
            }
            KvDtype::Q4_0 => {
                let blocks_per_stream = self.plan.k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES)
                    + self.plan.v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                (self.plan.layer_count
                    * self.plan.kv_head_count
                    * blocks_per_stream
                    * std::mem::size_of::<BlockQ4_0>()) as u64
            }
        }
    }

    /// THE canonical KV store: one (layer, position, kv_head) row of K and V,
    /// rounded through f16 exactly as the write path always has, into
    /// whichever dtype backs this cache. Every writer routes through here —
    /// including the CUDA prefill mirror-back, whose data is f16-exact
    /// already (re-rounding is idempotent), so the routing is bit-neutral
    /// and enforces the f16-exactness invariant structurally.
    pub(super) fn store_kv_head_row(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
        key_row: &[f32],
        value_row: &[f32],
    ) {
        let k_head_dim = self.plan.k_head_dim;
        let v_dim = value_row.len();
        debug_assert_eq!(key_row.len(), k_head_dim);
        // Every writer routes through here, so this is THE place the materialized-through
        // watermark advances — including the GPU→host mirror, which therefore needs no
        // special-casing to be trusted afterwards.
        self.materialized_through = self.materialized_through.max(position + 1);
        let dst = self.offset(layer_idx, position, kv_head);

        match self.dtype {
            KvDtype::F32 => {
                for (slot, &value) in self.keys[dst..dst + k_head_dim].iter_mut().zip(key_row) {
                    *slot = super::kv_f16::f16_to_f32_kv(super::kv_f16::f32_to_f16_kv(value));
                }
                if v_dim > 0 {
                    let v_dst = dst / k_head_dim * v_dim; // Adjust offset for value buffer if dims differ
                    for (slot, &value) in
                        self.values[v_dst..v_dst + v_dim].iter_mut().zip(value_row)
                    {
                        *slot = super::kv_f16::f16_to_f32_kv(super::kv_f16::f32_to_f16_kv(value));
                    }
                }
            }
            KvDtype::F16 => {
                super::kv_f16::convert_f32_slice_to_f16(
                    key_row,
                    &mut self.keys_f16[dst..dst + k_head_dim],
                );
                if v_dim > 0 {
                    let v_dst = dst / k_head_dim * v_dim;
                    super::kv_f16::convert_f32_slice_to_f16(
                        value_row,
                        &mut self.values_f16[v_dst..v_dst + v_dim],
                    );
                }
            }
            KvDtype::Q8_0 => {
                let key_blocks = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let key_dst = self.quantized_offset(layer_idx, position, kv_head, key_blocks);
                quantize_row_q8_0(key_row, &mut self.keys_q8_0[key_dst..key_dst + key_blocks]);
                if v_dim > 0 {
                    let value_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let value_dst =
                        self.quantized_offset(layer_idx, position, kv_head, value_blocks);
                    quantize_row_q8_0(
                        value_row,
                        &mut self.values_q8_0[value_dst..value_dst + value_blocks],
                    );
                }
            }
            KvDtype::Q4_0 => {
                let key_blocks = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let key_dst = self.quantized_offset(layer_idx, position, kv_head, key_blocks);
                quantize_row_q4_0(key_row, &mut self.keys_q4_0[key_dst..key_dst + key_blocks]);
                if v_dim > 0 {
                    let value_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let value_dst =
                        self.quantized_offset(layer_idx, position, kv_head, value_blocks);
                    quantize_row_q4_0(
                        value_row,
                        &mut self.values_q4_0[value_dst..value_dst + value_blocks],
                    );
                }
            }
        }
    }

    /// Copy one key row out as f32, whichever dtype backs it. Cold paths
    /// only (diagnostics, GPU seeding) — the decode hot path reads the
    /// buffers directly through the layout accessors.
    pub(super) fn copy_key_row_into(
        &self,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
        out: &mut [f32],
    ) {
        let k_head_dim = self.plan.k_head_dim;
        debug_assert_eq!(out.len(), k_head_dim);
        let src = self.offset(layer_idx, position, kv_head);
        match self.dtype {
            KvDtype::F32 => out.copy_from_slice(&self.keys[src..src + k_head_dim]),
            KvDtype::F16 => {
                for (slot, &bits) in out.iter_mut().zip(&self.keys_f16[src..src + k_head_dim]) {
                    *slot = super::kv_f16::f16_to_f32_kv(bits);
                }
            }
            KvDtype::Q8_0 => {
                let block_count = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let block_src = self.quantized_offset(layer_idx, position, kv_head, block_count);
                dequantize_row_q8_0(&self.keys_q8_0[block_src..block_src + block_count], out);
            }
            KvDtype::Q4_0 => {
                let block_count = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let block_src = self.quantized_offset(layer_idx, position, kv_head, block_count);
                dequantize_row_q4_0(&self.keys_q4_0[block_src..block_src + block_count], out);
            }
        }
    }

    /// Copy one value row out as f32; see [`Self::copy_key_row_into`].
    pub(super) fn copy_value_row_into(
        &self,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
        out: &mut [f32],
    ) {
        let v_head_dim = self.plan.v_head_dim;
        let k_head_dim = self.plan.k_head_dim;
        debug_assert_eq!(out.len(), v_head_dim);
        let key_src = self.offset(layer_idx, position, kv_head);
        let is_mla = self.plan.value_shape[3] == 0;
        match self.dtype {
            KvDtype::F32 => {
                let src_slice = if is_mla {
                    &self.keys[key_src..key_src + v_head_dim]
                } else {
                    let v_src = key_src / k_head_dim * v_head_dim;
                    &self.values[v_src..v_src + v_head_dim]
                };
                out.copy_from_slice(src_slice);
            }
            KvDtype::F16 => {
                let src_slice = if is_mla {
                    &self.keys_f16[key_src..key_src + v_head_dim]
                } else {
                    let v_src = key_src / k_head_dim * v_head_dim;
                    &self.values_f16[v_src..v_src + v_head_dim]
                };
                for (slot, &bits) in out.iter_mut().zip(src_slice) {
                    *slot = super::kv_f16::f16_to_f32_kv(bits);
                }
            }
            KvDtype::Q8_0 => {
                let blocks = if is_mla {
                    let key_block_count = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let block_src =
                        self.quantized_offset(layer_idx, position, kv_head, key_block_count);
                    &self.keys_q8_0
                        [block_src..block_src + v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES)]
                } else {
                    let block_count = v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let block_src =
                        self.quantized_offset(layer_idx, position, kv_head, block_count);
                    &self.values_q8_0[block_src..block_src + block_count]
                };
                dequantize_row_q8_0(blocks, out);
            }
            KvDtype::Q4_0 => {
                let blocks = if is_mla {
                    let key_block_count = k_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let block_src =
                        self.quantized_offset(layer_idx, position, kv_head, key_block_count);
                    &self.keys_q4_0
                        [block_src..block_src + v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES)]
                } else {
                    let block_count = v_head_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let block_src =
                        self.quantized_offset(layer_idx, position, kv_head, block_count);
                    &self.values_q4_0[block_src..block_src + block_count]
                };
                dequantize_row_q4_0(blocks, out);
            }
        }
    }
}

fn kv_cache_grow_tokens(max_sequence_length: usize) -> usize {
    if max_sequence_length < 512 {
        return 1;
    }
    env::var("CAMELID_KV_CACHE_GROW_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256)
}

/// Explicit override for the KV-cache predict-and-abort budget, in bytes. Mirrors
/// `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` for the weight-materialization guard.
const KV_CACHE_BUDGET_LIMIT_ENV: &str = "CAMELID_MAX_KV_CACHE_BYTES";
/// Default share of *available* physical RAM the KV cache may claim when no env override
/// is set — leaves headroom for activations + the OS so a long-context request fails
/// closed instead of pushing the host into paging / OOM.
const KV_CACHE_BUDGET_AVAILABLE_PERCENT: u64 = 80;
/// Floor for the auto-budget, as a share of *total* physical RAM. `available` is a live
/// reading that dips sharply right when a large model's weights fault into the working
/// set during the forward pass — exactly when a session's KV cache is first sized.
/// Without a floor the budget can collapse to a few MB with gigabytes actually free and
/// refuse even a short generation (observed: a 44 MB budget with 5.4 GB free while an 8B
/// generated on CPU → spurious `generation_step_failed`). A fraction of total RAM is a
/// stable floor: still far below a runaway-context ceiling, but always enough for normal
/// generation. The `available`-based value still governs when RAM is genuinely healthy.
const KV_CACHE_BUDGET_TOTAL_FLOOR_PERCENT: u64 = 25;

/// Resolve a new session's KV-cache memory budget (bytes): an explicit
/// `CAMELID_MAX_KV_CACHE_BYTES` wins (deterministic, for tuning or controlled runs);
/// otherwise `max(80% of available, 25% of total)` physical RAM (Windows
/// `GlobalMemoryStatusEx`).
fn resolve_kv_cache_budget_bytes() -> u64 {
    let env_value = env::var(KV_CACHE_BUDGET_LIMIT_ENV).ok();
    kv_cache_budget_from(env_value.as_deref(), crate::gait::host_ram_status())
}

/// Pure budget policy (extracted so it is testable without mutating process env or
/// querying the host): a parseable non-empty env override wins; else the larger of
/// 80%-of-available and a 25%-of-total floor (the floor guards against a transient
/// collapse in `available`); else (no env, no RAM probe — e.g. off Windows) unbounded.
fn kv_cache_budget_from(env_value: Option<&str>, ram: Option<(u64, u64)>) -> u64 {
    if let Some(trimmed) = env_value.map(str::trim) {
        if !trimmed.is_empty() {
            if let Ok(bytes) = trimmed.parse::<u64>() {
                return bytes;
            }
        }
    }
    match ram {
        Some((total, available)) => {
            let by_available = available.saturating_mul(KV_CACHE_BUDGET_AVAILABLE_PERCENT) / 100;
            let floor = total.saturating_mul(KV_CACHE_BUDGET_TOTAL_FLOOR_PERCENT) / 100;
            by_available.max(floor)
        }
        None => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_with(
        max_seq: usize,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> LlamaKvCachePlan {
        let shape = vec![layers, max_seq, kv_heads, head_dim];
        LlamaKvCachePlan {
            max_sequence_length: max_seq,
            layer_count: layers,
            kv_head_count: kv_heads,
            head_dim,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            key_shape: shape.clone(),
            value_shape: shape,
        }
    }

    #[test]
    fn kv_bytes_per_token_counts_k_and_v_f32() {
        // Llama 3.2 3B shape: 28 layers * 8 kv-heads * 128 head_dim = 28672 stride;
        // *2 (K+V) *4 (f32) = 229376 bytes/token.
        let cache = LlamaKvCache::new(
            plan_with(131072, 28, 8, 128),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        assert_eq!(cache.kv_bytes_per_token(), 229_376);
        // TinyLlama shape: 22 * 4 * 64 = 5632 stride; *8 = 45056 bytes/token.
        let cache = LlamaKvCache::new(
            plan_with(2048, 22, 4, 64),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        assert_eq!(cache.kv_bytes_per_token(), 45_056);
    }

    #[test]
    fn predict_and_abort_refuses_over_budget_before_allocating() {
        // max_seq < 512 => grow_tokens == 1, so target == required (no chunk rounding) —
        // deterministic regardless of CAMELID_KV_CACHE_GROW_TOKENS.
        let mut cache = LlamaKvCache::new(
            plan_with(400, 16, 8, 64),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        let per_token = cache.kv_bytes_per_token(); // 16*8*64*8 = 65536
        cache.kv_budget_bytes = 100 * per_token; // budget for exactly 100 tokens
                                                 // At budget: allowed, allocates exactly 100.
        assert!(cache.ensure_position_capacity(100).is_ok());
        assert_eq!(cache.allocated_sequence_length, 100);
        // One token over: refused BEFORE any new allocation.
        let err = cache.ensure_position_capacity(101).unwrap_err();
        assert!(
            matches!(err, BackendError::KvCacheBudgetExceeded { .. }),
            "over-budget growth must be the typed KV budget error, got: {err:?}"
        );
        // The user-facing message still names the override env so the operator sees the knob.
        let msg = err.to_string();
        assert!(
            msg.contains(KV_CACHE_BUDGET_LIMIT_ENV),
            "message should name the override env: {msg}"
        );
        assert_eq!(
            cache.allocated_sequence_length, 100,
            "the over-budget length must not have been allocated"
        );
    }

    #[test]
    fn unbounded_budget_allows_normal_growth() {
        let mut cache = LlamaKvCache::new(
            plan_with(4096, 16, 8, 64),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        cache.kv_budget_bytes = u64::MAX;
        assert!(cache.ensure_position_capacity(2048).is_ok());
        assert!(cache.allocated_sequence_length >= 2048);
    }

    #[test]
    fn budget_policy_env_override_wins_then_ram_fraction_then_unbounded() {
        // Explicit env override (parseable) wins verbatim.
        assert_eq!(
            kv_cache_budget_from(Some(" 4096 "), Some((32 << 30, 16 << 30))),
            4096
        );
        // Unparseable / empty env falls through to the RAM policy. Here available is
        // healthy (20 of 32 GiB) so 80%-of-available (16 GiB) dominates the 25%-of-total
        // floor (8 GiB).
        assert_eq!(
            kv_cache_budget_from(Some("not-a-number"), Some((32 << 30, 20 << 30))),
            (20u64 << 30) * KV_CACHE_BUDGET_AVAILABLE_PERCENT / 100
        );
        assert_eq!(
            kv_cache_budget_from(None, Some((32 << 30, 20 << 30))),
            (20u64 << 30) * KV_CACHE_BUDGET_AVAILABLE_PERCENT / 100
        );
        // No env and no RAM probe (e.g. off Windows) -> unbounded (env remains the gate).
        assert_eq!(kv_cache_budget_from(None, None), u64::MAX);
    }

    #[test]
    fn budget_floor_prevents_self_starvation_on_transient_low_available() {
        // A large model's weights faulting into the working set can crater `available`
        // right when a session's KV cache is sized. Without the floor the budget would be
        // 0.8 * 50 MiB = 40 MiB and refuse even a short generation on a machine with
        // gigabytes actually free; the 25%-of-total floor keeps it usable.
        let total = 16u64 << 30;
        let available = 50u64 << 20; // 50 MiB — the pathological transient reading
        let budget = kv_cache_budget_from(None, Some((total, available)));
        let floor = total * KV_CACHE_BUDGET_TOTAL_FLOOR_PERCENT / 100; // 4 GiB
        assert_eq!(budget, floor, "floor must govern when available collapses");
        assert!(
            budget > available.saturating_mul(KV_CACHE_BUDGET_AVAILABLE_PERCENT) / 100,
            "floor must exceed the collapsed available-based value"
        );
        // An explicit override still wins over the floor (deterministic controlled runs).
        assert_eq!(
            kv_cache_budget_from(Some("1048576"), Some((total, available))),
            1_048_576
        );
    }

    /// On macOS the host RAM probe now returns `Some`, so the no-env path resolves to
    /// the 80%-of-available RAM branch instead of the off-platform `u64::MAX` unbounded
    /// fallback — the auto-budget genuinely gates here now. (The per-token byte math and
    /// refuse-before-allocate tests above are platform-independent and already cover the
    /// Mac guard path; this pins the policy resolution that changed for Mac.)
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_ram_branch_engages_auto_budget() {
        // resolve_kv_cache_budget_bytes() below does a libc getenv; serialize it against every
        // other env-mutating test (ENV_LOCK) so the read can't race a concurrent set_var on the
        // shared `environ` under parallel `cargo test`. (See the review's phase1 F1/F2.)
        let _env = crate::test_support::env_lock();
        let ram = crate::gait::host_ram_status();
        assert!(ram.is_some(), "macOS host_ram_status must report Some");
        let budget = kv_cache_budget_from(None, ram);
        assert!(budget > 0, "auto-budget must be a positive byte count");
        assert!(
            budget < u64::MAX,
            "auto-budget must be bounded on Mac, not the off-platform unbounded fallback"
        );
        if let Some((total, available)) = ram {
            assert!(available <= total, "available must not exceed total");
            let by_available = available.saturating_mul(KV_CACHE_BUDGET_AVAILABLE_PERCENT) / 100;
            let floor = total.saturating_mul(KV_CACHE_BUDGET_TOTAL_FLOOR_PERCENT) / 100;
            assert_eq!(budget, by_available.max(floor));
        }
        // An explicit override still wins verbatim over the live RAM reading.
        assert_eq!(kv_cache_budget_from(Some("4096"), ram), 4096);
        // And the live end-to-end resolver agrees (no env set in this test).
        assert!(resolve_kv_cache_budget_bytes() < u64::MAX);
    }

    #[test]
    fn budget_excluded_from_state_equality() {
        let mut a = LlamaKvCache::new(
            plan_with(2048, 22, 4, 64),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        let mut b = LlamaKvCache::new(
            plan_with(2048, 22, 4, 64),
            crate::model::KvCacheQuantization::F16,
        )
        .unwrap();
        a.kv_budget_bytes = 1 << 20;
        b.kv_budget_bytes = 1 << 40;
        // Different host budgets, identical state -> still equal.
        assert_eq!(a, b);
    }

    #[test]
    fn quantized_cache_byte_accounting_includes_per_row_padding() {
        let plan = plan_with(100, 2, 3, 33);
        for (dtype, expected_per_token) in [(KvDtype::Q8_0, 816), (KvDtype::Q4_0, 432)] {
            let mut cache = LlamaKvCache::new_with_layout_and_dtype(
                plan.clone(),
                KvLayout::PositionMajor,
                dtype,
            )
            .unwrap();
            assert_eq!(cache.kv_bytes_per_token(), expected_per_token);
            cache.kv_budget_bytes = expected_per_token;
            cache.ensure_position_capacity(1).unwrap();
            assert_eq!(cache.allocated_bytes(), expected_per_token);
            assert_eq!(cache.allocated_elements(), 2 * 3 * 33 * 2);

            let error = cache.ensure_position_capacity(2).unwrap_err();
            assert!(matches!(error, BackendError::KvCacheBudgetExceeded { .. }));
            assert_eq!(cache.allocated_sequence_length, 1);
        }
    }

    #[test]
    fn quantized_cache_preserves_tail_rows_across_layout_growth() {
        let plan = plan_with(600, 2, 2, 40);
        for dtype in [KvDtype::Q8_0, KvDtype::Q4_0] {
            for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
                let mut cache =
                    LlamaKvCache::new_with_layout_and_dtype(plan.clone(), layout, dtype).unwrap();
                cache.kv_budget_bytes = u64::MAX;
                cache.ensure_position_capacity(1).unwrap();

                let key: Vec<f32> = (0..40).map(|i| (i as f32 - 17.0) * 0.125).collect();
                let value: Vec<f32> = (0..40).map(|i| (11.0 - i as f32) * 0.0625).collect();
                cache.store_kv_head_row(1, 0, 1, &key, &value);

                // Crossing the 256-position growth boundary re-lays head-major
                // storage and must preserve every padded quantized row.
                cache.ensure_position_capacity(257).unwrap();
                let mut actual_key = vec![0.0; 40];
                let mut actual_value = vec![0.0; 40];
                cache.copy_key_row_into(1, 0, 1, &mut actual_key);
                cache.copy_value_row_into(1, 0, 1, &mut actual_value);

                let tolerance = if dtype == KvDtype::Q8_0 { 0.02 } else { 0.2 };
                for (index, (actual, expected)) in actual_key.iter().zip(&key).enumerate() {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{dtype:?}/{layout:?} key tail mismatch at {index}: {actual} vs {expected}"
                    );
                }
                for (index, (actual, expected)) in actual_value.iter().zip(&value).enumerate() {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{dtype:?}/{layout:?} value tail mismatch at {index}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn quantized_storage_participates_in_cache_equality() {
        for dtype in [KvDtype::Q8_0, KvDtype::Q4_0] {
            let mut cache = LlamaKvCache::new_with_layout_and_dtype(
                plan_with(1, 1, 1, 32),
                KvLayout::PositionMajor,
                dtype,
            )
            .unwrap();
            cache.ensure_position_capacity(1).unwrap();
            cache.store_kv_head_row(0, 0, 0, &[1.0; 32], &[2.0; 32]);
            let mut different = cache.clone();
            match dtype {
                KvDtype::Q8_0 => different.keys_q8_0[0].qs[0] ^= 1,
                KvDtype::Q4_0 => different.keys_q4_0[0].qs[0] ^= 1,
                _ => unreachable!(),
            }
            assert_ne!(cache, different);
        }
    }
}
