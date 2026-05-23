Total output lines: 15326

use std::{
    cell::RefCell,
    collections::HashMap,
    env, mem,
    process::Command,
    sync::{atomic::AtomicU64, Arc},
    time::Instant,
};

use rayon::prelude::*;
use serde::Serialize;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::execution_plan::MAC_Q8_PREFILL_I8MM_MIN_ROWS;
use crate::metal;

mod diagnostic_config;
mod kv_cache;
mod q8_block_reader;
mod q8_runtime;
mod q8_telemetry;
mod rope;

#[cfg(test)]
use diagnostic_config::diagnostic_zero_delta_value;
use diagnostic_config::{
    apply_ffn_gate_up_order, attention_score_scale_value, map_attention_head_to_kv_head,
};
pub use diagnostic_config::{
    diagnostic_attention_score_scale, diagnostic_ffn_gate_up_order, diagnostic_gqa_head_mapping,
    diagnostic_linear_accumulation_precision, diagnostic_output_projection_layout,
    diagnostic_rectangular_linear_layout, diagnostic_rectangular_linear_layout_for_role,
    diagnostic_rms_norm_epsilon, diagnostic_square_linear_layout, diagnostic_zero_delta,
    diagnostic_zero_delta_selector, AttentionScoreScale, DeltaZeroTarget, FfnGateUpOrder,
    GqaHeadMapping, LinearAccumulationPrecision, OutputProjectionLayout, RectangularLinearLayout,
    SquareLinearLayout,
};
pub use kv_cache::{LlamaKvCache, LlamaKvCachePlan};
pub use q8_block_reader::Q8BlockReader;
use q8_runtime::{
    q8_0_env_flag_disabled, q8_0_env_flag_enabled_default_off,
    q8_0_env_flag_enabled_default_on_fail_closed, Q8RuntimeFlags, ResolvedRuntimePlan,
};
use q8_telemetry::{
    add_q8_schedule_counter, record_q8_schedule_output_projection_route_call,
    record_q8_schedule_projection_route_denial, record_q8_schedule_projection_route_elapsed,
    Q8_SCHED_FFN_DECODE_CHAIN_ACTIVATION_QUANTIZE_US, Q8_SCHED_FFN_DECODE_CHAIN_DOWN_US,
    Q8_SCHED_FFN_DECODE_CHAIN_INPUT_QUANTIZE_US, Q8_SCHED_FFN_DECODE_CHAIN_TAKEN,
    Q8_SCHED_FFN_DECODE_CHAIN_TOTAL_US, Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF,
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4, Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US, Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK,
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE, Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN,
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US,
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US, Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS,
};
#[cfg(test)]
use q8_telemetry::{q8_schedule_layer_index_for_projection_name, Q8_SCHEDULE_TELEMETRY_ENV};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use q8_telemetry::{
    q8_schedule_role_for_output_name, record_q8_schedule_activation_pack,
    record_q8_schedule_i8mm_single_projection_role_call,
    record_q8_schedule_i8mm_single_projection_role_gemm,
    record_q8_schedule_i8mm_single_projection_role_pack,
    record_q8_schedule_i8mm_single_projection_role_scheduler,
    record_q8_schedule_i8mm_single_projection_role_tail, Q8_SCHED_CONSERVATIVE_TAIL_ROWS,
    Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS, Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS,
    Q8_SCHED_Q8_GEMM_COMPUTE_US, Q8_SCHED_RAYON_FANOUT_BOUNDARIES,
};
pub use q8_telemetry::{
    q8_schedule_telemetry_enabled, reset_q8_schedule_telemetry, snapshot_q8_schedule_telemetry,
    LlamaQ8OutputProjectionLayerRouteTelemetry, LlamaQ8OutputProjectionRouteTelemetry,
    LlamaQ8ProjectionRouteDenialTelemetry, LlamaQ8ScheduleRoleTelemetry, LlamaQ8ScheduleTelemetry,
};
pub use rope::{
    diagnostic_rope_direction, diagnostic_rope_pairing, diagnostic_rope_position_mode,
    RopeDirection, RopePairing, RopePositionMode,
};

use crate::{
    gguf::GgufTensorType,
    model::{
        DenseLlamaDims, LlamaFfnTensors, LlamaModelConfig, LlamaMoeExpertTensors,
        LlamaTensorBinding,
    },
    tensor::{
        dot_product, parse_byte_count_env, q8_0_file_read_stats, should_parallelize_linear_output,
        with_q8_file_cache_capacity_override, CpuTensor, Q8_0Block, Q8_0FileBacking,
        Q8_0FileReadStats, Q8_0PackedRows4, Q8_0PackedRows4Block, Q8_0PackedRows4Interleave,
        Q8_0RuntimeStorage, Q8_0VnniPacked, Q8_0VnniTile16, TensorShape, TensorStore,
    },
    BackendError, Result,
};

#[cfg(test)]
use crate::tensor::record_q8_0_file_read;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use crate::tensor::Q8_0AmxPackedBlock;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe extern "C" {
    fn camelid_x86_q8_amx_supported() -> std::os::raw::c_int;
    fn camelid_q8_0_amx_compute_tile16(
        input_groups: *const Q8_0PackedRows4Block,
        blocks_per_row: usize,
        m_rows: usize,
        weight_blocks: *const Q8_0AmxPackedBlock,
        output: *mut f32,
        output_stride: usize,
    );
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceWorkspace {
    pub scratch_f32: Vec<f32>,
    pub activation_f32: Vec<f32>,
}

impl InferenceWorkspace {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            scratch_f32: vec![0.0; max_capacity],
            activation_f32: vec![0.0; max_capacity],
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.scratch_f32.fill(0.0);
        self.activation_f32.fill(0.0);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLayerWeights {
    pub attention_norm: CpuTensor,
    pub attention_q: CpuTensor,
    pub attention_k: CpuTensor,
    pub attention_v: CpuTensor,
    pub attention_output: CpuTensor,
    pub ffn_norm: CpuTensor,
    pub ffn_gate: CpuTensor,
    pub ffn_up: CpuTensor,
    pub ffn_down: CpuTensor,
    pub moe_router: Option<CpuTensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLoadedWeights {
    pub token_embedding: CpuTensor,
    pub output_norm: CpuTensor,
    pub output: Option<CpuTensor>,
    pub rope_freqs: Option<CpuTensor>,
    pub layers: Vec<LlamaLayerWeights>,
}

impl LlamaLoadedWeights {
    pub fn output_projection(&self) -> &CpuTensor {
        self.output.as_ref().unwrap_or(&self.token_embedding)
    }

    fn has_lazy_q8_0_file_backing(&self) -> bool {
        tensor_has_q8_0_file_backing(&self.token_embedding)
            || self
                .output
                .as_ref()
                .is_some_and(tensor_has_q8_0_file_backing)
            || self.layers.iter().any(|layer| {
                tensor_has_q8_0_file_backing(&layer.attention_q)
                    || tensor_has_q8_0_file_backing(&layer.attention_k)
                    || tensor_has_q8_0_file_backing(&layer.attention_v)
                    || tensor_has_q8_0_file_backing(&layer.attention_output)
                    || tensor_has_q8_0_file_backing(&layer.ffn_gate)
                    || tensor_has_q8_0_file_backing(&layer.ffn_up)
                    || tensor_has_q8_0_file_backing(&layer.ffn_down)
                    || layer
                        .moe_router
                        .as_ref()
                        .is_some_and(tensor_has_q8_0_file_backing)
            })
    }

    fn largest_q8_0_file_backed_layer_storage_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| {
                [
                    &layer.attention_q,
                    &layer.attention_k,
                    &layer.attention_v,
                    &layer.attention_output,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                    layer.moe_router.as_ref().unwrap_or(&layer.ffn_norm),
                ]
                .into_iter()
                .map(tensor_q8_0_file_backed_storage_bytes)
                .sum()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn load(store: &TensorStore, binding: &LlamaTensorBinding) -> Result<Self> {
        let auto_retain_q8_0_blocks = auto_retain_q8_0_blocks_for_fast_local_chat(binding);
        let load_linear = |name: &str| {
            if auto_retain_q8_0_blocks {
                store.load_q8_0_block_backed_linear(name)
            } else if lazy_q8_0_linear_enabled() {
                store.load_q8_0_file_backed_linear(name)
            } else {
                store.load_cpu_f32(name)
            }
        };
        let load_moe_experts = |experts: &LlamaMoeExpertTensors| match experts {
            LlamaMoeExpertTensors::Merged(desc) => store.load_q8_0_file_backed_tensor(&desc.name),
            LlamaMoeExpertTensors::Split(descs) => {
                let first = descs.first().ok_or_else(|| {
                    BackendError::InvalidModelMetadata(
                        "split MoE expert binding has no descriptors".to_string(),
                    )
                })?;
                let mut dims: Vec<usize> =
                    first.dimensions.iter().map(|dim| *dim as usize).collect();
                dims.push(descs.len());
                store.load_q8_0_split_file_backed_tensor(
                    format!("{}..{} split experts", first.name, descs.len()),
                    dims,
                    descs,
                )
            }
        };
        let token_embedding = normalize_token_embedding_shape(
            load_linear(&binding.token_embedding.name)?,
            &binding.token_embedding.name,
        )?;
        let output_norm = store.load_cpu_f32(&binding.output_norm.name)?;
        let output = if binding.output_is_tied_embedding {
            if auto_retain_q8_0_blocks {
                Some(store.load_q8_0_block_backed_linear_as(
                    &binding.token_embedding.name,
                    "output.weight",
                )?)
            } else if lazy_q8_0_linear_enabled() {
                Some(store.load_q8_0_file_backed_tensor_as(
                    &binding.token_embedding.name,
                    "output.weight",
                )?)
            } else {
                None
            }
        } else {
            Some(load_linear(&binding.output.name)?)
        };
        let rope_freqs = binding
            .rope_freqs
            .as_ref()
            .map(|desc| store.load_cpu_f32(&desc.name))
            .transpose()?;
        let mut layers = Vec::with_capacity(binding.layers.len());
        for layer in &binding.layers {
            let (ffn_gate, ffn_up, ffn_down, moe_router) = match &layer.ffn {
                LlamaFfnTensors::Dense { gate, up, down } => (
                    load_linear(&gate.name)?,
                    load_linear(&up.name)?,
                    load_linear(&down.name)?,
                    None,
                ),
                LlamaFfnTensors::MoE {
                    router,
                    gate_experts,
                    up_experts,
                    down_experts,
                } => (
                    load_moe_experts(gate_experts)?,
                    load_moe_experts(up_experts)?,
                    load_moe_experts(down_experts)?,
                    Some(store.load_cpu_f32(&router.name)?),
                ),
            };
            layers.push(LlamaLayerWeights {
                attention_norm: store.load_cpu_f32(&layer.attention_norm.name)?,
                attention_q: load_linear(&layer.attention_q.name)?,
                attention_k: load_linear(&layer.attention_k.name)?,
                attention_v: load_linear(&layer.attention_v.name)?,
                attention_output: load_linear(&layer.attention_output.name)?,
                ffn_norm: store.load_cpu_f32(&layer.ffn_norm.name)?,
                ffn_gate,
                ffn_up,
                ffn_down,
                moe_router,
            });
        }
        Ok(Self {
            token_embedding,
            output_norm,
            output,
            rope_freqs,
            layers,
        })
    }

    pub fn validate_dense_shapes(&self, config: &LlamaModelConfig) -> Result<()> {
        let dims = DenseLlamaDims::from_config(config)?;
        require_tensor_shape(
            &self.token_embedding,
            &[dims.vocab_size, dims.embedding_length],
            "token embedding",
        )?;
        require_tensor_shape(&self.output_norm, &[dims.embedding_length], "output norm")?;
        require_matrix_shape(
            self.output_projection(),
            dims.embedding_length,
            dims.vocab_size,
            "output projection",
        )?;
        if let Some(rope_freqs) = &self.rope_freqs {
            let rope_dim = config.rope_dimension_count.unwrap_or(dims.head_dim as u32) as usize;
            validate_rope_frequency_tensor(rope_freqs, rope_dim)?;
        }

        if self.layers.len() != dims.block_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "config block count {} does not match loaded layer count {}",
                dims.block_count,
                self.layers.len()
            )));
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            require_tensor_shape(
                &layer.attention_norm,
                &[dims.embedding_length],
                &format!("layer {idx} attention norm"),
            )?;
            require_matrix_shape(
                &layer.attention_q,
                dims.embedding_length,
                dims.embedding_length,
                &format!("layer {idx} attention q"),
            )?;
            require_matrix_shape(
                &layer.attention_k,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention k"),
            )?;
            require_matrix_shape(
                &layer.attention_v,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention v"),
            )?;
            require_matrix_shape(
                &layer.attention_output,
                dims.embedding_length,
                dims.embedding_length,
                &format!("layer {idx} attention output"),
            )?;
            require_tensor_shape(
                &layer.ffn_norm,
                &[dims.embedding_length],
                &format!("layer {idx} ffn norm"),
            )?;
            if let Some(moe) = &config.moe {
                let router = layer.moe_router.as_ref().ok_or_else(|| {
                    BackendError::RuntimeShapeMismatch(format!(
                        "layer {idx} Mixtral MoE router tensor is missing"
                    ))
                })?;
                require_matrix_shape(
                    router,
                    dims.embedding_length,
                    moe.expert_count as usize,
                    &format!("layer {idx} ffn router"),
                )?;
                require_tensor_shape(
                    &layer.ffn_gate,
                    &[
                        dims.embedding_length,
                        dims.feed_forward_length,
                        moe.expert_count as usize,
                    ],
                    &format!("layer {idx} ffn gate experts"),
                )?;
                require_tensor_shape(
                    &layer.ffn_up,
                    &[
                        dims.embedding_length,
                        dims.feed_forward_length,
                        moe.expert_count as usize,
                    ],
                    &format!("layer {idx} ffn up experts"),
                )?;
                require_tensor_shape(
                    &layer.ffn_down,
                    &[
                        dims.feed_forward_length,
                        dims.embedding_length,
                        moe.expert_count as usize,
                    ],
                    &format!("layer {idx} ffn down experts"),
                )?;
            } else {
                require_matrix_shape(
                    &layer.ffn_gate,
                    dims.embedding_length,
                    dims.feed_forward_length,
                    &format!("layer {idx} ffn gate"),
                )?;
                require_matrix_shape(
                    &layer.ffn_up,
                    dims.embedding_length,
                    dims.feed_forward_length,
                    &format!("layer {idx} ffn up"),
                )?;
                require_matrix_shape(
                    &layer.ffn_down,
                    dims.feed_forward_length,
                    dims.embedding_length,
                    &format!("layer {idx} ffn down"),
                )?;
            }
        }

        Ok(())
    }
}

fn tensor_has_q8_0_file_backing(tensor: &CpuTensor) -> bool {
    tensor.source_type == Some(GgufTensorType::Q8_0)
        && (tensor.q8_0_file_backing.is_some() || tensor.q8_0_split_file_backing.is_some())
}

fn tensor_q8_0_file_backed_storage_bytes(tensor: &CpuTensor) -> u64 {
    if tensor.source_type != Some(GgufTensorType::Q8_0) {
        return 0;
    }
    tensor
        .q8_0_file_backing
        .as_ref()
        .map(Q8_0FileBacking::storage_bytes)
        .unwrap_or(0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaForwardOutput {
    pub logits: CpuTensor,
    pub hidden_state: CpuTensor,
    pub output_norm_state: CpuTensor,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaTensorCheckpoint {
    pub shape: Vec<usize>,
    pub len: usize,
    pub first_values: Vec<f32>,
    pub max_abs_window_start: usize,
    pub max_abs_window: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaTensorStats {
    pub min: f32,
    pub min_index: usize,
    pub max: f32,
    pub max_index: usize,
    pub mean: f32,
    pub rms: f32,
    pub max_abs_index: usize,
    pub max_abs: f32,
    pub checkpoint: LlamaTensorCheckpoint,
}

impl LlamaTensorStats {
    pub fn from_tensor(tensor: &CpuTensor) -> Result<Self> {
        tensor_stats(tensor)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaForwardDiagnostics {
    pub embedding: LlamaTensorStats,
    pub final_hidden: LlamaTensorStats,
    pub final_norm: LlamaFinalNormDiagnostic,
    pub output_norm: LlamaTensorStats,
    pub logits: LlamaTensorStats,
    pub layers: Vec<LlamaLayerDiagnostics>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaFinalNormDiagnostic {
    pub epsilon: f32,
    pub hidden_mean_square: f32,
    pub hidden_rms: f32,
    pub scale: f32,
    pub hidden_first_values: Vec<f32>,
    pub weight_first_values: Vec<f32>,
    pub reconstructed_first_values: Vec<f32>,
    pub reported_first_values: Vec<f32>,
    pub reported_max_abs_index: usize,
    pub reported_max_abs: f32,
    pub reported_max_abs_window_start: usize,
    pub reported_max_abs_window: Vec<f32>,
    pub reconstructed_reported_max_abs_window: Vec<f32>,
    pub max_abs_delta_index: usize,
    pub max_abs_delta: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaOutputProjectionDiagnostic {
    pub token_id: u32,
    pub layout: &'static str,
    pub reported_logit: f32,
    pub reconstructed_logit: f32,
    pub decoded_component_reconstructed_logit: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_direct_reconstructed_logit: Option<f32>,
    pub absolute_delta: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_direct_absolute_delta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_direct_decoded_component_delta: Option<f32>,
    pub output_norm_rms: f32,
    pub output_row_rms: f32,
    pub cosine_similarity: f32,
    pub output_norm_first_values: Vec<f32>,
    pub outp…133593 tokens truncated…             scale,
                },
                &mut out[out_start..out_start + head_dim],
                &mut scores,
            )?;
        }
    }

    let tensor = CpuTensor::from_f32(name, vec![1, expected_width], out)?;
    let trace = collect_diagnostics
        .then(|| {
            attention_trace_with_params(AttentionTraceParams {
                kv_cache,
                layer_idx,
                query,
                context: &tensor,
                attention_heads,
                repeats,
                kv_heads,
                head_mapping,
                position_count,
                scale,
            })
        })
        .transpose()?;
    Ok(LlamaAttentionContextOutput { tensor, trace })
}

fn causal_attention_context_batch(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    base_position: usize,
    query: &CpuTensor,
    attention_heads: usize,
    kv_heads: usize,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    if kv_heads == 0 || !attention_heads.is_multiple_of(kv_heads) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention head count {attention_heads} must be a multiple of kv head count {kv_heads}"
        )));
    }
    let head_dim = kv_cache.plan.head_dim;
    let expected_width = attention_heads * head_dim;
    if query.rank() != 2 || query.dim(1)? != expected_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention query shape {:?} does not match expected [rows, {expected_width}]",
            query.shape.dims
        )));
    }
    if kv_heads != kv_cache.plan.kv_head_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention kv head count {kv_heads} does not match KV cache plan {}",
            kv_cache.plan.kv_head_count
        )));
    }
    if layer_idx >= kv_cache.plan.layer_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "layer index {layer_idx} is out of range for KV cache layer count {}",
            kv_cache.plan.layer_count
        )));
    }

    let rows = query.dim(0)?;
    let required_sequence_length = base_position.checked_add(rows).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(format!(
            "attention batch base position {base_position} plus {rows} row(s) overflows"
        ))
    })?;
    if required_sequence_length > kv_cache.allocated_sequence_length {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention batch needs {} cached position(s), but KV cache has {} allocated",
            required_sequence_length, kv_cache.allocated_sequence_length
        )));
    }
    let repeats = attention_heads / kv_heads;
    let head_mapping = diagnostic_gqa_head_mapping()?;
    let score_scale = diagnostic_attention_score_scale()?;
    let scale = attention_score_scale_value(head_dim, score_scale);
    let mut out = vec![0.0; rows * expected_width];

    let fill_row = |row: usize, out_row: &mut [f32], scores: &mut Vec<f32>| -> Result<()> {
        let position_count = base_position + row + 1;
        let query_row_start = row * expected_width;
        if position_count == 1 {
            for attention_head in 0..attention_heads {
                let kv_head =
                    map_attention_head_to_kv_head(attention_head, repeats, kv_heads, head_mapping);
                let out_start = attention_head * head_dim;
                let value_start = kv_cache_offset(kv_cache, layer_idx, 0, kv_head);
                out_row[out_start..out_start + head_dim]
                    .copy_from_slice(&kv_cache.values[value_start..value_start + head_dim]);
            }
        } else {
            for attention_head in 0..attention_heads {
                let kv_head =
                    map_attention_head_to_kv_head(attention_head, repeats, kv_heads, head_mapping);
                let query_start = query_row_start + attention_head * head_dim;
                let query_slice = &query.data[query_start..query_start + head_dim];
                let out_start = attention_head * head_dim;
                attention_context_for_head_into(
                    AttentionContextHeadParams {
                        kv_cache,
                        layer_idx,
                        kv_head,
                        query_slice,
                        position_count,
                        scale,
                    },
                    &mut out_row[out_start..out_start + head_dim],
                    scores,
                )?;
            }
        }
        Ok(())
    };

    if should_parallelize_attention_context_batch(rows, attention_heads) {
        out.par_chunks_mut(expected_width)
            .enumerate()
            .try_for_each(|(row, out_row)| {
                let mut scores = Vec::with_capacity(base_position + row + 1);
                fill_row(row, out_row, &mut scores)
            })?;
    } else {
        let mut scores = Vec::with_capacity(required_sequence_length);
        for (row, out_row) in out.chunks_mut(expected_width).enumerate() {
            fill_row(row, out_row, &mut scores)?;
        }
    }

    CpuTensor::from_f32(name, vec![rows, expected_width], out)
}

struct AttentionContextHeadParams<'a> {
    kv_cache: &'a LlamaKvCache,
    layer_idx: usize,
    kv_head: usize,
    query_slice: &'a [f32],
    position_count: usize,
    scale: f32,
}

fn attention_context_for_head_into(
    params: AttentionContextHeadParams<'_>,
    out_slice: &mut [f32],
    scores: &mut Vec<f32>,
) -> Result<()> {
    let head_dim = params.kv_cache.plan.head_dim;
    debug_assert_eq!(params.query_slice.len(), head_dim);
    debug_assert_eq!(out_slice.len(), head_dim);
    scores.clear();
    scores.reserve(params.position_count);
    let head_base = kv_cache_head_base_offset(params.kv_cache, params.layer_idx, params.kv_head);
    let position_stride = kv_cache_position_stride(params.kv_cache);

    let mut key_start = head_base;
    for position in 0..params.position_count {
        let key_slice = &params.kv_cache.keys[key_start..key_start + head_dim];
        let score = dot_product(params.query_slice, key_slice) * params.scale;
        scores.push(score);
        if position + 1 < params.position_count {
            key_start += position_stride;
        }
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut score_sum = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - max_score).exp();
        score_sum += *score;
    }
    if score_sum == 0.0 || !score_sum.is_finite() {
        return Err(BackendError::RuntimeShapeMismatch(
            "attention softmax produced invalid normalization sum".to_string(),
        ));
    }

    let inv_score_sum = 1.0 / score_sum;
    let mut value_start = head_base;
    for (position, score) in scores.iter().copied().enumerate() {
        let probability = score * inv_score_sum;
        let value_slice = &params.kv_cache.values[value_start..value_start + head_dim];
        for (out_value, value) in out_slice.iter_mut().zip(value_slice) {
            *out_value += probability * *value;
        }
        if position + 1 < params.position_count {
            value_start += position_stride;
        }
    }

    Ok(())
}

fn kv_cache_head_base_offset(kv_cache: &LlamaKvCache, layer_idx: usize, kv_head: usize) -> usize {
    ((layer_idx * kv_cache.plan.kv_head_count) + kv_head) * kv_cache.plan.head_dim
}

fn kv_cache_position_stride(kv_cache: &LlamaKvCache) -> usize {
    kv_cache.plan.layer_count * kv_cache.plan.kv_head_count * kv_cache.plan.head_dim
}

const PARALLEL_ATTENTION_CONTEXT_MIN_UNITS: usize = 256;

fn should_parallelize_attention_context_batch(rows: usize, attention_heads: usize) -> bool {
    rayon::current_num_threads() > 1
        && rows.saturating_mul(attention_heads) >= PARALLEL_ATTENTION_CONTEXT_MIN_UNITS
}

struct AttentionTraceParams<'a> {
    kv_cache: &'a LlamaKvCache,
    layer_idx: usize,
    query: &'a CpuTensor,
    context: &'a CpuTensor,
    attention_heads: usize,
    repeats: usize,
    kv_heads: usize,
    head_mapping: GqaHeadMapping,
    position_count: usize,
    scale: f32,
}

fn attention_trace_with_params(params: AttentionTraceParams<'_>) -> Result<LlamaAttentionTrace> {
    let head_dim = params.kv_cache.plan.head_dim;
    let sampled_heads = sampled_attention_trace_heads(
        params.attention_heads,
        params.repeats,
        params.kv_heads,
        params.head_mapping,
    );
    let mut heads = Vec::with_capacity(sampled_heads.len());
    for attention_head in sampled_heads {
        let kv_head = map_attention_head_to_kv_head(
            attention_head,
            params.repeats,
            params.kv_heads,
            params.head_mapping,
        );
        let query_start = attention_head * head_dim;
        let query_slice = &params.query.data[query_start..query_start + head_dim];
        let context_slice = &params.context.data[query_start..query_start + head_dim];
        let scores = attention_scores_for_head(
            params.kv_cache,
            params.layer_idx,
            kv_head,
            query_slice,
            params.position_count,
            params.scale,
        );
        let probabilities = attention_probabilities(&scores)?;
        let probability_sum = probabilities.iter().sum::<f32>();
        let probability_entropy = probabilities
            .iter()
            .copied()
            .filter(|probability| *probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum::<f32>();
        let probability_rms = (probabilities
            .iter()
            .copied()
            .map(|probability| probability * probability)
            .sum::<f32>()
            / probabilities.len() as f32)
            .sqrt();
        let (max_probability_position, max_probability) = probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, 0.0));
        let top_probability_positions = top_attention_probability_positions(
            params.kv_cache,
            params.layer_idx,
            kv_head,
            head_dim,
            &scores,
            &probabilities,
        );
        let reconstructed_context = reconstruct_attention_context_for_head(
            params.kv_cache,
            params.layer_idx,
            kv_head,
            head_dim,
            &probabilities,
        );
        let mut context_reconstruction_max_abs_delta_index = 0;
        let mut context_reconstruction_max_abs_delta = 0.0_f32;
        for (idx, (reconstructed, reported)) in reconstructed_context
            .iter()
            .zip(context_slice.iter())
            .enumerate()
        {
            let delta = (reconstructed - reported).abs();
            if delta > context_reconstruction_max_abs_delta {
                context_reconstruction_max_abs_delta = delta;
                context_reconstruction_max_abs_delta_index = idx;
            }
        }
        let sampled_positions = sampled_attention_trace_positions(params.position_count);
        let mut positions = Vec::with_capacity(sampled_positions.len());
        for position in sampled_positions {
            let key_start = kv_cache_offset(params.kv_cache, params.layer_idx, position, kv_head);
            let key_slice = &params.kv_cache.keys[key_start..key_start + head_dim];
            let value_slice = &params.kv_cache.values[key_start..key_start + head_dim];
            let qk_products = query_slice
                .iter()
                .zip(key_slice.iter())
                .map(|(query, key)| query * key)
                .collect::<Vec<_>>();
            let reconstructed_score = qk_products.iter().sum::<f32>() * params.scale;
            let qk_products_max_abs_index = max_abs_index(&qk_products);
            let (qk_products_max_abs_window_start, qk_products_max_abs_window) =
                tensor_window_around_index(
                    &qk_products,
                    qk_products_max_abs_index,
                    ATTENTION_TRACE_VALUE_LIMIT,
                );
            positions.push(LlamaAttentionPositionTrace {
                position,
                score: scores[position],
                reconstructed_score,
                score_reconstruction_delta: (scores[position] - reconstructed_score).abs(),
                probability: probabilities[position],
                key_first_values: sample_first_values(key_slice),
                qk_products_first_values: sample_first_values(&qk_products),
                qk_products_max_abs_window_start,
                qk_products_max_abs_window,
                value_first_values: sample_first_values(value_slice),
            });
        }
        heads.push(LlamaAttentionHeadTrace {
            attention_head,
            kv_head,
            query_first_values: sample_first_values(query_slice),
            context_first_values: sample_first_values(context_slice),
            reconstructed_context_first_values: sample_first_values(&reconstructed_context),
            context_reconstruction_max_abs_delta_index,
            context_reconstruction_max_abs_delta,
            probability_sum,
            probability_entropy,
            probability_rms,
            max_probability_position,
            max_probability,
            top_probability_positions,
            positions,
        });
    }

    Ok(LlamaAttentionTrace {
        scale: params.scale,
        position_count: params.position_count,
        head_dim,
        heads,
    })
}

fn attention_scores_for_head(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    kv_head: usize,
    query_slice: &[f32],
    position_count: usize,
    scale: f32,
) -> Vec<f32> {
    let head_dim = kv_cache.plan.head_dim;
    let mut scores = Vec::with_capacity(position_count);
    let mut key_start = kv_cache_head_base_offset(kv_cache, layer_idx, kv_head);
    let position_stride = kv_cache_position_stride(kv_cache);
    for position in 0..position_count {
        let key_slice = &kv_cache.keys[key_start..key_start + head_dim];
        let score = dot_product(query_slice, key_slice) * scale;
        scores.push(score);
        if position + 1 < position_count {
            key_start += position_stride;
        }
    }
    scores
}

fn attention_probabilities(scores: &[f32]) -> Result<Vec<f32>> {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exponentials = Vec::with_capacity(scores.len());
    let mut score_sum = 0.0;
    for score in scores {
        let exponential = (*score - max_score).exp();
        exponentials.push(exponential);
        score_sum += exponential;
    }
    if score_sum == 0.0 || !score_sum.is_finite() {
        return Err(BackendError::RuntimeShapeMismatch(
            "attention softmax produced invalid normalization sum".to_string(),
        ));
    }
    Ok(exponentials
        .into_iter()
        .map(|score| score / score_sum)
        .collect())
}

fn sample_first_values(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .take(ATTENTION_TRACE_VALUE_LIMIT)
        .copied()
        .collect()
}

fn reconstruct_attention_context_for_head(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    kv_head: usize,
    head_dim: usize,
    probabilities: &[f32],
) -> Vec<f32> {
    let mut context = vec![0.0; head_dim];
    for (position, probability) in probabilities.iter().copied().enumerate() {
        let value_start = kv_cache_offset(kv_cache, layer_idx, position, kv_head);
        let value_slice = &kv_cache.values[value_start..value_start + head_dim];
        for dim in 0..head_dim {
            context[dim] += probability * value_slice[dim];
        }
    }
    context
}

fn top_attention_probability_positions(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    kv_head: usize,
    head_dim: usize,
    scores: &[f32],
    probabilities: &[f32],
) -> Vec<LlamaAttentionTopProbabilityTrace> {
    let mut ranked = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_by(
        |(left_position, left_probability), (right_position, right_probability)| {
            right_probability
                .partial_cmp(left_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_position.cmp(right_position))
        },
    );

    ranked
        .into_iter()
        .take(ATTENTION_TRACE_TOP_PROBABILITY_LIMIT)
        .map(|(position, probability)| {
            let key_start = kv_cache_offset(kv_cache, layer_idx, position, kv_head);
            let key_slice = &kv_cache.keys[key_start..key_start + head_dim];
            let value_slice = &kv_cache.values[key_start..key_start + head_dim];
            LlamaAttentionTopProbabilityTrace {
                position,
                score: scores[position],
                probability,
                key_first_values: sample_first_values(key_slice),
                value_first_values: sample_first_values(value_slice),
            }
        })
        .collect()
}

fn sampled_attention_trace_positions(position_count: usize) -> Vec<usize> {
    if position_count <= ATTENTION_TRACE_POSITION_LIMIT {
        return (0..position_count).collect();
    }

    let mut positions = Vec::with_capacity(ATTENTION_TRACE_POSITION_LIMIT);
    positions.extend(0..ATTENTION_TRACE_EDGE_POSITION_LIMIT);
    positions
        .extend(position_count.saturating_sub(ATTENTION_TRACE_EDGE_POSITION_LIMIT)..position_count);
    positions
}

fn sampled_attention_trace_heads(
    attention_heads: usize,
    repeats: usize,
    kv_heads: usize,
    head_mapping: GqaHeadMapping,
) -> Vec<usize> {
    if attention_heads <= ATTENTION_TRACE_HEAD_LIMIT {
        return (0..attention_heads).collect();
    }

    let mut heads = Vec::with_capacity(ATTENTION_TRACE_HEAD_LIMIT);
    for kv_head in 0..kv_heads {
        if heads.len() >= ATTENTION_TRACE_HEAD_LIMIT {
            break;
        }
        if let Some(attention_head) = first_attention_head_for_kv_head(
            kv_head,
            attention_heads,
            repeats,
            kv_heads,
            head_mapping,
        ) {
            heads.push(attention_head);
        }
    }

    if heads.len() < ATTENTION_TRACE_HEAD_LIMIT {
        let tail_start = attention_heads.saturating_sub(ATTENTION_TRACE_HEAD_LIMIT - heads.len());
        for attention_head in tail_start..attention_heads {
            if heads.len() >= ATTENTION_TRACE_HEAD_LIMIT {
                break;
            }
            if !heads.contains(&attention_head) {
                heads.push(attention_head);
            }
        }
    }

    heads.sort_unstable();
    heads.dedup();
    heads
}

fn first_attention_head_for_kv_head(
    kv_head: usize,
    attention_heads: usize,
    repeats: usize,
    kv_heads: usize,
    head_mapping: GqaHeadMapping,
) -> Option<usize> {
    match head_mapping {
        GqaHeadMapping::Grouped => {
            let attention_head = kv_head.saturating_mul(repeats);
            (attention_head < attention_heads).then_some(attention_head)
        }
        GqaHeadMapping::Modulo => {
            (kv_head..attention_heads).find(|attention_head| attention_head % kv_heads == kv_head)
        }
    }
}

fn kv_cache_offset(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    position: usize,
    kv_head: usize,
) -> usize {
    (((position * kv_cache.plan.layer_count) + layer_idx) * kv_cache.plan.kv_head_count + kv_head)
        * kv_cache.plan.head_dim
}

pub fn tensor_map(tensors: impl IntoIterator<Item = CpuTensor>) -> HashMap<String, CpuTensor> {
    tensors
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect()
}

#[cfg(test)]
mod tests;
