use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    fs::File,
    io::{Error as IoError, ErrorKind, Result as IoResult},
    mem,
    os::unix::fs::FileExt,
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

use rayon::prelude::*;
use serde::Serialize;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::execution_plan::MAC_Q8_PREFILL_I8MM_MIN_ROWS;
use crate::metal;

mod kv_cache;
mod q8_runtime;
mod rope;

const Q8_SCHEDULE_TELEMETRY_ENV: &str = "CAMELID_Q8_SCHED_TELEMETRY";

pub use kv_cache::{LlamaKvCache, LlamaKvCachePlan};
use q8_runtime::{
    q8_0_env_flag_disabled, q8_0_env_flag_enabled_default_off,
    q8_0_env_flag_enabled_default_on_fail_closed, Q8RuntimeFlags, ResolvedRuntimePlan,
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
        dot_product, parse_byte_count_env, q8_0_file_read_stats, record_q8_0_file_read,
        should_parallelize_linear_output, with_q8_file_cache_capacity_override, CpuTensor,
        Q8_0Block, Q8_0FileBacking, Q8_0FileReadStats, Q8_0PackedRows4, Q8_0PackedRows4Block,
        Q8_0PackedRows4Interleave, Q8_0RuntimeStorage, Q8_0VnniPacked, Q8_0VnniTile16, TensorShape,
        TensorStore,
    },
    BackendError, Result,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8BlockReader {
    offset: u64,
    num_blocks: usize,
}

impl Q8BlockReader {
    pub const BLOCK_SIZE_BYTES: usize = 34;
    pub const WEIGHTS_PER_BLOCK: usize = 32;

    pub fn new(offset: u64, num_blocks: usize) -> Self {
        Self { offset, num_blocks }
    }

    pub fn dequantize_block_to_slice(
        &self,
        file: &File,
        block_idx: usize,
        dest: &mut [f32],
    ) -> IoResult<()> {
        if block_idx >= self.num_blocks {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "block index out of bounds",
            ));
        }

        let dest_offset = block_idx
            .checked_mul(Self::WEIGHTS_PER_BLOCK)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "destination offset overflow"))?;
        if dest_offset + Self::WEIGHTS_PER_BLOCK > dest.len() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "destination buffer too small",
            ));
        }

        let block_offset = self
            .offset
            .checked_add((block_idx * Self::BLOCK_SIZE_BYTES) as u64)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "block offset overflow"))?;
        let mut block_data = [0u8; Self::BLOCK_SIZE_BYTES];
        file.read_exact_at(&mut block_data, block_offset)?;
        record_q8_0_file_read(block_data.len());

        let scale_bits = u16::from_le_bytes(block_data[0..2].try_into().expect("2-byte scale"));
        let scale = f16_bits_to_f32(scale_bits);
        for i in 0..Self::WEIGHTS_PER_BLOCK {
            dest[dest_offset + i] = f32::from(block_data[2 + i] as i8) * scale;
        }
        Ok(())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputProjectionLayout {
    Descriptor,
    TokenMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquareLinearLayout {
    Descriptor,
    Transposed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RectangularLinearLayout {
    Auto,
    Descriptor,
    Transposed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaHeadMapping {
    Grouped,
    Modulo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionScoreScale {
    HeadDim,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearAccumulationPrecision {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnGateUpOrder {
    GateUp,
    UpGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaZeroTarget {
    Attention,
    Ffn,
}

impl OutputProjectionLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::Descriptor => "descriptor",
            Self::TokenMajor => "token_major",
        }
    }
}

impl SquareLinearLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::Descriptor => "descriptor",
            Self::Transposed => "transposed",
        }
    }
}

impl RectangularLinearLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Descriptor => "descriptor",
            Self::Transposed => "transposed",
        }
    }
}

impl GqaHeadMapping {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grouped => "grouped",
            Self::Modulo => "modulo",
        }
    }
}

impl AttentionScoreScale {
    pub fn label(self) -> &'static str {
        match self {
            Self::HeadDim => "head_dim",
            Self::None => "none",
        }
    }
}

impl LinearAccumulationPrecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl FfnGateUpOrder {
    pub fn label(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::UpGate => "up_gate",
        }
    }
}

pub fn diagnostic_zero_delta(target: DeltaZeroTarget, layer_idx: usize) -> Result<bool> {
    let key = diagnostic_zero_delta_key(target);
    match env::var(key) {
        Ok(value) => diagnostic_zero_delta_value(key, &value, layer_idx),
        Err(env::VarError::NotPresent) => Ok(false),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid {key}: {err}"
        ))),
    }
}

pub fn diagnostic_zero_delta_selector(target: DeltaZeroTarget) -> Result<String> {
    let key = diagnostic_zero_delta_key(target);
    match env::var(key) {
        Ok(value) => {
            let trimmed = value.trim();
            diagnostic_zero_delta_value(key, trimmed, 0)?;
            Ok(if trimmed.is_empty() {
                "none".to_string()
            } else {
                trimmed.to_string()
            })
        }
        Err(env::VarError::NotPresent) => Ok("none".to_string()),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid {key}: {err}"
        ))),
    }
}

fn diagnostic_zero_delta_key(target: DeltaZeroTarget) -> &'static str {
    match target {
        DeltaZeroTarget::Attention => "CAMELID_ZERO_ATTENTION_DELTA",
        DeltaZeroTarget::Ffn => "CAMELID_ZERO_FFN_DELTA",
    }
}

fn diagnostic_zero_delta_value(key: &str, value: &str, layer_idx: usize) -> Result<bool> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "none" || trimmed == "false" || trimmed == "off" {
        return Ok(false);
    }
    if trimmed == "all" || trimmed == "true" || trimmed == "on" {
        return Ok(true);
    }

    for item in trimmed.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let parsed = item.parse::<usize>().map_err(|err| {
            BackendError::InvalidModelMetadata(format!(
                "invalid {key} layer selector {item:?}: {err}; expected all, none, or comma-separated layer indices"
            ))
        })?;
        if parsed == layer_idx {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn diagnostic_gqa_head_mapping() -> Result<GqaHeadMapping> {
    match env::var("CAMELID_GQA_HEAD_MAPPING") {
        Ok(value) if value == "modulo" => Ok(GqaHeadMapping::Modulo),
        Ok(value) if value == "grouped" || value.is_empty() => Ok(GqaHeadMapping::Grouped),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_GQA_HEAD_MAPPING {value:?}; expected grouped or modulo"
        ))),
        Err(env::VarError::NotPresent) => Ok(GqaHeadMapping::Grouped),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_GQA_HEAD_MAPPING: {err}"
        ))),
    }
}

pub fn diagnostic_attention_score_scale() -> Result<AttentionScoreScale> {
    match env::var("CAMELID_ATTENTION_SCORE_SCALE") {
        Ok(value) if value == "none" => Ok(AttentionScoreScale::None),
        Ok(value) if value == "head_dim" || value.is_empty() => Ok(AttentionScoreScale::HeadDim),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ATTENTION_SCORE_SCALE {value:?}; expected head_dim or none"
        ))),
        Err(env::VarError::NotPresent) => Ok(AttentionScoreScale::HeadDim),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ATTENTION_SCORE_SCALE: {err}"
        ))),
    }
}

pub fn diagnostic_linear_accumulation_precision() -> Result<LinearAccumulationPrecision> {
    match env::var("CAMELID_LINEAR_ACCUMULATION") {
        Ok(value) if value == "f64" => Ok(LinearAccumulationPrecision::F64),
        Ok(value) if value == "f32" || value.is_empty() => Ok(LinearAccumulationPrecision::F32),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_LINEAR_ACCUMULATION {value:?}; expected f32 or f64"
        ))),
        Err(env::VarError::NotPresent) => Ok(LinearAccumulationPrecision::F32),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_LINEAR_ACCUMULATION: {err}"
        ))),
    }
}

pub fn diagnostic_ffn_gate_up_order() -> Result<FfnGateUpOrder> {
    match env::var("CAMELID_FFN_GATE_UP_ORDER") {
        Ok(value) if value == "up_gate" => Ok(FfnGateUpOrder::UpGate),
        Ok(value) if value == "gate_up" || value.is_empty() => Ok(FfnGateUpOrder::GateUp),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_FFN_GATE_UP_ORDER {value:?}; expected gate_up or up_gate"
        ))),
        Err(env::VarError::NotPresent) => Ok(FfnGateUpOrder::GateUp),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_FFN_GATE_UP_ORDER: {err}"
        ))),
    }
}

#[inline(always)]
fn apply_ffn_gate_up_order(gate_value: f32, up_value: f32, order: FfnGateUpOrder) -> f32 {
    match order {
        FfnGateUpOrder::GateUp => (gate_value / (1.0 + (-gate_value).exp())) * up_value,
        FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * gate_value,
    }
}

fn attention_score_scale_value(head_dim: usize, mode: AttentionScoreScale) -> f32 {
    match mode {
        AttentionScoreScale::HeadDim => 1.0 / (head_dim as f32).sqrt(),
        AttentionScoreScale::None => 1.0,
    }
}

fn map_attention_head_to_kv_head(
    attention_head: usize,
    repeats: usize,
    kv_heads: usize,
    mapping: GqaHeadMapping,
) -> usize {
    match mapping {
        GqaHeadMapping::Grouped => attention_head / repeats,
        GqaHeadMapping::Modulo => attention_head % kv_heads,
    }
}

pub fn diagnostic_output_projection_layout() -> Result<OutputProjectionLayout> {
    match env::var("CAMELID_OUTPUT_PROJECTION_LAYOUT") {
        Ok(value) if value == "descriptor" => Ok(OutputProjectionLayout::Descriptor),
        Ok(value) if value == "token_major" || value.is_empty() => Ok(OutputProjectionLayout::TokenMajor),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_OUTPUT_PROJECTION_LAYOUT {value:?}; expected descriptor or token_major"
        ))),
        Err(env::VarError::NotPresent) => Ok(OutputProjectionLayout::TokenMajor),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_OUTPUT_PROJECTION_LAYOUT: {err}"
        ))),
    }
}

pub fn diagnostic_square_linear_layout() -> Result<SquareLinearLayout> {
    match env::var("CAMELID_SQUARE_LINEAR_LAYOUT") {
        Ok(value) if value == "transposed" => Ok(SquareLinearLayout::Transposed),
        Ok(value) if value == "descriptor" || value.is_empty() => {
            Ok(SquareLinearLayout::Descriptor)
        }
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_SQUARE_LINEAR_LAYOUT {value:?}; expected descriptor or transposed"
        ))),
        Err(env::VarError::NotPresent) => Ok(SquareLinearLayout::Transposed),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_SQUARE_LINEAR_LAYOUT: {err}"
        ))),
    }
}

pub fn diagnostic_rectangular_linear_layout() -> Result<RectangularLinearLayout> {
    diagnostic_rectangular_linear_layout_env("CAMELID_RECTANGULAR_LINEAR_LAYOUT")
}

pub fn diagnostic_rectangular_linear_layout_for_role(
    role: &str,
) -> Result<RectangularLinearLayout> {
    let role_key = role
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let key = format!("CAMELID_RECTANGULAR_LINEAR_LAYOUT_{role_key}");
    match env::var(&key) {
        Ok(_) => diagnostic_rectangular_linear_layout_env(&key),
        Err(env::VarError::NotPresent) => diagnostic_rectangular_linear_layout(),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid {key}: {err}"
        ))),
    }
}

fn diagnostic_rectangular_linear_layout_env(key: &str) -> Result<RectangularLinearLayout> {
    match env::var(key) {
        Ok(value) if value == "descriptor" => Ok(RectangularLinearLayout::Descriptor),
        Ok(value) if value == "transposed" => Ok(RectangularLinearLayout::Transposed),
        Ok(value) if value == "auto" || value.is_empty() => Ok(RectangularLinearLayout::Auto),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported {key} {value:?}; expected auto, descriptor, or transposed"
        ))),
        Err(env::VarError::NotPresent) => Ok(RectangularLinearLayout::Auto),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid {key}: {err}"
        ))),
    }
}

pub fn diagnostic_rms_norm_epsilon(config_epsilon: f32) -> Result<f32> {
    match env::var("CAMELID_RMS_NORM_EPSILON") {
        Ok(value) if value.is_empty() => Ok(config_epsilon),
        Ok(value) => {
            let epsilon = value.parse::<f32>().map_err(|err| {
                BackendError::InvalidModelMetadata(format!(
                    "invalid CAMELID_RMS_NORM_EPSILON {value:?}: {err}"
                ))
            })?;
            if !epsilon.is_finite() || epsilon < 0.0 {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "unsupported CAMELID_RMS_NORM_EPSILON {value:?}; expected a finite non-negative float"
                )));
            }
            Ok(epsilon)
        }
        Err(env::VarError::NotPresent) => Ok(config_epsilon),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_RMS_NORM_EPSILON: {err}"
        ))),
    }
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
    pub output_row_first_values: Vec<f32>,
    pub component_products_first_values: Vec<f32>,
    pub component_products_max_abs_window_start: usize,
    pub component_products_max_abs_window: Vec<f32>,
    pub max_abs_component_index: usize,
    pub max_abs_component: f32,
    pub positive_component_sum: f32,
    pub negative_component_sum: f32,
    pub top_positive_components: Vec<LlamaOutputProjectionComponentDiagnostic>,
    pub top_negative_components: Vec<LlamaOutputProjectionComponentDiagnostic>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaOutputProjectionComponentDiagnostic {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_hidden_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_norm_weight_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_norm_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reconstructed_output_norm_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_norm_reconstruction_delta: Option<f32>,
    pub output_norm_value: f32,
    pub output_row_value: f32,
    pub component: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaLayerDiagnostics {
    pub layer_index: usize,
    pub residual_flow: LlamaResidualFlowDiagnostic,
    pub attention_norm: LlamaTensorStats,
    pub attention_norm_reconstruction: LlamaRmsNormDiagnostic,
    pub attention_q: LlamaTensorStats,
    pub attention_q_reconstruction: LlamaLinearProjectionDiagnostic,
    pub attention_k: LlamaTensorStats,
    pub attention_k_reconstruction: LlamaLinearProjectionDiagnostic,
    pub attention_q_rope: LlamaTensorStats,
    pub attention_q_rope_reconstruction: LlamaRopeDiagnostic,
    pub attention_k_rope: LlamaTensorStats,
    pub attention_k_rope_reconstruction: LlamaRopeDiagnostic,
    pub attention_v: LlamaTensorStats,
    pub attention_v_reconstruction: LlamaLinearProjectionDiagnostic,
    pub kv_cache_trace: LlamaKvCacheTrace,
    pub attention_trace: LlamaAttentionTrace,
    pub attention_context: LlamaTensorStats,
    pub attention_output: LlamaTensorStats,
    pub attention_output_reconstruction: LlamaLinearProjectionDiagnostic,
    pub attention_residual: LlamaTensorStats,
    pub ffn_norm: LlamaTensorStats,
    pub ffn_norm_reconstruction: LlamaRmsNormDiagnostic,
    pub ffn_gate: LlamaTensorStats,
    pub ffn_gate_reconstruction: LlamaLinearProjectionDiagnostic,
    pub ffn_up: LlamaTensorStats,
    pub ffn_up_reconstruction: LlamaLinearProjectionDiagnostic,
    pub ffn_activation: LlamaTensorStats,
    pub ffn_activation_reconstruction: LlamaFfnActivationDiagnostic,
    pub ffn_output: LlamaTensorStats,
    pub ffn_down_reconstruction: LlamaLinearProjectionDiagnostic,
    pub ffn_residual: LlamaTensorStats,
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

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaLinearProjectionDiagnostic {
    pub role: String,
    pub layout: String,
    pub input_width: usize,
    pub output_width: usize,
    pub weight_shape: Vec<usize>,
    pub input_first_values: Vec<f32>,
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
pub struct LlamaFfnActivationDiagnostic {
    pub gate_width: usize,
    pub activation_order: &'static str,
    pub gate_first_values: Vec<f32>,
    pub up_first_values: Vec<f32>,
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
pub struct LlamaResidualFlowDiagnostic {
    pub attention_input: LlamaTensorStats,
    pub attention_delta: LlamaResidualReconstructionDiagnostic,
    pub ffn_input: LlamaTensorStats,
    pub ffn_delta: LlamaResidualReconstructionDiagnostic,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaResidualReconstructionDiagnostic {
    pub input_rms: f32,
    pub delta_rms: f32,
    pub reported_rms: f32,
    pub delta_to_input_rms_ratio: f32,
    pub delta_input_cosine_similarity: f32,
    pub input_first_values: Vec<f32>,
    pub delta_first_values: Vec<f32>,
    pub reconstructed_first_values: Vec<f32>,
    pub reported_first_values: Vec<f32>,
    pub reported_max_abs_index: usize,
    pub reported_max_abs: f32,
    pub reported_max_abs_window_start: usize,
    pub reported_max_abs_window: Vec<f32>,
    pub reconstructed_reported_max_abs_window: Vec<f32>,
    pub delta_reported_max_abs_window: Vec<f32>,
    pub max_abs_delta_index: usize,
    pub max_abs_delta: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaRmsNormDiagnostic {
    pub epsilon: f32,
    pub input_mean_square: f32,
    pub input_rms: f32,
    pub scale: f32,
    pub input_first_values: Vec<f32>,
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
pub struct LlamaRopeDiagnostic {
    pub role: String,
    pub pairing: &'static str,
    pub direction: &'static str,
    pub position_mode: &'static str,
    pub frequency_source: &'static str,
    pub rope_freqs_count: Option<usize>,
    pub scaling_type: &'static str,
    pub scaling_factor: f32,
    pub scaling_original_context_length: Option<u32>,
    pub scaling_low_freq_factor: Option<f32>,
    pub scaling_high_freq_factor: Option<f32>,
    pub position: usize,
    pub effective_position: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub freq_base: f32,
    pub input_first_values: Vec<f32>,
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
pub struct LlamaAttentionTrace {
    pub scale: f32,
    pub position_count: usize,
    pub head_dim: usize,
    pub heads: Vec<LlamaAttentionHeadTrace>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaAttentionHeadTrace {
    pub attention_head: usize,
    pub kv_head: usize,
    pub query_first_values: Vec<f32>,
    pub context_first_values: Vec<f32>,
    pub reconstructed_context_first_values: Vec<f32>,
    pub context_reconstruction_max_abs_delta_index: usize,
    pub context_reconstruction_max_abs_delta: f32,
    pub probability_sum: f32,
    pub probability_entropy: f32,
    pub probability_rms: f32,
    pub max_probability_position: usize,
    pub max_probability: f32,
    pub top_probability_positions: Vec<LlamaAttentionTopProbabilityTrace>,
    pub positions: Vec<LlamaAttentionPositionTrace>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaAttentionTopProbabilityTrace {
    pub position: usize,
    pub score: f32,
    pub probability: f32,
    pub key_first_values: Vec<f32>,
    pub value_first_values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaAttentionPositionTrace {
    pub position: usize,
    pub score: f32,
    pub reconstructed_score: f32,
    pub score_reconstruction_delta: f32,
    pub probability: f32,
    pub key_first_values: Vec<f32>,
    pub qk_products_first_values: Vec<f32>,
    pub qk_products_max_abs_window_start: usize,
    pub qk_products_max_abs_window: Vec<f32>,
    pub value_first_values: Vec<f32>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LlamaForwardTimings {
    pub total: u128,
    pub embedding: u128,
    pub layers_total: u128,
    pub final_norm: u128,
    pub logits: u128,
    pub layers: Vec<LlamaLayerTimings>,
    pub memory: Option<LlamaForwardMemoryTimings>,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ScheduleTelemetry {
    pub rayon_fanout_boundaries: u64,
    pub i8mm_single_projection_calls: u64,
    pub i8mm_fused_gate_up_calls: u64,
    pub i8mm_single_projection_by_role: HashMap<String, LlamaQ8ScheduleRoleTelemetry>,
    pub output_projection_calls: u64,
    pub output_projection_by_route: HashMap<String, LlamaQ8OutputProjectionRouteTelemetry>,
    pub output_projection_by_layer_route:
        HashMap<String, LlamaQ8OutputProjectionLayerRouteTelemetry>,
    pub projection_route_denials: HashMap<String, LlamaQ8ProjectionRouteDenialTelemetry>,
    pub ffn_gate_up_decode_consumer_activation_us: u64,
    pub ffn_gate_up_decode_consumer_tensor_us: u64,
    pub activation_pack_calls: u64,
    pub activation_pack_rows: u64,
    pub activation_pack_bytes_requested: u64,
    pub scratch_allocation_count: u64,
    pub scratch_bytes_allocated: u64,
    pub scratch_bytes_reused: u64,
    pub scratch_peak_capacity_bytes: u64,
    pub activation_quantize_pack_us: u64,
    pub q8_gemm_compute_us: u64,
    pub conservative_tail_rows: u64,
    pub ffn_down_gemm4_prefill_candidates: u64,
    pub ffn_down_gemm4_prefill_reject_plan_off: u64,
    pub ffn_down_gemm4_prefill_reject_rows_lt4: u64,
    pub ffn_down_gemm4_prefill_reject_bad_input_width: u64,
    pub ffn_down_gemm4_prefill_reject_no_runtime_packed: u64,
    pub ffn_down_gemm4_prefill_reject_non_i8_interleave: u64,
    pub ffn_down_decode_consumer_taken: u64,
    pub ffn_down_vnni_decode_candidates: u64,
    pub ffn_down_vnni_decode_taken: u64,
    pub ffn_down_vnni_decode_quantize_us: u64,
    pub ffn_down_vnni_decode_kernel_us: u64,
    pub ffn_down_vnni_decode_reject_gate_off: u64,
    pub ffn_down_vnni_decode_reject_cpu_feature: u64,
    pub ffn_down_vnni_decode_reject_no_vnni_pack: u64,
    pub ffn_down_vnni_decode_reject_bad_input_width: u64,
    pub ffn_down_vnni_decode_reject_bad_output_width: u64,
    pub ffn_down_vnni_decode_reject_shape_or_role: u64,
    pub prefill_single_token_fallbacks: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ScheduleRoleTelemetry {
    pub calls: u64,
    pub rows: u64,
    pub pack_us: u64,
    pub gemm_us: u64,
    pub tail_rows: u64,
    pub rayon_fanout_boundaries: u64,
    pub scheduler_chunk_calls: u64,
    pub scheduler_output_groups: u64,
    pub scheduler_row_groups: u64,
    pub scheduler_groups_per_chunk: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8OutputProjectionRouteTelemetry {
    pub role: String,
    pub route: String,
    pub calls: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub elapsed_us: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8OutputProjectionLayerRouteTelemetry {
    pub layer_index: usize,
    pub role: String,
    pub route: String,
    pub calls: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub elapsed_us: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ProjectionRouteDenialTelemetry {
    pub role: String,
    pub route: String,
    pub reason: String,
    pub denials: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
}

static Q8_SCHED_RAYON_FANOUT_BOUNDARIES: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_OUTPUT_PROJECTION_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_ACTIVATION_PACK_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_ACTIVATION_PACK_ROWS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_SCRATCH_ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_SCRATCH_BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_SCRATCH_BYTES_REUSED: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_Q8_GEMM_COMPUTE_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_CONSERVATIVE_TAIL_ROWS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE: OnceLock<
    Mutex<HashMap<String, LlamaQ8ScheduleRoleTelemetry>>,
> = OnceLock::new();
static Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE: OnceLock<
    Mutex<HashMap<String, LlamaQ8OutputProjectionRouteTelemetry>>,
> = OnceLock::new();
static Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE: OnceLock<
    Mutex<HashMap<String, LlamaQ8OutputProjectionLayerRouteTelemetry>>,
> = OnceLock::new();
static Q8_SCHED_PROJECTION_ROUTE_DENIALS: OnceLock<
    Mutex<HashMap<String, LlamaQ8ProjectionRouteDenialTelemetry>>,
> = OnceLock::new();
#[cfg(not(test))]
static Q8_SCHED_TELEMETRY_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn q8_schedule_telemetry_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(Q8_SCHEDULE_TELEMETRY_ENV)
    }
    #[cfg(not(test))]
    *Q8_SCHED_TELEMETRY_ENABLED.get_or_init(|| env_flag_enabled(Q8_SCHEDULE_TELEMETRY_ENV))
}

pub fn reset_q8_schedule_telemetry() {
    Q8_SCHED_RAYON_FANOUT_BOUNDARIES.store(0, Ordering::Relaxed);
    Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_OUTPUT_PROJECTION_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_ROWS.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_BYTES_ALLOCATED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_BYTES_REUSED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.store(0, Ordering::Relaxed);
    Q8_SCHED_Q8_GEMM_COMPUTE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_CONSERVATIVE_TAIL_ROWS.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE.store(0, Ordering::Relaxed);
    Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS.store(0, Ordering::Relaxed);
    if let Some(by_role) = Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE.get() {
        by_role.lock().expect("q8 role telemetry mutex").clear();
    }
    if let Some(by_route) = Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE.get() {
        by_route
            .lock()
            .expect("q8 output projection telemetry mutex")
            .clear();
    }
    if let Some(by_layer_route) = Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE.get() {
        by_layer_route
            .lock()
            .expect("q8 output projection layer route telemetry mutex")
            .clear();
    }
    if let Some(denials) = Q8_SCHED_PROJECTION_ROUTE_DENIALS.get() {
        denials
            .lock()
            .expect("q8 projection route denial telemetry mutex")
            .clear();
    }
}

pub fn snapshot_q8_schedule_telemetry() -> LlamaQ8ScheduleTelemetry {
    LlamaQ8ScheduleTelemetry {
        rayon_fanout_boundaries: Q8_SCHED_RAYON_FANOUT_BOUNDARIES.load(Ordering::Relaxed),
        i8mm_single_projection_calls: Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS.load(Ordering::Relaxed),
        i8mm_fused_gate_up_calls: Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS.load(Ordering::Relaxed),
        i8mm_single_projection_by_role: Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE
            .get()
            .map(|by_role| by_role.lock().expect("q8 role telemetry mutex").clone())
            .unwrap_or_default(),
        output_projection_calls: Q8_SCHED_OUTPUT_PROJECTION_CALLS.load(Ordering::Relaxed),
        output_projection_by_route: Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE
            .get()
            .map(|by_route| {
                by_route
                    .lock()
                    .expect("q8 output projection telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        output_projection_by_layer_route: Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE
            .get()
            .map(|by_layer_route| {
                by_layer_route
                    .lock()
                    .expect("q8 output projection layer route telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        projection_route_denials: Q8_SCHED_PROJECTION_ROUTE_DENIALS
            .get()
            .map(|denials| {
                denials
                    .lock()
                    .expect("q8 projection route denial telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        ffn_gate_up_decode_consumer_activation_us:
            Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US.load(Ordering::Relaxed),
        ffn_gate_up_decode_consumer_tensor_us: Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US
            .load(Ordering::Relaxed),
        activation_pack_calls: Q8_SCHED_ACTIVATION_PACK_CALLS.load(Ordering::Relaxed),
        activation_pack_rows: Q8_SCHED_ACTIVATION_PACK_ROWS.load(Ordering::Relaxed),
        activation_pack_bytes_requested: Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED
            .load(Ordering::Relaxed),
        scratch_allocation_count: Q8_SCHED_SCRATCH_ALLOCATION_COUNT.load(Ordering::Relaxed),
        scratch_bytes_allocated: Q8_SCHED_SCRATCH_BYTES_ALLOCATED.load(Ordering::Relaxed),
        scratch_bytes_reused: Q8_SCHED_SCRATCH_BYTES_REUSED.load(Ordering::Relaxed),
        scratch_peak_capacity_bytes: Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES.load(Ordering::Relaxed),
        activation_quantize_pack_us: Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.load(Ordering::Relaxed),
        q8_gemm_compute_us: Q8_SCHED_Q8_GEMM_COMPUTE_US.load(Ordering::Relaxed),
        conservative_tail_rows: Q8_SCHED_CONSERVATIVE_TAIL_ROWS.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_candidates: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_plan_off: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_rows_lt4: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_bad_input_width:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_no_runtime_packed:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_non_i8_interleave:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE.load(Ordering::Relaxed),
        ffn_down_decode_consumer_taken: Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_candidates: Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_taken: Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN.load(Ordering::Relaxed),
        ffn_down_vnni_decode_quantize_us: Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_kernel_us: Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_gate_off: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_cpu_feature: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_no_vnni_pack: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_bad_input_width:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_bad_output_width:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_shape_or_role:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE.load(Ordering::Relaxed),
        prefill_single_token_fallbacks: Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS
            .load(Ordering::Relaxed),
    }
}

#[allow(dead_code)]
fn add_q8_schedule_counter(counter: &AtomicU64, value: u64) {
    if q8_schedule_telemetry_enabled() && value > 0 {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

#[allow(dead_code)]
fn update_q8_schedule_peak(counter: &AtomicU64, value: u64) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let mut current = counter.load(Ordering::Relaxed);
    while value > current {
        match counter.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

#[allow(dead_code)]
fn record_q8_schedule_activation_pack(
    packed_inputs: &mut Vec<Q8_0PackedRows4Block>,
    before_capacity: usize,
    packed_rows: usize,
    blocks_per_row: usize,
    elapsed_us: u128,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let requested_blocks = packed_rows / 4 * blocks_per_row;
    let requested_bytes = requested_blocks.saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    let before_capacity_bytes =
        before_capacity.saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    let after_capacity_bytes = packed_inputs
        .capacity()
        .saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    Q8_SCHED_ACTIVATION_PACK_CALLS.fetch_add(1, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_ROWS.fetch_add(packed_rows as u64, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED.fetch_add(requested_bytes as u64, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.fetch_add(elapsed_us as u64, Ordering::Relaxed);
    if before_capacity < requested_blocks {
        Q8_SCHED_SCRATCH_ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        Q8_SCHED_SCRATCH_BYTES_ALLOCATED.fetch_add(
            after_capacity_bytes.saturating_sub(before_capacity_bytes) as u64,
            Ordering::Relaxed,
        );
    } else {
        Q8_SCHED_SCRATCH_BYTES_REUSED.fetch_add(requested_bytes as u64, Ordering::Relaxed);
    }
    update_q8_schedule_peak(
        &Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES,
        after_capacity_bytes as u64,
    );
}

#[allow(dead_code)]
fn record_q8_schedule_i8mm_single_projection_role_call(
    role: &str,
    rows: u64,
    rayon_fanout_boundaries: u64,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.calls = entry.calls.saturating_add(1);
        entry.rows = entry.rows.saturating_add(rows);
        entry.rayon_fanout_boundaries = entry
            .rayon_fanout_boundaries
            .saturating_add(rayon_fanout_boundaries);
    });
}

#[allow(dead_code)]
fn record_q8_schedule_i8mm_single_projection_role_pack(role: &str, elapsed_us: u128) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.pack_us = entry.pack_us.saturating_add(elapsed_us as u64);
    });
}

#[allow(dead_code)]
fn record_q8_schedule_i8mm_single_projection_role_gemm(role: &str, elapsed_us: u128) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.gemm_us = entry.gemm_us.saturating_add(elapsed_us as u64);
    });
}

#[allow(dead_code)]
fn record_q8_schedule_i8mm_single_projection_role_tail(role: &str, tail_rows: u64) {
    if !q8_schedule_telemetry_enabled() || tail_rows == 0 {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.tail_rows = entry.tail_rows.saturating_add(tail_rows);
    });
}

#[allow(dead_code)]
fn record_q8_schedule_i8mm_single_projection_role_scheduler(
    role: &str,
    output_groups: u64,
    row_groups: u64,
    groups_per_chunk: u64,
) {
    if !q8_schedule_telemetry_enabled() || output_groups == 0 {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.scheduler_chunk_calls = entry.scheduler_chunk_calls.saturating_add(1);
        entry.scheduler_output_groups = entry.scheduler_output_groups.saturating_add(output_groups);
        entry.scheduler_row_groups = entry.scheduler_row_groups.saturating_add(row_groups);
        entry.scheduler_groups_per_chunk = entry
            .scheduler_groups_per_chunk
            .saturating_add(groups_per_chunk.max(1));
    });
}

#[allow(dead_code)]
fn update_q8_schedule_role(role: &str, update: impl FnOnce(&mut LlamaQ8ScheduleRoleTelemetry)) {
    let by_role =
        Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_role = by_role.lock().expect("q8 role telemetry mutex");
    update(by_role.entry(role.to_string()).or_default());
}

#[allow(dead_code)]
fn record_q8_schedule_output_projection_route_call(
    role: &str,
    route: &str,
    projection_name: Option<&str>,
    rows: usize,
    input_width: usize,
    output_width: usize,
    elapsed_us: u128,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    Q8_SCHED_OUTPUT_PROJECTION_CALLS.fetch_add(1, Ordering::Relaxed);
    let key = format!("{role}.{route}");
    let by_route = Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_route = by_route
        .lock()
        .expect("q8 output projection telemetry mutex");
    let entry = by_route
        .entry(key)
        .or_insert_with(|| LlamaQ8OutputProjectionRouteTelemetry {
            role: role.to_string(),
            route: route.to_string(),
            ..LlamaQ8OutputProjectionRouteTelemetry::default()
        });
    entry.calls = entry.calls.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
    entry.elapsed_us = entry.elapsed_us.saturating_add(elapsed_us as u64);

    let Some(layer_index) = projection_name.and_then(q8_schedule_layer_index_for_projection_name)
    else {
        return;
    };
    let key = format!("layer_{layer_index}.{role}.{route}");
    let by_layer_route =
        Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_layer_route = by_layer_route
        .lock()
        .expect("q8 output projection layer route telemetry mutex");
    let entry =
        by_layer_route
            .entry(key)
            .or_insert_with(|| LlamaQ8OutputProjectionLayerRouteTelemetry {
                layer_index,
                role: role.to_string(),
                route: route.to_string(),
                ..LlamaQ8OutputProjectionLayerRouteTelemetry::default()
            });
    entry.calls = entry.calls.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
    entry.elapsed_us = entry.elapsed_us.saturating_add(elapsed_us as u64);
}

fn q8_schedule_layer_index_for_projection_name(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("layer_")?;
    let (digits, _) = rest.split_once('_')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn record_q8_schedule_projection_route_elapsed(
    role: &str,
    route: &str,
    projection_name: &str,
    rows: usize,
    input_width: usize,
    output_width: usize,
    started: Option<Instant>,
) {
    if let Some(started) = started {
        record_q8_schedule_output_projection_route_call(
            role,
            route,
            Some(projection_name),
            rows,
            input_width,
            output_width,
            started.elapsed().as_micros(),
        );
    }
}

#[allow(dead_code)]
fn record_q8_schedule_projection_route_denial(
    role: &str,
    route: &str,
    reason: &str,
    rows: usize,
    input_width: usize,
    output_width: usize,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let key = format!("{role}.{route}.{reason}");
    let denials = Q8_SCHED_PROJECTION_ROUTE_DENIALS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut denials = denials
        .lock()
        .expect("q8 projection route denial telemetry mutex");
    let entry = denials
        .entry(key)
        .or_insert_with(|| LlamaQ8ProjectionRouteDenialTelemetry {
            role: role.to_string(),
            route: route.to_string(),
            reason: reason.to_string(),
            ..LlamaQ8ProjectionRouteDenialTelemetry::default()
        });
    entry.denials = entry.denials.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
}

#[allow(dead_code)]
fn q8_schedule_output_projection_route_kind(
    weight: BorrowedLinearWeight<'_>,
    input_width: usize,
    output_width: usize,
) -> &'static str {
    if weight.source_type == Some(GgufTensorType::Q8_0) {
        if q8_0_selected_borrowed_packed_rows4(weight)
            .filter(|(packed, _)| {
                packed.rows == output_width
                    && packed.blocks_per_row == input_width / Q8_0_BLOCK_VALUES
            })
            .is_some()
        {
            "q8_0_borrowed_packed_rows4"
        } else if weight.q8_0_blocks.is_some() {
            "q8_0_retained_blocks"
        } else if weight.q8_0_file_backing.is_some()
            && weight.cols == input_width
            && weight.rows == output_width
            && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
        {
            "q8_0_file_reader"
        } else {
            "q8_0_f32_fallback"
        }
    } else {
        "f32"
    }
}

#[allow(dead_code)]
fn q8_schedule_role_for_output_name(name: &str) -> &'static str {
    if name.contains("attention_q") || name.contains("attn_q") {
        "attention_q"
    } else if name.contains("attention_k") || name.contains("attn_k") {
        "attention_k"
    } else if name.contains("attention_v") || name.contains("attn_v") {
        "attention_v"
    } else if name.contains("attention_output") || name.contains("attn_output") {
        "attention_output"
    } else if name.contains("ffn_gate") {
        "ffn_gate"
    } else if name.contains("ffn_up") {
        "ffn_up"
    } else if name.contains("ffn_down") {
        "ffn_down"
    } else if name.contains("logits") {
        "logits"
    } else {
        "unknown"
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LlamaLayerTimings {
    pub layer_index: usize,
    pub total: u128,
    pub attention_norm: u128,
    pub attention_q: u128,
    pub attention_k: u128,
    pub attention_v: u128,
    pub attention_rope: u128,
    pub kv_cache_write: u128,
    pub attention_context: u128,
    pub attention_output: u128,
    pub attention_residual: u128,
    pub ffn_norm: u128,
    pub ffn_gate: u128,
    pub ffn_up: u128,
    pub ffn_activation: u128,
    pub ffn_down: u128,
    pub ffn_residual: u128,
    pub memory: Option<LlamaLayerMemoryTimings>,
}

impl From<&LlamaLayerTimings> for LlamaPrefillLayerMajorChunkTimings {
    fn from(value: &LlamaLayerTimings) -> Self {
        Self {
            total: value.total,
            attention_norm: value.attention_norm,
            attention_q: value.attention_q,
            attention_k: value.attention_k,
            attention_v: value.attention_v,
            attention_rope: value.attention_rope,
            kv_cache_write: value.kv_cache_write,
            attention_context: value.attention_context,
            attention_output: value.attention_output,
            attention_residual: value.attention_residual,
            ffn_norm: value.ffn_norm,
            ffn_gate: value.ffn_gate,
            ffn_up: value.ffn_up,
            ffn_activation: value.ffn_activation,
            ffn_down: value.ffn_down,
            ffn_residual: value.ffn_residual,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaForwardMemoryTimings {
    pub forward_passes: usize,
    pub materialization: LlamaWeightMaterializationStats,
    pub q8_file_reads: Q8_0FileReadStats,
    pub q8_file_read_phases: Vec<LlamaQ8FileReadPhaseTrace>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prefill_layer_major_attribution: Vec<LlamaPrefillLayerMajorChunkAttribution>,
    #[serde(skip)]
    q8_file_read_start: Q8_0FileReadStats,
    #[serde(skip)]
    q8_file_read_phase_start: Q8_0FileReadStats,
    pub start: LlamaMemorySample,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_embedding: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_layers: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_final_norm: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_logits: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_kib: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_delta_kib: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_phase: Option<String>,
    pub layers: Vec<LlamaLayerMemoryTimings>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaLayerMemoryTimings {
    pub layer_index: usize,
    pub forward_passes: usize,
    pub q8_file_reads: Q8_0FileReadStats,
    pub q8_file_read_phases: Vec<LlamaQ8FileReadPhaseTrace>,
    #[serde(skip)]
    q8_file_read_start: Q8_0FileReadStats,
    #[serde(skip)]
    q8_file_read_phase_start: Q8_0FileReadStats,
    pub start: LlamaMemorySample,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_norm: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_q: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_k: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_rope: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_v: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_kv_cache_write: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_context: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_output: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_attention_residual: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_ffn_norm: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_ffn_activation: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_ffn_down: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_ffn_residual: Option<LlamaMemorySample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_kib: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_delta_kib: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_phase: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaMemorySample {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_kib: Option<u64>,
    pub kv_cache_position: usize,
    pub kv_cache_allocated_sequence_length: usize,
    pub kv_cache_allocated_elements: usize,
    pub kv_cache_allocated_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8FileReadPhaseTrace {
    pub phase: String,
    pub q8_file_reads: Q8_0FileReadStats,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaPrefillLayerMajorChunkAttribution {
    pub layer_index: usize,
    pub chunk_start: usize,
    pub chunk_rows: usize,
    pub base_position: usize,
    pub hidden_bytes: u64,
    pub next_hidden_bytes: u64,
    pub chunk_input_bytes: u64,
    pub kv_cache_bytes_before: u64,
    pub kv_cache_bytes_after: u64,
    pub q8_file_reads: Q8_0FileReadStats,
    pub timings: LlamaPrefillLayerMajorChunkTimings,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaPrefillLayerMajorChunkTimings {
    pub total: u128,
    pub attention_norm: u128,
    pub attention_q: u128,
    pub attention_k: u128,
    pub attention_v: u128,
    pub attention_rope: u128,
    pub kv_cache_write: u128,
    pub attention_context: u128,
    pub attention_output: u128,
    pub attention_residual: u128,
    pub ffn_norm: u128,
    pub ffn_gate: u128,
    pub ffn_up: u128,
    pub ffn_activation: u128,
    pub ffn_down: u128,
    pub ffn_residual: u128,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaWeightMaterializationStats {
    pub tensor_count: usize,
    pub dense_f32_tensor_count: usize,
    pub dense_f32_bytes: u64,
    pub q8_0_source_tensor_count: usize,
    pub q8_0_f32_materialized_tensor_count: usize,
    pub q8_0_f32_materialized_bytes: u64,
    pub q8_0_file_backed_tensor_count: usize,
    pub q8_0_file_backed_storage_bytes: u64,
    pub q8_0_file_backed_f32_bytes_avoided: u64,
    pub q8_0_file_backed_retained_block_bytes_if_enabled: u64,
    pub q8_0_file_handle_cached_count: usize,
    pub q8_0_retained_block_tensor_count: usize,
    pub q8_0_retained_block_bytes: u64,
    pub has_q8_0_f32_materialization: bool,
    pub has_lazy_q8_0_file_backing: bool,
    pub has_retained_q8_0_blocks: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTimedForwardOutput {
    pub output: LlamaForwardOutput,
    pub timings: LlamaForwardTimings,
    pub diagnostics: LlamaForwardDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
struct LlamaFastForwardOutput {
    output: LlamaForwardOutput,
    timings: LlamaForwardTimings,
    diagnostics: Option<LlamaForwardDiagnostics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: Vec<(usize, f32)>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlamaSampler {
    Greedy,
    Sampling(SamplingConfig),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaGenerationStep {
    pub prompt_token_count: usize,
    pub prefill_token_count: usize,
    pub next_token_id: u32,
    pub logits: CpuTensor,
    pub hidden_state: CpuTensor,
    pub output_norm_state: CpuTensor,
    pub timings: LlamaForwardTimings,
    pub prefill_timings: LlamaForwardTimings,
    pub first_token_timings: LlamaForwardTimings,
    pub sample: u128,
    pub diagnostics: Option<LlamaForwardDiagnostics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaInferenceSession {
    pub config: LlamaModelConfig,
    pub weights: Arc<LlamaLoadedWeights>,
    pub kv_cache: LlamaKvCache,
}

impl LlamaInferenceSession {
    pub fn new(
        config: LlamaModelConfig,
        weights: impl Into<Arc<LlamaLoadedWeights>>,
    ) -> Result<Self> {
        let weights = weights.into();
        weights.validate_dense_shapes(&config)?;
        let plan = LlamaKvCachePlan::from_config(&config)?;
        Ok(Self {
            config,
            weights,
            kv_cache: LlamaKvCache::new(plan)?,
        })
    }

    pub fn forward_single_token(&mut self, token_id: u32) -> Result<LlamaForwardOutput> {
        Ok(self.forward_single_token_timed_fast(token_id)?.output)
    }

    pub fn forward_single_token_timed(&mut self, token_id: u32) -> Result<LlamaTimedForwardOutput> {
        let timed = self.forward_single_token_timed_internal(token_id, true, true)?;
        Ok(LlamaTimedForwardOutput {
            output: timed.output,
            timings: timed.timings,
            diagnostics: timed
                .diagnostics
                .expect("diagnostics requested for timed forward"),
        })
    }

    fn forward_single_token_timed_fast(&mut self, token_id: u32) -> Result<LlamaFastForwardOutput> {
        self.forward_single_token_timed_internal(token_id, false, true)
    }

    fn forward_prefill_chunk_timed_fast(
        &mut self,
        token_ids: &[u32],
    ) -> Result<LlamaForwardTimings> {
        if token_ids.is_empty() {
            return Ok(LlamaForwardTimings::default());
        }
        if token_ids.len() > self.kv_cache.plan.max_sequence_length - self.kv_cache.position {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "prefill chunk of {} token(s) exceeds remaining context capacity {}",
                token_ids.len(),
                self.kv_cache.plan.max_sequence_length - self.kv_cache.position
            )));
        }

        let chunk_base_position = self.kv_cache.position;
        let total_started = Instant::now();
        let mut memory = structured_forward_memory_enabled().then(|| {
            LlamaForwardMemoryTimings::new(
                capture_memory_sample(&self.kv_cache),
                collect_weight_materialization_stats(&self.weights),
                q8_0_file_read_stats(),
            )
        });
        trace_forward_memory("prefill_chunk_start");
        let embedding_started = Instant::now();
        let mut hidden = self
            .weights
            .token_embedding
            .embedding_lookup(token_ids, "token_embedding_prefill_chunk")?;
        let mut timings = LlamaForwardTimings {
            embedding: embedding_started.elapsed().as_micros(),
            ..LlamaForwardTimings::default()
        };
        if let Some(memory) = &mut memory {
            memory.record_after_embedding(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_chunk_embedding_done");
        let layers_started = Instant::now();
        let rms_norm_epsilon = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let timed = forward_prefill_layer_chunk_timed(
                &hidden,
                layer,
                PrefillLayerChunkParams {
                    config: &self.config,
                    rope_freqs: self.weights.rope_freqs.as_ref(),
                    rms_norm_epsilon,
                    layer_idx,
                    base_position: chunk_base_position,
                    chunk_start: 0,
                    chunk_rows: token_ids.len(),
                },
                &mut self.kv_cache,
            )?;
            hidden = timed.output;
            if let (Some(memory), Some(layer_memory)) = (&mut memory, &timed.timings.memory) {
                memory.record_layer(layer_memory.clone());
            }
            timings.layers.push(timed.timings);
        }
        timings.layers_total = layers_started.elapsed().as_micros();
        if let Some(memory) = &mut memory {
            memory.record_after_layers(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_chunk_layers_done");
        self.kv_cache.position += token_ids.len();
        timings.total = total_started.elapsed().as_micros();
        if let Some(memory) = &mut memory {
            memory.record_end(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_chunk_end");
        timings.memory = memory;
        Ok(timings)
    }

    fn forward_prefill_layer_major_timed_fast(
        &mut self,
        token_ids: &[u32],
        chunk_tokens: usize,
    ) -> Result<LlamaForwardTimings> {
        let forward_passes = if token_ids.is_empty() {
            0
        } else {
            token_ids.len().div_ceil(chunk_tokens)
        };
        let q8_file_cache_capacity =
            prefill_layer_major_q8_file_cache_capacity_override(&self.weights, forward_passes);
        with_q8_file_cache_capacity_override(q8_file_cache_capacity, || {
            self.forward_prefill_layer_major_timed_fast_inner(token_ids, chunk_tokens)
        })
    }

    fn forward_prefill_layer_major_timed_fast_inner(
        &mut self,
        token_ids: &[u32],
        chunk_tokens: usize,
    ) -> Result<LlamaForwardTimings> {
        if token_ids.is_empty() {
            return Ok(LlamaForwardTimings::default());
        }
        if token_ids.len() > self.kv_cache.plan.max_sequence_length - self.kv_cache.position {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "layer-major prefill of {} token(s) exceeds remaining context capacity {}",
                token_ids.len(),
                self.kv_cache.plan.max_sequence_length - self.kv_cache.position
            )));
        }

        let prefill_base_position = self.kv_cache.position;
        let forward_passes = token_ids.len().div_ceil(chunk_tokens);
        let total_started = Instant::now();
        let mut memory = structured_forward_memory_enabled().then(|| {
            LlamaForwardMemoryTimings::new(
                capture_memory_sample(&self.kv_cache),
                collect_weight_materialization_stats(&self.weights),
                q8_0_file_read_stats(),
            )
        });
        if let Some(memory) = &mut memory {
            memory.forward_passes = forward_passes;
        }
        trace_forward_memory("prefill_layer_major_start");
        let embedding_started = Instant::now();
        let mut hidden = self
            .weights
            .token_embedding
            .embedding_lookup(token_ids, "token_embedding_prefill_layer_major")?;
        let mut timings = LlamaForwardTimings {
            embedding: embedding_started.elapsed().as_micros(),
            ..LlamaForwardTimings::default()
        };
        if let Some(memory) = &mut memory {
            memory.record_after_embedding(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_layer_major_embedding_done");

        let layers_started = Instant::now();
        let rms_norm_epsilon = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        let hidden_width = hidden.dim(1)?;
        let hidden_dims = vec![token_ids.len(), hidden_width];
        let capture_prefill_attribution = prefill_layer_major_attribution_enabled();
        let mut next_hidden = vec![0.0_f32; hidden.data.len()];
        let mut chunk_input_buffer = Vec::with_capacity(chunk_tokens * hidden_width);
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let hidden_bytes = tensor_f32_bytes(&hidden);
            if next_hidden.len() != hidden.data.len() {
                next_hidden.resize(hidden.data.len(), 0.0);
            }
            let mut layer_timings = LlamaLayerTimings {
                layer_index: layer_idx,
                ..LlamaLayerTimings::default()
            };
            for chunk_start in (0..token_ids.len()).step_by(chunk_tokens) {
                let rows_this_chunk = chunk_tokens.min(token_ids.len() - chunk_start);
                let chunk_base_position = prefill_base_position + chunk_start;
                copy_tensor_rows_into_buffer(
                    &hidden,
                    chunk_start,
                    rows_this_chunk,
                    &mut chunk_input_buffer,
                )?;
                let hidden_chunk = CpuTensor::from_f32(
                    format!("layer_{layer_idx}_prefill_layer_major_input_{chunk_start}"),
                    vec![rows_this_chunk, hidden_width],
                    std::mem::take(&mut chunk_input_buffer),
                )?;
                let saved_position = self.kv_cache.position;
                self.kv_cache.position = chunk_base_position;
                let kv_cache_bytes_before = self.kv_cache.allocated_bytes();
                let q8_file_read_start = q8_0_file_read_stats();
                let timed = forward_prefill_layer_chunk_timed(
                    &hidden_chunk,
                    layer,
                    PrefillLayerChunkParams {
                        config: &self.config,
                        rope_freqs: self.weights.rope_freqs.as_ref(),
                        rms_norm_epsilon,
                        layer_idx,
                        base_position: chunk_base_position,
                        chunk_start,
                        chunk_rows: rows_this_chunk,
                    },
                    &mut self.kv_cache,
                );
                let kv_cache_bytes_after = self.kv_cache.allocated_bytes();
                let q8_file_reads =
                    q8_0_file_read_stats().saturating_delta_since(q8_file_read_start);
                self.kv_cache.position = saved_position;
                let timed = timed?;
                let chunk_input_bytes = tensor_f32_bytes(&hidden_chunk);
                chunk_input_buffer = hidden_chunk.data;
                copy_tensor_rows_into(&timed.output, &mut next_hidden, chunk_start, hidden_width)?;
                if capture_prefill_attribution {
                    if let Some(memory) = &mut memory {
                        memory.record_prefill_layer_major_attribution(
                            LlamaPrefillLayerMajorChunkAttribution {
                                layer_index: layer_idx,
                                chunk_start,
                                chunk_rows: rows_this_chunk,
                                base_position: chunk_base_position,
                                hidden_bytes,
                                next_hidden_bytes: vec_f32_bytes(&next_hidden),
                                chunk_input_bytes,
                                kv_cache_bytes_before,
                                kv_cache_bytes_after,
                                q8_file_reads,
                                timings: LlamaPrefillLayerMajorChunkTimings::from(&timed.timings),
                            },
                        );
                    }
                }
                layer_timings.add_assign(&timed.timings);
            }
            if let (Some(memory), Some(layer_memory)) = (&mut memory, &layer_timings.memory) {
                memory.record_layer(layer_memory.clone());
            }
            timings.layers.push(layer_timings);
            std::mem::swap(&mut hidden.data, &mut next_hidden);
            hidden.name = format!("layer_{layer_idx}_prefill_layer_major_output");
            hidden.shape = TensorShape {
                dims: hidden_dims.clone(),
            };
            hidden.source_type = None;
            hidden.q8_0_blocks = None;
            hidden.q8_0_packed_rows4_4x4 = None;
            hidden.q8_0_packed_rows4_4x8 = None;
            hidden.q8_0_runtime_storage = None;
            hidden.q8_0_file_backing = None;
            trace_forward_memory(&format!("prefill_layer_major_layer_{layer_idx}_done"));
        }
        timings.layers_total = layers_started.elapsed().as_micros();
        self.kv_cache.position = prefill_base_position + token_ids.len();
        if let Some(memory) = &mut memory {
            memory.record_after_layers(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_layer_major_layers_done");
        timings.total = total_started.elapsed().as_micros();
        if let Some(memory) = &mut memory {
            memory.record_end(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("prefill_layer_major_end");
        timings.memory = memory;
        Ok(timings)
    }

    fn forward_single_token_timed_internal(
        &mut self,
        token_id: u32,
        collect_diagnostics: bool,
        compute_logits: bool,
    ) -> Result<LlamaFastForwardOutput> {
        if !self.kv_cache.can_append() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV cache is full at context length {}",
                self.kv_cache.plan.max_sequence_length
            )));
        }

        let runtime_plan = ResolvedRuntimePlan::from_env()?;
        let total_started = Instant::now();
        let mut memory = structured_forward_memory_enabled().then(|| {
            LlamaForwardMemoryTimings::new(
                capture_memory_sample(&self.kv_cache),
                collect_weight_materialization_stats(&self.weights),
                q8_0_file_read_stats(),
            )
        });
        trace_forward_memory("forward_start");
        let embedding_started = Instant::now();
        let mut hidden = self
            .weights
            .token_embedding
            .embedding_lookup(&[token_id], "token_embedding")?;
        if let Some(memory) = &mut memory {
            memory.record_after_embedding(capture_memory_sample(&self.kv_cache));
        }
        trace_forward_memory("embedding_done");
        let embedding_stats = collect_diagnostics
            .then(|| LlamaTensorStats::from_tensor(&hidden))
            .transpose()?;
        let mut timings = LlamaForwardTimings {
            embedding: embedding_started.elapsed().as_micros(),
            ..LlamaForwardTimings::default()
        };
        let mut layer_diagnostics =
            collect_diagnostics.then(|| Vec::with_capacity(self.weights.layers.len()));
        let layers_started = Instant::now();
        let rms_norm_epsilon = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            trace_forward_memory(&format!("layer_{layer_idx}_start"));
            let timed = forward_layer_timed(
                &hidden,
                layer,
                ForwardLayerParams {
                    config: &self.config,
                    rope_freqs: self.weights.rope_freqs.as_ref(),
                    rms_norm_epsilon,
                    layer_idx,
                    collect_diagnostics,
                    runtime_plan: &runtime_plan,
                },
                &mut self.kv_cache,
            )?;
            hidden = timed.output;
            trace_forward_memory(&format!("layer_{layer_idx}_done"));
            if let (Some(memory), Some(layer_memory)) = (&mut memory, &timed.timings.memory) {
                memory.record_layer(layer_memory.clone());
            }
            timings.layers.push(timed.timings);
            if let (Some(layer_diagnostics), Some(diagnostics)) =
                (&mut layer_diagnostics, timed.diagnostics)
            {
                layer_diagnostics.push(diagnostics);
            }
        }
        timings.layers_total = layers_started.elapsed().as_micros();
        if let Some(memory) = &mut memory {
            memory.record_after_layers(capture_memory_sample(&self.kv_cache));
        }
        let final_hidden_stats = collect_diagnostics
            .then(|| LlamaTensorStats::from_tensor(&hidden))
            .transpose()?;
        let (norm, logits, final_norm_diagnostic, output_norm_stats, logits_stats) =
            if compute_logits {
                let final_norm_started = Instant::now();
                let norm =
                    hidden.rms_norm(&self.weights.output_norm, rms_norm_epsilon, "output_norm")?;
                trace_forward_memory("output_norm_done");
                let final_norm_diagnostic = collect_diagnostics
                    .then(|| {
                        final_norm_diagnostics(
                            &hidden,
                            &self.weights.output_norm,
                            &norm,
                            rms_norm_epsilon,
                        )
                    })
                    .transpose()?;
                let output_norm_stats = collect_diagnostics
                    .then(|| LlamaTensorStats::from_tensor(&norm))
                    .transpose()?;
                timings.final_norm = final_norm_started.elapsed().as_micros();
                if let Some(memory) = &mut memory {
                    memory.record_after_final_norm(capture_memory_sample(&self.kv_cache));
                }
                let logits_started = Instant::now();
                let logits = output_projection_runtime_with_plan(
                    &norm,
                    self.weights.output_projection(),
                    "logits",
                    &runtime_plan,
                    collect_diagnostics,
                )?;
                trace_forward_memory("logits_done");
                let logits_stats = collect_diagnostics
                    .then(|| LlamaTensorStats::from_tensor(&logits))
                    .transpose()?;
                timings.logits = logits_started.elapsed().as_micros();
                if let Some(memory) = &mut memory {
                    memory.record_after_logits(capture_memory_sample(&self.kv_cache));
                }
                (
                    norm,
                    logits,
                    final_norm_diagnostic,
                    output_norm_stats,
                    logits_stats,
                )
            } else {
                trace_forward_memory("logits_skipped");
                (
                    CpuTensor::from_f32("output_norm_skipped", vec![1, 0], Vec::new())?,
                    CpuTensor::from_f32("logits_skipped", vec![1, 0], Vec::new())?,
                    None,
                    None,
                    None,
                )
            };
        self.kv_cache.position += 1;
        timings.total = total_started.elapsed().as_micros();
        if let Some(memory) = &mut memory {
            memory.record_end(capture_memory_sample(&self.kv_cache));
        }
        timings.memory = memory;
        let diagnostics = if collect_diagnostics {
            Some(LlamaForwardDiagnostics {
                embedding: embedding_stats.expect("embedding diagnostics collected"),
                final_hidden: final_hidden_stats.expect("final hidden diagnostics collected"),
                final_norm: final_norm_diagnostic.expect("final norm diagnostics collected"),
                output_norm: output_norm_stats.expect("output norm diagnostics collected"),
                logits: logits_stats.expect("logit diagnostics collected"),
                layers: layer_diagnostics.expect("layer diagnostics collected"),
            })
        } else {
            None
        };
        Ok(LlamaFastForwardOutput {
            output: LlamaForwardOutput {
                logits,
                hidden_state: hidden,
                output_norm_state: norm,
            },
            timings,
            diagnostics,
        })
    }

    pub fn generate_next_token(
        &mut self,
        token_ids: &[u32],
        sampler: LlamaSampler,
    ) -> Result<LlamaGenerationStep> {
        self.generate_next_token_with_history(token_ids, sampler, token_ids)
    }

    pub fn generate_next_token_with_history(
        &mut self,
        token_ids: &[u32],
        sampler: LlamaSampler,
        token_history: &[u32],
    ) -> Result<LlamaGenerationStep> {
        self.generate_next_token_with_history_diagnostics(token_ids, sampler, token_history, true)
    }

    pub fn generate_next_token_with_history_diagnostics(
        &mut self,
        token_ids: &[u32],
        sampler: LlamaSampler,
        token_history: &[u32],
        collect_diagnostics: bool,
    ) -> Result<LlamaGenerationStep> {
        if token_ids.is_empty() {
            return Err(BackendError::RuntimeShapeMismatch(
                "generation requires at least one prompt token".to_string(),
            ));
        }
        if token_ids.len() > self.kv_cache.plan.max_sequence_length - self.kv_cache.position {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "generation prompt of {} token(s) exceeds remaining context capacity {}",
                token_ids.len(),
                self.kv_cache.plan.max_sequence_length - self.kv_cache.position
            )));
        }
        if let LlamaSampler::Sampling(config) = &sampler {
            config.validate()?;
        }

        let mut diagnostics = None;
        let mut timings = LlamaForwardTimings::default();
        let mut prefill_timings = LlamaForwardTimings::default();
        let mut first_token_timings = LlamaForwardTimings::default();
        let prefill_count = token_ids.len().saturating_sub(1);
        let prefill_chunk_tokens = prefill_chunk_token_count(prefill_count);
        if prefill_count > 0
            && prefill_chunk_tokens > 1
            && prefill_layer_major_enabled(&self.weights)
        {
            let prefill_token_ids = &token_ids[..prefill_count];
            let prefill_chunk_tokens = prefill_layer_major_chunk_token_count(prefill_count);
            let layer_major_timings = self
                .forward_prefill_layer_major_timed_fast(prefill_token_ids, prefill_chunk_tokens)?;
            timings.add_assign(&layer_major_timings);
            prefill_timings.add_assign(&layer_major_timings);
        } else if prefill_count > 0 && prefill_chunk_tokens > 1 {
            for chunk in token_ids[..prefill_count].chunks(prefill_chunk_tokens) {
                let chunk_timings = self.forward_prefill_chunk_timed_fast(chunk)?;
                timings.add_assign(&chunk_timings);
                prefill_timings.add_assign(&chunk_timings);
            }
        } else {
            for token_id in &token_ids[..prefill_count] {
                add_q8_schedule_counter(&Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS, 1);
                let timed = self.forward_single_token_timed_internal(*token_id, false, false)?;
                timings.add_assign(&timed.timings);
                prefill_timings.add_assign(&timed.timings);
            }
        }

        let last_token_id = *token_ids.last().expect("non-empty token_ids checked above");
        let timed =
            self.forward_single_token_timed_internal(last_token_id, collect_diagnostics, true)?;
        timings.add_assign(&timed.timings);
        first_token_timings.add_assign(&timed.timings);
        if let Some(step_diagnostics) = timed.diagnostics {
            diagnostics = Some(step_diagnostics);
        }

        let output = timed.output;
        let logits = output.logits;
        let hidden_state = output.hidden_state;
        let output_norm_state = output.output_norm_state;
        let sample_started = Instant::now();
        let next_token_id = sampler.sample_with_history(&logits, token_history)?;
        let sample = sample_started.elapsed().as_micros();
        Ok(LlamaGenerationStep {
            prompt_token_count: token_ids.len(),
            prefill_token_count: token_ids.len().saturating_sub(1),
            next_token_id,
            logits,
            hidden_state,
            output_norm_state,
            timings,
            prefill_timings,
            first_token_timings,
            sample,
            diagnostics,
        })
    }
}

fn copy_tensor_rows_into_buffer(
    tensor: &CpuTensor,
    row_start: usize,
    rows: usize,
    buffer: &mut Vec<f32>,
) -> Result<()> {
    if tensor.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "tensor row buffer copy expected rank-2 tensor {}, got {:?}",
            tensor.name, tensor.shape.dims
        )));
    }
    let total_rows = tensor.dim(0)?;
    let width = tensor.dim(1)?;
    let row_end = row_start.checked_add(rows).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row buffer copy range overflows".to_string())
    })?;
    if rows == 0 || row_end > total_rows {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "tensor row buffer copy {row_start}..{row_end} is outside row count {total_rows}"
        )));
    }
    let data_start = row_start.checked_mul(width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row buffer copy offset overflows".to_string())
    })?;
    let data_len = rows.checked_mul(width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row buffer copy length overflows".to_string())
    })?;
    let data_end = data_start.checked_add(data_len).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row buffer copy end overflows".to_string())
    })?;
    buffer.clear();
    buffer.extend_from_slice(&tensor.data[data_start..data_end]);
    Ok(())
}

fn copy_tensor_rows_into(
    source: &CpuTensor,
    dest: &mut [f32],
    dest_row_start: usize,
    dest_width: usize,
) -> Result<()> {
    if source.rank() != 2 || source.dim(1)? != dest_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "tensor row copy expected source width {dest_width}, got {:?}",
            source.shape.dims
        )));
    }
    let rows = source.dim(0)?;
    let dest_start = dest_row_start.checked_mul(dest_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row copy offset overflows".to_string())
    })?;
    let dest_len = rows.checked_mul(dest_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row copy length overflows".to_string())
    })?;
    let dest_end = dest_start.checked_add(dest_len).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("tensor row copy end overflows".to_string())
    })?;
    if dest_end > dest.len() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "tensor row copy destination range {dest_start}..{dest_end} exceeds {} values",
            dest.len()
        )));
    }
    dest[dest_start..dest_end].copy_from_slice(&source.data);
    Ok(())
}

fn forward_memory_trace_enabled() -> bool {
    env_flag_enabled("CAMELID_FORWARD_MEMORY_TRACE")
}

fn structured_forward_memory_enabled() -> bool {
    env_flag_enabled("CAMELID_FORWARD_RSS_TIMINGS")
        || forward_memory_trace_enabled()
        || prefill_layer_major_attribution_enabled()
}

fn prefill_layer_major_attribution_enabled() -> bool {
    env_flag_enabled("CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION")
}

const Q8_FILE_CACHE_BYTES_ENV: &str = "CAMELID_Q8_0_FILE_CACHE_BYTES";
const PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES_ENV: &str =
    "CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES";
const DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES: usize = 256 * 1024 * 1024;

fn prefill_layer_major_default_q8_file_cache_capacity(weights: &LlamaLoadedWeights) -> usize {
    DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES.max(
        weights
            .largest_q8_0_file_backed_layer_storage_bytes()
            .try_into()
            .unwrap_or(usize::MAX),
    )
}

fn prefill_layer_major_q8_file_cache_capacity_override(
    weights: &LlamaLoadedWeights,
    forward_passes: usize,
) -> Option<usize> {
    if !weights.has_lazy_q8_0_file_backing() {
        return None;
    }
    let default_capacity = prefill_layer_major_default_q8_file_cache_capacity(weights);
    if env::var(PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES_ENV).is_ok() {
        return Some(
            parse_byte_count_env(PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES_ENV)
                .unwrap_or(default_capacity),
        );
    }
    if env::var(Q8_FILE_CACHE_BYTES_ENV).is_ok() {
        return None;
    }
    if forward_passes <= 1 {
        return None;
    }
    Some(default_capacity)
}

fn env_flag_enabled(key: &str) -> bool {
    matches!(
        env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
    )
}

fn prefill_chunk_token_count(prefill_count: usize) -> usize {
    const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 256;
    prefill_chunk_token_count_from_env(
        "CAMELID_PREFILL_CHUNK_TOKENS",
        prefill_count,
        DEFAULT_PREFILL_CHUNK_TOKENS,
    )
}

fn prefill_layer_major_chunk_token_count(prefill_count: usize) -> usize {
    const DEFAULT_PREFILL_LAYER_MAJOR_CHUNK_TOKENS: usize = 512;
    if env::var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS").is_ok() {
        return prefill_chunk_token_count_from_env(
            "CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS",
            prefill_count,
            DEFAULT_PREFILL_LAYER_MAJOR_CHUNK_TOKENS,
        );
    }
    if env::var("CAMELID_PREFILL_CHUNK_TOKENS").is_ok() {
        return prefill_chunk_token_count(prefill_count);
    }
    DEFAULT_PREFILL_LAYER_MAJOR_CHUNK_TOKENS
}

fn prefill_chunk_token_count_from_env(key: &str, prefill_count: usize, default: usize) -> usize {
    match env::var(key) {
        Ok(value) => parse_prefill_chunk_token_count(&value, prefill_count).unwrap_or(default),
        Err(_) => default,
    }
}

fn parse_prefill_chunk_token_count(value: &str, prefill_count: usize) -> Option<usize> {
    let trimmed = value.trim();
    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "all" | "full" | "prompt" | "unbounded"
    ) {
        return Some(prefill_count.max(1));
    }
    trimmed.parse::<usize>().ok().filter(|value| *value > 0)
}

fn prefill_layer_major_enabled(weights: &LlamaLoadedWeights) -> bool {
    match env::var("CAMELID_PREFILL_LAYER_MAJOR") {
        Ok(value) => {
            let trimmed = value.trim();
            !(trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("disabled"))
        }
        Err(env::VarError::NotPresent) => weights.has_lazy_q8_0_file_backing(),
        Err(_) => weights.has_lazy_q8_0_file_backing(),
    }
}

fn trace_forward_memory(phase: &str) {
    if !forward_memory_trace_enabled() {
        return;
    }

    let rss_kib = current_process_rss_kib()
        .map(|rss| rss.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let vm = macos_vm_stat_snapshot();
    let (free_like_pages, free_like_mib, throttled_pages) = vm
        .map(|snapshot| {
            let free_like_pages =
                snapshot.free_pages + snapshot.speculative_pages + snapshot.purgeable_pages;
            let free_like_mib = (free_like_pages as f64 * 16.0) / 1024.0;
            (
                free_like_pages.to_string(),
                format!("{free_like_mib:.1}"),
                snapshot.throttled_pages.to_string(),
            )
        })
        .unwrap_or_else(|| {
            (
                "unknown".to_string(),
                "unknown".to_string(),
                "unknown".to_string(),
            )
        });

    let q8_reads = q8_0_file_read_stats();
    let q8_file_read_mib = q8_reads.read_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_hit_mib = q8_reads.cache_hit_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_miss_mib = q8_reads.cache_miss_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_insert_mib = q8_reads.cache_insert_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_evicted_mib = q8_reads.cache_evicted_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_merged_mib = q8_reads.cache_merged_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_mib = q8_reads.cache_bytes as f64 / (1024.0 * 1024.0);
    let q8_file_cache_capacity_mib = q8_reads.cache_capacity_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "camelid_forward_memory_trace phase={phase} rss_kib={rss_kib} free_like_pages={free_like_pages} free_like_mib={free_like_mib} throttled_pages={throttled_pages} q8_file_read_calls={} q8_file_read_bytes={} q8_file_read_mib={q8_file_read_mib:.2} q8_file_cache_hits={} q8_file_cache_hit_bytes={} q8_file_cache_hit_mib={q8_file_cache_hit_mib:.2} q8_file_cache_misses={} q8_file_cache_miss_bytes={} q8_file_cache_miss_mib={q8_file_cache_miss_mib:.2} q8_file_cache_inserts={} q8_file_cache_insert_bytes={} q8_file_cache_insert_mib={q8_file_cache_insert_mib:.2} q8_file_cache_evictions={} q8_file_cache_evicted_bytes={} q8_file_cache_evicted_mib={q8_file_cache_evicted_mib:.2} q8_file_cache_merges={} q8_file_cache_merged_bytes={} q8_file_cache_merged_mib={q8_file_cache_merged_mib:.2} q8_file_cache_decoded_scale_hits={} q8_file_cache_decoded_scale_hit_blocks={} q8_file_cache_entries={} q8_file_cache_bytes={} q8_file_cache_mib={q8_file_cache_mib:.2} q8_file_cache_capacity_bytes={} q8_file_cache_capacity_mib={q8_file_cache_capacity_mib:.2}",
        q8_reads.read_calls,
        q8_reads.read_bytes,
        q8_reads.cache_hits,
        q8_reads.cache_hit_bytes,
        q8_reads.cache_misses,
        q8_reads.cache_miss_bytes,
        q8_reads.cache_inserts,
        q8_reads.cache_insert_bytes,
        q8_reads.cache_evictions,
        q8_reads.cache_evicted_bytes,
        q8_reads.cache_merges,
        q8_reads.cache_merged_bytes,
        q8_reads.cache_decoded_scale_hits,
        q8_reads.cache_decoded_scale_hit_blocks,
        q8_reads.cache_entries,
        q8_reads.cache_bytes,
        q8_reads.cache_capacity_bytes
    );
}

fn trace_forward_layer_memory(layer_idx: usize, phase: &str) {
    if forward_memory_trace_enabled() {
        trace_forward_memory(&format!("layer_{layer_idx}_{phase}"));
    }
}

fn trace_forward_prefill_layer_chunk_memory(
    layer_idx: usize,
    chunk_start: usize,
    chunk_rows: usize,
    base_position: usize,
    phase: &str,
) {
    if forward_memory_trace_enabled() {
        trace_forward_memory(&format!(
            "layer_{layer_idx}_prefill_chunk_start_{chunk_start}_rows_{chunk_rows}_base_{base_position}_{phase}"
        ));
    }
}

fn capture_memory_sample(kv_cache: &LlamaKvCache) -> LlamaMemorySample {
    LlamaMemorySample {
        rss_kib: current_process_rss_kib(),
        kv_cache_position: kv_cache.position,
        kv_cache_allocated_sequence_length: kv_cache.allocated_sequence_length,
        kv_cache_allocated_elements: kv_cache.allocated_elements(),
        kv_cache_allocated_bytes: kv_cache.allocated_bytes(),
    }
}

fn tensor_f32_bytes(tensor: &CpuTensor) -> u64 {
    vec_f32_bytes(&tensor.data)
}

fn vec_f32_bytes(data: &[f32]) -> u64 {
    (data.len() as u64) * (std::mem::size_of::<f32>() as u64)
}

fn collect_weight_materialization_stats(
    weights: &LlamaLoadedWeights,
) -> LlamaWeightMaterializationStats {
    let mut stats = LlamaWeightMaterializationStats::default();
    record_tensor_materialization(&mut stats, &weights.token_embedding);
    record_tensor_materialization(&mut stats, &weights.output_norm);
    if let Some(output) = &weights.output {
        record_tensor_materialization(&mut stats, output);
    }
    if let Some(rope_freqs) = &weights.rope_freqs {
        record_tensor_materialization(&mut stats, rope_freqs);
    }
    for layer in &weights.layers {
        record_tensor_materialization(&mut stats, &layer.attention_norm);
        record_tensor_materialization(&mut stats, &layer.attention_q);
        record_tensor_materialization(&mut stats, &layer.attention_k);
        record_tensor_materialization(&mut stats, &layer.attention_v);
        record_tensor_materialization(&mut stats, &layer.attention_output);
        record_tensor_materialization(&mut stats, &layer.ffn_norm);
        record_tensor_materialization(&mut stats, &layer.ffn_gate);
        record_tensor_materialization(&mut stats, &layer.ffn_up);
        record_tensor_materialization(&mut stats, &layer.ffn_down);
    }
    stats.has_q8_0_f32_materialization = stats.q8_0_f32_materialized_tensor_count > 0;
    stats.has_lazy_q8_0_file_backing = stats.q8_0_file_backed_tensor_count > 0;
    stats.has_retained_q8_0_blocks = stats.q8_0_retained_block_tensor_count > 0;
    stats
}

fn record_tensor_materialization(stats: &mut LlamaWeightMaterializationStats, tensor: &CpuTensor) {
    stats.tensor_count += 1;
    let f32_bytes = (tensor.data.len() as u64) * (std::mem::size_of::<f32>() as u64);
    if !tensor.data.is_empty() {
        stats.dense_f32_tensor_count += 1;
        stats.dense_f32_bytes = stats.dense_f32_bytes.saturating_add(f32_bytes);
    }
    if tensor.source_type == Some(GgufTensorType::Q8_0) {
        stats.q8_0_source_tensor_count += 1;
        if !tensor.data.is_empty() {
            stats.q8_0_f32_materialized_tensor_count += 1;
            stats.q8_0_f32_materialized_bytes =
                stats.q8_0_f32_materialized_bytes.saturating_add(f32_bytes);
        }
        if let Some(backing) = &tensor.q8_0_file_backing {
            stats.q8_0_file_backed_tensor_count += 1;
            stats.q8_0_file_backed_storage_bytes = stats
                .q8_0_file_backed_storage_bytes
                .saturating_add(backing.storage_bytes());
            stats.q8_0_file_backed_f32_bytes_avoided = stats
                .q8_0_file_backed_f32_bytes_avoided
                .saturating_add(backing.f32_materialization_bytes());
            stats.q8_0_file_backed_retained_block_bytes_if_enabled = stats
                .q8_0_file_backed_retained_block_bytes_if_enabled
                .saturating_add(backing.retained_block_bytes());
            if backing.file_handle_cached() {
                stats.q8_0_file_handle_cached_count += 1;
            }
        }
        if let Some(blocks) = &tensor.q8_0_blocks {
            stats.q8_0_retained_block_tensor_count += 1;
            stats.q8_0_retained_block_bytes = stats
                .q8_0_retained_block_bytes
                .saturating_add((blocks.len() as u64) * (std::mem::size_of::<Q8_0Block>() as u64));
        }
    }
}

fn current_process_rss_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    if let Some(rss) = linux_current_process_rss_kib() {
        return Some(rss);
    }

    current_process_rss_kib_via_ps()
}

#[cfg(target_os = "linux")]
fn linux_current_process_rss_kib() -> Option<u64> {
    parse_proc_status_rss_kib(&std::fs::read_to_string("/proc/self/status").ok()?)
}

#[cfg(target_os = "linux")]
fn parse_proc_status_rss_kib(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("VmRSS:"))?;
    let mut fields = line.split_whitespace();
    let _label = fields.next()?;
    fields.next()?.parse::<u64>().ok()
}

fn current_process_rss_kib_via_ps() -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

impl LlamaForwardMemoryTimings {
    fn new(
        start: LlamaMemorySample,
        materialization: LlamaWeightMaterializationStats,
        q8_file_read_start: Q8_0FileReadStats,
    ) -> Self {
        let mut memory = Self {
            forward_passes: 1,
            materialization,
            q8_file_reads: Q8_0FileReadStats::default(),
            q8_file_read_phases: Vec::new(),
            prefill_layer_major_attribution: Vec::new(),
            q8_file_read_start,
            q8_file_read_phase_start: q8_file_read_start,
            peak_rss_kib: None,
            peak_rss_delta_kib: None,
            peak_phase: None,
            start,
            after_embedding: None,
            after_layers: None,
            after_final_norm: None,
            after_logits: None,
            end: None,
            layers: Vec::new(),
        };
        memory.consider_peak_sample("forward_start", &memory.start.clone());
        memory
    }

    fn record_after_embedding(&mut self, sample: LlamaMemorySample) {
        self.record("embedding_done", sample, |this, sample| {
            this.after_embedding = Some(sample)
        });
    }

    fn record_after_layers(&mut self, sample: LlamaMemorySample) {
        self.record("layers_done", sample, |this, sample| {
            this.after_layers = Some(sample)
        });
    }

    fn record_after_final_norm(&mut self, sample: LlamaMemorySample) {
        self.record("final_norm_done", sample, |this, sample| {
            this.after_final_norm = Some(sample)
        });
    }

    fn record_after_logits(&mut self, sample: LlamaMemorySample) {
        self.record("logits_done", sample, |this, sample| {
            this.after_logits = Some(sample)
        });
    }

    fn record_end(&mut self, sample: LlamaMemorySample) {
        self.q8_file_reads = q8_0_file_read_stats().saturating_delta_since(self.q8_file_read_start);
        self.record("forward_end", sample, |this, sample| {
            this.end = Some(sample)
        });
    }

    fn record_layer(&mut self, layer: LlamaLayerMemoryTimings) {
        let layer_index = layer.layer_index;
        if let Some(rss) = layer.peak_rss_kib {
            let phase = layer.peak_phase.as_deref().unwrap_or("layer_peak");
            self.consider_peak_rss(&format!("layers.{layer_index}.{phase}"), rss);
        }
        if self.layers.len() <= layer_index {
            self.layers.resize_with(layer_index + 1, || layer.clone());
        }
        self.layers[layer_index] = layer;
    }

    fn record_prefill_layer_major_attribution(
        &mut self,
        trace: LlamaPrefillLayerMajorChunkAttribution,
    ) {
        self.prefill_layer_major_attribution.push(trace);
    }

    fn record(
        &mut self,
        phase: &str,
        sample: LlamaMemorySample,
        set: impl FnOnce(&mut Self, LlamaMemorySample),
    ) {
        self.consider_peak_sample(phase, &sample);
        self.record_q8_file_read_phase(phase);
        set(self, sample);
    }

    fn record_q8_file_read_phase(&mut self, phase: &str) {
        let current = q8_0_file_read_stats();
        let delta = current.saturating_delta_since(self.q8_file_read_phase_start);
        self.q8_file_read_phase_start = current;
        if q8_file_read_stats_has_activity(delta) {
            add_q8_file_read_phase_trace(&mut self.q8_file_read_phases, phase, delta);
        }
    }

    fn consider_peak_sample(&mut self, phase: &str, sample: &LlamaMemorySample) {
        if let Some(rss) = sample.rss_kib {
            self.consider_peak_rss(phase, rss);
        }
    }

    fn consider_peak_rss(&mut self, phase: &str, rss: u64) {
        if self.peak_rss_kib.is_none_or(|peak| rss > peak) {
            self.peak_rss_kib = Some(rss);
            self.peak_rss_delta_kib = self.start.rss_kib.map(|start| rss as i64 - start as i64);
            self.peak_phase = Some(phase.to_string());
        }
    }

    fn merge_assign(&mut self, other: &Self) {
        self.forward_passes += other.forward_passes;
        self.materialization = other.materialization.clone();
        add_q8_file_read_stats_delta(&mut self.q8_file_reads, other.q8_file_reads);
        for phase in &other.q8_file_read_phases {
            add_q8_file_read_phase_trace(
                &mut self.q8_file_read_phases,
                &phase.phase,
                phase.q8_file_reads,
            );
        }
        self.prefill_layer_major_attribution
            .extend(other.prefill_layer_major_attribution.iter().cloned());
        self.after_embedding = other.after_embedding.clone();
        self.after_layers = other.after_layers.clone();
        self.after_final_norm = other.after_final_norm.clone();
        self.after_logits = other.after_logits.clone();
        self.end = other.end.clone();
        if let (Some(phase), Some(rss)) = (&other.peak_phase, other.peak_rss_kib) {
            self.consider_peak_rss(phase, rss);
        }
        if self.layers.len() < other.layers.len() {
            self.layers
                .resize_with(other.layers.len(), || other.layers[0].clone());
        }
        for (idx, source) in other.layers.iter().enumerate() {
            if self.layers[idx].layer_index == source.layer_index {
                self.layers[idx].merge_assign(source);
            } else {
                self.layers[idx] = source.clone();
            }
        }
    }
}

fn q8_file_read_stats_has_activity(stats: Q8_0FileReadStats) -> bool {
    stats.read_calls > 0
        || stats.read_bytes > 0
        || stats.cache_hits > 0
        || stats.cache_hit_bytes > 0
        || stats.cache_misses > 0
        || stats.cache_miss_bytes > 0
        || stats.cache_inserts > 0
        || stats.cache_insert_bytes > 0
        || stats.cache_evictions > 0
        || stats.cache_evicted_bytes > 0
        || stats.cache_merges > 0
        || stats.cache_merged_bytes > 0
        || stats.cache_decoded_scale_hits > 0
        || stats.cache_decoded_scale_hit_blocks > 0
}

fn add_q8_file_read_stats_delta(target: &mut Q8_0FileReadStats, delta: Q8_0FileReadStats) {
    target.read_calls = target.read_calls.saturating_add(delta.read_calls);
    target.read_bytes = target.read_bytes.saturating_add(delta.read_bytes);
    target.cache_hits = target.cache_hits.saturating_add(delta.cache_hits);
    target.cache_hit_bytes = target.cache_hit_bytes.saturating_add(delta.cache_hit_bytes);
    target.cache_misses = target.cache_misses.saturating_add(delta.cache_misses);
    target.cache_miss_bytes = target
        .cache_miss_bytes
        .saturating_add(delta.cache_miss_bytes);
    target.cache_inserts = target.cache_inserts.saturating_add(delta.cache_inserts);
    target.cache_insert_bytes = target
        .cache_insert_bytes
        .saturating_add(delta.cache_insert_bytes);
    target.cache_evictions = target.cache_evictions.saturating_add(delta.cache_evictions);
    target.cache_evicted_bytes = target
        .cache_evicted_bytes
        .saturating_add(delta.cache_evicted_bytes);
    target.cache_merges = target.cache_merges.saturating_add(delta.cache_merges);
    target.cache_merged_bytes = target
        .cache_merged_bytes
        .saturating_add(delta.cache_merged_bytes);
    target.cache_decoded_scale_hits = target
        .cache_decoded_scale_hits
        .saturating_add(delta.cache_decoded_scale_hits);
    target.cache_decoded_scale_hit_blocks = target
        .cache_decoded_scale_hit_blocks
        .saturating_add(delta.cache_decoded_scale_hit_blocks);
    // These fields are point-in-time cache state, not additive counters.  A merged timing window
    // can span a scoped Q8 cache override (for example layer-major prefill) followed by a later
    // single-token pass after the override has been restored to zero.  Keep the peak observed
    // state so aggregate diagnostics still show that bounded cache/reuse was active.
    target.cache_entries = target.cache_entries.max(delta.cache_entries);
    target.cache_bytes = target.cache_bytes.max(delta.cache_bytes);
    target.cache_capacity_bytes = target.cache_capacity_bytes.max(delta.cache_capacity_bytes);
}

fn add_q8_file_read_phase_trace(
    phases: &mut Vec<LlamaQ8FileReadPhaseTrace>,
    phase: &str,
    delta: Q8_0FileReadStats,
) {
    if let Some(existing) = phases.iter_mut().find(|entry| entry.phase == phase) {
        add_q8_file_read_stats_delta(&mut existing.q8_file_reads, delta);
        return;
    }
    phases.push(LlamaQ8FileReadPhaseTrace {
        phase: phase.to_string(),
        q8_file_reads: delta,
    });
}

impl LlamaLayerMemoryTimings {
    fn new(layer_index: usize, start: LlamaMemorySample) -> Self {
        let mut memory = Self {
            layer_index,
            forward_passes: 1,
            q8_file_reads: Q8_0FileReadStats::default(),
            q8_file_read_start: q8_0_file_read_stats(),
            q8_file_read_phases: Vec::new(),
            q8_file_read_phase_start: q8_0_file_read_stats(),
            peak_rss_kib: None,
            peak_rss_delta_kib: None,
            peak_phase: None,
            start,
            after_attention_norm: None,
            after_attention_q: None,
            after_attention_k: None,
            after_attention_rope: None,
            after_attention_v: None,
            after_kv_cache_write: None,
            after_attention_context: None,
            after_attention_output: None,
            after_attention_residual: None,
            after_ffn_norm: None,
            after_ffn_activation: None,
            after_ffn_down: None,
            after_ffn_residual: None,
        };
        memory.consider_peak("layer_start", &memory.start.clone());
        memory
    }

    fn record_after_attention_norm(&mut self, sample: LlamaMemorySample) {
        self.record("attention_norm_done", sample, |this, sample| {
            this.after_attention_norm = Some(sample)
        });
    }

    fn record_after_attention_q(&mut self, sample: LlamaMemorySample) {
        self.record("attention_q_done", sample, |this, sample| {
            this.after_attention_q = Some(sample)
        });
    }

    fn record_after_attention_k(&mut self, sample: LlamaMemorySample) {
        self.record("attention_k_done", sample, |this, sample| {
            this.after_attention_k = Some(sample)
        });
    }

    fn record_after_attention_rope(&mut self, sample: LlamaMemorySample) {
        self.record("attention_rope_done", sample, |this, sample| {
            this.after_attention_rope = Some(sample)
        });
    }

    fn record_after_attention_v(&mut self, sample: LlamaMemorySample) {
        self.record("attention_v_done", sample, |this, sample| {
            this.after_attention_v = Some(sample)
        });
    }

    fn record_after_kv_cache_write(&mut self, sample: LlamaMemorySample) {
        self.record("kv_cache_write_done", sample, |this, sample| {
            this.after_kv_cache_write = Some(sample)
        });
    }

    fn record_after_attention_context(&mut self, sample: LlamaMemorySample) {
        self.record("attention_context_done", sample, |this, sample| {
            this.after_attention_context = Some(sample)
        });
    }

    fn record_after_attention_output(&mut self, sample: LlamaMemorySample) {
        self.record("attention_output_done", sample, |this, sample| {
            this.after_attention_output = Some(sample)
        });
    }

    fn record_after_attention_residual(&mut self, sample: LlamaMemorySample) {
        self.record("attention_residual_done", sample, |this, sample| {
            this.after_attention_residual = Some(sample)
        });
    }

    fn record_after_ffn_norm(&mut self, sample: LlamaMemorySample) {
        self.record("ffn_norm_done", sample, |this, sample| {
            this.after_ffn_norm = Some(sample)
        });
    }

    fn record_after_ffn_activation(&mut self, sample: LlamaMemorySample) {
        self.record("ffn_activation_done", sample, |this, sample| {
            this.after_ffn_activation = Some(sample)
        });
    }

    fn record_after_ffn_down(&mut self, sample: LlamaMemorySample) {
        self.record("ffn_down_done", sample, |this, sample| {
            this.after_ffn_down = Some(sample)
        });
    }

    fn record_after_ffn_residual(&mut self, sample: LlamaMemorySample) {
        self.record("ffn_residual_done", sample, |this, sample| {
            this.after_ffn_residual = Some(sample)
        });
    }

    fn record_end(&mut self) {
        self.q8_file_reads = q8_0_file_read_stats().saturating_delta_since(self.q8_file_read_start);
        self.record_q8_file_read_phase("layer_end");
    }

    fn record(
        &mut self,
        phase: &str,
        sample: LlamaMemorySample,
        set: impl FnOnce(&mut Self, LlamaMemorySample),
    ) {
        self.consider_peak(phase, &sample);
        self.record_q8_file_read_phase(phase);
        set(self, sample);
    }

    fn record_q8_file_read_phase(&mut self, phase: &str) {
        let current = q8_0_file_read_stats();
        let delta = current.saturating_delta_since(self.q8_file_read_phase_start);
        self.q8_file_read_phase_start = current;
        if q8_file_read_stats_has_activity(delta) {
            add_q8_file_read_phase_trace(&mut self.q8_file_read_phases, phase, delta);
        }
    }

    fn consider_peak(&mut self, phase: &str, sample: &LlamaMemorySample) {
        if let Some(rss) = sample.rss_kib {
            self.consider_peak_rss(phase, rss);
        }
    }

    fn consider_peak_rss(&mut self, phase: &str, rss: u64) {
        if self.peak_rss_kib.is_none_or(|peak| rss > peak) {
            self.peak_rss_kib = Some(rss);
            self.peak_rss_delta_kib = self.start.rss_kib.map(|start| rss as i64 - start as i64);
            self.peak_phase = Some(phase.to_string());
        }
    }

    fn merge_assign(&mut self, other: &Self) {
        self.forward_passes += other.forward_passes;
        add_q8_file_read_stats_delta(&mut self.q8_file_reads, other.q8_file_reads);
        for phase in &other.q8_file_read_phases {
            add_q8_file_read_phase_trace(
                &mut self.q8_file_read_phases,
                &phase.phase,
                phase.q8_file_reads,
            );
        }
        self.after_attention_norm = other.after_attention_norm.clone();
        self.after_attention_q = other.after_attention_q.clone();
        self.after_attention_k = other.after_attention_k.clone();
        self.after_attention_rope = other.after_attention_rope.clone();
        self.after_attention_v = other.after_attention_v.clone();
        self.after_kv_cache_write = other.after_kv_cache_write.clone();
        self.after_attention_context = other.after_attention_context.clone();
        self.after_attention_output = other.after_attention_output.clone();
        self.after_attention_residual = other.after_attention_residual.clone();
        self.after_ffn_norm = other.after_ffn_norm.clone();
        self.after_ffn_activation = other.after_ffn_activation.clone();
        self.after_ffn_down = other.after_ffn_down.clone();
        self.after_ffn_residual = other.after_ffn_residual.clone();
        if let (Some(phase), Some(rss)) = (&other.peak_phase, other.peak_rss_kib) {
            self.consider_peak_rss(phase, rss);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VmStatSnapshot {
    free_pages: u64,
    speculative_pages: u64,
    purgeable_pages: u64,
    throttled_pages: u64,
}

#[cfg(target_os = "macos")]
fn macos_vm_stat_snapshot() -> Option<VmStatSnapshot> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Some(VmStatSnapshot {
        free_pages: parse_vm_stat_pages(&text, "Pages free")?,
        speculative_pages: parse_vm_stat_pages(&text, "Pages speculative")?,
        purgeable_pages: parse_vm_stat_pages(&text, "Pages purgeable")?,
        throttled_pages: parse_vm_stat_pages(&text, "Pages throttled")?,
    })
}

#[cfg(not(target_os = "macos"))]
fn macos_vm_stat_snapshot() -> Option<VmStatSnapshot> {
    None
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with(key))?;
    let digits = line
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse::<u64>().ok()
}

impl LlamaForwardTimings {
    pub fn add_assign(&mut self, other: &Self) {
        self.total += other.total;
        self.embedding += other.embedding;
        self.layers_total += other.layers_total;
        self.final_norm += other.final_norm;
        self.logits += other.logits;
        if self.layers.len() < other.layers.len() {
            for idx in self.layers.len()..other.layers.len() {
                self.layers.push(LlamaLayerTimings {
                    layer_index: idx,
                    ..LlamaLayerTimings::default()
                });
            }
        }
        for (target, source) in self.layers.iter_mut().zip(&other.layers) {
            target.add_assign(source);
        }
        match (&mut self.memory, &other.memory) {
            (Some(target), Some(source)) => target.merge_assign(source),
            (None, Some(source)) => self.memory = Some(source.clone()),
            _ => {}
        }
    }
}

impl LlamaLayerTimings {
    fn add_assign(&mut self, other: &Self) {
        self.layer_index = other.layer_index;
        self.total += other.total;
        self.attention_norm += other.attention_norm;
        self.attention_q += other.attention_q;
        self.attention_k += other.attention_k;
        self.attention_v += other.attention_v;
        self.attention_rope += other.attention_rope;
        self.kv_cache_write += other.kv_cache_write;
        self.attention_context += other.attention_context;
        self.attention_output += other.attention_output;
        self.attention_residual += other.attention_residual;
        self.ffn_norm += other.ffn_norm;
        self.ffn_gate += other.ffn_gate;
        self.ffn_up += other.ffn_up;
        self.ffn_activation += other.ffn_activation;
        self.ffn_down += other.ffn_down;
        self.ffn_residual += other.ffn_residual;
        match (&mut self.memory, &other.memory) {
            (Some(target), Some(source)) => target.merge_assign(source),
            (None, Some(source)) => self.memory = Some(source.clone()),
            _ => {}
        }
    }
}

impl LlamaSampler {
    pub fn sample(&self, logits: &CpuTensor) -> Result<u32> {
        self.sample_with_history(logits, &[])
    }

    pub fn sample_with_history(&self, logits: &CpuTensor, token_history: &[u32]) -> Result<u32> {
        match self {
            Self::Greedy => greedy_sample(logits),
            Self::Sampling(config) => sample_with_config(logits, config, token_history),
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "temperature must be finite and non-negative, got {}",
                self.temperature
            )));
        }
        if matches!(self.top_k, Some(0)) {
            return Err(BackendError::RuntimeShapeMismatch(
                "top_k must be greater than zero when provided".to_string(),
            ));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "top_p must be finite and in (0, 1], got {top_p}"
                )));
            }
        }
        if !self.presence_penalty.is_finite() || !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "presence_penalty must be finite and in [-2, 2], got {}",
                self.presence_penalty
            )));
        }
        if !self.frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "frequency_penalty must be finite and in [-2, 2], got {}",
                self.frequency_penalty
            )));
        }
        for (token_id, bias) in &self.logit_bias {
            if !bias.is_finite() || !(-100.0..=100.0).contains(bias) {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "logit_bias for token {token_id} must be finite and in [-100, 100], got {bias}"
                )));
            }
        }
        Ok(())
    }
}

fn validate_logits(logits: &CpuTensor) -> Result<()> {
    if logits.shape.dims.len() != 2 || logits.shape.dims[0] != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "sampler logits expected shape [1, vocab], got {:?}",
            logits.shape.dims
        )));
    }
    if logits.data.is_empty() {
        return Err(BackendError::RuntimeShapeMismatch(
            "sampler logits must not be empty".to_string(),
        ));
    }
    for (idx, value) in logits.data.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "sampler logits contain non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn greedy_sample(logits: &CpuTensor) -> Result<u32> {
    validate_logits(logits)?;

    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in logits.data.iter().copied().enumerate() {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }

    token_index_to_u32(best_idx)
}

fn sample_with_config(
    logits: &CpuTensor,
    config: &SamplingConfig,
    token_history: &[u32],
) -> Result<u32> {
    config.validate()?;
    validate_logits(logits)?;
    let adjusted = apply_sampling_adjustments(logits, config, token_history)?;
    if config.temperature == 0.0 {
        return greedy_sample(&adjusted);
    }

    let mut candidates: Vec<(usize, f32)> = adjusted.data.iter().copied().enumerate().collect();
    candidates.sort_by(|(left_idx, left), (right_idx, right)| {
        right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
    });
    if let Some(top_k) = config.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }

    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit / config.temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weighted: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|(idx, logit)| (idx, ((logit / config.temperature) - max_logit).exp()))
        .collect();
    let weight_sum: f32 = weighted.iter().map(|(_, weight)| *weight).sum();
    if weight_sum == 0.0 || !weight_sum.is_finite() {
        return Err(BackendError::RuntimeShapeMismatch(
            "sampler softmax produced invalid normalization sum".to_string(),
        ));
    }
    for (_, weight) in &mut weighted {
        *weight /= weight_sum;
    }

    if let Some(top_p) = config.top_p.filter(|top_p| *top_p < 1.0) {
        weighted.sort_by(|(left_idx, left), (right_idx, right)| {
            right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
        });
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, probability) in &weighted {
            cumulative += *probability;
            keep += 1;
            if cumulative >= top_p {
                break;
            }
        }
        weighted.truncate(keep.max(1));
        let renorm: f32 = weighted.iter().map(|(_, weight)| *weight).sum();
        if renorm == 0.0 || !renorm.is_finite() {
            return Err(BackendError::RuntimeShapeMismatch(
                "top_p filtering removed all sampler candidates".to_string(),
            ));
        }
        for (_, weight) in &mut weighted {
            *weight /= renorm;
        }
    }

    let draw = seeded_unit_interval(config.seed.unwrap_or(0));
    let mut cumulative = 0.0;
    for (idx, probability) in &weighted {
        cumulative += *probability;
        if draw < cumulative {
            return token_index_to_u32(*idx);
        }
    }
    token_index_to_u32(
        weighted
            .last()
            .expect("weighted candidates are non-empty")
            .0,
    )
}

fn apply_sampling_adjustments<'a>(
    logits: &'a CpuTensor,
    config: &SamplingConfig,
    token_history: &[u32],
) -> Result<std::borrow::Cow<'a, CpuTensor>> {
    if config.presence_penalty == 0.0
        && config.frequency_penalty == 0.0
        && config.logit_bias.is_empty()
    {
        return Ok(std::borrow::Cow::Borrowed(logits));
    }

    let mut adjusted = logits.clone();
    for (token_id, bias) in &config.logit_bias {
        let Some(value) = adjusted.data.get_mut(*token_id) else {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "logit_bias token {token_id} is outside vocabulary size {}",
                adjusted.data.len()
            )));
        };
        *value += *bias;
    }

    if config.presence_penalty != 0.0 || config.frequency_penalty != 0.0 {
        let mut counts = std::collections::HashMap::<usize, usize>::new();
        for token_id in token_history {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit sampler index space"
                ))
            })?;
            if token_idx < adjusted.data.len() {
                *counts.entry(token_idx).or_default() += 1;
            }
        }
        for (token_idx, count) in counts {
            let value = adjusted
                .data
                .get_mut(token_idx)
                .expect("token index was checked against vocabulary size");
            if config.presence_penalty != 0.0 {
                *value -= config.presence_penalty;
            }
            if config.frequency_penalty != 0.0 {
                *value -= config.frequency_penalty * count as f32;
            }
        }
    }

    Ok(std::borrow::Cow::Owned(adjusted))
}

fn seeded_unit_interval(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mantissa = (z >> 40) as u32;
    mantissa as f32 / (1u32 << 24) as f32
}

fn token_index_to_u32(idx: usize) -> Result<u32> {
    u32::try_from(idx).map_err(|_| {
        BackendError::RuntimeShapeMismatch(format!("sampled token index {idx} does not fit u32"))
    })
}

struct ForwardLayerParams<'a> {
    config: &'a LlamaModelConfig,
    rope_freqs: Option<&'a CpuTensor>,
    rms_norm_epsilon: f32,
    layer_idx: usize,
    collect_diagnostics: bool,
    runtime_plan: &'a ResolvedRuntimePlan,
}

struct PrefillLayerChunkParams<'a> {
    config: &'a LlamaModelConfig,
    rope_freqs: Option<&'a CpuTensor>,
    rms_norm_epsilon: f32,
    layer_idx: usize,
    base_position: usize,
    chunk_start: usize,
    chunk_rows: usize,
}

fn forward_layer_timed(
    hidden: &CpuTensor,
    layer: &LlamaLayerWeights,
    params: ForwardLayerParams<'_>,
    kv_cache: &mut LlamaKvCache,
) -> Result<LlamaTimedLayerOutput> {
    let config = params.config;
    let rope_freqs = params.rope_freqs;
    let rms_norm_epsilon = params.rms_norm_epsilon;
    let layer_idx = params.layer_idx;
    let collect_diagnostics = params.collect_diagnostics;
    let runtime_plan = params.runtime_plan;
    let total_started = Instant::now();
    let mut timings = LlamaLayerTimings {
        layer_index: layer_idx,
        ..LlamaLayerTimings::default()
    };
    let mut memory = structured_forward_memory_enabled()
        .then(|| LlamaLayerMemoryTimings::new(layer_idx, capture_memory_sample(kv_cache)));

    let started = Instant::now();
    let attn_norm = hidden.rms_norm(
        &layer.attention_norm,
        rms_norm_epsilon,
        format!("layer_{layer_idx}_attention_norm"),
    )?;
    let attention_norm_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&attn_norm))
        .transpose()?;
    let attention_norm_diagnostic = collect_diagnostics
        .then(|| rms_norm_diagnostics(hidden, &layer.attention_norm, &attn_norm, rms_norm_epsilon))
        .transpose()?;
    timings.attention_norm = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_norm(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_norm_done");

    let qkv_started = Instant::now();
    let shared_qkv = if collect_diagnostics {
        None
    } else if let Some(qkv) = try_x86_q8_attention_qkv_packed_rows4_matmul_path(
        &attn_norm,
        &layer.attention_q,
        &layer.attention_k,
        &layer.attention_v,
        runtime_plan,
    )? {
        Some(qkv)
    } else if let Some(qkv) = try_x86_q8_attention_qkv_decode_consumer_path(
        &attn_norm,
        &layer.attention_q,
        &layer.attention_k,
        &layer.attention_v,
        runtime_plan,
    )? {
        Some(qkv)
    } else {
        try_attention_qkv_shared_q8_0_block_dot(
            &attn_norm,
            &layer.attention_q,
            &layer.attention_k,
            &layer.attention_v,
        )?
    };

    let (q, k, v, shared_qkv_elapsed) = if let Some((q, k, v)) = shared_qkv {
        let elapsed = qkv_started.elapsed().as_micros();
        (q, k, v, Some(elapsed))
    } else {
        let started = Instant::now();
        let q = linear_runtime_with_plan(
            &attn_norm,
            &layer.attention_q,
            format!("layer_{layer_idx}_attention_q"),
            runtime_plan,
            collect_diagnostics,
        )?;
        timings.attention_q = started.elapsed().as_micros();

        let started = Instant::now();
        let k = linear_for_role_runtime_with_plan(
            &attn_norm,
            &layer.attention_k,
            format!("layer_{layer_idx}_attention_k"),
            "attention_k",
            runtime_plan,
            collect_diagnostics,
        )?;
        timings.attention_k = started.elapsed().as_micros();

        let started = Instant::now();
        let v = linear_for_role_runtime_with_plan(
            &attn_norm,
            &layer.attention_v,
            format!("layer_{layer_idx}_attention_v"),
            "attention_v",
            runtime_plan,
            collect_diagnostics,
        )?;
        timings.attention_v = started.elapsed().as_micros();
        (q, k, v, None)
    };
    if let Some(total_elapsed) = shared_qkv_elapsed {
        let base = total_elapsed / 3;
        timings.attention_q = base;
        timings.attention_k = base;
        timings.attention_v = total_elapsed - (base * 2);
    }
    let attention_q_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&q))
        .transpose()?;
    let attention_q_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_q, &q, "linear"))
        .transpose()?;
    if let Some(memory) = &mut memory {
        memory.record_after_attention_q(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_q_done");

    let attention_k_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&k))
        .transpose()?;
    let attention_k_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_k, &k, "attention_k"))
        .transpose()?;
    if let Some(memory) = &mut memory {
        memory.record_after_attention_k(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_k_done");

    let started = Instant::now();
    let q_before_rope = q;
    let k_before_rope = k;
    let q = apply_rope(
        &q_before_rope,
        kv_cache.position,
        config.attention_head_count as usize,
        config,
        rope_freqs,
        format!("layer_{layer_idx}_attention_q_rope"),
    )?;
    let k = apply_rope(
        &k_before_rope,
        kv_cache.position,
        config.attention_head_count_kv as usize,
        config,
        rope_freqs,
        format!("layer_{layer_idx}_attention_k_rope"),
    )?;
    let attention_q_rope_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&q))
        .transpose()?;
    let attention_q_rope_diagnostic = collect_diagnostics
        .then(|| {
            rope_diagnostics(
                &q_before_rope,
                &q,
                kv_cache.position,
                config.attention_head_count as usize,
                config,
                rope_freqs,
                "attention_q",
            )
        })
        .transpose()?;
    let attention_k_rope_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&k))
        .transpose()?;
    let attention_k_rope_diagnostic = collect_diagnostics
        .then(|| {
            rope_diagnostics(
                &k_before_rope,
                &k,
                kv_cache.position,
                config.attention_head_count_kv as usize,
                config,
                rope_freqs,
                "attention_k",
            )
        })
        .transpose()?;
    timings.attention_rope = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_rope(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_rope_done");

    let attention_v_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&v))
        .transpose()?;
    let attention_v_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_v, &v, "attention_v"))
        .transpose()?;
    if let Some(memory) = &mut memory {
        memory.record_after_attention_v(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_v_done");

    let started = Instant::now();
    write_kv_cache(kv_cache, layer_idx, &k, &v)?;
    let kv_cache_diagnostic = collect_diagnostics
        .then(|| kv_cache_trace(kv_cache, layer_idx, kv_cache.position + 1))
        .transpose()?;
    timings.kv_cache_write = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_kv_cache_write(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "kv_cache_write_done");

    let started = Instant::now();
    let attention_context = causal_attention_context(
        kv_cache,
        layer_idx,
        &q,
        config.attention_head_count as usize,
        config.attention_head_count_kv as usize,
        format!("layer_{layer_idx}_attention_context"),
        collect_diagnostics,
    )?;
    let attention_trace = attention_context.trace;
    let context = attention_context.tensor;
    let attention_context_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&context))
        .transpose()?;
    timings.attention_context = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_context(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_context_done");

    let started = Instant::now();
    let mut attn_out = linear_runtime_with_plan(
        &context,
        &layer.attention_output,
        format!("layer_{layer_idx}_attention_output"),
        runtime_plan,
        collect_diagnostics,
    )?;
    if collect_diagnostics && diagnostic_zero_delta(DeltaZeroTarget::Attention, layer_idx)? {
        attn_out = zero_like(
            &attn_out,
            format!("layer_{layer_idx}_attention_output_zeroed"),
        )?;
    }
    let attention_output_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&attn_out))
        .transpose()?;
    let attention_output_diagnostic = collect_diagnostics
        .then(|| {
            linear_projection_diagnostics(&context, &layer.attention_output, &attn_out, "linear")
        })
        .transpose()?;
    timings.attention_output = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_output(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_output_done");

    let started = Instant::now();
    let residual = hidden.add(&attn_out, format!("layer_{layer_idx}_attention_residual"))?;
    let attention_residual_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&residual))
        .transpose()?;
    let attention_input_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(hidden))
        .transpose()?;
    let attention_delta_diagnostic = collect_diagnostics
        .then(|| residual_reconstruction_diagnostic(hidden, &attn_out, &residual))
        .transpose()?;
    timings.attention_residual = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_residual(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_residual_done");

    let started = Instant::now();
    let ffn_norm = residual.rms_norm(
        &layer.ffn_norm,
        rms_norm_epsilon,
        format!("layer_{layer_idx}_ffn_norm"),
    )?;
    let ffn_norm_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&ffn_norm))
        .transpose()?;
    let ffn_norm_diagnostic = collect_diagnostics
        .then(|| rms_norm_diagnostics(&residual, &layer.ffn_norm, &ffn_norm, rms_norm_epsilon))
        .transpose()?;
    timings.ffn_norm = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_norm(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "ffn_norm_done");

    let (
        mut ffn_out,
        ffn_gate_stats,
        ffn_up_stats,
        ffn_gate_diagnostic,
        ffn_up_diagnostic,
        ffn_activation_diagnostic,
        ffn_activation_stats,
        ffn_down_diagnostic,
        ffn_output_stats,
        ffn_out_already_residual,
    ) = if let (Some(moe), Some(router)) = (&params.config.moe, &layer.moe_router) {
        if collect_diagnostics {
            return Err(BackendError::UnsupportedModelArchitecture(
                    "Mixtral MoE diagnostics are not implemented yet; generation remains runtime-only until parity evidence is collected".to_string(),
                ));
        }
        let (ffn_out, gate, up, activation, down) = mixtral_moe_ffn(
            &ffn_norm,
            router,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
            moe.expert_used_count as usize,
            format!("layer_{layer_idx}_mixtral_moe_ffn"),
        )?;
        timings.ffn_gate = gate;
        timings.ffn_up = up;
        timings.ffn_activation = activation;
        timings.ffn_down = down;
        (
            ffn_out, None, None, None, None, None, None, None, None, false,
        )
    } else {
        let activated = gated_ffn_activation_with_plan(
            &ffn_norm,
            &layer.ffn_gate,
            &layer.ffn_up,
            format!("layer_{layer_idx}_ffn_activated"),
            runtime_plan,
            collect_diagnostics,
        )?;
        timings.ffn_gate = activated.gate;
        timings.ffn_up = activated.up;
        timings.ffn_activation = activated.activation;
        let ffn_gate_stats = activated.gate_stats;
        let ffn_up_stats = activated.up_stats;
        let ffn_gate_diagnostic = activated.gate_diagnostic;
        let ffn_up_diagnostic = activated.up_diagnostic;
        let ffn_activation_diagnostic = activated.activation_diagnostic;
        let activated = activated.tensor;
        let ffn_activation_stats = collect_diagnostics
            .then(|| LlamaTensorStats::from_tensor(&activated))
            .transpose()?;
        if let Some(memory) = &mut memory {
            memory.record_after_ffn_activation(capture_memory_sample(kv_cache));
        }
        trace_forward_layer_memory(layer_idx, "ffn_gate_up_activation_done");
        let started = Instant::now();
        let ffn_out = linear_for_role_runtime_with_plan(
            &activated,
            &layer.ffn_down,
            format!("layer_{layer_idx}_ffn_down"),
            "ffn_down",
            runtime_plan,
            collect_diagnostics,
        )?;
        let ffn_out_already_residual = false;
        let ffn_output_stats = collect_diagnostics
            .then(|| LlamaTensorStats::from_tensor(&ffn_out))
            .transpose()?;
        let ffn_down_diagnostic = collect_diagnostics
            .then(|| {
                linear_projection_diagnostics(&activated, &layer.ffn_down, &ffn_out, "ffn_down")
            })
            .transpose()?;
        timings.ffn_down = started.elapsed().as_micros();
        (
            ffn_out,
            ffn_gate_stats,
            ffn_up_stats,
            ffn_gate_diagnostic,
            ffn_up_diagnostic,
            ffn_activation_diagnostic,
            ffn_activation_stats,
            ffn_down_diagnostic,
            ffn_output_stats,
            ffn_out_already_residual,
        )
    };
    if collect_diagnostics && diagnostic_zero_delta(DeltaZeroTarget::Ffn, layer_idx)? {
        ffn_out = zero_like(&ffn_out, format!("layer_{layer_idx}_ffn_down_zeroed"))?;
    }
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_down(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "ffn_down_done");

    let started = Instant::now();
    let output = if ffn_out_already_residual {
        ffn_out.clone()
    } else {
        residual.add(&ffn_out, format!("layer_{layer_idx}_ffn_residual"))?
    };
    let ffn_residual_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&output))
        .transpose()?;
    let ffn_input_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&residual))
        .transpose()?;
    let ffn_delta_diagnostic = collect_diagnostics
        .then(|| residual_reconstruction_diagnostic(&residual, &ffn_out, &output))
        .transpose()?;
    timings.ffn_residual = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_residual(capture_memory_sample(kv_cache));
        memory.record_end();
    }
    trace_forward_layer_memory(layer_idx, "ffn_residual_done");
    timings.total = total_started.elapsed().as_micros();
    timings.memory = memory;
    let diagnostics = if collect_diagnostics {
        Some(LlamaLayerDiagnostics {
            layer_index: layer_idx,
            residual_flow: LlamaResidualFlowDiagnostic {
                attention_input: attention_input_stats
                    .expect("attention input diagnostics collected"),
                attention_delta: attention_delta_diagnostic
                    .expect("attention residual diagnostics collected"),
                ffn_input: ffn_input_stats.expect("ffn input diagnostics collected"),
                ffn_delta: ffn_delta_diagnostic.expect("ffn residual diagnostics collected"),
            },
            attention_norm: attention_norm_stats.expect("attention norm diagnostics collected"),
            attention_norm_reconstruction: attention_norm_diagnostic
                .expect("attention norm reconstruction diagnostics collected"),
            attention_q: attention_q_stats.expect("attention q diagnostics collected"),
            attention_q_reconstruction: attention_q_diagnostic
                .expect("attention q reconstruction diagnostics collected"),
            attention_k: attention_k_stats.expect("attention k diagnostics collected"),
            attention_k_reconstruction: attention_k_diagnostic
                .expect("attention k reconstruction diagnostics collected"),
            attention_q_rope: attention_q_rope_stats
                .expect("attention q rope diagnostics collected"),
            attention_q_rope_reconstruction: attention_q_rope_diagnostic
                .expect("attention q rope reconstruction diagnostics collected"),
            attention_k_rope: attention_k_rope_stats
                .expect("attention k rope diagnostics collected"),
            attention_k_rope_reconstruction: attention_k_rope_diagnostic
                .expect("attention k rope reconstruction diagnostics collected"),
            attention_v: attention_v_stats.expect("attention v diagnostics collected"),
            attention_v_reconstruction: attention_v_diagnostic
                .expect("attention v reconstruction diagnostics collected"),
            kv_cache_trace: kv_cache_diagnostic.expect("KV cache diagnostics collected"),
            attention_trace: attention_trace.expect("attention trace diagnostics collected"),
            attention_context: attention_context_stats
                .expect("attention context diagnostics collected"),
            attention_output: attention_output_stats
                .expect("attention output diagnostics collected"),
            attention_output_reconstruction: attention_output_diagnostic
                .expect("attention output reconstruction diagnostics collected"),
            attention_residual: attention_residual_stats
                .expect("attention residual diagnostics collected"),
            ffn_norm: ffn_norm_stats.expect("ffn norm diagnostics collected"),
            ffn_norm_reconstruction: ffn_norm_diagnostic
                .expect("ffn norm reconstruction diagnostics collected"),
            ffn_gate: ffn_gate_stats.expect("ffn gate diagnostics collected"),
            ffn_gate_reconstruction: ffn_gate_diagnostic
                .expect("ffn gate reconstruction diagnostics collected"),
            ffn_up: ffn_up_stats.expect("ffn up diagnostics collected"),
            ffn_up_reconstruction: ffn_up_diagnostic
                .expect("ffn up reconstruction diagnostics collected"),
            ffn_activation: ffn_activation_stats.expect("ffn activation diagnostics collected"),
            ffn_activation_reconstruction: ffn_activation_diagnostic
                .expect("ffn activation reconstruction diagnostics collected"),
            ffn_output: ffn_output_stats.expect("ffn output diagnostics collected"),
            ffn_down_reconstruction: ffn_down_diagnostic
                .expect("ffn down reconstruction diagnostics collected"),
            ffn_residual: ffn_residual_stats.expect("ffn residual diagnostics collected"),
        })
    } else {
        None
    };

    Ok(LlamaTimedLayerOutput {
        output,
        timings,
        diagnostics,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct LlamaTimedLayerOutput {
    output: CpuTensor,
    timings: LlamaLayerTimings,
    diagnostics: Option<LlamaLayerDiagnostics>,
}

fn forward_prefill_layer_chunk_timed(
    hidden: &CpuTensor,
    layer: &LlamaLayerWeights,
    params: PrefillLayerChunkParams<'_>,
    kv_cache: &mut LlamaKvCache,
) -> Result<LlamaTimedLayerOutput> {
    let config = params.config;
    let layer_idx = params.layer_idx;
    if params.base_position != kv_cache.position {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "prefill chunk base position {} does not match KV cache cursor {}",
            params.base_position, kv_cache.position
        )));
    }
    let total_started = Instant::now();
    let mut timings = LlamaLayerTimings {
        layer_index: layer_idx,
        ..LlamaLayerTimings::default()
    };
    let mut memory = structured_forward_memory_enabled()
        .then(|| LlamaLayerMemoryTimings::new(layer_idx, capture_memory_sample(kv_cache)));
    let trace_chunk_memory = |phase: &str| {
        trace_forward_prefill_layer_chunk_memory(
            layer_idx,
            params.chunk_start,
            params.chunk_rows,
            params.base_position,
            phase,
        );
    };
    trace_chunk_memory("start");

    let started = Instant::now();
    let attn_norm = hidden.rms_norm(
        &layer.attention_norm,
        params.rms_norm_epsilon,
        format!("layer_{layer_idx}_prefill_attention_norm"),
    )?;
    timings.attention_norm = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_norm(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_norm_done");

    let qkv_started = Instant::now();
    let shared_qkv = if x86_q8_attention_qkv_prefill_consumer_enabled() {
        let runtime_plan = ResolvedRuntimePlan::from_env()?;
        try_x86_q8_attention_qkv_packed_rows4_matmul_path(
            &attn_norm,
            &layer.attention_q,
            &layer.attention_k,
            &layer.attention_v,
            &runtime_plan,
        )?
    } else {
        None
    };

    let (q, k, v, shared_qkv_elapsed) = if let Some((q, k, v)) = shared_qkv {
        let elapsed = qkv_started.elapsed().as_micros();
        (q, k, v, Some(elapsed))
    } else {
        let started = Instant::now();
        let q = linear_runtime(
            &attn_norm,
            &layer.attention_q,
            format!("layer_{layer_idx}_prefill_attention_q"),
            false,
        )?;
        timings.attention_q = started.elapsed().as_micros();

        let started = Instant::now();
        let k = linear_for_role_runtime(
            &attn_norm,
            &layer.attention_k,
            format!("layer_{layer_idx}_prefill_attention_k"),
            "attention_k",
            false,
        )?;
        timings.attention_k = started.elapsed().as_micros();

        let started = Instant::now();
        let v = linear_for_role_runtime(
            &attn_norm,
            &layer.attention_v,
            format!("layer_{layer_idx}_prefill_attention_v"),
            "attention_v",
            false,
        )?;
        timings.attention_v = started.elapsed().as_micros();
        (q, k, v, None)
    };
    if let Some(total_elapsed) = shared_qkv_elapsed {
        let base = total_elapsed / 3;
        timings.attention_q = base;
        timings.attention_k = base;
        timings.attention_v = total_elapsed - (base * 2);
    }
    if let Some(memory) = &mut memory {
        memory.record_after_attention_q(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_q_done");
    if let Some(memory) = &mut memory {
        memory.record_after_attention_k(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_k_done");

    let started = Instant::now();
    let q = apply_rope_batch(
        &q,
        params.base_position,
        config.attention_head_count as usize,
        config,
        params.rope_freqs,
        format!("layer_{layer_idx}_prefill_attention_q_rope"),
    )?;
    let k = apply_rope_batch(
        &k,
        params.base_position,
        config.attention_head_count_kv as usize,
        config,
        params.rope_freqs,
        format!("layer_{layer_idx}_prefill_attention_k_rope"),
    )?;
    timings.attention_rope = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_rope(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_rope_done");

    if let Some(memory) = &mut memory {
        memory.record_after_attention_v(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_v_done");

    let started = Instant::now();
    write_kv_cache_batch(kv_cache, layer_idx, params.base_position, &k, &v)?;
    timings.kv_cache_write = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_kv_cache_write(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("kv_cache_write_done");

    let started = Instant::now();
    let context = causal_attention_context_batch(
        kv_cache,
        layer_idx,
        params.base_position,
        &q,
        config.attention_head_count as usize,
        config.attention_head_count_kv as usize,
        format!("layer_{layer_idx}_prefill_attention_context"),
    )?;
    timings.attention_context = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_context(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_context_done");

    let started = Instant::now();
    let attn_out = linear_runtime(
        &context,
        &layer.attention_output,
        format!("layer_{layer_idx}_prefill_attention_output"),
        false,
    )?;
    timings.attention_output = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_output(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_output_done");

    let started = Instant::now();
    let mut residual = attn_out;
    add_tensor_in_place(
        &mut residual,
        hidden,
        format!("layer_{layer_idx}_prefill_attention_residual"),
    )?;
    timings.attention_residual = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_residual(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_residual_done");

    let started = Instant::now();
    let ffn_norm = residual.rms_norm(
        &layer.ffn_norm,
        params.rms_norm_epsilon,
        format!("layer_{layer_idx}_prefill_ffn_norm"),
    )?;
    timings.ffn_norm = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_norm(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("ffn_norm_done");

    let ffn_out = if let (Some(moe), Some(router)) = (&params.config.moe, &layer.moe_router) {
        let (ffn_out, gate, up, activation, down) = mixtral_moe_ffn(
            &ffn_norm,
            router,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
            moe.expert_used_count as usize,
            format!("layer_{layer_idx}_prefill_mixtral_moe_ffn"),
        )?;
        timings.ffn_gate = gate;
        timings.ffn_up = up;
        timings.ffn_activation = activation;
        timings.ffn_down = down;
        ffn_out
    } else {
        let activated = gated_ffn_activation_batch(
            &ffn_norm,
            &layer.ffn_gate,
            &layer.ffn_up,
            format!("layer_{layer_idx}_prefill_ffn_activated"),
        )?;
        timings.ffn_gate = activated.gate;
        timings.ffn_up = activated.up;
        timings.ffn_activation = activated.activation;
        let activated = activated.tensor;
        if let Some(memory) = &mut memory {
            memory.record_after_ffn_activation(capture_memory_sample(kv_cache));
        }
        trace_chunk_memory("ffn_gate_up_activation_done");
        let started = Instant::now();
        let ffn_out = linear_for_role_runtime(
            &activated,
            &layer.ffn_down,
            format!("layer_{layer_idx}_prefill_ffn_down"),
            "ffn_down",
            false,
        )?;
        timings.ffn_down = started.elapsed().as_micros();
        ffn_out
    };
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_down(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("ffn_down_done");

    let started = Instant::now();
    let mut output = ffn_out;
    add_tensor_in_place(
        &mut output,
        &residual,
        format!("layer_{layer_idx}_prefill_ffn_residual"),
    )?;
    timings.ffn_residual = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_ffn_residual(capture_memory_sample(kv_cache));
        memory.record_end();
    }
    trace_chunk_memory("ffn_residual_done");
    trace_chunk_memory("end");
    timings.total = total_started.elapsed().as_micros();
    timings.memory = memory;

    Ok(LlamaTimedLayerOutput {
        output,
        timings,
        diagnostics: None,
    })
}

const TENSOR_CHECKPOINT_SAMPLE: usize = 10;

fn residual_reconstruction_diagnostic(
    input: &CpuTensor,
    delta: &CpuTensor,
    reported: &CpuTensor,
) -> Result<LlamaResidualReconstructionDiagnostic> {
    if input.shape.dims != delta.shape.dims || input.shape.dims != reported.shape.dims {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "residual reconstruction shape mismatch: input {:?}, delta {:?}, reported {:?}",
            input.shape.dims, delta.shape.dims, reported.shape.dims
        )));
    }

    let mut max_abs_delta_index = 0;
    let mut max_abs_delta = 0.0_f32;
    let mut input_sum_square = 0.0_f32;
    let mut delta_sum_square = 0.0_f32;
    let mut reported_sum_square = 0.0_f32;
    let mut input_delta_dot = 0.0_f32;
    let mut reported_max_abs_index = 0;
    let mut reported_max_abs = 0.0_f32;
    let mut reconstructed_values = Vec::with_capacity(input.data.len());
    for (idx, ((input_value, delta_value), reported_value)) in input
        .data
        .iter()
        .zip(delta.data.iter())
        .zip(reported.data.iter())
        .enumerate()
    {
        input_sum_square += input_value * input_value;
        delta_sum_square += delta_value * delta_value;
        reported_sum_square += reported_value * reported_value;
        input_delta_dot += input_value * delta_value;
        let reconstructed = input_value + delta_value;
        reconstructed_values.push(reconstructed);
        let abs_delta = (reconstructed - reported_value).abs();
        if abs_delta > max_abs_delta {
            max_abs_delta = abs_delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported_value.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
    }

    let len = input.data.len() as f32;
    let input_rms = (input_sum_square / len).sqrt();
    let delta_rms = (delta_sum_square / len).sqrt();
    let reported_rms = (reported_sum_square / len).sqrt();
    let delta_to_input_rms_ratio = if input_rms > 0.0 {
        delta_rms / input_rms
    } else {
        0.0
    };
    let cosine_denominator = input_sum_square.sqrt() * delta_sum_square.sqrt();
    let delta_input_cosine_similarity = if cosine_denominator > 0.0 {
        input_delta_dot / cosine_denominator
    } else {
        0.0
    };

    let sample_len = input.data.len().min(8);
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &reported.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed_values,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, delta_reported_max_abs_window) = tensor_window_around_index(
        &delta.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    Ok(LlamaResidualReconstructionDiagnostic {
        input_rms,
        delta_rms,
        reported_rms,
        delta_to_input_rms_ratio,
        delta_input_cosine_similarity,
        input_first_values: input.data.iter().take(sample_len).copied().collect(),
        delta_first_values: delta.data.iter().take(sample_len).copied().collect(),
        reconstructed_first_values: reconstructed_values
            .iter()
            .take(sample_len)
            .copied()
            .collect(),
        reported_first_values: reported.data.iter().take(sample_len).copied().collect(),
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        delta_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

fn rms_norm_diagnostics(
    input: &CpuTensor,
    weight: &CpuTensor,
    reported: &CpuTensor,
    epsilon: f32,
) -> Result<LlamaRmsNormDiagnostic> {
    if input.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "rms_norm diagnostics expected input rank 2, got {:?}",
            input.shape.dims
        )));
    }
    if weight.rank() != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "rms_norm diagnostics expected weight rank 1, got {:?}",
            weight.shape.dims
        )));
    }
    require_tensor_shape(reported, &input.shape.dims, "rms_norm output")?;
    let rows = input.dim(0)?;
    let cols = input.dim(1)?;
    if rows != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "rms_norm diagnostics expected one row, got {rows}"
        )));
    }
    if weight.dim(0)? != cols {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "rms_norm weight shape {:?} does not match input shape {:?}",
            weight.shape.dims, input.shape.dims
        )));
    }

    let input_mean_square = input.data.iter().map(|value| value * value).sum::<f32>() / cols as f32;
    let input_rms = input_mean_square.sqrt();
    let scale = 1.0 / (input_mean_square + epsilon).sqrt();
    let mut max_abs_delta = 0.0_f32;
    let mut max_abs_delta_index = 0;
    let mut reported_max_abs = 0.0_f32;
    let mut reported_max_abs_index = 0;
    let mut reconstructed_values = Vec::with_capacity(cols);
    let mut reconstructed_first_values = Vec::new();
    let mut reported_first_values = Vec::new();
    let mut input_first_values = Vec::new();
    let mut weight_first_values = Vec::new();

    for idx in 0..cols {
        let reconstructed = input.data[idx] * scale * weight.data[idx];
        reconstructed_values.push(reconstructed);
        let reported_value = reported.data[idx];
        let delta = (reconstructed - reported_value).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported_value.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
        if idx < TENSOR_CHECKPOINT_SAMPLE {
            input_first_values.push(input.data[idx]);
            weight_first_values.push(weight.data[idx]);
            reconstructed_first_values.push(reconstructed);
            reported_first_values.push(reported_value);
        }
    }
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &reported.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed_values,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );

    Ok(LlamaRmsNormDiagnostic {
        epsilon,
        input_mean_square,
        input_rms,
        scale,
        input_first_values,
        weight_first_values,
        reconstructed_first_values,
        reported_first_values,
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

fn final_norm_diagnostics(
    hidden: &CpuTensor,
    weight: &CpuTensor,
    output_norm: &CpuTensor,
    epsilon: f32,
) -> Result<LlamaFinalNormDiagnostic> {
    if hidden.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "final rms_norm diagnostics expected hidden rank 2, got {:?}",
            hidden.shape.dims
        )));
    }
    if weight.rank() != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "final rms_norm diagnostics expected weight rank 1, got {:?}",
            weight.shape.dims
        )));
    }
    require_tensor_shape(output_norm, &hidden.shape.dims, "final rms_norm output")?;
    let rows = hidden.dim(0)?;
    let cols = hidden.dim(1)?;
    if rows != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "final rms_norm diagnostics expected one row, got {rows}"
        )));
    }
    if weight.dim(0)? != cols {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "final rms_norm weight shape {:?} does not match hidden shape {:?}",
            weight.shape.dims, hidden.shape.dims
        )));
    }

    let hidden_mean_square =
        hidden.data.iter().map(|value| value * value).sum::<f32>() / cols as f32;
    let hidden_rms = hidden_mean_square.sqrt();
    let scale = 1.0 / (hidden_mean_square + epsilon).sqrt();
    let mut max_abs_delta = 0.0_f32;
    let mut max_abs_delta_index = 0;
    let mut reported_max_abs = 0.0_f32;
    let mut reported_max_abs_index = 0;
    let mut reconstructed_values = Vec::with_capacity(cols);
    let mut reconstructed_first_values = Vec::new();
    let mut reported_first_values = Vec::new();
    let mut hidden_first_values = Vec::new();
    let mut weight_first_values = Vec::new();

    for idx in 0..cols {
        let reconstructed = hidden.data[idx] * scale * weight.data[idx];
        reconstructed_values.push(reconstructed);
        let reported = output_norm.data[idx];
        let delta = (reconstructed - reported).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
        if idx < TENSOR_CHECKPOINT_SAMPLE {
            hidden_first_values.push(hidden.data[idx]);
            weight_first_values.push(weight.data[idx]);
            reconstructed_first_values.push(reconstructed);
            reported_first_values.push(reported);
        }
    }
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &output_norm.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed_values,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );

    Ok(LlamaFinalNormDiagnostic {
        epsilon,
        hidden_mean_square,
        hidden_rms,
        scale,
        hidden_first_values,
        weight_first_values,
        reconstructed_first_values,
        reported_first_values,
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

fn tensor_stats(tensor: &CpuTensor) -> Result<LlamaTensorStats> {
    if tensor.data.is_empty() {
        return Err(BackendError::RuntimeShapeMismatch(
            "cannot compute tensor stats for an empty tensor".to_string(),
        ));
    }

    let data = &tensor.data;
    let mut min = f32::INFINITY;
    let mut min_index = 0;
    let mut max = f32::NEG_INFINITY;
    let mut max_index = 0;
    let mut max_abs = 0.0_f32;
    let mut max_abs_index = 0;
    let mut sum = 0.0f64;
    let mut sum_square = 0.0f64;
    for (idx, value) in data.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "cannot compute tensor stats: non-finite value at index {idx}"
            )));
        }
        if value < min {
            min = value;
            min_index = idx;
        }
        if value > max {
            max = value;
            max_index = idx;
        }
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_index = idx;
        }
        let value = value as f64;
        sum += value;
        sum_square += value * value;
    }
    let len = data.len() as f64;
    let max_abs_window_start = max_abs_index.saturating_sub(TENSOR_CHECKPOINT_SAMPLE / 2);
    let max_abs_window_end = data
        .len()
        .min(max_abs_window_start + TENSOR_CHECKPOINT_SAMPLE);
    Ok(LlamaTensorStats {
        min,
        min_index,
        max,
        max_index,
        mean: (sum / len) as f32,
        rms: (sum_square / len).sqrt() as f32,
        max_abs_index,
        max_abs,
        checkpoint: LlamaTensorCheckpoint {
            shape: tensor.shape.dims.clone(),
            len: data.len(),
            first_values: data
                .iter()
                .take(TENSOR_CHECKPOINT_SAMPLE)
                .copied()
                .collect(),
            max_abs_window_start,
            max_abs_window: data[max_abs_window_start..max_abs_window_end].to_vec(),
        },
    })
}

fn normalize_token_embedding_shape(mut tensor: CpuTensor, name: &str) -> Result<CpuTensor> {
    if tensor.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "token embedding tensor {name} expected rank 2, got {:?}",
            tensor.shape.dims
        )));
    }

    // GGUF/GGML records LLaMA token embeddings as [embedding_width, vocab_size],
    // while the runtime embedding lookup expects row-major [vocab_size, embedding_width].
    // The underlying bytes are already token-major for lookup, so this is a shape
    // reinterpretation, not a numerical transpose.
    if tensor.shape.dims[0] < tensor.shape.dims[1] {
        tensor.shape.dims.swap(0, 1);
    }
    Ok(tensor)
}

fn require_tensor_shape(tensor: &CpuTensor, expected: &[usize], role: &str) -> Result<()> {
    if tensor.shape.dims != expected {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{role} tensor {} expected shape {:?}, got {:?}",
            tensor.name, expected, tensor.shape.dims
        )));
    }
    Ok(())
}

fn require_matrix_shape(
    tensor: &CpuTensor,
    input_width: usize,
    output_width: usize,
    role: &str,
) -> Result<()> {
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if tensor.shape.dims.as_slice() != direct && tensor.shape.dims.as_slice() != transposed {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{role} tensor {} expected shape {:?} or {:?}, got {:?}",
            tensor.name, direct, transposed, tensor.shape.dims
        )));
    }
    Ok(())
}

#[cfg(test)]
fn linear(input: &CpuTensor, weight: &CpuTensor, name: impl Into<String>) -> Result<CpuTensor> {
    linear_for_role(input, weight, name, "linear")
}

fn linear_runtime(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    linear_runtime_with_plan(input, weight, name, &runtime_plan, collect_diagnostics)
}

fn linear_runtime_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    linear_for_role_runtime_with_plan(
        input,
        weight,
        name,
        "linear",
        runtime_plan,
        collect_diagnostics,
    )
}

fn linear_for_role(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    rectangular_role: &str,
) -> Result<CpuTensor> {
    linear_with_diagnostic_layouts(
        input,
        weight,
        name,
        diagnostic_square_linear_layout()?,
        diagnostic_rectangular_linear_layout_for_role(rectangular_role)?,
    )
}

fn linear_for_role_runtime(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    rectangular_role: &str,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    linear_for_role_runtime_with_plan(
        input,
        weight,
        name,
        rectangular_role,
        &runtime_plan,
        collect_diagnostics,
    )
}

fn linear_for_role_runtime_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    if collect_diagnostics {
        linear_for_role(input, weight, name, rectangular_role)
    } else {
        let name = name.into();
        if let Some(output) = try_x86_q8_attention_output_decode_consumer_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_attention_output_packed_rows4_matmul_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_attention_projection_decode_consumer_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_ffn_down_gemm4_prefill_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_ffn_down_single_owner_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_ffn_down_decode_consumer_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        if let Some(output) = try_x86_q8_ffn_down_packed_rows4_matmul_path(
            input,
            weight,
            &name,
            rectangular_role,
            runtime_plan,
        )? {
            return Ok(output);
        }
        linear_with_diagnostic_layouts_with_plan(
            input,
            weight,
            name,
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Auto,
            runtime_plan,
        )
    }
}

fn linear_projection_diagnostics(
    input: &CpuTensor,
    weight: &CpuTensor,
    reported: &CpuTensor,
    rectangular_role: &str,
) -> Result<LlamaLinearProjectionDiagnostic> {
    if input.rank() != 2 || input.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "linear projection diagnostics expected one input row, got {:?}",
            input.shape.dims
        )));
    }
    if reported.rank() != 2 || reported.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "linear projection diagnostics expected one reported row, got {:?}",
            reported.shape.dims
        )));
    }
    let layout = effective_linear_layout(
        input.dim(1)?,
        weight,
        diagnostic_square_linear_layout()?,
        diagnostic_rectangular_linear_layout_for_role(rectangular_role)?,
    )?;
    let reconstructed = match layout.kind {
        EffectiveLinearLayoutKind::Descriptor => {
            let descriptor_weight =
                linear_weight_reinterpreted_as_descriptor(weight, input.dim(1)?)?;
            matmul_descriptor_with_precision(
                input,
                &descriptor_weight,
                "linear_projection_diagnostic",
            )?
        }
        EffectiveLinearLayoutKind::Transposed => {
            let transposed_weight =
                linear_weight_reinterpreted_as_transposed(weight, input.dim(1)?)?;
            matmul_rhs_transposed_with_precision(
                input,
                &transposed_weight,
                "linear_projection_diagnostic",
            )?
        }
    };
    require_tensor_shape(
        &reconstructed,
        &reported.shape.dims,
        "linear projection diagnostic reconstruction",
    )?;

    let mut max_abs_delta = 0.0_f32;
    let mut max_abs_delta_index = 0;
    let mut reported_max_abs = 0.0_f32;
    let mut reported_max_abs_index = 0;
    for (idx, (reconstructed_value, reported_value)) in reconstructed
        .data
        .iter()
        .zip(reported.data.iter())
        .enumerate()
    {
        let delta = (reconstructed_value - reported_value).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported_value.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
    }
    let sample_len = reported.data.len().min(TENSOR_CHECKPOINT_SAMPLE);
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &reported.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );

    Ok(LlamaLinearProjectionDiagnostic {
        role: rectangular_role.to_string(),
        layout: layout.label,
        input_width: input.dim(1)?,
        output_width: reported.dim(1)?,
        weight_shape: weight.shape.dims.clone(),
        input_first_values: input
            .data
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        weight_first_values: weight
            .data
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        reconstructed_first_values: reconstructed
            .data
            .iter()
            .take(sample_len)
            .copied()
            .collect(),
        reported_first_values: reported.data.iter().take(sample_len).copied().collect(),
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

fn tensor_window_around_index(data: &[f32], index: usize, sample_len: usize) -> (usize, Vec<f32>) {
    let start = index.saturating_sub(sample_len / 2);
    let end = data.len().min(start + sample_len);
    (start, data[start..end].to_vec())
}

fn max_abs_index(data: &[f32]) -> usize {
    data.iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EffectiveLinearLayout {
    kind: EffectiveLinearLayoutKind,
    label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveLinearLayoutKind {
    Descriptor,
    Transposed,
}

fn effective_linear_layout(
    input_width: usize,
    weight: &CpuTensor,
    square_layout: SquareLinearLayout,
    rectangular_layout: RectangularLinearLayout,
) -> Result<EffectiveLinearLayout> {
    if weight.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "linear weight {} expected rank 2, got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let rows = weight.dim(0)?;
    let cols = weight.dim(1)?;
    if rows == input_width && cols == input_width {
        return Ok(match square_layout {
            SquareLinearLayout::Descriptor => EffectiveLinearLayout {
                kind: EffectiveLinearLayoutKind::Descriptor,
                label: "descriptor".to_string(),
            },
            SquareLinearLayout::Transposed => EffectiveLinearLayout {
                kind: EffectiveLinearLayoutKind::Transposed,
                label: "transposed".to_string(),
            },
        });
    }
    if rectangular_layout == RectangularLinearLayout::Descriptor {
        return Ok(EffectiveLinearLayout {
            kind: EffectiveLinearLayoutKind::Descriptor,
            label: "descriptor".to_string(),
        });
    }
    if rectangular_layout == RectangularLinearLayout::Transposed {
        return Ok(EffectiveLinearLayout {
            kind: EffectiveLinearLayoutKind::Transposed,
            label: "transposed".to_string(),
        });
    }
    if rows == input_width {
        return Ok(EffectiveLinearLayout {
            kind: EffectiveLinearLayoutKind::Transposed,
            label: "transposed_auto".to_string(),
        });
    }
    if cols == input_width {
        return Ok(EffectiveLinearLayout {
            kind: EffectiveLinearLayoutKind::Transposed,
            label: "transposed_auto".to_string(),
        });
    }
    Err(BackendError::RuntimeShapeMismatch(format!(
        "linear input width {input_width} is incompatible with weight {} shape {:?}",
        weight.name, weight.shape.dims
    )))
}

fn linear_with_diagnostic_layouts(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    square_layout: SquareLinearLayout,
    rectangular_layout: RectangularLinearLayout,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    linear_with_diagnostic_layouts_with_plan(
        input,
        weight,
        name,
        square_layout,
        rectangular_layout,
        &runtime_plan,
    )
}

fn linear_with_diagnostic_layouts_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    square_layout: SquareLinearLayout,
    rectangular_layout: RectangularLinearLayout,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    let input_width = input.dim(1)?;
    if weight.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "linear weight {} expected rank 2, got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let rows = weight.dim(0)?;
    let cols = weight.dim(1)?;
    if rows == input_width && cols == input_width {
        match square_layout {
            SquareLinearLayout::Descriptor => {
                matmul_descriptor_with_precision_with_plan(input, weight, name, runtime_plan)
            }
            SquareLinearLayout::Transposed => {
                matmul_rhs_transposed_with_precision_with_plan(input, weight, name, runtime_plan)
            }
        }
    } else if rectangular_layout == RectangularLinearLayout::Descriptor {
        let descriptor_weight = linear_weight_reinterpreted_as_descriptor(weight, input_width)?;
        matmul_descriptor_with_precision_with_plan(input, &descriptor_weight, name, runtime_plan)
    } else if rectangular_layout == RectangularLinearLayout::Transposed || rows == input_width {
        let transposed_weight = linear_weight_reinterpreted_as_transposed(weight, input_width)?;
        matmul_rhs_transposed_with_precision_with_plan(
            input,
            &transposed_weight,
            name,
            runtime_plan,
        )
    } else if cols == input_width {
        matmul_rhs_transposed_with_precision_with_plan(input, weight, name, runtime_plan)
    } else {
        Err(BackendError::RuntimeShapeMismatch(format!(
            "linear input shape {:?} is incompatible with weight {} shape {:?}",
            input.shape.dims, weight.name, weight.shape.dims
        )))
    }
}

fn linear_weight_reinterpreted_as_descriptor(
    weight: &CpuTensor,
    input_width: usize,
) -> Result<CpuTensor> {
    if weight.dim(0)? == input_width {
        Ok(weight.clone())
    } else if weight.dim(1)? == input_width {
        Ok(weight_with_swapped_matrix_shape(weight))
    } else {
        Err(BackendError::RuntimeShapeMismatch(format!(
            "linear descriptor diagnostic cannot reinterpret weight {} shape {:?} for input width {input_width}",
            weight.name, weight.shape.dims
        )))
    }
}

fn linear_weight_reinterpreted_as_transposed(
    weight: &CpuTensor,
    input_width: usize,
) -> Result<CpuTensor> {
    if weight.dim(1)? == input_width {
        Ok(weight.clone())
    } else if weight.dim(0)? == input_width {
        Ok(weight_with_swapped_matrix_shape(weight))
    } else {
        Err(BackendError::RuntimeShapeMismatch(format!(
            "linear transposed diagnostic cannot reinterpret weight {} shape {:?} for input width {input_width}",
            weight.name, weight.shape.dims
        )))
    }
}

fn weight_with_swapped_matrix_shape(weight: &CpuTensor) -> CpuTensor {
    let mut reinterpreted = weight.clone();
    reinterpreted.shape.dims.swap(0, 1);
    reinterpreted.q8_0_packed_rows4_4x4 = None;
    reinterpreted.q8_0_packed_rows4_4x8 = None;
    if !q8_0_runtime_storage_matches_matrix_shape(
        reinterpreted.q8_0_runtime_storage.as_ref(),
        reinterpreted.shape.dims[0],
        reinterpreted.shape.dims[1],
    ) {
        reinterpreted.q8_0_runtime_storage = None;
    }
    reinterpreted
}

fn q8_0_runtime_storage_matches_matrix_shape(
    storage: Option<&Q8_0RuntimeStorage>,
    rows: usize,
    cols: usize,
) -> bool {
    let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = storage else {
        return false;
    };
    cols.is_multiple_of(Q8_0_BLOCK_VALUES)
        && packed.rows == rows
        && packed.blocks_per_row == cols / Q8_0_BLOCK_VALUES
}

#[derive(Clone, Copy)]
struct BorrowedLinearWeight<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f32],
    source_type: Option<GgufTensorType>,
    q8_0_blocks: Option<&'a [Q8_0Block]>,
    q8_0_packed_rows4_4x4: Option<&'a Q8_0PackedRows4>,
    q8_0_packed_rows4_4x8: Option<&'a Q8_0PackedRows4>,
    q8_0_runtime_storage: Option<&'a Q8_0RuntimeStorage>,
    q8_0_file_backing: Option<&'a Q8_0FileBacking>,
}

impl<'a> BorrowedLinearWeight<'a> {
    fn from_tensor(weight: &'a CpuTensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "linear weight {} expected rank 2, got {:?}",
                weight.name, weight.shape.dims
            )));
        }
        Ok(Self {
            rows: weight.dim(0)?,
            cols: weight.dim(1)?,
            data: &weight.data,
            source_type: weight.source_type,
            q8_0_blocks: weight.q8_0_blocks.as_deref(),
            q8_0_packed_rows4_4x4: weight.q8_0_packed_rows4_4x4.as_ref(),
            q8_0_packed_rows4_4x8: weight.q8_0_packed_rows4_4x8.as_ref(),
            q8_0_runtime_storage: weight.q8_0_runtime_storage.as_ref(),
            q8_0_file_backing: weight.q8_0_file_backing.as_ref(),
        })
    }

    fn with_swapped_matrix_shape(self) -> Self {
        let rows = self.cols;
        let cols = self.rows;
        let q8_0_runtime_storage = self.q8_0_runtime_storage.filter(|storage| {
            q8_0_runtime_storage_matches_matrix_shape(Some(*storage), rows, cols)
        });
        Self {
            rows,
            cols,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage,
            ..self
        }
    }
}

fn borrowed_linear_weight_as_descriptor<'a>(
    weight: &'a CpuTensor,
    input_width: usize,
) -> Result<BorrowedLinearWeight<'a>> {
    let view = BorrowedLinearWeight::from_tensor(weight)?;
    if view.rows == input_width {
        Ok(view)
    } else if view.cols == input_width {
        Ok(view.with_swapped_matrix_shape())
    } else {
        Err(BackendError::RuntimeShapeMismatch(format!(
            "linear descriptor diagnostic cannot reinterpret weight {} shape {:?} for input width {input_width}",
            weight.name, weight.shape.dims
        )))
    }
}

fn borrowed_linear_weight_as_transposed<'a>(
    weight: &'a CpuTensor,
    input_width: usize,
) -> Result<BorrowedLinearWeight<'a>> {
    let view = BorrowedLinearWeight::from_tensor(weight)?;
    if view.cols == input_width {
        Ok(view)
    } else if view.rows == input_width {
        Ok(view.with_swapped_matrix_shape())
    } else {
        Err(BackendError::RuntimeShapeMismatch(format!(
            "linear transposed diagnostic cannot reinterpret weight {} shape {:?} for input width {input_width}",
            weight.name, weight.shape.dims
        )))
    }
}

fn zero_like(tensor: &CpuTensor, name: impl Into<String>) -> Result<CpuTensor> {
    CpuTensor::from_f32(
        name,
        tensor.shape.dims.clone(),
        vec![0.0; tensor.data.len()],
    )
}

fn add_tensor_in_place(
    tensor: &mut CpuTensor,
    rhs: &CpuTensor,
    name: impl Into<String>,
) -> Result<()> {
    if tensor.shape != rhs.shape {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "shape mismatch: lhs {:?}, rhs {:?}",
            tensor.shape.dims, rhs.shape.dims
        )));
    }
    for (left, right) in tensor.data.iter_mut().zip(&rhs.data) {
        *left += right;
    }
    tensor.name = name.into();
    tensor.source_type = None;
    tensor.q8_0_blocks = None;
    tensor.q8_0_packed_rows4_4x4 = None;
    tensor.q8_0_packed_rows4_4x8 = None;
    tensor.q8_0_runtime_storage = None;
    tensor.q8_0_file_backing = None;
    Ok(())
}

#[allow(dead_code)]
fn output_projection_runtime(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    output_projection_runtime_with_plan(input, weight, name, &runtime_plan, collect_diagnostics)
}

fn output_projection_runtime_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    let layout = if collect_diagnostics {
        diagnostic_output_projection_layout()?
    } else {
        OutputProjectionLayout::TokenMajor
    };
    output_projection_with_layout_with_plan(input, weight, name, layout, runtime_plan)
}

#[allow(dead_code)]
fn output_projection_with_layout(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    layout: OutputProjectionLayout,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    output_projection_with_layout_with_plan(input, weight, name, layout, &runtime_plan)
}

fn output_projection_with_layout_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    layout: OutputProjectionLayout,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    match layout {
        OutputProjectionLayout::Descriptor => {
            let input_width = input.dim(1)?;
            if weight.rank() != 2 {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "output projection weight {} expected rank 2, got {:?}",
                    weight.name, weight.shape.dims
                )));
            }
            if weight.dim(0)? == input_width {
                matmul_descriptor_with_precision_with_plan(input, weight, name, runtime_plan)
            } else if weight.dim(1)? == input_width {
                matmul_rhs_transposed_with_precision_with_plan(input, weight, name, runtime_plan)
            } else {
                Err(BackendError::RuntimeShapeMismatch(format!(
                    "output projection input shape {:?} is incompatible with weight {} shape {:?}",
                    input.shape.dims, weight.name, weight.shape.dims
                )))
            }
        }
        OutputProjectionLayout::TokenMajor => {
            let input_width = input.dim(1)?;
            if weight.rank() != 2 {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token-major output projection requires rank-2 weight {}, got {:?}",
                    weight.name, weight.shape.dims
                )));
            }
            if weight.dim(0)? != input_width && weight.dim(1)? != input_width {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token-major output projection input shape {:?} is incompatible with weight {} shape {:?}",
                    input.shape.dims, weight.name, weight.shape.dims
                )));
            }
            let name = name.into();
            if let Some(output) =
                try_x86_q8_output_packed_rows4_matmul_path(input, weight, &name, runtime_plan)?
            {
                return Ok(output);
            }
            if let Some(output) =
                try_x86_q8_output_decode_owner_path(input, weight, &name, runtime_plan)?
            {
                return Ok(output);
            }
            let token_major = borrowed_linear_weight_as_transposed(weight, input_width)?;
            let output_width = token_major.rows;
            let route =
                q8_schedule_output_projection_route_kind(token_major, input_width, output_width);
            let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
            let output = matmul_rhs_transposed_borrowed_with_precision_with_plan(
                input,
                token_major,
                name.as_str(),
                runtime_plan,
            )?;
            if let Some(telemetry_started) = telemetry_started {
                record_q8_schedule_output_projection_route_call(
                    "logits",
                    route,
                    Some(&name),
                    input.dim(0)?,
                    input_width,
                    output_width,
                    telemetry_started.elapsed().as_micros(),
                );
            }
            Ok(output)
        }
    }
}

fn matmul_descriptor_with_precision(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    matmul_descriptor_with_precision_with_plan(input, weight, name, &runtime_plan)
}

fn matmul_descriptor_with_precision_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    match runtime_plan.linear_accumulation_precision {
        LinearAccumulationPrecision::F32 => input.matmul(weight, name),
        LinearAccumulationPrecision::F64 => matmul_descriptor_f64(input, weight, name),
    }
}

fn matmul_rhs_transposed_with_precision(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    matmul_rhs_transposed_with_precision_with_plan(input, weight, name, &runtime_plan)
}

fn matmul_rhs_transposed_with_precision_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    let input_width = input.dim(1)?;
    if should_use_q8_0_block_dot_with_plan(weight, input_width, runtime_plan) {
        return matmul_rhs_transposed_q8_0_block_dot_with_plan(input, weight, name, runtime_plan);
    }
    if let Some(backing) = q8_0_reader_backing(weight, input_width)? {
        return matmul_rhs_transposed_q8_0_block_reader_with_flags(
            input,
            backing,
            Q8BlockReader::new(backing.absolute_offset, backing.num_blocks),
            weight.dim(0)?,
            name,
            &runtime_plan.q8,
        );
    }
    if let Some((packed, interleave)) = q8_0_selected_packed_rows4(weight) {
        return matmul_rhs_transposed_q8_0_packed_rows4_f32_input(
            input,
            packed,
            interleave,
            weight.dim(0)?,
            name,
        );
    }
    match runtime_plan.linear_accumulation_precision {
        LinearAccumulationPrecision::F32 => input.matmul_rhs_transposed(weight, name),
        LinearAccumulationPrecision::F64 => matmul_rhs_transposed_f64(input, weight, name),
    }
}

#[allow(dead_code)]
fn matmul_rhs_transposed_borrowed_with_precision(
    input: &CpuTensor,
    weight: BorrowedLinearWeight<'_>,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    matmul_rhs_transposed_borrowed_with_precision_with_plan(input, weight, name, &runtime_plan)
}

fn matmul_rhs_transposed_borrowed_with_precision_with_plan(
    input: &CpuTensor,
    weight: BorrowedLinearWeight<'_>,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if weight.cols != input_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "borrowed transposed matmul shape mismatch: lhs {:?}, rhs [{}, {}]",
            input.shape.dims, weight.rows, weight.cols
        )));
    }
    let output_width = weight.rows;
    if let Some(backing) = borrowed_q8_0_reader_backing(weight, input_width, output_width)? {
        return matmul_rhs_transposed_q8_0_block_reader_with_flags(
            input,
            backing,
            Q8BlockReader::new(backing.absolute_offset, backing.num_blocks),
            output_width,
            name,
            &runtime_plan.q8,
        );
    }
    let mut output = vec![0.0; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        let output_start = row * output_width;
        accumulate_transposed_linear_row_runtime_with_plan(
            &input.data[input_start..input_start + input_width],
            weight,
            &mut output[output_start..output_start + output_width],
            runtime_plan,
        )?;
    }
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn matmul_descriptor_f64(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if weight.rank() != 2 || weight.dim(0)? != input_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "f64 descriptor matmul shape mismatch: lhs {:?}, rhs {:?}",
            input.shape.dims, weight.shape.dims
        )));
    }
    let output_width = weight.dim(1)?;
    let mut output = vec![0.0; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        for output_idx in 0..output_width {
            let mut sum = 0.0_f64;
            for inner in 0..input_width {
                sum += f64::from(input.data[input_start + inner])
                    * f64::from(weight.data[inner * output_width + output_idx]);
            }
            output[row * output_width + output_idx] = sum as f32;
        }
    }
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn matmul_rhs_transposed_f64(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if weight.rank() != 2 || weight.dim(1)? != input_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "f64 transposed matmul shape mismatch: lhs {:?}, rhs {:?}",
            input.shape.dims, weight.shape.dims
        )));
    }
    let output_width = weight.dim(0)?;
    let mut output = vec![0.0; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        for output_idx in 0..output_width {
            let weight_start = output_idx * input_width;
            let mut sum = 0.0_f64;
            for inner in 0..input_width {
                sum += f64::from(input.data[input_start + inner])
                    * f64::from(weight.data[weight_start + inner]);
            }
            output[row * output_width + output_idx] = sum as f32;
        }
    }
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

#[derive(Debug, Clone, Copy)]
struct OutputProjectionFinalNormSources<'a> {
    final_hidden: &'a CpuTensor,
    output_norm_weight: &'a CpuTensor,
    output_norm_scale: f32,
}

pub fn output_projection_diagnostics(
    output_norm: &CpuTensor,
    output_weight: &CpuTensor,
    logits: &CpuTensor,
    token_ids: &[u32],
    final_hidden: Option<&CpuTensor>,
    output_norm_weight: Option<&CpuTensor>,
    output_norm_scale: Option<f32>,
) -> Result<Vec<LlamaOutputProjectionDiagnostic>> {
    if output_norm.rank() != 2 || output_norm.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics expected output norm shape [1, hidden], got {:?}",
            output_norm.shape.dims
        )));
    }
    if logits.rank() != 2 || logits.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics expected logits shape [1, vocab], got {:?}",
            logits.shape.dims
        )));
    }
    let hidden_width = output_norm.dim(1)?;
    let final_norm_sources = optional_final_norm_sources(
        final_hidden,
        output_norm_weight,
        output_norm_scale,
        hidden_width,
    )?;
    let vocab_size = logits.dim(1)?;
    let layout = validate_output_projection_row_layout(
        output_weight,
        hidden_width,
        vocab_size,
        diagnostic_output_projection_layout()?,
    )?;

    let mut diagnostics = Vec::with_capacity(token_ids.len());
    for &token_id in token_ids {
        let token_index = token_id as usize;
        if token_index >= vocab_size {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "output projection diagnostic token id {token_id} is outside vocabulary size {vocab_size}"
            )));
        }
        diagnostics.push(output_projection_token_diagnostic(
            output_norm,
            output_weight,
            logits.data[token_index],
            token_id,
            hidden_width,
            layout,
            final_norm_sources,
        )?);
    }
    Ok(diagnostics)
}

fn optional_final_norm_sources<'a>(
    final_hidden: Option<&'a CpuTensor>,
    output_norm_weight: Option<&'a CpuTensor>,
    output_norm_scale: Option<f32>,
    hidden_width: usize,
) -> Result<Option<OutputProjectionFinalNormSources<'a>>> {
    let provided = [
        final_hidden.is_some(),
        output_norm_weight.is_some(),
        output_norm_scale.is_some(),
    ];
    if provided.iter().any(|value| *value) && !provided.iter().all(|value| *value) {
        return Err(BackendError::RuntimeShapeMismatch(
            "output projection final-norm component diagnostics require final hidden, output norm weight, and output norm scale together".to_string(),
        ));
    }
    if let Some(hidden) = final_hidden {
        if hidden.rank() != 2 || hidden.dim(0)? != 1 || hidden.dim(1)? != hidden_width {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "output projection final-norm diagnostics expected final hidden shape [1, {hidden_width}], got {:?}",
                hidden.shape.dims
            )));
        }
    }
    if let Some(weight) = output_norm_weight {
        if weight.rank() != 1 || weight.dim(0)? != hidden_width {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "output projection final-norm diagnostics expected output norm weight shape [{hidden_width}], got {:?}",
                weight.shape.dims
            )));
        }
    }
    Ok(
        match (final_hidden, output_norm_weight, output_norm_scale) {
            (Some(final_hidden), Some(output_norm_weight), Some(output_norm_scale)) => {
                Some(OutputProjectionFinalNormSources {
                    final_hidden,
                    output_norm_weight,
                    output_norm_scale,
                })
            }
            _ => None,
        },
    )
}

fn validate_output_projection_row_layout(
    output_weight: &CpuTensor,
    hidden_width: usize,
    vocab_size: usize,
    layout: OutputProjectionLayout,
) -> Result<EffectiveOutputProjectionRowLayout> {
    if output_weight.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics expected rank-2 output weight, got {:?}",
            output_weight.shape.dims
        )));
    }
    match layout {
        OutputProjectionLayout::Descriptor => {
            let rows = output_weight.dim(0)?;
            let cols = output_weight.dim(1)?;
            if rows == hidden_width && cols == vocab_size {
                Ok(EffectiveOutputProjectionRowLayout::DescriptorInputOutput)
            } else if rows == vocab_size && cols == hidden_width {
                Ok(EffectiveOutputProjectionRowLayout::DescriptorOutputInput)
            } else {
                Err(BackendError::RuntimeShapeMismatch(format!(
                    "descriptor output projection diagnostics expected output weight [hidden, vocab] = [{hidden_width}, {vocab_size}] or tied/output-input [vocab, hidden] = [{vocab_size}, {hidden_width}], got {:?}",
                    output_weight.shape.dims
                )))
            }
        }
        OutputProjectionLayout::TokenMajor => {
            let rows = output_weight.dim(0)?;
            let cols = output_weight.dim(1)?;
            if rows == vocab_size && cols == hidden_width {
                Ok(EffectiveOutputProjectionRowLayout::DescriptorOutputInput)
            } else if rows == hidden_width
                && cols == vocab_size
                && (output_weight.data.len() == hidden_width * vocab_size
                    || output_weight.q8_0_file_backing.is_some())
            {
                Ok(EffectiveOutputProjectionRowLayout::TokenMajorReinterpret)
            } else {
                Err(BackendError::RuntimeShapeMismatch(format!(
                    "token-major output projection diagnostics expected tied/output-input [{vocab_size}, {hidden_width}] or a token-major reinterpretation of [{hidden_width}, {vocab_size}], got shape {:?} with {} values",
                    output_weight.shape.dims,
                    output_weight.data.len()
                )))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveOutputProjectionRowLayout {
    DescriptorInputOutput,
    DescriptorOutputInput,
    TokenMajorReinterpret,
}

impl EffectiveOutputProjectionRowLayout {
    fn label(self) -> &'static str {
        match self {
            Self::DescriptorInputOutput => "descriptor",
            Self::DescriptorOutputInput => "output_input",
            Self::TokenMajorReinterpret => "token_major",
        }
    }
}

struct OutputProjectionTokenRow {
    values: Vec<f32>,
    q8_0_row_bytes: Option<Vec<u8>>,
}

fn output_projection_token_row(
    output_weight: &CpuTensor,
    hidden_width: usize,
    token_index: usize,
    layout: EffectiveOutputProjectionRowLayout,
) -> Result<OutputProjectionTokenRow> {
    if !output_weight.data.is_empty() {
        let values = match layout {
            EffectiveOutputProjectionRowLayout::DescriptorInputOutput => {
                let output_row_width = output_weight.shape.dims[1];
                (0..hidden_width)
                    .map(|hidden_index| {
                        output_weight.data[hidden_index * output_row_width + token_index]
                    })
                    .collect()
            }
            EffectiveOutputProjectionRowLayout::DescriptorOutputInput
            | EffectiveOutputProjectionRowLayout::TokenMajorReinterpret => {
                let start = token_index.checked_mul(hidden_width).ok_or_else(|| {
                    BackendError::RuntimeShapeMismatch(
                        "output projection token row offset overflow".to_string(),
                    )
                })?;
                output_weight.data[start..start + hidden_width].to_vec()
            }
        };
        return Ok(OutputProjectionTokenRow {
            values,
            q8_0_row_bytes: None,
        });
    }

    let backing = output_weight.q8_0_file_backing.as_ref().ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics need dense values or q8_0 file backing for token row {}, got empty tensor {}",
            token_index, output_weight.name
        ))
    })?;
    if output_weight.source_type != Some(GgufTensorType::Q8_0) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics only support empty file-backed rows for q8_0 tensors, got {:?}",
            output_weight.source_type
        )));
    }
    if layout != EffectiveOutputProjectionRowLayout::TokenMajorReinterpret
        && layout != EffectiveOutputProjectionRowLayout::DescriptorOutputInput
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection diagnostics cannot lazily decode {} layout for tensor {}",
            layout.label(),
            output_weight.name
        )));
    }
    if !hidden_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection q8_0 diagnostic hidden width {hidden_width} is not block aligned"
        )));
    }

    let row_count = match layout {
        EffectiveOutputProjectionRowLayout::DescriptorOutputInput => output_weight.dim(0)?,
        EffectiveOutputProjectionRowLayout::TokenMajorReinterpret => output_weight.dim(1)?,
        EffectiveOutputProjectionRowLayout::DescriptorInputOutput => unreachable!(
            "descriptor input/output layout is rejected for file-backed q8_0 diagnostics"
        ),
    };
    if token_index >= row_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection q8_0 diagnostic token row {token_index} exceeds row count {row_count} for tensor {}",
            output_weight.name
        )));
    }

    let blocks_per_row = hidden_width / Q8_0_BLOCK_VALUES;
    let expected_blocks = row_count.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "output projection q8_0 diagnostic block count overflow".to_string(),
        )
    })?;
    if backing.num_blocks != expected_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection q8_0 diagnostic expected {expected_blocks} blocks for tensor {} shape {:?}, got {}",
            output_weight.name,
            output_weight.shape.dims,
            backing.num_blocks
        )));
    }
    let row_block_start = token_index.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "output projection token row block offset overflow".to_string(),
        )
    })?;
    let row_offset = backing
        .absolute_offset
        .checked_add((row_block_start * Q8BlockReader::BLOCK_SIZE_BYTES) as u64)
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "output projection token row byte offset overflow".to_string(),
            )
        })?;
    let row_bytes_len = blocks_per_row
        .checked_mul(Q8BlockReader::BLOCK_SIZE_BYTES)
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "output projection token row byte length overflow".to_string(),
            )
        })?;
    let mut row_bytes = vec![0_u8; row_bytes_len];
    backing.read_exact_at_cached(&mut row_bytes, row_offset)?;
    let mut dest = Vec::with_capacity(hidden_width);
    for block in row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        dest.extend(block[2..].iter().map(|q| scale * f32::from(*q as i8)));
    }
    Ok(OutputProjectionTokenRow {
        values: dest,
        q8_0_row_bytes: Some(row_bytes),
    })
}

fn output_projection_q8_0_reconstructed_logit(
    output_norm: &CpuTensor,
    output_weight: &CpuTensor,
    hidden_width: usize,
    token_index: usize,
    layout: EffectiveOutputProjectionRowLayout,
    q8_0_row_bytes: Option<&[u8]>,
) -> Result<Option<f32>> {
    let Some(backing) = output_weight.q8_0_file_backing.as_ref() else {
        return Ok(None);
    };
    if !output_weight.data.is_empty() {
        return Ok(None);
    }
    if output_weight.source_type != Some(GgufTensorType::Q8_0) {
        return Ok(None);
    }
    if layout != EffectiveOutputProjectionRowLayout::TokenMajorReinterpret
        && layout != EffectiveOutputProjectionRowLayout::DescriptorOutputInput
    {
        return Ok(None);
    }
    if !hidden_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "output projection q8_0 diagnostic hidden width {hidden_width} is not block aligned"
        )));
    }

    let blocks_per_row = hidden_width / Q8_0_BLOCK_VALUES;
    let row_bytes_len = blocks_per_row
        .checked_mul(Q8BlockReader::BLOCK_SIZE_BYTES)
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "output projection q8_0 diagnostic row byte length overflow".to_string(),
            )
        })?;
    let mut owned_row_bytes = Vec::new();
    let row_bytes = if let Some(row_bytes) = q8_0_row_bytes {
        if row_bytes.len() != row_bytes_len {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "output projection q8_0 diagnostic supplied row bytes length {} does not match expected {row_bytes_len}",
                row_bytes.len()
            )));
        }
        row_bytes
    } else {
        let row_block_start = token_index.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "output projection q8_0 diagnostic row block offset overflow".to_string(),
            )
        })?;
        let row_offset = backing
            .absolute_offset
            .checked_add((row_block_start * Q8BlockReader::BLOCK_SIZE_BYTES) as u64)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "output projection q8_0 diagnostic row byte offset overflow".to_string(),
                )
            })?;
        owned_row_bytes.resize(row_bytes_len, 0_u8);
        backing.read_exact_at_cached(&mut owned_row_bytes, row_offset)?;
        &owned_row_bytes
    };
    let mut scales = vec![0.0_f32; blocks_per_row];
    decode_q8_0_encoded_row_scales(row_bytes, &mut scales);
    if q8_0_file_reader_block_dot_enabled() {
        let quantized_input = quantize_q8_0_row(&output_norm.data[..hidden_width]);
        Ok(Some(dot_q8_0_encoded_row_quantized_input_with_scales(
            &quantized_input.blocks,
            row_bytes,
            &scales,
        )))
    } else {
        Ok(Some(dot_q8_0_encoded_row_f32_input_with_scales(
            &output_norm.data[..hidden_width],
            row_bytes,
            &scales,
        )))
    }
}

fn output_projection_token_diagnostic(
    output_norm: &CpuTensor,
    output_weight: &CpuTensor,
    reported_logit: f32,
    token_id: u32,
    hidden_width: usize,
    layout: EffectiveOutputProjectionRowLayout,
    final_norm_sources: Option<OutputProjectionFinalNormSources<'_>>,
) -> Result<LlamaOutputProjectionDiagnostic> {
    let token_index = token_id as usize;
    let output_row = output_projection_token_row(output_weight, hidden_width, token_index, layout)?;
    let q8_direct_reconstructed_logit = output_projection_q8_0_reconstructed_logit(
        output_norm,
        output_weight,
        hidden_width,
        token_index,
        layout,
        output_row.q8_0_row_bytes.as_deref(),
    )?;
    let mut reconstructed_logit = 0.0f32;
    let mut norm_sum_sq = 0.0f32;
    let mut row_sum_sq = 0.0f32;
    let mut output_norm_first_values = Vec::new();
    let mut output_row_first_values = Vec::new();
    let mut component_products_first_values = Vec::new();
    let mut component_products = Vec::with_capacity(hidden_width);
    let mut component_diagnostics = Vec::with_capacity(hidden_width);
    let mut max_abs_component_index = 0usize;
    let mut max_abs_component = 0.0f32;
    let mut positive_component_sum = 0.0f32;
    let mut negative_component_sum = 0.0f32;

    for (idx, row_value) in output_row.values.iter().enumerate().take(hidden_width) {
        let norm_value = output_norm.data[idx];
        let row_value = *row_value;
        let component = norm_value * row_value;
        component_products.push(component);
        let final_norm_component = match final_norm_sources {
            Some(sources) => {
                let hidden_value = sources.final_hidden.data[idx];
                let weight_value = sources.output_norm_weight.data[idx];
                let reconstructed = hidden_value * sources.output_norm_scale * weight_value;
                (
                    Some(hidden_value),
                    Some(weight_value),
                    Some(sources.output_norm_scale),
                    Some(reconstructed),
                    Some((norm_value - reconstructed).abs()),
                )
            }
            None => (None, None, None, None, None),
        };
        component_diagnostics.push(LlamaOutputProjectionComponentDiagnostic {
            index: idx,
            final_hidden_value: final_norm_component.0,
            output_norm_weight_value: final_norm_component.1,
            output_norm_scale: final_norm_component.2,
            reconstructed_output_norm_value: final_norm_component.3,
            output_norm_reconstruction_delta: final_norm_component.4,
            output_norm_value: norm_value,
            output_row_value: row_value,
            component,
        });
        reconstructed_logit += component;
        norm_sum_sq += norm_value * norm_value;
        row_sum_sq += row_value * row_value;
        if component >= 0.0 {
            positive_component_sum += component;
        } else {
            negative_component_sum += component;
        }
        if component.abs() > max_abs_component.abs() {
            max_abs_component_index = idx;
            max_abs_component = component;
        }
        if output_norm_first_values.len() < 8 {
            output_norm_first_values.push(norm_value);
            output_row_first_values.push(row_value);
            component_products_first_values.push(component);
        }
    }

    let top_positive_components = top_signed_output_components(&component_diagnostics, true);
    let top_negative_components = top_signed_output_components(&component_diagnostics, false);

    let component_products_max_abs_window_start = max_abs_component_index.saturating_sub(4);
    let component_products_max_abs_window = component_products
        .iter()
        .skip(component_products_max_abs_window_start)
        .take(8)
        .copied()
        .collect::<Vec<_>>();

    let decoded_component_reconstructed_logit = reconstructed_logit;
    if let Some(q8_direct_reconstructed_logit) = q8_direct_reconstructed_logit {
        reconstructed_logit = q8_direct_reconstructed_logit;
    }

    let output_norm_rms = (norm_sum_sq / hidden_width as f32).sqrt();
    let output_row_rms = (row_sum_sq / hidden_width as f32).sqrt();
    let cosine_denominator = norm_sum_sq.sqrt() * row_sum_sq.sqrt();
    let cosine_similarity = if cosine_denominator == 0.0 {
        0.0
    } else {
        reconstructed_logit / cosine_denominator
    };

    Ok(LlamaOutputProjectionDiagnostic {
        token_id,
        layout: layout.label(),
        reported_logit,
        reconstructed_logit,
        decoded_component_reconstructed_logit,
        q8_direct_reconstructed_logit,
        absolute_delta: (reported_logit - reconstructed_logit).abs(),
        q8_direct_absolute_delta: q8_direct_reconstructed_logit
            .map(|value| (reported_logit - value).abs()),
        q8_direct_decoded_component_delta: q8_direct_reconstructed_logit
            .map(|value| (value - decoded_component_reconstructed_logit).abs()),
        output_norm_rms,
        output_row_rms,
        cosine_similarity,
        output_norm_first_values,
        output_row_first_values,
        component_products_first_values,
        component_products_max_abs_window_start,
        component_products_max_abs_window,
        max_abs_component_index,
        max_abs_component,
        positive_component_sum,
        negative_component_sum,
        top_positive_components,
        top_negative_components,
    })
}

fn top_signed_output_components(
    components: &[LlamaOutputProjectionComponentDiagnostic],
    positive: bool,
) -> Vec<LlamaOutputProjectionComponentDiagnostic> {
    let mut filtered = components
        .iter()
        .filter(|component| {
            (positive && component.component > 0.0) || (!positive && component.component < 0.0)
        })
        .cloned()
        .collect::<Vec<_>>();
    filtered.sort_by(|left, right| {
        if positive {
            right
                .component
                .partial_cmp(&left.component)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            left.component
                .partial_cmp(&right.component)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    filtered.truncate(TENSOR_CHECKPOINT_SAMPLE);
    filtered
}

fn try_attention_qkv_shared_q8_0_block_dot(
    input: &CpuTensor,
    q_weight: &CpuTensor,
    k_weight: &CpuTensor,
    v_weight: &CpuTensor,
) -> Result<Option<(CpuTensor, CpuTensor, CpuTensor)>> {
    if input.rank() != 2 || input.dim(0)? != 1 {
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    let q_width = linear_output_width(input, q_weight, "attention q")?;
    let k_width = linear_output_width(input, k_weight, "attention k")?;
    let v_width = linear_output_width(input, v_weight, "attention v")?;
    let input_row = &input.data[..input_width];

    let (q_transposed, k_transposed, v_transposed) = match (
        borrowed_linear_weight_as_transposed(q_weight, input_width),
        borrowed_linear_weight_as_transposed(k_weight, input_width),
        borrowed_linear_weight_as_transposed(v_weight, input_width),
    ) {
        (Ok(q_transposed), Ok(k_transposed), Ok(v_transposed))
            if q_transposed.rows == q_width
                && k_transposed.rows == k_width
                && v_transposed.rows == v_width
                && should_use_borrowed_q8_0_block_dot(q_transposed, input_width)
                && should_use_borrowed_q8_0_block_dot(k_transposed, input_width)
                && should_use_borrowed_q8_0_block_dot(v_transposed, input_width) =>
        {
            (q_transposed, k_transposed, v_transposed)
        }
        _ => return Ok(None),
    };

    let quantized_input = quantize_q8_0_row(input_row);
    let mut q = vec![0.0; q_width];
    let mut k = vec![0.0; k_width];
    let mut v = vec![0.0; v_width];
    accumulate_transposed_linear_row_q8_0_block_dot_quantized(
        &quantized_input.blocks,
        q_transposed,
        &mut q,
    );
    accumulate_transposed_linear_row_q8_0_block_dot_quantized(
        &quantized_input.blocks,
        k_transposed,
        &mut k,
    );
    accumulate_transposed_linear_row_q8_0_block_dot_quantized(
        &quantized_input.blocks,
        v_transposed,
        &mut v,
    );

    Ok(Some((
        CpuTensor::from_f32("attention_q_shared_q8", vec![1, q_width], q)?,
        CpuTensor::from_f32("attention_k_shared_q8", vec![1, k_width], k)?,
        CpuTensor::from_f32("attention_v_shared_q8", vec![1, v_width], v)?,
    )))
}

fn gated_ffn_activation(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<GatedFfnActivation> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    gated_ffn_activation_with_plan(
        input,
        gate_weight,
        up_weight,
        name,
        &runtime_plan,
        collect_diagnostics,
    )
}

fn gated_ffn_activation_with_plan(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
    collect_diagnostics: bool,
) -> Result<GatedFfnActivation> {
    let name = name.into();
    let rows = input.dim(0)?;
    if input.rank() != 2 || rows != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN activation expects a single-row input, got {:?}",
            input.shape.dims
        )));
    }
    let input_width = input.dim(1)?;
    let gate_width = linear_output_width(input, gate_weight, "ffn gate")?;
    let up_width = linear_output_width(input, up_weight, "ffn up")?;
    if gate_width != up_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN gate/up width mismatch: gate output {gate_width}, up output {up_width}"
        )));
    }

    if !collect_diagnostics {
        if let Some(activated) = try_x86_q8_ffn_gate_up_single_owner_path(
            input,
            gate_weight,
            up_weight,
            &name,
            runtime_plan,
        )? {
            return Ok(activated);
        }
        if let Some(activated) = try_x86_q8_ffn_gate_up_decode_fused_activation_path(
            input,
            gate_weight,
            up_weight,
            &name,
            runtime_plan,
        )? {
            return Ok(activated);
        }
    }

    let input_row = &input.data[..input_width];
    let mut gate = vec![0.0; gate_width];
    let mut up = vec![0.0; up_width];

    let ffn_gate_up_decode_consumer = if collect_diagnostics {
        if runtime_plan.q8.ffn_gate_up_decode_consumer {
            record_q8_schedule_projection_route_denial(
                "ffn_gate_up",
                "decode_consumer",
                "stream_diagnostics_collect_projection_details",
                rows,
                input_width,
                gate_width,
            );
        }
        None
    } else {
        try_x86_q8_ffn_gate_up_decode_consumer_path(
            input,
            gate_weight,
            up_weight,
            &name,
            &mut gate,
            &mut up,
            runtime_plan,
        )?
    };

    let shared_q8_gate_up = if collect_diagnostics || ffn_gate_up_decode_consumer.is_some() {
        None
    } else {
        match (
            borrowed_linear_weight_as_transposed(gate_weight, input_width),
            borrowed_linear_weight_as_transposed(up_weight, input_width),
        ) {
            (Ok(gate_transposed), Ok(up_transposed))
                if gate_transposed.rows == gate_width
                    && up_transposed.rows == up_width
                    && should_use_borrowed_q8_0_block_dot_with_plan(
                        gate_transposed,
                        input_width,
                        runtime_plan,
                    )
                    && should_use_borrowed_q8_0_block_dot_with_plan(
                        up_transposed,
                        input_width,
                        runtime_plan,
                    ) =>
            {
                Some((gate_transposed, up_transposed))
            }
            _ => None,
        }
    };

    let used_ffn_gate_up_decode_consumer = ffn_gate_up_decode_consumer.is_some();
    let (gate_elapsed, up_elapsed) = if let Some(elapsed) = ffn_gate_up_decode_consumer {
        elapsed
    } else if let Some((gate_transposed, up_transposed)) = shared_q8_gate_up {
        let started = Instant::now();
        let quantized_input = quantize_q8_0_row(input_row);
        if let Some(total_elapsed) = try_gated_ffn_gate_up_hybrid_q8_0(
            &quantized_input.blocks,
            gate_transposed,
            up_transposed,
            &mut gate,
            &mut up,
            &runtime_plan.q8,
        ) {
            // Gate/up are submitted as one hybrid CPU+Metal batch, so split the measured
            // elapsed time across the two existing timing fields while preserving the total.
            let gate_elapsed = total_elapsed / 2;
            (gate_elapsed, total_elapsed - gate_elapsed)
        } else if let (Some(gate_blocks), Some(up_blocks)) =
            (gate_transposed.q8_0_blocks, up_transposed.q8_0_blocks)
        {
            accumulate_two_q8_0_block_dot_quantized_cpu(
                &quantized_input.blocks,
                gate_blocks,
                &mut gate,
                up_blocks,
                &mut up,
            );
            let total_elapsed = started.elapsed().as_micros();
            let gate_elapsed = total_elapsed / 2;
            (gate_elapsed, total_elapsed - gate_elapsed)
        } else {
            accumulate_transposed_linear_row_q8_0_block_dot_quantized(
                &quantized_input.blocks,
                gate_transposed,
                &mut gate,
            );
            accumulate_transposed_linear_row_q8_0_block_dot_quantized(
                &quantized_input.blocks,
                up_transposed,
                &mut up,
            );
            let total_elapsed = started.elapsed().as_micros();
            let gate_elapsed = total_elapsed / 2;
            (gate_elapsed, total_elapsed - gate_elapsed)
        }
    } else {
        let started = Instant::now();
        accumulate_linear_row(
            input_row,
            gate_weight,
            &mut gate,
            "ffn gate",
            collect_diagnostics,
        )?;
        let gate_elapsed = started.elapsed().as_micros();

        let started = Instant::now();
        accumulate_linear_row(input_row, up_weight, &mut up, "ffn up", collect_diagnostics)?;
        (gate_elapsed, started.elapsed().as_micros())
    };

    let gate_projection = collect_diagnostics
        .then(|| CpuTensor::from_f32("ffn_gate_diagnostic", vec![1, gate_width], gate.clone()))
        .transpose()?;
    let up_projection = collect_diagnostics
        .then(|| CpuTensor::from_f32("ffn_up_diagnostic", vec![1, up_width], up.clone()))
        .transpose()?;
    let gate_stats = gate_projection.as_ref().map(tensor_stats).transpose()?;
    let up_stats = up_projection.as_ref().map(tensor_stats).transpose()?;
    let gate_diagnostic = gate_projection
        .as_ref()
        .map(|tensor| linear_projection_diagnostics(input, gate_weight, tensor, "ffn gate"))
        .transpose()?;
    let up_diagnostic = up_projection
        .as_ref()
        .map(|tensor| linear_projection_diagnostics(input, up_weight, tensor, "ffn up"))
        .transpose()?;

    let order = diagnostic_ffn_gate_up_order()?;
    let started = Instant::now();
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = match order {
            FfnGateUpOrder::GateUp => (*gate_value / (1.0 + (-*gate_value).exp())) * up_value,
            FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * *gate_value,
        };
    }
    let activation_elapsed = started.elapsed().as_micros();
    if used_ffn_gate_up_decode_consumer {
        add_q8_schedule_counter(
            &Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US,
            activation_elapsed as u64,
        );
    }
    let tensor_started = q8_schedule_telemetry_enabled().then(Instant::now);
    let tensor = CpuTensor::from_f32(name, vec![1, gate_width], gate)?;
    if used_ffn_gate_up_decode_consumer {
        if let Some(started) = tensor_started {
            add_q8_schedule_counter(
                &Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US,
                started.elapsed().as_micros() as u64,
            );
        }
    }
    let activation_diagnostic = if collect_diagnostics {
        Some(ffn_activation_diagnostics(
            gate_projection
                .as_ref()
                .expect("gate projection collected for activation diagnostics"),
            up_projection
                .as_ref()
                .expect("up projection collected for activation diagnostics"),
            &tensor,
            order,
        )?)
    } else {
        None
    };

    Ok(GatedFfnActivation {
        tensor,
        gate: gate_elapsed,
        up: up_elapsed,
        activation: activation_elapsed,
        gate_stats,
        up_stats,
        gate_diagnostic,
        up_diagnostic,
        activation_diagnostic,
    })
}

fn try_x86_q8_ffn_gate_up_decode_fused_activation_path(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<GatedFfnActivation>> {
    let name = name.into();
    if !runtime_plan.q8.ffn_gate_up_decode_consumer
        || !runtime_plan.q8.ffn_gate_up_decode_fused_activation
        || input.rank() != 2
        || input.dim(0)? != 1
    {
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_fused_activation",
            "input_width_not_q8_block_multiple",
            1,
            input_width,
            0,
        );
        return Ok(None);
    }
    let Some((gate_packed, gate_width)) = q8_0_runtime_packed_projection(gate_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_fused_activation",
            "missing_gate_runtime_packed_rows4",
            1,
            input_width,
            0,
        );
        return Ok(None);
    };
    let Some((up_packed, up_width)) = q8_0_runtime_packed_projection(up_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_fused_activation",
            "missing_up_runtime_packed_rows4",
            1,
            input_width,
            gate_width,
        );
        return Ok(None);
    };
    if gate_width != up_width
        || gate_packed.interleave != Q8_0PackedRows4Interleave::I8
        || up_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_fused_activation",
            "packed_projection_shape_or_interleave_mismatch",
            1,
            input_width,
            gate_width,
        );
        return Ok(None);
    }

    let order = diagnostic_ffn_gate_up_order()?;
    let started = Instant::now();
    let quantized_input = quantize_q8_0_row(&input.data[..input_width]);
    let tensor = q8_0_packed_rows4_single_input_projection_pair_activated_from_quantized(
        gate_packed,
        up_packed,
        gate_width,
        &name,
        order,
        &quantized_input.blocks,
        runtime_plan.q8.ffn_gate_up_decode_paired_dot,
    )?;
    let total_elapsed = started.elapsed().as_micros();
    record_q8_schedule_output_projection_route_call(
        "ffn_gate_up",
        "decode_fused_activation",
        Some(&tensor.name),
        1,
        input_width,
        gate_width,
        total_elapsed,
    );
    let gate_elapsed = total_elapsed / 2;
    Ok(Some(GatedFfnActivation {
        tensor,
        gate: gate_elapsed,
        up: total_elapsed - gate_elapsed,
        activation: 0,
        gate_stats: None,
        up_stats: None,
        gate_diagnostic: None,
        up_diagnostic: None,
        activation_diagnostic: None,
    }))
}

fn try_gated_ffn_gate_up_hybrid_q8_0(
    quantized_input: &[Q8_0Block],
    gate_weight: BorrowedLinearWeight<'_>,
    up_weight: BorrowedLinearWeight<'_>,
    gate: &mut [f32],
    up: &mut [f32],
    q8_flags: &Q8RuntimeFlags,
) -> Option<u128> {
    if !q8_flags.hybrid_retained || gate.is_empty() || gate.len() != up.len() {
        return None;
    }
    let blocks_per_row = quantized_input.len();
    let gate_weight_blocks = gate_weight.q8_0_blocks?;
    let up_weight_blocks = up_weight.q8_0_blocks?;
    if gate_weight_blocks.len() != gate.len().saturating_mul(blocks_per_row)
        || up_weight_blocks.len() != up.len().saturating_mul(blocks_per_row)
    {
        return None;
    }
    let gpu_rows = q8_flags.hybrid_gpu_rows_for_output(gate.len());
    if gpu_rows == 0 || gpu_rows >= gate.len() {
        return None;
    }
    let cpu_rows = gate.len() - gpu_rows;
    let gpu_block_start = cpu_rows * blocks_per_row;

    let (gate_cpu_output, gate_gpu_output) = gate.split_at_mut(cpu_rows);
    let (up_cpu_output, up_gpu_output) = up.split_at_mut(cpu_rows);
    let gate_cpu_weight_blocks = &gate_weight_blocks[..gpu_block_start];
    let gate_gpu_weight_blocks = &gate_weight_blocks[gpu_block_start..];
    let up_cpu_weight_blocks = &up_weight_blocks[..gpu_block_start];
    let up_gpu_weight_blocks = &up_weight_blocks[gpu_block_start..];
    let gate_gpu_weight_bytes = q8_0_blocks_as_bytes(gate_gpu_weight_blocks);
    let up_gpu_weight_bytes = q8_0_blocks_as_bytes(up_gpu_weight_blocks);

    let started = Instant::now();
    if with_q8_0_block_scales_and_quants(quantized_input, |input_scales, input_quants| {
        metal::try_q8_0_block_two_linear_rows_with_cpu(
            input_scales,
            input_quants,
            gate_gpu_weight_bytes,
            up_gpu_weight_bytes,
            gpu_rows,
            blocks_per_row,
            gate_gpu_output,
            up_gpu_output,
            || {
                accumulate_two_q8_0_block_dot_quantized_cpu(
                    quantized_input,
                    gate_cpu_weight_blocks,
                    gate_cpu_output,
                    up_cpu_weight_blocks,
                    up_cpu_output,
                );
            },
        )
    }) {
        trace_q8_0_hybrid_retained_success(cpu_rows, gpu_rows, blocks_per_row);
        Some(started.elapsed().as_micros())
    } else {
        None
    }
}

fn gated_ffn_activation_batch(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<GatedFfnActivation> {
    let name = name.into();
    if input.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN batch activation expects rank-2 input, got {:?}",
            input.shape.dims
        )));
    }

    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    let rows = input.dim(0)?;
    if q8_schedule_telemetry_enabled()
        && rows == 1
        && runtime_plan.q8.ffn_gate_up_decode_consumer
        && !runtime_plan.q8.ffn_gate_up_single_owner
    {
        let input_width = input.dim(1)?;
        let gate_width = linear_output_width(input, gate_weight, "ffn gate")?;
        let up_width = linear_output_width(input, up_weight, "ffn up")?;
        if gate_width == up_width {
            record_q8_schedule_projection_route_denial(
                "ffn_gate_up",
                "decode_consumer",
                "batch_decode_requires_single_owner_or_direct_call",
                rows,
                input_width,
                gate_width,
            );
        }
    }
    if let Some(activated) = try_x86_q8_ffn_gate_up_single_owner_path(
        input,
        gate_weight,
        up_weight,
        &name,
        &runtime_plan,
    )? {
        return Ok(activated);
    }
    if let Some(activated) = try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        input,
        gate_weight,
        up_weight,
        &name,
        &runtime_plan,
    )? {
        return Ok(activated);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(activated) =
        try_gated_ffn_activation_batch_packed_prefill_i8mm(input, gate_weight, up_weight, &name)?
    {
        return Ok(activated);
    }

    let started = Instant::now();
    let mut gate =
        linear_for_role_runtime(input, gate_weight, "ffn_gate_prefill", "ffn gate", false)?;
    let gate_elapsed = started.elapsed().as_micros();

    let started = Instant::now();
    let up = linear_for_role_runtime(input, up_weight, "ffn_up_prefill", "ffn up", false)?;
    let up_elapsed = started.elapsed().as_micros();

    require_tensor_shape(&up, &gate.shape.dims, "gated FFN prefill up projection")?;
    let order = diagnostic_ffn_gate_up_order()?;
    let started = Instant::now();
    for (gate_value, up_value) in gate.data.iter_mut().zip(up.data) {
        *gate_value = apply_ffn_gate_up_order(*gate_value, up_value, order);
    }
    gate.name = name;
    let activation_elapsed = started.elapsed().as_micros();

    Ok(GatedFfnActivation {
        tensor: gate,
        gate: gate_elapsed,
        up: up_elapsed,
        activation: activation_elapsed,
        gate_stats: None,
        up_stats: None,
        gate_diagnostic: None,
        up_diagnostic: None,
        activation_diagnostic: None,
    })
}

fn try_x86_q8_ffn_gate_up_single_owner_path(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<GatedFfnActivation>> {
    if !runtime_plan.q8.ffn_gate_up_single_owner || input.rank() != 2 {
        return Ok(None);
    }
    let name = name.into();
    let rows = input.dim(0)?;
    if rows == 0 {
        return Ok(None);
    }

    if rows > 1 {
        let mut matmul_plan = *runtime_plan;
        matmul_plan.q8.ffn_gate_up_packed_rows4_matmul = true;
        return try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
            input,
            gate_weight,
            up_weight,
            &name,
            &matmul_plan,
        );
    }

    let input_width = input.dim(1)?;
    let gate_width = linear_output_width(input, gate_weight, "ffn gate")?;
    let up_width = linear_output_width(input, up_weight, "ffn up")?;
    if gate_width != up_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN gate/up width mismatch: gate output {gate_width}, up output {up_width}"
        )));
    }

    let mut gate = vec![0.0; gate_width];
    let mut up = vec![0.0; up_width];
    let mut decode_plan = *runtime_plan;
    decode_plan.q8.ffn_gate_up_decode_consumer = true;
    let Some((gate_elapsed, up_elapsed)) = try_x86_q8_ffn_gate_up_decode_consumer_path(
        input,
        gate_weight,
        up_weight,
        &name,
        &mut gate,
        &mut up,
        &decode_plan,
    )?
    else {
        let _ = input_width;
        return Ok(None);
    };

    let order = diagnostic_ffn_gate_up_order()?;
    let activation_started = Instant::now();
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = apply_ffn_gate_up_order(*gate_value, up_value, order);
    }

    Ok(Some(GatedFfnActivation {
        tensor: CpuTensor::from_f32(name, vec![1, gate_width], gate)?,
        gate: gate_elapsed,
        up: up_elapsed,
        activation: activation_started.elapsed().as_micros(),
        gate_stats: None,
        up_stats: None,
        gate_diagnostic: None,
        up_diagnostic: None,
        activation_diagnostic: None,
    }))
}

fn try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<GatedFfnActivation>> {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (input, gate_weight, up_weight, name, runtime_plan);
        Ok(None)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let Some(route) = resolve_x86_q8_ffn_gate_up_route(
            input,
            gate_weight,
            up_weight,
            runtime_plan,
            X86Q8FfnGateUpRouteKind::PackedRows4Matmul,
        )?
        else {
            return Ok(None);
        };

        let order = diagnostic_ffn_gate_up_order()?;
        let projection_started = Instant::now();
        let gate = with_q8_0_quantized_matmul_input_rows(
            input,
            route.gate_packed.blocks_per_row,
            |rows, quantized_inputs| {
                q8_0_packed_rows4_matmul_projection_pair_activated_from_quantized(
                    rows,
                    route.gate_packed,
                    route.up_packed,
                    route.output_width,
                    name,
                    order,
                    quantized_inputs,
                )
            },
        )?;
        let projection_elapsed = projection_started.elapsed().as_micros();
        record_q8_schedule_output_projection_route_call(
            "ffn_gate_up",
            X86Q8FfnGateUpRouteKind::PackedRows4Matmul.telemetry_name(),
            Some(name),
            route.rows,
            route.input_width,
            route.output_width,
            projection_elapsed,
        );
        let gate_elapsed = projection_elapsed / 2;
        let up_elapsed = projection_elapsed - gate_elapsed;

        Ok(Some(GatedFfnActivation {
            tensor: gate,
            gate: gate_elapsed,
            up: up_elapsed,
            activation: 0,
            gate_stats: None,
            up_stats: None,
            gate_diagnostic: None,
            up_diagnostic: None,
            activation_diagnostic: None,
        }))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn try_gated_ffn_activation_batch_packed_prefill_i8mm(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: &str,
) -> Result<Option<GatedFfnActivation>> {
    if !mac_q8_sched_packed_prefill_enabled() {
        return Ok(None);
    }
    let rows = input.dim(0)?;
    if rows < 2 {
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let gate_width = linear_output_width(input, gate_weight, "ffn gate")?;
    let up_width = linear_output_width(input, up_weight, "ffn up")?;
    if gate_width != up_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN gate/up width mismatch: gate output {gate_width}, up output {up_width}"
        )));
    }
    let Some((gate_packed, Q8_0PackedRows4Interleave::I8)) =
        q8_0_selected_packed_rows4(gate_weight)
    else {
        return Ok(None);
    };
    let Some((up_packed, Q8_0PackedRows4Interleave::I8)) = q8_0_selected_packed_rows4(up_weight)
    else {
        return Ok(None);
    };
    if gate_packed.rows != gate_width
        || up_packed.rows != up_width
        || gate_packed.blocks_per_row != blocks_per_row
        || up_packed.blocks_per_row != blocks_per_row
    {
        return Ok(None);
    }

    let collect_q8_schedule = q8_schedule_telemetry_enabled();
    let projection_started = Instant::now();
    let (mut gate, up) = if mac_q8_prefill_i8mm_enabled() && rows >= 4 {
        let mut gate = vec![0.0_f32; rows * gate_width];
        let mut up = vec![0.0_f32; rows * up_width];
        let packed_rows = rows / 4 * 4;
        if collect_q8_schedule {
            add_q8_schedule_counter(&Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS, 1);
        }
        with_q8_0_file_reader_quantized_inputs(|quantized_inputs| {
            quantized_inputs.clear();
            Q8_0_PREFILL_PACKED_INPUTS.with(|cell| {
                let mut packed_inputs = cell.borrow_mut();
                packed_inputs.clear();
                let before_capacity = packed_inputs.capacity();
                let pack_started = collect_q8_schedule.then(Instant::now);
                quantize_pack_q8_0_rows4_i8_direct_into(
                    &input.data[..packed_rows * input_width],
                    packed_rows,
                    input_width,
                    blocks_per_row,
                    &mut packed_inputs,
                );
                if let Some(pack_started) = pack_started {
                    record_q8_schedule_activation_pack(
                        &mut packed_inputs,
                        before_capacity,
                        packed_rows,
                        blocks_per_row,
                        pack_started.elapsed().as_micros(),
                    );
                }
                let gemm_started = collect_q8_schedule.then(Instant::now);
                run_q8_0_packed_rows4_prefill_i8mm_two_kernel(
                    gate_packed,
                    up_packed,
                    &packed_inputs,
                    packed_rows / 4,
                    &mut gate,
                    &mut up,
                    collect_q8_schedule,
                );
                if let Some(gemm_started) = gemm_started {
                    add_q8_schedule_counter(
                        &Q8_SCHED_Q8_GEMM_COMPUTE_US,
                        gemm_started.elapsed().as_micros() as u64,
                    );
                }
                packed_inputs.clear();
                cap_q8_0_file_reader_scratch(&mut packed_inputs, 0);
            });

            let tail_rows = rows - packed_rows;
            if collect_q8_schedule {
                add_q8_schedule_counter(&Q8_SCHED_CONSERVATIVE_TAIL_ROWS, tail_rows as u64);
            }
            if tail_rows > 0 {
                quantized_inputs.reserve(tail_rows * blocks_per_row);
                for row in input.data[packed_rows * input_width..].chunks_exact(input_width) {
                    quantize_q8_0_blocks_into(row, quantized_inputs);
                }
            }
            for tail_row in 0..tail_rows {
                let input_start = tail_row * blocks_per_row;
                let row_blocks = &quantized_inputs[input_start..input_start + blocks_per_row];
                let output_start = (packed_rows + tail_row) * gate_width;
                accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                    row_blocks,
                    gate_packed,
                    Q8_0PackedRows4Interleave::I8,
                    &mut gate[output_start..output_start + gate_width],
                );
                accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                    row_blocks,
                    up_packed,
                    Q8_0PackedRows4Interleave::I8,
                    &mut up[output_start..output_start + up_width],
                );
            }
            Ok(())
        })?;
        (gate, up)
    } else {
        let (gate, up) = with_q8_0_quantized_matmul_input_rows(
            input,
            blocks_per_row,
            |rows, quantized_inputs| {
                q8_0_packed_rows4_matmul_projection_pair_from_quantized(
                    rows,
                    gate_packed,
                    up_packed,
                    gate_width,
                    up_width,
                    "ffn_gate_mac_q8_gate_up_packed_prefill",
                    "ffn_up_mac_q8_gate_up_packed_prefill",
                    quantized_inputs,
                )
            },
        )?;
        (gate.data, up.data)
    };
    let projection_elapsed = projection_started.elapsed().as_micros();
    let gate_elapsed = projection_elapsed / 2;
    let up_elapsed = projection_elapsed - gate_elapsed;

    let order = diagnostic_ffn_gate_up_order()?;
    let activation_started = Instant::now();
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = match order {
            FfnGateUpOrder::GateUp => (*gate_value / (1.0 + (-*gate_value).exp())) * up_value,
            FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * *gate_value,
        };
    }
    let activation_elapsed = activation_started.elapsed().as_micros();

    Ok(Some(GatedFfnActivation {
        tensor: CpuTensor::from_f32(name.to_string(), vec![rows, gate_width], gate)?,
        gate: gate_elapsed,
        up: up_elapsed,
        activation: activation_elapsed,
        gate_stats: None,
        up_stats: None,
        gate_diagnostic: None,
        up_diagnostic: None,
        activation_diagnostic: None,
    }))
}

fn softmax_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut scored = logits
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, (*value - max).exp()))
        .collect::<Vec<_>>();
    let sum = scored.iter().map(|(_, value)| *value).sum::<f32>();
    for (_, value) in &mut scored {
        *value /= sum;
    }
    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    if env_flag_enabled("CAMELID_MOE_RENORMALIZE_TOP_K") {
        let selected_sum = scored.iter().map(|(_, value)| *value).sum::<f32>();
        if selected_sum > 0.0 {
            for (_, value) in &mut scored {
                *value /= selected_sum;
            }
        }
    }
    scored
}

fn expert_matrix_view(
    weight: &CpuTensor,
    expert_idx: usize,
    input_width: usize,
    output_width: usize,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    if weight.rank() != 3 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert tensor {} expected rank 3, got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let experts = weight.dim(2)?;
    if expert_idx >= experts {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert index {expert_idx} out of bounds for {} experts",
            experts
        )));
    }
    let expert_elements = input_width.checked_mul(output_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("MoE expert element count overflow".to_string())
    })?;
    if weight.dim(0)? != input_width || weight.dim(1)? != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert tensor {} expected per-expert dims [{input_width}, {output_width}], got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let block_offset = expert_elements.checked_mul(expert_idx).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("MoE expert offset overflow".to_string())
    })? / Q8_0_BLOCK_VALUES;
    let block_count = expert_elements / Q8_0_BLOCK_VALUES;
    let mut tensor = if let Some(split_backings) = &weight.q8_0_split_file_backing {
        let backing = split_backings.get(expert_idx).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(format!(
                "MoE split expert index {expert_idx} missing from {} split backings",
                split_backings.len()
            ))
        })?;
        if backing.num_blocks != block_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "MoE split expert backing expected {block_count} blocks, got {}",
                backing.num_blocks
            )));
        }
        CpuTensor::q8_0_file_backed_linear(
            name,
            TensorShape {
                dims: vec![output_width, input_width],
            },
            backing.clone(),
        )
    } else if let Some(backing) = &weight.q8_0_file_backing {
        CpuTensor::q8_0_file_backed_linear(
            name,
            TensorShape {
                dims: vec![output_width, input_width],
            },
            Q8_0FileBacking::new(
                backing.path.clone(),
                backing.absolute_offset + (block_offset * Q8BlockReader::BLOCK_SIZE_BYTES) as u64,
                block_count,
            ),
        )
    } else {
        let start = expert_elements * expert_idx;
        let end = start + expert_elements;
        let data = weight
            .data
            .get(start..end)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(format!(
                    "MoE expert slice {start}..{end} missing from {}",
                    weight.name
                ))
            })?
            .to_vec();
        CpuTensor::from_f32(name, vec![output_width, input_width], data)?
    };
    tensor.source_type = weight.source_type;
    if let Some(blocks) = &weight.q8_0_blocks {
        tensor.q8_0_blocks = Some(blocks[block_offset..block_offset + block_count].to_vec());
    }
    Ok(tensor)
}

fn mixtral_moe_ffn(
    input: &CpuTensor,
    router: &CpuTensor,
    gate_experts: &CpuTensor,
    up_experts: &CpuTensor,
    down_experts: &CpuTensor,
    expert_used_count: usize,
    name: impl Into<String>,
) -> Result<(CpuTensor, u128, u128, u128, u128)> {
    if input.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Mixtral MoE FFN expects rank-2 input, got {:?}",
            input.shape.dims
        )));
    }
    let rows = input.dim(0)?;
    let hidden = input.dim(1)?;
    let ff = gate_experts.dim(1)?;
    let router_started = Instant::now();
    let logits = linear_for_role_runtime(input, router, "mixtral_router", "linear", false)?;
    let router_elapsed = router_started.elapsed().as_micros();
    let expert_count = logits.dim(1)?;
    let mut output = vec![0.0_f32; rows * hidden];
    let mut gate_elapsed = 0;
    let mut up_elapsed = 0;
    let mut activation_elapsed = 0;
    let mut down_elapsed = 0;
    for row in 0..rows {
        let row_input = CpuTensor::from_f32(
            "mixtral_moe_row",
            vec![1, hidden],
            input.data[row * hidden..(row + 1) * hidden].to_vec(),
        )?;
        let top = softmax_top_k(
            &logits.data[row * expert_count..(row + 1) * expert_count],
            expert_used_count,
        );
        for (expert_idx, weight) in top {
            let gate =
                expert_matrix_view(gate_experts, expert_idx, hidden, ff, "mixtral_gate_expert")?;
            let up = expert_matrix_view(up_experts, expert_idx, hidden, ff, "mixtral_up_expert")?;
            let down =
                expert_matrix_view(down_experts, expert_idx, ff, hidden, "mixtral_down_expert")?;
            let activated =
                gated_ffn_activation(&row_input, &gate, &up, "mixtral_expert_activated", false)?;
            gate_elapsed += activated.gate;
            up_elapsed += activated.up;
            activation_elapsed += activated.activation;
            let started = Instant::now();
            let expert_out = linear_for_role_runtime(
                &activated.tensor,
                &down,
                "mixtral_expert_down",
                "ffn_down",
                false,
            )?;
            down_elapsed += started.elapsed().as_micros();
            for col in 0..hidden {
                output[row * hidden + col] += expert_out.data[col] * weight;
            }
        }
    }
    Ok((
        CpuTensor::from_f32(name, vec![rows, hidden], output)?,
        gate_elapsed + router_elapsed,
        up_elapsed,
        activation_elapsed,
        down_elapsed,
    ))
}

fn ffn_activation_diagnostics(
    gate: &CpuTensor,
    up: &CpuTensor,
    reported: &CpuTensor,
    order: FfnGateUpOrder,
) -> Result<LlamaFfnActivationDiagnostic> {
    require_tensor_shape(up, &gate.shape.dims, "ffn activation up projection")?;
    require_tensor_shape(reported, &gate.shape.dims, "ffn activation reported output")?;
    if gate.rank() != 2 || gate.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "ffn activation diagnostics expected one gate row, got {:?}",
            gate.shape.dims
        )));
    }

    let mut reconstructed = Vec::with_capacity(gate.data.len());
    let mut max_abs_delta = 0.0_f32;
    let mut max_abs_delta_index = 0;
    let mut reported_max_abs = 0.0_f32;
    let mut reported_max_abs_index = 0;
    for (idx, ((gate_value, up_value), reported_value)) in gate
        .data
        .iter()
        .zip(up.data.iter())
        .zip(reported.data.iter())
        .enumerate()
    {
        let reconstructed_value = match order {
            FfnGateUpOrder::GateUp => (*gate_value / (1.0 + (-*gate_value).exp())) * *up_value,
            FfnGateUpOrder::UpGate => (*up_value / (1.0 + (-*up_value).exp())) * *gate_value,
        };
        let delta = (reconstructed_value - *reported_value).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported_value.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
        reconstructed.push(reconstructed_value);
    }
    let sample_len = reported.data.len().min(TENSOR_CHECKPOINT_SAMPLE);
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &reported.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );

    Ok(LlamaFfnActivationDiagnostic {
        gate_width: gate.dim(1)?,
        activation_order: order.label(),
        gate_first_values: gate.data.iter().take(sample_len).copied().collect(),
        up_first_values: up.data.iter().take(sample_len).copied().collect(),
        reconstructed_first_values: reconstructed.iter().take(sample_len).copied().collect(),
        reported_first_values: reported.data.iter().take(sample_len).copied().collect(),
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

struct GatedFfnActivation {
    tensor: CpuTensor,
    gate: u128,
    up: u128,
    activation: u128,
    gate_stats: Option<LlamaTensorStats>,
    up_stats: Option<LlamaTensorStats>,
    gate_diagnostic: Option<LlamaLinearProjectionDiagnostic>,
    up_diagnostic: Option<LlamaLinearProjectionDiagnostic>,
    activation_diagnostic: Option<LlamaFfnActivationDiagnostic>,
}

fn linear_output_width(input: &CpuTensor, weight: &CpuTensor, role: &str) -> Result<usize> {
    let input_width = input.dim(1)?;
    if weight.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{role} weight {} expected rank 2, got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let rows = weight.dim(0)?;
    let cols = weight.dim(1)?;
    if rows == input_width && cols == input_width {
        Ok(cols)
    } else {
        match diagnostic_rectangular_linear_layout_for_role(role)? {
            RectangularLinearLayout::Descriptor => {
                Ok(linear_weight_reinterpreted_as_descriptor(weight, input_width)?.dim(1)?)
            }
            RectangularLinearLayout::Transposed => {
                Ok(linear_weight_reinterpreted_as_transposed(weight, input_width)?.dim(0)?)
            }
            RectangularLinearLayout::Auto if rows == input_width => Ok(cols),
            RectangularLinearLayout::Auto if cols == input_width => Ok(rows),
            RectangularLinearLayout::Auto => Err(BackendError::RuntimeShapeMismatch(format!(
                "linear input shape {:?} is incompatible with {role} weight {} shape {:?}",
                input.shape.dims, weight.name, weight.shape.dims
            ))),
        }
    }
}

fn accumulate_linear_row(
    input_row: &[f32],
    weight: &CpuTensor,
    output: &mut [f32],
    role: &str,
    collect_diagnostics: bool,
) -> Result<()> {
    let input_width = input_row.len();
    let rows = weight.dim(0)?;
    let cols = weight.dim(1)?;
    let precision = if collect_diagnostics {
        diagnostic_linear_accumulation_precision()?
    } else {
        LinearAccumulationPrecision::F32
    };
    if rows == input_width && cols == input_width {
        let square_layout = if collect_diagnostics {
            diagnostic_square_linear_layout()?
        } else {
            SquareLinearLayout::Transposed
        };
        match square_layout {
            SquareLinearLayout::Descriptor => accumulate_descriptor_linear_row_with_precision(
                input_row,
                BorrowedLinearWeight::from_tensor(weight)?,
                output,
                precision,
            ),
            SquareLinearLayout::Transposed => accumulate_transposed_linear_row_runtime(
                input_row,
                BorrowedLinearWeight::from_tensor(weight)?,
                output,
                precision,
            )?,
        }
    } else {
        let rectangular_layout = if collect_diagnostics {
            diagnostic_rectangular_linear_layout_for_role(role)?
        } else {
            RectangularLinearLayout::Auto
        };
        match rectangular_layout {
            RectangularLinearLayout::Descriptor => {
                let descriptor_weight = borrowed_linear_weight_as_descriptor(weight, input_width)?;
                accumulate_descriptor_linear_row_with_precision(
                    input_row,
                    descriptor_weight,
                    output,
                    precision,
                );
            }
            RectangularLinearLayout::Transposed => {
                let transposed_weight = borrowed_linear_weight_as_transposed(weight, input_width)?;
                accumulate_transposed_linear_row_runtime(
                    input_row,
                    transposed_weight,
                    output,
                    precision,
                )?;
            }
            RectangularLinearLayout::Auto if rows == input_width => {
                let transposed_weight = borrowed_linear_weight_as_transposed(weight, input_width)?;
                accumulate_transposed_linear_row_runtime(
                    input_row,
                    transposed_weight,
                    output,
                    precision,
                )?;
            }
            RectangularLinearLayout::Auto if cols == input_width => {
                accumulate_transposed_linear_row_runtime(
                    input_row,
                    BorrowedLinearWeight::from_tensor(weight)?,
                    output,
                    precision,
                )?;
            }
            RectangularLinearLayout::Auto => {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "linear input width {input_width} is incompatible with {role} weight {} shape {:?}",
                    weight.name, weight.shape.dims
                )));
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn should_use_q8_0_block_dot(weight: &CpuTensor, input_width: usize) -> bool {
    let runtime_plan = ResolvedRuntimePlan::from_env().unwrap_or(ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags::from_env(),
    });
    should_use_q8_0_block_dot_with_plan(weight, input_width, &runtime_plan)
}

fn should_use_q8_0_block_dot_with_plan(
    weight: &CpuTensor,
    input_width: usize,
    runtime_plan: &ResolvedRuntimePlan,
) -> bool {
    runtime_plan.q8.block_dot
        && weight.source_type == Some(GgufTensorType::Q8_0)
        && (weight.q8_0_blocks.is_some() || q8_0_selected_packed_rows4(weight).is_some())
        && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
}

#[allow(dead_code)]
fn should_use_borrowed_q8_0_block_dot(
    weight: BorrowedLinearWeight<'_>,
    input_width: usize,
) -> bool {
    let runtime_plan = ResolvedRuntimePlan::from_env().unwrap_or(ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags::from_env(),
    });
    should_use_borrowed_q8_0_block_dot_with_plan(weight, input_width, &runtime_plan)
}

fn should_use_borrowed_q8_0_block_dot_with_plan(
    weight: BorrowedLinearWeight<'_>,
    input_width: usize,
    runtime_plan: &ResolvedRuntimePlan,
) -> bool {
    runtime_plan.q8.block_dot
        && weight.source_type == Some(GgufTensorType::Q8_0)
        && (weight.q8_0_blocks.is_some() || q8_0_selected_borrowed_packed_rows4(weight).is_some())
        && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
}

#[allow(dead_code)]
fn q8_0_block_dot_enabled() -> bool {
    // Retained Q8_0 blocks should use the quantized-input block-dot path by default;
    // keep the existing explicit dequantized-f32 escape hatch for diagnostics.
    q8_0_env_flag_enabled_default_on_fail_closed("CAMELID_Q8_0_BLOCK_DOT")
}

fn q8_0_file_reader_block_dot_enabled() -> bool {
    // Lazy/file-backed Q8 rows should also use block-dot by default. Preserving this
    // default is required because Q8 runtime/packed tensors may not have row-major
    // f32 backing for a safe generic fallback.
    q8_0_env_flag_enabled_default_on_fail_closed("CAMELID_Q8_0_FILE_READER_BLOCK_DOT")
}

#[allow(dead_code)]
fn q8_0_metal_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8")
}

#[allow(dead_code)]
fn q8_0_metal_retained_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8_RETAINED")
}

#[allow(dead_code)]
fn q8_0_hybrid_retained_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_HYBRID_Q8_RETAINED")
}

fn q8_0_metal_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var("CAMELID_TRACE_Q8_METAL")
            .map(|value| {
                let value = value.trim();
                value.eq_ignore_ascii_case("1")
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("on")
                    || value.eq_ignore_ascii_case("enabled")
            })
            .unwrap_or(false)
    })
}

fn trace_q8_0_hybrid_retained_success(cpu_rows: usize, gpu_rows: usize, blocks_per_row: usize) {
    if !q8_0_metal_trace_enabled() {
        return;
    }
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if count <= 8 || count.is_multiple_of(100) {
        eprintln!(
            "camelid_q8_metal_trace path=hybrid_retained success_count={count} cpu_rows={cpu_rows} gpu_rows={gpu_rows} blocks_per_row={blocks_per_row}"
        );
    }
}

#[allow(dead_code)]
fn q8_0_hybrid_retained_gpu_rows(output_rows: usize) -> usize {
    if output_rows < 2 {
        return 0;
    }
    if let Ok(value) = env::var("CAMELID_HYBRID_Q8_GPU_ROWS") {
        if let Ok(rows) = value.trim().parse::<usize>() {
            return rows.min(output_rows.saturating_sub(1));
        }
    }
    let percent = env::var("CAMELID_HYBRID_Q8_GPU_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10)
        .min(90);
    ((output_rows * percent).div_ceil(100))
        .max(1)
        .min(output_rows.saturating_sub(1))
}

fn auto_retain_q8_0_blocks_for_fast_local_chat(binding: &LlamaTensorBinding) -> bool {
    const DEFAULT_FAST_RETAINED_Q8_0_MAX_BYTES: usize = 6 * 1024 * 1024 * 1024;

    if env::var("CAMELID_LAZY_Q8_0_LINEAR").is_ok() {
        return false;
    }
    if q8_0_env_flag_disabled("CAMELID_RETAIN_Q8_0_BLOCKS") {
        return false;
    }
    let max_bytes = parse_byte_count_env("CAMELID_FAST_RETAINED_Q8_0_MAX_BYTES")
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_FAST_RETAINED_Q8_0_MAX_BYTES);

    let mut estimated_bytes = 0usize;
    let mut saw_q8_linear = false;
    let mut add_linear = |desc: &crate::gguf::GgufTensorDescriptor| -> Option<()> {
        let element_count = desc
            .dimensions
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))?;
        estimated_bytes = estimated_bytes.checked_add(element_count.checked_mul(4)?)?;
        if desc.tensor_type == GgufTensorType::Q8_0 {
            saw_q8_linear = true;
            let block_count = element_count.checked_add(Q8_0_BLOCK_VALUES - 1)? / Q8_0_BLOCK_VALUES;
            // Fast local chat keeps Q8_0 linear tensors as compact retained blocks and
            // executes the block-dot path directly. Do not budget a second f32 copy here;
            // the old f32+Q8 retention shape made 3B fall back to repeated lazy disk reads.
            estimated_bytes = estimated_bytes.checked_sub(element_count.checked_mul(4)?)?;
            estimated_bytes = estimated_bytes
                .checked_add(block_count.checked_mul(mem::size_of::<Q8_0Block>())?)?;
        }
        Some(())
    };

    if add_linear(&binding.token_embedding).is_none() {
        return false;
    }
    if !binding.output_is_tied_embedding && add_linear(&binding.output).is_none() {
        return false;
    }
    for layer in &binding.layers {
        for desc in [
            &layer.attention_q,
            &layer.attention_k,
            &layer.attention_v,
            &layer.attention_output,
        ] {
            if add_linear(desc).is_none() {
                return false;
            }
        }
        match &layer.ffn {
            LlamaFfnTensors::Dense { gate, up, down } => {
                for desc in [gate, up, down] {
                    if add_linear(desc).is_none() {
                        return false;
                    }
                }
            }
            LlamaFfnTensors::MoE {
                router,
                gate_experts,
                up_experts,
                down_experts,
            } => {
                for desc in std::iter::once(router)
                    .chain(gate_experts.descriptors())
                    .chain(up_experts.descriptors())
                    .chain(down_experts.descriptors())
                {
                    if add_linear(desc).is_none() {
                        return false;
                    }
                }
            }
        }
    }

    saw_q8_linear && estimated_bytes <= max_bytes
}

fn lazy_q8_0_linear_enabled() -> bool {
    match env::var("CAMELID_LAZY_Q8_0_LINEAR") {
        Ok(value)
            if value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled") =>
        {
            false
        }
        Ok(_) | Err(env::VarError::NotPresent) => true,
        Err(_) => true,
    }
}

const Q8_0_BLOCK_VALUES: usize = 32;
const X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS: usize = 1024;
const X86_Q8_PACKED_ROWS4_MATMUL_PARALLEL_MIN_GROUPS: usize = 64;
const X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK: usize = 8;
const X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS: usize = 8;

#[derive(Debug, Clone)]
struct QuantizedQ8_0Row {
    blocks: Vec<Q8_0Block>,
}

#[cfg(test)]
struct BorrowedQuantizedQ8_0Rows<'a> {
    blocks_per_row: usize,
    blocks: &'a [Q8_0Block],
}

#[cfg(test)]
impl BorrowedQuantizedQ8_0Rows<'_> {
    fn row(&self, row: usize) -> &[Q8_0Block] {
        let start = row * self.blocks_per_row;
        &self.blocks[start..start + self.blocks_per_row]
    }

    fn rows(&self) -> impl ExactSizeIterator<Item = &[Q8_0Block]> {
        self.blocks.chunks_exact(self.blocks_per_row)
    }
}

fn try_x86_q8_output_packed_rows4_matmul_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    if !runtime_plan.q8.output_packed_rows4_matmul
        || input.rank() != 2
        || input.dim(0)? <= 1
        || weight.name != "output.weight"
        || weight.source_type != Some(GgufTensorType::Q8_0)
    {
        return Ok(None);
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let Some((packed, output_width)) = q8_0_runtime_packed_projection(weight, input_width)? else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8 {
        return Ok(None);
    }

    q8_0_packed_rows4_matmul_projection(input, packed, output_width, name).map(Some)
}

fn try_x86_q8_output_decode_owner_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    if !runtime_plan.q8.output_decode_owner {
        return Ok(None);
    }
    let rows_for_telemetry = input.dim(0).unwrap_or(0);
    let input_width_for_telemetry = input.dim(1).unwrap_or(0);
    let output_width_for_telemetry = if weight.rank() == 2 {
        let weight_rows = weight.dim(0).unwrap_or(0);
        let weight_cols = weight.dim(1).unwrap_or(0);
        if weight_rows == input_width_for_telemetry {
            weight_cols
        } else if weight_cols == input_width_for_telemetry {
            weight_rows
        } else {
            0
        }
    } else {
        0
    };
    if input.rank() != 2 || rows_for_telemetry != 1 {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "shape_or_role_mismatch",
            rows_for_telemetry,
            input_width_for_telemetry,
            output_width_for_telemetry,
        );
        return Ok(None);
    }
    if weight.name != "output.weight" {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "output_weight_not_materialized",
            rows_for_telemetry,
            input_width_for_telemetry,
            output_width_for_telemetry,
        );
        return Ok(None);
    }
    if weight.source_type != Some(GgufTensorType::Q8_0) {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "source_not_q8_0",
            rows_for_telemetry,
            input_width_for_telemetry,
            output_width_for_telemetry,
        );
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "input_width_not_q8_block_multiple",
            1,
            input_width,
            output_width_for_telemetry,
        );
        return Ok(None);
    }
    let borrowed = borrowed_linear_weight_as_transposed(weight, input_width)?;
    let Some((packed, interleave)) = q8_0_selected_borrowed_packed_rows4(borrowed) else {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "missing_runtime_packed_rows4",
            1,
            input_width,
            borrowed.rows,
        );
        return Ok(None);
    };
    if interleave != Q8_0PackedRows4Interleave::I8
        || packed.rows != borrowed.rows
        || packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
        || !borrowed.rows.is_multiple_of(4)
    {
        record_q8_schedule_projection_route_denial(
            "logits",
            "x86_output_decode_owner",
            "packed_projection_shape_or_interleave_mismatch",
            1,
            input_width,
            borrowed.rows,
        );
        return Ok(None);
    }
    let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
    let quantized_input = quantize_q8_0_row(&input.data[..input_width]);
    let output = q8_0_packed_rows4_single_input_projection(
        packed,
        &quantized_input.blocks,
        borrowed.rows,
        name,
    )?;
    record_q8_schedule_projection_route_elapsed(
        "logits",
        "x86_output_decode_owner",
        name,
        1,
        input_width,
        borrowed.rows,
        telemetry_started,
    );
    Ok(Some(output))
}

fn x86_q8_attention_qkv_prefill_consumer_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER: OnceLock<bool> = OnceLock::new();
        *X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER")
        })
    }
}

fn x86_q8_attention_qkv_decode_group_chunking_enabled() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn x86_q8_attention_qkv_decode_groups_per_chunk() -> usize {
    env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn x86_q8_ffn_gate_up_decode_groups_per_chunk() -> usize {
    env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86Q8AttentionQkvRouteKind {
    Decode,
    PackedRows4Matmul,
}

struct X86Q8AttentionQkvRoute<'a> {
    q_packed: &'a Q8_0PackedRows4,
    k_packed: &'a Q8_0PackedRows4,
    v_packed: &'a Q8_0PackedRows4,
    input_width: usize,
    q_width: usize,
    k_width: usize,
    v_width: usize,
}

fn resolve_x86_q8_attention_qkv_route<'a>(
    input: &CpuTensor,
    q_weight: &'a CpuTensor,
    k_weight: &'a CpuTensor,
    v_weight: &'a CpuTensor,
    runtime_plan: &ResolvedRuntimePlan,
    route: X86Q8AttentionQkvRouteKind,
) -> Result<Option<X86Q8AttentionQkvRoute<'a>>> {
    let route_enabled = match route {
        X86Q8AttentionQkvRouteKind::Decode => runtime_plan.q8.attention_qkv_decode_consumer,
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul => {
            runtime_plan.q8.attention_qkv_packed_rows4_matmul
        }
    };
    if !route_enabled || input.rank() != 2 {
        return Ok(None);
    }

    let rows = input.dim(0)?;
    match route {
        X86Q8AttentionQkvRouteKind::Decode if rows != 1 => return Ok(None),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul if rows <= 1 => return Ok(None),
        _ => {}
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }

    let Some((q_packed, q_width)) = q8_0_runtime_packed_projection(q_weight, input_width)? else {
        return Ok(None);
    };
    let Some((k_packed, k_width)) = q8_0_runtime_packed_projection(k_weight, input_width)? else {
        return Ok(None);
    };
    let Some((v_packed, v_width)) = q8_0_runtime_packed_projection(v_weight, input_width)? else {
        return Ok(None);
    };
    if q_packed.blocks_per_row != k_packed.blocks_per_row
        || q_packed.blocks_per_row != v_packed.blocks_per_row
    {
        return Ok(None);
    }

    Ok(Some(X86Q8AttentionQkvRoute {
        q_packed,
        k_packed,
        v_packed,
        input_width,
        q_width,
        k_width,
        v_width,
    }))
}

fn try_x86_q8_attention_qkv_decode_consumer_path(
    input: &CpuTensor,
    q_weight: &CpuTensor,
    k_weight: &CpuTensor,
    v_weight: &CpuTensor,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<(CpuTensor, CpuTensor, CpuTensor)>> {
    let Some(route) = resolve_x86_q8_attention_qkv_route(
        input,
        q_weight,
        k_weight,
        v_weight,
        runtime_plan,
        X86Q8AttentionQkvRouteKind::Decode,
    )?
    else {
        return Ok(None);
    };

    let quantized_input = quantize_q8_0_row(&input.data[..route.input_width]);
    let (q, k, v) = q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
        route.q_packed,
        route.k_packed,
        route.v_packed,
        route.q_width,
        route.k_width,
        route.v_width,
        &quantized_input.blocks,
        runtime_plan.q8.attention_qkv_decode_group_chunking
            && x86_q8_attention_qkv_decode_group_chunking_enabled(),
    )?;
    Ok(Some((q, k, v)))
}

fn try_x86_q8_attention_qkv_packed_rows4_matmul_path(
    input: &CpuTensor,
    q_weight: &CpuTensor,
    k_weight: &CpuTensor,
    v_weight: &CpuTensor,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<(CpuTensor, CpuTensor, CpuTensor)>> {
    let Some(route) = resolve_x86_q8_attention_qkv_route(
        input,
        q_weight,
        k_weight,
        v_weight,
        runtime_plan,
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )?
    else {
        return Ok(None);
    };

    let (q, k, v) = with_q8_0_quantized_matmul_input_rows(
        input,
        route.q_packed.blocks_per_row,
        |rows, quantized_inputs| {
            q8_0_packed_rows4_matmul_projection_triplet_from_quantized(
                rows,
                route.q_packed,
                route.k_packed,
                route.v_packed,
                route.q_width,
                route.k_width,
                route.v_width,
                quantized_inputs,
            )
        },
    )?;
    Ok(Some((q, k, v)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86Q8FfnGateUpRouteKind {
    PackedRows4Matmul,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl X86Q8FfnGateUpRouteKind {
    fn telemetry_name(self) -> &'static str {
        match self {
            Self::PackedRows4Matmul => "packed_rows4_matmul_prefill",
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
struct X86Q8FfnGateUpRoute<'a> {
    gate_packed: &'a Q8_0PackedRows4,
    up_packed: &'a Q8_0PackedRows4,
    rows: usize,
    input_width: usize,
    output_width: usize,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn resolve_x86_q8_ffn_gate_up_route<'a>(
    input: &CpuTensor,
    gate_weight: &'a CpuTensor,
    up_weight: &'a CpuTensor,
    runtime_plan: &ResolvedRuntimePlan,
    route: X86Q8FfnGateUpRouteKind,
) -> Result<Option<X86Q8FfnGateUpRoute<'a>>> {
    let route_enabled = match route {
        X86Q8FfnGateUpRouteKind::PackedRows4Matmul => {
            runtime_plan.q8.ffn_gate_up_packed_rows4_matmul
        }
    };
    let route_name = route.telemetry_name();
    if !route_enabled || input.rank() != 2 {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            if !route_enabled {
                "plan_off"
            } else {
                "rank_mismatch"
            },
            input.dim(0).unwrap_or(0),
            input.dim(1).unwrap_or(0),
            0,
        );
        return Ok(None);
    }

    let rows = input.dim(0)?;
    match route {
        X86Q8FfnGateUpRouteKind::PackedRows4Matmul if rows <= 1 => {
            record_q8_schedule_projection_route_denial(
                "ffn_gate_up",
                route_name,
                "decode_or_empty_input",
                rows,
                input.dim(1).unwrap_or(0),
                0,
            );
            return Ok(None);
        }
        _ => {}
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "bad_input_width",
            rows,
            input_width,
            0,
        );
        return Ok(None);
    }

    let gate_width = linear_output_width(input, gate_weight, "ffn gate")?;
    let up_width = linear_output_width(input, up_weight, "ffn up")?;
    if gate_width != up_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN gate/up width mismatch: gate output {gate_width}, up output {up_width}"
        )));
    }

    let Some((gate_packed, packed_gate_width)) =
        q8_0_runtime_packed_projection(gate_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "no_runtime_packed_gate",
            rows,
            input_width,
            gate_width,
        );
        return Ok(None);
    };
    let Some((up_packed, packed_up_width)) =
        q8_0_runtime_packed_projection(up_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "no_runtime_packed_up",
            rows,
            input_width,
            up_width,
        );
        return Ok(None);
    };
    if packed_gate_width != gate_width
        || packed_up_width != up_width
        || gate_packed.rows != gate_width
        || up_packed.rows != up_width
        || gate_packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
        || up_packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
    {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "packed_shape_mismatch",
            rows,
            input_width,
            gate_width,
        );
        return Ok(None);
    }
    if gate_packed.interleave != Q8_0PackedRows4Interleave::I8
        || up_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "non_i8_interleave",
            rows,
            input_width,
            gate_width,
        );
        return Ok(None);
    }
    if gate_packed.blocks_per_row != up_packed.blocks_per_row {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            route_name,
            "gate_up_block_stride_mismatch",
            rows,
            input_width,
            gate_width,
        );
        return Ok(None);
    }

    Ok(Some(X86Q8FfnGateUpRoute {
        gate_packed,
        up_packed,
        rows,
        input_width,
        output_width: gate_width,
    }))
}

fn try_x86_q8_ffn_gate_up_decode_consumer_path(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: &str,
    gate: &mut [f32],
    up: &mut [f32],
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<(u128, u128)>> {
    if !runtime_plan.q8.ffn_gate_up_decode_consumer
        || input.rank() != 2
        || input.dim(0)? != 1
        || gate.is_empty()
        || gate.len() != up.len()
    {
        return Ok(None);
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_consumer",
            "input_width_not_q8_block_multiple",
            1,
            input_width,
            gate.len(),
        );
        return Ok(None);
    }

    let Some((gate_packed, gate_width)) = q8_0_runtime_packed_projection(gate_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_consumer",
            "missing_gate_runtime_packed_rows4",
            1,
            input_width,
            gate.len(),
        );
        return Ok(None);
    };
    let Some((up_packed, up_width)) = q8_0_runtime_packed_projection(up_weight, input_width)?
    else {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_consumer",
            "missing_up_runtime_packed_rows4",
            1,
            input_width,
            up.len(),
        );
        return Ok(None);
    };
    if gate_width != gate.len()
        || up_width != up.len()
        || gate_width != up_width
        || gate_packed.interleave != Q8_0PackedRows4Interleave::I8
        || up_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        record_q8_schedule_projection_route_denial(
            "ffn_gate_up",
            "decode_consumer",
            "packed_projection_shape_or_interleave_mismatch",
            1,
            input_width,
            gate.len(),
        );
        return Ok(None);
    }

    let started = Instant::now();
    let quantized_input = quantize_q8_0_row(&input.data[..input_width]);
    q8_0_packed_rows4_single_input_projection_pair_into_with_decode_chunking(
        gate_packed,
        up_packed,
        &quantized_input.blocks,
        gate,
        up,
        runtime_plan.q8.ffn_gate_up_decode_group_chunking,
    )?;
    let total_elapsed = started.elapsed().as_micros();
    record_q8_schedule_output_projection_route_call(
        "ffn_gate_up",
        "decode_consumer",
        Some(name),
        1,
        input_width,
        gate_width,
        total_elapsed,
    );
    let gate_elapsed = total_elapsed / 2;
    Ok(Some((gate_elapsed, total_elapsed - gate_elapsed)))
}

fn q8_0_runtime_packed_projection(
    weight: &CpuTensor,
    input_width: usize,
) -> Result<Option<(&Q8_0PackedRows4, usize)>> {
    if weight.source_type != Some(GgufTensorType::Q8_0) {
        return Ok(None);
    }
    let weight_rows = weight.dim(0)?;
    let weight_cols = weight.dim(1)?;
    let output_width = if weight_rows == input_width {
        weight_cols
    } else if weight_cols == input_width {
        weight_rows
    } else {
        return Ok(None);
    };
    let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = weight.q8_0_runtime_storage.as_ref() else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8
        || packed.rows != output_width
        || packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
        || !output_width.is_multiple_of(4)
    {
        return Ok(None);
    }
    Ok(Some((packed, output_width)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86Q8FfnDownRouteKind {
    Decode,
    PackedRows4Matmul,
    Gemm4Prefill,
    SingleOwner,
}

impl X86Q8FfnDownRouteKind {
    fn telemetry_name(self) -> &'static str {
        match self {
            Self::Decode => "x86_decode_consumer",
            Self::PackedRows4Matmul => "x86_packed_rows4_matmul",
            Self::Gemm4Prefill => "x86_gemm4_prefill",
            Self::SingleOwner => "x86_single_owner",
        }
    }
}

struct X86Q8FfnDownRoute<'a> {
    packed: &'a Q8_0PackedRows4,
    input_width: usize,
    output_width: usize,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_ffn_down_decode_reference_gate_enabled() -> bool {
    x86_q8_kernel_avx2_enabled()
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_q8_ffn_down_decode_reference_gate_enabled() -> bool {
    false
}

fn resolve_x86_q8_ffn_down_route<'a>(
    input: &CpuTensor,
    weight: &'a CpuTensor,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
    route: X86Q8FfnDownRouteKind,
) -> Result<Option<X86Q8FfnDownRoute<'a>>> {
    let route_enabled = match route {
        X86Q8FfnDownRouteKind::Decode => {
            runtime_plan.q8.ffn_down_decode_consumer
                || runtime_plan.q8.ffn_down_vnni_decode
                || x86_q8_ffn_down_decode_reference_gate_enabled()
        }
        X86Q8FfnDownRouteKind::PackedRows4Matmul => runtime_plan.q8.ffn_down_packed_rows4_matmul,
        X86Q8FfnDownRouteKind::Gemm4Prefill => {
            runtime_plan.q8.ffn_down_gemm4_prefill || runtime_plan.q8.ffn_down_amx_prefill
        }
        X86Q8FfnDownRouteKind::SingleOwner => runtime_plan.q8.ffn_down_single_owner,
    };
    let route_name = route.telemetry_name();
    let is_gemm4_prefill = matches!(route, X86Q8FfnDownRouteKind::Gemm4Prefill);
    if !route_enabled || rectangular_role != "ffn_down" || input.rank() != 2 || weight.rank() != 2 {
        if is_gemm4_prefill
            && rectangular_role == "ffn_down"
            && input.rank() == 2
            && weight.rank() == 2
        {
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES, 1);
            if !route_enabled {
                add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF, 1);
            }
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                if !route_enabled {
                    "plan_off"
                } else {
                    "role_or_rank_mismatch"
                },
                input.dim(0).unwrap_or(0),
                input.dim(1).unwrap_or(0),
                0,
            );
        }
        return Ok(None);
    }

    if is_gemm4_prefill {
        add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES, 1);
    }

    let rows = input.dim(0)?;
    match route {
        X86Q8FfnDownRouteKind::Decode if rows != 1 => return Ok(None),
        X86Q8FfnDownRouteKind::PackedRows4Matmul if rows <= 1 => return Ok(None),
        X86Q8FfnDownRouteKind::Gemm4Prefill if rows < 4 => {
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4, 1);
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "rows_lt4",
                rows,
                input.dim(1).unwrap_or(0),
                0,
            );
            return Ok(None);
        }
        X86Q8FfnDownRouteKind::SingleOwner if rows == 0 => return Ok(None),
        _ => {}
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        if is_gemm4_prefill {
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH, 1);
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "bad_input_width",
                rows,
                input_width,
                0,
            );
        }
        return Ok(None);
    }
    if weight.source_type != Some(GgufTensorType::Q8_0) {
        if is_gemm4_prefill {
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "non_q8_storage",
                rows,
                input_width,
                0,
            );
        }
        return Ok(None);
    }

    let weight_rows = weight.dim(0)?;
    let weight_cols = weight.dim(1)?;
    let output_width = if weight_rows == input_width {
        weight_cols
    } else if weight_cols == input_width {
        weight_rows
    } else {
        if is_gemm4_prefill {
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "weight_shape_mismatch",
                rows,
                input_width,
                0,
            );
        }
        return Ok(None);
    };
    let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = weight.q8_0_runtime_storage.as_ref() else {
        if is_gemm4_prefill {
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED, 1);
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "no_runtime_packed",
                rows,
                input_width,
                output_width,
            );
        }
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8 {
        if is_gemm4_prefill {
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE, 1);
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "non_i8_interleave",
                rows,
                input_width,
                output_width,
            );
        }
        return Ok(None);
    }
    if packed.rows != output_width
        || packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
        || !output_width.is_multiple_of(4)
    {
        if is_gemm4_prefill {
            record_q8_schedule_projection_route_denial(
                "ffn_down",
                route_name,
                "packed_shape_mismatch",
                rows,
                input_width,
                output_width,
            );
        }
        return Ok(None);
    }

    Ok(Some(X86Q8FfnDownRoute {
        packed,
        input_width,
        output_width,
    }))
}

fn q8_0_packed_rows4_single_input_projection(
    packed: &Q8_0PackedRows4,
    quantized_input: &[Q8_0Block],
    output_width: usize,
    name: &str,
) -> Result<CpuTensor> {
    q8_0_packed_rows4_single_input_projection_with_decode_chunking(
        packed,
        quantized_input,
        output_width,
        name,
        false,
    )
}

fn q8_0_packed_rows4_single_input_projection_with_decode_chunking(
    packed: &Q8_0PackedRows4,
    quantized_input: &[Q8_0Block],
    output_width: usize,
    name: &str,
    decode_group_chunking: bool,
) -> Result<CpuTensor> {
    let mut output = vec![0.0_f32; output_width];
    q8_0_packed_rows4_single_input_projection_into_with_decode_chunking(
        packed,
        quantized_input,
        &mut output,
        decode_group_chunking,
    )?;
    CpuTensor::from_f32(name, vec![1, output_width], output)
}

fn x86_q8_packed_rows4_serial_decode_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE")
}

fn mac_q8_ffn_down_decode_group_chunking_enabled() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING")
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

#[allow(dead_code)]
fn mac_q8_ffn_down_decode_groups_per_chunk() -> usize {
    env::var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn mac_q8_ffn_down_decode_consumer_enabled() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER")
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
fn mac_q8_ffn_down_single_projection_scheduler_counters_enabled() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS")
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

fn q8_ffn_down_decode_consumer_route_name(decode_group_chunking: bool) -> &'static str {
    if decode_group_chunking {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            "mac_decode_consumer_group_chunking"
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            "x86_decode_consumer_group_chunking"
        }
    } else if mac_q8_ffn_down_decode_consumer_enabled() {
        "mac_decode_consumer"
    } else {
        "x86_decode_consumer"
    }
}

fn x86_q8_ffn_down_decode_group_chunking_enabled() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[allow(dead_code)]
fn x86_q8_ffn_down_decode_groups_per_chunk() -> usize {
    env::var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn q8_ffn_down_decode_groups_per_chunk() -> usize {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        mac_q8_ffn_down_decode_groups_per_chunk()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        x86_q8_ffn_down_decode_groups_per_chunk()
    }
}

fn record_q8_ffn_down_vnni_decode_reject(
    counter: &AtomicU64,
    reason: &'static str,
    rows: usize,
    input_width: usize,
    output_width: usize,
) {
    add_q8_schedule_counter(counter, 1);
    record_q8_schedule_projection_route_denial(
        "ffn_down",
        "x86_vnni_decode_consumer",
        reason,
        rows,
        input_width,
        output_width,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_vnni_decode_cpu_supported() -> bool {
    x86_q8_vnni_decode_avx512_supported() || std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_q8_vnni_decode_cpu_supported() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_vnni_decode_avx512_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni")
}

fn q8_0_vnni_decode_1x64_projection(
    packed: &Q8_0VnniPacked,
    quantized_input: &[Q8_0Block],
    output_width: usize,
    name: &str,
) -> Result<CpuTensor> {
    if packed.rows != output_width
        || packed.blocks_per_row != quantized_input.len()
        || !output_width.is_multiple_of(64)
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 VNNI decode requires matching 64-aligned packed output/input, got packed rows {}, output {}, packed blocks_per_row {}, input blocks {}",
            packed.rows,
            output_width,
            packed.blocks_per_row,
            quantized_input.len()
        )));
    }

    let mut output = vec![0.0_f32; output_width];
    q8_0_vnni_decode_1x64_projection_into(packed, quantized_input, &mut output)?;
    CpuTensor::from_f32(name, vec![1, output_width], output)
}

fn q8_0_vnni_decode_1x64_projection_into(
    packed: &Q8_0VnniPacked,
    quantized_input: &[Q8_0Block],
    output: &mut [f32],
) -> Result<()> {
    let output_width = output.len();
    if packed.rows != output_width
        || packed.blocks_per_row != quantized_input.len()
        || !output_width.is_multiple_of(64)
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 VNNI decode requires matching 64-aligned packed output/input, got packed rows {}, output {}, packed blocks_per_row {}, input blocks {}",
            packed.rows,
            output_width,
            packed.blocks_per_row,
            quantized_input.len()
        )));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if x86_q8_vnni_decode_rawptr_enabled() {
            // SAFETY: runtime feature detection in `x86_q8_vnni_decode_rawptr_enabled`
            // confirms the selected x86 SIMD support, and the shape guard above proves that
            // `packed.tiles`, `quantized_input`, and `output` cover every 64-row group.
            unsafe {
                if x86_q8_vnni_decode_avx512_supported() {
                    q8_0_vnni_decode_1x64_projection_rawptr_avx512(packed, quantized_input, output);
                } else {
                    q8_0_vnni_decode_1x64_projection_rawptr_avx2(packed, quantized_input, output);
                }
            }
            return Ok(());
        }
    }

    let compute_group64 = |group64: usize, output_chunk: &mut [f32]| {
        for tile_col in 0..4 {
            let mut sums = [0.0_f32; 16];
            for (block_idx, input_block) in quantized_input.iter().enumerate() {
                let tile_idx = (group64 * 4 + tile_col) * packed.blocks_per_row + block_idx;
                let tile = &packed.tiles[tile_idx];
                let int_sums = q8_0_vnni_tile16_dot(tile, input_block);
                for (lane, sum) in sums.iter_mut().enumerate() {
                    *sum += int_sums[lane] as f32 * input_block.scale * tile.scale_f32[lane];
                }
            }
            output_chunk[tile_col * 16..tile_col * 16 + 16].copy_from_slice(&sums);
        }
    };

    if output.len() >= 1024 && rayon::current_num_threads() > 1 {
        output
            .par_chunks_exact_mut(64)
            .enumerate()
            .for_each(|(group64, output_chunk)| compute_group64(group64, output_chunk));
    } else {
        for (group64, output_chunk) in output.chunks_exact_mut(64).enumerate() {
            compute_group64(group64, output_chunk);
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn x86_q8_vnni_decode_rawptr_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR")
        && (x86_q8_vnni_decode_avx512_supported() || std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_vnni_decode_1x64_projection_rawptr_avx512(
    packed: &Q8_0VnniPacked,
    quantized_input: &[Q8_0Block],
    output: &mut [f32],
) {
    let blocks_per_row = packed.blocks_per_row;
    let input_addr = quantized_input.as_ptr() as usize;
    let tiles_addr = packed.tiles.as_ptr() as usize;

    if output.len() >= 1024 && rayon::current_num_threads() > 1 {
        output
            .par_chunks_exact_mut(64)
            .enumerate()
            .for_each(|(group64, output_chunk)| {
                // SAFETY: each parallel chunk owns one disjoint 64-wide output group.
                // Tile indexing is bounded by the caller's shape guard.
                unsafe {
                    q8_0_vnni_decode_group64_rawptr_avx512(
                        (tiles_addr as *const Q8_0VnniTile16).add(group64 * 4 * blocks_per_row),
                        blocks_per_row,
                        input_addr as *const Q8_0Block,
                        output_chunk.as_mut_ptr(),
                    );
                }
            });
    } else {
        for (group64, output_chunk) in output.chunks_exact_mut(64).enumerate() {
            // SAFETY: each serial chunk owns one disjoint 64-wide output group.
            // Tile indexing is bounded by the caller's shape guard.
            unsafe {
                q8_0_vnni_decode_group64_rawptr_avx512(
                    (tiles_addr as *const Q8_0VnniTile16).add(group64 * 4 * blocks_per_row),
                    blocks_per_row,
                    input_addr as *const Q8_0Block,
                    output_chunk.as_mut_ptr(),
                );
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_vnni_decode_group64_rawptr_avx512(
    group_tiles: *const Q8_0VnniTile16,
    blocks_per_row: usize,
    quantized_input: *const Q8_0Block,
    output: *mut f32,
) {
    use std::arch::x86_64::{
        _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    for tile_col in 0..4 {
        let mut acc = _mm512_setzero_ps();
        for block_idx in 0..blocks_per_row {
            let tile = unsafe { &*group_tiles.add(tile_col * blocks_per_row + block_idx) };
            let input_block = unsafe { &*quantized_input.add(block_idx) };
            let ints = unsafe { q8_0_vnni_tile16_i32_avx512(tile, input_block) };
            let ints = _mm512_cvtepi32_ps(ints);
            let scales = _mm512_mul_ps(
                _mm512_loadu_ps(tile.scale_f32.as_ptr()),
                _mm512_set1_ps(input_block.scale),
            );
            acc = _mm512_fmadd_ps(ints, scales, acc);
        }
        unsafe {
            _mm512_storeu_ps(output.add(tile_col * 16), acc);
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_vnni_tile16_i32_avx512(
    tile: &Q8_0VnniTile16,
    input_block: &Q8_0Block,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_setzero_si512,
        _mm512_sub_epi32,
    };

    let mut acc = _mm512_setzero_si512();
    for g in 0..8 {
        let bytes = [
            input_block.quants[g * 4] as u8 ^ 0x80,
            input_block.quants[g * 4 + 1] as u8 ^ 0x80,
            input_block.quants[g * 4 + 2] as u8 ^ 0x80,
            input_block.quants[g * 4 + 3] as u8 ^ 0x80,
        ];
        let activation = _mm512_set1_epi32(i32::from_le_bytes(bytes));
        let weights = unsafe { _mm512_loadu_si512(tile.quants.as_ptr().add(g * 64).cast()) };
        acc = _mm512_dpbusd_epi32(acc, activation, weights);
    }
    let comp = unsafe { _mm512_loadu_si512(tile.comp.as_ptr().cast()) };
    _mm512_sub_epi32(acc, comp)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_vnni_decode_1x64_projection_rawptr_avx2(
    packed: &Q8_0VnniPacked,
    quantized_input: &[Q8_0Block],
    output: &mut [f32],
) {
    let blocks_per_row = packed.blocks_per_row;
    let input_addr = quantized_input.as_ptr() as usize;
    let tiles_addr = packed.tiles.as_ptr() as usize;

    if output.len() >= 1024 && rayon::current_num_threads() > 1 {
        output
            .par_chunks_exact_mut(64)
            .enumerate()
            .for_each(|(group64, output_chunk)| {
                // SAFETY: each parallel chunk owns one disjoint 64-wide output group.
                // Tile indexing is bounded by the caller's shape guard.
                unsafe {
                    q8_0_vnni_decode_group64_rawptr_avx2(
                        (tiles_addr as *const Q8_0VnniTile16).add(group64 * 4 * blocks_per_row),
                        blocks_per_row,
                        input_addr as *const Q8_0Block,
                        output_chunk.as_mut_ptr(),
                    );
                }
            });
    } else {
        for (group64, output_chunk) in output.chunks_exact_mut(64).enumerate() {
            // SAFETY: each serial chunk owns one disjoint 64-wide output group.
            // Tile indexing is bounded by the caller's shape guard.
            unsafe {
                q8_0_vnni_decode_group64_rawptr_avx2(
                    (tiles_addr as *const Q8_0VnniTile16).add(group64 * 4 * blocks_per_row),
                    blocks_per_row,
                    input_addr as *const Q8_0Block,
                    output_chunk.as_mut_ptr(),
                );
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_vnni_decode_group64_rawptr_avx2(
    group_tiles: *const Q8_0VnniTile16,
    blocks_per_row: usize,
    quantized_input: *const Q8_0Block,
    output: *mut f32,
) {
    use std::arch::x86_64::{
        _mm_add_ps, _mm_cvtepi32_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
        _mm_storeu_ps,
    };

    for tile_col in 0..4 {
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();
        for block_idx in 0..blocks_per_row {
            let tile = unsafe { &*group_tiles.add(tile_col * blocks_per_row + block_idx) };
            let input_block = unsafe { &*quantized_input.add(block_idx) };
            let (ints0, ints1, ints2, ints3) =
                unsafe { q8_0_vnni_tile16_i32x4_avx2(tile, input_block) };
            let input_scale = _mm_set1_ps(input_block.scale);
            let scales_ptr = tile.scale_f32.as_ptr();
            let scale0 = _mm_mul_ps(_mm_loadu_ps(scales_ptr), input_scale);
            let scale1 = _mm_mul_ps(_mm_loadu_ps(unsafe { scales_ptr.add(4) }), input_scale);
            let scale2 = _mm_mul_ps(_mm_loadu_ps(unsafe { scales_ptr.add(8) }), input_scale);
            let scale3 = _mm_mul_ps(_mm_loadu_ps(unsafe { scales_ptr.add(12) }), input_scale);
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(ints0), scale0));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(ints1), scale1));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_cvtepi32_ps(ints2), scale2));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_cvtepi32_ps(ints3), scale3));
        }
        unsafe {
            let output = output.add(tile_col * 16);
            _mm_storeu_ps(output, acc0);
            _mm_storeu_ps(output.add(4), acc1);
            _mm_storeu_ps(output.add(8), acc2);
            _mm_storeu_ps(output.add(12), acc3);
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_vnni_tile16_i32x4_avx2(
    tile: &Q8_0VnniTile16,
    input_block: &Q8_0Block,
) -> (
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16, _mm256_setzero_si256,
        _mm_loadu_si128, _mm_set1_epi32,
    };

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    for g in 0..8 {
        let activation = _mm256_cvtepi8_epi16(_mm_set1_epi32(i32::from_le_bytes([
            input_block.quants[g * 4] as u8,
            input_block.quants[g * 4 + 1] as u8,
            input_block.quants[g * 4 + 2] as u8,
            input_block.quants[g * 4 + 3] as u8,
        ])));
        let base = unsafe { tile.quants.as_ptr().add(g * 64) };
        let weights0 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.cast()) });
        let weights1 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(16).cast()) });
        let weights2 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(32).cast()) });
        let weights3 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(48).cast()) });

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(activation, weights0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(activation, weights1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(activation, weights2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(activation, weights3));
    }

    let ints0 = q8_0_vnni_avx2_pair_sums_i128(acc0);
    let ints1 = q8_0_vnni_avx2_pair_sums_i128(acc1);
    let ints2 = q8_0_vnni_avx2_pair_sums_i128(acc2);
    let ints3 = q8_0_vnni_avx2_pair_sums_i128(acc3);
    (ints0, ints1, ints2, ints3)
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
fn q8_0_vnni_avx2_pair_sums_i128(acc: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::{
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm_unpacklo_epi64,
    };

    let pair_sums = _mm256_hadd_epi32(acc, acc);
    let lo = _mm256_castsi256_si128(pair_sums);
    let hi = _mm256_extracti128_si256(pair_sums, 1);
    _mm_unpacklo_epi64(lo, hi)
}

#[cfg(target_arch = "x86")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
fn q8_0_vnni_avx2_pair_sums_i128(acc: std::arch::x86::__m256i) -> std::arch::x86::__m128i {
    use std::arch::x86::{
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm_unpacklo_epi64,
    };

    let pair_sums = _mm256_hadd_epi32(acc, acc);
    let lo = _mm256_castsi256_si128(pair_sums);
    let hi = _mm256_extracti128_si256(pair_sums, 1);
    _mm_unpacklo_epi64(lo, hi)
}

fn q8_0_vnni_tile16_dot(tile: &Q8_0VnniTile16, input_block: &Q8_0Block) -> [i32; 16] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_vnni_decode_avx512_supported() {
            // SAFETY: runtime feature detection confirms the AVX512-VNNI feature set.
            return unsafe { q8_0_vnni_tile16_dot_avx512(tile, input_block) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection confirms AVX2 support. The helper only
            // reads the fixed-size tile/input arrays passed by shared references.
            return unsafe { q8_0_vnni_tile16_dot_avx2(tile, input_block) };
        }
    }
    q8_0_vnni_tile16_dot_scalar(tile, input_block)
}

fn q8_0_vnni_tile16_dot_scalar(tile: &Q8_0VnniTile16, input_block: &Q8_0Block) -> [i32; 16] {
    let mut sums = [0_i32; 16];
    for (lane, sum) in sums.iter_mut().enumerate() {
        let mut acc = 0_i32;
        for g in 0..8 {
            for r in 0..4 {
                let a = i32::from(input_block.quants[g * 4 + r]) + 128;
                let b = i32::from(tile.quants[g * 64 + lane * 4 + r]);
                acc += a * b;
            }
        }
        *sum = acc - tile.comp[lane];
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_vnni_tile16_dot_avx512(tile: &Q8_0VnniTile16, input_block: &Q8_0Block) -> [i32; 16] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_setzero_si512,
        _mm512_storeu_si512, _mm512_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_setzero_si512,
        _mm512_storeu_si512, _mm512_sub_epi32,
    };

    let mut acc = _mm512_setzero_si512();
    for g in 0..8 {
        let bytes = [
            input_block.quants[g * 4] as u8 ^ 0x80,
            input_block.quants[g * 4 + 1] as u8 ^ 0x80,
            input_block.quants[g * 4 + 2] as u8 ^ 0x80,
            input_block.quants[g * 4 + 3] as u8 ^ 0x80,
        ];
        let activation = _mm512_set1_epi32(i32::from_le_bytes(bytes));
        let weights = unsafe { _mm512_loadu_si512(tile.quants.as_ptr().add(g * 64).cast()) };
        acc = _mm512_dpbusd_epi32(acc, activation, weights);
    }
    let comp = unsafe { _mm512_loadu_si512(tile.comp.as_ptr().cast()) };
    acc = _mm512_sub_epi32(acc, comp);

    let mut lanes = [0_i32; 16];
    unsafe {
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc);
    }
    lanes
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_vnni_tile16_dot_avx2(tile: &Q8_0VnniTile16, input_block: &Q8_0Block) -> [i32; 16] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16, _mm256_setzero_si256,
        _mm_loadu_si128, _mm_set1_epi32, _mm_storeu_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16, _mm256_setzero_si256,
        _mm_loadu_si128, _mm_set1_epi32, _mm_storeu_si128,
    };

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    for g in 0..8 {
        let activation = _mm256_cvtepi8_epi16(_mm_set1_epi32(i32::from_le_bytes([
            input_block.quants[g * 4] as u8,
            input_block.quants[g * 4 + 1] as u8,
            input_block.quants[g * 4 + 2] as u8,
            input_block.quants[g * 4 + 3] as u8,
        ])));
        let base = unsafe { tile.quants.as_ptr().add(g * 64) };
        let weights0 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.cast()) });
        let weights1 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(16).cast()) });
        let weights2 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(32).cast()) });
        let weights3 = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(base.add(48).cast()) });

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(activation, weights0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(activation, weights1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(activation, weights2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(activation, weights3));
    }

    let mut lanes = [0_i32; 16];
    unsafe {
        _mm_storeu_si128(
            lanes.as_mut_ptr().cast(),
            q8_0_vnni_avx2_pair_sums_i128(acc0),
        );
    }
    unsafe {
        _mm_storeu_si128(
            lanes.as_mut_ptr().add(4).cast(),
            q8_0_vnni_avx2_pair_sums_i128(acc1),
        );
    }
    unsafe {
        _mm_storeu_si128(
            lanes.as_mut_ptr().add(8).cast(),
            q8_0_vnni_avx2_pair_sums_i128(acc2),
        );
    }
    unsafe {
        _mm_storeu_si128(
            lanes.as_mut_ptr().add(12).cast(),
            q8_0_vnni_avx2_pair_sums_i128(acc3),
        );
    }
    lanes
}

fn should_parallelize_x86_q8_packed_rows4_decode_output(output_width: usize) -> bool {
    !x86_q8_packed_rows4_serial_decode_enabled()
        && output_width >= X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS
        && rayon::current_num_threads() > 1
}

#[allow(dead_code)]
fn q8_0_packed_rows4_single_input_projection_into(
    packed: &Q8_0PackedRows4,
    quantized_input: &[Q8_0Block],
    output: &mut [f32],
) -> Result<()> {
    q8_0_packed_rows4_single_input_projection_into_with_decode_chunking(
        packed,
        quantized_input,
        output,
        false,
    )
}

fn q8_0_packed_rows4_single_input_projection_into_with_decode_chunking(
    packed: &Q8_0PackedRows4,
    quantized_input: &[Q8_0Block],
    output: &mut [f32],
    decode_group_chunking: bool,
) -> Result<()> {
    let output_width = output.len();
    let output_groups = q8_0_packed_rows4_output_groups(output_width, "decode projection")?;
    let blocks_per_row = packed.blocks_per_row;
    if packed.interleave != Q8_0PackedRows4Interleave::I8
        || packed.rows != output_width
        || quantized_input.len() != blocks_per_row
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 decode projection requires matching I8 packed output/input, got interleave {:?}, packed rows {}, output {}, packed blocks_per_row {}, input blocks {}",
            packed.interleave,
            packed.rows,
            output_width,
            blocks_per_row,
            quantized_input.len()
        )));
    }

    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled();
    let compute_group = |group_idx: usize, output_chunk: &mut [f32]| {
        let group_start = group_idx * blocks_per_row;
        let group_blocks = &packed.blocks[group_start..group_start + blocks_per_row];
        let sums = q8_0_packed_rows4_dot_i8_matmul(group_blocks, quantized_input, use_hoisted_avx2);
        output_chunk.copy_from_slice(&sums);
    };

    if output_groups > 1 && should_parallelize_x86_q8_packed_rows4_decode_output(output_width) {
        if decode_group_chunking {
            let groups_per_chunk = q8_ffn_down_decode_groups_per_chunk().min(output_groups);
            let chunk_floats = groups_per_chunk * 4;
            output.par_chunks_mut(chunk_floats).enumerate().for_each(
                |(chunk_idx, output_chunk)| {
                    let first_group_idx = chunk_idx * groups_per_chunk;
                    for (local_group_idx, output_group) in
                        output_chunk.chunks_exact_mut(4).enumerate()
                    {
                        compute_group(first_group_idx + local_group_idx, output_group);
                    }
                },
            );
        } else {
            output
                .par_chunks_mut(4)
                .enumerate()
                .for_each(|(group_idx, output_chunk)| compute_group(group_idx, output_chunk));
        }
    } else {
        for (group_idx, output_chunk) in output.chunks_exact_mut(4).enumerate() {
            compute_group(group_idx, output_chunk);
        }
    }
    Ok(())
}

fn q8_0_packed_rows4_single_input_projection_pair_into_with_decode_chunking(
    left_packed: &Q8_0PackedRows4,
    right_packed: &Q8_0PackedRows4,
    quantized_input: &[Q8_0Block],
    left_output: &mut [f32],
    right_output: &mut [f32],
    decode_group_chunking: bool,
) -> Result<()> {
    let output_width = left_output.len();
    if right_output.len() != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair decode output width mismatch: left={}, right={}",
            left_output.len(),
            right_output.len()
        )));
    }
    let output_groups = q8_0_packed_rows4_output_groups(output_width, "pair decode projection")?;
    let blocks_per_row = left_packed.blocks_per_row;
    if right_packed.blocks_per_row != blocks_per_row || quantized_input.len() != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair decode blocks_per_row mismatch: left={}, right={}, input={}",
            left_packed.blocks_per_row,
            right_packed.blocks_per_row,
            quantized_input.len()
        )));
    }
    if left_packed.interleave != Q8_0PackedRows4Interleave::I8
        || right_packed.interleave != Q8_0PackedRows4Interleave::I8
        || left_packed.rows != output_width
        || right_packed.rows != output_width
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair decode requires matching I8 packed outputs, got left {:?}/{} and right {:?}/{} for output {output_width}",
            left_packed.interleave, left_packed.rows, right_packed.interleave, right_packed.rows
        )));
    }

    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled();
    let compute_group = |group_idx: usize, left_chunk: &mut [f32], right_chunk: &mut [f32]| {
        let group_start = group_idx * blocks_per_row;
        let left_blocks = &left_packed.blocks[group_start..group_start + blocks_per_row];
        let right_blocks = &right_packed.blocks[group_start..group_start + blocks_per_row];
        let left_sums =
            q8_0_packed_rows4_dot_i8_matmul(left_blocks, quantized_input, use_hoisted_avx2);
        let right_sums =
            q8_0_packed_rows4_dot_i8_matmul(right_blocks, quantized_input, use_hoisted_avx2);
        left_chunk.copy_from_slice(&left_sums);
        right_chunk.copy_from_slice(&right_sums);
    };

    if output_groups > 1 && should_parallelize_x86_q8_packed_rows4_decode_output(output_width) {
        if decode_group_chunking {
            let groups_per_chunk = x86_q8_ffn_gate_up_decode_groups_per_chunk().min(output_groups);
            let chunk_floats = groups_per_chunk * 4;
            left_output
                .par_chunks_mut(chunk_floats)
                .zip(right_output.par_chunks_mut(chunk_floats))
                .enumerate()
                .for_each(|(chunk_idx, (left_chunk, right_chunk))| {
                    let first_group_idx = chunk_idx * groups_per_chunk;
                    for (local_group_idx, (left_group, right_group)) in left_chunk
                        .chunks_exact_mut(4)
                        .zip(right_chunk.chunks_exact_mut(4))
                        .enumerate()
                    {
                        compute_group(first_group_idx + local_group_idx, left_group, right_group);
                    }
                });
        } else {
            left_output
                .par_chunks_mut(4)
                .zip(right_output.par_chunks_mut(4))
                .enumerate()
                .for_each(|(group_idx, (left_chunk, right_chunk))| {
                    compute_group(group_idx, left_chunk, right_chunk)
                });
        }
    } else {
        for (group_idx, (left_chunk, right_chunk)) in left_output
            .chunks_exact_mut(4)
            .zip(right_output.chunks_exact_mut(4))
            .enumerate()
            .take(output_groups)
        {
            compute_group(group_idx, left_chunk, right_chunk);
        }
    }
    Ok(())
}

fn q8_0_packed_rows4_single_input_projection_pair_activated_from_quantized(
    gate_packed: &Q8_0PackedRows4,
    up_packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
    order: FfnGateUpOrder,
    quantized_input: &[Q8_0Block],
    use_paired_dot: bool,
) -> Result<CpuTensor> {
    let output_groups = q8_0_packed_rows4_output_groups(output_width, "pair activated decode")?;
    let blocks_per_row = gate_packed.blocks_per_row;
    if up_packed.blocks_per_row != blocks_per_row || quantized_input.len() != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair activated decode blocks_per_row mismatch: gate={}, up={}, input={}",
            gate_packed.blocks_per_row,
            up_packed.blocks_per_row,
            quantized_input.len()
        )));
    }
    if gate_packed.interleave != Q8_0PackedRows4Interleave::I8
        || up_packed.interleave != Q8_0PackedRows4Interleave::I8
        || gate_packed.rows != output_width
        || up_packed.rows != output_width
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair activated decode requires matching I8 packed outputs, got gate {:?}/{} and up {:?}/{} for output {output_width}",
            gate_packed.interleave, gate_packed.rows, up_packed.interleave, up_packed.rows
        )));
    }

    let mut output = vec![0.0_f32; output_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled();
    let compute_group = |group_idx: usize, output_chunk: &mut [f32]| {
        let group_start = group_idx * blocks_per_row;
        let gate_blocks = &gate_packed.blocks[group_start..group_start + blocks_per_row];
        let up_blocks = &up_packed.blocks[group_start..group_start + blocks_per_row];
        let (gate_sums, up_sums) = if use_paired_dot {
            q8_0_packed_rows4_dot_i8_matmul_pair(
                gate_blocks,
                up_blocks,
                quantized_input,
                use_hoisted_avx2,
            )
        } else {
            (
                q8_0_packed_rows4_dot_i8_matmul(gate_blocks, quantized_input, use_hoisted_avx2),
                q8_0_packed_rows4_dot_i8_matmul(up_blocks, quantized_input, use_hoisted_avx2),
            )
        };
        for lane in 0..4 {
            output_chunk[lane] = apply_ffn_gate_up_order(gate_sums[lane], up_sums[lane], order);
        }
    };

    if output_groups > 1 && should_parallelize_x86_q8_packed_rows4_decode_output(output_width) {
        output
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(group_idx, output_chunk)| compute_group(group_idx, output_chunk));
    } else {
        for (group_idx, output_chunk) in output.chunks_exact_mut(4).enumerate().take(output_groups)
        {
            compute_group(group_idx, output_chunk);
        }
    }
    CpuTensor::from_f32(name, vec![1, output_width], output)
}

#[allow(clippy::too_many_arguments)]
fn q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
    q_packed: &Q8_0PackedRows4,
    k_packed: &Q8_0PackedRows4,
    v_packed: &Q8_0PackedRows4,
    q_width: usize,
    k_width: usize,
    v_width: usize,
    quantized_input: &[Q8_0Block],
    decode_group_chunking: bool,
) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
    let blocks_per_row = q_packed.blocks_per_row;
    if k_packed.blocks_per_row != blocks_per_row
        || v_packed.blocks_per_row != blocks_per_row
        || quantized_input.len() != blocks_per_row
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 QKV decode blocks_per_row mismatch: q={}, k={}, v={}, input={}",
            q_packed.blocks_per_row,
            k_packed.blocks_per_row,
            v_packed.blocks_per_row,
            quantized_input.len()
        )));
    }
    if q_packed.interleave != Q8_0PackedRows4Interleave::I8
        || k_packed.interleave != Q8_0PackedRows4Interleave::I8
        || v_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        return Err(BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 QKV decode requires I8 interleave".to_string(),
        ));
    }
    if q_packed.rows != q_width || k_packed.rows != k_width || v_packed.rows != v_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 QKV decode output width mismatch: q packed/requested={}/{}, k packed/requested={}/{}, v packed/requested={}/{}",
            q_packed.rows, q_width, k_packed.rows, k_width, v_packed.rows, v_width
        )));
    }

    let q_groups = q8_0_packed_rows4_output_groups(q_width, "QKV decode q projection")?;
    let k_groups = q8_0_packed_rows4_output_groups(k_width, "QKV decode k projection")?;
    let v_groups = q8_0_packed_rows4_output_groups(v_width, "QKV decode v projection")?;
    let mut q_output = vec![0.0_f32; q_width];
    let mut k_output = vec![0.0_f32; k_width];
    let mut v_output = vec![0.0_f32; v_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled();

    if q_width == k_width
        && q_width == v_width
        && q_groups > 1
        && should_parallelize_x86_q8_packed_rows4_decode_output(q_width)
    {
        if decode_group_chunking {
            let groups_per_chunk = x86_q8_attention_qkv_decode_groups_per_chunk().min(q_groups);
            let chunk_floats = groups_per_chunk * 4;
            q_output
                .par_chunks_mut(chunk_floats)
                .zip(k_output.par_chunks_mut(chunk_floats))
                .zip(v_output.par_chunks_mut(chunk_floats))
                .enumerate()
                .for_each(|(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
                    let first_group_idx = chunk_idx * groups_per_chunk;
                    for (local_group_idx, ((q_group, k_group), v_group)) in q_chunk
                        .chunks_exact_mut(4)
                        .zip(k_chunk.chunks_exact_mut(4))
                        .zip(v_chunk.chunks_exact_mut(4))
                        .enumerate()
                    {
                        let group_start = (first_group_idx + local_group_idx) * blocks_per_row;
                        q_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                            &q_packed.blocks[group_start..group_start + blocks_per_row],
                            quantized_input,
                            use_hoisted_avx2,
                        ));
                        k_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                            &k_packed.blocks[group_start..group_start + blocks_per_row],
                            quantized_input,
                            use_hoisted_avx2,
                        ));
                        v_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                            &v_packed.blocks[group_start..group_start + blocks_per_row],
                            quantized_input,
                            use_hoisted_avx2,
                        ));
                    }
                });
        } else {
            q_output
                .par_chunks_mut(4)
                .zip(k_output.par_chunks_mut(4))
                .zip(v_output.par_chunks_mut(4))
                .enumerate()
                .for_each(|(group_idx, ((q_chunk, k_chunk), v_chunk))| {
                    let group_start = group_idx * blocks_per_row;
                    q_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &q_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_input,
                        use_hoisted_avx2,
                    ));
                    k_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &k_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_input,
                        use_hoisted_avx2,
                    ));
                    v_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &v_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_input,
                        use_hoisted_avx2,
                    ));
                });
        }
    } else if q_width == k_width && q_width == v_width {
        for (group_idx, ((q_chunk, k_chunk), v_chunk)) in q_output
            .chunks_exact_mut(4)
            .zip(k_output.chunks_exact_mut(4))
            .zip(v_output.chunks_exact_mut(4))
            .enumerate()
        {
            let group_start = group_idx * blocks_per_row;
            q_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &q_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
            k_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &k_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
            v_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &v_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
        }
    } else {
        for (group_idx, q_chunk) in q_output.chunks_exact_mut(4).enumerate().take(q_groups) {
            let group_start = group_idx * blocks_per_row;
            q_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &q_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
        }
        for (group_idx, k_chunk) in k_output.chunks_exact_mut(4).enumerate().take(k_groups) {
            let group_start = group_idx * blocks_per_row;
            k_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &k_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
        }
        for (group_idx, v_chunk) in v_output.chunks_exact_mut(4).enumerate().take(v_groups) {
            let group_start = group_idx * blocks_per_row;
            v_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                &v_packed.blocks[group_start..group_start + blocks_per_row],
                quantized_input,
                use_hoisted_avx2,
            ));
        }
    }

    Ok((
        CpuTensor::from_f32(
            "attention_q_x86_q8_qkv_consumer",
            vec![1, q_width],
            q_output,
        )?,
        CpuTensor::from_f32(
            "attention_k_x86_q8_qkv_consumer",
            vec![1, k_width],
            k_output,
        )?,
        CpuTensor::from_f32(
            "attention_v_x86_q8_qkv_consumer",
            vec![1, v_width],
            v_output,
        )?,
    ))
}

fn q8_0_packed_rows4_matmul_projection(
    input: &CpuTensor,
    packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
) -> Result<CpuTensor> {
    with_q8_0_quantized_matmul_input_rows(input, packed.blocks_per_row, |rows, quantized_inputs| {
        q8_0_packed_rows4_matmul_projection_from_quantized(
            rows,
            packed,
            output_width,
            name,
            quantized_inputs,
        )
    })
}

fn q8_0_packed_rows4_output_groups(output_width: usize, context: &str) -> Result<usize> {
    if !output_width.is_multiple_of(4) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} output width {output_width} is not divisible by 4"
        )));
    }
    Ok(output_width / 4)
}

fn should_parallelize_q8_packed_rows4_matmul(total_output_groups: usize) -> bool {
    total_output_groups >= X86_Q8_PACKED_ROWS4_MATMUL_PARALLEL_MIN_GROUPS
        && rayon::current_num_threads() > 1
}

fn x86_q8_ffn_down_gemm4_row_group_min_input_groups() -> usize {
    #[cfg(test)]
    {
        env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS)
    }
    #[cfg(not(test))]
    {
        static X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS_ONCE: OnceLock<usize> =
            OnceLock::new();
        *X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS_ONCE.get_or_init(|| {
            env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS)
        })
    }
}

fn should_use_x86_q8_ffn_down_gemm4_row_group_schedule(enabled: bool, input_groups: usize) -> bool {
    enabled
        && rayon::current_num_threads() > 1
        && input_groups >= x86_q8_ffn_down_gemm4_row_group_min_input_groups()
}

fn x86_q8_packed_rows4_matmul_groups_per_chunk() -> usize {
    #[cfg(test)]
    {
        env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK)
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_MATMUL_CHUNK_GROUPS: OnceLock<usize> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_MATMUL_CHUNK_GROUPS.get_or_init(|| {
            env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK)
        })
    }
}

fn q8_packed_rows4_matmul_parallel_chunk_floats(total_output_groups: usize) -> usize {
    let groups_per_chunk =
        total_output_groups.clamp(1, x86_q8_packed_rows4_matmul_groups_per_chunk());
    groups_per_chunk * 4
}

fn x86_q8_parallel_matmul_input_quantize_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PARALLEL_INPUT_QUANTIZE: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PARALLEL_INPUT_QUANTIZE.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE")
        })
    }
}

#[cfg(test)]
fn q8_0_quantized_matmul_input_rows(
    input: &CpuTensor,
    blocks_per_row: usize,
) -> Result<Vec<Q8_0Block>> {
    let rows = input.dim(0)?;
    let mut quantized_inputs = Vec::with_capacity(rows * blocks_per_row);
    q8_0_fill_quantized_matmul_input_rows(input, blocks_per_row, &mut quantized_inputs)?;
    Ok(quantized_inputs)
}

fn with_q8_0_quantized_matmul_input_rows<T>(
    input: &CpuTensor,
    blocks_per_row: usize,
    f: impl FnOnce(usize, &[Q8_0Block]) -> Result<T>,
) -> Result<T> {
    with_q8_0_file_reader_quantized_inputs(|quantized_inputs| {
        let rows = q8_0_fill_quantized_matmul_input_rows(input, blocks_per_row, quantized_inputs)?;
        f(rows, quantized_inputs)
    })
}

fn q8_0_fill_quantized_matmul_input_rows(
    input: &CpuTensor,
    blocks_per_row: usize,
    quantized_inputs: &mut Vec<Q8_0Block>,
) -> Result<usize> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    let expected_blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    if blocks_per_row != expected_blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 matmul expected {expected_blocks_per_row} input blocks per row, got {blocks_per_row}"
        )));
    }

    quantized_inputs.clear();
    quantized_inputs.reserve(rows * blocks_per_row);
    if x86_q8_parallel_matmul_input_quantize_enabled()
        && rows >= 8
        && rayon::current_num_threads() > 1
    {
        let rows_per_chunk = rows.clamp(1, 8);
        let floats_per_chunk = rows_per_chunk * input_width;
        let quantized_chunks: Vec<Vec<Q8_0Block>> = input
            .data
            .par_chunks(floats_per_chunk)
            .map(|input_rows| {
                let mut local_blocks = Vec::with_capacity(input_rows.len() / Q8_0_BLOCK_VALUES);
                for row in input_rows.chunks_exact(input_width) {
                    quantize_q8_0_blocks_into(row, &mut local_blocks);
                }
                local_blocks
            })
            .collect();
        for chunk in quantized_chunks {
            quantized_inputs.extend(chunk);
        }
    } else {
        for row in input.data.chunks_exact(input_width) {
            quantize_q8_0_blocks_into(row, quantized_inputs);
        }
    }
    Ok(rows)
}

fn q8_0_packed_rows4_matmul_projection_from_quantized(
    rows: usize,
    packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
    quantized_inputs: &[Q8_0Block],
) -> Result<CpuTensor> {
    let blocks_per_row = packed.blocks_per_row;
    let expected_quantized_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 matmul input block count overflow".to_string(),
        )
    })?;
    if quantized_inputs.len() != expected_quantized_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 matmul expected {expected_quantized_blocks} quantized input blocks, got {}",
            quantized_inputs.len()
        )));
    }

    if packed.interleave != Q8_0PackedRows4Interleave::I8 || packed.rows != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 matmul requires matching I8 packed output, got interleave {:?}, packed rows {}, output {}",
            packed.interleave, packed.rows, output_width
        )));
    }

    let output_groups_per_row = q8_0_packed_rows4_output_groups(output_width, "matmul projection")?;
    let total_output_groups = rows * output_groups_per_row;
    let mut output = vec![0.0_f32; rows * output_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_hoist_enabled();
    if should_parallelize_q8_packed_rows4_matmul(total_output_groups) {
        let chunk_floats = q8_packed_rows4_matmul_parallel_chunk_floats(total_output_groups);
        output
            .par_chunks_mut(chunk_floats)
            .enumerate()
            .for_each(|(chunk_idx, output_chunk)| {
                let first_group_idx = chunk_idx * (chunk_floats / 4);
                for (local_group_idx, output_group) in output_chunk.chunks_exact_mut(4).enumerate()
                {
                    let flat_group_idx = first_group_idx + local_group_idx;
                    let row_idx = flat_group_idx / output_groups_per_row;
                    let group_idx = flat_group_idx % output_groups_per_row;
                    let input_start = row_idx * blocks_per_row;
                    let group_start = group_idx * blocks_per_row;
                    let group_blocks = &packed.blocks[group_start..group_start + blocks_per_row];
                    let sums = q8_0_packed_rows4_dot_i8_matmul(
                        group_blocks,
                        &quantized_inputs[input_start..input_start + blocks_per_row],
                        use_hoisted_avx2,
                    );
                    output_group.copy_from_slice(&sums);
                }
            });
    } else {
        for row_idx in 0..rows {
            let input_start = row_idx * blocks_per_row;
            let quantized_row = &quantized_inputs[input_start..input_start + blocks_per_row];
            let output_start = row_idx * output_width;
            for (group_idx, output_chunk) in output[output_start..output_start + output_width]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                let group_blocks = &packed.blocks[group_start..group_start + blocks_per_row];
                let sums =
                    q8_0_packed_rows4_dot_i8_matmul(group_blocks, quantized_row, use_hoisted_avx2);
                output_chunk.copy_from_slice(&sums);
            }
        }
    }

    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn q8_0_packed_rows4_gemm4_accumulate_block(
    input_block: &Q8_0PackedRows4Block,
    weight_block: &Q8_0PackedRows4Block,
    sums: &mut [[f32; 4]; 4],
    use_avx2: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if use_avx2 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection confirms AVX2 support; pointers reference
            // complete packed rows4/I8 Q8_0 blocks and a contiguous 4x4 f32 accumulator.
            unsafe {
                q8_0_packed_rows4_gemm4_accumulate_block_avx2(
                    input_block.quants.as_ptr(),
                    input_block.scales.as_ptr(),
                    weight_block.quants.as_ptr(),
                    weight_block.scales.as_ptr(),
                    sums.as_mut_ptr().cast::<f32>(),
                );
            }
            return;
        }
    }

    let _ = use_avx2;
    let int_sums = q8_0_packed_rows4_gemm4_block_scalar(input_block, weight_block);
    for input_lane in 0..4 {
        let input_scale = input_block.scales[input_lane];
        for output_lane in 0..4 {
            sums[input_lane][output_lane] += int_sums[input_lane][output_lane] as f32
                * weight_block.scales[output_lane]
                * input_scale;
        }
    }
}

#[inline(always)]
fn q8_0_packed_rows4_prefetch_block(block: &Q8_0PackedRows4Block) {
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(block.quants.as_ptr(), std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(block.quants.as_ptr(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = block;
}

fn q8_0_packed_rows4_gemm4_block_scalar(
    input_block: &Q8_0PackedRows4Block,
    weight_block: &Q8_0PackedRows4Block,
) -> [[i32; 4]; 4] {
    let mut sums = [[0_i32; 4]; 4];
    for chunk in 0..4 {
        for k in 0..8 {
            for (input_lane, row_sums) in sums.iter_mut().enumerate() {
                let input = input_block.quants[chunk * 32 + input_lane * 8 + k] as i32;
                for (output_lane, sum) in row_sums.iter_mut().enumerate() {
                    let weight = weight_block.quants[chunk * 32 + output_lane * 8 + k] as i32;
                    *sum += input * weight;
                }
            }
        }
    }
    sums
}

fn run_q8_0_packed_rows4_prefill_gemm4_kernel_row_group_parallel(
    packed_weight: &Q8_0PackedRows4,
    packed_inputs: &[Q8_0PackedRows4Block],
    input_groups: usize,
    output: &mut [f32],
    use_avx2: bool,
) {
    let rows = packed_weight.rows;
    let blocks_per_row = packed_weight.blocks_per_row;
    debug_assert_eq!(packed_inputs.len(), input_groups * blocks_per_row);
    output
        .par_chunks_mut(4 * rows)
        .take(input_groups)
        .enumerate()
        .for_each(|(input_group, group_output)| {
            let input_blocks =
                &packed_inputs[input_group * blocks_per_row..(input_group + 1) * blocks_per_row];
            let (row0, rest) = group_output.split_at_mut(rows);
            let (row1, rest) = rest.split_at_mut(rows);
            let (row2, row3) = rest.split_at_mut(rows);
            for output_group in 0..rows / 4 {
                let weight_group = &packed_weight.blocks
                    [output_group * blocks_per_row..(output_group + 1) * blocks_per_row];
                let mut sums = [[0.0_f32; 4]; 4];
                for (block_idx, (input_block, weight_block)) in
                    input_blocks.iter().zip(weight_group).enumerate()
                {
                    if block_idx + 1 < blocks_per_row {
                        q8_0_packed_rows4_prefetch_block(&input_blocks[block_idx + 1]);
                        q8_0_packed_rows4_prefetch_block(&weight_group[block_idx + 1]);
                    }
                    q8_0_packed_rows4_gemm4_accumulate_block(
                        input_block,
                        weight_block,
                        &mut sums,
                        use_avx2,
                    );
                }
                let start = output_group * 4;
                row0[start..start + 4].copy_from_slice(&sums[0]);
                row1[start..start + 4].copy_from_slice(&sums[1]);
                row2[start..start + 4].copy_from_slice(&sums[2]);
                row3[start..start + 4].copy_from_slice(&sums[3]);
            }
        });
}

fn run_q8_0_packed_rows4_prefill_gemm4_kernel(
    packed_weight: &Q8_0PackedRows4,
    packed_inputs: &[Q8_0PackedRows4Block],
    input_groups: usize,
    output: &mut [f32],
    use_avx2: bool,
) {
    let rows = packed_weight.rows;
    let blocks_per_row = packed_weight.blocks_per_row;
    debug_assert_eq!(packed_inputs.len(), input_groups * blocks_per_row);
    for input_group in 0..input_groups {
        let input_blocks =
            &packed_inputs[input_group * blocks_per_row..(input_group + 1) * blocks_per_row];
        let group_output = &mut output[input_group * 4 * rows..(input_group + 1) * 4 * rows];
        let (row0, rest) = group_output.split_at_mut(rows);
        let (row1, rest) = rest.split_at_mut(rows);
        let (row2, row3) = rest.split_at_mut(rows);
        row0.par_chunks_mut(4)
            .zip(row1.par_chunks_mut(4))
            .zip(row2.par_chunks_mut(4))
            .zip(row3.par_chunks_mut(4))
            .enumerate()
            .for_each(
                |(output_group, (((row0_chunk, row1_chunk), row2_chunk), row3_chunk))| {
                    let weight_group = &packed_weight.blocks
                        [output_group * blocks_per_row..(output_group + 1) * blocks_per_row];
                    let mut sums = [[0.0_f32; 4]; 4];
                    for (block_idx, (input_block, weight_block)) in
                        input_blocks.iter().zip(weight_group).enumerate()
                    {
                        if block_idx + 1 < blocks_per_row {
                            q8_0_packed_rows4_prefetch_block(&input_blocks[block_idx + 1]);
                            q8_0_packed_rows4_prefetch_block(&weight_group[block_idx + 1]);
                        }
                        q8_0_packed_rows4_gemm4_accumulate_block(
                            input_block,
                            weight_block,
                            &mut sums,
                            use_avx2,
                        );
                    }
                    row0_chunk.copy_from_slice(&sums[0]);
                    row1_chunk.copy_from_slice(&sums[1]);
                    row2_chunk.copy_from_slice(&sums[2]);
                    row3_chunk.copy_from_slice(&sums[3]);
                },
            );
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn try_q8_0_packed_rows4_amx_prefill_projection(
    input: &CpuTensor,
    packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
) -> Result<Option<CpuTensor>> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let Some(amx_blocks) = packed.amx_blocks.as_ref() else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8
        || packed.rows != output_width
        || packed.blocks_per_row != blocks_per_row
        || !output_width.is_multiple_of(16)
        || rows < 4
    {
        return Ok(None);
    }
    // SAFETY: FFI call only checks CPU/XSTATE support and does not dereference Rust pointers.
    if unsafe { camelid_x86_q8_amx_supported() } == 0 {
        return Ok(None);
    }

    let packed_rows = rows / 4 * 4;
    let mut output = vec![0.0_f32; rows * output_width];
    Q8_0_PREFILL_PACKED_INPUTS.with(|cell| {
        let mut packed_inputs = cell.borrow_mut();
        packed_inputs.clear();
        quantize_pack_q8_0_rows4_i8_direct_into(
            &input.data[..packed_rows * input_width],
            packed_rows,
            input_width,
            blocks_per_row,
            &mut packed_inputs,
        );

        let output_tiles = output_width / 16;
        for row_chunk_start in (0..packed_rows).step_by(16) {
            let chunk_rows = (packed_rows - row_chunk_start).min(16);
            let input_group_start = (row_chunk_start / 4) * blocks_per_row;
            let input_ptr = packed_inputs[input_group_start..].as_ptr();
            for output_tile in 0..output_tiles {
                let weight_start = output_tile * blocks_per_row;
                let output_start = row_chunk_start * output_width + output_tile * 16;
                // SAFETY: `packed_inputs` contains complete rows4/I8 blocks for this row
                // chunk, `amx_blocks` contains one 16-row AMX tile per output tile/block,
                // and `output_start` points at a 16-wide tile in each row-strided output.
                unsafe {
                    camelid_q8_0_amx_compute_tile16(
                        input_ptr,
                        blocks_per_row,
                        chunk_rows,
                        amx_blocks[weight_start..].as_ptr(),
                        output[output_start..].as_mut_ptr(),
                        output_width,
                    );
                }
            }
        }
        packed_inputs.clear();
        cap_q8_0_file_reader_scratch(&mut packed_inputs, 0);
    });

    let tail_rows = rows - packed_rows;
    if tail_rows > 0 {
        with_q8_0_file_reader_quantized_inputs(|quantized_inputs| {
            quantized_inputs.clear();
            quantized_inputs.reserve(tail_rows * blocks_per_row);
            for row in input.data[packed_rows * input_width..].chunks_exact(input_width) {
                quantize_q8_0_blocks_into(row, quantized_inputs);
            }
            for tail_row in 0..tail_rows {
                let input_start = tail_row * blocks_per_row;
                let output_start = (packed_rows + tail_row) * output_width;
                accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                    &quantized_inputs[input_start..input_start + blocks_per_row],
                    packed,
                    Q8_0PackedRows4Interleave::I8,
                    &mut output[output_start..output_start + output_width],
                );
            }
            Ok(())
        })?;
    }

    CpuTensor::from_f32(name, vec![rows, output_width], output).map(Some)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn try_q8_0_packed_rows4_amx_prefill_projection(
    _input: &CpuTensor,
    _packed: &Q8_0PackedRows4,
    _output_width: usize,
    _name: &str,
) -> Result<Option<CpuTensor>> {
    Ok(None)
}

fn q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
    input: &CpuTensor,
    packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
    row_group_schedule: bool,
    use_avx2: bool,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    if blocks_per_row != packed.blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 gemm4 expected {} input blocks per row, got {blocks_per_row}",
            packed.blocks_per_row
        )));
    }
    if packed.interleave != Q8_0PackedRows4Interleave::I8 || packed.rows != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 gemm4 requires matching I8 packed output, got interleave {:?}, packed rows {}, output {}",
            packed.interleave, packed.rows, output_width
        )));
    }
    q8_0_packed_rows4_output_groups(output_width, "gemm4 projection")?;

    let packed_rows = rows / 4 * 4;
    if packed_rows == 0 {
        return q8_0_packed_rows4_matmul_projection(input, packed, output_width, name);
    }

    let mut output = vec![0.0_f32; rows * output_width];
    Q8_0_PREFILL_PACKED_INPUTS.with(|cell| {
        let mut packed_inputs = cell.borrow_mut();
        packed_inputs.clear();
        quantize_pack_q8_0_rows4_i8_direct_into(
            &input.data[..packed_rows * input_width],
            packed_rows,
            input_width,
            blocks_per_row,
            &mut packed_inputs,
        );
        let input_groups = packed_rows / 4;
        if should_use_x86_q8_ffn_down_gemm4_row_group_schedule(row_group_schedule, input_groups) {
            run_q8_0_packed_rows4_prefill_gemm4_kernel_row_group_parallel(
                packed,
                &packed_inputs,
                input_groups,
                &mut output,
                use_avx2,
            );
        } else {
            run_q8_0_packed_rows4_prefill_gemm4_kernel(
                packed,
                &packed_inputs,
                input_groups,
                &mut output,
                use_avx2,
            );
        }
        packed_inputs.clear();
        cap_q8_0_file_reader_scratch(&mut packed_inputs, 0);
    });

    let tail_rows = rows - packed_rows;
    if tail_rows > 0 {
        with_q8_0_file_reader_quantized_inputs(|quantized_inputs| {
            quantized_inputs.clear();
            quantized_inputs.reserve(tail_rows * blocks_per_row);
            for row in input.data[packed_rows * input_width..].chunks_exact(input_width) {
                quantize_q8_0_blocks_into(row, quantized_inputs);
            }
            for tail_row in 0..tail_rows {
                let input_start = tail_row * blocks_per_row;
                let output_start = (packed_rows + tail_row) * output_width;
                accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                    &quantized_inputs[input_start..input_start + blocks_per_row],
                    packed,
                    Q8_0PackedRows4Interleave::I8,
                    &mut output[output_start..output_start + output_width],
                );
            }
            Ok(())
        })?;
    }

    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_packed_rows4_gemm4_accumulate_block_avx2(
    input_packed: *const i8,
    input_scales: *const f32,
    weight_packed: *const i8,
    weight_scales: *const f32,
    sums: *mut f32,
) {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_broadcastsi128_si256, _mm256_castsi256_si128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sign_epi8, _mm_add_ps, _mm_cvtepi32_ps,
        _mm_hadd_epi32, _mm_loadl_epi64, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_si128,
        _mm_storeu_ps, _mm_unpacklo_epi64,
    };

    #[inline(always)]
    unsafe fn reduce_pairs_i32x8_to_i32x4(
        acc: std::arch::x86_64::__m256i,
    ) -> std::arch::x86_64::__m128i {
        let zero = _mm_setzero_si128();
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256::<1>(acc);
        let lo = _mm_hadd_epi32(lo, zero);
        let hi = _mm_hadd_epi32(hi, zero);
        _mm_unpacklo_epi64(lo, hi)
    }

    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    for chunk in 0..4usize {
        let chunk_start = chunk * 32;
        let weight32 = unsafe { _mm256_loadu_si256(weight_packed.add(chunk_start).cast()) };
        let abs_weight = _mm256_sign_epi8(weight32, weight32);

        let lane0 = unsafe { _mm_loadl_epi64(input_packed.add(chunk_start).cast()) };
        let input0 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(lane0, lane0));
        let signed0 = _mm256_sign_epi8(input0, weight32);
        acc0 = _mm256_add_epi32(
            acc0,
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed0), ones),
        );

        let lane1 = unsafe { _mm_loadl_epi64(input_packed.add(chunk_start + 8).cast()) };
        let input1 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(lane1, lane1));
        let signed1 = _mm256_sign_epi8(input1, weight32);
        acc1 = _mm256_add_epi32(
            acc1,
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed1), ones),
        );

        let lane2 = unsafe { _mm_loadl_epi64(input_packed.add(chunk_start + 16).cast()) };
        let input2 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(lane2, lane2));
        let signed2 = _mm256_sign_epi8(input2, weight32);
        acc2 = _mm256_add_epi32(
            acc2,
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed2), ones),
        );

        let lane3 = unsafe { _mm_loadl_epi64(input_packed.add(chunk_start + 24).cast()) };
        let input3 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(lane3, lane3));
        let signed3 = _mm256_sign_epi8(input3, weight32);
        acc3 = _mm256_add_epi32(
            acc3,
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed3), ones),
        );
    }

    let weight_scale = unsafe { _mm_loadu_ps(weight_scales) };

    let values0 = _mm_cvtepi32_ps(unsafe { reduce_pairs_i32x8_to_i32x4(acc0) });
    let scaled0 = _mm_mul_ps(
        _mm_mul_ps(values0, weight_scale),
        _mm_set1_ps(unsafe { *input_scales.add(0) }),
    );
    let sum0 = unsafe { _mm_loadu_ps(sums.add(0)) };
    unsafe { _mm_storeu_ps(sums.add(0), _mm_add_ps(sum0, scaled0)) };

    let values1 = _mm_cvtepi32_ps(unsafe { reduce_pairs_i32x8_to_i32x4(acc1) });
    let scaled1 = _mm_mul_ps(
        _mm_mul_ps(values1, weight_scale),
        _mm_set1_ps(unsafe { *input_scales.add(1) }),
    );
    let sum1 = unsafe { _mm_loadu_ps(sums.add(4)) };
    unsafe { _mm_storeu_ps(sums.add(4), _mm_add_ps(sum1, scaled1)) };

    let values2 = _mm_cvtepi32_ps(unsafe { reduce_pairs_i32x8_to_i32x4(acc2) });
    let scaled2 = _mm_mul_ps(
        _mm_mul_ps(values2, weight_scale),
        _mm_set1_ps(unsafe { *input_scales.add(2) }),
    );
    let sum2 = unsafe { _mm_loadu_ps(sums.add(8)) };
    unsafe { _mm_storeu_ps(sums.add(8), _mm_add_ps(sum2, scaled2)) };

    let values3 = _mm_cvtepi32_ps(unsafe { reduce_pairs_i32x8_to_i32x4(acc3) });
    let scaled3 = _mm_mul_ps(
        _mm_mul_ps(values3, weight_scale),
        _mm_set1_ps(unsafe { *input_scales.add(3) }),
    );
    let sum3 = unsafe { _mm_loadu_ps(sums.add(12)) };
    unsafe { _mm_storeu_ps(sums.add(12), _mm_add_ps(sum3, scaled3)) };
}

#[allow(dead_code, clippy::too_many_arguments)]
fn q8_0_packed_rows4_matmul_projection_pair_from_quantized(
    rows: usize,
    left_packed: &Q8_0PackedRows4,
    right_packed: &Q8_0PackedRows4,
    left_output_width: usize,
    right_output_width: usize,
    left_name: &str,
    right_name: &str,
    quantized_inputs: &[Q8_0Block],
) -> Result<(CpuTensor, CpuTensor)> {
    let blocks_per_row = left_packed.blocks_per_row;
    if right_packed.blocks_per_row != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair matmul blocks_per_row mismatch: left={}, right={}",
            left_packed.blocks_per_row, right_packed.blocks_per_row
        )));
    }
    if left_packed.interleave != Q8_0PackedRows4Interleave::I8
        || right_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        return Err(BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 pair matmul requires I8 interleave".to_string(),
        ));
    }
    let expected_quantized_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 pair matmul input block count overflow".to_string(),
        )
    })?;
    if quantized_inputs.len() != expected_quantized_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair matmul expected {expected_quantized_blocks} quantized input blocks, got {}",
            quantized_inputs.len()
        )));
    }
    if left_packed.rows != left_output_width || right_packed.rows != right_output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 pair matmul output width mismatch: left packed rows={}, left requested={}, right packed rows={}, right requested={}",
            left_packed.rows, left_output_width, right_packed.rows, right_output_width
        )));
    }

    let left_output_groups_per_row =
        q8_0_packed_rows4_output_groups(left_output_width, "pair matmul left projection")?;
    let right_output_groups_per_row =
        q8_0_packed_rows4_output_groups(right_output_width, "pair matmul right projection")?;
    let mut left_output = vec![0.0_f32; rows * left_output_width];
    let mut right_output = vec![0.0_f32; rows * right_output_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_hoist_enabled();

    let total_left_output_groups = rows * left_output_groups_per_row;
    if left_output_width == right_output_width
        && should_parallelize_q8_packed_rows4_matmul(total_left_output_groups)
    {
        let chunk_floats = q8_packed_rows4_matmul_parallel_chunk_floats(total_left_output_groups);
        left_output
            .par_chunks_mut(chunk_floats)
            .zip(right_output.par_chunks_mut(chunk_floats))
            .enumerate()
            .for_each(|(chunk_idx, (left_chunk, right_chunk))| {
                let first_group_idx = chunk_idx * (chunk_floats / 4);
                for (local_group_idx, (left_group, right_group)) in left_chunk
                    .chunks_exact_mut(4)
                    .zip(right_chunk.chunks_exact_mut(4))
                    .enumerate()
                {
                    let flat_group_idx = first_group_idx + local_group_idx;
                    let row_idx = flat_group_idx / left_output_groups_per_row;
                    let group_idx = flat_group_idx % left_output_groups_per_row;
                    let input_start = row_idx * blocks_per_row;
                    let group_start = group_idx * blocks_per_row;
                    let quantized_row =
                        &quantized_inputs[input_start..input_start + blocks_per_row];
                    let left_blocks =
                        &left_packed.blocks[group_start..group_start + blocks_per_row];
                    let right_blocks =
                        &right_packed.blocks[group_start..group_start + blocks_per_row];
                    let left_sums = q8_0_packed_rows4_dot_i8_matmul(
                        left_blocks,
                        quantized_row,
                        use_hoisted_avx2,
                    );
                    let right_sums = q8_0_packed_rows4_dot_i8_matmul(
                        right_blocks,
                        quantized_row,
                        use_hoisted_avx2,
                    );
                    left_group.copy_from_slice(&left_sums);
                    right_group.copy_from_slice(&right_sums);
                }
            });
    } else {
        for row_idx in 0..rows {
            let input_start = row_idx * blocks_per_row;
            let quantized_row = &quantized_inputs[input_start..input_start + blocks_per_row];
            let left_output_start = row_idx * left_output_width;
            for (group_idx, output_chunk) in left_output
                [left_output_start..left_output_start + left_output_width]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                let group_blocks = &left_packed.blocks[group_start..group_start + blocks_per_row];
                let sums =
                    q8_0_packed_rows4_dot_i8_matmul(group_blocks, quantized_row, use_hoisted_avx2);
                output_chunk.copy_from_slice(&sums);
            }
            let right_output_start = row_idx * right_output_width;
            for (group_idx, output_chunk) in right_output
                [right_output_start..right_output_start + right_output_groups_per_row * 4]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                let group_blocks = &right_packed.blocks[group_start..group_start + blocks_per_row];
                let sums =
                    q8_0_packed_rows4_dot_i8_matmul(group_blocks, quantized_row, use_hoisted_avx2);
                output_chunk.copy_from_slice(&sums);
            }
        }
    }

    Ok((
        CpuTensor::from_f32(left_name, vec![rows, left_output_width], left_output)?,
        CpuTensor::from_f32(right_name, vec![rows, right_output_width], right_output)?,
    ))
}

#[cfg(any(test, target_arch = "x86", target_arch = "x86_64"))]
fn validate_q8_0_packed_rows4_pair_matmul_inputs(
    rows: usize,
    left_packed: &Q8_0PackedRows4,
    right_packed: &Q8_0PackedRows4,
    output_width: usize,
    quantized_inputs: &[Q8_0Block],
    context: &str,
) -> Result<(usize, usize)> {
    let blocks_per_row = left_packed.blocks_per_row;
    if right_packed.blocks_per_row != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} blocks_per_row mismatch: left={}, right={}",
            left_packed.blocks_per_row, right_packed.blocks_per_row
        )));
    }
    if left_packed.interleave != Q8_0PackedRows4Interleave::I8
        || right_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} requires I8 interleave"
        )));
    }
    let expected_quantized_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} input block count overflow"
        ))
    })?;
    if quantized_inputs.len() != expected_quantized_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} expected {expected_quantized_blocks} quantized input blocks, got {}",
            quantized_inputs.len()
        )));
    }
    if left_packed.rows != output_width || right_packed.rows != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 {context} output width mismatch: left packed rows={}, requested={}, right packed rows={}",
            left_packed.rows, output_width, right_packed.rows
        )));
    }
    let output_groups_per_row =
        q8_0_packed_rows4_output_groups(output_width, "pair matmul fused projection")?;
    Ok((blocks_per_row, output_groups_per_row))
}

#[cfg(any(test, target_arch = "x86", target_arch = "x86_64"))]
fn q8_0_packed_rows4_matmul_projection_pair_activated_from_quantized(
    rows: usize,
    gate_packed: &Q8_0PackedRows4,
    up_packed: &Q8_0PackedRows4,
    output_width: usize,
    name: &str,
    order: FfnGateUpOrder,
    quantized_inputs: &[Q8_0Block],
) -> Result<CpuTensor> {
    let (blocks_per_row, output_groups_per_row) = validate_q8_0_packed_rows4_pair_matmul_inputs(
        rows,
        gate_packed,
        up_packed,
        output_width,
        quantized_inputs,
        "fused gate/up matmul activation",
    )?;

    let mut output = vec![0.0_f32; rows * output_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_hoist_enabled();
    let total_output_groups = rows * output_groups_per_row;
    let compute_group = |flat_group_idx: usize, output_group: &mut [f32]| {
        let row_idx = flat_group_idx / output_groups_per_row;
        let group_idx = flat_group_idx % output_groups_per_row;
        let input_start = row_idx * blocks_per_row;
        let group_start = group_idx * blocks_per_row;
        let quantized_row = &quantized_inputs[input_start..input_start + blocks_per_row];
        let gate_sums = q8_0_packed_rows4_dot_i8_matmul(
            &gate_packed.blocks[group_start..group_start + blocks_per_row],
            quantized_row,
            use_hoisted_avx2,
        );
        let up_sums = q8_0_packed_rows4_dot_i8_matmul(
            &up_packed.blocks[group_start..group_start + blocks_per_row],
            quantized_row,
            use_hoisted_avx2,
        );
        for lane in 0..4 {
            output_group[lane] = apply_ffn_gate_up_order(gate_sums[lane], up_sums[lane], order);
        }
    };

    if should_parallelize_q8_packed_rows4_matmul(total_output_groups) {
        let chunk_floats = q8_packed_rows4_matmul_parallel_chunk_floats(total_output_groups);
        output
            .par_chunks_mut(chunk_floats)
            .enumerate()
            .for_each(|(chunk_idx, output_chunk)| {
                let first_group_idx = chunk_idx * (chunk_floats / 4);
                for (local_group_idx, output_group) in output_chunk.chunks_exact_mut(4).enumerate()
                {
                    compute_group(first_group_idx + local_group_idx, output_group);
                }
            });
    } else {
        for (flat_group_idx, output_group) in output.chunks_exact_mut(4).enumerate() {
            compute_group(flat_group_idx, output_group);
        }
    }

    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

#[allow(clippy::too_many_arguments)]
fn q8_0_packed_rows4_matmul_projection_triplet_from_quantized(
    rows: usize,
    q_packed: &Q8_0PackedRows4,
    k_packed: &Q8_0PackedRows4,
    v_packed: &Q8_0PackedRows4,
    q_width: usize,
    k_width: usize,
    v_width: usize,
    quantized_inputs: &[Q8_0Block],
) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
    let blocks_per_row = q_packed.blocks_per_row;
    if k_packed.blocks_per_row != blocks_per_row || v_packed.blocks_per_row != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 QKV matmul blocks_per_row mismatch: q={}, k={}, v={}",
            q_packed.blocks_per_row, k_packed.blocks_per_row, v_packed.blocks_per_row
        )));
    }
    if q_packed.interleave != Q8_0PackedRows4Interleave::I8
        || k_packed.interleave != Q8_0PackedRows4Interleave::I8
        || v_packed.interleave != Q8_0PackedRows4Interleave::I8
    {
        return Err(BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 QKV matmul requires I8 interleave".to_string(),
        ));
    }
    let expected_quantized_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "Q8_0 packed rows4 QKV matmul input block count overflow".to_string(),
        )
    })?;
    if quantized_inputs.len() != expected_quantized_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 QKV matmul expected {expected_quantized_blocks} quantized input blocks, got {}",
            quantized_inputs.len()
        )));
    }
    if q_packed.rows != q_width || k_packed.rows != k_width || v_packed.rows != v_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Q8_0 packed rows4 QKV matmul output width mismatch: q packed/requested={}/{}, k packed/requested={}/{}, v packed/requested={}/{}",
            q_packed.rows, q_width, k_packed.rows, k_width, v_packed.rows, v_width
        )));
    }

    let q_groups_per_row = q8_0_packed_rows4_output_groups(q_width, "QKV q projection")?;
    let k_groups_per_row = q8_0_packed_rows4_output_groups(k_width, "QKV k projection")?;
    let v_groups_per_row = q8_0_packed_rows4_output_groups(v_width, "QKV v projection")?;
    let mut q_output = vec![0.0_f32; rows * q_width];
    let mut k_output = vec![0.0_f32; rows * k_width];
    let mut v_output = vec![0.0_f32; rows * v_width];
    let use_hoisted_avx2 = x86_q8_packed_rows4_avx2_dot_hoist_enabled();

    let total_q_output_groups = rows * q_groups_per_row;
    if q_width == k_width
        && q_width == v_width
        && should_parallelize_q8_packed_rows4_matmul(total_q_output_groups)
    {
        let chunk_floats = q8_packed_rows4_matmul_parallel_chunk_floats(total_q_output_groups);
        q_output
            .par_chunks_mut(chunk_floats)
            .zip(k_output.par_chunks_mut(chunk_floats))
            .zip(v_output.par_chunks_mut(chunk_floats))
            .enumerate()
            .for_each(|(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
                let first_group_idx = chunk_idx * (chunk_floats / 4);
                for (local_group_idx, ((q_group, k_group), v_group)) in q_chunk
                    .chunks_exact_mut(4)
                    .zip(k_chunk.chunks_exact_mut(4))
                    .zip(v_chunk.chunks_exact_mut(4))
                    .enumerate()
                {
                    let flat_group_idx = first_group_idx + local_group_idx;
                    let row_idx = flat_group_idx / q_groups_per_row;
                    let group_idx = flat_group_idx % q_groups_per_row;
                    let input_start = row_idx * blocks_per_row;
                    let group_start = group_idx * blocks_per_row;
                    let quantized_row =
                        &quantized_inputs[input_start..input_start + blocks_per_row];
                    q_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &q_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_row,
                        use_hoisted_avx2,
                    ));
                    k_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &k_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_row,
                        use_hoisted_avx2,
                    ));
                    v_group.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                        &v_packed.blocks[group_start..group_start + blocks_per_row],
                        quantized_row,
                        use_hoisted_avx2,
                    ));
                }
            });
    } else {
        for row_idx in 0..rows {
            let input_start = row_idx * blocks_per_row;
            let quantized_row = &quantized_inputs[input_start..input_start + blocks_per_row];
            let q_output_start = row_idx * q_width;
            for (group_idx, output_chunk) in q_output
                [q_output_start..q_output_start + q_groups_per_row * 4]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                output_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                    &q_packed.blocks[group_start..group_start + blocks_per_row],
                    quantized_row,
                    use_hoisted_avx2,
                ));
            }
            let k_output_start = row_idx * k_width;
            for (group_idx, output_chunk) in k_output
                [k_output_start..k_output_start + k_groups_per_row * 4]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                output_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                    &k_packed.blocks[group_start..group_start + blocks_per_row],
                    quantized_row,
                    use_hoisted_avx2,
                ));
            }
            let v_output_start = row_idx * v_width;
            for (group_idx, output_chunk) in v_output
                [v_output_start..v_output_start + v_groups_per_row * 4]
                .chunks_exact_mut(4)
                .enumerate()
            {
                let group_start = group_idx * blocks_per_row;
                output_chunk.copy_from_slice(&q8_0_packed_rows4_dot_i8_matmul(
                    &v_packed.blocks[group_start..group_start + blocks_per_row],
                    quantized_row,
                    use_hoisted_avx2,
                ));
            }
        }
    }

    Ok((
        CpuTensor::from_f32(
            "attention_q_x86_q8_qkv_packed_rows4_matmul",
            vec![rows, q_width],
            q_output,
        )?,
        CpuTensor::from_f32(
            "attention_k_x86_q8_qkv_packed_rows4_matmul",
            vec![rows, k_width],
            k_output,
        )?,
        CpuTensor::from_f32(
            "attention_v_x86_q8_qkv_packed_rows4_matmul",
            vec![rows, v_width],
            v_output,
        )?,
    ))
}

fn try_x86_q8_attention_output_decode_consumer_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    if !runtime_plan.q8.attention_output_decode_consumer
        || rectangular_role != "linear"
        || input.rank() != 2
        || input.dim(0)? != 1
        || !weight.name.ends_with(".attn_output.weight")
    {
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let Some((packed, output_width)) = q8_0_runtime_packed_projection(weight, input_width)? else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8 {
        return Ok(None);
    }

    let quantized_input = quantize_q8_0_row(&input.data[..input_width]);
    q8_0_packed_rows4_single_input_projection(packed, &quantized_input.blocks, output_width, name)
        .map(Some)
}

fn try_x86_q8_attention_output_packed_rows4_matmul_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    if !runtime_plan.q8.attention_output_packed_rows4_matmul
        || rectangular_role != "linear"
        || input.rank() != 2
        || input.dim(0)? <= 1
        || !weight.name.ends_with(".attn_output.weight")
    {
        return Ok(None);
    }
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let Some((packed, output_width)) = q8_0_runtime_packed_projection(weight, input_width)? else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8 {
        return Ok(None);
    }

    q8_0_packed_rows4_matmul_projection(input, packed, output_width, name).map(Some)
}

fn try_x86_q8_attention_projection_decode_consumer_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    if !runtime_plan.q8.attention_projection_decode_consumer
        || !matches!(
            rectangular_role,
            "attention_q"
                | "attention_k"
                | "attention_v"
                | "attention q"
                | "attention k"
                | "attention v"
        )
        || input.rank() != 2
        || input.dim(0)? != 1
        || weight.source_type != Some(GgufTensorType::Q8_0)
    {
        return Ok(None);
    }

    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let weight_rows = weight.dim(0)?;
    let weight_cols = weight.dim(1)?;
    let output_width = if weight_rows == input_width {
        weight_cols
    } else if weight_cols == input_width {
        weight_rows
    } else {
        return Ok(None);
    };
    let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = weight.q8_0_runtime_storage.as_ref() else {
        return Ok(None);
    };
    if packed.interleave != Q8_0PackedRows4Interleave::I8
        || packed.rows != output_width
        || packed.blocks_per_row != input_width / Q8_0_BLOCK_VALUES
        || !output_width.is_multiple_of(4)
    {
        return Ok(None);
    }

    let quantized_input = quantize_q8_0_row(&input.data[..input_width]);
    q8_0_packed_rows4_single_input_projection(packed, &quantized_input.blocks, output_width, name)
        .map(Some)
}

fn try_x86_q8_ffn_down_packed_rows4_matmul_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    let Some(route) = resolve_x86_q8_ffn_down_route(
        input,
        weight,
        rectangular_role,
        runtime_plan,
        X86Q8FfnDownRouteKind::PackedRows4Matmul,
    )?
    else {
        return Ok(None);
    };

    let rows = input.dim(0)?;
    let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
    let output =
        q8_0_packed_rows4_matmul_projection(input, route.packed, route.output_width, name)?;
    record_q8_schedule_projection_route_elapsed(
        "ffn_down",
        "x86_packed_rows4_matmul",
        name,
        rows,
        route.input_width,
        route.output_width,
        telemetry_started,
    );
    Ok(Some(output))
}

fn try_x86_q8_ffn_down_gemm4_prefill_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    let Some(route) = resolve_x86_q8_ffn_down_route(
        input,
        weight,
        rectangular_role,
        runtime_plan,
        X86Q8FfnDownRouteKind::Gemm4Prefill,
    )?
    else {
        return Ok(None);
    };

    let rows = input.dim(0)?;
    let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
    let (output, route_name) = if runtime_plan.q8.ffn_down_amx_prefill {
        if let Some(output) = try_q8_0_packed_rows4_amx_prefill_projection(
            input,
            route.packed,
            route.output_width,
            name,
        )? {
            (output, "x86_amx_prefill")
        } else if !runtime_plan.q8.ffn_down_gemm4_prefill {
            return Ok(None);
        } else {
            let output = q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
                input,
                route.packed,
                route.output_width,
                name,
                runtime_plan.q8.ffn_down_gemm4_row_group_schedule,
                runtime_plan.q8.ffn_down_gemm4_avx2,
            )?;
            let route_name = if runtime_plan.q8.ffn_down_gemm4_row_group_schedule {
                "x86_gemm4_prefill_row_group"
            } else {
                "x86_gemm4_prefill"
            };
            (output, route_name)
        }
    } else {
        let output = q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
            input,
            route.packed,
            route.output_width,
            name,
            runtime_plan.q8.ffn_down_gemm4_row_group_schedule,
            runtime_plan.q8.ffn_down_gemm4_avx2,
        )?;
        let route_name = if runtime_plan.q8.ffn_down_gemm4_row_group_schedule {
            "x86_gemm4_prefill_row_group"
        } else {
            "x86_gemm4_prefill"
        };
        (output, route_name)
    };
    record_q8_schedule_projection_route_elapsed(
        "ffn_down",
        route_name,
        name,
        rows,
        route.input_width,
        route.output_width,
        telemetry_started,
    );
    Ok(Some(output))
}

fn try_x86_q8_ffn_down_single_owner_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    let Some(route) = resolve_x86_q8_ffn_down_route(
        input,
        weight,
        rectangular_role,
        runtime_plan,
        X86Q8FfnDownRouteKind::SingleOwner,
    )?
    else {
        return Ok(None);
    };

    let rows = input.dim(0)?;
    let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
    let output = if rows == 1 {
        let quantized_input = quantize_q8_0_row(&input.data[..route.input_width]);
        q8_0_packed_rows4_single_input_projection(
            route.packed,
            &quantized_input.blocks,
            route.output_width,
            name,
        )?
    } else {
        q8_0_packed_rows4_matmul_projection(input, route.packed, route.output_width, name)?
    };
    let route_name = if rows == 1 {
        "x86_single_owner_decode"
    } else {
        "x86_single_owner_prefill"
    };
    record_q8_schedule_projection_route_elapsed(
        "ffn_down",
        route_name,
        name,
        rows,
        route.input_width,
        route.output_width,
        telemetry_started,
    );
    Ok(Some(output))
}

fn try_x86_q8_ffn_down_decode_consumer_path(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: &str,
    rectangular_role: &str,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<Option<CpuTensor>> {
    let rows_for_telemetry = input.dim(0).unwrap_or(0);
    let input_width_for_telemetry = input.dim(1).unwrap_or(0);
    let weight_rows_for_telemetry = weight.dim(0).unwrap_or(0);
    let weight_cols_for_telemetry = weight.dim(1).unwrap_or(0);
    let output_width_for_telemetry = if weight_rows_for_telemetry == input_width_for_telemetry {
        weight_cols_for_telemetry
    } else if weight_cols_for_telemetry == input_width_for_telemetry {
        weight_rows_for_telemetry
    } else {
        0
    };
    if q8_schedule_telemetry_enabled() {
        add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES, 1);
        if !runtime_plan.q8.ffn_down_vnni_decode {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF,
                "gate_off",
                rows_for_telemetry,
                input_width_for_telemetry,
                output_width_for_telemetry,
            );
        } else if rectangular_role != "ffn_down" || input.rank() != 2 || weight.rank() != 2 {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE,
                "shape_or_role_mismatch",
                rows_for_telemetry,
                input_width_for_telemetry,
                output_width_for_telemetry,
            );
        }
    }
    let Some(route) = resolve_x86_q8_ffn_down_route(
        input,
        weight,
        rectangular_role,
        runtime_plan,
        X86Q8FfnDownRouteKind::Decode,
    )?
    else {
        return Ok(None);
    };

    if runtime_plan.q8.ffn_down_vnni_decode {
        if !x86_q8_vnni_decode_cpu_supported() {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE,
                "cpu_feature_missing",
                1,
                route.input_width,
                route.output_width,
            );
        } else if !route.input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH,
                "bad_input_width",
                1,
                route.input_width,
                route.output_width,
            );
        } else if !route.output_width.is_multiple_of(64) {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH,
                "bad_output_width",
                1,
                route.input_width,
                route.output_width,
            );
        } else if let Some(vnni_packed) = route.packed.vnni_packed.as_ref() {
            let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
            let quantize_started = q8_schedule_telemetry_enabled().then(Instant::now);
            let quantized_input = quantize_q8_0_row(&input.data[..route.input_width]);
            if let Some(started) = quantize_started {
                add_q8_schedule_counter(
                    &Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US,
                    started.elapsed().as_micros() as u64,
                );
            }
            let kernel_started = q8_schedule_telemetry_enabled().then(Instant::now);
            let output = q8_0_vnni_decode_1x64_projection(
                vnni_packed,
                &quantized_input.blocks,
                route.output_width,
                name,
            )?;
            if let Some(started) = kernel_started {
                add_q8_schedule_counter(
                    &Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US,
                    started.elapsed().as_micros() as u64,
                );
            }
            add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN, 1);
            record_q8_schedule_projection_route_elapsed(
                "ffn_down",
                "x86_vnni_decode_consumer",
                name,
                1,
                route.input_width,
                route.output_width,
                telemetry_started,
            );
            return Ok(Some(output));
        } else {
            record_q8_ffn_down_vnni_decode_reject(
                &Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK,
                "no_vnni_pack",
                1,
                route.input_width,
                route.output_width,
            );
        }
    }

    let telemetry_started = q8_schedule_telemetry_enabled().then(Instant::now);
    add_q8_schedule_counter(&Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN, 1);
    let quantized_input = quantize_q8_0_row(&input.data[..route.input_width]);
    let decode_group_chunking = mac_q8_ffn_down_decode_group_chunking_enabled()
        || x86_q8_ffn_down_decode_group_chunking_enabled();
    let output = q8_0_packed_rows4_single_input_projection_with_decode_chunking(
        route.packed,
        &quantized_input.blocks,
        route.output_width,
        name,
        decode_group_chunking,
    )?;
    let route_name = q8_ffn_down_decode_consumer_route_name(decode_group_chunking);
    record_q8_schedule_projection_route_elapsed(
        "ffn_down",
        route_name,
        name,
        1,
        route.input_width,
        route.output_width,
        telemetry_started,
    );
    Ok(Some(output))
}

#[allow(dead_code)]
fn matmul_rhs_transposed_q8_0_block_dot(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let runtime_plan = ResolvedRuntimePlan::from_env()?;
    matmul_rhs_transposed_q8_0_block_dot_with_plan(input, weight, name, &runtime_plan)
}

fn matmul_rhs_transposed_q8_0_block_dot_with_plan(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<CpuTensor> {
    let name = name.into();
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    let output_width = weight.dim(0)?;
    let rhs_k = weight.dim(1)?;
    if input_width != rhs_k {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 block-dot shape mismatch: lhs {:?}, rhs {:?}",
            input.shape.dims, weight.shape.dims
        )));
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let expected_blocks = output_width * blocks_per_row;
    if let Some(weight_blocks) = weight.q8_0_blocks.as_ref() {
        if weight_blocks.len() != expected_blocks {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "q8_0 block-dot expected {expected_blocks} blocks for weight {} shape {:?}, got {}",
                weight.name,
                weight.shape.dims,
                weight_blocks.len()
            )));
        }
    } else if q8_0_selected_packed_rows4(weight)
        .filter(|(packed, _)| {
            packed.rows == output_width && packed.blocks_per_row == blocks_per_row
        })
        .is_none()
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 block-dot requested for {} without q8_0 blocks or matching packed rows4",
            weight.name
        )));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if mac_q8_prefill_i8mm_enabled() && mac_q8_prefill_i8mm_row_threshold_met(rows) {
        if let Some((packed, Q8_0PackedRows4Interleave::I8)) = q8_0_selected_packed_rows4(weight) {
            if packed.rows == output_width && packed.blocks_per_row == blocks_per_row {
                return matmul_rhs_transposed_q8_0_packed_rows4_prefill_i8mm(
                    input,
                    packed,
                    output_width,
                    q8_schedule_role_for_output_name(&name),
                    name,
                );
            }
        }
    }

    let mut output = vec![0.0_f32; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        let quantized_input =
            quantize_q8_0_row(&input.data[input_start..input_start + input_width]);
        let out_start = row * output_width;
        let output_row = &mut output[out_start..out_start + output_width];
        if let Some((packed, interleave)) = q8_0_selected_packed_rows4(weight) {
            if packed.rows == output_width && packed.blocks_per_row == blocks_per_row {
                accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                    &quantized_input.blocks,
                    packed,
                    interleave,
                    output_row,
                );
                continue;
            }
        }
        let weight_blocks = weight
            .q8_0_blocks
            .as_ref()
            .expect("q8_0 block-dot precondition checked");
        if runtime_plan.q8.metal_retained {
            let weight_bytes = q8_0_blocks_as_bytes(weight_blocks);
            if with_q8_0_block_scales_and_quants(
                &quantized_input.blocks,
                |input_scales, input_quants| {
                    metal::try_q8_0_block_linear_row(
                        input_scales,
                        input_quants,
                        weight_bytes,
                        output_width,
                        blocks_per_row,
                        output_row,
                    )
                },
            ) {
                continue;
            }
        }
        if should_parallelize_q8_0_file_reader_output(output_width) {
            output_row
                .par_iter_mut()
                .enumerate()
                .for_each(|(output_idx, out_value)| {
                    let weight_start = output_idx * blocks_per_row;
                    *out_value = q8_0_dot_rows(
                        &weight_blocks[weight_start..weight_start + blocks_per_row],
                        &quantized_input.blocks,
                    );
                });
        } else {
            for (output_idx, out_value) in output_row.iter_mut().enumerate() {
                let weight_start = output_idx * blocks_per_row;
                *out_value = q8_0_dot_rows(
                    &weight_blocks[weight_start..weight_start + blocks_per_row],
                    &quantized_input.blocks,
                );
            }
        }
    }
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn quantize_q8_0_row(input: &[f32]) -> QuantizedQ8_0Row {
    QuantizedQ8_0Row {
        blocks: quantize_q8_0_blocks(input),
    }
}

#[cfg(test)]
fn quantize_q8_0_rows_into<'a>(
    input: &CpuTensor,
    input_width: usize,
    blocks: &'a mut Vec<Q8_0Block>,
) -> Result<BorrowedQuantizedQ8_0Rows<'a>> {
    let rows = input.dim(0)?;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    blocks.clear();
    blocks.reserve(rows * blocks_per_row);
    for row in input.data.chunks_exact(input_width) {
        quantize_q8_0_blocks_into(row, blocks);
    }
    Ok(BorrowedQuantizedQ8_0Rows {
        blocks_per_row,
        blocks,
    })
}

fn quantize_q8_0_blocks(input: &[f32]) -> Vec<Q8_0Block> {
    let mut blocks = Vec::with_capacity(input.len() / Q8_0_BLOCK_VALUES);
    quantize_q8_0_blocks_into(input, &mut blocks);
    blocks
}

fn quantize_q8_0_blocks_into(input: &[f32], blocks: &mut Vec<Q8_0Block>) {
    debug_assert!(input.len().is_multiple_of(Q8_0_BLOCK_VALUES));
    blocks.extend(
        input
            .chunks_exact(Q8_0_BLOCK_VALUES)
            .map(quantize_q8_0_block),
    );
}

fn quantize_q8_0_block(block: &[f32]) -> Q8_0Block {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_VALUES);
    let max_abs = block
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let unrounded_scale = max_abs / 127.0;
    let scale_bits = f32_to_f16_bits(unrounded_scale);
    let scale = f16_bits_to_f32(scale_bits);
    let inv_scale = if unrounded_scale == 0.0 {
        0.0
    } else {
        1.0 / unrounded_scale
    };
    let mut quants = [0_i8; Q8_0_BLOCK_VALUES];
    for (idx, value) in block.iter().enumerate() {
        quants[idx] = (value * inv_scale).round().clamp(-128.0, 127.0) as i8;
    }
    Q8_0Block { scale, quants }
}

fn q8_0_dot_rows(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if aarch64_dotprod_enabled() {
            // SAFETY: runtime feature detection confirms dot-product support; the slice
            // iterator only passes complete Q8_0 blocks.
            return unsafe { q8_0_dot_rows_dotprod(weight, input) };
        }
    }

    weight
        .iter()
        .zip(input)
        .map(|(weight_block, input_block)| {
            let int_sum =
                q8_0_block_int_dot_horizontal_sum(&weight_block.quants, &input_block.quants);
            int_sum as f32 * weight_block.scale * input_block.scale
        })
        .sum()
}

fn q8_0_two_dot_rows(
    first_weight: &[Q8_0Block],
    second_weight: &[Q8_0Block],
    input: &[Q8_0Block],
) -> (f32, f32) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if aarch64_dotprod_enabled() {
            // SAFETY: runtime feature detection confirms dot-product support; the slice
            // iterator only passes complete Q8_0 blocks.
            return unsafe { q8_0_two_dot_rows_dotprod(first_weight, second_weight, input) };
        }
    }

    let mut first_sum = 0.0_f32;
    let mut second_sum = 0.0_f32;
    for ((first_block, second_block), input_block) in
        first_weight.iter().zip(second_weight).zip(input)
    {
        let first_int_sum =
            q8_0_block_int_dot_horizontal_sum(&first_block.quants, &input_block.quants);
        let second_int_sum =
            q8_0_block_int_dot_horizontal_sum(&second_block.quants, &input_block.quants);
        first_sum += first_int_sum as f32 * first_block.scale * input_block.scale;
        second_sum += second_int_sum as f32 * second_block.scale * input_block.scale;
    }
    (first_sum, second_sum)
}

fn q8_0_block_scales_and_quants(blocks: &[Q8_0Block]) -> (Vec<f32>, Vec<i8>) {
    let mut scales = Vec::with_capacity(blocks.len());
    let mut quants = Vec::with_capacity(blocks.len() * Q8_0_BLOCK_VALUES);
    for block in blocks {
        scales.push(block.scale);
        quants.extend_from_slice(&block.quants);
    }
    (scales, quants)
}

fn with_q8_0_block_scales_and_quants<T>(
    blocks: &[Q8_0Block],
    f: impl FnOnce(&[f32], &[i8]) -> T,
) -> T {
    Q8_0_RETAINED_INPUT_SCALES.with(|scales_cell| {
        Q8_0_RETAINED_INPUT_QUANTS.with(|quants_cell| {
            let mut scales = scales_cell.borrow_mut();
            let mut quants = quants_cell.borrow_mut();
            scales.clear();
            quants.clear();
            scales.reserve(blocks.len());
            quants.reserve(blocks.len() * Q8_0_BLOCK_VALUES);
            for block in blocks {
                scales.push(block.scale);
                quants.extend_from_slice(&block.quants);
            }
            let result = f(&scales, &quants);
            cap_q8_0_file_reader_scratch(&mut scales, 0);
            cap_q8_0_file_reader_scratch(&mut quants, 0);
            result
        })
    })
}

fn q8_0_blocks_as_bytes(blocks: &[Q8_0Block]) -> &[u8] {
    debug_assert_eq!(mem::size_of::<Q8_0Block>(), 36);
    // SAFETY: Q8_0Block is #[repr(C)] with f32 scale followed by 32 i8 quants.
    // The Metal retained-Q8 kernel treats this exact byte layout as immutable input.
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), std::mem::size_of_val(blocks))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_dot_rows_dotprod(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    weight
        .iter()
        .zip(input)
        .map(|(weight_block, input_block)| {
            // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
            let int_sum = unsafe {
                q8_0_i8_block_dotprod(weight_block.quants.as_ptr(), input_block.quants.as_ptr())
            };
            int_sum as f32 * weight_block.scale * input_block.scale
        })
        .sum()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_two_dot_rows_dotprod(
    first_weight: &[Q8_0Block],
    second_weight: &[Q8_0Block],
    input: &[Q8_0Block],
) -> (f32, f32) {
    let mut first_sum = 0.0_f32;
    let mut second_sum = 0.0_f32;
    for ((first_block, second_block), input_block) in
        first_weight.iter().zip(second_weight).zip(input)
    {
        // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
        let first_int_sum = unsafe {
            q8_0_i8_block_dotprod(first_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
        let second_int_sum = unsafe {
            q8_0_i8_block_dotprod(second_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        first_sum += first_int_sum as f32 * first_block.scale * input_block.scale;
        second_sum += second_int_sum as f32 * second_block.scale * input_block.scale;
    }
    (first_sum, second_sum)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_packed_4x4_block_dotprod(
    packed_quants: *const i8,
    input_quants: *const i8,
) -> [i32; 4] {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers provide a packed 128-i8 block and a 32-i8 input block.
    let b0 = unsafe { vld1q_s8(packed_quants) };
    let b1 = unsafe { vld1q_s8(packed_quants.add(16)) };
    let b2 = unsafe { vld1q_s8(packed_quants.add(32)) };
    let b3 = unsafe { vld1q_s8(packed_quants.add(48)) };
    let b4 = unsafe { vld1q_s8(packed_quants.add(64)) };
    let b5 = unsafe { vld1q_s8(packed_quants.add(80)) };
    let b6 = unsafe { vld1q_s8(packed_quants.add(96)) };
    let b7 = unsafe { vld1q_s8(packed_quants.add(112)) };
    let a0 = unsafe { vld1q_s8(input_quants) };
    let a1 = unsafe { vld1q_s8(input_quants.add(16)) };

    let mut acc = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT. This mirrors llama.cpp's q8_0 4x4
    // GEMV lane-dot shape: one output row per accumulator lane.
    unsafe {
        asm!(
            "sdot {acc:v}.4s, {b0:v}.16b, {a0:v}.4b[0]",
            "sdot {acc:v}.4s, {b1:v}.16b, {a0:v}.4b[1]",
            "sdot {acc:v}.4s, {b2:v}.16b, {a0:v}.4b[2]",
            "sdot {acc:v}.4s, {b3:v}.16b, {a0:v}.4b[3]",
            "sdot {acc:v}.4s, {b4:v}.16b, {a1:v}.4b[0]",
            "sdot {acc:v}.4s, {b5:v}.16b, {a1:v}.4b[1]",
            "sdot {acc:v}.4s, {b6:v}.16b, {a1:v}.4b[2]",
            "sdot {acc:v}.4s, {b7:v}.16b, {a1:v}.4b[3]",
            acc = inout(vreg) acc,
            b0 = in(vreg) b0,
            b1 = in(vreg) b1,
            b2 = in(vreg) b2,
            b3 = in(vreg) b3,
            b4 = in(vreg) b4,
            b5 = in(vreg) b5,
            b6 = in(vreg) b6,
            b7 = in(vreg) b7,
            a0 = in(vreg) a0,
            a1 = in(vreg) a1,
            options(nostack, preserves_flags)
        );
    }
    // SAFETY: int32x4_t is a four-lane i32 vector; lane order is output-row order.
    unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_packed_4x8_block_dotprod(
    packed_quants: *const i8,
    input_quants: *const i8,
) -> [i32; 4] {
    use std::arch::aarch64::{vcombine_s8, vdupq_n_s32, vld1_s8, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers provide a packed 128-i8 block and a 32-i8 input block.
    let b0 = unsafe { vld1q_s8(packed_quants) };
    let b1 = unsafe { vld1q_s8(packed_quants.add(16)) };
    let b2 = unsafe { vld1q_s8(packed_quants.add(32)) };
    let b3 = unsafe { vld1q_s8(packed_quants.add(48)) };
    let b4 = unsafe { vld1q_s8(packed_quants.add(64)) };
    let b5 = unsafe { vld1q_s8(packed_quants.add(80)) };
    let b6 = unsafe { vld1q_s8(packed_quants.add(96)) };
    let b7 = unsafe { vld1q_s8(packed_quants.add(112)) };
    let a0_half = unsafe { vld1_s8(input_quants) };
    let a1_half = unsafe { vld1_s8(input_quants.add(8)) };
    let a2_half = unsafe { vld1_s8(input_quants.add(16)) };
    let a3_half = unsafe { vld1_s8(input_quants.add(24)) };
    let a0 = vcombine_s8(a0_half, a0_half);
    let a1 = vcombine_s8(a1_half, a1_half);
    let a2 = vcombine_s8(a2_half, a2_half);
    let a3 = vcombine_s8(a3_half, a3_half);

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT. This mirrors llama.cpp's q8_0 4x8
    // GEMV dot shape; pairwise lane sums below mirror vpaddq_s32(ret0, ret1).
    unsafe {
        asm!(
            "sdot {acc0:v}.4s, {b0:v}.16b, {a0:v}.16b",
            "sdot {acc1:v}.4s, {b1:v}.16b, {a0:v}.16b",
            "sdot {acc0:v}.4s, {b2:v}.16b, {a1:v}.16b",
            "sdot {acc1:v}.4s, {b3:v}.16b, {a1:v}.16b",
            "sdot {acc0:v}.4s, {b4:v}.16b, {a2:v}.16b",
            "sdot {acc1:v}.4s, {b5:v}.16b, {a2:v}.16b",
            "sdot {acc0:v}.4s, {b6:v}.16b, {a3:v}.16b",
            "sdot {acc1:v}.4s, {b7:v}.16b, {a3:v}.16b",
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            b0 = in(vreg) b0,
            b1 = in(vreg) b1,
            b2 = in(vreg) b2,
            b3 = in(vreg) b3,
            b4 = in(vreg) b4,
            b5 = in(vreg) b5,
            b6 = in(vreg) b6,
            b7 = in(vreg) b7,
            a0 = in(vreg) a0,
            a1 = in(vreg) a1,
            a2 = in(vreg) a2,
            a3 = in(vreg) a3,
            options(nostack, preserves_flags)
        );
    }
    // SAFETY: int32x4_t is a four-lane i32 vector.
    let lanes0 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc0) };
    let lanes1 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc1) };
    [
        lanes0[0] + lanes0[1],
        lanes0[2] + lanes0[3],
        lanes1[0] + lanes1[1],
        lanes1[2] + lanes1[3],
    ]
}

fn q8_0_reader_backing(weight: &CpuTensor, input_width: usize) -> Result<Option<&Q8_0FileBacking>> {
    if weight.source_type != Some(GgufTensorType::Q8_0) || weight.q8_0_blocks.is_some() {
        return Ok(None);
    }
    if q8_0_selected_packed_rows4(weight).is_some() {
        return Ok(None);
    }
    let Some(backing) = weight.q8_0_file_backing.as_ref() else {
        return Ok(None);
    };
    if weight.rank() != 2 || weight.dim(1)? != input_width {
        return Ok(None);
    }
    let output_width = weight.dim(0)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let expected_blocks = output_width.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("q8_0 reader-backed block count overflow".to_string())
    })?;
    if backing.num_blocks != expected_blocks {
        return Ok(None);
    }
    Ok(Some(backing))
}

#[allow(dead_code)]
fn matmul_rhs_transposed_q8_0_block_reader(
    input: &CpuTensor,
    backing: &Q8_0FileBacking,
    reader: Q8BlockReader,
    output_width: usize,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let q8_flags = Q8RuntimeFlags::from_env();
    matmul_rhs_transposed_q8_0_block_reader_with_flags(
        input,
        backing,
        reader,
        output_width,
        name,
        &q8_flags,
    )
}

fn matmul_rhs_transposed_q8_0_block_reader_with_flags(
    input: &CpuTensor,
    backing: &Q8_0FileBacking,
    reader: Q8BlockReader,
    output_width: usize,
    name: impl Into<String>,
    q8_flags: &Q8RuntimeFlags,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 block-reader input width {input_width} is not a multiple of {Q8_0_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let expected_blocks = output_width.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("q8_0 block-reader block count overflow".to_string())
    })?;
    if reader.num_blocks != expected_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 block-reader expected {expected_blocks} blocks, got {}",
            reader.num_blocks
        )));
    }
    let row_bytes_len = blocks_per_row
        .checked_mul(Q8BlockReader::BLOCK_SIZE_BYTES)
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "q8_0 block-reader row byte count overflow".to_string(),
            )
        })?;
    let output_len = rows.checked_mul(output_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("q8_0 block-reader output size overflow".to_string())
    })?;
    if output_len == 0 {
        return CpuTensor::from_f32(name, vec![rows, output_width], Vec::new());
    }
    let mut output = vec![0.0_f32; output_len];
    let parallelize_output =
        should_use_q8_0_file_reader_parallel_output(row_bytes_len, output_width, rows)?;
    let use_q8_0_block_dot = q8_flags.file_reader_block_dot;
    let chunk_rows = q8_0_file_reader_chunk_rows_for_batch(
        row_bytes_len,
        output_width,
        rows,
        parallelize_output,
    )?;
    let row_chunk_len = row_bytes_len.checked_mul(chunk_rows).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "q8_0 block-reader chunk byte count overflow".to_string(),
        )
    })?;
    with_q8_0_file_reader_row_chunk(row_chunk_len, |row_chunk| {
        with_q8_0_file_reader_quantized_inputs(|quantized_input_blocks| {
            quantized_input_blocks.clear();
            if use_q8_0_block_dot {
                let quantized_input_block_count =
                    rows.checked_mul(blocks_per_row).ok_or_else(|| {
                        BackendError::RuntimeShapeMismatch(
                            "q8_0 block-reader quantized input block count overflow".to_string(),
                        )
                    })?;
                quantized_input_blocks.reserve(quantized_input_block_count);
                for row in input.data.chunks_exact(input_width) {
                    quantize_q8_0_blocks_into(row, quantized_input_blocks);
                }
            }
            if rows == 1 {
                let chunk_scales_len = chunk_rows.checked_mul(blocks_per_row).ok_or_else(|| {
                    BackendError::RuntimeShapeMismatch(
                        "q8_0 block-reader chunk scale count overflow".to_string(),
                    )
                })?;
                return with_q8_0_file_reader_chunk_scales(chunk_scales_len, |chunk_scales| {
                    let quantized_input = if use_q8_0_block_dot {
                        Some(&quantized_input_blocks[..blocks_per_row])
                    } else {
                        None
                    };
                    let mut output_idx = 0usize;
                    while output_idx < output_width {
                        let rows_this_chunk = chunk_rows.min(output_width - output_idx);
                        let chunk_bytes_len =
                            row_bytes_len.checked_mul(rows_this_chunk).ok_or_else(|| {
                                BackendError::RuntimeShapeMismatch(
                                    "q8_0 block-reader chunk byte count overflow".to_string(),
                                )
                            })?;
                        let block_start = output_idx * blocks_per_row;
                        let chunk_offset = reader
                            .offset
                            .checked_add((block_start * Q8BlockReader::BLOCK_SIZE_BYTES) as u64)
                            .ok_or_else(|| {
                                BackendError::RuntimeShapeMismatch(
                                    "q8_0 block-reader chunk offset overflow".to_string(),
                                )
                            })?;
                        let chunk = &mut row_chunk[..chunk_bytes_len];
                        let scales = &mut chunk_scales[..rows_this_chunk * blocks_per_row];
                        backing.read_exact_at_cached_with_q8_0_scales(
                            chunk,
                            chunk_offset,
                            scales,
                        )?;
                        let output_chunk = &mut output[output_idx..output_idx + rows_this_chunk];
                        let completed_with_metal = if use_q8_0_block_dot && q8_flags.metal {
                            let (input_scales, input_quants) = q8_0_block_scales_and_quants(
                                &quantized_input_blocks[..blocks_per_row],
                            );
                            metal::try_q8_0_encoded_linear_row(
                                &input_scales,
                                &input_quants,
                                chunk,
                                scales,
                                rows_this_chunk,
                                blocks_per_row,
                                output_chunk,
                            )
                        } else {
                            false
                        };
                        if completed_with_metal {
                            // Opt-in experimental Metal Q8 path completed this chunk.
                        } else if parallelize_output {
                            output_chunk
                                .par_iter_mut()
                                .zip(chunk.par_chunks_exact(row_bytes_len))
                                .zip(scales.par_chunks_exact(blocks_per_row))
                                .for_each(|((out_value, row_bytes), row_scales)| {
                                    *out_value = if let Some(quantized_input) = quantized_input {
                                        dot_q8_0_encoded_row_quantized_input_with_scales(
                                            quantized_input,
                                            row_bytes,
                                            row_scales,
                                        )
                                    } else {
                                        dot_q8_0_encoded_row_f32_input_with_scales(
                                            &input.data[..input_width],
                                            row_bytes,
                                            row_scales,
                                        )
                                    };
                                });
                        } else {
                            for ((out_value, row_bytes), row_scales) in output_chunk
                                .iter_mut()
                                .zip(chunk.chunks_exact(row_bytes_len))
                                .zip(scales.chunks_exact(blocks_per_row))
                            {
                                *out_value = if let Some(quantized_input) = quantized_input {
                                    dot_q8_0_encoded_row_quantized_input_with_scales(
                                        quantized_input,
                                        row_bytes,
                                        row_scales,
                                    )
                                } else {
                                    dot_q8_0_encoded_row_f32_input_with_scales(
                                        &input.data[..input_width],
                                        row_bytes,
                                        row_scales,
                                    )
                                };
                            }
                        }
                        output_idx += rows_this_chunk;
                    }
                    Ok(())
                });
            }

            let chunk_scales_len = chunk_rows.checked_mul(blocks_per_row).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "q8_0 block-reader chunk scale count overflow".to_string(),
                )
            })?;
            with_q8_0_file_reader_chunk_scales(chunk_scales_len, |chunk_scales| {
                let mut output_idx = 0usize;
                while output_idx < output_width {
                    let rows_this_chunk = chunk_rows.min(output_width - output_idx);
                    let chunk_bytes_len =
                        row_bytes_len.checked_mul(rows_this_chunk).ok_or_else(|| {
                            BackendError::RuntimeShapeMismatch(
                                "q8_0 block-reader chunk byte count overflow".to_string(),
                            )
                        })?;
                    let block_start = output_idx * blocks_per_row;
                    let chunk_offset = reader
                        .offset
                        .checked_add((block_start * Q8BlockReader::BLOCK_SIZE_BYTES) as u64)
                        .ok_or_else(|| {
                            BackendError::RuntimeShapeMismatch(
                                "q8_0 block-reader chunk offset overflow".to_string(),
                            )
                        })?;
                    let chunk = &mut row_chunk[..chunk_bytes_len];
                    let scales = &mut chunk_scales[..rows_this_chunk * blocks_per_row];
                    backing.read_exact_at_cached_with_q8_0_scales(chunk, chunk_offset, scales)?;
                    // Multi-row prefill reuses the same file-backed Q8 weight chunk across
                    // every input row. Decode each weight-block scale once per chunk row,
                    // then walk output columns outermost so each compact weight row stays hot
                    // while it is dotted against all input rows. The older row-major
                    // loop rescanned the full chunk once per input row; that kept RSS low but
                    // turned long-context prefill into unnecessary memory-bandwidth pressure.
                    let output_rows = &mut output[..rows * output_width];
                    if parallelize_output {
                        let scratch_len = rows_this_chunk.checked_mul(rows).ok_or_else(|| {
                            BackendError::RuntimeShapeMismatch(
                                "q8_0 block-reader output scratch size overflow".to_string(),
                            )
                        })?;
                        with_q8_0_file_reader_output_chunk(scratch_len, |output_chunk_scratch| {
                            let completed_with_metal = if use_q8_0_block_dot && q8_flags.metal {
                                let (input_scales, input_quants) =
                                    q8_0_block_scales_and_quants(quantized_input_blocks);
                                metal::try_q8_0_encoded_linear_rows(
                                    &input_scales,
                                    &input_quants,
                                    chunk,
                                    scales,
                                    rows,
                                    rows_this_chunk,
                                    blocks_per_row,
                                    output_chunk_scratch,
                                )
                            } else {
                                false
                            };
                            if !completed_with_metal {
                                output_chunk_scratch
                                    .par_chunks_mut(rows)
                                    .zip(chunk.par_chunks_exact(row_bytes_len))
                                    .zip(scales.par_chunks_exact(blocks_per_row))
                                    .for_each(|((column_outputs, row_bytes), row_scales)| {
                                        for (row_idx, out_value) in
                                            column_outputs.iter_mut().enumerate()
                                        {
                                            let row_start = row_idx * input_width;
                                            let row_end = row_start + input_width;
                                            *out_value = if use_q8_0_block_dot {
                                                let block_start = row_idx * blocks_per_row;
                                                let block_end = block_start + blocks_per_row;
                                                dot_q8_0_encoded_row_quantized_input_with_scales(
                                                    &quantized_input_blocks[block_start..block_end],
                                                    row_bytes,
                                                    row_scales,
                                                )
                                            } else {
                                                dot_q8_0_encoded_row_f32_input_with_scales(
                                                    &input.data[row_start..row_end],
                                                    row_bytes,
                                                    row_scales,
                                                )
                                            };
                                        }
                                    });
                            }
                            for (row, output_row) in
                                output_rows.chunks_mut(output_width).enumerate()
                            {
                                let output_chunk =
                                    &mut output_row[output_idx..output_idx + rows_this_chunk];
                                for (chunk_col, out_value) in output_chunk.iter_mut().enumerate() {
                                    *out_value = output_chunk_scratch[chunk_col * rows + row];
                                }
                            }
                            Ok(())
                        })?;
                    } else {
                        for (chunk_col, (row_bytes, row_scales)) in chunk
                            .chunks_exact(row_bytes_len)
                            .zip(scales.chunks_exact(blocks_per_row))
                            .enumerate()
                        {
                            let absolute_col = output_idx + chunk_col;
                            for row in 0..rows {
                                let row_start = row * input_width;
                                let row_end = row_start + input_width;
                                output_rows[row * output_width + absolute_col] =
                                    if use_q8_0_block_dot {
                                        let block_start = row * blocks_per_row;
                                        let block_end = block_start + blocks_per_row;
                                        dot_q8_0_encoded_row_quantized_input_with_scales(
                                            &quantized_input_blocks[block_start..block_end],
                                            row_bytes,
                                            row_scales,
                                        )
                                    } else {
                                        dot_q8_0_encoded_row_f32_input_with_scales(
                                            &input.data[row_start..row_end],
                                            row_bytes,
                                            row_scales,
                                        )
                                    };
                            }
                        }
                    }
                    output_idx += rows_this_chunk;
                }
                Ok(())
            })
        })
    })?;
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn matmul_rhs_transposed_q8_0_packed_rows4_f32_input(
    input: &CpuTensor,
    packed: &Q8_0PackedRows4,
    interleave: Q8_0PackedRows4Interleave,
    output_width: usize,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 packed rows4 f32 fallback input width {input_width} is not a multiple of {Q8_0_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    if packed.rows != output_width || packed.blocks_per_row != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 packed rows4 f32 fallback shape mismatch: packed rows={}, blocks_per_row={}, requested output_width={output_width}, input blocks_per_row={blocks_per_row}",
            packed.rows, packed.blocks_per_row
        )));
    }
    let mut output = vec![0.0_f32; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        let output_start = row * output_width;
        accumulate_q8_0_packed_rows4_f32_input(
            &input.data[input_start..input_start + input_width],
            packed,
            interleave,
            &mut output[output_start..output_start + output_width],
        );
    }
    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

#[cfg(test)]
fn dot_q8_0_encoded_row(input: &[Q8_0Block], row_bytes: &[u8]) -> f32 {
    let mut sum = 0.0_f32;
    for (input_block, block) in input
        .iter()
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
    {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let int_sum = q8_0_block_int_dot_horizontal_sum_encoded(&block[2..], &input_block.quants);
        sum += int_sum as f32 * scale * input_block.scale;
    }
    sum
}

fn decode_q8_0_encoded_row_scales(row_bytes: &[u8], scales: &mut [f32]) {
    debug_assert_eq!(
        row_bytes.len(),
        scales.len() * Q8BlockReader::BLOCK_SIZE_BYTES
    );
    for (scale, block) in scales
        .iter_mut()
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
    {
        *scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    }
}

fn dot_q8_0_encoded_row_with_scales(input: &[Q8_0Block], row_bytes: &[u8], scales: &[f32]) -> f32 {
    debug_assert_eq!(input.len(), scales.len());
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if aarch64_dotprod_enabled() {
            // SAFETY: runtime feature detection confirms dot-product support; row_bytes is
            // traversed as exact Q8_0 encoded blocks.
            return unsafe { dot_q8_0_encoded_row_with_scales_dotprod(input, row_bytes, scales) };
        }
    }

    let mut sum = 0.0_f32;
    for ((input_block, block), scale) in input
        .iter()
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
        .zip(scales)
    {
        let int_sum = q8_0_block_int_dot_horizontal_sum_encoded(&block[2..], &input_block.quants);
        sum += int_sum as f32 * *scale * input_block.scale;
    }
    sum
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q8_0_encoded_row_with_scales_dotprod(
    input: &[Q8_0Block],
    row_bytes: &[u8],
    scales: &[f32],
) -> f32 {
    let mut sum = 0.0_f32;
    for ((input_block, block), scale) in input
        .iter()
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
        .zip(scales)
    {
        // SAFETY: each encoded Q8_0 block stores 32 contiguous signed quant bytes after
        // the two-byte f16 scale header.
        let int_sum = unsafe {
            q8_0_i8_block_dotprod(
                block[2..].as_ptr().cast::<i8>(),
                input_block.quants.as_ptr(),
            )
        };
        sum += int_sum as f32 * *scale * input_block.scale;
    }
    sum
}

fn dot_q8_0_encoded_row_quantized_input_with_scales(
    input: &[Q8_0Block],
    row_bytes: &[u8],
    scales: &[f32],
) -> f32 {
    dot_q8_0_encoded_row_with_scales(input, row_bytes, scales)
}

fn dot_q8_0_encoded_row_f32_input_with_scales(
    input: &[f32],
    row_bytes: &[u8],
    scales: &[f32],
) -> f32 {
    debug_assert_eq!(input.len(), scales.len() * Q8_0_BLOCK_VALUES);
    let mut sum = 0.0_f32;
    for ((input_block, block), scale) in input
        .chunks_exact(Q8_0_BLOCK_VALUES)
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
        .zip(scales)
    {
        for (input_value, quant) in input_block.iter().zip(block[2..].iter()) {
            sum += *input_value * (*scale * f32::from(*quant as i8));
        }
    }
    sum
}

fn q8_0_block_int_dot_horizontal_sum_encoded(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    debug_assert_eq!(weight.len(), Q8_0_BLOCK_VALUES);
    q8_0_block_int_dot_horizontal_sum_encoded_impl(weight, input)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn q8_0_block_int_dot_horizontal_sum_encoded_impl(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    // SAFETY: both pointers address one complete Q8_0 block (32 signed bytes).
    unsafe { q8_0_i8_block_neon(weight.as_ptr().cast::<i8>(), input.as_ptr()) }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_block_int_dot_horizontal_sum_encoded_impl(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_kernel_avx2_enabled() && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection confirms AVX2 support; callers pass one
            // complete encoded Q8_0 block (32 signed bytes) and one complete input block.
            return unsafe { q8_0_i8_block_avx2(weight.as_ptr().cast::<i8>(), input.as_ptr()) };
        }
    }
    q8_0_block_int_dot_horizontal_sum_encoded_scalar(weight, input)
}

fn q8_0_block_int_dot_horizontal_sum(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    q8_0_block_int_dot_horizontal_sum_impl(weight, input)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn q8_0_block_int_dot_horizontal_sum_impl(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    // SAFETY: both pointers address one complete Q8_0 block (32 signed bytes).
    unsafe { q8_0_i8_block_neon(weight.as_ptr(), input.as_ptr()) }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_block_int_dot_horizontal_sum_impl(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_kernel_avx2_enabled() && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection confirms AVX2 support; callers pass complete
            // Q8_0 blocks containing 32 contiguous signed bytes each.
            return unsafe { q8_0_i8_block_avx2(weight.as_ptr(), input.as_ptr()) };
        }
    }
    q8_0_block_int_dot_horizontal_sum_scalar(weight, input)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_block_int_dot_horizontal_sum_scalar(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    // The generic q8_0 x q8_0 dot sums products in deterministic four-product lanes.
    let lanes = [
        q8_0_dot_group4(weight, input, 0) + q8_0_dot_group4(weight, input, 16),
        q8_0_dot_group4(weight, input, 4) + q8_0_dot_group4(weight, input, 20),
        q8_0_dot_group4(weight, input, 8) + q8_0_dot_group4(weight, input, 24),
        q8_0_dot_group4(weight, input, 12) + q8_0_dot_group4(weight, input, 28),
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_block_int_dot_horizontal_sum_encoded_scalar(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    let lanes = [
        q8_0_dot_group4_encoded(weight, input, 0) + q8_0_dot_group4_encoded(weight, input, 16),
        q8_0_dot_group4_encoded(weight, input, 4) + q8_0_dot_group4_encoded(weight, input, 20),
        q8_0_dot_group4_encoded(weight, input, 8) + q8_0_dot_group4_encoded(weight, input, 24),
        q8_0_dot_group4_encoded(weight, input, 12) + q8_0_dot_group4_encoded(weight, input, 28),
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_dot_group4(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
    start: usize,
) -> i32 {
    i32::from(weight[start]) * i32::from(input[start])
        + i32::from(weight[start + 1]) * i32::from(input[start + 1])
        + i32::from(weight[start + 2]) * i32::from(input[start + 2])
        + i32::from(weight[start + 3]) * i32::from(input[start + 3])
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn q8_0_dot_group4_encoded(weight: &[u8], input: &[i8; Q8_0_BLOCK_VALUES], start: usize) -> i32 {
    i32::from(weight[start] as i8) * i32::from(input[start])
        + i32::from(weight[start + 1] as i8) * i32::from(input[start + 1])
        + i32::from(weight[start + 2] as i8) * i32::from(input[start + 2])
        + i32::from(weight[start + 3] as i8) * i32::from(input[start + 3])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_kernel_avx2_enabled() -> bool {
    #[cfg(test)]
    {
        x86_q8_kernel_avx2_enabled_from_env()
    }
    #[cfg(not(test))]
    {
        static X86_Q8_KERNEL_AVX2_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_KERNEL_AVX2_ENABLED.get_or_init(x86_q8_kernel_avx2_enabled_from_env)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_kernel_avx2_enabled_from_env() -> bool {
    matches!(
        env::var("CAMELID_X86_Q8_KERNEL").as_deref(),
        Ok("avx2") | Ok("AVX2") | Ok("on") | Ok("ON") | Ok("1") | Ok("true") | Ok("TRUE")
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx2_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX2_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX2_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT")
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx2_dot_hoist_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST")
            && std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST")
                && std::arch::is_x86_feature_detected!("avx2")
        })
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_q8_packed_rows4_avx2_dot_hoist_enabled() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST")
            && std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST")
                && std::arch::is_x86_feature_detected!("avx2")
        })
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_q8_packed_rows4_avx2_dot_decode_hoist_enabled() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_i8_block_avx2(weight: *const i8, input: *const i8) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_epi32, _mm256_cmpeq_epi8, _mm256_cvtepi8_epi16, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_movemask_epi8, _mm256_mullo_epi16,
        _mm256_set1_epi16, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_sign_epi8,
        _mm256_storeu_si256, _mm_loadu_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_cmpeq_epi8, _mm256_cvtepi8_epi16, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_movemask_epi8, _mm256_mullo_epi16,
        _mm256_set1_epi16, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_sign_epi8,
        _mm256_storeu_si256, _mm_loadu_si128,
    };

    let ones = _mm256_set1_epi16(1);
    let weight_i8 = unsafe { _mm256_loadu_si256(weight.cast()) };
    let input_i8 = unsafe { _mm256_loadu_si256(input.cast()) };
    let min_i8 = _mm256_set1_epi8(i8::MIN);
    let has_min_i8 = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(weight_i8, min_i8))
        | _mm256_movemask_epi8(_mm256_cmpeq_epi8(input_i8, min_i8)))
        != 0;
    let acc = if has_min_i8 {
        let mut acc = _mm256_setzero_si256();
        for offset in [0usize, 16] {
            let weight_half = unsafe { _mm_loadu_si128(weight.add(offset).cast()) };
            let input_half = unsafe { _mm_loadu_si128(input.add(offset).cast()) };
            let products = _mm256_mullo_epi16(
                _mm256_cvtepi8_epi16(weight_half),
                _mm256_cvtepi8_epi16(input_half),
            );
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(products, ones));
        }
        acc
    } else {
        // Mirrors llama.cpp's x86 q8_0 dot for well-formed Q8_0 blocks, whose
        // quantized values are in [-127, 127].
        let abs_weight = _mm256_sign_epi8(weight_i8, weight_i8);
        let signed_input = _mm256_sign_epi8(input_i8, weight_i8);
        _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed_input), ones)
    };

    let mut lanes = [0_i32; 8];
    // SAFETY: lanes has exactly 32 bytes of storage for one __m256i value.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
    lanes.iter().sum()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn aarch64_dotprod_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !q8_0_env_flag_disabled("CAMELID_AARCH64_DOTPROD")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn q8_0_i8_block_neon(weight: *const i8, input: *const i8) -> i32 {
    if aarch64_dotprod_enabled() {
        // SAFETY: feature detection above guarantees the dot-product instructions are
        // available, and callers pass pointers to at least 32 contiguous i8 values.
        return unsafe { q8_0_i8_block_dotprod(weight, input) };
    }

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    unsafe { q8_0_i8_block_neon_mul(weight, input) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn q8_0_i8_block_dotprod(weight: *const i8, input: *const i8) -> i32 {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    let weight_lo = unsafe { vld1q_s8(weight) };
    let input_lo = unsafe { vld1q_s8(input) };
    let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
    let input_hi = unsafe { vld1q_s8(input.add(16)) };

    let mut acc = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT for this function. The operands are full
    // 128-bit vector registers loaded above, and the instruction only updates `acc`.
    unsafe {
        asm!(
            "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
            acc = inout(vreg) acc,
            weight_lo = in(vreg) weight_lo,
            input_lo = in(vreg) input_lo,
            weight_hi = in(vreg) weight_hi,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );
    }
    horizontal_sum_i32x4(acc)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn q8_0_i8_block_neon_mul(weight: *const i8, input: *const i8) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vpaddlq_s16,
    };

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    let weight_lo = unsafe { vld1q_s8(weight) };
    let input_lo = unsafe { vld1q_s8(input) };
    let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
    let input_hi = unsafe { vld1q_s8(input.add(16)) };

    let mut acc = vdupq_n_s32(0);
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_low_s8(weight_lo), vget_low_s8(input_lo))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_high_s8(weight_lo), vget_high_s8(input_lo))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_low_s8(weight_hi), vget_low_s8(input_hi))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_high_s8(weight_hi), vget_high_s8(input_hi))),
    );
    horizontal_sum_i32x4(acc)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[inline(always)]
fn horizontal_sum_i32x4(acc: std::arch::aarch64::int32x4_t) -> i32 {
    // SAFETY: int32x4_t is a four-lane i32 vector; extracting via transmute preserves lanes.
    let lanes: [i32; 4] =
        unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc) };
    lanes.iter().sum()
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = 14 - half_exp;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        if (mantissa & round_bit) != 0
            && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)
        {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14_i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

fn accumulate_descriptor_linear_row_with_precision(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    precision: LinearAccumulationPrecision,
) {
    match precision {
        LinearAccumulationPrecision::F32 => {
            accumulate_descriptor_linear_row(input_row, weight, output)
        }
        LinearAccumulationPrecision::F64 => {
            let output_width = output.len();
            if should_parallelize_linear_output(output_width) {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(output_idx, out_value)| {
                        let mut sum = 0.0_f64;
                        for (inner, lhs_value) in input_row.iter().copied().enumerate() {
                            sum += f64::from(lhs_value)
                                * f64::from(weight.data[inner * output_width + output_idx]);
                        }
                        *out_value = sum as f32;
                    });
            } else {
                for (output_idx, out_value) in output.iter_mut().enumerate() {
                    let mut sum = 0.0_f64;
                    for (inner, lhs_value) in input_row.iter().copied().enumerate() {
                        sum += f64::from(lhs_value)
                            * f64::from(weight.data[inner * output_width + output_idx]);
                    }
                    *out_value = sum as f32;
                }
            }
        }
    }
}

fn accumulate_descriptor_linear_row(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
    if try_accumulate_descriptor_linear_row_metal(input_row, weight, output) {
        return;
    }
    if try_accumulate_descriptor_linear_row_accelerate(input_row, weight, output) {
        return;
    }

    if should_parallelize_linear_output(output.len()) {
        let output_width = output.len();
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_idx, out_value)| {
                let mut sum = *out_value;
                for (inner, lhs_value) in input_row.iter().copied().enumerate() {
                    if lhs_value == 0.0 {
                        continue;
                    }
                    sum += lhs_value * weight.data[inner * output_width + output_idx];
                }
                *out_value = sum;
            });
        return;
    }

    for (inner, lhs_value) in input_row.iter().copied().enumerate() {
        if lhs_value == 0.0 {
            continue;
        }
        let rhs_start = inner * output.len();
        let rhs_row = &weight.data[rhs_start..rhs_start + output.len()];
        for (out_value, rhs_value) in output.iter_mut().zip(rhs_row) {
            *out_value += lhs_value * rhs_value;
        }
    }
}

fn try_accumulate_descriptor_linear_row_metal(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) -> bool {
    #[cfg(target_os = "macos")]
    {
        if std::env::var("CAMELID_METAL_LINEAR").ok().as_deref() != Some("1") {
            return false;
        }
        if weight.q8_0_blocks.is_some() || weight.q8_0_file_backing.is_some() {
            return false;
        }
        metal::try_linear_row_f32(input_row, weight.data, weight.rows, weight.cols, output)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (input_row, weight, output);
        false
    }
}

fn try_accumulate_descriptor_linear_row_accelerate(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) -> bool {
    #[cfg(target_os = "macos")]
    {
        if !apple_accelerate_blas_enabled() {
            return false;
        }
        if weight.rows != input_row.len() || weight.cols != output.len() {
            return false;
        }
        let Ok(m) = i32::try_from(weight.rows) else {
            return false;
        };
        let Ok(n) = i32::try_from(weight.cols) else {
            return false;
        };
        let Some(element_count) = weight.rows.checked_mul(weight.cols) else {
            return false;
        };
        if element_count < apple_accelerate_min_elements() {
            return false;
        }
        // SAFETY: dimensions are checked above. `weight.data` is row-major [M, N], so
        // CblasTrans computes output[N] += input[M]^T * weight[M, N].
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                m,
                n,
                1.0,
                weight.data.as_ptr(),
                n,
                input_row.as_ptr(),
                1,
                1.0,
                output.as_mut_ptr(),
                1,
            );
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (input_row, weight, output);
        false
    }
}

#[cfg(target_os = "macos")]
fn apple_accelerate_blas_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !q8_0_env_flag_disabled("CAMELID_APPLE_ACCELERATE"))
}

#[cfg(target_os = "macos")]
fn apple_accelerate_min_elements() -> usize {
    static MIN_ELEMENTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_ELEMENTS.get_or_init(|| {
        env::var("CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(262_144)
    })
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans_a: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        inc_x: i32,
        beta: f32,
        y: *mut f32,
        inc_y: i32,
    );
}

fn accumulate_transposed_linear_row_runtime(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    precision: LinearAccumulationPrecision,
) -> Result<()> {
    let runtime_plan = ResolvedRuntimePlan {
        linear_accumulation_precision: precision,
        q8: Q8RuntimeFlags::from_env(),
    };
    accumulate_transposed_linear_row_runtime_with_plan(input_row, weight, output, &runtime_plan)
}

fn accumulate_transposed_linear_row_runtime_with_plan(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    runtime_plan: &ResolvedRuntimePlan,
) -> Result<()> {
    if let Some(backing) = borrowed_q8_0_reader_backing(weight, input_row.len(), output.len())? {
        accumulate_transposed_linear_row_q8_0_file_reader_with_flags(
            input_row,
            backing,
            output,
            &runtime_plan.q8,
        )?;
        return Ok(());
    }
    accumulate_transposed_linear_row_with_precision_with_plan(
        input_row,
        weight,
        output,
        runtime_plan,
    );
    Ok(())
}

fn borrowed_q8_0_reader_backing<'a>(
    weight: BorrowedLinearWeight<'a>,
    input_width: usize,
    output_width: usize,
) -> Result<Option<&'a Q8_0FileBacking>> {
    if weight.source_type != Some(GgufTensorType::Q8_0) || weight.q8_0_blocks.is_some() {
        return Ok(None);
    }
    if q8_0_selected_borrowed_packed_rows4(weight).is_some() {
        return Ok(None);
    }
    let Some(backing) = weight.q8_0_file_backing else {
        return Ok(None);
    };
    if weight.cols != input_width || weight.rows != output_width {
        return Ok(None);
    }
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let expected_blocks = output_width.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "q8_0 borrowed reader-backed block count overflow".to_string(),
        )
    })?;
    if backing.num_blocks != expected_blocks {
        return Ok(None);
    }
    Ok(Some(backing))
}

#[allow(dead_code)]
fn accumulate_transposed_linear_row_q8_0_file_reader(
    input_row: &[f32],
    backing: &Q8_0FileBacking,
    output: &mut [f32],
) -> Result<()> {
    let q8_flags = Q8RuntimeFlags::from_env();
    accumulate_transposed_linear_row_q8_0_file_reader_with_flags(
        input_row, backing, output, &q8_flags,
    )
}

fn accumulate_transposed_linear_row_q8_0_file_reader_with_flags(
    input_row: &[f32],
    backing: &Q8_0FileBacking,
    output: &mut [f32],
    q8_flags: &Q8RuntimeFlags,
) -> Result<()> {
    let input_width = input_row.len();
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 borrowed block-reader input width {input_width} is not a multiple of {Q8_0_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let row_bytes_len = blocks_per_row
        .checked_mul(Q8BlockReader::BLOCK_SIZE_BYTES)
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "q8_0 borrowed block-reader row byte count overflow".to_string(),
            )
        })?;
    let chunk_rows = q8_0_file_reader_chunk_rows(row_bytes_len, output.len())?;
    let row_chunk_len = row_bytes_len.checked_mul(chunk_rows).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "q8_0 borrowed block-reader chunk byte count overflow".to_string(),
        )
    })?;
    let output_width = output.len();
    let use_q8_0_block_dot = q8_flags.file_reader_block_dot;
    with_q8_0_file_reader_row_chunk(row_chunk_len, |row_chunk| {
        with_q8_0_file_reader_quantized_inputs(|quantized_input_blocks| {
            quantized_input_blocks.clear();
            if use_q8_0_block_dot {
                quantize_q8_0_blocks_into(input_row, quantized_input_blocks);
            }
            let chunk_scales_len = chunk_rows.checked_mul(blocks_per_row).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "q8_0 borrowed block-reader chunk scale count overflow".to_string(),
                )
            })?;
            with_q8_0_file_reader_chunk_scales(chunk_scales_len, |chunk_scales| {
                let quantized_input = if use_q8_0_block_dot {
                    Some(&quantized_input_blocks[..blocks_per_row])
                } else {
                    None
                };
                let mut output_start = 0usize;
                while output_start < output_width {
                    let rows_this_chunk = chunk_rows.min(output_width - output_start);
                    let chunk_bytes_len =
                        row_bytes_len.checked_mul(rows_this_chunk).ok_or_else(|| {
                            BackendError::RuntimeShapeMismatch(
                                "q8_0 borrowed block-reader chunk byte count overflow".to_string(),
                            )
                        })?;
                    let chunk_offset = backing
                        .absolute_offset
                        .checked_add((output_start * row_bytes_len) as u64)
                        .ok_or_else(|| {
                            BackendError::RuntimeShapeMismatch(
                                "q8_0 borrowed block-reader chunk offset overflow".to_string(),
                            )
                        })?;
                    let chunk = &mut row_chunk[..chunk_bytes_len];
                    let scales = &mut chunk_scales[..rows_this_chunk * blocks_per_row];
                    backing.read_exact_at_cached_with_q8_0_scales(chunk, chunk_offset, scales)?;
                    let output_end = output_start + rows_this_chunk;
                    let output_chunk = &mut output[output_start..output_end];
                    if should_parallelize_q8_0_file_reader_output(output_width) {
                        output_chunk
                            .par_iter_mut()
                            .zip(chunk.par_chunks_exact(row_bytes_len))
                            .zip(scales.par_chunks_exact(blocks_per_row))
                            .for_each(|((out_value, row_bytes), row_scales)| {
                                *out_value = if let Some(quantized_input) = quantized_input {
                                    dot_q8_0_encoded_row_quantized_input_with_scales(
                                        quantized_input,
                                        row_bytes,
                                        row_scales,
                                    )
                                } else {
                                    dot_q8_0_encoded_row_f32_input_with_scales(
                                        input_row, row_bytes, row_scales,
                                    )
                                };
                            });
                    } else {
                        for ((out_value, row_bytes), row_scales) in output_chunk
                            .iter_mut()
                            .zip(chunk.chunks_exact(row_bytes_len))
                            .zip(scales.chunks_exact(blocks_per_row))
                        {
                            *out_value = if let Some(quantized_input) = quantized_input {
                                dot_q8_0_encoded_row_quantized_input_with_scales(
                                    quantized_input,
                                    row_bytes,
                                    row_scales,
                                )
                            } else {
                                dot_q8_0_encoded_row_f32_input_with_scales(
                                    input_row, row_bytes, row_scales,
                                )
                            };
                        }
                    }
                    output_start += rows_this_chunk;
                }
                Ok(())
            })
        })
    })
}

thread_local! {
    static Q8_0_FILE_READER_ROW_CHUNK: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_CHUNK_SCALES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_QUANTIZED_INPUTS: RefCell<Vec<Q8_0Block>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_OUTPUT_CHUNK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static Q8_0_PREFILL_PACKED_INPUTS: RefCell<Vec<Q8_0PackedRows4Block>> = const { RefCell::new(Vec::new()) };
    static Q8_0_RETAINED_INPUT_SCALES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static Q8_0_RETAINED_INPUT_QUANTS: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

fn q8_0_file_reader_retained_scratch_bytes() -> usize {
    const DEFAULT_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
    parse_byte_count_env("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES")
        .unwrap_or(DEFAULT_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES)
}

fn q8_0_file_reader_retained_scratch_entries<T>() -> usize {
    q8_0_file_reader_retained_scratch_bytes() / mem::size_of::<T>().max(1)
}

fn cap_q8_0_file_reader_scratch<T>(scratch: &mut Vec<T>, retained_len: usize) {
    let retained_entries = q8_0_file_reader_retained_scratch_entries::<T>();
    if scratch.capacity() > retained_entries {
        *scratch = Vec::with_capacity(retained_len.min(retained_entries));
    } else if scratch.len() > retained_len {
        scratch.truncate(retained_len);
    }
}

fn with_q8_0_file_reader_row_chunk<T>(
    len: usize,
    f: impl FnOnce(&mut [u8]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_ROW_CHUNK.with(|cell| {
        let mut row_chunk = cell.borrow_mut();
        if row_chunk.len() < len {
            row_chunk.resize(len, 0);
        }
        let result = f(&mut row_chunk[..len]);
        cap_q8_0_file_reader_scratch(&mut row_chunk, len);
        result
    })
}

fn with_q8_0_file_reader_chunk_scales<T>(
    len: usize,
    f: impl FnOnce(&mut [f32]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_CHUNK_SCALES.with(|cell| {
        let mut scales = cell.borrow_mut();
        if scales.len() < len {
            scales.resize(len, 0.0);
        }
        let result = f(&mut scales[..len]);
        cap_q8_0_file_reader_scratch(&mut scales, len);
        result
    })
}

fn with_q8_0_file_reader_quantized_inputs<T>(
    f: impl FnOnce(&mut Vec<Q8_0Block>) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_QUANTIZED_INPUTS.with(|cell| {
        let mut quantized_inputs = cell.borrow_mut();
        let result = f(&mut quantized_inputs);
        // Keep the allocation as reusable scratch capacity, but do not leave the
        // previous activation blocks logically live between file-backed Q8 calls.
        quantized_inputs.clear();
        cap_q8_0_file_reader_scratch(&mut quantized_inputs, 0);
        result
    })
}

fn with_q8_0_file_reader_output_chunk<T>(
    len: usize,
    f: impl FnOnce(&mut [f32]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_OUTPUT_CHUNK.with(|cell| {
        let mut output_chunk = cell.borrow_mut();
        if output_chunk.len() < len {
            output_chunk.resize(len, 0.0);
        }
        let result = f(&mut output_chunk[..len]);
        cap_q8_0_file_reader_scratch(&mut output_chunk, len);
        result
    })
}

#[cfg(test)]
fn q8_0_file_reader_scratch_capacities() -> (usize, usize, usize, usize) {
    let row_chunk = Q8_0_FILE_READER_ROW_CHUNK.with(|cell| cell.borrow().capacity());
    let chunk_scales = Q8_0_FILE_READER_CHUNK_SCALES.with(|cell| cell.borrow().capacity());
    let quantized_inputs = Q8_0_FILE_READER_QUANTIZED_INPUTS.with(|cell| cell.borrow().capacity());
    let output_chunk = Q8_0_FILE_READER_OUTPUT_CHUNK.with(|cell| cell.borrow().capacity());
    (row_chunk, chunk_scales, quantized_inputs, output_chunk)
}

fn should_parallelize_q8_0_file_reader_output(output_width: usize) -> bool {
    const DEFAULT_Q8_0_FILE_READER_PARALLEL_MIN_OUTPUTS: usize = 1024;
    if rayon::current_num_threads() <= 1 {
        return false;
    }
    if env::var("CAMELID_PARALLEL_LINEAR").is_ok() {
        return should_parallelize_linear_output(output_width);
    }
    let min_outputs = env::var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_Q8_0_FILE_READER_PARALLEL_MIN_OUTPUTS);
    output_width >= min_outputs
}

fn should_use_q8_0_file_reader_parallel_output(
    row_bytes_len: usize,
    output_width: usize,
    input_rows: usize,
) -> Result<bool> {
    let parallelize_output = should_parallelize_q8_0_file_reader_output(output_width);
    if !parallelize_output || input_rows <= 1 || output_width == 0 {
        return Ok(parallelize_output);
    }
    if env::var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES").is_ok() {
        return Ok(parallelize_output);
    }

    let weight_chunk_rows = q8_0_file_reader_chunk_rows(row_bytes_len, output_width)?;
    if weight_chunk_rows < output_width {
        return Ok(parallelize_output);
    }
    let scratch_chunk_rows = q8_0_file_reader_output_scratch_chunk_rows(input_rows, output_width)?;
    if scratch_chunk_rows < weight_chunk_rows {
        // With the default scratch budget, prefer the existing no-scratch traversal when a
        // whole file-backed Q8 tensor can otherwise be read as one coalesced burst. This keeps
        // long-prefill/full-prompt probes from fragmenting 8B FFN reads solely to feed the
        // parallel output scratch path, while explicit scratch-budget overrides still win.
        return Ok(false);
    }
    Ok(parallelize_output)
}

fn q8_0_file_reader_chunk_rows(row_bytes_len: usize, output_width: usize) -> Result<usize> {
    const DEFAULT_Q8_0_FILE_READER_CHUNK_BYTES: usize = 32 * 1024 * 1024;
    if row_bytes_len == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "q8_0 borrowed block-reader row byte count must be non-zero".to_string(),
        ));
    }
    if output_width == 0 {
        return Ok(1);
    }
    let chunk_bytes = parse_byte_count_env("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES")
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_Q8_0_FILE_READER_CHUNK_BYTES);
    let budget_rows = (chunk_bytes / row_bytes_len).max(1).min(output_width);
    let tensor_bytes = row_bytes_len.checked_mul(output_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(
            "q8_0 file-reader tensor byte count overflow".to_string(),
        )
    })?;
    let two_chunk_bytes = chunk_bytes.saturating_mul(2);
    // Llama 3 8B Q8_0 FFN matrices sit just under two 32 MiB chunks; other exact
    // tracked Q8 shapes may land exactly on the two-chunk boundary. Reading those
    // one-or-two-chunk tensors as one bounded burst cuts one syscall/read phase per
    // tensor without changing file-backed output values or enabling the global Q8 cache.
    if budget_rows < output_width && tensor_bytes <= two_chunk_bytes {
        return Ok(output_width);
    }
    Ok(budget_rows)
}

fn q8_0_file_reader_chunk_rows_for_batch(
    row_bytes_len: usize,
    output_width: usize,
    input_rows: usize,
    uses_output_scratch: bool,
) -> Result<usize> {
    let weight_chunk_rows = q8_0_file_reader_chunk_rows(row_bytes_len, output_width)?;
    if input_rows <= 1 || output_width == 0 || !uses_output_scratch {
        return Ok(weight_chunk_rows);
    }
    let scratch_chunk_rows = q8_0_file_reader_output_scratch_chunk_rows(input_rows, output_width)?;
    Ok(weight_chunk_rows.min(scratch_chunk_rows).max(1))
}

fn q8_0_file_reader_output_scratch_chunk_rows(
    input_rows: usize,
    output_width: usize,
) -> Result<usize> {
    const DEFAULT_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
    if output_width == 0 {
        return Ok(1);
    }
    let scratch_bytes = parse_byte_count_env("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES")
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES);
    let bytes_per_output_row = input_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "q8_0 file-reader output scratch row byte count overflow".to_string(),
            )
        })?;
    Ok((scratch_bytes / bytes_per_output_row)
        .max(1)
        .min(output_width))
}

#[allow(dead_code)]
fn accumulate_transposed_linear_row_with_precision(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    precision: LinearAccumulationPrecision,
) {
    let runtime_plan = ResolvedRuntimePlan {
        linear_accumulation_precision: precision,
        q8: Q8RuntimeFlags::from_env(),
    };
    accumulate_transposed_linear_row_with_precision_with_plan(
        input_row,
        weight,
        output,
        &runtime_plan,
    )
}

fn accumulate_transposed_linear_row_with_precision_with_plan(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    runtime_plan: &ResolvedRuntimePlan,
) {
    if should_use_borrowed_q8_0_block_dot_with_plan(weight, input_row.len(), runtime_plan) {
        accumulate_transposed_linear_row_q8_0_block_dot_with_flags(
            input_row,
            weight,
            output,
            &runtime_plan.q8,
        );
        return;
    }
    if let Some((packed, interleave)) = q8_0_selected_borrowed_packed_rows4(weight) {
        if input_row.len().is_multiple_of(Q8_0_BLOCK_VALUES)
            && packed.rows == output.len()
            && packed.blocks_per_row == input_row.len() / Q8_0_BLOCK_VALUES
        {
            accumulate_q8_0_packed_rows4_f32_input(input_row, packed, interleave, output);
            return;
        }
    }
    match runtime_plan.linear_accumulation_precision {
        LinearAccumulationPrecision::F32 => {
            accumulate_transposed_linear_row(input_row, weight, output)
        }
        LinearAccumulationPrecision::F64 => {
            if should_parallelize_linear_output(output.len()) {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(out_idx, out_value)| {
                        let rhs_start = out_idx * input_row.len();
                        let rhs_row = &weight.data[rhs_start..rhs_start + input_row.len()];
                        *out_value = dot_product_row_f64(input_row, rhs_row);
                    });
            } else {
                for (out_idx, out_value) in output.iter_mut().enumerate() {
                    let rhs_start = out_idx * input_row.len();
                    let rhs_row = &weight.data[rhs_start..rhs_start + input_row.len()];
                    *out_value = dot_product_row_f64(input_row, rhs_row);
                }
            }
        }
    }
}

#[allow(dead_code)]
fn accumulate_transposed_linear_row_q8_0_block_dot(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
    let q8_flags = Q8RuntimeFlags::from_env();
    accumulate_transposed_linear_row_q8_0_block_dot_with_flags(input_row, weight, output, &q8_flags)
}

fn accumulate_transposed_linear_row_q8_0_block_dot_with_flags(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    q8_flags: &Q8RuntimeFlags,
) {
    let quantized_input = quantize_q8_0_row(input_row);
    accumulate_transposed_linear_row_q8_0_block_dot_quantized_with_flags(
        &quantized_input.blocks,
        weight,
        output,
        q8_flags,
    );
}

#[allow(dead_code)]
fn accumulate_transposed_linear_row_q8_0_block_dot_quantized(
    quantized_input: &[Q8_0Block],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
    let q8_flags = Q8RuntimeFlags::from_env();
    accumulate_transposed_linear_row_q8_0_block_dot_quantized_with_flags(
        quantized_input,
        weight,
        output,
        &q8_flags,
    )
}

fn accumulate_transposed_linear_row_q8_0_block_dot_quantized_with_flags(
    quantized_input: &[Q8_0Block],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    q8_flags: &Q8RuntimeFlags,
) {
    let blocks_per_row = quantized_input.len();
    if let Some((packed, interleave)) = q8_0_selected_borrowed_packed_rows4(weight) {
        if packed.rows == output.len() && packed.blocks_per_row == blocks_per_row {
            accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                quantized_input,
                packed,
                interleave,
                output,
            );
            return;
        }
    }
    let weight_blocks = weight
        .q8_0_blocks
        .expect("q8_0 block-dot precondition checked");
    debug_assert_eq!(weight_blocks.len(), output.len() * blocks_per_row);
    if q8_flags.hybrid_retained {
        let gpu_rows = q8_flags.hybrid_gpu_rows_for_output(output.len());
        if gpu_rows > 0 && gpu_rows < output.len() {
            let cpu_rows = output.len() - gpu_rows;
            let gpu_block_start = cpu_rows * blocks_per_row;
            let (cpu_output, gpu_output) = output.split_at_mut(cpu_rows);
            let cpu_weight_blocks = &weight_blocks[..gpu_block_start];
            let gpu_weight_blocks = &weight_blocks[gpu_block_start..];
            let gpu_weight_bytes = q8_0_blocks_as_bytes(gpu_weight_blocks);
            if with_q8_0_block_scales_and_quants(quantized_input, |input_scales, input_quants| {
                metal::try_q8_0_block_linear_row_with_cpu(
                    input_scales,
                    input_quants,
                    gpu_weight_bytes,
                    gpu_rows,
                    blocks_per_row,
                    gpu_output,
                    || {
                        accumulate_q8_0_block_dot_quantized_cpu(
                            quantized_input,
                            cpu_weight_blocks,
                            cpu_output,
                        )
                    },
                )
            }) {
                trace_q8_0_hybrid_retained_success(cpu_rows, gpu_rows, blocks_per_row);
                return;
            }
        }
    }
    if q8_flags.metal_retained {
        let weight_bytes = q8_0_blocks_as_bytes(weight_blocks);
        if with_q8_0_block_scales_and_quants(quantized_input, |input_scales, input_quants| {
            metal::try_q8_0_block_linear_row(
                input_scales,
                input_quants,
                weight_bytes,
                output.len(),
                blocks_per_row,
                output,
            )
        }) {
            return;
        }
    }
    accumulate_q8_0_block_dot_quantized_cpu(quantized_input, weight_blocks, output);
}

fn q8_0_packed_4x4_dot_enabled() -> bool {
    env_flag_enabled("CAMELID_Q8_0_PACKED_4X4_DOT")
}

fn q8_0_packed_4x8_dot_enabled() -> bool {
    env_flag_enabled("CAMELID_Q8_0_PACKED_4X8_DOT")
}

fn q8_0_selected_packed_rows4(
    weight: &CpuTensor,
) -> Option<(&Q8_0PackedRows4, Q8_0PackedRows4Interleave)> {
    if let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = weight.q8_0_runtime_storage.as_ref() {
        return Some((packed, packed.interleave));
    }
    if q8_0_packed_4x8_dot_enabled() {
        if let Some(packed) = weight.q8_0_packed_rows4_4x8.as_ref() {
            return Some((packed, Q8_0PackedRows4Interleave::I8));
        }
    }
    if q8_0_packed_4x4_dot_enabled() {
        if let Some(packed) = weight.q8_0_packed_rows4_4x4.as_ref() {
            return Some((packed, Q8_0PackedRows4Interleave::I4));
        }
    }
    None
}

fn q8_0_selected_borrowed_packed_rows4(
    weight: BorrowedLinearWeight<'_>,
) -> Option<(&Q8_0PackedRows4, Q8_0PackedRows4Interleave)> {
    if let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = weight.q8_0_runtime_storage {
        return Some((packed, packed.interleave));
    }
    if q8_0_packed_4x8_dot_enabled() {
        if let Some(packed) = weight.q8_0_packed_rows4_4x8 {
            return Some((packed, Q8_0PackedRows4Interleave::I8));
        }
    }
    if q8_0_packed_4x4_dot_enabled() {
        if let Some(packed) = weight.q8_0_packed_rows4_4x4 {
            return Some((packed, Q8_0PackedRows4Interleave::I4));
        }
    }
    None
}

fn accumulate_q8_0_packed_rows4_dot_quantized_cpu(
    quantized_input: &[Q8_0Block],
    packed: &Q8_0PackedRows4,
    interleave: Q8_0PackedRows4Interleave,
    output: &mut [f32],
) {
    let blocks_per_row = quantized_input.len();
    debug_assert_eq!(packed.blocks_per_row, blocks_per_row);
    debug_assert_eq!(packed.rows, output.len());
    debug_assert!(output.len().is_multiple_of(4));

    if should_parallelize_linear_output(output.len()) {
        output
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(group_idx, output_chunk)| {
                accumulate_q8_0_packed_rows4_output_group(
                    quantized_input,
                    packed,
                    interleave,
                    blocks_per_row,
                    group_idx,
                    output_chunk,
                )
            });
        return;
    }

    accumulate_q8_0_packed_rows4_dot_quantized_cpu_serial(
        quantized_input,
        packed,
        interleave,
        output,
    );
}

fn accumulate_q8_0_packed_rows4_dot_quantized_cpu_serial(
    quantized_input: &[Q8_0Block],
    packed: &Q8_0PackedRows4,
    interleave: Q8_0PackedRows4Interleave,
    output: &mut [f32],
) {
    let blocks_per_row = quantized_input.len();
    debug_assert_eq!(packed.blocks_per_row, blocks_per_row);
    debug_assert_eq!(packed.rows, output.len());
    debug_assert!(output.len().is_multiple_of(4));

    for (group_idx, output_chunk) in output.chunks_mut(4).enumerate() {
        accumulate_q8_0_packed_rows4_output_group(
            quantized_input,
            packed,
            interleave,
            blocks_per_row,
            group_idx,
            output_chunk,
        );
    }
}

fn accumulate_q8_0_packed_rows4_output_group(
    quantized_input: &[Q8_0Block],
    packed: &Q8_0PackedRows4,
    interleave: Q8_0PackedRows4Interleave,
    blocks_per_row: usize,
    group_idx: usize,
    output_chunk: &mut [f32],
) {
    debug_assert_eq!(output_chunk.len(), 4);
    let group_blocks = &packed.blocks[group_idx * blocks_per_row..(group_idx + 1) * blocks_per_row];
    let sums = q8_0_packed_rows4_dot(group_blocks, quantized_input, interleave);
    output_chunk.copy_from_slice(&sums);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mac_q8_prefill_i8mm_enabled() -> bool {
    env_flag_enabled("CAMELID_MAC_Q8_PREFILL_I8MM")
        && std::arch::is_aarch64_feature_detected!("i8mm")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mac_q8_prefill_i8mm_row_threshold_met(rows: usize) -> bool {
    rows >= MAC_Q8_PREFILL_I8MM_MIN_ROWS
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mac_q8_sched_packed_prefill_enabled() -> bool {
    env::var("CAMELID_MAC_Q8_SCHED")
        .map(|value| value.trim().eq_ignore_ascii_case("packed_prefill"))
        .unwrap_or(false)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn matmul_rhs_transposed_q8_0_packed_rows4_prefill_i8mm(
    input: &CpuTensor,
    packed_weight: &Q8_0PackedRows4,
    output_width: usize,
    q8_role: &str,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let rows = input.dim(0)?;
    let input_width = input.dim(1)?;
    if !input_width.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 prefill i8mm input width {input_width} is not a multiple of {Q8_0_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    if packed_weight.rows != output_width || packed_weight.blocks_per_row != blocks_per_row {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 prefill i8mm shape mismatch: packed rows={}, blocks_per_row={}, requested output_width={output_width}, input blocks_per_row={blocks_per_row}",
            packed_weight.rows, packed_weight.blocks_per_row
        )));
    }

    let mut output = vec![0.0_f32; rows * output_width];
    let packed_rows = rows / 4 * 4;
    let collect_q8_schedule = q8_schedule_telemetry_enabled();
    if collect_q8_schedule {
        let row_groups = (packed_rows / 4) as u64;
        add_q8_schedule_counter(&Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS, 1);
        record_q8_schedule_i8mm_single_projection_role_call(q8_role, rows as u64, row_groups);
        if q8_role == "ffn_down" && mac_q8_ffn_down_single_projection_scheduler_counters_enabled() {
            let output_groups = (output_width / 4) as u64;
            record_q8_schedule_i8mm_single_projection_role_scheduler(
                q8_role,
                output_groups,
                row_groups,
                1,
            );
        }
    }
    with_q8_0_file_reader_quantized_inputs(|quantized_inputs| {
        quantized_inputs.clear();
        Q8_0_PREFILL_PACKED_INPUTS.with(|cell| {
            let mut packed_inputs = cell.borrow_mut();
            packed_inputs.clear();
            let before_capacity = packed_inputs.capacity();
            let pack_started = collect_q8_schedule.then(Instant::now);
            quantize_pack_q8_0_rows4_i8_direct_into(
                &input.data[..packed_rows * input_width],
                packed_rows,
                input_width,
                blocks_per_row,
                &mut packed_inputs,
            );
            if let Some(pack_started) = pack_started {
                let elapsed_us = pack_started.elapsed().as_micros();
                record_q8_schedule_activation_pack(
                    &mut packed_inputs,
                    before_capacity,
                    packed_rows,
                    blocks_per_row,
                    elapsed_us,
                );
                record_q8_schedule_i8mm_single_projection_role_pack(q8_role, elapsed_us);
            }
            let gemm_started = collect_q8_schedule.then(Instant::now);
            run_q8_0_packed_rows4_prefill_i8mm_kernel(
                packed_weight,
                &packed_inputs,
                packed_rows / 4,
                &mut output,
                collect_q8_schedule,
            );
            if let Some(gemm_started) = gemm_started {
                let elapsed_us = gemm_started.elapsed().as_micros();
                add_q8_schedule_counter(&Q8_SCHED_Q8_GEMM_COMPUTE_US, elapsed_us as u64);
                record_q8_schedule_i8mm_single_projection_role_gemm(q8_role, elapsed_us);
            }
            packed_inputs.clear();
            cap_q8_0_file_reader_scratch(&mut packed_inputs, 0);
        });

        let tail_rows = rows - packed_rows;
        if collect_q8_schedule {
            add_q8_schedule_counter(&Q8_SCHED_CONSERVATIVE_TAIL_ROWS, tail_rows as u64);
            record_q8_schedule_i8mm_single_projection_role_tail(q8_role, tail_rows as u64);
        }
        if tail_rows > 0 {
            quantized_inputs.reserve(tail_rows * blocks_per_row);
            for row in input.data[packed_rows * input_width..].chunks_exact(input_width) {
                quantize_q8_0_blocks_into(row, quantized_inputs);
            }
        }
        for tail_row in 0..tail_rows {
            let input_start = tail_row * blocks_per_row;
            let output_start = (packed_rows + tail_row) * output_width;
            accumulate_q8_0_packed_rows4_dot_quantized_cpu(
                &quantized_inputs[input_start..input_start + blocks_per_row],
                packed_weight,
                Q8_0PackedRows4Interleave::I8,
                &mut output[output_start..output_start + output_width],
            );
        }
        Ok(())
    })?;

    CpuTensor::from_f32(name, vec![rows, output_width], output)
}

fn quantize_pack_q8_0_rows4_i8_direct_into(
    row_major_input: &[f32],
    rows_to_pack: usize,
    input_width: usize,
    blocks_per_row: usize,
    output: &mut Vec<Q8_0PackedRows4Block>,
) {
    debug_assert!(rows_to_pack.is_multiple_of(4));
    debug_assert_eq!(row_major_input.len(), rows_to_pack * input_width);
    debug_assert_eq!(input_width, blocks_per_row * Q8_0_BLOCK_VALUES);
    output.reserve((rows_to_pack / 4) * blocks_per_row);
    for row_group in (0..rows_to_pack).step_by(4) {
        for block_idx in 0..blocks_per_row {
            let mut scales = [0.0_f32; 4];
            let mut quants = [0_i8; 128];
            for (lane, scale) in scales.iter_mut().enumerate() {
                let row_start = (row_group + lane) * input_width;
                let block_start = row_start + block_idx * Q8_0_BLOCK_VALUES;
                let block = quantize_q8_0_block(
                    &row_major_input[block_start..block_start + Q8_0_BLOCK_VALUES],
                );
                *scale = block.scale;
                for chunk in 0..4 {
                    let src_start = chunk * 8;
                    let dst_start = chunk * 32 + lane * 8;
                    quants[dst_start..dst_start + 8]
                        .copy_from_slice(&block.quants[src_start..src_start + 8]);
                }
            }
            output.push(Q8_0PackedRows4Block { scales, quants });
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run_q8_0_packed_rows4_prefill_i8mm_kernel(
    packed_weight: &Q8_0PackedRows4,
    packed_inputs: &[Q8_0PackedRows4Block],
    input_groups: usize,
    output: &mut [f32],
    collect_q8_schedule: bool,
) {
    if collect_q8_schedule {
        add_q8_schedule_counter(&Q8_SCHED_RAYON_FANOUT_BOUNDARIES, input_groups as u64);
    }
    let rows = packed_weight.rows;
    let blocks_per_row = packed_weight.blocks_per_row;
    debug_assert_eq!(packed_inputs.len(), input_groups * blocks_per_row);
    for input_group in 0..input_groups {
        let input_blocks =
            &packed_inputs[input_group * blocks_per_row..(input_group + 1) * blocks_per_row];
        let group_output = &mut output[input_group * 4 * rows..(input_group + 1) * 4 * rows];
        let (row0, rest) = group_output.split_at_mut(rows);
        let (row1, rest) = rest.split_at_mut(rows);
        let (row2, row3) = rest.split_at_mut(rows);
        row0.par_chunks_mut(4)
            .zip(row1.par_chunks_mut(4))
            .zip(row2.par_chunks_mut(4))
            .zip(row3.par_chunks_mut(4))
            .enumerate()
            .for_each(
                |(output_group, (((row0_chunk, row1_chunk), row2_chunk), row3_chunk))| {
                    let weight_group = &packed_weight.blocks
                        [output_group * blocks_per_row..(output_group + 1) * blocks_per_row];
                    let mut sums = [[0.0_f32; 4]; 4];
                    for (input_block, weight_block) in input_blocks.iter().zip(weight_group) {
                        // SAFETY: mac_q8_prefill_i8mm_enabled checked runtime I8MM support before this path;
                        // both operands are q8_0_4x8 packed blocks with 4 rows/columns and 32 K values.
                        let int_sums = unsafe {
                            q8_0_packed_4x8_gemm4_block_i8mm(
                                input_block.quants.as_ptr(),
                                weight_block.quants.as_ptr(),
                            )
                        };
                        for input_lane in 0..4 {
                            for output_lane in 0..4 {
                                sums[input_lane][output_lane] += int_sums[input_lane][output_lane]
                                    as f32
                                    * weight_block.scales[output_lane]
                                    * input_block.scales[input_lane];
                            }
                        }
                    }
                    row0_chunk.copy_from_slice(&sums[0]);
                    row1_chunk.copy_from_slice(&sums[1]);
                    row2_chunk.copy_from_slice(&sums[2]);
                    row3_chunk.copy_from_slice(&sums[3]);
                },
            );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run_q8_0_packed_rows4_prefill_i8mm_two_kernel(
    gate_weight: &Q8_0PackedRows4,
    up_weight: &Q8_0PackedRows4,
    packed_inputs: &[Q8_0PackedRows4Block],
    input_groups: usize,
    gate_output: &mut [f32],
    up_output: &mut [f32],
    collect_q8_schedule: bool,
) {
    let rows = gate_weight.rows;
    let blocks_per_row = gate_weight.blocks_per_row;
    debug_assert_eq!(up_weight.rows, rows);
    debug_assert_eq!(up_weight.blocks_per_row, blocks_per_row);
    debug_assert_eq!(packed_inputs.len(), input_groups * blocks_per_row);
    debug_assert!(gate_output.len() >= input_groups * 4 * rows);
    debug_assert!(up_output.len() >= input_groups * 4 * rows);
    if collect_q8_schedule {
        add_q8_schedule_counter(&Q8_SCHED_RAYON_FANOUT_BOUNDARIES, input_groups as u64);
    }
    for input_group in 0..input_groups {
        let input_blocks =
            &packed_inputs[input_group * blocks_per_row..(input_group + 1) * blocks_per_row];
        let gate_group_output =
            &mut gate_output[input_group * 4 * rows..(input_group + 1) * 4 * rows];
        let up_group_output = &mut up_output[input_group * 4 * rows..(input_group + 1) * 4 * rows];
        let gate_base = gate_group_output.as_mut_ptr() as usize;
        let up_base = up_group_output.as_mut_ptr() as usize;
        (0..rows / 4).into_par_iter().for_each(|output_group| {
            let gate_weight_group = &gate_weight.blocks
                [output_group * blocks_per_row..(output_group + 1) * blocks_per_row];
            let up_weight_group = &up_weight.blocks
                [output_group * blocks_per_row..(output_group + 1) * blocks_per_row];
            let mut gate_sums = [[0.0_f32; 4]; 4];
            let mut up_sums = [[0.0_f32; 4]; 4];
            for ((input_block, gate_block), up_block) in input_blocks
                .iter()
                .zip(gate_weight_group)
                .zip(up_weight_group)
            {
                // SAFETY: mac_q8_prefill_i8mm_enabled checked runtime I8MM support before this path;
                // all operands are q8_0_4x8 packed blocks with 4 rows/columns and 32 K values.
                let gate_int_sums = unsafe {
                    q8_0_packed_4x8_gemm4_block_i8mm(
                        input_block.quants.as_ptr(),
                        gate_block.quants.as_ptr(),
                    )
                };
                // SAFETY: same I8MM/layout preconditions as gate_int_sums above.
                let up_int_sums = unsafe {
                    q8_0_packed_4x8_gemm4_block_i8mm(
                        input_block.quants.as_ptr(),
                        up_block.quants.as_ptr(),
                    )
                };
                for input_lane in 0..4 {
                    for output_lane in 0..4 {
                        let input_scale = input_block.scales[input_lane];
                        gate_sums[input_lane][output_lane] += gate_int_sums[input_lane][output_lane]
                            as f32
                            * gate_block.scales[output_lane]
                            * input_scale;
                        up_sums[input_lane][output_lane] += up_int_sums[input_lane][output_lane]
                            as f32
                            * up_block.scales[output_lane]
                            * input_scale;
                    }
                }
            }
            let output_start = output_group * 4;
            for lane in 0..4 {
                // SAFETY: each parallel output_group writes a disjoint 4-column range in
                // each row lane of this input group; gate_base/up_base point to the unique
                // mutable slices above and remain valid for the duration of this scope.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        gate_sums[lane].as_ptr(),
                        (gate_base as *mut f32).add(lane * rows + output_start),
                        4,
                    );
                    std::ptr::copy_nonoverlapping(
                        up_sums[lane].as_ptr(),
                        (up_base as *mut f32).add(lane * rows + output_start),
                        4,
                    );
                }
            }
        });
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[target_feature(enable = "i8mm")]
unsafe fn q8_0_packed_4x8_gemm4_block_i8mm(
    input_quants: *const i8,
    weight_quants: *const i8,
) -> [[i32; 4]; 4] {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    for chunk in 0..4 {
        let offset = chunk * 32;
        // SAFETY: callers provide complete 128-byte q8_0_4x8 quant arrays.
        let a01 = unsafe { vld1q_s8(input_quants.add(offset)) };
        let a23 = unsafe { vld1q_s8(input_quants.add(offset + 16)) };
        let b01 = unsafe { vld1q_s8(weight_quants.add(offset)) };
        let b23 = unsafe { vld1q_s8(weight_quants.add(offset + 16)) };
        unsafe {
            asm!(
                "smmla {acc0:v}.4s, {a01:v}.16b, {b01:v}.16b",
                "smmla {acc1:v}.4s, {a01:v}.16b, {b23:v}.16b",
                "smmla {acc2:v}.4s, {a23:v}.16b, {b01:v}.16b",
                "smmla {acc3:v}.4s, {a23:v}.16b, {b23:v}.16b",
                acc0 = inout(vreg) acc0,
                acc1 = inout(vreg) acc1,
                acc2 = inout(vreg) acc2,
                acc3 = inout(vreg) acc3,
                a01 = in(vreg) a01,
                a23 = in(vreg) a23,
                b01 = in(vreg) b01,
                b23 = in(vreg) b23,
                options(nostack, preserves_flags)
            );
        }
    }
    let acc0 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc0) };
    let acc1 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc1) };
    let acc2 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc2) };
    let acc3 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc3) };
    [
        [acc0[0], acc0[1], acc1[0], acc1[1]],
        [acc0[2], acc0[3], acc1[2], acc1[3]],
        [acc2[0], acc2[1], acc3[0], acc3[1]],
        [acc2[2], acc2[3], acc3[2], acc3[3]],
    ]
}

fn accumulate_q8_0_packed_rows4_f32_input(
    input_row: &[f32],
    packed: &Q8_0PackedRows4,
    interleave: Q8_0PackedRows4Interleave,
    output: &mut [f32],
) {
    let blocks_per_row = input_row.len() / Q8_0_BLOCK_VALUES;
    debug_assert_eq!(packed.blocks_per_row, blocks_per_row);
    debug_assert_eq!(packed.rows, output.len());
    debug_assert!(output.len().is_multiple_of(4));

    let block_len = interleave.block_len();
    let compute_group = |group_idx: usize, output_chunk: &mut [f32]| {
        debug_assert_eq!(output_chunk.len(), 4);
        let mut sums = [0.0_f32; 4];
        let group_blocks =
            &packed.blocks[group_idx * blocks_per_row..(group_idx + 1) * blocks_per_row];
        for (block_idx, packed_block) in group_blocks.iter().enumerate() {
            let input_block_start = block_idx * Q8_0_BLOCK_VALUES;
            for idx in 0..Q8_0_BLOCK_VALUES {
                let chunk = idx / block_len;
                let lane_offset = idx % block_len;
                let packed_chunk_start = chunk * 4 * block_len;
                let input_value = input_row[input_block_start + idx];
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let packed_idx = packed_chunk_start + lane * block_len + lane_offset;
                    *sum += input_value
                        * f32::from(packed_block.quants[packed_idx])
                        * packed_block.scales[lane];
                }
            }
        }
        output_chunk.copy_from_slice(&sums);
    };

    if should_parallelize_linear_output(output.len()) {
        output
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(group_idx, output_chunk)| compute_group(group_idx, output_chunk));
        return;
    }

    for (group_idx, output_chunk) in output.chunks_mut(4).enumerate() {
        compute_group(group_idx, output_chunk);
    }
}

fn q8_0_packed_rows4_dot_i8_matmul(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
    use_hoisted_avx2: bool,
) -> [f32; 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled() {
            // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI support.
            return unsafe { q8_0_packed_rows4_dot_i8_avx512vnni_dpbusd(packed_blocks, input) };
        }
        if x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled() {
            // SAFETY: runtime feature detection in
            // `x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled` confirms support.
            return unsafe { q8_0_packed_rows4_dot_i8_avx512vnni_dpwssd(packed_blocks, input) };
        }
        if use_hoisted_avx2 {
            // SAFETY: `use_hoisted_avx2` is only true after runtime AVX2 detection.
            return unsafe { q8_0_packed_rows4_dot_i8_avx2(packed_blocks, input) };
        }
    }
    let _ = use_hoisted_avx2;
    q8_0_packed_rows4_dot(packed_blocks, input, Q8_0PackedRows4Interleave::I8)
}

fn q8_0_packed_rows4_dot_i8_matmul_pair(
    left_packed_blocks: &[Q8_0PackedRows4Block],
    right_packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
    use_hoisted_avx2: bool,
) -> ([f32; 4], [f32; 4]) {
    debug_assert_eq!(left_packed_blocks.len(), input.len());
    debug_assert_eq!(right_packed_blocks.len(), input.len());
    let mut left_sums = [0.0_f32; 4];
    let mut right_sums = [0.0_f32; 4];
    for ((left_block, right_block), input_block) in left_packed_blocks
        .iter()
        .zip(right_packed_blocks)
        .zip(input)
    {
        let (left_int_sums, right_int_sums) = q8_0_packed_rows4_block_dot_i8_pair(
            left_block,
            right_block,
            input_block,
            use_hoisted_avx2,
        );
        let input_scale = input_block.scale;
        for lane in 0..4 {
            left_sums[lane] += left_int_sums[lane] as f32 * left_block.scales[lane] * input_scale;
            right_sums[lane] +=
                right_int_sums[lane] as f32 * right_block.scales[lane] * input_scale;
        }
    }
    (left_sums, right_sums)
}

fn q8_0_packed_rows4_block_dot_i8_pair(
    left_block: &Q8_0PackedRows4Block,
    right_block: &Q8_0PackedRows4Block,
    input_block: &Q8_0Block,
    use_hoisted_avx2: bool,
) -> ([i32; 4], [i32; 4]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled() {
            // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI support.
            return unsafe {
                (
                    q8_0_packed_4x8_block_avx512vnni_dpbusd(
                        left_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                    q8_0_packed_4x8_block_avx512vnni_dpbusd(
                        right_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                )
            };
        }
        if x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled() {
            // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI and AVX2 support.
            return unsafe {
                (
                    q8_0_packed_4x8_block_avx512vnni_dpwssd(
                        left_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                    q8_0_packed_4x8_block_avx512vnni_dpwssd(
                        right_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                )
            };
        }
        if (use_hoisted_avx2
            || x86_q8_packed_rows4_avx2_dot_enabled()
            || x86_q8_kernel_avx2_enabled())
            && std::arch::is_x86_feature_detected!("avx2")
        {
            // SAFETY: runtime feature detection confirms AVX2 support.
            return unsafe {
                (
                    q8_0_packed_4x8_block_avx2(
                        left_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                    q8_0_packed_4x8_block_avx2(
                        right_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                )
            };
        }
    }
    let _ = use_hoisted_avx2;
    (
        q8_0_packed_rows4_block_dot_scalar(
            &left_block.quants,
            &input_block.quants,
            Q8_0PackedRows4Interleave::I8,
        ),
        q8_0_packed_rows4_block_dot_scalar(
            &right_block.quants,
            &input_block.quants,
            Q8_0PackedRows4Interleave::I8,
        ),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_rows4_dot_i8_avx512vnni_dpbusd(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        let int_sums = unsafe {
            q8_0_packed_4x8_block_avx512vnni_dpbusd(
                packed_block.quants.as_ptr(),
                input_block.quants.as_ptr(),
            )
        };
        let input_scale = input_block.scale;
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_scale;
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_rows4_dot_i8_avx512vnni_dpwssd(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        let int_sums = unsafe {
            q8_0_packed_4x8_block_avx512vnni_dpwssd(
                packed_block.quants.as_ptr(),
                input_block.quants.as_ptr(),
            )
        };
        let input_scale = input_block.scale;
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_scale;
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_packed_rows4_dot_i8_avx2(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        let int_sums = unsafe {
            q8_0_packed_4x8_block_avx2(packed_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        let input_scale = input_block.scale;
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_scale;
        }
    }
    sums
}

fn q8_0_packed_rows4_dot(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
    interleave: Q8_0PackedRows4Interleave,
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let int_sums = if aarch64_dotprod_enabled() {
            // SAFETY: runtime feature detection confirms dot-product support; packed quants
            // contain 128 i8 values and input quants contain 32 contiguous i8 values.
            unsafe {
                match interleave {
                    Q8_0PackedRows4Interleave::I4 => q8_0_packed_4x4_block_dotprod(
                        packed_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                    Q8_0PackedRows4Interleave::I8 => q8_0_packed_4x8_block_dotprod(
                        packed_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                }
            }
        } else {
            q8_0_packed_rows4_block_dot_scalar(
                &packed_block.quants,
                &input_block.quants,
                interleave,
            )
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let int_sums = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if interleave == Q8_0PackedRows4Interleave::I8
                    && x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled()
                {
                    // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI support;
                    // packed quants contain one complete rows4/I8 block and input quants
                    // contain one Q8_0 block.
                    unsafe {
                        q8_0_packed_4x8_block_avx512vnni_dpbusd(
                            packed_block.quants.as_ptr(),
                            input_block.quants.as_ptr(),
                        )
                    }
                } else if interleave == Q8_0PackedRows4Interleave::I8
                    && x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled()
                {
                    // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI and AVX2
                    // support; packed quants contain one complete rows4/I8 block and input
                    // quants contain one Q8_0 block.
                    unsafe {
                        q8_0_packed_4x8_block_avx512vnni_dpwssd(
                            packed_block.quants.as_ptr(),
                            input_block.quants.as_ptr(),
                        )
                    }
                } else if interleave == Q8_0PackedRows4Interleave::I8
                    && (x86_q8_packed_rows4_avx2_dot_enabled() || x86_q8_kernel_avx2_enabled())
                    && std::arch::is_x86_feature_detected!("avx2")
                {
                    // SAFETY: runtime feature detection confirms AVX2 support; packed quants
                    // contain one complete rows4/I8 block and input quants contain one Q8_0 block.
                    unsafe {
                        q8_0_packed_4x8_block_avx2(
                            packed_block.quants.as_ptr(),
                            input_block.quants.as_ptr(),
                        )
                    }
                } else {
                    q8_0_packed_rows4_block_dot_scalar(
                        &packed_block.quants,
                        &input_block.quants,
                        interleave,
                    )
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                q8_0_packed_rows4_block_dot_scalar(
                    &packed_block.quants,
                    &input_block.quants,
                    interleave,
                )
            }
        };
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_block.scale;
        }
    }
    sums
}

fn q8_0_packed_rows4_block_dot_scalar(
    packed: &[i8; 128],
    input: &[i8; 32],
    interleave: Q8_0PackedRows4Interleave,
) -> [i32; 4] {
    let block_len = interleave.block_len();
    let chunks = 32 / block_len;
    let mut sums = [0_i32; 4];
    for chunk in 0..chunks {
        for lane in 0..4 {
            for idx in 0..block_len {
                sums[lane] += i32::from(packed[chunk * 4 * block_len + lane * block_len + idx])
                    * i32::from(input[chunk * block_len + idx]);
            }
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_4x8_block_avx512vnni_dpbusd(packed: *const i8, input: *const i8) -> [i32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm512_abs_epi8, _mm512_cmplt_epi8_mask, _mm512_dpbusd_epi32, _mm512_loadu_si512,
        _mm512_mask_mov_epi8, _mm512_set_epi64, _mm512_setzero_si512, _mm512_storeu_si512,
        _mm512_sub_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm512_abs_epi8, _mm512_cmplt_epi8_mask, _mm512_dpbusd_epi32, _mm512_loadu_si512,
        _mm512_mask_mov_epi8, _mm512_set_epi64, _mm512_setzero_si512, _mm512_storeu_si512,
        _mm512_sub_epi8,
    };

    let zero = _mm512_setzero_si512();
    let mut acc = zero;
    for pair in 0..2usize {
        let chunk = pair * 2;
        let packed64 = unsafe { _mm512_loadu_si512(packed.add(chunk * 32).cast()) };
        let low_input = unsafe { std::ptr::read_unaligned(input.add(chunk * 8).cast::<i64>()) };
        let high_input =
            unsafe { std::ptr::read_unaligned(input.add((chunk + 1) * 8).cast::<i64>()) };
        let input64 = _mm512_set_epi64(
            high_input, high_input, high_input, high_input, low_input, low_input, low_input,
            low_input,
        );

        // Mirror llama.cpp Q8_0 VNNI dot strategy: convert signed*signed bytes into
        // unsigned(abs(weight)) * signed(input adjusted by weight sign), then use DPBUSD.
        let abs_packed = _mm512_abs_epi8(packed64);
        let neg_input = _mm512_sub_epi8(zero, input64);
        let negative_weight_mask = _mm512_cmplt_epi8_mask(packed64, zero);
        let signed_input = _mm512_mask_mov_epi8(input64, negative_weight_mask, neg_input);
        acc = _mm512_dpbusd_epi32(acc, abs_packed, signed_input);
    }

    let mut lanes = [0_i32; 16];
    unsafe {
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc);
    }
    [
        lanes[0] + lanes[1] + lanes[8] + lanes[9],
        lanes[2] + lanes[3] + lanes[10] + lanes[11],
        lanes[4] + lanes[5] + lanes[12] + lanes[13],
        lanes[6] + lanes[7] + lanes[14] + lanes[15],
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_4x8_block_avx512vnni_dpwssd(packed: *const i8, input: *const i8) -> [i32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_dpwssd_epi32,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_dpwssd_epi32,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };

    let mut acc = _mm512_setzero_si512();
    for chunk in 0..4usize {
        let packed32 = unsafe { _mm256_loadu_si256(packed.add(chunk * 32).cast()) };
        let packed_i16 = _mm512_cvtepi8_epi16(packed32);

        let input8 = unsafe { _mm_loadl_epi64(input.add(chunk * 8).cast()) };
        let input16 = _mm_unpacklo_epi64(input8, input8);
        let input32 = _mm256_broadcastsi128_si256(input16);
        let input_i16 = _mm512_cvtepi8_epi16(input32);

        acc = _mm512_dpwssd_epi32(acc, packed_i16, input_i16);
    }

    let mut lanes = [0_i32; 16];
    unsafe {
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc);
    }
    [
        lanes[0..4].iter().sum(),
        lanes[4..8].iter().sum(),
        lanes[8..12].iter().sum(),
        lanes[12..16].iter().sum(),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_packed_4x8_block_avx2(packed: *const i8, input: *const i8) -> [i32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_epi32, _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sign_epi8,
        _mm256_storeu_si256, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sign_epi8,
        _mm256_storeu_si256, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };

    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for chunk in 0..4usize {
        let chunk_start = chunk * 32;
        let packed32 = unsafe { _mm256_loadu_si256(packed.add(chunk_start).cast()) };
        // Signed i8 x signed i8 via the same abs/sign lowering used by llama.cpp's
        // q8_0 dot path: abs(lhs) as unsigned bytes, sign(rhs, lhs), then
        // maddubs+madd. This avoids the old widen-to-i16 + mullo sequence.
        let abs_packed = _mm256_sign_epi8(packed32, packed32);
        let input8 = unsafe { _mm_loadl_epi64(input.add(chunk * 8).cast()) };
        let input32 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(input8, input8));
        let signed_input = _mm256_sign_epi8(input32, packed32);
        acc = _mm256_add_epi32(
            acc,
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_packed, signed_input), ones),
        );
    }

    let mut lanes = [0_i32; 8];
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
    }
    [
        lanes[0] + lanes[1],
        lanes[2] + lanes[3],
        lanes[4] + lanes[5],
        lanes[6] + lanes[7],
    ]
}

fn accumulate_q8_0_block_dot_quantized_cpu(
    quantized_input: &[Q8_0Block],
    weight_blocks: &[Q8_0Block],
    output: &mut [f32],
) {
    let blocks_per_row = quantized_input.len();
    debug_assert_eq!(weight_blocks.len(), output.len() * blocks_per_row);
    if should_parallelize_linear_output(output.len()) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_value)| {
                let weight_start = out_idx * blocks_per_row;
                *out_value = q8_0_dot_rows(
                    &weight_blocks[weight_start..weight_start + blocks_per_row],
                    quantized_input,
                );
            });
        return;
    }
    for (out_idx, out_value) in output.iter_mut().enumerate() {
        let weight_start = out_idx * blocks_per_row;
        *out_value = q8_0_dot_rows(
            &weight_blocks[weight_start..weight_start + blocks_per_row],
            quantized_input,
        );
    }
}

fn accumulate_two_q8_0_block_dot_quantized_cpu(
    quantized_input: &[Q8_0Block],
    first_weight_blocks: &[Q8_0Block],
    first_output: &mut [f32],
    second_weight_blocks: &[Q8_0Block],
    second_output: &mut [f32],
) {
    let blocks_per_row = quantized_input.len();
    debug_assert_eq!(first_output.len(), second_output.len());
    debug_assert_eq!(
        first_weight_blocks.len(),
        first_output.len() * blocks_per_row
    );
    debug_assert_eq!(
        second_weight_blocks.len(),
        second_output.len() * blocks_per_row
    );
    if should_parallelize_linear_output(first_output.len()) {
        first_output
            .par_iter_mut()
            .zip(second_output.par_iter_mut())
            .enumerate()
            .for_each(|(out_idx, (first_value, second_value))| {
                let weight_start = out_idx * blocks_per_row;
                let weight_end = weight_start + blocks_per_row;
                let (first_sum, second_sum) = q8_0_two_dot_rows(
                    &first_weight_blocks[weight_start..weight_end],
                    &second_weight_blocks[weight_start..weight_end],
                    quantized_input,
                );
                *first_value = first_sum;
                *second_value = second_sum;
            });
        return;
    }
    for (out_idx, (first_value, second_value)) in first_output
        .iter_mut()
        .zip(second_output.iter_mut())
        .enumerate()
    {
        let weight_start = out_idx * blocks_per_row;
        let weight_end = weight_start + blocks_per_row;
        let (first_sum, second_sum) = q8_0_two_dot_rows(
            &first_weight_blocks[weight_start..weight_end],
            &second_weight_blocks[weight_start..weight_end],
            quantized_input,
        );
        *first_value = first_sum;
        *second_value = second_sum;
    }
}

fn accumulate_transposed_linear_row(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
    if try_accumulate_transposed_linear_row_metal(input_row, weight, output) {
        return;
    }
    if should_parallelize_linear_output(output.len()) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_value)| {
                let rhs_start = out_idx * input_row.len();
                let rhs_row = &weight.data[rhs_start..rhs_start + input_row.len()];
                *out_value = dot_product_row(input_row, rhs_row);
            });
        return;
    }

    for (out_idx, out_value) in output.iter_mut().enumerate() {
        let rhs_start = out_idx * input_row.len();
        let rhs_row = &weight.data[rhs_start..rhs_start + input_row.len()];
        *out_value = dot_product_row(input_row, rhs_row);
    }
}

fn try_accumulate_transposed_linear_row_metal(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) -> bool {
    #[cfg(target_os = "macos")]
    {
        if std::env::var("CAMELID_METAL_LINEAR").ok().as_deref() != Some("1") {
            return false;
        }
        if weight.q8_0_blocks.is_some() || weight.q8_0_file_backing.is_some() {
            return false;
        }
        metal::try_linear_row_transposed_f32(
            input_row,
            weight.data,
            input_row.len(),
            output.len(),
            output,
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (input_row, weight, output);
        false
    }
}

fn dot_product_row_f64(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>() as f32
}

fn dot_product_row(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = 0.0;
    let mut idx = 0;
    while idx + 4 <= lhs.len() {
        sum += lhs[idx] * rhs[idx];
        sum += lhs[idx + 1] * rhs[idx + 1];
        sum += lhs[idx + 2] * rhs[idx + 2];
        sum += lhs[idx + 3] * rhs[idx + 3];
        idx += 4;
    }
    while idx < lhs.len() {
        sum += lhs[idx] * rhs[idx];
        idx += 1;
    }
    sum
}

fn apply_rope(
    tensor: &CpuTensor,
    position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    if head_count == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "RoPE head count must be greater than zero".to_string(),
        ));
    }
    if tensor.rank() != 2 || tensor.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE input {} expected shape [1, width], got {:?}",
            tensor.name, tensor.shape.dims
        )));
    }
    let width = tensor.dim(1)?;
    if !width.is_multiple_of(head_count) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE input width {width} is not divisible by head count {head_count}"
        )));
    }
    let head_dim = width / head_count;
    let rope_dim = config.rope_dimension_count.unwrap_or(head_dim as u32) as usize;
    if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE dimension count {rope_dim} must be even and within head dimension {head_dim}"
        )));
    }
    let freq_base = config.rope_freq_base.unwrap_or(10_000.0);
    if freq_base <= 0.0 || !freq_base.is_finite() {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE frequency base {freq_base} must be finite and positive"
        )));
    }
    let scaling = rope_scaling_from_config(config)?;
    let rope_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;

    apply_rope_with_pairing(
        tensor,
        RopeParams {
            position,
            head_count,
            head_dim,
            rope_dim,
            freq_base,
            pairing: diagnostic_rope_pairing()?,
            direction: diagnostic_rope_direction()?,
            position_mode: diagnostic_rope_position_mode()?,
            scaling,
            rope_freqs,
        },
        name,
    )
}

fn apply_rope_batch(
    tensor: &CpuTensor,
    base_position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    if head_count == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "RoPE head count must be greater than zero".to_string(),
        ));
    }
    if tensor.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE batch input {} expected rank 2, got {:?}",
            tensor.name, tensor.shape.dims
        )));
    }
    let rows = tensor.dim(0)?;
    let width = tensor.dim(1)?;
    if !width.is_multiple_of(head_count) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE batch input width {width} is not divisible by head count {head_count}"
        )));
    }
    let head_dim = width / head_count;
    let rope_dim = config.rope_dimension_count.unwrap_or(head_dim as u32) as usize;
    if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE dimension count {rope_dim} must be even and within head dimension {head_dim}"
        )));
    }
    let freq_base = config.rope_freq_base.unwrap_or(10_000.0);
    if freq_base <= 0.0 || !freq_base.is_finite() {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE frequency base {freq_base} must be finite and positive"
        )));
    }
    let scaling = rope_scaling_from_config(config)?;
    let rope_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;
    let params = RopeParams {
        position: base_position,
        head_count,
        head_dim,
        rope_dim,
        freq_base,
        pairing: diagnostic_rope_pairing()?,
        direction: diagnostic_rope_direction()?,
        position_mode: diagnostic_rope_position_mode()?,
        scaling,
        rope_freqs,
    };

    let mut data = tensor.data.clone();
    for row in 0..rows {
        apply_rope_to_row(
            &mut data[row * width..(row + 1) * width],
            base_position + row,
            params,
        );
    }
    CpuTensor::from_f32(name, tensor.shape.dims.clone(), data)
}

fn validate_rope_frequency_tensor(rope_freqs: &CpuTensor, rope_dim: usize) -> Result<&[f32]> {
    let expected_count = rope_dim / 2;
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE dimension count {rope_dim} must be even and greater than zero"
        )));
    }
    if rope_freqs.shape.dims != [expected_count] {
        return Err(BackendError::InvalidModelMetadata(format!(
            "rope_freqs.weight expected shape [{expected_count}], got {:?}",
            rope_freqs.shape.dims
        )));
    }
    if let Some((idx, frequency)) = rope_freqs
        .data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, frequency)| *frequency <= 0.0 || !frequency.is_finite())
    {
        return Err(BackendError::InvalidModelMetadata(format!(
            "rope_freqs.weight[{idx}] frequency factor {frequency} must be finite and positive"
        )));
    }
    Ok(&rope_freqs.data)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RopeScaling {
    kind: RopeScalingKind,
    factor: f32,
    original_context_length: Option<u32>,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RopeScalingKind {
    None,
    Linear,
    Llama3,
}

impl RopeScalingKind {
    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Llama3 => "llama3",
        }
    }
}

fn rope_scaling_from_config(config: &LlamaModelConfig) -> Result<RopeScaling> {
    let kind = match config.rope_scaling_type.as_deref().map(str::trim) {
        None | Some("") | Some("none") => RopeScalingKind::None,
        Some("linear") => RopeScalingKind::Linear,
        Some("llama3") => RopeScalingKind::Llama3,
        Some(other) => {
            return Err(BackendError::InvalidModelMetadata(format!(
                "unsupported llama.rope.scaling.type {other:?}; expected none, linear, or llama3"
            )))
        }
    };

    let factor = config.rope_scaling_factor.unwrap_or(1.0);
    if factor <= 0.0 || !factor.is_finite() {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE scaling factor {factor} must be finite and positive"
        )));
    }

    match kind {
        RopeScalingKind::None => Ok(RopeScaling {
            kind,
            factor: 1.0,
            original_context_length: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }),
        RopeScalingKind::Linear => Ok(RopeScaling {
            kind,
            factor,
            original_context_length: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }),
        RopeScalingKind::Llama3 => {
            let original_context_length =
                config.rope_scaling_original_context_length.unwrap_or(8_192);
            if original_context_length == 0 {
                return Err(BackendError::InvalidModelMetadata(
                    "llama3 RoPE scaling original context length must be greater than zero"
                        .to_string(),
                ));
            }
            let low_freq_factor = config.rope_scaling_low_freq_factor.unwrap_or(1.0);
            let high_freq_factor = config.rope_scaling_high_freq_factor.unwrap_or(4.0);
            if low_freq_factor <= 0.0
                || high_freq_factor <= 0.0
                || !low_freq_factor.is_finite()
                || !high_freq_factor.is_finite()
                || high_freq_factor <= low_freq_factor
            {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "llama3 RoPE scaling frequency factors must be finite, positive, and high > low; got low={low_freq_factor}, high={high_freq_factor}"
                )));
            }
            Ok(RopeScaling {
                kind,
                factor,
                original_context_length: Some(original_context_length),
                low_freq_factor: Some(low_freq_factor),
                high_freq_factor: Some(high_freq_factor),
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RopeParams<'a> {
    position: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    pairing: RopePairing,
    direction: RopeDirection,
    position_mode: RopePositionMode,
    scaling: RopeScaling,
    rope_freqs: Option<&'a [f32]>,
}

fn apply_rope_with_pairing(
    tensor: &CpuTensor,
    params: RopeParams<'_>,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let mut data = tensor.data.clone();
    apply_rope_to_row(&mut data, params.position, params);

    CpuTensor::from_f32(name, tensor.shape.dims.clone(), data)
}

fn apply_rope_to_row(data: &mut [f32], position: usize, mut params: RopeParams<'_>) {
    params.position = position;
    for head in 0..params.head_count {
        let head_start = head * params.head_dim;
        for pair_idx in 0..(params.rope_dim / 2) {
            let (dim0, dim1) = match params.pairing {
                RopePairing::AdjacentEvenOdd => {
                    let dim0 = head_start + (pair_idx * 2);
                    (dim0, dim0 + 1)
                }
                RopePairing::SplitHalf => (
                    head_start + pair_idx,
                    head_start + pair_idx + (params.rope_dim / 2),
                ),
            };
            let theta = rope_pair_frequency(pair_idx, &params);
            let angle = params.position_mode.effective_position(params.position) as f32 * theta;
            let (sin, cos) = angle.sin_cos();
            let x0 = data[dim0];
            let x1 = data[dim1];
            match params.direction {
                RopeDirection::Forward => {
                    data[dim0] = (x0 * cos) - (x1 * sin);
                    data[dim1] = (x0 * sin) + (x1 * cos);
                }
                RopeDirection::Inverse => {
                    data[dim0] = (x0 * cos) + (x1 * sin);
                    data[dim1] = (-x0 * sin) + (x1 * cos);
                }
            }
        }
    }
}

fn rope_pair_frequency(pair_idx: usize, params: &RopeParams<'_>) -> f32 {
    let base_frequency = params
        .freq_base
        .powf(-(pair_idx as f32 * 2.0) / params.rope_dim as f32);
    // GGUF's `rope_freqs.weight` follows llama.cpp's `freq_factors` contract:
    // the stored value divides the metadata-derived base frequency for the pair,
    // rather than replacing it as an absolute frequency.
    let effective_base_frequency = if let Some(rope_freqs) = params.rope_freqs {
        base_frequency / rope_freqs[pair_idx]
    } else {
        base_frequency
    };
    match params.scaling.kind {
        RopeScalingKind::None => effective_base_frequency,
        RopeScalingKind::Linear => effective_base_frequency / params.scaling.factor,
        RopeScalingKind::Llama3 => {
            llama3_scaled_rope_frequency(effective_base_frequency, params.scaling)
        }
    }
}

fn llama3_scaled_rope_frequency(frequency: f32, scaling: RopeScaling) -> f32 {
    let original_context_length = scaling
        .original_context_length
        .expect("validated llama3 scaling has original context length")
        as f32;
    let low_freq_factor = scaling
        .low_freq_factor
        .expect("validated llama3 scaling has low freq factor");
    let high_freq_factor = scaling
        .high_freq_factor
        .expect("validated llama3 scaling has high freq factor");

    let wavelength = (2.0 * std::f32::consts::PI) / frequency;
    let low_freq_wavelength = original_context_length / low_freq_factor;
    let high_freq_wavelength = original_context_length / high_freq_factor;
    if wavelength < high_freq_wavelength {
        frequency
    } else if wavelength > low_freq_wavelength {
        frequency / scaling.factor
    } else {
        let smooth = (original_context_length / wavelength - low_freq_factor)
            / (high_freq_factor - low_freq_factor);
        ((1.0 - smooth) * frequency / scaling.factor) + (smooth * frequency)
    }
}

fn rope_diagnostics(
    input: &CpuTensor,
    reported: &CpuTensor,
    position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
    role: &str,
) -> Result<LlamaRopeDiagnostic> {
    if head_count == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "RoPE diagnostic head count must be greater than zero".to_string(),
        ));
    }
    if input.rank() != 2 || input.dim(0)? != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE diagnostic input {} expected shape [1, width], got {:?}",
            input.name, input.shape.dims
        )));
    }
    require_tensor_shape(reported, &input.shape.dims, "RoPE diagnostic output")?;
    let width = input.dim(1)?;
    if !width.is_multiple_of(head_count) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "RoPE diagnostic input width {width} is not divisible by head count {head_count}"
        )));
    }
    let head_dim = width / head_count;
    let rope_dim = config.rope_dimension_count.unwrap_or(head_dim as u32) as usize;
    if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE diagnostic dimension count {rope_dim} must be even and within head dimension {head_dim}"
        )));
    }
    let freq_base = config.rope_freq_base.unwrap_or(10_000.0);
    if freq_base <= 0.0 || !freq_base.is_finite() {
        return Err(BackendError::InvalidModelMetadata(format!(
            "RoPE diagnostic frequency base {freq_base} must be finite and positive"
        )));
    }
    let pairing = diagnostic_rope_pairing()?;
    let direction = diagnostic_rope_direction()?;
    let position_mode = diagnostic_rope_position_mode()?;
    let scaling = rope_scaling_from_config(config)?;
    let rope_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;
    let reconstructed = apply_rope_with_pairing(
        input,
        RopeParams {
            position,
            head_count,
            head_dim,
            rope_dim,
            freq_base,
            pairing,
            direction,
            position_mode,
            scaling,
            rope_freqs,
        },
        format!("{role}_rope_diagnostic"),
    )?;

    let mut max_abs_delta = 0.0_f32;
    let mut max_abs_delta_index = 0usize;
    let mut reported_max_abs = 0.0_f32;
    let mut reported_max_abs_index = 0usize;
    for (idx, (reconstructed, reported)) in reconstructed
        .data
        .iter()
        .copied()
        .zip(reported.data.iter().copied())
        .enumerate()
    {
        let delta = (reconstructed - reported).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_index = idx;
        }
        let reported_abs = reported.abs();
        if reported_abs > reported_max_abs {
            reported_max_abs = reported_abs;
            reported_max_abs_index = idx;
        }
    }
    let (reported_max_abs_window_start, reported_max_abs_window) = tensor_window_around_index(
        &reported.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );
    let (_, reconstructed_reported_max_abs_window) = tensor_window_around_index(
        &reconstructed.data,
        reported_max_abs_index,
        TENSOR_CHECKPOINT_SAMPLE,
    );

    Ok(LlamaRopeDiagnostic {
        role: role.to_string(),
        pairing: pairing.label(),
        direction: direction.label(),
        position_mode: position_mode.label(),
        frequency_source: if rope_freqs.is_some() {
            "rope_freqs.weight"
        } else {
            "metadata"
        },
        rope_freqs_count: rope_freqs.map(<[f32]>::len),
        scaling_type: scaling.kind.label(),
        scaling_factor: scaling.factor,
        scaling_original_context_length: scaling.original_context_length,
        scaling_low_freq_factor: scaling.low_freq_factor,
        scaling_high_freq_factor: scaling.high_freq_factor,
        position,
        effective_position: position_mode.effective_position(position),
        head_count,
        head_dim,
        rope_dim,
        freq_base,
        input_first_values: input
            .data
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        reconstructed_first_values: reconstructed
            .data
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        reported_first_values: reported
            .data
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        reported_max_abs_index,
        reported_max_abs,
        reported_max_abs_window_start,
        reported_max_abs_window,
        reconstructed_reported_max_abs_window,
        max_abs_delta_index,
        max_abs_delta,
    })
}

fn write_kv_cache(
    kv_cache: &mut LlamaKvCache,
    layer_idx: usize,
    key: &CpuTensor,
    value: &CpuTensor,
) -> Result<()> {
    let expected_width = kv_cache.plan.kv_head_count * kv_cache.plan.head_dim;
    if key.shape.dims != [1, expected_width] || value.shape.dims != [1, expected_width] {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "KV projection shapes must be [1, {expected_width}], got key {:?}, value {:?}",
            key.shape.dims, value.shape.dims
        )));
    }
    if layer_idx >= kv_cache.plan.layer_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "layer index {layer_idx} is out of range for KV cache layer count {}",
            kv_cache.plan.layer_count
        )));
    }
    kv_cache.ensure_position_capacity(kv_cache.position + 1)?;
    let offset = kv_cache_offset(kv_cache, layer_idx, kv_cache.position, 0);
    let end = offset + expected_width;
    copy_to_f16_kv_cache_storage(&mut kv_cache.keys[offset..end], &key.data);
    copy_to_f16_kv_cache_storage(&mut kv_cache.values[offset..end], &value.data);
    Ok(())
}

fn write_kv_cache_batch(
    kv_cache: &mut LlamaKvCache,
    layer_idx: usize,
    base_position: usize,
    key: &CpuTensor,
    value: &CpuTensor,
) -> Result<()> {
    let expected_width = kv_cache.plan.kv_head_count * kv_cache.plan.head_dim;
    if key.rank() != 2
        || value.rank() != 2
        || key.dim(1)? != expected_width
        || value.dim(1)? != expected_width
        || key.dim(0)? != value.dim(0)?
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "KV batch projection shapes must be [rows, {expected_width}], got key {:?}, value {:?}",
            key.shape.dims, value.shape.dims
        )));
    }
    if layer_idx >= kv_cache.plan.layer_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "layer index {layer_idx} is out of range for KV cache layer count {}",
            kv_cache.plan.layer_count
        )));
    }
    let rows = key.dim(0)?;
    kv_cache.ensure_position_capacity(base_position + rows)?;
    for row in 0..rows {
        let position = base_position + row;
        let offset = kv_cache_offset(kv_cache, layer_idx, position, 0);
        let end = offset + expected_width;
        let row_start = row * expected_width;
        let row_end = row_start + expected_width;
        copy_to_f16_kv_cache_storage(
            &mut kv_cache.keys[offset..end],
            &key.data[row_start..row_end],
        );
        copy_to_f16_kv_cache_storage(
            &mut kv_cache.values[offset..end],
            &value.data[row_start..row_end],
        );
    }
    Ok(())
}

fn kv_cache_trace(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    position_count: usize,
) -> Result<LlamaKvCacheTrace> {
    if layer_idx >= kv_cache.plan.layer_count {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "layer index {layer_idx} is out of range for KV cache layer count {}",
            kv_cache.plan.layer_count
        )));
    }
    if position_count > kv_cache.plan.max_sequence_length {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "KV trace position count {position_count} exceeds cache capacity {}",
            kv_cache.plan.max_sequence_length
        )));
    }
    if position_count == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "KV trace requires at least one cached position".to_string(),
        ));
    }

    let key_value_width = kv_cache.plan.kv_head_count * kv_cache.plan.head_dim;
    if key_value_width == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "KV trace requires non-empty key/value rows".to_string(),
        ));
    }
    let mut key_sum_square = 0.0_f64;
    let mut value_sum_square = 0.0_f64;
    let mut key_checksum = 0.0_f64;
    let mut value_checksum = 0.0_f64;
    let mut key_max_abs = 0.0_f32;
    let mut key_max_abs_position = 0;
    let mut key_max_abs_index = 0;
    let mut value_max_abs = 0.0_f32;
    let mut value_max_abs_position = 0;
    let mut value_max_abs_index = 0;

    for position in 0..position_count {
        let start = kv_cache_offset(kv_cache, layer_idx, position, 0);
        let end = start + key_value_width;
        for (idx, (&key, &value)) in kv_cache.keys[start..end]
            .iter()
            .zip(kv_cache.values[start..end].iter())
            .enumerate()
        {
            if !key.is_finite() || !value.is_finite() {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "KV trace found non-finite value at layer {layer_idx} position {position} index {idx}"
                )));
            }
            let ordinal = ((position * key_value_width) + idx + 1) as f64;
            let key64 = key as f64;
            let value64 = value as f64;
            key_sum_square += key64 * key64;
            value_sum_square += value64 * value64;
            key_checksum += ordinal * key64;
            value_checksum += ordinal * value64;
            let key_abs = key.abs();
            if key_abs > key_max_abs {
                key_max_abs = key_abs;
                key_max_abs_position = position;
                key_max_abs_index = idx;
            }
            let value_abs = value.abs();
            if value_abs > value_max_abs {
                value_max_abs = value_abs;
                value_max_abs_position = position;
                value_max_abs_index = idx;
            }
        }
    }

    let value_count = (position_count * key_value_width) as f64;
    let sampled_positions = sampled_attention_trace_positions(position_count)
        .into_iter()
        .map(|position| kv_cache_position_trace(kv_cache, layer_idx, position, key_value_width))
        .collect::<Result<Vec<_>>>()?;

    Ok(LlamaKvCacheTrace {
        layer_index: layer_idx,
        position_count,
        kv_head_count: kv_cache.plan.kv_head_count,
        head_dim: kv_cache.plan.head_dim,
        key_value_width,
        key_checksum,
        value_checksum,
        key_rms: (key_sum_square / value_count).sqrt() as f32,
        value_rms: (value_sum_square / value_count).sqrt() as f32,
        key_max_abs,
        key_max_abs_position,
        key_max_abs_index,
        value_max_abs,
        value_max_abs_position,
        value_max_abs_index,
        sampled_positions,
    })
}

fn kv_cache_position_trace(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    position: usize,
    key_value_width: usize,
) -> Result<LlamaKvCachePositionTrace> {
    let start = kv_cache_offset(kv_cache, layer_idx, position, 0);
    let end = start + key_value_width;
    let key_slice = &kv_cache.keys[start..end];
    let value_slice = &kv_cache.values[start..end];
    let mut key_sum_square = 0.0_f64;
    let mut value_sum_square = 0.0_f64;
    let mut key_checksum = 0.0_f64;
    let mut value_checksum = 0.0_f64;
    let mut key_max_abs = 0.0_f32;
    let mut value_max_abs = 0.0_f32;
    for (idx, (&key, &value)) in key_slice.iter().zip(value_slice.iter()).enumerate() {
        if !key.is_finite() || !value.is_finite() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV position trace found non-finite value at layer {layer_idx} position {position} index {idx}"
            )));
        }
        let ordinal = (idx + 1) as f64;
        let key64 = key as f64;
        let value64 = value as f64;
        key_sum_square += key64 * key64;
        value_sum_square += value64 * value64;
        key_checksum += ordinal * key64;
        value_checksum += ordinal * value64;
        key_max_abs = key_max_abs.max(key.abs());
        value_max_abs = value_max_abs.max(value.abs());
    }
    let width = key_value_width as f64;
    Ok(LlamaKvCachePositionTrace {
        position,
        key_checksum,
        value_checksum,
        key_rms: (key_sum_square / width).sqrt() as f32,
        value_rms: (value_sum_square / width).sqrt() as f32,
        key_max_abs,
        value_max_abs,
        key_first_values: key_slice
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
        value_first_values: value_slice
            .iter()
            .take(TENSOR_CHECKPOINT_SAMPLE)
            .copied()
            .collect(),
    })
}

fn copy_to_f16_kv_cache_storage(dest: &mut [f32], source: &[f32]) {
    debug_assert_eq!(dest.len(), source.len());
    for (dest_value, source_value) in dest.iter_mut().zip(source.iter().copied()) {
        *dest_value = f16_bits_to_f32(f32_to_f16_bits(source_value));
    }
}

const ATTENTION_TRACE_HEAD_LIMIT: usize = 8;
const ATTENTION_TRACE_POSITION_LIMIT: usize = 8;
const ATTENTION_TRACE_VALUE_LIMIT: usize = 10;
const ATTENTION_TRACE_EDGE_POSITION_LIMIT: usize = ATTENTION_TRACE_POSITION_LIMIT / 2;
const ATTENTION_TRACE_TOP_PROBABILITY_LIMIT: usize = 4;

#[derive(Debug, Clone, PartialEq)]
struct LlamaAttentionContextOutput {
    tensor: CpuTensor,
    trace: Option<LlamaAttentionTrace>,
}

fn causal_attention_context(
    kv_cache: &LlamaKvCache,
    layer_idx: usize,
    query: &CpuTensor,
    attention_heads: usize,
    kv_heads: usize,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<LlamaAttentionContextOutput> {
    if kv_heads == 0 || !attention_heads.is_multiple_of(kv_heads) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention head count {attention_heads} must be a multiple of kv head count {kv_heads}"
        )));
    }
    let head_dim = kv_cache.plan.head_dim;
    let expected_width = attention_heads * head_dim;
    if query.shape.dims != [1, expected_width] {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "attention query shape {:?} does not match expected [1, {expected_width}]",
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

    let position_count = kv_cache.position + 1;
    let repeats = attention_heads / kv_heads;
    let head_mapping = diagnostic_gqa_head_mapping()?;
    let score_scale = diagnostic_attention_score_scale()?;
    let scale = attention_score_scale_value(head_dim, score_scale);
    let mut out = vec![0.0; expected_width];

    if position_count == 1 {
        for attention_head in 0..attention_heads {
            let kv_head =
                map_attention_head_to_kv_head(attention_head, repeats, kv_heads, head_mapping);
            let out_start = attention_head * head_dim;
            let value_start = kv_cache_offset(kv_cache, layer_idx, 0, kv_head);
            out[out_start..out_start + head_dim]
                .copy_from_slice(&kv_cache.values[value_start..value_start + head_dim]);
        }
    } else {
        let mut scores = Vec::with_capacity(position_count);
        for attention_head in 0..attention_heads {
            let kv_head =
                map_attention_head_to_kv_head(attention_head, repeats, kv_heads, head_mapping);
            let query_start = attention_head * head_dim;
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
