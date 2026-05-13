use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    fs::File,
    io::{Error as IoError, ErrorKind, Result as IoResult},
    mem,
    os::unix::fs::FileExt,
    process::Command,
    sync::Arc,
    time::Instant,
};

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    gguf::GgufTensorType,
    model::{
        DenseLlamaDims, LlamaFfnTensors, LlamaModelConfig, LlamaMoeExpertTensors,
        LlamaTensorBinding,
    },
    tensor::{
        dot_product, parse_byte_count_env, q8_0_file_read_stats, record_q8_0_file_read,
        should_parallelize_linear_output, with_q8_file_cache_capacity_override, CpuTensor,
        Q8_0Block, Q8_0FileBacking, Q8_0FileReadStats, TensorShape, TensorStore,
    },
    BackendError, Result,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaKvCachePlan {
    pub max_sequence_length: usize,
    pub layer_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
}

impl LlamaKvCachePlan {
    pub fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let dims = DenseLlamaDims::from_config(config)?;
        let max_sequence_length = config.context_length as usize;
        let shape = vec![
            dims.block_count,
            max_sequence_length,
            dims.attention_head_count_kv,
            dims.head_dim,
        ];
        Ok(Self {
            max_sequence_length,
            layer_count: dims.block_count,
            kv_head_count: dims.attention_head_count_kv,
            head_dim: dims.head_dim,
            key_shape: shape.clone(),
            value_shape: shape,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaKvCache {
    pub plan: LlamaKvCachePlan,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub allocated_sequence_length: usize,
    pub position: usize,
}

impl LlamaKvCache {
    pub fn new(plan: LlamaKvCachePlan) -> Result<Self> {
        Ok(Self {
            plan,
            keys: Vec::new(),
            values: Vec::new(),
            allocated_sequence_length: 0,
            position: 0,
        })
    }

    pub fn can_append(&self) -> bool {
        self.position < self.plan.max_sequence_length
    }

    fn ensure_position_capacity(&mut self, required_sequence_length: usize) -> Result<()> {
        if required_sequence_length > self.plan.max_sequence_length {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV cache position {required_sequence_length} exceeds context length {}",
                self.plan.max_sequence_length
            )));
        }
        if required_sequence_length <= self.allocated_sequence_length {
            return Ok(());
        }
        let values = required_sequence_length
            .checked_mul(self.plan.layer_count)
            .and_then(|value| value.checked_mul(self.plan.kv_head_count))
            .and_then(|value| value.checked_mul(self.plan.head_dim))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("KV cache element count overflow".to_string())
            })?;
        self.keys.resize(values, 0.0);
        self.values.resize(values, 0.0);
        self.allocated_sequence_length = required_sequence_length;
        Ok(())
    }

    pub fn allocated_elements(&self) -> usize {
        self.keys.len() + self.values.len()
    }

    pub fn allocated_bytes(&self) -> u64 {
        (self.allocated_elements() as u64) * (std::mem::size_of::<f32>() as u64)
    }
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
pub enum RopePairing {
    AdjacentEvenOdd,
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePositionMode {
    ZeroBased,
    OneBased,
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

impl RopePairing {
    pub fn label(self) -> &'static str {
        match self {
            Self::AdjacentEvenOdd => "adjacent_even_odd",
            Self::SplitHalf => "split_half",
        }
    }
}

impl RopeDirection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Inverse => "inverse",
        }
    }
}

impl RopePositionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ZeroBased => "zero_based",
            Self::OneBased => "one_based",
        }
    }

    fn effective_position(self, position: usize) -> usize {
        match self {
            Self::ZeroBased => position,
            Self::OneBased => position + 1,
        }
    }
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

pub fn diagnostic_rope_pairing() -> Result<RopePairing> {
    match env::var("CAMELID_ROPE_PAIRING") {
        Ok(value) if value == "split_half" => Ok(RopePairing::SplitHalf),
        Ok(value) if value == "adjacent_even_odd" || value.is_empty() => {
            Ok(RopePairing::AdjacentEvenOdd)
        }
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_PAIRING {value:?}; expected adjacent_even_odd or split_half"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopePairing::AdjacentEvenOdd),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_PAIRING: {err}"
        ))),
    }
}

pub fn diagnostic_rope_direction() -> Result<RopeDirection> {
    match env::var("CAMELID_ROPE_DIRECTION") {
        Ok(value) if value == "inverse" => Ok(RopeDirection::Inverse),
        Ok(value) if value == "forward" || value.is_empty() => Ok(RopeDirection::Forward),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_DIRECTION {value:?}; expected forward or inverse"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopeDirection::Forward),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_DIRECTION: {err}"
        ))),
    }
}

pub fn diagnostic_rope_position_mode() -> Result<RopePositionMode> {
    match env::var("CAMELID_ROPE_POSITION_MODE") {
        Ok(value) if value == "one_based" => Ok(RopePositionMode::OneBased),
        Ok(value) if value == "zero_based" || value.is_empty() => Ok(RopePositionMode::ZeroBased),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_POSITION_MODE {value:?}; expected zero_based or one_based"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopePositionMode::ZeroBased),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_POSITION_MODE: {err}"
        ))),
    }
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
                store.load_cpu_f32_with_q8_0_block_retention(name, true)
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
                if retain_moe_q8_0_blocks_enabled() {
                    let mut blocks = Vec::new();
                    for desc in descs {
                        let expert_blocks = store.load_q8_0_blocks(&desc.name)?;
                        blocks.extend(expert_blocks.blocks);
                    }
                    CpuTensor::from_q8_0_blocks(
                        format!("{}..{} retained split experts", first.name, descs.len()),
                        TensorShape { dims },
                        blocks,
                    )
                } else {
                    store.load_q8_0_split_file_backed_tensor(
                        format!("{}..{} split experts", first.name, descs.len()),
                        dims,
                        descs,
                    )
                }
            }
        };
        let token_embedding = normalize_token_embedding_shape(
            load_linear(&binding.token_embedding.name)?,
            &binding.token_embedding.name,
        )?;
        let output_norm = store.load_cpu_f32(&binding.output_norm.name)?;
        let output = if binding.output_is_tied_embedding {
            None
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
                let logits = output_projection_runtime(
                    &norm,
                    self.weights.output_projection(),
                    "logits",
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

fn apply_sampling_adjustments(
    logits: &CpuTensor,
    config: &SamplingConfig,
    token_history: &[u32],
) -> Result<CpuTensor> {
    if config.presence_penalty == 0.0
        && config.frequency_penalty == 0.0
        && config.logit_bias.is_empty()
    {
        return Ok(logits.clone());
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

    Ok(adjusted)
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

    let started = Instant::now();
    let q = linear_runtime(
        &attn_norm,
        &layer.attention_q,
        format!("layer_{layer_idx}_attention_q"),
        collect_diagnostics,
    )?;
    let attention_q_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&q))
        .transpose()?;
    let attention_q_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_q, &q, "linear"))
        .transpose()?;
    timings.attention_q = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_q(capture_memory_sample(kv_cache));
    }
    trace_forward_layer_memory(layer_idx, "attention_q_done");

    let started = Instant::now();
    let k = linear_for_role_runtime(
        &attn_norm,
        &layer.attention_k,
        format!("layer_{layer_idx}_attention_k"),
        "attention_k",
        collect_diagnostics,
    )?;
    let attention_k_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&k))
        .transpose()?;
    let attention_k_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_k, &k, "attention_k"))
        .transpose()?;
    timings.attention_k = started.elapsed().as_micros();
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

    let started = Instant::now();
    let v = linear_for_role_runtime(
        &attn_norm,
        &layer.attention_v,
        format!("layer_{layer_idx}_attention_v"),
        "attention_v",
        collect_diagnostics,
    )?;
    let attention_v_stats = collect_diagnostics
        .then(|| LlamaTensorStats::from_tensor(&v))
        .transpose()?;
    let attention_v_diagnostic = collect_diagnostics
        .then(|| linear_projection_diagnostics(&attn_norm, &layer.attention_v, &v, "attention_v"))
        .transpose()?;
    timings.attention_v = started.elapsed().as_micros();
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
    let mut attn_out = linear_runtime(
        &context,
        &layer.attention_output,
        format!("layer_{layer_idx}_attention_output"),
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
        (ffn_out, None, None, None, None, None, None, None, None)
    } else {
        let activated = gated_ffn_activation(
            &ffn_norm,
            &layer.ffn_gate,
            &layer.ffn_up,
            format!("layer_{layer_idx}_ffn_activated"),
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
        let ffn_out = linear_for_role_runtime(
            &activated,
            &layer.ffn_down,
            format!("layer_{layer_idx}_ffn_down"),
            "ffn_down",
            collect_diagnostics,
        )?;
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
    let output = residual.add(&ffn_out, format!("layer_{layer_idx}_ffn_residual"))?;
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

    let started = Instant::now();
    let q = linear_runtime(
        &attn_norm,
        &layer.attention_q,
        format!("layer_{layer_idx}_prefill_attention_q"),
        false,
    )?;
    timings.attention_q = started.elapsed().as_micros();
    if let Some(memory) = &mut memory {
        memory.record_after_attention_q(capture_memory_sample(kv_cache));
    }
    trace_chunk_memory("attention_q_done");

    let started = Instant::now();
    let k = linear_for_role_runtime(
        &attn_norm,
        &layer.attention_k,
        format!("layer_{layer_idx}_prefill_attention_k"),
        "attention_k",
        false,
    )?;
    timings.attention_k = started.elapsed().as_micros();
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

    let started = Instant::now();
    let v = linear_for_role_runtime(
        &attn_norm,
        &layer.attention_v,
        format!("layer_{layer_idx}_prefill_attention_v"),
        "attention_v",
        false,
    )?;
    timings.attention_v = started.elapsed().as_micros();
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
    linear_for_role_runtime(input, weight, name, "linear", collect_diagnostics)
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
    if collect_diagnostics {
        linear_for_role(input, weight, name, rectangular_role)
    } else {
        linear_with_diagnostic_layouts(
            input,
            weight,
            name,
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Auto,
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
            SquareLinearLayout::Descriptor => matmul_descriptor_with_precision(input, weight, name),
            SquareLinearLayout::Transposed => {
                matmul_rhs_transposed_with_precision(input, weight, name)
            }
        }
    } else if rectangular_layout == RectangularLinearLayout::Descriptor {
        let descriptor_weight = linear_weight_reinterpreted_as_descriptor(weight, input_width)?;
        matmul_descriptor_with_precision(input, &descriptor_weight, name)
    } else if rectangular_layout == RectangularLinearLayout::Transposed || rows == input_width {
        let transposed_weight = linear_weight_reinterpreted_as_transposed(weight, input_width)?;
        matmul_rhs_transposed_with_precision(input, &transposed_weight, name)
    } else if cols == input_width {
        matmul_rhs_transposed_with_precision(input, weight, name)
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
    reinterpreted
}

#[derive(Clone, Copy)]
struct BorrowedLinearWeight<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f32],
    source_type: Option<GgufTensorType>,
    q8_0_blocks: Option<&'a [Q8_0Block]>,
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
            q8_0_file_backing: weight.q8_0_file_backing.as_ref(),
        })
    }

    fn with_swapped_matrix_shape(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
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
    tensor.q8_0_file_backing = None;
    Ok(())
}

fn output_projection_runtime(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<CpuTensor> {
    let layout = if collect_diagnostics {
        diagnostic_output_projection_layout()?
    } else {
        OutputProjectionLayout::TokenMajor
    };
    output_projection_with_layout(input, weight, name, layout)
}

fn output_projection_with_layout(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
    layout: OutputProjectionLayout,
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
                matmul_descriptor_with_precision(input, weight, name)
            } else if weight.dim(1)? == input_width {
                matmul_rhs_transposed_with_precision(input, weight, name)
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
            let token_major = borrowed_linear_weight_as_transposed(weight, input_width)?;
            matmul_rhs_transposed_borrowed_with_precision(input, token_major, name)
        }
    }
}

fn matmul_descriptor_with_precision(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    match diagnostic_linear_accumulation_precision()? {
        LinearAccumulationPrecision::F32 => input.matmul(weight, name),
        LinearAccumulationPrecision::F64 => matmul_descriptor_f64(input, weight, name),
    }
}

fn matmul_rhs_transposed_with_precision(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
    let input_width = input.dim(1)?;
    if should_use_q8_0_block_dot(weight, input_width) {
        return matmul_rhs_transposed_q8_0_block_dot(input, weight, name);
    }
    if let Some(backing) = q8_0_reader_backing(weight, input_width)? {
        return matmul_rhs_transposed_q8_0_block_reader(
            input,
            backing,
            Q8BlockReader::new(backing.absolute_offset, backing.num_blocks),
            weight.dim(0)?,
            name,
        );
    }
    match diagnostic_linear_accumulation_precision()? {
        LinearAccumulationPrecision::F32 => input.matmul_rhs_transposed(weight, name),
        LinearAccumulationPrecision::F64 => matmul_rhs_transposed_f64(input, weight, name),
    }
}

fn matmul_rhs_transposed_borrowed_with_precision(
    input: &CpuTensor,
    weight: BorrowedLinearWeight<'_>,
    name: impl Into<String>,
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
        return matmul_rhs_transposed_q8_0_block_reader(
            input,
            backing,
            Q8BlockReader::new(backing.absolute_offset, backing.num_blocks),
            output_width,
            name,
        );
    }
    let mut output = vec![0.0; rows * output_width];
    let precision = diagnostic_linear_accumulation_precision()?;
    for row in 0..rows {
        let input_start = row * input_width;
        let output_start = row * output_width;
        accumulate_transposed_linear_row_runtime(
            &input.data[input_start..input_start + input_width],
            weight,
            &mut output[output_start..output_start + output_width],
            precision,
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

fn gated_ffn_activation(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
    collect_diagnostics: bool,
) -> Result<GatedFfnActivation> {
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

    let input_row = &input.data[..input_width];
    let (mut gate, up, gate_elapsed, up_elapsed) = if collect_diagnostics {
        let mut gate = vec![0.0; gate_width];
        let mut up = vec![0.0; up_width];

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
        let up_elapsed = started.elapsed().as_micros();
        (gate, up, gate_elapsed, up_elapsed)
    } else {
        let (gate_result, up_result) = rayon::join(
            || {
                let mut gate = vec![0.0; gate_width];
                let started = Instant::now();
                accumulate_linear_row(input_row, gate_weight, &mut gate, "ffn gate", false)?;
                Result::<_>::Ok((gate, started.elapsed().as_micros()))
            },
            || {
                let mut up = vec![0.0; up_width];
                let started = Instant::now();
                accumulate_linear_row(input_row, up_weight, &mut up, "ffn up", false)?;
                Result::<_>::Ok((up, started.elapsed().as_micros()))
            },
        );
        let (gate, gate_elapsed) = gate_result?;
        let (up, up_elapsed) = up_result?;
        (gate, up, gate_elapsed, up_elapsed)
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
    let tensor = CpuTensor::from_f32(name, vec![1, gate_width], gate)?;
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

fn gated_ffn_activation_batch(
    input: &CpuTensor,
    gate_weight: &CpuTensor,
    up_weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<GatedFfnActivation> {
    if input.rank() != 2 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "gated FFN batch activation expects rank-2 input, got {:?}",
            input.shape.dims
        )));
    }

    let (gate_result, up_result) = rayon::join(
        || {
            let started = Instant::now();
            let gate =
                linear_for_role_runtime(input, gate_weight, "ffn_gate_prefill", "ffn gate", false)?;
            Result::<_>::Ok((gate, started.elapsed().as_micros()))
        },
        || {
            let started = Instant::now();
            let up = linear_for_role_runtime(input, up_weight, "ffn_up_prefill", "ffn up", false)?;
            Result::<_>::Ok((up, started.elapsed().as_micros()))
        },
    );
    let (mut gate, gate_elapsed) = gate_result?;
    let (up, up_elapsed) = up_result?;

    require_tensor_shape(&up, &gate.shape.dims, "gated FFN prefill up projection")?;
    let order = diagnostic_ffn_gate_up_order()?;
    let started = Instant::now();
    for (gate_value, up_value) in gate.data.iter_mut().zip(up.data) {
        *gate_value = match order {
            FfnGateUpOrder::GateUp => (*gate_value / (1.0 + (-*gate_value).exp())) * up_value,
            FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * *gate_value,
        };
    }
    gate.name = name.into();
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
    let selected_sum = scored.iter().map(|(_, value)| *value).sum::<f32>();
    if selected_sum > 0.0 {
        for (_, value) in &mut scored {
            *value /= selected_sum;
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
    let mut tensor = if let Some(blocks) = &weight.q8_0_blocks {
        CpuTensor::from_q8_0_blocks(
            name,
            TensorShape {
                dims: vec![output_width, input_width],
            },
            blocks[block_offset..block_offset + block_count].to_vec(),
        )?
    } else if let Some(split_backings) = &weight.q8_0_split_file_backing {
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
    if mixtral_moe_retained_q8_fast_path_available(gate_experts, up_experts, down_experts) {
        return mixtral_moe_ffn_retained_q8(
            input,
            router,
            gate_experts,
            up_experts,
            down_experts,
            expert_used_count,
            name,
        );
    }
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

fn mixtral_moe_retained_q8_fast_path_available(
    gate_experts: &CpuTensor,
    up_experts: &CpuTensor,
    down_experts: &CpuTensor,
) -> bool {
    gate_experts.q8_0_blocks.is_some()
        && up_experts.q8_0_blocks.is_some()
        && down_experts.q8_0_blocks.is_some()
}

fn borrowed_retained_expert_matrix_view<'a>(
    weight: &'a CpuTensor,
    expert_idx: usize,
    input_width: usize,
    output_width: usize,
) -> Result<BorrowedLinearWeight<'a>> {
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
    if weight.dim(0)? != input_width || weight.dim(1)? != output_width {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert tensor {} expected per-expert dims [{input_width}, {output_width}], got {:?}",
            weight.name, weight.shape.dims
        )));
    }
    let expert_elements = input_width.checked_mul(output_width).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("MoE expert element count overflow".to_string())
    })?;
    if !expert_elements.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert element count {expert_elements} is not Q8_0 block aligned"
        )));
    }
    let block_count = expert_elements / Q8_0_BLOCK_VALUES;
    let block_offset = block_count.checked_mul(expert_idx).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("MoE expert block offset overflow".to_string())
    })?;
    let blocks = weight.q8_0_blocks.as_deref().ok_or_else(|| {
        BackendError::RuntimeShapeMismatch(format!(
            "MoE retained Q8 fast path requested without retained blocks for {}",
            weight.name
        ))
    })?;
    let block_end = block_offset.checked_add(block_count).ok_or_else(|| {
        BackendError::RuntimeShapeMismatch("MoE expert block slice overflow".to_string())
    })?;
    if block_end > blocks.len() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "MoE expert block slice {block_offset}..{block_end} exceeds {} blocks for {}",
            blocks.len(),
            weight.name
        )));
    }
    Ok(BorrowedLinearWeight {
        rows: output_width,
        cols: input_width,
        data: &[],
        source_type: weight.source_type,
        q8_0_blocks: Some(&blocks[block_offset..block_end]),
        q8_0_file_backing: None,
    })
}

fn mixtral_moe_ffn_retained_q8(
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
    let output_name = name.into();
    let router_started = Instant::now();
    let logits = linear_for_role_runtime(input, router, "mixtral_router", "linear", false)?;
    let router_elapsed = router_started.elapsed().as_micros();
    let expert_count = logits.dim(1)?;
    let mut output = vec![0.0_f32; rows * hidden];
    let mut gate_elapsed = 0;
    let mut up_elapsed = 0;
    let mut activation_elapsed = 0;
    let mut down_elapsed = 0;
    let order = diagnostic_ffn_gate_up_order()?;

    for row in 0..rows {
        let input_start = row * hidden;
        let input_row = &input.data[input_start..input_start + hidden];
        let top = softmax_top_k(
            &logits.data[row * expert_count..(row + 1) * expert_count],
            expert_used_count,
        );
        for (expert_idx, weight) in top {
            let gate_weight =
                borrowed_retained_expert_matrix_view(gate_experts, expert_idx, hidden, ff)?;
            let up_weight = borrowed_retained_expert_matrix_view(up_experts, expert_idx, hidden, ff)?;
            let down_weight =
                borrowed_retained_expert_matrix_view(down_experts, expert_idx, ff, hidden)?;

            let mut gate = vec![0.0_f32; ff];
            let mut up = vec![0.0_f32; ff];
            let started = Instant::now();
            accumulate_transposed_linear_row_runtime(
                input_row,
                gate_weight,
                &mut gate,
                LinearAccumulationPrecision::F32,
            )?;
            gate_elapsed += started.elapsed().as_micros();
            let started = Instant::now();
            accumulate_transposed_linear_row_runtime(
                input_row,
                up_weight,
                &mut up,
                LinearAccumulationPrecision::F32,
            )?;
            up_elapsed += started.elapsed().as_micros();
            let started = Instant::now();
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value = match order {
                    FfnGateUpOrder::GateUp => (*gate_value / (1.0 + (-*gate_value).exp())) * up_value,
                    FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * *gate_value,
                };
            }
            activation_elapsed += started.elapsed().as_micros();

            let mut expert_out = vec![0.0_f32; hidden];
            let started = Instant::now();
            accumulate_transposed_linear_row_runtime(
                &gate,
                down_weight,
                &mut expert_out,
                LinearAccumulationPrecision::F32,
            )?;
            down_elapsed += started.elapsed().as_micros();
            let output_start = row * hidden;
            for col in 0..hidden {
                output[output_start + col] += expert_out[col] * weight;
            }
        }
    }

    Ok((
        CpuTensor::from_f32(output_name, vec![rows, hidden], output)?,
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

fn should_use_q8_0_block_dot(weight: &CpuTensor, input_width: usize) -> bool {
    q8_0_block_dot_enabled()
        && weight.source_type == Some(GgufTensorType::Q8_0)
        && weight.q8_0_blocks.is_some()
        && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
}

fn should_use_borrowed_q8_0_block_dot(
    weight: BorrowedLinearWeight<'_>,
    input_width: usize,
) -> bool {
    q8_0_block_dot_enabled()
        && weight.source_type == Some(GgufTensorType::Q8_0)
        && weight.q8_0_blocks.is_some()
        && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
}

fn q8_0_block_dot_enabled() -> bool {
    // Retained Q8_0 blocks should use the same quantized-input block-dot path as the
    // lazy/file-backed reader. This is both the llama.cpp-style execution shape and the only
    // interactive path for local Q8 chat; keep a dequantized-f32 escape hatch for diagnostics.
    !q8_0_env_flag_disabled("CAMELID_Q8_0_BLOCK_DOT")
}

fn q8_0_file_reader_block_dot_enabled() -> bool {
    // Lazy/file-backed Q8 rows are Camelid's production Q8_0 surface. Use the
    // quantized-input block-dot path by default to match llama.cpp-style Q8_0 matmul
    // tie-break behavior, while keeping an explicit dequantized-f32 escape hatch.
    !q8_0_env_flag_disabled("CAMELID_Q8_0_FILE_READER_BLOCK_DOT")
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

fn q8_0_env_flag_disabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("dequantized")
                || value.eq_ignore_ascii_case("f32")
        })
        .unwrap_or(false)
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
fn retain_moe_q8_0_blocks_enabled() -> bool {
    env_flag_enabled("CAMELID_RETAIN_MOE_Q8_0_BLOCKS")
}


const Q8_0_BLOCK_VALUES: usize = 32;

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

fn matmul_rhs_transposed_q8_0_block_dot(
    input: &CpuTensor,
    weight: &CpuTensor,
    name: impl Into<String>,
) -> Result<CpuTensor> {
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
    let weight_blocks = weight.q8_0_blocks.as_ref().ok_or_else(|| {
        BackendError::InvalidTensorData(format!(
            "q8_0 block-dot requested for {} without q8_0 blocks",
            weight.name
        ))
    })?;
    let expected_blocks = output_width * blocks_per_row;
    if weight_blocks.len() != expected_blocks {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "q8_0 block-dot expected {expected_blocks} blocks for weight {} shape {:?}, got {}",
            weight.name,
            weight.shape.dims,
            weight_blocks.len()
        )));
    }

    let mut output = vec![0.0_f32; rows * output_width];
    for row in 0..rows {
        let input_start = row * input_width;
        let quantized_input =
            quantize_q8_0_row(&input.data[input_start..input_start + input_width]);
        let out_start = row * output_width;
        let output_row = &mut output[out_start..out_start + output_width];
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
    blocks.extend(input.chunks_exact(Q8_0_BLOCK_VALUES).map(|block| {
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
    }));
}

fn q8_0_dot_rows(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    #[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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

fn q8_0_reader_backing(weight: &CpuTensor, input_width: usize) -> Result<Option<&Q8_0FileBacking>> {
    if weight.source_type != Some(GgufTensorType::Q8_0) || weight.q8_0_blocks.is_some() {
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

fn matmul_rhs_transposed_q8_0_block_reader(
    input: &CpuTensor,
    backing: &Q8_0FileBacking,
    reader: Q8BlockReader,
    output_width: usize,
    name: impl Into<String>,
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
    let use_q8_0_block_dot = q8_0_file_reader_block_dot_enabled();
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
                        if parallelize_output {
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
    #[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
fn q8_0_block_int_dot_horizontal_sum_encoded_impl(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    // SAFETY: both pointers address one complete Q8_0 block (32 signed bytes).
    unsafe { q8_0_i8_block_neon(weight.as_ptr().cast::<i8>(), input.as_ptr()) }
}

#[cfg(not(target_arch = "aarch64"))]
fn q8_0_block_int_dot_horizontal_sum_encoded_impl(
    weight: &[u8],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_64_avx2_enabled() {
            // SAFETY: AVX2 support is runtime-detected above and both slices contain one full
            // Q8_0 block (32 signed quant bytes). GGUF stores the quants as raw bytes; casting
            // through i8 preserves the two's-complement signed values used by the scalar path.
            return unsafe { q8_0_i8_block_avx2(weight.as_ptr().cast::<i8>(), input.as_ptr()) };
        }
    }
    let lanes = [
        q8_0_dot_group4_encoded(weight, input, 0) + q8_0_dot_group4_encoded(weight, input, 16),
        q8_0_dot_group4_encoded(weight, input, 4) + q8_0_dot_group4_encoded(weight, input, 20),
        q8_0_dot_group4_encoded(weight, input, 8) + q8_0_dot_group4_encoded(weight, input, 24),
        q8_0_dot_group4_encoded(weight, input, 12) + q8_0_dot_group4_encoded(weight, input, 28),
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

fn q8_0_block_int_dot_horizontal_sum(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    q8_0_block_int_dot_horizontal_sum_impl(weight, input)
}

#[cfg(target_arch = "aarch64")]
fn q8_0_block_int_dot_horizontal_sum_impl(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    // SAFETY: both pointers address one complete Q8_0 block (32 signed bytes).
    unsafe { q8_0_i8_block_neon(weight.as_ptr(), input.as_ptr()) }
}

#[cfg(not(target_arch = "aarch64"))]
fn q8_0_block_int_dot_horizontal_sum_impl(
    weight: &[i8; Q8_0_BLOCK_VALUES],
    input: &[i8; Q8_0_BLOCK_VALUES],
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_64_avx2_enabled() {
            // SAFETY: AVX2 support is runtime-detected above and both pointers address one full
            // Q8_0 block (32 signed quant bytes).
            return unsafe { q8_0_i8_block_avx2(weight.as_ptr(), input.as_ptr()) };
        }
    }
    // The generic q8_0 x q8_0 dot sums the 32 products in deterministic grouped scalar order.
    let lanes = [
        q8_0_dot_group4(weight, input, 0) + q8_0_dot_group4(weight, input, 16),
        q8_0_dot_group4(weight, input, 4) + q8_0_dot_group4(weight, input, 20),
        q8_0_dot_group4(weight, input, 8) + q8_0_dot_group4(weight, input, 24),
        q8_0_dot_group4(weight, input, 12) + q8_0_dot_group4(weight, input, 28),
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[cfg(not(target_arch = "aarch64"))]
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

#[cfg(not(target_arch = "aarch64"))]
fn q8_0_dot_group4_encoded(weight: &[u8], input: &[i8; Q8_0_BLOCK_VALUES], start: usize) -> i32 {
    i32::from(weight[start] as i8) * i32::from(input[start])
        + i32::from(weight[start + 1] as i8) * i32::from(input[start + 1])
        + i32::from(weight[start + 2] as i8) * i32::from(input[start + 2])
        + i32::from(weight[start + 3] as i8) * i32::from(input[start + 3])
}

#[cfg(target_arch = "x86_64")]
fn x86_64_avx2_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !q8_0_env_flag_disabled("CAMELID_X86_64_AVX2_Q8_DOT")
            && std::arch::is_x86_feature_detected!("avx2")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_i8_block_avx2(weight: *const i8, input: *const i8) -> i32 {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_madd_epi16,
    };

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    let weight_lo = unsafe { _mm_loadu_si128(weight.cast::<__m128i>()) };
    let input_lo = unsafe { _mm_loadu_si128(input.cast::<__m128i>()) };
    let weight_hi = unsafe { _mm_loadu_si128(weight.add(16).cast::<__m128i>()) };
    let input_hi = unsafe { _mm_loadu_si128(input.add(16).cast::<__m128i>()) };

    let weight_lo_i16 = _mm256_cvtepi8_epi16(weight_lo);
    let input_lo_i16 = _mm256_cvtepi8_epi16(input_lo);
    let weight_hi_i16 = _mm256_cvtepi8_epi16(weight_hi);
    let input_hi_i16 = _mm256_cvtepi8_epi16(input_hi);
    let products_lo = _mm256_madd_epi16(weight_lo_i16, input_lo_i16);
    let products_hi = _mm256_madd_epi16(weight_hi_i16, input_hi_i16);
    let products = _mm256_add_epi32(products_lo, products_hi);

    // Keep the horizontal reduction in registers. The old path spilled every 8-lane
    // partial dot to a stack array, which is brutal inside millions of Q8 block dots.
    let lo = _mm256_castsi256_si128(products);
    let hi = _mm256_extracti128_si256(products, 1);
    let sum4 = _mm_add_epi32(lo, hi);
    let shuffled = _mm_shuffle_epi32(sum4, 0b10_11_00_01);
    let sum2 = _mm_add_epi32(sum4, shuffled);
    let shuffled = _mm_shuffle_epi32(sum2, 0b01_00_11_10);
    let sum1 = _mm_add_epi32(sum2, shuffled);
    _mm_cvtsi128_si32(sum1)
}

#[cfg(target_arch = "aarch64")]
fn aarch64_dotprod_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !q8_0_env_flag_disabled("CAMELID_AARCH64_DOTPROD")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

#[cfg(target_arch = "aarch64")]
unsafe fn q8_0_i8_block_neon(weight: *const i8, input: *const i8) -> i32 {
    if aarch64_dotprod_enabled() {
        // SAFETY: feature detection above guarantees the dot-product instructions are
        // available, and callers pass pointers to at least 32 contiguous i8 values.
        return unsafe { q8_0_i8_block_dotprod(weight, input) };
    }

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    unsafe { q8_0_i8_block_neon_mul(weight, input) }
}

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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
    if let Some(backing) = borrowed_q8_0_reader_backing(weight, input_row.len(), output.len())? {
        accumulate_transposed_linear_row_q8_0_file_reader(input_row, backing, output)?;
        return Ok(());
    }
    accumulate_transposed_linear_row_with_precision(input_row, weight, output, precision);
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

fn accumulate_transposed_linear_row_q8_0_file_reader(
    input_row: &[f32],
    backing: &Q8_0FileBacking,
    output: &mut [f32],
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
    let use_q8_0_block_dot = q8_0_file_reader_block_dot_enabled();
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

fn accumulate_transposed_linear_row_with_precision(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
    precision: LinearAccumulationPrecision,
) {
    if should_use_borrowed_q8_0_block_dot(weight, input_row.len()) {
        accumulate_transposed_linear_row_q8_0_block_dot(input_row, weight, output);
        return;
    }
    match precision {
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

fn accumulate_transposed_linear_row_q8_0_block_dot(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
    let blocks_per_row = input_row.len() / Q8_0_BLOCK_VALUES;
    let weight_blocks = weight
        .q8_0_blocks
        .expect("q8_0 block-dot precondition checked");
    debug_assert_eq!(weight_blocks.len(), output.len() * blocks_per_row);
    let quantized_input = quantize_q8_0_row(input_row);
    if should_parallelize_linear_output(output.len()) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_value)| {
                let weight_start = out_idx * blocks_per_row;
                *out_value = q8_0_dot_rows(
                    &weight_blocks[weight_start..weight_start + blocks_per_row],
                    &quantized_input.blocks,
                );
            });
        return;
    }
    for (out_idx, out_value) in output.iter_mut().enumerate() {
        let weight_start = out_idx * blocks_per_row;
        *out_value = q8_0_dot_rows(
            &weight_blocks[weight_start..weight_start + blocks_per_row],
            &quantized_input.blocks,
        );
    }
}

fn accumulate_transposed_linear_row(
    input_row: &[f32],
    weight: BorrowedLinearWeight<'_>,
    output: &mut [f32],
) {
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
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use std::io::Write;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_slice_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "slice length mismatch");
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() < 1e-5,
                "expected index {idx} to be {expected}, got {actual}"
            );
        }
    }

    fn no_rope_scaling() -> RopeScaling {
        RopeScaling {
            kind: RopeScalingKind::None,
            factor: 1.0,
            original_context_length: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }
    }

    #[test]
    fn q8_0_block_reader_smoke() {
        let _q8_guard = crate::test_support::q8_file_state_lock();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let scale_bits = 0x3c00u16;
        let mut block_data = vec![0u8; Q8BlockReader::BLOCK_SIZE_BYTES];
        block_data[0..2].copy_from_slice(&scale_bits.to_le_bytes());
        block_data[2] = 10i8 as u8;
        block_data[3] = 20i8 as u8;

        temp_file.write_all(&block_data).unwrap();
        temp_file.flush().unwrap();

        let reader = Q8BlockReader::new(0, 1);
        let file = temp_file.reopen().unwrap();
        let mut dest = vec![0.0; Q8BlockReader::WEIGHTS_PER_BLOCK];
        reader
            .dequantize_block_to_slice(&file, 0, &mut dest)
            .unwrap();

        assert_eq!(dest[0], 10.0);
        assert_eq!(dest[1], 20.0);
        assert!(dest[2..].iter().all(|value| *value == 0.0));
    }

    fn write_q8_0_test_block(
        out: &mut impl Write,
        scale: f32,
        quants: [i8; Q8_0_BLOCK_VALUES],
    ) -> Q8_0Block {
        let scale_bits = f32_to_f16_bits(scale);
        out.write_all(&scale_bits.to_le_bytes()).unwrap();
        out.write_all(&quants.map(|value| value as u8)).unwrap();
        Q8_0Block {
            scale: f16_bits_to_f32(scale_bits),
            quants,
        }
    }

    #[test]
    fn q8_file_backed_output_projection_reuses_weight_read_across_batch_rows() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut first_row = [0_i8; Q8_0_BLOCK_VALUES];
        let mut second_row = [0_i8; Q8_0_BLOCK_VALUES];
        for idx in 0..Q8_0_BLOCK_VALUES {
            first_row[idx] = (idx as i8 % 7) - 3;
            second_row[idx] = 4 - (idx as i8 % 9);
        }
        let weight_blocks = [
            write_q8_0_test_block(&mut temp_file, 0.5, first_row),
            write_q8_0_test_block(&mut temp_file, 0.25, second_row),
        ];
        temp_file.flush().unwrap();

        let input = CpuTensor::from_f32(
            "prefill-output-norm",
            vec![3, Q8_0_BLOCK_VALUES],
            (0..(3 * Q8_0_BLOCK_VALUES))
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.05)
                .collect(),
        )
        .unwrap();
        let weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            TensorShape {
                dims: vec![2, Q8_0_BLOCK_VALUES],
            },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, weight_blocks.len()),
        );
        let start = q8_0_file_read_stats();

        let actual = output_projection_with_layout(
            &input,
            &weight,
            "logits",
            OutputProjectionLayout::TokenMajor,
        )
        .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        let mut expected = Vec::new();
        for input_row in input.data.chunks_exact(Q8_0_BLOCK_VALUES) {
            let quantized_input = quantize_q8_0_row(input_row);
            expected.push(q8_0_dot_rows(&weight_blocks[0..1], &quantized_input.blocks));
            expected.push(q8_0_dot_rows(&weight_blocks[1..2], &quantized_input.blocks));
        }
        assert_eq!(actual.shape.dims, vec![3, 2]);
        assert_slice_close(&actual.data, &expected);
        assert_eq!(reads.read_calls, 1);
        assert_eq!(
            reads.read_bytes,
            (weight_blocks.len() * Q8BlockReader::BLOCK_SIZE_BYTES) as u64
        );
    }

    #[test]
    fn q8_file_backed_output_projection_empty_batch_skips_weight_reads() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let weight_blocks = [
            write_q8_0_test_block(&mut temp_file, 1.0, [1_i8; Q8_0_BLOCK_VALUES]),
            write_q8_0_test_block(&mut temp_file, 1.0, [-1_i8; Q8_0_BLOCK_VALUES]),
        ];
        temp_file.flush().unwrap();

        let input = CpuTensor::from_f32(
            "empty-prefill-output-norm",
            vec![0, Q8_0_BLOCK_VALUES],
            Vec::new(),
        )
        .unwrap();
        let weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            TensorShape {
                dims: vec![weight_blocks.len(), Q8_0_BLOCK_VALUES],
            },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, weight_blocks.len()),
        );
        let start = q8_0_file_read_stats();

        let actual = output_projection_with_layout(
            &input,
            &weight,
            "logits",
            OutputProjectionLayout::TokenMajor,
        )
        .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(actual.shape.dims, vec![0, weight_blocks.len()]);
        assert!(actual.data.is_empty());
        assert_eq!(reads.read_calls, 0);
        assert_eq!(reads.read_bytes, 0);
        assert!(!weight
            .q8_0_file_backing
            .as_ref()
            .unwrap()
            .file_handle_cached());
    }

    fn memory_sample(
        rss_kib: u64,
        kv_position: usize,
        allocated_sequence_length: usize,
    ) -> LlamaMemorySample {
        let elements = allocated_sequence_length * 2;
        LlamaMemorySample {
            rss_kib: Some(rss_kib),
            kv_cache_position: kv_position,
            kv_cache_allocated_sequence_length: allocated_sequence_length,
            kv_cache_allocated_elements: elements,
            kv_cache_allocated_bytes: (elements * std::mem::size_of::<f32>()) as u64,
        }
    }

    fn test_forward_memory(start: LlamaMemorySample) -> LlamaForwardMemoryTimings {
        LlamaForwardMemoryTimings::new(
            start,
            LlamaWeightMaterializationStats::default(),
            Q8_0FileReadStats::default(),
        )
    }

    fn tiny_prefill_schedule_weights(attention_q: CpuTensor) -> LlamaLoadedWeights {
        LlamaLoadedWeights {
            token_embedding: CpuTensor::from_f32("token_embd.weight", vec![2, 2], vec![1.0; 4])
                .unwrap(),
            output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0; 2]).unwrap(),
            output: None,
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: CpuTensor::from_f32(
                    "blk.0.attn_norm.weight",
                    vec![2],
                    vec![1.0; 2],
                )
                .unwrap(),
                attention_q,
                attention_k: CpuTensor::from_f32("blk.0.attn_k.weight", vec![2, 2], vec![1.0; 4])
                    .unwrap(),
                attention_v: CpuTensor::from_f32("blk.0.attn_v.weight", vec![2, 2], vec![1.0; 4])
                    .unwrap(),
                attention_output: CpuTensor::from_f32(
                    "blk.0.attn_output.weight",
                    vec![2, 2],
                    vec![1.0; 4],
                )
                .unwrap(),
                ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0; 2])
                    .unwrap(),
                ffn_gate: CpuTensor::from_f32("blk.0.ffn_gate.weight", vec![2, 2], vec![1.0; 4])
                    .unwrap(),
                ffn_up: CpuTensor::from_f32("blk.0.ffn_up.weight", vec![2, 2], vec![1.0; 4])
                    .unwrap(),
                ffn_down: CpuTensor::from_f32("blk.0.ffn_down.weight", vec![2, 2], vec![1.0; 4])
                    .unwrap(),
                moe_router: None,
            }],
        }
    }

    #[test]
    fn prefill_chunk_token_count_accepts_full_prompt_probe() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
        assert_eq!(prefill_chunk_token_count(2047), 256);

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "256");
        assert_eq!(prefill_chunk_token_count(2047), 256);

        for value in ["all", "full", "prompt", "unbounded", " FULL "] {
            std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", value);
            assert_eq!(prefill_chunk_token_count(2047), 2047);
        }

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "0");
        assert_eq!(prefill_chunk_token_count(2047), 256);
        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    }

    #[test]
    fn prefill_layer_major_chunk_token_count_has_separate_headroom_default() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
        assert_eq!(prefill_chunk_token_count(2047), 256);
        assert_eq!(prefill_layer_major_chunk_token_count(2047), 512);

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "128");
        assert_eq!(prefill_layer_major_chunk_token_count(2047), 128);

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "1024");
        assert_eq!(prefill_layer_major_chunk_token_count(2047), 1024);

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "all");
        assert_eq!(prefill_layer_major_chunk_token_count(2047), 2047);

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "0");
        assert_eq!(prefill_layer_major_chunk_token_count(2047), 512);
        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
    }

    #[test]
    fn q8_file_reader_batch_chunk_rows_respect_output_scratch_budget() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "1024");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64");

        assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 32);
        assert_eq!(
            q8_0_file_reader_chunk_rows_for_batch(32, 100, 1, true).unwrap(),
            32
        );
        assert_eq!(
            q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, true).unwrap(),
            2
        );
        assert_eq!(
            q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, false).unwrap(),
            32
        );
        assert_eq!(q8_0_file_reader_chunk_rows(32, 63).unwrap(), 63);
        assert_eq!(q8_0_file_reader_chunk_rows(32, 64).unwrap(), 64);
        assert_eq!(q8_0_file_reader_chunk_rows(32, 65).unwrap(), 32);

        std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "1 KiB");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64_B");
        assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 32);
        assert_eq!(
            q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, true).unwrap(),
            2
        );
        assert_eq!(
            q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, false).unwrap(),
            32
        );

        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");
    }

    #[test]
    fn q8_file_reader_parallel_output_falls_back_when_default_scratch_fragments_full_tensor_read() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "4096");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            assert!(should_parallelize_q8_0_file_reader_output(100));
            assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 100);
            assert_eq!(
                q8_0_file_reader_output_scratch_chunk_rows(1_000_000, 100).unwrap(),
                16
            );
            assert!(!should_use_q8_0_file_reader_parallel_output(32, 100, 1_000_000).unwrap());

            std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64");
            assert!(should_use_q8_0_file_reader_parallel_output(32, 100, 8).unwrap());
        });

        std::env::remove_var("CAMELID_PARALLEL_LINEAR");
        std::env::remove_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");
    }

    #[test]
    fn q8_file_reader_default_coalesces_llama3_8b_ffn_q8_shapes() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");

        let llama3_8b_hidden_row_bytes = 4096 / Q8_0_BLOCK_VALUES * Q8BlockReader::BLOCK_SIZE_BYTES;
        let llama3_8b_ffn_row_bytes = 14336 / Q8_0_BLOCK_VALUES * Q8BlockReader::BLOCK_SIZE_BYTES;

        assert_eq!(
            q8_0_file_reader_chunk_rows(llama3_8b_hidden_row_bytes, 14336).unwrap(),
            14336
        );
        assert_eq!(
            q8_0_file_reader_chunk_rows(llama3_8b_ffn_row_bytes, 4096).unwrap(),
            4096
        );
    }

    #[test]
    fn prefill_layer_major_defaults_only_for_lazy_q8_backing() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let dense_weights = tiny_prefill_schedule_weights(
            CpuTensor::from_f32("blk.0.attn_q.weight", vec![2, 2], vec![1.0; 4]).unwrap(),
        );
        assert!(!prefill_layer_major_enabled(&dense_weights));

        let lazy_q8_attention_q = CpuTensor::from_f32_with_source_type(
            "blk.0.attn_q.weight",
            vec![2, 2],
            vec![1.0; 4],
            Some(GgufTensorType::Q8_0),
        )
        .unwrap()
        .with_q8_0_file_backing(Q8_0FileBacking::new("unused.gguf".into(), 0, 1));
        let lazy_q8_weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);
        assert!(prefill_layer_major_enabled(&lazy_q8_weights));

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", "1");
        assert!(prefill_layer_major_enabled(&dense_weights));

        for value in ["0", "false", "off", "disabled", " FALSE ", "Off"] {
            std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", value);
            assert!(!prefill_layer_major_enabled(&lazy_q8_weights));
        }

        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR");
    }

    #[test]
    fn prefill_layer_major_q8_cache_uses_scoped_default_only_for_lazy_q8() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let dense_weights = tiny_prefill_schedule_weights(
            CpuTensor::from_f32("blk.0.attn_q.weight", vec![2, 2], vec![1.0; 4]).unwrap(),
        );
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&dense_weights, 2),
            None
        );

        let lazy_q8_attention_q = CpuTensor::from_f32_with_source_type(
            "blk.0.attn_q.weight",
            vec![2, 2],
            vec![1.0; 4],
            Some(GgufTensorType::Q8_0),
        )
        .unwrap()
        .with_q8_0_file_backing(Q8_0FileBacking::new("unused.gguf".into(), 0, 1));
        let lazy_q8_weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
            None
        );
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 2),
            Some(DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES)
        );

        let large_layer_blocks =
            (DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES / Q8BlockReader::BLOCK_SIZE_BYTES) + 1;
        let large_layer_capacity = large_layer_blocks * Q8BlockReader::BLOCK_SIZE_BYTES;
        let large_lazy_q8_attention_q = CpuTensor::q8_0_file_backed_linear(
            "blk.0.attn_q.weight",
            TensorShape { dims: vec![1, 32] },
            Q8_0FileBacking::new("unused.gguf".into(), 0, large_layer_blocks),
        );
        let large_lazy_q8_weights = tiny_prefill_schedule_weights(large_lazy_q8_attention_q);
        assert_eq!(
            large_lazy_q8_weights.largest_q8_0_file_backed_layer_storage_bytes(),
            large_layer_capacity as u64
        );
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&large_lazy_q8_weights, 2),
            Some(large_layer_capacity)
        );

        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "64 MiB");
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 2),
            None
        );

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES", "0");
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
            Some(0)
        );

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES", "1 MiB");
        assert_eq!(
            prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
            Some(1024 * 1024)
        );
    }

    #[test]
    fn prefill_layer_major_scoped_q8_cache_reuses_file_reads_across_chunks() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        let _ = q8_0_file_read_stats();

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for _ in 0..32 {
            temp_file
                .write_all(&f32_to_f16_bits(1.0).to_le_bytes())
                .unwrap();
            temp_file.write_all(&[0_u8; Q8_0_BLOCK_VALUES]).unwrap();
        }
        temp_file.flush().unwrap();

        let config = LlamaModelConfig {
            context_length: 2,
            embedding_length: 32,
            block_count: 1,
            feed_forward_length: 32,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(32),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1.0e-5,
            vocab_size: Some(2),
            file_type: None,
            moe: None,
        };
        let dense_vector = |name: &str| CpuTensor::from_f32(name, vec![32], vec![1.0; 32]).unwrap();
        let dense_matrix =
            |name: &str| CpuTensor::from_f32(name, vec![32, 32], vec![0.0; 32 * 32]).unwrap();
        let attention_q = CpuTensor::q8_0_file_backed_linear(
            "blk.0.attn_q.weight",
            TensorShape { dims: vec![32, 32] },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 32),
        );
        let weights = LlamaLoadedWeights {
            token_embedding: CpuTensor::from_f32(
                "token_embd.weight",
                vec![2, 32],
                (0..64).map(|idx| idx as f32 * 0.001).collect(),
            )
            .unwrap(),
            output_norm: dense_vector("output_norm.weight"),
            output: None,
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: dense_vector("blk.0.attn_norm.weight"),
                attention_q,
                attention_k: dense_matrix("blk.0.attn_k.weight"),
                attention_v: dense_matrix("blk.0.attn_v.weight"),
                attention_output: dense_matrix("blk.0.attn_output.weight"),
                ffn_norm: dense_vector("blk.0.ffn_norm.weight"),
                ffn_gate: dense_matrix("blk.0.ffn_gate.weight"),
                ffn_up: dense_matrix("blk.0.ffn_up.weight"),
                ffn_down: dense_matrix("blk.0.ffn_down.weight"),
                moe_router: None,
            }],
        };
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let start = q8_0_file_read_stats();

        let timings = session
            .forward_prefill_layer_major_timed_fast(&[0, 1], 1)
            .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(timings.layers.len(), 1);
        assert_eq!(session.kv_cache.position, 2);
        assert_eq!(reads.read_calls, 1);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * 32) as u64
        );
        assert_eq!(reads.cache_hits, 1);
        assert_eq!(
            reads.cache_hit_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * 32) as u64
        );
    }

    #[test]
    fn materialization_stats_quantify_lazy_q8_file_backing_tradeoff() {
        let lazy_q8_attention_q = CpuTensor::q8_0_file_backed_linear(
            "blk.0.attn_q.weight",
            crate::tensor::TensorShape { dims: vec![2, 64] },
            Q8_0FileBacking::new("unused.gguf".into(), 0, 4),
        );
        let weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);

        let stats = collect_weight_materialization_stats(&weights);

        assert_eq!(stats.q8_0_file_backed_tensor_count, 1);
        assert_eq!(stats.q8_0_file_backed_storage_bytes, 4 * 34);
        assert_eq!(stats.q8_0_file_backed_f32_bytes_avoided, 4 * 32 * 4);
        assert_eq!(
            stats.q8_0_file_backed_retained_block_bytes_if_enabled,
            4 * std::mem::size_of::<Q8_0Block>() as u64
        );
        assert_eq!(stats.q8_0_retained_block_bytes, 0);
        assert!(stats.has_lazy_q8_0_file_backing);
        assert!(!stats.has_retained_q8_0_blocks);
        assert!(!stats.has_q8_0_f32_materialization);
    }

    #[test]
    fn memory_timing_merge_tracks_forward_passes_and_peak_rss() {
        let mut first = LlamaForwardTimings {
            memory: Some(test_forward_memory(memory_sample(100, 0, 0))),
            ..LlamaForwardTimings::default()
        };
        first
            .memory
            .as_mut()
            .unwrap()
            .record_after_logits(memory_sample(110, 0, 1));
        first
            .memory
            .as_mut()
            .unwrap()
            .q8_file_read_phases
            .push(LlamaQ8FileReadPhaseTrace {
                phase: "logits_done".to_string(),
                q8_file_reads: Q8_0FileReadStats {
                    read_calls: 3,
                    read_bytes: 256,
                    cache_hits: 1,
                    cache_hit_bytes: 64,
                    cache_entries: 2,
                    cache_bytes: 512,
                    cache_capacity_bytes: 1024,
                    ..Q8_0FileReadStats::default()
                },
            });

        let mut second = LlamaForwardTimings {
            memory: Some(test_forward_memory(memory_sample(105, 1, 1))),
            ..LlamaForwardTimings::default()
        };
        second
            .memory
            .as_mut()
            .unwrap()
            .record_after_layers(memory_sample(140, 1, 2));
        second
            .memory
            .as_mut()
            .unwrap()
            .q8_file_read_phases
            .push(LlamaQ8FileReadPhaseTrace {
                phase: "layers_done".to_string(),
                q8_file_reads: Q8_0FileReadStats {
                    read_calls: 4,
                    read_bytes: 1024,
                    cache_hits: 2,
                    cache_hit_bytes: 128,
                    cache_entries: 3,
                    cache_bytes: 768,
                    cache_capacity_bytes: 1024,
                    ..Q8_0FileReadStats::default()
                },
            });
        first.memory.as_mut().unwrap().q8_file_reads = Q8_0FileReadStats {
            read_calls: 3,
            read_bytes: 256,
            cache_hits: 1,
            cache_hit_bytes: 64,
            cache_entries: 2,
            cache_bytes: 512,
            cache_capacity_bytes: 1024,
            ..Q8_0FileReadStats::default()
        };
        second.memory.as_mut().unwrap().q8_file_reads = Q8_0FileReadStats {
            read_calls: 4,
            read_bytes: 1024,
            cache_hits: 2,
            cache_hit_bytes: 128,
            cache_entries: 3,
            cache_bytes: 768,
            cache_capacity_bytes: 1024,
            ..Q8_0FileReadStats::default()
        };

        first.add_assign(&second);

        let memory = first.memory.expect("merged memory timings");
        assert_eq!(memory.forward_passes, 2);
        assert_eq!(
            memory.q8_file_reads,
            Q8_0FileReadStats {
                read_calls: 7,
                read_bytes: 1280,
                cache_hits: 3,
                cache_hit_bytes: 192,
                cache_entries: 3,
                cache_bytes: 768,
                cache_capacity_bytes: 1024,
                ..Q8_0FileReadStats::default()
            }
        );
        assert_eq!(memory.peak_rss_kib, Some(140));
        assert_eq!(memory.peak_rss_delta_kib, Some(40));
        assert_eq!(memory.peak_phase.as_deref(), Some("layers_done"));
        assert_eq!(memory.q8_file_read_phases.len(), 2);
        assert_eq!(memory.q8_file_read_phases[0].phase, "logits_done");
        assert_eq!(
            memory.q8_file_read_phases[0].q8_file_reads.cache_hit_bytes,
            64
        );
        assert_eq!(memory.q8_file_read_phases[1].phase, "layers_done");
        assert_eq!(memory.q8_file_read_phases[1].q8_file_reads.read_bytes, 1024);
        assert_eq!(memory.end, None);
        assert_eq!(
            memory
                .after_layers
                .unwrap()
                .kv_cache_allocated_sequence_length,
            2
        );
    }

    #[test]
    fn q8_file_read_stats_merge_keeps_peak_cache_state() {
        let mut target = Q8_0FileReadStats {
            read_calls: 2,
            read_bytes: 128,
            cache_entries: 4,
            cache_bytes: 1024,
            cache_capacity_bytes: 2048,
            ..Q8_0FileReadStats::default()
        };
        let delta = Q8_0FileReadStats {
            read_calls: 3,
            read_bytes: 256,
            cache_hits: 1,
            cache_hit_bytes: 64,
            cache_entries: 0,
            cache_bytes: 0,
            cache_capacity_bytes: 0,
            ..Q8_0FileReadStats::default()
        };

        add_q8_file_read_stats_delta(&mut target, delta);

        assert_eq!(target.read_calls, 5);
        assert_eq!(target.read_bytes, 384);
        assert_eq!(target.cache_hits, 1);
        assert_eq!(target.cache_hit_bytes, 64);
        assert_eq!(target.cache_entries, 4);
        assert_eq!(target.cache_bytes, 1024);
        assert_eq!(target.cache_capacity_bytes, 2048);
    }

    #[test]
    fn layer_memory_merge_accumulates_q8_file_reads() {
        let mut first = LlamaLayerMemoryTimings::new(3, memory_sample(100, 0, 0));
        first.q8_file_reads = Q8_0FileReadStats {
            read_calls: 2,
            read_bytes: 128,
            cache_hits: 1,
            cache_hit_bytes: 32,
            cache_entries: 1,
            cache_bytes: 256,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        };
        let mut second = LlamaLayerMemoryTimings::new(3, memory_sample(105, 1, 1));
        second.q8_file_reads = Q8_0FileReadStats {
            read_calls: 5,
            read_bytes: 512,
            cache_hits: 3,
            cache_hit_bytes: 96,
            cache_entries: 2,
            cache_bytes: 384,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        };
        first.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
            phase: "attention_q_done".to_string(),
            q8_file_reads: Q8_0FileReadStats {
                read_calls: 2,
                read_bytes: 128,
                cache_hits: 1,
                cache_hit_bytes: 32,
                cache_entries: 1,
                cache_bytes: 256,
                cache_capacity_bytes: 512,
                ..Q8_0FileReadStats::default()
            },
        });
        second.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
            phase: "attention_q_done".to_string(),
            q8_file_reads: Q8_0FileReadStats {
                read_calls: 3,
                read_bytes: 384,
                cache_hits: 2,
                cache_hit_bytes: 64,
                cache_entries: 2,
                cache_bytes: 384,
                cache_capacity_bytes: 512,
                ..Q8_0FileReadStats::default()
            },
        });
        second.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
            phase: "ffn_down_done".to_string(),
            q8_file_reads: Q8_0FileReadStats {
                read_calls: 2,
                read_bytes: 128,
                cache_hits: 1,
                cache_hit_bytes: 32,
                cache_entries: 2,
                cache_bytes: 384,
                cache_capacity_bytes: 512,
                ..Q8_0FileReadStats::default()
            },
        });

        first.merge_assign(&second);

        assert_eq!(first.forward_passes, 2);
        assert_eq!(
            first.q8_file_reads,
            Q8_0FileReadStats {
                read_calls: 7,
                read_bytes: 640,
                cache_hits: 4,
                cache_hit_bytes: 128,
                cache_entries: 2,
                cache_bytes: 384,
                cache_capacity_bytes: 512,
                ..Q8_0FileReadStats::default()
            }
        );
        assert_eq!(first.q8_file_read_phases.len(), 2);
        assert_eq!(first.q8_file_read_phases[0].phase, "attention_q_done");
        assert_eq!(first.q8_file_read_phases[0].q8_file_reads.read_calls, 5);
        assert_eq!(first.q8_file_read_phases[0].q8_file_reads.read_bytes, 512);
        assert_eq!(first.q8_file_read_phases[0].q8_file_reads.cache_hits, 3);
        assert_eq!(
            first.q8_file_read_phases[0].q8_file_reads.cache_hit_bytes,
            96
        );
        assert_eq!(first.q8_file_read_phases[1].phase, "ffn_down_done");
        assert_eq!(first.q8_file_read_phases[1].q8_file_reads.read_calls, 2);
        assert_eq!(
            first.q8_file_read_phases[1].q8_file_reads.cache_hit_bytes,
            32
        );

        first.record_after_attention_output(memory_sample(160, 1, 1));
        assert_eq!(first.peak_rss_kib, Some(160));
        assert_eq!(first.peak_rss_delta_kib, Some(60));
        assert_eq!(first.peak_phase.as_deref(), Some("attention_output_done"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_proc_status_rss_kib() {
        assert_eq!(
            parse_proc_status_rss_kib("Name:\tcamelid\nVmRSS:\t  12345 kB\n"),
            Some(12_345)
        );
    }

    fn clear_dense_diagnostic_env() {
        for key in [
            "CAMELID_ATTENTION_SCORE_SCALE",
            "CAMELID_FFN_GATE_UP_ORDER",
            "CAMELID_FORWARD_MEMORY_TRACE",
            "CAMELID_FORWARD_RSS_TIMINGS",
            "CAMELID_GQA_HEAD_MAPPING",
            "CAMELID_LINEAR_ACCUMULATION",
            "CAMELID_OUTPUT_PROJECTION_LAYOUT",
            "CAMELID_PREFILL_LAYER_MAJOR",
            "CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES",
            "CAMELID_PARALLEL_LINEAR",
            "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS",
            "CAMELID_Q8_0_BLOCK_DOT",
            "CAMELID_Q8_0_FILE_READER_BLOCK_DOT",
            "CAMELID_Q8_0_FILE_CACHE_BYTES",
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            "CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES",
            "CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES",
            "CAMELID_PARALLEL_LINEAR",
            "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_V",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_DOWN",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_GATE",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_UP",
            "CAMELID_RMS_NORM_EPSILON",
            "CAMELID_ROPE_DIRECTION",
            "CAMELID_ROPE_PAIRING",
            "CAMELID_ROPE_POSITION_MODE",
            "CAMELID_SQUARE_LINEAR_LAYOUT",
        ] {
            std::env::remove_var(key);
        }
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    #[test]
    fn final_norm_diagnostics_reconstruct_output_norm_values() {
        let hidden = CpuTensor::from_f32("hidden", vec![1, 4], vec![3.0, 4.0, 0.0, -5.0]).unwrap();
        let weight =
            CpuTensor::from_f32("output_norm.weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output_norm = hidden.rms_norm(&weight, 1e-5, "output_norm").unwrap();

        let diagnostic = final_norm_diagnostics(&hidden, &weight, &output_norm, 1e-5).unwrap();

        assert_close(diagnostic.hidden_mean_square, 12.5);
        assert_close(diagnostic.hidden_rms, 12.5_f32.sqrt());
        assert_eq!(diagnostic.hidden_first_values, vec![3.0, 4.0, 0.0, -5.0]);
        assert_eq!(diagnostic.weight_first_values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(diagnostic.reconstructed_first_values.len(), 4);
        assert_eq!(diagnostic.reported_first_values, output_norm.data);
        assert_eq!(diagnostic.reported_max_abs_index, 3);
        assert_close(diagnostic.reported_max_abs, output_norm.data[3].abs());
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, output_norm.data);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            output_norm.data
        );
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn rms_norm_diagnostics_report_peak_window() {
        let input = CpuTensor::from_f32("input", vec![1, 4], vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let weight = CpuTensor::from_f32("norm.weight", vec![4], vec![0.5, 1.0, 1.5, 2.0]).unwrap();
        let reported = input.rms_norm(&weight, 1e-5, "reported").unwrap();

        let diagnostic = rms_norm_diagnostics(&input, &weight, &reported, 1e-5).unwrap();

        assert_eq!(diagnostic.reported_max_abs_index, 3);
        assert_close(diagnostic.reported_max_abs, reported.data[3].abs());
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, reported.data);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            reported.data
        );
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn residual_diagnostics_report_delta_scale_and_alignment() {
        let input = CpuTensor::from_f32("input", vec![1, 4], vec![3.0, 4.0, 0.0, -5.0]).unwrap();
        let delta = CpuTensor::from_f32("delta", vec![1, 4], vec![1.0, -2.0, 0.0, 2.0]).unwrap();
        let reported = input.add(&delta, "reported").unwrap();

        let diagnostic = residual_reconstruction_diagnostic(&input, &delta, &reported).unwrap();

        assert_close(diagnostic.input_rms, 12.5_f32.sqrt());
        assert_close(diagnostic.delta_rms, 2.25_f32.sqrt());
        assert_close(diagnostic.reported_rms, 7.25_f32.sqrt());
        assert_close(
            diagnostic.delta_to_input_rms_ratio,
            2.25_f32.sqrt() / 12.5_f32.sqrt(),
        );
        assert_close(
            diagnostic.delta_input_cosine_similarity,
            -15.0 / (50.0_f32.sqrt() * 9.0_f32.sqrt()),
        );
        assert_eq!(diagnostic.reconstructed_first_values, reported.data);
        assert_eq!(diagnostic.reported_max_abs_index, 0);
        assert_close(diagnostic.reported_max_abs, 4.0);
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, reported.data);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            reported.data
        );
        assert_eq!(diagnostic.delta_reported_max_abs_window, delta.data);
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn linear_projection_diagnostics_reconstruct_descriptor_layout() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K");
        std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
        let weight =
            CpuTensor::from_f32("weight", vec![3, 2], vec![1.0, 2.0, -3.0, 4.0, 0.5, -2.0])
                .unwrap();
        let reported = linear_with_diagnostic_layouts(
            &input,
            &weight,
            "reported",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Descriptor,
        )
        .unwrap();

        let diagnostic =
            linear_projection_diagnostics(&input, &weight, &reported, "attention_k").unwrap();

        assert_eq!(diagnostic.layout, "descriptor");
        assert_eq!(diagnostic.input_width, 3);
        assert_eq!(diagnostic.output_width, 2);
        assert_eq!(diagnostic.weight_shape, vec![3, 2]);
        assert_eq!(diagnostic.reconstructed_first_values, reported.data);
        assert_eq!(diagnostic.reported_max_abs_index, 0);
        assert_close(diagnostic.reported_max_abs, 5.25);
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, reported.data);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            reported.data
        );
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn linear_projection_diagnostics_reconstruct_transposed_layout() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_V");
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
        let weight =
            CpuTensor::from_f32("weight", vec![2, 3], vec![1.0, -3.0, 0.5, 2.0, 4.0, -2.0])
                .unwrap();
        let reported = linear_with_diagnostic_layouts(
            &input,
            &weight,
            "reported",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Auto,
        )
        .unwrap();

        let diagnostic =
            linear_projection_diagnostics(&input, &weight, &reported, "attention_v").unwrap();

        assert_eq!(diagnostic.layout, "transposed_auto");
        assert_eq!(diagnostic.input_width, 3);
        assert_eq!(diagnostic.output_width, 2);
        assert_eq!(diagnostic.weight_shape, vec![2, 3]);
        assert_eq!(diagnostic.reconstructed_first_values, reported.data);
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn linear_projection_diagnostics_report_nonzero_reconstruction_delta() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_Q");
        std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
        let weight =
            CpuTensor::from_f32("weight", vec![3, 2], vec![1.0, 2.0, -3.0, 4.0, 0.5, -2.0])
                .unwrap();
        let reported = CpuTensor::from_f32("reported", vec![1, 2], vec![5.25, -2.75]).unwrap();

        let diagnostic =
            linear_projection_diagnostics(&input, &weight, &reported, "attention_q").unwrap();

        assert_eq!(diagnostic.layout, "descriptor");
        assert_eq!(diagnostic.reconstructed_first_values, vec![5.25, -1.0]);
        assert_eq!(diagnostic.reported_first_values, vec![5.25, -2.75]);
        assert_eq!(diagnostic.reported_max_abs_index, 0);
        assert_close(diagnostic.reported_max_abs, 5.25);
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, vec![5.25, -2.75]);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            vec![5.25, -1.0]
        );
        assert_eq!(diagnostic.max_abs_delta_index, 1);
        assert_close(diagnostic.max_abs_delta, 1.75);
    }

    #[test]
    fn parallel_linear_matches_serial_descriptor_transposed_and_q8_rows() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");

        let input = CpuTensor::from_f32("input", vec![1, 4], vec![2.0, -1.0, 0.5, 3.0]).unwrap();
        let descriptor_weight = CpuTensor::from_f32(
            "descriptor.weight",
            vec![4, 3],
            vec![
                1.0, -2.0, 0.25, -3.0, 4.0, 0.5, 0.5, -1.0, 2.0, 2.0, 0.25, -0.75,
            ],
        )
        .unwrap();
        let transposed_weight = CpuTensor::from_f32(
            "transposed.weight",
            vec![3, 4],
            vec![
                1.0, -3.0, 0.5, 2.0, -2.0, 4.0, -1.0, 0.25, 0.25, 0.5, 2.0, -0.75,
            ],
        )
        .unwrap();

        let serial_descriptor = linear_with_diagnostic_layouts(
            &input,
            &descriptor_weight,
            "serial_descriptor",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Descriptor,
        )
        .unwrap();
        let serial_transposed = linear_with_diagnostic_layouts(
            &input,
            &transposed_weight,
            "serial_transposed",
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Transposed,
        )
        .unwrap();

        std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
        let parallel_descriptor = linear_with_diagnostic_layouts(
            &input,
            &descriptor_weight,
            "parallel_descriptor",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Descriptor,
        )
        .unwrap();
        let parallel_transposed = linear_with_diagnostic_layouts(
            &input,
            &transposed_weight,
            "parallel_transposed",
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Transposed,
        )
        .unwrap();

        assert_eq!(parallel_descriptor.data, serial_descriptor.data);
        assert_eq!(parallel_transposed.data, serial_transposed.data);

        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");
        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let q8_input = CpuTensor::from_f32("q8_input", vec![1, 32], input_values).unwrap();
        let row0 = Q8_0Block {
            scale: 0.5,
            quants: std::array::from_fn(|idx| idx as i8 - 16),
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
        };
        let mut dequantized_weight = Vec::with_capacity(64);
        for block in [&row0, &row1] {
            dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
        }
        let q8_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "q8_weight",
            vec![2, 32],
            dequantized_weight,
            vec![row0, row1],
        )
        .unwrap();
        let serial_q8 =
            matmul_rhs_transposed_with_precision(&q8_input, &q8_weight, "serial_q8").unwrap();
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
        let parallel_q8 =
            matmul_rhs_transposed_with_precision(&q8_input, &q8_weight, "parallel_q8").unwrap();

        assert_eq!(parallel_q8.data, serial_q8.data);
    }

    #[test]
    fn q8_0_block_dot_defaults_to_quantized_fast_path() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let input_values = vec![0.25; 32];
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values).unwrap();
        let block = Q8_0Block {
            scale: 1.0,
            quants: [1; 32],
        };
        let weight =
            CpuTensor::from_f32_with_q8_0_blocks("weight", vec![1, 32], vec![1.0; 32], vec![block])
                .unwrap();

        assert!(should_use_q8_0_block_dot(&weight, 32));
        let actual = matmul_rhs_transposed_with_precision(&input, &weight, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![1, 1]);
        assert!(
            (actual.data[0] - 8.0).abs() < 1.0e-3,
            "expected quantized fast path to stay close to dequantized output, got {}",
            actual.data[0]
        );
    }

    #[test]
    fn q8_0_block_dot_defaults_on_with_separate_escape_hatches() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        assert!(q8_0_block_dot_enabled());
        assert!(q8_0_file_reader_block_dot_enabled());

        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "off");
        assert!(!q8_0_file_reader_block_dot_enabled());

        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
        assert!(!q8_0_block_dot_enabled());
        assert!(!q8_0_file_reader_block_dot_enabled());

        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
        assert!(q8_0_file_reader_block_dot_enabled());
    }

    #[test]
    fn q8_0_block_dot_env_flags_ignore_outer_whitespace() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", " on ");
        assert!(q8_0_block_dot_enabled());

        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", " f32 ");
        assert!(!q8_0_file_reader_block_dot_enabled());

        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", " dequantized ");
        assert!(!q8_0_file_reader_block_dot_enabled());

        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn q8_0_input_quantization_uses_unrounded_scale_for_quants() {
        let unrounded_scale = 1.0_f32 / 127.0;
        let mut input_values = vec![0.0; Q8_0_BLOCK_VALUES];
        input_values[0] = 1.0;
        input_values[1] = 2.49995 * unrounded_scale;

        let quantized = quantize_q8_0_row(&input_values);
        let block = &quantized.blocks[0];

        assert_eq!(
            block.scale,
            f16_bits_to_f32(f32_to_f16_bits(unrounded_scale))
        );
        assert_eq!(block.quants[0], 127);
        assert_eq!(block.quants[1], 2);
        assert_eq!((input_values[1] / block.scale).round() as i8, 3);
    }

    #[test]
    fn q8_0_file_reader_quantized_input_buffer_reuses_capacity() {
        let first = CpuTensor::from_f32(
            "first",
            vec![2, Q8_0_BLOCK_VALUES],
            (0..2 * Q8_0_BLOCK_VALUES)
                .map(|idx| idx as f32 - 17.0)
                .collect(),
        )
        .unwrap();
        let second = CpuTensor::from_f32(
            "second",
            vec![1, Q8_0_BLOCK_VALUES],
            (0..Q8_0_BLOCK_VALUES).map(|idx| idx as f32).collect(),
        )
        .unwrap();

        let retained_capacity = with_q8_0_file_reader_quantized_inputs(|blocks| {
            *blocks = Vec::new();

            {
                let quantized = quantize_q8_0_rows_into(&first, Q8_0_BLOCK_VALUES, blocks)?;
                assert_eq!(quantized.rows().len(), 2);
                assert_eq!(quantized.row(0)[0].quants[0], -127);
            }
            let retained_capacity = blocks.capacity();

            {
                let quantized = quantize_q8_0_rows_into(&second, Q8_0_BLOCK_VALUES, blocks)?;
                assert_eq!(quantized.rows().len(), 1);
                assert_eq!(quantized.row(0)[0].quants[0], 0);
            }

            assert_eq!(blocks.capacity(), retained_capacity);
            Ok(blocks.capacity())
        })
        .unwrap();

        with_q8_0_file_reader_quantized_inputs(|blocks| {
            assert!(blocks.is_empty());
            assert_eq!(blocks.capacity(), retained_capacity);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn q8_0_file_reader_scratch_retention_is_bounded() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES", "128");

        with_q8_0_file_reader_row_chunk(512, |row_chunk| {
            row_chunk.fill(7);
            Ok(())
        })
        .unwrap();
        let (row_capacity, _, _, _) = q8_0_file_reader_scratch_capacities();
        assert!(
            row_capacity <= 128,
            "row scratch capacity should be capped after an oversized use, got {row_capacity}"
        );

        with_q8_0_file_reader_chunk_scales(256, |scales| {
            scales.fill(3.0);
            Ok(())
        })
        .unwrap();
        let (_, scale_capacity, _, _) = q8_0_file_reader_scratch_capacities();
        assert!(
            scale_capacity * mem::size_of::<f32>() <= 128,
            "scale scratch capacity should be capped after an oversized use, got {scale_capacity} entries"
        );

        with_q8_0_file_reader_output_chunk(256, |output_chunk| {
            output_chunk.fill(5.0);
            Ok(())
        })
        .unwrap();
        let (_, _, _, output_capacity) = q8_0_file_reader_scratch_capacities();
        assert!(
            output_capacity * mem::size_of::<f32>() <= 128,
            "output scratch capacity should be capped after an oversized use, got {output_capacity} entries"
        );

        with_q8_0_file_reader_quantized_inputs(|blocks| {
            blocks.resize(
                32,
                Q8_0Block {
                    scale: 1.0,
                    quants: [0; Q8_0_BLOCK_VALUES],
                },
            );
            Ok(())
        })
        .unwrap();
        let (_, _, quantized_capacity, _) = q8_0_file_reader_scratch_capacities();
        assert!(
            quantized_capacity * mem::size_of::<Q8_0Block>() <= 128,
            "quantized-input scratch capacity should be capped after an oversized use, got {quantized_capacity} entries"
        );

        std::env::remove_var("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES");
    }

    #[test]
    fn q8_0_block_reader_linear_matches_existing_q8_path() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "off");
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let row0 = Q8_0Block {
            scale: 0.5,
            quants: std::array::from_fn(|idx| idx as i8 - 16),
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
        };
        for block in [&row0, &row1] {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
            temp_file.write_all(&bytes).unwrap();
        }
        temp_file.flush().unwrap();

        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

        let mut dequantized_weight = Vec::with_capacity(64);
        for block in [&row0, &row1] {
            dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
        }
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![2, 32],
            dequantized_weight,
            vec![row0.clone(), row1.clone()],
        )
        .unwrap();

        let expected = matmul_rhs_transposed_with_precision(&input, &weight, "expected").unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 2);
        let reader = Q8BlockReader::new(0, 2);
        let actual =
            matmul_rhs_transposed_q8_0_block_reader(&input, &backing, reader, 2, "actual").unwrap();

        assert_eq!(actual.shape.dims, expected.shape.dims);
        assert_slice_close(&actual.data, &expected.data);
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "off");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let rows: Vec<Q8_0Block> = (0..5)
            .map(|row| Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
                quants: std::array::from_fn(|idx| idx as i8 - 12 + row as i8),
            })
            .collect();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
            temp_file.write_all(&bytes).unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..64)
            .map(|idx| idx as f32 * 0.25 - 4.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![2, 32], input_values).unwrap();
        let mut dequantized_weight = Vec::with_capacity(rows.len() * 32);
        for block in &rows {
            dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
        }
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![rows.len(), 32],
            dequantized_weight,
            rows,
        )
        .unwrap();
        let expected = matmul_rhs_transposed_with_precision(&input, &weight, "expected").unwrap();

        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 5);
        let reader = Q8BlockReader::new(0, 5);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let actual = pool
            .install(|| {
                assert!(should_parallelize_linear_output(5));
                matmul_rhs_transposed_q8_0_block_reader(&input, &backing, reader, 5, "actual")
            })
            .unwrap();

        assert_eq!(actual.shape.dims, expected.shape.dims);
        assert_slice_close(&actual.data, &expected.data);
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn q8_0_file_reader_parallelizes_wide_outputs_by_default() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        pool.install(|| {
            assert!(!should_parallelize_q8_0_file_reader_output(1023));
            assert!(should_parallelize_q8_0_file_reader_output(1024));
        });
    }

    #[test]
    fn q8_0_file_reader_parallel_respects_explicit_linear_off() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        pool.install(|| assert!(!should_parallelize_q8_0_file_reader_output(14336)));
    }

    #[test]
    fn q8_0_file_reader_parallel_uses_existing_linear_threshold_env() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "2048");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        pool.install(|| {
            assert!(!should_parallelize_q8_0_file_reader_output(2047));
            assert!(should_parallelize_q8_0_file_reader_output(2048));
        });
    }

    #[test]
    fn q8_0_encoded_row_matches_decoded_scale_helper() {
        let row = Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.375)),
            quants: std::array::from_fn(|idx| idx as i8 - 12),
        };
        let input = QuantizedQ8_0Row {
            blocks: vec![Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.25)),
                quants: std::array::from_fn(|idx| 15 - idx as i8),
            }],
        };
        let mut row_bytes = Vec::with_capacity(Q8BlockReader::BLOCK_SIZE_BYTES);
        row_bytes.extend_from_slice(&f32_to_f16_bits(row.scale).to_le_bytes());
        row_bytes.extend(row.quants.iter().map(|q| *q as u8));
        let mut scales = vec![0.0; 1];
        decode_q8_0_encoded_row_scales(&row_bytes, &mut scales);

        let direct = dot_q8_0_encoded_row(&input.blocks, &row_bytes);
        let decoded = dot_q8_0_encoded_row_with_scales(&input.blocks, &row_bytes, &scales);

        assert!((direct - decoded).abs() < 1e-6);
    }

    #[test]
    fn q8_0_file_backed_accumulate_matches_q8_block_dot_across_chunks() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let rows: Vec<Q8_0Block> = (0..5)
            .map(|row| Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
                quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
            })
            .collect();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
            temp_file.write_all(&bytes).unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..32)
            .map(|idx| idx as f32 * 0.5 - 3.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();
        let mut dequantized_weight = Vec::with_capacity(rows.len() * 32);
        for block in &rows {
            dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
        }
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![rows.len(), 32],
            dequantized_weight,
            rows.clone(),
        )
        .unwrap();
        let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();

        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
        let mut actual = vec![0.0; rows.len()];
        let start = q8_0_file_read_stats();
        accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
            .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_slice_close(&actual, &expected.data);
        assert_eq!(reads.read_calls, 3);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    }

    #[test]
    fn q8_0_file_backed_accumulate_coalesces_exact_two_chunk_tensor_read() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let rows: Vec<Q8_0Block> = (0..4)
            .map(|row| Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
                quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..32)
            .map(|idx| idx as f32 * 0.5 - 3.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![rows.len(), 32],
            dequantized_q8_0_rows(&rows),
            rows.clone(),
        )
        .unwrap();
        let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
        let mut actual = vec![0.0; rows.len()];
        let start = q8_0_file_read_stats();

        accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
            .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_slice_close(&actual, &expected.data);
        assert_eq!(reads.read_calls, 1);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    }

    #[test]
    fn q8_0_file_backed_accumulate_can_use_quantized_input_block_dot() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");

        let rows: Vec<Q8_0Block> = (0..3)
            .map(|row| Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.125 + row as f32 * 0.0625)),
                quants: std::array::from_fn(|idx| idx as i8 - 9 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..32)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.37)
            .collect::<Vec<_>>();
        let quantized_input = quantize_q8_0_row(&input_values);
        let expected = rows
            .iter()
            .map(|row| q8_0_dot_rows(std::slice::from_ref(row), &quantized_input.blocks))
            .collect::<Vec<_>>();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
        let mut actual = vec![0.0; rows.len()];

        accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
            .unwrap();

        assert_slice_close(&actual, &expected);
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn q8_0_file_backed_accumulate_rejects_unaligned_input_width() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 1);
        let input = vec![0.0_f32; Q8_0_BLOCK_VALUES + 1];
        let mut output = vec![0.0_f32; 1];

        let err = accumulate_transposed_linear_row_q8_0_file_reader(&input, &backing, &mut output)
            .unwrap_err()
            .to_string();

        assert!(err.contains("not a multiple of 32"));
    }

    #[test]
    fn q8_0_file_backed_batch_matmul_reuses_chunk_reads_across_input_rows() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let rows: Vec<Q8_0Block> = (0..5)
            .map(|row| Q8_0Block {
                scale: 0.125 + row as f32 * 0.03125,
                quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for row in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..96)
            .map(|idx| idx as f32 * 0.1 - 3.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![3, 32], input_values).unwrap();
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![rows.len(), 32],
            dequantized_q8_0_rows(&rows),
            rows.clone(),
        )
        .unwrap();
        let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
        let start = q8_0_file_read_stats();

        let actual = matmul_rhs_transposed_q8_0_block_reader(
            &input,
            &backing,
            Q8BlockReader::new(0, rows.len()),
            rows.len(),
            "actual",
        )
        .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_slice_close(&actual.data, &expected.data);
        assert_eq!(reads.read_calls, 3);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    }

    #[test]
    fn q8_0_file_backed_batch_matmul_can_use_quantized_input_block_dot() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");

        let rows: Vec<Q8_0Block> = (0..4)
            .map(|row| Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.1875 + row as f32 * 0.03125)),
                quants: std::array::from_fn(|idx| idx as i8 - 11 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..64)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.21)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![2, 32], input_values.clone()).unwrap();
        let mut expected = Vec::new();
        for input_row in input_values.chunks_exact(32) {
            let quantized_input = quantize_q8_0_row(input_row);
            expected.extend(
                rows.iter()
                    .map(|row| q8_0_dot_rows(std::slice::from_ref(row), &quantized_input.blocks)),
            );
        }
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());

        let actual = matmul_rhs_transposed_q8_0_block_reader(
            &input,
            &backing,
            Q8BlockReader::new(0, rows.len()),
            rows.len(),
            "actual",
        )
        .unwrap();

        assert_slice_close(&actual.data, &expected);
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn q8_0_file_backed_batch_matmul_reuses_cached_chunks_across_calls() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "1024");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let rows: Vec<Q8_0Block> = (0..5)
            .map(|row| Q8_0Block {
                scale: 0.125 + row as f32 * 0.03125,
                quants: std::array::from_fn(|idx| idx as i8 - 7 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for row in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..96)
            .map(|idx| idx as f32 * 0.075 - 2.5)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("input", vec![3, 32], input_values).unwrap();
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![rows.len(), 32],
            dequantized_q8_0_rows(&rows),
            rows.clone(),
        )
        .unwrap();
        let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());

        let start = q8_0_file_read_stats();
        let first = matmul_rhs_transposed_q8_0_block_reader(
            &input,
            &backing,
            Q8BlockReader::new(0, rows.len()),
            rows.len(),
            "first",
        )
        .unwrap();
        let after_first = q8_0_file_read_stats();
        let first_reads = after_first.saturating_delta_since(start);

        let second = matmul_rhs_transposed_q8_0_block_reader(
            &input,
            &backing,
            Q8BlockReader::new(0, rows.len()),
            rows.len(),
            "second",
        )
        .unwrap();
        let second_reads = q8_0_file_read_stats().saturating_delta_since(after_first);

        assert_slice_close(&first.data, &expected.data);
        assert_slice_close(&second.data, &expected.data);
        assert_eq!(first_reads.read_calls, 3);
        assert_eq!(
            first_reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        assert_eq!(second_reads.read_calls, 0);
        assert_eq!(second_reads.read_bytes, 0);
        assert_eq!(second_reads.cache_hits, 3);
        assert_eq!(
            second_reads.cache_hit_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    }

    #[test]
    fn q8_0_file_backed_borrowed_batch_matmul_reuses_chunk_reads_across_input_rows() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        std::env::set_var(
            "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
        );

        let rows: Vec<Q8_0Block> = (0..5)
            .map(|row| Q8_0Block {
                scale: 0.125 + row as f32 * 0.03125,
                quants: std::array::from_fn(|idx| idx as i8 - 9 + row as i8),
            })
            .collect();
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for row in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let input_values = (0..96)
            .map(|idx| idx as f32 * 0.05 - 2.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("output_norm_batch", vec![3, 32], input_values).unwrap();
        let expected_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "expected.weight",
            vec![rows.len(), 32],
            dequantized_q8_0_rows(&rows),
            rows.clone(),
        )
        .unwrap();
        let expected =
            matmul_rhs_transposed_q8_0_block_dot(&input, &expected_weight, "expected").unwrap();
        let output_weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            crate::tensor::TensorShape {
                dims: vec![32, rows.len()],
            },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len()),
        );
        let start = q8_0_file_read_stats();

        let actual = output_projection_runtime(&input, &output_weight, "actual", false).unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(actual.shape.dims, vec![3, 5]);
        assert_slice_close(&actual.data, &expected.data);
        assert_eq!(reads.read_calls, 3);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
        );
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    }

    #[test]
    fn q8_0_file_backing_cache_reuses_exact_chunk_reads() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "1024");

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        temp_file.write_all(&[1_u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        temp_file.flush().unwrap();
        let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 1);
        let start = q8_0_file_read_stats();
        let mut first = [0_u8; 4];
        let mut second = [0_u8; 4];

        backing.read_exact_at_cached(&mut first, 2).unwrap();
        let after_first = q8_0_file_read_stats().saturating_delta_since(start);
        backing.read_exact_at_cached(&mut second, 2).unwrap();
        let after_second = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(first, [3, 4, 5, 6]);
        assert_eq!(second, first);
        assert_eq!(after_first.read_calls, 1);
        assert_eq!(after_first.read_bytes, 4);
        assert_eq!(after_first.cache_hits, 0);
        assert_eq!(after_first.cache_entries, 1);
        assert_eq!(after_first.cache_bytes, 4);
        assert_eq!(after_first.cache_capacity_bytes, 1024);
        assert_eq!(after_second.read_calls, after_first.read_calls);
        assert_eq!(after_second.read_bytes, after_first.read_bytes);
        assert_eq!(after_second.cache_hits, 1);
        assert_eq!(after_second.cache_entries, 1);
        assert_eq!(after_second.cache_bytes, 4);
        assert_eq!(after_second.cache_capacity_bytes, 1024);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_0_block_dot_uses_raw_weight_blocks_and_quantized_input_when_opted_in() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

        let row0 = Q8_0Block {
            scale: 0.5,
            quants: std::array::from_fn(|idx| idx as i8 - 16),
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
        };
        let mut dequantized_weight = Vec::with_capacity(64);
        for block in [&row0, &row1] {
            dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
        }
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![2, 32],
            dequantized_weight,
            vec![row0.clone(), row1.clone()],
        )
        .unwrap();

        let actual = matmul_rhs_transposed_with_precision(&input, &weight, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![1, 2]);
        assert_slice_close(
            &actual.data,
            &[
                expected_q8_0_block_dot(&input_values, &row0),
                expected_q8_0_block_dot(&input_values, &row1),
            ],
        );
    }

    #[test]
    fn rectangular_shape_reinterpretation_preserves_q8_0_blocks_for_transposed_dot() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

        let block = Q8_0Block {
            scale: 1.0,
            quants: [0; 32],
        };
        let weight = CpuTensor::from_f32_with_q8_0_blocks(
            "weight",
            vec![32, 64],
            vec![0.0; 2048],
            vec![block; 64],
        )
        .unwrap();

        let reinterpreted = weight_with_swapped_matrix_shape(&weight);

        assert_eq!(reinterpreted.shape.dims, vec![64, 32]);
        assert_eq!(reinterpreted.source_type, Some(GgufTensorType::Q8_0));
        assert!(reinterpreted.q8_0_blocks.is_some());
        assert!(should_use_q8_0_block_dot(&reinterpreted, 32));
    }

    #[test]
    fn q8_0_block_dot_reads_descriptor_shaped_blocks_as_transposed_rows_when_opted_in() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

        let row0 = Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| (idx % 5) as i8 - 2),
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 3 } else { -1 }),
        };
        let descriptor_shaped_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "descriptor.weight",
            vec![32, 2],
            dequantized_q8_0_rows(&[row0.clone(), row1.clone()]),
            vec![row0.clone(), row1.clone()],
        )
        .unwrap();

        let actual = linear_with_diagnostic_layouts(
            &input,
            &descriptor_shaped_weight,
            "out",
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Auto,
        )
        .unwrap();

        assert_eq!(actual.shape.dims, vec![1, 2]);
        assert_slice_close(
            &actual.data,
            &[
                expected_q8_0_block_dot(&input_values, &row0),
                expected_q8_0_block_dot(&input_values, &row1),
            ],
        );
    }

    #[test]
    fn output_projection_q8_0_descriptor_shape_uses_storage_token_rows() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values.clone()).unwrap();

        let token_0 = Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| (idx % 7) as i8 - 3),
        };
        let token_1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 5 } else { -4 }),
        };
        let output_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "output.weight",
            vec![32, 2],
            dequantized_q8_0_rows(&[token_0.clone(), token_1.clone()]),
            vec![token_0.clone(), token_1.clone()],
        )
        .unwrap();

        let runtime =
            output_projection_runtime(&input, &output_weight, "runtime_logits", false).unwrap();
        let token_major = output_projection_with_layout(
            &input,
            &output_weight,
            "token_major_logits",
            OutputProjectionLayout::TokenMajor,
        )
        .unwrap();
        let descriptor = output_projection_with_layout(
            &input,
            &output_weight,
            "descriptor_logits",
            OutputProjectionLayout::Descriptor,
        )
        .unwrap();
        let expected = [
            expected_q8_0_block_dot(&input_values, &token_0),
            expected_q8_0_block_dot(&input_values, &token_1),
        ];

        assert_eq!(runtime.shape.dims, vec![1, 2]);
        assert_eq!(token_major.shape.dims, vec![1, 2]);
        assert_slice_close(&runtime.data, &expected);
        assert_slice_close(&token_major.data, &expected);
        assert!(
            descriptor
                .data
                .iter()
                .zip(expected.iter())
                .any(|(actual, expected)| (actual - expected).abs() > 1e-3),
            "descriptor-column interpretation should not alias token-major Q8_0 storage rows"
        );
    }

    #[test]
    fn gated_ffn_activation_uses_q8_0_descriptor_blocks_for_gate_and_up_when_opted_in() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

        let mut input_values = Vec::with_capacity(32);
        input_values.push(127.0);
        input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
        let input = CpuTensor::from_f32("ffn_norm", vec![1, 32], input_values.clone()).unwrap();

        let gate0 = Q8_0Block {
            scale: 0.0625,
            quants: std::array::from_fn(|idx| (idx % 7) as i8 - 3),
        };
        let gate1 = Q8_0Block {
            scale: 0.03125,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 4 } else { -2 }),
        };
        let up0 = Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 1 } else { -3 }),
        };
        let up1 = Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| (idx % 5) as i8 - 2),
        };
        let gate_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "blk.0.ffn_gate.weight",
            vec![32, 2],
            dequantized_q8_0_rows(&[gate0.clone(), gate1.clone()]),
            vec![gate0.clone(), gate1.clone()],
        )
        .unwrap();
        let up_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "blk.0.ffn_up.weight",
            vec![32, 2],
            dequantized_q8_0_rows(&[up0.clone(), up1.clone()]),
            vec![up0.clone(), up1.clone()],
        )
        .unwrap();

        let actual = gated_ffn_activation(&input, &gate_weight, &up_weight, "ffn", false)
            .unwrap()
            .tensor;

        let expected_gate = [
            expected_q8_0_block_dot(&input_values, &gate0),
            expected_q8_0_block_dot(&input_values, &gate1),
        ];
        let expected_up = [
            expected_q8_0_block_dot(&input_values, &up0),
            expected_q8_0_block_dot(&input_values, &up1),
        ];
        let expected = [
            silu(expected_gate[0]) * expected_up[0],
            silu(expected_gate[1]) * expected_up[1],
        ];

        assert_eq!(actual.shape.dims, vec![1, 2]);
        assert_slice_close(&actual.data, &expected);
    }

    #[test]
    fn q8_0_horizontal_sum_matches_linear_int_sum() {
        let weight = std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(111));
        let input = std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(97));
        let linear_sum: i32 = weight
            .iter()
            .zip(input.iter())
            .map(|(w, x)| i32::from(*w) * i32::from(*x))
            .sum();

        assert_eq!(
            q8_0_block_int_dot_horizontal_sum(&weight, &input),
            linear_sum
        );
    }

    #[test]
    fn q8_0_encoded_horizontal_sum_matches_linear_int_sum() {
        let weight: [i8; Q8_0_BLOCK_VALUES] =
            std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(111));
        let input: [i8; Q8_0_BLOCK_VALUES] =
            std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(97));
        let encoded_weight = weight.map(|quant| quant as u8);
        let linear_sum: i32 = weight
            .iter()
            .zip(input.iter())
            .map(|(w, x)| i32::from(*w) * i32::from(*x))
            .sum();

        assert_eq!(
            q8_0_block_int_dot_horizontal_sum_encoded(&encoded_weight, &input),
            linear_sum
        );
    }

    fn expected_q8_0_block_dot(input_values: &[f32], weight: &Q8_0Block) -> f32 {
        // The input vector deliberately contains a 127.0 max-absolute value, so Camelid's
        // Q8_0 activation quantizer uses an exactly representable scale of 1.0 and preserves
        // these integer samples as their Q8 quants. That keeps the expected dot independent
        // from the production quantization helper while still exercising the block-dot path.
        input_values
            .iter()
            .zip(weight.quants.iter())
            .map(|(input, weight_quant)| input * f32::from(*weight_quant) * weight.scale)
            .sum()
    }

    fn dequantized_q8_0_rows(rows: &[Q8_0Block]) -> Vec<f32> {
        rows.iter()
            .flat_map(|block| block.quants.iter().map(|q| block.scale * f32::from(*q)))
            .collect()
    }

    #[test]
    fn applies_rope_to_each_attention_head() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ROPE_PAIRING");
        std::env::remove_var("CAMELID_ROPE_DIRECTION");
        std::env::remove_var("CAMELID_ROPE_POSITION_MODE");

        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let rotated = apply_rope(&tensor, 1, 2, &config, None, "query_rope").unwrap();

        let (sin, cos) = 1.0_f32.sin_cos();
        assert_eq!(rotated.shape.dims, vec![1, 4]);
        assert_close(rotated.data[0], cos);
        assert_close(rotated.data[1], sin);
        assert_close(rotated.data[2], -sin);
        assert_close(rotated.data[3], cos);
    }

    #[test]
    fn apply_rope_uses_configured_frequency_base() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 8192,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(4),
            rope_freq_base: Some(500_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-5,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let rotated = apply_rope(&tensor, 1, 1, &config, None, "query_rope").unwrap();
        let diagnostic =
            rope_diagnostics(&tensor, &rotated, 1, 1, &config, None, "attention_q").unwrap();

        let theta_500k = 500_000.0_f32.powf(-0.5);
        let (sin_500k, cos_500k) = theta_500k.sin_cos();
        let theta_10k = 10_000.0_f32.powf(-0.5);
        let (sin_10k, _) = theta_10k.sin_cos();

        assert_eq!(rotated.shape.dims, vec![1, 4]);
        assert_close(rotated.data[2], cos_500k);
        assert_close(rotated.data[3], sin_500k);
        assert!(
            (rotated.data[3] - sin_10k).abs() > 1e-3,
            "RoPE rotation unexpectedly matched the TinyLlama 10000 fallback instead of GGUF freq_base=500000"
        );
        assert_eq!(diagnostic.freq_base, 500_000.0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn apply_rope_uses_llama3_frequency_scaling_metadata() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 32,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(4),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: Some("llama3".to_string()),
            rope_scaling_factor: Some(8.0),
            rope_scaling_original_context_length: Some(16),
            rope_scaling_low_freq_factor: Some(1.0),
            rope_scaling_high_freq_factor: Some(4.0),
            rms_norm_epsilon: 1e-5,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let rotated = apply_rope(&tensor, 8, 1, &config, None, "query_rope").unwrap();
        let diagnostic =
            rope_diagnostics(&tensor, &rotated, 8, 1, &config, None, "attention_q").unwrap();

        let base_theta = 10_000.0_f32.powf(-0.5);
        let scaled_theta = base_theta / 8.0;
        let (scaled_sin, scaled_cos) = (8.0 * scaled_theta).sin_cos();
        let (unscaled_sin, _) = (8.0 * base_theta).sin_cos();

        assert_eq!(rotated.shape.dims, vec![1, 4]);
        assert_close(rotated.data[2], scaled_cos);
        assert_close(rotated.data[3], scaled_sin);
        assert!(
            (rotated.data[3] - unscaled_sin).abs() > 1e-2,
            "RoPE rotation unexpectedly ignored llama3 scaling metadata"
        );
        assert_eq!(diagnostic.scaling_type, "llama3");
        assert_eq!(diagnostic.scaling_factor, 8.0);
        assert_eq!(diagnostic.scaling_original_context_length, Some(16));
        assert_eq!(diagnostic.scaling_low_freq_factor, Some(1.0));
        assert_eq!(diagnostic.scaling_high_freq_factor, Some(4.0));
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn apply_rope_uses_gguf_rope_frequency_factors() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 32,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(4),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-5,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let rope_freqs = CpuTensor::from_f32("rope_freqs.weight", vec![2], vec![1.0, 4.0]).unwrap();

        let rotated = apply_rope(&tensor, 8, 1, &config, Some(&rope_freqs), "query_rope").unwrap();
        let diagnostic = rope_diagnostics(
            &tensor,
            &rotated,
            8,
            1,
            &config,
            Some(&rope_freqs),
            "attention_q",
        )
        .unwrap();

        let derived_theta = 10_000.0_f32.powf(-0.5);
        let factor_theta = derived_theta / 4.0;
        let (factor_sin, factor_cos) = (8.0_f32 * factor_theta).sin_cos();
        let (derived_sin, _) = (8.0_f32 * derived_theta).sin_cos();

        assert_close(rotated.data[2], factor_cos);
        assert_close(rotated.data[3], factor_sin);
        assert!(
            (rotated.data[3] - derived_sin).abs() > 0.05,
            "RoPE rotation unexpectedly ignored rope_freqs.weight factors"
        );
        assert_eq!(diagnostic.frequency_source, "rope_freqs.weight");
        assert_eq!(diagnostic.rope_freqs_count, Some(2));
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn rope_diagnostics_reconstruct_reported_rotation() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ROPE_PAIRING");
        std::env::remove_var("CAMELID_ROPE_DIRECTION");
        std::env::remove_var("CAMELID_ROPE_POSITION_MODE");

        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let reported = apply_rope(&tensor, 1, 2, &config, None, "query_rope").unwrap();

        let diagnostic =
            rope_diagnostics(&tensor, &reported, 1, 2, &config, None, "attention_q").unwrap();

        assert_eq!(diagnostic.role, "attention_q");
        assert_eq!(diagnostic.pairing, "adjacent_even_odd");
        assert_eq!(diagnostic.direction, "forward");
        assert_eq!(diagnostic.position_mode, "zero_based");
        assert_eq!(diagnostic.position, 1);
        assert_eq!(diagnostic.effective_position, 1);
        assert_eq!(diagnostic.head_count, 2);
        assert_eq!(diagnostic.head_dim, 2);
        assert_eq!(diagnostic.rope_dim, 2);
        assert_eq!(diagnostic.input_first_values, tensor.data);
        assert_eq!(diagnostic.reported_first_values, reported.data);
        assert_eq!(diagnostic.reconstructed_first_values, reported.data);
        assert_eq!(diagnostic.reported_max_abs_index, 1);
        assert_close(diagnostic.reported_max_abs, reported.data[1]);
        assert_eq!(diagnostic.reported_max_abs_window_start, 0);
        assert_eq!(diagnostic.reported_max_abs_window, reported.data);
        assert_eq!(
            diagnostic.reconstructed_reported_max_abs_window,
            reported.data
        );
        assert_eq!(diagnostic.max_abs_delta_index, 0);
        assert!(diagnostic.max_abs_delta < 1e-7);
    }

    #[test]
    fn zero_delta_selector_accepts_all_none_and_layer_lists() {
        assert!(diagnostic_zero_delta_value("TEST_ZERO", "all", 7).unwrap());
        assert!(diagnostic_zero_delta_value("TEST_ZERO", "true", 7).unwrap());
        assert!(!diagnostic_zero_delta_value("TEST_ZERO", "none", 7).unwrap());
        assert!(!diagnostic_zero_delta_value("TEST_ZERO", "", 7).unwrap());
        assert!(diagnostic_zero_delta_value("TEST_ZERO", "1, 7, 9", 7).unwrap());
        assert!(!diagnostic_zero_delta_value("TEST_ZERO", "1, 2, 9", 7).unwrap());
        assert!(diagnostic_zero_delta_value("TEST_ZERO", "oops", 7).is_err());
    }

    #[test]
    fn split_half_rope_pairing_is_available_for_diagnostics() {
        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(4),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let head_dim = 4;
        let rope_dim = 4;
        let freq_base = config.rope_freq_base.unwrap();

        let adjacent = apply_rope_with_pairing(
            &tensor,
            RopeParams {
                position: 1,
                head_count: 1,
                head_dim,
                rope_dim,
                freq_base,
                pairing: RopePairing::AdjacentEvenOdd,
                direction: RopeDirection::Forward,
                position_mode: RopePositionMode::ZeroBased,
                scaling: no_rope_scaling(),
                rope_freqs: None,
            },
            "adjacent",
        )
        .unwrap();
        let split = apply_rope_with_pairing(
            &tensor,
            RopeParams {
                position: 1,
                head_count: 1,
                head_dim,
                rope_dim,
                freq_base,
                pairing: RopePairing::SplitHalf,
                direction: RopeDirection::Forward,
                position_mode: RopePositionMode::ZeroBased,
                scaling: no_rope_scaling(),
                rope_freqs: None,
            },
            "split",
        )
        .unwrap();

        let (sin, cos) = 1.0_f32.sin_cos();
        assert_eq!(adjacent.shape.dims, vec![1, 4]);
        assert_eq!(split.shape.dims, vec![1, 4]);
        assert_close(adjacent.data[0], cos);
        assert_close(adjacent.data[1], sin);
        assert_close(adjacent.data[2], 0.0);
        assert_close(adjacent.data[3], 0.0);
        assert_close(split.data[0], cos);
        assert_close(split.data[1], 0.0);
        assert_close(split.data[2], sin);
        assert_close(split.data[3], 0.0);
    }

    #[test]
    fn inverse_rope_direction_is_available_for_diagnostics() {
        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
        let head_dim = 2;
        let rope_dim = 2;
        let freq_base = config.rope_freq_base.unwrap();

        let forward = apply_rope_with_pairing(
            &tensor,
            RopeParams {
                position: 1,
                head_count: 1,
                head_dim,
                rope_dim,
                freq_base,
                pairing: RopePairing::AdjacentEvenOdd,
                direction: RopeDirection::Forward,
                position_mode: RopePositionMode::ZeroBased,
                scaling: no_rope_scaling(),
                rope_freqs: None,
            },
            "forward",
        )
        .unwrap();
        let inverse = apply_rope_with_pairing(
            &tensor,
            RopeParams {
                position: 1,
                head_count: 1,
                head_dim,
                rope_dim,
                freq_base,
                pairing: RopePairing::AdjacentEvenOdd,
                direction: RopeDirection::Inverse,
                position_mode: RopePositionMode::ZeroBased,
                scaling: no_rope_scaling(),
                rope_freqs: None,
            },
            "inverse",
        )
        .unwrap();

        let (sin, cos) = 1.0_f32.sin_cos();
        assert_eq!(forward.shape.dims, vec![1, 2]);
        assert_eq!(inverse.shape.dims, vec![1, 2]);
        assert_close(forward.data[0], cos);
        assert_close(forward.data[1], sin);
        assert_close(inverse.data[0], cos);
        assert_close(inverse.data[1], -sin);
    }

    #[test]
    fn one_based_rope_position_mode_is_available_for_diagnostics() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_ROPE_POSITION_MODE", "one_based");

        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: None,
            file_type: None,
            moe: None,
        };
        let tensor = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();

        let rotated = apply_rope(&tensor, 0, 1, &config, None, "query_rope").unwrap();
        let diagnostic =
            rope_diagnostics(&tensor, &rotated, 0, 1, &config, None, "attention_q").unwrap();

        let (sin, cos) = 1.0_f32.sin_cos();
        assert_close(rotated.data[0], cos);
        assert_close(rotated.data[1], sin);
        assert_eq!(diagnostic.position_mode, "one_based");
        assert_eq!(diagnostic.position, 0);
        assert_eq!(diagnostic.effective_position, 1);
        assert!(diagnostic.max_abs_delta < 1e-7);

        std::env::set_var("CAMELID_ROPE_POSITION_MODE", "diagonal");
        assert!(diagnostic_rope_position_mode().is_err());
        std::env::remove_var("CAMELID_ROPE_POSITION_MODE");
    }

    #[test]
    fn tied_output_projection_uses_token_major_embedding_layout() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let embedding = CpuTensor::from_f32(
            "token_embd.weight",
            vec![3, 2],
            vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                2.0, 3.0, // token 2
            ],
        )
        .unwrap();
        let hidden = CpuTensor::from_f32("hidden", vec![1, 2], vec![2.0, 3.0]).unwrap();

        let logits = linear(&hidden, &embedding, "logits").unwrap();

        assert_eq!(logits.shape.dims, vec![1, 3]);
        assert_close(logits.data[0], 2.0);
        assert_close(logits.data[1], 3.0);
        assert_close(logits.data[2], 13.0);
    }

    #[test]
    fn output_projection_diagnostics_reconstruct_tied_output_rows() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

        let output_norm =
            CpuTensor::from_f32("output_norm", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
        let tied_output = CpuTensor::from_f32(
            "token_embd.weight",
            vec![4, 3],
            vec![
                0.5, 1.0, -2.0, // token 0
                -1.0, 0.25, 0.75, // token 1
                2.0, -0.5, 1.5, // token 2
                0.0, 3.0, -1.0, // token 3
            ],
        )
        .unwrap();
        let logits = output_projection_with_layout(
            &output_norm,
            &tied_output,
            "logits",
            OutputProjectionLayout::Descriptor,
        )
        .unwrap();

        let diagnostics = output_projection_diagnostics(
            &output_norm,
            &tied_output,
            &logits,
            &[2],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(logits.shape.dims, vec![1, 4]);
        assert_close(logits.data[2], 5.25);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].token_id, 2);
        assert_eq!(diagnostics[0].layout, "output_input");
        assert_close(diagnostics[0].reported_logit, 5.25);
        assert_close(diagnostics[0].reconstructed_logit, 5.25);
        assert_close(diagnostics[0].absolute_delta, 0.0);
        assert_eq!(diagnostics[0].output_row_first_values, vec![2.0, -0.5, 1.5]);
        assert_eq!(
            diagnostics[0].component_products_first_values,
            vec![4.0, 0.5, 0.75]
        );
    }

    #[test]
    fn token_major_output_projection_diagnostic_reinterprets_descriptor_shape() {
        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let descriptor_weight = CpuTensor::from_f32(
            "output.weight",
            vec![2, 3],
            vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                2.0, 3.0, // token 2
            ],
        )
        .unwrap();

        let logits = output_projection_with_layout(
            &input,
            &descriptor_weight,
            "logits",
            OutputProjectionLayout::TokenMajor,
        )
        .unwrap();
        assert_eq!(logits.shape.dims, vec![1, 3]);
        assert_close(logits.data[0], 2.0);
        assert_close(logits.data[1], 3.0);
        assert_close(logits.data[2], 13.0);
    }

    #[test]
    fn output_projection_defaults_to_token_major_runtime_layout() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        assert_eq!(
            diagnostic_output_projection_layout().unwrap(),
            OutputProjectionLayout::TokenMajor
        );

        let input = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let token_major_weight = CpuTensor::from_f32(
            "output.weight",
            vec![2, 3],
            vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                2.0, 3.0, // token 2
            ],
        )
        .unwrap();

        let logits =
            output_projection_runtime(&input, &token_major_weight, "logits", false).unwrap();
        assert_eq!(logits.shape.dims, vec![1, 3]);
        assert_close(logits.data[0], 2.0);
        assert_close(logits.data[1], 3.0);
        assert_close(logits.data[2], 13.0);
    }

    #[test]
    fn output_projection_diagnostics_support_q8_0_file_backed_token_major_rows() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

        let input_values = (0..32)
            .map(|idx| idx as f32 * 0.25 - 2.0)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values.clone()).unwrap();
        let row0 = Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| idx as i8 - 8),
        };
        let row1 = Q8_0Block {
            scale: 0.0625,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 6 } else { -5 }),
        };
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for block in [&row0, &row1] {
            use std::io::Write;
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
            temp_file.write_all(&bytes).unwrap();
        }
        temp_file.flush().unwrap();

        let output_weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            crate::tensor::TensorShape { dims: vec![32, 2] },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 2),
        );

        let logits = output_projection_runtime(&input, &output_weight, "logits", false).unwrap();
        let read_start = q8_0_file_read_stats();
        let diagnostics = output_projection_diagnostics(
            &input,
            &output_weight,
            &logits,
            &[0, 1],
            None,
            None,
            None,
        )
        .unwrap();
        let reads = q8_0_file_read_stats().saturating_delta_since(read_start);

        assert_eq!(diagnostics.len(), 2);
        assert_close(diagnostics[0].reconstructed_logit, logits.data[0]);
        assert_close(diagnostics[1].reconstructed_logit, logits.data[1]);
        assert_close(
            diagnostics[0].q8_direct_reconstructed_logit.unwrap(),
            logits.data[0],
        );
        assert_close(
            diagnostics[1].q8_direct_reconstructed_logit.unwrap(),
            logits.data[1],
        );
        assert_eq!(diagnostics[0].q8_direct_absolute_delta, Some(0.0));
        assert_eq!(diagnostics[1].q8_direct_absolute_delta, Some(0.0));
        assert!(diagnostics[0]
            .q8_direct_decoded_component_delta
            .is_some_and(|delta| delta.is_finite()));
        assert_eq!(reads.read_calls, 2);
        assert_eq!(
            reads.read_bytes,
            (Q8BlockReader::BLOCK_SIZE_BYTES * 2) as u64
        );
    }

    #[test]
    fn output_projection_diagnostics_reject_q8_0_file_backed_unaligned_rows_before_read() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

        let output_norm = CpuTensor::from_f32("output_norm", vec![1, 33], vec![0.0; 33]).unwrap();
        let logits = CpuTensor::from_f32("logits", vec![1, 1], vec![0.0]).unwrap();
        let output_weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            crate::tensor::TensorShape { dims: vec![33, 1] },
            Q8_0FileBacking::new("unused-q8-output.gguf".into(), 0, 1),
        );

        let start = q8_0_file_read_stats();
        let err = output_projection_diagnostics(
            &output_norm,
            &output_weight,
            &logits,
            &[0],
            None,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert!(err.contains("hidden width 33 is not block aligned"));
        assert_eq!(reads.read_calls, 0);
        assert_eq!(reads.read_bytes, 0);
    }

    #[test]
    fn output_projection_diagnostics_reject_q8_0_file_backing_block_mismatch_before_read() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

        let output_norm = CpuTensor::from_f32("output_norm", vec![1, 32], vec![0.0; 32]).unwrap();
        let logits = CpuTensor::from_f32("logits", vec![1, 2], vec![0.0, 0.0]).unwrap();
        let output_weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            crate::tensor::TensorShape { dims: vec![32, 2] },
            Q8_0FileBacking::new("unused-q8-output.gguf".into(), 0, 1),
        );

        let start = q8_0_file_read_stats();
        let err = output_projection_diagnostics(
            &output_norm,
            &output_weight,
            &logits,
            &[1],
            None,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        let reads = q8_0_file_read_stats().saturating_delta_since(start);

        assert!(err.contains("expected 2 blocks"));
        assert!(err.contains("got 1"));
        assert_eq!(reads.read_calls, 0);
        assert_eq!(reads.read_bytes, 0);
    }

    #[test]
    fn output_projection_diagnostics_match_q8_0_file_backed_block_dot_probe() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
        std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

        let input_values = (0..32)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.17)
            .collect::<Vec<_>>();
        let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values).unwrap();
        let rows = [
            Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.15625)),
                quants: std::array::from_fn(|idx| idx as i8 - 10),
            },
            Q8_0Block {
                scale: f16_bits_to_f32(f32_to_f16_bits(0.09375)),
                quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 7 } else { -4 }),
            },
        ];
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        for block in &rows {
            temp_file
                .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
                .unwrap();
            temp_file
                .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
                .unwrap();
        }
        temp_file.flush().unwrap();

        let output_weight = CpuTensor::q8_0_file_backed_linear(
            "output.weight",
            crate::tensor::TensorShape { dims: vec![32, 2] },
            Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len()),
        );

        let logits = output_projection_runtime(&input, &output_weight, "logits", false).unwrap();
        let diagnostics = output_projection_diagnostics(
            &input,
            &output_weight,
            &logits,
            &[0, 1],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(diagnostics.len(), 2);
        assert_close(
            diagnostics[0].q8_direct_reconstructed_logit.unwrap(),
            logits.data[0],
        );
        assert_close(
            diagnostics[1].q8_direct_reconstructed_logit.unwrap(),
            logits.data[1],
        );
        assert_eq!(diagnostics[0].q8_direct_absolute_delta, Some(0.0));
        assert_eq!(diagnostics[1].q8_direct_absolute_delta, Some(0.0));
        assert!(diagnostics[0]
            .q8_direct_decoded_component_delta
            .is_some_and(|delta| delta.is_finite()));
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
    }

    #[test]
    fn output_projection_runtime_ignores_diagnostic_layout_env_without_dense_collection() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

        let input = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let token_major_weight = CpuTensor::from_f32(
            "output.weight",
            vec![2, 3],
            vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                2.0, 3.0, // token 2
            ],
        )
        .unwrap();

        let runtime_logits =
            output_projection_runtime(&input, &token_major_weight, "runtime_logits", false)
                .unwrap();
        let diagnostic_logits =
            output_projection_runtime(&input, &token_major_weight, "diagnostic_logits", true)
                .unwrap();

        assert_eq!(runtime_logits.shape.dims, vec![1, 3]);
        assert_close(runtime_logits.data[0], 2.0);
        assert_close(runtime_logits.data[1], 3.0);
        assert_close(runtime_logits.data[2], 13.0);
        assert_eq!(diagnostic_logits.shape.dims, vec![1, 3]);
        assert_close(diagnostic_logits.data[0], 5.0);
        assert_close(diagnostic_logits.data[1], 6.0);
        assert_close(diagnostic_logits.data[2], 9.0);
    }

    #[test]
    fn output_projection_diagnostics_reconstruct_selected_logits() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

        let output_norm = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let output_weight = CpuTensor::from_f32(
            "output.weight",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // hidden dim 0 to tokens 0..2
                4.0, 5.0, 6.0, // hidden dim 1 to tokens 0..2
            ],
        )
        .unwrap();
        let logits = output_projection_with_layout(
            &output_norm,
            &output_weight,
            "logits",
            OutputProjectionLayout::Descriptor,
        )
        .unwrap();

        let final_hidden = CpuTensor::from_f32("final_hidden", vec![1, 2], vec![4.0, 6.0]).unwrap();
        let output_norm_weight =
            CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0, 1.0]).unwrap();
        let diagnostics = output_projection_diagnostics(
            &output_norm,
            &output_weight,
            &logits,
            &[2],
            Some(&final_hidden),
            Some(&output_norm_weight),
            Some(0.5),
        )
        .unwrap();

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].token_id, 2);
        assert_eq!(diagnostics[0].layout, "descriptor");
        assert_close(diagnostics[0].reported_logit, 24.0);
        assert_close(diagnostics[0].reconstructed_logit, 24.0);
        assert_close(diagnostics[0].absolute_delta, 0.0);
        assert_eq!(diagnostics[0].output_row_first_values, vec![3.0, 6.0]);
        assert_eq!(
            diagnostics[0].component_products_first_values,
            vec![6.0, 18.0]
        );
        assert_eq!(diagnostics[0].component_products_max_abs_window_start, 0);
        assert_eq!(
            diagnostics[0].component_products_max_abs_window,
            vec![6.0, 18.0]
        );
        assert_eq!(diagnostics[0].max_abs_component_index, 1);
        assert_close(diagnostics[0].max_abs_component, 18.0);
        assert_close(diagnostics[0].positive_component_sum, 24.0);
        assert_close(diagnostics[0].negative_component_sum, 0.0);
        assert_eq!(diagnostics[0].top_positive_components.len(), 2);
        assert_eq!(diagnostics[0].top_positive_components[0].index, 1);
        assert_close(
            diagnostics[0].top_positive_components[0].output_norm_value,
            3.0,
        );
        assert_close(
            diagnostics[0].top_positive_components[0].output_row_value,
            6.0,
        );
        assert_close(diagnostics[0].top_positive_components[0].component, 18.0);
        assert_eq!(
            diagnostics[0].top_positive_components[0].final_hidden_value,
            Some(6.0)
        );
        assert_eq!(
            diagnostics[0].top_positive_components[0].output_norm_weight_value,
            Some(1.0)
        );
        assert_eq!(
            diagnostics[0].top_positive_components[0].output_norm_scale,
            Some(0.5)
        );
        assert_eq!(
            diagnostics[0].top_positive_components[0].reconstructed_output_norm_value,
            Some(3.0)
        );
        assert_eq!(
            diagnostics[0].top_positive_components[0].output_norm_reconstruction_delta,
            Some(0.0)
        );
        assert_eq!(diagnostics[0].top_positive_components[1].index, 0);
        assert_close(diagnostics[0].top_positive_components[1].component, 6.0);
        assert!(diagnostics[0].top_negative_components.is_empty());
    }

    #[test]
    fn output_projection_diagnostics_report_signed_component_extremes() {
        let output_norm =
            CpuTensor::from_f32("output_norm", vec![1, 4], vec![2.0, -3.0, 4.0, -5.0]).unwrap();
        let output_weight =
            CpuTensor::from_f32("output.weight", vec![4, 1], vec![1.5, 2.0, -0.5, -4.0]).unwrap();
        let logits = output_projection_with_layout(
            &output_norm,
            &output_weight,
            "logits",
            OutputProjectionLayout::Descriptor,
        )
        .unwrap();

        let diagnostics = output_projection_diagnostics(
            &output_norm,
            &output_weight,
            &logits,
            &[0],
            None,
            None,
            None,
        )
        .unwrap();

        assert_close(diagnostics[0].reported_logit, 15.0);
        assert_close(diagnostics[0].positive_component_sum, 23.0);
        assert_close(diagnostics[0].negative_component_sum, -8.0);
        assert_eq!(diagnostics[0].top_positive_components.len(), 2);
        assert_eq!(diagnostics[0].top_positive_components[0].index, 3);
        assert_close(diagnostics[0].top_positive_components[0].component, 20.0);
        assert_eq!(diagnostics[0].top_positive_components[1].index, 0);
        assert_close(diagnostics[0].top_positive_components[1].component, 3.0);
        assert_eq!(diagnostics[0].top_negative_components.len(), 2);
        assert_eq!(diagnostics[0].top_negative_components[0].index, 1);
        assert_close(diagnostics[0].top_negative_components[0].component, -6.0);
        assert_eq!(diagnostics[0].top_negative_components[1].index, 2);
        assert_close(diagnostics[0].top_negative_components[1].component, -2.0);
    }

    #[test]
    fn square_linear_transposed_diagnostic_reinterprets_ambiguous_square_weight() {
        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let square_weight = CpuTensor::from_f32(
            "attention_q.weight",
            vec![2, 2],
            vec![
                1.0, 2.0, // descriptor row for input dim 0
                3.0, 4.0, // descriptor row for input dim 1
            ],
        )
        .unwrap();

        let descriptor = linear_with_diagnostic_layouts(
            &input,
            &square_weight,
            "descriptor",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Auto,
        )
        .unwrap();
        let transposed = linear_with_diagnostic_layouts(
            &input,
            &square_weight,
            "transposed",
            SquareLinearLayout::Transposed,
            RectangularLinearLayout::Auto,
        )
        .unwrap();

        assert_eq!(descriptor.shape.dims, vec![1, 2]);
        assert_eq!(transposed.shape.dims, vec![1, 2]);
        assert_close(descriptor.data[0], 11.0);
        assert_close(descriptor.data[1], 16.0);
        assert_close(transposed.data[0], 8.0);
        assert_close(transposed.data[1], 18.0);
    }

    #[test]
    fn rectangular_linear_role_override_reinterprets_only_named_projection() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let descriptor_weight = CpuTensor::from_f32(
            "attention_k.weight",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // descriptor row for input dim 0
                4.0, 5.0, 6.0, // descriptor row for input dim 1
            ],
        )
        .unwrap();

        std::env::set_var(
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K",
            "transposed",
        );
        let overridden =
            linear_for_role(&input, &descriptor_weight, "overridden", "attention_k").unwrap();
        let unaffected =
            linear_for_role(&input, &descriptor_weight, "unaffected", "attention_v").unwrap();
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K");
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");

        assert_eq!(overridden.shape.dims, vec![1, 3]);
        assert_eq!(unaffected.shape.dims, vec![1, 3]);
        assert_close(overridden.data[0], 8.0);
        assert_close(overridden.data[1], 18.0);
        assert_close(overridden.data[2], 28.0);
        assert_close(unaffected.data[0], 14.0);
        assert_close(unaffected.data[1], 19.0);
        assert_close(unaffected.data[2], 24.0);
    }

    #[test]
    fn linear_accumulation_precision_f64_reconstructs_descriptor_layout() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_LINEAR_ACCUMULATION", "f64");
        std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, 1.0e-3, -2.0]).unwrap();
        let weight = CpuTensor::from_f32(
            "weight",
            vec![3, 2],
            vec![1.0e8, -1.0e8, -1.0e8, 1.0e8, 0.25, -0.5],
        )
        .unwrap();

        let actual = linear(&input, &weight, "out").unwrap();

        std::env::remove_var("CAMELID_LINEAR_ACCUMULATION");
        std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");
        let expected = vec![
            (1.0_f64 * 1.0e8 + 1.0e-3 * -1.0e8 + -2.0 * 0.25) as f32,
            (1.0_f64 * -1.0e8 + 1.0e-3 * 1.0e8 + -2.0 * -0.5) as f32,
        ];
        assert_eq!(actual.shape.dims, vec![1, 2]);
        assert_eq!(actual.data, expected);
    }

    #[test]
    fn linear_accumulation_precision_f64_reconstructs_transposed_layout() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_LINEAR_ACCUMULATION", "f64");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, 1.0e-3, -2.0]).unwrap();
        let weight = CpuTensor::from_f32(
            "weight",
            vec![2, 3],
            vec![1.0e8, -1.0e8, 0.25, -1.0e8, 1.0e8, -0.5],
        )
        .unwrap();

        let actual = linear(&input, &weight, "out").unwrap();

        std::env::remove_var("CAMELID_LINEAR_ACCUMULATION");
        let expected = vec![
            (1.0_f64 * 1.0e8 + 1.0e-3 * -1.0e8 + -2.0 * 0.25) as f32,
            (1.0_f64 * -1.0e8 + 1.0e-3 * 1.0e8 + -2.0 * -0.5) as f32,
        ];
        assert_eq!(actual.shape.dims, vec![1, 2]);
        assert_eq!(actual.data, expected);
    }

    #[test]
    fn rectangular_linear_transposed_diagnostic_reinterprets_descriptor_weight() {
        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
        let descriptor_weight = CpuTensor::from_f32(
            "attention_k.weight",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // descriptor row for input dim 0
                4.0, 5.0, 6.0, // descriptor row for input dim 1
            ],
        )
        .unwrap();

        let auto = linear_with_diagnostic_layouts(
            &input,
            &descriptor_weight,
            "auto",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Auto,
        )
        .unwrap();
        let forced_transposed = linear_with_diagnostic_layouts(
            &input,
            &descriptor_weight,
            "forced_transposed",
            SquareLinearLayout::Descriptor,
            RectangularLinearLayout::Transposed,
        )
        .unwrap();

        assert_eq!(auto.shape.dims, vec![1, 3]);
        assert_eq!(forced_transposed.shape.dims, vec![1, 3]);
        assert_close(auto.data[0], 8.0);
        assert_close(auto.data[1], 18.0);
        assert_close(auto.data[2], 28.0);
        assert_close(forced_transposed.data[0], 8.0);
        assert_close(forced_transposed.data[1], 18.0);
        assert_close(forced_transposed.data[2], 28.0);
    }

    #[test]
    fn gated_ffn_activation_matches_separate_linear_silu_mul_for_transposed_weights() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");

        let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, -2.0, 0.5]).unwrap();
        let gate = CpuTensor::from_f32(
            "gate",
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0, // x
                0.0, 1.0, 0.0, // y
                0.0, 0.0, 1.0, // z
                0.5, -0.5, 1.0,
            ],
        )
        .unwrap();
        let up = CpuTensor::from_f32(
            "up",
            vec![4, 3],
            vec![
                -1.0, 0.0, 0.0, // -x
                0.0, 2.0, 0.0, // 2y
                0.0, 0.0, 3.0, // 3z
                1.0, 1.0, 1.0,
            ],
        )
        .unwrap();

        let separate = linear(&input, &gate, "gate_out")
            .unwrap()
            .silu_mul(&linear(&input, &up, "up_out").unwrap(), "separate")
            .unwrap();
        let fused = gated_ffn_activation(&input, &gate, &up, "fused", true)
            .unwrap()
            .tensor;

        assert_eq!(fused.shape.dims, vec![1, 4]);
        for (actual, expected) in fused.data.iter().zip(separate.data) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn gated_ffn_activation_matches_reference_for_wide_transposed_weights() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let input_values = vec![1.0, -2.0, 0.5];
        let output_width = 1031;
        let input = CpuTensor::from_f32("input", vec![1, input_values.len()], input_values.clone())
            .unwrap();
        let mut gate_values = Vec::with_capacity(output_width * input_values.len());
        let mut up_values = Vec::with_capacity(output_width * input_values.len());
        for idx in 0..output_width {
            gate_values.extend_from_slice(&[
                0.01 * ((idx % 7) as f32 - 3.0),
                -0.02 * ((idx % 5) as f32 - 2.0),
                0.03 * ((idx % 11) as f32 - 5.0),
            ]);
            up_values.extend_from_slice(&[
                -0.015 * ((idx % 13) as f32 - 6.0),
                0.012 * ((idx % 17) as f32 - 8.0),
                0.02 * ((idx % 19) as f32 - 9.0),
            ]);
        }
        let gate = CpuTensor::from_f32(
            "gate",
            vec![output_width, input_values.len()],
            gate_values.clone(),
        )
        .unwrap();
        let up = CpuTensor::from_f32(
            "up",
            vec![output_width, input_values.len()],
            up_values.clone(),
        )
        .unwrap();

        let fused = gated_ffn_activation(&input, &gate, &up, "fused", false)
            .unwrap()
            .tensor;

        assert_eq!(fused.shape.dims, vec![1, output_width]);
        for idx in 0..output_width {
            let row_start = idx * input_values.len();
            let gate_value = input_values
                .iter()
                .zip(&gate_values[row_start..row_start + input_values.len()])
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let up_value = input_values
                .iter()
                .zip(&up_values[row_start..row_start + input_values.len()])
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert_close(fused.data[idx], silu(gate_value) * up_value);
        }
    }

    #[test]
    fn gated_ffn_activation_matches_separate_linear_silu_mul_for_direct_weights() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");

        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, -1.0]).unwrap();
        let gate = CpuTensor::from_f32(
            "gate",
            vec![2, 3],
            vec![
                1.0, 0.0, -1.0, // input col 0 contributions
                0.5, 2.0, 1.0, // input col 1 contributions
            ],
        )
        .unwrap();
        let up =
            CpuTensor::from_f32("up", vec![2, 3], vec![-1.0, 0.25, 0.5, 1.5, -0.5, 2.0]).unwrap();

        let separate = linear(&input, &gate, "gate_out")
            .unwrap()
            .silu_mul(&linear(&input, &up, "up_out").unwrap(), "separate")
            .unwrap();
        let fused = gated_ffn_activation(&input, &gate, &up, "fused", true)
            .unwrap()
            .tensor;

        assert_eq!(fused.shape.dims, vec![1, 3]);
        for (actual, expected) in fused.data.iter().zip(separate.data) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn ffn_gate_up_order_diagnostic_can_apply_silu_to_up_projection() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_FFN_GATE_UP_ORDER", "up_gate");

        let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, -1.0]).unwrap();
        let gate = CpuTensor::from_f32(
            "gate",
            vec![2, 3],
            vec![
                1.0, 0.0, -1.0, // input col 0 contributions
                0.5, 2.0, 1.0, // input col 1 contributions
            ],
        )
        .unwrap();
        let up =
            CpuTensor::from_f32("up", vec![2, 3], vec![-1.0, 0.25, 0.5, 1.5, -0.5, 2.0]).unwrap();

        let separate = linear(&input, &up, "up_out")
            .unwrap()
            .silu_mul(&linear(&input, &gate, "gate_out").unwrap(), "separate")
            .unwrap();
        let fused = gated_ffn_activation(&input, &gate, &up, "fused", true).unwrap();

        assert_eq!(fused.tensor.shape.dims, vec![1, 3]);
        for (actual, expected) in fused.tensor.data.iter().zip(separate.data) {
            assert_close(*actual, expected);
        }
        let diagnostic = fused.activation_diagnostic.expect("activation diagnostic");
        assert_eq!(diagnostic.activation_order, "up_gate");
        assert_close(diagnostic.max_abs_delta, 0.0);

        std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");
    }

    #[test]
    fn single_token_forward_diagnostics_follow_llama_stage_order() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        std::env::set_var("CAMELID_SQUARE_LINEAR_LAYOUT", "descriptor");
        std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");
        std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");
        std::env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "1");

        let config = LlamaModelConfig {
            context_length: 4,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 0.0,
            vocab_size: Some(3),
            file_type: None,
            moe: None,
        };
        let weights = Arc::new(LlamaLoadedWeights {
            token_embedding: CpuTensor::from_f32(
                "token_embd.weight",
                vec![3, 2],
                vec![
                    1.0, 1.0, // token 0, selected by the prompt
                    -1.0, 0.5, // token 1
                    0.25, -0.75, // token 2
                ],
            )
            .unwrap(),
            output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0, 1.0])
                .unwrap(),
            output: Some(
                CpuTensor::from_f32(
                    "output.weight",
                    vec![2, 3],
                    vec![
                        1.0, 0.0, -1.0, // hidden dim 0 -> vocab logits
                        0.0, 1.0, -1.0, // hidden dim 1 -> vocab logits
                    ],
                )
                .unwrap(),
            ),
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: CpuTensor::from_f32(
                    "blk.0.attn_norm.weight",
                    vec![2],
                    vec![1.0, 1.0],
                )
                .unwrap(),
                attention_q: CpuTensor::from_f32(
                    "blk.0.attn_q.weight",
                    vec![2, 2],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .unwrap(),
                attention_k: CpuTensor::from_f32(
                    "blk.0.attn_k.weight",
                    vec![2, 2],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .unwrap(),
                attention_v: CpuTensor::from_f32(
                    "blk.0.attn_v.weight",
                    vec![2, 2],
                    vec![0.25, 0.25, 0.25, 0.25],
                )
                .unwrap(),
                attention_output: CpuTensor::from_f32(
                    "blk.0.attn_output.weight",
                    vec![2, 2],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .unwrap(),
                ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0, 1.0])
                    .unwrap(),
                ffn_gate: CpuTensor::from_f32(
                    "blk.0.ffn_gate.weight",
                    vec![2, 2],
                    vec![1.0, 0.0, 0.0, 2.0],
                )
                .unwrap(),
                ffn_up: CpuTensor::from_f32(
                    "blk.0.ffn_up.weight",
                    vec![2, 2],
                    vec![3.0, 0.0, 0.0, 4.0],
                )
                .unwrap(),
                ffn_down: CpuTensor::from_f32(
                    "blk.0.ffn_down.weight",
                    vec![2, 2],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .unwrap(),
                moe_router: None,
            }],
        });
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();

        let step = session
            .generate_next_token_with_history_diagnostics(&[0], LlamaSampler::Greedy, &[0], true)
            .unwrap();

        assert_eq!(step.prompt_token_count, 1);
        assert_eq!(step.prefill_token_count, 0);
        assert_eq!(step.prefill_timings.total, 0);
        assert_eq!(
            step.first_token_timings
                .memory
                .as_ref()
                .unwrap()
                .forward_passes,
            1
        );
        assert_eq!(step.next_token_id, 1);
        let memory = step
            .timings
            .memory
            .as_ref()
            .expect("memory timings requested");
        assert_eq!(memory.forward_passes, 1);
        assert!(memory.after_embedding.is_some());
        assert!(memory.after_layers.is_some());
        assert!(memory.after_logits.is_some());
        assert_eq!(memory.layers.len(), 1);
        assert_eq!(memory.layers[0].layer_index, 0);
        assert!(memory.layers[0].after_kv_cache_write.is_some());
        assert_eq!(memory.end.as_ref().unwrap().kv_cache_position, 1);
        let diagnostics = step.diagnostics.expect("dense diagnostics requested");
        assert_slice_close(&diagnostics.embedding.checkpoint.first_values, &[1.0, 1.0]);
        assert_eq!(diagnostics.layers.len(), 1);
        let layer = &diagnostics.layers[0];
        assert_eq!(layer.layer_index, 0);

        assert_slice_close(
            &layer.residual_flow.attention_input.checkpoint.first_values,
            &[1.0, 1.0],
        );
        assert_slice_close(&layer.attention_norm.checkpoint.first_values, &[1.0, 1.0]);
        assert_close(layer.attention_norm_reconstruction.input_rms, 1.0);
        assert_close(layer.attention_norm_reconstruction.max_abs_delta, 0.0);
        assert_slice_close(&layer.attention_q.checkpoint.first_values, &[1.0, 1.0]);
        assert_slice_close(&layer.attention_k.checkpoint.first_values, &[1.0, 1.0]);
        assert_slice_close(&layer.attention_q_rope.checkpoint.first_values, &[1.0, 1.0]);
        assert_slice_close(&layer.attention_k_rope.checkpoint.first_values, &[1.0, 1.0]);
        assert_slice_close(&layer.attention_v.checkpoint.first_values, &[0.5, 0.5]);
        assert_eq!(layer.kv_cache_trace.layer_index, 0);
        assert_eq!(layer.kv_cache_trace.position_count, 1);
        assert_eq!(layer.kv_cache_trace.key_value_width, 2);
        assert_close(layer.kv_cache_trace.key_checksum as f32, 3.0);
        assert_close(layer.kv_cache_trace.value_checksum as f32, 1.5);
        assert_close(layer.kv_cache_trace.key_rms, 1.0);
        assert_close(layer.kv_cache_trace.value_rms, 0.5);
        assert_eq!(layer.kv_cache_trace.sampled_positions.len(), 1);
        assert_slice_close(
            &layer.kv_cache_trace.sampled_positions[0].key_first_values,
            &[1.0, 1.0],
        );
        assert_slice_close(
            &layer.kv_cache_trace.sampled_positions[0].value_first_values,
            &[0.5, 0.5],
        );
        assert_slice_close(
            &layer.attention_context.checkpoint.first_values,
            &[0.5, 0.5],
        );
        assert_eq!(layer.attention_trace.position_count, 1);
        assert_close(layer.attention_trace.heads[0].positions[0].probability, 1.0);
        assert_slice_close(&layer.attention_output.checkpoint.first_values, &[0.5, 0.5]);
        assert_slice_close(
            &layer.attention_residual.checkpoint.first_values,
            &[1.5, 1.5],
        );
        assert_slice_close(
            &layer.residual_flow.attention_delta.delta_first_values,
            &[0.5, 0.5],
        );
        assert_close(layer.residual_flow.attention_delta.max_abs_delta, 0.0);

        assert_slice_close(&layer.ffn_norm.checkpoint.first_values, &[1.0, 1.0]);
        assert_slice_close(&layer.ffn_gate.checkpoint.first_values, &[1.0, 2.0]);
        assert_slice_close(&layer.ffn_up.checkpoint.first_values, &[3.0, 4.0]);
        let expected_activation = vec![silu(1.0) * 3.0, silu(2.0) * 4.0];
        assert_slice_close(
            &layer.ffn_activation.checkpoint.first_values,
            &expected_activation,
        );
        assert_eq!(
            layer.ffn_activation_reconstruction.activation_order,
            "gate_up"
        );
        assert_close(layer.ffn_activation_reconstruction.max_abs_delta, 0.0);
        assert_slice_close(
            &layer.ffn_output.checkpoint.first_values,
            &expected_activation,
        );

        let expected_hidden = vec![1.5 + expected_activation[0], 1.5 + expected_activation[1]];
        assert_slice_close(
            &layer.ffn_residual.checkpoint.first_values,
            &expected_hidden,
        );
        assert_slice_close(
            &diagnostics.final_hidden.checkpoint.first_values,
            &expected_hidden,
        );
        assert_close(layer.residual_flow.ffn_delta.max_abs_delta, 0.0);

        let final_mean_square = expected_hidden
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / expected_hidden.len() as f32;
        let final_scale = 1.0 / final_mean_square.sqrt();
        let expected_output_norm = expected_hidden
            .iter()
            .map(|value| value * final_scale)
            .collect::<Vec<_>>();
        assert_slice_close(
            &diagnostics.output_norm.checkpoint.first_values,
            &expected_output_norm,
        );
        assert_close(diagnostics.final_norm.max_abs_delta, 0.0);

        let expected_logits = vec![
            expected_output_norm[0],
            expected_output_norm[1],
            -expected_output_norm[0] - expected_output_norm[1],
        ];
        assert_slice_close(&step.logits.data, &expected_logits);
        assert_slice_close(
            &diagnostics.logits.checkpoint.first_values,
            &expected_logits,
        );
    }

    #[test]
    fn chunked_prefill_matches_sequential_prefill_outputs_and_cache() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 12,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1.0e-5,
            vocab_size: Some(4),
            file_type: None,
            moe: None,
        };
        let weights = Arc::new(LlamaLoadedWeights {
            token_embedding: CpuTensor::from_f32(
                "token_embd.weight",
                vec![4, 2],
                vec![1.0, 0.25, -0.5, 0.75, 0.3, -0.8, 0.2, 0.4],
            )
            .unwrap(),
            output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![0.9, 1.1])
                .unwrap(),
            output: Some(
                CpuTensor::from_f32(
                    "output.weight",
                    vec![4, 2],
                    vec![0.7, -0.2, -0.4, 0.6, 0.1, 0.3, -0.5, -0.1],
                )
                .unwrap(),
            ),
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: CpuTensor::from_f32(
                    "blk.0.attn_norm.weight",
                    vec![2],
                    vec![1.0, 0.8],
                )
                .unwrap(),
                attention_q: CpuTensor::from_f32(
                    "blk.0.attn_q.weight",
                    vec![2, 2],
                    vec![0.5, -0.1, 0.25, 0.7],
                )
                .unwrap(),
                attention_k: CpuTensor::from_f32(
                    "blk.0.attn_k.weight",
                    vec![2, 2],
                    vec![0.3, 0.2, -0.4, 0.6],
                )
                .unwrap(),
                attention_v: CpuTensor::from_f32(
                    "blk.0.attn_v.weight",
                    vec![2, 2],
                    vec![0.2, -0.3, 0.5, 0.4],
                )
                .unwrap(),
                attention_output: CpuTensor::from_f32(
                    "blk.0.attn_output.weight",
                    vec![2, 2],
                    vec![0.6, 0.1, -0.2, 0.9],
                )
                .unwrap(),
                ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.2, 0.7])
                    .unwrap(),
                ffn_gate: CpuTensor::from_f32(
                    "blk.0.ffn_gate.weight",
                    vec![2, 2],
                    vec![0.4, -0.6, 0.8, 0.2],
                )
                .unwrap(),
                ffn_up: CpuTensor::from_f32(
                    "blk.0.ffn_up.weight",
                    vec![2, 2],
                    vec![-0.3, 0.9, 0.5, 0.1],
                )
                .unwrap(),
                ffn_down: CpuTensor::from_f32(
                    "blk.0.ffn_down.weight",
                    vec![2, 2],
                    vec![0.7, -0.2, 0.4, 0.3],
                )
                .unwrap(),
                moe_router: None,
            }],
        });

        let prompt = [0, 1, 2, 3, 0, 1, 2];

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "1");
        let mut sequential = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
        let sequential_step = sequential
            .generate_next_token_with_history_diagnostics(
                &prompt,
                LlamaSampler::Greedy,
                &prompt,
                false,
            )
            .unwrap();

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "2");
        std::env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "1");
        let mut chunked = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
        let chunked_step = chunked
            .generate_next_token_with_history_diagnostics(
                &prompt,
                LlamaSampler::Greedy,
                &prompt,
                false,
            )
            .unwrap();

        let prefill_memory = chunked_step
            .prefill_timings
            .memory
            .as_ref()
            .expect("chunked prefill records structured memory timings");
        assert_eq!(prefill_memory.forward_passes, 3);
        assert_eq!(prefill_memory.layers.len(), 1);
        assert_eq!(prefill_memory.end.as_ref().unwrap().kv_cache_position, 6);
        for layer_memory in &prefill_memory.layers {
            assert_eq!(layer_memory.forward_passes, 3);
            assert!(layer_memory.after_attention_norm.is_some());
            assert!(layer_memory.after_attention_q.is_some());
            assert!(layer_memory.after_attention_k.is_some());
            assert!(layer_memory.after_attention_rope.is_some());
            assert!(layer_memory.after_attention_v.is_some());
            assert!(layer_memory.after_kv_cache_write.is_some());
            assert!(layer_memory.after_attention_context.is_some());
            assert!(layer_memory.after_attention_output.is_some());
            assert!(layer_memory.after_attention_residual.is_some());
            assert!(layer_memory.after_ffn_norm.is_some());
            assert!(layer_memory.after_ffn_activation.is_some());
            assert!(layer_memory.after_ffn_down.is_some());
            assert!(layer_memory.after_ffn_residual.is_some());
            assert_eq!(layer_memory.q8_file_reads, Q8_0FileReadStats::default());
        }
        assert_eq!(prefill_memory.q8_file_reads, Q8_0FileReadStats::default());

        assert_eq!(chunked_step.next_token_id, sequential_step.next_token_id);
        assert_slice_close(&chunked_step.logits.data, &sequential_step.logits.data);
        assert_slice_close(
            &chunked_step.hidden_state.data,
            &sequential_step.hidden_state.data,
        );
        assert_eq!(chunked.kv_cache.position, sequential.kv_cache.position);
        assert_slice_close(&chunked.kv_cache.keys, &sequential.kv_cache.keys);
        assert_slice_close(&chunked.kv_cache.values, &sequential.kv_cache.values);

        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", "1");
        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION", "1");
        let mut layer_major = LlamaInferenceSession::new(config, weights).unwrap();
        let layer_major_step = layer_major
            .generate_next_token_with_history_diagnostics(
                &prompt,
                LlamaSampler::Greedy,
                &prompt,
                false,
            )
            .unwrap();
        let layer_major_memory = layer_major_step
            .prefill_timings
            .memory
            .as_ref()
            .expect("layer-major attribution enables structured prefill memory");
        assert!(!layer_major_memory
            .prefill_layer_major_attribution
            .is_empty());
        let first_attribution = &layer_major_memory.prefill_layer_major_attribution[0];
        assert_eq!(first_attribution.layer_index, 0);
        assert_eq!(first_attribution.chunk_start, 0);
        assert!(first_attribution.chunk_rows > 0);
        assert!(first_attribution.hidden_bytes > 0);
        assert!(first_attribution.next_hidden_bytes > 0);
        assert!(first_attribution.chunk_input_bytes > 0);
        assert!(first_attribution.kv_cache_bytes_after >= first_attribution.kv_cache_bytes_before);
        let serialized_memory = serde_json::to_value(layer_major_memory).unwrap();
        assert!(serialized_memory
            .get("prefill_layer_major_attribution")
            .and_then(|value| value.as_array())
            .is_some_and(|value| !value.is_empty()));
        assert_eq!(
            layer_major_step.next_token_id,
            sequential_step.next_token_id
        );
        assert_slice_close(&layer_major_step.logits.data, &sequential_step.logits.data);
        assert_slice_close(
            &layer_major_step.hidden_state.data,
            &sequential_step.hidden_state.data,
        );
        assert_eq!(layer_major.kv_cache.position, sequential.kv_cache.position);
        assert_slice_close(&layer_major.kv_cache.keys, &sequential.kv_cache.keys);
        assert_slice_close(&layer_major.kv_cache.values, &sequential.kv_cache.values);

        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR");
        std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION");
        std::env::remove_var("CAMELID_FORWARD_RSS_TIMINGS");
    }

    #[test]
    fn prefill_layer_rejects_misaligned_kv_cache_cursor() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 8,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1.0e-5,
            vocab_size: Some(4),
            file_type: None,
            moe: None,
        };
        let layer = LlamaLayerWeights {
            attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0, 1.0])
                .unwrap(),
            attention_q: CpuTensor::from_f32(
                "blk.0.attn_q.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_k: CpuTensor::from_f32(
                "blk.0.attn_k.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_v: CpuTensor::from_f32(
                "blk.0.attn_v.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_output: CpuTensor::from_f32(
                "blk.0.attn_output.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0, 1.0])
                .unwrap(),
            ffn_gate: CpuTensor::from_f32(
                "blk.0.ffn_gate.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            ffn_up: CpuTensor::from_f32(
                "blk.0.ffn_up.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            ffn_down: CpuTensor::from_f32(
                "blk.0.ffn_down.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            moe_router: None,
        };
        let hidden = CpuTensor::from_f32("hidden", vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let plan = LlamaKvCachePlan::from_config(&config).unwrap();
        let mut kv_cache = LlamaKvCache::new(plan).unwrap();
        kv_cache.position = 1;

        let err = forward_prefill_layer_chunk_timed(
            &hidden,
            &layer,
            PrefillLayerChunkParams {
                config: &config,
                rope_freqs: None,
                rms_norm_epsilon: config.rms_norm_epsilon,
                layer_idx: 0,
                base_position: 0,
                chunk_start: 0,
                chunk_rows: 2,
            },
            &mut kv_cache,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("prefill chunk base position 0 does not match KV cache cursor 1"));
    }

    #[test]
    fn batch_attention_rejects_reads_beyond_allocated_kv_cache() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 2,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1.0e-5,
            vocab_size: Some(4),
            file_type: None,
            moe: None,
        };
        let kv_cache = LlamaKvCache::new(LlamaKvCachePlan::from_config(&config).unwrap()).unwrap();
        let query = CpuTensor::from_f32("query", vec![1, 2], vec![0.1, 0.2]).unwrap();

        let err =
            causal_attention_context_batch(&kv_cache, 0, 0, &query, 1, 1, "context").unwrap_err();

        assert!(err
            .to_string()
            .contains("attention batch needs 1 cached position(s), but KV cache has 0 allocated"));
    }

    #[test]
    fn batch_attention_parallel_context_matches_serial() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let rows = 8;
        let attention_heads = 32;
        let kv_heads = 8;
        let head_dim = 2;
        let expected_width = attention_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let plan = LlamaKvCachePlan {
            max_sequence_length: rows,
            layer_count: 1,
            kv_head_count: kv_heads,
            head_dim,
            key_shape: vec![1, rows, kv_heads, head_dim],
            value_shape: vec![1, rows, kv_heads, head_dim],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
        let key_data: Vec<f32> = (0..rows * kv_width)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
            .collect();
        let value_data: Vec<f32> = (0..rows * kv_width)
            .map(|idx| 10.0 + ((idx % 17) as f32) * 0.25)
            .collect();
        let query_data: Vec<f32> = (0..rows * expected_width)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.0625)
            .collect();

        let key = CpuTensor::from_f32("key", vec![rows, kv_width], key_data).unwrap();
        let value = CpuTensor::from_f32("value", vec![rows, kv_width], value_data).unwrap();
        write_kv_cache_batch(&mut kv_cache, 0, 0, &key, &value).unwrap();
        let query = CpuTensor::from_f32("query", vec![rows, expected_width], query_data).unwrap();

        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let serial = serial_pool
            .install(|| {
                assert!(!should_parallelize_attention_context_batch(
                    rows,
                    attention_heads
                ));
                causal_attention_context_batch(
                    &kv_cache,
                    0,
                    0,
                    &query,
                    attention_heads,
                    kv_heads,
                    "serial",
                )
            })
            .unwrap();

        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let parallel = parallel_pool
            .install(|| {
                assert!(should_parallelize_attention_context_batch(
                    rows,
                    attention_heads
                ));
                causal_attention_context_batch(
                    &kv_cache,
                    0,
                    0,
                    &query,
                    attention_heads,
                    kv_heads,
                    "parallel",
                )
            })
            .unwrap();

        assert_eq!(parallel.shape.dims, serial.shape.dims);
        assert_slice_close(&parallel.data, &serial.data);
    }

    #[test]
    fn batch_attention_parallel_context_respects_threshold_and_thread_count() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        let single_thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        single_thread_pool.install(|| {
            assert!(!should_parallelize_attention_context_batch(16, 32));
        });

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            assert!(!should_parallelize_attention_context_batch(7, 32));
            assert!(should_parallelize_attention_context_batch(8, 32));
        });
    }

    #[test]
    fn zero_prefill_chunk_env_falls_back_without_panicking() {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();

        let config = LlamaModelConfig {
            context_length: 8,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1.0e-5,
            vocab_size: Some(4),
            file_type: None,
            moe: None,
        };
        let weights = Arc::new(LlamaLoadedWeights {
            token_embedding: CpuTensor::from_f32(
                "token_embd.weight",
                vec![4, 2],
                vec![1.0, 0.25, -0.5, 0.75, 0.3, -0.8, 0.2, 0.4],
            )
            .unwrap(),
            output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![0.9, 1.1])
                .unwrap(),
            output: Some(
                CpuTensor::from_f32(
                    "output.weight",
                    vec![4, 2],
                    vec![0.7, -0.2, -0.4, 0.6, 0.1, 0.3, -0.5, -0.1],
                )
                .unwrap(),
            ),
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: CpuTensor::from_f32(
                    "blk.0.attn_norm.weight",
                    vec![2],
                    vec![1.0, 0.8],
                )
                .unwrap(),
                attention_q: CpuTensor::from_f32(
                    "blk.0.attn_q.weight",
                    vec![2, 2],
                    vec![0.5, -0.1, 0.25, 0.7],
                )
                .unwrap(),
                attention_k: CpuTensor::from_f32(
                    "blk.0.attn_k.weight",
                    vec![2, 2],
                    vec![0.3, 0.2, -0.4, 0.6],
                )
                .unwrap(),
                attention_v: CpuTensor::from_f32(
                    "blk.0.attn_v.weight",
                    vec![2, 2],
                    vec![0.2, -0.3, 0.5, 0.4],
                )
                .unwrap(),
                attention_output: CpuTensor::from_f32(
                    "blk.0.attn_output.weight",
                    vec![2, 2],
                    vec![0.6, 0.1, -0.2, 0.9],
                )
                .unwrap(),
                ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.2, 0.7])
                    .unwrap(),
                ffn_gate: CpuTensor::from_f32(
                    "blk.0.ffn_gate.weight",
                    vec![2, 2],
                    vec![0.4, -0.6, 0.8, 0.2],
                )
                .unwrap(),
                ffn_up: CpuTensor::from_f32(
                    "blk.0.ffn_up.weight",
                    vec![2, 2],
                    vec![-0.3, 0.9, 0.5, 0.1],
                )
                .unwrap(),
                ffn_down: CpuTensor::from_f32(
                    "blk.0.ffn_down.weight",
                    vec![2, 2],
                    vec![0.7, -0.2, 0.4, 0.3],
                )
                .unwrap(),
                moe_router: None,
            }],
        });

        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "0");
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[0, 1, 2],
                LlamaSampler::Greedy,
                &[0, 1, 2],
                false,
            )
            .unwrap();

        assert_eq!(step.prefill_token_count, 2);
        assert!(step.prefill_timings.total > 0);

        std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    }

    #[test]
    fn kv_cache_allocates_positions_lazily_without_losing_prior_layers() {
        let plan = LlamaKvCachePlan {
            max_sequence_length: 10,
            layer_count: 2,
            kv_head_count: 1,
            head_dim: 2,
            key_shape: vec![2, 10, 1, 2],
            value_shape: vec![2, 10, 1, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
        assert_eq!(kv_cache.allocated_sequence_length, 0);
        assert!(kv_cache.keys.is_empty());
        assert!(kv_cache.values.is_empty());

        let layer0_key = CpuTensor::from_f32("layer0_key", vec![1, 2], vec![1.0, 2.0]).unwrap();
        let layer0_value = CpuTensor::from_f32("layer0_value", vec![1, 2], vec![3.0, 4.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &layer0_key, &layer0_value).unwrap();
        assert_eq!(kv_cache.allocated_sequence_length, 1);
        assert_eq!(kv_cache.keys.len(), 4);
        assert_eq!(kv_cache.values.len(), 4);

        let layer1_key = CpuTensor::from_f32("layer1_key", vec![1, 2], vec![5.0, 6.0]).unwrap();
        let layer1_value = CpuTensor::from_f32("layer1_value", vec![1, 2], vec![7.0, 8.0]).unwrap();
        write_kv_cache(&mut kv_cache, 1, &layer1_key, &layer1_value).unwrap();

        kv_cache.position = 1;
        let layer0_next_key =
            CpuTensor::from_f32("layer0_next_key", vec![1, 2], vec![9.0, 10.0]).unwrap();
        let layer0_next_value =
            CpuTensor::from_f32("layer0_next_value", vec![1, 2], vec![11.0, 12.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &layer0_next_key, &layer0_next_value).unwrap();
        assert_eq!(kv_cache.allocated_sequence_length, 2);
        assert_eq!(kv_cache.keys.len(), 8);
        assert_eq!(kv_cache.values.len(), 8);

        let prior_layer1_start = kv_cache_offset(&kv_cache, 1, 0, 0);
        assert_eq!(
            &kv_cache.keys[prior_layer1_start..prior_layer1_start + 2],
            &[5.0, 6.0]
        );
        assert_eq!(
            &kv_cache.values[prior_layer1_start..prior_layer1_start + 2],
            &[7.0, 8.0]
        );
    }

    #[test]
    fn kv_cache_storage_matches_llama_cpp_f16_rounding() {
        let plan = LlamaKvCachePlan {
            max_sequence_length: 1,
            layer_count: 1,
            kv_head_count: 1,
            head_dim: 2,
            key_shape: vec![1, 1, 1, 2],
            value_shape: vec![1, 1, 1, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
        let key = CpuTensor::from_f32("key", vec![1, 2], vec![1.0001, -2.0003]).unwrap();
        let value = CpuTensor::from_f32("value", vec![1, 2], vec![3.0007, -4.0009]).unwrap();

        write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();

        assert_eq!(
            kv_cache.keys,
            key.data
                .iter()
                .copied()
                .map(|value| f16_bits_to_f32(f32_to_f16_bits(value)))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            kv_cache.values,
            value
                .data
                .iter()
                .copied()
                .map(|value| f16_bits_to_f32(f32_to_f16_bits(value)))
                .collect::<Vec<_>>()
        );
        assert_ne!(kv_cache.keys, key.data);
        assert_ne!(kv_cache.values, value.data);
    }

    #[test]
    fn causal_attention_context_attends_over_prior_and_current_positions() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

        let plan = LlamaKvCachePlan {
            max_sequence_length: 3,
            layer_count: 1,
            kv_head_count: 1,
            head_dim: 2,
            key_shape: vec![1, 3, 1, 2],
            value_shape: vec![1, 3, 1, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

        let prior_key = CpuTensor::from_f32("prior_key", vec![1, 2], vec![1.0, 0.0]).unwrap();
        let prior_value = CpuTensor::from_f32("prior_value", vec![1, 2], vec![10.0, 0.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &prior_key, &prior_value).unwrap();
        kv_cache.position = 1;
        let current_key = CpuTensor::from_f32("current_key", vec![1, 2], vec![0.0, 1.0]).unwrap();
        let current_value =
            CpuTensor::from_f32("current_value", vec![1, 2], vec![0.0, 20.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &current_key, &current_value).unwrap();

        let query = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
        let context =
            causal_attention_context(&kv_cache, 0, &query, 1, 1, "context", true).unwrap();

        let first_score = (1.0_f32 / 2.0_f32.sqrt()).exp();
        let first_probability = first_score / (first_score + 1.0);
        assert_eq!(context.tensor.shape.dims, vec![1, 2]);
        assert_close(context.tensor.data[0], first_probability * 10.0);
        assert_close(context.tensor.data[1], (1.0 - first_probability) * 20.0);
        let trace = context.trace.expect("trace diagnostics requested");
        assert_eq!(trace.position_count, 2);
        assert_eq!(trace.head_dim, 2);
        assert_eq!(trace.heads.len(), 1);
        let head = &trace.heads[0];
        assert_eq!(head.attention_head, 0);
        assert_eq!(head.kv_head, 0);
        assert_eq!(head.query_first_values, vec![1.0, 0.0]);
        assert_close(head.probability_sum, 1.0);
        assert_close(
            head.probability_entropy,
            -(first_probability * first_probability.ln()
                + (1.0 - first_probability) * (1.0 - first_probability).ln()),
        );
        assert_close(
            head.probability_rms,
            ((first_probability * first_probability
                + (1.0 - first_probability) * (1.0 - first_probability))
                / 2.0)
                .sqrt(),
        );
        assert_eq!(head.max_probability_position, 0);
        assert_close(head.max_probability, first_probability);
        assert_eq!(head.top_probability_positions.len(), 2);
        assert_eq!(head.top_probability_positions[0].position, 0);
        assert_close(
            head.top_probability_positions[0].score,
            1.0 / 2.0_f32.sqrt(),
        );
        assert_close(
            head.top_probability_positions[0].probability,
            first_probability,
        );
        assert_eq!(
            head.top_probability_positions[0].key_first_values,
            vec![1.0, 0.0]
        );
        assert_eq!(
            head.top_probability_positions[0].value_first_values,
            vec![10.0, 0.0]
        );
        assert_eq!(head.context_reconstruction_max_abs_delta_index, 0);
        assert_close(head.context_reconstruction_max_abs_delta, 0.0);
        assert_eq!(head.positions.len(), 2);
        assert_close(head.positions[0].score, 1.0 / 2.0_f32.sqrt());
        assert_close(head.positions[0].probability, first_probability);
        assert_eq!(head.positions[0].key_first_values, vec![1.0, 0.0]);
        assert_eq!(head.positions[0].value_first_values, vec![10.0, 0.0]);
        assert_close(head.context_first_values[0], first_probability * 10.0);
        assert_close(
            head.context_first_values[1],
            (1.0 - first_probability) * 20.0,
        );
        assert_eq!(head.reconstructed_context_first_values.len(), 2);
        assert_close(
            head.reconstructed_context_first_values[0],
            first_probability * 10.0,
        );
        assert_close(
            head.reconstructed_context_first_values[1],
            (1.0 - first_probability) * 20.0,
        );
    }

    #[test]
    fn causal_attention_context_repeats_grouped_kv_heads_for_single_position() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

        let plan = LlamaKvCachePlan {
            max_sequence_length: 1,
            layer_count: 1,
            kv_head_count: 2,
            head_dim: 2,
            key_shape: vec![1, 1, 2, 2],
            value_shape: vec![1, 1, 2, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

        let key = CpuTensor::from_f32("key", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let value = CpuTensor::from_f32("value", vec![1, 4], vec![10.0, 11.0, 20.0, 21.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();

        let query = CpuTensor::from_f32(
            "query",
            vec![1, 8],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
        )
        .unwrap();
        let context =
            causal_attention_context(&kv_cache, 0, &query, 4, 2, "context", true).unwrap();

        assert_eq!(
            context.tensor.data,
            vec![10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]
        );

        let trace = context.trace.expect("trace diagnostics requested");
        assert_eq!(trace.position_count, 1);
        assert_eq!(trace.heads.len(), 4);
        assert_eq!(trace.heads[0].attention_head, 0);
        assert_eq!(trace.heads[0].kv_head, 0);
        assert_eq!(trace.heads[1].attention_head, 1);
        assert_eq!(trace.heads[1].kv_head, 0);
        assert_eq!(trace.heads[2].attention_head, 2);
        assert_eq!(trace.heads[2].kv_head, 1);
        assert_eq!(trace.heads[3].attention_head, 3);
        assert_eq!(trace.heads[3].kv_head, 1);
        assert_eq!(trace.heads[1].context_first_values, vec![10.0, 11.0]);
        assert_eq!(trace.heads[1].positions.len(), 1);
        assert_close(trace.heads[1].probability_entropy, 0.0);
        assert_close(trace.heads[1].probability_rms, 1.0);
        assert_close(trace.heads[1].positions[0].score, 0.0);
        assert_close(trace.heads[1].positions[0].reconstructed_score, 0.0);
        assert_close(trace.heads[1].positions[0].score_reconstruction_delta, 0.0);
        assert_eq!(
            trace.heads[1].positions[0].qk_products_first_values,
            vec![0.0, 0.0]
        );
        assert_close(trace.heads[1].context_reconstruction_max_abs_delta, 0.0);
    }

    #[test]
    fn causal_attention_context_repeats_grouped_kv_heads_across_positions() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

        let plan = LlamaKvCachePlan {
            max_sequence_length: 2,
            layer_count: 1,
            kv_head_count: 2,
            head_dim: 2,
            key_shape: vec![1, 2, 2, 2],
            value_shape: vec![1, 2, 2, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

        let prior_key =
            CpuTensor::from_f32("prior_key", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let prior_value =
            CpuTensor::from_f32("prior_value", vec![1, 4], vec![10.0, 0.0, 20.0, 0.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &prior_key, &prior_value).unwrap();
        kv_cache.position = 1;
        let current_key =
            CpuTensor::from_f32("current_key", vec![1, 4], vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let current_value =
            CpuTensor::from_f32("current_value", vec![1, 4], vec![0.0, 11.0, 0.0, 21.0]).unwrap();
        write_kv_cache(&mut kv_cache, 0, &current_key, &current_value).unwrap();

        let query = CpuTensor::from_f32(
            "query",
            vec![1, 8],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let context =
            causal_attention_context(&kv_cache, 0, &query, 4, 2, "context", true).unwrap();

        let high_score = (1.0_f32 / 2.0_f32.sqrt()).exp();
        let high_probability = high_score / (high_score + 1.0);
        let low_probability = 1.0 - high_probability;
        assert_eq!(context.tensor.shape.dims, vec![1, 8]);
        assert_close(context.tensor.data[0], high_probability * 10.0);
        assert_close(context.tensor.data[1], low_probability * 11.0);
        assert_close(context.tensor.data[2], low_probability * 10.0);
        assert_close(context.tensor.data[3], high_probability * 11.0);
        assert_close(context.tensor.data[4], high_probability * 20.0);
        assert_close(context.tensor.data[5], low_probability * 21.0);
        assert_close(context.tensor.data[6], low_probability * 20.0);
        assert_close(context.tensor.data[7], high_probability * 21.0);

        let trace = context.trace.expect("trace diagnostics requested");
        assert_eq!(trace.position_count, 2);
        assert_eq!(trace.heads.len(), 4);
        assert_eq!(trace.heads[0].attention_head, 0);
        assert_eq!(trace.heads[0].kv_head, 0);
        assert_eq!(trace.heads[1].attention_head, 1);
        assert_eq!(trace.heads[1].kv_head, 0);
        assert_eq!(trace.heads[2].attention_head, 2);
        assert_eq!(trace.heads[2].kv_head, 1);
        assert_eq!(trace.heads[3].attention_head, 3);
        assert_eq!(trace.heads[3].kv_head, 1);
        assert_close(trace.heads[0].positions[0].probability, high_probability);
        assert_close(
            trace.heads[0].positions[0].reconstructed_score,
            1.0 / 2.0_f32.sqrt(),
        );
        assert_close(trace.heads[0].positions[0].score_reconstruction_delta, 0.0);
        assert_eq!(
            trace.heads[0].positions[0].qk_products_first_values,
            vec![1.0, 0.0]
        );
        assert_eq!(
            trace.heads[0].positions[0].qk_products_max_abs_window_start,
            0
        );
        assert_eq!(
            trace.heads[0].positions[0].qk_products_max_abs_window,
            vec![1.0, 0.0]
        );
        assert_close(trace.heads[1].positions[1].probability, high_probability);
        assert_close(trace.heads[0].context_reconstruction_max_abs_delta, 0.0);
        assert_close(trace.heads[1].context_reconstruction_max_abs_delta, 0.0);
    }

    #[test]
    fn attention_trace_reports_top_probability_positions_outside_edge_samples() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

        let plan = LlamaKvCachePlan {
            max_sequence_length: 10,
            layer_count: 1,
            kv_head_count: 1,
            head_dim: 2,
            key_shape: vec![1, 10, 1, 2],
            value_shape: vec![1, 10, 1, 2],
        };
        let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

        for position in 0..10 {
            kv_cache.position = position;
            let key_values = if position == 5 {
                vec![10.0, 0.0]
            } else {
                vec![0.0, 0.0]
            };
            let key = CpuTensor::from_f32("key", vec![1, 2], key_values).unwrap();
            let value = CpuTensor::from_f32(
                "value",
                vec![1, 2],
                vec![position as f32, -(position as f32)],
            )
            .unwrap();
            write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();
        }
        kv_cache.position = 9;

        let query = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
        let context =
            causal_attention_context(&kv_cache, 0, &query, 1, 1, "context", true).unwrap();
        let trace = context.trace.expect("trace diagnostics requested");
        let head = &trace.heads[0];

        assert_eq!(
            head.positions
                .iter()
                .map(|position| position.position)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 6, 7, 8, 9]
        );
        assert_eq!(head.top_probability_positions.len(), 4);
        assert_eq!(head.top_probability_positions[0].position, 5);
        assert!(
            head.top_probability_positions[0].probability
                > head.top_probability_positions[1].probability
        );
        assert_eq!(
            head.top_probability_positions[0].key_first_values,
            vec![10.0, 0.0]
        );
        assert_eq!(
            head.top_probability_positions[0].value_first_values,
            vec![5.0, -5.0]
        );
        assert_close(
            head.top_probability_positions[0].score,
            10.0 / 2.0_f32.sqrt(),
        );
    }

    #[test]
    fn attention_score_scale_diagnostic_supports_default_and_unscaled_modes() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        assert_eq!(
            diagnostic_attention_score_scale().unwrap(),
            AttentionScoreScale::HeadDim
        );
        assert_close(
            attention_score_scale_value(4, diagnostic_attention_score_scale().unwrap()),
            0.5,
        );

        std::env::set_var("CAMELID_ATTENTION_SCORE_SCALE", "none");
        assert_eq!(
            diagnostic_attention_score_scale().unwrap(),
            AttentionScoreScale::None
        );
        assert_close(
            attention_score_scale_value(4, diagnostic_attention_score_scale().unwrap()),
            1.0,
        );

        std::env::set_var("CAMELID_ATTENTION_SCORE_SCALE", "bogus");
        assert!(diagnostic_attention_score_scale().is_err());
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    }

    #[test]
    fn gqa_head_mapping_supports_grouped_and_modulo_indexing() {
        assert_eq!(
            map_attention_head_to_kv_head(0, 2, 2, GqaHeadMapping::Grouped),
            0
        );
        assert_eq!(
            map_attention_head_to_kv_head(1, 2, 2, GqaHeadMapping::Grouped),
            0
        );
        assert_eq!(
            map_attention_head_to_kv_head(2, 2, 2, GqaHeadMapping::Grouped),
            1
        );
        assert_eq!(
            map_attention_head_to_kv_head(3, 2, 2, GqaHeadMapping::Grouped),
            1
        );

        assert_eq!(
            map_attention_head_to_kv_head(0, 2, 2, GqaHeadMapping::Modulo),
            0
        );
        assert_eq!(
            map_attention_head_to_kv_head(1, 2, 2, GqaHeadMapping::Modulo),
            1
        );
        assert_eq!(
            map_attention_head_to_kv_head(2, 2, 2, GqaHeadMapping::Modulo),
            0
        );
        assert_eq!(
            map_attention_head_to_kv_head(3, 2, 2, GqaHeadMapping::Modulo),
            1
        );
    }

    #[test]
    fn attention_trace_samples_prompt_prefix_and_current_tail_positions() {
        assert_eq!(sampled_attention_trace_positions(0), Vec::<usize>::new());
        assert_eq!(sampled_attention_trace_positions(3), vec![0, 1, 2]);
        assert_eq!(
            sampled_attention_trace_positions(8),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            sampled_attention_trace_positions(18),
            vec![0, 1, 2, 3, 14, 15, 16, 17]
        );
    }

    #[test]
    fn attention_trace_samples_gqa_kv_group_anchors_and_tail_heads() {
        assert_eq!(
            sampled_attention_trace_heads(4, 2, 2, GqaHeadMapping::Grouped),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            sampled_attention_trace_heads(32, 8, 4, GqaHeadMapping::Grouped),
            vec![0, 8, 16, 24, 28, 29, 30, 31]
        );
        assert_eq!(
            sampled_attention_trace_heads(32, 8, 4, GqaHeadMapping::Modulo),
            vec![0, 1, 2, 3, 28, 29, 30, 31]
        );
    }

    #[test]
    fn softmax_top_k_renormalizes_selected_experts() {
        let top = softmax_top_k(&[0.0, 1.0, 2.0], 2);
        assert_eq!(top[0].0, 2);
        assert_eq!(top[1].0, 1);
        let selected_sum = top.iter().map(|(_, weight)| *weight).sum::<f32>();
        assert!((selected_sum - 1.0).abs() < 1.0e-6, "{top:?}");
        let expected_first = 2.0_f32.exp() / (2.0_f32.exp() + 1.0_f32.exp());
        assert!((top[0].1 - expected_first).abs() < 1.0e-6, "{top:?}");
    }

    #[test]
    fn mixtral_moe_ffn_routes_top_k_experts() {
        let input = CpuTensor::from_f32("input", vec![1, 2], vec![1.0, 1.0]).unwrap();
        let router = CpuTensor::from_f32("router", vec![2, 2], vec![10.0, 0.0, 0.0, 0.0]).unwrap();
        let gate_experts = CpuTensor::from_f32(
            "gate_experts",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let up_experts = CpuTensor::from_f32(
            "up_experts",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let down_experts = CpuTensor::from_f32(
            "down_experts",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let (out, ..) = mixtral_moe_ffn(
            &input,
            &router,
            &gate_experts,
            &up_experts,
            &down_experts,
            2,
            "out",
        )
        .unwrap();

        let expected = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!((out.data[0] - expected).abs() < 1.0e-3, "{:?}", out.data);
        assert!((out.data[1] - expected).abs() < 1.0e-3, "{:?}", out.data);
    }
}
