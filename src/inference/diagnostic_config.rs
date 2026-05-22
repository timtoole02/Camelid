use std::env;

use crate::{BackendError, Result};

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

pub(super) fn diagnostic_zero_delta_value(
    key: &str,
    value: &str,
    layer_idx: usize,
) -> Result<bool> {
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
pub(super) fn apply_ffn_gate_up_order(
    gate_value: f32,
    up_value: f32,
    order: FfnGateUpOrder,
) -> f32 {
    match order {
        FfnGateUpOrder::GateUp => (gate_value / (1.0 + (-gate_value).exp())) * up_value,
        FfnGateUpOrder::UpGate => (up_value / (1.0 + (-up_value).exp())) * gate_value,
    }
}

pub(super) fn attention_score_scale_value(head_dim: usize, mode: AttentionScoreScale) -> f32 {
    match mode {
        AttentionScoreScale::HeadDim => 1.0 / (head_dim as f32).sqrt(),
        AttentionScoreScale::None => 1.0,
    }
}

pub(super) fn map_attention_head_to_kv_head(
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
        Ok(value) if value == "token_major" || value.is_empty() => {
            Ok(OutputProjectionLayout::TokenMajor)
        }
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
