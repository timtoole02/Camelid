use std::env;

use crate::{model::LlamaModelConfig, tensor::CpuTensor, BackendError, Result};

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

    pub(super) fn effective_position(self, position: usize) -> usize {
        match self {
            Self::ZeroBased => position,
            Self::OneBased => position + 1,
        }
    }
}

/// RoPE pairing for a specific model: the `CAMELID_ROPE_PAIRING` env override
/// wins (for diagnostics); otherwise the model's config decides. Qwen3 sets
/// `rope_neox_pairing = true` (split-half / NEOX) because its GGUF weights are
/// not permuted; LLaMA-family rows keep adjacent even/odd because llama.cpp
/// permutes their weights at conversion.
pub(crate) fn rope_pairing_for_config(config: &LlamaModelConfig) -> Result<RopePairing> {
    if env::var_os("CAMELID_ROPE_PAIRING").is_some() {
        return diagnostic_rope_pairing();
    }
    Ok(if config.rope_neox_pairing {
        RopePairing::SplitHalf
    } else {
        RopePairing::AdjacentEvenOdd
    })
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
    // Resolved once per process (non-test): `apply_rope` consults this per
    // call on the decode hot loop, and env reads allocate on Windows. Invalid
    // values stay uncached (the error re-reads; it aborts the forward anyway).
    #[cfg(not(test))]
    {
        static RESOLVED: std::sync::OnceLock<RopeDirection> = std::sync::OnceLock::new();
        if let Some(direction) = RESOLVED.get() {
            return Ok(*direction);
        }
        let direction = diagnostic_rope_direction_uncached()?;
        Ok(*RESOLVED.get_or_init(|| direction))
    }
    #[cfg(test)]
    diagnostic_rope_direction_uncached()
}

fn diagnostic_rope_direction_uncached() -> Result<RopeDirection> {
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
    #[cfg(not(test))]
    {
        static RESOLVED: std::sync::OnceLock<RopePositionMode> = std::sync::OnceLock::new();
        if let Some(mode) = RESOLVED.get() {
            return Ok(*mode);
        }
        let mode = diagnostic_rope_position_mode_uncached()?;
        Ok(*RESOLVED.get_or_init(|| mode))
    }
    #[cfg(test)]
    diagnostic_rope_position_mode_uncached()
}

fn diagnostic_rope_position_mode_uncached() -> Result<RopePositionMode> {
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

pub(super) fn apply_rope(
    tensor: &CpuTensor,
    position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
    rope_offset: usize,
    name: &str,
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
            pairing: rope_pairing_for_config(config)?,
            direction: diagnostic_rope_direction()?,
            position_mode: diagnostic_rope_position_mode()?,
            scaling,
            rope_freqs,
            rope_offset,
        },
        name,
    )
}

pub(super) fn apply_rope_batch(
    tensor: &CpuTensor,
    base_position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
    rope_offset: usize,
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

    let mut data = tensor.data.clone();
    use crate::tensor::should_parallelize_linear_output;
    use rayon::prelude::*;

    let params = RopeParams {
        position: base_position,
        head_count,
        head_dim,
        rope_dim,
        freq_base,
        pairing: rope_pairing_for_config(config)?,
        direction: diagnostic_rope_direction()?,
        position_mode: diagnostic_rope_position_mode()?,
        scaling,
        rope_freqs,
        rope_offset,
    };

    if should_parallelize_linear_output(rows * width) {
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, row_data)| {
                apply_rope_to_row(row_data, base_position + row, params);
            });
    } else {
        for row in 0..rows {
            apply_rope_to_row(
                &mut data[row * width..(row + 1) * width],
                base_position + row,
                params,
            );
        }
    }
    CpuTensor::from_f32(name, tensor.shape.dims.clone(), data)
}

pub(super) fn validate_rope_frequency_tensor(
    rope_freqs: &CpuTensor,
    rope_dim: usize,
) -> Result<&[f32]> {
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
pub(super) struct RopeScaling {
    pub(super) kind: RopeScalingKind,
    pub(super) factor: f32,
    pub(super) original_context_length: Option<u32>,
    pub(super) low_freq_factor: Option<f32>,
    pub(super) high_freq_factor: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RopeScalingKind {
    None,
    Linear,
    Llama3,
    Yarn,
}

impl RopeScalingKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Llama3 => "llama3",
            Self::Yarn => "yarn",
        }
    }
}

// --- YaRN (NTK-by-parts) RoPE, faithful to llama.cpp ggml `rope_yarn`. Defaults match
// ggml: beta_fast=32, beta_slow=1, ext_factor=1, attn_factor=1 (no per-model overrides in
// the GGUF we target). The per-pair frequency is interpolated between the un-scaled
// ("extrapolation") and 1/factor-scaled ("interpolation") angles via a correction ramp,
// and the cos/sin magnitude is multiplied by `mscale`.
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

fn yarn_corr_dim(rope_dim: usize, n_ctx_orig: f32, n_rot: f32, base: f32) -> f32 {
    (rope_dim as f32) * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

fn yarn_corr_dims(rope_dim: usize, n_ctx_orig: u32, base: f32) -> (f32, f32) {
    let start = yarn_corr_dim(rope_dim, n_ctx_orig as f32, YARN_BETA_FAST, base).floor();
    let end = yarn_corr_dim(rope_dim, n_ctx_orig as f32, YARN_BETA_SLOW, base).ceil();
    (start.max(0.0), end.min(rope_dim as f32 - 1.0))
}

fn yarn_ramp(low: f32, high: f32, pair_idx: usize) -> f32 {
    let y = ((pair_idx as f32) - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// cos/sin magnitude scaling (1.0 unless YaRN). ext_factor=1, attn_factor=1, so
/// mscale = 1 + 0.1*ln(1/freq_scale) = 1 + 0.1*ln(factor).
pub(super) fn rope_magnitude_scale(scaling: RopeScaling) -> f32 {
    match scaling.kind {
        RopeScalingKind::Yarn => 1.0 + 0.1 * scaling.factor.ln(),
        _ => 1.0,
    }
}

pub(super) fn rope_scaling_from_config(config: &LlamaModelConfig) -> Result<RopeScaling> {
    let kind = match config.rope_scaling_type.as_deref().map(str::trim) {
        None | Some("") | Some("none") => RopeScalingKind::None,
        Some("linear") => RopeScalingKind::Linear,
        Some("llama3") => RopeScalingKind::Llama3,
        Some("yarn") => RopeScalingKind::Yarn,
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
        RopeScalingKind::Yarn => {
            let original_context_length =
                config.rope_scaling_original_context_length.unwrap_or(8_192);
            if original_context_length == 0 {
                return Err(BackendError::InvalidModelMetadata(
                    "yarn RoPE scaling original context length must be greater than zero"
                        .to_string(),
                ));
            }
            Ok(RopeScaling {
                kind,
                factor,
                original_context_length: Some(original_context_length),
                low_freq_factor: None,
                high_freq_factor: None,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RopeParams<'a> {
    pub(super) position: usize,
    pub(super) head_count: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dim: usize,
    pub(super) freq_base: f32,
    pub(super) pairing: RopePairing,
    pub(super) direction: RopeDirection,
    pub(super) position_mode: RopePositionMode,
    pub(super) scaling: RopeScaling,
    pub(super) rope_freqs: Option<&'a [f32]>,
    pub(super) rope_offset: usize,
}

pub(super) fn apply_rope_with_pairing(
    tensor: &CpuTensor,
    params: RopeParams<'_>,
    name: &str,
) -> Result<CpuTensor> {
    // Pooled copy of the input, bit-identical to `tensor.data.clone()` (RoPE
    // rewrites the rope dims in place and leaves the rest as copied).
    let mut data = super::decode_scratch::take(tensor.data.len());
    data.copy_from_slice(&tensor.data);
    apply_rope_to_row(&mut data, params.position, params);

    super::decode_scratch::tensor_from_pooled(name, &tensor.shape.dims, data)
}

fn apply_rope_to_row(data: &mut [f32], position: usize, mut params: RopeParams<'_>) {
    params.position = position;
    let half_rope_dim = params.rope_dim / 2;
    let mscale = rope_magnitude_scale(params.scaling);
    for pair_idx in 0..half_rope_dim {
        let theta = rope_pair_frequency(pair_idx, &params);
        let angle = params.position_mode.effective_position(params.position) as f32 * theta;
        let (mut sin, mut cos) = angle.sin_cos();
        sin *= mscale;
        cos *= mscale;
        for head in 0..params.head_count {
            let head_start = head * params.head_dim + params.rope_offset;
            let (dim0, dim1) = match params.pairing {
                RopePairing::AdjacentEvenOdd => {
                    let dim0 = head_start + (pair_idx * 2);
                    (dim0, dim0 + 1)
                }
                RopePairing::SplitHalf => {
                    (head_start + pair_idx, head_start + pair_idx + half_rope_dim)
                }
            };
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
        RopeScalingKind::Yarn => {
            // theta = interp*(1-ramp_mix) + extrap*ramp_mix, ext_factor=1 so ramp_mix=ramp.
            let n_ctx_orig = params.scaling.original_context_length.unwrap_or(8_192);
            let (low, high) = yarn_corr_dims(params.rope_dim, n_ctx_orig, params.freq_base);
            let ramp_mix = yarn_ramp(low, high, pair_idx);
            let theta_extrap = effective_base_frequency;
            let theta_interp = theta_extrap / params.scaling.factor; // freq_scale = 1/factor
            theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
        }
    }
}

/// RoPE cos/sin tables for a single position, for the GPU resident-decode kernel (which
/// consumes per-pair cos/sin and applies the rotation itself). Mirrors `apply_rope_to_row`'s
/// angle math exactly — same `rope_pair_frequency` (incl. llama3 scaling and `rope_freqs`)
/// and `effective_position` — so the GPU rotation matches the CPU path bit-for-bit given the
/// same projected K/Q. Returns Ok(None) when the configured RoPE direction is Inverse (the
/// kernel is forward-only), signalling the caller to fall back to the CPU path.
pub(super) struct ResidentRopeTables {
    pub(super) cos: Vec<f32>,
    pub(super) sin: Vec<f32>,
    pub(super) split_half_pairing: bool,
}

/// gemma3 dual-theta RoPE tables for one position: the VERBATIM runnable-lane
/// frequency form `1.0 / base.powf(2*i/rope_dim)` (`RunnableModel::apply_rope`,
/// src/runnable/model.rs) — deliberately NOT `rope_pair_frequency`'s
/// negated-exponent form, whose last-ULP drift against the CPU oracle can flip
/// a near-tie greedy token (same lesson as `qwen35_rope_tables`,
/// src/runnable/model.rs). gemma3 carries no rope scaling (the loader rejects
/// it) and no rope_freqs tensor, so the plain form is exact. Returns
/// `(cos, sin)`, each `rope_dim / 2` long. The caller picks `base` per layer
/// type (`Gemma3Metadata::rope_freq_base_at`).
pub(crate) fn gemma3_rope_tables(
    position: usize,
    rope_dim: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let mut cos_t = vec![0.0f32; half];
    let mut sin_t = vec![0.0f32; half];
    for i in 0..half {
        let freq = 1.0f32 / base.powf(2.0 * i as f32 / rope_dim as f32);
        let (s, c) = (position as f32 * freq).sin_cos();
        cos_t[i] = c;
        sin_t[i] = s;
    }
    (cos_t, sin_t)
}

pub(super) fn resident_decode_rope_tables(
    position: usize,
    head_dim: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
) -> Result<Option<ResidentRopeTables>> {
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
    if diagnostic_rope_direction()? != RopeDirection::Forward {
        return Ok(None);
    }
    let pairing = rope_pairing_for_config(config)?;
    let position_mode = diagnostic_rope_position_mode()?;
    let scaling = rope_scaling_from_config(config)?;
    let rope_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;
    let params = RopeParams {
        position,
        head_count: 1,
        head_dim,
        rope_dim,
        freq_base,
        pairing,
        direction: RopeDirection::Forward,
        position_mode,
        scaling,
        rope_freqs,
        rope_offset: 0,
    };
    let half = rope_dim / 2;
    let eff = position_mode.effective_position(position) as f32;
    let mscale = rope_magnitude_scale(scaling);
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for pair in 0..half {
        let (s, c) = (eff * rope_pair_frequency(pair, &params)).sin_cos();
        cos.push(c * mscale);
        sin.push(s * mscale);
    }
    Ok(Some(ResidentRopeTables {
        cos,
        sin,
        split_half_pairing: matches!(pairing, RopePairing::SplitHalf),
    }))
}

/// Flattened RoPE cos/sin tables for positions `0..n_positions`, for the GPU resident
/// prefill. Identical angle math to `resident_decode_rope_tables` (same
/// `rope_pair_frequency` incl. llama3 scaling and `rope_freqs`, same
/// `effective_position`), but the position-independent per-pair frequencies are derived
/// once instead of once per position.
pub(super) fn resident_prefill_rope_tables(
    n_positions: usize,
    head_dim: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
) -> Result<Option<ResidentRopeTables>> {
    resident_prefill_rope_tables_at(0, n_positions, head_dim, config, rope_freqs)
}

pub(super) fn resident_prefill_rope_tables_at(
    base_position: usize,
    n_positions: usize,
    head_dim: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
) -> Result<Option<ResidentRopeTables>> {
    let first = match resident_decode_rope_tables(base_position, head_dim, config, rope_freqs)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let half = first.cos.len();
    let position_mode = diagnostic_rope_position_mode()?;

    // Recover the per-pair frequencies the same way the per-position builder derives
    // them, then sweep positions with only sin_cos in the inner loop.
    let rope_dim = config.rope_dimension_count.unwrap_or(head_dim as u32) as usize;
    let freq_base = config.rope_freq_base.unwrap_or(10_000.0);
    let pairing = rope_pairing_for_config(config)?;
    let scaling = rope_scaling_from_config(config)?;
    let validated_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;
    let params = RopeParams {
        position: 0,
        head_count: 1,
        head_dim,
        rope_dim,
        freq_base,
        pairing,
        direction: RopeDirection::Forward,
        position_mode,
        scaling,
        rope_freqs: validated_freqs,
        rope_offset: 0,
    };
    let freqs: Vec<f32> = (0..half)
        .map(|pair| rope_pair_frequency(pair, &params))
        .collect();

    let mscale = rope_magnitude_scale(scaling);
    let mut cos_all = Vec::with_capacity(n_positions * half);
    let mut sin_all = Vec::with_capacity(n_positions * half);
    if n_positions > 0 {
        cos_all.extend_from_slice(&first.cos);
        sin_all.extend_from_slice(&first.sin);
    }
    for pos in base_position.saturating_add(1)..base_position.saturating_add(n_positions) {
        let eff = position_mode.effective_position(pos) as f32;
        for &f in &freqs {
            let (s, c) = (eff * f).sin_cos();
            cos_all.push(c * mscale);
            sin_all.push(s * mscale);
        }
    }
    Ok(Some(ResidentRopeTables {
        cos: cos_all,
        sin: sin_all,
        split_half_pairing: first.split_half_pairing,
    }))
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
