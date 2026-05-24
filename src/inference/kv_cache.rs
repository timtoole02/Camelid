use std::env;

use serde::Serialize;

use super::{f16_bits_to_f32, f32_to_f16_bits, TENSOR_CHECKPOINT_SAMPLE};
use crate::{
    model::{DenseLlamaDims, LlamaModelConfig},
    tensor::CpuTensor,
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
        let values = target_sequence_length
            .checked_mul(self.plan.layer_count)
            .and_then(|value| value.checked_mul(self.plan.kv_head_count))
            .and_then(|value| value.checked_mul(self.plan.head_dim))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("KV cache element count overflow".to_string())
            })?;
        self.keys.resize(values, 0.0);
        self.values.resize(values, 0.0);
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
        self.keys.len() + self.values.len()
    }

    pub fn allocated_bytes(&self) -> u64 {
        (self.allocated_elements() as u64) * (std::mem::size_of::<f32>() as u64)
    }

    pub(super) fn offset(&self, layer_idx: usize, position: usize, kv_head: usize) -> usize {
        (((position * self.plan.layer_count) + layer_idx) * self.plan.kv_head_count + kv_head)
            * self.plan.head_dim
    }

    pub(super) fn head_base_offset(&self, layer_idx: usize, kv_head: usize) -> usize {
        ((layer_idx * self.plan.kv_head_count) + kv_head) * self.plan.head_dim
    }

    pub(super) fn position_stride(&self) -> usize {
        self.plan.layer_count * self.plan.kv_head_count * self.plan.head_dim
    }
}

pub(super) fn write_kv_cache(
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
    let offset = kv_cache.offset(layer_idx, kv_cache.position, 0);
    let end = offset + expected_width;
    copy_to_f16_kv_cache_storage(&mut kv_cache.keys[offset..end], &key.data);
    copy_to_f16_kv_cache_storage(&mut kv_cache.values[offset..end], &value.data);
    Ok(())
}

pub(super) fn write_kv_cache_batch(
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
        let offset = kv_cache.offset(layer_idx, position, 0);
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

pub(super) fn kv_cache_trace(
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
        let start = kv_cache.offset(layer_idx, position, 0);
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
    let sampled_positions = sampled_kv_cache_trace_positions(position_count)
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
    let start = kv_cache.offset(layer_idx, position, 0);
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

fn sampled_kv_cache_trace_positions(position_count: usize) -> Vec<usize> {
    const POSITION_LIMIT: usize = 8;
    const EDGE_POSITION_LIMIT: usize = POSITION_LIMIT / 2;

    if position_count <= POSITION_LIMIT {
        return (0..position_count).collect();
    }

    let mut positions = Vec::with_capacity(POSITION_LIMIT);
    positions.extend(0..EDGE_POSITION_LIMIT);
    positions.extend(position_count.saturating_sub(EDGE_POSITION_LIMIT)..position_count);
    positions
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
