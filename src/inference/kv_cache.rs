use std::env;

use serde::Serialize;

use crate::{
    model::{DenseLlamaDims, LlamaModelConfig},
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
