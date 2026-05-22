use serde::Serialize;

use crate::{
    tensor::{dot_product, CpuTensor},
    BackendError, Result,
};

use super::{
    kv_cache_head_base_offset, kv_cache_offset, kv_cache_position_stride,
    map_attention_head_to_kv_head, max_abs_index, tensor_window_around_index, GqaHeadMapping,
    LlamaKvCache,
};

const ATTENTION_TRACE_HEAD_LIMIT: usize = 8;
const ATTENTION_TRACE_POSITION_LIMIT: usize = 8;
const ATTENTION_TRACE_VALUE_LIMIT: usize = 10;
const ATTENTION_TRACE_EDGE_POSITION_LIMIT: usize = ATTENTION_TRACE_POSITION_LIMIT / 2;
const ATTENTION_TRACE_TOP_PROBABILITY_LIMIT: usize = 4;

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

pub(super) struct AttentionTraceParams<'a> {
    pub(super) kv_cache: &'a LlamaKvCache,
    pub(super) layer_idx: usize,
    pub(super) query: &'a CpuTensor,
    pub(super) context: &'a CpuTensor,
    pub(super) attention_heads: usize,
    pub(super) repeats: usize,
    pub(super) kv_heads: usize,
    pub(super) head_mapping: GqaHeadMapping,
    pub(super) position_count: usize,
    pub(super) scale: f32,
}

pub(super) fn attention_trace_with_params(
    params: AttentionTraceParams<'_>,
) -> Result<LlamaAttentionTrace> {
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

pub(super) fn sampled_attention_trace_positions(position_count: usize) -> Vec<usize> {
    if position_count <= ATTENTION_TRACE_POSITION_LIMIT {
        return (0..position_count).collect();
    }

    let mut positions = Vec::with_capacity(ATTENTION_TRACE_POSITION_LIMIT);
    positions.extend(0..ATTENTION_TRACE_EDGE_POSITION_LIMIT);
    positions
        .extend(position_count.saturating_sub(ATTENTION_TRACE_EDGE_POSITION_LIMIT)..position_count);
    positions
}

pub(super) fn sampled_attention_trace_heads(
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
