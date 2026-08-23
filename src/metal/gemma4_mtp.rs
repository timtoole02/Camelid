//! Isolated native Metal runner for the official Gemma 4 26B-A4B MTP assistant.
//!
//! Construction is explicit, validates the exact official artifact, locks its
//! internal-file mapping resident, and runs only under target-runtime callbacks
//! that scope the borrowed target buffers. The established per-draft synchronized
//! path remains the default; the device-fed chain is separately opt-in.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize, MTLStorageMode,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    command_buffer_gpu_times_us, encode_binary, encode_scale_f32, metal_linear_kernel,
    read_buffer_f32, write_buffer_f32, Gemma4MtpTargetEmbeddingFormat,
    Gemma4MtpTargetEmbeddingView, Gemma4MtpTargetKvLayerView, Gemma4MtpTargetKvView,
    MetalLinearKernel,
};
#[cfg(test)]
use super::{encode_rms_norm_f32, encode_rms_norm_per_head};
use crate::{wire_mmap::GgufWireMmap, BackendError, Result};

/// Exact staged artifact used by the isolated experiment. Loading is still
/// explicit; no production path opens this file automatically.
pub const OFFICIAL_STAGED_ASSISTANT_PATH: &str =
    "/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors";

const EXPECTED_FILE_BYTES: u64 = 839_427_840;
const EXPECTED_HEADER_BYTES: usize = 5_360;
const PAYLOAD_FILE_OFFSET: usize = 8 + EXPECTED_HEADER_BYTES;
const EXPECTED_PAYLOAD_BYTES: usize = 839_422_472;
const EXPECTED_SHA256: &str = "c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801";
const EXPECTED_CONFIG_SHA256: &str =
    "23d2bc4a8920f24c23653ff6871437bbd95e52527bf50007aaad05b0b6cab510";

const TARGET_HIDDEN: usize = 2_816;
const ASSISTANT_HIDDEN: usize = 1_024;
const FFN_HIDDEN: usize = 8_192;
const VOCAB: usize = 262_144;
const N_HEADS: usize = 16;
const LOCAL_HEAD_DIM: usize = 256;
const LOCAL_KV_HEADS: usize = 8;
// This is the radius encoded by the official assistant config. Transformers'
// bidirectional sliding-window mask uses an inclusive `distance <= window`
// predicate and then flips the KV axis for the assistant's past-facing view.
// A one-query assistant therefore consumes window + 1 KV rows when available.
const LOCAL_WINDOW: usize = 1_024;
const LOCAL_ATTENTION_SPAN: usize = LOCAL_WINDOW + 1;
const FULL_HEAD_DIM: usize = 512;
const FULL_KV_HEADS: usize = 2;
const MTP_CHAIN_MAX_DRAFTS: usize = 16;
// Official Gemma 4 proportional RoPE keeps the normal split-half geometry for
// the entire 512-wide head, but gives only the first quarter of dimensions a
// non-zero angle: 512 * 0.25 / 2 = 64 active pairs.  In particular, pair d is
// (d, d + 256), not (d, d + 64).  This must match the target layer-29 K cache.
const FULL_ROPE_ACTIVE_PAIRS: usize = FULL_HEAD_DIM / 8;
const RMS_EPS: f32 = 1.0e-6;
const MATRIX_BYTES_PER_PROPOSAL: u64 = 839_385_088;

const MTP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float mtp_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// IEEE round-to-nearest, ties-to-even BF16 store followed by an exact widen
// back to f32. Scratch remains f32 so it can share Camelid's common Metal
// primitives, but every value represents an exact BF16 tensor element.
inline float mtp_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        // Preserve the sign/high payload and force a quiet NaN even when the
        // payload exists only in the discarded low 16 bits.
        uint upper = (bits >> 16) | 0x0040u;
        return as_type<float>(upper << 16);
    }
    uint bias = 0x00007fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + bias) & 0xffff0000u);
}

kernel void mtp_round_bf16_widen_f32(
    device float* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) data[gid] = mtp_round_bf16(data[gid]);
}

// Test diagnostics must snapshot scratch before the next stage reuses it.
// Keeping this as a Metal copy preserves command-buffer ordering and avoids a
// synchronization/readback at every checkpoint.
kernel void mtp_copy_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = input[gid];
}

inline int mtp_q6k_code(device const uchar* block, uint index) {
    const uint half_index = index >> 7;
    const uint position = index & 127u;
    const uint lane = position & 31u;
    const uint ql_base = half_index * 64u;
    const uint qh_base = 128u + half_index * 32u;
    if (position < 32u) {
        return int((block[ql_base + lane] & 0x0fu) |
                   ((block[qh_base + lane] & 3u) << 4u)) - 32;
    }
    if (position < 64u) {
        return int((block[ql_base + 32u + lane] & 0x0fu) |
                   (((block[qh_base + lane] >> 2u) & 3u) << 4u)) - 32;
    }
    if (position < 96u) {
        return int((block[ql_base + lane] >> 4u) |
                   (((block[qh_base + lane] >> 4u) & 3u) << 4u)) - 32;
    }
    return int((block[ql_base + 32u + lane] >> 4u) |
               (((block[qh_base + lane] >> 6u) & 3u) << 4u)) - 32;
}

kernel void mtp_gather_q6k_embed_and_recurrent(
    device const uint* token_buf [[buffer(0)]],
    device const uchar* target_embedding [[buffer(1)]],
    constant ulong& target_embedding_offset [[buffer(2)]],
    device const float* recurrent_hidden [[buffer(3)]],
    device float* pre_input [[buffer(4)]],
    constant uint& target_hidden [[buffer(5)]],
    constant uint& target_vocab [[buffer(6)]],
    constant uint& q6k_superblocks_per_row [[buffer(7)]],
    constant float& embedding_scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < target_hidden) {
        uint token = token_buf[0];
        if (token >= target_vocab) {
            const float invalid = as_type<float>(0x7fc00000u);
            pre_input[gid] = invalid;
            pre_input[target_hidden + gid] = invalid;
            return;
        }
        const ulong row_block = (ulong)token * (ulong)q6k_superblocks_per_row;
        device const uchar* block = target_embedding + target_embedding_offset
            + (row_block + (ulong)(gid >> 8)) * 210ul;
        const uint block_index = gid & 255u;
        const float d = float(*reinterpret_cast<device const half*>(block + 208));
        const int group_scale = int(reinterpret_cast<device const char*>(block + 192)[block_index >> 4]);
        const float decoded = (d * float(group_scale))
            * float(mtp_q6k_code(block, block_index));
        pre_input[gid] = mtp_round_bf16(decoded * embedding_scale);
        pre_input[target_hidden + gid] = mtp_round_bf16(recurrent_hidden[gid]);
    }
}

// Row-major BF16 weights with the pinned ATen AArch64 reduction order. The
// whole safetensors file is bound at offset zero; weight_byte_offset can
// therefore address tensors whose starts are only two-byte aligned. The 32
// residue lanes fold 16/8/4, then finish through vaddvq_f32's adjacent-pair
// horizontal order: (lane0 + lane1) + (lane2 + lane3).
kernel void mtp_bf16_gemv_f32acc(
    device const uchar* file_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& weight_byte_offset [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    device const ushort* weights =
        reinterpret_cast<device const ushort*>(file_bytes + weight_byte_offset);
    const ulong row_base = ulong(row) * ulong(cols);
    float partial = 0.0f;
    const uint cols_vec = (cols / 128) * 128;
    for (uint col = 0; col < cols_vec; col += 128) {
        device const packed_ushort4* w4 =
            reinterpret_cast<device const packed_ushort4*>(weights + row_base + col + lane * 4);
        const ushort4 wv = ushort4(*w4);
        const uint in_base = col + lane * 4;
        partial += mtp_round_bf16(input[in_base]) * mtp_bf16_to_f32(wv.x)
                 + mtp_round_bf16(input[in_base + 1]) * mtp_bf16_to_f32(wv.y)
                 + mtp_round_bf16(input[in_base + 2]) * mtp_bf16_to_f32(wv.z)
                 + mtp_round_bf16(input[in_base + 3]) * mtp_bf16_to_f32(wv.w);
    }
    for (uint col = cols_vec + lane; col < cols; col += 32) {
        const float weight = mtp_bf16_to_f32(weights[row_base + col]);
        partial += mtp_round_bf16(input[col]) * weight;
    }

    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 =
        simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 =
        simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    const float value = pair01 + pair23;
    if (lane == 0) output[row] = mtp_round_bf16(value);
}

kernel void mtp_q4_0_gemv_f32acc(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32;
    device const uchar* row_bytes = q4_weights + ulong(row) * ulong(blocks_per_row) * 18ul;
    float partial = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32) {
        device const uchar* block = row_bytes + ulong(b) * 18ul;
        const float d = float(*reinterpret_cast<device const half*>(block));
        device const packed_uchar4* q4 = reinterpret_cast<device const packed_uchar4*>(block + 2);
        const uint in_base = b * 32;
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            const uchar4 wb = uchar4(q4[k]);
            const float x0 = input[in_base + k * 4];
            const float x1 = input[in_base + k * 4 + 1];
            const float x2 = input[in_base + k * 4 + 2];
            const float x3 = input[in_base + k * 4 + 3];
            const float x16 = input[in_base + 16 + k * 4];
            const float x17 = input[in_base + 16 + k * 4 + 1];
            const float x18 = input[in_base + 16 + k * 4 + 2];
            const float x19 = input[in_base + 16 + k * 4 + 3];
            partial += d * (float(int(wb.x & 0x0f) - 8) * x0 + float(int(wb.x >> 4) - 8) * x16
                          + float(int(wb.y & 0x0f) - 8) * x1 + float(int(wb.y >> 4) - 8) * x17
                          + float(int(wb.z & 0x0f) - 8) * x2 + float(int(wb.z >> 4) - 8) * x18
                          + float(int(wb.w & 0x0f) - 8) * x3 + float(int(wb.w >> 4) - 8) * x19);
        }
    }
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 = simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 = simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    const float value = pair01 + pair23;
    if (lane == 0) output[row] = value;
}

// Gemma 4 uses split-half RoPE. The tables always cover head_dim/2 entries;
// proportional full attention represents inactive pairs as BF16 (1, 0), so
// the partner of d remains d + head_dim/2 even when only 64 pairs rotate.
kernel void mtp_rope_split_bf16_f32(
    device float* data [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& half_head [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = head_count * half_head;
    if (gid >= total) return;
    const uint head = gid / half_head;
    const uint pair = gid - head * half_head;
    const uint dim0 = head * head_dim + pair;
    const uint dim1 = dim0 + half_head;
    const float x0 = mtp_round_bf16(data[dim0]);
    const float x1 = mtp_round_bf16(data[dim1]);
    const float c = mtp_round_bf16(cos_table[pair]);
    const float s = mtp_round_bf16(sin_table[pair]);
    const float first = mtp_round_bf16(x0 * c);
    const float first_cross = mtp_round_bf16((-x1) * s);
    const float second = mtp_round_bf16(x1 * c);
    const float second_cross = mtp_round_bf16(x0 * s);
    data[dim0] = mtp_round_bf16(first + first_cross);
    data[dim1] = mtp_round_bf16(second + second_cross);
}

kernel void mtp_gelu_tanh_mul_bf16_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float x = mtp_round_bf16(gate[gid]);
    const float x3 = x * x * x;
    const float activated = mtp_round_bf16(
        0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x3))));
    output[gid] = mtp_round_bf16(activated * mtp_round_bf16(up[gid]));
}

// Target KV is held in Camelid f32 cache buffers. The official assistant sees
// BF16 shared KV, so K/V are RNE-rounded at each load without allocating a
// second cache. The three explicit phases preserve the official BF16 tensor
// boundaries: BF16 QK scores, f32 softmax -> BF16 probabilities, and BF16
// probability/value matmul output.
inline float4 mtp_load_rounded_bf16x4(
    device const float* values,
    uint base) {
    return float4(
        mtp_round_bf16(values[base + 0]),
        mtp_round_bf16(values[base + 1]),
        mtp_round_bf16(values[base + 2]),
        mtp_round_bf16(values[base + 3]));
}

// Pinned ATen AArch64 BF16 QK dot: eight float4 accumulators, 4/2/1 fold,
// then vaddvq_f32's adjacent-pair horizontal order. One lane still owns each
// position in a 32-position stripe, so only the arithmetic flavor changes.
kernel void mtp_attention_scores_bf16_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& position_stride [[buffer(8)]],
    constant uint& kv_head_stride [[buffer(9)]],
    constant uint& kv_base_offset [[buffer(10)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads || (head_dim & 31u) != 0u) return;
    const uint kv_head = head / group;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    for (uint p = lane; p < position_count; p += 32) {
        const uint k_base = kv_base + p * position_stride;
        float4 acc0 = 0.0f;
        float4 acc1 = 0.0f;
        float4 acc2 = 0.0f;
        float4 acc3 = 0.0f;
        float4 acc4 = 0.0f;
        float4 acc5 = 0.0f;
        float4 acc6 = 0.0f;
        float4 acc7 = 0.0f;
        for (uint d = 0; d < head_dim; d += 32) {
            acc0 += mtp_load_rounded_bf16x4(query, q_base + d + 0) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 0);
            acc1 += mtp_load_rounded_bf16x4(query, q_base + d + 4) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 4);
            acc2 += mtp_load_rounded_bf16x4(query, q_base + d + 8) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 8);
            acc3 += mtp_load_rounded_bf16x4(query, q_base + d + 12) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 12);
            acc4 += mtp_load_rounded_bf16x4(query, q_base + d + 16) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 16);
            acc5 += mtp_load_rounded_bf16x4(query, q_base + d + 20) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 20);
            acc6 += mtp_load_rounded_bf16x4(query, q_base + d + 24) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 24);
            acc7 += mtp_load_rounded_bf16x4(query, q_base + d + 28) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 28);
        }
        acc0 += acc4;
        acc1 += acc5;
        acc2 += acc6;
        acc3 += acc7;
        acc0 += acc2;
        acc1 += acc3;
        acc0 += acc1;
        const float dot = (acc0.x + acc0.y) + (acc0.z + acc0.w);
        scores[score_base + p] = mtp_round_bf16(dot * scale);
    }
}

kernel void mtp_attention_softmax_bf16_f32(
    device float* scores [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads) return;
    const uint score_base = head * position_count;
    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32) {
        local_max = max(local_max, scores[score_base + p]);
    }
    const float max_score = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint p = lane; p < position_count; p += 32) {
        const float value = exp(scores[score_base + p] - max_score);
        scores[score_base + p] = value;
        local_sum += value;
    }
    const float denominator = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint p = lane; p < position_count; p += 32) {
        scores[score_base + p] =
            mtp_round_bf16(scores[score_base + p] / denominator);
    }
}

kernel void mtp_attention_context_bf16_f32(
    device const float* values [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads || position_stride == 0u ||
        compact_base + position_count != physical_logical_k) return;
    const uint kv_head = head / group;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    // ATen vectorizes the full physical K prefix. A compact local suffix must
    // therefore preserve its absolute-position modulo-four phase, and every
    // physical tail term is accumulated into p0.
    const uint physical_vector_end = physical_logical_k & ~3u;
    for (uint d = lane; d < head_dim; d += 32) {
        float p0 = 0.0f;
        float p1 = 0.0f;
        float p2 = 0.0f;
        float p3 = 0.0f;
        for (uint p = 0; p < position_count; ++p) {
            const uint absolute_position = compact_base + p;
            const float product =
                mtp_round_bf16(probabilities[score_base + p]) *
                mtp_round_bf16(values[kv_base + p * position_stride + d]);
            if (absolute_position >= physical_vector_end) {
                p0 += product;
            } else {
                switch (absolute_position & 3u) {
                    case 0u: p0 += product; break;
                    case 1u: p1 += product; break;
                    case 2u: p2 += product; break;
                    default: p3 += product; break;
                }
            }
        }
        const float result = ((p0 + p1) + p2) + p3;
        output[q_base + d] = mtp_round_bf16(result);
    }
}

// Pinned ATen contiguous-f32 RMS sum-of-squares path for the assistant's
// widths 256/512/1024. Sixteen logical residues cover each 256-element slab;
// slab sums and both horizontal folds are explicitly ordered. One threadgroup
// owns each row/head and all 256 threads apply the shared inverse RMS.
kernel void mtp_rms_norm_aten_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& use_weight [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float residue[16];
    threadgroup float inverse_rms;
    const uint base = row * width;

    if (tid < 16u) {
        float row_residue = 0.0f;
        for (uint slab = 0u; slab < width; slab += 256u) {
            float slab_residue = 0.0f;
            for (uint item = 0u; item < 16u; ++item) {
                const float value = input[base + slab + tid + item * 16u];
                const float square = value * value;
                slab_residue = slab_residue + square;
            }
            row_residue = row_residue + slab_residue;
        }
        residue[tid] = row_residue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4u) {
        float partial = residue[tid];
        partial = partial + residue[4u + tid];
        partial = partial + residue[8u + tid];
        partial = partial + residue[12u + tid];
        residue[tid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float sum_squares = residue[0];
        sum_squares = sum_squares + residue[1];
        sum_squares = sum_squares + residue[2];
        sum_squares = sum_squares + residue[3];
        const float mean_squares = sum_squares / float(width);
        const float stabilized = mean_squares + eps;
        inverse_rms = 1.0f / sqrt(stabilized);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < width; index += 256u) {
        float value = input[base + index] * inverse_rms;
        if (use_weight != 0u) value = value * weight[index];
        output[base + index] = value;
    }
}

// One deterministic reduction returns only the tied-head top-1 token. NaNs do
// not win; equal finite logits choose the lower token id.
kernel void mtp_argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* output_id [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint best_id = 0;
    for (uint i = tid; i < count; i += tgsize) {
        const float candidate = logits[i];
        if (!isnan(candidate) &&
            (candidate > best || (candidate == best && i < best_id))) {
            best = candidate;
            best_id = i;
        }
    }
    values[tid] = best;
    indices[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride];
            const uint other_id = indices[tid + stride];
            if (other > values[tid] ||
                (other == values[tid] && other_id < indices[tid])) {
                values[tid] = other;
                indices[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output_id[0] = indices[0];
}
"#;

// Stage-oracle checkpoints deliberately retain only the last `sliding_window`
// columns even though the official assistant computes over the inclusive
// `sliding_window + 1` span. Keep this row-strided diagnostic copy in a
// test-only library so correcting the compute geometry does not alter the
// production Metal library or its selected pipelines.
#[cfg(test)]
const MTP_TEST_DIAGNOSTIC_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float mtp_test_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        uint upper = (bits >> 16) | 0x0040u;
        return as_type<float>(upper << 16);
    }
    uint bias = 0x00007fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + bias) & 0xffff0000u);
}

inline float mtp_test_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// Test-only negative control preserving the former assistant GEMV reduction.
// Production uses the pinned adjacent-final ATen tree; simd_sum remains here
// solely to prove the old 1-ULP discriminator and official legacy hashes.
kernel void mtp_test_bf16_gemv_legacy_f32acc(
    device const uchar* file_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& weight_byte_offset [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    device const ushort* weights =
        reinterpret_cast<device const ushort*>(file_bytes + weight_byte_offset);
    const ulong row_base = ulong(row) * ulong(cols);
    float partial = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const float weight = mtp_test_bf16_to_f32(weights[row_base + col]);
        partial += mtp_test_round_bf16(input[col]) * weight;
    }

    const float value = simd_sum(partial);
    if (lane == 0) output[row] = mtp_test_round_bf16(value);
}

// Test-only negative control preserving the former scalar-sequential QK dot.
kernel void mtp_test_attention_scores_legacy_bf16_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& position_stride [[buffer(8)]],
    constant uint& kv_head_stride [[buffer(9)]],
    constant uint& kv_base_offset [[buffer(10)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads) return;
    const uint kv_head = head / group;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    for (uint p = lane; p < position_count; p += 32) {
        const uint k_base = kv_base + p * position_stride;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += mtp_test_round_bf16(query[q_base + d]) *
                   mtp_test_round_bf16(keys[k_base + d]);
        }
        scores[score_base + p] = mtp_test_round_bf16(dot * scale);
    }
}

// Test-only negative control preserving the former scalar-sequential
// probability @ value accumulation.
kernel void mtp_test_attention_context_legacy_bf16_f32(
    device const float* values [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads || compact_base + position_count > physical_logical_k) return;
    const uint kv_head = head / group;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    for (uint d = lane; d < head_dim; d += 32) {
        float value = 0.0f;
        for (uint p = 0; p < position_count; ++p) {
            value += mtp_test_round_bf16(probabilities[score_base + p]) *
                     mtp_test_round_bf16(values[kv_base + p * position_stride + d]);
        }
        output[q_base + d] = mtp_test_round_bf16(value);
    }
}

// Scalar spelling of the pinned SLEEF ADVSIMD expf_u10 polynomial. Explicit
// fma calls mirror the wheel's FMLA sequence; split exponent scaling mirrors
// SLEEF's vldexp2 and avoids relying on Metal's native exp approximation.
inline float mtp_test_sleef_expf_u10(float d) {
    const int q = int(rint(d * 1.44269504088896340736f));
    float s = fma(float(q), -0.693145751953125f, d);
    s = fma(float(q), -1.428606765330187045e-6f, s);

    float u = 0.000198527617612853646278381f;
    u = fma(u, s, 0.00139304355252534151077271f);
    u = fma(u, s, 0.00833336077630519866943359f);
    u = fma(u, s, 0.0416664853692054748535156f);
    u = fma(u, s, 0.166666671633720397949219f);
    u = fma(u, s, 0.5f);
    u = 1.0f + fma(s * s, u, s);

    const int q_hi = q >> 1;
    const int q_lo = q - q_hi;
    const float scale_hi = as_type<float>(uint(q_hi + 0x7f) << 23);
    const float scale_lo = as_type<float>(uint(q_lo + 0x7f) << 23);
    u = (u * scale_hi) * scale_lo;
    if (d < -104.0f) u = 0.0f;
    if (d > 100.0f) u = INFINITY;
    return u;
}

kernel void mtp_test_copy_row_tail_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& input_columns [[buffer(2)]],
    constant uint& output_columns [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint output_count = row_count * output_columns;
    if (gid >= output_count) return;
    const uint row = gid / output_columns;
    const uint column = gid - row * output_columns;
    const uint first_column = input_columns - output_columns;
    output[gid] = input[row * input_columns + first_column + column];
}

inline float mtp_test_attention_exp(float value, bool use_sleef) {
    return use_sleef ? mtp_test_sleef_expf_u10(value) : exp(value);
}

// Test-only candidate for the pinned ATen CPU float softmax path. One Metal
// thread owns one head, making the Vec4 lane accumulation and ARM horizontal
// tree explicit and repeatable. `use_sleef=false` isolates reduction geometry;
// `true` additionally substitutes the pinned SLEEF exp approximation.
inline void mtp_test_attention_softmax_aten_impl(
    device float* scores,
    uint n_heads,
    uint position_count,
    uint head,
    uint lane,
    bool use_sleef) {
    if (head >= n_heads || lane != 0 || position_count < 4) return;
    const uint score_base = head * position_count;
    const uint vector_count = position_count & ~3u;

    float max0 = scores[score_base + 0];
    float max1 = scores[score_base + 1];
    float max2 = scores[score_base + 2];
    float max3 = scores[score_base + 3];
    for (uint p = 4; p < vector_count; p += 4) {
        max0 = max(max0, scores[score_base + p + 0]);
        max1 = max(max1, scores[score_base + p + 1]);
        max2 = max(max2, scores[score_base + p + 2]);
        max3 = max(max3, scores[score_base + p + 3]);
    }
    if (vector_count + 0 < position_count)
        max0 = max(max0, scores[score_base + vector_count + 0]);
    if (vector_count + 1 < position_count)
        max1 = max(max1, scores[score_base + vector_count + 1]);
    if (vector_count + 2 < position_count)
        max2 = max(max2, scores[score_base + vector_count + 2]);
    const float max02 = max(max0, max2);
    const float max13 = max(max1, max3);
    const float max_score = max(max02, max13);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint p = 0; p < vector_count; p += 4) {
        const float e0 = mtp_test_attention_exp(
            scores[score_base + p + 0] - max_score, use_sleef);
        const float e1 = mtp_test_attention_exp(
            scores[score_base + p + 1] - max_score, use_sleef);
        const float e2 = mtp_test_attention_exp(
            scores[score_base + p + 2] - max_score, use_sleef);
        const float e3 = mtp_test_attention_exp(
            scores[score_base + p + 3] - max_score, use_sleef);
        scores[score_base + p + 0] = e0;
        scores[score_base + p + 1] = e1;
        scores[score_base + p + 2] = e2;
        scores[score_base + p + 3] = e3;
        sum0 += e0;
        sum1 += e1;
        sum2 += e2;
        sum3 += e3;
    }
    if (vector_count + 0 < position_count) {
        const float e = mtp_test_attention_exp(
            scores[score_base + vector_count + 0] - max_score, use_sleef);
        scores[score_base + vector_count + 0] = e;
        sum0 += e;
    }
    if (vector_count + 1 < position_count) {
        const float e = mtp_test_attention_exp(
            scores[score_base + vector_count + 1] - max_score, use_sleef);
        scores[score_base + vector_count + 1] = e;
        sum1 += e;
    }
    if (vector_count + 2 < position_count) {
        const float e = mtp_test_attention_exp(
            scores[score_base + vector_count + 2] - max_score, use_sleef);
        scores[score_base + vector_count + 2] = e;
        sum2 += e;
    }

    const float denominator = (sum0 + sum2) + (sum1 + sum3);
    const float reciprocal = 1.0f / denominator;
    for (uint p = 0; p < position_count; ++p) {
        scores[score_base + p] =
            mtp_test_round_bf16(scores[score_base + p] * reciprocal);
    }
}

// Both wrappers live only in the diagnostic library. Production keeps using
// mtp_attention_softmax_bf16_f32 unless an ignored test explicitly opts in.
kernel void mtp_test_attention_softmax_aten_geometry_f32(
    device float* scores [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    mtp_test_attention_softmax_aten_impl(
        scores, n_heads, position_count, head, lane, false);
}

kernel void mtp_test_attention_softmax_aten_sleef_f32(
    device float* scores [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    mtp_test_attention_softmax_aten_impl(
        scores, n_heads, position_count, head, lane, true);
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedTensor {
    name: &'static str,
    shape: &'static [u64],
    start: u64,
    end: u64,
}

macro_rules! tensor {
    ($name:literal, [$($dim:expr),+], $start:expr, $end:expr) => {
        ExpectedTensor {
            name: $name,
            shape: &[$($dim),+],
            start: $start,
            end: $end,
        }
    };
}

const EXPECTED_TENSORS: &[ExpectedTensor] = &[
    tensor!(
        "model.embed_tokens.weight",
        [262_144, 1_024],
        0,
        536_870_912
    ),
    tensor!(
        "model.layers.0.input_layernorm.weight",
        [1_024],
        536_870_912,
        536_872_960
    ),
    tensor!("model.layers.0.layer_scalar", [1], 536_872_960, 536_872_962),
    tensor!(
        "model.layers.0.mlp.down_proj.weight",
        [1_024, 8_192],
        536_872_962,
        553_650_178
    ),
    tensor!(
        "model.layers.0.mlp.gate_proj.weight",
        [8_192, 1_024],
        553_650_178,
        570_427_394
    ),
    tensor!(
        "model.layers.0.mlp.up_proj.weight",
        [8_192, 1_024],
        570_427_394,
        587_204_610
    ),
    tensor!(
        "model.layers.0.post_attention_layernorm.weight",
        [1_024],
        587_204_610,
        587_206_658
    ),
    tensor!(
        "model.layers.0.post_feedforward_layernorm.weight",
        [1_024],
        587_206_658,
        587_208_706
    ),
    tensor!(
        "model.layers.0.pre_feedforward_layernorm.weight",
        [1_024],
        587_208_706,
        587_210_754
    ),
    tensor!(
        "model.layers.0.self_attn.o_proj.weight",
        [1_024, 4_096],
        587_210_754,
        595_599_362
    ),
    tensor!(
        "model.layers.0.self_attn.q_norm.weight",
        [256],
        595_599_362,
        595_599_874
    ),
    tensor!(
        "model.layers.0.self_attn.q_proj.weight",
        [4_096, 1_024],
        595_599_874,
        603_988_482
    ),
    tensor!(
        "model.layers.1.input_layernorm.weight",
        [1_024],
        603_988_482,
        603_990_530
    ),
    tensor!("model.layers.1.layer_scalar", [1], 603_990_530, 603_990_532),
    tensor!(
        "model.layers.1.mlp.down_proj.weight",
        [1_024, 8_192],
        603_990_532,
        620_767_748
    ),
    tensor!(
        "model.layers.1.mlp.gate_proj.weight",
        [8_192, 1_024],
        620_767_748,
        637_544_964
    ),
    tensor!(
        "model.layers.1.mlp.up_proj.weight",
        [8_192, 1_024],
        637_544_964,
        654_322_180
    ),
    tensor!(
        "model.layers.1.post_attention_layernorm.weight",
        [1_024],
        654_322_180,
        654_324_228
    ),
    tensor!(
        "model.layers.1.post_feedforward_layernorm.weight",
        [1_024],
        654_324_228,
        654_326_276
    ),
    tensor!(
        "model.layers.1.pre_feedforward_layernorm.weight",
        [1_024],
        654_326_276,
        654_328_324
    ),
    tensor!(
        "model.layers.1.self_attn.o_proj.weight",
        [1_024, 4_096],
        654_328_324,
        662_716_932
    ),
    tensor!(
        "model.layers.1.self_attn.q_norm.weight",
        [256],
        662_716_932,
        662_717_444
    ),
    tensor!(
        "model.layers.1.self_attn.q_proj.weight",
        [4_096, 1_024],
        662_717_444,
        671_106_052
    ),
    tensor!(
        "model.layers.2.input_layernorm.weight",
        [1_024],
        671_106_052,
        671_108_100
    ),
    tensor!("model.layers.2.layer_scalar", [1], 671_108_100, 671_108_102),
    tensor!(
        "model.layers.2.mlp.down_proj.weight",
        [1_024, 8_192],
        671_108_102,
        687_885_318
    ),
    tensor!(
        "model.layers.2.mlp.gate_proj.weight",
        [8_192, 1_024],
        687_885_318,
        704_662_534
    ),
    tensor!(
        "model.layers.2.mlp.up_proj.weight",
        [8_192, 1_024],
        704_662_534,
        721_439_750
    ),
    tensor!(
        "model.layers.2.post_attention_layernorm.weight",
        [1_024],
        721_439_750,
        721_441_798
    ),
    tensor!(
        "model.layers.2.post_feedforward_layernorm.weight",
        [1_024],
        721_441_798,
        721_443_846
    ),
    tensor!(
        "model.layers.2.pre_feedforward_layernorm.weight",
        [1_024],
        721_443_846,
        721_445_894
    ),
    tensor!(
        "model.layers.2.self_attn.o_proj.weight",
        [1_024, 4_096],
        721_445_894,
        729_834_502
    ),
    tensor!(
        "model.layers.2.self_attn.q_norm.weight",
        [256],
        729_834_502,
        729_835_014
    ),
    tensor!(
        "model.layers.2.self_attn.q_proj.weight",
        [4_096, 1_024],
        729_835_014,
        738_223_622
    ),
    tensor!(
        "model.layers.3.input_layernorm.weight",
        [1_024],
        738_223_622,
        738_225_670
    ),
    tensor!("model.layers.3.layer_scalar", [1], 738_225_670, 738_225_672),
    tensor!(
        "model.layers.3.mlp.down_proj.weight",
        [1_024, 8_192],
        738_225_672,
        755_002_888
    ),
    tensor!(
        "model.layers.3.mlp.gate_proj.weight",
        [8_192, 1_024],
        755_002_888,
        771_780_104
    ),
    tensor!(
        "model.layers.3.mlp.up_proj.weight",
        [8_192, 1_024],
        771_780_104,
        788_557_320
    ),
    tensor!(
        "model.layers.3.post_attention_layernorm.weight",
        [1_024],
        788_557_320,
        788_559_368
    ),
    tensor!(
        "model.layers.3.post_feedforward_layernorm.weight",
        [1_024],
        788_559_368,
        788_561_416
    ),
    tensor!(
        "model.layers.3.pre_feedforward_layernorm.weight",
        [1_024],
        788_561_416,
        788_563_464
    ),
    tensor!(
        "model.layers.3.self_attn.o_proj.weight",
        [1_024, 8_192],
        788_563_464,
        805_340_680
    ),
    tensor!(
        "model.layers.3.self_attn.q_norm.weight",
        [512],
        805_340_680,
        805_341_704
    ),
    tensor!(
        "model.layers.3.self_attn.q_proj.weight",
        [8_192, 1_024],
        805_341_704,
        822_118_920
    ),
    tensor!("model.norm.weight", [1_024], 822_118_920, 822_120_968),
    tensor!(
        "post_projection.weight",
        [2_816, 1_024],
        822_120_968,
        827_888_136
    ),
    tensor!(
        "pre_projection.weight",
        [1_024, 5_632],
        827_888_136,
        839_422_472
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorRef {
    absolute_offset: u32,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Debug)]
struct TensorEntry {
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct AssistantManifest {
    tensors: BTreeMap<String, TensorEntry>,
}

#[derive(Deserialize)]
struct SafetensorsEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl AssistantManifest {
    fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors.get(name).ok_or_else(|| {
            BackendError::TensorNotFound(format!("official Gemma 4 MTP tensor {name}"))
        })
    }

    fn matrix(&self, name: &str) -> Result<TensorRef> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 2 {
            return Err(BackendError::InvalidTensorData(format!(
                "MTP matrix {name} has rank {}, expected 2",
                tensor.shape.len()
            )));
        }
        Ok(TensorRef {
            absolute_offset: u32::try_from(PAYLOAD_FILE_OFFSET + tensor.start).map_err(|_| {
                BackendError::InvalidTensorData(format!("MTP tensor {name} offset exceeds u32"))
            })?,
            rows: tensor.shape[0] as u32,
            cols: tensor.shape[1] as u32,
        })
    }
}

fn invalid(detail: impl Into<String>) -> BackendError {
    BackendError::InvalidTensorData(format!("Gemma 4 MTP assistant: {}", detail.into()))
}

fn parse_device_chain_opt_in(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some(value) if value.eq_ignore_ascii_case("false") => Ok(false),
        Some("1") => Ok(true),
        Some(value) if value.eq_ignore_ascii_case("true") => Ok(true),
        Some(_) => Err("expected 0, 1, false, or true"),
    }
}

/// Whether the experimental one-wait assistant chain was explicitly selected.
/// Missing means false; malformed/non-Unicode values fail closed instead of
/// accidentally changing the production drafting path.
pub fn device_chain_requested_from_environment() -> Result<bool> {
    const NAME: &str = "CAMELID_GEMMA4_MTP_DEVICE_CHAIN";
    match std::env::var(NAME) {
        Ok(value) => parse_device_chain_opt_in(Some(&value))
            .map_err(|detail| invalid(format!("{NAME} {detail}, got {value:?}"))),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(invalid(format!("{NAME} must contain Unicode text")))
        }
    }
}

fn parse_and_validate_header(header: &[u8], payload_bytes: usize) -> Result<AssistantManifest> {
    let root: serde_json::Value = serde_json::from_slice(header)
        .map_err(|error| invalid(format!("invalid safetensors header JSON: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| invalid("safetensors header root is not an object"))?;
    let metadata = object
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid("missing safetensors __metadata__"))?;
    if metadata.len() != 1
        || metadata.get("format").and_then(serde_json::Value::as_str) != Some("pt")
    {
        return Err(invalid("safetensors metadata is not exactly format=pt"));
    }

    if object.len() != EXPECTED_TENSORS.len() + 1 {
        return Err(invalid(format!(
            "tensor count {} does not match expected {}",
            object.len().saturating_sub(1),
            EXPECTED_TENSORS.len()
        )));
    }

    let mut tensors = BTreeMap::new();
    for expected in EXPECTED_TENSORS {
        let value = object
            .get(expected.name)
            .ok_or_else(|| invalid(format!("missing tensor {}", expected.name)))?;
        let actual: SafetensorsEntry = serde_json::from_value(value.clone()).map_err(|error| {
            invalid(format!("invalid descriptor for {}: {error}", expected.name))
        })?;
        if actual.dtype != "BF16"
            || actual.shape.as_slice() != expected.shape
            || actual.data_offsets != [expected.start, expected.end]
        {
            return Err(invalid(format!(
                "tensor {} mismatch: dtype={} shape={:?} offsets={:?}",
                expected.name, actual.dtype, actual.shape, actual.data_offsets
            )));
        }
        let expected_len = expected
            .shape
            .iter()
            .try_fold(2u64, |bytes, dim| bytes.checked_mul(*dim))
            .ok_or_else(|| invalid(format!("tensor {} byte size overflow", expected.name)))?;
        if expected.end - expected.start != expected_len {
            return Err(invalid(format!(
                "internal manifest byte size mismatch for {}",
                expected.name
            )));
        }
        let start = usize::try_from(expected.start)
            .map_err(|_| invalid(format!("tensor {} start exceeds usize", expected.name)))?;
        let end = usize::try_from(expected.end)
            .map_err(|_| invalid(format!("tensor {} end exceeds usize", expected.name)))?;
        tensors.insert(
            expected.name.to_string(),
            TensorEntry {
                shape: actual.shape,
                start,
                end,
            },
        );
    }

    if EXPECTED_TENSORS.first().map(|entry| entry.start) != Some(0)
        || EXPECTED_TENSORS.last().map(|entry| entry.end) != Some(payload_bytes as u64)
        || EXPECTED_TENSORS
            .windows(2)
            .any(|pair| pair[0].end != pair[1].start)
    {
        return Err(invalid(
            "internal tensor manifest is not a contiguous full payload",
        ));
    }
    Ok(AssistantManifest { tensors })
}

fn parse_official_manifest(mapping: &GgufWireMmap) -> Result<AssistantManifest> {
    if mapping.file_len() != EXPECTED_FILE_BYTES {
        return Err(invalid(format!(
            "file size {} does not match expected {EXPECTED_FILE_BYTES}",
            mapping.file_len()
        )));
    }
    let prefix = mapping.bytes(0, 8)?;
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(prefix);
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes))
        .map_err(|_| invalid("header length exceeds usize"))?;
    if header_len != EXPECTED_HEADER_BYTES {
        return Err(invalid(format!(
            "header length {header_len} does not match expected {EXPECTED_HEADER_BYTES}"
        )));
    }
    let header = mapping.bytes(8, header_len)?;
    parse_and_validate_header(header, EXPECTED_PAYLOAD_BYTES)
}

fn validate_official_config(weight_path: &Path) -> Result<()> {
    let config_path = weight_path
        .parent()
        .ok_or_else(|| invalid("weight path has no parent directory"))?
        .join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|source| BackendError::Io {
        path: config_path.clone(),
        source,
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(format!("invalid assistant config.json: {error}")))?;
    let text = value
        .get("text_config")
        .ok_or_else(|| invalid("assistant config lacks text_config"))?;
    let exact = [
        (
            "backbone_hidden_size",
            value.get("backbone_hidden_size"),
            2_816,
        ),
        ("hidden_size", text.get("hidden_size"), 1_024),
        ("intermediate_size", text.get("intermediate_size"), 8_192),
        ("num_hidden_layers", text.get("num_hidden_layers"), 4),
        ("num_attention_heads", text.get("num_attention_heads"), 16),
        ("num_key_value_heads", text.get("num_key_value_heads"), 8),
        (
            "num_global_key_value_heads",
            text.get("num_global_key_value_heads"),
            2,
        ),
        ("head_dim", text.get("head_dim"), 256),
        ("global_head_dim", text.get("global_head_dim"), 512),
        ("num_kv_shared_layers", text.get("num_kv_shared_layers"), 4),
        ("sliding_window", text.get("sliding_window"), 1_024),
        ("vocab_size", text.get("vocab_size"), 262_144),
    ];
    for (name, actual, expected) in exact {
        if actual.and_then(serde_json::Value::as_u64) != Some(expected) {
            return Err(invalid(format!(
                "assistant config {name} is {:?}, expected {expected}",
                actual
            )));
        }
    }
    if value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|v| v.first())
        .and_then(serde_json::Value::as_str)
        != Some("Gemma4AssistantForCausalLM")
        || value.get("dtype").and_then(serde_json::Value::as_str) != Some("bfloat16")
        || value
            .get("use_ordered_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || text
            .get("attention_k_eq_v")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || text
            .get("hidden_activation")
            .and_then(serde_json::Value::as_str)
            != Some("gelu_pytorch_tanh")
        || text
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        return Err(invalid(
            "assistant config semantic flags do not match the official 26B-A4B assistant",
        ));
    }
    let layer_types = text
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("assistant config lacks layer_types"))?;
    let actual_layers: Vec<&str> = layer_types
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    if actual_layers
        != [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
    {
        return Err(invalid(format!(
            "assistant layer_types mismatch: {actual_layers:?}"
        )));
    }
    let eps = text
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| invalid("assistant config lacks rms_norm_eps"))?;
    if eps.to_bits() != 1.0e-6f64.to_bits() {
        return Err(invalid(format!(
            "assistant rms_norm_eps {eps} does not equal {RMS_EPS}"
        )));
    }
    let rope = text
        .get("rope_parameters")
        .ok_or_else(|| invalid("assistant config lacks rope_parameters"))?;
    let local_rope = rope
        .get("sliding_attention")
        .ok_or_else(|| invalid("assistant config lacks sliding RoPE parameters"))?;
    let full_rope = rope
        .get("full_attention")
        .ok_or_else(|| invalid("assistant config lacks full RoPE parameters"))?;
    if local_rope
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        != Some(10_000.0)
        || local_rope
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            != Some("default")
        || full_rope
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            != Some(1_000_000.0)
        || full_rope
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            != Some("proportional")
        || full_rope
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            != Some(0.25)
        || !text
            .get("final_logit_softcapping")
            .is_some_and(serde_json::Value::is_null)
    {
        return Err(invalid(
            "assistant RoPE or final-logit configuration does not match the official artifact",
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[derive(Debug)]
struct LockedAssistantMapping {
    mapping: Arc<GgufWireMmap>,
    locked_bytes: usize,
    resident_pages: usize,
    total_pages: usize,
}

impl Drop for LockedAssistantMapping {
    fn drop(&mut self) {
        // SAFETY: the pointer and length are the live mapping locked by
        // `lock_fully_resident`; unlocking does not invalidate the mapping.
        unsafe {
            libc::munlock(
                self.mapping.base_ptr().cast_mut().cast::<c_void>(),
                self.locked_bytes,
            );
        }
    }
}

fn lock_fully_resident(mapping: Arc<GgufWireMmap>) -> Result<LockedAssistantMapping> {
    let locked_bytes = mapping.mapped_len();
    // SAFETY: mapping covers this exact page-aligned range. `mlock` is the
    // experiment's fail-closed resident-RAM contract.
    let lock_result =
        unsafe { libc::mlock(mapping.base_ptr().cast_mut().cast::<c_void>(), locked_bytes) };
    if lock_result != 0 {
        return Err(invalid(format!(
            "mlock of {locked_bytes} bytes failed: {}",
            std::io::Error::last_os_error()
        )));
    }

    let page = crate::wire_mmap::page_size();
    let total_pages = locked_bytes.div_ceil(page);
    let mut status = vec![0u8; total_pages];
    // SAFETY: `status` has one byte per page and the queried range is the live
    // mapping. macOS/POSIX reports residency in bit zero of each byte.
    let mincore_result = unsafe {
        libc::mincore(
            mapping.base_ptr().cast_mut().cast::<c_void>(),
            locked_bytes,
            status.as_mut_ptr().cast::<libc::c_char>(),
        )
    };
    if mincore_result != 0 {
        // SAFETY: paired with the successful mlock above.
        unsafe {
            libc::munlock(mapping.base_ptr().cast_mut().cast::<c_void>(), locked_bytes);
        }
        return Err(invalid(format!(
            "mincore verification failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let resident_pages = status.iter().filter(|entry| **entry & 1 != 0).count();
    if resident_pages != total_pages {
        // SAFETY: paired with the successful mlock above.
        unsafe {
            libc::munlock(mapping.base_ptr().cast_mut().cast::<c_void>(), locked_bytes);
        }
        return Err(invalid(format!(
            "only {resident_pages}/{total_pages} assistant pages are resident after mlock"
        )));
    }
    Ok(LockedAssistantMapping {
        mapping,
        locked_bytes,
        resident_pages,
        total_pages,
    })
}

struct MtpPipelines {
    bf16_gemv: ComputePipelineState,
    #[cfg(test)]
    bf16_gemv_legacy: ComputePipelineState,
    round_bf16: ComputePipelineState,
    copy_f32: ComputePipelineState,
    #[cfg(test)]
    copy_row_tail_f32: ComputePipelineState,
    #[cfg(test)]
    attention_softmax_aten_geometry_f32: ComputePipelineState,
    #[cfg(test)]
    attention_softmax_aten_sleef_f32: ComputePipelineState,
    rope_bf16: ComputePipelineState,
    gelu_mul_bf16: ComputePipelineState,
    attention_scores_bf16: ComputePipelineState,
    #[cfg(test)]
    attention_scores_legacy_bf16: ComputePipelineState,
    attention_softmax_bf16: ComputePipelineState,
    attention_context_bf16: ComputePipelineState,
    #[cfg(test)]
    attention_context_legacy_bf16: ComputePipelineState,
    rms_norm_aten_f32: ComputePipelineState,
    argmax: ComputePipelineState,
    gather_q6k_embed_and_recurrent: ComputePipelineState,
    q4_0_gemv: ComputePipelineState,
}

impl MtpPipelines {
    fn new(device: &Device) -> Result<Self> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(MTP_SHADER, &options)
            .map_err(|error| invalid(format!("Metal shader compilation failed: {error}")))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|error| invalid(format!("Metal function {name} missing: {error}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| invalid(format!("Metal pipeline {name} failed: {error}")))
        };
        #[cfg(test)]
        let test_diagnostic_library = device
            .new_library_with_source(MTP_TEST_DIAGNOSTIC_SHADER, &options)
            .map_err(|error| {
                invalid(format!(
                    "MTP test-diagnostic Metal shader compilation failed: {error}"
                ))
            })?;
        #[cfg(test)]
        let test_diagnostic_pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = test_diagnostic_library
                .get_function(name, None)
                .map_err(|error| {
                    invalid(format!(
                        "MTP test-diagnostic Metal function {name} missing: {error}"
                    ))
                })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    invalid(format!(
                        "MTP test-diagnostic Metal pipeline {name} failed: {error}"
                    ))
                })
        };
        Ok(Self {
            bf16_gemv: pipeline("mtp_bf16_gemv_f32acc")?,
            #[cfg(test)]
            bf16_gemv_legacy: test_diagnostic_pipeline("mtp_test_bf16_gemv_legacy_f32acc")?,
            round_bf16: pipeline("mtp_round_bf16_widen_f32")?,
            copy_f32: pipeline("mtp_copy_f32")?,
            #[cfg(test)]
            copy_row_tail_f32: test_diagnostic_pipeline("mtp_test_copy_row_tail_f32")?,
            #[cfg(test)]
            attention_softmax_aten_geometry_f32: test_diagnostic_pipeline(
                "mtp_test_attention_softmax_aten_geometry_f32",
            )?,
            #[cfg(test)]
            attention_softmax_aten_sleef_f32: test_diagnostic_pipeline(
                "mtp_test_attention_softmax_aten_sleef_f32",
            )?,
            rope_bf16: pipeline("mtp_rope_split_bf16_f32")?,
            gelu_mul_bf16: pipeline("mtp_gelu_tanh_mul_bf16_f32")?,
            attention_scores_bf16: pipeline("mtp_attention_scores_bf16_f32")?,
            #[cfg(test)]
            attention_scores_legacy_bf16: test_diagnostic_pipeline(
                "mtp_test_attention_scores_legacy_bf16_f32",
            )?,
            attention_softmax_bf16: pipeline("mtp_attention_softmax_bf16_f32")?,
            attention_context_bf16: pipeline("mtp_attention_context_bf16_f32")?,
            #[cfg(test)]
            attention_context_legacy_bf16: test_diagnostic_pipeline(
                "mtp_test_attention_context_legacy_bf16_f32",
            )?,
            rms_norm_aten_f32: pipeline("mtp_rms_norm_aten_f32")?,
            argmax: pipeline("mtp_argmax_f32")?,
            gather_q6k_embed_and_recurrent: pipeline("mtp_gather_q6k_embed_and_recurrent")?,
            q4_0_gemv: pipeline("mtp_q4_0_gemv_f32acc")?,
        })
    }

    fn selected_bf16_gemv(&self) -> &ComputePipelineState {
        &self.bf16_gemv
    }

    fn selected_attention_scores_bf16(&self) -> &ComputePipelineState {
        &self.attention_scores_bf16
    }

    fn selected_attention_context_bf16(&self) -> &ComputePipelineState {
        &self.attention_context_bf16
    }
}

struct LayerWeights {
    input_norm: Buffer,
    post_attention_norm: Buffer,
    pre_feedforward_norm: Buffer,
    post_feedforward_norm: Buffer,
    q_norm: Buffer,
    q: TensorRef,
    o: TensorRef,
    gate: TensorRef,
    up: TensorRef,
    down: TensorRef,
    scale_scalar: Buffer,
}

struct MtpScratch {
    pre_input: Buffer,
    hidden: Buffer,
    normed: Buffer,
    query: Buffer,
    query_normed: Buffer,
    context: Buffer,
    attention_projection: Buffer,
    attention_normalized: Buffer,
    attention_residual: Buffer,
    gate: Buffer,
    up: Buffer,
    gated: Buffer,
    down: Buffer,
    down_normalized: Buffer,
    next_hidden: Buffer,
    final_normalized: Buffer,
    recurrent_hidden: Buffer,
    chain_recurrent_hidden: Buffer,
    logits: Buffer,
    output_token: Buffer,
    hidden_rms_scalar: Buffer,
    hidden_count: Buffer,
    ffn_count: Buffer,
    local_qnorm_scalar: Buffer,
    full_qnorm_scalar: Buffer,
    local_rope_scalar: Buffer,
    full_rope_scalar: Buffer,
    local_cos: Buffer,
    local_sin: Buffer,
    full_cos: Buffer,
    full_sin: Buffer,
    local_attention_scalar: Buffer,
    full_attention_scalar: Buffer,
    attention_denom: Buffer,
    attention_blocks: Buffer,
}

/// Resident-memory accounting captured after the artifact is admitted.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4MtpResidentLedger {
    pub file_bytes: u64,
    pub mapped_bytes: u64,
    pub locked_bytes: u64,
    pub resident_pages: u64,
    pub total_pages: u64,
    pub payload_bytes: u64,
    pub decoded_norm_bytes: u64,
    pub fixed_scratch_bytes: u64,
    pub hash_us: u128,
    pub lock_and_residency_us: u128,
    pub pipeline_compile_us: u128,
    pub load_wall_us: u128,
}

/// Per-proposal byte accounting. `target_kv_read_bytes` is the exact logical K
/// plus V span traversed by the three sliding and one full attention layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4MtpProposalLedger {
    pub assistant_matrix_bytes: u64,
    /// Unique physical capacity of the borrowed target layer-28/layer-29 K/V
    /// buffers. Unlike `target_kv_read_bytes`, this counts each shared buffer
    /// once and uses its allocated stride rather than the logical prefix.
    pub borrowed_target_kv_capacity_bytes: u64,
    pub target_kv_read_bytes: u64,
    pub dynamic_attention_scratch_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4MtpProposalTiming {
    pub encode_us: u128,
    pub wait_us: u128,
    pub wall_us: u128,
    pub gpu_us: u128,
    pub kernel_us: u128,
}

#[derive(Clone, Debug)]
pub struct Gemma4MtpProposal {
    pub token: u32,
    pub recurrent_hidden: Vec<f32>,
    pub timing: Gemma4MtpProposalTiming,
    pub ledger: Gemma4MtpProposalLedger,
    #[cfg(test)]
    stage_snapshots: Vec<MtpStageSnapshot>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct MtpStageSnapshot {
    name: String,
    values: Vec<f32>,
    bf16_sha256: String,
}

#[cfg(test)]
struct PendingMtpStageSnapshot {
    name: String,
    buffer: Buffer,
    count: usize,
}

/// Explicit, default-off native assistant. The locked file mapping owns the
/// physical BF16 pages and the Metal buffer points directly at those pages.
pub struct Gemma4MtpAssistantMetal {
    weight_file: Buffer,
    pipelines: MtpPipelines,
    layers: Vec<LayerWeights>,
    final_norm: Buffer,
    embedding: TensorRef,
    q4_embedding: Option<Buffer>,
    pre_projection: TensorRef,
    post_projection: TensorRef,
    scratch: MtpScratch,
    queue: metal::CommandQueue,
    resident_ledger: Gemma4MtpResidentLedger,
    last_proposal_ledger: Option<Gemma4MtpProposalLedger>,
    source_path: PathBuf,
    // Must remain last: Rust drops struct fields in declaration order, so all
    // no-copy Metal buffers are released before this unlocks/unmaps the pages.
    _locked_mapping: LockedAssistantMapping,
}

fn shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16_rne_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let mut half_exp = (unbiased + 15) as u32;
        let mut half_mant = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) != 0) {
            half_mant += 1;
            if half_mant >= 0x400 {
                half_mant = 0;
                half_exp += 1;
                if half_exp > 30 {
                    return sign | 0x7c00;
                }
            }
        }
        sign | ((half_exp as u16) << 10) | (half_mant as u16)
    } else if unbiased >= -24 {
        let shift = (-14 - unbiased) as u32 + 13;
        let full_mant = mant | 0x0080_0000;
        let mut half_mant = full_mant >> shift;
        let rem_mask = (1 << shift) - 1;
        let half_bit = 1 << (shift - 1);
        let rem = full_mant & rem_mask;
        if rem > half_bit || (rem == half_bit && (half_mant & 1) != 0) {
            half_mant += 1;
        }
        sign | (half_mant as u16)
    } else {
        sign
    }
}

fn round_to_bf16_f32(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_rne_bits(value))
}

fn decode_bf16(mapping: &GgufWireMmap, tensor: &TensorEntry) -> Result<Vec<f32>> {
    let bytes = mapping.bytes(
        (PAYLOAD_FILE_OFFSET + tensor.start) as u64,
        tensor.end - tensor.start,
    )?;
    if !bytes.len().is_multiple_of(2) {
        return Err(invalid("BF16 tensor has an odd byte count"));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn f32_buffer(device: &Device, values: &[f32]) -> Buffer {
    let buffer = shared_buffer(device, std::mem::size_of_val(values));
    write_buffer_f32(&buffer, values);
    buffer
}

fn set_rms_scalar(buffer: &Buffer, width: usize) {
    unsafe {
        let ptr = buffer.contents().cast::<u8>();
        *ptr.cast::<u32>() = width as u32;
        *ptr.add(4).cast::<f32>() = RMS_EPS;
    }
}

fn set_count(buffer: &Buffer, count: usize) {
    unsafe {
        *buffer.contents().cast::<u32>() = count as u32;
    }
}

fn set_qnorm_scalar(buffer: &Buffer, head_dim: usize) {
    unsafe {
        let ptr = buffer.contents().cast::<u8>();
        *ptr.cast::<u32>() = head_dim as u32;
        *ptr.add(4).cast::<f32>() = RMS_EPS;
        *ptr.add(8).cast::<u32>() = 1;
    }
}

fn set_rope_scalar(buffer: &Buffer, head_dim: usize, half_rope: usize) {
    unsafe {
        let ptr = buffer.contents().cast::<u32>();
        *ptr = N_HEADS as u32;
        *ptr.add(1) = head_dim as u32;
        *ptr.add(2) = half_rope as u32;
        *ptr.add(3) = 1; // split-half pairing
    }
}

impl MtpScratch {
    fn new(device: &Device) -> Self {
        let f32s = |count: usize| shared_buffer(device, count * std::mem::size_of::<f32>());
        let scratch = Self {
            pre_input: f32s(TARGET_HIDDEN * 2),
            hidden: f32s(ASSISTANT_HIDDEN),
            normed: f32s(ASSISTANT_HIDDEN),
            query: f32s(N_HEADS * FULL_HEAD_DIM),
            query_normed: f32s(N_HEADS * FULL_HEAD_DIM),
            context: f32s(N_HEADS * FULL_HEAD_DIM),
            attention_projection: f32s(ASSISTANT_HIDDEN),
            attention_normalized: f32s(ASSISTANT_HIDDEN),
            attention_residual: f32s(ASSISTANT_HIDDEN),
            gate: f32s(FFN_HIDDEN),
            up: f32s(FFN_HIDDEN),
            gated: f32s(FFN_HIDDEN),
            down: f32s(ASSISTANT_HIDDEN),
            down_normalized: f32s(ASSISTANT_HIDDEN),
            next_hidden: f32s(ASSISTANT_HIDDEN),
            final_normalized: f32s(ASSISTANT_HIDDEN),
            recurrent_hidden: f32s(TARGET_HIDDEN),
            chain_recurrent_hidden: f32s(MTP_CHAIN_MAX_DRAFTS * TARGET_HIDDEN),
            logits: f32s(VOCAB),
            output_token: shared_buffer(
                device,
                MTP_CHAIN_MAX_DRAFTS * std::mem::size_of::<u32>(),
            ),
            hidden_rms_scalar: shared_buffer(device, 8),
            hidden_count: shared_buffer(device, 4),
            ffn_count: shared_buffer(device, 4),
            local_qnorm_scalar: shared_buffer(device, 12),
            full_qnorm_scalar: shared_buffer(device, 12),
            local_rope_scalar: shared_buffer(device, 16),
            full_rope_scalar: shared_buffer(device, 16),
            local_cos: f32s(LOCAL_HEAD_DIM / 2),
            local_sin: f32s(LOCAL_HEAD_DIM / 2),
            full_cos: f32s(FULL_HEAD_DIM / 2),
            full_sin: f32s(FULL_HEAD_DIM / 2),
            // Ten u32/f32 words: common score geometry followed by explicit
            // compact-base and physical-K values for exact context reduction.
            local_attention_scalar: shared_buffer(device, 40),
            full_attention_scalar: shared_buffer(device, 40),
            attention_denom: f32s(N_HEADS),
            attention_blocks: shared_buffer(device, 8),
        };
        set_rms_scalar(&scratch.hidden_rms_scalar, ASSISTANT_HIDDEN);
        set_count(&scratch.hidden_count, ASSISTANT_HIDDEN);
        set_count(&scratch.ffn_count, FFN_HIDDEN);
        set_qnorm_scalar(&scratch.local_qnorm_scalar, LOCAL_HEAD_DIM);
        set_qnorm_scalar(&scratch.full_qnorm_scalar, FULL_HEAD_DIM);
        set_rope_scalar(
            &scratch.local_rope_scalar,
            LOCAL_HEAD_DIM,
            LOCAL_HEAD_DIM / 2,
        );
        set_rope_scalar(&scratch.full_rope_scalar, FULL_HEAD_DIM, FULL_HEAD_DIM / 2);
        scratch
    }

    fn byte_len(&self) -> u64 {
        [
            &self.pre_input,
            &self.hidden,
            &self.normed,
            &self.query,
            &self.query_normed,
            &self.context,
            &self.attention_projection,
            &self.attention_normalized,
            &self.attention_residual,
            &self.gate,
            &self.up,
            &self.gated,
            &self.down,
            &self.down_normalized,
            &self.next_hidden,
            &self.final_normalized,
            &self.recurrent_hidden,
            &self.chain_recurrent_hidden,
            &self.logits,
            &self.output_token,
            &self.hidden_rms_scalar,
            &self.hidden_count,
            &self.ffn_count,
            &self.local_qnorm_scalar,
            &self.full_qnorm_scalar,
            &self.local_rope_scalar,
            &self.full_rope_scalar,
            &self.local_cos,
            &self.local_sin,
            &self.full_cos,
            &self.full_sin,
            &self.local_attention_scalar,
            &self.full_attention_scalar,
            &self.attention_denom,
            &self.attention_blocks,
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }
}

impl Gemma4MtpAssistantMetal {
    /// Load and hard-pin the exact staged official artifact. This is never
    /// called by the production drafter.
    pub fn load_staged_official() -> Result<Self> {
        Self::load(Path::new(OFFICIAL_STAGED_ASSISTANT_PATH))
    }

    /// Load an exact byte-identical copy of the official artifact. Shape,
    /// offsets, config, file length and SHA-256 are all pinned before mlock.
    pub fn load(path: &Path) -> Result<Self> {
        let load_started = Instant::now();
        validate_official_config(path)?;
        let mapping = GgufWireMmap::map(path)?;
        mapping.advise_sequential();
        mapping.advise_willneed();
        let manifest = parse_official_manifest(&mapping)?;

        // Hashing the file-backed mapping verifies the staged bytes and faults
        // every page once before the hard-residency admission.
        let hash_started = Instant::now();
        let all_bytes = mapping.bytes(0, EXPECTED_FILE_BYTES as usize)?;
        let actual_hash = sha256_hex(all_bytes);
        let hash_us = hash_started.elapsed().as_micros();
        if actual_hash != EXPECTED_SHA256 {
            return Err(invalid(format!(
                "SHA-256 {actual_hash} does not match expected {EXPECTED_SHA256}"
            )));
        }
        let lock_started = Instant::now();
        let locked_mapping = lock_fully_resident(mapping)?;
        let lock_and_residency_us = lock_started.elapsed().as_micros();

        let kernel =
            metal_linear_kernel().ok_or_else(|| invalid("Metal common core is unavailable"))?;
        let pipeline_started = Instant::now();
        let pipelines = MtpPipelines::new(&kernel.device)?;
        let pipeline_compile_us = pipeline_started.elapsed().as_micros();
        let weight_file = kernel.device.new_buffer_with_bytes_no_copy(
            locked_mapping.mapping.base_ptr().cast::<c_void>(),
            locked_mapping.mapping.mapped_len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let norm = |name: &str| -> Result<Buffer> {
            let values = decode_bf16(&locked_mapping.mapping, manifest.tensor(name)?)?;
            Ok(f32_buffer(&kernel.device, &values))
        };
        let mut layers = Vec::with_capacity(4);
        for layer in 0..4 {
            let prefix = format!("model.layers.{layer}");
            let scale_values = decode_bf16(
                &locked_mapping.mapping,
                manifest.tensor(&format!("{prefix}.layer_scalar"))?,
            )?;
            if scale_values.len() != 1 || !scale_values[0].is_finite() {
                return Err(invalid(format!("layer {layer} scalar is invalid")));
            }
            let scale_scalar = shared_buffer(&kernel.device, 8);
            unsafe {
                let ptr = scale_scalar.contents().cast::<u8>();
                *ptr.cast::<u32>() = ASSISTANT_HIDDEN as u32;
                *ptr.add(4).cast::<f32>() = scale_values[0];
            }
            layers.push(LayerWeights {
                input_norm: norm(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: norm(&format!("{prefix}.post_attention_layernorm.weight"))?,
                pre_feedforward_norm: norm(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                post_feedforward_norm: norm(&format!(
                    "{prefix}.post_feedforward_layernorm.weight"
                ))?,
                q_norm: norm(&format!("{prefix}.self_attn.q_norm.weight"))?,
                q: manifest.matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
                o: manifest.matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
                gate: manifest.matrix(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up: manifest.matrix(&format!("{prefix}.mlp.up_proj.weight"))?,
                down: manifest.matrix(&format!("{prefix}.mlp.down_proj.weight"))?,
                scale_scalar,
            });
        }
        let final_norm = norm("model.norm.weight")?;
        let embedding = manifest.matrix("model.embed_tokens.weight")?;
        let q4_embedding = quantize_embedding_to_q4_0(&kernel.device, &locked_mapping.mapping, embedding).ok();
        let pre_projection = manifest.matrix("pre_projection.weight")?;
        let post_projection = manifest.matrix("post_projection.weight")?;
        let scratch = MtpScratch::new(&kernel.device);
        let resident_ledger = Gemma4MtpResidentLedger {
            file_bytes: locked_mapping.mapping.file_len(),
            mapped_bytes: locked_mapping.mapping.mapped_len() as u64,
            locked_bytes: locked_mapping.locked_bytes as u64,
            resident_pages: locked_mapping.resident_pages as u64,
            total_pages: locked_mapping.total_pages as u64,
            payload_bytes: EXPECTED_PAYLOAD_BYTES as u64,
            decoded_norm_bytes: layers
                .iter()
                .map(|layer| {
                    layer.input_norm.length()
                        + layer.post_attention_norm.length()
                        + layer.pre_feedforward_norm.length()
                        + layer.post_feedforward_norm.length()
                        + layer.q_norm.length()
                        + layer.scale_scalar.length()
                })
                .sum::<u64>()
                + final_norm.length(),
            fixed_scratch_bytes: scratch.byte_len(),
            hash_us,
            lock_and_residency_us,
            pipeline_compile_us,
            load_wall_us: load_started.elapsed().as_micros(),
        };

        Ok(Self {
            weight_file,
            pipelines,
            layers,
            final_norm,
            embedding,
            q4_embedding,
            pre_projection,
            post_projection,
            scratch,
            queue: kernel.device.new_command_queue(),
            resident_ledger,
            last_proposal_ledger: None,
            source_path: path.to_path_buf(),
            _locked_mapping: locked_mapping,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn resident_ledger(&self) -> Gemma4MtpResidentLedger {
        self.resident_ledger
    }

    /// Whether the admitted assistant is executing its tied output projection
    /// through the bounded Q4_0 copy. Structured hybrid serve telemetry is
    /// omitted when this is false so the receipt contract can never claim a Q4
    /// assistant head while the runtime is using the BF16 fallback.
    #[doc(hidden)]
    pub fn q4_head_enabled(&self) -> bool {
        self.q4_embedding.is_some()
    }

    /// Ledger from the latest successful target-backed proposal. The
    /// target-free warmup restores this value, so its private one-position KV
    /// buffers can never be mistaken for the measured target allocation.
    #[doc(hidden)]
    pub fn last_proposal_ledger(&self) -> Option<Gemma4MtpProposalLedger> {
        self.last_proposal_ledger
    }

    /// Warm the isolated assistant without borrowing target KV or entering the
    /// target runtime. A single zero-valued synthetic position exercises every
    /// assistant matrix/pipeline using private, bounded buffers; consequently
    /// it cannot perturb Ghost expert LFU state or target sequence residency.
    #[doc(hidden)]
    pub fn warm_target_free(&mut self) -> Result<Gemma4MtpProposalTiming> {
        let kernel =
            metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        let local_elements = LOCAL_KV_HEADS * LOCAL_HEAD_DIM;
        let full_elements = FULL_KV_HEADS * FULL_HEAD_DIM;
        let sliding_key = f32_buffer(&kernel.device, &vec![0.0; local_elements]);
        let sliding_value = f32_buffer(&kernel.device, &vec![0.0; local_elements]);
        let full_key = f32_buffer(&kernel.device, &vec![0.0; full_elements]);
        let full_value = f32_buffer(&kernel.device, &vec![0.0; full_elements]);
        let target_kv = Gemma4MtpTargetKvView {
            sliding: Gemma4MtpTargetKvLayerView {
                layer_index: 28,
                key: &sliding_key,
                value: &sliding_value,
                logical_len: 1,
                kv_stride: 1,
                kv_heads: LOCAL_KV_HEADS,
                head_dim: LOCAL_HEAD_DIM,
                sliding_window: Some(LOCAL_WINDOW),
            },
            full: Gemma4MtpTargetKvLayerView {
                layer_index: 29,
                key: &full_key,
                value: &full_value,
                logical_len: 1,
                kv_stride: 1,
                kv_heads: FULL_KV_HEADS,
                head_dim: FULL_HEAD_DIM,
                sliding_window: None,
            },
        };
        let zero_target = vec![0.0f32; TARGET_HIDDEN];
        let previous_ledger = self.last_proposal_ledger;
        let proposal = self.propose(&zero_target, &zero_target, target_kv)?;
        self.last_proposal_ledger = previous_ledger;
        Ok(proposal.timing)
    }

    /// Run one official assistant proposal against the target's scoped shared
    /// K/V pair. `target_scaled_embedding` is the target model's embedding for
    /// its freshly sampled authoritative token; `pending_target_hidden` is the
    /// target's final-normalized hidden row immediately preceding that token.
    pub fn propose(
        &mut self,
        target_scaled_embedding: &[f32],
        pending_target_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
    ) -> Result<Gemma4MtpProposal> {
        let wall_started = Instant::now();
        if target_scaled_embedding.len() != TARGET_HIDDEN
            || pending_target_hidden.len() != TARGET_HIDDEN
        {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 MTP input widths are embedding={} hidden={}, expected {TARGET_HIDDEN}",
                target_scaled_embedding.len(),
                pending_target_hidden.len()
            )));
        }
        if target_scaled_embedding
            .iter()
            .chain(pending_target_hidden)
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP input contains non-finite values".into(),
            ));
        }
        validate_target_kv(&target_kv)?;
        let logical_len = target_kv.logical_len();
        if logical_len == 0 {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP target KV is empty".into(),
            ));
        }

        let mut pre_input = Vec::with_capacity(TARGET_HIDDEN * 2);
        pre_input.extend(
            target_scaled_embedding
                .iter()
                .copied()
                .map(round_to_bf16_f32),
        );
        pre_input.extend(pending_target_hidden.iter().copied().map(round_to_bf16_f32));
        write_buffer_f32(&self.scratch.pre_input, &pre_input);
        // The shared target KV contains only the processed target prefix. The
        // current authoritative anchor is the target's unforwarded bonus token:
        // its embedding is paired with the final-normalized hidden row of the
        // preceding processed token. Both pinned Transformers and llama.cpp
        // therefore run the assistant query at position == shared-KV length;
        // every autoregressive assistant proposal reuses this same position.
        let proposal_position = logical_len;
        write_rope_tables(
            proposal_position,
            10_000.0,
            LOCAL_HEAD_DIM,
            LOCAL_HEAD_DIM / 2,
            &self.scratch.local_cos,
            &self.scratch.local_sin,
        );
        write_rope_tables(
            proposal_position,
            1_000_000.0,
            FULL_HEAD_DIM,
            FULL_ROPE_ACTIVE_PAIRS,
            &self.scratch.full_cos,
            &self.scratch.full_sin,
        );

        let (local_start, local_count) = assistant_local_attention_bounds(logical_len);
        write_attention_scalar(
            &self.scratch.local_attention_scalar,
            target_kv.sliding(),
            local_count,
            local_start,
        )?;
        write_attention_scalar(
            &self.scratch.full_attention_scalar,
            target_kv.full(),
            logical_len,
            0,
        )?;

        let score_elements = N_HEADS
            .checked_mul(logical_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let score_bytes = score_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid("attention score byte size overflow"))?;
        let kernel =
            metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        let attention_scores = shared_buffer(&kernel.device, score_bytes);

        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        #[cfg(test)]
        let mut pending_stage_snapshots = (std::env::var("CAMELID_GEMMA4_MTP_STAGE_DIAGNOSTICS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            || std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON").is_some())
        .then(Vec::new);
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.pre_input,
            &self.scratch.hidden,
            self.pre_projection,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.hidden,
            ASSISTANT_HIDDEN,
            "pre_projection",
            &mut pending_stage_snapshots,
        );

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let (kv, head_dim, position_count, cos, sin, qnorm_scalar, rope_scalar, attn_scalar) =
                if layer_index < 3 {
                    (
                        target_kv.sliding(),
                        LOCAL_HEAD_DIM,
                        local_count,
                        &self.scratch.local_cos,
                        &self.scratch.local_sin,
                        &self.scratch.local_qnorm_scalar,
                        &self.scratch.local_rope_scalar,
                        &self.scratch.local_attention_scalar,
                    )
                } else {
                    (
                        target_kv.full(),
                        FULL_HEAD_DIM,
                        logical_len,
                        &self.scratch.full_cos,
                        &self.scratch.full_sin,
                        &self.scratch.full_qnorm_scalar,
                        &self.scratch.full_rope_scalar,
                        &self.scratch.full_attention_scalar,
                    )
                };
            self.encode_layer(
                encoder,
                kernel,
                layer_index,
                layer,
                kv,
                head_dim,
                position_count,
                cos,
                sin,
                qnorm_scalar,
                rope_scalar,
                attn_scalar,
                &attention_scores,
                #[cfg(test)]
                &mut pending_stage_snapshots,
            );
        }

        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.hidden,
            &self.final_norm,
            &self.scratch.final_normalized,
            &self.scratch.hidden_rms_scalar,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
            "final_norm",
            &mut pending_stage_snapshots,
        );
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.final_normalized,
            &self.scratch.recurrent_hidden,
            self.post_projection,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.recurrent_hidden,
            TARGET_HIDDEN,
            "post_projection",
            &mut pending_stage_snapshots,
        );
        if let Some(q4_emb) = self.q4_embedding.as_ref() {
            encode_q4_0_gemv(
                encoder,
                &self.pipelines.q4_0_gemv,
                q4_emb,
                &self.scratch.final_normalized,
                &self.scratch.logits,
                self.embedding.cols as u32,
                self.embedding.rows as u32,
            );
        } else {
            encode_bf16_gemv(
                encoder,
                self.pipelines.selected_bf16_gemv(),
                &self.weight_file,
                &self.scratch.final_normalized,
                &self.scratch.logits,
                self.embedding,
            );
        }
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.logits,
            VOCAB,
            "lm_head",
            &mut pending_stage_snapshots,
        );
        encode_argmax(
            encoder,
            &self.pipelines.argmax,
            &self.scratch.logits,
            &self.scratch.output_token,
            VOCAB,
        );
        encoder.end_encoding();
        let encode_us = encode_started.elapsed().as_micros();

        command_buffer.commit();
        let wait_started = Instant::now();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "Metal command buffer ended with status {:?}",
                command_buffer.status()
            )));
        }
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(command_buffer);
        let token = unsafe { *self.scratch.output_token.contents().cast::<u32>() };
        if token as usize >= VOCAB {
            return Err(invalid(format!(
                "Metal argmax returned invalid token {token}"
            )));
        }
        let mut recurrent_hidden = vec![0.0f32; TARGET_HIDDEN];
        read_buffer_f32(&self.scratch.recurrent_hidden, &mut recurrent_hidden);
        #[cfg(test)]
        let stage_snapshots = finish_stage_snapshots(pending_stage_snapshots);
        let borrowed_target_kv_capacity_bytes = borrowed_target_kv_capacity_bytes(&target_kv)?;
        let target_kv_read_bytes = target_kv_read_bytes(local_count, logical_len)?;

        let ledger = Gemma4MtpProposalLedger {
            assistant_matrix_bytes: MATRIX_BYTES_PER_PROPOSAL,
            borrowed_target_kv_capacity_bytes,
            target_kv_read_bytes,
            dynamic_attention_scratch_bytes: attention_scores.length(),
            readback_bytes: (std::mem::size_of::<u32>()
                + TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
        };
        self.last_proposal_ledger = Some(ledger);
        Ok(Gemma4MtpProposal {
            token,
            recurrent_hidden,
            timing: Gemma4MtpProposalTiming {
                encode_us,
                wait_us,
                wall_us: wall_started.elapsed().as_micros(),
                gpu_us,
                kernel_us,
            },
            ledger,
            #[cfg(test)]
            stage_snapshots,
        })
    }

    /// Established correctness path for a chained assistant sequence. Each
    /// draft performs a CPU target-embedding callback, one Metal commit/wait,
    /// and token/recurrent readback. This remains the default fallback.
    pub fn propose_chain<F>(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        draft_limit: usize,
        eot: &[u32],
        mut get_token_embedding: F,
    ) -> Result<Vec<Gemma4MtpProposal>>
    where
        F: FnMut(u32, &mut [f32]) -> Result<()>,
    {
        if draft_limit == 0 {
            return Ok(Vec::new());
        }
        let draft_limit = draft_limit.min(MTP_CHAIN_MAX_DRAFTS);
        if initial_recurrent_hidden.len() != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 MTP recurrent hidden width is {}, expected {TARGET_HIDDEN}",
                initial_recurrent_hidden.len()
            )));
        }
        validate_target_kv(&target_kv)?;
        let logical_len = target_kv.logical_len();
        if logical_len == 0 {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP target KV is empty".into(),
            ));
        }

        let proposal_position = logical_len;
        write_rope_tables(
            proposal_position,
            10_000.0,
            LOCAL_HEAD_DIM,
            LOCAL_HEAD_DIM / 2,
            &self.scratch.local_cos,
            &self.scratch.local_sin,
        );
        write_rope_tables(
            proposal_position,
            1_000_000.0,
            FULL_HEAD_DIM,
            FULL_HEAD_DIM / 2,
            &self.scratch.full_cos,
            &self.scratch.full_sin,
        );

        let (local_start, local_count) = assistant_local_attention_bounds(logical_len);
        write_attention_scalar(
            &self.scratch.local_attention_scalar,
            target_kv.sliding(),
            local_count,
            local_start,
        )?;
        write_attention_scalar(
            &self.scratch.full_attention_scalar,
            target_kv.full(),
            logical_len,
            0,
        )?;

        let score_elements = N_HEADS
            .checked_mul(logical_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let score_bytes = score_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid("attention score byte size overflow"))?;
        let kernel =
            metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        let attention_scores = shared_buffer(&kernel.device, score_bytes);

        let mut proposals = Vec::with_capacity(draft_limit);
        let mut current_token = anchor_token;
        let mut current_recurrent_hidden = initial_recurrent_hidden.to_vec();
        let pre_input_ptr = self.scratch.pre_input.contents().cast::<f32>();
        let mut embed_buf = [0.0f32; TARGET_HIDDEN];

        // Pre-fill initial recurrent hidden into second half of pre_input
        for i in 0..TARGET_HIDDEN {
            unsafe {
                *pre_input_ptr.add(TARGET_HIDDEN + i) = round_to_bf16_f32(initial_recurrent_hidden[i]);
            }
        }
        for step in 0..draft_limit {
            let step_start = Instant::now();
            get_token_embedding(current_token, &mut embed_buf)?;

            // Write target embedding directly into first half of pre_input buffer
            for i in 0..TARGET_HIDDEN {
                unsafe {
                    *pre_input_ptr.add(i) = round_to_bf16_f32(embed_buf[i]);
                }
            }

            if step > 0 {
                let rec_ptr = self.scratch.recurrent_hidden.contents().cast::<f32>();
                for i in 0..TARGET_HIDDEN {
                    unsafe {
                        *pre_input_ptr.add(TARGET_HIDDEN + i) = round_to_bf16_f32(*rec_ptr.add(i));
                    }
                }
            }

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let encode_started = Instant::now();

            encode_bf16_gemv(
                encoder,
                self.pipelines.selected_bf16_gemv(),
                &self.weight_file,
                &self.scratch.pre_input,
                &self.scratch.hidden,
                self.pre_projection,
            );

            for (layer_index, layer) in self.layers.iter().enumerate() {
                let (kv, head_dim, position_count, cos, sin, qnorm_scalar, rope_scalar, attn_scalar) =
                    if layer_index < 3 {
                        (
                            target_kv.sliding(),
                            LOCAL_HEAD_DIM,
                            local_count,
                            &self.scratch.local_cos,
                            &self.scratch.local_sin,
                            &self.scratch.local_qnorm_scalar,
                            &self.scratch.local_rope_scalar,
                            &self.scratch.local_attention_scalar,
                        )
                    } else {
                        (
                            target_kv.full(),
                            FULL_HEAD_DIM,
                            logical_len,
                            &self.scratch.full_cos,
                            &self.scratch.full_sin,
                            &self.scratch.full_qnorm_scalar,
                            &self.scratch.full_rope_scalar,
                            &self.scratch.full_attention_scalar,
                        )
                    };
                self.encode_layer(
                    encoder,
                    kernel,
                    layer_index,
                    layer,
                    kv,
                    head_dim,
                    position_count,
                    cos,
                    sin,
                    qnorm_scalar,
                    rope_scalar,
                    attn_scalar,
                    &attention_scores,
                    #[cfg(test)]
                    &mut None,
                );
            }

            encode_assistant_rms_norm_f32(
                encoder,
                &self.pipelines,
                &self.scratch.hidden,
                &self.final_norm,
                &self.scratch.final_normalized,
                &self.scratch.hidden_rms_scalar,
            );
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.final_normalized,
                ASSISTANT_HIDDEN,
            );

            encode_bf16_gemv(
                encoder,
                self.pipelines.selected_bf16_gemv(),
                &self.weight_file,
                &self.scratch.final_normalized,
                &self.scratch.recurrent_hidden,
                self.post_projection,
            );

            if let Some(q4_emb) = self.q4_embedding.as_ref() {
                encode_q4_0_gemv(
                    encoder,
                    &self.pipelines.q4_0_gemv,
                    q4_emb,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    self.embedding.cols as u32,
                    self.embedding.rows as u32,
                );
            } else {
                encode_bf16_gemv(
                    encoder,
                    self.pipelines.selected_bf16_gemv(),
                    &self.weight_file,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    self.embedding,
                );
            }

            encode_argmax(
                encoder,
                &self.pipelines.argmax,
                &self.scratch.logits,
                &self.scratch.output_token,
                VOCAB,
            );
            encoder.end_encoding();
            let encode_us = encode_started.elapsed().as_micros();

            command_buffer.commit();
            let wait_started = Instant::now();
            command_buffer.wait_until_completed();
            let wait_us = wait_started.elapsed().as_micros();

            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(invalid(format!(
                    "Metal command buffer ended with status {:?}",
                    command_buffer.status()
                )));
            }
            let (gpu_us, kernel_us) = command_buffer_gpu_times_us(command_buffer);
            let token = unsafe { *self.scratch.output_token.contents().cast::<u32>() };
            if token as usize >= VOCAB {
                return Err(invalid(format!(
                    "Metal argmax returned invalid token {token}"
                )));
            }

            read_buffer_f32(&self.scratch.recurrent_hidden, &mut current_recurrent_hidden);
            current_token = token;

            proposals.push(Gemma4MtpProposal {
                token,
                recurrent_hidden: current_recurrent_hidden.clone(),
                timing: Gemma4MtpProposalTiming {
                    encode_us,
                    wait_us,
                    wall_us: step_start.elapsed().as_micros(),
                    gpu_us,
                    kernel_us,
                },
                ledger: Gemma4MtpProposalLedger::default(),
                #[cfg(test)]
                stage_snapshots: Vec::new(),
            });

            if eot.contains(&token) {
                break;
            }
        }

        Ok(proposals)
    }

    /// Experimental device-fed assistant chain. This is called only after the
    /// explicit `CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1` gate in the target runtime.
    ///
    /// All draft steps are encoded into one command buffer. The token selected
    /// at step N feeds the target Q6_K embedding gather at step N+1, and the
    /// post-projection recurrent row stays in Metal scratch. Per-step token ids
    /// and recurrent rows are copied to retained shared buffers, then read once
    /// after the single final wait. No command or borrowed target buffer escapes
    /// this call.
    pub fn propose_chain_device_resident(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        target_embedding: Gemma4MtpTargetEmbeddingView<'_>,
        draft_limit: usize,
        eot: &[u32],
    ) -> Result<Vec<Gemma4MtpProposal>> {
        if draft_limit == 0 {
            return Ok(Vec::new());
        }
        let draft_limit = draft_limit.min(MTP_CHAIN_MAX_DRAFTS);
        if anchor_token as usize >= VOCAB {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 MTP anchor token {anchor_token} exceeds vocab {VOCAB}"
            )));
        }
        if initial_recurrent_hidden.len() != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 MTP recurrent hidden width is {}, expected {TARGET_HIDDEN}",
                initial_recurrent_hidden.len()
            )));
        }
        if initial_recurrent_hidden
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP recurrent hidden contains non-finite values".into(),
            ));
        }
        validate_target_kv(&target_kv)?;
        let logical_len = target_kv.logical_len();
        if logical_len == 0 {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP target KV is empty".into(),
            ));
        }
        let device_registry_id = self.queue.device().registry_id();
        validate_target_embedding(&target_embedding, device_registry_id)?;
        validate_target_kv_device(&target_kv, device_registry_id)?;

        let wall_started = Instant::now();
        write_rope_tables(
            logical_len,
            10_000.0,
            LOCAL_HEAD_DIM,
            LOCAL_HEAD_DIM / 2,
            &self.scratch.local_cos,
            &self.scratch.local_sin,
        );
        write_rope_tables(
            logical_len,
            1_000_000.0,
            FULL_HEAD_DIM,
            FULL_HEAD_DIM / 2,
            &self.scratch.full_cos,
            &self.scratch.full_sin,
        );

        let (local_start, local_count) = assistant_local_attention_bounds(logical_len);
        write_attention_scalar(
            &self.scratch.local_attention_scalar,
            target_kv.sliding(),
            local_count,
            local_start,
        )?;
        write_attention_scalar(
            &self.scratch.full_attention_scalar,
            target_kv.full(),
            logical_len,
            0,
        )?;

        let score_elements = N_HEADS
            .checked_mul(logical_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let score_bytes = score_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid("attention score byte size overflow"))?;
        let kernel =
            metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        if kernel.device.registry_id() != device_registry_id {
            return Err(invalid(
                "assistant queue and common Metal core use different devices",
            ));
        }
        let attention_scores = shared_buffer(&kernel.device, score_bytes);

        write_buffer_f32(&self.scratch.recurrent_hidden, initial_recurrent_hidden);
        unsafe {
            *self.scratch.output_token.contents().cast::<u32>() = anchor_token;
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        for step in 0..draft_limit {
            let input_token_offset = if step == 0 {
                0
            } else {
                ((step - 1) * std::mem::size_of::<u32>()) as u64
            };
            encode_mtp_gather_q6k_embed_and_recurrent(
                encoder,
                &self.pipelines.gather_q6k_embed_and_recurrent,
                &self.scratch.output_token,
                input_token_offset,
                &target_embedding,
                &self.scratch.recurrent_hidden,
                &self.scratch.pre_input,
                TARGET_HIDDEN,
            );
            encode_bf16_gemv(
                encoder,
                self.pipelines.selected_bf16_gemv(),
                &self.weight_file,
                &self.scratch.pre_input,
                &self.scratch.hidden,
                self.pre_projection,
            );

            for (layer_index, layer) in self.layers.iter().enumerate() {
                let (kv, head_dim, position_count, cos, sin, qnorm_scalar, rope_scalar, attn_scalar) =
                    if layer_index < 3 {
                        (
                            target_kv.sliding(),
                            LOCAL_HEAD_DIM,
                            local_count,
                            &self.scratch.local_cos,
                            &self.scratch.local_sin,
                            &self.scratch.local_qnorm_scalar,
                            &self.scratch.local_rope_scalar,
                            &self.scratch.local_attention_scalar,
                        )
                    } else {
                        (
                            target_kv.full(),
                            FULL_HEAD_DIM,
                            logical_len,
                            &self.scratch.full_cos,
                            &self.scratch.full_sin,
                            &self.scratch.full_qnorm_scalar,
                            &self.scratch.full_rope_scalar,
                            &self.scratch.full_attention_scalar,
                        )
                    };
                self.encode_layer(
                    encoder,
                    kernel,
                    layer_index,
                    layer,
                    kv,
                    head_dim,
                    position_count,
                    cos,
                    sin,
                    qnorm_scalar,
                    rope_scalar,
                    attn_scalar,
                    &attention_scores,
                    #[cfg(test)]
                    &mut None,
                );
            }

            encode_assistant_rms_norm_f32(
                encoder,
                &self.pipelines,
                &self.scratch.hidden,
                &self.final_norm,
                &self.scratch.final_normalized,
                &self.scratch.hidden_rms_scalar,
            );
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.final_normalized,
                ASSISTANT_HIDDEN,
            );
            encode_bf16_gemv(
                encoder,
                self.pipelines.selected_bf16_gemv(),
                &self.weight_file,
                &self.scratch.final_normalized,
                &self.scratch.recurrent_hidden,
                self.post_projection,
            );
            encode_copy_f32_to_offset(
                encoder,
                &self.pipelines.copy_f32,
                &self.scratch.recurrent_hidden,
                &self.scratch.chain_recurrent_hidden,
                (step * TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
                TARGET_HIDDEN,
            );

            if let Some(q4_emb) = self.q4_embedding.as_ref() {
                encode_q4_0_gemv(
                    encoder,
                    &self.pipelines.q4_0_gemv,
                    q4_emb,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    self.embedding.cols as u32,
                    self.embedding.rows as u32,
                );
            } else {
                encode_bf16_gemv(
                    encoder,
                    self.pipelines.selected_bf16_gemv(),
                    &self.weight_file,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    self.embedding,
                );
            }
            encode_argmax_offset(
                encoder,
                &self.pipelines.argmax,
                &self.scratch.logits,
                &self.scratch.output_token,
                (step * std::mem::size_of::<u32>()) as u64,
                VOCAB,
            );
        }
        encoder.end_encoding();
        let encode_us = encode_started.elapsed().as_micros();

        command_buffer.commit();
        let wait_started = Instant::now();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "device-chain Metal command buffer ended with status {:?}",
                command_buffer.status()
            )));
        }
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(command_buffer);

        let mut recurrent_rows = vec![0.0f32; draft_limit * TARGET_HIDDEN];
        read_buffer_f32(&self.scratch.chain_recurrent_hidden, &mut recurrent_rows);
        let token_ptr = self.scratch.output_token.contents().cast::<u32>();
        let tokens = unsafe { std::slice::from_raw_parts(token_ptr, draft_limit) }.to_vec();

        let borrowed_target_kv_capacity_bytes = borrowed_target_kv_capacity_bytes(&target_kv)?;
        let per_step_kv_read_bytes = target_kv_read_bytes(local_count, logical_len)?;
        let draft_count_u64 = draft_limit as u64;
        let ledger = Gemma4MtpProposalLedger {
            assistant_matrix_bytes: MATRIX_BYTES_PER_PROPOSAL
                .checked_mul(draft_count_u64)
                .ok_or_else(|| invalid("device-chain matrix-byte ledger overflow"))?,
            borrowed_target_kv_capacity_bytes,
            target_kv_read_bytes: per_step_kv_read_bytes
                .checked_mul(draft_count_u64)
                .ok_or_else(|| invalid("device-chain target-KV ledger overflow"))?,
            dynamic_attention_scratch_bytes: attention_scores.length(),
            readback_bytes: draft_count_u64
                .checked_mul(
                    (std::mem::size_of::<u32>() + TARGET_HIDDEN * std::mem::size_of::<f32>())
                        as u64,
                )
                .ok_or_else(|| invalid("device-chain readback ledger overflow"))?,
        };
        let total_timing = Gemma4MtpProposalTiming {
            encode_us,
            wait_us,
            wall_us: wall_started.elapsed().as_micros(),
            gpu_us,
            kernel_us,
        };

        let mut proposals = Vec::with_capacity(draft_limit);
        for (step, token) in tokens.into_iter().enumerate() {
            if token as usize >= VOCAB {
                return Err(invalid(format!(
                    "device-chain argmax step {step} returned invalid token {token}"
                )));
            }
            let row_start = step * TARGET_HIDDEN;
            let recurrent_hidden = recurrent_rows[row_start..row_start + TARGET_HIDDEN].to_vec();
            if recurrent_hidden.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "device-chain recurrent row {step} contains non-finite values"
                )));
            }
            proposals.push(Gemma4MtpProposal {
                token,
                recurrent_hidden,
                // The first row owns aggregate one-command timing/ledger so
                // existing callers that sum proposals remain numerically true.
                timing: if step == 0 {
                    total_timing
                } else {
                    Gemma4MtpProposalTiming::default()
                },
                ledger: if step == 0 {
                    ledger
                } else {
                    Gemma4MtpProposalLedger::default()
                },
                #[cfg(test)]
                stage_snapshots: Vec::new(),
            });
            if eot.contains(&token) {
                break;
            }
        }
        self.last_proposal_ledger = Some(ledger);
        Ok(proposals)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(unused_variables))]
    fn encode_layer(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        kernel: &MetalLinearKernel,
        layer_index: usize,
        layer: &LayerWeights,
        kv: &Gemma4MtpTargetKvLayerView<'_>,
        head_dim: usize,
        position_count: usize,
        cos: &Buffer,
        sin: &Buffer,
        qnorm_scalar: &Buffer,
        rope_scalar: &Buffer,
        attention_scalar: &Buffer,
        attention_scores: &Buffer,
        #[cfg(test)] pending_stage_snapshots: &mut Option<Vec<PendingMtpStageSnapshot>>,
    ) {
        let q_dim = N_HEADS * head_dim;
        debug_assert!(position_count > 0);
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.hidden,
            &layer.input_norm,
            &self.scratch.normed,
            &self.scratch.hidden_rms_scalar,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.input_norm"),
            pending_stage_snapshots,
        );
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.normed,
            &self.scratch.query,
            layer.q,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.query,
            q_dim,
            format!("layer.{layer_index}.q_proj"),
            pending_stage_snapshots,
        );
        encode_assistant_rms_norm_per_head(
            encoder,
            &self.pipelines,
            &self.scratch.query,
            &layer.q_norm,
            &self.scratch.query_normed,
            qnorm_scalar,
            N_HEADS,
            head_dim,
            0,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.query_normed,
            q_dim,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.query_normed,
            q_dim,
            format!("layer.{layer_index}.q_norm"),
            pending_stage_snapshots,
        );
        encode_rope_bf16(
            encoder,
            &self.scratch.query_normed,
            cos,
            sin,
            rope_scalar,
            &self.pipelines.rope_bf16,
            N_HEADS,
            cos.length() as usize / std::mem::size_of::<f32>(),
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.query_normed,
            q_dim,
            format!("layer.{layer_index}.q_rope"),
            pending_stage_snapshots,
        );
        encode_attention_bf16(
            encoder,
            &self.pipelines,
            &self.scratch.query_normed,
            kv.key_buffer(),
            kv.value_buffer(),
            attention_scores,
            &self.scratch.context,
            attention_scalar,
            N_HEADS,
            #[cfg(test)]
            layer_index,
            #[cfg(test)]
            position_count,
            #[cfg(test)]
            q_dim,
            #[cfg(test)]
            pending_stage_snapshots,
        );
        debug_assert_eq!(layer.o.cols as usize, q_dim);
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.context,
            &self.scratch.attention_projection,
            layer.o,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.attention_projection,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.o_proj"),
            pending_stage_snapshots,
        );
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.attention_projection,
            &layer.post_attention_norm,
            &self.scratch.attention_normalized,
            &self.scratch.hidden_rms_scalar,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.attention_normalized,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.attention_normalized,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.post_attention_norm"),
            pending_stage_snapshots,
        );
        encode_binary(
            encoder,
            &kernel.residual_add_pipeline,
            &self.scratch.hidden,
            &self.scratch.attention_normalized,
            &self.scratch.attention_residual,
            &self.scratch.hidden_count,
            ASSISTANT_HIDDEN,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.attention_residual,
            ASSISTANT_HIDDEN,
        );
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.attention_residual,
            &layer.pre_feedforward_norm,
            &self.scratch.normed,
            &self.scratch.hidden_rms_scalar,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.pre_feedforward_norm"),
            pending_stage_snapshots,
        );
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.normed,
            &self.scratch.gate,
            layer.gate,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.gate,
            FFN_HIDDEN,
            format!("layer.{layer_index}.gate_proj"),
            pending_stage_snapshots,
        );
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.normed,
            &self.scratch.up,
            layer.up,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.up,
            FFN_HIDDEN,
            format!("layer.{layer_index}.up_proj"),
            pending_stage_snapshots,
        );
        encode_gelu_mul_bf16(
            encoder,
            &self.pipelines.gelu_mul_bf16,
            &self.scratch.gate,
            &self.scratch.up,
            &self.scratch.gated,
            FFN_HIDDEN,
        );
        encode_bf16_gemv(
            encoder,
            self.pipelines.selected_bf16_gemv(),
            &self.weight_file,
            &self.scratch.gated,
            &self.scratch.down,
            layer.down,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.down,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.down_proj"),
            pending_stage_snapshots,
        );
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.down,
            &layer.post_feedforward_norm,
            &self.scratch.down_normalized,
            &self.scratch.hidden_rms_scalar,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.down_normalized,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.down_normalized,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.post_feedforward_norm"),
            pending_stage_snapshots,
        );
        encode_binary(
            encoder,
            &kernel.residual_add_pipeline,
            &self.scratch.attention_residual,
            &self.scratch.down_normalized,
            &self.scratch.next_hidden,
            &self.scratch.hidden_count,
            ASSISTANT_HIDDEN,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.next_hidden,
            ASSISTANT_HIDDEN,
        );
        encode_scale_f32(
            encoder,
            kernel,
            &self.scratch.next_hidden,
            &self.scratch.hidden,
            &layer.scale_scalar,
            ASSISTANT_HIDDEN,
        );
        encode_round_bf16(
            encoder,
            &self.pipelines.round_bf16,
            &self.scratch.hidden,
            ASSISTANT_HIDDEN,
        );
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.hidden,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.output"),
            pending_stage_snapshots,
        );
    }
}

fn encode_bf16_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weight_file: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: TensorRef,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weight_file), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &matrix.absolute_offset as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &matrix.cols as *const u32 as *const c_void);
    encoder.set_bytes(5, 4, &matrix.rows as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: matrix.rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

fn dispatch_1d(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    count: usize,
) {
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (count as u64).div_ceil(width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_round_bf16(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    data: &Buffer,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(data), 0);
    encoder.set_bytes(1, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

#[cfg(test)]
fn encode_stage_snapshot(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    source: &Buffer,
    count: usize,
    name: impl Into<String>,
    pending: &mut Option<Vec<PendingMtpStageSnapshot>>,
) {
    let Some(pending) = pending.as_mut() else {
        return;
    };
    let buffer = shared_buffer(
        &metal_linear_kernel()
            .expect("MTP stage diagnostics require the common Metal core")
            .device,
        count * std::mem::size_of::<f32>(),
    );
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(source), 0);
    encoder.set_buffer(1, Some(&buffer), 0);
    encoder.set_bytes(2, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
    pending.push(PendingMtpStageSnapshot {
        name: name.into(),
        buffer,
        count,
    });
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_stage_row_tail_snapshot(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    source: &Buffer,
    input_columns: usize,
    output_columns: usize,
    row_count: usize,
    name: impl Into<String>,
    pending: &mut Option<Vec<PendingMtpStageSnapshot>>,
) {
    let Some(pending) = pending.as_mut() else {
        return;
    };
    assert!(output_columns <= input_columns);
    let count = row_count
        .checked_mul(output_columns)
        .expect("MTP row-tail diagnostic element count overflow");
    let buffer = shared_buffer(
        &metal_linear_kernel()
            .expect("MTP stage diagnostics require the common Metal core")
            .device,
        count * std::mem::size_of::<f32>(),
    );
    let input_columns = input_columns as u32;
    let output_columns = output_columns as u32;
    let row_count = row_count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(source), 0);
    encoder.set_buffer(1, Some(&buffer), 0);
    encoder.set_bytes(2, 4, &input_columns as *const u32 as *const c_void);
    encoder.set_bytes(3, 4, &output_columns as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &row_count as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
    pending.push(PendingMtpStageSnapshot {
        name: name.into(),
        buffer,
        count,
    });
}

#[cfg(test)]
fn finish_stage_snapshots(pending: Option<Vec<PendingMtpStageSnapshot>>) -> Vec<MtpStageSnapshot> {
    pending
        .unwrap_or_default()
        .into_iter()
        .map(|pending| {
            let mut values = vec![0.0f32; pending.count];
            read_buffer_f32(&pending.buffer, &mut values);
            let mut digest = Sha256::new();
            for value in &values {
                digest.update(f32_to_bf16_rne_bits(*value).to_le_bytes());
            }
            MtpStageSnapshot {
                name: pending.name,
                values,
                bf16_sha256: format!("{:x}", digest.finalize()),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn encode_rope_bf16(
    encoder: &metal::ComputeCommandEncoderRef,
    data: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    scalar: &Buffer,
    pipeline: &ComputePipelineState,
    head_count: usize,
    half_head: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(data), 0);
    encoder.set_buffer(1, Some(cos), 0);
    encoder.set_buffer(2, Some(sin), 0);
    encoder.set_buffer(3, Some(scalar), 0);
    encoder.set_buffer(4, Some(scalar), 4);
    encoder.set_buffer(5, Some(scalar), 8);
    dispatch_1d(encoder, pipeline, head_count * half_head);
}

fn encode_gelu_mul_bf16(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    output: &Buffer,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(gate), 0);
    encoder.set_buffer(1, Some(up), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_bf16(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &MtpPipelines,
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    scores: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    head_count: usize,
    #[cfg(test)] layer_index: usize,
    #[cfg(test)] position_count: usize,
    #[cfg(test)] context_elements: usize,
    #[cfg(test)] pending_stage_snapshots: &mut Option<Vec<PendingMtpStageSnapshot>>,
) {
    let groups = MTLSize {
        width: head_count as u64,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    #[cfg(test)]
    let checkpoint_columns = stage_attention_checkpoint_columns(layer_index, position_count);

    encoder.set_compute_pipeline_state(pipelines.selected_attention_scores_bf16());
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(keys), 0);
    encoder.set_buffer(2, Some(scores), 0);
    for index in 0..8u64 {
        encoder.set_buffer(3 + index, Some(scalar), index * 4);
    }
    encoder.dispatch_thread_groups(groups, threads);
    #[cfg(test)]
    encode_stage_row_tail_snapshot(
        encoder,
        &pipelines.copy_row_tail_f32,
        scores,
        position_count,
        checkpoint_columns,
        head_count,
        format!("layer.{layer_index}.attention_scores"),
        pending_stage_snapshots,
    );

    #[cfg(test)]
    let softmax_pipeline = match test_strict_softmax_mode() {
        Some(TestStrictSoftmaxMode::Geometry) => &pipelines.attention_softmax_aten_geometry_f32,
        Some(TestStrictSoftmaxMode::Sleef) => &pipelines.attention_softmax_aten_sleef_f32,
        None => &pipelines.attention_softmax_bf16,
    };
    #[cfg(not(test))]
    let softmax_pipeline = &pipelines.attention_softmax_bf16;
    encoder.set_compute_pipeline_state(softmax_pipeline);
    encoder.set_buffer(0, Some(scores), 0);
    encoder.set_buffer(1, Some(scalar), 0);
    encoder.set_buffer(2, Some(scalar), 8);
    encoder.dispatch_thread_groups(groups, threads);
    #[cfg(test)]
    encode_stage_row_tail_snapshot(
        encoder,
        &pipelines.copy_row_tail_f32,
        scores,
        position_count,
        checkpoint_columns,
        head_count,
        format!("layer.{layer_index}.attention_probs"),
        pending_stage_snapshots,
    );

    encoder.set_compute_pipeline_state(pipelines.selected_attention_context_bf16());
    encoder.set_buffer(0, Some(values), 0);
    encoder.set_buffer(1, Some(scores), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_buffer(3, Some(scalar), 0);
    encoder.set_buffer(4, Some(scalar), 4);
    encoder.set_buffer(5, Some(scalar), 8);
    encoder.set_buffer(6, Some(scalar), 12);
    encoder.set_buffer(7, Some(scalar), 20);
    encoder.set_buffer(8, Some(scalar), 24);
    encoder.set_buffer(9, Some(scalar), 28);
    encoder.set_buffer(10, Some(scalar), 32);
    encoder.set_buffer(11, Some(scalar), 36);
    encoder.dispatch_thread_groups(groups, threads);
    #[cfg(test)]
    encode_stage_snapshot(
        encoder,
        &pipelines.copy_f32,
        output,
        context_elements,
        format!("layer.{layer_index}.attention_context"),
        pending_stage_snapshots,
    );
}

#[cfg(test)]
fn stage_attention_checkpoint_columns(layer_index: usize, position_count: usize) -> usize {
    if layer_index < 3 {
        // The pinned stage oracle intentionally crops local attention to the
        // final configured-window columns, while computation includes the
        // extra inclusive-boundary row.
        position_count.min(LOCAL_WINDOW)
    } else {
        position_count
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestStrictSoftmaxMode {
    Geometry,
    Sleef,
}

#[cfg(test)]
fn test_strict_softmax_mode() -> Option<TestStrictSoftmaxMode> {
    let value = std::env::var("CAMELID_GEMMA4_MTP_TEST_STRICT_SOFTMAX").ok()?;
    if value.eq_ignore_ascii_case("geometry") {
        Some(TestStrictSoftmaxMode::Geometry)
    } else if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("sleef")
    {
        Some(TestStrictSoftmaxMode::Sleef)
    } else if value == "0" || value.eq_ignore_ascii_case("false") || value.is_empty() {
        None
    } else {
        panic!(
            "CAMELID_GEMMA4_MTP_TEST_STRICT_SOFTMAX must be geometry, sleef, 1/true, or 0/false"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_assistant_aten_rms_norm(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    width: usize,
    head_count: usize,
    row_off: u64,
    use_weight: bool,
) {
    assert!(
        matches!(width, 256 | 512 | 1_024),
        "official assistant RMS only supports width 256/512/1024, got {width}"
    );
    let use_weight = u32::from(use_weight);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), row_off);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output), row_off);
    encoder.set_buffer(3, Some(scalar), 0);
    encoder.set_buffer(4, Some(scalar), 4);
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &use_weight as *const u32 as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: head_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_assistant_rms_norm_f32(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &MtpPipelines,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
) {
    encode_assistant_aten_rms_norm(
        encoder,
        &pipelines.rms_norm_aten_f32,
        input,
        weight,
        output,
        scalar,
        ASSISTANT_HIDDEN,
        1,
        0,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_assistant_rms_norm_per_head(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &MtpPipelines,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    head_count: usize,
    head_dim: usize,
    row_off: u64,
) {
    encode_assistant_aten_rms_norm(
        encoder,
        &pipelines.rms_norm_aten_f32,
        input,
        weight,
        output,
        scalar,
        head_dim,
        head_count,
        row_off,
        true,
    );
}

fn encode_argmax(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    logits: &Buffer,
    output_token: &Buffer,
    count: usize,
) {
    encode_argmax_offset(encoder, pipeline, logits, output_token, 0, count);
}

fn encode_argmax_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    logits: &Buffer,
    output_token: &Buffer,
    token_offset: u64,
    count: usize,
) {
    let count = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(output_token), token_offset);
    encoder.set_bytes(2, 4, &count as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_mtp_gather_q6k_embed_and_recurrent(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    output_token: &Buffer,
    token_offset: u64,
    target_embedding: &Gemma4MtpTargetEmbeddingView<'_>,
    recurrent_hidden: &Buffer,
    pre_input: &Buffer,
    target_hidden: usize,
) {
    let target_hidden_u32 = target_hidden as u32;
    let target_vocab_u32 = target_embedding.vocab() as u32;
    let q6k_superblocks_per_row = (target_hidden / 256) as u32;
    let embedding_scale = (target_hidden as f32).sqrt();
    let target_embedding_offset = target_embedding.byte_offset() as u64;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(output_token), token_offset);
    encoder.set_buffer(1, Some(target_embedding.buffer()), 0);
    encoder.set_bytes(
        2,
        8,
        &target_embedding_offset as *const u64 as *const c_void,
    );
    encoder.set_buffer(3, Some(recurrent_hidden), 0);
    encoder.set_buffer(4, Some(pre_input), 0);
    encoder.set_bytes(5, 4, &target_hidden_u32 as *const u32 as *const c_void);
    encoder.set_bytes(6, 4, &target_vocab_u32 as *const u32 as *const c_void);
    encoder.set_bytes(
        7,
        4,
        &q6k_superblocks_per_row as *const u32 as *const c_void,
    );
    encoder.set_bytes(8, 4, &embedding_scale as *const f32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: ((target_hidden + 255) / 256) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_copy_f32_to_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    source: &Buffer,
    destination: &Buffer,
    destination_offset: u64,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(source), 0);
    encoder.set_buffer(1, Some(destination), destination_offset);
    encoder.set_bytes(2, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

fn encode_q4_0_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q4_weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    cols: u32,
    rows: u32,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q4_weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &rows as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

fn quantize_embedding_to_q4_0(
    device: &Device,
    mapping: &GgufWireMmap,
    tensor: TensorRef,
) -> Result<Buffer> {
    let rows = tensor.rows as usize;
    let cols = tensor.cols as usize;
    let blocks_per_row = cols / 32;
    let total_bytes = rows * blocks_per_row * 18;
    let buf = shared_buffer(device, total_bytes);
    let out_addr = buf.contents() as usize;
    let in_bytes = mapping.bytes(tensor.absolute_offset as u64, rows * cols * 2)?;
    let in_u16 = unsafe {
        std::slice::from_raw_parts(in_bytes.as_ptr() as *const u16, rows * cols)
    };

    use rayon::prelude::*;
    (0..rows).into_par_iter().for_each(|row| {
        let row_in = &in_u16[row * cols..(row + 1) * cols];
        let row_out = (out_addr + row * blocks_per_row * 18) as *mut u8;
        for b in 0..blocks_per_row {
            let block_in = &row_in[b * 32..(b + 1) * 32];
            let mut f32_vals = [0.0f32; 32];
            let mut max_abs = 0.0f32;
            for i in 0..32 {
                let f = bf16_bits_to_f32(block_in[i]);
                f32_vals[i] = f;
                let abs = f.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
            }
            let scale = max_abs / -8.0;
            let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            let d_bits = f32_to_f16_bits(scale);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &d_bits as *const u16 as *const u8,
                    row_out.add(b * 18),
                    2,
                );
                let qs = row_out.add(b * 18 + 2);
                for i in 0..16 {
                    let v0 = (f32_vals[i] * inv_scale + 8.5).floor().clamp(0.0, 15.0) as u8;
                    let v1 = (f32_vals[i + 16] * inv_scale + 8.5).floor().clamp(0.0, 15.0) as u8;
                    *qs.add(i) = (v0 & 0x0f) | ((v1 & 0x0f) << 4);
                }
            }
        }
    });

    Ok(buf)
}

fn validate_target_embedding_geometry(
    format: Gemma4MtpTargetEmbeddingFormat,
    hidden: usize,
    vocab: usize,
    byte_offset: usize,
    byte_len: usize,
    buffer_len: usize,
) -> Result<()> {
    const Q6K_VALUES: usize = 256;
    const Q6K_WIRE: usize = 210;
    if format != Gemma4MtpTargetEmbeddingFormat::Q6K
        || hidden != TARGET_HIDDEN
        || vocab != VOCAB
        || !hidden.is_multiple_of(Q6K_VALUES)
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 MTP device-chain target embedding mismatch: format={format:?} hidden={hidden} vocab={vocab}; expected Q6_K {VOCAB}x{TARGET_HIDDEN}"
        )));
    }
    let expected_len = vocab
        .checked_mul(hidden / Q6K_VALUES)
        .and_then(|value| value.checked_mul(Q6K_WIRE))
        .ok_or_else(|| invalid("target embedding byte length overflow"))?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| invalid("target embedding range overflow"))?;
    if byte_len != expected_len
        || !byte_offset.is_multiple_of(std::mem::align_of::<u16>())
        || byte_end > buffer_len
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 MTP device-chain target embedding range mismatch: offset={byte_offset} bytes={byte_len} buffer={buffer_len}; expected bytes={expected_len} with 2-byte alignment"
        )));
    }
    Ok(())
}

fn validate_target_embedding(
    view: &Gemma4MtpTargetEmbeddingView<'_>,
    required_device_registry_id: u64,
) -> Result<()> {
    validate_target_embedding_geometry(
        view.format(),
        view.hidden(),
        view.vocab(),
        view.byte_offset(),
        view.byte_len(),
        usize::try_from(view.buffer().length())
            .map_err(|_| invalid("target embedding buffer length exceeds usize"))?,
    )?;
    if view.buffer().storage_mode() != MTLStorageMode::Shared {
        return Err(invalid(format!(
            "target embedding storage mode is {:?}, expected Shared no-copy GGUF pages",
            view.buffer().storage_mode()
        )));
    }
    let actual_device_registry_id = view.buffer().device().registry_id();
    if actual_device_registry_id != required_device_registry_id {
        return Err(invalid(format!(
            "target embedding Metal device {actual_device_registry_id} differs from assistant device {required_device_registry_id}"
        )));
    }
    Ok(())
}

fn validate_target_kv_device(
    view: &Gemma4MtpTargetKvView<'_>,
    required_device_registry_id: u64,
) -> Result<()> {
    let buffers = [
        view.sliding().key_buffer(),
        view.sliding().value_buffer(),
        view.full().key_buffer(),
        view.full().value_buffer(),
    ];
    for (index, buffer) in buffers.into_iter().enumerate() {
        let actual = buffer.device().registry_id();
        if actual != required_device_registry_id {
            return Err(invalid(format!(
                "target KV buffer {index} Metal device {actual} differs from assistant device {required_device_registry_id}"
            )));
        }
    }
    Ok(())
}

fn validate_target_kv(view: &Gemma4MtpTargetKvView<'_>) -> Result<()> {
    let sliding = view.sliding();
    let full = view.full();
    let layer_buffers_cover_capacity = |kv: &Gemma4MtpTargetKvLayerView<'_>| {
        kv.kv_heads()
            .checked_mul(kv.kv_stride())
            .and_then(|value| value.checked_mul(kv.head_dim()))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .is_some_and(|required| {
                kv.key_buffer().length() >= required as u64
                    && kv.value_buffer().length() >= required as u64
            })
    };
    if sliding.layer_index() != 28
        || sliding.kv_heads() != LOCAL_KV_HEADS
        || sliding.head_dim() != LOCAL_HEAD_DIM
        || sliding.sliding_window() != Some(LOCAL_WINDOW)
        || full.layer_index() != 29
        || full.kv_heads() != FULL_KV_HEADS
        || full.head_dim() != FULL_HEAD_DIM
        || full.sliding_window().is_some()
        || sliding.logical_len() != full.logical_len()
        || sliding.logical_len() > sliding.kv_stride()
        || full.logical_len() > full.kv_stride()
        || !layer_buffers_cover_capacity(sliding)
        || !layer_buffers_cover_capacity(full)
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 MTP target KV geometry mismatch: sliding=(layer {}, len {}, stride {}, heads {}, dim {}, window {:?}), full=(layer {}, len {}, stride {}, heads {}, dim {}, window {:?})",
            sliding.layer_index(),
            sliding.logical_len(),
            sliding.kv_stride(),
            sliding.kv_heads(),
            sliding.head_dim(),
            sliding.sliding_window(),
            full.layer_index(),
            full.logical_len(),
            full.kv_stride(),
            full.kv_heads(),
            full.head_dim(),
            full.sliding_window(),
        )));
    }
    Ok(())
}

fn write_rope_tables(
    position: usize,
    theta: f32,
    head_dim: usize,
    active_pairs: usize,
    cos_buffer: &Buffer,
    sin_buffer: &Buffer,
) {
    let (cos, sin) = rope_table_values(position, theta, head_dim, active_pairs);
    debug_assert_eq!(
        cos_buffer.length() as usize,
        cos.len() * std::mem::size_of::<f32>()
    );
    debug_assert_eq!(
        sin_buffer.length() as usize,
        sin.len() * std::mem::size_of::<f32>()
    );
    write_buffer_f32(cos_buffer, &cos);
    write_buffer_f32(sin_buffer, &sin);
}

/// Build a full split-half RoPE table. `active_pairs` controls how many
/// frequencies rotate; it never changes the split-half partner stride.  This
/// mirrors Transformers proportional RoPE (real frequencies followed by
/// zeros) and llama.cpp's equivalent [1, ..., 1e30] frequency-factor tensor.
fn rope_table_values(
    position: usize,
    theta: f32,
    head_dim: usize,
    active_pairs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half_head = head_dim / 2;
    assert_eq!(head_dim % 2, 0);
    assert!(active_pairs <= half_head);
    let mut cos = vec![1.0; half_head];
    let mut sin = vec![0.0; half_head];
    for pair in 0..active_pairs {
        let frequency = theta.powf(-(2.0 * pair as f32) / head_dim as f32);
        let (s, c) = (position as f32 * frequency).sin_cos();
        cos[pair] = round_to_bf16_f32(c);
        sin[pair] = round_to_bf16_f32(s);
    }
    (cos, sin)
}

/// Return the past-facing KV slice selected by the official assistant's
/// bidirectional sliding-window mask for a one-token query.
///
/// `sliding_window` is an inclusive distance, so a radius of 1,024 covers
/// 1,025 stored prefix positions. The target cache remains a full, contiguous
/// logical prefix; this helper only selects the assistant mask's visible
/// suffix and is intentionally scoped to the isolated assistant runner.
fn assistant_local_attention_bounds(logical_len: usize) -> (usize, usize) {
    let count = logical_len.min(LOCAL_ATTENTION_SPAN);
    (logical_len - count, count)
}

fn write_attention_scalar(
    scalar: &Buffer,
    kv: &Gemma4MtpTargetKvLayerView<'_>,
    position_count: usize,
    window_start: usize,
) -> Result<()> {
    let suffix_end = window_start
        .checked_add(position_count)
        .ok_or_else(|| invalid("attention suffix end overflow"))?;
    if suffix_end != kv.logical_len() {
        return Err(invalid(format!(
            "attention compact suffix [{window_start}, {suffix_end}) does not end at physical logical K {}",
            kv.logical_len()
        )));
    }
    if kv.logical_len() > kv.kv_stride() {
        return Err(invalid(format!(
            "attention physical logical K {} exceeds KV stride {}",
            kv.logical_len(),
            kv.kv_stride()
        )));
    }
    let kv_head_stride = kv
        .kv_stride()
        .checked_mul(kv.head_dim())
        .ok_or_else(|| invalid("KV head stride overflow"))?;
    let kv_base_offset = window_start
        .checked_mul(kv.head_dim())
        .ok_or_else(|| invalid("KV base offset overflow"))?;
    let values = [
        N_HEADS as u32,
        kv.head_dim() as u32,
        position_count as u32,
        (N_HEADS / kv.kv_heads()) as u32,
        1.0f32.to_bits(),
        kv.head_dim() as u32,
        u32::try_from(kv_head_stride).map_err(|_| invalid("KV head stride exceeds u32"))?,
        u32::try_from(kv_base_offset).map_err(|_| invalid("KV base offset exceeds u32"))?,
        u32::try_from(window_start).map_err(|_| invalid("KV compact base exceeds u32"))?,
        u32::try_from(kv.logical_len()).map_err(|_| invalid("KV logical K exceeds u32"))?,
    ];
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            scalar.contents().cast::<u32>(),
            values.len(),
        );
    }
    Ok(())
}

fn target_kv_read_bytes(local_count: usize, full_count: usize) -> Result<u64> {
    let local_per_layer = LOCAL_KV_HEADS
        .checked_mul(LOCAL_HEAD_DIM)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| value.checked_mul(local_count))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid("local target-KV ledger overflow"))?;
    let full = FULL_KV_HEADS
        .checked_mul(FULL_HEAD_DIM)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| value.checked_mul(full_count))
        .ok_or_else(|| invalid("full target-KV ledger overflow"))?;
    Ok((local_per_layer + full) as u64)
}

fn borrowed_target_kv_capacity_bytes(target_kv: &Gemma4MtpTargetKvView<'_>) -> Result<u64> {
    let sliding = target_kv_layer_capacity_bytes(
        target_kv.sliding().kv_heads(),
        target_kv.sliding().kv_stride(),
        target_kv.sliding().head_dim(),
    )?;
    let full = target_kv_layer_capacity_bytes(
        target_kv.full().kv_heads(),
        target_kv.full().kv_stride(),
        target_kv.full().head_dim(),
    )?;
    u64::try_from(
        sliding
            .checked_add(full)
            .ok_or_else(|| invalid("borrowed target-KV capacity sum overflow"))?,
    )
    .map_err(|_| invalid("borrowed target-KV capacity exceeds u64"))
}

fn target_kv_layer_capacity_bytes(
    kv_heads: usize,
    kv_stride: usize,
    head_dim: usize,
) -> Result<usize> {
    kv_heads
        .checked_mul(kv_stride)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| invalid("borrowed target-KV capacity ledger overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORACLE_MANIFEST: &str =
        include_str!("../../qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/manifest.json");
    const EXPECTED_RECURRENCE_ORACLE_SHA256: &str =
        "08cf02bedfec09074eeebaa24b1ffaaa5362badd412b523de6a0d52952e94109";
    const EXPECTED_STAGE_ORACLE_SHA256: &str =
        "f18ed0a4ae9538ed5b41e74c39f9242e9b7a842a7c2d3cc5d7abf9765c4b983e";
    const EXPECTED_STEP1_STAGE_ORACLE_SHA256: &str =
        "0808a6bcbfb0500af0a99eb2ceb621577af97a045499af89e58dbc68f81bb031";
    const EXPECTED_STEP3_STAGE_ORACLE_SHA256: &str =
        "580af32c7c04065d8ba87a4952addea8b292604b3067ee213891cef852be51ff";
    const EXPECTED_STEP4_STAGE_ORACLE_SHA256: &str =
        "c37376fcee30cf3ae682068bf27fb98467d2aaf98d3419075444ed8d3f206c90";
    const EXPECTED_RECURRENCE_GENERATION_RECEIPT_SHA256: &str =
        "573864f252b29b63932a440d9224733e086ae4de42b10caf457da975c623053d";
    const EXPECTED_RECURRENCE_RUN1_LOG_SHA256: &str =
        "654d0c1fa5221056efb9bc4cc02a7f9c5c1c79044124fd23e1d94b076a05a7a5";
    const EXPECTED_RECURRENCE_RUN2_LOG_SHA256: &str =
        "afdebdb0642f130583f3b56d9c1050451541c3da3c545415d62a01bd026184d3";
    const EXPECTED_RECURRENCE_FAILED_ATTEMPTS_LOG_SHA256: &str =
        "a56f1a59faaa14d1760c1f8ea8b06f63a67f9b87de01802707b2c5f5c05fc0b3";

    fn synthetic_official_header() -> Vec<u8> {
        let mut root = serde_json::Map::new();
        root.insert("__metadata__".into(), serde_json::json!({ "format": "pt" }));
        for tensor in EXPECTED_TENSORS {
            root.insert(
                tensor.name.into(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": tensor.shape,
                    "data_offsets": [tensor.start, tensor.end],
                }),
            );
        }
        serde_json::to_vec(&serde_json::Value::Object(root)).unwrap()
    }

    #[test]
    fn pinned_tensor_manifest_is_contiguous_and_full() {
        assert_eq!(EXPECTED_TENSORS.len(), 48);
        assert_eq!(EXPECTED_TENSORS[0].start, 0);
        assert_eq!(
            EXPECTED_TENSORS.last().unwrap().end,
            EXPECTED_PAYLOAD_BYTES as u64
        );
        for pair in EXPECTED_TENSORS.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        let matrix_bytes: u64 = EXPECTED_TENSORS
            .iter()
            .filter(|tensor| tensor.shape.len() == 2)
            .map(|tensor| tensor.end - tensor.start)
            .sum();
        assert_eq!(matrix_bytes, MATRIX_BYTES_PER_PROPOSAL);
    }

    #[test]
    fn device_chain_opt_in_is_explicit_and_malformed_values_fail_closed() {
        assert_eq!(parse_device_chain_opt_in(None), Ok(false));
        assert_eq!(parse_device_chain_opt_in(Some("0")), Ok(false));
        assert_eq!(parse_device_chain_opt_in(Some("FALSE")), Ok(false));
        assert_eq!(parse_device_chain_opt_in(Some("1")), Ok(true));
        assert_eq!(parse_device_chain_opt_in(Some("TrUe")), Ok(true));
        assert!(parse_device_chain_opt_in(Some("yes")).is_err());
        assert!(parse_device_chain_opt_in(Some("")).is_err());
    }

    #[test]
    fn device_chain_embedding_geometry_admits_only_exact_q6k_target() {
        let bytes = VOCAB * (TARGET_HIDDEN / 256) * 210;
        validate_target_embedding_geometry(
            Gemma4MtpTargetEmbeddingFormat::Q6K,
            TARGET_HIDDEN,
            VOCAB,
            4_096,
            bytes,
            4_096 + bytes,
        )
        .unwrap();

        for format in [
            Gemma4MtpTargetEmbeddingFormat::Q4K,
            Gemma4MtpTargetEmbeddingFormat::Q8_0,
            Gemma4MtpTargetEmbeddingFormat::Bf16,
        ] {
            assert!(validate_target_embedding_geometry(
                format,
                TARGET_HIDDEN,
                VOCAB,
                4_096,
                bytes,
                4_096 + bytes,
            )
            .is_err());
        }
        assert!(validate_target_embedding_geometry(
            Gemma4MtpTargetEmbeddingFormat::Q6K,
            TARGET_HIDDEN,
            VOCAB,
            4_097,
            bytes,
            4_097 + bytes,
        )
        .is_err());
        assert!(validate_target_embedding_geometry(
            Gemma4MtpTargetEmbeddingFormat::Q6K,
            TARGET_HIDDEN,
            VOCAB,
            4_096,
            bytes,
            4_096 + bytes - 1,
        )
        .is_err());
    }

    #[test]
    fn strict_parser_accepts_only_the_pinned_manifest() {
        let header = synthetic_official_header();
        let manifest = parse_and_validate_header(&header, EXPECTED_PAYLOAD_BYTES).unwrap();
        assert_eq!(manifest.tensors.len(), EXPECTED_TENSORS.len());
        assert_eq!(
            manifest.tensor("pre_projection.weight").unwrap().shape,
            vec![1_024, 5_632]
        );

        let mut malformed: serde_json::Value = serde_json::from_slice(&header).unwrap();
        malformed["model.layers.3.self_attn.q_proj.weight"]["shape"] =
            serde_json::json!([8_191, 1_024]);
        let malformed = serde_json::to_vec(&malformed).unwrap();
        let error = parse_and_validate_header(&malformed, EXPECTED_PAYLOAD_BYTES).unwrap_err();
        assert!(error.to_string().contains("q_proj.weight mismatch"));
    }

    fn f32_to_bf16_exact(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn cpu_bf16_gemv(input: &[f32], weights: &[u16], rows: usize) -> Vec<f32> {
        let cols = input.len();
        (0..rows)
            .map(|row| {
                let value = input
                    .iter()
                    .enumerate()
                    .map(|(col, input)| {
                        round_to_bf16_f32(*input) * bf16_bits_to_f32(weights[row * cols + col])
                    })
                    .sum();
                round_to_bf16_f32(value)
            })
            .collect()
    }

    fn deterministic_bf16_values(
        count: usize,
        seed: u32,
        exponent_base: u32,
        exponent_span: u32,
    ) -> (Vec<f32>, String) {
        let mut values = Vec::with_capacity(count);
        let mut digest = Sha256::new();
        for index in 0..count {
            let mut state = (index as u32).wrapping_add(seed);
            state ^= state >> 16;
            state = state.wrapping_mul(0x7feb_352d);
            state ^= state >> 15;
            state = state.wrapping_mul(0x846c_a68b);
            state ^= state >> 16;
            let sign = ((state >> 31) & 1) << 15;
            let exponent = (exponent_base + ((state >> 24) % exponent_span)) << 7;
            let mantissa = (state >> 16) & 0x7f;
            let bits = (sign | exponent | mantissa) as u16;
            digest.update(bits.to_le_bytes());
            values.push(bf16_bits_to_f32(bits));
        }
        (values, format!("{:x}", digest.finalize()))
    }

    fn widened_bf16_sha256(values: &[f32]) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(f32_to_bf16_rne_bits(*value).to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn raw_f32_sha256(values: &[f32]) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    // Source-faithful scalar view of pinned ATen SumKernel.cpp's
    // Vec4/ILP4 cascade for one contiguous row. This deliberately does not
    // use iterators/sum(): every statement is an arithmetic-order boundary.
    fn cpu_aten_rms_sum_squares(input: &[f32]) -> f32 {
        assert!(matches!(input.len(), 256 | 512 | 1_024));
        let mut residue = [0.0f32; 16];
        for r in 0..16 {
            let mut row_residue = 0.0f32;
            for slab in (0..input.len()).step_by(256) {
                let mut slab_residue = 0.0f32;
                for item in 0..16 {
                    let value = input[slab + r + item * 16];
                    let square = value * value;
                    slab_residue += square;
                }
                row_residue += slab_residue;
            }
            residue[r] = row_residue;
        }
        let mut partial = [0.0f32; 4];
        for lane in 0..4 {
            let mut value = residue[lane];
            value += residue[4 + lane];
            value += residue[8 + lane];
            value += residue[12 + lane];
            partial[lane] = value;
        }
        let mut sum_squares = partial[0];
        sum_squares += partial[1];
        sum_squares += partial[2];
        sum_squares += partial[3];
        sum_squares
    }

    fn cpu_aten_rms_norm(input: &[f32], weight: &[f32]) -> (f32, f32, Vec<f32>) {
        assert_eq!(input.len(), weight.len());
        let sum_squares = cpu_aten_rms_sum_squares(input);
        let mean_squares = sum_squares / input.len() as f32;
        let stabilized = mean_squares + RMS_EPS;
        let inverse_rms = 1.0f32 / stabilized.sqrt();
        let output = input
            .iter()
            .zip(weight)
            .map(|(input, weight)| {
                let value = *input * inverse_rms;
                value * *weight
            })
            .collect();
        (sum_squares, inverse_rms, output)
    }

    fn run_test_rms_norm(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        input: &[f32],
        weight: &[f32],
        width: usize,
        head_count: usize,
        production: bool,
    ) -> Vec<f32> {
        assert_eq!(input.len(), width * head_count);
        assert_eq!(weight.len(), width);
        let input_buffer = f32_buffer(&kernel.device, input);
        let weight_buffer = f32_buffer(&kernel.device, weight);
        let output_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(input));
        let scalar = shared_buffer(&kernel.device, 8);
        set_rms_scalar(&scalar, width);
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        if production {
            encode_assistant_aten_rms_norm(
                encoder,
                &pipelines.rms_norm_aten_f32,
                &input_buffer,
                &weight_buffer,
                &output_buffer,
                &scalar,
                width,
                head_count,
                0,
                true,
            );
        } else if head_count == 1 {
            encode_rms_norm_f32(
                encoder,
                kernel,
                &input_buffer,
                &weight_buffer,
                &output_buffer,
                &scalar,
            );
        } else {
            let per_head_scalar = shared_buffer(&kernel.device, 12);
            set_qnorm_scalar(&per_head_scalar, width);
            encode_rms_norm_per_head(
                encoder,
                kernel,
                &input_buffer,
                &weight_buffer,
                &output_buffer,
                &per_head_scalar,
                head_count,
                0,
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; input.len()];
        read_buffer_f32(&output_buffer, &mut output);
        output
    }

    fn manifest_string<'a>(manifest: &'a serde_json::Value, path: &[&str]) -> &'a str {
        let mut value = manifest;
        for key in path {
            value = &value[*key];
        }
        value
            .as_str()
            .unwrap_or_else(|| panic!("oracle manifest field {path:?} is not a string"))
    }

    fn oracle_values(
        manifest: &serde_json::Value,
        name: &str,
        count: usize,
        seed: u32,
        exponent_base: u32,
        exponent_span: u32,
    ) -> Vec<f32> {
        let (values, hash) = deterministic_bf16_values(count, seed, exponent_base, exponent_span);
        assert_eq!(
            hash,
            manifest_string(manifest, &["input_tensor_sha256", name]),
            "target-free oracle input generator drifted for {name}"
        );
        values
    }

    #[test]
    fn bf16_rne_ties_choose_even_mantissa() {
        let even_low = f32::from_bits(0x3f80_8000);
        let odd_low = f32::from_bits(0x3f81_8000);
        assert_eq!(f32_to_bf16_rne_bits(even_low), 0x3f80);
        assert_eq!(f32_to_bf16_rne_bits(odd_low), 0x3f82);
        assert_eq!(f32_to_bf16_rne_bits(-even_low), 0xbf80);
        assert_eq!(f32_to_bf16_rne_bits(-odd_low), 0xbf82);
        assert_eq!(f32_to_bf16_rne_bits(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_rne_bits(f32::NEG_INFINITY), 0xff80);
        assert!(round_to_bf16_f32(f32::NAN).is_nan());
    }

    #[test]
    fn bf16_decode_and_cpu_gemv_cover_signed_rows() {
        let input = [1.0f32, 2.0, -3.0, 4.0];
        let weights: Vec<u16> = [1.0f32, 0.5, -2.0, 3.0, -1.0, 2.0, 1.0, -0.5]
            .into_iter()
            .map(f32_to_bf16_exact)
            .collect();
        assert_eq!(cpu_bf16_gemv(&input, &weights, 2), vec![20.0, -2.0]);
    }

    #[test]
    fn target_kv_ledgers_distinguish_unique_capacity_from_repeated_reads() {
        let capacity_1024 = target_kv_layer_capacity_bytes(LOCAL_KV_HEADS, 1_024, LOCAL_HEAD_DIM)
            .unwrap()
            + target_kv_layer_capacity_bytes(FULL_KV_HEADS, 1_024, FULL_HEAD_DIM).unwrap();
        assert_eq!(capacity_1024, 25_165_824);
        assert_eq!(target_kv_read_bytes(1_024, 1_024).unwrap(), 58_720_256);
        assert_eq!(target_kv_read_bytes(1_025, 1_031).unwrap(), 58_826_752);

        // The target-free warmup owns one position of each unique K/V pair;
        // its three local assistant layers reread the same layer-28 buffers.
        let warm_capacity = target_kv_layer_capacity_bytes(LOCAL_KV_HEADS, 1, LOCAL_HEAD_DIM)
            .unwrap()
            + target_kv_layer_capacity_bytes(FULL_KV_HEADS, 1, FULL_HEAD_DIM).unwrap();
        assert_eq!(warm_capacity, 24_576);
        assert_eq!(target_kv_read_bytes(1, 1).unwrap(), 57_344);
    }

    #[test]
    fn assistant_sliding_window_uses_the_inclusive_mask_boundary() {
        assert_eq!(LOCAL_WINDOW, 1_024);
        assert_eq!(LOCAL_ATTENTION_SPAN, 1_025);
        assert_eq!(assistant_local_attention_bounds(0), (0, 0));
        assert_eq!(assistant_local_attention_bounds(1), (0, 1));
        assert_eq!(assistant_local_attention_bounds(1_024), (0, 1_024));
        assert_eq!(assistant_local_attention_bounds(1_025), (0, 1_025));
        // Pinned recurrence geometry: six positions are masked and the final
        // 1,025 positions are visible to each local assistant layer.
        assert_eq!(assistant_local_attention_bounds(1_031), (6, 1_025));

        // Stage evidence remains pinned to its deliberately cropped 1,024
        // local columns without shrinking the arithmetic span back to 1,024.
        assert_eq!(stage_attention_checkpoint_columns(0, 1_025), 1_024);
        assert_eq!(stage_attention_checkpoint_columns(2, 1_025), 1_024);
        assert_eq!(stage_attention_checkpoint_columns(3, 1_031), 1_031);
    }

    #[test]
    fn proportional_rope_preserves_full_head_pair_stride() {
        assert_eq!(FULL_HEAD_DIM / 2, 256);
        assert_eq!(FULL_ROPE_ACTIVE_PAIRS, 64);
        let (cos, sin) =
            rope_table_values(1_030, 1_000_000.0, FULL_HEAD_DIM, FULL_ROPE_ACTIVE_PAIRS);
        assert_eq!(cos.len(), 256);
        assert_eq!(sin.len(), 256);
        assert!(cos[FULL_ROPE_ACTIVE_PAIRS..]
            .iter()
            .all(|value| value.to_bits() == 1.0f32.to_bits()));
        assert!(sin[FULL_ROPE_ACTIVE_PAIRS..]
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits()));

        // Exercise the exact split-half geometry used by encode_rope.  The
        // first active value must couple to dimension 256; dimension 64 is an
        // inactive first-half value and therefore remains untouched.
        let mut head = vec![0.0f32; FULL_HEAD_DIM];
        head[0] = 1.0;
        head[64] = 3.0;
        head[256] = 2.0;
        head[320] = 4.0;
        for pair in 0..FULL_HEAD_DIM / 2 {
            let partner = pair + FULL_HEAD_DIM / 2;
            let (x0, x1) = (head[pair], head[partner]);
            head[pair] = x0 * cos[pair] - x1 * sin[pair];
            head[partner] = x0 * sin[pair] + x1 * cos[pair];
        }
        assert_eq!(head[0], cos[0] - 2.0 * sin[0]);
        assert_eq!(head[256], sin[0] + 2.0 * cos[0]);
        assert_eq!(head[64].to_bits(), 3.0f32.to_bits());
        assert_eq!(head[320].to_bits(), 4.0f32.to_bits());
    }

    #[test]
    fn production_rms_matches_pinned_cascade_at_n256_n512_n1024() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let fixtures: &[(usize, usize, &[u32], &[u32], &str)] = &[
            (
                256,
                4,
                &[0x49a9_ffb5, 0x49ab_550f, 0x49aa_ba34, 0x49ab_86b3],
                &[0x3c5e_2335, 0x3c5d_457e, 0x3c5d_a9c1, 0x3c5d_2577],
                "bd092c5928aaf23d0a6e843e35dc999711a2313194e8ded6a888aa97834d865b",
            ),
            (
                512,
                4,
                &[0x4a2a_72c9, 0x4a2b_604b, 0x4a2a_6548, 0x4a2b_31cf],
                &[0x3c5d_d82b, 0x3c5d_3e3d, 0x3c5d_e0f6, 0x3c5d_5c45],
                "0c76f881bf3e83cf03447e58d16a3d832e66151548cdc36a688c578bcbb8c969",
            ),
            (
                1_024,
                1,
                &[0x4aaa_b72d],
                &[0x3c5d_abb7],
                "4751b6ef2bbc38d10696d2a775d530a40e61dba97356f3547cf908e7f7f1cb29",
            ),
        ];

        for &(width, head_count, expected_sums, expected_inverses, expected_sha256) in fixtures {
            let mut input = Vec::with_capacity(width * head_count);
            let mut expected = Vec::with_capacity(width * head_count);
            let weight = vec![1.0f32; width];
            for row in 0..head_count {
                let row_input: Vec<f32> = (0..width)
                    .map(|index| {
                        let bits = (index as u32)
                            .wrapping_mul(0xbc50_8947)
                            .wrapping_add(0xb744_855e)
                            .wrapping_add((row as u32).wrapping_mul(0x9e37_79b9))
                            & 0xffff;
                        (bits as i32 - 32_768) as f32 / 256.0f32
                    })
                    .collect();
                let (sum_squares, inverse_rms, row_expected) =
                    cpu_aten_rms_norm(&row_input, &weight);
                assert_eq!(
                    sum_squares.to_bits(),
                    expected_sums[row],
                    "pinned ATen sum-of-squares drift at width {width} row {row}"
                );
                assert_eq!(
                    inverse_rms.to_bits(),
                    expected_inverses[row],
                    "pinned ATen inverse-RMS drift at width {width} row {row}"
                );
                input.extend_from_slice(&row_input);
                expected.extend_from_slice(&row_expected);
            }
            assert_eq!(
                raw_f32_sha256(&expected),
                expected_sha256,
                "pinned ATen raw-f32 fixture drift at width {width}"
            );

            let production =
                run_test_rms_norm(kernel, &pipelines, &input, &weight, width, head_count, true);
            assert_eq!(
                production
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "production ATen RMS Metal path differs at width {width}"
            );
            assert_eq!(raw_f32_sha256(&production), expected_sha256);

            let legacy = run_test_rms_norm(
                kernel, &pipelines, &input, &weight, width, head_count, false,
            );
            assert_ne!(
                legacy
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "fixture failed to discriminate legacy RMS at width {width}"
            );
        }
    }

    #[test]
    fn metal_bf16_gemv_matches_synthetic_reference() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let cols = 33usize;
        let rows = 5usize;
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let weights: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let value = ((i * 3 + 1) % 9) as f32 - 4.0;
                f32_to_bf16_exact(value)
            })
            .collect();
        let expected = cpu_bf16_gemv(&input, &weights, rows);
        let weight_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(&weights[..]));
        unsafe {
            std::ptr::copy_nonoverlapping(
                weights.as_ptr().cast::<u8>(),
                weight_buffer.contents().cast::<u8>(),
                std::mem::size_of_val(&weights[..]),
            );
        }
        let input_buffer = f32_buffer(&kernel.device, &input);
        let output_buffer = shared_buffer(&kernel.device, rows * std::mem::size_of::<f32>());
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_bf16_gemv(
            encoder,
            &pipelines.bf16_gemv,
            &weight_buffer,
            &input_buffer,
            &output_buffer,
            TensorRef {
                absolute_offset: 0,
                rows: rows as u32,
                cols: cols as u32,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut actual = vec![0.0f32; rows];
        read_buffer_f32(&output_buffer, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn production_bf16_gemv_uses_adjacent_final_pair_reduction() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        // Target-free discriminator captured from the pinned Torch 2.13.0
        // AArch64 BF16 dot path. The first 16/8/4 folds are shared; only the
        // final vaddvq_f32 adjacent-pair order distinguishes these raw bits.
        let input_bits: [u16; 32] = [
            48425, 48424, 48227, 49571, 15509, 15627, 16782, 16759, 49363, 48564, 49009, 15508,
            15435, 48577, 15588, 48362, 49291, 49470, 48707, 16491, 49402, 15860, 48182, 16476,
            16370, 48309, 16894, 49070, 16387, 49507, 48587, 15994,
        ];
        let weight_bits: [u16; 32] = [
            16407, 48402, 48991, 16752, 48617, 15793, 16632, 49229, 16349, 49370, 17012, 48186,
            16164, 48969, 48221, 48370, 48213, 49588, 49446, 15976, 48272, 49660, 48705, 49559,
            48488, 16638, 16351, 48578, 16314, 16126, 16328, 49470,
        ];
        let input: Vec<f32> = input_bits.iter().copied().map(bf16_bits_to_f32).collect();
        let weight_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(&weight_bits));
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight_bits.as_ptr().cast::<u8>(),
                weight_buffer.contents().cast::<u8>(),
                std::mem::size_of_val(&weight_bits),
            );
        }
        let input_buffer = f32_buffer(&kernel.device, &input);
        let run = |pipeline: &ComputePipelineState| {
            let output = shared_buffer(&kernel.device, std::mem::size_of::<f32>());
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode_bf16_gemv(
                encoder,
                pipeline,
                &weight_buffer,
                &input_buffer,
                &output,
                TensorRef {
                    absolute_offset: 0,
                    rows: 1,
                    cols: 32,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
            let mut value = [0.0f32; 1];
            read_buffer_f32(&output, &mut value);
            f32_to_bf16_rne_bits(value[0])
        };

        assert_eq!(run(&pipelines.bf16_gemv), 49_680);
        assert_eq!(run(&pipelines.bf16_gemv_legacy), 49_681);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_test_attention_scores(
        kernel: &MetalLinearKernel,
        pipeline: &ComputePipelineState,
        query: &[f32],
        keys: &[f32],
        head_count: usize,
        head_dim: usize,
        position_count: usize,
        group: usize,
        position_stride: usize,
        kv_head_stride: usize,
        kv_base_offset: usize,
    ) -> Vec<f32> {
        let query_buffer = f32_buffer(&kernel.device, query);
        let key_buffer = f32_buffer(&kernel.device, keys);
        let scores = shared_buffer(
            &kernel.device,
            head_count * position_count * std::mem::size_of::<f32>(),
        );
        let scalar = shared_buffer(&kernel.device, 32);
        let words = [
            head_count as u32,
            head_dim as u32,
            position_count as u32,
            group as u32,
            1.0f32.to_bits(),
            position_stride as u32,
            kv_head_stride as u32,
            kv_base_offset as u32,
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                scalar.contents().cast::<u32>(),
                words.len(),
            );
        }
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&key_buffer), 0);
        encoder.set_buffer(2, Some(&scores), 0);
        for index in 0..8u64 {
            encoder.set_buffer(3 + index, Some(&scalar), index * 4);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: head_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; head_count * position_count];
        read_buffer_f32(&scores, &mut output);
        output
    }

    #[test]
    fn production_qk_uses_pinned_float4_reduction_for_k32_and_k512() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let query_bits: [u16; 32] = [
            48425, 48424, 48227, 49571, 15509, 15627, 16782, 16759, 49363, 48564, 49009, 15508,
            15435, 48577, 15588, 48362, 49291, 49470, 48707, 16491, 49402, 15860, 48182, 16476,
            16370, 48309, 16894, 49070, 16387, 49507, 48587, 15994,
        ];
        let key_bits: [u16; 32] = [
            16407, 48402, 48991, 16752, 48617, 15793, 16632, 49229, 16349, 49370, 17012, 48186,
            16164, 48969, 48221, 48370, 48213, 49588, 49446, 15976, 48272, 49660, 48705, 49559,
            48488, 16638, 16351, 48578, 16314, 16126, 16328, 49470,
        ];
        for head_dim in [32usize, 512] {
            let mut query = vec![0.0f32; head_dim];
            let mut keys = vec![0.0f32; head_dim];
            for index in 0..32 {
                query[index] = bf16_bits_to_f32(query_bits[index]);
                keys[index] = bf16_bits_to_f32(key_bits[index]);
            }
            let run = |pipeline: &ComputePipelineState| {
                let scores = run_test_attention_scores(
                    kernel, pipeline, &query, &keys, 1, head_dim, 1, 1, head_dim, head_dim, 0,
                );
                f32_to_bf16_rne_bits(scores[0])
            };
            assert_eq!(run(&pipelines.attention_scores_bf16), 49_680);
            assert_eq!(run(&pipelines.attention_scores_legacy_bf16), 49_681);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_test_attention_context(
        kernel: &MetalLinearKernel,
        pipeline: &ComputePipelineState,
        probabilities: &[f32],
        values: &[f32],
        head_count: usize,
        head_dim: usize,
        position_count: usize,
        group: usize,
        position_stride: usize,
        kv_head_stride: usize,
        kv_base_offset: usize,
        compact_base: usize,
        physical_logical_k: usize,
    ) -> Vec<f32> {
        let probability_buffer = f32_buffer(&kernel.device, probabilities);
        let value_buffer = f32_buffer(&kernel.device, values);
        let output = shared_buffer(
            &kernel.device,
            head_count * head_dim * std::mem::size_of::<f32>(),
        );
        assert_eq!(compact_base + position_count, physical_logical_k);
        let scalar = shared_buffer(&kernel.device, 40);
        let words = [
            head_count as u32,
            head_dim as u32,
            position_count as u32,
            group as u32,
            1.0f32.to_bits(),
            position_stride as u32,
            kv_head_stride as u32,
            kv_base_offset as u32,
            compact_base as u32,
            physical_logical_k as u32,
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                scalar.contents().cast::<u32>(),
                words.len(),
            );
        }
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&value_buffer), 0);
        encoder.set_buffer(1, Some(&probability_buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_buffer(3, Some(&scalar), 0);
        encoder.set_buffer(4, Some(&scalar), 4);
        encoder.set_buffer(5, Some(&scalar), 8);
        encoder.set_buffer(6, Some(&scalar), 12);
        encoder.set_buffer(7, Some(&scalar), 20);
        encoder.set_buffer(8, Some(&scalar), 24);
        encoder.set_buffer(9, Some(&scalar), 28);
        encoder.set_buffer(10, Some(&scalar), 32);
        encoder.set_buffer(11, Some(&scalar), 36);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: head_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut result = vec![0.0f32; head_count * head_dim];
        read_buffer_f32(&output, &mut result);
        result
    }

    #[test]
    fn production_context_uses_explicit_physical_k_phase_and_tail() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let probability_bits: [u16; 32] = [
            15428, 15592, 15899, 15457, 16469, 16013, 15650, 16085, 16830, 15616, 16150, 16608,
            15489, 16114, 15499, 15524, 15621, 15982, 16634, 15992, 15438, 16123, 15824, 15461,
            16751, 16764, 16491, 16008, 16686, 16789, 16323, 15701,
        ];
        let value_bits: [u16; 32] = [
            49252, 16884, 49366, 15766, 48218, 16143, 49083, 48423, 49014, 15394, 48517, 48837,
            49515, 15504, 49558, 16130, 48359, 16067, 48213, 48774, 16144, 16060, 16654, 49230,
            16714, 48800, 49182, 16610, 48162, 49413, 49139, 48903,
        ];
        let make_inputs = |compact_base: usize, count: usize| {
            let physical_k = compact_base + count;
            let mut probabilities = vec![0.0f32; count];
            let mut values = vec![0.0f32; physical_k];
            for index in 0..32 {
                probabilities[index] = bf16_bits_to_f32(probability_bits[index]);
                values[compact_base + index] = bf16_bits_to_f32(value_bits[index]);
            }
            (probabilities, values)
        };
        let run = |pipeline: &ComputePipelineState,
                   probabilities: &[f32],
                   values: &[f32],
                   compact_base: usize| {
            let result = run_test_attention_context(
                kernel,
                pipeline,
                probabilities,
                values,
                1,
                1,
                probabilities.len(),
                1,
                1,
                values.len(),
                compact_base,
                compact_base,
                values.len(),
            );
            f32_to_bf16_rne_bits(result[0])
        };

        // Local K=1,025 begins at absolute position six. This tail adjustment
        // puts the correct absolute-position lane phase on the opposite BF16
        // side from both compact-index phasing and scalar sequential sum.
        let (mut local_probabilities, mut local_values) = make_inputs(6, 1_025);
        local_probabilities[1_022] = 1.0;
        local_values[1_028] = bf16_bits_to_f32(0x40d5);
        assert_eq!(
            run(
                &pipelines.attention_context_bf16,
                &local_probabilities,
                &local_values,
                6,
            ),
            0x3c2a
        );
        assert_eq!(
            run(
                &pipelines.attention_context_legacy_bf16,
                &local_probabilities,
                &local_values,
                6,
            ),
            0x3c2b
        );

        // Full K=1,031 has a three-element physical tail. Equal and opposite
        // finite BF16 terms at tail positions 1,028 and 1,030 distinguish
        // ATen's all-to-p0 rule from modulo-four tail routing and sequential
        // scalar accumulation.
        let (mut full_probabilities, mut full_values) = make_inputs(0, 1_031);
        full_probabilities[1_028] = 1.0;
        full_values[1_028] = bf16_bits_to_f32(0xe76c);
        full_probabilities[1_030] = 1.0;
        full_values[1_030] = bf16_bits_to_f32(0x676c);
        assert_eq!(
            run(
                &pipelines.attention_context_bf16,
                &full_probabilities,
                &full_values,
                0,
            ),
            0xc32c
        );
        assert_eq!(
            run(
                &pipelines.attention_context_legacy_bf16,
                &full_probabilities,
                &full_values,
                0,
            ),
            0x0000
        );
    }

    fn run_test_strict_softmax(
        kernel: &MetalLinearKernel,
        pipeline: &ComputePipelineState,
        values: &[f32],
    ) -> Vec<f32> {
        assert!(values.len() >= 4);
        let rounded: Vec<f32> = values.iter().copied().map(round_to_bf16_f32).collect();
        let scores = f32_buffer(&kernel.device, &rounded);
        let scalar = shared_buffer(&kernel.device, 12);
        unsafe {
            let words = scalar.contents().cast::<u32>();
            *words = 1;
            *words.add(1) = 0;
            *words.add(2) = values.len() as u32;
        }
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&scores), 0);
        encoder.set_buffer(1, Some(&scalar), 0);
        encoder.set_buffer(2, Some(&scalar), 8);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; values.len()];
        read_buffer_f32(&scores, &mut output);
        output
    }

    #[test]
    fn metal_test_strict_softmax_matches_pinned_aten_bf16_fixtures() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let tree33: Vec<f32> = (0..33)
            .map(|index| ((((index * 37) % 67) as i64 - 33) as f64 / 8.0) as f32)
            .collect();
        let formula = |index: usize| {
            ((((index * 73 + 19) % 509) as i64 - 254) as f64 / 32.0
                + ((index % 7) as i64 - 3) as f64 / 256.0) as f32
        };
        let sliding1024: Vec<f32> = (0..1_024).map(formula).collect();
        let full1031: Vec<f32> = (0..1_031).map(formula).collect();
        let fixtures = [
            (
                "tail5",
                vec![0.0, -0.5, -1.0, -2.0, -10.0],
                Some("dc4f1196cc621e5fe8d69bc049e4cddc114b7cc7e23b0e98fabbd6d0188201be"),
            ),
            (
                "tail7",
                vec![0.0, -0.125, -0.5, -1.0, -2.0, -4.0, -16.0],
                Some("f25b573e77944f83cc321ae54f9c2187009315d6e40c0c25123a31cc213875d6"),
            ),
            (
                "tree33",
                tree33,
                Some("ab3e7cb1bad654bae8567a94f32bfa0f12441df81ab572d931df8b41a0271c2d"),
            ),
            (
                "sliding1024",
                sliding1024,
                Some("d5f8a24863233b411d7bec7edc3784f8555d3c624a5d16b53e4e1d9101f3ac9d"),
            ),
            (
                "full1031",
                full1031,
                Some("e65b9d2402ef32bfec8783a3d12a83a807f98d73aa8ede9f6fce0fc83bfd9575"),
            ),
        ];
        for (name, values, expected_sha256) in fixtures {
            let geometry = run_test_strict_softmax(
                kernel,
                &pipelines.attention_softmax_aten_geometry_f32,
                &values,
            );
            let sleef = run_test_strict_softmax(
                kernel,
                &pipelines.attention_softmax_aten_sleef_f32,
                &values,
            );
            let geometry_sha256 = widened_bf16_sha256(&geometry);
            let sleef_sha256 = widened_bf16_sha256(&sleef);
            eprintln!(
                "MTP_STRICT_SOFTMAX_SYNTHETIC name={name} geometry_sha256={geometry_sha256} sleef_sha256={sleef_sha256} expected_sha256={}",
                expected_sha256.unwrap_or("pending-full-reference-hash")
            );
            if let Some(expected_sha256) = expected_sha256 {
                assert_eq!(
                    sleef_sha256, expected_sha256,
                    "pinned ATen BF16 SLEEF softmax drift for {name}"
                );
            }
        }
    }

    #[derive(Deserialize)]
    struct StageOracleDocument {
        schema_version: u32,
        #[serde(default)]
        authority: Option<StageOracleAuthority>,
        stages: Vec<StageOracleEntry>,
    }

    #[derive(Deserialize)]
    struct StageOracleAuthority {
        #[serde(default)]
        model_sha256: Option<String>,
        #[serde(default)]
        recurrence_oracle_sha256: Option<String>,
        #[serde(default)]
        recurrence_step: Option<usize>,
    }

    #[derive(Deserialize)]
    struct StageOracleEntry {
        name: String,
        elements: usize,
        bf16_sha256: String,
        bf16_bits: Vec<u16>,
    }

    #[test]
    #[ignore = "reads one bounded official BF16 norm and the external step-4 stage oracle"]
    fn official_step4_layer0_production_rms_matches_input_norm_oracle() {
        let model_path = std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH");
        let oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON");
        let oracle_bytes = std::fs::read(&oracle_path).expect("read step-4 stage oracle");
        assert_eq!(
            sha256_hex(&oracle_bytes),
            EXPECTED_STEP4_STAGE_ORACLE_SHA256,
            "step-4 stage oracle bytes changed"
        );
        let oracle: StageOracleDocument =
            serde_json::from_slice(&oracle_bytes).expect("parse step-4 stage oracle");
        assert_eq!(oracle.schema_version, 1);
        let authority = oracle.authority.as_ref().expect("stage oracle authority");
        assert_eq!(authority.model_sha256.as_deref(), Some(EXPECTED_SHA256));
        assert_eq!(
            authority.recurrence_oracle_sha256.as_deref(),
            Some(EXPECTED_RECURRENCE_ORACLE_SHA256)
        );
        assert_eq!(authority.recurrence_step, Some(4));
        let stage = |name: &str| {
            oracle
                .stages
                .iter()
                .find(|stage| stage.name == name)
                .unwrap_or_else(|| panic!("missing stage {name}"))
        };
        let input_stage = stage("pre_projection");
        let output_stage = stage("layer.0.input_norm");
        assert_eq!(input_stage.elements, ASSISTANT_HIDDEN);
        assert_eq!(output_stage.elements, ASSISTANT_HIDDEN);
        assert_eq!(
            input_stage.bf16_sha256,
            "cb28fc9f1e35e102b473f0e1970168caa9c18b5c96dc6bc5e66897d02ee5d6b0"
        );
        assert_eq!(
            output_stage.bf16_sha256,
            "690299b97eab047be36f92605f99664b4c0b3f4b274a7ec586451376d5706058"
        );

        let mapping = GgufWireMmap::map(&model_path).expect("map official assistant weights");
        let manifest = parse_official_manifest(&mapping).expect("validate official tensor map");
        let weight = decode_bf16(
            &mapping,
            manifest
                .tensor("model.layers.0.input_layernorm.weight")
                .expect("official layer-0 input norm"),
        )
        .expect("decode official layer-0 input norm");
        assert_eq!(weight.len(), ASSISTANT_HIDDEN);
        let input: Vec<f32> = input_stage
            .bf16_bits
            .iter()
            .copied()
            .map(bf16_bits_to_f32)
            .collect();
        let (sum_squares, inverse_rms, cpu_output) = cpu_aten_rms_norm(&input, &weight);
        assert_eq!(sum_squares.to_bits(), 0x45d1_4d0a);
        assert_eq!(inverse_rms.to_bits(), 0x3ec8_32a4);
        let cpu_bits: Vec<u16> = cpu_output
            .iter()
            .copied()
            .map(f32_to_bf16_rne_bits)
            .collect();
        assert_eq!(cpu_bits, output_stage.bf16_bits);

        let kernel = metal_linear_kernel().expect("Metal kernel");
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let legacy = run_test_rms_norm(
            kernel,
            &pipelines,
            &input,
            &weight,
            ASSISTANT_HIDDEN,
            1,
            false,
        );
        assert_eq!(
            widened_bf16_sha256(&legacy),
            "6ac672b455aa9da46dbac4b3ea80cb5fde79f2978bf18c3235e8addf4a459ba4"
        );
        let legacy_bits: Vec<u16> = legacy.iter().copied().map(f32_to_bf16_rne_bits).collect();
        let differences: Vec<usize> = legacy_bits
            .iter()
            .zip(&output_stage.bf16_bits)
            .enumerate()
            .filter_map(|(index, (actual, expected))| (actual != expected).then_some(index))
            .collect();
        assert_eq!(differences, vec![771]);
        assert_eq!(input_stage.bf16_bits[771], 0xc093);
        assert_eq!(f32_to_bf16_rne_bits(weight[771]), 0x4053);
        assert_eq!(legacy_bits[771], 0xc0be);
        assert_eq!(output_stage.bf16_bits[771], 0xc0bd);

        let production = run_test_rms_norm(
            kernel,
            &pipelines,
            &input,
            &weight,
            ASSISTANT_HIDDEN,
            1,
            true,
        );
        let production_bits: Vec<u16> = production
            .iter()
            .copied()
            .map(f32_to_bf16_rne_bits)
            .collect();
        assert_eq!(production_bits, output_stage.bf16_bits);
        assert_eq!(widened_bf16_sha256(&production), output_stage.bf16_sha256);
        eprintln!(
            "MTP_ATEN_RMS_OFFICIAL name=layer.0.input_norm elements={} legacy_sha256={} production_sha256={} expected_sha256={} mismatch_index=771 legacy_bits=c0be expected_bits=c0bd",
            output_stage.elements,
            widened_bf16_sha256(&legacy),
            widened_bf16_sha256(&production),
            output_stage.bf16_sha256,
        );
    }

    #[test]
    #[ignore = "reads the external step-3 stage oracle; no model weights are loaded"]
    fn official_step3_layer0_production_qk_scores_match_oracle() {
        let oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON");
        let oracle_bytes = std::fs::read(&oracle_path).expect("read step-3 stage oracle");
        assert_eq!(
            sha256_hex(&oracle_bytes),
            EXPECTED_STEP3_STAGE_ORACLE_SHA256,
            "step-3 stage oracle bytes changed"
        );
        let oracle: StageOracleDocument =
            serde_json::from_slice(&oracle_bytes).expect("parse step-3 stage oracle");
        assert_eq!(oracle.schema_version, 1);
        let authority = oracle.authority.as_ref().expect("stage oracle authority");
        assert_eq!(authority.model_sha256.as_deref(), Some(EXPECTED_SHA256));
        assert_eq!(
            authority.recurrence_oracle_sha256.as_deref(),
            Some(EXPECTED_RECURRENCE_ORACLE_SHA256)
        );
        assert_eq!(authority.recurrence_step, Some(3));
        let stage = |name: &str| {
            oracle
                .stages
                .iter()
                .find(|stage| stage.name == name)
                .unwrap_or_else(|| panic!("missing stage {name}"))
        };
        let query_stage = stage("layer.0.q_rope");
        let score_stage = stage("layer.0.attention_scores");
        assert_eq!(query_stage.elements, N_HEADS * LOCAL_HEAD_DIM);
        assert_eq!(score_stage.elements, N_HEADS * LOCAL_WINDOW);
        assert_eq!(
            score_stage.bf16_sha256,
            "3aabb57dfcb21ca89ccc8b9b8c84975b7abc2dd0ff5db1b998de07c6085dd7ed"
        );

        let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
        let kv_len = 1_031usize;
        let keys = oracle_values(
            &manifest,
            "sliding_key_layer28",
            LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
            0x3141_5926,
            125,
            3,
        );
        let query: Vec<f32> = query_stage
            .bf16_bits
            .iter()
            .copied()
            .map(bf16_bits_to_f32)
            .collect();
        let (local_start, local_count) = assistant_local_attention_bounds(kv_len);
        assert_eq!((local_start, local_count), (6, 1_025));
        let kernel = metal_linear_kernel().expect("Metal kernel");
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let run = |pipeline: &ComputePipelineState| {
            let full = run_test_attention_scores(
                kernel,
                pipeline,
                &query,
                &keys,
                N_HEADS,
                LOCAL_HEAD_DIM,
                local_count,
                N_HEADS / LOCAL_KV_HEADS,
                LOCAL_HEAD_DIM,
                kv_len * LOCAL_HEAD_DIM,
                local_start * LOCAL_HEAD_DIM,
            );
            let mut cropped = Vec::with_capacity(N_HEADS * LOCAL_WINDOW);
            for head in 0..N_HEADS {
                let row = &full[head * local_count..(head + 1) * local_count];
                cropped.extend_from_slice(&row[local_count - LOCAL_WINDOW..]);
            }
            cropped
        };

        let legacy = run(&pipelines.attention_scores_legacy_bf16);
        assert_eq!(
            widened_bf16_sha256(&legacy),
            "21f9fca548f4954427345af24baf5529bf151814e2e33e6ba517cc7124ac9cc1"
        );
        let legacy_bits: Vec<u16> = legacy.iter().copied().map(f32_to_bf16_rne_bits).collect();
        let differences: Vec<usize> = legacy_bits
            .iter()
            .zip(&score_stage.bf16_bits)
            .enumerate()
            .filter_map(|(index, (actual, expected))| (actual != expected).then_some(index))
            .collect();
        assert_eq!(differences, vec![15 * LOCAL_WINDOW + 572]);
        assert_eq!(legacy_bits[differences[0]], 0xb7c8);
        assert_eq!(score_stage.bf16_bits[differences[0]], 0xb7c0);

        let production = run(&pipelines.attention_scores_bf16);
        let production_bits: Vec<u16> = production
            .iter()
            .copied()
            .map(f32_to_bf16_rne_bits)
            .collect();
        assert_eq!(production_bits, score_stage.bf16_bits);
        assert_eq!(widened_bf16_sha256(&production), score_stage.bf16_sha256);
    }

    #[test]
    #[ignore = "reads the external step-3 stage oracle; no model weights are loaded"]
    fn official_step3_layer0_production_context_matches_oracle() {
        let oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON");
        let oracle_bytes = std::fs::read(&oracle_path).expect("read step-3 stage oracle");
        assert_eq!(
            sha256_hex(&oracle_bytes),
            EXPECTED_STEP3_STAGE_ORACLE_SHA256,
            "step-3 stage oracle bytes changed"
        );
        let oracle: StageOracleDocument =
            serde_json::from_slice(&oracle_bytes).expect("parse step-3 stage oracle");
        let authority = oracle.authority.as_ref().expect("stage oracle authority");
        assert_eq!(authority.model_sha256.as_deref(), Some(EXPECTED_SHA256));
        assert_eq!(
            authority.recurrence_oracle_sha256.as_deref(),
            Some(EXPECTED_RECURRENCE_ORACLE_SHA256)
        );
        assert_eq!(authority.recurrence_step, Some(3));
        let stage = |name: &str| {
            oracle
                .stages
                .iter()
                .find(|stage| stage.name == name)
                .unwrap_or_else(|| panic!("missing stage {name}"))
        };
        let query_stage = stage("layer.0.q_rope");
        let score_stage = stage("layer.0.attention_scores");
        let probability_stage = stage("layer.0.attention_probs");
        let context_stage = stage("layer.0.attention_context");
        assert_eq!(context_stage.elements, N_HEADS * LOCAL_HEAD_DIM);
        assert_eq!(
            context_stage.bf16_sha256,
            "3b52f5b48259ed1c88c3aae3aa277cfe0a4c2d51226ca1d15ff854170fdb55e9"
        );

        let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
        let kv_len = 1_031usize;
        let keys = oracle_values(
            &manifest,
            "sliding_key_layer28",
            LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
            0x3141_5926,
            125,
            3,
        );
        let values = oracle_values(
            &manifest,
            "sliding_value_layer28",
            LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
            0x2718_2818,
            125,
            3,
        );
        let query: Vec<f32> = query_stage
            .bf16_bits
            .iter()
            .copied()
            .map(bf16_bits_to_f32)
            .collect();
        let (local_start, local_count) = assistant_local_attention_bounds(kv_len);
        assert_eq!((local_start, local_count), (6, 1_025));
        let kernel = metal_linear_kernel().expect("Metal kernel");
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        let scores = run_test_attention_scores(
            kernel,
            &pipelines.attention_scores_bf16,
            &query,
            &keys,
            N_HEADS,
            LOCAL_HEAD_DIM,
            local_count,
            N_HEADS / LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            kv_len * LOCAL_HEAD_DIM,
            local_start * LOCAL_HEAD_DIM,
        );
        let crop_bits = |rows: &[f32]| {
            let mut bits = Vec::with_capacity(N_HEADS * LOCAL_WINDOW);
            for head in 0..N_HEADS {
                let row = &rows[head * local_count..(head + 1) * local_count];
                bits.extend(
                    row[local_count - LOCAL_WINDOW..]
                        .iter()
                        .copied()
                        .map(f32_to_bf16_rne_bits),
                );
            }
            bits
        };
        assert_eq!(crop_bits(&scores), score_stage.bf16_bits);

        // The stage oracle deliberately retains only the final 1,024 columns.
        // Prepend the separately pinned CPU-oracle probability for absolute
        // position 6 in each head so this test isolates context reduction and
        // does not depend on either experimental Metal softmax candidate.
        let omitted_probability_bits: [u16; N_HEADS] = [
            0x1699, 0x0a17, 0x0bc3, 0x170f, 0x39aa, 0x13bd, 0x1da4, 0x1d06, 0x1439, 0x1f23, 0x044f,
            0x1d3c, 0x1396, 0x2484, 0x15fb, 0x1db0,
        ];
        let mut omitted_digest = Sha256::new();
        for bits in omitted_probability_bits {
            omitted_digest.update(bits.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", omitted_digest.finalize()),
            "af5ff6372196707f7bb0b50e83b1ddbdf0193b27790a86a03de960c0a2e62586"
        );
        let mut probabilities = Vec::with_capacity(N_HEADS * local_count);
        for (head, omitted_bits) in omitted_probability_bits.into_iter().enumerate() {
            probabilities.push(bf16_bits_to_f32(omitted_bits));
            probabilities.extend(
                probability_stage.bf16_bits[head * LOCAL_WINDOW..(head + 1) * LOCAL_WINDOW]
                    .iter()
                    .copied()
                    .map(bf16_bits_to_f32),
            );
        }
        assert_eq!(
            widened_bf16_sha256(&probabilities),
            "b7457428ac8fba4fd26aef264c77385c172c7cf56a42758a6fc0bd709689af7c"
        );
        assert_eq!(crop_bits(&probabilities), probability_stage.bf16_bits);
        let run = |pipeline: &ComputePipelineState| {
            run_test_attention_context(
                kernel,
                pipeline,
                &probabilities,
                &values,
                N_HEADS,
                LOCAL_HEAD_DIM,
                local_count,
                N_HEADS / LOCAL_KV_HEADS,
                LOCAL_HEAD_DIM,
                kv_len * LOCAL_HEAD_DIM,
                local_start * LOCAL_HEAD_DIM,
                local_start,
                kv_len,
            )
        };
        let legacy = run(&pipelines.attention_context_legacy_bf16);
        assert_eq!(
            widened_bf16_sha256(&legacy),
            "54fb45ab9c3f95ea8ca82695986bdcba8a4e66919b52087505b98b2e177d6a3f"
        );
        let production = run(&pipelines.attention_context_bf16);
        let production_bits: Vec<u16> = production
            .iter()
            .copied()
            .map(f32_to_bf16_rne_bits)
            .collect();
        assert_eq!(production_bits, context_stage.bf16_bits);
        assert_eq!(widened_bf16_sha256(&production), context_stage.bf16_sha256);
    }

    #[test]
    #[ignore = "reads two bounded official BF16 matrices and the external step-1 stage oracle"]
    fn official_step1_layer1_production_gemv_matches_oracle() {
        let model_path = std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH");
        let oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .map(PathBuf::from)
            .expect("set CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON");
        let oracle_bytes = std::fs::read(&oracle_path).expect("read step-1 stage oracle");
        assert_eq!(
            sha256_hex(&oracle_bytes),
            EXPECTED_STEP1_STAGE_ORACLE_SHA256,
            "step-1 stage oracle bytes changed"
        );
        let oracle: StageOracleDocument =
            serde_json::from_slice(&oracle_bytes).expect("parse step-1 stage oracle");
        assert_eq!(oracle.schema_version, 1);
        let authority = oracle.authority.as_ref().expect("stage oracle authority");
        assert_eq!(authority.model_sha256.as_deref(), Some(EXPECTED_SHA256));
        assert_eq!(
            authority.recurrence_oracle_sha256.as_deref(),
            Some(EXPECTED_RECURRENCE_ORACLE_SHA256)
        );
        assert_eq!(authority.recurrence_step, Some(1));

        let mapping = GgufWireMmap::map(&model_path).expect("map official assistant weights");
        let manifest = parse_official_manifest(&mapping).expect("validate official tensor map");
        let kernel = metal_linear_kernel().expect("Metal kernel");
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let stage = |name: &str| {
            oracle
                .stages
                .iter()
                .find(|stage| stage.name == name)
                .unwrap_or_else(|| panic!("missing stage {name}"))
        };
        let cases = [
            (
                "layer.1.attention_context",
                "layer.1.o_proj",
                "model.layers.1.self_attn.o_proj.weight",
                "03e2fddde434b174c8d562eaffad9be7d2d896447c306852bf870c83b23d0f86",
            ),
            (
                "layer.1.pre_feedforward_norm",
                "layer.1.up_proj",
                "model.layers.1.mlp.up_proj.weight",
                "47edf273b1f5f0ae8cb8879ddc76256ab55b7fe74918bdf710b1e2ed9c1bc1e4",
            ),
        ];

        for (input_name, output_name, weight_name, expected_native_sha256) in cases {
            let input_stage = stage(input_name);
            let output_stage = stage(output_name);
            let matrix = manifest.matrix(weight_name).expect("official matrix");
            assert_eq!(input_stage.elements, matrix.cols as usize);
            assert_eq!(output_stage.elements, matrix.rows as usize);
            let matrix_bytes = (matrix.rows as usize)
                .checked_mul(matrix.cols as usize)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .expect("matrix byte size");
            let weights = mapping
                .bytes(matrix.absolute_offset as u64, matrix_bytes)
                .expect("official matrix bytes");
            let weight_buffer = shared_buffer(&kernel.device, matrix_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    weights.as_ptr(),
                    weight_buffer.contents().cast::<u8>(),
                    matrix_bytes,
                );
            }
            let input: Vec<f32> = input_stage
                .bf16_bits
                .iter()
                .copied()
                .map(bf16_bits_to_f32)
                .collect();
            let input_buffer = f32_buffer(&kernel.device, &input);
            let run = |pipeline: &ComputePipelineState| {
                let output = shared_buffer(
                    &kernel.device,
                    output_stage.elements * std::mem::size_of::<f32>(),
                );
                let command_buffer = kernel.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encode_bf16_gemv(
                    encoder,
                    pipeline,
                    &weight_buffer,
                    &input_buffer,
                    &output,
                    TensorRef {
                        absolute_offset: 0,
                        rows: matrix.rows,
                        cols: matrix.cols,
                    },
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
                let mut values = vec![0.0f32; output_stage.elements];
                read_buffer_f32(&output, &mut values);
                values
            };

            let legacy = run(&pipelines.bf16_gemv_legacy);
            assert_eq!(widened_bf16_sha256(&legacy), expected_native_sha256);
            let production = run(&pipelines.bf16_gemv);
            let production_bits: Vec<u16> = production
                .iter()
                .copied()
                .map(f32_to_bf16_rne_bits)
                .collect();
            assert_eq!(
                production_bits, output_stage.bf16_bits,
                "production GEMV raw BF16 mismatch for {output_name}"
            );
            assert_eq!(widened_bf16_sha256(&production), output_stage.bf16_sha256);
            eprintln!(
                "MTP_ATEN_GEMV_OFFICIAL name={output_name} elements={} legacy_sha256={} production_sha256={} expected_sha256={}",
                output_stage.elements,
                expected_native_sha256,
                widened_bf16_sha256(&production),
                output_stage.bf16_sha256,
            );
        }
    }

    #[derive(Deserialize)]
    struct RecurrenceOracleDocument {
        schema_version: u32,
        authority: RecurrenceOracleAuthority,
        feedback_embedding: RecurrenceFeedbackEmbedding,
        admission: RecurrenceAdmission,
        steps: Vec<RecurrenceOracleStep>,
    }

    #[derive(Deserialize)]
    struct RecurrenceOracleAuthority {
        target_model_loaded: bool,
        assistant_model_sha256: String,
        assistant_config_sha256: String,
        canonical_manifest_sha256: String,
        kv_len: usize,
        position_id: usize,
        proposal_count: usize,
        stop_token_ids: Vec<u32>,
        argmax_tie_policy: String,
        minimum_reference_margin_bf16_ulp: u32,
    }

    #[derive(Deserialize)]
    struct RecurrenceFeedbackEmbedding {
        seed_xor: u32,
        token_multiplier: u32,
        shape: Vec<usize>,
        exponent_base: u32,
        exponent_span: u32,
    }

    #[derive(Deserialize)]
    struct RecurrenceAdmission {
        required_fresh_python_runs: usize,
        required_native_repetitions: usize,
        required_exact_top1_steps: usize,
        minimum_top16_set_overlap_per_step: usize,
        native_margin_floor_rule: String,
        native_margin_cap_bf16_ulp: u32,
        minimum_recurrent_cosine_per_step: f64,
        maximum_recurrent_relative_l2_per_step: f64,
        stop_on_first_top1_mismatch: bool,
        teacher_force_after_mismatch: bool,
        require_native_bf16_lattice: bool,
        require_native_repeat_bit_determinism: bool,
    }

    #[derive(Deserialize)]
    struct RecurrenceGenerationReceipt {
        schema_version: u32,
        assistant_model_sha256: String,
        canonical_oracle: RecurrenceGeneratedArtifact,
        rerun_oracle: RecurrenceGeneratedArtifact,
        byte_identical: bool,
        top1_token_ids: Vec<u32>,
        fresh_process_runs: usize,
        swapouts_start: u64,
        swapouts_end: u64,
        t7_model_safetensors_opened: bool,
        salt_search: bool,
        teacher_forcing: bool,
        run_logs: Vec<String>,
        failed_attempt_log: String,
    }

    #[derive(Deserialize)]
    struct RecurrenceGeneratedArtifact {
        path: String,
        bytes: u64,
        sha256: String,
    }

    #[derive(Deserialize)]
    struct RecurrenceOracleStep {
        index: usize,
        input_embedding_bf16_sha256: String,
        recurrent_input_bf16_sha256: String,
        top1_token_id: u32,
        top16_token_ids: Vec<u32>,
        top16_logits_bf16_bits: Vec<u16>,
        top1_margin_bf16_ulp: u32,
        logits_bf16_sha256: String,
        recurrent_hidden_bf16_sha256: String,
        recurrent_hidden_bf16_bits: Vec<u16>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NativeRecurrenceStep {
        token: u32,
        top16_token_ids: Vec<usize>,
        logits_bf16_sha256: String,
        recurrent_hidden_bf16_sha256: String,
    }

    #[derive(Clone, Debug, serde::Serialize)]
    struct NativeAdmissionStepEvidence {
        step: usize,
        token_id: u32,
        top16_overlap: usize,
        recurrent_cosine: f64,
        recurrent_relative_l2: f64,
        native_margin_bf16_ulp: u32,
        required_margin_bf16_ulp: u32,
    }

    #[derive(Debug, serde::Serialize)]
    struct NativeAdmissionThresholds {
        exact_top1_steps: usize,
        minimum_top16_overlap_per_step: usize,
        minimum_recurrent_cosine_per_step: f64,
        maximum_recurrent_relative_l2_per_step: f64,
        native_margin_cap_bf16_ulp: u32,
        native_margin_floor_rule: String,
        required_margin_floors_bf16_ulp: Vec<u32>,
        require_bf16_lattice: bool,
        require_repeat_bit_determinism: bool,
    }

    #[derive(Debug, serde::Serialize)]
    struct NativeAdmissionPlatform {
        os: String,
        os_version: String,
        machine_arch: String,
        machine_model: String,
        metal_device_name: String,
    }

    #[derive(Debug, serde::Serialize)]
    struct NativeAdmissionEvidence {
        schema_version: u32,
        pass: bool,
        assistant_model_sha256: String,
        recurrence_oracle_sha256: String,
        recurrence_generation_receipt_sha256: String,
        stage_oracle_sha256: String,
        structural_stage_pass: bool,
        admission_test_exe_sha256: String,
        native_source_sha256: String,
        metal_rs_source_sha256: String,
        gemma4_runtime_rs_source_sha256: String,
        cargo_lock_sha256: String,
        platform: NativeAdmissionPlatform,
        created_unix_ms: u64,
        run_nonce: String,
        test_name: String,
        top1_token_ids: Vec<u32>,
        native_repetitions: usize,
        repeat_bit_deterministic: bool,
        per_step: Vec<NativeAdmissionStepEvidence>,
        min_top16_overlap: usize,
        min_recurrent_cosine: f64,
        max_recurrent_relative_l2: f64,
        tie_policy: String,
        tie_policy_test_pass: bool,
        teacher_forcing: bool,
        thresholds: NativeAdmissionThresholds,
    }

    fn native_admission_command_output(program: &str, arguments: &[&str]) -> String {
        let output = std::process::Command::new(program)
            .args(arguments)
            .output()
            .unwrap_or_else(|error| {
                panic!("run {program} for native admission provenance: {error}")
            });
        assert!(
            output.status.success(),
            "{program} failed while collecting native admission provenance: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value = String::from_utf8(output.stdout)
            .expect("native admission provenance command emitted non-UTF-8")
            .trim()
            .to_owned();
        assert!(
            !value.is_empty(),
            "{program} emitted empty native admission provenance"
        );
        value
    }

    fn validate_recurrence_generation_receipt(
        path: &Path,
        expected_top1_token_ids: &[u32],
    ) -> String {
        let bytes = std::fs::read(path).expect("read recurrence generation receipt");
        let receipt_sha256 = sha256_hex(&bytes);
        assert_eq!(
            receipt_sha256, EXPECTED_RECURRENCE_GENERATION_RECEIPT_SHA256,
            "recurrence generation receipt SHA-256 is not pinned"
        );
        let receipt: RecurrenceGenerationReceipt =
            serde_json::from_slice(&bytes).expect("parse recurrence generation receipt");
        assert_eq!(receipt.schema_version, 1);
        assert_eq!(receipt.assistant_model_sha256, EXPECTED_SHA256);
        assert_eq!(receipt.canonical_oracle.bytes, 127_153);
        assert_eq!(receipt.rerun_oracle.bytes, 127_153);
        assert_eq!(
            receipt.canonical_oracle.sha256,
            EXPECTED_RECURRENCE_ORACLE_SHA256
        );
        assert_eq!(
            receipt.rerun_oracle.sha256,
            EXPECTED_RECURRENCE_ORACLE_SHA256
        );
        assert_eq!(
            receipt.canonical_oracle.path,
            "qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/assistant_recurrence7_bf16_cpu.json"
        );
        assert!(receipt
            .rerun_oracle
            .path
            .ends_with("/assistant_recurrence7_bf16_cpu.rerun.json"));
        assert!(receipt.byte_identical);
        assert_eq!(receipt.top1_token_ids, expected_top1_token_ids);
        assert_eq!(receipt.fresh_process_runs, 2);
        assert_eq!(receipt.swapouts_start, receipt.swapouts_end);
        assert!(!receipt.t7_model_safetensors_opened);
        assert!(!receipt.salt_search);
        assert!(!receipt.teacher_forcing);
        assert_eq!(
            receipt.run_logs,
            vec!["recurrence7_run1.log", "recurrence7_run2.log"]
        );
        assert_eq!(
            receipt.failed_attempt_log,
            "recurrence7_failed_margin_floor_attempts.log"
        );

        let parent = path
            .parent()
            .expect("recurrence generation receipt must have a parent");
        let run1_sha256 = sha256_hex(
            &std::fs::read(parent.join(&receipt.run_logs[0]))
                .expect("read pinned recurrence generation run-1 log"),
        );
        let run2_sha256 = sha256_hex(
            &std::fs::read(parent.join(&receipt.run_logs[1]))
                .expect("read pinned recurrence generation run-2 log"),
        );
        let failed_attempts_sha256 = sha256_hex(
            &std::fs::read(parent.join(&receipt.failed_attempt_log))
                .expect("read pinned recurrence failed-attempt log"),
        );
        assert_eq!(run1_sha256, EXPECTED_RECURRENCE_RUN1_LOG_SHA256);
        assert_eq!(run2_sha256, EXPECTED_RECURRENCE_RUN2_LOG_SHA256);
        assert_eq!(
            failed_attempts_sha256,
            EXPECTED_RECURRENCE_FAILED_ATTEMPTS_LOG_SHA256
        );
        receipt_sha256
    }

    fn validate_native_admission_evidence_path(path: &Path) {
        assert!(
            path.is_absolute(),
            "native admission evidence path must be absolute"
        );
        let parent = path
            .parent()
            .expect("native admission evidence path must have a parent");
        let canonical_parent = parent
            .canonicalize()
            .expect("native admission evidence parent must already exist");
        assert!(
            canonical_parent.starts_with("/Users/timtoole/"),
            "native admission evidence must be emitted to the internal volume, got {}",
            canonical_parent.display()
        );
        assert!(
            !path.exists(),
            "refusing to retain or replace stale native admission evidence at {}",
            path.display()
        );
    }

    fn emit_native_admission_evidence_atomically(path: &Path, evidence: &NativeAdmissionEvidence) {
        use std::io::Write as _;

        let parent = path.parent().unwrap();
        let file_name = path.file_name().unwrap().to_string_lossy();
        let temporary = parent.join(format!(
            ".{file_name}.tmp-native-admission-{}",
            std::process::id()
        ));
        assert!(
            !temporary.exists(),
            "native admission temporary path already exists: {}",
            temporary.display()
        );
        let encoded = serde_json::to_vec(evidence).expect("serialize native admission evidence");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .expect("create native admission evidence temporary file");
        file.write_all(&encoded)
            .expect("write native admission evidence temporary file");
        file.write_all(b"\n")
            .expect("terminate native admission evidence with newline");
        file.sync_all()
            .expect("sync native admission evidence temporary file");
        drop(file);
        if let Err(error) = std::fs::hard_link(&temporary, path) {
            let _ = std::fs::remove_file(&temporary);
            panic!(
                "publish native admission evidence without replacing a raced final path: {error}"
            );
        }
        if let Err(error) = std::fs::remove_file(&temporary) {
            let _ = std::fs::remove_file(path);
            panic!("remove native admission evidence temporary link: {error}");
        }
        if let Err(error) = std::fs::File::open(parent).and_then(|directory| directory.sync_all()) {
            std::fs::remove_file(path)
                .expect("remove published evidence after directory sync failure");
            let _ = std::fs::File::open(parent).and_then(|directory| directory.sync_all());
            panic!("sync native admission evidence directory: {error}");
        }
    }

    #[derive(Debug)]
    struct Bf16DiffMetrics {
        exact: usize,
        max_abs: f32,
        mean_abs: f64,
        max_ulp: u32,
        mean_ulp: f64,
    }

    fn ordered_bf16(bits: u16) -> u32 {
        if bits & 0x8000 != 0 {
            (!bits) as u32
        } else {
            (bits | 0x8000) as u32
        }
    }

    fn bf16_diff_metrics(actual: &[f32], expected_bits: &[u16]) -> Bf16DiffMetrics {
        assert_eq!(actual.len(), expected_bits.len());
        let mut exact = 0usize;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut max_ulp = 0u32;
        let mut sum_ulp = 0u64;
        for (&actual, &expected_bits) in actual.iter().zip(expected_bits) {
            let actual_bits = f32_to_bf16_rne_bits(actual);
            exact += usize::from(actual_bits == expected_bits);
            let expected = bf16_bits_to_f32(expected_bits);
            let abs = (actual - expected).abs();
            max_abs = max_abs.max(abs);
            sum_abs += abs as f64;
            let ulp = ordered_bf16(actual_bits).abs_diff(ordered_bf16(expected_bits));
            max_ulp = max_ulp.max(ulp);
            sum_ulp += ulp as u64;
        }
        Bf16DiffMetrics {
            exact,
            max_abs,
            mean_abs: sum_abs / actual.len().max(1) as f64,
            max_ulp,
            mean_ulp: sum_ulp as f64 / actual.len().max(1) as f64,
        }
    }

    fn top_token_ids(values: &[f32], count: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_unstable_by(|&left, &right| {
            values[right]
                .total_cmp(&values[left])
                .then_with(|| left.cmp(&right))
        });
        indices.truncate(count.min(indices.len()));
        indices
    }

    fn gpu_argmax_exact_tie_prefers_lowest_id(assistant: &Gemma4MtpAssistantMetal) -> bool {
        let kernel = metal_linear_kernel().expect("Metal kernel for argmax tie-policy test");
        let logits = f32_buffer(
            &kernel.device,
            &[f32::NAN, -5.0, 7.0, 1.0, 7.0, 7.0, 6.5, -1.0],
        );
        let output_token = shared_buffer(&kernel.device, std::mem::size_of::<u32>());
        unsafe {
            *output_token.contents().cast::<u32>() = u32::MAX;
        }
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_argmax(
            encoder,
            &assistant.pipelines.argmax,
            &logits,
            &output_token,
            8,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let actual = unsafe { *output_token.contents().cast::<u32>() };
        eprintln!("MTP_ARGMAX_EXACT_TIE expected_low_id=2 actual_id={actual}");
        actual == 2
    }

    fn recurrence_feedback_embedding(
        token: u32,
        feedback: &RecurrenceFeedbackEmbedding,
    ) -> (Vec<f32>, String) {
        let seed = feedback.seed_xor ^ token.wrapping_mul(feedback.token_multiplier);
        deterministic_bf16_values(
            TARGET_HIDDEN,
            seed,
            feedback.exponent_base,
            feedback.exponent_span,
        )
    }

    fn recurrence_hidden_similarity(actual: &[f32], expected_bits: &[u16]) -> (f64, f64) {
        assert_eq!(actual.len(), expected_bits.len());
        let mut dot = 0.0f64;
        let mut actual_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut diff_sq = 0.0f64;
        for (&actual, &expected_bits) in actual.iter().zip(expected_bits) {
            let actual = actual as f64;
            let expected = bf16_bits_to_f32(expected_bits) as f64;
            dot += actual * expected;
            actual_sq += actual * actual;
            expected_sq += expected * expected;
            let diff = actual - expected;
            diff_sq += diff * diff;
        }
        let denominator = (actual_sq.sqrt() * expected_sq.sqrt()).max(f64::MIN_POSITIVE);
        let cosine = dot / denominator;
        let relative_l2 = diff_sq.sqrt() / expected_sq.sqrt().max(f64::MIN_POSITIVE);
        (cosine, relative_l2)
    }

    fn recurrence_target_kv_view<'a>(
        sliding_key: &'a Buffer,
        sliding_value: &'a Buffer,
        full_key: &'a Buffer,
        full_value: &'a Buffer,
        kv_len: usize,
    ) -> Gemma4MtpTargetKvView<'a> {
        Gemma4MtpTargetKvView {
            sliding: Gemma4MtpTargetKvLayerView {
                layer_index: 28,
                key: sliding_key,
                value: sliding_value,
                logical_len: kv_len,
                kv_stride: kv_len,
                kv_heads: LOCAL_KV_HEADS,
                head_dim: LOCAL_HEAD_DIM,
                sliding_window: Some(LOCAL_WINDOW),
            },
            full: Gemma4MtpTargetKvLayerView {
                layer_index: 29,
                key: full_key,
                value: full_value,
                logical_len: kv_len,
                kv_stride: kv_len,
                kv_heads: FULL_KV_HEADS,
                head_dim: FULL_HEAD_DIM,
                sliding_window: None,
            },
        }
    }

    fn report_stage_diagnostics(
        snapshots: &[MtpStageSnapshot],
        oracle: Option<&StageOracleDocument>,
    ) {
        for snapshot in snapshots {
            eprintln!(
                "MTP_STAGE_NATIVE name={} elements={} bf16_sha256={}",
                snapshot.name,
                snapshot.values.len(),
                snapshot.bf16_sha256,
            );
        }
        let Some(oracle) = oracle else {
            return;
        };
        assert_eq!(oracle.schema_version, 1);
        assert_eq!(snapshots.len(), oracle.stages.len(), "stage count mismatch");
        let mut first_divergence = None;
        let mut first_material_divergence = None;
        let mut q_rope_stages = 0usize;
        let mut attention_score_stages = 0usize;
        let mut attention_probability_stages = 0usize;
        let mut structurally_enforced_q_rope_stages = 0usize;
        let mut structurally_enforced_attention_score_stages = 0usize;
        for (index, (snapshot, expected)) in snapshots.iter().zip(&oracle.stages).enumerate() {
            assert_eq!(
                snapshot.name, expected.name,
                "stage order mismatch at {index}"
            );
            assert_eq!(snapshot.values.len(), expected.elements);
            assert_eq!(expected.bf16_bits.len(), expected.elements);
            let expected_widened: Vec<f32> = expected
                .bf16_bits
                .iter()
                .copied()
                .map(bf16_bits_to_f32)
                .collect();
            assert_eq!(
                widened_bf16_sha256(&expected_widened),
                expected.bf16_sha256,
                "stage oracle hash is internally inconsistent for {}",
                expected.name,
            );
            let metrics = bf16_diff_metrics(&snapshot.values, &expected.bf16_bits);
            let exact_percent = 100.0 * metrics.exact as f64 / expected.elements.max(1) as f64;
            if snapshot.bf16_sha256 != expected.bf16_sha256 && first_divergence.is_none() {
                first_divergence = Some(expected.name.clone());
            }
            if metrics.max_ulp > 1 && first_material_divergence.is_none() {
                first_material_divergence = Some(expected.name.clone());
            }
            if expected.name.ends_with(".q_rope") {
                q_rope_stages += 1;
                if !expected.name.starts_with("layer.3.") {
                    structurally_enforced_q_rope_stages += 1;
                    assert_eq!(
                        metrics.exact, expected.elements,
                        "{} must remain bit-exact to prove RoPE geometry",
                        expected.name
                    );
                }
            } else if expected.name.ends_with(".attention_scores") {
                attention_score_stages += 1;
                if !expected.name.starts_with("layer.3.") {
                    structurally_enforced_attention_score_stages += 1;
                    assert!(
                        expected.elements - metrics.exact <= 2 && metrics.max_ulp <= 1,
                        "{} QK scores exceed the structural tolerance: exact={}/{}, max_bf16_ulp={}",
                        expected.name,
                        metrics.exact,
                        expected.elements,
                        metrics.max_ulp
                    );
                }
            } else if expected.name.ends_with(".attention_probs") {
                attention_probability_stages += 1;
            }
            eprintln!(
                "MTP_STAGE_COMPARE index={} name={} actual_sha256={} expected_sha256={} exact={}/{} exact_percent={:.6} max_abs={:.9e} mean_abs={:.9e} max_bf16_ulp={} mean_bf16_ulp={:.6}",
                index,
                expected.name,
                snapshot.bf16_sha256,
                expected.bf16_sha256,
                metrics.exact,
                expected.elements,
                exact_percent,
                metrics.max_abs,
                metrics.mean_abs,
                metrics.max_ulp,
                metrics.mean_ulp,
            );
            if expected.name == "lm_head" {
                let actual_top = top_token_ids(&snapshot.values, 16);
                let expected_top = top_token_ids(&expected_widened, 16);
                let overlap = actual_top
                    .iter()
                    .filter(|token| expected_top.contains(token))
                    .count();
                let actual_margin = snapshot.values[actual_top[0]] - snapshot.values[actual_top[1]];
                let expected_margin =
                    expected_widened[expected_top[0]] - expected_widened[expected_top[1]];
                eprintln!(
                    "MTP_LOGIT_COMPARE actual_top1={} expected_top1={} top16_overlap={}/16 actual_margin={:.9e} expected_margin={:.9e} actual_top16={:?} expected_top16={:?}",
                    actual_top[0],
                    expected_top[0],
                    overlap,
                    actual_margin,
                    expected_margin,
                    actual_top,
                    expected_top,
                );
            }
        }
        eprintln!(
            "MTP_STAGE_FIRST_DIVERGENCE name={}",
            first_divergence.as_deref().unwrap_or("none")
        );
        assert_eq!(
            q_rope_stages, 4,
            "stage oracle must cover all q_rope stages"
        );
        assert_eq!(
            attention_score_stages, 4,
            "stage oracle must cover all QK score stages"
        );
        assert_eq!(
            attention_probability_stages, 4,
            "stage oracle must cover all softmax stages"
        );
        assert_eq!(
            structurally_enforced_q_rope_stages, 3,
            "layers 0..2 must enforce q_rope geometry"
        );
        assert_eq!(
            structurally_enforced_attention_score_stages, 3,
            "layers 0..2 must enforce QK geometry"
        );
        assert!(
            first_material_divergence
                .as_deref()
                .is_some_and(|name| name.ends_with(".attention_probs")),
            "first >1-BF16-ULP divergence must be localized to softmax, got {:?}",
            first_material_divergence
        );
        eprintln!(
            "MTP_STAGE_FIRST_MATERIAL_DIVERGENCE name={}",
            first_material_divergence.as_deref().unwrap()
        );
    }

    #[test]
    #[ignore = "maps+mlocks the 801 MiB official assistant and executes the structural single-proposal diagnostic"]
    fn official_target_free_bf16_single_proposal_structural_diagnostic() {
        let Some(path) = std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH") else {
            eprintln!(
                "SKIP: set CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH to the hash-pinned model.safetensors"
            );
            return;
        };
        let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
        assert_eq!(manifest["scope"]["target_model_loaded"], false);
        assert_eq!(manifest["scope"]["kv_len"], 1_031);
        assert_eq!(manifest["scope"]["position_id"], 1_031);
        assert_eq!(manifest["oracle_output"]["top1_token_id"], 53_965);

        let mut assistant = Gemma4MtpAssistantMetal::load(Path::new(&path)).unwrap();
        let resident = assistant.resident_ledger();
        eprintln!(
            "MTP_RESIDENT_LEDGER source_path={} file_bytes={} mapped_bytes={} locked_bytes={} resident_pages={} total_pages={} payload_bytes={} decoded_norm_bytes={} fixed_scratch_bytes={} hash_us={} lock_and_residency_us={} pipeline_compile_us={} load_wall_us={}",
            assistant.source_path().display(),
            resident.file_bytes,
            resident.mapped_bytes,
            resident.locked_bytes,
            resident.resident_pages,
            resident.total_pages,
            resident.payload_bytes,
            resident.decoded_norm_bytes,
            resident.fixed_scratch_bytes,
            resident.hash_us,
            resident.lock_and_residency_us,
            resident.pipeline_compile_us,
            resident.load_wall_us,
        );
        let device = &metal_linear_kernel().unwrap().device;
        let kv_len = 1_031usize;
        let embedding = oracle_values(
            &manifest,
            "target_scaled_embedding",
            TARGET_HIDDEN,
            0x1357_9bdf,
            120,
            3,
        );
        let target_hidden = oracle_values(
            &manifest,
            "target_final_normalized_hidden",
            TARGET_HIDDEN,
            0x2468_ace1,
            125,
            3,
        );
        let sliding_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_key_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x3141_5926,
                125,
                3,
            ),
        );
        let sliding_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_value_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x2718_2818,
                125,
                3,
            ),
        );
        let full_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_key_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x1618_0339,
                125,
                3,
            ),
        );
        let full_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_value_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x5772_1566,
                125,
                3,
            ),
        );
        let target_kv = Gemma4MtpTargetKvView {
            sliding: Gemma4MtpTargetKvLayerView {
                layer_index: 28,
                key: &sliding_key,
                value: &sliding_value,
                logical_len: kv_len,
                kv_stride: kv_len,
                kv_heads: LOCAL_KV_HEADS,
                head_dim: LOCAL_HEAD_DIM,
                sliding_window: Some(LOCAL_WINDOW),
            },
            full: Gemma4MtpTargetKvLayerView {
                layer_index: 29,
                key: &full_key,
                value: &full_value,
                logical_len: kv_len,
                kv_stride: kv_len,
                kv_heads: FULL_KV_HEADS,
                head_dim: FULL_HEAD_DIM,
                sliding_window: None,
            },
        };
        let proposal = assistant
            .propose(&embedding, &target_hidden, target_kv)
            .unwrap();
        let recurrent_hash = widened_bf16_sha256(&proposal.recurrent_hidden);
        let mut logits = vec![0.0f32; VOCAB];
        read_buffer_f32(&assistant.scratch.logits, &mut logits);
        let logits_hash = widened_bf16_sha256(&logits);
        eprintln!(
            "MTP_PROPOSAL_LEDGER top1={} recurrent_bf16_sha256={} logits_bf16_sha256={} encode_us={} wait_us={} wall_us={} gpu_us={} kernel_us={} assistant_matrix_bytes={} borrowed_target_kv_capacity_bytes={} target_kv_read_bytes={} dynamic_attention_scratch_bytes={} readback_bytes={}",
            proposal.token,
            recurrent_hash,
            logits_hash,
            proposal.timing.encode_us,
            proposal.timing.wait_us,
            proposal.timing.wall_us,
            proposal.timing.gpu_us,
            proposal.timing.kernel_us,
            proposal.ledger.assistant_matrix_bytes,
            proposal.ledger.borrowed_target_kv_capacity_bytes,
            proposal.ledger.target_kv_read_bytes,
            proposal.ledger.dynamic_attention_scratch_bytes,
            proposal.ledger.readback_bytes,
        );
        let stage_oracle = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON").map(|path| {
            let bytes = std::fs::read(path).expect("read optional structural stage oracle JSON");
            serde_json::from_slice::<StageOracleDocument>(&bytes)
                .expect("parse optional structural stage oracle JSON")
        });
        report_stage_diagnostics(&proposal.stage_snapshots, stage_oracle.as_ref());
        let expected_top16 = manifest["oracle_output"]["top16_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let expected_top16_bits = manifest["oracle_output"]["top16_logits_bf16_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u16)
            .collect::<Vec<_>>();
        let actual_top16 = top_token_ids(&logits, 16);
        let top16_overlap = actual_top16
            .iter()
            .filter(|token| expected_top16.contains(token))
            .count();
        let actual_margin = ordered_bf16(f32_to_bf16_rne_bits(logits[actual_top16[0]]))
            .abs_diff(ordered_bf16(f32_to_bf16_rne_bits(logits[actual_top16[1]])));
        let expected_margin =
            ordered_bf16(expected_top16_bits[0]).abs_diff(ordered_bf16(expected_top16_bits[1]));
        assert_eq!(proposal.token, 53_965);
        assert_eq!(proposal.token as usize, actual_top16[0]);
        assert!(
            top16_overlap >= 15,
            "single-proposal structural top-16 overlap {top16_overlap}/16 is below 15/16"
        );
        assert!(
            actual_margin >= expected_margin.min(2),
            "single-proposal structural margin {actual_margin} BF16 ULP is below {}",
            expected_margin.min(2)
        );
        assert!(proposal
            .recurrent_hidden
            .iter()
            .all(|value| value.is_finite() && value.to_bits() & 0xffff == 0));
        assert!(logits
            .iter()
            .all(|value| value.is_finite() && value.to_bits() & 0xffff == 0));
        eprintln!(
            "MTP_SINGLE_PROPOSAL_STRUCTURAL top1={} top16_overlap={}/16 native_margin_bf16_ulp={} required_margin_bf16_ulp={} actual_recurrent_sha256={} expected_recurrent_sha256={} actual_logits_sha256={} expected_logits_sha256={} whole_hashes_are_diagnostic_only=true",
            proposal.token,
            top16_overlap,
            actual_margin,
            expected_margin.min(2),
            recurrent_hash,
            manifest_string(
                &manifest,
                &["oracle_output", "recurrent_hidden_bf16_sha256"]
            ),
            logits_hash,
            manifest_string(&manifest, &["oracle_output", "logits_bf16_sha256"]),
        );
    }

    /// Diagnostic-only recurrence-stage comparator. Unlike native admission,
    /// this accepts an explicitly supplied, unpinned stage oracle and emits no
    /// evidence artifact. The recurrence oracle still supplies the exact BF16
    /// input state, so selecting step N never teacher-forces through N native
    /// proposals: it executes exactly one proposal from the authoritative
    /// step-N embedding/recurrent pair.
    #[test]
    #[ignore = "maps+mlocks the official assistant and compares one recurrence step against an explicitly supplied stage oracle"]
    fn official_target_free_bf16_recurrence_step_stage_diagnostic() {
        let assistant_path = std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH")
            .expect("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH is required");
        let recurrence_oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_RECURRENCE_ORACLE_JSON")
            .expect("CAMELID_GEMMA4_MTP_RECURRENCE_ORACLE_JSON is required");
        // Reuse the existing name so `propose` enables its cfg(test)-only GPU
        // snapshots. This test deliberately does not require the admission-
        // pinned stage hash.
        let stage_oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .expect("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON is required");
        let step_index = std::env::var("CAMELID_GEMMA4_MTP_STAGE_STEP_INDEX")
            .expect("CAMELID_GEMMA4_MTP_STAGE_STEP_INDEX is required")
            .parse::<usize>()
            .expect("CAMELID_GEMMA4_MTP_STAGE_STEP_INDEX must be an unsigned integer");

        let recurrence_oracle_bytes =
            std::fs::read(&recurrence_oracle_path).expect("read diagnostic recurrence oracle JSON");
        let recurrence_oracle_sha256 = sha256_hex(&recurrence_oracle_bytes);
        let oracle: RecurrenceOracleDocument = serde_json::from_slice(&recurrence_oracle_bytes)
            .expect("parse diagnostic recurrence oracle JSON");
        assert_eq!(oracle.schema_version, 2);
        assert!(!oracle.authority.target_model_loaded);
        assert_eq!(oracle.authority.assistant_model_sha256, EXPECTED_SHA256);
        assert_eq!(
            oracle.authority.assistant_config_sha256,
            EXPECTED_CONFIG_SHA256
        );
        assert_eq!(
            oracle.authority.canonical_manifest_sha256,
            sha256_hex(ORACLE_MANIFEST.as_bytes())
        );
        assert_eq!(oracle.authority.kv_len, 1_031);
        assert_eq!(oracle.authority.position_id, 1_031);
        assert_eq!(oracle.steps.len(), oracle.authority.proposal_count);
        let expected = oracle
            .steps
            .get(step_index)
            .unwrap_or_else(|| panic!("recurrence step {step_index} is outside the oracle"));
        assert_eq!(expected.index, step_index);

        let stage_oracle_bytes =
            std::fs::read(&stage_oracle_path).expect("read diagnostic stage oracle JSON");
        let stage_oracle: StageOracleDocument = serde_json::from_slice(&stage_oracle_bytes)
            .expect("parse diagnostic stage oracle JSON");
        assert_eq!(stage_oracle.schema_version, 1);
        let stage_authority = stage_oracle
            .authority
            .as_ref()
            .expect("diagnostic stage oracle must include authority");
        assert_eq!(
            stage_authority.model_sha256.as_deref(),
            Some(EXPECTED_SHA256),
            "diagnostic stage oracle assistant hash mismatch"
        );
        assert_eq!(
            stage_authority.recurrence_oracle_sha256.as_deref(),
            Some(recurrence_oracle_sha256.as_str()),
            "diagnostic stage oracle was generated from a different recurrence oracle"
        );
        assert_eq!(
            stage_authority.recurrence_step,
            Some(step_index),
            "diagnostic stage oracle recurrence step mismatch"
        );

        let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
        let (embedding, recurrent_hidden) = if step_index == 0 {
            (
                oracle_values(
                    &manifest,
                    "target_scaled_embedding",
                    TARGET_HIDDEN,
                    0x1357_9bdf,
                    120,
                    3,
                ),
                oracle_values(
                    &manifest,
                    "target_final_normalized_hidden",
                    TARGET_HIDDEN,
                    0x2468_ace1,
                    125,
                    3,
                ),
            )
        } else {
            let previous = &oracle.steps[step_index - 1];
            assert_eq!(
                expected.recurrent_input_bf16_sha256,
                previous.recurrent_hidden_bf16_sha256
            );
            let recurrent_hidden: Vec<f32> = previous
                .recurrent_hidden_bf16_bits
                .iter()
                .copied()
                .map(bf16_bits_to_f32)
                .collect();
            let (embedding, embedding_sha256) =
                recurrence_feedback_embedding(previous.top1_token_id, &oracle.feedback_embedding);
            assert_eq!(embedding_sha256, expected.input_embedding_bf16_sha256);
            (embedding, recurrent_hidden)
        };
        assert_eq!(
            widened_bf16_sha256(&embedding),
            expected.input_embedding_bf16_sha256
        );
        assert_eq!(
            widened_bf16_sha256(&recurrent_hidden),
            expected.recurrent_input_bf16_sha256
        );

        let mut assistant = Gemma4MtpAssistantMetal::load(Path::new(&assistant_path)).unwrap();
        let device = &metal_linear_kernel().unwrap().device;
        let kv_len = oracle.authority.kv_len;
        let sliding_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_key_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x3141_5926,
                125,
                3,
            ),
        );
        let sliding_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_value_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x2718_2818,
                125,
                3,
            ),
        );
        let full_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_key_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x1618_0339,
                125,
                3,
            ),
        );
        let full_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_value_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x5772_1566,
                125,
                3,
            ),
        );
        let proposal = assistant
            .propose(
                &embedding,
                &recurrent_hidden,
                recurrence_target_kv_view(
                    &sliding_key,
                    &sliding_value,
                    &full_key,
                    &full_value,
                    kv_len,
                ),
            )
            .unwrap();
        eprintln!(
            "MTP_RECURRENCE_STEP_STAGE_DIAGNOSTIC step={} actual_top1={} expected_top1={} snapshots={} recurrence_oracle_sha256={} stage_oracle_sha256={}",
            step_index,
            proposal.token,
            expected.top1_token_id,
            proposal.stage_snapshots.len(),
            recurrence_oracle_sha256,
            sha256_hex(&stage_oracle_bytes),
        );
        report_stage_diagnostics(&proposal.stage_snapshots, Some(&stage_oracle));
    }

    #[test]
    #[ignore = "maps+mlocks the official assistant and executes the target-free seven-proposal recurrence oracle"]
    fn official_target_free_bf16_oracle_matches_native_seven_proposal_recurrence() {
        const PROPOSALS: usize = 7;
        const MIN_TOP16_OVERLAP: usize = 15;
        const NATIVE_MARGIN_CAP_BF16_ULP: u32 = 2;
        const MIN_RECURRENT_COSINE: f64 = 0.999_95;
        const MAX_RECURRENT_RELATIVE_L2: f64 = 0.01;
        const PINNED_REQUIRED_MARGIN_FLOORS: [u32; PROPOSALS] = [2, 2, 2, 0, 2, 1, 2];

        let assistant_path = std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH")
            .expect("CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH is required");
        let oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_RECURRENCE_ORACLE_JSON")
            .expect("CAMELID_GEMMA4_MTP_RECURRENCE_ORACLE_JSON is required");
        let stage_oracle_path = std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON")
            .expect("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON is required");
        let generation_receipt_path =
            std::env::var_os("CAMELID_GEMMA4_MTP_RECURRENCE_GENERATION_RECEIPT_JSON")
                .expect("CAMELID_GEMMA4_MTP_RECURRENCE_GENERATION_RECEIPT_JSON is required");
        let evidence_output = PathBuf::from(
            std::env::var_os("CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON")
                .expect("CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON is required"),
        );
        validate_native_admission_evidence_path(&evidence_output);
        let evidence_run_nonce = std::env::var("CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE")
            .expect("CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE is required");
        assert!(
            (24..=128).contains(&evidence_run_nonce.len())
                && evidence_run_nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "native admission run nonce must be 24..=128 URL-safe ASCII characters"
        );

        let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
        let oracle_bytes = std::fs::read(&oracle_path).expect("read recurrence oracle JSON");
        let oracle_sha256 = sha256_hex(&oracle_bytes);
        let stage_oracle_bytes =
            std::fs::read(&stage_oracle_path).expect("read structural stage oracle JSON");
        let stage_oracle_sha256 = sha256_hex(&stage_oracle_bytes);
        assert_eq!(
            oracle_sha256, EXPECTED_RECURRENCE_ORACLE_SHA256,
            "recurrence oracle SHA-256 is not pinned"
        );
        assert_eq!(
            stage_oracle_sha256, EXPECTED_STAGE_ORACLE_SHA256,
            "structural stage oracle SHA-256 is not pinned"
        );
        let oracle: RecurrenceOracleDocument =
            serde_json::from_slice(&oracle_bytes).expect("parse recurrence oracle JSON");
        let stage_oracle: StageOracleDocument = serde_json::from_slice(&stage_oracle_bytes)
            .expect("parse structural stage oracle JSON");
        let pinned_top1_token_ids: Vec<u32> =
            oracle.steps.iter().map(|step| step.top1_token_id).collect();
        let recurrence_generation_receipt_sha256 = validate_recurrence_generation_receipt(
            Path::new(&generation_receipt_path),
            &pinned_top1_token_ids,
        );
        assert_eq!(oracle.schema_version, 2);
        assert!(!oracle.authority.target_model_loaded);
        assert_eq!(oracle.authority.assistant_model_sha256, EXPECTED_SHA256);
        assert_eq!(
            oracle.authority.assistant_config_sha256,
            EXPECTED_CONFIG_SHA256
        );
        assert_eq!(
            oracle.authority.canonical_manifest_sha256,
            sha256_hex(ORACLE_MANIFEST.as_bytes())
        );
        assert_eq!(oracle.authority.kv_len, 1_031);
        assert_eq!(oracle.authority.position_id, 1_031);
        assert_eq!(oracle.authority.proposal_count, PROPOSALS);
        assert_eq!(oracle.authority.stop_token_ids, vec![1, 106]);
        assert_eq!(oracle.authority.argmax_tie_policy, "lowest_token_id");
        assert_eq!(oracle.authority.minimum_reference_margin_bf16_ulp, 0);
        assert_eq!(oracle.steps.len(), PROPOSALS);
        assert_eq!(oracle.feedback_embedding.seed_xor, 0x6a09_e667);
        assert_eq!(oracle.feedback_embedding.token_multiplier, 0x9e37_79b9);
        assert_eq!(oracle.feedback_embedding.shape, vec![1, 1, TARGET_HIDDEN]);
        assert_eq!(oracle.feedback_embedding.exponent_base, 120);
        assert_eq!(oracle.feedback_embedding.exponent_span, 3);
        assert_eq!(oracle.admission.required_fresh_python_runs, 2);
        assert_eq!(oracle.admission.required_native_repetitions, 2);
        assert_eq!(oracle.admission.required_exact_top1_steps, PROPOSALS);
        assert_eq!(
            oracle.admission.minimum_top16_set_overlap_per_step,
            MIN_TOP16_OVERLAP
        );
        assert_eq!(
            oracle.admission.native_margin_floor_rule,
            "min(reference_top1_margin_bf16_ulp, native_margin_cap_bf16_ulp)"
        );
        assert_eq!(
            oracle.admission.native_margin_cap_bf16_ulp,
            NATIVE_MARGIN_CAP_BF16_ULP
        );
        assert_eq!(
            oracle.admission.minimum_recurrent_cosine_per_step,
            MIN_RECURRENT_COSINE
        );
        assert_eq!(
            oracle.admission.maximum_recurrent_relative_l2_per_step,
            MAX_RECURRENT_RELATIVE_L2
        );
        assert!(oracle.admission.stop_on_first_top1_mismatch);
        assert!(!oracle.admission.teacher_force_after_mismatch);
        assert!(oracle.admission.require_native_bf16_lattice);
        assert!(oracle.admission.require_native_repeat_bit_determinism);

        let mut assistant = Gemma4MtpAssistantMetal::load(Path::new(&assistant_path)).unwrap();
        let tie_policy_test_pass = gpu_argmax_exact_tie_prefers_lowest_id(&assistant);
        assert!(
            tie_policy_test_pass,
            "production Metal argmax did not select the lowest token ID for an exact tie"
        );
        let device = &metal_linear_kernel().unwrap().device;
        let kv_len = oracle.authority.kv_len;
        let initial_embedding = oracle_values(
            &manifest,
            "target_scaled_embedding",
            TARGET_HIDDEN,
            0x1357_9bdf,
            120,
            3,
        );
        let initial_hidden = oracle_values(
            &manifest,
            "target_final_normalized_hidden",
            TARGET_HIDDEN,
            0x2468_ace1,
            125,
            3,
        );
        let sliding_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_key_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x3141_5926,
                125,
                3,
            ),
        );
        let sliding_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "sliding_value_layer28",
                LOCAL_KV_HEADS * kv_len * LOCAL_HEAD_DIM,
                0x2718_2818,
                125,
                3,
            ),
        );
        let full_key = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_key_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x1618_0339,
                125,
                3,
            ),
        );
        let full_value = f32_buffer(
            device,
            &oracle_values(
                &manifest,
                "full_value_layer29",
                FULL_KV_HEADS * kv_len * FULL_HEAD_DIM,
                0x5772_1566,
                125,
                3,
            ),
        );

        let mut first_native_run: Option<Vec<NativeRecurrenceStep>> = None;
        let mut first_admission_steps: Option<Vec<NativeAdmissionStepEvidence>> = None;
        let mut structural_stage_pass = false;
        for repetition in 0..oracle.admission.required_native_repetitions {
            let mut embedding = initial_embedding.clone();
            let mut recurrent_hidden = initial_hidden.clone();
            let mut native_run = Vec::with_capacity(PROPOSALS);
            let mut admission_steps = Vec::with_capacity(PROPOSALS);
            let mut expected_previous_recurrent_hash = manifest_string(
                &manifest,
                &["input_tensor_sha256", "target_final_normalized_hidden"],
            );

            for (step_index, expected) in oracle.steps.iter().enumerate() {
                assert_eq!(expected.index, step_index);
                assert_eq!(
                    expected.recurrent_input_bf16_sha256,
                    expected_previous_recurrent_hash
                );
                assert_eq!(expected.top16_token_ids.len(), 16);
                assert_eq!(expected.top16_logits_bf16_bits.len(), 16);
                assert_eq!(expected.recurrent_hidden_bf16_bits.len(), TARGET_HIDDEN);
                assert_eq!(expected.logits_bf16_sha256.len(), 64);
                let expected_recurrent: Vec<f32> = expected
                    .recurrent_hidden_bf16_bits
                    .iter()
                    .copied()
                    .map(bf16_bits_to_f32)
                    .collect();
                assert_eq!(
                    widened_bf16_sha256(&expected_recurrent),
                    expected.recurrent_hidden_bf16_sha256
                );
                let expected_margin = ordered_bf16(expected.top16_logits_bf16_bits[0])
                    .abs_diff(ordered_bf16(expected.top16_logits_bf16_bits[1]));
                assert_eq!(expected_margin, expected.top1_margin_bf16_ulp);
                assert!(expected_margin >= oracle.authority.minimum_reference_margin_bf16_ulp);
                if expected_margin == 0 {
                    let tied_low_id = expected
                        .top16_token_ids
                        .iter()
                        .zip(&expected.top16_logits_bf16_bits)
                        .take_while(|(_, bits)| **bits == expected.top16_logits_bf16_bits[0])
                        .map(|(token, _)| *token)
                        .min()
                        .unwrap();
                    assert_eq!(
                        expected.top1_token_id, tied_low_id,
                        "schema-v2 oracle did not apply the pinned low-ID argmax tie policy"
                    );
                }
                if step_index + 1 < PROPOSALS {
                    assert!(!oracle
                        .authority
                        .stop_token_ids
                        .contains(&expected.top1_token_id));
                }

                let embedding_hash = widened_bf16_sha256(&embedding);
                assert_eq!(embedding_hash, expected.input_embedding_bf16_sha256);
                if step_index == 0 {
                    assert_eq!(expected.top1_token_id, 53_965);
                    assert_eq!(
                        expected.top16_token_ids,
                        manifest["oracle_output"]["top16_token_ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|value| value.as_u64().unwrap() as u32)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        expected.top16_logits_bf16_bits,
                        manifest["oracle_output"]["top16_logits_bf16_bits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|value| value.as_u64().unwrap() as u16)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        expected.logits_bf16_sha256,
                        manifest_string(&manifest, &["oracle_output", "logits_bf16_sha256"])
                    );
                    assert_eq!(
                        expected.recurrent_hidden_bf16_sha256,
                        manifest_string(
                            &manifest,
                            &["oracle_output", "recurrent_hidden_bf16_sha256"]
                        )
                    );
                }

                let proposal = assistant
                    .propose(
                        &embedding,
                        &recurrent_hidden,
                        recurrence_target_kv_view(
                            &sliding_key,
                            &sliding_value,
                            &full_key,
                            &full_value,
                            kv_len,
                        ),
                    )
                    .unwrap();
                assert!(proposal
                    .recurrent_hidden
                    .iter()
                    .all(|value| { value.is_finite() && value.to_bits() & 0xffff == 0 }));
                if repetition == 0 && step_index == 0 {
                    report_stage_diagnostics(&proposal.stage_snapshots, Some(&stage_oracle));
                    structural_stage_pass = true;
                }
                let mut logits = vec![0.0f32; VOCAB];
                read_buffer_f32(&assistant.scratch.logits, &mut logits);
                assert!(logits
                    .iter()
                    .all(|value| value.is_finite() && value.to_bits() & 0xffff == 0));
                let actual_top16 = top_token_ids(&logits, 16);
                assert_eq!(proposal.token as usize, actual_top16[0]);
                assert_eq!(
                    proposal.token, expected.top1_token_id,
                    "recurrence token diverged at step {step_index}; do not teacher-force the next input"
                );
                let overlap = actual_top16
                    .iter()
                    .filter(|token| expected.top16_token_ids.contains(&(**token as u32)))
                    .count();
                assert!(
                    overlap >= MIN_TOP16_OVERLAP,
                    "step {step_index} top-16 overlap {overlap}/16 is below {MIN_TOP16_OVERLAP}/16"
                );
                let native_margin = ordered_bf16(f32_to_bf16_rne_bits(logits[actual_top16[0]]))
                    .abs_diff(ordered_bf16(f32_to_bf16_rne_bits(logits[actual_top16[1]])));
                let required_native_margin =
                    expected_margin.min(oracle.admission.native_margin_cap_bf16_ulp);
                assert_eq!(
                    required_native_margin, PINNED_REQUIRED_MARGIN_FLOORS[step_index],
                    "step {step_index} pinned native margin floor changed"
                );
                assert!(
                    native_margin >= required_native_margin,
                    "step {step_index} native top-1 margin {native_margin} BF16 ULP is below the per-step floor {required_native_margin}"
                );
                let (cosine, relative_l2) = recurrence_hidden_similarity(
                    &proposal.recurrent_hidden,
                    &expected.recurrent_hidden_bf16_bits,
                );
                assert!(
                    cosine >= MIN_RECURRENT_COSINE,
                    "step {step_index} recurrent cosine {cosine:.9} is below {MIN_RECURRENT_COSINE}"
                );
                assert!(
                    relative_l2 <= MAX_RECURRENT_RELATIVE_L2,
                    "step {step_index} recurrent relative-L2 {relative_l2:.9} exceeds {MAX_RECURRENT_RELATIVE_L2}"
                );
                let recurrent_hash = widened_bf16_sha256(&proposal.recurrent_hidden);
                let logits_hash = widened_bf16_sha256(&logits);
                eprintln!(
                    "MTP_RECURRENCE_COMPARE repetition={} step={} expected_top1={} actual_top1={} top16_overlap={}/16 expected_margin_bf16_ulp={} actual_margin_bf16_ulp={} recurrent_cosine={:.9} recurrent_relative_l2={:.9} expected_logits_sha256={} actual_logits_sha256={} expected_recurrent_sha256={} actual_recurrent_sha256={} borrowed_target_kv_capacity_bytes={}",
                    repetition,
                    step_index,
                    expected.top1_token_id,
                    proposal.token,
                    overlap,
                    expected_margin,
                    native_margin,
                    cosine,
                    relative_l2,
                    expected.logits_bf16_sha256,
                    logits_hash,
                    expected.recurrent_hidden_bf16_sha256,
                    recurrent_hash,
                    proposal.ledger.borrowed_target_kv_capacity_bytes,
                );
                native_run.push(NativeRecurrenceStep {
                    token: proposal.token,
                    top16_token_ids: actual_top16,
                    logits_bf16_sha256: logits_hash,
                    recurrent_hidden_bf16_sha256: recurrent_hash,
                });
                admission_steps.push(NativeAdmissionStepEvidence {
                    step: step_index,
                    token_id: proposal.token,
                    top16_overlap: overlap,
                    recurrent_cosine: cosine,
                    recurrent_relative_l2: relative_l2,
                    native_margin_bf16_ulp: native_margin,
                    required_margin_bf16_ulp: required_native_margin,
                });
                recurrent_hidden = proposal.recurrent_hidden;
                expected_previous_recurrent_hash = &expected.recurrent_hidden_bf16_sha256;
                if step_index + 1 < PROPOSALS {
                    let (next_embedding, next_embedding_hash) =
                        recurrence_feedback_embedding(proposal.token, &oracle.feedback_embedding);
                    assert_eq!(
                        next_embedding_hash,
                        oracle.steps[step_index + 1].input_embedding_bf16_sha256
                    );
                    embedding = next_embedding;
                }
            }

            if let Some(first) = first_native_run.as_ref() {
                assert_eq!(
                    &native_run, first,
                    "native seven-step recurrence was not bit-deterministic across repetitions"
                );
            } else {
                first_native_run = Some(native_run);
                first_admission_steps = Some(admission_steps);
            }
        }

        assert!(
            structural_stage_pass,
            "schema-v2 admission did not execute the step-0 structural stage validator"
        );
        {
            let native_run = first_native_run.expect("at least one native recurrence repetition");
            let per_step =
                first_admission_steps.expect("at least one native recurrence metrics pass");
            assert_eq!(native_run.len(), PROPOSALS);
            assert_eq!(per_step.len(), PROPOSALS);
            let min_top16_overlap = per_step
                .iter()
                .map(|step| step.top16_overlap)
                .min()
                .unwrap();
            let min_recurrent_cosine = per_step
                .iter()
                .map(|step| step.recurrent_cosine)
                .fold(f64::INFINITY, f64::min);
            let max_recurrent_relative_l2 = per_step
                .iter()
                .map(|step| step.recurrent_relative_l2)
                .fold(0.0f64, f64::max);
            let current_exe = std::env::current_exe().expect("resolve admission test executable");
            let admission_test_exe_sha256 = sha256_hex(
                &std::fs::read(&current_exe).expect("read admission test executable for SHA-256"),
            );
            let platform = NativeAdmissionPlatform {
                os: std::env::consts::OS.to_owned(),
                os_version: native_admission_command_output(
                    "/usr/bin/sw_vers",
                    &["-productVersion"],
                ),
                machine_arch: std::env::consts::ARCH.to_owned(),
                machine_model: native_admission_command_output(
                    "/usr/sbin/sysctl",
                    &["-n", "hw.model"],
                ),
                metal_device_name: device.name().to_owned(),
            };
            let created_unix_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock predates Unix epoch")
                .as_millis()
                .try_into()
                .expect("native admission timestamp does not fit u64");
            let evidence = NativeAdmissionEvidence {
                schema_version: 1,
                pass: true,
                assistant_model_sha256: EXPECTED_SHA256.to_owned(),
                recurrence_oracle_sha256: oracle_sha256,
                recurrence_generation_receipt_sha256,
                stage_oracle_sha256,
                structural_stage_pass,
                admission_test_exe_sha256,
                native_source_sha256: sha256_hex(include_str!("gemma4_mtp.rs").as_bytes()),
                metal_rs_source_sha256: sha256_hex(include_str!("../metal.rs").as_bytes()),
                gemma4_runtime_rs_source_sha256: sha256_hex(
                    include_str!("../gemma4_runtime.rs").as_bytes(),
                ),
                cargo_lock_sha256: sha256_hex(include_str!("../../Cargo.lock").as_bytes()),
                platform,
                created_unix_ms,
                run_nonce: evidence_run_nonce,
                test_name: concat!(
                    module_path!(),
                    "::official_target_free_bf16_oracle_matches_native_seven_proposal_recurrence"
                )
                .to_owned(),
                top1_token_ids: native_run.iter().map(|step| step.token).collect(),
                native_repetitions: oracle.admission.required_native_repetitions,
                repeat_bit_deterministic: true,
                per_step,
                min_top16_overlap,
                min_recurrent_cosine,
                max_recurrent_relative_l2,
                tie_policy: oracle.authority.argmax_tie_policy,
                tie_policy_test_pass,
                teacher_forcing: oracle.admission.teacher_force_after_mismatch,
                thresholds: NativeAdmissionThresholds {
                    exact_top1_steps: PROPOSALS,
                    minimum_top16_overlap_per_step: MIN_TOP16_OVERLAP,
                    minimum_recurrent_cosine_per_step: MIN_RECURRENT_COSINE,
                    maximum_recurrent_relative_l2_per_step: MAX_RECURRENT_RELATIVE_L2,
                    native_margin_cap_bf16_ulp: NATIVE_MARGIN_CAP_BF16_ULP,
                    native_margin_floor_rule: oracle.admission.native_margin_floor_rule,
                    required_margin_floors_bf16_ulp: PINNED_REQUIRED_MARGIN_FLOORS.to_vec(),
                    require_bf16_lattice: true,
                    require_repeat_bit_determinism: true,
                },
            };
            emit_native_admission_evidence_atomically(&evidence_output, &evidence);
            eprintln!(
                "MTP_NATIVE_ADMISSION_EVIDENCE path={} oracle_sha256={} native_source_sha256={} test_exe_sha256={}",
                evidence_output.display(),
                evidence.recurrence_oracle_sha256,
                evidence.native_source_sha256,
                evidence.admission_test_exe_sha256,
            );
        }
    }
}
