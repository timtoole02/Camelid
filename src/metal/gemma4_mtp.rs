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
const MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX: usize = 3;
const MTP_DEVICE_CHAIN_K4_WARM_DRAFTS: usize = MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX + 1;
const MTP_DEVICE_CHAIN_DISPATCHES_PER_DRAFT: usize = 112;
const MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT: usize = 33;
const MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT: usize = 50_176;
const MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT: usize =
    MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT * std::mem::size_of::<f32>() * 2;
const MTP_DEVICE_CHAIN_K4_WARM_DISPATCHES: usize =
    MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * MTP_DEVICE_CHAIN_DISPATCHES_PER_DRAFT + 1;
const MTP_DEVICE_CHAIN_K4_WARM_RESTORE_BYTES: usize =
    (MTP_DEVICE_CHAIN_K4_WARM_DRAFTS + 1) * TARGET_HIDDEN * std::mem::size_of::<f32>()
        + MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * std::mem::size_of::<u32>();
const TARGET_Q6K_VALUES_PER_BLOCK: usize = 256;
const TARGET_Q6K_WIRE_BYTES_PER_BLOCK: usize = 210;
const TARGET_Q6K_ROW_BYTES: usize =
    (TARGET_HIDDEN / TARGET_Q6K_VALUES_PER_BLOCK) * TARGET_Q6K_WIRE_BYTES_PER_BLOCK;
const MTP_STEP3_LOGIT_TRACE_ENV: &str = "CAMELID_GEMMA4_MTP_STEP3_LOGIT_TRACE";
const MTP_LOGIT_TRACE_DRAFT_INDEX_ENV: &str = "CAMELID_GEMMA4_MTP_LOGIT_TRACE_DRAFT_INDEX";
const MTP_BF16_PRODUCER_FUSION_ENV: &str = "CAMELID_GEMMA4_MTP_BF16_PRODUCER_FUSION";
const MTP_BF16_LATTICE_LOADS_ENV: &str = "CAMELID_GEMMA4_MTP_BF16_LATTICE_LOADS";
const MTP_BF16_LATTICE_DISPATCHES_ELIDED: usize = 0;
const MTP_BF16_LATTICE_BYTES_ELIDED: usize = 0;
const MTP_BF16_LATTICE_SCRATCH_BYTES_ADDED: usize = 0;
const MTP_BF16_LATTICE_CANDIDATE_PSOS_COMPILED: usize = 2;
// Official Gemma 4 proportional RoPE keeps the normal split-half geometry for
// the entire 512-wide head, but gives only the first quarter of dimensions a
// non-zero angle: 512 * 0.25 / 2 = 64 active pairs.  In particular, pair d is
// (d, d + 256), not (d, d + 64).  This must match the target layer-29 K cache.
const FULL_ROPE_ACTIVE_PAIRS: usize = FULL_HEAD_DIM / 8;
const RMS_EPS: f32 = 1.0e-6;
const MATRIX_BYTES_PER_PROPOSAL: u64 = 839_385_088;
const EMBEDDING_BF16_BYTES: u64 = 536_870_912;
const FULL_Q4_MATRIX_BYTES: u64 = 236_077_056;
const Q4_0_BLOCK_VALUES: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;

const fn mtp_device_chain_k4_warm_dispatches(bf16_producer_fusion: bool) -> usize {
    if bf16_producer_fusion {
        MTP_DEVICE_CHAIN_K4_WARM_DISPATCHES
            - MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT
    } else {
        MTP_DEVICE_CHAIN_K4_WARM_DISPATCHES
    }
}

/// Scalar BF16 RNE operations skipped by the H62 attention consumers for one
/// assistant draft at a particular target logical K. Query is reused once per
/// visible key in QK and each probability is reused once per output dimension
/// in context, so both sides contribute the same count. Dispatches, bytes and
/// scratch are unchanged.
fn mtp_bf16_lattice_rounds_elided_per_draft(logical_len: usize) -> Option<usize> {
    let (_, local_count) = assistant_local_attention_bounds(logical_len);
    let local_q_dim = N_HEADS.checked_mul(LOCAL_HEAD_DIM)?;
    let full_q_dim = N_HEADS.checked_mul(FULL_HEAD_DIM)?;
    let local = 3usize
        .checked_mul(2)?
        .checked_mul(local_q_dim)?
        .checked_mul(local_count)?;
    let full = 2usize.checked_mul(full_q_dim)?.checked_mul(logical_len)?;
    local.checked_add(full)
}

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

kernel void mtp_residual_add_bf16_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = mtp_round_bf16(a[gid] + b[gid]);
}

kernel void mtp_scale_bf16_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = mtp_round_bf16(input[gid] * scale);
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
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32;
    device const uchar* row_bytes = q4_weights + weight_byte_offset
        + ulong(row) * ulong(blocks_per_row) * 18ul;
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
    if (lane == 0) output[row] = round_output_bf16 != 0u
        ? mtp_round_bf16(value)
        : value;
}

// H66 exact assistant-head probe. The established kernel launches one
// SIMDgroup for every vocabulary row even though all rows consume the same
// 1,024-value input. These variants keep each row's scalar accumulation and
// reduction tree unchanged while one SIMDgroup evaluates two or four adjacent
// rows and reuses the input loads. They add no barriers or threadgroup memory.
inline float mtp_q4_0_nibble_term_f32acc(
    float d,
    uchar4 wb,
    float x0,
    float x1,
    float x2,
    float x3,
    float x16,
    float x17,
    float x18,
    float x19) {
    return d * (float(int(wb.x & 0x0f) - 8) * x0 + float(int(wb.x >> 4) - 8) * x16
              + float(int(wb.y & 0x0f) - 8) * x1 + float(int(wb.y >> 4) - 8) * x17
              + float(int(wb.z & 0x0f) - 8) * x2 + float(int(wb.z >> 4) - 8) * x18
              + float(int(wb.w & 0x0f) - 8) * x3 + float(int(wb.w >> 4) - 8) * x19);
}

inline float mtp_q4_0_reduce_f32acc(float partial) {
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 = simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 = simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    return pair01 + pair23;
}

kernel void mtp_q4_0_gemv_rows2_f32acc(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint row0 = row_group * 2u;
    if (row0 >= rows) return;
    const uint row1 = row0 + 1u;
    const uint blocks_per_row = cols / 32;
    const ulong row_bytes = ulong(blocks_per_row) * 18ul;
    device const uchar* matrix = q4_weights + weight_byte_offset;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32) {
        const uint in_base = b * 32;
        const ulong block_offset = ulong(b) * 18ul;
        device const uchar* block0 = matrix + ulong(row0) * row_bytes + block_offset;
        device const uchar* block1 = block0;
        if (row1 < rows) {
            block1 = matrix + ulong(row1) * row_bytes + block_offset;
        }
        const float d0 = float(*reinterpret_cast<device const half*>(block0));
        const float d1 = float(*reinterpret_cast<device const half*>(block1));
        device const packed_uchar4* q40 =
            reinterpret_cast<device const packed_uchar4*>(block0 + 2);
        device const packed_uchar4* q41 =
            reinterpret_cast<device const packed_uchar4*>(block1 + 2);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            const uint k4 = k * 4;
            const float x0 = input[in_base + k4];
            const float x1 = input[in_base + k4 + 1];
            const float x2 = input[in_base + k4 + 2];
            const float x3 = input[in_base + k4 + 3];
            const float x16 = input[in_base + 16 + k4];
            const float x17 = input[in_base + 17 + k4];
            const float x18 = input[in_base + 18 + k4];
            const float x19 = input[in_base + 19 + k4];
            partial0 += mtp_q4_0_nibble_term_f32acc(
                d0, uchar4(q40[k]), x0, x1, x2, x3, x16, x17, x18, x19);
            partial1 += mtp_q4_0_nibble_term_f32acc(
                d1, uchar4(q41[k]), x0, x1, x2, x3, x16, x17, x18, x19);
        }
    }
    const float value0 = mtp_q4_0_reduce_f32acc(partial0);
    const float value1 = mtp_q4_0_reduce_f32acc(partial1);
    if (lane == 0) {
        output[row0] = round_output_bf16 != 0u ? mtp_round_bf16(value0) : value0;
        if (row1 < rows) {
            output[row1] = round_output_bf16 != 0u ? mtp_round_bf16(value1) : value1;
        }
    }
}

kernel void mtp_q4_0_gemv_rows4_f32acc(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint row0 = row_group * 4u;
    if (row0 >= rows) return;
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const uint blocks_per_row = cols / 32;
    const ulong row_bytes = ulong(blocks_per_row) * 18ul;
    device const uchar* matrix = q4_weights + weight_byte_offset;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32) {
        const uint in_base = b * 32;
        const ulong block_offset = ulong(b) * 18ul;
        device const uchar* block0 = matrix + ulong(row0) * row_bytes + block_offset;
        device const uchar* block1 = block0;
        device const uchar* block2 = block0;
        device const uchar* block3 = block0;
        if (row1 < rows) {
            block1 = matrix + ulong(row1) * row_bytes + block_offset;
        }
        if (row2 < rows) {
            block2 = matrix + ulong(row2) * row_bytes + block_offset;
        }
        if (row3 < rows) {
            block3 = matrix + ulong(row3) * row_bytes + block_offset;
        }
        const float d0 = float(*reinterpret_cast<device const half*>(block0));
        const float d1 = float(*reinterpret_cast<device const half*>(block1));
        const float d2 = float(*reinterpret_cast<device const half*>(block2));
        const float d3 = float(*reinterpret_cast<device const half*>(block3));
        device const packed_uchar4* q40 =
            reinterpret_cast<device const packed_uchar4*>(block0 + 2);
        device const packed_uchar4* q41 =
            reinterpret_cast<device const packed_uchar4*>(block1 + 2);
        device const packed_uchar4* q42 =
            reinterpret_cast<device const packed_uchar4*>(block2 + 2);
        device const packed_uchar4* q43 =
            reinterpret_cast<device const packed_uchar4*>(block3 + 2);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            const uint k4 = k * 4;
            const float x0 = input[in_base + k4];
            const float x1 = input[in_base + k4 + 1];
            const float x2 = input[in_base + k4 + 2];
            const float x3 = input[in_base + k4 + 3];
            const float x16 = input[in_base + 16 + k4];
            const float x17 = input[in_base + 17 + k4];
            const float x18 = input[in_base + 18 + k4];
            const float x19 = input[in_base + 19 + k4];
            partial0 += mtp_q4_0_nibble_term_f32acc(
                d0, uchar4(q40[k]), x0, x1, x2, x3, x16, x17, x18, x19);
            partial1 += mtp_q4_0_nibble_term_f32acc(
                d1, uchar4(q41[k]), x0, x1, x2, x3, x16, x17, x18, x19);
            partial2 += mtp_q4_0_nibble_term_f32acc(
                d2, uchar4(q42[k]), x0, x1, x2, x3, x16, x17, x18, x19);
            partial3 += mtp_q4_0_nibble_term_f32acc(
                d3, uchar4(q43[k]), x0, x1, x2, x3, x16, x17, x18, x19);
        }
    }
    const float value0 = mtp_q4_0_reduce_f32acc(partial0);
    const float value1 = mtp_q4_0_reduce_f32acc(partial1);
    const float value2 = mtp_q4_0_reduce_f32acc(partial2);
    const float value3 = mtp_q4_0_reduce_f32acc(partial3);
    if (lane == 0) {
        output[row0] = round_output_bf16 != 0u ? mtp_round_bf16(value0) : value0;
        if (row1 < rows) {
            output[row1] = round_output_bf16 != 0u ? mtp_round_bf16(value1) : value1;
        }
        if (row2 < rows) {
            output[row2] = round_output_bf16 != 0u ? mtp_round_bf16(value2) : value2;
        }
        if (row3 < rows) {
            output[row3] = round_output_bf16 != 0u ? mtp_round_bf16(value3) : value3;
        }
    }
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

// H62 is admitted only for buffers whose producer has already stored widened
// BF16. Every production query offset is float4-aligned: head dimensions are
// 256/512 and the pinned QK loop advances in multiples of four.
inline float4 mtp_load_bf16_latticex4(
    device const float* values,
    uint base) {
    return *reinterpret_cast<device const float4*>(values + base);
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

// Exact H62 twin of mtp_attention_scores_bf16_f32. RoPE has already stored
// query as widened BF16, so its idempotent RNE load may be removed. Target KV
// remains f32 and deliberately retains RNE at every key load.
kernel void mtp_attention_scores_bf16_lattice_query_f32(
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
            acc0 += mtp_load_bf16_latticex4(query, q_base + d + 0) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 0);
            acc1 += mtp_load_bf16_latticex4(query, q_base + d + 4) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 4);
            acc2 += mtp_load_bf16_latticex4(query, q_base + d + 8) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 8);
            acc3 += mtp_load_bf16_latticex4(query, q_base + d + 12) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 12);
            acc4 += mtp_load_bf16_latticex4(query, q_base + d + 16) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 16);
            acc5 += mtp_load_bf16_latticex4(query, q_base + d + 20) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 20);
            acc6 += mtp_load_bf16_latticex4(query, q_base + d + 24) *
                    mtp_load_rounded_bf16x4(keys, k_base + d + 24);
            acc7 += mtp_load_bf16_latticex4(query, q_base + d + 28) *
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

// Exact H62 twin of mtp_attention_context_bf16_f32. Softmax stores widened
// BF16 probabilities in place, so their idempotent RNE load may be removed.
// Target V remains f32 and deliberately retains RNE at every value load.
kernel void mtp_attention_context_bf16_lattice_probabilities_f32(
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
    const uint physical_vector_end = physical_logical_k & ~3u;
    for (uint d = lane; d < head_dim; d += 32) {
        float p0 = 0.0f;
        float p1 = 0.0f;
        float p2 = 0.0f;
        float p3 = 0.0f;
        for (uint p = 0; p < position_count; ++p) {
            const uint absolute_position = compact_base + p;
            const float product =
                probabilities[score_base + p] *
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
    constant uint& round_output_bf16 [[buffer(6)]],
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
        output[base + index] = round_output_bf16 != 0u
            ? mtp_round_bf16(value)
            : value;
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Q4TensorRef {
    byte_offset: u64,
    byte_len: u64,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FullQ4LayerWeights {
    q: Q4TensorRef,
    o: Q4TensorRef,
    gate: Q4TensorRef,
    up: Q4TensorRef,
    down: Q4TensorRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FullQ4Layout {
    embedding: Q4TensorRef,
    layers: [FullQ4LayerWeights; 4],
    pre_projection: Q4TensorRef,
    post_projection: Q4TensorRef,
    matrix_bytes: u64,
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

fn parse_full_q4_opt_in(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    parse_device_chain_opt_in(value)
}

fn parse_bf16_producer_fusion_opt_in(
    value: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    parse_device_chain_opt_in(value)
}

fn parse_bf16_lattice_loads_opt_in(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    parse_device_chain_opt_in(value)
}

fn mtp_step3_logit_trace_enabled_value(value: Option<&str>) -> bool {
    value.map(str::trim).is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn mtp_step3_logit_capture_enabled_values(
    trace_value: Option<&str>,
    explicit_capture: bool,
) -> bool {
    mtp_step3_logit_trace_enabled_value(trace_value) || explicit_capture
}

fn mtp_step3_logit_capture_enabled(explicit_capture: bool) -> bool {
    let trace_value = std::env::var(MTP_STEP3_LOGIT_TRACE_ENV).ok();
    mtp_step3_logit_capture_enabled_values(trace_value.as_deref(), explicit_capture)
}

fn mtp_logit_trace_draft_index_value(value: Option<&str>) -> Option<usize> {
    value
        .map(str::trim)?
        .parse::<usize>()
        .ok()
        .filter(|&index| index < MTP_CHAIN_MAX_DRAFTS)
}

fn mtp_logit_capture_draft_index(explicit_step3_capture: bool) -> Option<usize> {
    if mtp_step3_logit_capture_enabled(explicit_step3_capture) {
        Some(MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX)
    } else {
        mtp_logit_trace_draft_index_value(
            std::env::var(MTP_LOGIT_TRACE_DRAFT_INDEX_ENV)
                .ok()
                .as_deref(),
        )
    }
}

fn full_q4_requested_from_environment() -> Result<bool> {
    const NAME: &str = "CAMELID_GEMMA4_MTP_FULL_Q4";
    match std::env::var(NAME) {
        Ok(value) => parse_full_q4_opt_in(Some(&value))
            .map_err(|detail| invalid(format!("{NAME} {detail}, got {value:?}"))),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(invalid(format!("{NAME} must contain Unicode text")))
        }
    }
}

fn bf16_producer_fusion_requested_from_environment() -> Result<bool> {
    match std::env::var(MTP_BF16_PRODUCER_FUSION_ENV) {
        Ok(value) => parse_bf16_producer_fusion_opt_in(Some(&value)).map_err(|detail| {
            invalid(format!(
                "{MTP_BF16_PRODUCER_FUSION_ENV} {detail}, got {value:?}"
            ))
        }),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err(invalid(format!(
            "{MTP_BF16_PRODUCER_FUSION_ENV} must contain Unicode text"
        ))),
    }
}

fn bf16_lattice_loads_requested_from_environment() -> Result<bool> {
    match std::env::var(MTP_BF16_LATTICE_LOADS_ENV) {
        Ok(value) => parse_bf16_lattice_loads_opt_in(Some(&value)).map_err(|detail| {
            invalid(format!(
                "{MTP_BF16_LATTICE_LOADS_ENV} {detail}, got {value:?}"
            ))
        }),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err(invalid(format!(
            "{MTP_BF16_LATTICE_LOADS_ENV} must contain Unicode text"
        ))),
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
    residual_add_bf16: ComputePipelineState,
    scale_bf16: ComputePipelineState,
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
    attention_scores_bf16_lattice_query: ComputePipelineState,
    #[cfg(test)]
    attention_scores_legacy_bf16: ComputePipelineState,
    attention_softmax_bf16: ComputePipelineState,
    attention_context_bf16: ComputePipelineState,
    attention_context_bf16_lattice_probabilities: ComputePipelineState,
    #[cfg(test)]
    attention_context_legacy_bf16: ComputePipelineState,
    rms_norm_aten_f32: ComputePipelineState,
    argmax: ComputePipelineState,
    gather_q6k_embed_and_recurrent: ComputePipelineState,
    q4_0_gemv: ComputePipelineState,
    #[cfg(test)]
    q4_0_gemv_rows2: ComputePipelineState,
    #[cfg(test)]
    q4_0_gemv_rows4: ComputePipelineState,
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
            residual_add_bf16: pipeline("mtp_residual_add_bf16_f32")?,
            scale_bf16: pipeline("mtp_scale_bf16_f32")?,
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
            attention_scores_bf16_lattice_query: pipeline(
                "mtp_attention_scores_bf16_lattice_query_f32",
            )?,
            #[cfg(test)]
            attention_scores_legacy_bf16: test_diagnostic_pipeline(
                "mtp_test_attention_scores_legacy_bf16_f32",
            )?,
            attention_softmax_bf16: pipeline("mtp_attention_softmax_bf16_f32")?,
            attention_context_bf16: pipeline("mtp_attention_context_bf16_f32")?,
            attention_context_bf16_lattice_probabilities: pipeline(
                "mtp_attention_context_bf16_lattice_probabilities_f32",
            )?,
            #[cfg(test)]
            attention_context_legacy_bf16: test_diagnostic_pipeline(
                "mtp_test_attention_context_legacy_bf16_f32",
            )?,
            rms_norm_aten_f32: pipeline("mtp_rms_norm_aten_f32")?,
            argmax: pipeline("mtp_argmax_f32")?,
            gather_q6k_embed_and_recurrent: pipeline("mtp_gather_q6k_embed_and_recurrent")?,
            q4_0_gemv: pipeline("mtp_q4_0_gemv_f32acc")?,
            #[cfg(test)]
            q4_0_gemv_rows2: pipeline("mtp_q4_0_gemv_rows2_f32acc")?,
            #[cfg(test)]
            q4_0_gemv_rows4: pipeline("mtp_q4_0_gemv_rows4_f32acc")?,
        })
    }

    fn selected_bf16_gemv(&self) -> &ComputePipelineState {
        &self.bf16_gemv
    }

    fn selected_attention_scores_bf16(&self, bf16_lattice_loads: bool) -> &ComputePipelineState {
        if bf16_lattice_loads {
            &self.attention_scores_bf16_lattice_query
        } else {
            &self.attention_scores_bf16
        }
    }

    fn selected_attention_context_bf16(&self, bf16_lattice_loads: bool) -> &ComputePipelineState {
        if bf16_lattice_loads {
            &self.attention_context_bf16_lattice_probabilities
        } else {
            &self.attention_context_bf16
        }
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

struct FullQ4Weights {
    buffer: Buffer,
    layout: FullQ4Layout,
    quantize_us: u128,
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
    /// Anonymous packed Q4_0 assistant matrices. Zero on the established
    /// BF16-linear path (the separately packed tied head is not included).
    pub full_q4_matrix_bytes: u64,
    pub full_q4_quantize_us: u128,
    pub hash_us: u128,
    pub lock_and_residency_us: u128,
    pub pipeline_compile_us: u128,
    pub load_wall_us: u128,
}

/// Per-proposal byte accounting. `target_kv_read_bytes` is the exact logical K
/// plus V span traversed by the three sliding and one full attention layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// Default-off one-shot diagnostic snapshot. Boundary arbitration attaches
    /// it to draft 3; the indexed trace can attach it to another draft.
    pub(crate) step3_assistant_logits: Option<Vec<f32>>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpDeviceChainInvocation {
    Production,
    TargetFreeK4Warmup,
}

/// Explicit, default-off native assistant. The established path retains a
/// locked BF16 file mapping; full-Q4 releases it after every matrix and norm
/// needed at runtime has been copied into independently owned Metal buffers.
pub struct Gemma4MtpAssistantMetal {
    weight_file: Buffer,
    pipelines: MtpPipelines,
    layers: Vec<LayerWeights>,
    final_norm: Buffer,
    embedding: TensorRef,
    q4_embedding: Option<Buffer>,
    full_q4: Option<FullQ4Weights>,
    pre_projection: TensorRef,
    post_projection: TensorRef,
    scratch: MtpScratch,
    queue: metal::CommandQueue,
    bf16_producer_fusion: bool,
    bf16_lattice_loads: bool,
    resident_ledger: Gemma4MtpResidentLedger,
    last_proposal_ledger: Option<Gemma4MtpProposalLedger>,
    source_path: PathBuf,
    // Must remain last: Rust drops struct fields in declaration order, so all
    // no-copy Metal buffers are released before this unlocks/unmaps the pages.
    _locked_mapping: Option<LockedAssistantMapping>,
}

fn shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
}

fn read_buffer_prefix_bytes(buffer: &Buffer, byte_len: usize) -> Result<Vec<u8>> {
    let buffer_len = usize::try_from(buffer.length())
        .map_err(|_| invalid("Metal buffer length exceeds usize"))?;
    if byte_len > buffer_len {
        return Err(invalid(format!(
            "Metal buffer prefix read {byte_len} exceeds buffer length {buffer_len}"
        )));
    }
    let mut bytes = vec![0u8; byte_len];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents().cast::<u8>(), bytes.as_mut_ptr(), byte_len);
    }
    Ok(bytes)
}

fn write_buffer_prefix_bytes(buffer: &Buffer, bytes: &[u8]) -> Result<()> {
    let buffer_len = usize::try_from(buffer.length())
        .map_err(|_| invalid("Metal buffer length exceeds usize"))?;
    if bytes.len() > buffer_len {
        return Err(invalid(format!(
            "Metal buffer prefix write {} exceeds buffer length {buffer_len}",
            bytes.len()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast::<u8>(), bytes.len());
    }
    Ok(())
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
            output_token: shared_buffer(device, MTP_CHAIN_MAX_DRAFTS * std::mem::size_of::<u32>()),
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
    /// Load the exact staged official artifact. The established BF16 path
    /// retains its hard pin; full-Q4 releases the source after packing.
    pub fn load_staged_official() -> Result<Self> {
        Self::load(Path::new(OFFICIAL_STAGED_ASSISTANT_PATH))
    }

    /// Load an exact byte-identical copy of the official artifact. Shape,
    /// offsets, config, file length and SHA-256 are all pinned before mlock.
    pub fn load(path: &Path) -> Result<Self> {
        let full_q4 = full_q4_requested_from_environment()?;
        let bf16_producer_fusion = bf16_producer_fusion_requested_from_environment()?;
        let bf16_lattice_loads = bf16_lattice_loads_requested_from_environment()?;
        Self::load_with_options(path, full_q4, bf16_producer_fusion, bf16_lattice_loads)
    }

    fn load_with_full_q4(path: &Path, full_q4_requested: bool) -> Result<Self> {
        Self::load_with_options(path, full_q4_requested, false, false)
    }

    fn load_with_options(
        path: &Path,
        full_q4_requested: bool,
        bf16_producer_fusion: bool,
        bf16_lattice_loads: bool,
    ) -> Result<Self> {
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
        let pre_projection = manifest.matrix("pre_projection.weight")?;
        let post_projection = manifest.matrix("post_projection.weight")?;
        let full_q4 = full_q4_requested
            .then(|| {
                quantize_full_assistant_to_q4_0(
                    &kernel.device,
                    &locked_mapping.mapping,
                    embedding,
                    &layers,
                    pre_projection,
                    post_projection,
                )
            })
            .transpose()?;
        // Preserve the established best-effort Q4 tied-head optimization when
        // full-Q4 is off. The explicit full-Q4 admission above is fail-closed:
        // it never falls back to a partially quantized assistant.
        let q4_embedding = if full_q4.is_none() {
            quantize_embedding_to_q4_0(&kernel.device, &locked_mapping.mapping, embedding).ok()
        } else {
            None
        };
        let scratch = MtpScratch::new(&kernel.device);
        let full_q4_matrix_bytes = full_q4
            .as_ref()
            .map_or(0, |weights| weights.layout.matrix_bytes);
        let full_q4_quantize_us = full_q4.as_ref().map_or(0, |weights| weights.quantize_us);
        let file_bytes = locked_mapping.mapping.file_len();
        let decoded_norm_bytes = layers
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
            + final_norm.length();

        // Every full-Q4 runtime matrix must be backed by an exact, contiguous
        // packed slice before the sole BF16 source mapping may be released.
        // The four-byte buffer preserves existing encoder signatures; the
        // validated full-Q4 branch never dereferences it.
        let (
            weight_file,
            retained_mapping,
            mapped_bytes,
            locked_bytes,
            resident_pages,
            total_pages,
        ) = if let Some(weights) = full_q4.as_ref() {
            weights.layout.validate_complete(
                embedding,
                &layers,
                pre_projection,
                post_projection,
                weights.buffer.length(),
            )?;
            drop(locked_mapping);
            (shared_buffer(&kernel.device, 4), None, 0, 0, 0, 0)
        } else {
            let mapped_bytes = locked_mapping.mapping.mapped_len() as u64;
            let locked_bytes = locked_mapping.locked_bytes as u64;
            let resident_pages = locked_mapping.resident_pages as u64;
            let total_pages = locked_mapping.total_pages as u64;
            let weight_file = kernel.device.new_buffer_with_bytes_no_copy(
                locked_mapping.mapping.base_ptr().cast::<c_void>(),
                locked_mapping.mapping.mapped_len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            (
                weight_file,
                Some(locked_mapping),
                mapped_bytes,
                locked_bytes,
                resident_pages,
                total_pages,
            )
        };
        let resident_ledger = Gemma4MtpResidentLedger {
            file_bytes,
            mapped_bytes,
            locked_bytes,
            resident_pages,
            total_pages,
            payload_bytes: EXPECTED_PAYLOAD_BYTES as u64,
            decoded_norm_bytes,
            fixed_scratch_bytes: scratch.byte_len(),
            full_q4_matrix_bytes,
            full_q4_quantize_us,
            hash_us,
            lock_and_residency_us,
            pipeline_compile_us,
            load_wall_us: load_started.elapsed().as_micros(),
        };

        if let Some(weights) = full_q4.as_ref() {
            eprintln!(
                "[gemma4-mtp full-q4] enabled=true source_sha256={} matrices=23 packed_bytes={} bf16_matrix_bytes={} quantize_us={} norms_quantized=false fallback=false",
                EXPECTED_SHA256,
                weights.layout.matrix_bytes,
                MATRIX_BYTES_PER_PROPOSAL,
                weights.quantize_us,
            );
            eprintln!(
                "[gemma4-mtp full-q4 residency] source_retained=false mapped_bytes={} locked_bytes={} resident_pages={} total_pages={} packed_bytes={}",
                resident_ledger.mapped_bytes,
                resident_ledger.locked_bytes,
                resident_ledger.resident_pages,
                resident_ledger.total_pages,
                weights.layout.matrix_bytes,
            );
        }
        eprintln!(
            "[gemma4-mtp bf16-producer-fusion] enabled={} standalone_round_dispatches_per_draft={} standalone_round_elements_per_draft={} standalone_round_rw_bytes_per_draft={} elided_round_dispatches_per_draft={} elided_round_elements_per_draft={} elided_round_rw_bytes_per_draft={} scratch_bytes_added=0",
            usize::from(bf16_producer_fusion),
            if bf16_producer_fusion {
                0
            } else {
                MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT
            },
            if bf16_producer_fusion {
                0
            } else {
                MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT
            },
            if bf16_producer_fusion {
                0
            } else {
                MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT
            },
            if bf16_producer_fusion {
                MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT
            } else {
                0
            },
            if bf16_producer_fusion {
                MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT
            } else {
                0
            },
            if bf16_producer_fusion {
                MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT
            } else {
                0
            },
        );
        eprintln!(
            "[gemma4-mtp bf16-lattice-loads] enabled={} direct_query_qk={} direct_probability_context={} dispatches_elided={} bytes_elided={} scratch_bytes_added={} candidate_psos_compiled={}",
            usize::from(bf16_lattice_loads),
            usize::from(bf16_lattice_loads),
            usize::from(bf16_lattice_loads),
            MTP_BF16_LATTICE_DISPATCHES_ELIDED,
            MTP_BF16_LATTICE_BYTES_ELIDED,
            MTP_BF16_LATTICE_SCRATCH_BYTES_ADDED,
            MTP_BF16_LATTICE_CANDIDATE_PSOS_COMPILED,
        );

        Ok(Self {
            weight_file,
            pipelines,
            layers,
            final_norm,
            embedding,
            q4_embedding,
            full_q4,
            pre_projection,
            post_projection,
            scratch,
            queue: kernel.device.new_command_queue(),
            bf16_producer_fusion,
            bf16_lattice_loads,
            resident_ledger,
            last_proposal_ledger: None,
            source_path: path.to_path_buf(),
            _locked_mapping: retained_mapping,
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
        self.q4_embedding.is_some() || self.full_q4.is_some()
    }

    #[doc(hidden)]
    pub fn full_q4_enabled(&self) -> bool {
        self.full_q4.is_some()
    }

    fn assistant_matrix_bytes_per_proposal(&self) -> u64 {
        if let Some(full_q4) = self.full_q4.as_ref() {
            full_q4.layout.matrix_bytes
        } else if let Some(q4_embedding) = self.q4_embedding.as_ref() {
            MATRIX_BYTES_PER_PROPOSAL - EMBEDDING_BF16_BYTES + q4_embedding.length()
        } else {
            MATRIX_BYTES_PER_PROPOSAL
        }
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
        self.warm_target_free_with_queue(false)
    }

    /// Warm the same target-free workload on the private queue used by the
    /// one-command device-resident draft chain. This is a separate opt-in so
    /// the established common-queue warmup remains the default control.
    #[doc(hidden)]
    pub fn warm_target_free_on_private_queue(&mut self) -> Result<Gemma4MtpProposalTiming> {
        self.warm_target_free_with_queue(true)
    }

    fn warm_target_free_with_queue(
        &mut self,
        private_queue: bool,
    ) -> Result<Gemma4MtpProposalTiming> {
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
        let proposal =
            self.propose_with_queue(&zero_target, &zero_target, target_kv, private_queue)?;
        self.last_proposal_ledger = previous_ledger;
        Ok(proposal.timing)
    }

    /// Warm the exact four-draft, step-3-capturing device-chain command graph
    /// on its private queue without borrowing any target-owned resource. The
    /// compact Q6_K binding contains one all-zero row and advertises a one-token
    /// synthetic vocabulary, so feedback can never address beyond that row.
    #[doc(hidden)]
    pub fn warm_target_free_device_chain_k4_step3_capture(
        &mut self,
    ) -> Result<Gemma4MtpProposalTiming> {
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

        let target_embedding_buffer = shared_buffer(&kernel.device, TARGET_Q6K_ROW_BYTES);
        write_buffer_prefix_bytes(&target_embedding_buffer, &vec![0u8; TARGET_Q6K_ROW_BYTES])?;
        let target_embedding = Gemma4MtpTargetEmbeddingView {
            buffer: &target_embedding_buffer,
            byte_offset: 0,
            byte_len: TARGET_Q6K_ROW_BYTES,
            hidden: TARGET_HIDDEN,
            vocab: 1,
            format: Gemma4MtpTargetEmbeddingFormat::Q6K,
        };
        let zero_recurrent = vec![0.0f32; TARGET_HIDDEN];

        let recurrent_bytes = TARGET_HIDDEN * std::mem::size_of::<f32>();
        let chain_recurrent_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * recurrent_bytes;
        let token_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * std::mem::size_of::<u32>();
        let saved_recurrent =
            read_buffer_prefix_bytes(&self.scratch.recurrent_hidden, recurrent_bytes)?;
        let saved_chain_recurrent =
            read_buffer_prefix_bytes(&self.scratch.chain_recurrent_hidden, chain_recurrent_bytes)?;
        let saved_tokens = read_buffer_prefix_bytes(&self.scratch.output_token, token_bytes)?;
        let previous_ledger = self.last_proposal_ledger;

        let warm_result = self.propose_chain_device_resident_with_step3_logit_capture_inner(
            0,
            &zero_recurrent,
            target_kv,
            target_embedding,
            MTP_DEVICE_CHAIN_K4_WARM_DRAFTS,
            &[],
            true,
            MtpDeviceChainInvocation::TargetFreeK4Warmup,
        );

        // Attempt every restoration before propagating either a warm failure or
        // a restoration failure. No result path may expose synthetic recurrence,
        // feedback tokens, or the synthetic per-proposal ledger.
        let recurrent_restore =
            write_buffer_prefix_bytes(&self.scratch.recurrent_hidden, &saved_recurrent);
        let chain_recurrent_restore =
            write_buffer_prefix_bytes(&self.scratch.chain_recurrent_hidden, &saved_chain_recurrent);
        let token_restore = write_buffer_prefix_bytes(&self.scratch.output_token, &saved_tokens);
        self.last_proposal_ledger = previous_ledger;
        recurrent_restore?;
        chain_recurrent_restore?;
        token_restore?;

        let proposals = warm_result?;
        let returned_drafts = proposals.len();
        let tokens_zero = proposals.iter().all(|proposal| proposal.token == 0);
        let recurrent_zero = proposals.iter().all(|proposal| {
            proposal
                .recurrent_hidden
                .iter()
                .all(|value| value.to_bits() & 0x7fff_ffff == 0)
        });
        let step3_snapshot_zero = proposals
            .get(MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX)
            .and_then(|proposal| proposal.step3_assistant_logits.as_ref())
            .is_some_and(|logits| {
                logits.len() == VOCAB
                    && logits
                        .iter()
                        .all(|value| value.to_bits() & 0x7fff_ffff == 0)
            });
        if returned_drafts != MTP_DEVICE_CHAIN_K4_WARM_DRAFTS
            || !tokens_zero
            || !recurrent_zero
            || !step3_snapshot_zero
        {
            return Err(invalid(format!(
                "target-free K4 device-chain warmup invariant failed: returned_drafts={returned_drafts} tokens_zero={} recurrent_zero={} step3_snapshot_zero={}",
                usize::from(tokens_zero),
                usize::from(recurrent_zero),
                usize::from(step3_snapshot_zero),
            )));
        }
        let timing = proposals
            .first()
            .expect("four-draft warmup invariant checked")
            .timing;
        eprintln!(
            "[gemma4-mtp device-chain-warmup] graph=k4-step3-capture requested_drafts={} returned_drafts={} command_buffers=1 commits=1 waits=1 dispatches={} bf16_producer_fusion={} standalone_round_dispatches_per_draft={} standalone_round_rw_bytes_per_draft={} elided_round_dispatches_per_draft={} elided_round_rw_bytes_per_draft={} queue=private-device-chain explicit_step3_capture=1 synthetic_embedding_rows=1 synthetic_embedding_bytes={} synthetic_vocab=1 synthetic_kv_len=1 target_buffers_borrowed=0 tokens_zero=1 recurrent_zero=1 step3_snapshot_zero=1 recurrent_hidden_restored=1 chain_recurrent_restored=1 token_scratch_restored=1 restored_scratch_bytes={} ledger_restored=1 target_state_mutation=0 output_published=0 encode_us={} wait_us={} gpu_us={} kernel_us={} wall_us={}",
            MTP_DEVICE_CHAIN_K4_WARM_DRAFTS,
            returned_drafts,
            mtp_device_chain_k4_warm_dispatches(self.bf16_producer_fusion),
            usize::from(self.bf16_producer_fusion),
            if self.bf16_producer_fusion {
                0
            } else {
                MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT
            },
            if self.bf16_producer_fusion {
                0
            } else {
                MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT
            },
            if self.bf16_producer_fusion {
                MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT
            } else {
                0
            },
            if self.bf16_producer_fusion {
                MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT
            } else {
                0
            },
            TARGET_Q6K_ROW_BYTES,
            MTP_DEVICE_CHAIN_K4_WARM_RESTORE_BYTES,
            timing.encode_us,
            timing.wait_us,
            timing.gpu_us,
            timing.kernel_us,
            timing.wall_us,
        );
        Ok(timing)
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
        self.propose_with_queue(
            target_scaled_embedding,
            pending_target_hidden,
            target_kv,
            false,
        )
    }

    fn propose_with_queue(
        &mut self,
        target_scaled_embedding: &[f32],
        pending_target_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        private_queue: bool,
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
        write_assistant_rope_tables(proposal_position, &self.scratch);

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

        let command_buffer = if private_queue {
            self.queue.new_command_buffer()
        } else {
            kernel.queue.new_command_buffer()
        };
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        #[cfg(test)]
        let mut pending_stage_snapshots = (std::env::var("CAMELID_GEMMA4_MTP_STAGE_DIAGNOSTICS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            || std::env::var_os("CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON").is_some())
        .then(Vec::new);
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            self.full_q4
                .as_ref()
                .map(|weights| weights.layout.pre_projection),
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
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.final_normalized,
                ASSISTANT_HIDDEN,
            );
        }
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
            "final_norm",
            &mut pending_stage_snapshots,
        );
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            self.full_q4
                .as_ref()
                .map(|weights| weights.layout.post_projection),
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
        if let Some(full_q4) = self.full_q4.as_ref() {
            encode_q4_0_gemv_packed(
                encoder,
                &self.pipelines.q4_0_gemv,
                &full_q4.buffer,
                &self.scratch.final_normalized,
                &self.scratch.logits,
                full_q4.layout.embedding,
                false,
            );
        } else if let Some(q4_emb) = self.q4_embedding.as_ref() {
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
            assistant_matrix_bytes: self.assistant_matrix_bytes_per_proposal(),
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
            step3_assistant_logits: None,
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
        get_token_embedding: F,
    ) -> Result<Vec<Gemma4MtpProposal>>
    where
        F: FnMut(u32, &mut [f32]) -> Result<()>,
    {
        self.propose_chain_with_step3_logit_capture(
            anchor_token,
            initial_recurrent_hidden,
            target_kv,
            draft_limit,
            eot,
            false,
            get_token_embedding,
        )
    }

    /// Chained assistant sequence with an optional one-shot step-3 logit
    /// snapshot. The existing trace environment remains an independent opt-in;
    /// callers use `capture_step3_logits` to request capture for this call only.
    pub(crate) fn propose_chain_with_step3_logit_capture<F>(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        draft_limit: usize,
        eot: &[u32],
        capture_step3_logits: bool,
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
        write_assistant_rope_tables(proposal_position, &self.scratch);

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
        let capture_step3_logits = draft_limit > MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX
            && mtp_step3_logit_capture_enabled(capture_step3_logits);

        let mut proposals = Vec::with_capacity(draft_limit);
        let mut current_token = anchor_token;
        let mut current_recurrent_hidden = initial_recurrent_hidden.to_vec();
        let pre_input_ptr = self.scratch.pre_input.contents().cast::<f32>();
        let mut embed_buf = [0.0f32; TARGET_HIDDEN];

        // Pre-fill initial recurrent hidden into second half of pre_input
        for i in 0..TARGET_HIDDEN {
            unsafe {
                *pre_input_ptr.add(TARGET_HIDDEN + i) =
                    round_to_bf16_f32(initial_recurrent_hidden[i]);
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

            encode_assistant_matrix(
                encoder,
                &self.pipelines,
                &self.weight_file,
                self.full_q4.as_ref(),
                self.full_q4
                    .as_ref()
                    .map(|weights| weights.layout.pre_projection),
                &self.scratch.pre_input,
                &self.scratch.hidden,
                self.pre_projection,
            );

            for (layer_index, layer) in self.layers.iter().enumerate() {
                let (
                    kv,
                    head_dim,
                    position_count,
                    cos,
                    sin,
                    qnorm_scalar,
                    rope_scalar,
                    attn_scalar,
                ) = if layer_index < 3 {
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
                self.bf16_producer_fusion,
            );
            if !self.bf16_producer_fusion {
                encode_round_bf16(
                    encoder,
                    &self.pipelines.round_bf16,
                    &self.scratch.final_normalized,
                    ASSISTANT_HIDDEN,
                );
            }

            encode_assistant_matrix(
                encoder,
                &self.pipelines,
                &self.weight_file,
                self.full_q4.as_ref(),
                self.full_q4
                    .as_ref()
                    .map(|weights| weights.layout.post_projection),
                &self.scratch.final_normalized,
                &self.scratch.recurrent_hidden,
                self.post_projection,
            );

            if let Some(full_q4) = self.full_q4.as_ref() {
                encode_q4_0_gemv_packed(
                    encoder,
                    &self.pipelines.q4_0_gemv,
                    &full_q4.buffer,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    full_q4.layout.embedding,
                    false,
                );
            } else if let Some(q4_emb) = self.q4_embedding.as_ref() {
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

            let step3_assistant_logits =
                if capture_step3_logits && step == MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX {
                    let mut logits = vec![0.0f32; VOCAB];
                    read_buffer_f32(&self.scratch.logits, &mut logits);
                    Some(logits)
                } else {
                    None
                };

            read_buffer_f32(
                &self.scratch.recurrent_hidden,
                &mut current_recurrent_hidden,
            );
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
                step3_assistant_logits,
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
        self.propose_chain_device_resident_with_step3_logit_capture(
            anchor_token,
            initial_recurrent_hidden,
            target_kv,
            target_embedding,
            draft_limit,
            eot,
            false,
        )
    }

    /// Device-fed chain with an optional one-shot logit snapshot. Boundary
    /// arbitration still owns explicit step-3 capture; the default-off indexed
    /// trace environment can select another draft on ordinary rounds.
    pub(crate) fn propose_chain_device_resident_with_step3_logit_capture(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        target_embedding: Gemma4MtpTargetEmbeddingView<'_>,
        draft_limit: usize,
        eot: &[u32],
        capture_step3_logits: bool,
    ) -> Result<Vec<Gemma4MtpProposal>> {
        self.propose_chain_device_resident_with_step3_logit_capture_inner(
            anchor_token,
            initial_recurrent_hidden,
            target_kv,
            target_embedding,
            draft_limit,
            eot,
            capture_step3_logits,
            MtpDeviceChainInvocation::Production,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_chain_device_resident_with_step3_logit_capture_inner(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_kv: Gemma4MtpTargetKvView<'_>,
        target_embedding: Gemma4MtpTargetEmbeddingView<'_>,
        draft_limit: usize,
        eot: &[u32],
        capture_step3_logits: bool,
        invocation: MtpDeviceChainInvocation,
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
        match invocation {
            MtpDeviceChainInvocation::Production => {
                validate_target_embedding(&target_embedding, device_registry_id)?;
            }
            MtpDeviceChainInvocation::TargetFreeK4Warmup => {
                validate_target_free_device_chain_embedding(&target_embedding, device_registry_id)?;
            }
        }
        validate_target_kv_device(&target_kv, device_registry_id)?;

        let wall_started = Instant::now();
        write_assistant_rope_tables(logical_len, &self.scratch);

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
        let logit_capture_index = mtp_logit_capture_draft_index(capture_step3_logits)
            .filter(|&index| index < draft_limit);
        let logits_snapshot = logit_capture_index
            .map(|_| shared_buffer(&kernel.device, VOCAB * std::mem::size_of::<f32>()));

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
            encode_assistant_matrix(
                encoder,
                &self.pipelines,
                &self.weight_file,
                self.full_q4.as_ref(),
                self.full_q4
                    .as_ref()
                    .map(|weights| weights.layout.pre_projection),
                &self.scratch.pre_input,
                &self.scratch.hidden,
                self.pre_projection,
            );

            for (layer_index, layer) in self.layers.iter().enumerate() {
                let (
                    kv,
                    head_dim,
                    position_count,
                    cos,
                    sin,
                    qnorm_scalar,
                    rope_scalar,
                    attn_scalar,
                ) = if layer_index < 3 {
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
                self.bf16_producer_fusion,
            );
            if !self.bf16_producer_fusion {
                encode_round_bf16(
                    encoder,
                    &self.pipelines.round_bf16,
                    &self.scratch.final_normalized,
                    ASSISTANT_HIDDEN,
                );
            }
            encode_assistant_matrix(
                encoder,
                &self.pipelines,
                &self.weight_file,
                self.full_q4.as_ref(),
                self.full_q4
                    .as_ref()
                    .map(|weights| weights.layout.post_projection),
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

            if let Some(full_q4) = self.full_q4.as_ref() {
                encode_q4_0_gemv_packed(
                    encoder,
                    &self.pipelines.q4_0_gemv,
                    &full_q4.buffer,
                    &self.scratch.final_normalized,
                    &self.scratch.logits,
                    full_q4.layout.embedding,
                    false,
                );
            } else if let Some(q4_emb) = self.q4_embedding.as_ref() {
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
            if logit_capture_index == Some(step) {
                if let Some(snapshot) = logits_snapshot.as_ref() {
                    encode_copy_f32_to_offset(
                        encoder,
                        &self.pipelines.copy_f32,
                        &self.scratch.logits,
                        snapshot,
                        0,
                        VOCAB,
                    );
                }
            }
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
        let mut captured_assistant_logits = logits_snapshot.as_ref().map(|snapshot| {
            let mut logits = vec![0.0f32; VOCAB];
            read_buffer_f32(snapshot, &mut logits);
            logits
        });

        let borrowed_target_kv_capacity_bytes = borrowed_target_kv_capacity_bytes(&target_kv)?;
        let per_step_kv_read_bytes = target_kv_read_bytes(local_count, logical_len)?;
        let draft_count_u64 = draft_limit as u64;
        let readback_bytes = draft_count_u64
            .checked_mul(
                (std::mem::size_of::<u32>() + TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    logits_snapshot
                        .as_ref()
                        .map_or(0, |snapshot| snapshot.length()),
                )
            })
            .ok_or_else(|| invalid("device-chain readback ledger overflow"))?;
        let ledger = Gemma4MtpProposalLedger {
            assistant_matrix_bytes: self
                .assistant_matrix_bytes_per_proposal()
                .checked_mul(draft_count_u64)
                .ok_or_else(|| invalid("device-chain matrix-byte ledger overflow"))?,
            borrowed_target_kv_capacity_bytes,
            target_kv_read_bytes: per_step_kv_read_bytes
                .checked_mul(draft_count_u64)
                .ok_or_else(|| invalid("device-chain target-KV ledger overflow"))?,
            dynamic_attention_scratch_bytes: attention_scores.length(),
            readback_bytes,
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
                step3_assistant_logits: if logit_capture_index == Some(step) {
                    captured_assistant_logits.take()
                } else {
                    None
                },
                #[cfg(test)]
                stage_snapshots: Vec::new(),
            });
            if eot.contains(&token) {
                break;
            }
        }
        if invocation == MtpDeviceChainInvocation::Production {
            let (lattice_rounds_per_draft, lattice_rounds_total) = if self.bf16_lattice_loads {
                let per_draft = mtp_bf16_lattice_rounds_elided_per_draft(logical_len)
                    .ok_or_else(|| invalid("BF16 lattice-load accounting overflow"))?;
                let total = per_draft
                    .checked_mul(draft_limit)
                    .ok_or_else(|| invalid("BF16 lattice-load request accounting overflow"))?;
                (per_draft, total)
            } else {
                (0, 0)
            };
            eprintln!(
                "[gemma4-mtp bf16-lattice-loads] requested_drafts={draft_limit} returned_drafts={} target_logical_k={} enabled={} direct_query_qk={} direct_probability_context={} bf16_round_ops_elided_per_draft={} bf16_round_ops_elided_total={} dispatches_elided={} bytes_elided={} scratch_bytes_added={}",
                proposals.len(),
                logical_len,
                usize::from(self.bf16_lattice_loads),
                usize::from(self.bf16_lattice_loads),
                usize::from(self.bf16_lattice_loads),
                lattice_rounds_per_draft,
                lattice_rounds_total,
                MTP_BF16_LATTICE_DISPATCHES_ELIDED,
                MTP_BF16_LATTICE_BYTES_ELIDED,
                MTP_BF16_LATTICE_SCRATCH_BYTES_ADDED,
            );
            eprintln!(
                "[gemma4-mtp bf16-producer-fusion] requested_drafts={draft_limit} returned_drafts={} enabled={} standalone_round_dispatches_per_draft={} standalone_round_elements_per_draft={} standalone_round_rw_bytes_per_draft={} elided_round_dispatches_per_draft={} elided_round_elements_per_draft={} elided_round_rw_bytes_per_draft={}",
                proposals.len(),
                usize::from(self.bf16_producer_fusion),
                if self.bf16_producer_fusion { 0 } else { MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT },
                if self.bf16_producer_fusion { 0 } else { MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT },
                if self.bf16_producer_fusion { 0 } else { MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT },
                if self.bf16_producer_fusion { MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT } else { 0 },
                if self.bf16_producer_fusion { MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT } else { 0 },
                if self.bf16_producer_fusion { MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT } else { 0 },
            );
            eprintln!(
                "[gemma4-mtp device-chain] requested_drafts={draft_limit} returned_drafts={} command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 linear_format={} matrix_bytes_per_draft={} encode_us={} wait_us={} gpu_us={} kernel_us={} wall_us={}",
                proposals.len(),
                if self.full_q4.is_some() { "q4_0_all" } else if self.q4_embedding.is_some() { "bf16_q4_0_head" } else { "bf16" },
                self.assistant_matrix_bytes_per_proposal(),
                total_timing.encode_us,
                total_timing.wait_us,
                total_timing.gpu_us,
                total_timing.kernel_us,
                total_timing.wall_us,
            );
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
        let q4_layer = self
            .full_q4
            .as_ref()
            .map(|weights| weights.layout.layers[layer_index]);
        debug_assert!(position_count > 0);
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.hidden,
            &layer.input_norm,
            &self.scratch.normed,
            &self.scratch.hidden_rms_scalar,
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
            );
        }
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.input_norm"),
            pending_stage_snapshots,
        );
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            q4_layer.map(|weights| weights.q),
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
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.query_normed,
                q_dim,
            );
        }
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
            self.bf16_lattice_loads,
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
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            q4_layer.map(|weights| weights.o),
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
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.attention_normalized,
                ASSISTANT_HIDDEN,
            );
        }
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
            if self.bf16_producer_fusion {
                &self.pipelines.residual_add_bf16
            } else {
                &kernel.residual_add_pipeline
            },
            &self.scratch.hidden,
            &self.scratch.attention_normalized,
            &self.scratch.attention_residual,
            &self.scratch.hidden_count,
            ASSISTANT_HIDDEN,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.attention_residual,
                ASSISTANT_HIDDEN,
            );
        }
        encode_assistant_rms_norm_f32(
            encoder,
            &self.pipelines,
            &self.scratch.attention_residual,
            &layer.pre_feedforward_norm,
            &self.scratch.normed,
            &self.scratch.hidden_rms_scalar,
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
            );
        }
        #[cfg(test)]
        encode_stage_snapshot(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            format!("layer.{layer_index}.pre_feedforward_norm"),
            pending_stage_snapshots,
        );
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            q4_layer.map(|weights| weights.gate),
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
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            q4_layer.map(|weights| weights.up),
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
        encode_assistant_matrix(
            encoder,
            &self.pipelines,
            &self.weight_file,
            self.full_q4.as_ref(),
            q4_layer.map(|weights| weights.down),
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
            self.bf16_producer_fusion,
        );
        if !self.bf16_producer_fusion {
            encode_round_bf16(
                encoder,
                &self.pipelines.round_bf16,
                &self.scratch.down_normalized,
                ASSISTANT_HIDDEN,
            );
        }
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
            if self.bf16_producer_fusion {
                &self.pipelines.residual_add_bf16
            } else {
                &kernel.residual_add_pipeline
            },
            &self.scratch.attention_residual,
            &self.scratch.down_normalized,
            &self.scratch.next_hidden,
            &self.scratch.hidden_count,
            ASSISTANT_HIDDEN,
        );
        if self.bf16_producer_fusion {
            encode_scale_bf16(
                encoder,
                &self.pipelines.scale_bf16,
                &self.scratch.next_hidden,
                &self.scratch.hidden,
                &layer.scale_scalar,
                ASSISTANT_HIDDEN,
            );
        } else {
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
        }
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

fn encode_scale_bf16(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    count: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(scalar), 0);
    encoder.set_buffer(3, Some(scalar), 4);
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
    bf16_lattice_loads: bool,
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

    encoder
        .set_compute_pipeline_state(pipelines.selected_attention_scores_bf16(bf16_lattice_loads));
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

    encoder
        .set_compute_pipeline_state(pipelines.selected_attention_context_bf16(bf16_lattice_loads));
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
    round_output_bf16: bool,
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
    let round_output_bf16 = u32::from(round_output_bf16);
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &round_output_bf16 as *const u32 as *const c_void,
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
    round_output_bf16: bool,
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
        round_output_bf16,
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
    round_output_bf16: bool,
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
        round_output_bf16,
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
    encode_q4_0_gemv_packed(
        encoder,
        pipeline,
        q4_weights,
        input,
        output,
        Q4TensorRef {
            byte_offset: 0,
            byte_len: q4_weights.length(),
            rows,
            cols,
        },
        false,
    );
}

fn encode_q4_0_gemv_packed(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q4_weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
) {
    debug_assert!(matrix.byte_offset + matrix.byte_len <= q4_weights.length());
    let cols = matrix.cols;
    let rows = matrix.rows;
    let weight_byte_offset = matrix.byte_offset;
    let round_output_bf16 = u32::from(round_output_bf16);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q4_weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &weight_byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round_output_bf16 as *const u32 as *const c_void);
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

#[cfg(test)]
fn encode_q4_0_gemv_packed_rows(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q4_weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
    rows_per_group: u32,
) {
    assert!(matches!(rows_per_group, 2 | 4));
    assert!(matrix.byte_offset + matrix.byte_len <= q4_weights.length());
    let cols = matrix.cols;
    let rows = matrix.rows;
    let weight_byte_offset = matrix.byte_offset;
    let round_output_bf16 = u32::from(round_output_bf16);
    let groups = rows.div_ceil(rows_per_group);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q4_weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &weight_byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round_output_bf16 as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups as u64,
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

fn encode_assistant_matrix(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &MtpPipelines,
    weight_file: &Buffer,
    full_q4: Option<&FullQ4Weights>,
    q4_matrix: Option<Q4TensorRef>,
    input: &Buffer,
    output: &Buffer,
    bf16_matrix: TensorRef,
) {
    if let Some(full_q4) = full_q4 {
        let q4_matrix = q4_matrix.expect("full-Q4 admission populated every matrix");
        debug_assert_eq!(
            (q4_matrix.rows, q4_matrix.cols),
            (bf16_matrix.rows, bf16_matrix.cols)
        );
        encode_q4_0_gemv_packed(
            encoder,
            &pipelines.q4_0_gemv,
            &full_q4.buffer,
            input,
            output,
            q4_matrix,
            true,
        );
    } else {
        encode_bf16_gemv(
            encoder,
            pipelines.selected_bf16_gemv(),
            weight_file,
            input,
            output,
            bf16_matrix,
        );
    }
}

fn q4_0_matrix_bytes(tensor: TensorRef) -> Result<usize> {
    let rows = tensor.rows as usize;
    let cols = tensor.cols as usize;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4_0_BLOCK_VALUES) {
        return Err(invalid(format!(
            "Q4_0 matrix geometry {}x{} is empty or not divisible by {Q4_0_BLOCK_VALUES}",
            tensor.rows, tensor.cols
        )));
    }
    rows.checked_mul(cols / Q4_0_BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(Q4_0_BLOCK_BYTES))
        .ok_or_else(|| invalid("Q4_0 matrix byte size overflow"))
}

fn append_q4_0_layout(tensor: TensorRef, cursor: &mut u64) -> Result<Q4TensorRef> {
    let byte_len = u64::try_from(q4_0_matrix_bytes(tensor)?)
        .map_err(|_| invalid("Q4_0 matrix byte size exceeds u64"))?;
    let result = Q4TensorRef {
        byte_offset: *cursor,
        byte_len,
        rows: tensor.rows,
        cols: tensor.cols,
    };
    *cursor = cursor
        .checked_add(byte_len)
        .ok_or_else(|| invalid("full-Q4 layout byte size overflow"))?;
    Ok(result)
}

fn validate_q4_layout_pairs(pairs: &[(TensorRef, Q4TensorRef)], buffer_len: u64) -> Result<u64> {
    if pairs.is_empty() {
        return Err(invalid("full-Q4 layout contains no matrices"));
    }
    let mut cursor = 0u64;
    for (index, (source, packed)) in pairs.iter().enumerate() {
        let expected_len = u64::try_from(q4_0_matrix_bytes(*source)?)
            .map_err(|_| invalid("Q4_0 matrix byte size exceeds u64"))?;
        if packed.byte_offset != cursor {
            return Err(invalid(format!(
                "full-Q4 matrix {index} begins at {}, expected contiguous offset {cursor}",
                packed.byte_offset
            )));
        }
        if (packed.rows, packed.cols) != (source.rows, source.cols) {
            return Err(invalid(format!(
                "full-Q4 matrix {index} geometry {}x{} does not match source {}x{}",
                packed.rows, packed.cols, source.rows, source.cols
            )));
        }
        if packed.byte_len != expected_len {
            return Err(invalid(format!(
                "full-Q4 matrix {index} has {} bytes, expected {expected_len}",
                packed.byte_len
            )));
        }
        cursor = cursor
            .checked_add(packed.byte_len)
            .ok_or_else(|| invalid("full-Q4 layout byte size overflow"))?;
    }
    if cursor != buffer_len {
        return Err(invalid(format!(
            "full-Q4 layout covers {cursor} bytes, but packed buffer has {buffer_len}"
        )));
    }
    Ok(cursor)
}

impl FullQ4Layout {
    fn build(
        embedding: TensorRef,
        layers: &[LayerWeights],
        pre_projection: TensorRef,
        post_projection: TensorRef,
    ) -> Result<Self> {
        if layers.len() != 4 {
            return Err(invalid(format!(
                "full-Q4 layout has {} layers, expected 4",
                layers.len()
            )));
        }
        let mut cursor = 0u64;
        let embedding = append_q4_0_layout(embedding, &mut cursor)?;
        let mut packed_layers = [FullQ4LayerWeights::default(); 4];
        for (destination, source) in packed_layers.iter_mut().zip(layers) {
            *destination = FullQ4LayerWeights {
                q: append_q4_0_layout(source.q, &mut cursor)?,
                o: append_q4_0_layout(source.o, &mut cursor)?,
                gate: append_q4_0_layout(source.gate, &mut cursor)?,
                up: append_q4_0_layout(source.up, &mut cursor)?,
                down: append_q4_0_layout(source.down, &mut cursor)?,
            };
        }
        let pre_projection = append_q4_0_layout(pre_projection, &mut cursor)?;
        let post_projection = append_q4_0_layout(post_projection, &mut cursor)?;
        Ok(Self {
            embedding,
            layers: packed_layers,
            pre_projection,
            post_projection,
            matrix_bytes: cursor,
        })
    }

    fn validate_complete(
        &self,
        embedding: TensorRef,
        layers: &[LayerWeights],
        pre_projection: TensorRef,
        post_projection: TensorRef,
        buffer_len: u64,
    ) -> Result<()> {
        if layers.len() != self.layers.len() {
            return Err(invalid(format!(
                "full-Q4 validation has {} source layers and {} packed layers",
                layers.len(),
                self.layers.len()
            )));
        }
        let mut pairs = Vec::with_capacity(23);
        pairs.push((embedding, self.embedding));
        for (source, packed) in layers.iter().zip(self.layers.iter()) {
            pairs.extend([
                (source.q, packed.q),
                (source.o, packed.o),
                (source.gate, packed.gate),
                (source.up, packed.up),
                (source.down, packed.down),
            ]);
        }
        pairs.extend([
            (pre_projection, self.pre_projection),
            (post_projection, self.post_projection),
        ]);
        if pairs.len() != 23 {
            return Err(invalid(format!(
                "full-Q4 layout has {} matrices, expected 23",
                pairs.len()
            )));
        }
        let validated_bytes = validate_q4_layout_pairs(&pairs, buffer_len)?;
        if validated_bytes != self.matrix_bytes {
            return Err(invalid(format!(
                "full-Q4 layout records {} bytes after validating {validated_bytes}",
                self.matrix_bytes
            )));
        }
        if self.matrix_bytes != FULL_Q4_MATRIX_BYTES {
            return Err(invalid(format!(
                "full-Q4 official pack has {} bytes, expected {FULL_Q4_MATRIX_BYTES}",
                self.matrix_bytes
            )));
        }
        Ok(())
    }
}

fn quantize_q4_0_row(input: &[u16], output: &mut [u8]) {
    debug_assert_eq!(input.len() % Q4_0_BLOCK_VALUES, 0);
    debug_assert_eq!(
        output.len(),
        input.len() / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES
    );
    for (block_in, block_out) in input
        .chunks_exact(Q4_0_BLOCK_VALUES)
        .zip(output.chunks_exact_mut(Q4_0_BLOCK_BYTES))
    {
        let mut f32_vals = [0.0f32; Q4_0_BLOCK_VALUES];
        let mut max_abs = 0.0f32;
        let mut signed_max = 0.0f32;
        for (destination, bits) in f32_vals.iter_mut().zip(block_in) {
            let value = bf16_bits_to_f32(*bits);
            *destination = value;
            let absolute = value.abs();
            if absolute > max_abs {
                max_abs = absolute;
                signed_max = value;
            }
        }
        // Match ggml Q4_0 exactly: the sign of the first max-magnitude
        // element selects which side of the block receives the -8 code.
        let scale = signed_max / -8.0;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        block_out[..2].copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
        for index in 0..16 {
            let low = (f32_vals[index] * inv_scale + 8.5).floor().clamp(0.0, 15.0) as u8;
            let high = (f32_vals[index + 16] * inv_scale + 8.5)
                .floor()
                .clamp(0.0, 15.0) as u8;
            block_out[2 + index] = (low & 0x0f) | ((high & 0x0f) << 4);
        }
    }
}

fn quantize_matrix_into_q4_0(
    mapping: &GgufWireMmap,
    tensor: TensorRef,
    destination: &Buffer,
    packed: Q4TensorRef,
) -> Result<()> {
    let rows = tensor.rows as usize;
    let cols = tensor.cols as usize;
    let row_bytes = cols / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES;
    let expected_bytes = q4_0_matrix_bytes(tensor)?;
    if packed.rows != tensor.rows
        || packed.cols != tensor.cols
        || packed.byte_len != expected_bytes as u64
        || packed
            .byte_offset
            .checked_add(packed.byte_len)
            .is_none_or(|end| end > destination.length())
    {
        return Err(invalid(
            "full-Q4 matrix layout does not cover its destination",
        ));
    }
    let input_bytes = mapping.bytes(tensor.absolute_offset as u64, rows * cols * 2)?;
    let input =
        unsafe { std::slice::from_raw_parts(input_bytes.as_ptr().cast::<u16>(), rows * cols) };
    let output_address = destination.contents() as usize + packed.byte_offset as usize;

    use rayon::prelude::*;
    (0..rows).into_par_iter().for_each(|row| {
        let row_input = &input[row * cols..(row + 1) * cols];
        let row_output = unsafe {
            std::slice::from_raw_parts_mut((output_address + row * row_bytes) as *mut u8, row_bytes)
        };
        quantize_q4_0_row(row_input, row_output);
    });
    Ok(())
}

fn quantize_full_assistant_to_q4_0(
    device: &Device,
    mapping: &GgufWireMmap,
    embedding: TensorRef,
    layers: &[LayerWeights],
    pre_projection: TensorRef,
    post_projection: TensorRef,
) -> Result<FullQ4Weights> {
    let layout = FullQ4Layout::build(embedding, layers, pre_projection, post_projection)?;
    layout.validate_complete(
        embedding,
        layers,
        pre_projection,
        post_projection,
        layout.matrix_bytes,
    )?;
    let buffer = shared_buffer(
        device,
        usize::try_from(layout.matrix_bytes)
            .map_err(|_| invalid("full-Q4 matrix pack exceeds usize"))?,
    );
    let started = Instant::now();
    quantize_matrix_into_q4_0(mapping, embedding, &buffer, layout.embedding)?;
    for ((source, destination), layer_index) in
        layers.iter().zip(layout.layers.iter()).zip(0usize..)
    {
        for (name, tensor, packed) in [
            ("q", source.q, destination.q),
            ("o", source.o, destination.o),
            ("gate", source.gate, destination.gate),
            ("up", source.up, destination.up),
            ("down", source.down, destination.down),
        ] {
            quantize_matrix_into_q4_0(mapping, tensor, &buffer, packed).map_err(|error| {
                invalid(format!(
                    "full-Q4 layer {layer_index} {name} packing failed: {error}"
                ))
            })?;
        }
    }
    quantize_matrix_into_q4_0(mapping, pre_projection, &buffer, layout.pre_projection)?;
    quantize_matrix_into_q4_0(mapping, post_projection, &buffer, layout.post_projection)?;
    layout.validate_complete(
        embedding,
        layers,
        pre_projection,
        post_projection,
        buffer.length(),
    )?;
    Ok(FullQ4Weights {
        buffer,
        layout,
        quantize_us: started.elapsed().as_micros(),
    })
}

fn quantize_embedding_to_q4_0(
    device: &Device,
    mapping: &GgufWireMmap,
    tensor: TensorRef,
) -> Result<Buffer> {
    let byte_len = q4_0_matrix_bytes(tensor)?;
    let buffer = shared_buffer(device, byte_len);
    quantize_matrix_into_q4_0(
        mapping,
        tensor,
        &buffer,
        Q4TensorRef {
            byte_offset: 0,
            byte_len: byte_len as u64,
            rows: tensor.rows,
            cols: tensor.cols,
        },
    )?;
    Ok(buffer)
}

fn validate_target_embedding_geometry(
    format: Gemma4MtpTargetEmbeddingFormat,
    hidden: usize,
    vocab: usize,
    byte_offset: usize,
    byte_len: usize,
    buffer_len: usize,
) -> Result<()> {
    if format != Gemma4MtpTargetEmbeddingFormat::Q6K
        || hidden != TARGET_HIDDEN
        || vocab != VOCAB
        || !hidden.is_multiple_of(TARGET_Q6K_VALUES_PER_BLOCK)
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 MTP device-chain target embedding mismatch: format={format:?} hidden={hidden} vocab={vocab}; expected Q6_K {VOCAB}x{TARGET_HIDDEN}"
        )));
    }
    let expected_len = vocab
        .checked_mul(hidden / TARGET_Q6K_VALUES_PER_BLOCK)
        .and_then(|value| value.checked_mul(TARGET_Q6K_WIRE_BYTES_PER_BLOCK))
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

fn validate_target_free_device_chain_embedding_geometry(
    format: Gemma4MtpTargetEmbeddingFormat,
    hidden: usize,
    vocab: usize,
    byte_offset: usize,
    byte_len: usize,
    buffer_len: usize,
) -> Result<()> {
    if format != Gemma4MtpTargetEmbeddingFormat::Q6K
        || hidden != TARGET_HIDDEN
        || !hidden.is_multiple_of(TARGET_Q6K_VALUES_PER_BLOCK)
        || vocab != 1
        || byte_offset != 0
        || byte_len != TARGET_Q6K_ROW_BYTES
        || buffer_len != TARGET_Q6K_ROW_BYTES
    {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 MTP target-free device-chain embedding mismatch: format={format:?} hidden={hidden} vocab={vocab} offset={byte_offset} bytes={byte_len} buffer={buffer_len}; expected one exact Q6_K {TARGET_HIDDEN}-wide row ({TARGET_Q6K_ROW_BYTES} bytes) at offset zero"
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

fn validate_target_free_device_chain_embedding(
    view: &Gemma4MtpTargetEmbeddingView<'_>,
    required_device_registry_id: u64,
) -> Result<()> {
    validate_target_free_device_chain_embedding_geometry(
        view.format(),
        view.hidden(),
        view.vocab(),
        view.byte_offset(),
        view.byte_len(),
        usize::try_from(view.buffer().length())
            .map_err(|_| invalid("target-free embedding buffer length exceeds usize"))?,
    )?;
    if view.buffer().storage_mode() != MTLStorageMode::Shared {
        return Err(invalid(format!(
            "target-free embedding storage mode is {:?}, expected Shared private warmup storage",
            view.buffer().storage_mode()
        )));
    }
    let actual_device_registry_id = view.buffer().device().registry_id();
    if actual_device_registry_id != required_device_registry_id {
        return Err(invalid(format!(
            "target-free embedding Metal device {actual_device_registry_id} differs from assistant device {required_device_registry_id}"
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

/// Populate the one-position RoPE tables shared by the reference, CPU-chain,
/// and device-chain proposal paths. Keeping the proportional full-attention
/// geometry behind this parameter-free seam prevents a chain from silently
/// rotating the 192 inactive split-half pairs.
fn write_assistant_rope_tables(position: usize, scratch: &MtpScratch) {
    write_rope_tables(
        position,
        10_000.0,
        LOCAL_HEAD_DIM,
        LOCAL_HEAD_DIM / 2,
        &scratch.local_cos,
        &scratch.local_sin,
    );
    let (cos, sin) = full_rope_table_values(position);
    debug_assert_eq!(
        scratch.full_cos.length() as usize,
        cos.len() * std::mem::size_of::<f32>()
    );
    debug_assert_eq!(
        scratch.full_sin.length() as usize,
        sin.len() * std::mem::size_of::<f32>()
    );
    write_buffer_f32(&scratch.full_cos, &cos);
    write_buffer_f32(&scratch.full_sin, &sin);
}

fn full_rope_table_values(position: usize) -> (Vec<f32>, Vec<f32>) {
    rope_table_values(position, 1_000_000.0, FULL_HEAD_DIM, FULL_ROPE_ACTIVE_PAIRS)
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
    fn bf16_producer_fusion_is_explicit_default_off_and_fail_closed() {
        assert_eq!(parse_bf16_producer_fusion_opt_in(None), Ok(false));
        assert_eq!(parse_bf16_producer_fusion_opt_in(Some("0")), Ok(false));
        assert_eq!(parse_bf16_producer_fusion_opt_in(Some("FALSE")), Ok(false));
        assert_eq!(parse_bf16_producer_fusion_opt_in(Some("1")), Ok(true));
        assert_eq!(parse_bf16_producer_fusion_opt_in(Some("TrUe")), Ok(true));
        assert!(parse_bf16_producer_fusion_opt_in(Some("yes")).is_err());
        assert!(parse_bf16_producer_fusion_opt_in(Some("")).is_err());
    }

    #[test]
    fn bf16_lattice_loads_are_explicit_default_off_and_fail_closed() {
        assert_eq!(parse_bf16_lattice_loads_opt_in(None), Ok(false));
        assert_eq!(parse_bf16_lattice_loads_opt_in(Some("0")), Ok(false));
        assert_eq!(parse_bf16_lattice_loads_opt_in(Some("FALSE")), Ok(false));
        assert_eq!(parse_bf16_lattice_loads_opt_in(Some("1")), Ok(true));
        assert_eq!(parse_bf16_lattice_loads_opt_in(Some("TrUe")), Ok(true));
        assert!(parse_bf16_lattice_loads_opt_in(Some("yes")).is_err());
        assert!(parse_bf16_lattice_loads_opt_in(Some("")).is_err());
    }

    fn mtp_shader_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start = source
            .find(start)
            .unwrap_or_else(|| panic!("missing shader section start {start:?}"));
        let source = &source[start..];
        let end = source
            .find(end)
            .unwrap_or_else(|| panic!("missing shader section end {end:?}"));
        &source[..end]
    }

    #[test]
    fn bf16_lattice_loads_source_scope_pins_bf16_producers_and_f32_kv_consumers() {
        let control_qk = mtp_shader_section(
            MTP_SHADER,
            "kernel void mtp_attention_scores_bf16_f32(",
            "kernel void mtp_attention_scores_bf16_lattice_query_f32(",
        );
        assert!(control_qk.contains("mtp_load_rounded_bf16x4(query"));
        assert!(control_qk.contains("mtp_load_rounded_bf16x4(keys"));

        let lattice_qk = mtp_shader_section(
            MTP_SHADER,
            "kernel void mtp_attention_scores_bf16_lattice_query_f32(",
            "kernel void mtp_attention_softmax_bf16_f32(",
        );
        assert!(lattice_qk.contains("mtp_load_bf16_latticex4(query"));
        assert!(!lattice_qk.contains("mtp_load_rounded_bf16x4(query"));
        assert!(lattice_qk.contains("mtp_load_rounded_bf16x4(keys"));
        assert!(!lattice_qk.contains("mtp_load_bf16_latticex4(keys"));

        let control_context = mtp_shader_section(
            MTP_SHADER,
            "kernel void mtp_attention_context_bf16_f32(",
            "kernel void mtp_attention_context_bf16_lattice_probabilities_f32(",
        );
        assert!(control_context.contains("mtp_round_bf16(probabilities"));
        assert!(control_context.contains("mtp_round_bf16(values"));

        let lattice_context = mtp_shader_section(
            MTP_SHADER,
            "kernel void mtp_attention_context_bf16_lattice_probabilities_f32(",
            "// Pinned ATen contiguous-f32 RMS",
        );
        assert!(lattice_context.contains("probabilities[score_base + p] *"));
        assert!(!lattice_context.contains("mtp_round_bf16(probabilities"));
        assert!(lattice_context.contains("mtp_round_bf16(values"));

        // The candidate is sound only because both query and probability
        // producers store widened BF16 before these consumers execute.
        assert!(MTP_SHADER.contains("data[dim0] = mtp_round_bf16(first + first_cross)"));
        assert!(MTP_SHADER.contains("mtp_round_bf16(scores[score_base + p] / denominator)"));
        assert!(MTP_TEST_DIAGNOSTIC_SHADER
            .contains("mtp_test_round_bf16(scores[score_base + p] * reciprocal)"));
    }

    #[test]
    fn step3_logit_trace_is_default_off_and_accepts_only_explicit_truthy_values() {
        assert!(!mtp_step3_logit_trace_enabled_value(None));
        assert!(!mtp_step3_logit_trace_enabled_value(Some("0")));
        assert!(!mtp_step3_logit_trace_enabled_value(Some("false")));
        assert!(!mtp_step3_logit_trace_enabled_value(Some("unexpected")));
        assert!(mtp_step3_logit_trace_enabled_value(Some("1")));
        assert!(mtp_step3_logit_trace_enabled_value(Some(" TrUe ")));
        assert!(mtp_step3_logit_trace_enabled_value(Some("ON")));
        assert!(mtp_step3_logit_trace_enabled_value(Some("yes")));
    }

    #[test]
    fn step3_logit_capture_is_enabled_by_trace_or_explicit_per_call_request() {
        assert!(!mtp_step3_logit_capture_enabled_values(None, false));
        assert!(!mtp_step3_logit_capture_enabled_values(
            Some("false"),
            false
        ));
        assert!(mtp_step3_logit_capture_enabled_values(Some("1"), false));
        assert!(mtp_step3_logit_capture_enabled_values(None, true));
        assert!(mtp_step3_logit_capture_enabled_values(Some("false"), true));
        assert!(mtp_step3_logit_capture_enabled_values(Some("true"), true));
    }

    #[test]
    fn target_free_device_chain_warmup_pins_k4_step3_graph_and_restore_geometry() {
        let recurrent_bytes = TARGET_HIDDEN * std::mem::size_of::<f32>();
        let chain_recurrent_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * recurrent_bytes;
        let token_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * std::mem::size_of::<u32>();

        assert_eq!(MTP_DEVICE_CHAIN_K4_WARM_DRAFTS, 4);
        assert_eq!(MTP_STEP3_LOGIT_TRACE_DRAFT_INDEX, 3);
        assert!(mtp_step3_logit_capture_enabled_values(None, true));
        assert_eq!(MTP_DEVICE_CHAIN_DISPATCHES_PER_DRAFT, 112);
        assert_eq!(MTP_DEVICE_CHAIN_K4_WARM_DISPATCHES, 4 * 112 + 1);
        assert_eq!(mtp_device_chain_k4_warm_dispatches(false), 449);
        assert_eq!(mtp_device_chain_k4_warm_dispatches(true), 317);
        assert_eq!(recurrent_bytes, 11_264);
        assert_eq!(chain_recurrent_bytes, 45_056);
        assert_eq!(token_bytes, 16);
        assert_eq!(
            recurrent_bytes + chain_recurrent_bytes + token_bytes,
            56_336
        );
        assert_eq!(MTP_DEVICE_CHAIN_K4_WARM_RESTORE_BYTES, 56_336);
        assert_eq!(TARGET_Q6K_ROW_BYTES, 2_310);
    }

    #[test]
    fn bf16_producer_fusion_accounting_pins_the_exact_request_graph() {
        assert_eq!(MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT, 33);
        assert_eq!(MTP_STANDALONE_BF16_ROUND_ELEMENTS_PER_DRAFT, 50_176);
        assert_eq!(MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT, 401_408);
        assert_eq!(mtp_device_chain_k4_warm_dispatches(false), 449);
        assert_eq!(mtp_device_chain_k4_warm_dispatches(true), 317);
        assert_eq!(44 * MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT, 1_452);
        assert_eq!(
            44 * MTP_STANDALONE_BF16_ROUND_RW_BYTES_PER_DRAFT,
            17_661_952
        );
    }

    #[test]
    fn bf16_lattice_loads_accounting_pins_the_exact_44_proposal_request() {
        let request_rounds = [(13usize, 104usize), (12, 118), (13, 131), (6, 145)];
        let expected_per_draft = [4_259_840usize, 4_833_280, 5_365_760, 5_939_200];
        assert_eq!(
            request_rounds
                .iter()
                .map(|(drafts, _)| drafts)
                .sum::<usize>(),
            44
        );
        for ((_, logical_k), expected) in request_rounds.iter().zip(expected_per_draft) {
            assert_eq!(
                mtp_bf16_lattice_rounds_elided_per_draft(*logical_k),
                Some(expected)
            );
        }
        let request_total = request_rounds
            .iter()
            .map(|(drafts, logical_k)| {
                drafts * mtp_bf16_lattice_rounds_elided_per_draft(*logical_k).unwrap()
            })
            .sum::<usize>();
        assert_eq!(request_total, 218_767_360);
        assert_eq!(44 * 4 * 3, 528);
        assert_eq!(MTP_BF16_LATTICE_DISPATCHES_ELIDED, 0);
        assert_eq!(MTP_BF16_LATTICE_BYTES_ELIDED, 0);
        assert_eq!(MTP_BF16_LATTICE_SCRATCH_BYTES_ADDED, 0);
        assert_eq!(MTP_BF16_LATTICE_CANDIDATE_PSOS_COMPILED, 2);
    }

    #[test]
    fn indexed_logit_trace_accepts_only_a_bounded_zero_based_draft_index() {
        assert_eq!(mtp_logit_trace_draft_index_value(None), None);
        assert_eq!(mtp_logit_trace_draft_index_value(Some("")), None);
        assert_eq!(mtp_logit_trace_draft_index_value(Some(" 10 ")), Some(10));
        assert_eq!(mtp_logit_trace_draft_index_value(Some("15")), Some(15));
        assert_eq!(mtp_logit_trace_draft_index_value(Some("16")), None);
        assert_eq!(mtp_logit_trace_draft_index_value(Some("-1")), None);
        assert_eq!(mtp_logit_trace_draft_index_value(Some("nope")), None);
    }

    #[test]
    fn full_q4_opt_in_is_explicit_and_malformed_values_fail_closed() {
        assert_eq!(parse_full_q4_opt_in(None), Ok(false));
        assert_eq!(parse_full_q4_opt_in(Some("0")), Ok(false));
        assert_eq!(parse_full_q4_opt_in(Some("FALSE")), Ok(false));
        assert_eq!(parse_full_q4_opt_in(Some("1")), Ok(true));
        assert_eq!(parse_full_q4_opt_in(Some("TrUe")), Ok(true));
        assert!(parse_full_q4_opt_in(Some("yes")).is_err());
        assert!(parse_full_q4_opt_in(Some("")).is_err());
    }

    #[test]
    fn full_q4_layout_covers_every_matrix_without_padding_or_overlap() {
        let mut cursor = 0u64;
        let mut matrix_count = 0usize;
        let mut pairs = Vec::with_capacity(23);
        for expected in EXPECTED_TENSORS
            .iter()
            .filter(|tensor| tensor.shape.len() == 2)
        {
            let tensor = TensorRef {
                absolute_offset: (PAYLOAD_FILE_OFFSET as u64 + expected.start) as u32,
                rows: expected.shape[0] as u32,
                cols: expected.shape[1] as u32,
            };
            let packed = append_q4_0_layout(tensor, &mut cursor).unwrap();
            assert_eq!(packed.byte_offset + packed.byte_len, cursor);
            assert_eq!((packed.rows, packed.cols), (tensor.rows, tensor.cols));
            pairs.push((tensor, packed));
            matrix_count += 1;
        }
        assert_eq!(matrix_count, 23);
        assert_eq!(cursor, FULL_Q4_MATRIX_BYTES);
        assert_eq!(cursor, 225 * 1_048_576 + 147_456);
        assert_eq!(validate_q4_layout_pairs(&pairs, cursor).unwrap(), cursor);

        let mut noncontiguous = pairs.clone();
        noncontiguous[1].1.byte_offset += 1;
        assert!(validate_q4_layout_pairs(&noncontiguous, cursor).is_err());
        assert!(validate_q4_layout_pairs(&pairs, cursor + 1).is_err());
    }

    #[test]
    fn q4_row_encoder_has_pinned_zero_and_finite_block_contract() {
        let zeros = [0u16; Q4_0_BLOCK_VALUES];
        let mut zero_block = [0u8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_row(&zeros, &mut zero_block);
        assert_eq!(&zero_block[..2], &[0, 0x80]);
        assert!(zero_block[2..].iter().all(|byte| *byte == 0x88));

        let input: Vec<u16> = (0..Q4_0_BLOCK_VALUES)
            .map(|index| {
                let value = (index as f32 - 15.5) / 7.0;
                f32_to_bf16_rne_bits(value)
            })
            .collect();
        let mut first = [0u8; Q4_0_BLOCK_BYTES];
        let mut second = [0u8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_row(&input, &mut first);
        quantize_q4_0_row(&input, &mut second);
        assert_eq!(first, second);
        assert_ne!(&first[..2], &[0, 0]);
        assert!(first[2..].iter().any(|byte| *byte != 0x88));
    }

    #[test]
    fn q4_row_encoder_matches_canonical_signed_max_for_both_dominant_signs() {
        fn assert_matches_reference(values: [f32; Q4_0_BLOCK_VALUES]) {
            let bf16: [u16; Q4_0_BLOCK_VALUES] = values.map(f32_to_bf16_rne_bits);
            let reference_input: [f32; Q4_0_BLOCK_VALUES] = bf16.map(bf16_bits_to_f32);
            let reference = crate::tensor::kv_quant::quantize_block_q4_0(&reference_input);
            let mut actual = [0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_row(&bf16, &mut actual);
            assert_eq!(&actual[..2], &reference.scale.to_le_bytes());
            assert_eq!(&actual[2..], &reference.qs);
        }

        let mut negative_dominant = [0.0f32; Q4_0_BLOCK_VALUES];
        negative_dominant[0] = -8.0;
        negative_dominant[Q4_0_BLOCK_VALUES / 2] = 7.0;
        assert_matches_reference(negative_dominant);

        let mut positive_dominant = [0.0f32; Q4_0_BLOCK_VALUES];
        positive_dominant[0] = 8.0;
        positive_dominant[Q4_0_BLOCK_VALUES / 2] = -7.0;
        assert_matches_reference(positive_dominant);

        // Exercise an f16 subnormal scale just above the halfway-to-zero tie.
        // This catches local conversion shortcuts that incorrectly flush the
        // entire unbiased-exponent -25 range instead of rounding to 0x0001.
        let tiny = f32::from_bits(0x3481_0000);
        let mut tiny_positive_dominant = [0.0f32; Q4_0_BLOCK_VALUES];
        tiny_positive_dominant[0] = tiny;
        tiny_positive_dominant[Q4_0_BLOCK_VALUES / 2] = -tiny * 0.5;
        assert_matches_reference(tiny_positive_dominant);

        let mut tiny_negative_dominant = [0.0f32; Q4_0_BLOCK_VALUES];
        tiny_negative_dominant[0] = -tiny;
        tiny_negative_dominant[Q4_0_BLOCK_VALUES / 2] = tiny * 0.5;
        assert_matches_reference(tiny_negative_dominant);
    }

    #[test]
    fn q4_packed_nonzero_offset_is_raw_bit_identical() {
        const ROWS: usize = 64;
        const COLS: usize = 1_024;
        const OFFSET: usize = 256;
        let kernel = metal_linear_kernel().expect("Metal kernel");
        let pipelines = MtpPipelines::new(&kernel.device).expect("MTP pipelines");
        let input: Vec<f32> = (0..COLS)
            .map(|index| ((index as f32 * 0.03125).sin() * 0.75) + 0.125)
            .collect();
        let weights: Vec<u16> = (0..ROWS * COLS)
            .map(|index| {
                f32_to_bf16_rne_bits(((index as f32 * 0.000_976_562_5).cos() * 0.5) - 0.0625)
            })
            .collect();
        let row_bytes = COLS / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES;
        let mut packed = vec![0u8; ROWS * row_bytes];
        for row in 0..ROWS {
            quantize_q4_0_row(
                &weights[row * COLS..(row + 1) * COLS],
                &mut packed[row * row_bytes..(row + 1) * row_bytes],
            );
        }
        let standalone = shared_buffer(&kernel.device, packed.len());
        let offset_buffer = shared_buffer(&kernel.device, OFFSET + packed.len() + 256);
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                standalone.contents().cast::<u8>(),
                packed.len(),
            );
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                offset_buffer.contents().cast::<u8>().add(OFFSET),
                packed.len(),
            );
        }
        let input = f32_buffer(&kernel.device, &input);
        let run = |weights: &Buffer, offset: u64| {
            let output = shared_buffer(&kernel.device, ROWS * std::mem::size_of::<f32>());
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode_q4_0_gemv_packed(
                encoder,
                &pipelines.q4_0_gemv,
                weights,
                &input,
                &output,
                Q4TensorRef {
                    byte_offset: offset,
                    byte_len: packed.len() as u64,
                    rows: ROWS as u32,
                    cols: COLS as u32,
                },
                true,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
            let mut values = vec![0.0f32; ROWS];
            read_buffer_f32(&output, &mut values);
            values
        };
        let zero_offset = run(&standalone, 0);
        let nonzero_offset = run(&offset_buffer, OFFSET as u64);
        assert_eq!(
            zero_offset
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            nonzero_offset
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum H66HeadPath {
        Established,
        Rows2,
        Rows4,
    }

    impl H66HeadPath {
        const fn label(self) -> &'static str {
            match self {
                Self::Established => "established",
                Self::Rows2 => "rows2",
                Self::Rows4 => "rows4",
            }
        }
    }

    fn encode_h66_head_path(
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MtpPipelines,
        path: H66HeadPath,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        matrix: Q4TensorRef,
        round_output_bf16: bool,
    ) {
        match path {
            H66HeadPath::Established => encode_q4_0_gemv_packed(
                encoder,
                &pipelines.q4_0_gemv,
                weights,
                input,
                output,
                matrix,
                round_output_bf16,
            ),
            H66HeadPath::Rows2 => encode_q4_0_gemv_packed_rows(
                encoder,
                &pipelines.q4_0_gemv_rows2,
                weights,
                input,
                output,
                matrix,
                round_output_bf16,
                2,
            ),
            H66HeadPath::Rows4 => encode_q4_0_gemv_packed_rows(
                encoder,
                &pipelines.q4_0_gemv_rows4,
                weights,
                input,
                output,
                matrix,
                round_output_bf16,
                4,
            ),
        }
    }

    fn run_h66_head_once_bits(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        path: H66HeadPath,
        weights: &Buffer,
        input: &Buffer,
        matrix: Q4TensorRef,
        round_output_bf16: bool,
        guard_rows: usize,
    ) -> Vec<u32> {
        const GUARD_BITS: u32 = 0x4e91_2345;
        let output_rows = matrix.rows as usize + guard_rows;
        let initial = vec![f32::from_bits(GUARD_BITS); output_rows];
        let output = f32_buffer(&kernel.device, &initial);
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_h66_head_path(
            encoder,
            pipelines,
            path,
            weights,
            input,
            &output,
            matrix,
            round_output_bf16,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(
            command_buffer.status(),
            MTLCommandBufferStatus::Completed,
            "H66 {:?} command failed",
            path
        );
        let mut values = vec![0.0f32; output_rows];
        read_buffer_f32(&output, &mut values);
        let bits: Vec<u32> = values.into_iter().map(f32::to_bits).collect();
        assert!(
            bits[matrix.rows as usize..]
                .iter()
                .all(|bits| *bits == GUARD_BITS),
            "H66 {:?} overwrote the output guard",
            path
        );
        bits
    }

    #[test]
    fn q4_head_rows2_rows4_are_raw_bit_exact_with_ragged_rows_and_guards() {
        const ROWS: usize = 7;
        const COLS: usize = 8_192;
        const PREFIX: usize = 256;
        const SUFFIX: usize = 256;
        const GUARD_ROWS: usize = 5;
        const WEIGHT_CANARY: u8 = 0xa5;
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let row_bytes = COLS / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES;
        let packed_bytes = ROWS * row_bytes;
        let mut packed = vec![0u8; packed_bytes];
        for row in 0..ROWS {
            let source: Vec<u16> = (0..COLS)
                .map(|col| {
                    let phase = ((row * COLS + col) as f32 * 0.000_976_562_5).sin();
                    f32_to_bf16_rne_bits(phase * (0.25 + row as f32 * 0.03125))
                })
                .collect();
            quantize_q4_0_row(&source, &mut packed[row * row_bytes..(row + 1) * row_bytes]);
        }
        let mut storage = vec![WEIGHT_CANARY; PREFIX + packed_bytes + SUFFIX];
        storage[PREFIX..PREFIX + packed_bytes].copy_from_slice(&packed);
        let weights = shared_buffer(&kernel.device, storage.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                storage.as_ptr(),
                weights.contents().cast::<u8>(),
                storage.len(),
            );
        }
        let matrix = Q4TensorRef {
            byte_offset: PREFIX as u64,
            byte_len: packed_bytes as u64,
            rows: ROWS as u32,
            cols: COLS as u32,
        };
        let mut production_input: Vec<f32> = (0..COLS)
            .map(|index| round_to_bf16_f32(((index as f32 * 0.003_906_25).cos() * 0.75) - 0.125))
            .collect();
        production_input[..6].copy_from_slice(&[
            0.0,
            -0.0,
            f32::from_bits(0x3f80_0000),
            f32::from_bits(0xbf80_0000),
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
        ]);
        let mut adversarial_input = bf16_fusion_adversarial_finite_values(COLS, 0x66);
        adversarial_input[..4].copy_from_slice(&[
            0.0,
            -0.0,
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0xbf80_8000),
        ]);
        assert!(production_input.iter().all(|value| value.is_finite()));
        assert!(adversarial_input.iter().all(|value| value.is_finite()));

        for (input_label, input_values) in [
            ("production-bf16", production_input),
            ("adversarial-finite", adversarial_input),
        ] {
            let input = f32_buffer(&kernel.device, &input_values);
            for round_output_bf16 in [false, true] {
                let established = run_h66_head_once_bits(
                    kernel,
                    &pipelines,
                    H66HeadPath::Established,
                    &weights,
                    &input,
                    matrix,
                    round_output_bf16,
                    GUARD_ROWS,
                );
                for path in [H66HeadPath::Rows2, H66HeadPath::Rows4] {
                    let candidate = run_h66_head_once_bits(
                        kernel,
                        &pipelines,
                        path,
                        &weights,
                        &input,
                        matrix,
                        round_output_bf16,
                        GUARD_ROWS,
                    );
                    assert_eq!(
                        candidate,
                        established,
                        "H66 {} drifted for {input_label}, round_output_bf16={round_output_bf16}",
                        path.label()
                    );
                }
            }
        }

        let final_storage =
            unsafe { std::slice::from_raw_parts(weights.contents().cast::<u8>(), storage.len()) };
        assert!(final_storage[..PREFIX]
            .iter()
            .all(|byte| *byte == WEIGHT_CANARY));
        assert!(final_storage[PREFIX + packed_bytes..]
            .iter()
            .all(|byte| *byte == WEIGHT_CANARY));
    }

    #[derive(Clone, Copy, Debug)]
    struct H66HeadRequestTiming {
        segments_us: [u128; 5],
        total_us: u128,
    }

    fn time_h66_head_request(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        path: H66HeadPath,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        matrix: Q4TensorRef,
    ) -> H66HeadRequestTiming {
        const SEGMENTS: [usize; 5] = [4, 9, 12, 13, 6];
        let mut segments_us = [0u128; SEGMENTS.len()];
        for (segment_index, drafts) in SEGMENTS.into_iter().enumerate() {
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            for _ in 0..drafts {
                encode_h66_head_path(
                    encoder, pipelines, path, weights, input, output, matrix, false,
                );
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(
                command_buffer.status(),
                MTLCommandBufferStatus::Completed,
                "H66 timing {:?} segment {segment_index} failed",
                path
            );
            segments_us[segment_index] = command_buffer_gpu_times_us(command_buffer).0;
        }
        H66HeadRequestTiming {
            total_us: segments_us.iter().sum(),
            segments_us,
        }
    }

    #[derive(Clone, Copy)]
    struct H66VmCounters {
        swapins: u64,
        swapouts: u64,
    }

    #[allow(deprecated)]
    fn h66_vm_counters() -> Option<H66VmCounters> {
        let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let result = unsafe {
            libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast::<libc::integer_t>(),
                &mut count,
            )
        };
        if result != libc::KERN_SUCCESS {
            return None;
        }
        let stats = unsafe { stats.assume_init() };
        Some(H66VmCounters {
            swapins: stats.swapins,
            swapouts: stats.swapouts,
        })
    }

    fn h66_median_i128(samples: &[i128]) -> i128 {
        let mut ordered = samples.to_vec();
        ordered.sort_unstable();
        ordered[ordered.len() / 2]
    }

    fn h66_current_swap_is_zero() -> bool {
        std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .is_some_and(|text| text.contains("used = 0.00M"))
    }

    fn h66_synthetic_official_geometry_head(device: &Device) -> (Buffer, Q4TensorRef) {
        const PREFIX: usize = 256;
        const SUFFIX: usize = 256;
        const HEAD_BYTES: usize = 150_994_944;
        const CANARY: u8 = 0x5a;
        const SCALES: [u16; 8] = [
            0x3000, 0x3400, 0x3800, 0x3c00, 0xb000, 0xb400, 0xb800, 0xbc00,
        ];
        let buffer = shared_buffer(device, PREFIX + HEAD_BYTES + SUFFIX);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().cast::<u8>(),
                PREFIX + HEAD_BYTES + SUFFIX,
            )
        };
        bytes.fill(CANARY);
        let blocks_per_row = ASSISTANT_HIDDEN / Q4_0_BLOCK_VALUES;
        for row in 0..VOCAB {
            for block in 0..blocks_per_row {
                let block_index = row * blocks_per_row + block;
                let base = PREFIX + block_index * Q4_0_BLOCK_BYTES;
                let scale = SCALES[(row.wrapping_mul(17) + block.wrapping_mul(5)) % SCALES.len()];
                bytes[base..base + 2].copy_from_slice(&scale.to_le_bytes());
                for q in 0..16 {
                    bytes[base + 2 + q] = (row as u8)
                        .wrapping_mul(29)
                        .wrapping_add((block as u8).wrapping_mul(11))
                        .wrapping_add((q as u8).wrapping_mul(7));
                }
            }
        }
        assert!(bytes[..PREFIX].iter().all(|byte| *byte == CANARY));
        assert!(bytes[PREFIX + HEAD_BYTES..]
            .iter()
            .all(|byte| *byte == CANARY));
        (
            buffer,
            Q4TensorRef {
                byte_offset: PREFIX as u64,
                byte_len: HEAD_BYTES as u64,
                rows: VOCAB as u32,
                cols: ASSISTANT_HIDDEN as u32,
            },
        )
    }

    #[test]
    #[ignore = "real-Metal H66 exact synthetic-official-geometry 44-head timing gate"]
    fn q4_head_rows2_rows4_synthetic_official_geometry_exact_44_head_timing() {
        const SEGMENTS: [usize; 5] = [4, 9, 12, 13, 6];
        const OFFICIAL_HEAD_BYTES: u64 = 150_994_944;
        const OUTPUT_GUARD_ROWS: usize = 16;
        const OUTPUT_GUARD_BITS: u32 = 0x4e91_2345;
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP H66 synthetic official-geometry head timing: no Metal device");
            return;
        };
        assert!(h66_current_swap_is_zero(), "H66 requires zero current swap");
        let before_vm = h66_vm_counters().expect("H66 requires VM swap counters");
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile H66 pipelines");
        for pipeline in [&pipelines.q4_0_gemv_rows2, &pipelines.q4_0_gemv_rows4] {
            assert_eq!(pipeline.thread_execution_width(), 32);
            assert!(pipeline.max_total_threads_per_threadgroup() >= 32);
            assert_eq!(pipeline.static_threadgroup_memory_length(), 0);
        }
        let (head_weights, head) = h66_synthetic_official_geometry_head(&kernel.device);
        assert_eq!(
            (head.rows, head.cols),
            (VOCAB as u32, ASSISTANT_HIDDEN as u32)
        );
        assert_eq!(head.byte_len, OFFICIAL_HEAD_BYTES);
        assert_eq!(SEGMENTS.iter().sum::<usize>(), 44);
        let mut input_values: Vec<f32> = (0..ASSISTANT_HIDDEN)
            .map(|index| round_to_bf16_f32(((index as f32 * 0.015_625).sin() * 0.625) + 0.03125))
            .collect();
        input_values[..4].copy_from_slice(&[0.0, -0.0, 1.0, -1.0]);
        let input = f32_buffer(&kernel.device, &input_values);
        let initial_output = vec![f32::from_bits(OUTPUT_GUARD_BITS); VOCAB + OUTPUT_GUARD_ROWS];
        let output_established = f32_buffer(&kernel.device, &initial_output);
        let output_candidate = f32_buffer(&kernel.device, &initial_output);

        let exact_once = |path: H66HeadPath, output: &Buffer| {
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode_h66_head_path(
                encoder,
                &pipelines,
                path,
                &head_weights,
                &input,
                output,
                head,
                false,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
            let mut values = vec![0.0f32; VOCAB];
            read_buffer_f32(output, &mut values);
            values.into_iter().map(f32::to_bits).collect::<Vec<_>>()
        };
        let established_bits = exact_once(H66HeadPath::Established, &output_established);
        let rows2_bits = exact_once(H66HeadPath::Rows2, &output_candidate);
        assert_eq!(
            rows2_bits, established_bits,
            "H66 full-shape Row2 logits drifted"
        );
        let rows4_bits = exact_once(H66HeadPath::Rows4, &output_candidate);
        assert_eq!(
            rows4_bits, established_bits,
            "H66 full-shape Row4 logits drifted"
        );

        for warmup in 0..2 {
            let order = if warmup == 0 {
                [
                    H66HeadPath::Established,
                    H66HeadPath::Rows2,
                    H66HeadPath::Rows4,
                ]
            } else {
                [
                    H66HeadPath::Rows4,
                    H66HeadPath::Rows2,
                    H66HeadPath::Established,
                ]
            };
            for path in order {
                let output = match path {
                    H66HeadPath::Established => &output_established,
                    H66HeadPath::Rows2 | H66HeadPath::Rows4 => &output_candidate,
                };
                let _ = time_h66_head_request(
                    kernel,
                    &pipelines,
                    path,
                    &head_weights,
                    &input,
                    output,
                    head,
                );
            }
        }

        let mut established = Vec::with_capacity(9);
        let mut rows2 = Vec::with_capacity(9);
        let mut rows4 = Vec::with_capacity(9);
        for sample in 0..9 {
            let order = match sample % 3 {
                0 => [
                    H66HeadPath::Established,
                    H66HeadPath::Rows2,
                    H66HeadPath::Rows4,
                ],
                1 => [
                    H66HeadPath::Rows2,
                    H66HeadPath::Rows4,
                    H66HeadPath::Established,
                ],
                _ => [
                    H66HeadPath::Rows4,
                    H66HeadPath::Established,
                    H66HeadPath::Rows2,
                ],
            };
            for path in order {
                let output = match path {
                    H66HeadPath::Established => &output_established,
                    H66HeadPath::Rows2 | H66HeadPath::Rows4 => &output_candidate,
                };
                let timing = time_h66_head_request(
                    kernel,
                    &pipelines,
                    path,
                    &head_weights,
                    &input,
                    output,
                    head,
                );
                match path {
                    H66HeadPath::Established => established.push(timing),
                    H66HeadPath::Rows2 => rows2.push(timing),
                    H66HeadPath::Rows4 => rows4.push(timing),
                }
            }
        }

        let paired_savings = |candidate: &[H66HeadRequestTiming]| -> (i128, [i128; 5]) {
            let request: Vec<i128> = established
                .iter()
                .zip(candidate)
                .map(|(control, candidate)| control.total_us as i128 - candidate.total_us as i128)
                .collect();
            let segment_medians = std::array::from_fn(|segment| {
                let deltas: Vec<i128> = established
                    .iter()
                    .zip(candidate)
                    .map(|(control, candidate)| {
                        control.segments_us[segment] as i128
                            - candidate.segments_us[segment] as i128
                    })
                    .collect();
                h66_median_i128(&deltas)
            });
            (h66_median_i128(&request), segment_medians)
        };
        let (rows2_saving, rows2_segments) = paired_savings(&rows2);
        let (rows4_saving, rows4_segments) = paired_savings(&rows4);
        eprintln!(
            "[gemma4-mtp h66-head] official=0 official_geometry=1 synthetic=1 admission_only=1 head_rows={} head_cols={} head_bytes={} drafts=44 request_segments=4,9,12,13,6 warmups_per_path=2 samples=9 exact_logits={}/{} rows2_paired_saving_us={} rows4_paired_saving_us={} rows2_segment_saving_us={:?} rows4_segment_saving_us={:?}",
            head.rows,
            head.cols,
            head.byte_len,
            established_bits.len(),
            VOCAB,
            rows2_saving,
            rows4_saving,
            rows2_segments,
            rows4_segments,
        );
        for sample in 0..9 {
            eprintln!(
                "[gemma4-mtp h66-head] sample={} established_total_us={} rows2_total_us={} rows4_total_us={} established_segments_us={:?} rows2_segments_us={:?} rows4_segments_us={:?}",
                sample,
                established[sample].total_us,
                rows2[sample].total_us,
                rows4[sample].total_us,
                established[sample].segments_us,
                rows2[sample].segments_us,
                rows4[sample].segments_us,
            );
        }
        for (label, output) in [
            ("established", &output_established),
            ("candidate", &output_candidate),
        ] {
            let values = unsafe {
                std::slice::from_raw_parts(
                    output.contents().cast::<f32>(),
                    VOCAB + OUTPUT_GUARD_ROWS,
                )
            };
            assert!(
                values[VOCAB..]
                    .iter()
                    .all(|value| value.to_bits() == OUTPUT_GUARD_BITS),
                "H66 {label} overwrote the full-shape output guard"
            );
        }
        let head_storage = unsafe {
            std::slice::from_raw_parts(
                head_weights.contents().cast::<u8>(),
                head_weights.length() as usize,
            )
        };
        let head_start = head.byte_offset as usize;
        let head_end = head_start + head.byte_len as usize;
        assert!(head_storage[..head_start].iter().all(|byte| *byte == 0x5a));
        assert!(head_storage[head_end..].iter().all(|byte| *byte == 0x5a));
        let after_vm = h66_vm_counters().expect("H66 requires VM swap counters");
        assert_eq!(after_vm.swapins, before_vm.swapins, "H66 timing swapped in");
        assert_eq!(
            after_vm.swapouts, before_vm.swapouts,
            "H66 timing swapped out"
        );
        assert!(
            h66_current_swap_is_zero(),
            "H66 ended with nonzero current swap"
        );
        let (winner, winner_saving, winner_segments) = if rows4_saving >= rows2_saving {
            (H66HeadPath::Rows4, rows4_saving, rows4_segments)
        } else {
            (H66HeadPath::Rows2, rows2_saving, rows2_segments)
        };
        assert!(
            winner_segments.iter().all(|saving| *saving >= 0),
            "H66 {} regressed at least one request segment: {:?}",
            winner.label(),
            winner_segments
        );
        assert!(
            winner_saving >= 20_000,
            "H66 best exact path {} saved only {winner_saving}us/request; require at least 20000us",
            winner.label()
        );
    }

    #[test]
    #[ignore = "loads and quantizes the pinned 800 MiB official artifact"]
    fn official_full_q4_k7_warm_benchmark_receipt() {
        #[derive(Debug)]
        struct BenchResult {
            load_wall_us: u128,
            quantize_us: u128,
            mapped_bytes: u64,
            locked_bytes: u64,
            resident_pages: u64,
            total_pages: u64,
            packed_bytes: u64,
            matrix_bytes_per_draft: u64,
            k7_gpu_us: u128,
            k7_proposal_wall_us: u128,
            k7_outer_wall_us: u128,
        }

        fn run(full_q4: bool) -> BenchResult {
            let path = Path::new(OFFICIAL_STAGED_ASSISTANT_PATH);
            assert!(path.is_file(), "missing official assistant at {path:?}");
            let load_started = Instant::now();
            let mut assistant = Gemma4MtpAssistantMetal::load_with_full_q4(path, full_q4)
                .expect("load official assistant");
            let load_wall_us = load_started.elapsed().as_micros();
            assert_eq!(assistant.full_q4_enabled(), full_q4);
            assistant.warm_target_free().expect("warm assistant");

            let kernel = metal_linear_kernel().expect("Metal kernel");
            let sliding_elements = LOCAL_KV_HEADS * LOCAL_HEAD_DIM;
            let full_elements = FULL_KV_HEADS * FULL_HEAD_DIM;
            let sliding_key = f32_buffer(&kernel.device, &vec![0.0; sliding_elements]);
            let sliding_value = f32_buffer(&kernel.device, &vec![0.0; sliding_elements]);
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
            let initial_recurrent = vec![0.0f32; TARGET_HIDDEN];
            let outer_started = Instant::now();
            let proposals = assistant
                .propose_chain(
                    0,
                    &initial_recurrent,
                    target_kv,
                    7,
                    &[],
                    |_token, output| {
                        output.fill(0.0);
                        Ok(())
                    },
                )
                .expect("run K7 assistant chain");
            let k7_outer_wall_us = outer_started.elapsed().as_micros();
            assert_eq!(proposals.len(), 7);
            assert!(proposals.iter().all(|proposal| {
                proposal.recurrent_hidden.len() == TARGET_HIDDEN
                    && proposal
                        .recurrent_hidden
                        .iter()
                        .all(|value| value.is_finite())
            }));
            let ledger = assistant.resident_ledger();
            BenchResult {
                load_wall_us,
                quantize_us: ledger.full_q4_quantize_us,
                mapped_bytes: ledger.mapped_bytes,
                locked_bytes: ledger.locked_bytes,
                resident_pages: ledger.resident_pages,
                total_pages: ledger.total_pages,
                packed_bytes: ledger.full_q4_matrix_bytes,
                matrix_bytes_per_draft: assistant.assistant_matrix_bytes_per_proposal(),
                k7_gpu_us: proposals
                    .iter()
                    .map(|proposal| proposal.timing.gpu_us)
                    .sum(),
                k7_proposal_wall_us: proposals
                    .iter()
                    .map(|proposal| proposal.timing.wall_us)
                    .sum(),
                k7_outer_wall_us,
            }
        }

        let baseline = run(false);
        let full_q4 = run(true);
        assert_eq!(full_q4.packed_bytes, FULL_Q4_MATRIX_BYTES);
        assert_eq!(full_q4.matrix_bytes_per_draft, full_q4.packed_bytes);
        assert_eq!(baseline.matrix_bytes_per_draft, 453_509_120);
        assert!(baseline.mapped_bytes > 0);
        assert!(baseline.locked_bytes > 0);
        assert!(baseline.resident_pages > 0);
        assert_eq!(baseline.resident_pages, baseline.total_pages);
        assert_eq!(full_q4.mapped_bytes, 0);
        assert_eq!(full_q4.locked_bytes, 0);
        assert_eq!(full_q4.resident_pages, 0);
        assert_eq!(full_q4.total_pages, 0);
        assert!(full_q4.quantize_us > 0);
        eprintln!(
            "[gemma4-mtp full-q4 benchmark] baseline={baseline:?} full_q4={full_q4:?} traffic_ratio={:.3} gpu_speedup={:.3} proposal_wall_speedup={:.3}",
            baseline.matrix_bytes_per_draft as f64 / full_q4.matrix_bytes_per_draft as f64,
            baseline.k7_gpu_us as f64 / full_q4.k7_gpu_us as f64,
            baseline.k7_proposal_wall_us as f64 / full_q4.k7_proposal_wall_us as f64,
        );
    }

    #[test]
    #[ignore = "loads and quantizes the pinned 800 MiB official artifact"]
    fn official_full_q4_k4_device_chain_warmup_restores_exact_private_state() {
        fn pattern(byte_len: usize, salt: u8) -> Vec<u8> {
            (0..byte_len)
                .map(|index| (index as u8).wrapping_mul(37).wrapping_add(salt))
                .collect()
        }

        let path = Path::new(OFFICIAL_STAGED_ASSISTANT_PATH);
        assert!(path.is_file(), "missing official assistant at {path:?}");
        let mut assistant = Gemma4MtpAssistantMetal::load_with_full_q4(path, true)
            .expect("load official full-Q4 assistant");

        let recurrent_bytes = TARGET_HIDDEN * std::mem::size_of::<f32>();
        let chain_recurrent_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * recurrent_bytes;
        let token_bytes = MTP_DEVICE_CHAIN_K4_WARM_DRAFTS * std::mem::size_of::<u32>();
        assert_eq!(
            recurrent_bytes + chain_recurrent_bytes + token_bytes,
            56_336
        );

        let recurrent_before = pattern(recurrent_bytes, 11);
        let chain_recurrent_before = pattern(chain_recurrent_bytes, 29);
        let tokens_before = pattern(token_bytes, 47);
        write_buffer_prefix_bytes(&assistant.scratch.recurrent_hidden, &recurrent_before)
            .expect("seed recurrent scratch");
        write_buffer_prefix_bytes(
            &assistant.scratch.chain_recurrent_hidden,
            &chain_recurrent_before,
        )
        .expect("seed chain recurrent scratch");
        write_buffer_prefix_bytes(&assistant.scratch.output_token, &tokens_before)
            .expect("seed token scratch");
        let ledger_before = Some(Gemma4MtpProposalLedger {
            assistant_matrix_bytes: 101,
            borrowed_target_kv_capacity_bytes: 103,
            target_kv_read_bytes: 107,
            dynamic_attention_scratch_bytes: 109,
            readback_bytes: 113,
        });
        assistant.last_proposal_ledger = ledger_before;

        assistant
            .warm_target_free_device_chain_k4_step3_capture()
            .expect("run target-free K4 step-3-capturing device-chain warmup");

        assert_eq!(
            read_buffer_prefix_bytes(&assistant.scratch.recurrent_hidden, recurrent_bytes)
                .expect("read restored recurrent scratch"),
            recurrent_before,
        );
        assert_eq!(
            read_buffer_prefix_bytes(
                &assistant.scratch.chain_recurrent_hidden,
                chain_recurrent_bytes,
            )
            .expect("read restored chain recurrent scratch"),
            chain_recurrent_before,
        );
        assert_eq!(
            read_buffer_prefix_bytes(&assistant.scratch.output_token, token_bytes)
                .expect("read restored token scratch"),
            tokens_before,
        );
        assert_eq!(assistant.last_proposal_ledger, ledger_before);
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
    fn target_free_device_chain_embedding_geometry_admits_only_one_exact_q6k_row() {
        validate_target_free_device_chain_embedding_geometry(
            Gemma4MtpTargetEmbeddingFormat::Q6K,
            TARGET_HIDDEN,
            1,
            0,
            TARGET_Q6K_ROW_BYTES,
            TARGET_Q6K_ROW_BYTES,
        )
        .unwrap();

        // The compact warm-only binding must remain impossible to admit through
        // the production full-vocabulary validator.
        assert!(validate_target_embedding_geometry(
            Gemma4MtpTargetEmbeddingFormat::Q6K,
            TARGET_HIDDEN,
            1,
            0,
            TARGET_Q6K_ROW_BYTES,
            TARGET_Q6K_ROW_BYTES,
        )
        .is_err());

        let malformed = [
            (
                Gemma4MtpTargetEmbeddingFormat::Q4K,
                TARGET_HIDDEN,
                1,
                0,
                TARGET_Q6K_ROW_BYTES,
                TARGET_Q6K_ROW_BYTES,
            ),
            (
                Gemma4MtpTargetEmbeddingFormat::Q6K,
                TARGET_HIDDEN - 1,
                1,
                0,
                TARGET_Q6K_ROW_BYTES,
                TARGET_Q6K_ROW_BYTES,
            ),
            (
                Gemma4MtpTargetEmbeddingFormat::Q6K,
                TARGET_HIDDEN,
                2,
                0,
                TARGET_Q6K_ROW_BYTES,
                TARGET_Q6K_ROW_BYTES,
            ),
            (
                Gemma4MtpTargetEmbeddingFormat::Q6K,
                TARGET_HIDDEN,
                1,
                2,
                TARGET_Q6K_ROW_BYTES,
                TARGET_Q6K_ROW_BYTES + 2,
            ),
            (
                Gemma4MtpTargetEmbeddingFormat::Q6K,
                TARGET_HIDDEN,
                1,
                0,
                TARGET_Q6K_ROW_BYTES - 1,
                TARGET_Q6K_ROW_BYTES,
            ),
            (
                Gemma4MtpTargetEmbeddingFormat::Q6K,
                TARGET_HIDDEN,
                1,
                0,
                TARGET_Q6K_ROW_BYTES,
                TARGET_Q6K_ROW_BYTES + 1,
            ),
        ];
        for (format, hidden, vocab, byte_offset, byte_len, buffer_len) in malformed {
            assert!(validate_target_free_device_chain_embedding_geometry(
                format,
                hidden,
                vocab,
                byte_offset,
                byte_len,
                buffer_len,
            )
            .is_err());
        }
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
                false,
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

    fn bf16_fusion_adversarial_finite_values(count: usize, phase: usize) -> Vec<f32> {
        const EDGE_BITS: [u32; 20] = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3e80_0000,
            0xbe80_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x3f7f_8000,
            0xbf7f_8000,
            0x3f80_8000,
            0xbf80_8000,
            0x3f81_8000,
            0xbf81_8000,
            0x4000_4000,
            0xc000_4000,
        ];
        (0..count)
            .map(|index| f32::from_bits(EDGE_BITS[(index + phase) % EDGE_BITS.len()]))
            .collect()
    }

    fn raw_f32_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn run_test_rms_bf16_boundary(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        input: &[f32],
        weight: &[f32],
        width: usize,
        head_count: usize,
        fused: bool,
    ) -> Vec<u32> {
        assert_eq!(input.len(), width * head_count);
        assert_eq!(weight.len(), width);
        let input_buffer = f32_buffer(&kernel.device, input);
        let weight_buffer = f32_buffer(&kernel.device, weight);
        let output_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(input));
        let scalar = shared_buffer(&kernel.device, 8);
        set_rms_scalar(&scalar, width);
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
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
            fused,
        );
        if !fused {
            encode_round_bf16(encoder, &pipelines.round_bf16, &output_buffer, input.len());
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; input.len()];
        read_buffer_f32(&output_buffer, &mut output);
        raw_f32_bits(&output)
    }

    fn run_test_residual_bf16_boundary(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        a: &[f32],
        b: &[f32],
        fused: bool,
    ) -> Vec<u32> {
        assert_eq!(a.len(), b.len());
        let a_buffer = f32_buffer(&kernel.device, a);
        let b_buffer = f32_buffer(&kernel.device, b);
        let output_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(a));
        let count = shared_buffer(&kernel.device, std::mem::size_of::<u32>());
        set_count(&count, a.len());
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_binary(
            encoder,
            if fused {
                &pipelines.residual_add_bf16
            } else {
                &kernel.residual_add_pipeline
            },
            &a_buffer,
            &b_buffer,
            &output_buffer,
            &count,
            a.len(),
        );
        if !fused {
            encode_round_bf16(encoder, &pipelines.round_bf16, &output_buffer, a.len());
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; a.len()];
        read_buffer_f32(&output_buffer, &mut output);
        raw_f32_bits(&output)
    }

    fn run_test_scale_bf16_boundary(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        input: &[f32],
        scale: f32,
        fused: bool,
    ) -> Vec<u32> {
        let input_buffer = f32_buffer(&kernel.device, input);
        let output_buffer = shared_buffer(&kernel.device, std::mem::size_of_val(input));
        let scalar = shared_buffer(&kernel.device, 8);
        unsafe {
            let ptr = scalar.contents().cast::<u8>();
            *ptr.cast::<u32>() = input.len() as u32;
            *ptr.add(4).cast::<f32>() = scale;
        }
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        if fused {
            encode_scale_bf16(
                encoder,
                &pipelines.scale_bf16,
                &input_buffer,
                &output_buffer,
                &scalar,
                input.len(),
            );
        } else {
            encode_scale_f32(
                encoder,
                kernel,
                &input_buffer,
                &output_buffer,
                &scalar,
                input.len(),
            );
            encode_round_bf16(encoder, &pipelines.round_bf16, &output_buffer, input.len());
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let mut output = vec![0.0f32; input.len()];
        read_buffer_f32(&output_buffer, &mut output);
        raw_f32_bits(&output)
    }

    struct Bf16ProducerBoundaryBench {
        hidden_input: Buffer,
        hidden_weight: Buffer,
        hidden_output: Buffer,
        hidden_scalar: Buffer,
        local_query_input: Buffer,
        local_query_weight: Buffer,
        local_query_output: Buffer,
        local_query_scalar: Buffer,
        full_query_input: Buffer,
        full_query_weight: Buffer,
        full_query_output: Buffer,
        full_query_scalar: Buffer,
        residual_a: Buffer,
        residual_b: Buffer,
        residual_output: Buffer,
        hidden_count: Buffer,
        scale_output: Buffer,
        scale_scalar: Buffer,
    }

    impl Bf16ProducerBoundaryBench {
        fn new(device: &Device) -> Self {
            let hidden_input_values = bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 3);
            let hidden_weight_values: Vec<f32> =
                bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 9)
                    .into_iter()
                    .map(|value| 1.0f32 + value * 0.125f32)
                    .collect();
            let local_query_values =
                bf16_fusion_adversarial_finite_values(N_HEADS * LOCAL_HEAD_DIM, 5);
            let local_query_weight_values: Vec<f32> =
                bf16_fusion_adversarial_finite_values(LOCAL_HEAD_DIM, 11)
                    .into_iter()
                    .map(|value| 1.0f32 + value * 0.125f32)
                    .collect();
            let full_query_values =
                bf16_fusion_adversarial_finite_values(N_HEADS * FULL_HEAD_DIM, 7);
            let full_query_weight_values: Vec<f32> =
                bf16_fusion_adversarial_finite_values(FULL_HEAD_DIM, 13)
                    .into_iter()
                    .map(|value| 1.0f32 + value * 0.125f32)
                    .collect();
            let residual_b_values = bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 17);
            let hidden_scalar = shared_buffer(device, 8);
            let local_query_scalar = shared_buffer(device, 8);
            let full_query_scalar = shared_buffer(device, 8);
            set_rms_scalar(&hidden_scalar, ASSISTANT_HIDDEN);
            set_rms_scalar(&local_query_scalar, LOCAL_HEAD_DIM);
            set_rms_scalar(&full_query_scalar, FULL_HEAD_DIM);
            let hidden_count = shared_buffer(device, std::mem::size_of::<u32>());
            set_count(&hidden_count, ASSISTANT_HIDDEN);
            let scale_scalar = shared_buffer(device, 8);
            unsafe {
                let ptr = scale_scalar.contents().cast::<u8>();
                *ptr.cast::<u32>() = ASSISTANT_HIDDEN as u32;
                *ptr.add(4).cast::<f32>() = 0.75f32;
            }
            Self {
                hidden_input: f32_buffer(device, &hidden_input_values),
                hidden_weight: f32_buffer(device, &hidden_weight_values),
                hidden_output: shared_buffer(device, ASSISTANT_HIDDEN * std::mem::size_of::<f32>()),
                hidden_scalar,
                local_query_input: f32_buffer(device, &local_query_values),
                local_query_weight: f32_buffer(device, &local_query_weight_values),
                local_query_output: shared_buffer(
                    device,
                    N_HEADS * LOCAL_HEAD_DIM * std::mem::size_of::<f32>(),
                ),
                local_query_scalar,
                full_query_input: f32_buffer(device, &full_query_values),
                full_query_weight: f32_buffer(device, &full_query_weight_values),
                full_query_output: shared_buffer(
                    device,
                    N_HEADS * FULL_HEAD_DIM * std::mem::size_of::<f32>(),
                ),
                full_query_scalar,
                residual_a: f32_buffer(device, &hidden_input_values),
                residual_b: f32_buffer(device, &residual_b_values),
                residual_output: shared_buffer(
                    device,
                    ASSISTANT_HIDDEN * std::mem::size_of::<f32>(),
                ),
                hidden_count,
                scale_output: shared_buffer(device, ASSISTANT_HIDDEN * std::mem::size_of::<f32>()),
                scale_scalar,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_bf16_boundary_bench_rms(
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MtpPipelines,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        scalar: &Buffer,
        width: usize,
        head_count: usize,
        fused: bool,
    ) {
        encode_assistant_aten_rms_norm(
            encoder,
            &pipelines.rms_norm_aten_f32,
            input,
            weight,
            output,
            scalar,
            width,
            head_count,
            0,
            true,
            fused,
        );
        if !fused {
            encode_round_bf16(encoder, &pipelines.round_bf16, output, width * head_count);
        }
    }

    fn encode_bf16_producer_boundary_bench_proposal(
        encoder: &metal::ComputeCommandEncoderRef,
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        buffers: &Bf16ProducerBoundaryBench,
        fused: bool,
    ) {
        for layer in 0..4 {
            encode_bf16_boundary_bench_rms(
                encoder,
                pipelines,
                &buffers.hidden_input,
                &buffers.hidden_weight,
                &buffers.hidden_output,
                &buffers.hidden_scalar,
                ASSISTANT_HIDDEN,
                1,
                fused,
            );
            let (query_input, query_weight, query_output, query_scalar, head_dim) = if layer < 3 {
                (
                    &buffers.local_query_input,
                    &buffers.local_query_weight,
                    &buffers.local_query_output,
                    &buffers.local_query_scalar,
                    LOCAL_HEAD_DIM,
                )
            } else {
                (
                    &buffers.full_query_input,
                    &buffers.full_query_weight,
                    &buffers.full_query_output,
                    &buffers.full_query_scalar,
                    FULL_HEAD_DIM,
                )
            };
            encode_bf16_boundary_bench_rms(
                encoder,
                pipelines,
                query_input,
                query_weight,
                query_output,
                query_scalar,
                head_dim,
                N_HEADS,
                fused,
            );
            encode_bf16_boundary_bench_rms(
                encoder,
                pipelines,
                &buffers.hidden_input,
                &buffers.hidden_weight,
                &buffers.hidden_output,
                &buffers.hidden_scalar,
                ASSISTANT_HIDDEN,
                1,
                fused,
            );
            encode_binary(
                encoder,
                if fused {
                    &pipelines.residual_add_bf16
                } else {
                    &kernel.residual_add_pipeline
                },
                &buffers.residual_a,
                &buffers.residual_b,
                &buffers.residual_output,
                &buffers.hidden_count,
                ASSISTANT_HIDDEN,
            );
            if !fused {
                encode_round_bf16(
                    encoder,
                    &pipelines.round_bf16,
                    &buffers.residual_output,
                    ASSISTANT_HIDDEN,
                );
            }
            for _ in 0..2 {
                encode_bf16_boundary_bench_rms(
                    encoder,
                    pipelines,
                    &buffers.hidden_input,
                    &buffers.hidden_weight,
                    &buffers.hidden_output,
                    &buffers.hidden_scalar,
                    ASSISTANT_HIDDEN,
                    1,
                    fused,
                );
            }
            encode_binary(
                encoder,
                if fused {
                    &pipelines.residual_add_bf16
                } else {
                    &kernel.residual_add_pipeline
                },
                &buffers.residual_a,
                &buffers.residual_b,
                &buffers.residual_output,
                &buffers.hidden_count,
                ASSISTANT_HIDDEN,
            );
            if !fused {
                encode_round_bf16(
                    encoder,
                    &pipelines.round_bf16,
                    &buffers.residual_output,
                    ASSISTANT_HIDDEN,
                );
            }
            if fused {
                encode_scale_bf16(
                    encoder,
                    &pipelines.scale_bf16,
                    &buffers.hidden_input,
                    &buffers.scale_output,
                    &buffers.scale_scalar,
                    ASSISTANT_HIDDEN,
                );
            } else {
                encode_scale_f32(
                    encoder,
                    kernel,
                    &buffers.hidden_input,
                    &buffers.scale_output,
                    &buffers.scale_scalar,
                    ASSISTANT_HIDDEN,
                );
                encode_round_bf16(
                    encoder,
                    &pipelines.round_bf16,
                    &buffers.scale_output,
                    ASSISTANT_HIDDEN,
                );
            }
        }
        encode_bf16_boundary_bench_rms(
            encoder,
            pipelines,
            &buffers.hidden_input,
            &buffers.hidden_weight,
            &buffers.hidden_output,
            &buffers.hidden_scalar,
            ASSISTANT_HIDDEN,
            1,
            fused,
        );
    }

    fn time_bf16_producer_boundary_request(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        buffers: &Bf16ProducerBoundaryBench,
        fused: bool,
    ) -> u128 {
        const REQUEST_DRAFTS: usize = 44;
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for _ in 0..REQUEST_DRAFTS {
            encode_bf16_producer_boundary_bench_proposal(
                encoder, kernel, pipelines, buffers, fused,
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let (gpu_us, _) = command_buffer_gpu_times_us(command_buffer);
        gpu_us
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
        let (cos, sin) = full_rope_table_values(1_030);
        assert_eq!(cos.len(), 256);
        assert_eq!(sin.len(), 256);
        assert_ne!(cos[FULL_ROPE_ACTIVE_PAIRS - 1].to_bits(), 1.0f32.to_bits());
        assert_ne!(sin[FULL_ROPE_ACTIVE_PAIRS - 1].to_bits(), 0.0f32.to_bits());
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
    fn cpu_and_device_chains_share_proportional_full_rope_tables() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let scratch = MtpScratch::new(&kernel.device);
        write_assistant_rope_tables(1_030, &scratch);

        let mut cos = vec![0.0f32; FULL_HEAD_DIM / 2];
        let mut sin = vec![0.0f32; FULL_HEAD_DIM / 2];
        read_buffer_f32(&scratch.full_cos, &mut cos);
        read_buffer_f32(&scratch.full_sin, &mut sin);
        assert_ne!(cos[FULL_ROPE_ACTIVE_PAIRS - 1].to_bits(), 1.0f32.to_bits());
        assert_ne!(sin[FULL_ROPE_ACTIVE_PAIRS - 1].to_bits(), 0.0f32.to_bits());
        assert!(cos[FULL_ROPE_ACTIVE_PAIRS..]
            .iter()
            .all(|value| value.to_bits() == 1.0f32.to_bits()));
        assert!(sin[FULL_ROPE_ACTIVE_PAIRS..]
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits()));
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
    fn bf16_producer_fusion_rms_is_raw_u32_identical_at_all_production_widths() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        for (width, head_count) in [(256usize, 16usize), (512, 16), (1_024, 1)] {
            let input = bf16_fusion_adversarial_finite_values(width * head_count, width);
            let raw_weight = bf16_fusion_adversarial_finite_values(width, head_count);
            let weight: Vec<f32> = raw_weight
                .into_iter()
                .map(|value| 1.0f32 + value * 0.125f32)
                .collect();
            assert!(input.iter().chain(&weight).all(|value| value.is_finite()));
            let control = run_test_rms_bf16_boundary(
                kernel, &pipelines, &input, &weight, width, head_count, false,
            );
            let fused = run_test_rms_bf16_boundary(
                kernel, &pipelines, &input, &weight, width, head_count, true,
            );
            assert_eq!(
                fused, control,
                "fused RMS BF16 boundary drifted at width={width} heads={head_count}"
            );
        }
    }

    #[test]
    fn bf16_producer_fusion_residual_is_raw_u32_identical_on_edge_values() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let mut a = bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 0);
        let mut b = bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 7);
        a[..6].copy_from_slice(&[
            0.0f32,
            -0.0f32,
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
            f32::from_bits(0x0000_0001),
            f32::from_bits(0x8000_0001),
        ]);
        b[..6].copy_from_slice(&[-0.0f32, -0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]);
        let control = run_test_residual_bf16_boundary(kernel, &pipelines, &a, &b, false);
        let fused = run_test_residual_bf16_boundary(kernel, &pipelines, &a, &b, true);
        assert_eq!(fused, control);
    }

    #[test]
    fn bf16_producer_fusion_scale_is_raw_u32_identical_on_edge_values() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let input = bf16_fusion_adversarial_finite_values(ASSISTANT_HIDDEN, 13);
        for scale in [1.0f32, 0.75f32, -0.5f32] {
            let control = run_test_scale_bf16_boundary(kernel, &pipelines, &input, scale, false);
            let fused = run_test_scale_bf16_boundary(kernel, &pipelines, &input, scale, true);
            assert_eq!(fused, control, "fused scale drifted for scale={scale}");
        }
    }

    #[test]
    #[ignore = "real-Metal exact 44-proposal BF16 producer-fusion timing gate"]
    fn bf16_producer_fusion_exact_44_proposal_microbenchmark() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP BF16 producer-fusion benchmark: no Metal device");
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let buffers = Bf16ProducerBoundaryBench::new(&kernel.device);
        assert_eq!(MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT, 33);
        assert_eq!(44 * MTP_STANDALONE_BF16_ROUND_DISPATCHES_PER_DRAFT, 1_452);

        for _ in 0..2 {
            let _ = time_bf16_producer_boundary_request(kernel, &pipelines, &buffers, false);
            let _ = time_bf16_producer_boundary_request(kernel, &pipelines, &buffers, true);
        }

        let mut control = Vec::with_capacity(9);
        let mut fused = Vec::with_capacity(9);
        for sample in 0..9 {
            if sample % 2 == 0 {
                control.push(time_bf16_producer_boundary_request(
                    kernel, &pipelines, &buffers, false,
                ));
                fused.push(time_bf16_producer_boundary_request(
                    kernel, &pipelines, &buffers, true,
                ));
            } else {
                fused.push(time_bf16_producer_boundary_request(
                    kernel, &pipelines, &buffers, true,
                ));
                control.push(time_bf16_producer_boundary_request(
                    kernel, &pipelines, &buffers, false,
                ));
            }
        }
        let median = |samples: &[u128]| {
            let mut ordered = samples.to_vec();
            ordered.sort_unstable();
            ordered[ordered.len() / 2]
        };
        let control_median_us = median(&control);
        let fused_median_us = median(&fused);
        let saving_us = control_median_us as i128 - fused_median_us as i128;
        eprintln!(
            "[gemma4-mtp bf16-producer-fusion benchmark] drafts=44 warmups_per_path=2 samples=9 control_gpu_us={} fused_gpu_us={} saving_us={} standalone_round_dispatches_elided=1452 standalone_round_rw_bytes_elided=17661952",
            control_median_us,
            fused_median_us,
            saving_us,
        );
        for sample in 0..9 {
            eprintln!(
                "[gemma4-mtp bf16-producer-fusion benchmark] sample={} control_gpu_us={} fused_gpu_us={}",
                sample,
                control[sample],
                fused[sample],
            );
        }
        assert!(
            saving_us >= 5_000,
            "BF16 producer fusion saved only {saving_us} us; require at least 5000 us/request"
        );
    }

    struct Bf16LatticeAttentionBenchRound {
        drafts: usize,
        logical_k: usize,
        local_scalar: Buffer,
        full_scalar: Buffer,
    }

    struct Bf16LatticeAttentionBench {
        local_query: Buffer,
        local_keys: Buffer,
        local_values: Buffer,
        local_scores: Buffer,
        local_output: Buffer,
        full_query: Buffer,
        full_keys: Buffer,
        full_values: Buffer,
        full_scores: Buffer,
        full_output: Buffer,
        rounds: Vec<Bf16LatticeAttentionBenchRound>,
    }

    impl Bf16LatticeAttentionBench {
        fn new(device: &Device) -> Self {
            const KV_STRIDE: usize = 192;
            const MAX_LOGICAL_K: usize = 145;
            let local_query_values =
                bf16_lattice_test_values(N_HEADS * LOCAL_HEAD_DIM, 0x9100, false);
            let local_key_values =
                dirty_low16_test_values(LOCAL_KV_HEADS * KV_STRIDE * LOCAL_HEAD_DIM, 0x9200);
            let local_value_values =
                dirty_low16_test_values(LOCAL_KV_HEADS * KV_STRIDE * LOCAL_HEAD_DIM, 0x9300);
            let full_query_values =
                bf16_lattice_test_values(N_HEADS * FULL_HEAD_DIM, 0x9400, false);
            let full_key_values =
                dirty_low16_test_values(FULL_KV_HEADS * KV_STRIDE * FULL_HEAD_DIM, 0x9500);
            let full_value_values =
                dirty_low16_test_values(FULL_KV_HEADS * KV_STRIDE * FULL_HEAD_DIM, 0x9600);
            let rounds = [(13usize, 104usize), (12, 118), (13, 131), (6, 145)]
                .into_iter()
                .map(|(drafts, logical_k)| Bf16LatticeAttentionBenchRound {
                    drafts,
                    logical_k,
                    local_scalar: test_attention_scalar(
                        device,
                        N_HEADS,
                        LOCAL_HEAD_DIM,
                        logical_k,
                        N_HEADS / LOCAL_KV_HEADS,
                        KV_STRIDE,
                        0,
                        logical_k,
                    ),
                    full_scalar: test_attention_scalar(
                        device,
                        N_HEADS,
                        FULL_HEAD_DIM,
                        logical_k,
                        N_HEADS / FULL_KV_HEADS,
                        KV_STRIDE,
                        0,
                        logical_k,
                    ),
                })
                .collect();
            Self {
                local_query: f32_buffer(device, &local_query_values),
                local_keys: f32_buffer(device, &local_key_values),
                local_values: f32_buffer(device, &local_value_values),
                local_scores: shared_buffer(
                    device,
                    N_HEADS * MAX_LOGICAL_K * std::mem::size_of::<f32>(),
                ),
                local_output: shared_buffer(
                    device,
                    N_HEADS * LOCAL_HEAD_DIM * std::mem::size_of::<f32>(),
                ),
                full_query: f32_buffer(device, &full_query_values),
                full_keys: f32_buffer(device, &full_key_values),
                full_values: f32_buffer(device, &full_value_values),
                full_scores: shared_buffer(
                    device,
                    N_HEADS * MAX_LOGICAL_K * std::mem::size_of::<f32>(),
                ),
                full_output: shared_buffer(
                    device,
                    N_HEADS * FULL_HEAD_DIM * std::mem::size_of::<f32>(),
                ),
                rounds,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_bf16_lattice_attention_bench_layer(
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MtpPipelines,
        query: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        scores: &Buffer,
        output: &Buffer,
        scalar: &Buffer,
        lattice_loads: bool,
    ) {
        let groups = MTLSize {
            width: N_HEADS as u64,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.set_compute_pipeline_state(pipelines.selected_attention_scores_bf16(lattice_loads));
        encoder.set_buffer(0, Some(query), 0);
        encoder.set_buffer(1, Some(keys), 0);
        encoder.set_buffer(2, Some(scores), 0);
        for index in 0..8u64 {
            encoder.set_buffer(3 + index, Some(scalar), index * 4);
        }
        encoder.dispatch_thread_groups(groups, threads);

        encoder.set_compute_pipeline_state(&pipelines.attention_softmax_bf16);
        encoder.set_buffer(0, Some(scores), 0);
        encoder.set_buffer(1, Some(scalar), 0);
        encoder.set_buffer(2, Some(scalar), 8);
        encoder.dispatch_thread_groups(groups, threads);

        encoder
            .set_compute_pipeline_state(pipelines.selected_attention_context_bf16(lattice_loads));
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
    }

    fn time_bf16_lattice_attention_request(
        kernel: &MetalLinearKernel,
        pipelines: &MtpPipelines,
        buffers: &Bf16LatticeAttentionBench,
        lattice_loads: bool,
    ) -> u128 {
        let command_buffer = kernel.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for round in &buffers.rounds {
            for _ in 0..round.drafts {
                for _ in 0..3 {
                    encode_bf16_lattice_attention_bench_layer(
                        encoder,
                        pipelines,
                        &buffers.local_query,
                        &buffers.local_keys,
                        &buffers.local_values,
                        &buffers.local_scores,
                        &buffers.local_output,
                        &round.local_scalar,
                        lattice_loads,
                    );
                }
                encode_bf16_lattice_attention_bench_layer(
                    encoder,
                    pipelines,
                    &buffers.full_query,
                    &buffers.full_keys,
                    &buffers.full_values,
                    &buffers.full_scores,
                    &buffers.full_output,
                    &round.full_scalar,
                    lattice_loads,
                );
            }
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let (gpu_us, _) = command_buffer_gpu_times_us(command_buffer);
        gpu_us
    }

    fn read_bf16_lattice_bench_outputs(buffers: &Bf16LatticeAttentionBench) -> Vec<u32> {
        let mut local = vec![0.0f32; N_HEADS * LOCAL_HEAD_DIM];
        let mut full = vec![0.0f32; N_HEADS * FULL_HEAD_DIM];
        read_buffer_f32(&buffers.local_output, &mut local);
        read_buffer_f32(&buffers.full_output, &mut full);
        local.into_iter().chain(full).map(f32::to_bits).collect()
    }

    #[test]
    #[ignore = "real-Metal exact 44-proposal BF16 lattice-load timing gate"]
    fn bf16_lattice_loads_exact_44_proposal_interleaved_timing() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP BF16 lattice-load benchmark: no Metal device");
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP kernels");
        let buffers = Bf16LatticeAttentionBench::new(&kernel.device);
        assert_eq!(
            buffers
                .rounds
                .iter()
                .map(|round| round.drafts)
                .sum::<usize>(),
            44
        );
        // This is a continuation/admission microbenchmark, not a request-wall
        // prediction: 528 attention dispatches share one command buffer and
        // deliberately reuse the same compact buffers across all 44 drafts.
        assert_eq!(44 * 4 * 3, 528);
        let round_ops = buffers
            .rounds
            .iter()
            .map(|round| {
                round.drafts * mtp_bf16_lattice_rounds_elided_per_draft(round.logical_k).unwrap()
            })
            .sum::<usize>();
        assert_eq!(round_ops, 218_767_360);

        let _ = time_bf16_lattice_attention_request(kernel, &pipelines, &buffers, false);
        let control_bits = read_bf16_lattice_bench_outputs(&buffers);
        let _ = time_bf16_lattice_attention_request(kernel, &pipelines, &buffers, true);
        let lattice_bits = read_bf16_lattice_bench_outputs(&buffers);
        assert_eq!(lattice_bits, control_bits, "timing workload output drifted");

        for _ in 0..2 {
            let _ = time_bf16_lattice_attention_request(kernel, &pipelines, &buffers, false);
            let _ = time_bf16_lattice_attention_request(kernel, &pipelines, &buffers, true);
        }

        let mut control = Vec::with_capacity(9);
        let mut lattice = Vec::with_capacity(9);
        for sample in 0..9 {
            if sample % 2 == 0 {
                control.push(time_bf16_lattice_attention_request(
                    kernel, &pipelines, &buffers, false,
                ));
                lattice.push(time_bf16_lattice_attention_request(
                    kernel, &pipelines, &buffers, true,
                ));
            } else {
                lattice.push(time_bf16_lattice_attention_request(
                    kernel, &pipelines, &buffers, true,
                ));
                control.push(time_bf16_lattice_attention_request(
                    kernel, &pipelines, &buffers, false,
                ));
            }
        }
        let median = |samples: &[u128]| {
            let mut ordered = samples.to_vec();
            ordered.sort_unstable();
            ordered[ordered.len() / 2]
        };
        let control_median_us = median(&control);
        let lattice_median_us = median(&lattice);
        let saving_us = control_median_us as i128 - lattice_median_us as i128;
        eprintln!(
            "[gemma4-mtp bf16-lattice-loads benchmark] drafts=44 request_segments=13@104,12@118,13@131,6@145 exactness_runs_per_path=1 warmups_per_path=2 premeasurement_runs_per_path=3 samples=9 control_gpu_us={} lattice_gpu_us={} saving_us={} bf16_round_ops_elided=218767360 attention_dispatches=528 command_buffers=1 buffers_reused=1 admission_only=1 dispatches_elided=0 bytes_elided=0 scratch_bytes_added=0",
            control_median_us,
            lattice_median_us,
            saving_us,
        );
        for sample in 0..9 {
            eprintln!(
                "[gemma4-mtp bf16-lattice-loads benchmark] sample={} control_gpu_us={} lattice_gpu_us={}",
                sample,
                control[sample],
                lattice[sample],
            );
        }
        assert!(
            saving_us >= 5_000,
            "BF16 lattice loads saved only {saving_us} us; require at least 5000 us/request"
        );
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
            assert_eq!(run(&pipelines.attention_scores_bf16_lattice_query), 49_680);
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

    fn bf16_lattice_test_values(count: usize, seed: u32, positive: bool) -> Vec<f32> {
        let (mut values, _) = deterministic_bf16_values(count, seed, 116, 10);
        if positive {
            for value in &mut values {
                *value = value.abs();
            }
        }
        assert!(values.iter().all(|value| value.to_bits() & 0xffff == 0));
        values
    }

    fn dirty_low16_test_values(count: usize, seed: u32) -> Vec<f32> {
        let (values, _) = deterministic_bf16_values(count, seed, 116, 10);
        let values: Vec<f32> = values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let low = ((index as u32).wrapping_mul(0x9e37).wrapping_add(seed) & 0xffff) | 1;
                f32::from_bits(value.to_bits() | low)
            })
            .collect();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| value.to_bits() & 0xffff != 0));
        values
    }

    fn test_attention_scalar(
        device: &Device,
        head_count: usize,
        head_dim: usize,
        position_count: usize,
        group: usize,
        kv_stride: usize,
        compact_base: usize,
        physical_logical_k: usize,
    ) -> Buffer {
        let scalar = shared_buffer(device, 40);
        let words = [
            head_count as u32,
            head_dim as u32,
            position_count as u32,
            group as u32,
            1.0f32.to_bits(),
            head_dim as u32,
            (kv_stride * head_dim) as u32,
            (compact_base * head_dim) as u32,
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
        scalar
    }

    #[test]
    fn bf16_lattice_loads_match_raw_u32_at_local_and_full_production_widths() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        const KV_STRIDE: usize = 192;
        for (name, head_dim, kv_heads, logical_k) in [
            ("local", LOCAL_HEAD_DIM, LOCAL_KV_HEADS, 145usize),
            ("full", FULL_HEAD_DIM, FULL_KV_HEADS, 145usize),
        ] {
            let group = N_HEADS / kv_heads;
            let query = bf16_lattice_test_values(N_HEADS * head_dim, head_dim as u32, false);
            let keys = dirty_low16_test_values(kv_heads * KV_STRIDE * head_dim, 0x5130);
            let control_scores = run_test_attention_scores(
                kernel,
                pipelines.selected_attention_scores_bf16(false),
                &query,
                &keys,
                N_HEADS,
                head_dim,
                logical_k,
                group,
                head_dim,
                KV_STRIDE * head_dim,
                0,
            );
            let lattice_scores = run_test_attention_scores(
                kernel,
                pipelines.selected_attention_scores_bf16(true),
                &query,
                &keys,
                N_HEADS,
                head_dim,
                logical_k,
                group,
                head_dim,
                KV_STRIDE * head_dim,
                0,
            );
            assert_eq!(
                raw_f32_bits(&lattice_scores),
                raw_f32_bits(&control_scores),
                "{name} QK direct-query load drifted"
            );

            let probabilities =
                bf16_lattice_test_values(N_HEADS * logical_k, 0x6200 + head_dim as u32, true);
            let values = dirty_low16_test_values(kv_heads * KV_STRIDE * head_dim, 0x7310);
            let control_context = run_test_attention_context(
                kernel,
                pipelines.selected_attention_context_bf16(false),
                &probabilities,
                &values,
                N_HEADS,
                head_dim,
                logical_k,
                group,
                head_dim,
                KV_STRIDE * head_dim,
                0,
                0,
                logical_k,
            );
            let lattice_context = run_test_attention_context(
                kernel,
                pipelines.selected_attention_context_bf16(true),
                &probabilities,
                &values,
                N_HEADS,
                head_dim,
                logical_k,
                group,
                head_dim,
                KV_STRIDE * head_dim,
                0,
                0,
                logical_k,
            );
            assert_eq!(
                raw_f32_bits(&lattice_context),
                raw_f32_bits(&control_context),
                "{name} context direct-probability load drifted"
            );
        }
    }

    #[test]
    fn bf16_lattice_loads_preserve_shared_buffer_alias_and_sentinel_guards() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        let pipelines = MtpPipelines::new(&kernel.device).expect("compile MTP test kernels");
        const HEAD_DIM: usize = 32;
        const POSITIONS: usize = 3;
        const POISON_BITS: u32 = 0x7fc1_2345;
        let poison = f32::from_bits(POISON_BITS);

        let q_start = 4usize;
        let k_start = q_start + HEAD_DIM + 4;
        let score_start = k_start + POSITIONS * HEAD_DIM + 4;
        let qk_len = score_start + POSITIONS + 4;
        let mut qk_backing = vec![poison; qk_len];
        qk_backing[q_start..q_start + HEAD_DIM]
            .copy_from_slice(&bf16_lattice_test_values(HEAD_DIM, 0x8110, false));
        qk_backing[k_start..k_start + POSITIONS * HEAD_DIM]
            .copy_from_slice(&dirty_low16_test_values(POSITIONS * HEAD_DIM, 0x8220));
        let qk_scalar = test_attention_scalar(
            &kernel.device,
            1,
            HEAD_DIM,
            POSITIONS,
            1,
            POSITIONS,
            0,
            POSITIONS,
        );
        let run_qk = |pipeline: &ComputePipelineState| {
            let backing = f32_buffer(&kernel.device, &qk_backing);
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&backing), (q_start * 4) as u64);
            encoder.set_buffer(1, Some(&backing), (k_start * 4) as u64);
            encoder.set_buffer(2, Some(&backing), (score_start * 4) as u64);
            for index in 0..8u64 {
                encoder.set_buffer(3 + index, Some(&qk_scalar), index * 4);
            }
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
            let mut output = vec![0.0f32; qk_len];
            read_buffer_f32(&backing, &mut output);
            output
        };
        let qk_control = run_qk(&pipelines.attention_scores_bf16);
        let qk_lattice = run_qk(&pipelines.attention_scores_bf16_lattice_query);
        assert_eq!(raw_f32_bits(&qk_lattice), raw_f32_bits(&qk_control));
        for index in 0..qk_len {
            if !(score_start..score_start + POSITIONS).contains(&index) {
                assert_eq!(qk_lattice[index].to_bits(), qk_backing[index].to_bits());
            }
        }
        assert!(qk_lattice[score_start..score_start + POSITIONS]
            .iter()
            .any(|value| value.to_bits() != POISON_BITS));

        let value_start = 4usize;
        let probability_start = value_start + POSITIONS * HEAD_DIM + 4;
        let output_start = probability_start + POSITIONS + 5;
        let context_len = output_start + HEAD_DIM + 4;
        let mut context_backing = vec![poison; context_len];
        context_backing[value_start..value_start + POSITIONS * HEAD_DIM]
            .copy_from_slice(&dirty_low16_test_values(POSITIONS * HEAD_DIM, 0x8330));
        context_backing[probability_start..probability_start + POSITIONS]
            .copy_from_slice(&bf16_lattice_test_values(POSITIONS, 0x8440, true));
        let context_scalar = test_attention_scalar(
            &kernel.device,
            1,
            HEAD_DIM,
            POSITIONS,
            1,
            POSITIONS,
            0,
            POSITIONS,
        );
        let run_context = |pipeline: &ComputePipelineState| {
            let backing = f32_buffer(&kernel.device, &context_backing);
            let command_buffer = kernel.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&backing), (value_start * 4) as u64);
            encoder.set_buffer(
                1,
                Some(&backing),
                (probability_start * std::mem::size_of::<f32>()) as u64,
            );
            encoder.set_buffer(2, Some(&backing), (output_start * 4) as u64);
            encoder.set_buffer(3, Some(&context_scalar), 0);
            encoder.set_buffer(4, Some(&context_scalar), 4);
            encoder.set_buffer(5, Some(&context_scalar), 8);
            encoder.set_buffer(6, Some(&context_scalar), 12);
            encoder.set_buffer(7, Some(&context_scalar), 20);
            encoder.set_buffer(8, Some(&context_scalar), 24);
            encoder.set_buffer(9, Some(&context_scalar), 28);
            encoder.set_buffer(10, Some(&context_scalar), 32);
            encoder.set_buffer(11, Some(&context_scalar), 36);
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
            let mut output = vec![0.0f32; context_len];
            read_buffer_f32(&backing, &mut output);
            output
        };
        let context_control = run_context(&pipelines.attention_context_bf16);
        let context_lattice = run_context(&pipelines.attention_context_bf16_lattice_probabilities);
        assert_eq!(
            raw_f32_bits(&context_lattice),
            raw_f32_bits(&context_control)
        );
        for index in 0..context_len {
            if !(output_start..output_start + HEAD_DIM).contains(&index) {
                assert_eq!(
                    context_lattice[index].to_bits(),
                    context_backing[index].to_bits()
                );
            }
        }
        assert!(context_lattice[output_start..output_start + HEAD_DIM]
            .iter()
            .any(|value| value.to_bits() != POISON_BITS));
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
                &pipelines.attention_context_bf16_lattice_probabilities,
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
                &pipelines.attention_context_bf16_lattice_probabilities,
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
