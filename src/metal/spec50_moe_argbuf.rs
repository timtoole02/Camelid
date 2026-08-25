//! Tier-2 argument-buffer variants of the fixed-geometry Gemma 4 expert
//! kernels.
//!
//! The only shader change is expert-address selection. The copied-slab kernels
//! derive a record address from `(slab, expert_weight_offset)`; these variants
//! use the same offset to index a static 128-pointer argument buffer. Every
//! integer dot, floating-point expression, loop, lane partition, and reduction
//! following that selection is kept textually identical to `spec50_moe.rs`.
//!
//! The production API below builds the same table from persistent anonymous
//! per-slot buffers and declares only the exact active slot union. The original
//! file-backed experiment remains as falsification evidence; its ignored
//! real-model tests pin the address-selection and paging behavior.

#![allow(dead_code)]

use super::*;

use metal::{
    Buffer, ComputePipelineState, Device, Function, MTLArgumentBuffersTier, MTLResourceOptions,
    MTLResourceUsage,
};
use std::sync::OnceLock;

use super::spec50_moe::{
    S50_DOWN_ROWS_PER_SIMDGROUP, S50_FF, S50_GU_BLOCKS, S50_HIDDEN, S50_RECORD_BYTES,
    S50_SLOT_STRIDE,
};

const SPEC50_MOE_ARGBUF_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define G4Q4_HIDDEN 2816u
#define G4Q4_FF 704u
#define G4Q4_ROUTES 8u
#define G4Q4_GU_BLOCKS 88u
#define G4Q4_DOWN_BLOCKS 22u
#define G4Q4_WIRE 18ul
#define G4Q4_GU_ROW_BYTES 1584ul
#define G4Q4_DOWN_ROW_BYTES 396ul
#define G4Q4_GATE_UP_BYTES 2230272ul
#define G4Q4_SLOT_STRIDE 3358720u
#define S50_DOWN_TERMS 6u
#define S50_DOWN_ROWS 4u

struct Gemma4UniqueExpertWork {
    ulong candidate_mask;
    uint expert_weight_offset;
    uint slab_index;
};

struct Gemma4CandidateRouteEntry {
    uint unique_expert_idx;
    float weight;
};

struct Gemma4ExpertPointerTable {
    array<device const uchar *, 128> records [[id(0)]];
};

kernel void spec50_moe_argbuf_gateup_geglu_quant_batch_k(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint b = group % G4Q4_DOWN_BLOCKS;
    const uint u = group / G4Q4_DOWN_BLOCKS;
    if (u >= num_unique_experts) return;
    if (k_candidates == 0u || k_candidates > 16u) return;

    const Gemma4UniqueExpertWork work = work_list[u];
    const ulong mask = work.candidate_mask;
    if (mask == 0ULL) return;

    // The address-selection seam is the sole arithmetic-program difference.
    // `expert_weight_offset` remains the existing fixed-stride expert ID.
    const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
    device const uchar* weights = expert_table.records[expert_id];
    const uint row = b * 32u + lane;
    device const uchar* gate_row = weights + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

    float gate_acc[16];
    float up_acc[16];
    #pragma unroll
    for (uint t = 0; t < 16; ++t) {
        gate_acc[t] = 0.0f;
        up_acc[t] = 0.0f;
    }

    for (uint gb = 0; gb < G4Q4_GU_BLOCKS; ++gb) {
        device const uchar* b_gate = gate_row + ulong(gb) * G4Q4_WIRE;
        device const uchar* b_up = up_row + ulong(gb) * G4Q4_WIRE;
        const float w_scale_gate = float(*reinterpret_cast<device const half*>(b_gate));
        const float w_scale_up = float(*reinterpret_cast<device const half*>(b_up));

        device const packed_uchar4* pg4 =
            reinterpret_cast<device const packed_uchar4*>(b_gate + 2);
        device const packed_uchar4* pu4 =
            reinterpret_cast<device const packed_uchar4*>(b_up + 2);

        uchar4 rg[4], ru[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            rg[k] = uchar4(pg4[k]);
            ru[k] = uchar4(pu4[k]);
        }

        #pragma unroll
        for (uint t = 0; t < 16; ++t) {
            if (t >= k_candidates) continue;
            if ((mask & (1ULL << t)) == 0ULL) continue;
            device const char* x = input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
            device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
            device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
            const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

            int4 ag = int4(0);
            int4 au = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const int4 xl = int4(xlo4[k]);
                const int4 xh = int4(xhi4[k]);
                ag += (int4(rg[k] & uchar4(0x0f)) - 8) * xl + (int4(rg[k] >> 4) - 8) * xh;
                au += (int4(ru[k] & uchar4(0x0f)) - 8) * xl + (int4(ru[k] >> 4) - 8) * xh;
            }
            const int isum_gate = (ag.x + ag.y) + (ag.z + ag.w);
            const int isum_up = (au.x + au.y) + (au.z + au.w);

            gate_acc[t] += (float(isum_gate) * w_scale_gate) * in_scale;
            up_acc[t] += (float(isum_up) * w_scale_up) * in_scale;
        }
    }

    #pragma unroll
    for (uint t = 0; t < 16; ++t) {
        if (t >= k_candidates) continue;
        if ((mask & (1ULL << t)) == 0ULL) continue;
        const float gate = gate_acc[t];
        const float up = up_acc[t];
        const float inner = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        const float gelu = 0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
        const float act_val = gelu * up;

        const float max_abs = simd_max(fabs(act_val));
        const float unrounded = max_abs / 127.0f;
        const float stored_scale = float(half(unrounded));
        const float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;

        if (lane == 0) {
            const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
                + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
            output_scales[scale_idx] = stored_scale;
        }

        const int q = clamp(int(round(act_val * inverse)), -127, 127);
        const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(row);
        output_quants[quant_idx] = char(q);
    }
}

// Exact K<=12 register-narrow twin of the general K=16 kernel. This keeps only
// the 24 floating-point accumulators that K=9..12 can address while preserving
// every load, integer fold, floating-point expression, and output index above.
kernel void spec50_moe_argbuf_gateup_geglu_quant_batch_k12(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint b = group % G4Q4_DOWN_BLOCKS;
    const uint u = group / G4Q4_DOWN_BLOCKS;
    if (u >= num_unique_experts) return;
    if (k_candidates == 0u || k_candidates > 12u) return;

    const Gemma4UniqueExpertWork work = work_list[u];
    const ulong mask = work.candidate_mask;
    if (mask == 0ULL) return;

    const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
    device const uchar* weights = expert_table.records[expert_id];
    const uint row = b * 32u + lane;
    device const uchar* gate_row = weights + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

    float gate_acc[12];
    float up_acc[12];
    #pragma unroll
    for (uint t = 0; t < 12; ++t) {
        gate_acc[t] = 0.0f;
        up_acc[t] = 0.0f;
    }

    for (uint gb = 0; gb < G4Q4_GU_BLOCKS; ++gb) {
        device const uchar* b_gate = gate_row + ulong(gb) * G4Q4_WIRE;
        device const uchar* b_up = up_row + ulong(gb) * G4Q4_WIRE;
        const float w_scale_gate = float(*reinterpret_cast<device const half*>(b_gate));
        const float w_scale_up = float(*reinterpret_cast<device const half*>(b_up));

        device const packed_uchar4* pg4 =
            reinterpret_cast<device const packed_uchar4*>(b_gate + 2);
        device const packed_uchar4* pu4 =
            reinterpret_cast<device const packed_uchar4*>(b_up + 2);

        uchar4 rg[4], ru[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            rg[k] = uchar4(pg4[k]);
            ru[k] = uchar4(pu4[k]);
        }

        #pragma unroll
        for (uint t = 0; t < 12; ++t) {
            if (t >= k_candidates) continue;
            if ((mask & (1ULL << t)) == 0ULL) continue;
            device const char* x = input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
            device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
            device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
            const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

            int4 ag = int4(0);
            int4 au = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const int4 xl = int4(xlo4[k]);
                const int4 xh = int4(xhi4[k]);
                ag += (int4(rg[k] & uchar4(0x0f)) - 8) * xl + (int4(rg[k] >> 4) - 8) * xh;
                au += (int4(ru[k] & uchar4(0x0f)) - 8) * xl + (int4(ru[k] >> 4) - 8) * xh;
            }
            const int isum_gate = (ag.x + ag.y) + (ag.z + ag.w);
            const int isum_up = (au.x + au.y) + (au.z + au.w);

            gate_acc[t] += (float(isum_gate) * w_scale_gate) * in_scale;
            up_acc[t] += (float(isum_up) * w_scale_up) * in_scale;
        }
    }

    #pragma unroll
    for (uint t = 0; t < 12; ++t) {
        if (t >= k_candidates) continue;
        if ((mask & (1ULL << t)) == 0ULL) continue;
        const float gate = gate_acc[t];
        const float up = up_acc[t];
        const float inner = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        const float gelu = 0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
        const float act_val = gelu * up;

        const float max_abs = simd_max(fabs(act_val));
        const float unrounded = max_abs / 127.0f;
        const float stored_scale = float(half(unrounded));
        const float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;

        if (lane == 0) {
            const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
                + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
            output_scales[scale_idx] = stored_scale;
        }

        const int q = clamp(int(round(act_val * inverse)), -127, 127);
        const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(row);
        output_quants[quant_idx] = char(q);
    }
}

// Preserve the original K<=8 register geometry after widening the general
// argument-buffer kernel to K=16.  The widened kernel keeps 32 floating-point
// accumulators live per lane (gate + up), even for the shipping K=8 round.
// This exact twin keeps only the 16 accumulators K=8 can address.  Every load,
// integer fold, floating-point expression, and output index is otherwise
// identical to the general kernel above.
kernel void spec50_moe_argbuf_gateup_geglu_quant_batch_k8(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint b = group % G4Q4_DOWN_BLOCKS;
    const uint u = group / G4Q4_DOWN_BLOCKS;
    if (u >= num_unique_experts) return;
    if (k_candidates == 0u || k_candidates > 8u) return;

    const Gemma4UniqueExpertWork work = work_list[u];
    const ulong mask = work.candidate_mask;
    if (mask == 0ULL) return;

    const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
    device const uchar* weights = expert_table.records[expert_id];
    const uint row = b * 32u + lane;
    device const uchar* gate_row = weights + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

    float gate_acc[8];
    float up_acc[8];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        gate_acc[t] = 0.0f;
        up_acc[t] = 0.0f;
    }

    for (uint gb = 0; gb < G4Q4_GU_BLOCKS; ++gb) {
        device const uchar* b_gate = gate_row + ulong(gb) * G4Q4_WIRE;
        device const uchar* b_up = up_row + ulong(gb) * G4Q4_WIRE;
        const float w_scale_gate = float(*reinterpret_cast<device const half*>(b_gate));
        const float w_scale_up = float(*reinterpret_cast<device const half*>(b_up));

        device const packed_uchar4* pg4 =
            reinterpret_cast<device const packed_uchar4*>(b_gate + 2);
        device const packed_uchar4* pu4 =
            reinterpret_cast<device const packed_uchar4*>(b_up + 2);

        uchar4 rg[4], ru[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            rg[k] = uchar4(pg4[k]);
            ru[k] = uchar4(pu4[k]);
        }

        #pragma unroll
        for (uint t = 0; t < 8; ++t) {
            if (t >= k_candidates) continue;
            if ((mask & (1ULL << t)) == 0ULL) continue;
            device const char* x = input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
            device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
            device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
            const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

            int4 ag = int4(0);
            int4 au = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const int4 xl = int4(xlo4[k]);
                const int4 xh = int4(xhi4[k]);
                ag += (int4(rg[k] & uchar4(0x0f)) - 8) * xl + (int4(rg[k] >> 4) - 8) * xh;
                au += (int4(ru[k] & uchar4(0x0f)) - 8) * xl + (int4(ru[k] >> 4) - 8) * xh;
            }
            const int isum_gate = (ag.x + ag.y) + (ag.z + ag.w);
            const int isum_up = (au.x + au.y) + (au.z + au.w);

            gate_acc[t] += (float(isum_gate) * w_scale_gate) * in_scale;
            up_acc[t] += (float(isum_up) * w_scale_up) * in_scale;
        }
    }

    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        if (t >= k_candidates) continue;
        if ((mask & (1ULL << t)) == 0ULL) continue;
        const float gate = gate_acc[t];
        const float up = up_acc[t];
        const float inner = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        const float gelu = 0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
        const float act_val = gelu * up;

        const float max_abs = simd_max(fabs(act_val));
        const float unrounded = max_abs / 127.0f;
        const float stored_scale = float(half(unrounded));
        const float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;

        if (lane == 0) {
            const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
                + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
            output_scales[scale_idx] = stored_scale;
        }

        const int q = clamp(int(round(act_val * inverse)), -127, 127);
        const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(row);
        output_quants[quant_idx] = char(q);
    }
}

// Exact low-register-pressure K<=8 twin. One threadgroup owns one
// (unique expert, candidate, 32-row block), so each lane keeps only the gate
// and up accumulator for that candidate live. Candidate-local arithmetic and
// accumulation order are identical to the shipping K8 kernel; only the grid
// partition changes. Inactive (expert, candidate) pairs return before touching
// expert pages. This deliberately trades repeated warm weight reads for lower
// register pressure. It remains benchmark-only because the M4 result did not
// beat the current K8 kernel materially.
kernel void spec50_moe_argbuf_gateup_geglu_quant_batch_k8_candidate(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (k_candidates == 0u || k_candidates > 8u) return;
    const uint b = group % G4Q4_DOWN_BLOCKS;
    const uint pair = group / G4Q4_DOWN_BLOCKS;
    const uint t = pair % k_candidates;
    const uint u = pair / k_candidates;
    if (u >= num_unique_experts) return;

    const Gemma4UniqueExpertWork work = work_list[u];
    if ((work.candidate_mask & (1ULL << t)) == 0ULL) return;

    const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
    device const uchar* weights = expert_table.records[expert_id];
    const uint row = b * 32u + lane;
    device const uchar* gate_row = weights + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint gb = 0; gb < G4Q4_GU_BLOCKS; ++gb) {
        device const uchar* b_gate = gate_row + ulong(gb) * G4Q4_WIRE;
        device const uchar* b_up = up_row + ulong(gb) * G4Q4_WIRE;
        const float w_scale_gate = float(*reinterpret_cast<device const half*>(b_gate));
        const float w_scale_up = float(*reinterpret_cast<device const half*>(b_up));

        device const packed_uchar4* pg4 =
            reinterpret_cast<device const packed_uchar4*>(b_gate + 2);
        device const packed_uchar4* pu4 =
            reinterpret_cast<device const packed_uchar4*>(b_up + 2);

        uchar4 rg[4], ru[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            rg[k] = uchar4(pg4[k]);
            ru[k] = uchar4(pu4[k]);
        }

        device const char* x =
            input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
        device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
        device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
        const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

        int4 ag = int4(0);
        int4 au = int4(0);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            const int4 xl = int4(xlo4[k]);
            const int4 xh = int4(xhi4[k]);
            ag += (int4(rg[k] & uchar4(0x0f)) - 8) * xl
                + (int4(rg[k] >> 4) - 8) * xh;
            au += (int4(ru[k] & uchar4(0x0f)) - 8) * xl
                + (int4(ru[k] >> 4) - 8) * xh;
        }
        const int isum_gate = (ag.x + ag.y) + (ag.z + ag.w);
        const int isum_up = (au.x + au.y) + (au.z + au.w);

        gate_acc += (float(isum_gate) * w_scale_gate) * in_scale;
        up_acc += (float(isum_up) * w_scale_up) * in_scale;
    }

    const float gate = gate_acc;
    const float up = up_acc;
    const float inner = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
    const float gelu = 0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
    const float act_val = gelu * up;

    const float max_abs = simd_max(fabs(act_val));
    const float unrounded = max_abs / 127.0f;
    const float stored_scale = float(half(unrounded));
    const float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;

    if (lane == 0) {
        const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
            + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        output_scales[scale_idx] = stored_scale;
    }

    const int q = clamp(int(round(act_val * inverse)), -127, 127);
    const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF
        + ulong(t) * G4Q4_FF + ulong(row);
    output_quants[quant_idx] = char(q);
}

// Exact four-candidate tiled twin. Compared with the current K8 kernel, each
// lane keeps eight rather than sixteen floating accumulators live. Compared
// with the one-candidate twin, it reuses each resident weight row across up to
// four candidates. Candidate-local accumulation order remains unchanged.
kernel void spec50_moe_argbuf_gateup_geglu_quant_batch_k8_tile4(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (k_candidates == 0u || k_candidates > 8u) return;
    const uint tiles = (k_candidates + 3u) / 4u;
    const uint b = group % G4Q4_DOWN_BLOCKS;
    const uint packed = group / G4Q4_DOWN_BLOCKS;
    const uint tile = packed % tiles;
    const uint u = packed / tiles;
    if (u >= num_unique_experts) return;

    const Gemma4UniqueExpertWork work = work_list[u];
    const uint first_t = tile * 4u;
    const ulong tile_mask = 0xFULL << first_t;
    if ((work.candidate_mask & tile_mask) == 0ULL) return;

    const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
    device const uchar* weights = expert_table.records[expert_id];
    const uint row = b * 32u + lane;
    device const uchar* gate_row = weights + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

    float gate_acc[4];
    float up_acc[4];
    #pragma unroll
    for (uint local_t = 0; local_t < 4; ++local_t) {
        gate_acc[local_t] = 0.0f;
        up_acc[local_t] = 0.0f;
    }

    for (uint gb = 0; gb < G4Q4_GU_BLOCKS; ++gb) {
        device const uchar* b_gate = gate_row + ulong(gb) * G4Q4_WIRE;
        device const uchar* b_up = up_row + ulong(gb) * G4Q4_WIRE;
        const float w_scale_gate = float(*reinterpret_cast<device const half*>(b_gate));
        const float w_scale_up = float(*reinterpret_cast<device const half*>(b_up));

        device const packed_uchar4* pg4 =
            reinterpret_cast<device const packed_uchar4*>(b_gate + 2);
        device const packed_uchar4* pu4 =
            reinterpret_cast<device const packed_uchar4*>(b_up + 2);

        uchar4 rg[4], ru[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            rg[k] = uchar4(pg4[k]);
            ru[k] = uchar4(pu4[k]);
        }

        #pragma unroll
        for (uint local_t = 0; local_t < 4; ++local_t) {
            const uint t = first_t + local_t;
            if (t >= k_candidates) continue;
            if ((work.candidate_mask & (1ULL << t)) == 0ULL) continue;
            device const char* x =
                input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
            device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
            device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
            const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

            int4 ag = int4(0);
            int4 au = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const int4 xl = int4(xlo4[k]);
                const int4 xh = int4(xhi4[k]);
                ag += (int4(rg[k] & uchar4(0x0f)) - 8) * xl
                    + (int4(rg[k] >> 4) - 8) * xh;
                au += (int4(ru[k] & uchar4(0x0f)) - 8) * xl
                    + (int4(ru[k] >> 4) - 8) * xh;
            }
            const int isum_gate = (ag.x + ag.y) + (ag.z + ag.w);
            const int isum_up = (au.x + au.y) + (au.z + au.w);

            gate_acc[local_t] += (float(isum_gate) * w_scale_gate) * in_scale;
            up_acc[local_t] += (float(isum_up) * w_scale_up) * in_scale;
        }
    }

    #pragma unroll
    for (uint local_t = 0; local_t < 4; ++local_t) {
        const uint t = first_t + local_t;
        if (t >= k_candidates) continue;
        if ((work.candidate_mask & (1ULL << t)) == 0ULL) continue;
        const float gate = gate_acc[local_t];
        const float up = up_acc[local_t];
        const float inner = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        const float gelu = 0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
        const float act_val = gelu * up;

        const float max_abs = simd_max(fabs(act_val));
        const float unrounded = max_abs / 127.0f;
        const float stored_scale = float(half(unrounded));
        const float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;

        if (lane == 0) {
            const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
                + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
            output_scales[scale_idx] = stored_scale;
        }

        const int q = clamp(int(round(act_val * inverse)), -127, 127);
        const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(row);
        output_quants[quant_idx] = char(q);
    }
}

kernel void spec50_moe_argbuf_down_union_batch_k(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint t = sg;
    if (t >= k_candidates) return;

    const uint row0 = group * S50_DOWN_ROWS;

    float lane_total[S50_DOWN_ROWS];
    #pragma unroll
    for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
        lane_total[r] = 0.0f;
    }

    #pragma unroll
    for (uint jj = 0; jj < S50_DOWN_TERMS; ++jj) {
        const uint flat = lane + jj * 32u;
        if (flat >= G4Q4_ROUTES * G4Q4_DOWN_BLOCKS) continue;
        const uint slot = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - slot * G4Q4_DOWN_BLOCKS;
        const Gemma4CandidateRouteEntry route = candidate_routes[t * G4Q4_ROUTES + slot];
        if (route.weight == 0.0f || route.unique_expert_idx >= 128u) continue;
        const uint u = route.unique_expert_idx;
        const Gemma4UniqueExpertWork work = work_list[u];
        // Only expert-address selection changes from the copied-slab kernel.
        const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
        device const uchar* weights = expert_table.records[expert_id];
        device const uchar* down_rows = weights + G4Q4_GATE_UP_BYTES
            + ulong(row0) * G4Q4_DOWN_ROW_BYTES;

        const ulong act_quant_base = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(b) * 32ul;
        device const char* x = act_quants + act_quant_base;
        device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
        device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
        int4 xl4[4], xh4[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            xl4[k] = int4(xlo4[k]);
            xh4[k] = int4(xhi4[k]);
        }
        const ulong act_scale_base = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
            + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        const float act_scale = act_scales[act_scale_base];

        #pragma unroll
        for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
            device const uchar* block =
                down_rows + ulong(r) * G4Q4_DOWN_ROW_BYTES + ulong(b) * G4Q4_WIRE;
            const float weight_scale = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* wq =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            int4 a4 = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const uchar4 wb = uchar4(wq[k]);
                a4 += (int4(wb & uchar4(0x0f)) - 8) * xl4[k]
                    + (int4(wb >> 4) - 8) * xh4[k];
            }
            const int isum = (a4.x + a4.y) + (a4.z + a4.w);
            const float term_scale = (weight_scale * act_scale) * route.weight;
            lane_total[r] += float(isum) * term_scale;
        }
    }

    #pragma unroll
    for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
        const float total = simd_sum(lane_total[r]);
        if (lane == 0) {
            output_moe_acc[ulong(t) * G4Q4_HIDDEN + ulong(row0 + r)] = total;
        }
    }
}

// Exact row-reuse probe. Eight output rows share each route lookup and
// activation block load instead of four; every row retains the same lane term
// order and simd_sum reduction as the current kernel.
kernel void spec50_moe_argbuf_down_union_batch_k_rows8(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint t = sg;
    if (t >= k_candidates) return;

    const uint row0 = group * 8u;

    float lane_total[8];
    #pragma unroll
    for (uint r = 0; r < 8u; ++r) {
        lane_total[r] = 0.0f;
    }

    #pragma unroll
    for (uint jj = 0; jj < S50_DOWN_TERMS; ++jj) {
        const uint flat = lane + jj * 32u;
        if (flat >= G4Q4_ROUTES * G4Q4_DOWN_BLOCKS) continue;
        const uint slot = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - slot * G4Q4_DOWN_BLOCKS;
        const Gemma4CandidateRouteEntry route = candidate_routes[t * G4Q4_ROUTES + slot];
        if (route.weight == 0.0f || route.unique_expert_idx >= 128u) continue;
        const uint u = route.unique_expert_idx;
        const Gemma4UniqueExpertWork work = work_list[u];
        const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
        device const uchar* weights = expert_table.records[expert_id];
        device const uchar* down_rows = weights + G4Q4_GATE_UP_BYTES
            + ulong(row0) * G4Q4_DOWN_ROW_BYTES;

        const ulong act_quant_base = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(b) * 32ul;
        device const char* x = act_quants + act_quant_base;
        device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
        device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
        int4 xl4[4], xh4[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            xl4[k] = int4(xlo4[k]);
            xh4[k] = int4(xhi4[k]);
        }
        const ulong act_scale_base = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
            + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        const float act_scale = act_scales[act_scale_base];

        #pragma unroll
        for (uint r = 0; r < 8u; ++r) {
            device const uchar* block =
                down_rows + ulong(r) * G4Q4_DOWN_ROW_BYTES + ulong(b) * G4Q4_WIRE;
            const float weight_scale = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* wq =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            int4 a4 = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const uchar4 wb = uchar4(wq[k]);
                a4 += (int4(wb & uchar4(0x0f)) - 8) * xl4[k]
                    + (int4(wb >> 4) - 8) * xh4[k];
            }
            const int isum = (a4.x + a4.y) + (a4.z + a4.w);
            const float term_scale = (weight_scale * act_scale) * route.weight;
            lane_total[r] += float(isum) * term_scale;
        }
    }

    #pragma unroll
    for (uint r = 0; r < 8u; ++r) {
        const float total = simd_sum(lane_total[r]);
        if (lane == 0) {
            output_moe_acc[ulong(t) * G4Q4_HIDDEN + ulong(row0 + r)] = total;
        }
    }
}

// Exact threadgroup-pressure probe. The current kernel places all K<=8
// candidate SIMD groups in one threadgroup. This twin caps each threadgroup at
// four candidates and tiles the candidate dimension across the grid. Per-row,
// per-candidate lane ownership and reduction order are unchanged.
kernel void spec50_moe_argbuf_down_union_batch_k_tg4(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row_groups = G4Q4_HIDDEN / S50_DOWN_ROWS;
    const uint row_group = group % row_groups;
    const uint candidate_tile = group / row_groups;
    const uint t = candidate_tile * 4u + sg;
    if (t >= k_candidates) return;

    const uint row0 = row_group * S50_DOWN_ROWS;

    float lane_total[S50_DOWN_ROWS];
    #pragma unroll
    for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
        lane_total[r] = 0.0f;
    }

    #pragma unroll
    for (uint jj = 0; jj < S50_DOWN_TERMS; ++jj) {
        const uint flat = lane + jj * 32u;
        if (flat >= G4Q4_ROUTES * G4Q4_DOWN_BLOCKS) continue;
        const uint slot = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - slot * G4Q4_DOWN_BLOCKS;
        const Gemma4CandidateRouteEntry route = candidate_routes[t * G4Q4_ROUTES + slot];
        if (route.weight == 0.0f || route.unique_expert_idx >= 128u) continue;
        const uint u = route.unique_expert_idx;
        const Gemma4UniqueExpertWork work = work_list[u];
        const uint expert_id = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
        device const uchar* weights = expert_table.records[expert_id];
        device const uchar* down_rows = weights + G4Q4_GATE_UP_BYTES
            + ulong(row0) * G4Q4_DOWN_ROW_BYTES;

        const ulong act_quant_base = ulong(u) * ulong(k_candidates) * G4Q4_FF
            + ulong(t) * G4Q4_FF + ulong(b) * 32ul;
        device const char* x = act_quants + act_quant_base;
        device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
        device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
        int4 xl4[4], xh4[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            xl4[k] = int4(xlo4[k]);
            xh4[k] = int4(xhi4[k]);
        }
        const ulong act_scale_base = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS
            + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        const float act_scale = act_scales[act_scale_base];

        #pragma unroll
        for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
            device const uchar* block =
                down_rows + ulong(r) * G4Q4_DOWN_ROW_BYTES + ulong(b) * G4Q4_WIRE;
            const float weight_scale = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* wq =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            int4 a4 = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const uchar4 wb = uchar4(wq[k]);
                a4 += (int4(wb & uchar4(0x0f)) - 8) * xl4[k]
                    + (int4(wb >> 4) - 8) * xh4[k];
            }
            const int isum = (a4.x + a4.y) + (a4.z + a4.w);
            const float term_scale = (weight_scale * act_scale) * route.weight;
            lane_total[r] += float(isum) * term_scale;
        }
    }

    #pragma unroll
    for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
        const float total = simd_sum(lane_total[r]);
        if (lane == 0) {
            output_moe_acc[ulong(t) * G4Q4_HIDDEN + ulong(row0 + r)] = total;
        }
    }
}

// HEAD K=1 twins. Every operation after selecting the record pointer is kept
// textually identical to the shipping copied-slab kernels in metal.rs.
inline float gemma4_q4_expert_argbuf_row_dot(
    device const uchar* weight_row,
    device const float* input_scales,
    device const char* input_quants,
    uint blocks
) {
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* block = weight_row + ulong(b) * G4Q4_WIRE;
        const float weight_scale =
            float(*reinterpret_cast<device const half*>(block));
        device const char* x = input_quants + ulong(b) * 32ul;
        // The 16 nibble bytes start 2 bytes into an 18-byte block, so the run is
        // only 2-byte aligned: packed_uchar4 (alignment 1) is the legal wide load
        // here, exactly as the GateUp path above already does. Four of these
        // replace 16 scalar byte loads per block. `isum` is an INTEGER
        // accumulator, and integer addition is exact and associative, so the
        // regrouping is bit-identical; the float terms are untouched.
        int isum = 0;
        {
            device const packed_uchar4* p4 =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            const uchar4 w0 = uchar4(p4[0]);
            const uchar4 w1 = uchar4(p4[1]);
            const uchar4 w2 = uchar4(p4[2]);
            const uchar4 w3 = uchar4(p4[3]);
            const uchar wq[16] = {w0.x, w0.y, w0.z, w0.w,
                                  w1.x, w1.y, w1.z, w1.w,
                                  w2.x, w2.y, w2.z, w2.w,
                                  w3.x, w3.y, w3.z, w3.w};
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const uint packed = uint(wq[j]);
                const int lo = int(packed & 0x0fu) - 8;
                const int hi = int(packed >> 4) - 8;
                isum += lo * int(x[j]);
                isum += hi * int(x[j + 16]);
            }
        }
        const float term =
            (float(isum) * weight_scale) * input_scales[b];
        acc = acc + term;
    }
    return acc;
}

inline float gemma4_q4_expert_argbuf_block_term_simd(
    device const uchar* weight_row,
    device const float* input_scales,
    device const char* input_quants,
    uint b
) {
    device const uchar* block = weight_row + ulong(b) * G4Q4_WIRE;
    const float weight_scale =
        float(*reinterpret_cast<device const half*>(block));
    device const char* x = input_quants + ulong(b) * 32ul;
    int isum = 0;
    device const packed_uchar4* p4 =
        reinterpret_cast<device const packed_uchar4*>(block + 2);
    const uchar4 w0 = uchar4(p4[0]);
    const uchar4 w1 = uchar4(p4[1]);
    const uchar4 w2 = uchar4(p4[2]);
    const uchar4 w3 = uchar4(p4[3]);
    const uchar wq[16] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                          w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w};
    #pragma unroll
    for (uint j = 0; j < 16; ++j) {
        const uint packed = uint(wq[j]);
        const int lo = int(packed & 0x0fu) - 8;
        const int hi = int(packed >> 4) - 8;
        isum += lo * int(x[j]);
        isum += hi * int(x[j + 16]);
    }
    return (float(isum) * weight_scale) * input_scales[b];
}

kernel void gemma4_q4_expert_argbuf_gate_up_split(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device float* gate_up [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = G4Q4_ROUTES * G4Q4_FF;
    if (gid >= total) return;
    const uint route = gid / G4Q4_FF;
    const uint row = gid - route * G4Q4_FF;
    device const uchar* rows_base = expert_table.records[route_slots[route]];
    device const uchar* gate_row =
        rows_base + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = rows_base
        + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;
    const uint output_base = route * (2u * G4Q4_FF);
    gate_up[output_base + row] = gemma4_q4_expert_argbuf_row_dot(
        gate_row, input_scales, input_quants, G4Q4_GU_BLOCKS);
    gate_up[output_base + G4Q4_FF + row] = gemma4_q4_expert_argbuf_row_dot(
        up_row, input_scales, input_quants, G4Q4_GU_BLOCKS);
}

kernel void gemma4_q4_expert_argbuf_gate_up_geglu(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device float* activated [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = G4Q4_ROUTES * G4Q4_FF;
    if (gid >= total) return;
    const uint route = gid / G4Q4_FF;
    const uint row = gid - route * G4Q4_FF;
    device const uchar* rows_base = expert_table.records[route_slots[route]];
    device const uchar* gate_row =
        rows_base + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = rows_base
        + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;
    const float gate = gemma4_q4_expert_argbuf_row_dot(
        gate_row, input_scales, input_quants, G4Q4_GU_BLOCKS);
    const float up = gemma4_q4_expert_argbuf_row_dot(
        up_row, input_scales, input_quants, G4Q4_GU_BLOCKS);
    const float inner =
        0.7978845608f * (gate + 0.044715f * gate * gate * gate);
    const float gelu =
        0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
    activated[gid] = gelu * up;
}

kernel void gemma4_q4_expert_argbuf_gate_up_geglu_simd(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device float* activated [[buffer(4)]],
    threadgroup float* block_terms [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint total = G4Q4_ROUTES * G4Q4_FF;
    if (gid >= total) return;
    const uint route = gid / G4Q4_FF;
    const uint row = gid - route * G4Q4_FF;
    device const uchar* rows_base = expert_table.records[route_slots[route]];
    device const uchar* gate_row =
        rows_base + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = rows_base
        + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;
    for (uint b = lane; b < G4Q4_GU_BLOCKS; b += 32u) {
        block_terms[b] = gemma4_q4_expert_argbuf_block_term_simd(
            gate_row, input_scales, input_quants, b);
        block_terms[G4Q4_GU_BLOCKS + b] =
            gemma4_q4_expert_argbuf_block_term_simd(
                up_row, input_scales, input_quants, b);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float gate = 0.0f;
        float up = 0.0f;
        for (uint b = 0; b < G4Q4_GU_BLOCKS; ++b) {
            gate = gate + block_terms[b];
        }
        for (uint b = 0; b < G4Q4_GU_BLOCKS; ++b) {
            up = up + block_terms[G4Q4_GU_BLOCKS + b];
        }
        const float inner =
            0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        const float gelu =
            0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
        activated[gid] = gelu * up;
    }
}

kernel void gemma4_q4_expert_argbuf_gate_up_geglu_turbo(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device float* activated [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint total = G4Q4_ROUTES * G4Q4_FF;
    const uint gid0 = (group * 4u + sgitg) * 4u;
    if (gid0 >= total) return;
    const uint route = gid0 / G4Q4_FF;
    const uint row0 = gid0 - route * G4Q4_FF;
    device const uchar* rows_base = expert_table.records[route_slots[route]];
    float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane; b < G4Q4_GU_BLOCKS; b += 32u) {
        for (uint r = 0; r < 4u; ++r) {
            gate_acc[r] += gemma4_q4_expert_argbuf_block_term_simd(
                rows_base + ulong(row0 + r) * G4Q4_GU_ROW_BYTES,
                input_scales, input_quants, b);
            up_acc[r] += gemma4_q4_expert_argbuf_block_term_simd(
                rows_base + ulong(row0 + r + G4Q4_FF) * G4Q4_GU_ROW_BYTES,
                input_scales, input_quants, b);
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        const float gate = simd_sum(gate_acc[r]);
        const float up = simd_sum(up_acc[r]);
        if (lane == 0) {
            const float inner =
                0.7978845608f * (gate + 0.044715f * gate * gate * gate);
            const float gelu =
                0.5f * gate * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
            activated[gid0 + r] = gelu * up;
        }
    }
}

kernel void gemma4_q4_expert_argbuf_down_reduce(
    device const float* activation_scales [[buffer(0)]],
    device const char* activation_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device const float* route_scales [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= G4Q4_HIDDEN) return;
    float total = 0.0f;
    for (uint route = 0; route < G4Q4_ROUTES; ++route) {
        device const uchar* rows_base = expert_table.records[route_slots[route]];
        device const uchar* down_row = rows_base
            + G4Q4_GATE_UP_BYTES + ulong(row) * G4Q4_DOWN_ROW_BYTES;
        const float y = gemma4_q4_expert_argbuf_row_dot(
            down_row,
            activation_scales + route * G4Q4_DOWN_BLOCKS,
            activation_quants + route * G4Q4_FF,
            G4Q4_DOWN_BLOCKS);
        const float weighted = y * route_scales[route];
        total = total + weighted;
    }
    output[row] = total;
}

kernel void gemma4_q4_expert_argbuf_down_reduce_simd(
    device const float* activation_scales [[buffer(0)]],
    device const char* activation_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device const float* route_scales [[buffer(4)]],
    device float* output [[buffer(5)]],
    threadgroup float* block_terms [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= G4Q4_HIDDEN) return;
    const uint terms = G4Q4_ROUTES * G4Q4_DOWN_BLOCKS;
    for (uint flat = lane; flat < terms; flat += 32u) {
        const uint route = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - route * G4Q4_DOWN_BLOCKS;
        device const uchar* rows_base = expert_table.records[route_slots[route]];
        device const uchar* down_row = rows_base
            + G4Q4_GATE_UP_BYTES + ulong(row) * G4Q4_DOWN_ROW_BYTES;
        device const uchar* block = down_row + ulong(b) * G4Q4_WIRE;
        const float weight_scale =
            float(*reinterpret_cast<device const half*>(block));
        device const char* x = activation_quants
            + route * G4Q4_FF + b * 32u;
        int isum = 0;
        // packed_uchar4 (alignment 1) is the legal wide load for a run that starts
        // 2 bytes into an 18-byte block; 4 loads replace 16 scalar byte loads.
        // isum is an INTEGER accumulator, so regrouping is bit-identical.
        device const packed_uchar4* dp4 =
            reinterpret_cast<device const packed_uchar4*>(block + 2);
        const uchar4 d0 = uchar4(dp4[0]);
        const uchar4 d1 = uchar4(dp4[1]);
        const uchar4 d2 = uchar4(dp4[2]);
        const uchar4 d3 = uchar4(dp4[3]);
        const uchar dq[16] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w,
                              d2.x, d2.y, d2.z, d2.w, d3.x, d3.y, d3.z, d3.w};
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            const uint packed = uint(dq[j]);
            const int lo = int(packed & 0x0fu) - 8;
            const int hi = int(packed >> 4) - 8;
            isum += lo * int(x[j]);
            isum += hi * int(x[j + 16]);
        }
        block_terms[flat] = (float(isum) * weight_scale)
            * activation_scales[route * G4Q4_DOWN_BLOCKS + b];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0.0f;
        for (uint route = 0; route < G4Q4_ROUTES; ++route) {
            float y = 0.0f;
            for (uint b = 0; b < G4Q4_DOWN_BLOCKS; ++b) {
                y = y + block_terms[route * G4Q4_DOWN_BLOCKS + b];
            }
            const float weighted = y * route_scales[route];
            total = total + weighted;
        }
        output[row] = total;
    }
}

kernel void gemma4_q4_expert_argbuf_down_reduce_turbo(
    device const float* activation_scales [[buffer(0)]],
    device const char* activation_quants [[buffer(1)]],
    device Gemma4ExpertPointerTable& expert_table [[buffer(2)]],
    device const uint* route_slots [[buffer(3)]],
    device const float* route_scales [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (group * 4u + sgitg) * 4u;
    if (row0 >= G4Q4_HIDDEN) return;
    const uint terms = G4Q4_ROUTES * G4Q4_DOWN_BLOCKS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint flat = lane; flat < terms; flat += 32u) {
        const uint route = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - route * G4Q4_DOWN_BLOCKS;
        device const uchar* rows_base = expert_table.records[route_slots[route]]
            + G4Q4_GATE_UP_BYTES + ulong(b) * G4Q4_WIRE;
        device const char* x = activation_quants
            + route * G4Q4_FF + b * 32u;
        const float block_scale = activation_scales[route * G4Q4_DOWN_BLOCKS + b]
            * route_scales[route];
        for (uint r = 0; r < 4u; ++r) {
            device const uchar* block =
                rows_base + ulong(row0 + r) * G4Q4_DOWN_ROW_BYTES;
            const float weight_scale =
                float(*reinterpret_cast<device const half*>(block));
            int isum = 0;
            // packed_uchar4 (alignment 1) is the legal wide load for a run that starts
            // 2 bytes into an 18-byte block; 4 loads replace 16 scalar byte loads.
            // isum is an INTEGER accumulator, so regrouping is bit-identical.
            device const packed_uchar4* dp4 =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            const uchar4 d0 = uchar4(dp4[0]);
            const uchar4 d1 = uchar4(dp4[1]);
            const uchar4 d2 = uchar4(dp4[2]);
            const uchar4 d3 = uchar4(dp4[3]);
            const uchar dq[16] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w,
                                  d2.x, d2.y, d2.z, d2.w, d3.x, d3.y, d3.z, d3.w};
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const uint packed = uint(dq[j]);
                const int lo = int(packed & 0x0fu) - 8;
                const int hi = int(packed >> 4) - 8;
                isum += lo * int(x[j]);
                isum += hi * int(x[j + 16]);
            }
            acc[r] += (float(isum) * weight_scale) * block_scale;
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        const float total = simd_sum(acc[r]);
        if (lane == 0) output[row0 + r] = total;
    }
}
"#;

pub(crate) struct Spec50MoeArgbufKernels {
    pub(crate) gateup: ComputePipelineState,
    /// Register-narrow K<=12 twin. It is optional and default-off so a Metal
    /// compiler or GPU that rejects it retains the proven K=16 pipeline.
    gateup_k12: Option<ComputePipelineState>,
    /// Register-narrow K<=8 twin. Optional so an older Metal compiler can
    /// retain the proven K=16 pipeline instead of failing construction.
    gateup_k8: Option<ComputePipelineState>,
    /// Exact candidate-sliced K<=8 twin. It is kept independent from the
    /// shipping K8 pipeline so production can fail closed when unavailable.
    gateup_k8_candidate: Option<ComputePipelineState>,
    /// Exact four-candidate tiled K<=8 twin.
    gateup_k8_tile4: Option<ComputePipelineState>,
    pub(crate) down: ComputePipelineState,
    /// Exact eight-row reuse probe retained independently from production.
    down_rows8: Option<ComputePipelineState>,
    /// Exact four-candidate threadgroup probe retained independently.
    down_tg4: Option<ComputePipelineState>,
    head_gateup_split: ComputePipelineState,
    head_gateup_scalar: ComputePipelineState,
    head_gateup_simd: ComputePipelineState,
    head_gateup_turbo: ComputePipelineState,
    head_down_scalar: ComputePipelineState,
    head_down_simd: ComputePipelineState,
    head_down_turbo: ComputePipelineState,
    gateup_function: Function,
}

static SPEC50_MOE_ARGBUF_KERNELS: OnceLock<Option<Spec50MoeArgbufKernels>> = OnceLock::new();
const ARGBUF_GATEUP_K12_ENV: &str = "CAMELID_GEMMA4_ARGBUF_GATEUP_K12";

fn argbuf_gateup_k12_enabled_from(value: Option<&str>) -> bool {
    value == Some("1")
}

fn argbuf_gateup_k12_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var(ARGBUF_GATEUP_K12_ENV).ok();
        argbuf_gateup_k12_enabled_from(value.as_deref())
    })
}

pub(crate) fn spec50_moe_argbuf_kernels(
    device: &Device,
) -> Option<&'static Spec50MoeArgbufKernels> {
    SPEC50_MOE_ARGBUF_KERNELS
        .get_or_init(|| {
            // The copied-slab runtime kernels are selected from the strict
            // library. Keep the same setting so pointer selection is the only
            // code-generation variable.
            let options = metal::CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = device
                .new_library_with_source(SPEC50_MOE_ARGBUF_SHADER, &options)
                .map_err(|err| eprintln!("[metal] SPEC50_MOE_ARGBUF_SHADER compile failed: {err}"))
                .ok()?;
            let gateup_function = library
                .get_function("spec50_moe_argbuf_gateup_geglu_quant_batch_k", None)
                .map_err(|err| eprintln!("[metal] argbuf GateUp function missing: {err}"))
                .ok()?;
            let down_function = library
                .get_function("spec50_moe_argbuf_down_union_batch_k", None)
                .map_err(|err| eprintln!("[metal] argbuf Down function missing: {err}"))
                .ok()?;
            let gateup = device
                .new_compute_pipeline_state_with_function(&gateup_function)
                .map_err(|err| eprintln!("[metal] argbuf GateUp pipeline failed: {err}"))
                .ok()?;
            let gateup_k12 = library
                .get_function(
                    "spec50_moe_argbuf_gateup_geglu_quant_batch_k12",
                    None,
                )
                .and_then(|function| device.new_compute_pipeline_state_with_function(&function))
                .map_err(|err| {
                    eprintln!(
                        "[metal] argbuf K12-specialized GateUp unavailable; using K16 pipeline: {err}"
                    )
                })
                .ok();
            let gateup_k8 = library
                .get_function(
                    "spec50_moe_argbuf_gateup_geglu_quant_batch_k8",
                    None,
                )
                .and_then(|function| device.new_compute_pipeline_state_with_function(&function))
                .map_err(|err| {
                    eprintln!(
                        "[metal] argbuf K8-specialized GateUp unavailable; using K16 pipeline: {err}"
                    )
                })
                .ok();
            let gateup_k8_candidate = library
                .get_function(
                    "spec50_moe_argbuf_gateup_geglu_quant_batch_k8_candidate",
                    None,
                )
                .and_then(|function| device.new_compute_pipeline_state_with_function(&function))
                .map_err(|err| {
                    eprintln!(
                        "[metal] argbuf candidate-sliced K8 GateUp unavailable; using current K8 pipeline: {err}"
                    )
                })
                .ok();
            let gateup_k8_tile4 = library
                .get_function(
                    "spec50_moe_argbuf_gateup_geglu_quant_batch_k8_tile4",
                    None,
                )
                .and_then(|function| device.new_compute_pipeline_state_with_function(&function))
                .map_err(|err| {
                    eprintln!(
                        "[metal] argbuf four-candidate tiled K8 GateUp unavailable: {err}"
                    )
                })
                .ok();
            let down = device
                .new_compute_pipeline_state_with_function(&down_function)
                .map_err(|err| eprintln!("[metal] argbuf Down pipeline failed: {err}"))
                .ok()?;
            let make_pipeline = |name: &str| {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] argbuf function {name} missing: {err}"))
                    .ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] argbuf pipeline {name} failed: {err}"))
                    .ok()
            };
            let down_rows8 = make_pipeline("spec50_moe_argbuf_down_union_batch_k_rows8");
            let down_tg4 = make_pipeline("spec50_moe_argbuf_down_union_batch_k_tg4");
            let head_gateup_split = make_pipeline("gemma4_q4_expert_argbuf_gate_up_split")?;
            let head_gateup_scalar = make_pipeline("gemma4_q4_expert_argbuf_gate_up_geglu")?;
            let head_gateup_simd = make_pipeline("gemma4_q4_expert_argbuf_gate_up_geglu_simd")?;
            let head_gateup_turbo = make_pipeline("gemma4_q4_expert_argbuf_gate_up_geglu_turbo")?;
            let head_down_scalar = make_pipeline("gemma4_q4_expert_argbuf_down_reduce")?;
            let head_down_simd = make_pipeline("gemma4_q4_expert_argbuf_down_reduce_simd")?;
            let head_down_turbo = make_pipeline("gemma4_q4_expert_argbuf_down_reduce_turbo")?;
            Some(Spec50MoeArgbufKernels {
                gateup,
                gateup_k12,
                gateup_k8,
                gateup_k8_candidate,
                gateup_k8_tile4,
                down,
                down_rows8,
                down_tg4,
                head_gateup_split,
                head_gateup_scalar,
                head_gateup_simd,
                head_gateup_turbo,
                head_down_scalar,
                head_down_simd,
                head_down_turbo,
                gateup_function,
            })
        })
        .as_ref()
}

const UNBOUND_RECORD_INDEX: u8 = u8::MAX;
const EXPERT_RECORD_ALIGNMENT: usize = 16 * 1024;

fn expert_record_binding_is_valid(record: &Buffer, offset: usize) -> bool {
    let Some(end) = offset.checked_add(S50_SLOT_STRIDE) else {
        return false;
    };
    let Some(address) = (record.contents() as usize).checked_add(offset) else {
        return false;
    };
    !record.contents().is_null()
        && u64::try_from(end).is_ok_and(|end| end <= record.length())
        && address.is_multiple_of(EXPERT_RECORD_ALIGNMENT)
}

fn same_metal_buffer(left: &Buffer, right: &Buffer) -> bool {
    std::ptr::eq::<metal::BufferRef>(&**left, &**right)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GateupRangeLayout {
    work_offset: usize,
    work_end: usize,
    output_scales_offset: usize,
    output_scales_end: usize,
    output_quants_offset: usize,
    output_quants_end: usize,
}

fn gateup_range_layout(
    work_base: usize,
    work_count: usize,
    k_candidates: usize,
) -> Option<GateupRangeLayout> {
    if work_count == 0 || !(1..=16).contains(&k_candidates) {
        return None;
    }
    let work_end_index = work_base.checked_add(work_count)?;
    let work_stride = std::mem::size_of::<Gemma4UniqueExpertWork>();
    let scale_stride = k_candidates
        .checked_mul(S50_FF / 32)?
        .checked_mul(std::mem::size_of::<f32>())?;
    let quant_stride = k_candidates.checked_mul(S50_FF)?;
    Some(GateupRangeLayout {
        work_offset: work_base.checked_mul(work_stride)?,
        work_end: work_end_index.checked_mul(work_stride)?,
        output_scales_offset: work_base.checked_mul(scale_stride)?,
        output_scales_end: work_end_index.checked_mul(scale_stride)?,
        output_quants_offset: work_base.checked_mul(quant_stride)?,
        output_quants_end: work_end_index.checked_mul(quant_stride)?,
    })
}

fn buffer_contains_bytes(buffer: &Buffer, end: usize) -> bool {
    u64::try_from(end).is_ok_and(|end| end <= buffer.length())
}

/// Encode selected records at their original slot IDs in the GateUp function's
/// reflected 128-entry argument-buffer layout. The same struct occupies buffer
/// 2 in Down. Slots without a corresponding record remain deterministic nulls.
fn new_indexed_expert_table_with_offsets(
    device: &Device,
    kernels: &Spec50MoeArgbufKernels,
    addressable_slot_count: usize,
    record_slots: &[usize],
    records: &[Buffer],
    record_offsets: &[usize],
) -> Option<(Buffer, [u8; 128])> {
    if !(1..=128).contains(&addressable_slot_count)
        || records.is_empty()
        || records.len() != record_slots.len()
        || records.len() != record_offsets.len()
        || records.len() > addressable_slot_count
        || records
            .iter()
            .zip(record_offsets)
            .any(|(record, &offset)| !expert_record_binding_is_valid(record, offset))
    {
        return None;
    }
    let mut slot_to_record = [UNBOUND_RECORD_INDEX; 128];
    for (record_index, &slot) in record_slots.iter().enumerate() {
        if slot >= addressable_slot_count || slot_to_record[slot] != UNBOUND_RECORD_INDEX {
            return None;
        }
        slot_to_record[slot] = u8::try_from(record_index).ok()?;
    }
    let encoder = kernels.gateup_function.new_argument_encoder(2);
    let table = device.new_buffer(
        encoder.encoded_length(),
        MTLResourceOptions::StorageModeShared,
    );
    // Inactive table entries must be null rather than whatever happened to be
    // in a newly allocated shared page. Production kernels validate every slot
    // before encoding, but deterministic null padding keeps that fail-closed
    // contract true even under GPU validation.
    unsafe {
        std::ptr::write_bytes(
            table.contents().cast::<u8>(),
            0,
            encoder.encoded_length() as usize,
        );
    }
    encoder.set_argument_buffer(&table, 0);
    for ((&slot, record), &offset) in record_slots.iter().zip(records).zip(record_offsets) {
        encoder.set_buffer(slot as u64, record, offset as u64);
    }
    Some((table, slot_to_record))
}

/// Zero-offset compatibility wrapper for existing per-record buffers.
fn new_indexed_expert_table(
    device: &Device,
    kernels: &Spec50MoeArgbufKernels,
    addressable_slot_count: usize,
    record_slots: &[usize],
    records: &[Buffer],
) -> Option<(Buffer, [u8; 128])> {
    let record_offsets = vec![0; records.len()];
    new_indexed_expert_table_with_offsets(
        device,
        kernels,
        addressable_slot_count,
        record_slots,
        records,
        &record_offsets,
    )
}

/// Dense compatibility wrapper used by the original prototype and its tests.
fn new_static_expert_table(
    device: &Device,
    kernels: &Spec50MoeArgbufKernels,
    records: &[Buffer],
) -> Option<Buffer> {
    let record_slots = (0..records.len()).collect::<Vec<_>>();
    new_indexed_expert_table(device, kernels, records.len(), &record_slots, records)
        .map(|(table, _)| table)
}

/// Tier-2 pointer table for persistent anonymous expert-slot records.
///
/// Cloning this object clones Objective-C references only; it neither copies
/// record bytes nor creates another Metal allocation. Construction is lazy,
/// but the first compute binding of the argument table may materialize every
/// record encoded into it. [`Self::declare_active_slots`] remains the required
/// correctness/resource-usage declaration; it does not cap physical residency.
#[derive(Clone)]
pub(crate) struct Gemma4MoeSlotArgTable {
    records: Vec<Buffer>,
    /// Exact selected-record split for this committed table. Anonymous dense
    /// tables report every bound record as hot; mapped-only tables report
    /// every bound record as mapped; mixed tables receipt both independently.
    hot_bound_record_count: usize,
    mapped_bound_record_count: usize,
    /// Number of legal shader-visible slot IDs. Sparse mapped tables retain the
    /// canonical width of 128 while owning only the exact selected records.
    addressable_slot_count: usize,
    /// Original slot ID -> index in `records`; `u8::MAX` is an unbound/null
    /// argument-table entry.
    slot_to_record: [u8; 128],
    table: Buffer,
    /// A no-copy Metal record retains the caller's bytes, not the Rust owner
    /// that mapped them. Production file-backed tables therefore keep the
    /// complete read-only mapping alive for at least as long as any cloned GPU
    /// binding. Anonymous slot tables leave this empty.
    _mapped_owner: Option<std::sync::Arc<crate::wire_mmap::GgufWireMmap>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gemma4HeadArgbufGateMode {
    Split,
    FusedScalar,
    FusedOrderedSimd,
    FusedTurbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gemma4HeadArgbufDownMode {
    Scalar,
    OrderedSimd,
    Turbo,
}

impl Gemma4MoeSlotArgTable {
    /// Build a canonical-ID pointer table whose entries may be byte ranges of
    /// larger Metal buffers. Repeating a buffer is intentional: each argument
    /// entry retains the same resource while `record_offsets` selects a
    /// distinct fixed-stride expert record within it.
    ///
    /// The effective record address must retain the existing 16-KiB alignment
    /// and the complete fixed stride must fit in its backing buffer. Resource
    /// declarations are de-duplicated by underlying Metal-buffer identity, not
    /// by table slot, so a slab referenced by many entries is declared once.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_indexed_slot_buffer_offsets(
        device: &Device,
        addressable_slot_count: usize,
        record_slots: &[usize],
        records: &[Buffer],
        record_offsets: &[usize],
        mapped_bound_record_count: usize,
        mapped_owner: Option<std::sync::Arc<crate::wire_mmap::GgufWireMmap>>,
    ) -> Option<Self> {
        if records.is_empty()
            || records.len() != record_slots.len()
            || records.len() != record_offsets.len()
            || mapped_bound_record_count > records.len()
            || (mapped_bound_record_count == 0) != mapped_owner.is_none()
            || records
                .iter()
                .zip(record_offsets)
                .any(|(record, &offset)| !expert_record_binding_is_valid(record, offset))
            || device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2
        {
            return None;
        }
        let kernels = spec50_moe_argbuf_kernels(device)?;
        let (table, slot_to_record) = new_indexed_expert_table_with_offsets(
            device,
            kernels,
            addressable_slot_count,
            record_slots,
            records,
            record_offsets,
        )?;
        Some(Self {
            // Keep one retained Objective-C reference per table entry. Cloned
            // references to the same MTLBuffer are cheap and make the pointer
            // table's complete lifetime independent of the caller.
            records: records.to_vec(),
            hot_bound_record_count: records.len() - mapped_bound_record_count,
            mapped_bound_record_count,
            addressable_slot_count,
            slot_to_record,
            table,
            _mapped_owner: mapped_owner,
        })
    }

    pub(crate) fn from_indexed_slot_buffers(
        device: &Device,
        addressable_slot_count: usize,
        record_slots: &[usize],
        records: &[Buffer],
        mapped_bound_record_count: usize,
        mapped_owner: Option<std::sync::Arc<crate::wire_mmap::GgufWireMmap>>,
    ) -> Option<Self> {
        if records
            .iter()
            .any(|record| record.length() != S50_SLOT_STRIDE as u64)
        {
            return None;
        }
        let record_offsets = vec![0; records.len()];
        Self::from_indexed_slot_buffer_offsets(
            device,
            addressable_slot_count,
            record_slots,
            records,
            &record_offsets,
            mapped_bound_record_count,
            mapped_owner,
        )
    }

    /// Build a fixed pointer table over already allocated anonymous slot
    /// records. No record byte is read or written here, so construction itself
    /// remains physically lazy. Callers must nevertheless encode only their
    /// admitted physical prefix because first compute use may materialize the
    /// table's complete referenced set.
    pub(crate) fn from_slot_buffers(device: &Device, records: &[Buffer]) -> Option<Self> {
        if records.is_empty() || records.len() > 128 {
            return None;
        }
        let record_slots = (0..records.len()).collect::<Vec<_>>();
        Self::from_indexed_slot_buffers(device, records.len(), &record_slots, records, 0, None)
    }

    fn from_mixed_active_slots_with_addressable_count(
        mmap: std::sync::Arc<crate::wire_mmap::GgufWireMmap>,
        offset: u64,
        byte_len: usize,
        addressable_slot_count: usize,
        active_slots: &[usize],
        hot_slot_ids: &[usize],
        hot_records: &[Buffer],
    ) -> Option<Self> {
        if !(1..=128).contains(&addressable_slot_count)
            || S50_RECORD_BYTES > S50_SLOT_STRIDE
            || active_slots.is_empty()
            || hot_slot_ids.len() != hot_records.len()
            || hot_records.len() > addressable_slot_count
        {
            return None;
        }
        let required = addressable_slot_count.checked_mul(S50_SLOT_STRIDE)?;
        let offset = usize::try_from(offset).ok()?;
        let end = offset.checked_add(required)?;
        let page_size = crate::wire_mmap::page_size();
        if byte_len != required
            || end > mmap.mapped_len()
            || page_size == 0
            || !S50_SLOT_STRIDE.is_multiple_of(page_size)
        {
            return None;
        }
        let base = unsafe { mmap.base_ptr().add(offset) };
        if !(base as usize).is_multiple_of(page_size) {
            return None;
        }

        // Validate the complete bounded hot directory before constructing the
        // first no-copy view. Any malformed canonical ID, duplicate override,
        // or record geometry therefore fails without partially materializing a
        // mixed residency set.
        let mut hot_slot_to_record = [UNBOUND_RECORD_INDEX; 128];
        for (record_index, (&slot, record)) in hot_slot_ids.iter().zip(hot_records).enumerate() {
            if slot >= addressable_slot_count
                || hot_slot_to_record[slot] != UNBOUND_RECORD_INDEX
                || record.length() as usize != S50_SLOT_STRIDE
                || record.contents().is_null()
                || !(record.contents() as usize).is_multiple_of(page_size)
            {
                return None;
            }
            hot_slot_to_record[slot] = u8::try_from(record_index).ok()?;
        }

        let mut record_slots = active_slots.to_vec();
        record_slots.sort_unstable();
        record_slots.dedup();
        if record_slots
            .iter()
            .any(|&slot| slot >= addressable_slot_count)
        {
            return None;
        }

        let kernel = metal_linear_kernel()?;
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2
            || S50_SLOT_STRIDE > device.max_buffer_length() as usize
        {
            return None;
        }
        let mut records = Vec::with_capacity(record_slots.len());
        let mut mapped_record_count = 0usize;
        for &slot in &record_slots {
            let hot_record_index = hot_slot_to_record[slot];
            if hot_record_index != UNBOUND_RECORD_INDEX {
                records.push(hot_records[usize::from(hot_record_index)].clone());
                continue;
            }
            let record_offset = slot.checked_mul(S50_SLOT_STRIDE)?;
            let pointer = unsafe { base.add(record_offset) };
            if !(pointer as usize).is_multiple_of(page_size) {
                return None;
            }
            records.push(device.new_buffer_with_bytes_no_copy(
                pointer.cast(),
                S50_SLOT_STRIDE as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            ));
            mapped_record_count += 1;
        }
        let mapped_owner = (mapped_record_count > 0).then_some(mmap);
        Self::from_indexed_slot_buffers(
            device,
            addressable_slot_count,
            &record_slots,
            &records,
            mapped_record_count,
            mapped_owner,
        )
    }

    /// Build one transient canonical-ID table over a clean mapped cold tier and
    /// a bounded anonymous hot tier. Hot records override the same canonical
    /// IDs in the mapped layer; all other active IDs receive exact no-copy
    /// views. Inactive hot and cold records are not encoded into the table.
    ///
    /// The returned table retains every selected anonymous buffer and, when at
    /// least one cold record is selected, the mmap owner. It must remain alive
    /// until the command buffer reaches a terminal state, and callers must not
    /// refill a selected hot record before that same terminal state.
    pub(crate) fn from_mixed_active_slots(
        mmap: std::sync::Arc<crate::wire_mmap::GgufWireMmap>,
        offset: u64,
        byte_len: usize,
        active_slots: &[usize],
        hot_slot_ids: &[usize],
        hot_records: &[Buffer],
    ) -> Option<Self> {
        Self::from_mixed_active_slots_with_addressable_count(
            mmap,
            offset,
            byte_len,
            128,
            active_slots,
            hot_slot_ids,
            hot_records,
        )
    }

    /// Build a transient sparse table over exactly the selected canonical
    /// `.cghost` records. Record buffers remain indexed by their original slot
    /// IDs, so HEAD route slots and chained fixed-stride work offsets require no
    /// translation. The returned owner must live until the GPU command reaches
    /// a terminal state.
    pub(crate) fn from_mapped_active_slots(
        mmap: std::sync::Arc<crate::wire_mmap::GgufWireMmap>,
        offset: u64,
        byte_len: usize,
        active_slots: &[usize],
    ) -> Option<Self> {
        Self::from_mixed_active_slots(mmap, offset, byte_len, active_slots, &[], &[])
    }

    pub(crate) const fn addressable_slot_count(&self) -> usize {
        self.addressable_slot_count
    }

    /// Compatibility name for existing dense callers. For sparse mapped tables
    /// this is the legal address width, not the number of owned record views.
    pub(crate) const fn slot_count(&self) -> usize {
        self.addressable_slot_count()
    }

    pub(crate) fn bound_record_count(&self) -> usize {
        self.records.len()
    }

    pub(crate) const fn hot_bound_record_count(&self) -> usize {
        self.hot_bound_record_count
    }

    pub(crate) const fn mapped_bound_record_count(&self) -> usize {
        self.mapped_bound_record_count
    }

    pub(crate) fn argument_buffer(&self) -> &Buffer {
        &self.table
    }

    /// Declare exactly the distinct active resources for one encoder.
    /// Multiple table slots backed by byte ranges of the same `MTLBuffer`
    /// require just one `use_resource` declaration.
    /// Validation is completed before the first declaration, so an invalid
    /// union cannot partially mutate the command encoder's residency set.
    pub(crate) fn declare_active_slots(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        active_slots: &[usize],
    ) -> Option<usize> {
        if active_slots.is_empty()
            || active_slots.iter().any(|&slot| {
                if slot >= self.addressable_slot_count {
                    return true;
                }
                let record_index = self.slot_to_record[slot];
                record_index == UNBOUND_RECORD_INDEX
                    || usize::from(record_index) >= self.records.len()
            })
        {
            return None;
        }
        let mut seen_slots = [false; 128];
        let mut distinct_slots = 0usize;
        let mut record_indices = Vec::with_capacity(active_slots.len());
        for &slot in active_slots {
            if seen_slots[slot] {
                continue;
            }
            seen_slots[slot] = true;
            distinct_slots += 1;
            let record_index = usize::from(self.slot_to_record[slot]);
            if !record_indices.iter().any(|&selected| {
                same_metal_buffer(&self.records[selected], &self.records[record_index])
            }) {
                record_indices.push(record_index);
            }
        }
        for &record_index in &record_indices {
            encoder.use_resource(&self.records[record_index], MTLResourceUsage::Read);
        }
        Some(distinct_slots)
    }

    /// Declare all distinct backing resources for one encoder.
    pub(crate) fn declare_all_records(&self, encoder: &metal::ComputeCommandEncoderRef) -> usize {
        let mut record_indices = Vec::with_capacity(self.records.len());
        for record_index in 0..self.records.len() {
            if !record_indices.iter().any(|&selected| {
                same_metal_buffer(&self.records[selected], &self.records[record_index])
            }) {
                record_indices.push(record_index);
            }
        }
        for &record_index in &record_indices {
            encoder.use_resource(&self.records[record_index], MTLResourceUsage::Read);
        }
        self.records.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_chained_gateup_k8(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_scales: &Buffer,
        input_quants: &Buffer,
        work_list: &Buffer,
        output_scales: &Buffer,
        output_quants: &Buffer,
        num_unique_experts: usize,
        k_candidates: usize,
    ) -> bool {
        self.encode_chained_gateup_k8_range(
            encoder,
            input_scales,
            input_quants,
            work_list,
            output_scales,
            output_quants,
            0,
            num_unique_experts,
            k_candidates,
        )
    }

    /// Encode one contiguous unique-expert range into its normal positions in
    /// the full-union activation buffers. A later full-union Down dispatch can
    /// therefore consume several independently encoded GateUp ranges without
    /// translating `unique_expert_idx` values.
    ///
    /// This method only records commands. The caller must order/barrier all
    /// range encodes before Down reads their shared full-union outputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_chained_gateup_k8_range(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_scales: &Buffer,
        input_quants: &Buffer,
        work_list: &Buffer,
        output_scales: &Buffer,
        output_quants: &Buffer,
        work_base: usize,
        work_count: usize,
        k_candidates: usize,
    ) -> bool {
        let Some(work_end) = work_base.checked_add(work_count) else {
            return false;
        };
        let Some(layout) = gateup_range_layout(work_base, work_count, k_candidates) else {
            return false;
        };
        let input_scales_end = k_candidates
            .checked_mul(S50_GU_BLOCKS)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()));
        let input_quants_end = k_candidates.checked_mul(S50_HIDDEN);
        if work_end > self.bound_record_count()
            || input_scales_end.is_none_or(|end| !buffer_contains_bytes(input_scales, end))
            || input_quants_end.is_none_or(|end| !buffer_contains_bytes(input_quants, end))
            || !buffer_contains_bytes(work_list, layout.work_end)
            || !buffer_contains_bytes(output_scales, layout.output_scales_end)
            || !buffer_contains_bytes(output_quants, layout.output_quants_end)
        {
            return false;
        }
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        let gateup = if k_candidates <= 8 {
            pipelines.gateup_k8.as_ref().unwrap_or(&pipelines.gateup)
        } else if (9..=12).contains(&k_candidates) && argbuf_gateup_k12_enabled() {
            pipelines.gateup_k12.as_ref().unwrap_or(&pipelines.gateup)
        } else {
            &pipelines.gateup
        };
        encode_argbuf_gateup_range(
            encoder,
            gateup,
            input_scales,
            input_quants,
            &self.table,
            work_list,
            output_scales,
            output_quants,
            work_base,
            work_count as u32,
            k_candidates as u32,
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_chained_down_k8(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        act_scales: &Buffer,
        act_quants: &Buffer,
        candidate_routes: &Buffer,
        work_list: &Buffer,
        output: &Buffer,
        k_candidates: usize,
    ) -> bool {
        if !(1..=16).contains(&k_candidates) {
            return false;
        }
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        if k_candidates <= 8 && pipelines.down_rows8.is_some() {
            encode_argbuf_down_rows8(
                encoder,
                pipelines.down_rows8.as_ref().unwrap(),
                act_scales,
                act_quants,
                &self.table,
                candidate_routes,
                work_list,
                output,
                k_candidates as u32,
            );
        } else {
            encode_argbuf_down(
                encoder,
                &pipelines.down,
                act_scales,
                act_quants,
                &self.table,
                candidate_routes,
                work_list,
                output,
                k_candidates as u32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_head_gateup(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_scales: &Buffer,
        input_quants: &Buffer,
        route_slots: &Buffer,
        output: &Buffer,
        mode: Gemma4HeadArgbufGateMode,
    ) -> bool {
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        let pipeline = match mode {
            Gemma4HeadArgbufGateMode::Split => &pipelines.head_gateup_split,
            Gemma4HeadArgbufGateMode::FusedScalar => &pipelines.head_gateup_scalar,
            Gemma4HeadArgbufGateMode::FusedOrderedSimd => &pipelines.head_gateup_simd,
            Gemma4HeadArgbufGateMode::FusedTurbo => &pipelines.head_gateup_turbo,
        };
        if matches!(
            mode,
            Gemma4HeadArgbufGateMode::FusedOrderedSimd | Gemma4HeadArgbufGateMode::FusedTurbo
        ) && pipeline.thread_execution_width() != 32
        {
            return false;
        }
        if mode == Gemma4HeadArgbufGateMode::FusedTurbo
            && pipeline.max_total_threads_per_threadgroup() < 128
        {
            return false;
        }
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input_scales), 0);
        encoder.set_buffer(1, Some(input_quants), 0);
        encoder.set_buffer(2, Some(&self.table), 0);
        encoder.set_buffer(3, Some(route_slots), 0);
        encoder.set_buffer(4, Some(output), 0);
        let rows = GEMMA4_Q4_EXPERT_ROUTES * GEMMA4_Q4_EXPERT_FF;
        match mode {
            Gemma4HeadArgbufGateMode::Split | Gemma4HeadArgbufGateMode::FusedScalar => {
                dispatch_1d(encoder, pipeline, rows);
            }
            Gemma4HeadArgbufGateMode::FusedOrderedSimd => {
                encoder.set_threadgroup_memory_length(
                    0,
                    (2 * GEMMA4_Q4_EXPERT_INPUT_BLOCKS * std::mem::size_of::<f32>()) as u64,
                );
                dispatch_one_simdgroup_per_row(encoder, rows);
            }
            Gemma4HeadArgbufGateMode::FusedTurbo => {
                dispatch_four_simdgroup_rows(encoder, rows);
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_head_down(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        activation_scales: &Buffer,
        activation_quants: &Buffer,
        route_slots: &Buffer,
        route_scales: &Buffer,
        output: &Buffer,
        mode: Gemma4HeadArgbufDownMode,
    ) -> bool {
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        let pipeline = match mode {
            Gemma4HeadArgbufDownMode::Scalar => &pipelines.head_down_scalar,
            Gemma4HeadArgbufDownMode::OrderedSimd => &pipelines.head_down_simd,
            Gemma4HeadArgbufDownMode::Turbo => &pipelines.head_down_turbo,
        };
        if matches!(
            mode,
            Gemma4HeadArgbufDownMode::OrderedSimd | Gemma4HeadArgbufDownMode::Turbo
        ) && pipeline.thread_execution_width() != 32
        {
            return false;
        }
        if mode == Gemma4HeadArgbufDownMode::Turbo
            && pipeline.max_total_threads_per_threadgroup() < 128
        {
            return false;
        }
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(activation_scales), 0);
        encoder.set_buffer(1, Some(activation_quants), 0);
        encoder.set_buffer(2, Some(&self.table), 0);
        encoder.set_buffer(3, Some(route_slots), 0);
        encoder.set_buffer(4, Some(route_scales), 0);
        encoder.set_buffer(5, Some(output), 0);
        match mode {
            Gemma4HeadArgbufDownMode::Scalar => {
                dispatch_1d(encoder, pipeline, GEMMA4_Q4_EXPERT_HIDDEN);
            }
            Gemma4HeadArgbufDownMode::OrderedSimd => {
                encoder.set_threadgroup_memory_length(
                    0,
                    (GEMMA4_Q4_EXPERT_ROUTES
                        * GEMMA4_Q4_EXPERT_ACT_BLOCKS
                        * std::mem::size_of::<f32>()) as u64,
                );
                dispatch_one_simdgroup_per_row(encoder, GEMMA4_Q4_EXPERT_HIDDEN);
            }
            Gemma4HeadArgbufDownMode::Turbo => {
                dispatch_four_simdgroup_rows(encoder, GEMMA4_Q4_EXPERT_HIDDEN);
            }
        }
        true
    }
}

/// One canonical `.cghost` MoE layer exposed as 128 independent Metal
/// resources plus a static expert-ID pointer table. Splitting at record
/// boundaries is essential: Metal residency follows `use_resource`, so a
/// monolithic 410-MiB layer resource faults every expert even when the shader
/// dereferences only the active union.
pub(crate) struct Spec50MoeArgbufLayer {
    records: Vec<Buffer>,
    expert_table: Buffer,
    mapped_bytes: usize,
    _mmap: std::sync::Arc<crate::wire_mmap::GgufWireMmap>,
}

impl Spec50MoeArgbufLayer {
    pub(crate) fn new(
        mmap: std::sync::Arc<crate::wire_mmap::GgufWireMmap>,
        offset: u64,
        byte_len: usize,
    ) -> Option<Self> {
        const EXPERTS: usize = 128;
        let required = EXPERTS.checked_mul(S50_SLOT_STRIDE)?;
        if S50_RECORD_BYTES > S50_SLOT_STRIDE || byte_len < required {
            return None;
        }
        let offset = usize::try_from(offset).ok()?;
        let end = offset.checked_add(required)?;
        if end > mmap.mapped_len() {
            return None;
        }
        let base = unsafe { mmap.base_ptr().add(offset) };
        if !(base as usize).is_multiple_of(crate::wire_mmap::page_size()) {
            return None;
        }

        let kernel = metal_linear_kernel()?;
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2
            || S50_SLOT_STRIDE > device.max_buffer_length() as usize
        {
            return None;
        }
        let kernels = spec50_moe_argbuf_kernels(device)?;
        let mut records = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let record_offset = expert.checked_mul(S50_SLOT_STRIDE)?;
            let pointer = unsafe { base.add(record_offset) };
            if !(pointer as usize).is_multiple_of(crate::wire_mmap::page_size()) {
                return None;
            }
            records.push(device.new_buffer_with_bytes_no_copy(
                pointer.cast(),
                S50_SLOT_STRIDE as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            ));
        }
        let expert_table = new_static_expert_table(device, kernels, &records)?;
        Some(Self {
            records,
            expert_table,
            mapped_bytes: required,
            _mmap: mmap,
        })
    }

    pub(crate) const fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    /// Snapshot the source mapping's physical residency without touching any
    /// record bytes. SPEC50 consumes the complete 3,345,408-byte payload, and
    /// the 3,358,720-byte stride is exactly 205 Apple-Silicon 16-KiB pages, so
    /// a non-resident page here is a direct first-order proxy for a page the
    /// all-mapped kernel may have to fault when this expert is active.
    ///
    /// This is instrumentation only. `mincore` is intentionally kept behind
    /// the runtime's explicit shadow flag because scanning all 30 layers can
    /// perturb a timing run even though it does not make pages resident.
    pub(crate) fn mapped_nonresident_pages_by_expert(&self) -> Option<[u16; 128]> {
        let page_size = crate::wire_mmap::page_size();
        if page_size == 0
            || !S50_SLOT_STRIDE.is_multiple_of(page_size)
            || !self.mapped_bytes.is_multiple_of(page_size)
        {
            return None;
        }
        let pages_per_expert = S50_SLOT_STRIDE / page_size;
        if pages_per_expert > u16::MAX as usize {
            return None;
        }
        let page_count = self.mapped_bytes / page_size;
        let base = self.records.first()?.contents();
        if base.is_null() || !(base as usize).is_multiple_of(page_size) {
            return None;
        }
        let mut residency = vec![0 as libc::c_char; page_count];
        let status =
            unsafe { libc::mincore(base.cast_const(), self.mapped_bytes, residency.as_mut_ptr()) };
        if status != 0 {
            return None;
        }
        let mut nonresident = [0u16; 128];
        for (expert, pages) in residency.chunks_exact(pages_per_expert).enumerate() {
            if expert >= nonresident.len() {
                return None;
            }
            nonresident[expert] = pages
                .iter()
                .filter(|&&state| (state as libc::c_int & libc::MINCORE_INCORE) == 0)
                .count() as u16;
        }
        (residency.len() == 128 * pages_per_expert).then_some(nonresident)
    }

    /// Declare exactly the host-proven active expert union. Inactive pointer
    /// entries remain encoded in the static table but are deliberately not
    /// declared: declaring all 128 was measured to fault all 128×205 pages.
    /// Returns the number of distinct resources declared, or `None` for an
    /// empty/invalid union.
    pub(crate) fn declare_active(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        active_experts: &[usize],
    ) -> Option<usize> {
        let mut seen = [false; 128];
        let mut declared = 0usize;
        for &expert in active_experts {
            if expert >= self.records.len() {
                return None;
            }
            if !seen[expert] {
                encoder.use_resource(&self.records[expert], MTLResourceUsage::Read);
                seen[expert] = true;
                declared += 1;
            }
        }
        (declared > 0).then_some(declared)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gateup(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_scales: &Buffer,
        input_quants: &Buffer,
        work_list: &Buffer,
        output_scales: &Buffer,
        output_quants: &Buffer,
        num_unique_experts: usize,
        k_candidates: usize,
    ) -> bool {
        if num_unique_experts == 0 || num_unique_experts > 128 || !(1..=16).contains(&k_candidates)
        {
            return false;
        }
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        encode_argbuf_gateup(
            encoder,
            &pipelines.gateup,
            input_scales,
            input_quants,
            &self.expert_table,
            work_list,
            output_scales,
            output_quants,
            num_unique_experts as u32,
            k_candidates as u32,
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_down(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        act_scales: &Buffer,
        act_quants: &Buffer,
        candidate_routes: &Buffer,
        work_list: &Buffer,
        output_moe_acc: &Buffer,
        k_candidates: usize,
    ) -> bool {
        if !(1..=16).contains(&k_candidates) {
            return false;
        }
        let Some(kernel) = metal_linear_kernel() else {
            return false;
        };
        let Some(pipelines) = spec50_moe_argbuf_kernels(&kernel.device) else {
            return false;
        };
        encode_argbuf_down(
            encoder,
            &pipelines.down,
            act_scales,
            act_quants,
            &self.expert_table,
            candidate_routes,
            work_list,
            output_moe_acc,
            k_candidates as u32,
        );
        true
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_gateup(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input_scales: &Buffer,
    input_quants: &Buffer,
    expert_table: &Buffer,
    work_list: &Buffer,
    output_scales: &Buffer,
    output_quants: &Buffer,
    num_unique_experts: u32,
    k_candidates: u32,
) {
    encode_argbuf_gateup_range(
        encoder,
        pipeline,
        input_scales,
        input_quants,
        expert_table,
        work_list,
        output_scales,
        output_quants,
        0,
        num_unique_experts,
        k_candidates,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_gateup_range(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input_scales: &Buffer,
    input_quants: &Buffer,
    expert_table: &Buffer,
    work_list: &Buffer,
    output_scales: &Buffer,
    output_quants: &Buffer,
    work_base: usize,
    work_count: u32,
    k_candidates: u32,
) {
    debug_assert!((1..=16).contains(&k_candidates));
    let layout = gateup_range_layout(work_base, work_count as usize, k_candidates as usize)
        .expect("validated GateUp work range");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(input_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(work_list), layout.work_offset as u64);
    encoder.set_buffer(4, Some(output_scales), layout.output_scales_offset as u64);
    encoder.set_buffer(5, Some(output_quants), layout.output_quants_offset as u64);
    encoder.set_bytes(6, 4, &work_count as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_candidates as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: u64::from(work_count) * (S50_FF as u64 / 32),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_gateup_candidate(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input_scales: &Buffer,
    input_quants: &Buffer,
    expert_table: &Buffer,
    work_list: &Buffer,
    output_scales: &Buffer,
    output_quants: &Buffer,
    num_unique_experts: u32,
    k_candidates: u32,
) {
    debug_assert!((1..=8).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(input_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(work_list), 0);
    encoder.set_buffer(4, Some(output_scales), 0);
    encoder.set_buffer(5, Some(output_quants), 0);
    encoder.set_bytes(6, 4, &num_unique_experts as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_candidates as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: u64::from(num_unique_experts) * u64::from(k_candidates) * (S50_FF as u64 / 32),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_gateup_tile4(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input_scales: &Buffer,
    input_quants: &Buffer,
    expert_table: &Buffer,
    work_list: &Buffer,
    output_scales: &Buffer,
    output_quants: &Buffer,
    num_unique_experts: u32,
    k_candidates: u32,
) {
    debug_assert!((1..=8).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(input_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(work_list), 0);
    encoder.set_buffer(4, Some(output_scales), 0);
    encoder.set_buffer(5, Some(output_quants), 0);
    encoder.set_bytes(6, 4, &num_unique_experts as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_candidates as *const u32 as *const _);
    let tiles = (k_candidates + 3) / 4;
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: u64::from(num_unique_experts) * u64::from(tiles) * (S50_FF as u64 / 32),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_down(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    act_scales: &Buffer,
    act_quants: &Buffer,
    expert_table: &Buffer,
    candidate_routes: &Buffer,
    work_list: &Buffer,
    output_moe_acc: &Buffer,
    k_candidates: u32,
) {
    debug_assert!((1..=16).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(act_scales), 0);
    encoder.set_buffer(1, Some(act_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(candidate_routes), 0);
    encoder.set_buffer(4, Some(work_list), 0);
    encoder.set_buffer(5, Some(output_moe_acc), 0);
    encoder.set_bytes(6, 4, &k_candidates as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (S50_HIDDEN / S50_DOWN_ROWS_PER_SIMDGROUP) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32 * u64::from(k_candidates),
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_down_rows8(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    act_scales: &Buffer,
    act_quants: &Buffer,
    expert_table: &Buffer,
    candidate_routes: &Buffer,
    work_list: &Buffer,
    output_moe_acc: &Buffer,
    k_candidates: u32,
) {
    debug_assert!((1..=8).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(act_scales), 0);
    encoder.set_buffer(1, Some(act_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(candidate_routes), 0);
    encoder.set_buffer(4, Some(work_list), 0);
    encoder.set_buffer(5, Some(output_moe_acc), 0);
    encoder.set_bytes(6, 4, &k_candidates as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (S50_HIDDEN / 8) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32 * u64::from(k_candidates),
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_argbuf_down_tg4(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    act_scales: &Buffer,
    act_quants: &Buffer,
    expert_table: &Buffer,
    candidate_routes: &Buffer,
    work_list: &Buffer,
    output_moe_acc: &Buffer,
    k_candidates: u32,
) {
    debug_assert!((1..=8).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(act_scales), 0);
    encoder.set_buffer(1, Some(act_quants), 0);
    encoder.set_buffer(2, Some(expert_table), 0);
    encoder.set_buffer(3, Some(candidate_routes), 0);
    encoder.set_buffer(4, Some(work_list), 0);
    encoder.set_buffer(5, Some(output_moe_acc), 0);
    encoder.set_bytes(6, 4, &k_candidates as *const u32 as *const _);
    let candidate_tiles = (k_candidates + 3) / 4;
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (S50_HIDDEN / S50_DOWN_ROWS_PER_SIMDGROUP) as u64 * u64::from(candidate_tiles),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32 * u64::from(k_candidates.min(4)),
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{MTLArgumentBuffersTier, MTLCommandBufferStatus, MTLResourceUsage};

    use super::super::spec50_moe::{
        encode_spec50_down, encode_spec50_gateup, spec50_moe_pipelines, S50_DOWN_BLOCKS,
        S50_RECORD_BYTES, S50_ROUTES,
    };

    const EXPERTS: usize = 128;
    const ACTIVE_EXPERTS: usize = 30;
    // The copied-slab oracle used by this module is deliberately the shipping
    // K<=8 SPEC50 path. K=9..16 is covered by `spec50_widen`'s independent
    // parity tests; asking this oracle for K>8 trips its admission guard before
    // the argument-buffer result can be compared.
    const MAX_K: usize = 8;

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    fn new_buffer(device: &Device, len: usize) -> Buffer {
        device.new_buffer(len.max(4) as u64, MTLResourceOptions::StorageModeShared)
    }

    #[test]
    fn gateup_range_layout_keeps_outputs_in_full_union_coordinates() {
        let layout = gateup_range_layout(3, 5, 8).unwrap();
        assert_eq!(
            layout.work_offset,
            3 * std::mem::size_of::<Gemma4UniqueExpertWork>()
        );
        assert_eq!(
            layout.work_end,
            8 * std::mem::size_of::<Gemma4UniqueExpertWork>()
        );
        assert_eq!(layout.output_scales_offset, 3 * 8 * (S50_FF / 32) * 4);
        assert_eq!(layout.output_scales_end, 8 * 8 * (S50_FF / 32) * 4);
        assert_eq!(layout.output_quants_offset, 3 * 8 * S50_FF);
        assert_eq!(layout.output_quants_end, 8 * 8 * S50_FF);
        assert!(gateup_range_layout(0, 0, 8).is_none());
        assert!(gateup_range_layout(0, 1, 0).is_none());
        assert!(gateup_range_layout(0, 1, 17).is_none());
        assert!(gateup_range_layout(usize::MAX, 1, 8).is_none());
    }

    #[test]
    fn gateup_k12_experiment_gate_is_strictly_opt_in() {
        assert!(!argbuf_gateup_k12_enabled_from(None));
        assert!(!argbuf_gateup_k12_enabled_from(Some("")));
        assert!(!argbuf_gateup_k12_enabled_from(Some("0")));
        assert!(!argbuf_gateup_k12_enabled_from(Some("true")));
        assert!(argbuf_gateup_k12_enabled_from(Some("1")));
    }

    #[test]
    fn indexed_table_accepts_aligned_views_of_one_stage_slab() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP indexed stage-slab structure gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            return;
        }
        let stage = new_buffer(device, 3 * S50_SLOT_STRIDE);
        let records = vec![stage.clone(), stage.clone(), stage.clone()];
        let offsets = [0, S50_SLOT_STRIDE, 2 * S50_SLOT_STRIDE];
        let table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            3,
            &[0, 1, 2],
            &records,
            &offsets,
            0,
            None,
        )
        .expect("three aligned record views of one stage slab");
        assert_eq!(table.slot_count(), 3);
        assert_eq!(table.bound_record_count(), 3);

        let cb = kernel.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(table.declare_active_slots(encoder, &[0, 1, 2]), Some(3));
        encoder.end_encoding();

        let mut misaligned = offsets;
        misaligned[1] += 1;
        assert!(Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            3,
            &[0, 1, 2],
            &records,
            &misaligned,
            0,
            None,
        )
        .is_none());
        assert!(Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            3,
            &[0, 1, 2],
            &records,
            &[0, S50_SLOT_STRIDE, 3 * S50_SLOT_STRIDE],
            0,
            None,
        )
        .is_none());
    }

    fn write_bytes<T>(buffer: &Buffer, values: &[T]) {
        let len = std::mem::size_of_val(values);
        assert!(len <= buffer.length() as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                len,
            );
        }
    }

    fn fill_zero(buffer: &Buffer) {
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, buffer.length() as usize);
        }
    }

    fn read_bytes(buffer: &Buffer, len: usize) -> Vec<u8> {
        assert!(len <= buffer.length() as usize);
        let mut bytes = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.contents().cast::<u8>(), bytes.as_mut_ptr(), len);
        }
        bytes
    }

    fn assert_raw_eq(label: &str, expected: &[u8], actual: &[u8]) {
        assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
        if let Some(index) = expected
            .iter()
            .zip(actual)
            .position(|(expected, actual)| expected != actual)
        {
            panic!(
                "{label}: raw mismatch at byte {index}/{}: expected 0x{:02x}, got 0x{:02x}",
                expected.len(),
                expected[index],
                actual[index]
            );
        }
    }

    struct Routing {
        work: Vec<Gemma4UniqueExpertWork>,
        routes: Vec<Gemma4CandidateRouteEntry>,
        active_experts: Vec<usize>,
    }

    fn route_fixture() -> ([[u32; S50_ROUTES]; MAX_K], [[f32; S50_ROUTES]; MAX_K]) {
        let mut experts = [[0u32; S50_ROUTES]; MAX_K];
        let mut weights = [[0.0f32; S50_ROUTES]; MAX_K];
        for token in 0..MAX_K {
            for rank in 0..S50_ROUTES {
                experts[token][rank] = ((token * 4 + rank * 3) % ACTIVE_EXPERTS) as u32;
                weights[token][rank] = 0.05 + rank as f32 * 0.01 + token as f32 * 0.001;
            }
        }
        (experts, weights)
    }

    fn build_routing(k: usize) -> Routing {
        let (experts, weights) = route_fixture();
        let mut active_map = [u32::MAX; EXPERTS];
        let mut work = vec![Gemma4UniqueExpertWork::default(); EXPERTS];
        let mut active_experts = Vec::new();

        for expert_id in 0..ACTIVE_EXPERTS {
            let mut mask = 0u64;
            for token in 0..k {
                for rank in 0..S50_ROUTES {
                    if experts[token][rank] as usize == expert_id && weights[token][rank] != 0.0 {
                        mask |= 1u64 << token;
                    }
                }
            }
            if mask == 0 {
                continue;
            }
            let unique = active_experts.len();
            active_map[expert_id] = unique as u32;
            work[unique] = Gemma4UniqueExpertWork {
                candidate_mask: mask,
                expert_weight_offset: (expert_id * S50_SLOT_STRIDE) as u32,
                slab_index: 0,
            };
            active_experts.push(expert_id);
        }

        let mut routes = vec![Gemma4CandidateRouteEntry::default(); MAX_K * S50_ROUTES];
        for token in 0..k {
            for rank in 0..S50_ROUTES {
                let expert_id = experts[token][rank] as usize;
                let unique = active_map[expert_id];
                assert_ne!(unique, u32::MAX);
                routes[token * S50_ROUTES + rank] = Gemma4CandidateRouteEntry {
                    unique_expert_idx: unique,
                    weight: weights[token][rank],
                };
            }
        }
        Routing {
            work,
            routes,
            active_experts,
        }
    }

    struct Buffers {
        input_scales: Buffer,
        input_quants: Buffer,
        work: Buffer,
        routes: Buffer,
        copy_scales: Buffer,
        copy_quants: Buffer,
        arg_scales: Buffer,
        arg_quants: Buffer,
        copy_down: Buffer,
        arg_down: Buffer,
    }

    impl Buffers {
        fn new(device: &Device) -> Self {
            let mut rng = Rng::new(0xa26b_5001);
            let input_scales = new_buffer(device, MAX_K * S50_GU_BLOCKS * 4);
            let scales: Vec<f32> = (0..MAX_K * S50_GU_BLOCKS)
                .map(|_| 0.0005 + rng.next_f32() * 0.01)
                .collect();
            write_bytes(&input_scales, &scales);

            let input_quants = new_buffer(device, MAX_K * S50_HIDDEN);
            let quants: Vec<i8> = (0..MAX_K * S50_HIDDEN)
                .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
                .collect();
            write_bytes(&input_quants, &quants);

            let scale_len = ACTIVE_EXPERTS * MAX_K * S50_DOWN_BLOCKS * 4;
            let quant_len = ACTIVE_EXPERTS * MAX_K * S50_FF;
            let down_len = MAX_K * S50_HIDDEN * 4;
            Self {
                input_scales,
                input_quants,
                work: new_buffer(
                    device,
                    EXPERTS * std::mem::size_of::<Gemma4UniqueExpertWork>(),
                ),
                routes: new_buffer(
                    device,
                    MAX_K * S50_ROUTES * std::mem::size_of::<Gemma4CandidateRouteEntry>(),
                ),
                copy_scales: new_buffer(device, scale_len),
                copy_quants: new_buffer(device, quant_len),
                arg_scales: new_buffer(device, scale_len),
                arg_quants: new_buffer(device, quant_len),
                copy_down: new_buffer(device, down_len),
                arg_down: new_buffer(device, down_len),
            }
        }

        fn upload(&self, routing: &Routing) {
            write_bytes(&self.work, &routing.work);
            write_bytes(&self.routes, &routing.routes);
        }

        fn zero_outputs(&self) {
            for output in [
                &self.copy_scales,
                &self.copy_quants,
                &self.arg_scales,
                &self.arg_quants,
                &self.copy_down,
                &self.arg_down,
            ] {
                fill_zero(output);
            }
        }
    }

    const SYNTHETIC_EXPERTS: usize = S50_ROUTES;

    /// Create valid, deterministic Q4_0 records without consulting a model
    /// file. Every 18-byte wire block carries a finite f16 scale, so parity
    /// failures cannot be hidden by NaNs from arbitrary test bytes.
    fn synthetic_slot_backings_count(
        device: &Device,
        expert_count: usize,
    ) -> (Vec<Buffer>, Buffer) {
        assert!((1..=EXPERTS).contains(&expert_count));
        assert_eq!(S50_RECORD_BYTES % 18, 0);
        let copied_slab = new_buffer(device, expert_count * S50_SLOT_STRIDE);
        let mut records = Vec::with_capacity(expert_count);
        for slot in 0..expert_count {
            let record = new_buffer(device, S50_SLOT_STRIDE);
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(record.contents().cast::<u8>(), S50_SLOT_STRIDE)
            };
            bytes.fill(0);
            for (block_idx, block) in bytes[..S50_RECORD_BYTES].chunks_exact_mut(18).enumerate() {
                let scale = 0.0005 + slot as f32 * 0.000_031 + (block_idx % 17) as f32 * 0.000_007;
                block[..2].copy_from_slice(&super::super::f32_to_f16_bits(scale).to_le_bytes());
                for (byte_idx, quant) in block[2..].iter_mut().enumerate() {
                    *quant = (slot
                        .wrapping_mul(37)
                        .wrapping_add(block_idx.wrapping_mul(13))
                        .wrapping_add(byte_idx.wrapping_mul(17)))
                        as u8;
                }
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    copied_slab
                        .contents()
                        .cast::<u8>()
                        .add(slot * S50_SLOT_STRIDE),
                    S50_SLOT_STRIDE,
                );
            }
            records.push(record);
        }
        (records, copied_slab)
    }

    fn synthetic_slot_backings(device: &Device) -> (Vec<Buffer>, Buffer) {
        synthetic_slot_backings_count(device, SYNTHETIC_EXPERTS)
    }

    /// Every token routes to all eight synthetic records in a different order.
    /// That keeps the active union fixed while exercising candidate masks and
    /// route-table indexing at every admitted K.
    fn build_synthetic_routing(k: usize) -> Routing {
        assert!((1..=MAX_K).contains(&k));
        let mut work = vec![Gemma4UniqueExpertWork::default(); EXPERTS];
        let candidate_mask = (1u64 << k) - 1;
        for (slot, entry) in work.iter_mut().take(SYNTHETIC_EXPERTS).enumerate() {
            *entry = Gemma4UniqueExpertWork {
                candidate_mask,
                expert_weight_offset: (slot * S50_SLOT_STRIDE) as u32,
                slab_index: 0,
            };
        }
        let mut routes = vec![Gemma4CandidateRouteEntry::default(); MAX_K * S50_ROUTES];
        for token in 0..k {
            for rank in 0..S50_ROUTES {
                let slot = (token * 5 + rank * 3) % SYNTHETIC_EXPERTS;
                routes[token * S50_ROUTES + rank] = Gemma4CandidateRouteEntry {
                    unique_expert_idx: slot as u32,
                    weight: 0.05 + rank as f32 * 0.01 + token as f32 * 0.001,
                };
            }
        }
        Routing {
            work,
            routes,
            active_experts: (0..SYNTHETIC_EXPERTS).collect(),
        }
    }

    fn shipping_head_gateup_pipeline<'a>(
        kernel: &'a super::super::MetalLinearKernel,
        mode: Gemma4HeadArgbufGateMode,
    ) -> Option<&'a ComputePipelineState> {
        let pipeline = match mode {
            Gemma4HeadArgbufGateMode::Split => {
                Some(&kernel.gemma4_q4_expert_gate_up_split_pipeline)
            }
            Gemma4HeadArgbufGateMode::FusedScalar => {
                Some(&kernel.gemma4_q4_expert_gate_up_geglu_pipeline)
            }
            Gemma4HeadArgbufGateMode::FusedOrderedSimd => {
                kernel.gemma4_q4_expert_gate_up_geglu_simd_pipeline.as_ref()
            }
            Gemma4HeadArgbufGateMode::FusedTurbo => kernel
                .gemma4_q4_expert_gate_up_geglu_turbo_pipeline
                .as_ref(),
        }?;
        if matches!(
            mode,
            Gemma4HeadArgbufGateMode::FusedOrderedSimd | Gemma4HeadArgbufGateMode::FusedTurbo
        ) && pipeline.thread_execution_width() != 32
        {
            return None;
        }
        if mode == Gemma4HeadArgbufGateMode::FusedTurbo
            && pipeline.max_total_threads_per_threadgroup() < 128
        {
            return None;
        }
        Some(pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shipping_head_gateup(
        kernel: &super::super::MetalLinearKernel,
        encoder: &metal::ComputeCommandEncoderRef,
        input_scales: &Buffer,
        input_quants: &Buffer,
        copied_slab: &Buffer,
        route_slots: &Buffer,
        output: &Buffer,
        mode: Gemma4HeadArgbufGateMode,
    ) -> bool {
        let Some(pipeline) = shipping_head_gateup_pipeline(kernel, mode) else {
            return false;
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input_scales), 0);
        encoder.set_buffer(1, Some(input_quants), 0);
        encoder.set_buffer(2, Some(copied_slab), 0);
        encoder.set_buffer(3, Some(route_slots), 0);
        encoder.set_buffer(4, Some(output), 0);
        let rows = GEMMA4_Q4_EXPERT_ROUTES * GEMMA4_Q4_EXPERT_FF;
        match mode {
            Gemma4HeadArgbufGateMode::Split | Gemma4HeadArgbufGateMode::FusedScalar => {
                super::super::dispatch_1d(encoder, pipeline, rows);
            }
            Gemma4HeadArgbufGateMode::FusedOrderedSimd => {
                encoder.set_threadgroup_memory_length(
                    0,
                    (2 * GEMMA4_Q4_EXPERT_INPUT_BLOCKS * std::mem::size_of::<f32>()) as u64,
                );
                super::super::dispatch_one_simdgroup_per_row(encoder, rows);
            }
            Gemma4HeadArgbufGateMode::FusedTurbo => {
                super::super::dispatch_four_simdgroup_rows(encoder, rows);
            }
        }
        true
    }

    fn shipping_head_down_pipeline<'a>(
        kernel: &'a super::super::MetalLinearKernel,
        mode: Gemma4HeadArgbufDownMode,
    ) -> Option<&'a ComputePipelineState> {
        let pipeline = match mode {
            Gemma4HeadArgbufDownMode::Scalar => Some(&kernel.gemma4_q4_expert_down_reduce_pipeline),
            Gemma4HeadArgbufDownMode::OrderedSimd => {
                kernel.gemma4_q4_expert_down_reduce_simd_pipeline.as_ref()
            }
            Gemma4HeadArgbufDownMode::Turbo => {
                kernel.gemma4_q4_expert_down_reduce_turbo_pipeline.as_ref()
            }
        }?;
        if matches!(
            mode,
            Gemma4HeadArgbufDownMode::OrderedSimd | Gemma4HeadArgbufDownMode::Turbo
        ) && pipeline.thread_execution_width() != 32
        {
            return None;
        }
        if mode == Gemma4HeadArgbufDownMode::Turbo
            && pipeline.max_total_threads_per_threadgroup() < 128
        {
            return None;
        }
        Some(pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shipping_head_down(
        kernel: &super::super::MetalLinearKernel,
        encoder: &metal::ComputeCommandEncoderRef,
        activation_scales: &Buffer,
        activation_quants: &Buffer,
        copied_slab: &Buffer,
        route_slots: &Buffer,
        route_scales: &Buffer,
        output: &Buffer,
        mode: Gemma4HeadArgbufDownMode,
    ) -> bool {
        let Some(pipeline) = shipping_head_down_pipeline(kernel, mode) else {
            return false;
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(activation_scales), 0);
        encoder.set_buffer(1, Some(activation_quants), 0);
        encoder.set_buffer(2, Some(copied_slab), 0);
        encoder.set_buffer(3, Some(route_slots), 0);
        encoder.set_buffer(4, Some(route_scales), 0);
        encoder.set_buffer(5, Some(output), 0);
        match mode {
            Gemma4HeadArgbufDownMode::Scalar => {
                super::super::dispatch_1d(encoder, pipeline, GEMMA4_Q4_EXPERT_HIDDEN);
            }
            Gemma4HeadArgbufDownMode::OrderedSimd => {
                encoder.set_threadgroup_memory_length(
                    0,
                    (GEMMA4_Q4_EXPERT_ROUTES
                        * GEMMA4_Q4_EXPERT_ACT_BLOCKS
                        * std::mem::size_of::<f32>()) as u64,
                );
                super::super::dispatch_one_simdgroup_per_row(encoder, GEMMA4_Q4_EXPERT_HIDDEN);
            }
            Gemma4HeadArgbufDownMode::Turbo => {
                super::super::dispatch_four_simdgroup_rows(encoder, GEMMA4_Q4_EXPERT_HIDDEN);
            }
        }
        true
    }

    #[test]
    fn gemma4_moe_sparse_slot_table_preserves_original_ids_and_fails_closed() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP sparse argbuf structure gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            return;
        }
        let records = vec![
            new_buffer(device, S50_SLOT_STRIDE),
            new_buffer(device, S50_SLOT_STRIDE),
            new_buffer(device, S50_SLOT_STRIDE),
        ];
        let table = Gemma4MoeSlotArgTable::from_indexed_slot_buffers(
            device,
            128,
            &[127, 3, 91],
            &records,
            0,
            None,
        )
        .expect("sparse original-ID table");
        assert_eq!(table.addressable_slot_count(), 128);
        assert_eq!(table.bound_record_count(), 3);
        assert_eq!(table.hot_bound_record_count(), 3);
        assert_eq!(table.mapped_bound_record_count(), 0);
        assert!(Gemma4MoeSlotArgTable::from_indexed_slot_buffers(
            device,
            128,
            &[3, 3, 91],
            &records,
            0,
            None,
        )
        .is_none());
        let cb = kernel.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(table.declare_active_slots(encoder, &[3, 4]), None);
        assert_eq!(table.declare_active_slots(encoder, &[127, 128]), None);
        assert_eq!(
            table.declare_active_slots(encoder, &[127, 3, 91, 127]),
            Some(3)
        );
        encoder.end_encoding();
    }

    #[test]
    fn gemma4_moe_mapped_active_slots_are_sparse_and_retain_owner() {
        let Some(kernel) = metal_linear_kernel() else {
            return;
        };
        if kernel.device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            return;
        }
        let mapped_bytes = 128usize * S50_SLOT_STRIDE;
        let file = tempfile::NamedTempFile::new().expect("temporary sparse mapped fixture");
        file.as_file()
            .set_len(mapped_bytes as u64)
            .expect("size sparse mapped fixture");
        let mmap =
            crate::wire_mmap::GgufWireMmap::map(file.path()).expect("map sparse mapped fixture");
        let owner = std::sync::Arc::downgrade(&mmap);
        let table = Gemma4MoeSlotArgTable::from_mapped_active_slots(
            mmap,
            0,
            mapped_bytes,
            &[127, 3, 3, 91, 0],
        )
        .expect("selected-only mapped table");
        assert_eq!(table.addressable_slot_count(), 128);
        assert_eq!(table.bound_record_count(), 4);
        assert_eq!(table.hot_bound_record_count(), 0);
        assert_eq!(table.mapped_bound_record_count(), 4);
        assert!(owner.upgrade().is_some());
        let cb = kernel.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(table.declare_active_slots(encoder, &[3, 42]), None);
        assert_eq!(
            table.declare_active_slots(encoder, &[127, 3, 91, 0]),
            Some(4)
        );
        encoder.end_encoding();
        drop(table);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn gemma4_moe_slot_arg_table_chained_k1_to_k8_raw_bit_parity_and_dedup() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP anonymous argbuf parity gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP anonymous argbuf parity gate: Tier-2 argument buffers unavailable");
            return;
        }

        assert!(Gemma4MoeSlotArgTable::from_slot_buffers(device, &[]).is_none());
        let wrong_size = new_buffer(device, S50_SLOT_STRIDE - 1);
        assert!(Gemma4MoeSlotArgTable::from_slot_buffers(
            device,
            std::slice::from_ref(&wrong_size),
        )
        .is_none());

        let (records, copied_slab) = synthetic_slot_backings(device);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("anonymous per-slot Tier-2 table");
        assert_eq!(table.slot_count(), SYNTHETIC_EXPERTS);
        assert!(table.argument_buffer().length() > 0);
        let copied = spec50_moe_pipelines(device).expect("copied-slab SPEC50 pipelines");
        let buffers = Buffers::new(device);

        for k in 1..=MAX_K {
            let routing = build_synthetic_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            assert_eq!(table.declare_active_slots(encoder, &[]), None);
            assert_eq!(
                table.declare_active_slots(encoder, &[0, SYNTHETIC_EXPERTS]),
                None
            );
            let declared_with_duplicates: Vec<usize> = routing
                .active_experts
                .iter()
                .copied()
                .chain(routing.active_experts.iter().rev().copied())
                .collect();
            assert_eq!(
                table.declare_active_slots(encoder, &declared_with_duplicates),
                Some(SYNTHETIC_EXPERTS)
            );

            encode_spec50_gateup(
                encoder,
                &copied.gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                &copied_slab,
                0,
                &buffers.work,
                &buffers.copy_scales,
                &buffers.copy_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
                None,
                None,
            );
            assert!(table.encode_chained_gateup_k8(
                encoder,
                &buffers.input_scales,
                &buffers.input_quants,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                SYNTHETIC_EXPERTS,
                k,
            ));
            encoder.memory_barrier_with_resources(&[
                &buffers.copy_scales,
                &buffers.copy_quants,
                &buffers.arg_scales,
                &buffers.arg_quants,
            ]);
            encode_spec50_down(
                encoder,
                &copied.down,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &copied_slab,
                0,
                &buffers.routes,
                &buffers.work,
                &buffers.copy_down,
                k as u32,
                None,
            );
            assert!(table.encode_chained_down_k8(
                encoder,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "anonymous argbuf K={k} command failed: {}",
                command_buffer_error_details(cb)
            );

            let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
            let down_bytes = k * S50_HIDDEN * 4;
            assert_raw_eq(
                &format!("anonymous K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("anonymous K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );
            assert_raw_eq(
                &format!("anonymous K={k} Down output"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );
        }
    }

    /// Real-Metal proof that the candidate-sliced grid changes scheduling,
    /// not arithmetic. Compare the complete scale and quant payload at every
    /// admitted width; byte equality also covers inactive output holes.
    #[test]
    fn gemma4_moe_candidate_sliced_gateup_k1_to_k8_raw_bit_parity() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP candidate-sliced GateUp parity: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP candidate-sliced GateUp parity: Tier-2 argument buffers unavailable");
            return;
        }
        let pipelines = spec50_moe_argbuf_kernels(device).expect("argument-buffer MoE pipelines");
        let fast = pipelines
            .gateup_k8_candidate
            .as_ref()
            .expect("candidate-sliced K8 GateUp pipeline");
        let tile4 = pipelines
            .gateup_k8_tile4
            .as_ref()
            .expect("four-candidate tiled K8 GateUp pipeline");
        let current = pipelines.gateup_k8.as_ref().unwrap_or(&pipelines.gateup);
        let (records, _) = synthetic_slot_backings(device);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("anonymous per-slot Tier-2 table");
        let buffers = Buffers::new(device);

        for k in 1..=MAX_K {
            let routing = build_synthetic_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &routing.active_experts),
                Some(SYNTHETIC_EXPERTS)
            );
            encode_argbuf_gateup(
                encoder,
                current,
                &buffers.input_scales,
                &buffers.input_quants,
                table.argument_buffer(),
                &buffers.work,
                &buffers.copy_scales,
                &buffers.copy_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encode_argbuf_gateup_candidate(
                encoder,
                fast,
                &buffers.input_scales,
                &buffers.input_quants,
                table.argument_buffer(),
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "candidate-sliced GateUp K={k} command failed: {}",
                command_buffer_error_details(cb)
            );

            let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
            assert_raw_eq(
                &format!("candidate-sliced K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("candidate-sliced K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );

            fill_zero(&buffers.arg_scales);
            fill_zero(&buffers.arg_quants);
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &routing.active_experts),
                Some(SYNTHETIC_EXPERTS)
            );
            encode_argbuf_gateup_tile4(
                encoder,
                tile4,
                &buffers.input_scales,
                &buffers.input_quants,
                table.argument_buffer(),
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "four-candidate tiled GateUp K={k} command failed: {}",
                command_buffer_error_details(cb)
            );
            assert_raw_eq(
                &format!("four-candidate tiled K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("four-candidate tiled K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );
        }
    }

    /// Real-Metal proof that narrowing the live accumulator arrays from 16 to
    /// 12 does not alter any GateUp scale or quant bit at the four widths that
    /// can select the experimental pipeline.
    #[test]
    fn gemma4_moe_argbuf_gateup_k12_raw_bit_parity() {
        const WIDE_K: usize = 12;

        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP K12 GateUp parity: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP K12 GateUp parity: Tier-2 argument buffers unavailable");
            return;
        }
        let pipelines = spec50_moe_argbuf_kernels(device).expect("argument-buffer MoE pipelines");
        let narrow = pipelines
            .gateup_k12
            .as_ref()
            .expect("K12-specialized GateUp pipeline");
        let (records, _) = synthetic_slot_backings(device);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("anonymous per-slot Tier-2 table");

        let mut rng = Rng::new(0x12a2_6b50_01);
        let input_scales = new_buffer(device, WIDE_K * S50_GU_BLOCKS * 4);
        let scales: Vec<f32> = (0..WIDE_K * S50_GU_BLOCKS)
            .map(|_| 0.0005 + rng.next_f32() * 0.01)
            .collect();
        write_bytes(&input_scales, &scales);
        let input_quants = new_buffer(device, WIDE_K * S50_HIDDEN);
        let quants: Vec<i8> = (0..WIDE_K * S50_HIDDEN)
            .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
            .collect();
        write_bytes(&input_quants, &quants);

        let work_list = new_buffer(
            device,
            EXPERTS * std::mem::size_of::<Gemma4UniqueExpertWork>(),
        );
        let scale_len = SYNTHETIC_EXPERTS * WIDE_K * S50_DOWN_BLOCKS * 4;
        let quant_len = SYNTHETIC_EXPERTS * WIDE_K * S50_FF;
        let reference_scales = new_buffer(device, scale_len);
        let reference_quants = new_buffer(device, quant_len);
        let narrow_scales = new_buffer(device, scale_len);
        let narrow_quants = new_buffer(device, quant_len);
        let active_experts: Vec<usize> = (0..SYNTHETIC_EXPERTS).collect();

        for k in 9..=WIDE_K {
            let candidate_mask = (1u64 << k) - 1;
            let mut work = vec![Gemma4UniqueExpertWork::default(); EXPERTS];
            for (slot, entry) in work.iter_mut().take(SYNTHETIC_EXPERTS).enumerate() {
                *entry = Gemma4UniqueExpertWork {
                    candidate_mask,
                    expert_weight_offset: (slot * S50_SLOT_STRIDE) as u32,
                    slab_index: 0,
                };
            }
            write_bytes(&work_list, &work);
            for output in [
                &reference_scales,
                &reference_quants,
                &narrow_scales,
                &narrow_quants,
            ] {
                fill_zero(output);
            }

            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &active_experts),
                Some(SYNTHETIC_EXPERTS)
            );
            encode_argbuf_gateup(
                encoder,
                &pipelines.gateup,
                &input_scales,
                &input_quants,
                table.argument_buffer(),
                &work_list,
                &reference_scales,
                &reference_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encode_argbuf_gateup(
                encoder,
                narrow,
                &input_scales,
                &input_quants,
                table.argument_buffer(),
                &work_list,
                &narrow_scales,
                &narrow_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "K12 GateUp parity K={k} command failed: {}",
                command_buffer_error_details(cb)
            );

            let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
            assert_raw_eq(
                &format!("K12 K={k} GateUp scales"),
                &read_bytes(&reference_scales, scale_bytes),
                &read_bytes(&narrow_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("K12 K={k} GateUp quants"),
                &read_bytes(&reference_quants, quant_bytes),
                &read_bytes(&narrow_quants, quant_bytes),
            );
        }
    }

    /// Real-Metal proof that both Down scheduling probes preserve every
    /// candidate row's lane ownership, term order, and SIMD reduction bits.
    #[test]
    fn gemma4_moe_down_schedule_k1_to_k8_raw_bit_parity() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP Down schedule parity: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP Down schedule parity: Tier-2 argument buffers unavailable");
            return;
        }
        let pipelines = spec50_moe_argbuf_kernels(device).expect("argument-buffer MoE pipelines");
        let rows8 = pipelines
            .down_rows8
            .as_ref()
            .expect("eight-row Down pipeline");
        let tg4 = pipelines
            .down_tg4
            .as_ref()
            .expect("four-candidate threadgroup Down pipeline");
        let gateup = pipelines.gateup_k8.as_ref().unwrap_or(&pipelines.gateup);
        let (records, _) = synthetic_slot_backings(device);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("anonymous per-slot Tier-2 table");
        let buffers = Buffers::new(device);

        for k in 1..=MAX_K {
            let routing = build_synthetic_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &routing.active_experts),
                Some(SYNTHETIC_EXPERTS)
            );
            encode_argbuf_gateup(
                encoder,
                gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                table.argument_buffer(),
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
            );
            encoder.memory_barrier_with_resources(&[&buffers.arg_scales, &buffers.arg_quants]);
            encode_argbuf_down(
                encoder,
                &pipelines.down,
                &buffers.arg_scales,
                &buffers.arg_quants,
                table.argument_buffer(),
                &buffers.routes,
                &buffers.work,
                &buffers.copy_down,
                k as u32,
            );
            encode_argbuf_down_rows8(
                encoder,
                rows8,
                &buffers.arg_scales,
                &buffers.arg_quants,
                table.argument_buffer(),
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "Down rows8 K={k} command failed: {}",
                command_buffer_error_details(cb)
            );
            let down_bytes = k * S50_HIDDEN * std::mem::size_of::<f32>();
            assert_raw_eq(
                &format!("Down rows8 K={k}"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );

            fill_zero(&buffers.arg_down);
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &routing.active_experts),
                Some(SYNTHETIC_EXPERTS)
            );
            encode_argbuf_down_tg4(
                encoder,
                tg4,
                &buffers.arg_scales,
                &buffers.arg_quants,
                table.argument_buffer(),
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "Down tg4 K={k} command failed: {}",
                command_buffer_error_details(cb)
            );
            assert_raw_eq(
                &format!("Down tg4 K={k}"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );
        }
    }

    #[test]
    fn gemma4_moe_mixed_hot_cold_table_chained_k1_to_k8_raw_bit_parity_and_lifetime() {
        use std::io::Write as _;

        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP mixed hot/cold argbuf parity gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!(
                "SKIP mixed hot/cold argbuf parity gate: Tier-2 argument buffers unavailable"
            );
            return;
        }

        let (records, copied_slab) = synthetic_slot_backings(device);
        let hot_slot_ids = [1usize, 4, 6];
        let hot_records = hot_slot_ids
            .iter()
            .map(|&slot| records[slot].clone())
            .collect::<Vec<_>>();

        // The mapped copy deliberately poisons every hot slot while retaining
        // valid Q4_0 block structure. Exact parity with `copied_slab` therefore
        // proves that canonical hot overrides won and every other slot came
        // from the cold mapping.
        let mapped_bytes = SYNTHETIC_EXPERTS * S50_SLOT_STRIDE;
        let mut file = tempfile::NamedTempFile::new().expect("temporary mixed expert fixture");
        file.as_file()
            .set_len(mapped_bytes as u64)
            .expect("size mixed expert fixture");
        for (slot, record) in records.iter().enumerate() {
            let mut bytes = read_bytes(record, S50_SLOT_STRIDE);
            if hot_slot_ids.contains(&slot) {
                for block in bytes[..S50_RECORD_BYTES].chunks_exact_mut(18) {
                    block[2] ^= 0x5a;
                }
            }
            file.as_file_mut()
                .write_all(&bytes)
                .expect("write mixed expert fixture record");
        }
        file.as_file()
            .sync_all()
            .expect("sync mixed expert fixture");

        let mmap =
            crate::wire_mmap::GgufWireMmap::map(file.path()).expect("map mixed expert fixture");
        let mapped_owner = std::sync::Arc::downgrade(&mmap);

        // The complete hot directory and active union are validated before any
        // no-copy view exists. Duplicates, mismatched pairs, and out-of-range
        // canonical IDs must all fail closed.
        assert!(
            Gemma4MoeSlotArgTable::from_mixed_active_slots_with_addressable_count(
                std::sync::Arc::clone(&mmap),
                0,
                mapped_bytes,
                SYNTHETIC_EXPERTS,
                &[0],
                &[1, 1],
                &hot_records[..2],
            )
            .is_none()
        );
        assert!(
            Gemma4MoeSlotArgTable::from_mixed_active_slots_with_addressable_count(
                std::sync::Arc::clone(&mmap),
                0,
                mapped_bytes,
                SYNTHETIC_EXPERTS,
                &[0],
                &hot_slot_ids,
                &hot_records[..2],
            )
            .is_none()
        );
        assert!(
            Gemma4MoeSlotArgTable::from_mixed_active_slots_with_addressable_count(
                std::sync::Arc::clone(&mmap),
                0,
                mapped_bytes,
                SYNTHETIC_EXPERTS,
                &[0, SYNTHETIC_EXPERTS],
                &hot_slot_ids,
                &hot_records,
            )
            .is_none()
        );

        let split_receipt = Gemma4MoeSlotArgTable::from_mixed_active_slots_with_addressable_count(
            std::sync::Arc::clone(&mmap),
            0,
            mapped_bytes,
            SYNTHETIC_EXPERTS,
            &[1, 0, 1],
            &hot_slot_ids,
            &hot_records,
        )
        .expect("one-hot/one-cold deduplicated receipt");
        assert_eq!(split_receipt.bound_record_count(), 2);
        assert_eq!(split_receipt.hot_bound_record_count(), 1);
        assert_eq!(split_receipt.mapped_bound_record_count(), 1);
        assert_eq!(
            split_receipt.hot_bound_record_count() + split_receipt.mapped_bound_record_count(),
            split_receipt.bound_record_count()
        );
        drop(split_receipt);

        let table = Gemma4MoeSlotArgTable::from_mixed_active_slots_with_addressable_count(
            std::sync::Arc::clone(&mmap),
            0,
            mapped_bytes,
            SYNTHETIC_EXPERTS,
            &(0..SYNTHETIC_EXPERTS).collect::<Vec<_>>(),
            &hot_slot_ids,
            &hot_records,
        )
        .expect("mixed canonical-ID expert table");
        assert_eq!(table.addressable_slot_count(), SYNTHETIC_EXPERTS);
        assert_eq!(table.bound_record_count(), SYNTHETIC_EXPERTS);
        assert_eq!(table.hot_bound_record_count(), hot_slot_ids.len());
        assert_eq!(
            table.mapped_bound_record_count(),
            SYNTHETIC_EXPERTS - hot_slot_ids.len()
        );
        assert_eq!(
            table.hot_bound_record_count() + table.mapped_bound_record_count(),
            table.bound_record_count()
        );

        // The table is the lifetime-safe binding: its Objective-C references
        // retain selected anonymous records, and its Arc retains the mapping.
        // Prove both by dropping every source-side handle before GPU use.
        drop(mmap);
        drop(hot_records);
        drop(records);
        assert!(mapped_owner.upgrade().is_some());
        let retained_table = table.clone();
        drop(table);
        let table = retained_table;

        let copied = spec50_moe_pipelines(device).expect("copied-slab SPEC50 pipelines");
        let buffers = Buffers::new(device);
        for k in 1..=MAX_K {
            let routing = build_synthetic_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &routing.active_experts),
                Some(SYNTHETIC_EXPERTS)
            );

            encode_spec50_gateup(
                encoder,
                &copied.gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                &copied_slab,
                0,
                &buffers.work,
                &buffers.copy_scales,
                &buffers.copy_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
                None,
                None,
            );
            assert!(table.encode_chained_gateup_k8(
                encoder,
                &buffers.input_scales,
                &buffers.input_quants,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                SYNTHETIC_EXPERTS,
                k,
            ));
            encoder.memory_barrier_with_resources(&[
                &buffers.copy_scales,
                &buffers.copy_quants,
                &buffers.arg_scales,
                &buffers.arg_quants,
            ]);
            encode_spec50_down(
                encoder,
                &copied.down,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &copied_slab,
                0,
                &buffers.routes,
                &buffers.work,
                &buffers.copy_down,
                k as u32,
                None,
            );
            assert!(table.encode_chained_down_k8(
                encoder,
                &buffers.arg_scales,
                &buffers.arg_quants,
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "mixed hot/cold argbuf K={k} command failed: {}",
                command_buffer_error_details(cb)
            );

            let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
            let down_bytes = k * S50_HIDDEN * 4;
            assert_raw_eq(
                &format!("mixed hot/cold K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("mixed hot/cold K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );
            assert_raw_eq(
                &format!("mixed hot/cold K={k} Down output"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );
        }

        drop(table);
        assert!(
            mapped_owner.upgrade().is_none(),
            "dropping the final mixed table must release the mmap owner"
        );
    }

    #[test]
    fn gemma4_moe_three_prefix_command_buffers_are_raw_bit_exact_and_resource_disjoint() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP three-prefix argbuf parity gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP three-prefix argbuf parity gate: Tier-2 unavailable");
            return;
        }

        const HOT: usize = 3;
        const READY: usize = 2;
        const DEMAND: usize = SYNTHETIC_EXPERTS - HOT - READY;
        let (records, copied_slab) = synthetic_slot_backings(device);
        let ready_slab = new_buffer(device, READY * S50_SLOT_STRIDE);
        let demand_slab = new_buffer(device, DEMAND * S50_SLOT_STRIDE);
        for slot in 0..READY {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    records[HOT + slot].contents().cast::<u8>(),
                    ready_slab
                        .contents()
                        .cast::<u8>()
                        .add(slot * S50_SLOT_STRIDE),
                    S50_SLOT_STRIDE,
                );
            }
        }
        for slot in 0..DEMAND {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    records[HOT + READY + slot].contents().cast::<u8>(),
                    demand_slab
                        .contents()
                        .cast::<u8>()
                        .add(slot * S50_SLOT_STRIDE),
                    S50_SLOT_STRIDE,
                );
            }
        }

        let hot_records = records[..HOT].to_vec();
        let hot_offsets = vec![0; HOT];
        let mut ready_records = hot_records.clone();
        let mut ready_offsets = hot_offsets.clone();
        for slot in 0..READY {
            ready_records.push(ready_slab.clone());
            ready_offsets.push(slot * S50_SLOT_STRIDE);
        }
        let mut full_records = ready_records.clone();
        let mut full_offsets = ready_offsets.clone();
        for slot in 0..DEMAND {
            full_records.push(demand_slab.clone());
            full_offsets.push(slot * S50_SLOT_STRIDE);
        }
        let hot_slots = (0..HOT).collect::<Vec<_>>();
        let ready_slots = (0..HOT + READY).collect::<Vec<_>>();
        let all_slots = (0..SYNTHETIC_EXPERTS).collect::<Vec<_>>();
        let hot_table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            HOT,
            &hot_slots,
            &hot_records,
            &hot_offsets,
            0,
            None,
        )
        .expect("hot prefix table");
        let ready_table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            HOT + READY,
            &ready_slots,
            &ready_records,
            &ready_offsets,
            0,
            None,
        )
        .expect("hot+ready prefix table");
        let full_table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
            device,
            SYNTHETIC_EXPERTS,
            &all_slots,
            &full_records,
            &full_offsets,
            0,
            None,
        )
        .expect("full three-wave table");
        assert_eq!(hot_table.bound_record_count(), HOT);
        assert_eq!(ready_table.bound_record_count(), HOT + READY);
        assert_eq!(full_table.bound_record_count(), SYNTHETIC_EXPERTS);
        assert!(ready_table
            .records
            .iter()
            .all(|record| { !super::same_metal_buffer(record, &demand_slab) }));
        assert!(full_table
            .records
            .iter()
            .any(|record| { super::same_metal_buffer(record, &demand_slab) }));

        let copied = spec50_moe_pipelines(device).expect("copied-slab SPEC50 pipelines");
        let buffers = Buffers::new(device);
        for k in 1..=MAX_K {
            let routing = build_synthetic_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();

            let oracle = kernel.queue.new_command_buffer();
            let oracle_enc = oracle.new_compute_command_encoder();
            encode_spec50_gateup(
                oracle_enc,
                &copied.gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                &copied_slab,
                0,
                &buffers.work,
                &buffers.copy_scales,
                &buffers.copy_quants,
                SYNTHETIC_EXPERTS as u32,
                k as u32,
                None,
                None,
            );
            oracle_enc.memory_barrier_with_resources(&[&buffers.copy_scales, &buffers.copy_quants]);
            encode_spec50_down(
                oracle_enc,
                &copied.down,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &copied_slab,
                0,
                &buffers.routes,
                &buffers.work,
                &buffers.copy_down,
                k as u32,
                None,
            );
            oracle_enc.end_encoding();
            oracle.commit();

            let hot_cb = kernel.queue.new_command_buffer();
            let hot_enc = hot_cb.new_compute_command_encoder();
            assert_eq!(
                hot_table.declare_active_slots(hot_enc, &hot_slots),
                Some(HOT)
            );
            assert!(hot_table.encode_chained_gateup_k8_range(
                hot_enc,
                &buffers.input_scales,
                &buffers.input_quants,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                0,
                HOT,
                k,
            ));
            hot_enc.end_encoding();

            let ready_cb = kernel.queue.new_command_buffer();
            let ready_enc = ready_cb.new_compute_command_encoder();
            let ready_range = (HOT..HOT + READY).collect::<Vec<_>>();
            assert_eq!(
                ready_table.declare_active_slots(ready_enc, &ready_range),
                Some(READY)
            );
            assert!(ready_table.encode_chained_gateup_k8_range(
                ready_enc,
                &buffers.input_scales,
                &buffers.input_quants,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                HOT,
                READY,
                k,
            ));
            ready_enc.end_encoding();

            let final_cb = kernel.queue.new_command_buffer();
            let final_enc = final_cb.new_compute_command_encoder();
            let demand_range = (HOT + READY..SYNTHETIC_EXPERTS).collect::<Vec<_>>();
            assert_eq!(
                full_table.declare_active_slots(final_enc, &demand_range),
                Some(DEMAND)
            );
            assert!(full_table.encode_chained_gateup_k8_range(
                final_enc,
                &buffers.input_scales,
                &buffers.input_quants,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                HOT + READY,
                DEMAND,
                k,
            ));
            final_enc.memory_barrier_with_resources(&[&buffers.arg_scales, &buffers.arg_quants]);
            assert_eq!(
                full_table.declare_active_slots(final_enc, &all_slots),
                Some(SYNTHETIC_EXPERTS)
            );
            assert!(full_table.encode_chained_down_k8(
                final_enc,
                &buffers.arg_scales,
                &buffers.arg_quants,
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k,
            ));
            final_enc.end_encoding();

            hot_cb.commit();
            ready_cb.commit();
            final_cb.commit();
            final_cb.wait_until_completed();
            assert_eq!(hot_cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(ready_cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(
                final_cb.status(),
                MTLCommandBufferStatus::Completed,
                "three-prefix K={k} command failed: {}",
                command_buffer_error_details(final_cb)
            );

            let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
            let down_bytes = k * S50_HIDDEN * 4;
            assert_raw_eq(
                &format!("three-prefix K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("three-prefix K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );
            assert_raw_eq(
                &format!("three-prefix K={k} Down output"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );
        }
    }

    #[test]
    fn gemma4_moe_async_two_wave_terminal_is_raw_bit_exact_and_hot_resource_disjoint() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP async-two-wave argbuf parity gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP async-two-wave argbuf parity gate: Tier-2 unavailable");
            return;
        }

        const HOT: usize = 3;
        const COLD: usize = SYNTHETIC_EXPERTS - HOT;
        for (ready_count, demand_count, variant) in [
            (2, COLD - 2, "mixed"),
            (0, COLD, "ready-empty"),
            (COLD, 0, "demand-empty"),
        ] {
            assert_eq!(ready_count + demand_count, COLD);
            let (records, copied_slab) = synthetic_slot_backings(device);
            let ready_slab = new_buffer(device, ready_count.max(1) * S50_SLOT_STRIDE);
            let demand_slab = new_buffer(device, demand_count.max(1) * S50_SLOT_STRIDE);
            for slot in 0..ready_count {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        records[HOT + slot].contents().cast::<u8>(),
                        ready_slab
                            .contents()
                            .cast::<u8>()
                            .add(slot * S50_SLOT_STRIDE),
                        S50_SLOT_STRIDE,
                    );
                }
            }
            for slot in 0..demand_count {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        records[HOT + ready_count + slot].contents().cast::<u8>(),
                        demand_slab
                            .contents()
                            .cast::<u8>()
                            .add(slot * S50_SLOT_STRIDE),
                        S50_SLOT_STRIDE,
                    );
                }
            }

            let hot_records = records[..HOT].to_vec();
            let hot_offsets = vec![0; HOT];
            let mut full_records = hot_records.clone();
            let mut full_offsets = hot_offsets.clone();
            for slot in 0..ready_count {
                full_records.push(ready_slab.clone());
                full_offsets.push(slot * S50_SLOT_STRIDE);
            }
            for slot in 0..demand_count {
                full_records.push(demand_slab.clone());
                full_offsets.push(slot * S50_SLOT_STRIDE);
            }
            let hot_slots = (0..HOT).collect::<Vec<_>>();
            let all_slots = (0..SYNTHETIC_EXPERTS).collect::<Vec<_>>();
            let cold_slots = (HOT..SYNTHETIC_EXPERTS).collect::<Vec<_>>();
            let hot_table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
                device,
                HOT,
                &hot_slots,
                &hot_records,
                &hot_offsets,
                0,
                None,
            )
            .expect("H55 hot-only table");
            let full_table = Gemma4MoeSlotArgTable::from_indexed_slot_buffer_offsets(
                device,
                SYNTHETIC_EXPERTS,
                &all_slots,
                &full_records,
                &full_offsets,
                0,
                None,
            )
            .expect("H55 full terminal table");
            assert_eq!(hot_table.bound_record_count(), HOT);
            assert_eq!(full_table.bound_record_count(), SYNTHETIC_EXPERTS);
            assert!(hot_table.records.iter().all(|record| {
                !super::same_metal_buffer(record, &ready_slab)
                    && !super::same_metal_buffer(record, &demand_slab)
            }));
            assert_eq!(
                full_table
                    .records
                    .iter()
                    .any(|record| super::same_metal_buffer(record, &ready_slab)),
                ready_count > 0
            );
            assert_eq!(
                full_table
                    .records
                    .iter()
                    .any(|record| super::same_metal_buffer(record, &demand_slab)),
                demand_count > 0
            );

            let copied = spec50_moe_pipelines(device).expect("copied-slab SPEC50 pipelines");
            let buffers = Buffers::new(device);
            for k in 1..=MAX_K {
                let routing = build_synthetic_routing(k);
                buffers.upload(&routing);
                buffers.zero_outputs();

                let oracle = kernel.queue.new_command_buffer();
                let oracle_enc = oracle.new_compute_command_encoder();
                encode_spec50_gateup(
                    oracle_enc,
                    &copied.gateup,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    &copied_slab,
                    0,
                    &buffers.work,
                    &buffers.copy_scales,
                    &buffers.copy_quants,
                    SYNTHETIC_EXPERTS as u32,
                    k as u32,
                    None,
                    None,
                );
                oracle_enc
                    .memory_barrier_with_resources(&[&buffers.copy_scales, &buffers.copy_quants]);
                encode_spec50_down(
                    oracle_enc,
                    &copied.down,
                    &buffers.copy_scales,
                    &buffers.copy_quants,
                    &copied_slab,
                    0,
                    &buffers.routes,
                    &buffers.work,
                    &buffers.copy_down,
                    k as u32,
                    None,
                );
                oracle_enc.end_encoding();
                oracle.commit();

                let hot_cb = kernel.queue.new_command_buffer();
                let hot_enc = hot_cb.new_compute_command_encoder();
                assert_eq!(
                    hot_table.declare_active_slots(hot_enc, &hot_slots),
                    Some(HOT)
                );
                assert!(hot_table.encode_chained_gateup_k8_range(
                    hot_enc,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    &buffers.work,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    0,
                    HOT,
                    k,
                ));
                hot_enc.end_encoding();

                let terminal_cb = kernel.queue.new_command_buffer();
                let terminal_enc = terminal_cb.new_compute_command_encoder();
                assert_eq!(
                    full_table.declare_active_slots(terminal_enc, &cold_slots),
                    Some(ready_count + demand_count)
                );
                assert!(full_table.encode_chained_gateup_k8_range(
                    terminal_enc,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    &buffers.work,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    HOT,
                    ready_count + demand_count,
                    k,
                ));
                terminal_enc
                    .memory_barrier_with_resources(&[&buffers.arg_scales, &buffers.arg_quants]);
                assert_eq!(
                    full_table.declare_active_slots(terminal_enc, &all_slots),
                    Some(SYNTHETIC_EXPERTS)
                );
                assert!(full_table.encode_chained_down_k8(
                    terminal_enc,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    &buffers.routes,
                    &buffers.work,
                    &buffers.arg_down,
                    k,
                ));
                terminal_enc.end_encoding();

                hot_cb.commit();
                terminal_cb.commit();
                terminal_cb.wait_until_completed();
                assert_eq!(hot_cb.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(
                    terminal_cb.status(),
                    MTLCommandBufferStatus::Completed,
                    "async-two-wave {variant} K={k} command failed: {}",
                    command_buffer_error_details(terminal_cb)
                );

                let scale_bytes = SYNTHETIC_EXPERTS * k * S50_DOWN_BLOCKS * 4;
                let quant_bytes = SYNTHETIC_EXPERTS * k * S50_FF;
                let down_bytes = k * S50_HIDDEN * 4;
                assert_raw_eq(
                    &format!("async-two-wave {variant} K={k} GateUp scales"),
                    &read_bytes(&buffers.copy_scales, scale_bytes),
                    &read_bytes(&buffers.arg_scales, scale_bytes),
                );
                assert_raw_eq(
                    &format!("async-two-wave {variant} K={k} GateUp quants"),
                    &read_bytes(&buffers.copy_quants, quant_bytes),
                    &read_bytes(&buffers.arg_quants, quant_bytes),
                );
                assert_raw_eq(
                    &format!("async-two-wave {variant} K={k} Down output"),
                    &read_bytes(&buffers.copy_down, down_bytes),
                    &read_bytes(&buffers.arg_down, down_bytes),
                );
            }
        }
    }

    #[test]
    fn gemma4_moe_slot_arg_table_head_k1_all_modes_raw_bit_parity() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP anonymous HEAD argbuf parity gate: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!(
                "SKIP anonymous HEAD argbuf parity gate: Tier-2 argument buffers unavailable"
            );
            return;
        }
        let (records, copied_slab) = synthetic_slot_backings(device);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("anonymous per-slot Tier-2 table");
        let route_slots_values = [7u32, 0, 5, 2, 6, 1, 4, 3];
        let route_slots = new_buffer(device, std::mem::size_of_val(&route_slots_values));
        write_bytes(&route_slots, &route_slots_values);
        let declared_with_duplicates = [7usize, 0, 5, 2, 6, 1, 4, 3, 7, 2, 0];

        let mut rng = Rng::new(0x51a7_9e11);
        let input_scales_values: Vec<f32> = (0..GEMMA4_Q4_EXPERT_INPUT_BLOCKS)
            .map(|_| 0.0005 + rng.next_f32() * 0.01)
            .collect();
        let input_quants_values: Vec<i8> = (0..GEMMA4_Q4_EXPERT_HIDDEN)
            .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
            .collect();
        let input_scales = new_buffer(device, std::mem::size_of_val(&input_scales_values[..]));
        let input_quants = new_buffer(device, std::mem::size_of_val(&input_quants_values[..]));
        write_bytes(&input_scales, &input_scales_values);
        write_bytes(&input_quants, &input_quants_values);
        let gate_output_bytes =
            GEMMA4_Q4_EXPERT_ROUTES * 2 * GEMMA4_Q4_EXPERT_FF * std::mem::size_of::<f32>();
        let copied_gate = new_buffer(device, gate_output_bytes);
        let arg_gate = new_buffer(device, gate_output_bytes);

        for mode in [
            Gemma4HeadArgbufGateMode::Split,
            Gemma4HeadArgbufGateMode::FusedScalar,
            Gemma4HeadArgbufGateMode::FusedOrderedSimd,
            Gemma4HeadArgbufGateMode::FusedTurbo,
        ] {
            if shipping_head_gateup_pipeline(kernel, mode).is_none() {
                continue;
            }
            fill_zero(&copied_gate);
            fill_zero(&arg_gate);
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &declared_with_duplicates),
                Some(SYNTHETIC_EXPERTS)
            );
            assert!(encode_shipping_head_gateup(
                kernel,
                encoder,
                &input_scales,
                &input_quants,
                &copied_slab,
                &route_slots,
                &copied_gate,
                mode,
            ));
            assert!(table.encode_head_gateup(
                encoder,
                &input_scales,
                &input_quants,
                &route_slots,
                &arg_gate,
                mode,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "anonymous HEAD GateUp {mode:?} failed: {}",
                command_buffer_error_details(cb)
            );
            let output_bytes = if mode == Gemma4HeadArgbufGateMode::Split {
                gate_output_bytes
            } else {
                GEMMA4_Q4_EXPERT_ROUTES * GEMMA4_Q4_EXPERT_FF * std::mem::size_of::<f32>()
            };
            assert_raw_eq(
                &format!("anonymous HEAD GateUp {mode:?}"),
                &read_bytes(&copied_gate, output_bytes),
                &read_bytes(&arg_gate, output_bytes),
            );
        }

        let activation_scales_values: Vec<f32> = (0..GEMMA4_Q4_EXPERT_ROUTES
            * GEMMA4_Q4_EXPERT_ACT_BLOCKS)
            .map(|_| 0.0005 + rng.next_f32() * 0.01)
            .collect();
        let activation_quants_values: Vec<i8> = (0..GEMMA4_Q4_EXPERT_ROUTES * GEMMA4_Q4_EXPERT_FF)
            .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
            .collect();
        let route_scales_values = [0.07f32, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31];
        let activation_scales =
            new_buffer(device, std::mem::size_of_val(&activation_scales_values[..]));
        let activation_quants =
            new_buffer(device, std::mem::size_of_val(&activation_quants_values[..]));
        let route_scales = new_buffer(device, std::mem::size_of_val(&route_scales_values));
        write_bytes(&activation_scales, &activation_scales_values);
        write_bytes(&activation_quants, &activation_quants_values);
        write_bytes(&route_scales, &route_scales_values);
        let down_output_bytes = GEMMA4_Q4_EXPERT_HIDDEN * std::mem::size_of::<f32>();
        let copied_down = new_buffer(device, down_output_bytes);
        let arg_down = new_buffer(device, down_output_bytes);

        for mode in [
            Gemma4HeadArgbufDownMode::Scalar,
            Gemma4HeadArgbufDownMode::OrderedSimd,
            Gemma4HeadArgbufDownMode::Turbo,
        ] {
            if shipping_head_down_pipeline(kernel, mode).is_none() {
                continue;
            }
            fill_zero(&copied_down);
            fill_zero(&arg_down);
            let cb = kernel.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            assert_eq!(
                table.declare_active_slots(encoder, &declared_with_duplicates),
                Some(SYNTHETIC_EXPERTS)
            );
            assert!(encode_shipping_head_down(
                kernel,
                encoder,
                &activation_scales,
                &activation_quants,
                &copied_slab,
                &route_slots,
                &route_scales,
                &copied_down,
                mode,
            ));
            assert!(table.encode_head_down(
                encoder,
                &activation_scales,
                &activation_quants,
                &route_slots,
                &route_scales,
                &arg_down,
                mode,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "anonymous HEAD Down {mode:?} failed: {}",
                command_buffer_error_details(cb)
            );
            assert_raw_eq(
                &format!("anonymous HEAD Down {mode:?}"),
                &read_bytes(&copied_down, down_output_bytes),
                &read_bytes(&arg_down, down_output_bytes),
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ResourceDeclaration {
        Active,
        All,
        None,
    }

    impl ResourceDeclaration {
        fn from_env() -> Self {
            match std::env::var("CAMELID_GEMMA4_ARGBUF_RESOURCE_MODE").as_deref() {
                Ok("all") => Self::All,
                Ok("none") => Self::None,
                Ok("active") | Err(_) => Self::Active,
                Ok(value) => panic!(
                    "CAMELID_GEMMA4_ARGBUF_RESOURCE_MODE must be active, all, or none; got {value:?}"
                ),
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::Active => "active",
                Self::All => "all",
                Self::None => "none",
            }
        }
    }

    fn declare_expert_records(
        encoder: &metal::ComputeCommandEncoderRef,
        records: &[Buffer],
        active_experts: &[usize],
        declaration: ResourceDeclaration,
    ) {
        match declaration {
            ResourceDeclaration::Active => {
                for &expert_id in active_experts {
                    encoder.use_resource(&records[expert_id], MTLResourceUsage::Read);
                }
            }
            ResourceDeclaration::All => {
                for record in records {
                    encoder.use_resource(record, MTLResourceUsage::Read);
                }
            }
            ResourceDeclaration::None => {}
        }
    }

    #[derive(Clone, Copy)]
    struct VmCounters {
        pageins: u64,
        faults: u64,
        decompressions: u64,
        swapins: u64,
    }

    #[allow(deprecated)]
    fn vm_counters() -> Option<VmCounters> {
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
        Some(VmCounters {
            pageins: stats.pageins,
            faults: stats.faults,
            decompressions: stats.decompressions,
            swapins: stats.swapins,
        })
    }

    fn vm_delta(
        after: Option<VmCounters>,
        before: Option<VmCounters>,
        field: fn(VmCounters) -> u64,
    ) -> u64 {
        after.zip(before).map_or(0, |(after, before)| {
            field(after).saturating_sub(field(before))
        })
    }

    #[derive(Debug, Clone, Copy)]
    struct Timing {
        wall_us: u128,
        gpu_us: u128,
        kernel_us: u128,
        pageins: u64,
        faults: u64,
        decompressions: u64,
        swapins: u64,
    }

    #[allow(clippy::too_many_arguments)]
    fn time_gateup_layers(
        queue: &metal::CommandQueue,
        pipeline: &ComputePipelineState,
        candidate_tile: usize,
        table: &Gemma4MoeSlotArgTable,
        buffers: &Buffers,
        routing: &Routing,
        k: usize,
        layers: usize,
    ) -> Timing {
        buffers.upload(routing);
        buffers.zero_outputs();
        let before = vm_counters();
        let started = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(
            table.declare_active_slots(encoder, &routing.active_experts),
            Some(routing.active_experts.len())
        );
        for _ in 0..layers {
            match candidate_tile {
                1 => encode_argbuf_gateup_candidate(
                    encoder,
                    pipeline,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    table.argument_buffer(),
                    &buffers.work,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    routing.active_experts.len() as u32,
                    k as u32,
                ),
                4 => encode_argbuf_gateup_tile4(
                    encoder,
                    pipeline,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    table.argument_buffer(),
                    &buffers.work,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    routing.active_experts.len() as u32,
                    k as u32,
                ),
                0 => encode_argbuf_gateup(
                    encoder,
                    pipeline,
                    &buffers.input_scales,
                    &buffers.input_quants,
                    table.argument_buffer(),
                    &buffers.work,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    routing.active_experts.len() as u32,
                    k as u32,
                ),
                _ => panic!("unsupported GateUp candidate tile {candidate_tile}"),
            }
        }
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_us = started.elapsed().as_micros();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "GateUp benchmark command failed: {}",
            command_buffer_error_details(cb)
        );
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(cb);
        let after = vm_counters();
        Timing {
            wall_us,
            gpu_us,
            kernel_us,
            pageins: vm_delta(after, before, |stats| stats.pageins),
            faults: vm_delta(after, before, |stats| stats.faults),
            decompressions: vm_delta(after, before, |stats| stats.decompressions),
            swapins: vm_delta(after, before, |stats| stats.swapins),
        }
    }

    fn prepare_down_activations(
        queue: &metal::CommandQueue,
        pipelines: &Spec50MoeArgbufKernels,
        table: &Gemma4MoeSlotArgTable,
        buffers: &Buffers,
        routing: &Routing,
        k: usize,
    ) {
        buffers.upload(routing);
        buffers.zero_outputs();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(
            table.declare_active_slots(encoder, &routing.active_experts),
            Some(routing.active_experts.len())
        );
        let gateup = pipelines.gateup_k8.as_ref().unwrap_or(&pipelines.gateup);
        encode_argbuf_gateup(
            encoder,
            gateup,
            &buffers.input_scales,
            &buffers.input_quants,
            table.argument_buffer(),
            &buffers.work,
            &buffers.arg_scales,
            &buffers.arg_quants,
            routing.active_experts.len() as u32,
            k as u32,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "Down activation preparation failed: {}",
            command_buffer_error_details(cb)
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn time_down_layers(
        queue: &metal::CommandQueue,
        pipeline: &ComputePipelineState,
        schedule: usize,
        table: &Gemma4MoeSlotArgTable,
        buffers: &Buffers,
        routing: &Routing,
        k: usize,
        layers: usize,
    ) -> Timing {
        buffers.upload(routing);
        fill_zero(&buffers.arg_down);
        let before = vm_counters();
        let started = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        assert_eq!(
            table.declare_active_slots(encoder, &routing.active_experts),
            Some(routing.active_experts.len())
        );
        for _ in 0..layers {
            match schedule {
                8 => encode_argbuf_down_rows8(
                    encoder,
                    pipeline,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    table.argument_buffer(),
                    &buffers.routes,
                    &buffers.work,
                    &buffers.arg_down,
                    k as u32,
                ),
                4 => encode_argbuf_down_tg4(
                    encoder,
                    pipeline,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    table.argument_buffer(),
                    &buffers.routes,
                    &buffers.work,
                    &buffers.arg_down,
                    k as u32,
                ),
                0 => encode_argbuf_down(
                    encoder,
                    pipeline,
                    &buffers.arg_scales,
                    &buffers.arg_quants,
                    table.argument_buffer(),
                    &buffers.routes,
                    &buffers.work,
                    &buffers.arg_down,
                    k as u32,
                ),
                _ => panic!("unsupported Down schedule {schedule}"),
            }
        }
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_us = started.elapsed().as_micros();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "Down benchmark command failed: {}",
            command_buffer_error_details(cb)
        );
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(cb);
        let after = vm_counters();
        Timing {
            wall_us,
            gpu_us,
            kernel_us,
            pageins: vm_delta(after, before, |stats| stats.pageins),
            faults: vm_delta(after, before, |stats| stats.faults),
            decompressions: vm_delta(after, before, |stats| stats.decompressions),
            swapins: vm_delta(after, before, |stats| stats.swapins),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn time_argbuf_round(
        queue: &metal::CommandQueue,
        pipelines: &Spec50MoeArgbufKernels,
        table: &Buffer,
        records: &[Buffer],
        buffers: &Buffers,
        routing: &Routing,
        k: usize,
        declaration: ResourceDeclaration,
    ) -> Timing {
        buffers.upload(routing);
        buffers.zero_outputs();
        let before = vm_counters();
        let started = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        declare_expert_records(encoder, records, &routing.active_experts, declaration);
        encode_argbuf_gateup(
            encoder,
            &pipelines.gateup,
            &buffers.input_scales,
            &buffers.input_quants,
            table,
            &buffers.work,
            &buffers.arg_scales,
            &buffers.arg_quants,
            routing.active_experts.len() as u32,
            k as u32,
        );
        encoder.memory_barrier_with_resources(&[&buffers.arg_scales, &buffers.arg_quants]);
        encode_argbuf_down(
            encoder,
            &pipelines.down,
            &buffers.arg_scales,
            &buffers.arg_quants,
            table,
            &buffers.routes,
            &buffers.work,
            &buffers.arg_down,
            k as u32,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_us = started.elapsed().as_micros();
        if matches!(declaration, ResourceDeclaration::None) {
            eprintln!(
                "[spec50-argbuf] no-useResource command status={:?} details={}",
                cb.status(),
                command_buffer_error_details(cb),
            );
        }
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "argbuf MoE command failed: {}",
            command_buffer_error_details(cb)
        );
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(cb);
        let after = vm_counters();
        Timing {
            wall_us,
            gpu_us,
            kernel_us,
            pageins: vm_delta(after, before, |stats| stats.pageins),
            faults: vm_delta(after, before, |stats| stats.faults),
            decompressions: vm_delta(after, before, |stats| stats.decompressions),
            swapins: vm_delta(after, before, |stats| stats.swapins),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn time_copied_round(
        queue: &metal::CommandQueue,
        pipelines: &super::super::spec50_moe::Spec50MoeVariant,
        copied_slab: &Buffer,
        buffers: &Buffers,
        routing: &Routing,
        k: usize,
    ) -> Timing {
        buffers.upload(routing);
        buffers.zero_outputs();
        let before = vm_counters();
        let started = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encode_spec50_gateup(
            encoder,
            &pipelines.gateup,
            &buffers.input_scales,
            &buffers.input_quants,
            copied_slab,
            0,
            &buffers.work,
            &buffers.copy_scales,
            &buffers.copy_quants,
            routing.active_experts.len() as u32,
            k as u32,
            None,
            None,
        );
        encoder.memory_barrier_with_resources(&[&buffers.copy_scales, &buffers.copy_quants]);
        encode_spec50_down(
            encoder,
            &pipelines.down,
            &buffers.copy_scales,
            &buffers.copy_quants,
            copied_slab,
            0,
            &buffers.routes,
            &buffers.work,
            &buffers.copy_down,
            k as u32,
            None,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_us = started.elapsed().as_micros();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "copied-slab MoE command failed: {}",
            command_buffer_error_details(cb)
        );
        let (gpu_us, kernel_us) = command_buffer_gpu_times_us(cb);
        let after = vm_counters();
        Timing {
            wall_us,
            gpu_us,
            kernel_us,
            pageins: vm_delta(after, before, |stats| stats.pageins),
            faults: vm_delta(after, before, |stats| stats.faults),
            decompressions: vm_delta(after, before, |stats| stats.decompressions),
            swapins: vm_delta(after, before, |stats| stats.swapins),
        }
    }

    /// Warm, real-Metal comparison shaped like one complete 30-layer target
    /// pass: K=8, 30 unique experts, and 30 consecutive GateUp dispatches.
    /// The explicit ignore keeps a timing-sensitive result out of ordinary CI.
    #[test]
    #[ignore = "real-Metal 30-layer GateUp microbenchmark"]
    fn spec50_moe_candidate_sliced_gateup_30_layer_microbenchmark() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP candidate-sliced GateUp benchmark: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP candidate-sliced GateUp benchmark: Tier-2 unavailable");
            return;
        }
        let pipelines = spec50_moe_argbuf_kernels(device).expect("argument-buffer MoE pipelines");
        let fast = pipelines
            .gateup_k8_candidate
            .as_ref()
            .expect("candidate-sliced K8 GateUp pipeline");
        let tile4 = pipelines
            .gateup_k8_tile4
            .as_ref()
            .expect("four-candidate tiled K8 GateUp pipeline");
        let current = pipelines.gateup_k8.as_ref().unwrap_or(&pipelines.gateup);
        let (records, _) = synthetic_slot_backings_count(device, ACTIVE_EXPERTS);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("30-expert anonymous Tier-2 table");
        let buffers = Buffers::new(device);
        let routing = build_routing(MAX_K);
        assert_eq!(routing.active_experts.len(), ACTIVE_EXPERTS);

        // Discard one pipeline warm-up of each shape, then alternate to reduce
        // thermal/frequency bias. The middle GPU sample is the headline.
        let _ = time_gateup_layers(
            &kernel.queue,
            current,
            0,
            &table,
            &buffers,
            &routing,
            MAX_K,
            30,
        );
        let _ = time_gateup_layers(
            &kernel.queue,
            fast,
            1,
            &table,
            &buffers,
            &routing,
            MAX_K,
            30,
        );
        let _ = time_gateup_layers(
            &kernel.queue,
            tile4,
            4,
            &table,
            &buffers,
            &routing,
            MAX_K,
            30,
        );
        let mut current_samples = Vec::with_capacity(7);
        let mut fast_samples = Vec::with_capacity(7);
        let mut tile4_samples = Vec::with_capacity(7);
        for _ in 0..7 {
            current_samples.push(time_gateup_layers(
                &kernel.queue,
                current,
                0,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
            fast_samples.push(time_gateup_layers(
                &kernel.queue,
                fast,
                1,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
            tile4_samples.push(time_gateup_layers(
                &kernel.queue,
                tile4,
                4,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
        }
        let median_gpu = |samples: &[Timing]| {
            let mut ordered = samples.to_vec();
            ordered.sort_unstable_by_key(|timing| timing.gpu_us);
            ordered[ordered.len() / 2]
        };
        let current_median = median_gpu(&current_samples);
        let fast_median = median_gpu(&fast_samples);
        let tile4_median = median_gpu(&tile4_samples);
        let candidate_speedup = current_median.gpu_us as f64 / fast_median.gpu_us as f64;
        let tile4_speedup = current_median.gpu_us as f64 / tile4_median.gpu_us as f64;
        eprintln!(
            "[spec50-fast-moe-gateup] K=8 U=30 layers=30 current_gpu_us={} candidate_gpu_us={} candidate_speedup={candidate_speedup:.4} tile4_gpu_us={} tile4_speedup={tile4_speedup:.4}",
            current_median.gpu_us,
            fast_median.gpu_us,
            tile4_median.gpu_us,
        );
        for (sample, ((current, fast), tile4)) in current_samples
            .iter()
            .zip(&fast_samples)
            .zip(&tile4_samples)
            .enumerate()
        {
            eprintln!(
                "[spec50-fast-moe-gateup] sample={sample} current_wall_us={} current_gpu_us={} candidate_wall_us={} candidate_gpu_us={} tile4_wall_us={} tile4_gpu_us={}",
                current.wall_us,
                current.gpu_us,
                fast.wall_us,
                fast.gpu_us,
                tile4.wall_us,
                tile4.gpu_us,
            );
        }
    }

    /// Warm, real-Metal K8/U30 comparison of one complete 30-layer Down stage.
    /// Kept ignored because GPU timings are machine- and thermal-state-specific.
    #[test]
    #[ignore = "real-Metal 30-layer Down scheduling microbenchmark"]
    fn spec50_moe_down_schedule_30_layer_microbenchmark() {
        let Some(kernel) = metal_linear_kernel() else {
            eprintln!("SKIP Down schedule benchmark: no Metal device");
            return;
        };
        let device = &kernel.device;
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            eprintln!("SKIP Down schedule benchmark: Tier-2 unavailable");
            return;
        }
        let pipelines = spec50_moe_argbuf_kernels(device).expect("argument-buffer MoE pipelines");
        let rows8 = pipelines
            .down_rows8
            .as_ref()
            .expect("eight-row Down pipeline");
        let tg4 = pipelines
            .down_tg4
            .as_ref()
            .expect("four-candidate threadgroup Down pipeline");
        let (records, _) = synthetic_slot_backings_count(device, ACTIVE_EXPERTS);
        let table = Gemma4MoeSlotArgTable::from_slot_buffers(device, &records)
            .expect("30-expert anonymous Tier-2 table");
        let buffers = Buffers::new(device);
        let routing = build_routing(MAX_K);
        assert_eq!(routing.active_experts.len(), ACTIVE_EXPERTS);
        prepare_down_activations(&kernel.queue, pipelines, &table, &buffers, &routing, MAX_K);

        for (pipeline, schedule) in [(&pipelines.down, 0), (rows8, 8), (tg4, 4)] {
            let _ = time_down_layers(
                &kernel.queue,
                pipeline,
                schedule,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            );
        }
        let mut current_samples = Vec::with_capacity(7);
        let mut rows8_samples = Vec::with_capacity(7);
        let mut tg4_samples = Vec::with_capacity(7);
        for _ in 0..7 {
            current_samples.push(time_down_layers(
                &kernel.queue,
                &pipelines.down,
                0,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
            rows8_samples.push(time_down_layers(
                &kernel.queue,
                rows8,
                8,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
            tg4_samples.push(time_down_layers(
                &kernel.queue,
                tg4,
                4,
                &table,
                &buffers,
                &routing,
                MAX_K,
                30,
            ));
        }
        let median_gpu = |samples: &[Timing]| {
            let mut ordered = samples.to_vec();
            ordered.sort_unstable_by_key(|timing| timing.gpu_us);
            ordered[ordered.len() / 2]
        };
        let current_median = median_gpu(&current_samples);
        let rows8_median = median_gpu(&rows8_samples);
        let tg4_median = median_gpu(&tg4_samples);
        let rows8_speedup = current_median.gpu_us as f64 / rows8_median.gpu_us as f64;
        let tg4_speedup = current_median.gpu_us as f64 / tg4_median.gpu_us as f64;
        eprintln!(
            "[spec50-fast-moe-down] K=8 U=30 layers=30 current_gpu_us={} rows8_gpu_us={} rows8_speedup={rows8_speedup:.4} tg4_gpu_us={} tg4_speedup={tg4_speedup:.4}",
            current_median.gpu_us,
            rows8_median.gpu_us,
            tg4_median.gpu_us,
        );
        for (sample, ((current, rows8), tg4)) in current_samples
            .iter()
            .zip(&rows8_samples)
            .zip(&tg4_samples)
            .enumerate()
        {
            eprintln!(
                "[spec50-fast-moe-down] sample={sample} current_gpu_us={} rows8_gpu_us={} tg4_gpu_us={} current_wall_us={} rows8_wall_us={} tg4_wall_us={}",
                current.gpu_us,
                rows8.gpu_us,
                tg4.gpu_us,
                current.wall_us,
                rows8.wall_us,
                tg4.wall_us,
            );
        }
    }

    /// Bound-check the page-residency probe against one real mapped layer.
    /// The count is deliberately not asserted to be cold or warm: residency
    /// is external state, while the invariant needed by shadow accounting is
    /// simply that every record reports within its 205-page stride.
    #[test]
    #[ignore = "requires the local Gemma 4 26B .cghost and a Tier-2 Metal device"]
    fn spec50_moe_argbuf_real_mapping_mincore_bounds() {
        if std::env::var("CAMELID_GEMMA4_ARGBUF_MOE_TEST").as_deref() != Ok("1") {
            eprintln!("SKIP argbuf mincore gate: set CAMELID_GEMMA4_ARGBUF_MOE_TEST=1");
            return;
        }
        if !detect_metal_device().available {
            eprintln!("SKIP argbuf mincore gate: no Metal device");
            return;
        }
        let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost")
            });
        if !cghost_path.is_file() {
            eprintln!(
                "SKIP argbuf mincore gate: {} not found",
                cghost_path.display()
            );
            return;
        }
        let layer_idx = std::env::var("CAMELID_GEMMA4_ARGBUF_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(26);
        let ghost = crate::ghost::GhostFile::open(&cghost_path).expect("open .cghost");
        let (mmap, layer_offset, layer_bytes) = ghost
            .mapped_moe_layer_slab(layer_idx, EXPERTS, S50_RECORD_BYTES, S50_SLOT_STRIDE)
            .expect("validate mapped expert layer")
            .expect("normal GhostFile open retains mmap");
        let layer = Spec50MoeArgbufLayer::new(mmap, layer_offset, layer_bytes)
            .expect("construct record-granular mapped layer");
        let nonresident = layer
            .mapped_nonresident_pages_by_expert()
            .expect("mincore mapped layer");
        let pages_per_expert = S50_SLOT_STRIDE / crate::wire_mmap::page_size();
        assert_eq!(pages_per_expert, 205);
        assert!(
            nonresident
                .iter()
                .all(|&pages| usize::from(pages) <= pages_per_expert),
            "every expert count must fit its mapped stride"
        );
    }

    /// Real-model admission gate for replacing anonymous copied expert slots
    /// with a static Tier-2 pointer table. It remains ignored and env-gated
    /// until a production integration is explicitly approved.
    #[test]
    #[ignore = "requires the local Gemma 4 26B .cghost and performs a cold paging benchmark"]
    fn spec50_moe_argbuf_real_layer_bitwise_and_paging_gate() {
        if std::env::var("CAMELID_GEMMA4_ARGBUF_MOE_TEST").as_deref() != Ok("1") {
            eprintln!("SKIP argbuf MoE gate: set CAMELID_GEMMA4_ARGBUF_MOE_TEST=1");
            return;
        }
        if !detect_metal_device().available {
            eprintln!("SKIP argbuf MoE gate: no Metal device");
            return;
        }
        let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost")
            });
        if !cghost_path.is_file() {
            eprintln!("SKIP argbuf MoE gate: {} not found", cghost_path.display());
            return;
        }
        let layer_idx = std::env::var("CAMELID_GEMMA4_ARGBUF_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(26);

        let ghost = crate::ghost::GhostFile::open(&cghost_path).expect("open .cghost");
        let (mmap, layer_offset, layer_bytes) = ghost
            .mapped_moe_layer_slab(layer_idx, EXPERTS, S50_RECORD_BYTES, S50_SLOT_STRIDE)
            .expect("validate mapped expert layer")
            .expect("normal GhostFile open retains mmap");
        assert_eq!(layer_bytes, EXPERTS * S50_SLOT_STRIDE);

        let kernel = metal_linear_kernel().expect("Metal kernel");
        let device = kernel.device.clone();
        let queue = kernel.queue.clone();
        assert_eq!(
            device.argument_buffers_support(),
            MTLArgumentBuffersTier::Tier2,
            "expert pointer table requires Tier-2 argument buffers"
        );
        let argbuf = spec50_moe_argbuf_kernels(&device).expect("argbuf MoE pipelines");
        let copied = spec50_moe_pipelines(&device).expect("copied-slab MoE pipelines");
        let buffers = Buffers::new(&device);
        let resource_declaration = ResourceDeclaration::from_env();

        let layer_base = usize::try_from(layer_offset).expect("layer offset fits usize");
        let mut record_buffers = Vec::with_capacity(EXPERTS);
        for expert_id in 0..EXPERTS {
            let record_offset = expert_id * S50_SLOT_STRIDE;
            let pointer = unsafe { mmap.base_ptr().add(layer_base + record_offset) };
            assert_eq!(pointer as usize % crate::wire_mmap::page_size(), 0);
            record_buffers.push(device.new_buffer_with_bytes_no_copy(
                pointer.cast(),
                S50_SLOT_STRIDE as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            ));
        }
        let table_started = std::time::Instant::now();
        let expert_table =
            new_static_expert_table(&device, argbuf, &record_buffers).expect("static expert table");
        let table_us = table_started.elapsed().as_micros();

        let routing_k8 = build_routing(MAX_K);
        assert_eq!(routing_k8.active_experts.len(), ACTIVE_EXPERTS);

        // Run the file-backed variant before allocating/copying the anonymous
        // slab so the K8 measurement preserves the layer's cold residency.
        let arg_cold = time_argbuf_round(
            &queue,
            argbuf,
            &expert_table,
            &record_buffers,
            &buffers,
            &routing_k8,
            MAX_K,
            resource_declaration,
        );
        eprintln!(
            "[spec50-argbuf] immediate arg-cold resource_declaration={}: wall_us={} gpu_us={} \
             kernel_us={} pageins={} faults={} decompressions={} swapins={}",
            resource_declaration.label(),
            arg_cold.wall_us,
            arg_cold.gpu_us,
            arg_cold.kernel_us,
            arg_cold.pageins,
            arg_cold.faults,
            arg_cold.decompressions,
            arg_cold.swapins,
        );

        let copied_slab = new_buffer(&device, ACTIVE_EXPERTS * S50_SLOT_STRIDE);
        let copy_started = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                mmap.base_ptr().add(layer_base),
                copied_slab.contents().cast::<u8>(),
                ACTIVE_EXPERTS * S50_SLOT_STRIDE,
            );
        }
        let anonymous_copy_us = copy_started.elapsed().as_micros();

        let copy_first =
            time_copied_round(&queue, copied, &copied_slab, &buffers, &routing_k8, MAX_K);
        // Compare each K with a shared copied-GateUp activation for Down, so a
        // GateUp defect cannot hide or manufacture a Down equality.
        for k in 1..=MAX_K {
            let routing = build_routing(k);
            buffers.upload(&routing);
            buffers.zero_outputs();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            declare_expert_records(
                encoder,
                &record_buffers,
                &routing.active_experts,
                resource_declaration,
            );
            encode_spec50_gateup(
                encoder,
                &copied.gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                &copied_slab,
                0,
                &buffers.work,
                &buffers.copy_scales,
                &buffers.copy_quants,
                routing.active_experts.len() as u32,
                k as u32,
                None,
                None,
            );
            encode_argbuf_gateup(
                encoder,
                &argbuf.gateup,
                &buffers.input_scales,
                &buffers.input_quants,
                &expert_table,
                &buffers.work,
                &buffers.arg_scales,
                &buffers.arg_quants,
                routing.active_experts.len() as u32,
                k as u32,
            );
            encoder.memory_barrier_with_resources(&[
                &buffers.copy_scales,
                &buffers.copy_quants,
                &buffers.arg_scales,
                &buffers.arg_quants,
            ]);
            encode_spec50_down(
                encoder,
                &copied.down,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &copied_slab,
                0,
                &buffers.routes,
                &buffers.work,
                &buffers.copy_down,
                k as u32,
                None,
            );
            encode_argbuf_down(
                encoder,
                &argbuf.down,
                &buffers.copy_scales,
                &buffers.copy_quants,
                &expert_table,
                &buffers.routes,
                &buffers.work,
                &buffers.arg_down,
                k as u32,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "K={k} parity command failed: {}",
                command_buffer_error_details(cb)
            );

            let scale_bytes = routing.active_experts.len() * k * S50_DOWN_BLOCKS * 4;
            let quant_bytes = routing.active_experts.len() * k * S50_FF;
            let down_bytes = k * S50_HIDDEN * 4;
            assert_raw_eq(
                &format!("K={k} GateUp scales"),
                &read_bytes(&buffers.copy_scales, scale_bytes),
                &read_bytes(&buffers.arg_scales, scale_bytes),
            );
            assert_raw_eq(
                &format!("K={k} GateUp quants"),
                &read_bytes(&buffers.copy_quants, quant_bytes),
                &read_bytes(&buffers.arg_quants, quant_bytes),
            );
            assert_raw_eq(
                &format!("K={k} Down output"),
                &read_bytes(&buffers.copy_down, down_bytes),
                &read_bytes(&buffers.arg_down, down_bytes),
            );
            eprintln!(
                "[spec50-argbuf] K={k} U={}: GateUp scales/quants and Down raw-bit exact",
                routing.active_experts.len()
            );
        }

        // Alternate variants to keep thermal/frequency drift symmetric. Raw
        // samples are printed, and the summary uses the median GPU sample.
        let mut arg_warm_samples = Vec::with_capacity(5);
        let mut copy_warm_samples = Vec::with_capacity(5);
        for _ in 0..5 {
            arg_warm_samples.push(time_argbuf_round(
                &queue,
                argbuf,
                &expert_table,
                &record_buffers,
                &buffers,
                &routing_k8,
                MAX_K,
                resource_declaration,
            ));
            copy_warm_samples.push(time_copied_round(
                &queue,
                copied,
                &copied_slab,
                &buffers,
                &routing_k8,
                MAX_K,
            ));
        }
        let median_gpu = |samples: &[Timing]| {
            let mut ordered = samples.to_vec();
            ordered.sort_unstable_by_key(|timing| timing.gpu_us);
            ordered[ordered.len() / 2]
        };
        let arg_warm = median_gpu(&arg_warm_samples);
        let copy_warm = median_gpu(&copy_warm_samples);

        eprintln!(
            "[spec50-argbuf] layer={layer_idx} K=8 U=30 resource_declaration={} active_mib={:.2} \
             table_us={table_us} anonymous_copy_us={anonymous_copy_us}",
            resource_declaration.label(),
            (ACTIVE_EXPERTS * S50_SLOT_STRIDE) as f64 / (1024.0 * 1024.0),
        );
        for (label, timing) in [
            ("arg-cold", &arg_cold),
            ("arg-warm", &arg_warm),
            ("copy-first", &copy_first),
            ("copy-warm", &copy_warm),
        ] {
            eprintln!(
                "[spec50-argbuf] {label}: wall_us={} gpu_us={} kernel_us={} \
                 pageins={} faults={} decompressions={} swapins={}",
                timing.wall_us,
                timing.gpu_us,
                timing.kernel_us,
                timing.pageins,
                timing.faults,
                timing.decompressions,
                timing.swapins,
            );
        }
        for (variant, samples) in [
            ("arg-warm-sample", &arg_warm_samples),
            ("copy-warm-sample", &copy_warm_samples),
        ] {
            for (sample, timing) in samples.iter().enumerate() {
                eprintln!(
                    "[spec50-argbuf] {variant}-{sample}: wall_us={} gpu_us={} kernel_us={} pageins={}",
                    timing.wall_us, timing.gpu_us, timing.kernel_us, timing.pageins,
                );
            }
        }
    }
}
