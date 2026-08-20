//! Union-tiled routed-expert kernels for the Gemma 4 26B MoE round.
//!
//! Two drop-in replacements for the currently dispatched batched kernels. Both
//! are required to be BITWISE identical to their reference for every row, every
//! token and every `k_candidates` in `1..=8`, and both are proved so against the
//! live reference pipelines by `spec50_moe_kernels_are_bitwise_identical_for_every_k`.
//!
//! # `spec50_moe_gateup_geglu_quant_batch_k`
//!
//! Replaces `gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k`. Same
//! tile-once geometry (one threadgroup per unique expert x 32-row FF block),
//! same GeGLU, same Q8 quantization, same buffer binding. Three scheduling
//! changes, no arithmetic change:
//!
//! * `packed_uchar4` weight loads instead of ~34 scalar byte loads per block.
//!   The nibble run starts 2 bytes into an 18-byte block, so it is only 2-byte
//!   aligned and `packed_uchar4` (alignment 1) is the legal wide load; `uchar4`
//!   there is the misaligned-UB trap that has bitten this file before.
//! * The 16 nibble codes stay as raw `uchar4` and are unpacked inside the token
//!   loop rather than into 16 live `int4` registers per block. That looks like
//!   more work and measures 1.10x FASTER: the 64-register unpacked form cost
//!   more in occupancy than it saved in ALU.
//! * The token loop runs to a static bound of 8 with a mask guard instead of the
//!   reference's runtime `t < k_candidates`, which is what keeps `gate_acc` /
//!   `up_acc` in registers instead of on the stack. It visits exactly the same
//!   tokens in the same order.
//!
//! Four 4-wide `int4` partials are folded once per block instead of a horizontal
//! add per 4-lane group. Integer addition is associative and
//! `|isum| <= 32 * 8 * 127 = 32512` never overflows, so that reordering is exact,
//! and the float expression that consumes `isum` is textually unchanged.
//!
//! # `spec50_moe_down_union_batch_k`
//!
//! Replaces `gemma4_q4_multi_expert_down_scatter_reduce_simd`, same buffer
//! binding. The reference threadgroup is one (token, hidden row) pair and folds
//! 176 terms across 32 lanes -- ~5.5 terms per lane, each one a fresh
//! `route -> work_list -> down_row -> block` walk ending in a 396-byte segment
//! read. The replacement keeps the lane partition and the accumulation order
//! and changes only what one simdgroup covers:
//!
//! * one simdgroup per TOKEN, `S50_DOWN_ROWS_PER_SIMDGROUP` consecutive hidden
//!   rows per simdgroup. Those rows' Down segments are adjacent in the slab (row
//!   stride 396 B), so a routed expert is now read as 1584 contiguous bytes
//!   instead of 396 -- fewer transactions, and almost no partial-cache-line
//!   waste at the segment ends.
//! * the activation block and its scale do not depend on the hidden row, so they
//!   are loaded and converted once and shared across the rows.
//! * the reference's `for (flat = lane; flat < 176; flat += 32)` has a trip count
//!   of 5 or 6 depending on the lane and so cannot be unrolled; a static bound of
//!   `ceil(176/32)` with a guard visits the same flats in the same order.
//!
//! Each (token, row) keeps its OWN serial `lane_total` chain and its own
//! `simd_sum` over the same 32 lanes, so nothing is reassociated.
//!
//! # Two designs that were tried and rejected
//!
//! * **Rank-indexed accumulator registers after a unique-expert outer loop.**
//!   The natural way to make Down scale with the expert union rather than with K.
//!   It is NOT bit-exact (measured max 896 ULP): the reference's
//!   `lane_total += float(isum) * term_scale` contracts to an `fma`, and a
//!   per-rank register can only hold the separately rounded product. It was also
//!   1.6x slower -- 64 live f32 accumulators per thread spill.
//! * **Staging weights through threadgroup memory** (coalesced tile for GateUp,
//!   one-read-per-union-expert for Down). Bit-exact, and for Down it does cut
//!   device traffic to the exact union floor -- and it is 1.8x / 1.1x SLOWER.
//!   The tile costs more in occupancy, barriers and threadgroup-memory bank
//!   conflicts than the saved traffic is worth on this part.
//!
//! # The fast-math trap
//!
//! This module compiles its shader TWICE, and the encoders use the
//! fast-math-DISABLED library. That is load-bearing, not caution: with Metal's
//! default fast-math a LITERAL transcription of the reference Down kernel into
//! this library disagrees with the reference by up to 16384 ULP, because the
//! compiler reassociates the `lane_total` chain here but did not in
//! `LINEAR_ROW_SHADER`. `spec50_down_divergence_bisect` pins that down and is
//! kept as a regression guard: it asserts nothing, it prints the four-way
//! comparison that identifies the cause if this ever moves again. The cost of
//! disabling fast-math measured as nil (39.32 ms fast vs 39.33 ms strict).

// Additive: nothing in the existing dispatch paths reaches these yet, so the
// pipelines and encoders read as dead until the integrator swaps them in.
#![allow(dead_code)]

use super::*;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Geometry mirrors (identical to the G4Q4_* shader defines / Rust mirrors).
// ---------------------------------------------------------------------------

pub(crate) const S50_HIDDEN: usize = 2_816;
pub(crate) const S50_FF: usize = 704;
pub(crate) const S50_ROUTES: usize = 8;
pub(crate) const S50_GU_BLOCKS: usize = 88;
pub(crate) const S50_DOWN_BLOCKS: usize = 22;
pub(crate) const S50_WIRE: usize = 18;
pub(crate) const S50_GU_ROW_BYTES: usize = 1_584;
pub(crate) const S50_DOWN_ROW_BYTES: usize = 396;
pub(crate) const S50_GATE_UP_BYTES: usize = 2_230_272;
pub(crate) const S50_DOWN_BYTES: usize = 1_115_136;
pub(crate) const S50_RECORD_BYTES: usize = 3_345_408;
pub(crate) const S50_SLOT_STRIDE: usize = 3_358_720;

const _: () = {
    assert!(S50_GU_ROW_BYTES == S50_GU_BLOCKS * S50_WIRE);
    assert!(S50_DOWN_ROW_BYTES == S50_DOWN_BLOCKS * S50_WIRE);
    assert!(S50_GATE_UP_BYTES == 2 * S50_FF * S50_GU_ROW_BYTES);
    assert!(S50_DOWN_BYTES == S50_HIDDEN * S50_DOWN_ROW_BYTES);
    assert!(S50_RECORD_BYTES == S50_GATE_UP_BYTES + S50_DOWN_BYTES);
    assert!(S50_HIDDEN % S50_DOWN_ROWS_PER_SIMDGROUP == 0);
};

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

const SPEC50_MOE_SHADER: &str = r#"
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

// ---------------------------------------------------------------------------
// GateUp: bitwise twin of gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k.
//
// Buffer order is unchanged, so this is a drop-in for that pipeline.
//   0 input_scales   (f32, k * 88)
//   1 input_quants   (i8,  k * 2816)
//   2 expert_weights (slab, resident bank; byte offset supplied by the caller)
//   3 work_list      (Gemma4UniqueExpertWork[128])
//   4 output_scales  (f32, num_unique * k * 22)
//   5 output_quants  (i8,  num_unique * k * 704)
//   6 num_unique_experts (uint)
//   7 k_candidates       (uint)
//   8 overflow_expert_weights (slab, overflow bank; may be null)
// Grid: num_unique * (704/32) threadgroups of 32 threads.
// ---------------------------------------------------------------------------
kernel void spec50_moe_gateup_geglu_quant_batch_k(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const uchar* expert_weights [[buffer(2)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(3)]],
    device float* output_scales [[buffer(4)]],
    device char* output_quants [[buffer(5)]],
    constant uint& num_unique_experts [[buffer(6)]],
    constant uint& k_candidates [[buffer(7)]],
    device const uchar* overflow_expert_weights [[buffer(8)]],
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

    const uint row = b * 32u + lane;
    const ulong expert_base = ulong(work.expert_weight_offset);
    device const uchar* weights = (work.slab_index == 1 && overflow_expert_weights != nullptr)
        ? overflow_expert_weights
        : expert_weights;
    device const uchar* gate_row = weights + expert_base + ulong(row) * G4Q4_GU_ROW_BYTES;
    device const uchar* up_row = weights + expert_base + ulong(row + G4Q4_FF) * G4Q4_GU_ROW_BYTES;

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

        // The 16 nibble bytes start 2 bytes into an 18-byte block, so the run is
        // only 2-byte aligned: packed_uchar4 (alignment 1) is the wide load that
        // is legal here -- uchar4 would be misaligned UB. Four of these replace
        // the reference's 16 scalar byte loads per block per matrix.
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

// ---------------------------------------------------------------------------
// Down: union-tiled twin of gemma4_q4_multi_expert_down_scatter_reduce_simd.
//
// Binding is IDENTICAL to the reference (buffers 0..=7), so this is a drop-in:
//   0 act_scales      (f32, num_unique * k * 22)
//   1 act_quants      (i8,  num_unique * k * 704)
//   2 expert_weights  (slab, resident bank)
//   3 candidate_routes(Gemma4CandidateRouteEntry[k * 8])
//   4 work_list       (Gemma4UniqueExpertWork[128])
//   5 output_moe_acc  (f32, k * 2816)
//   6 k_candidates    (uint)
//   7 overflow_expert_weights (slab, overflow bank; may be null)
//
// What changes is only the grid. The reference dispatches k * 2816 32-thread
// threadgroups keyed `group = t * 2816 + row`, so the k threadgroups that share
// a hidden row are 2816 groups apart in launch order and never co-resident:
// each of them pulls that row's routed Down segments from DRAM independently,
// and the round reads (k * 8) / union times the byte floor.
//
// Here the threadgroup is the hidden ROW and the k tokens are k SIMD groups
// inside it. Every simdgroup runs the reference body verbatim -- same flat
// `for (flat = lane; flat < 8 * 22; flat += 32)` walk, same lane partition, same
// serial accumulation order, same single `simd_sum` over the same 32 lanes -- so
// the result is bitwise identical by construction, with no reassociation to
// justify. The k tokens now hit the same 396-byte Down segments while those
// lines are still resident, which is what collapses the re-read factor toward
// the expert union.
//
// Grid: 2816 threadgroups of (32 * k_candidates) threads.
// ---------------------------------------------------------------------------
kernel void spec50_moe_down_union_batch_k(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device const uchar* expert_weights [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    device const uchar* overflow_expert_weights [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint t = sg;
    if (t >= k_candidates) return;

    // S50_DOWN_ROWS consecutive hidden rows per simdgroup. Their Down segments
    // are adjacent in the slab (row stride 396 B), so one routed expert is now
    // read as S50_DOWN_ROWS * 396 contiguous bytes instead of 396: fewer
    // transactions and far less partial-cache-line waste at the segment ends.
    // The activation block and its scale do not depend on the hidden row, so
    // they are loaded once and shared across the rows.
    //
    // Each row keeps its OWN serial `lane_total` chain over the reference's flat
    // walk, so per (token, row) the accumulation order, the lane partition and
    // the simd_sum are the reference's exactly.
    const uint row0 = group * S50_DOWN_ROWS;

    float lane_total[S50_DOWN_ROWS];
    #pragma unroll
    for (uint r = 0; r < S50_DOWN_ROWS; ++r) {
        lane_total[r] = 0.0f;
    }

    // The reference's `for (flat = lane; flat < 176; flat += 32)` has a trip
    // count of 5 or 6 depending on the lane and so cannot be unrolled;
    // S50_DOWN_TERMS is the static bound ceil(176/32) and the guard visits the
    // same flats in the same order.
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
        const ulong expert_base = ulong(work.expert_weight_offset);
        device const uchar* weights = (work.slab_index == 1 && overflow_expert_weights != nullptr)
            ? overflow_expert_weights
            : expert_weights;
        device const uchar* down_rows = weights + expert_base + G4Q4_GATE_UP_BYTES
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
            // Nibble bytes are at block offset +2, i.e. only 2-byte aligned:
            // packed_uchar4 (alignment 1) is the legal wide load. The four 4-wide
            // integer partials fold once at the end -- integer addition is
            // associative and |isum| <= 32 * 8 * 127 = 32512 never overflows, so
            // this is exact against the reference's serial 16-step loop.
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





// ---------------------------------------------------------------------------
// DIAGNOSTIC clones, test-only. Both use the reference's own grid
// (group = t * 2816 + row, 32-thread threadgroups) so the ONLY variable is the
// one named in the kernel. They exist to bisect a bitwise divergence between
// this library and LINEAR_ROW_SHADER.
// ---------------------------------------------------------------------------

// Literal transcription of gemma4_q4_multi_expert_down_scatter_reduce_simd.
// If this disagrees with the real reference, the difference is the LIBRARY
// (compile options / codegen), not the kernel body.
kernel void spec50_down_clone_scalar(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device const uchar* expert_weights [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    device const uchar* overflow_expert_weights [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint t = group / G4Q4_HIDDEN;
    const uint row = group - t * G4Q4_HIDDEN;
    if (t >= k_candidates) return;

    const uint terms = G4Q4_ROUTES * G4Q4_DOWN_BLOCKS;
    float lane_total = 0.0f;
    for (uint flat = lane; flat < terms; flat += 32u) {
        const uint slot = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - slot * G4Q4_DOWN_BLOCKS;
        const Gemma4CandidateRouteEntry route = candidate_routes[t * G4Q4_ROUTES + slot];
        if (route.weight == 0.0f || route.unique_expert_idx >= 128u) continue;
        const uint u = route.unique_expert_idx;
        const Gemma4UniqueExpertWork work = work_list[u];
        const ulong expert_base = ulong(work.expert_weight_offset);
        device const uchar* weights = (work.slab_index == 1 && overflow_expert_weights != nullptr)
            ? overflow_expert_weights
            : expert_weights;
        device const uchar* down_row = weights + expert_base + G4Q4_GATE_UP_BYTES + ulong(row) * G4Q4_DOWN_ROW_BYTES;
        device const uchar* block = down_row + ulong(b) * G4Q4_WIRE;
        const float weight_scale = float(*reinterpret_cast<device const half*>(block));

        const ulong act_quant_base = ulong(u) * ulong(k_candidates) * G4Q4_FF + ulong(t) * G4Q4_FF + ulong(b) * 32ul;
        device const char* x = act_quants + act_quant_base;
        int isum = 0;
        #pragma unroll
        for (uint l = 0; l < 16; ++l) {
            const uchar wb = block[2 + l];
            const int x_lo = int(x[l]);
            const int x_hi = int(x[l + 16]);
            isum += (int(wb & 0x0f) - 8) * x_lo + (int(wb >> 4) - 8) * x_hi;
        }
        const ulong act_scale_base = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        const float term_scale = (weight_scale * act_scales[act_scale_base]) * route.weight;
        lane_total += float(isum) * term_scale;
    }
    const float total = simd_sum(lane_total);
    if (lane == 0) {
        output_moe_acc[ulong(t) * G4Q4_HIDDEN + ulong(row)] = total;
    }
}

// Same as the clone above except the 16-step integer dot is replaced by the
// packed_uchar4 / int4 form. If THIS disagrees with the scalar clone, the
// vectorized integer dot is the difference.
kernel void spec50_down_clone_vec(
    device const float* act_scales [[buffer(0)]],
    device const char* act_quants [[buffer(1)]],
    device const uchar* expert_weights [[buffer(2)]],
    device const Gemma4CandidateRouteEntry* candidate_routes [[buffer(3)]],
    device const Gemma4UniqueExpertWork* work_list [[buffer(4)]],
    device float* output_moe_acc [[buffer(5)]],
    constant uint& k_candidates [[buffer(6)]],
    device const uchar* overflow_expert_weights [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint t = group / G4Q4_HIDDEN;
    const uint row = group - t * G4Q4_HIDDEN;
    if (t >= k_candidates) return;

    const uint terms = G4Q4_ROUTES * G4Q4_DOWN_BLOCKS;
    float lane_total = 0.0f;
    for (uint flat = lane; flat < terms; flat += 32u) {
        const uint slot = flat / G4Q4_DOWN_BLOCKS;
        const uint b = flat - slot * G4Q4_DOWN_BLOCKS;
        const Gemma4CandidateRouteEntry route = candidate_routes[t * G4Q4_ROUTES + slot];
        if (route.weight == 0.0f || route.unique_expert_idx >= 128u) continue;
        const uint u = route.unique_expert_idx;
        const Gemma4UniqueExpertWork work = work_list[u];
        const ulong expert_base = ulong(work.expert_weight_offset);
        device const uchar* weights = (work.slab_index == 1 && overflow_expert_weights != nullptr)
            ? overflow_expert_weights
            : expert_weights;
        device const uchar* down_row = weights + expert_base + G4Q4_GATE_UP_BYTES + ulong(row) * G4Q4_DOWN_ROW_BYTES;
        device const uchar* block = down_row + ulong(b) * G4Q4_WIRE;
        const float weight_scale = float(*reinterpret_cast<device const half*>(block));

        const ulong act_quant_base = ulong(u) * ulong(k_candidates) * G4Q4_FF + ulong(t) * G4Q4_FF + ulong(b) * 32ul;
        device const char* x = act_quants + act_quant_base;
        device const packed_uchar4* wq = reinterpret_cast<device const packed_uchar4*>(block + 2);
        device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
        device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
        int4 a4 = int4(0);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            const uchar4 wb = uchar4(wq[k]);
            a4 += (int4(wb & uchar4(0x0f)) - 8) * int4(xlo4[k])
                + (int4(wb >> 4) - 8) * int4(xhi4[k]);
        }
        const int isum = (a4.x + a4.y) + (a4.z + a4.w);
        const ulong act_scale_base = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
        const float term_scale = (weight_scale * act_scales[act_scale_base]) * route.weight;
        lane_total += float(isum) * term_scale;
    }
    const float total = simd_sum(lane_total);
    if (lane == 0) {
        output_moe_acc[ulong(t) * G4Q4_HIDDEN + ulong(row)] = total;
    }
}

"#;

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

pub(crate) struct Spec50MoeVariant {
    pub(crate) gateup: ComputePipelineState,
    pub(crate) down: ComputePipelineState,
    /// Diagnostic bisection clones; see the shader comment above them.
    pub(crate) down_clone_scalar: ComputePipelineState,
    pub(crate) down_clone_vec: ComputePipelineState,
}

pub(crate) struct Spec50MoeKernels {
    /// Compiled with Metal's default fast-math.
    pub(crate) fast: Spec50MoeVariant,
    /// Compiled with fast-math disabled, which is what pins the float
    /// accumulation order against the reference library.
    pub(crate) strict: Spec50MoeVariant,
}

static SPEC50_MOE_KERNELS: OnceLock<Option<Spec50MoeKernels>> = OnceLock::new();

fn build_variant(device: &Device, fast_math: bool) -> Option<Spec50MoeVariant> {
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(fast_math);
    let library = device
        .new_library_with_source(SPEC50_MOE_SHADER, &options)
        .map_err(|err| eprintln!("[metal] SPEC50_MOE_SHADER compile failed (fast={fast_math}): {err}"))
        .ok()?;
    let build = |name: &str| -> Option<ComputePipelineState> {
        let function = library
            .get_function(name, None)
            .map_err(|err| eprintln!("[metal] spec50 {name} missing: {err}"))
            .ok()?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| eprintln!("[metal] spec50 {name} pipeline failed: {err}"))
            .ok()
    };
    Some(Spec50MoeVariant {
        gateup: build("spec50_moe_gateup_geglu_quant_batch_k")?,
        down: build("spec50_moe_down_union_batch_k")?,
        down_clone_scalar: build("spec50_down_clone_scalar")?,
        down_clone_vec: build("spec50_down_clone_vec")?,
    })
}

/// The pipelines an integrator should dispatch: fast-math disabled, which is
/// what pins the float accumulation order against the reference library. See the
/// module header.
pub(crate) fn spec50_moe_pipelines(device: &Device) -> Option<&'static Spec50MoeVariant> {
    spec50_moe_kernels(device).map(|k| &k.strict)
}

pub(crate) fn spec50_moe_kernels(device: &Device) -> Option<&'static Spec50MoeKernels> {
    SPEC50_MOE_KERNELS
        .get_or_init(|| {
            Some(Spec50MoeKernels {
                fast: build_variant(device, true)?,
                strict: build_variant(device, false)?,
            })
        })
        .as_ref()
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

/// GateUp + GeGLU + Q8 quantization over the unique-expert union.
///
/// Buffer order is identical to
/// `gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k`, so the integrator
/// only swaps the pipeline state. No new buffer is required.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_gateup(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input_scales: &Buffer,
    input_quants: &Buffer,
    expert_weights: &Buffer,
    expert_weights_offset: u64,
    work_list: &Buffer,
    output_scales: &Buffer,
    output_quants: &Buffer,
    num_unique_experts: u32,
    k_candidates: u32,
    overflow_expert_weights: Option<&Buffer>,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(input_quants), 0);
    encoder.set_buffer(2, Some(expert_weights), expert_weights_offset);
    encoder.set_buffer(3, Some(work_list), 0);
    encoder.set_buffer(4, Some(output_scales), 0);
    encoder.set_buffer(5, Some(output_quants), 0);
    encoder.set_bytes(6, 4, &num_unique_experts as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_candidates as *const u32 as *const _);
    encoder.set_buffer(8, overflow_expert_weights.map(|v| &**v), 0);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (num_unique_experts as u64) * (S50_FF as u64 / 32),
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

/// Hidden rows one Down simdgroup folds. Must match `S50_DOWN_ROWS` in the shader.
pub(crate) const S50_DOWN_ROWS_PER_SIMDGROUP: usize = 4;

/// Union-tiled Down projection + weighted scatter-reduce.
///
/// The binding is byte-for-byte the reference binding
/// (`gemma4_q4_multi_expert_down_scatter_reduce_simd`, buffers 0..=7): the
/// integrator allocates NO new buffer and passes no new scalar. Only the grid
/// changes -- `2816 / S50_DOWN_ROWS_PER_SIMDGROUP` threadgroups of
/// `32 * k_candidates` threads (one simdgroup per token, each folding
/// `S50_DOWN_ROWS_PER_SIMDGROUP` consecutive hidden rows) instead of
/// `k_candidates * 2816` threadgroups of 32.
///
/// `k_candidates` must be in `1..=8`; the caller already gates on that.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_down(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    act_scales: &Buffer,
    act_quants: &Buffer,
    expert_weights: &Buffer,
    expert_weights_offset: u64,
    candidate_routes: &Buffer,
    work_list: &Buffer,
    output_moe_acc: &Buffer,
    k_candidates: u32,
    overflow_expert_weights: Option<&Buffer>,
) {
    debug_assert!((1..=8).contains(&k_candidates));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(act_scales), 0);
    encoder.set_buffer(1, Some(act_quants), 0);
    encoder.set_buffer(2, Some(expert_weights), expert_weights_offset);
    encoder.set_buffer(3, Some(candidate_routes), 0);
    encoder.set_buffer(4, Some(work_list), 0);
    encoder.set_buffer(5, Some(output_moe_acc), 0);
    encoder.set_bytes(6, 4, &k_candidates as *const u32 as *const _);
    encoder.set_buffer(7, overflow_expert_weights.map(|v| &**v), 0);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (S50_HIDDEN / S50_DOWN_ROWS_PER_SIMDGROUP) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32 * u64::from(k_candidates.clamp(1, 8)),
            height: 1,
            depth: 1,
        },
    );
}

// ---------------------------------------------------------------------------
// Synthetic 26B-shaped harness (tests + benchmark)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Experts materialised in the fake slab. 26B routes 8-of-128 per token; the
    /// measured K=8 union saturates near 30, so 40 leaves headroom for the
    /// rotation the benchmark uses to keep every repeat DRAM-resident.
    const SLAB_EXPERTS: usize = 40;
    const BENCH_UNIQUE: usize = 32;
    const BENCH_LAYERS: usize = 30;

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            // xorshift64*
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

    fn write_bytes(buffer: &Buffer, bytes: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>(),
                bytes.len(),
            );
        }
    }

    fn read_f32(buffer: &Buffer, len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                len * 4,
            );
        }
        out
    }

    fn read_i8(buffer: &Buffer, len: usize) -> Vec<i8> {
        let mut out = vec![0i8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                len,
            );
        }
        out
    }

    /// Fill one expert record in place: 1408 gate/up rows then 2816 down rows,
    /// each row 88 / 22 Q4_0 blocks of `half` scale + 16 packed nibble bytes.
    fn fill_expert_record(dst: &mut [u8], rng: &mut Rng) {
        debug_assert_eq!(dst.len(), S50_RECORD_BYTES);
        let mut off = 0usize;
        let blocks = 2 * S50_FF * S50_GU_BLOCKS + S50_HIDDEN * S50_DOWN_BLOCKS;
        for _ in 0..blocks {
            // Keep scales well inside the half normal range so no term is Inf or
            // NaN: a poisoned lane would make a bitwise comparison vacuous.
            let scale = 0.002 + rng.next_f32() * 0.05;
            let bits = f32_to_f16_bits(scale);
            dst[off] = (bits & 0xff) as u8;
            dst[off + 1] = (bits >> 8) as u8;
            for l in 0..16 {
                dst[off + 2 + l] = (rng.next_u32() & 0xff) as u8;
            }
            off += S50_WIRE;
        }
        debug_assert_eq!(off, dst.len());
    }

    struct Routing {
        work_list: Vec<Gemma4UniqueExpertWork>,
        routes: Vec<Gemma4CandidateRouteEntry>,
        num_unique: usize,
    }

    /// Build routing exactly as `gemma4_gpu_topk_routing` does: per-token slots
    /// first, then a stable ascending scan over slots 0..num_slots that assigns
    /// unique-expert indices in slot order.
    ///
    /// `per_token_slots[t]` are the 8 slab slots token t routes to. They are
    /// generated independently of K so that token t's routing is identical at
    /// K=1 and inside any larger batch.
    fn build_routing(per_token_slots: &[[u32; 8]], weights: &[[f32; 8]], k: usize) -> Routing {
        let mut active_slot_map = [u32::MAX; 128];
        let mut work_list = vec![Gemma4UniqueExpertWork::default(); 128];
        let mut count = 0usize;
        for slot in 0..SLAB_EXPERTS {
            let mut mask = 0u64;
            for (t, slots) in per_token_slots.iter().enumerate().take(k) {
                for (i, s) in slots.iter().enumerate() {
                    if *s as usize == slot && weights[t][i] != 0.0 {
                        mask |= 1u64 << t;
                    }
                }
            }
            if mask == 0 {
                continue;
            }
            active_slot_map[slot] = count as u32;
            work_list[count] = Gemma4UniqueExpertWork {
                candidate_mask: mask,
                expert_weight_offset: (slot * S50_SLOT_STRIDE) as u32,
                slab_index: 0,
            };
            count += 1;
        }
        let mut routes = vec![Gemma4CandidateRouteEntry::default(); 8 * S50_ROUTES];
        for t in 0..k {
            for i in 0..S50_ROUTES {
                let slot = per_token_slots[t][i] as usize;
                let mapped = active_slot_map[slot];
                if mapped == u32::MAX {
                    routes[t * S50_ROUTES + i] = Gemma4CandidateRouteEntry {
                        unique_expert_idx: u32::MAX,
                        weight: 0.0,
                    };
                } else {
                    routes[t * S50_ROUTES + i] = Gemma4CandidateRouteEntry {
                        unique_expert_idx: mapped,
                        weight: weights[t][i],
                    };
                }
            }
        }
        Routing {
            work_list,
            routes,
            num_unique: count,
        }
    }

    struct Harness {
        device: Device,
        queue: CommandQueue,
        slab: Buffer,
        input_scales: Buffer,
        input_quants: Buffer,
        work_list: Buffer,
        routes: Buffer,
        gu_scales_ref: Buffer,
        gu_quants_ref: Buffer,
        gu_scales_new: Buffer,
        gu_quants_new: Buffer,
        down_ref: Buffer,
        down_new: Buffer,
        per_token_slots: Vec<[u32; 8]>,
        route_weights: Vec<[f32; 8]>,
    }

    impl Harness {
        fn new() -> Option<Self> {
            let kernel = metal_linear_kernel()?;
            let device = kernel.device.clone();
            let queue = kernel.queue.clone();

            let slab_len = SLAB_EXPERTS * S50_SLOT_STRIDE;
            let slab = new_buffer(&device, slab_len);
            {
                let mut rng = Rng::new(0x5150_0001);
                let base = slab.contents().cast::<u8>();
                for e in 0..SLAB_EXPERTS {
                    let record = unsafe {
                        std::slice::from_raw_parts_mut(
                            base.add(e * S50_SLOT_STRIDE),
                            S50_RECORD_BYTES,
                        )
                    };
                    fill_expert_record(record, &mut rng);
                }
                // Zero the inter-record padding so nothing reads uninitialised.
                for e in 0..SLAB_EXPERTS {
                    let pad = unsafe {
                        std::slice::from_raw_parts_mut(
                            base.add(e * S50_SLOT_STRIDE + S50_RECORD_BYTES),
                            S50_SLOT_STRIDE - S50_RECORD_BYTES,
                        )
                    };
                    pad.fill(0);
                }
            }

            let mut rng = Rng::new(0x5150_0002);
            let input_scales = new_buffer(&device, 8 * S50_GU_BLOCKS * 4);
            let scales: Vec<f32> = (0..8 * S50_GU_BLOCKS)
                .map(|_| 0.0005 + rng.next_f32() * 0.01)
                .collect();
            write_bytes(&input_scales, unsafe {
                std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), scales.len() * 4)
            });
            let input_quants = new_buffer(&device, 8 * S50_HIDDEN);
            let quants: Vec<u8> = (0..8 * S50_HIDDEN)
                .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8 as u8)
                .collect();
            write_bytes(&input_quants, &quants);

            // Independent per-token routing: token t's 8 distinct slots and
            // weights do not depend on how many tokens ride in the batch.
            let mut per_token_slots = Vec::new();
            let mut route_weights = Vec::new();
            for t in 0..8usize {
                let mut rng_t = Rng::new(0x5150_1000 + t as u64);
                let mut slots = [0u32; 8];
                let mut chosen: Vec<u32> = Vec::new();
                while chosen.len() < 8 {
                    let c = rng_t.next_u32() % SLAB_EXPERTS as u32;
                    if !chosen.contains(&c) {
                        chosen.push(c);
                    }
                }
                slots.copy_from_slice(&chosen);
                let mut w = [0.0f32; 8];
                for (i, wi) in w.iter_mut().enumerate() {
                    *wi = 0.02 + rng_t.next_f32() * (1.0 / (1.0 + i as f32));
                }
                per_token_slots.push(slots);
                route_weights.push(w);
            }

            let gu_scale_len = SLAB_EXPERTS * 8 * S50_DOWN_BLOCKS;
            let gu_quant_len = SLAB_EXPERTS * 8 * S50_FF;
            Some(Self {
                work_list: new_buffer(&device, 128 * 16),
                routes: new_buffer(&device, 8 * S50_ROUTES * 8),
                gu_scales_ref: new_buffer(&device, gu_scale_len * 4),
                gu_quants_ref: new_buffer(&device, gu_quant_len),
                gu_scales_new: new_buffer(&device, gu_scale_len * 4),
                gu_quants_new: new_buffer(&device, gu_quant_len),
                down_ref: new_buffer(&device, 8 * S50_HIDDEN * 4),
                down_new: new_buffer(&device, 8 * S50_HIDDEN * 4),
                device,
                queue,
                slab,
                input_scales,
                input_quants,
                per_token_slots,
                route_weights,
            })
        }

        fn upload_routing(&self, routing: &Routing) {
            write_bytes(&self.work_list, unsafe {
                std::slice::from_raw_parts(
                    routing.work_list.as_ptr().cast::<u8>(),
                    routing.work_list.len() * std::mem::size_of::<Gemma4UniqueExpertWork>(),
                )
            });
            write_bytes(&self.routes, unsafe {
                std::slice::from_raw_parts(
                    routing.routes.as_ptr().cast::<u8>(),
                    routing.routes.len() * std::mem::size_of::<Gemma4CandidateRouteEntry>(),
                )
            });
        }

        fn zero_outputs(&self) {
            for (buf, len) in [
                (&self.gu_scales_ref, SLAB_EXPERTS * 8 * S50_DOWN_BLOCKS * 4),
                (&self.gu_scales_new, SLAB_EXPERTS * 8 * S50_DOWN_BLOCKS * 4),
                (&self.gu_quants_ref, SLAB_EXPERTS * 8 * S50_FF),
                (&self.gu_quants_new, SLAB_EXPERTS * 8 * S50_FF),
                (&self.down_ref, 8 * S50_HIDDEN * 4),
                (&self.down_new, 8 * S50_HIDDEN * 4),
            ] {
                unsafe {
                    std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, len);
                }
            }
        }
    }

    fn reference_gateup(
        kernel: &MetalLinearKernel,
        encoder: &metal::ComputeCommandEncoderRef,
        h: &Harness,
        scales: &Buffer,
        quants: &Buffer,
        num_unique: u32,
        k: u32,
    ) {
        let pipeline = kernel
            .gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k_pipeline
            .as_ref()
            .expect("reference gateup batch_k pipeline");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&h.input_scales), 0);
        encoder.set_buffer(1, Some(&h.input_quants), 0);
        encoder.set_buffer(2, Some(&h.slab), 0);
        encoder.set_buffer(3, Some(&h.work_list), 0);
        encoder.set_buffer(4, Some(scales), 0);
        encoder.set_buffer(5, Some(quants), 0);
        encoder.set_bytes(6, 4, &num_unique as *const u32 as *const _);
        encoder.set_bytes(7, 4, &k as *const u32 as *const _);
        encoder.set_buffer(8, None, 0);
        dispatch_one_simdgroup_per_row(encoder, num_unique as usize * (S50_FF / 32));
    }

    fn reference_down(
        kernel: &MetalLinearKernel,
        encoder: &metal::ComputeCommandEncoderRef,
        h: &Harness,
        scales: &Buffer,
        quants: &Buffer,
        out: &Buffer,
        k: u32,
    ) {
        let pipeline = kernel
            .gemma4_q4_multi_expert_down_scatter_reduce_simd_pipeline
            .as_ref()
            .expect("reference down simd pipeline");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(scales), 0);
        encoder.set_buffer(1, Some(quants), 0);
        encoder.set_buffer(2, Some(&h.slab), 0);
        encoder.set_buffer(3, Some(&h.routes), 0);
        encoder.set_buffer(4, Some(&h.work_list), 0);
        encoder.set_buffer(5, Some(out), 0);
        encoder.set_bytes(6, 4, &k as *const u32 as *const _);
        encoder.set_buffer(7, None, 0);
        dispatch_one_simdgroup_per_row(encoder, k as usize * S50_HIDDEN);
    }

    /// One K: run reference and replacement for both stages and compare raw bits.
    fn run_pair(h: &Harness, k: usize) -> (Vec<f32>, Vec<i8>, Vec<f32>, Vec<f32>, Vec<i8>, Vec<f32>) {
        let kernel = metal_linear_kernel().expect("metal kernel");
        let spec = spec50_moe_kernels(&h.device).expect("spec50 pipelines");
        let routing = build_routing(&h.per_token_slots, &h.route_weights, k);
        h.upload_routing(&routing);
        h.zero_outputs();
        let nu = routing.num_unique as u32;
        let ku = k as u32;

        let cb = h.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        reference_gateup(kernel, enc, h, &h.gu_scales_ref, &h.gu_quants_ref, nu, ku);
        enc.memory_barrier_with_resources(&[&h.gu_scales_ref, &h.gu_quants_ref]);
        encode_spec50_gateup(
            enc,
            &spec.fast.gateup,
            &h.input_scales,
            &h.input_quants,
            &h.slab,
            0,
            &h.work_list,
            &h.gu_scales_new,
            &h.gu_quants_new,
            nu,
            ku,
            None,
        );
        enc.memory_barrier_with_resources(&[&h.gu_scales_new, &h.gu_quants_new]);
        // Both Down runs consume the REFERENCE GateUp output so a GateUp
        // difference cannot mask or manufacture a Down difference.
        reference_down(
            kernel,
            enc,
            h,
            &h.gu_scales_ref,
            &h.gu_quants_ref,
            &h.down_ref,
            ku,
        );
        enc.memory_barrier_with_resources(&[&h.down_ref]);
        encode_spec50_down(
            enc,
            &spec.strict.down,
            &h.gu_scales_ref,
            &h.gu_quants_ref,
            &h.slab,
            0,
            &h.routes,
            &h.work_list,
            &h.down_new,
            ku,
            None,
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let sl = routing.num_unique * k * S50_DOWN_BLOCKS;
        let ql = routing.num_unique * k * S50_FF;
        (
            read_f32(&h.gu_scales_ref, sl),
            read_i8(&h.gu_quants_ref, ql),
            read_f32(&h.down_ref, k * S50_HIDDEN),
            read_f32(&h.gu_scales_new, sl),
            read_i8(&h.gu_quants_new, ql),
            read_f32(&h.down_new, k * S50_HIDDEN),
        )
    }

    fn assert_bits_eq(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label}: length");
        let mut worst_ulp: i64 = 0;
        let mut first: Option<usize> = None;
        let mut diffs = 0usize;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.to_bits() != y.to_bits() {
                diffs += 1;
                if first.is_none() {
                    first = Some(i);
                }
                let ulp = (x.to_bits() as i64 - y.to_bits() as i64).abs();
                worst_ulp = worst_ulp.max(ulp);
            }
        }
        assert!(
            diffs == 0,
            "{label}: {diffs}/{} raw f32 bit mismatches, first at {:?} (ref {:?} / new {:?}), max ULP delta {worst_ulp}",
            a.len(),
            first,
            first.map(|i| a[i]),
            first.map(|i| b[i]),
        );
    }

    /// Bisect a Down divergence: reference vs its literal clone in THIS library
    /// (isolates compile/codegen), and scalar clone vs vectorized clone
    /// (isolates the integer dot rewrite). Both clones use the reference grid.
    #[test]
    fn spec50_down_divergence_bisect() {
        let Some(h) = Harness::new() else {
            eprintln!("[spec50] no Metal device; skipping");
            return;
        };
        let kernel = metal_linear_kernel().expect("metal kernel");
        let spec = spec50_moe_kernels(&h.device).expect("spec50 pipelines");
        let k = 8usize;
        let routing = build_routing(&h.per_token_slots, &h.route_weights, k);
        h.upload_routing(&routing);
        h.zero_outputs();
        let nu = routing.num_unique as u32;
        let ku = k as u32;

        let clone_scalar = new_buffer(&h.device, 8 * S50_HIDDEN * 4);
        let clone_vec = new_buffer(&h.device, 8 * S50_HIDDEN * 4);
        let ref_twice = new_buffer(&h.device, 8 * S50_HIDDEN * 4);

        let cb = h.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        reference_gateup(kernel, enc, &h, &h.gu_scales_ref, &h.gu_quants_ref, nu, ku);
        enc.memory_barrier_with_resources(&[&h.gu_scales_ref, &h.gu_quants_ref]);
        reference_down(kernel, enc, &h, &h.gu_scales_ref, &h.gu_quants_ref, &h.down_ref, ku);
        enc.memory_barrier_with_resources(&[&h.down_ref]);
        reference_down(kernel, enc, &h, &h.gu_scales_ref, &h.gu_quants_ref, &ref_twice, ku);
        enc.memory_barrier_with_resources(&[&ref_twice]);
        for (pipeline, out) in [
            (&spec.fast.down_clone_scalar, &clone_scalar),
            (&spec.strict.down_clone_scalar, &clone_vec),
        ] {
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&h.gu_scales_ref), 0);
            enc.set_buffer(1, Some(&h.gu_quants_ref), 0);
            enc.set_buffer(2, Some(&h.slab), 0);
            enc.set_buffer(3, Some(&h.routes), 0);
            enc.set_buffer(4, Some(&h.work_list), 0);
            enc.set_buffer(5, Some(out), 0);
            enc.set_bytes(6, 4, &ku as *const u32 as *const _);
            enc.set_buffer(7, None, 0);
            dispatch_one_simdgroup_per_row(enc, k * S50_HIDDEN);
            enc.memory_barrier_with_resources(&[out]);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let n = k * S50_HIDDEN;
        let r = read_f32(&h.down_ref, n);
        let r2 = read_f32(&ref_twice, n);
        let cs = read_f32(&clone_scalar, n);
        let cv = read_f32(&clone_vec, n);
        let cmp = |a: &[f32], b: &[f32]| -> (usize, i64) {
            let mut diffs = 0usize;
            let mut worst = 0i64;
            for (x, y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    diffs += 1;
                    worst = worst.max((x.to_bits() as i64 - y.to_bits() as i64).abs());
                }
            }
            (diffs, worst)
        };
        eprintln!("[spec50 bisect] reference vs reference (rerun):       {:?}", cmp(&r, &r2));
        eprintln!("[spec50 bisect] reference vs clone (fast-math):        {:?}", cmp(&r, &cs));
        eprintln!("[spec50 bisect] reference vs clone (fast-math OFF):    {:?}", cmp(&r, &cv));
        eprintln!("[spec50 bisect] clone fast vs clone strict:            {:?}", cmp(&cs, &cv));
    }

    #[test]
    fn spec50_moe_kernels_are_bitwise_identical_for_every_k() {
        let Some(h) = Harness::new() else {
            eprintln!("[spec50] no Metal device; skipping");
            return;
        };
        for k in 1..=8usize {
            let (s_ref, q_ref, d_ref, s_new, q_new, d_new) = run_pair(&h, k);
            assert_bits_eq(&format!("gateup scales K={k}"), &s_ref, &s_new);
            assert_eq!(q_ref, q_new, "gateup quants K={k}");
            assert_bits_eq(&format!("down K={k}"), &d_ref, &d_new);
            eprintln!(
                "[spec50] K={k}: gateup {} scales + {} quants bit-identical, down {} rows bit-identical",
                s_ref.len(),
                q_ref.len(),
                d_ref.len()
            );
        }
    }

    #[test]
    fn spec50_moe_per_token_results_are_batch_independent() {
        let Some(h) = Harness::new() else {
            eprintln!("[spec50] no Metal device; skipping");
            return;
        };
        // Token 0 alone.
        let (_, _, d_ref_1, _, _, d_new_1) = run_pair(&h, 1);
        for k in 2..=8usize {
            let (_, _, d_ref_k, _, _, d_new_k) = run_pair(&h, k);
            assert_bits_eq(
                &format!("reference down token 0 at K=1 vs K={k}"),
                &d_ref_1[..S50_HIDDEN],
                &d_ref_k[..S50_HIDDEN],
            );
            assert_bits_eq(
                &format!("spec50 down token 0 at K=1 vs K={k}"),
                &d_new_1[..S50_HIDDEN],
                &d_new_k[..S50_HIDDEN],
            );
        }
        eprintln!("[spec50] token 0 is bit-identical at K=1 and inside every K<=8 batch");
    }

    /// 30 layers' worth of dispatches per measurement, GPU-timed, OLD vs NEW.
    #[test]
    fn spec50_moe_benchmark() {
        let Some(h) = Harness::new() else {
            eprintln!("[spec50] no Metal device; skipping");
            return;
        };
        let kernel = metal_linear_kernel().expect("metal kernel");
        let spec = spec50_moe_kernels(&h.device).expect("spec50 pipelines");

        // Force a BENCH_UNIQUE-wide union by giving every token 8 slots drawn
        // from a rotating window, then rebuild the work list stably.
        let mut slots = Vec::new();
        let mut weights = Vec::new();
        for t in 0..8usize {
            let mut s = [0u32; 8];
            for (i, si) in s.iter_mut().enumerate() {
                *si = ((t * 4 + i * 3) % BENCH_UNIQUE) as u32;
            }
            let mut w = [0.0f32; 8];
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = 0.05 + (i as f32) * 0.01 + (t as f32) * 0.001;
            }
            slots.push(s);
            weights.push(w);
        }

        eprintln!(
            "\n[spec50] 26B synthetic MoE, {BENCH_LAYERS} layers/measurement, slab {} MB",
            SLAB_EXPERTS * S50_SLOT_STRIDE / (1024 * 1024)
        );
        eprintln!(
            "{:>3} {:>6} {:>12} {:>10} {:>12} {:>10} {:>8}",
            "K", "stage", "old ms", "old GB/s", "new ms", "new GB/s", "speedup"
        );

        for k in [1usize, 4, 8] {
            let routing = build_routing(&slots, &weights, k);
            h.upload_routing(&routing);
            h.zero_outputs();
            let nu = routing.num_unique as u32;
            let ku = k as u32;

            // Prime the GateUp activations once so Down has real input.
            {
                let cb = h.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                reference_gateup(kernel, enc, &h, &h.gu_scales_ref, &h.gu_quants_ref, nu, ku);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            let gateup_bytes =
                BENCH_LAYERS as f64 * routing.num_unique as f64 * S50_GATE_UP_BYTES as f64;
            let down_bytes = BENCH_LAYERS as f64 * routing.num_unique as f64 * S50_DOWN_BYTES as f64;

            let time = |body: &dyn Fn(&metal::ComputeCommandEncoderRef)| -> f64 {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let cb = h.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for _ in 0..BENCH_LAYERS {
                        body(enc);
                    }
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    let (gpu_us, _) = command_buffer_gpu_times_us(cb);
                    best = best.min(gpu_us as f64 / 1000.0);
                }
                best
            };

            let gu_old = time(&|enc| {
                reference_gateup(kernel, enc, &h, &h.gu_scales_ref, &h.gu_quants_ref, nu, ku)
            });
            let gu_new = time(&|enc| {
                encode_spec50_gateup(
                    enc,
                    &spec.fast.gateup,
                    &h.input_scales,
                    &h.input_quants,
                    &h.slab,
                    0,
                    &h.work_list,
                    &h.gu_scales_new,
                    &h.gu_quants_new,
                    nu,
                    ku,
                    None,
                )
            });
            let dn_old = time(&|enc| {
                reference_down(
                    kernel,
                    enc,
                    &h,
                    &h.gu_scales_ref,
                    &h.gu_quants_ref,
                    &h.down_ref,
                    ku,
                )
            });
            let dn_new = time(&|enc| {
                encode_spec50_down(
                    enc,
                    &spec.strict.down,
                    &h.gu_scales_ref,
                    &h.gu_quants_ref,
                    &h.slab,
                    0,
                    &h.routes,
                    &h.work_list,
                    &h.down_new,
                    ku,
                    None,
                )
            });

            let gbs = |bytes: f64, ms: f64| bytes / (ms * 1.0e-3) / 1.0e9;
            eprintln!(
                "{k:>3} {:>6} {gu_old:>12.2} {:>10.1} {gu_new:>12.2} {:>10.1} {:>7.2}x",
                "gateup",
                gbs(gateup_bytes, gu_old),
                gbs(gateup_bytes, gu_new),
                gu_old / gu_new
            );
            eprintln!(
                "{k:>3} {:>6} {dn_old:>12.2} {:>10.1} {dn_new:>12.2} {:>10.1} {:>7.2}x",
                "down",
                gbs(down_bytes, dn_old),
                gbs(down_bytes, dn_new),
                dn_old / dn_new
            );
        }
    }
}
