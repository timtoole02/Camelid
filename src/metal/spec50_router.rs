//! spec50 router lane: a batched, bit-exact replacement for the gemma4 router GEMV.
//!
//! The shipping kernel (`gemma4_router_batch_k_f32`) launches one 256-thread
//! threadgroup per token, leaves 128 of those threads idle, and has each of the 128
//! active threads walk a serial 2816-long strided dot. Measured on a 26B-shaped
//! synthetic (hidden 2816, 128 experts, 30 layers per command buffer) it costs ~5.7 ms
//! at K=8 -- and ~5.9 ms at K=1. That flatness is the finding: the kernel is nowhere
//! near the ~0.48 ms byte floor for its 1.44 MB/layer matrix, it is bound by dependent
//! FMA latency at an occupancy of 8 threadgroups. Bytes were never the problem.
//!
//! What this module changes, in order of what it bought:
//!
//! * Layout. Each lane owns one `(token, expert)` pair, `lane = expert_sub * 8 +
//!   token_sub`, so a 32-lane simdgroup covers four expert rows for eight tokens and
//!   the matrix is read once per chunk. 32 threadgroups instead of 8, no idle half.
//!   Worth ~1.1x on its own -- confirming the diagnosis above.
//! * Unrolling. The dot is a chain of 2816 dependent FMAs with three loads each and
//!   nothing to hide the latency behind. Hoisting many iterations' loads ahead of the
//!   FMAs is the real lever, and it is worth ~4.6x.
//! * A prepared `r` stream. `r[t][i] = x[t][i] * factor[t] * scale[i]` is recomputed by
//!   all four lanes that share a token. Computing it once for the batch drops two
//!   multiplies and a whole load stream per element, which also frees the registers to
//!   unroll deeper. Worth the last ~1.2x, for 5.5x total and ~1.04 ms at K=8.
//!
//! EXACTNESS. Every logit is bit-identical to `gemma4_router_batch_k_f32` for every
//! K in 1..=8. Getting there needs the two halves of the lane compiled with DIFFERENT
//! options, which is the non-obvious part and is enforced by
//! `spec50_router_kernels()`:
//!
//! * The GEMV is compiled with fast math OFF. Fast math permits reassociating an FP
//!   reduction and does so the moment the loop is unrolled by hand -- measured at
//!   37-38 ULP across 4374/4608 logits, for every unrolled shape tried. Turning fast
//!   math off forbids that while leaving `dot += w * r` contracted to an FMA exactly as
//!   the shipping kernel has it, so the unrolled variants come out bit-identical.
//!   `spec50_router_gemv_exact` keeps the shipping loop verbatim as the control that
//!   proves the second half of that claim.
//! * The prepare pass is compiled with fast math ON, because the shipping kernel is,
//!   and its `1.0f / sqrt(...)` is an rsqrt there. Building the same source strict
//!   instead moves 1614/4608 logits by 6 ULP.
//!
//! `spec50_router_gemv_variant()` keeps all fourteen measured shapes alive, and
//! `spec50_router_gemv_variant_exactness_sweep` pins the ULP cost of each, so the
//! reasoning above is a test result rather than a comment.
//!
//! Everything here is additive: its own Metal libraries behind a `OnceLock`, free
//! encode functions with explicit buffers, no existing dispatch site touched.
#![allow(dead_code)]

use super::*;

/// Tokens one router-GEMV threadgroup handles (one chunk); K=8 is the whole scope.
pub(crate) const SPEC50_ROUTER_CHUNK_TOKENS: usize = 8;
/// Threads per router-GEMV threadgroup: one simdgroup = 4 experts x 8 tokens.
pub(crate) const SPEC50_ROUTER_TG_THREADS: u64 = 32;
/// Experts resolved by one router-GEMV threadgroup.
pub(crate) const SPEC50_ROUTER_EXPERTS_PER_TG: usize = 4;
/// Threads per RMS-factor threadgroup. MUST be 256: the halving tree below is the
/// shipping kernel's tree, and that tree's rounding depends on the thread count.
pub(crate) const SPEC50_ROUTER_RMS_TG_THREADS: u64 = 256;

/// Bytes the integrator must allocate for the RMS-factor scratch buffer: one f32 per
/// token in the batch. Metal buffers are page-backed, so this is a 16-byte floor.
pub(crate) fn spec50_router_factor_bytes(k_tokens: usize) -> u64 {
    ((k_tokens.max(1) * 4) as u64).max(16)
}

/// Bytes for the optional `r_stage` scratch used by the r-staged variants:
/// `k_tokens x hidden` f32.
pub(crate) fn spec50_router_r_stage_bytes(k_tokens: usize, hidden: usize) -> u64 {
    ((k_tokens.max(1) * hidden.max(1) * 4) as u64).max(16)
}

/// One source, compiled TWICE — once fast-math ON, once OFF. Which library each kernel
/// is taken from is load-bearing for bit equality; see the module header and
/// `spec50_router_kernels()`.
const SPEC50_ROUTER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define SPEC50_CHUNK 8u

// Phase 1 of gemma4_router_batch_k_f32, verbatim, hoisted out of the per-expert loop
// so the K factors are computed once per batch instead of once per (token, expert)
// threadgroup. Must be dispatched with exactly 256 threads per threadgroup.
kernel void spec50_router_rms_factor(
    device const float* input [[buffer(0)]],
    device float* out_factor [[buffer(1)]],
    constant uint& hidden [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& k_tokens [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tgsize [[threads_per_threadgroup]]
) {
    if (token_idx >= k_tokens) return;
    device const float* in_tok = input + token_idx * hidden;

    threadgroup float partial[256];
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tgsize) {
        float v = in_tok[i];
        local_ss += v * v;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = 1.0f / sqrt(partial[0] / float(hidden) + eps);
    float factor = rms_inv * (1.0f / sqrt(float(hidden)));
    if (tid == 0) {
        out_factor[token_idx] = factor;
    }
}

// Phase 2: one simdgroup = 4 expert rows x 8 tokens, one serial dot per lane.
// lane = expert_sub * 8 + token_sub, so the eight lanes sharing an expert broadcast
// the same weight address and the four lanes sharing a token broadcast the same
// activation address. No constraint on `hidden`.
kernel void spec50_router_gemv_exact(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    // Inactive lanes still walk the loop (they are in the same simdgroup anyway);
    // clamp their pointers so they stay inside the allocations.
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    // SPEC50 exactness note. Two independent things can break bit equality here:
    //   * reassociation of the accumulator, which fast math permits and performs the
    //     moment the loop is unrolled by hand (measured: 37-38 ULP);
    //   * the per-operation semantics, i.e. whether `dot += w * r` contracts to an FMA.
    // Compiling the GEMV with fast math OFF forbids the first and, as the strict-exact
    // control measures, leaves the second unchanged -- contraction still happens. That
    // is what makes the unrolled variants exact. This kernel keeps the loop verbatim so
    // it is exact from EITHER library and can serve as the control.
    // Textually the shipping kernel's dot loop, character for character. That is the
    // whole bit-exactness argument: same source text + same compile options => the
    // compiler makes the same contraction and reassociation decisions. Widening these
    // loads to float4 measurably breaks it (fast math then vectorises the accumulator
    // itself: 4374/9216 logits move, up to 38 ULP) — see spec50_router_gemv_vec4.
    float dot_val = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Same layout, loads for four iterations hoisted ahead of four strictly serial FMAs.
// Purely a memory-level-parallelism change: the accumulator order is untouched, so
// whether this stays bit-exact depends only on how fast math treats the unrolled adds.
kernel void spec50_router_gemv_u4(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 4u <= hidden; i += 4u) {
        float w0 = w_row[i + 0u];
        float w1 = w_row[i + 1u];
        float w2 = w_row[i + 2u];
        float w3 = w_row[i + 3u];
        float x0 = in_tok[i + 0u];
        float x1 = in_tok[i + 1u];
        float x2 = in_tok[i + 2u];
        float x3 = in_tok[i + 3u];
        float c0 = gate_inp_scale[i + 0u];
        float c1 = gate_inp_scale[i + 1u];
        float c2 = gate_inp_scale[i + 2u];
        float c3 = gate_inp_scale[i + 3u];
        float r0 = x0 * factor * c0;
        dot_val += w0 * r0;
        float r1 = x1 * factor * c1;
        dot_val += w1 * r1;
        float r2 = x2 * factor * c2;
        dot_val += w2 * r2;
        float r3 = x3 * factor * c3;
        dot_val += w3 * r3;
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// As spec50_router_gemv_u4 with eight iterations of load lookahead.
kernel void spec50_router_gemv_u8(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 8u <= hidden; i += 8u) {
        float w0 = w_row[i + 0u];
        float w1 = w_row[i + 1u];
        float w2 = w_row[i + 2u];
        float w3 = w_row[i + 3u];
        float w4 = w_row[i + 4u];
        float w5 = w_row[i + 5u];
        float w6 = w_row[i + 6u];
        float w7 = w_row[i + 7u];
        float x0 = in_tok[i + 0u];
        float x1 = in_tok[i + 1u];
        float x2 = in_tok[i + 2u];
        float x3 = in_tok[i + 3u];
        float x4 = in_tok[i + 4u];
        float x5 = in_tok[i + 5u];
        float x6 = in_tok[i + 6u];
        float x7 = in_tok[i + 7u];
        float c0 = gate_inp_scale[i + 0u];
        float c1 = gate_inp_scale[i + 1u];
        float c2 = gate_inp_scale[i + 2u];
        float c3 = gate_inp_scale[i + 3u];
        float c4 = gate_inp_scale[i + 4u];
        float c5 = gate_inp_scale[i + 5u];
        float c6 = gate_inp_scale[i + 6u];
        float c7 = gate_inp_scale[i + 7u];
        float r0 = x0 * factor * c0;
        dot_val += w0 * r0;
        float r1 = x1 * factor * c1;
        dot_val += w1 * r1;
        float r2 = x2 * factor * c2;
        dot_val += w2 * r2;
        float r3 = x3 * factor * c3;
        dot_val += w3 * r3;
        float r4 = x4 * factor * c4;
        dot_val += w4 * r4;
        float r5 = x5 * factor * c5;
        dot_val += w5 * r5;
        float r6 = x6 * factor * c6;
        dot_val += w6 * r6;
        float r7 = x7 * factor * c7;
        dot_val += w7 * r7;
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Sixteen iterations of scalar load lookahead ahead of sixteen strictly serial FMAs.
kernel void spec50_router_gemv_u16(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 16u <= hidden; i += 16u) {
        float w0 = w_row[i + 0u];
        float w1 = w_row[i + 1u];
        float w2 = w_row[i + 2u];
        float w3 = w_row[i + 3u];
        float w4 = w_row[i + 4u];
        float w5 = w_row[i + 5u];
        float w6 = w_row[i + 6u];
        float w7 = w_row[i + 7u];
        float w8 = w_row[i + 8u];
        float w9 = w_row[i + 9u];
        float w10 = w_row[i + 10u];
        float w11 = w_row[i + 11u];
        float w12 = w_row[i + 12u];
        float w13 = w_row[i + 13u];
        float w14 = w_row[i + 14u];
        float w15 = w_row[i + 15u];
        float x0 = in_tok[i + 0u];
        float x1 = in_tok[i + 1u];
        float x2 = in_tok[i + 2u];
        float x3 = in_tok[i + 3u];
        float x4 = in_tok[i + 4u];
        float x5 = in_tok[i + 5u];
        float x6 = in_tok[i + 6u];
        float x7 = in_tok[i + 7u];
        float x8 = in_tok[i + 8u];
        float x9 = in_tok[i + 9u];
        float x10 = in_tok[i + 10u];
        float x11 = in_tok[i + 11u];
        float x12 = in_tok[i + 12u];
        float x13 = in_tok[i + 13u];
        float x14 = in_tok[i + 14u];
        float x15 = in_tok[i + 15u];
        float c0 = gate_inp_scale[i + 0u];
        float c1 = gate_inp_scale[i + 1u];
        float c2 = gate_inp_scale[i + 2u];
        float c3 = gate_inp_scale[i + 3u];
        float c4 = gate_inp_scale[i + 4u];
        float c5 = gate_inp_scale[i + 5u];
        float c6 = gate_inp_scale[i + 6u];
        float c7 = gate_inp_scale[i + 7u];
        float c8 = gate_inp_scale[i + 8u];
        float c9 = gate_inp_scale[i + 9u];
        float c10 = gate_inp_scale[i + 10u];
        float c11 = gate_inp_scale[i + 11u];
        float c12 = gate_inp_scale[i + 12u];
        float c13 = gate_inp_scale[i + 13u];
        float c14 = gate_inp_scale[i + 14u];
        float c15 = gate_inp_scale[i + 15u];
        float r0 = x0 * factor * c0;
        dot_val += w0 * r0;
        float r1 = x1 * factor * c1;
        dot_val += w1 * r1;
        float r2 = x2 * factor * c2;
        dot_val += w2 * r2;
        float r3 = x3 * factor * c3;
        dot_val += w3 * r3;
        float r4 = x4 * factor * c4;
        dot_val += w4 * r4;
        float r5 = x5 * factor * c5;
        dot_val += w5 * r5;
        float r6 = x6 * factor * c6;
        dot_val += w6 * r6;
        float r7 = x7 * factor * c7;
        dot_val += w7 * r7;
        float r8 = x8 * factor * c8;
        dot_val += w8 * r8;
        float r9 = x9 * factor * c9;
        dot_val += w9 * r9;
        float r10 = x10 * factor * c10;
        dot_val += w10 * r10;
        float r11 = x11 * factor * c11;
        dot_val += w11 * r11;
        float r12 = x12 * factor * c12;
        dot_val += w12 * r12;
        float r13 = x13 * factor * c13;
        dot_val += w13 * r13;
        float r14 = x14 * factor * c14;
        dot_val += w14 * r14;
        float r15 = x15 * factor * c15;
        dot_val += w15 * r15;
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Sixteen elements per step as four float4 loads per stream: wide loads AND deep
// lookahead. Serial accumulation, so it is exact under a fast-math-OFF compile.
kernel void spec50_router_gemv_v4x4(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 16u <= hidden; i += 16u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 x0 = *reinterpret_cast<device const float4*>(in_tok + i + 0u);
        float4 x1 = *reinterpret_cast<device const float4*>(in_tok + i + 4u);
        float4 x2 = *reinterpret_cast<device const float4*>(in_tok + i + 8u);
        float4 x3 = *reinterpret_cast<device const float4*>(in_tok + i + 12u);
        float4 c0 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 0u);
        float4 c1 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 4u);
        float4 c2 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 8u);
        float4 c3 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 12u);
        dot_val += w0.x * (x0.x * factor * c0.x);
        dot_val += w0.y * (x0.y * factor * c0.y);
        dot_val += w0.z * (x0.z * factor * c0.z);
        dot_val += w0.w * (x0.w * factor * c0.w);
        dot_val += w1.x * (x1.x * factor * c1.x);
        dot_val += w1.y * (x1.y * factor * c1.y);
        dot_val += w1.z * (x1.z * factor * c1.z);
        dot_val += w1.w * (x1.w * factor * c1.w);
        dot_val += w2.x * (x2.x * factor * c2.x);
        dot_val += w2.y * (x2.y * factor * c2.y);
        dot_val += w2.z * (x2.z * factor * c2.z);
        dot_val += w2.w * (x2.w * factor * c2.w);
        dot_val += w3.x * (x3.x * factor * c3.x);
        dot_val += w3.y * (x3.y * factor * c3.y);
        dot_val += w3.z * (x3.z * factor * c3.z);
        dot_val += w3.w * (x3.w * factor * c3.w);
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Thirty-two elements per step, eight float4 loads per stream.
kernel void spec50_router_gemv_v4x8(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 32u <= hidden; i += 32u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i + 16u);
        float4 w5 = *reinterpret_cast<device const float4*>(w_row + i + 20u);
        float4 w6 = *reinterpret_cast<device const float4*>(w_row + i + 24u);
        float4 w7 = *reinterpret_cast<device const float4*>(w_row + i + 28u);
        float4 x0 = *reinterpret_cast<device const float4*>(in_tok + i + 0u);
        float4 x1 = *reinterpret_cast<device const float4*>(in_tok + i + 4u);
        float4 x2 = *reinterpret_cast<device const float4*>(in_tok + i + 8u);
        float4 x3 = *reinterpret_cast<device const float4*>(in_tok + i + 12u);
        float4 x4 = *reinterpret_cast<device const float4*>(in_tok + i + 16u);
        float4 x5 = *reinterpret_cast<device const float4*>(in_tok + i + 20u);
        float4 x6 = *reinterpret_cast<device const float4*>(in_tok + i + 24u);
        float4 x7 = *reinterpret_cast<device const float4*>(in_tok + i + 28u);
        float4 c0 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 0u);
        float4 c1 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 4u);
        float4 c2 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 8u);
        float4 c3 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 12u);
        float4 c4 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 16u);
        float4 c5 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 20u);
        float4 c6 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 24u);
        float4 c7 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 28u);
        dot_val += w0.x * (x0.x * factor * c0.x);
        dot_val += w0.y * (x0.y * factor * c0.y);
        dot_val += w0.z * (x0.z * factor * c0.z);
        dot_val += w0.w * (x0.w * factor * c0.w);
        dot_val += w1.x * (x1.x * factor * c1.x);
        dot_val += w1.y * (x1.y * factor * c1.y);
        dot_val += w1.z * (x1.z * factor * c1.z);
        dot_val += w1.w * (x1.w * factor * c1.w);
        dot_val += w2.x * (x2.x * factor * c2.x);
        dot_val += w2.y * (x2.y * factor * c2.y);
        dot_val += w2.z * (x2.z * factor * c2.z);
        dot_val += w2.w * (x2.w * factor * c2.w);
        dot_val += w3.x * (x3.x * factor * c3.x);
        dot_val += w3.y * (x3.y * factor * c3.y);
        dot_val += w3.z * (x3.z * factor * c3.z);
        dot_val += w3.w * (x3.w * factor * c3.w);
        dot_val += w4.x * (x4.x * factor * c4.x);
        dot_val += w4.y * (x4.y * factor * c4.y);
        dot_val += w4.z * (x4.z * factor * c4.z);
        dot_val += w4.w * (x4.w * factor * c4.w);
        dot_val += w5.x * (x5.x * factor * c5.x);
        dot_val += w5.y * (x5.y * factor * c5.y);
        dot_val += w5.z * (x5.z * factor * c5.z);
        dot_val += w5.w * (x5.w * factor * c5.w);
        dot_val += w6.x * (x6.x * factor * c6.x);
        dot_val += w6.y * (x6.y * factor * c6.y);
        dot_val += w6.z * (x6.z * factor * c6.z);
        dot_val += w6.w * (x6.w * factor * c6.w);
        dot_val += w7.x * (x7.x * factor * c7.x);
        dot_val += w7.y * (x7.y * factor * c7.y);
        dot_val += w7.z * (x7.z * factor * c7.z);
        dot_val += w7.w * (x7.w * factor * c7.w);
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}


// Precomputes r[t][i] = x[t][i] * factor[t] * scale[i] once for the batch, in the
// shipping kernel's operand order. The GEMV then reads ONE prepared stream instead of
// recomputing r in all four lanes that share a token: it drops two multiplies per
// element and a whole load stream, which is what the register budget needs to unroll
// deep. Costs a k x hidden f32 scratch buffer.
// Fused prepare pass for the r-staged variants. MUST be taken from the fast-math-ON
// library: the shipping kernel is compiled with fast math, so `1.0f / sqrt(...)` there
// is an rsqrt and `partial[0] / float(hidden)` a reciprocal multiply. Building this
// same source with fast math OFF instead moves 1614/4608 logits by 6 ULP -- the factor
// itself changes, and every expert of the affected tokens moves with it. The dot loop
// wants the opposite (see SPEC50 exactness note on the GEMV kernels), so the two halves
// of this lane are deliberately compiled from two different libraries.
//
// One 256-thread threadgroup per token
// computes that token's RMS factor with the shipping kernel's phase 1 verbatim and
// then writes r[t][i] = x[t][i] * factor * scale[i] for the whole row. Fusing the two
// prepare dispatches matters at this size: the GEMV is down to ~1 ms for 30 layers,
// so 30 saved dispatches are a measurable share of it.
kernel void spec50_router_prepare(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device float* out_r [[buffer(2)]],
    device float* out_factor [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& k_tokens [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tgsize [[threads_per_threadgroup]]
) {
    if (token_idx >= k_tokens) return;
    device const float* in_tok = input + token_idx * hidden;

    threadgroup float partial[256];
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tgsize) {
        float v = in_tok[i];
        local_ss += v * v;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = 1.0f / sqrt(partial[0] / float(hidden) + eps);
    float factor = rms_inv * (1.0f / sqrt(float(hidden)));
    if (tid == 0) {
        out_factor[token_idx] = factor;
    }
    device float* r_tok = out_r + token_idx * hidden;
    for (uint i = tid; i < hidden; i += tgsize) {
        r_tok[i] = in_tok[i] * factor * gate_inp_scale[i];
    }
}

kernel void spec50_router_r_precompute(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device float* out_r [[buffer(2)]],
    device const float* factors [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& k_tokens [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint t = gid.y;
    if (i >= hidden || t >= k_tokens) return;
    device const float* in_tok = input + t * hidden;
    const float factor = factors[t];
    out_r[t * hidden + i] = in_tok[i] * factor * gate_inp_scale[i];
}

// Sixty-four elements per step, three streams. Register-pressure probe.
kernel void spec50_router_gemv_v4x16(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 64u <= hidden; i += 64u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i + 16u);
        float4 w5 = *reinterpret_cast<device const float4*>(w_row + i + 20u);
        float4 w6 = *reinterpret_cast<device const float4*>(w_row + i + 24u);
        float4 w7 = *reinterpret_cast<device const float4*>(w_row + i + 28u);
        float4 w8 = *reinterpret_cast<device const float4*>(w_row + i + 32u);
        float4 w9 = *reinterpret_cast<device const float4*>(w_row + i + 36u);
        float4 w10 = *reinterpret_cast<device const float4*>(w_row + i + 40u);
        float4 w11 = *reinterpret_cast<device const float4*>(w_row + i + 44u);
        float4 w12 = *reinterpret_cast<device const float4*>(w_row + i + 48u);
        float4 w13 = *reinterpret_cast<device const float4*>(w_row + i + 52u);
        float4 w14 = *reinterpret_cast<device const float4*>(w_row + i + 56u);
        float4 w15 = *reinterpret_cast<device const float4*>(w_row + i + 60u);
        float4 x0 = *reinterpret_cast<device const float4*>(in_tok + i + 0u);
        float4 x1 = *reinterpret_cast<device const float4*>(in_tok + i + 4u);
        float4 x2 = *reinterpret_cast<device const float4*>(in_tok + i + 8u);
        float4 x3 = *reinterpret_cast<device const float4*>(in_tok + i + 12u);
        float4 x4 = *reinterpret_cast<device const float4*>(in_tok + i + 16u);
        float4 x5 = *reinterpret_cast<device const float4*>(in_tok + i + 20u);
        float4 x6 = *reinterpret_cast<device const float4*>(in_tok + i + 24u);
        float4 x7 = *reinterpret_cast<device const float4*>(in_tok + i + 28u);
        float4 x8 = *reinterpret_cast<device const float4*>(in_tok + i + 32u);
        float4 x9 = *reinterpret_cast<device const float4*>(in_tok + i + 36u);
        float4 x10 = *reinterpret_cast<device const float4*>(in_tok + i + 40u);
        float4 x11 = *reinterpret_cast<device const float4*>(in_tok + i + 44u);
        float4 x12 = *reinterpret_cast<device const float4*>(in_tok + i + 48u);
        float4 x13 = *reinterpret_cast<device const float4*>(in_tok + i + 52u);
        float4 x14 = *reinterpret_cast<device const float4*>(in_tok + i + 56u);
        float4 x15 = *reinterpret_cast<device const float4*>(in_tok + i + 60u);
        float4 c0 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 0u);
        float4 c1 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 4u);
        float4 c2 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 8u);
        float4 c3 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 12u);
        float4 c4 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 16u);
        float4 c5 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 20u);
        float4 c6 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 24u);
        float4 c7 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 28u);
        float4 c8 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 32u);
        float4 c9 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 36u);
        float4 c10 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 40u);
        float4 c11 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 44u);
        float4 c12 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 48u);
        float4 c13 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 52u);
        float4 c14 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 56u);
        float4 c15 = *reinterpret_cast<device const float4*>(gate_inp_scale + i + 60u);
        dot_val += w0.x * (x0.x * factor * c0.x);
        dot_val += w0.y * (x0.y * factor * c0.y);
        dot_val += w0.z * (x0.z * factor * c0.z);
        dot_val += w0.w * (x0.w * factor * c0.w);
        dot_val += w1.x * (x1.x * factor * c1.x);
        dot_val += w1.y * (x1.y * factor * c1.y);
        dot_val += w1.z * (x1.z * factor * c1.z);
        dot_val += w1.w * (x1.w * factor * c1.w);
        dot_val += w2.x * (x2.x * factor * c2.x);
        dot_val += w2.y * (x2.y * factor * c2.y);
        dot_val += w2.z * (x2.z * factor * c2.z);
        dot_val += w2.w * (x2.w * factor * c2.w);
        dot_val += w3.x * (x3.x * factor * c3.x);
        dot_val += w3.y * (x3.y * factor * c3.y);
        dot_val += w3.z * (x3.z * factor * c3.z);
        dot_val += w3.w * (x3.w * factor * c3.w);
        dot_val += w4.x * (x4.x * factor * c4.x);
        dot_val += w4.y * (x4.y * factor * c4.y);
        dot_val += w4.z * (x4.z * factor * c4.z);
        dot_val += w4.w * (x4.w * factor * c4.w);
        dot_val += w5.x * (x5.x * factor * c5.x);
        dot_val += w5.y * (x5.y * factor * c5.y);
        dot_val += w5.z * (x5.z * factor * c5.z);
        dot_val += w5.w * (x5.w * factor * c5.w);
        dot_val += w6.x * (x6.x * factor * c6.x);
        dot_val += w6.y * (x6.y * factor * c6.y);
        dot_val += w6.z * (x6.z * factor * c6.z);
        dot_val += w6.w * (x6.w * factor * c6.w);
        dot_val += w7.x * (x7.x * factor * c7.x);
        dot_val += w7.y * (x7.y * factor * c7.y);
        dot_val += w7.z * (x7.z * factor * c7.z);
        dot_val += w7.w * (x7.w * factor * c7.w);
        dot_val += w8.x * (x8.x * factor * c8.x);
        dot_val += w8.y * (x8.y * factor * c8.y);
        dot_val += w8.z * (x8.z * factor * c8.z);
        dot_val += w8.w * (x8.w * factor * c8.w);
        dot_val += w9.x * (x9.x * factor * c9.x);
        dot_val += w9.y * (x9.y * factor * c9.y);
        dot_val += w9.z * (x9.z * factor * c9.z);
        dot_val += w9.w * (x9.w * factor * c9.w);
        dot_val += w10.x * (x10.x * factor * c10.x);
        dot_val += w10.y * (x10.y * factor * c10.y);
        dot_val += w10.z * (x10.z * factor * c10.z);
        dot_val += w10.w * (x10.w * factor * c10.w);
        dot_val += w11.x * (x11.x * factor * c11.x);
        dot_val += w11.y * (x11.y * factor * c11.y);
        dot_val += w11.z * (x11.z * factor * c11.z);
        dot_val += w11.w * (x11.w * factor * c11.w);
        dot_val += w12.x * (x12.x * factor * c12.x);
        dot_val += w12.y * (x12.y * factor * c12.y);
        dot_val += w12.z * (x12.z * factor * c12.z);
        dot_val += w12.w * (x12.w * factor * c12.w);
        dot_val += w13.x * (x13.x * factor * c13.x);
        dot_val += w13.y * (x13.y * factor * c13.y);
        dot_val += w13.z * (x13.z * factor * c13.z);
        dot_val += w13.w * (x13.w * factor * c13.w);
        dot_val += w14.x * (x14.x * factor * c14.x);
        dot_val += w14.y * (x14.y * factor * c14.y);
        dot_val += w14.z * (x14.z * factor * c14.z);
        dot_val += w14.w * (x14.w * factor * c14.w);
        dot_val += w15.x * (x15.x * factor * c15.x);
        dot_val += w15.y * (x15.y * factor * c15.y);
        dot_val += w15.z * (x15.z * factor * c15.z);
        dot_val += w15.w * (x15.w * factor * c15.w);
    }
    for (; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Thirty-two elements per step over the precomputed r stream (two streams).
kernel void spec50_router_gemv_rv4x8(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    device const float* r_stage [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];
    device const float* r_row = r_stage + (tok0 + tc) * hidden;

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 32u <= hidden; i += 32u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i + 16u);
        float4 w5 = *reinterpret_cast<device const float4*>(w_row + i + 20u);
        float4 w6 = *reinterpret_cast<device const float4*>(w_row + i + 24u);
        float4 w7 = *reinterpret_cast<device const float4*>(w_row + i + 28u);
        float4 r0 = *reinterpret_cast<device const float4*>(r_row + i + 0u);
        float4 r1 = *reinterpret_cast<device const float4*>(r_row + i + 4u);
        float4 r2 = *reinterpret_cast<device const float4*>(r_row + i + 8u);
        float4 r3 = *reinterpret_cast<device const float4*>(r_row + i + 12u);
        float4 r4 = *reinterpret_cast<device const float4*>(r_row + i + 16u);
        float4 r5 = *reinterpret_cast<device const float4*>(r_row + i + 20u);
        float4 r6 = *reinterpret_cast<device const float4*>(r_row + i + 24u);
        float4 r7 = *reinterpret_cast<device const float4*>(r_row + i + 28u);
        dot_val += w0.x * r0.x;
        dot_val += w0.y * r0.y;
        dot_val += w0.z * r0.z;
        dot_val += w0.w * r0.w;
        dot_val += w1.x * r1.x;
        dot_val += w1.y * r1.y;
        dot_val += w1.z * r1.z;
        dot_val += w1.w * r1.w;
        dot_val += w2.x * r2.x;
        dot_val += w2.y * r2.y;
        dot_val += w2.z * r2.z;
        dot_val += w2.w * r2.w;
        dot_val += w3.x * r3.x;
        dot_val += w3.y * r3.y;
        dot_val += w3.z * r3.z;
        dot_val += w3.w * r3.w;
        dot_val += w4.x * r4.x;
        dot_val += w4.y * r4.y;
        dot_val += w4.z * r4.z;
        dot_val += w4.w * r4.w;
        dot_val += w5.x * r5.x;
        dot_val += w5.y * r5.y;
        dot_val += w5.z * r5.z;
        dot_val += w5.w * r5.w;
        dot_val += w6.x * r6.x;
        dot_val += w6.y * r6.y;
        dot_val += w6.z * r6.z;
        dot_val += w6.w * r6.w;
        dot_val += w7.x * r7.x;
        dot_val += w7.y * r7.y;
        dot_val += w7.z * r7.z;
        dot_val += w7.w * r7.w;
    }
    for (; i < hidden; ++i) {
        float r_i = r_row[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// Sixty-four elements per step over the precomputed r stream.
kernel void spec50_router_gemv_rv4x16(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    device const float* r_stage [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];
    device const float* r_row = r_stage + (tok0 + tc) * hidden;

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 64u <= hidden; i += 64u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i + 16u);
        float4 w5 = *reinterpret_cast<device const float4*>(w_row + i + 20u);
        float4 w6 = *reinterpret_cast<device const float4*>(w_row + i + 24u);
        float4 w7 = *reinterpret_cast<device const float4*>(w_row + i + 28u);
        float4 w8 = *reinterpret_cast<device const float4*>(w_row + i + 32u);
        float4 w9 = *reinterpret_cast<device const float4*>(w_row + i + 36u);
        float4 w10 = *reinterpret_cast<device const float4*>(w_row + i + 40u);
        float4 w11 = *reinterpret_cast<device const float4*>(w_row + i + 44u);
        float4 w12 = *reinterpret_cast<device const float4*>(w_row + i + 48u);
        float4 w13 = *reinterpret_cast<device const float4*>(w_row + i + 52u);
        float4 w14 = *reinterpret_cast<device const float4*>(w_row + i + 56u);
        float4 w15 = *reinterpret_cast<device const float4*>(w_row + i + 60u);
        float4 r0 = *reinterpret_cast<device const float4*>(r_row + i + 0u);
        float4 r1 = *reinterpret_cast<device const float4*>(r_row + i + 4u);
        float4 r2 = *reinterpret_cast<device const float4*>(r_row + i + 8u);
        float4 r3 = *reinterpret_cast<device const float4*>(r_row + i + 12u);
        float4 r4 = *reinterpret_cast<device const float4*>(r_row + i + 16u);
        float4 r5 = *reinterpret_cast<device const float4*>(r_row + i + 20u);
        float4 r6 = *reinterpret_cast<device const float4*>(r_row + i + 24u);
        float4 r7 = *reinterpret_cast<device const float4*>(r_row + i + 28u);
        float4 r8 = *reinterpret_cast<device const float4*>(r_row + i + 32u);
        float4 r9 = *reinterpret_cast<device const float4*>(r_row + i + 36u);
        float4 r10 = *reinterpret_cast<device const float4*>(r_row + i + 40u);
        float4 r11 = *reinterpret_cast<device const float4*>(r_row + i + 44u);
        float4 r12 = *reinterpret_cast<device const float4*>(r_row + i + 48u);
        float4 r13 = *reinterpret_cast<device const float4*>(r_row + i + 52u);
        float4 r14 = *reinterpret_cast<device const float4*>(r_row + i + 56u);
        float4 r15 = *reinterpret_cast<device const float4*>(r_row + i + 60u);
        dot_val += w0.x * r0.x;
        dot_val += w0.y * r0.y;
        dot_val += w0.z * r0.z;
        dot_val += w0.w * r0.w;
        dot_val += w1.x * r1.x;
        dot_val += w1.y * r1.y;
        dot_val += w1.z * r1.z;
        dot_val += w1.w * r1.w;
        dot_val += w2.x * r2.x;
        dot_val += w2.y * r2.y;
        dot_val += w2.z * r2.z;
        dot_val += w2.w * r2.w;
        dot_val += w3.x * r3.x;
        dot_val += w3.y * r3.y;
        dot_val += w3.z * r3.z;
        dot_val += w3.w * r3.w;
        dot_val += w4.x * r4.x;
        dot_val += w4.y * r4.y;
        dot_val += w4.z * r4.z;
        dot_val += w4.w * r4.w;
        dot_val += w5.x * r5.x;
        dot_val += w5.y * r5.y;
        dot_val += w5.z * r5.z;
        dot_val += w5.w * r5.w;
        dot_val += w6.x * r6.x;
        dot_val += w6.y * r6.y;
        dot_val += w6.z * r6.z;
        dot_val += w6.w * r6.w;
        dot_val += w7.x * r7.x;
        dot_val += w7.y * r7.y;
        dot_val += w7.z * r7.z;
        dot_val += w7.w * r7.w;
        dot_val += w8.x * r8.x;
        dot_val += w8.y * r8.y;
        dot_val += w8.z * r8.z;
        dot_val += w8.w * r8.w;
        dot_val += w9.x * r9.x;
        dot_val += w9.y * r9.y;
        dot_val += w9.z * r9.z;
        dot_val += w9.w * r9.w;
        dot_val += w10.x * r10.x;
        dot_val += w10.y * r10.y;
        dot_val += w10.z * r10.z;
        dot_val += w10.w * r10.w;
        dot_val += w11.x * r11.x;
        dot_val += w11.y * r11.y;
        dot_val += w11.z * r11.z;
        dot_val += w11.w * r11.w;
        dot_val += w12.x * r12.x;
        dot_val += w12.y * r12.y;
        dot_val += w12.z * r12.z;
        dot_val += w12.w * r12.w;
        dot_val += w13.x * r13.x;
        dot_val += w13.y * r13.y;
        dot_val += w13.z * r13.z;
        dot_val += w13.w * r13.w;
        dot_val += w14.x * r14.x;
        dot_val += w14.y * r14.y;
        dot_val += w14.z * r14.z;
        dot_val += w14.w * r14.w;
        dot_val += w15.x * r15.x;
        dot_val += w15.y * r15.y;
        dot_val += w15.z * r15.z;
        dot_val += w15.w * r15.w;
    }
    for (; i < hidden; ++i) {
        float r_i = r_row[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// 96 elements per step over the precomputed r stream.
kernel void spec50_router_gemv_rv4x24(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    device const float* r_stage [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];
    device const float* r_row = r_stage + (tok0 + tc) * hidden;

    float dot_val = 0.0f;
    uint i = 0;
    for (; i + 96u <= hidden; i += 96u) {
        float4 w0 = *reinterpret_cast<device const float4*>(w_row + i + 0u);
        float4 w1 = *reinterpret_cast<device const float4*>(w_row + i + 4u);
        float4 w2 = *reinterpret_cast<device const float4*>(w_row + i + 8u);
        float4 w3 = *reinterpret_cast<device const float4*>(w_row + i + 12u);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i + 16u);
        float4 w5 = *reinterpret_cast<device const float4*>(w_row + i + 20u);
        float4 w6 = *reinterpret_cast<device const float4*>(w_row + i + 24u);
        float4 w7 = *reinterpret_cast<device const float4*>(w_row + i + 28u);
        float4 w8 = *reinterpret_cast<device const float4*>(w_row + i + 32u);
        float4 w9 = *reinterpret_cast<device const float4*>(w_row + i + 36u);
        float4 w10 = *reinterpret_cast<device const float4*>(w_row + i + 40u);
        float4 w11 = *reinterpret_cast<device const float4*>(w_row + i + 44u);
        float4 w12 = *reinterpret_cast<device const float4*>(w_row + i + 48u);
        float4 w13 = *reinterpret_cast<device const float4*>(w_row + i + 52u);
        float4 w14 = *reinterpret_cast<device const float4*>(w_row + i + 56u);
        float4 w15 = *reinterpret_cast<device const float4*>(w_row + i + 60u);
        float4 w16 = *reinterpret_cast<device const float4*>(w_row + i + 64u);
        float4 w17 = *reinterpret_cast<device const float4*>(w_row + i + 68u);
        float4 w18 = *reinterpret_cast<device const float4*>(w_row + i + 72u);
        float4 w19 = *reinterpret_cast<device const float4*>(w_row + i + 76u);
        float4 w20 = *reinterpret_cast<device const float4*>(w_row + i + 80u);
        float4 w21 = *reinterpret_cast<device const float4*>(w_row + i + 84u);
        float4 w22 = *reinterpret_cast<device const float4*>(w_row + i + 88u);
        float4 w23 = *reinterpret_cast<device const float4*>(w_row + i + 92u);
        float4 r0 = *reinterpret_cast<device const float4*>(r_row + i + 0u);
        float4 r1 = *reinterpret_cast<device const float4*>(r_row + i + 4u);
        float4 r2 = *reinterpret_cast<device const float4*>(r_row + i + 8u);
        float4 r3 = *reinterpret_cast<device const float4*>(r_row + i + 12u);
        float4 r4 = *reinterpret_cast<device const float4*>(r_row + i + 16u);
        float4 r5 = *reinterpret_cast<device const float4*>(r_row + i + 20u);
        float4 r6 = *reinterpret_cast<device const float4*>(r_row + i + 24u);
        float4 r7 = *reinterpret_cast<device const float4*>(r_row + i + 28u);
        float4 r8 = *reinterpret_cast<device const float4*>(r_row + i + 32u);
        float4 r9 = *reinterpret_cast<device const float4*>(r_row + i + 36u);
        float4 r10 = *reinterpret_cast<device const float4*>(r_row + i + 40u);
        float4 r11 = *reinterpret_cast<device const float4*>(r_row + i + 44u);
        float4 r12 = *reinterpret_cast<device const float4*>(r_row + i + 48u);
        float4 r13 = *reinterpret_cast<device const float4*>(r_row + i + 52u);
        float4 r14 = *reinterpret_cast<device const float4*>(r_row + i + 56u);
        float4 r15 = *reinterpret_cast<device const float4*>(r_row + i + 60u);
        float4 r16 = *reinterpret_cast<device const float4*>(r_row + i + 64u);
        float4 r17 = *reinterpret_cast<device const float4*>(r_row + i + 68u);
        float4 r18 = *reinterpret_cast<device const float4*>(r_row + i + 72u);
        float4 r19 = *reinterpret_cast<device const float4*>(r_row + i + 76u);
        float4 r20 = *reinterpret_cast<device const float4*>(r_row + i + 80u);
        float4 r21 = *reinterpret_cast<device const float4*>(r_row + i + 84u);
        float4 r22 = *reinterpret_cast<device const float4*>(r_row + i + 88u);
        float4 r23 = *reinterpret_cast<device const float4*>(r_row + i + 92u);
        dot_val += w0.x * r0.x;
        dot_val += w0.y * r0.y;
        dot_val += w0.z * r0.z;
        dot_val += w0.w * r0.w;
        dot_val += w1.x * r1.x;
        dot_val += w1.y * r1.y;
        dot_val += w1.z * r1.z;
        dot_val += w1.w * r1.w;
        dot_val += w2.x * r2.x;
        dot_val += w2.y * r2.y;
        dot_val += w2.z * r2.z;
        dot_val += w2.w * r2.w;
        dot_val += w3.x * r3.x;
        dot_val += w3.y * r3.y;
        dot_val += w3.z * r3.z;
        dot_val += w3.w * r3.w;
        dot_val += w4.x * r4.x;
        dot_val += w4.y * r4.y;
        dot_val += w4.z * r4.z;
        dot_val += w4.w * r4.w;
        dot_val += w5.x * r5.x;
        dot_val += w5.y * r5.y;
        dot_val += w5.z * r5.z;
        dot_val += w5.w * r5.w;
        dot_val += w6.x * r6.x;
        dot_val += w6.y * r6.y;
        dot_val += w6.z * r6.z;
        dot_val += w6.w * r6.w;
        dot_val += w7.x * r7.x;
        dot_val += w7.y * r7.y;
        dot_val += w7.z * r7.z;
        dot_val += w7.w * r7.w;
        dot_val += w8.x * r8.x;
        dot_val += w8.y * r8.y;
        dot_val += w8.z * r8.z;
        dot_val += w8.w * r8.w;
        dot_val += w9.x * r9.x;
        dot_val += w9.y * r9.y;
        dot_val += w9.z * r9.z;
        dot_val += w9.w * r9.w;
        dot_val += w10.x * r10.x;
        dot_val += w10.y * r10.y;
        dot_val += w10.z * r10.z;
        dot_val += w10.w * r10.w;
        dot_val += w11.x * r11.x;
        dot_val += w11.y * r11.y;
        dot_val += w11.z * r11.z;
        dot_val += w11.w * r11.w;
        dot_val += w12.x * r12.x;
        dot_val += w12.y * r12.y;
        dot_val += w12.z * r12.z;
        dot_val += w12.w * r12.w;
        dot_val += w13.x * r13.x;
        dot_val += w13.y * r13.y;
        dot_val += w13.z * r13.z;
        dot_val += w13.w * r13.w;
        dot_val += w14.x * r14.x;
        dot_val += w14.y * r14.y;
        dot_val += w14.z * r14.z;
        dot_val += w14.w * r14.w;
        dot_val += w15.x * r15.x;
        dot_val += w15.y * r15.y;
        dot_val += w15.z * r15.z;
        dot_val += w15.w * r15.w;
        dot_val += w16.x * r16.x;
        dot_val += w16.y * r16.y;
        dot_val += w16.z * r16.z;
        dot_val += w16.w * r16.w;
        dot_val += w17.x * r17.x;
        dot_val += w17.y * r17.y;
        dot_val += w17.z * r17.z;
        dot_val += w17.w * r17.w;
        dot_val += w18.x * r18.x;
        dot_val += w18.y * r18.y;
        dot_val += w18.z * r18.z;
        dot_val += w18.w * r18.w;
        dot_val += w19.x * r19.x;
        dot_val += w19.y * r19.y;
        dot_val += w19.z * r19.z;
        dot_val += w19.w * r19.w;
        dot_val += w20.x * r20.x;
        dot_val += w20.y * r20.y;
        dot_val += w20.z * r20.z;
        dot_val += w20.w * r20.w;
        dot_val += w21.x * r21.x;
        dot_val += w21.y * r21.y;
        dot_val += w21.z * r21.z;
        dot_val += w21.w * r21.w;
        dot_val += w22.x * r22.x;
        dot_val += w22.y * r22.y;
        dot_val += w22.z * r22.z;
        dot_val += w22.w * r22.w;
        dot_val += w23.x * r23.x;
        dot_val += w23.y * r23.y;
        dot_val += w23.z * r23.z;
        dot_val += w23.w * r23.w;
    }
    for (; i < hidden; ++i) {
        float r_i = r_row[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}



// NOT BIT-EXACT. Identical layout to spec50_router_gemv_exact, but with float4 loads
// and a hand-unrolled body. Kept only so the benchmark can price what exactness costs;
// it must never be wired into the runtime. `hidden` must be a multiple of 4.
kernel void spec50_router_gemv_vec4(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    for (uint i = 0; i < hidden; i += 4u) {
        float4 x4 = *reinterpret_cast<device const float4*>(in_tok + i);
        float4 s4 = *reinterpret_cast<device const float4*>(gate_inp_scale + i);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i);
        dot_val += w4.x * (x4.x * factor * s4.x);
        dot_val += w4.y * (x4.y * factor * s4.y);
        dot_val += w4.z * (x4.z * factor * s4.z);
        dot_val += w4.w * (x4.w * factor * s4.w);
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}
"#;

#[cfg(target_os = "macos")]
pub(crate) struct Spec50RouterKernels {
    pub(crate) rms_factor_pipeline: ComputePipelineState,
    pub(crate) router_gemv_pipeline: ComputePipelineState,
    pub(crate) router_gemv_u4_pipeline: ComputePipelineState,
    pub(crate) router_gemv_u8_pipeline: ComputePipelineState,
    /// NOT bit-exact; benchmark-only.
    pub(crate) router_gemv_vec4_pipeline: ComputePipelineState,
    /// Same three sources again, from a fast-math-OFF library. Without fast math the
    /// compiler may not reassociate an FP reduction, so a hand-unrolled loop keeps the
    /// shipping kernel's summation order by construction -- IF the per-operation
    /// semantics (notably whether `dot += w * r` still contracts to an FMA) survive
    /// turning fast math off. `strict_exact` is the control that answers that.
    pub(crate) router_gemv_strict_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_u8_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_vec4_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_u16_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_v4x4_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_v4x8_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_v4x16_pipeline: ComputePipelineState,
    pub(crate) r_precompute_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_rv4x8_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_rv4x16_pipeline: ComputePipelineState,
    pub(crate) router_gemv_strict_rv4x24_pipeline: ComputePipelineState,
    pub(crate) prepare_pipeline: ComputePipelineState,
}

/// Which dot-loop shape to dispatch. Only `Exact` is proven bit-identical to
/// `gemma4_router_batch_k_f32`; the others exist so the tests can measure what each
/// step away from the shipping loop text costs in ULP and what it buys in ms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spec50RouterVariant {
    Exact,
    Unroll4,
    Unroll8,
    Vec4,
    StrictExact,
    StrictUnroll8,
    StrictVec4,
    StrictUnroll16,
    StrictVec4x4,
    StrictVec4x8,
    StrictVec4x16,
    /// Needs the `r_stage` scratch buffer.
    StrictRVec4x8,
    /// Needs the `r_stage` scratch buffer.
    StrictRVec4x16,
    /// Needs the `r_stage` scratch buffer.
    StrictRVec4x24,
}

/// What [`encode_spec50_router_gemv`] dispatches: the fastest variant that is
/// bit-identical to `gemma4_router_batch_k_f32` across K=1..=8.
pub(crate) const SPEC50_ROUTER_DEFAULT_VARIANT: Spec50RouterVariant =
    Spec50RouterVariant::StrictRVec4x16;

impl Spec50RouterVariant {
    /// True when the variant needs the `k_tokens x hidden` f32 `r_stage` scratch.
    pub(crate) fn needs_r_stage(self) -> bool {
        matches!(
            self,
            Self::StrictRVec4x8 | Self::StrictRVec4x16 | Self::StrictRVec4x24
        )
    }
}

#[cfg(target_os = "macos")]
static SPEC50_ROUTER_KERNELS: OnceLock<Option<Spec50RouterKernels>> = OnceLock::new();

/// Compiles (once) and returns the spec50 router pipelines, or `None` — with a stderr
/// line saying why — if the Metal compile fails. Unlike `metal_linear_kernel()`, a
/// failure here disables only this lane.
#[cfg(target_os = "macos")]
pub(crate) fn spec50_router_kernels() -> Option<&'static Spec50RouterKernels> {
    SPEC50_ROUTER_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SPEC50_ROUTER_SHADER, &options)
                .map_err(|err| eprintln!("[metal] SPEC50_ROUTER_SHADER compile failed: {err}"))
                .ok()?;
            let strict_options = CompileOptions::new();
            strict_options.set_fast_math_enabled(false);
            let strict_library = device
                .new_library_with_source(SPEC50_ROUTER_SHADER, &strict_options)
                .map_err(|err| {
                    eprintln!("[metal] SPEC50_ROUTER_SHADER (strict) compile failed: {err}")
                })
                .ok()?;
            let pipeline_from =
                |lib: &metal::Library, name: &str| -> Option<ComputePipelineState> {
                    let function = lib
                        .get_function(name, None)
                        .map_err(|err| eprintln!("[metal] spec50 function {name} missing: {err}"))
                        .ok()?;
                    device
                        .new_compute_pipeline_state_with_function(&function)
                        .map_err(|err| eprintln!("[metal] spec50 pipeline {name} failed: {err}"))
                        .ok()
                };
            let pipeline = |name: &str| pipeline_from(&library, name);
            let strict = |name: &str| pipeline_from(&strict_library, name);
            Some(Spec50RouterKernels {
                rms_factor_pipeline: pipeline("spec50_router_rms_factor")?,
                router_gemv_pipeline: pipeline("spec50_router_gemv_exact")?,
                router_gemv_u4_pipeline: pipeline("spec50_router_gemv_u4")?,
                router_gemv_u8_pipeline: pipeline("spec50_router_gemv_u8")?,
                router_gemv_vec4_pipeline: pipeline("spec50_router_gemv_vec4")?,
                router_gemv_strict_pipeline: strict("spec50_router_gemv_exact")?,
                router_gemv_strict_u8_pipeline: strict("spec50_router_gemv_u8")?,
                router_gemv_strict_vec4_pipeline: strict("spec50_router_gemv_vec4")?,
                router_gemv_strict_u16_pipeline: strict("spec50_router_gemv_u16")?,
                router_gemv_strict_v4x4_pipeline: strict("spec50_router_gemv_v4x4")?,
                router_gemv_strict_v4x8_pipeline: strict("spec50_router_gemv_v4x8")?,
                router_gemv_strict_v4x16_pipeline: strict("spec50_router_gemv_v4x16")?,
                r_precompute_pipeline: strict("spec50_router_r_precompute")?,
                router_gemv_strict_rv4x8_pipeline: strict("spec50_router_gemv_rv4x8")?,
                router_gemv_strict_rv4x16_pipeline: strict("spec50_router_gemv_rv4x16")?,
                router_gemv_strict_rv4x24_pipeline: strict("spec50_router_gemv_rv4x24")?,
                prepare_pipeline: pipeline("spec50_router_prepare")?,
            })
        })
        .as_ref()
}

/// Batched router GEMV, bit-identical twin of `encode_gemma4_router_batch_k`.
///
/// Encodes TWO dispatches into `encoder` (a compute encoder is serial, so the
/// ordering is guaranteed without an explicit barrier):
///   1. `spec50_router_prepare`     -> `factors[0..k_tokens]` and `r_stage`
///   2. `spec50_router_gemv_rv4x16` -> `out_logits[k_tokens][num_experts]`
///
/// Buffers the integrator supplies:
///   `input`            `[k_tokens][hidden]` f32
///   `gate_inp_scale`   `[hidden]` f32
///   `gate_inp_weights` `[num_experts][hidden]` f32
///   `out_logits`       `[k_tokens][num_experts]` f32, at `out_logits_offset`
///   `factors`          NEW scratch, `spec50_router_factor_bytes(k_tokens)` bytes
///   `r_stage`          NEW scratch, `spec50_router_r_stage_bytes(k_tokens, hidden)`
///                      bytes (90 KB at K=8, hidden 2816)
///
/// Both scratch buffers are written every call and carry nothing between calls, so one
/// pair sized for the largest K can be shared by every layer.
///
/// Returns `None` (encode nothing) unless `num_experts % 4 == 0` and `hidden % 4 == 0`
/// — the caller should fall back to the shipping encode then.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_router_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50RouterKernels,
    input: &Buffer,
    gate_inp_scale: &Buffer,
    gate_inp_weights: &Buffer,
    out_logits: &Buffer,
    factors: &Buffer,
    r_stage: &Buffer,
    hidden: usize,
    eps: f32,
    num_experts: usize,
    k_tokens: usize,
    out_logits_offset: u64,
) -> Option<()> {
    encode_spec50_router_gemv_variant(
        encoder,
        kernels,
        SPEC50_ROUTER_DEFAULT_VARIANT,
        input,
        Some(r_stage),
        gate_inp_scale,
        gate_inp_weights,
        out_logits,
        factors,
        hidden,
        eps,
        num_experts,
        k_tokens,
        out_logits_offset,
    )
}

/// Variant-selecting sibling of [`encode_spec50_router_gemv`], for tests and
/// benchmarks. Same buffers, same grid; only the dot-loop pipeline changes.
///
/// ONLY [`Spec50RouterVariant::Exact`] may reach the runtime. The others are here
/// to be measured, and the test suite pins what each one costs in ULP.
/// `Vec4` additionally requires `hidden % 4 == 0`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_router_gemv_variant(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50RouterKernels,
    variant: Spec50RouterVariant,
    input: &Buffer,
    r_stage: Option<&Buffer>,
    gate_inp_scale: &Buffer,
    gate_inp_weights: &Buffer,
    out_logits: &Buffer,
    factors: &Buffer,
    hidden: usize,
    eps: f32,
    num_experts: usize,
    k_tokens: usize,
    out_logits_offset: u64,
) -> Option<()> {
    if k_tokens == 0
        || num_experts == 0
        || hidden == 0
        || !num_experts.is_multiple_of(SPEC50_ROUTER_EXPERTS_PER_TG)
        || (variant.needs_r_stage() && r_stage.is_none())
        || (matches!(
            variant,
            Spec50RouterVariant::Vec4
                | Spec50RouterVariant::StrictVec4
                | Spec50RouterVariant::StrictVec4x4
                | Spec50RouterVariant::StrictVec4x8
                | Spec50RouterVariant::StrictVec4x16
                | Spec50RouterVariant::StrictRVec4x8
                | Spec50RouterVariant::StrictRVec4x16
                | Spec50RouterVariant::StrictRVec4x24
        ) && !hidden.is_multiple_of(4))
    {
        return None;
    }
    let hidden_u32 = hidden as u32;
    let num_experts_u32 = num_experts as u32;
    let k_tokens_u32 = k_tokens as u32;

    // Dispatch 1: the batch-wide prepare pass. r-staged variants take the fused
    // kernel (factors AND r in one dispatch); the others only need the factors.
    // A compute encoder is serial, so the GEMV observes this without a barrier.
    if variant.needs_r_stage() {
        let r_buf = r_stage?;
        encoder.set_compute_pipeline_state(&kernels.prepare_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(gate_inp_scale), 0);
        encoder.set_buffer(2, Some(r_buf), 0);
        encoder.set_buffer(3, Some(factors), 0);
        encoder.set_bytes(4, 4, &hidden_u32 as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        encoder.set_bytes(6, 4, &k_tokens_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: k_tokens as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: SPEC50_ROUTER_RMS_TG_THREADS,
                height: 1,
                depth: 1,
            },
        );
    } else {
        encoder.set_compute_pipeline_state(&kernels.rms_factor_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(factors), 0);
        encoder.set_bytes(2, 4, &hidden_u32 as *const u32 as *const _);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
        encoder.set_bytes(4, 4, &k_tokens_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: k_tokens as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: SPEC50_ROUTER_RMS_TG_THREADS,
                height: 1,
                depth: 1,
            },
        );
    }

    // Dispatch 2: the GEMV itself.
    let pipeline = match variant {
        Spec50RouterVariant::Exact => &kernels.router_gemv_pipeline,
        Spec50RouterVariant::Unroll4 => &kernels.router_gemv_u4_pipeline,
        Spec50RouterVariant::Unroll8 => &kernels.router_gemv_u8_pipeline,
        Spec50RouterVariant::Vec4 => &kernels.router_gemv_vec4_pipeline,
        Spec50RouterVariant::StrictExact => &kernels.router_gemv_strict_pipeline,
        Spec50RouterVariant::StrictUnroll8 => &kernels.router_gemv_strict_u8_pipeline,
        Spec50RouterVariant::StrictVec4 => &kernels.router_gemv_strict_vec4_pipeline,
        Spec50RouterVariant::StrictUnroll16 => &kernels.router_gemv_strict_u16_pipeline,
        Spec50RouterVariant::StrictVec4x4 => &kernels.router_gemv_strict_v4x4_pipeline,
        Spec50RouterVariant::StrictVec4x8 => &kernels.router_gemv_strict_v4x8_pipeline,
        Spec50RouterVariant::StrictVec4x16 => &kernels.router_gemv_strict_v4x16_pipeline,
        Spec50RouterVariant::StrictRVec4x8 => &kernels.router_gemv_strict_rv4x8_pipeline,
        Spec50RouterVariant::StrictRVec4x16 => &kernels.router_gemv_strict_rv4x16_pipeline,
        Spec50RouterVariant::StrictRVec4x24 => &kernels.router_gemv_strict_rv4x24_pipeline,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(gate_inp_scale), 0);
    encoder.set_buffer(2, Some(gate_inp_weights), 0);
    encoder.set_buffer(3, Some(out_logits), out_logits_offset);
    encoder.set_buffer(4, Some(factors), 0);
    encoder.set_bytes(5, 4, &hidden_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &num_experts_u32 as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_tokens_u32 as *const u32 as *const _);
    if let Some(r_buf) = r_stage {
        encoder.set_buffer(8, Some(r_buf), 0);
    }
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (num_experts / SPEC50_ROUTER_EXPERTS_PER_TG) as u64,
            height: k_tokens.div_ceil(SPEC50_ROUTER_CHUNK_TOKENS) as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: SPEC50_ROUTER_TG_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Some(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    const HIDDEN: usize = 2816;
    const EXPERTS: usize = 128;
    const EPS: f32 = 1e-6;
    const LAYERS: usize = 30;

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u32() as f64 / u32::MAX as f64) as f32
        }
        fn sym(&mut self) -> f32 {
            self.unit() * 2.0 - 1.0
        }
    }

    fn shared(device: &Device, bytes: usize) -> Buffer {
        device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn buf_f32(device: &Device, values: &[f32]) -> Buffer {
        let b = shared(device, values.len() * 4);
        write_buffer_f32(&b, values);
        b
    }

    fn read_f32(buffer: &Buffer, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        read_buffer_f32(buffer, &mut out);
        out
    }

    /// Runs one command buffer and returns its hardware GPU-busy window in ms.
    fn run<F: FnOnce(&metal::ComputeCommandEncoderRef)>(queue: &CommandQueue, f: F) -> f64 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        f(e);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let (us, _) = command_buffer_gpu_times_us(cb);
        us as f64 / 1000.0
    }

    struct RouterFixture {
        device: Device,
        queue: CommandQueue,
        input: Vec<f32>,
        scale: Vec<f32>,
        weights: Vec<f32>,
        input_buf: Buffer,
        scale_buf: Buffer,
        weights_buf: Buffer,
        factors: Buffer,
        r_stage: Buffer,
        old_out: Buffer,
        new_out: Buffer,
    }

    fn router_fixture(seed: u64, k_max: usize) -> RouterFixture {
        let device = Device::system_default().expect("metal device");
        let queue = device.new_command_queue();
        let mut rng = Lcg(seed);
        let input: Vec<f32> = (0..k_max * HIDDEN).map(|_| rng.sym() * 3.0).collect();
        let scale: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + rng.sym() * 0.25).collect();
        let weights: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.sym() * 0.05).collect();
        let input_buf = buf_f32(&device, &input);
        let scale_buf = buf_f32(&device, &scale);
        let weights_buf = buf_f32(&device, &weights);
        let factors = shared(&device, spec50_router_factor_bytes(k_max) as usize);
        let r_stage = shared(&device, spec50_router_r_stage_bytes(k_max, HIDDEN) as usize);
        let old_out = shared(&device, k_max * EXPERTS * 4);
        let new_out = shared(&device, k_max * EXPERTS * 4);
        RouterFixture {
            device,
            queue,
            input,
            scale,
            weights,
            input_buf,
            scale_buf,
            weights_buf,
            factors,
            r_stage,
            old_out,
            new_out,
        }
    }

    /// f64 oracle: the exact logits plus the L1 norm of the summed terms, so the f32
    /// error can be judged against the dot product's own backward-error scale.
    fn cpu_router_ref_f64(fx: &RouterFixture, k: usize) -> (Vec<f64>, Vec<f64>) {
        let mut out = vec![0f64; k * EXPERTS];
        let mut l1 = vec![0f64; k * EXPERTS];
        for t in 0..k {
            let x = &fx.input[t * HIDDEN..(t + 1) * HIDDEN];
            let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rms_inv = 1.0 / (ss / HIDDEN as f64 + EPS as f64).sqrt();
            let factor = rms_inv * (1.0 / (HIDDEN as f64).sqrt());
            for e in 0..EXPERTS {
                let w = &fx.weights[e * HIDDEN..(e + 1) * HIDDEN];
                let mut acc = 0f64;
                let mut abs = 0f64;
                for (i, &wi) in w.iter().enumerate() {
                    let term = wi as f64 * (x[i] as f64 * factor * fx.scale[i] as f64);
                    acc += term;
                    abs += term.abs();
                }
                out[t * EXPERTS + e] = acc;
                l1[t * EXPERTS + e] = abs;
            }
        }
        (out, l1)
    }

    const VARIANTS: [(Spec50RouterVariant, &str); 14] = [
        (Spec50RouterVariant::Exact, "exact"),
        (Spec50RouterVariant::Unroll4, "unroll4"),
        (Spec50RouterVariant::Unroll8, "unroll8"),
        (Spec50RouterVariant::Vec4, "vec4"),
        (Spec50RouterVariant::StrictExact, "strict-exact"),
        (Spec50RouterVariant::StrictUnroll8, "strict-unroll8"),
        (Spec50RouterVariant::StrictVec4, "strict-vec4"),
        (Spec50RouterVariant::StrictUnroll16, "strict-unroll16"),
        (Spec50RouterVariant::StrictVec4x4, "strict-vec4x4"),
        (Spec50RouterVariant::StrictVec4x8, "strict-vec4x8"),
        (Spec50RouterVariant::StrictVec4x16, "strict-vec4x16"),
        (Spec50RouterVariant::StrictRVec4x8, "strict-r-vec4x8"),
        (Spec50RouterVariant::StrictRVec4x16, "strict-r-vec4x16"),
        (Spec50RouterVariant::StrictRVec4x24, "strict-r-vec4x24"),
    ];

    #[allow(clippy::too_many_arguments)]
    fn encode_variant(
        e: &metal::ComputeCommandEncoderRef,
        kernels: &Spec50RouterKernels,
        variant: Spec50RouterVariant,
        input: &Buffer,
        fx: &RouterFixture,
        weights: &Buffer,
        out: &Buffer,
        k: usize,
        offset: u64,
    ) {
        encode_spec50_router_gemv_variant(
            e,
            kernels,
            variant,
            input,
            Some(&fx.r_stage),
            &fx.scale_buf,
            weights,
            out,
            &fx.factors,
            HIDDEN,
            EPS,
            EXPERTS,
            k,
            offset,
        )
        .expect("spec50 router encode rejected a supported shape");
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_new(
        e: &metal::ComputeCommandEncoderRef,
        kernels: &Spec50RouterKernels,
        input: &Buffer,
        fx: &RouterFixture,
        weights: &Buffer,
        out: &Buffer,
        k: usize,
        offset: u64,
    ) {
        encode_spec50_router_gemv(
            e,
            kernels,
            input,
            &fx.scale_buf,
            weights,
            out,
            &fx.factors,
            &fx.r_stage,
            HIDDEN,
            EPS,
            EXPERTS,
            k,
            offset,
        )
        .expect("spec50 router encode rejected a supported shape");
    }

    /// The contract: bit-identical to `gemma4_router_batch_k_f32` for every K in
    /// 1..=8, and every token's logits identical whether it is evaluated alone
    /// (K=1) or inside a batch (per-token batch independence).
    #[test]
    fn spec50_router_gemv_bitwise_vs_old_and_batch_independent() {
        if !detect_metal_device().available {
            eprintln!("[spec50 router] no Metal device, skipping");
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels failed to compile");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0001, 8);

        // Each token evaluated completely alone, as its own K=1 dispatch.
        let mut alone = vec![0f32; 8 * EXPERTS];
        for t in 0..8 {
            let tok_buf = buf_f32(&fx.device, &fx.input[t * HIDDEN..(t + 1) * HIDDEN]);
            run(&fx.queue, |e| {
                encode_new(
                    e,
                    kernels,
                    &tok_buf,
                    &fx,
                    &fx.weights_buf,
                    &fx.new_out,
                    1,
                    0,
                );
            });
            alone[t * EXPERTS..(t + 1) * EXPERTS].copy_from_slice(&read_f32(&fx.new_out, EXPERTS));
        }

        let mut mismatches_total = 0usize;
        let mut max_abs_diff = 0f32;
        let mut max_ulp: u32 = 0;
        for k in 1..=8usize {
            run(&fx.queue, |e| {
                encode_gemma4_router_batch_k(
                    e,
                    old,
                    &fx.input_buf,
                    &fx.scale_buf,
                    &fx.weights_buf,
                    &fx.old_out,
                    HIDDEN,
                    EPS,
                    EXPERTS,
                    k,
                    0,
                )
                .unwrap();
                encode_new(
                    e,
                    kernels,
                    &fx.input_buf,
                    &fx,
                    &fx.weights_buf,
                    &fx.new_out,
                    k,
                    0,
                );
            });
            let got_old = read_f32(&fx.old_out, k * EXPERTS);
            let got_new = read_f32(&fx.new_out, k * EXPERTS);
            for i in 0..k * EXPERTS {
                let (a, b) = (got_old[i], got_new[i]);
                if a.to_bits() != b.to_bits() {
                    mismatches_total += 1;
                    max_abs_diff = max_abs_diff.max((a - b).abs());
                    max_ulp = max_ulp.max(a.to_bits().abs_diff(b.to_bits()));
                }
                assert_eq!(
                    b.to_bits(),
                    alone[i].to_bits(),
                    "K={k} token {} expert {}: batched result differs from its K=1 result",
                    i / EXPERTS,
                    i % EXPERTS
                );
            }
        }
        eprintln!(
            "[spec50 router] K=1..8 vs gemma4_router_batch_k_f32: {mismatches_total} mismatching \
             logits (max abs diff {max_abs_diff:.3e}, max ULP delta {max_ulp})"
        );
        assert_eq!(
            mismatches_total, 0,
            "new router logits are NOT bit-identical to gemma4_router_batch_k_f32 \
             (max ULP delta {max_ulp}, max abs diff {max_abs_diff:.3e})"
        );

        // Sanity floor: the f32 chain must still track the f64 oracle.
        let (reference, l1) = cpu_router_ref_f64(&fx, 8);
        run(&fx.queue, |e| {
            encode_new(
                e,
                kernels,
                &fx.input_buf,
                &fx,
                &fx.weights_buf,
                &fx.new_out,
                8,
                0,
            );
        });
        let got = read_f32(&fx.new_out, 8 * EXPERTS);
        let mut worst_rel_l1 = 0f64;
        for i in 0..8 * EXPERTS {
            worst_rel_l1 = worst_rel_l1.max((got[i] as f64 - reference[i]).abs() / l1[i]);
        }
        eprintln!("[spec50 router] vs f64 oracle: max err / term-L1 = {worst_rel_l1:.3e}");
        assert!(
            worst_rel_l1 <= 1e-5,
            "router f32 error {worst_rel_l1:.3e} exceeds 1e-5 of the term L1 norm"
        );
    }

    /// Documents WHY the shipping kernel keeps the scalar dot loop verbatim: every
    /// variant that touches the loop's text — even ones that only hoist loads and
    /// leave the accumulator order alone — is a chance for fast math to reassociate
    /// the reduction. This prints the ULP cost of each so the trade is on the record.
    #[test]
    fn spec50_router_gemv_variant_exactness_sweep() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0001, 8);
        eprintln!("| variant | mismatching logits (K=1..8) | max ULP delta | max abs diff |");
        eprintln!("|---------|-----------------------------|---------------|--------------|");
        let mut exact_mismatches = usize::MAX;
        for (variant, name) in VARIANTS {
            let mut mismatches = 0usize;
            let mut max_ulp: u32 = 0;
            let mut max_abs = 0f32;
            let mut total = 0usize;
            for k in 1..=8usize {
                run(&fx.queue, |e| {
                    encode_gemma4_router_batch_k(
                        e,
                        old,
                        &fx.input_buf,
                        &fx.scale_buf,
                        &fx.weights_buf,
                        &fx.old_out,
                        HIDDEN,
                        EPS,
                        EXPERTS,
                        k,
                        0,
                    )
                    .unwrap();
                    encode_variant(
                        e,
                        kernels,
                        variant,
                        &fx.input_buf,
                        &fx,
                        &fx.weights_buf,
                        &fx.new_out,
                        k,
                        0,
                    );
                });
                let a = read_f32(&fx.old_out, k * EXPERTS);
                let b = read_f32(&fx.new_out, k * EXPERTS);
                for i in 0..k * EXPERTS {
                    total += 1;
                    if a[i].to_bits() != b[i].to_bits() {
                        mismatches += 1;
                        max_ulp = max_ulp.max(a[i].to_bits().abs_diff(b[i].to_bits()));
                        max_abs = max_abs.max((a[i] - b[i]).abs());
                    }
                }
            }
            eprintln!("| {name} | {mismatches}/{total} | {max_ulp} | {max_abs:.3e} |");
            if variant == SPEC50_ROUTER_DEFAULT_VARIANT {
                exact_mismatches = mismatches;
            }
        }
        assert_eq!(
            exact_mismatches, 0,
            "the default variant ({SPEC50_ROUTER_DEFAULT_VARIANT:?}) must be \
             bit-identical to gemma4_router_batch_k_f32"
        );
    }

    /// Ragged K (not a multiple of the 8-token chunk) and a non-zero output offset.
    #[test]
    fn spec50_router_gemv_ragged_k_and_offset() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        for k in [3usize, 5, 7, 8] {
            let fx = router_fixture(0x5eed_0002 + k as u64, k);
            let old_out = shared(&fx.device, (k + 4) * EXPERTS * 4);
            let new_out = shared(&fx.device, (k + 4) * EXPERTS * 4);
            let offset = (4 * EXPERTS * 4) as u64;
            run(&fx.queue, |e| {
                encode_gemma4_router_batch_k(
                    e,
                    old,
                    &fx.input_buf,
                    &fx.scale_buf,
                    &fx.weights_buf,
                    &old_out,
                    HIDDEN,
                    EPS,
                    EXPERTS,
                    k,
                    offset,
                )
                .unwrap();
                encode_new(
                    e,
                    kernels,
                    &fx.input_buf,
                    &fx,
                    &fx.weights_buf,
                    &new_out,
                    k,
                    offset,
                );
            });
            let a = read_f32(&old_out, (k + 4) * EXPERTS);
            let b = read_f32(&new_out, (k + 4) * EXPERTS);
            for i in 4 * EXPERTS..(k + 4) * EXPERTS {
                assert_eq!(a[i].to_bits(), b[i].to_bits(), "K={k}: mismatch at {i}");
            }
            // Nothing may be written before the offset.
            for i in 0..4 * EXPERTS {
                assert_eq!(b[i].to_bits(), 0, "K={k}: wrote below out_logits_offset");
            }
        }
    }

    /// 30 layers' worth of router GEMV per command buffer with a distinct 1.44 MB
    /// matrix per layer, old vs new, at K = 1 / 4 / 8. `--nocapture` to see it.
    #[test]
    fn spec50_router_gemv_bench_30_layers() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0003, 8);
        let mut rng = Lcg(77);
        // 30 x 1.44 MB = 43 MB of router matrices, so no layer is served from L2.
        let layer_weights: Vec<Buffer> = (0..LAYERS)
            .map(|_| {
                let w: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.sym() * 0.05).collect();
                buf_f32(&fx.device, &w)
            })
            .collect();
        let matrix_bytes = (EXPERTS * HIDDEN * 4 * LAYERS) as f64;
        eprintln!(
            "[spec50 router bench] {LAYERS} layers/cb, hidden {HIDDEN}, experts {EXPERTS}, \
             {:.1} MB of router matrix per measurement",
            matrix_bytes / 1e6
        );
        eprintln!("| K | old ms | variant | new ms | old GB/s | new GB/s | speedup | bit-exact |");
        eprintln!("|---|--------|---------|--------|----------|----------|---------|-----------|");
        for &k in &[1usize, 4, 8] {
            let mut best_old = f64::MAX;
            let mut best = [f64::MAX; VARIANTS.len()];
            for _round in 0..5 {
                let t_old = run(&fx.queue, |e| {
                    for w in &layer_weights {
                        encode_gemma4_router_batch_k(
                            e,
                            old,
                            &fx.input_buf,
                            &fx.scale_buf,
                            w,
                            &fx.old_out,
                            HIDDEN,
                            EPS,
                            EXPERTS,
                            k,
                            0,
                        )
                        .unwrap();
                    }
                });
                best_old = best_old.min(t_old);
                for (idx, (variant, _)) in VARIANTS.iter().enumerate() {
                    let t = run(&fx.queue, |e| {
                        for w in &layer_weights {
                            encode_variant(
                                e,
                                kernels,
                                *variant,
                                &fx.input_buf,
                                &fx,
                                w,
                                &fx.new_out,
                                k,
                                0,
                            );
                        }
                    });
                    best[idx] = best[idx].min(t);
                }
            }
            // GB/s counts the router matrix only: 1.44 MB/layer, the byte floor this
            // lane is measured against.
            let gbs = |ms: f64| matrix_bytes / (ms * 1e-3) / 1e9;
            for (idx, (variant, name)) in VARIANTS.iter().enumerate() {
                let exact = if *variant == SPEC50_ROUTER_DEFAULT_VARIANT {
                    "default"
                } else {
                    "see sweep"
                };
                eprintln!(
                    "| {k} | {best_old:.3} | {name} | {:.3} | {:.1} | {:.1} | {:.2}x | {exact} |",
                    best[idx],
                    gbs(best_old),
                    gbs(best[idx]),
                    best_old / best[idx]
                );
            }
        }
    }
}
