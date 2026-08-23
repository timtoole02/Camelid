//! Dense Q4_0 x f32 K=8 batch GEMM replacements for the chained decode lane.
//!
//! # Why
//!
//! The shipped `q4_0_*_block_linear_batch_k8` kernels give every 32-thread
//! simdgroup four output rows, and every simdgroup streams the whole activation
//! batch `y` (K x hidden f32). The activation traffic therefore scales with
//! `rows / 4` while the weight traffic is fixed, and on the 26B shapes the
//! activation re-reads are 7-14x the weight bytes. Measured on this machine the
//! shared-MLP stage already runs at ~84% of the M4 bandwidth wall counting
//! those re-reads, so the only win available is *redundant-traffic reduction*:
//! amortize each staged copy of `y` over more output rows per threadgroup.
//!
//! # What changed
//!
//! Nothing arithmetic. These kernels reproduce the K=8 reference kernels'
//! floating-point program instruction for instruction: the same
//! lane -> (block residue, byte quad) mapping (`ix = lane/4`,
//! `ilb = (lane%4)*4`), the same strictly ascending per-lane `ib` accumulation
//! chain, the same `dot(lo,ylo) + dot(hi,yhi)` association, the same scale
//! placement (folded into the weight vector), the same `simd_sum` tree, and the
//! verbatim GeGLU expression. Three things change, all outside the FP program:
//!
//! * `NR` (rows per simdgroup) is a compile-time template instead of a hard 4.
//!   A row's accumulation chain does not depend on which simdgroup owns it.
//! * `SG` simdgroups per threadgroup, each owning its own `NR` rows. Per-row
//!   arithmetic is per-simdgroup and untouched. The prior attempt broke bitwise
//!   equality at SG>1 because the pipeline's `maxTotalThreadsPerThreadgroup`
//!   could fall below the dispatched width; here every entry point carries
//!   `[[max_total_threads_per_threadgroup(32*SG)]]` and the builder verifies
//!   the compiled pipeline honors the width, refusing the config otherwise.
//! * In the `tiled` variants, `y` is staged once per threadgroup into
//!   threadgroup memory in tiles of `TB` Q4_0-block columns (K x hidden f32 is
//!   90 KB at the 26B hidden, past the 32 KB threadgroup limit, hence tiling)
//!   and all `SG` simdgroups consume the staged copy. `TB % 8 == 0` keeps each
//!   lane's `ib` visit order identical to the reference (`ib ≡ ix (mod 8)`,
//!   ascending), so the accumulation chain is bit-identical; the staged floats
//!   are bit-copies. The `flat` variants skip staging (at SG=1 a simdgroup
//!   already reads each `y` element exactly once, so staging is pure overhead
//!   there — flat + NR is the SG=1 shape of the same traffic reduction).
//!
//! * Weights' packed nibbles are read through `packed_uchar4` (alignment 1).
//!   The reference's `uchar4` reinterpret at `wb + 2` is under-aligned for
//!   every even block index with the 18-byte block stride (the same UB pattern
//!   fixed elsewhere in this file's parent); the loaded bytes are identical.
//!
//! # Scope
//!
//! K=8 only. The generic (runtime `k_batch`) reference kernel's loop structure
//! blocks FMA contraction that a static-K template invites, so K != 8
//! templates cannot match it bitwise; production speculative width is K=8 and
//! every other K keeps today's kernels. The `encode_*_v4` entry points return
//! `false` for `k_batch != 8` (and when the library failed to build) so the
//! caller falls back to the existing encode functions unchanged.
//!
//! # Measured outcome (M4, 26B shapes)
//!
//! Bit-exactness is empirically fragile to any body restructuring: the
//! compiled FMA contraction of the accumulation chain changes with the row
//! count (plain diverges at NR=8, gateup at NR=16, a per-row-routed QKV
//! rewrite at NR=4 — up to hundreds of ULP on loop-carried shapes), while
//! NR=4 with any SG and any tile size is bit-exact for all three kernels.
//! So scaling comes from SG at NR=4.
//!
//! Perf: only the fused QKV wins — 1.44x (12.3 -> 8.6 ms / 30 dispatches) on
//! the local-layer 8192x88-block shape at (NR=4, SG=4, TB=16), so that is
//! what `encode_gemma4_q4_0_qkv_matmul_batch_k_v4` ships. Every plain and
//! gateup 26B shape is 0.6-1.0x under every bit-exact config: those stages
//! are execution-bound (their activation re-reads are served by the cache
//! hierarchy, and K=8 costs ~3x K=1 at identical weight traffic), and the
//! parity contract pins the per-row instruction stream, so their `_v4`
//! encodes return `false` and the shipped kernels keep those dispatches.
//!
//! The tests at the bottom assert raw `f32::to_bits` equality against the
//! kernels being replaced for every variant in the sweep, plus per-token batch
//! independence at K=8.

#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Shader: template bodies. Entry points are generated per config by
// `spec50_shader_src`.
// ---------------------------------------------------------------------------

const SPEC50_DENSE_PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// plain: output[k * rows + r] = sum_j W[r][j] * y[k][j]
// Body mirrors q4_0_block_linear_batch_k8 with the row count 4 replaced by the
// template parameter NR. `gsg` is the global simdgroup index (tg * SG + sg).
// ---------------------------------------------------------------------------
template <uint NR>
static inline void spec50_plain_flat_body(
    device const float* y,
    device const char* weight_blocks,
    device float* output,
    uint blocks_per_row,
    uint rows,
    uint gsg,
    uint lane
) {
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = gsg * NR;
    if (r0 >= rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;

    bool has_row[NR];
    #pragma unroll
    for (uint i = 0; i < NR; ++i) {
        has_row[i] = (r0 + i < rows);
    }

    float sums[NR][KB] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 scaled_lo[NR], scaled_hi[NR];

        #pragma unroll
        for (uint i = 0; i < NR; ++i) {
            if (has_row[i]) {
                const uint r = r0 + i;
                device const char* wb = weight_blocks + r * row_stride + ib * q4_block_bytes;
                const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                const uchar4 wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb + 2 + ilb));
                const float4 lo = float4(int4(wq & 0x0F) - 8);
                const float4 hi = float4(int4(wq >> 4) - 8);
                scaled_lo[i] = lo * w_scale;
                scaled_hi[i] = hi * w_scale;
            }
        }

        #pragma unroll
        for (uint k = 0; k < KB; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            #pragma unroll
            for (uint i = 0; i < NR; ++i) {
                if (has_row[i]) {
                    sums[i][k] += dot(scaled_lo[i], ylo) + dot(scaled_hi[i], yhi);
                }
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < NR; ++i) {
            if (has_row[i]) {
                const float tot = simd_sum(sums[i][k]);
                if (lane == 0) {
                    output[k * rows + r0 + i] = tot;
                }
            }
        }
    }
}

// Tiled twin: y staged tile-by-tile in threadgroup memory, shared by all SG
// simdgroups. No early return (barriers must stay uniform); has_row guards
// everything row-shaped. tile4 holds KB * TB * 8 float4s.
template <uint NR, uint SG, uint TB>
static inline void spec50_plain_tiled_body(
    device const float* y,
    device const char* weight_blocks,
    device float* output,
    uint blocks_per_row,
    uint rows,
    uint tg,
    uint sg,
    uint lane,
    threadgroup float4* tile4
) {
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = (tg * SG + sg) * NR;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint iq = lane % 4;
    const uint tid = sg * 32 + lane;

    device const float4* y4 = reinterpret_cast<device const float4*>(y);

    bool has_row[NR];
    #pragma unroll
    for (uint i = 0; i < NR; ++i) {
        has_row[i] = (r0 + i < rows);
    }

    float sums[NR][KB] = {{0.0f}};

    for (uint ts = 0; ts < blocks_per_row; ts += TB) {
        const uint tb = min(TB, blocks_per_row - ts);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < KB * TB * 8; idx += 32 * SG) {
            const uint k = idx / (TB * 8);
            const uint f = idx % (TB * 8);
            if (f < tb * 8) {
                tile4[idx] = y4[k * (hidden >> 2) + (ts << 3) + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ib = ts + ix; ib < ts + tb; ib += NQ) {
            float4 scaled_lo[NR], scaled_hi[NR];

            #pragma unroll
            for (uint i = 0; i < NR; ++i) {
                if (has_row[i]) {
                    const uint r = r0 + i;
                    device const char* wb = weight_blocks + r * row_stride + ib * q4_block_bytes;
                    const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                    const uchar4 wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb + 2 + ilb));
                    const float4 lo = float4(int4(wq & 0x0F) - 8);
                    const float4 hi = float4(int4(wq >> 4) - 8);
                    scaled_lo[i] = lo * w_scale;
                    scaled_hi[i] = hi * w_scale;
                }
            }

            const uint lb8 = (ib - ts) * 8;
            #pragma unroll
            for (uint k = 0; k < KB; ++k) {
                const float4 ylo = tile4[k * (TB * 8) + lb8 + iq];
                const float4 yhi = tile4[k * (TB * 8) + lb8 + 4 + iq];

                #pragma unroll
                for (uint i = 0; i < NR; ++i) {
                    if (has_row[i]) {
                        sums[i][k] += dot(scaled_lo[i], ylo) + dot(scaled_hi[i], yhi);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < NR; ++i) {
            if (has_row[i]) {
                const float tot = simd_sum(sums[i][k]);
                if (lane == 0) {
                    output[k * rows + r0 + i] = tot;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fused QKV. The body is q4_0_qkv_block_linear_batch_k8's, verbatim: FOUR rows
// per simdgroup, the Q/K/V base resolved once per 4-row group, four scalar
// accumulator arrays, unconditional processing/stores. The compiled FMA
// contraction of the accumulation chain is sensitive to any body
// restructuring (a per-row-routed rewrite diverged by up to 128 ULP on
// loop-carried shapes), so the row count stays hard-coded at 4 and scaling
// comes from SG. Like the reference, this is only well defined when q_rows,
// k_rows and v_rows are each multiples of 4 (the 26B fused-QKV splits are
// 4096/2048/2048).
// ---------------------------------------------------------------------------
#define SPEC50_QKV_ROUTE4 \
    device const char* w_base; \
    uint target_r0; \
    uint target_kind; \
    if (r0 < q_rows) { \
        w_base = q_weight; \
        target_r0 = r0; \
        target_kind = 0; \
    } else if (r0 < q_rows + k_rows) { \
        w_base = k_weight; \
        target_r0 = r0 - q_rows; \
        target_kind = 1; \
    } else { \
        w_base = v_weight; \
        target_r0 = r0 - (q_rows + k_rows); \
        target_kind = 2; \
    }

#define SPEC50_QKV_LOAD4 \
    device const char* wb0 = w_base + (target_r0 + 0) * row_stride + ib * q4_block_bytes; \
    device const char* wb1 = w_base + (target_r0 + 1) * row_stride + ib * q4_block_bytes; \
    device const char* wb2 = w_base + (target_r0 + 2) * row_stride + ib * q4_block_bytes; \
    device const char* wb3 = w_base + (target_r0 + 3) * row_stride + ib * q4_block_bytes; \
    const float w_scale0 = float(*reinterpret_cast<device const half*>(wb0)); \
    const float w_scale1 = float(*reinterpret_cast<device const half*>(wb1)); \
    const float w_scale2 = float(*reinterpret_cast<device const half*>(wb2)); \
    const float w_scale3 = float(*reinterpret_cast<device const half*>(wb3)); \
    const uchar4 wq0 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb0 + 2 + ilb)); \
    const uchar4 wq1 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb1 + 2 + ilb)); \
    const uchar4 wq2 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb2 + 2 + ilb)); \
    const uchar4 wq3 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb3 + 2 + ilb)); \
    const float4 lo0 = float4(int4(wq0 & 0x0F) - 8) * w_scale0; \
    const float4 hi0 = float4(int4(wq0 >> 4) - 8) * w_scale0; \
    const float4 lo1 = float4(int4(wq1 & 0x0F) - 8) * w_scale1; \
    const float4 hi1 = float4(int4(wq1 >> 4) - 8) * w_scale1; \
    const float4 lo2 = float4(int4(wq2 & 0x0F) - 8) * w_scale2; \
    const float4 hi2 = float4(int4(wq2 >> 4) - 8) * w_scale2; \
    const float4 lo3 = float4(int4(wq3 & 0x0F) - 8) * w_scale3; \
    const float4 hi3 = float4(int4(wq3 >> 4) - 8) * w_scale3;

#define SPEC50_QKV_STORE4 \
    const float tot0 = simd_sum(sum0[k]); \
    const float tot1 = simd_sum(sum1[k]); \
    const float tot2 = simd_sum(sum2[k]); \
    const float tot3 = simd_sum(sum3[k]); \
    if (lane == 0) { \
        if (target_kind == 0) { \
            query_out[k * q_rows + target_r0 + 0] = tot0; \
            query_out[k * q_rows + target_r0 + 1] = tot1; \
            query_out[k * q_rows + target_r0 + 2] = tot2; \
            query_out[k * q_rows + target_r0 + 3] = tot3; \
        } else if (target_kind == 1) { \
            key_out[k * k_rows + target_r0 + 0] = tot0; \
            key_out[k * k_rows + target_r0 + 1] = tot1; \
            key_out[k * k_rows + target_r0 + 2] = tot2; \
            key_out[k * k_rows + target_r0 + 3] = tot3; \
        } else { \
            val_out[k * v_rows + target_r0 + 0] = tot0; \
            val_out[k * v_rows + target_r0 + 1] = tot1; \
            val_out[k * v_rows + target_r0 + 2] = tot2; \
            val_out[k * v_rows + target_r0 + 3] = tot3; \
        } \
    }

template <uint NR>
static inline void spec50_qkv_flat_body(
    device const float* y,
    device const char* q_weight,
    device const char* k_weight,
    device const char* v_weight,
    device float* query_out,
    device float* key_out,
    device float* val_out,
    uint blocks_per_row,
    uint q_rows,
    uint k_rows,
    uint v_rows,
    uint gsg,
    uint lane
) {
    static_assert(NR == 4, "the QKV body is the reference's verbatim 4-row program");
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint r0 = gsg * NR0;
    if (r0 >= total_rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;

    SPEC50_QKV_ROUTE4

    float sum0[KB] = {0.0f};
    float sum1[KB] = {0.0f};
    float sum2[KB] = {0.0f};
    float sum3[KB] = {0.0f};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        SPEC50_QKV_LOAD4

#pragma unroll
        for (uint k = 0; k < KB; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);
            sum0[k] += dot(lo0, ylo) + dot(hi0, yhi);
            sum1[k] += dot(lo1, ylo) + dot(hi1, yhi);
            sum2[k] += dot(lo2, ylo) + dot(hi2, yhi);
            sum3[k] += dot(lo3, ylo) + dot(hi3, yhi);
        }
    }

#pragma unroll
    for (uint k = 0; k < KB; ++k) {
        SPEC50_QKV_STORE4
    }
}

template <uint NR, uint SG, uint TB>
static inline void spec50_qkv_tiled_body(
    device const float* y,
    device const char* q_weight,
    device const char* k_weight,
    device const char* v_weight,
    device float* query_out,
    device float* key_out,
    device float* val_out,
    uint blocks_per_row,
    uint q_rows,
    uint k_rows,
    uint v_rows,
    uint tg,
    uint sg,
    uint lane,
    threadgroup float4* tile4
) {
    static_assert(NR == 4, "the QKV body is the reference's verbatim 4-row program");
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint r0 = (tg * SG + sg) * NR0;
    const bool live = r0 < total_rows;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint iq = lane % 4;
    const uint tid = sg * 32 + lane;

    device const float4* y4 = reinterpret_cast<device const float4*>(y);

    // Dead simdgroups still stage and hit the barriers; route against row 0 so
    // their (discarded) loads stay in bounds.
    const uint r0_safe = live ? r0 : 0u;
    device const char* w_base;
    uint target_r0;
    uint target_kind;
    if (r0_safe < q_rows) {
        w_base = q_weight;
        target_r0 = r0_safe;
        target_kind = 0;
    } else if (r0_safe < q_rows + k_rows) {
        w_base = k_weight;
        target_r0 = r0_safe - q_rows;
        target_kind = 1;
    } else {
        w_base = v_weight;
        target_r0 = r0_safe - (q_rows + k_rows);
        target_kind = 2;
    }

    float sum0[KB] = {0.0f};
    float sum1[KB] = {0.0f};
    float sum2[KB] = {0.0f};
    float sum3[KB] = {0.0f};

    for (uint ts = 0; ts < blocks_per_row; ts += TB) {
        const uint tb = min(TB, blocks_per_row - ts);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < KB * TB * 8; idx += 32 * SG) {
            const uint k = idx / (TB * 8);
            const uint f = idx % (TB * 8);
            if (f < tb * 8) {
                tile4[idx] = y4[k * (hidden >> 2) + (ts << 3) + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ib = ts + ix; ib < ts + tb; ib += NQ) {
            SPEC50_QKV_LOAD4

            const uint lb8 = (ib - ts) * 8;
#pragma unroll
            for (uint k = 0; k < KB; ++k) {
                const float4 ylo = tile4[k * (TB * 8) + lb8 + iq];
                const float4 yhi = tile4[k * (TB * 8) + lb8 + 4 + iq];
                sum0[k] += dot(lo0, ylo) + dot(hi0, yhi);
                sum1[k] += dot(lo1, ylo) + dot(hi1, yhi);
                sum2[k] += dot(lo2, ylo) + dot(hi2, yhi);
                sum3[k] += dot(lo3, ylo) + dot(hi3, yhi);
            }
        }
    }

    if (!live) return;
#pragma unroll
    for (uint k = 0; k < KB; ++k) {
        SPEC50_QKV_STORE4
    }
}

// ---------------------------------------------------------------------------
// fused shared gate/up + GeGLU. Activation expression and tanh clamp are the
// reference's, verbatim.
// ---------------------------------------------------------------------------
#define SPEC50_GATEUP_LOAD \
    _Pragma("unroll") \
    for (uint i = 0; i < NR; ++i) { \
        if (has_row[i]) { \
            const uint r = r0 + i; \
            device const char* g_wb = gate_weight + r * row_stride + ib * q4_block_bytes; \
            const float g_w_scale = float(*reinterpret_cast<device const half*>(g_wb)); \
            const uchar4 g_wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(g_wb + 2 + ilb)); \
            const float4 g_lo = float4(int4(g_wq & 0x0F) - 8); \
            const float4 g_hi = float4(int4(g_wq >> 4) - 8); \
            g_scaled_lo[i] = g_lo * g_w_scale; \
            g_scaled_hi[i] = g_hi * g_w_scale; \
            device const char* u_wb = up_weight + r * row_stride + ib * q4_block_bytes; \
            const float u_w_scale = float(*reinterpret_cast<device const half*>(u_wb)); \
            const uchar4 u_wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(u_wb + 2 + ilb)); \
            const float4 u_lo = float4(int4(u_wq & 0x0F) - 8); \
            const float4 u_hi = float4(int4(u_wq >> 4) - 8); \
            u_scaled_lo[i] = u_lo * u_w_scale; \
            u_scaled_hi[i] = u_hi * u_w_scale; \
        } \
    }

#define SPEC50_GEGLU_STORE \
    if (lane == 0) { \
        const float in_v = 0.7978845608f * (g_tot + 0.044715f * g_tot * g_tot * g_tot); \
        const float gelu = 0.5f * g_tot * (1.0f + tanh(clamp(in_v, -15.0f, 15.0f))); \
        act_output[k * rows + r0 + i] = gelu * u_tot; \
    }

template <uint NR>
static inline void spec50_gateup_flat_body(
    device const float* y,
    device const char* gate_weight,
    device const char* up_weight,
    device float* act_output,
    uint blocks_per_row,
    uint rows,
    uint gsg,
    uint lane
) {
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = gsg * NR;
    if (r0 >= rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;

    bool has_row[NR];
    #pragma unroll
    for (uint i = 0; i < NR; ++i) {
        has_row[i] = (r0 + i < rows);
    }

    float g_sums[NR][KB] = {{0.0f}};
    float u_sums[NR][KB] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 g_scaled_lo[NR], g_scaled_hi[NR];
        float4 u_scaled_lo[NR], u_scaled_hi[NR];

        SPEC50_GATEUP_LOAD

        #pragma unroll
        for (uint k = 0; k < KB; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            #pragma unroll
            for (uint i = 0; i < NR; ++i) {
                if (has_row[i]) {
                    g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                    u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
                }
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < NR; ++i) {
            if (has_row[i]) {
                const float g_tot = simd_sum(g_sums[i][k]);
                const float u_tot = simd_sum(u_sums[i][k]);
                SPEC50_GEGLU_STORE
            }
        }
    }
}

template <uint NR, uint SG, uint TB>
static inline void spec50_gateup_tiled_body(
    device const float* y,
    device const char* gate_weight,
    device const char* up_weight,
    device float* act_output,
    uint blocks_per_row,
    uint rows,
    uint tg,
    uint sg,
    uint lane,
    threadgroup float4* tile4
) {
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = (tg * SG + sg) * NR;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint iq = lane % 4;
    const uint tid = sg * 32 + lane;

    device const float4* y4 = reinterpret_cast<device const float4*>(y);

    bool has_row[NR];
    #pragma unroll
    for (uint i = 0; i < NR; ++i) {
        has_row[i] = (r0 + i < rows);
    }

    float g_sums[NR][KB] = {{0.0f}};
    float u_sums[NR][KB] = {{0.0f}};

    for (uint ts = 0; ts < blocks_per_row; ts += TB) {
        const uint tb = min(TB, blocks_per_row - ts);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < KB * TB * 8; idx += 32 * SG) {
            const uint k = idx / (TB * 8);
            const uint f = idx % (TB * 8);
            if (f < tb * 8) {
                tile4[idx] = y4[k * (hidden >> 2) + (ts << 3) + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ib = ts + ix; ib < ts + tb; ib += NQ) {
            float4 g_scaled_lo[NR], g_scaled_hi[NR];
            float4 u_scaled_lo[NR], u_scaled_hi[NR];

            SPEC50_GATEUP_LOAD

            const uint lb8 = (ib - ts) * 8;
            #pragma unroll
            for (uint k = 0; k < KB; ++k) {
                const float4 ylo = tile4[k * (TB * 8) + lb8 + iq];
                const float4 yhi = tile4[k * (TB * 8) + lb8 + 4 + iq];

                #pragma unroll
                for (uint i = 0; i < NR; ++i) {
                    if (has_row[i]) {
                        g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                        u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < NR; ++i) {
            if (has_row[i]) {
                const float g_tot = simd_sum(g_sums[i][k]);
                const float u_tot = simd_sum(u_sums[i][k]);
                SPEC50_GEGLU_STORE
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Exact-partition K=8 experiments.
//
// These bodies deliberately preserve the *runtime-width* kernels' floating
// point program.  In particular, `k_batch` remains a runtime loop bound and
// QKV keeps its post-dot scale multiply.  The legacy fixed-K8 kernels and the
// v4 kernels above instead expose a compile-time K=8 loop to the compiler;
// that changes FMA contraction and is not partition-safe against K=1.
//
// The only transformations below are scheduling and bit-copying activation
// values through threadgroup memory.  `SG` groups independent SIMD groups in
// one threadgroup.  `TB`, when non-zero, shares an activation tile across those
// SIMD groups while preserving every lane's strictly increasing `ib` order.
//
// M4 falsification result (724.8 MB distinct-weight 30-layer sweep): every
// candidate below was raw-bit exact, but the best result was SG=4/TB=0 at
// 37.912 ms versus the runtime-width oracle's 38.196 ms (1.007x, within run
// noise). Tiled candidates ranged from 0.438x to 0.964x. Consequently none is
// connected to a production selector, and
// `CAMELID_GEMMA4_MTP_FAST_DENSE_K8` is intentionally not admitted.
// ---------------------------------------------------------------------------

const SPEC50_EXACT_DENSE_PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;

template <uint SG>
static inline void spec50_exact_plain_flat_body(
    device const float* y,
    device const char* weight_blocks,
    device float* output,
    uint blocks_per_row,
    uint rows,
    uint k_batch,
    uint tg,
    uint sg,
    uint lane
) {
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = (tg * SG + sg) * NR0;
    if (r0 >= rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;
    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;

    bool has_row[4];
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
        has_row[i] = (r0 + i < rows);
    }
    float sums[4][8] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 scaled_lo[4], scaled_hi[4];
#pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const uint r = r0 + i;
                device const char* wb = weight_blocks + r * row_stride + ib * q4_block_bytes;
                const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                const uchar4 wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb + 2 + ilb));
                const float4 lo = float4(int4(wq & 0x0F) - 8);
                const float4 hi = float4(int4(wq >> 4) - 8);
                scaled_lo[i] = lo * w_scale;
                scaled_hi[i] = hi * w_scale;
            }
        }

#pragma unroll
        for (uint k = 0; k < k_batch; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);
#pragma unroll
            for (uint i = 0; i < 4; ++i) {
                if (has_row[i]) {
                    sums[i][k] += dot(scaled_lo[i], ylo) + dot(scaled_hi[i], yhi);
                }
            }
        }
    }

    for (uint k = 0; k < k_batch; ++k) {
#pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const float tot = simd_sum(sums[i][k]);
                if (lane == 0) output[k * rows + r0 + i] = tot;
            }
        }
    }
}

template <uint SG, uint TB>
static inline void spec50_exact_plain_tiled_body(
    device const float* y,
    device const char* weight_blocks,
    device float* output,
    uint blocks_per_row,
    uint rows,
    uint k_batch,
    uint tg,
    uint sg,
    uint lane,
    threadgroup float4* tile4
) {
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = (tg * SG + sg) * NR0;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;
    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint iq = lane % 4;
    const uint tid = sg * 32 + lane;
    device const float4* y4 = reinterpret_cast<device const float4*>(y);

    bool has_row[4];
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
        has_row[i] = (r0 + i < rows);
    }
    float sums[4][8] = {{0.0f}};

    for (uint ts = 0; ts < blocks_per_row; ts += TB) {
        const uint tb = min(TB, blocks_per_row - ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < k_batch * TB * 8; idx += 32 * SG) {
            const uint k = idx / (TB * 8);
            const uint f = idx % (TB * 8);
            if (f < tb * 8) tile4[idx] = y4[k * (hidden >> 2) + (ts << 3) + f];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ib = ts + ix; ib < ts + tb; ib += NQ) {
            float4 scaled_lo[4], scaled_hi[4];
#pragma unroll
            for (uint i = 0; i < 4; ++i) {
                if (has_row[i]) {
                    const uint r = r0 + i;
                    device const char* wb = weight_blocks + r * row_stride + ib * q4_block_bytes;
                    const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                    const uchar4 wq = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb + 2 + ilb));
                    const float4 lo = float4(int4(wq & 0x0F) - 8);
                    const float4 hi = float4(int4(wq >> 4) - 8);
                    scaled_lo[i] = lo * w_scale;
                    scaled_hi[i] = hi * w_scale;
                }
            }
            const uint lb8 = (ib - ts) * 8;
#pragma unroll
            for (uint k = 0; k < k_batch; ++k) {
                const float4 ylo = tile4[k * (TB * 8) + lb8 + iq];
                const float4 yhi = tile4[k * (TB * 8) + lb8 + 4 + iq];
#pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    if (has_row[i]) {
                        sums[i][k] += dot(scaled_lo[i], ylo) + dot(scaled_hi[i], yhi);
                    }
                }
            }
        }
    }

    for (uint k = 0; k < k_batch; ++k) {
#pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const float tot = simd_sum(sums[i][k]);
                if (lane == 0) output[k * rows + r0 + i] = tot;
            }
        }
    }
}

template <uint SG>
static inline void spec50_exact_qkv_flat_body(
    device const float* y,
    device const char* q_weight,
    device const char* k_weight,
    device const char* v_weight,
    device float* query_out,
    device float* key_out,
    device float* val_out,
    uint blocks_per_row,
    uint q_rows,
    uint k_rows,
    uint v_rows,
    uint k_batch,
    uint tg,
    uint sg,
    uint lane
) {
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint r0 = (tg * SG + sg) * NR0;
    if (r0 >= total_rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;
    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const bool has_row1 = (r0 + 1 < total_rows);

    device const char* w_base0;
    uint target_r0;
    uint target_kind0;
    if (r0 < q_rows) { w_base0 = q_weight; target_r0 = r0; target_kind0 = 0; }
    else if (r0 < q_rows + k_rows) { w_base0 = k_weight; target_r0 = r0 - q_rows; target_kind0 = 1; }
    else { w_base0 = v_weight; target_r0 = r0 - (q_rows + k_rows); target_kind0 = 2; }

    device const char* w_base1 = nullptr;
    uint target_r1 = 0;
    uint target_kind1 = 0;
    if (has_row1) {
        const uint r1 = r0 + 1;
        if (r1 < q_rows) { w_base1 = q_weight; target_r1 = r1; target_kind1 = 0; }
        else if (r1 < q_rows + k_rows) { w_base1 = k_weight; target_r1 = r1 - q_rows; target_kind1 = 1; }
        else { w_base1 = v_weight; target_r1 = r1 - (q_rows + k_rows); target_kind1 = 2; }
    }

    float sum0[8] = {0.0f};
    float sum1[8] = {0.0f};
    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        device const char* wb0 = w_base0 + target_r0 * row_stride + ib * q4_block_bytes;
        const float w_scale0 = float(*reinterpret_cast<device const half*>(wb0));
        const uchar4 wq0 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb0 + 2 + ilb));
        const float4 lo0 = float4(int4(wq0 & 0x0F) - 8);
        const float4 hi0 = float4(int4(wq0 >> 4) - 8);
        float w_scale1 = 0.0f;
        float4 lo1 = 0.0f;
        float4 hi1 = 0.0f;
        if (has_row1) {
            device const char* wb1 = w_base1 + target_r1 * row_stride + ib * q4_block_bytes;
            w_scale1 = float(*reinterpret_cast<device const half*>(wb1));
            const uchar4 wq1 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb1 + 2 + ilb));
            lo1 = float4(int4(wq1 & 0x0F) - 8);
            hi1 = float4(int4(wq1 >> 4) - 8);
        }
        for (uint k = 0; k < k_batch; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);
            sum0[k] += (dot(lo0, ylo) + dot(hi0, yhi)) * w_scale0;
            if (has_row1) sum1[k] += (dot(lo1, ylo) + dot(hi1, yhi)) * w_scale1;
        }
    }
    for (uint k = 0; k < k_batch; ++k) {
        const float tot0 = simd_sum(sum0[k]);
        if (lane == 0) {
            if (target_kind0 == 0) query_out[k * q_rows + target_r0] = tot0;
            else if (target_kind0 == 1) key_out[k * k_rows + target_r0] = tot0;
            else val_out[k * v_rows + target_r0] = tot0;
        }
        if (has_row1) {
            const float tot1 = simd_sum(sum1[k]);
            if (lane == 0) {
                if (target_kind1 == 0) query_out[k * q_rows + target_r1] = tot1;
                else if (target_kind1 == 1) key_out[k * k_rows + target_r1] = tot1;
                else val_out[k * v_rows + target_r1] = tot1;
            }
        }
    }
}

template <uint SG, uint TB>
static inline void spec50_exact_qkv_tiled_body(
    device const float* y,
    device const char* q_weight,
    device const char* k_weight,
    device const char* v_weight,
    device float* query_out,
    device float* key_out,
    device float* val_out,
    uint blocks_per_row,
    uint q_rows,
    uint k_rows,
    uint v_rows,
    uint k_batch,
    uint tg,
    uint sg,
    uint lane,
    threadgroup float4* tile4
) {
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint r0 = (tg * SG + sg) * NR0;
    const bool live0 = r0 < total_rows;
    const uint safe_r0 = live0 ? r0 : 0;
    const bool has_row1 = live0 && (r0 + 1 < total_rows);
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;
    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint iq = lane % 4;
    const uint tid = sg * 32 + lane;
    device const float4* y4 = reinterpret_cast<device const float4*>(y);

    device const char* w_base0;
    uint target_r0;
    uint target_kind0;
    if (safe_r0 < q_rows) { w_base0 = q_weight; target_r0 = safe_r0; target_kind0 = 0; }
    else if (safe_r0 < q_rows + k_rows) { w_base0 = k_weight; target_r0 = safe_r0 - q_rows; target_kind0 = 1; }
    else { w_base0 = v_weight; target_r0 = safe_r0 - (q_rows + k_rows); target_kind0 = 2; }

    device const char* w_base1 = nullptr;
    uint target_r1 = 0;
    uint target_kind1 = 0;
    if (has_row1) {
        const uint r1 = r0 + 1;
        if (r1 < q_rows) { w_base1 = q_weight; target_r1 = r1; target_kind1 = 0; }
        else if (r1 < q_rows + k_rows) { w_base1 = k_weight; target_r1 = r1 - q_rows; target_kind1 = 1; }
        else { w_base1 = v_weight; target_r1 = r1 - (q_rows + k_rows); target_kind1 = 2; }
    }

    float sum0[8] = {0.0f};
    float sum1[8] = {0.0f};
    for (uint ts = 0; ts < blocks_per_row; ts += TB) {
        const uint tb = min(TB, blocks_per_row - ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < k_batch * TB * 8; idx += 32 * SG) {
            const uint k = idx / (TB * 8);
            const uint f = idx % (TB * 8);
            if (f < tb * 8) tile4[idx] = y4[k * (hidden >> 2) + (ts << 3) + f];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ib = ts + ix; ib < ts + tb; ib += NQ) {
            device const char* wb0 = w_base0 + target_r0 * row_stride + ib * q4_block_bytes;
            const float w_scale0 = float(*reinterpret_cast<device const half*>(wb0));
            const uchar4 wq0 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb0 + 2 + ilb));
            const float4 lo0 = float4(int4(wq0 & 0x0F) - 8);
            const float4 hi0 = float4(int4(wq0 >> 4) - 8);
            float w_scale1 = 0.0f;
            float4 lo1 = 0.0f;
            float4 hi1 = 0.0f;
            if (has_row1) {
                device const char* wb1 = w_base1 + target_r1 * row_stride + ib * q4_block_bytes;
                w_scale1 = float(*reinterpret_cast<device const half*>(wb1));
                const uchar4 wq1 = uchar4(*reinterpret_cast<device const packed_uchar4*>(wb1 + 2 + ilb));
                lo1 = float4(int4(wq1 & 0x0F) - 8);
                hi1 = float4(int4(wq1 >> 4) - 8);
            }
            const uint lb8 = (ib - ts) * 8;
            for (uint k = 0; k < k_batch; ++k) {
                const float4 ylo = tile4[k * (TB * 8) + lb8 + iq];
                const float4 yhi = tile4[k * (TB * 8) + lb8 + 4 + iq];
                sum0[k] += (dot(lo0, ylo) + dot(hi0, yhi)) * w_scale0;
                if (has_row1) sum1[k] += (dot(lo1, ylo) + dot(hi1, yhi)) * w_scale1;
            }
        }
    }
    if (!live0) return;
    for (uint k = 0; k < k_batch; ++k) {
        const float tot0 = simd_sum(sum0[k]);
        if (lane == 0) {
            if (target_kind0 == 0) query_out[k * q_rows + target_r0] = tot0;
            else if (target_kind0 == 1) key_out[k * k_rows + target_r0] = tot0;
            else val_out[k * v_rows + target_r0] = tot0;
        }
        if (has_row1) {
            const float tot1 = simd_sum(sum1[k]);
            if (lane == 0) {
                if (target_kind1 == 0) query_out[k * q_rows + target_r1] = tot1;
                else if (target_kind1 == 1) key_out[k * k_rows + target_r1] = tot1;
                else val_out[k * v_rows + target_r1] = tot1;
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Entry-point generation
// ---------------------------------------------------------------------------

/// One kernel configuration. `tb == 0` selects the flat (unstaged) variant;
/// `tb > 0` stages `y` in threadgroup tiles of `tb` Q4_0-block columns
/// (`tb % 8 == 0` required to preserve the reference accumulation order,
/// `8 * tb * 128` bytes of threadgroup memory, so `tb <= 32`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Spec50Cfg {
    pub nr: u32,
    pub sg: u32,
    pub tb: u32,
}

impl Spec50Cfg {
    pub(crate) const fn rows_per_tg(&self) -> u64 {
        (self.nr * self.sg) as u64
    }
    pub(crate) const fn tg_width(&self) -> u64 {
        32 * self.sg as u64
    }
    fn suffix(&self) -> String {
        format!("nr{}_sg{}_tb{}", self.nr, self.sg, self.tb)
    }
    fn valid(&self) -> bool {
        self.nr >= 1
            && self.sg >= 1
            && self.sg <= 32
            && (self.tb == 0 || (self.tb % 8 == 0 && self.tb <= 32))
    }
}

fn push_plain(src: &mut String, c: Spec50Cfg) {
    let Spec50Cfg { nr, sg, tb } = c;
    let w = 32 * sg;
    let body = if tb == 0 {
        format!("spec50_plain_flat_body<{nr}u>(y, weight_blocks, output, blocks_per_row, rows, tg * {sg}u + sg, lane);")
    } else {
        let t4 = 8 * tb * 8;
        format!(
            "threadgroup float4 tile4[{t4}];\n    \
             spec50_plain_tiled_body<{nr}u, {sg}u, {tb}u>(y, weight_blocks, output, blocks_per_row, rows, tg, sg, lane, tile4);"
        )
    };
    src.push_str(&format!(
        r#"
[[max_total_threads_per_threadgroup({w})]]
kernel void spec50_plain_k8_{suffix}(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    (void)k_batch;
    {body}
}}
"#,
        suffix = c.suffix(),
    ));
}

fn push_qkv(src: &mut String, c: Spec50Cfg) {
    // The QKV body is the reference's verbatim 4-row program; it scales via SG
    // (and tiling) only.
    let Spec50Cfg { nr, sg, tb } = Spec50Cfg { nr: 4, ..c };
    let w = 32 * sg;
    let body = if tb == 0 {
        format!(
            "spec50_qkv_flat_body<{nr}u>(y, q_weight, k_weight, v_weight, query_out, key_out, val_out, \
             blocks_per_row, q_rows, k_rows, v_rows, tg * {sg}u + sg, lane);"
        )
    } else {
        let t4 = 8 * tb * 8;
        format!(
            "threadgroup float4 tile4[{t4}];\n    \
             spec50_qkv_tiled_body<{nr}u, {sg}u, {tb}u>(y, q_weight, k_weight, v_weight, query_out, key_out, val_out, \
             blocks_per_row, q_rows, k_rows, v_rows, tg, sg, lane, tile4);"
        )
    };
    src.push_str(&format!(
        r#"
[[max_total_threads_per_threadgroup({w})]]
kernel void spec50_qkv_k8_{suffix}(
    device const float* y [[buffer(0)]],
    device const char* q_weight [[buffer(1)]],
    device const char* k_weight [[buffer(2)]],
    device const char* v_weight [[buffer(3)]],
    device float* query_out [[buffer(4)]],
    device float* key_out [[buffer(5)]],
    device float* val_out [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& q_rows [[buffer(8)]],
    constant uint& k_rows [[buffer(9)]],
    constant uint& v_rows [[buffer(10)]],
    constant uint& k_batch [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    (void)k_batch;
    {body}
}}
"#,
        suffix = c.suffix(),
    ));
}

fn push_gateup(src: &mut String, c: Spec50Cfg) {
    let Spec50Cfg { nr, sg, tb } = c;
    let w = 32 * sg;
    let body = if tb == 0 {
        format!(
            "spec50_gateup_flat_body<{nr}u>(y, gate_weight, up_weight, act_output, blocks_per_row, rows, tg * {sg}u + sg, lane);"
        )
    } else {
        let t4 = 8 * tb * 8;
        format!(
            "threadgroup float4 tile4[{t4}];\n    \
             spec50_gateup_tiled_body<{nr}u, {sg}u, {tb}u>(y, gate_weight, up_weight, act_output, blocks_per_row, rows, tg, sg, lane, tile4);"
        )
    };
    src.push_str(&format!(
        r#"
[[max_total_threads_per_threadgroup({w})]]
kernel void spec50_gateup_k8_{suffix}(
    device const float* y [[buffer(0)]],
    device const char* gate_weight [[buffer(1)]],
    device const char* up_weight [[buffer(2)]],
    device float* act_output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    (void)k_batch;
    {body}
}}
"#,
        suffix = c.suffix(),
    ));
}

/// Build the MSL source containing entry points for the requested configs
/// (one per kernel kind).
fn spec50_shader_src(plain: Spec50Cfg, qkv: Spec50Cfg, gateup: Spec50Cfg) -> String {
    let mut src = String::with_capacity(SPEC50_DENSE_PRELUDE.len() + 8192);
    src.push_str(SPEC50_DENSE_PRELUDE);
    push_plain(&mut src, plain);
    push_qkv(&mut src, qkv);
    push_gateup(&mut src, gateup);
    src
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Spec50ExactCfg {
    pub sg: u32,
    pub tb: u32,
}

impl Spec50ExactCfg {
    fn valid(self) -> bool {
        self.sg >= 1 && self.sg <= 16 && (self.tb == 0 || (self.tb % 8 == 0 && self.tb <= 32))
    }

    fn suffix(self) -> String {
        format!("sg{}_tb{}", self.sg, self.tb)
    }

    fn tg_width(self) -> u64 {
        32 * self.sg as u64
    }
}

fn spec50_exact_dense_shader_src(cfg: Spec50ExactCfg) -> String {
    let mut src = String::with_capacity(SPEC50_EXACT_DENSE_PRELUDE.len() + 4096);
    src.push_str(SPEC50_EXACT_DENSE_PRELUDE);
    let sg = cfg.sg;
    let w = cfg.tg_width();
    let suffix = cfg.suffix();
    let plain_body = if cfg.tb == 0 {
        format!(
            "spec50_exact_plain_flat_body<{sg}u>(y, weight_blocks, output, blocks_per_row, rows, k_batch, tg, sg, lane);"
        )
    } else {
        let tb = cfg.tb;
        let tile4 = 8 * tb * 8;
        format!(
            "threadgroup float4 tile4[{tile4}];\n    spec50_exact_plain_tiled_body<{sg}u, {tb}u>(y, weight_blocks, output, blocks_per_row, rows, k_batch, tg, sg, lane, tile4);"
        )
    };
    src.push_str(&format!(
        r#"
[[max_total_threads_per_threadgroup({w})]]
kernel void spec50_exact_plain_k8_{suffix}(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    {plain_body}
}}
"#
    ));

    let qkv_body = if cfg.tb == 0 {
        format!(
            "spec50_exact_qkv_flat_body<{sg}u>(y, q_weight, k_weight, v_weight, query_out, key_out, val_out, blocks_per_row, q_rows, k_rows, v_rows, k_batch, tg, sg, lane);"
        )
    } else {
        let tb = cfg.tb;
        let tile4 = 8 * tb * 8;
        format!(
            "threadgroup float4 tile4[{tile4}];\n    spec50_exact_qkv_tiled_body<{sg}u, {tb}u>(y, q_weight, k_weight, v_weight, query_out, key_out, val_out, blocks_per_row, q_rows, k_rows, v_rows, k_batch, tg, sg, lane, tile4);"
        )
    };
    src.push_str(&format!(
        r#"
[[max_total_threads_per_threadgroup({w})]]
kernel void spec50_exact_qkv_k8_{suffix}(
    device const float* y [[buffer(0)]],
    device const char* q_weight [[buffer(1)]],
    device const char* k_weight [[buffer(2)]],
    device const char* v_weight [[buffer(3)]],
    device float* query_out [[buffer(4)]],
    device float* key_out [[buffer(5)]],
    device float* val_out [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& q_rows [[buffer(8)]],
    constant uint& k_rows [[buffer(9)]],
    constant uint& v_rows [[buffer(10)]],
    constant uint& k_batch [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    {qkv_body}
}}
"#
    ));
    src
}

pub(crate) struct Spec50ExactDenseKernels {
    plain: ComputePipelineState,
    qkv: ComputePipelineState,
    cfg: Spec50ExactCfg,
}

impl Spec50ExactDenseKernels {
    pub(crate) fn build(device: &Device, cfg: Spec50ExactCfg) -> Option<Self> {
        if !cfg.valid() {
            return None;
        }
        let src = spec50_exact_dense_shader_src(cfg);
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&src, &options)
            .map_err(|err| eprintln!("[metal] exact dense K8 shader compile failed: {err}"))
            .ok()?;
        let pipe = |prefix: &str| -> Option<ComputePipelineState> {
            let name = format!("{prefix}_{}", cfg.suffix());
            let function = library
                .get_function(&name, None)
                .map_err(|err| eprintln!("[metal] exact dense K8 missing {name}: {err}"))
                .ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|err| eprintln!("[metal] exact dense K8 pipeline {name} failed: {err}"))
                .ok()?;
            if pipeline.max_total_threads_per_threadgroup() < cfg.tg_width() {
                eprintln!(
                    "[metal] exact dense K8 {name} width {} exceeds pipeline maximum {}",
                    cfg.tg_width(),
                    pipeline.max_total_threads_per_threadgroup()
                );
                return None;
            }
            Some(pipeline)
        };
        Some(Self {
            plain: pipe("spec50_exact_plain_k8")?,
            qkv: pipe("spec50_exact_qkv_k8")?,
            cfg,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// Shipped configurations (chosen by the in-file benchmarks on the M4 26B
/// shapes; see the sweep tests). All bit-identical to the reference kernels.
///
/// Measured outcome (M4, 30 dispatches/measurement, best of 6,
/// `command_buffer_gpu_times_us`): only the fused QKV wins — 12.3 -> 8.6 ms
/// on the 25-local-layer sweep shape (8192 rows x 88 blocks), 1.44x. Every
/// plain shape (o-proj 2816x128b/2816x256b, q 8192x88b, k 1024x88b, down
/// 2816x66b) and the gateup 2112x88b are 0.6-1.0x under every bit-exact
/// config: those kernels are execution-bound at K=8, not DRAM-bound — the
/// activation re-reads the traffic model counted are served by the cache
/// hierarchy (the "counted" old traffic would imply 146-413 GB/s, past the
/// M4 wall), and K=8 costing ~3x K=1 with identical weight traffic confirms
/// the bound is instruction issue. The parity contract pins the per-row FP
/// program, so the instruction stream per row cannot shrink; plain/gateup
/// therefore keep the shipped kernels (their `_v4` encodes return false).
pub(crate) const SPEC50_V4_PLAIN: Spec50Cfg = Spec50Cfg {
    nr: 4,
    sg: 4,
    tb: 16,
};
pub(crate) const SPEC50_V4_QKV: Spec50Cfg = Spec50Cfg {
    nr: 4,
    sg: 4,
    tb: 16,
};
pub(crate) const SPEC50_V4_GATEUP: Spec50Cfg = Spec50Cfg {
    nr: 4,
    sg: 4,
    tb: 16,
};

pub(crate) struct Spec50V4Kernels {
    plain: ComputePipelineState,
    qkv: ComputePipelineState,
    gateup: ComputePipelineState,
    plain_cfg: Spec50Cfg,
    qkv_cfg: Spec50Cfg,
    gateup_cfg: Spec50Cfg,
}

impl Spec50V4Kernels {
    pub(crate) fn build(
        device: &Device,
        plain_cfg: Spec50Cfg,
        qkv_cfg: Spec50Cfg,
        gateup_cfg: Spec50Cfg,
    ) -> Option<Self> {
        // QKV always runs the verbatim 4-row reference program.
        let qkv_cfg = Spec50Cfg { nr: 4, ..qkv_cfg };
        if !(plain_cfg.valid() && qkv_cfg.valid() && gateup_cfg.valid()) {
            return None;
        }
        let src = spec50_shader_src(plain_cfg, qkv_cfg, gateup_cfg);
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&src, &options)
            .map_err(|err| eprintln!("[metal] SPEC50_DENSE shader compile failed: {err}"))
            .ok()?;

        let pipe = |name: String, cfg: Spec50Cfg| -> Option<ComputePipelineState> {
            let f = library
                .get_function(&name, None)
                .map_err(|err| eprintln!("[metal] spec50: missing {name}: {err}"))
                .ok()?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|err| eprintln!("[metal] spec50: pipeline {name} failed: {err}"))
                .ok()?;
            // The compiled pipeline must honor the declared threadgroup width.
            // Silently dispatching wider than maxTotalThreadsPerThreadgroup is
            // exactly the failure mode that broke bitwise equality at SG>1 in
            // the prior attempt — refuse the config instead.
            if p.max_total_threads_per_threadgroup() < cfg.tg_width() {
                eprintln!(
                    "[metal] spec50: {name} maxTotalThreadsPerThreadgroup {} < dispatch width {}; config refused",
                    p.max_total_threads_per_threadgroup(),
                    cfg.tg_width()
                );
                return None;
            }
            Some(p)
        };

        Some(Self {
            plain: pipe(format!("spec50_plain_k8_{}", plain_cfg.suffix()), plain_cfg)?,
            qkv: pipe(format!("spec50_qkv_k8_{}", qkv_cfg.suffix()), qkv_cfg)?,
            gateup: pipe(
                format!("spec50_gateup_k8_{}", gateup_cfg.suffix()),
                gateup_cfg,
            )?,
            plain_cfg,
            qkv_cfg,
            gateup_cfg,
        })
    }
}

static SPEC50_V4_KERNELS: OnceLock<Option<Spec50V4Kernels>> = OnceLock::new();

pub(crate) fn spec50_v4_kernels() -> Option<&'static Spec50V4Kernels> {
    SPEC50_V4_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            Spec50V4Kernels::build(&device, SPEC50_V4_PLAIN, SPEC50_V4_QKV, SPEC50_V4_GATEUP)
        })
        .as_ref()
}

// ---------------------------------------------------------------------------
// Encode entry points
// ---------------------------------------------------------------------------

fn spec50_grid(rows: usize, rows_per_tg: u64) -> metal::MTLSize {
    metal::MTLSize {
        width: (rows as u64).div_ceil(rows_per_tg.max(1)),
        height: 1,
        depth: 1,
    }
}

fn spec50_tg(width: u64) -> metal::MTLSize {
    metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

#[allow(clippy::too_many_arguments)]
fn spec50_encode_exact_plain_with(
    ks: &Spec50ExactDenseKernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    weight: &Buffer,
    weight_offset: u64,
    output: &Buffer,
    rows: usize,
    blocks_per_row: u32,
    k_batch: u32,
) {
    e.set_compute_pipeline_state(&ks.plain);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), weight_offset);
    e.set_buffer(3, Some(output), 0);
    e.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    let rows_u32 = rows as u32;
    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    e.set_bytes(6, 4, &k_batch as *const u32 as *const _);
    e.dispatch_thread_groups(
        spec50_grid(rows, 4 * ks.cfg.sg as u64),
        spec50_tg(ks.cfg.tg_width()),
    );
}

#[allow(clippy::too_many_arguments)]
fn spec50_encode_exact_qkv_with(
    ks: &Spec50ExactDenseKernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    q_weight: &Buffer,
    k_weight: &Buffer,
    v_weight: &Buffer,
    query_out: &Buffer,
    key_out: &Buffer,
    val_out: &Buffer,
    total_rows: usize,
    scalars: (u32, u32, u32, u32),
    k_batch: u32,
) {
    let (blocks, q_rows, k_rows, v_rows) = scalars;
    e.set_compute_pipeline_state(&ks.qkv);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(1, Some(q_weight), 0);
    e.set_buffer(2, Some(k_weight), 0);
    e.set_buffer(3, Some(v_weight), 0);
    e.set_buffer(4, Some(query_out), 0);
    e.set_buffer(5, Some(key_out), 0);
    e.set_buffer(6, Some(val_out), 0);
    e.set_bytes(7, 4, &blocks as *const u32 as *const _);
    e.set_bytes(8, 4, &q_rows as *const u32 as *const _);
    e.set_bytes(9, 4, &k_rows as *const u32 as *const _);
    e.set_bytes(10, 4, &v_rows as *const u32 as *const _);
    e.set_bytes(11, 4, &k_batch as *const u32 as *const _);
    e.dispatch_thread_groups(
        spec50_grid(total_rows, 2 * ks.cfg.sg as u64),
        spec50_tg(ks.cfg.tg_width()),
    );
}

/// Byte offsets for one canonical eight-token fused-QKV tile.
///
/// A wider wave is composed by rebasing the token-major input and outputs at
/// the Metal binding boundary.  The shader still sees exactly eight local
/// tokens and therefore executes the already-pinned K=8 floating-point
/// program unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Spec50QkvBufferOffsets {
    pub(crate) y: u64,
    pub(crate) query_out: u64,
    pub(crate) key_out: u64,
    pub(crate) val_out: u64,
}

fn spec50_encode_plain_with(
    ks: &Spec50V4Kernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    weight: &Buffer,
    weight_offset: u64,
    output: &Buffer,
    rows: usize,
    blocks_per_row: u32,
) {
    let rows_u32 = rows as u32;
    let k_batch_u32 = 8u32;
    e.set_compute_pipeline_state(&ks.plain);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), weight_offset);
    e.set_buffer(3, Some(output), 0);
    e.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    e.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(
        spec50_grid(rows, ks.plain_cfg.rows_per_tg()),
        spec50_tg(ks.plain_cfg.tg_width()),
    );
}

#[allow(clippy::too_many_arguments)]
fn spec50_encode_qkv_with(
    ks: &Spec50V4Kernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    q_weight: &Buffer,
    k_weight: &Buffer,
    v_weight: &Buffer,
    query_out: &Buffer,
    key_out: &Buffer,
    val_out: &Buffer,
    total_rows: usize,
    scalars: (u32, u32, u32, u32),
) {
    spec50_encode_qkv_at_offsets(
        ks,
        e,
        y,
        q_weight,
        k_weight,
        v_weight,
        query_out,
        key_out,
        val_out,
        total_rows,
        scalars,
        Spec50QkvBufferOffsets::default(),
    );
}

/// Offset-aware form of [`spec50_encode_qkv_with`] used to prove that a K=16
/// token-major wave can be scheduled as two invocations of the exact same K=8
/// arithmetic primitive.
#[allow(clippy::too_many_arguments)]
fn spec50_encode_qkv_at_offsets(
    ks: &Spec50V4Kernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    q_weight: &Buffer,
    k_weight: &Buffer,
    v_weight: &Buffer,
    query_out: &Buffer,
    key_out: &Buffer,
    val_out: &Buffer,
    total_rows: usize,
    scalars: (u32, u32, u32, u32),
    offsets: Spec50QkvBufferOffsets,
) {
    let (bpr, q_rows, k_rows, v_rows) = scalars;
    let k_batch_u32 = 8u32;
    e.set_compute_pipeline_state(&ks.qkv);
    e.set_buffer(0, Some(y), offsets.y);
    e.set_buffer(1, Some(q_weight), 0);
    e.set_buffer(2, Some(k_weight), 0);
    e.set_buffer(3, Some(v_weight), 0);
    e.set_buffer(4, Some(query_out), offsets.query_out);
    e.set_buffer(5, Some(key_out), offsets.key_out);
    e.set_buffer(6, Some(val_out), offsets.val_out);
    e.set_bytes(7, 4, &bpr as *const u32 as *const _);
    e.set_bytes(8, 4, &q_rows as *const u32 as *const _);
    e.set_bytes(9, 4, &k_rows as *const u32 as *const _);
    e.set_bytes(10, 4, &v_rows as *const u32 as *const _);
    e.set_bytes(11, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(
        spec50_grid(total_rows, ks.qkv_cfg.rows_per_tg()),
        spec50_tg(ks.qkv_cfg.tg_width()),
    );
}

#[allow(clippy::too_many_arguments)]
fn spec50_encode_gateup_with(
    ks: &Spec50V4Kernels,
    e: &metal::ComputeCommandEncoderRef,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    rows: usize,
    blocks_per_row: u32,
) {
    let rows_u32 = rows as u32;
    let k_batch_u32 = 8u32;
    e.set_compute_pipeline_state(&ks.gateup);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(1, Some(gate_weight), 0);
    e.set_buffer(2, Some(up_weight), 0);
    e.set_buffer(3, Some(act_output), 0);
    e.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    e.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(
        spec50_grid(rows, ks.gateup_cfg.rows_per_tg()),
        spec50_tg(ks.gateup_cfg.tg_width()),
    );
}

/// Drop-in twin of `encode_gemma4_q4_0_matmul_batch_k`.
///
/// Always returns `false` today: no bit-exact configuration beats the shipped
/// plain kernel on any 26B dispatch shape (see [`SPEC50_V4_PLAIN`]'s doc for
/// the measured table) — the stage is execution-bound and the parity contract
/// pins the per-row instruction stream. The tiled kernel stays available (and
/// bit-exactness-gated) through the tests for future revisiting.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemma4_q4_0_matmul_batch_k_v4(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    weight_offset: u64,
    output: &Buffer,
    rows: usize,
    scalar_buf: &Buffer,
    k_batch: usize,
) -> bool {
    let _ = (
        e,
        k,
        y,
        weight,
        weight_offset,
        output,
        rows,
        scalar_buf,
        k_batch,
    );
    false
}

/// Drop-in twin of `encode_gemma4_q4_0_qkv_matmul_batch_k` for `k_batch == 8`.
///
/// Returns `false` without encoding for any other `k_batch`, when any of the
/// Q/K/V row splits is not a multiple of 4 (the 4-row group routing — the
/// reference kernel's own precondition; the 26B splits are 4096/2048/2048),
/// or when the library failed to build; the caller must then use the existing
/// encode fn, which stays bit-identical by definition.
///
/// Bit-identical to `q4_0_qkv_block_linear_batch_k8` (asserted via
/// `f32::to_bits` by the in-file tests) and measured 1.44x on the 26B
/// local-layer shape (8192 rows x 88 blocks): 12.3 -> 8.6 ms per 30
/// dispatches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemma4_q4_0_qkv_matmul_batch_k_v4(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    q_weight: &Buffer,
    k_weight: &Buffer,
    v_weight: &Buffer,
    query_out: &Buffer,
    key_out: &Buffer,
    val_out: &Buffer,
    qkv_scalar: &Buffer,
    total_rows: usize,
    k_batch: usize,
) -> bool {
    let _ = k;
    if k_batch != 8 {
        return false;
    }
    let Some(ks) = spec50_v4_kernels() else {
        return false;
    };
    let scalars = unsafe {
        let ptr = qkv_scalar.contents() as *const u32;
        (*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3))
    };
    if scalars.1 % 4 != 0 || scalars.2 % 4 != 0 || scalars.3 % 4 != 0 {
        return false;
    }
    spec50_encode_qkv_with(
        ks, e, y, q_weight, k_weight, v_weight, query_out, key_out, val_out, total_rows, scalars,
    );
    true
}

/// Drop-in twin of `encode_gemma4_q4_0_gateup_matmul_batch_k`.
///
/// Always returns `false` today, for the same measured reason as
/// [`encode_gemma4_q4_0_matmul_batch_k_v4`]: every bit-exact configuration is
/// 0.6-0.96x on the 26B gateup shape (2112 rows x 88 blocks).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemma4_q4_0_gateup_matmul_batch_k_v4(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    gateup_scalar: &Buffer,
    rows: usize,
    k_batch: usize,
) -> bool {
    let _ = (
        e,
        k,
        y,
        gate_weight,
        up_weight,
        act_output,
        gateup_scalar,
        rows,
        k_batch,
    );
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const Q4_BLOCK: usize = 18;
    const K8: usize = 8;

    /// The full config sweep proven bitwise. (4,1,0) is the literal reference
    /// row/threadgroup geometry (packed nibbles aside) and must hold by
    /// construction; the rest is what the campaign proposes.
    const SWEEP: [Spec50Cfg; 18] = [
        Spec50Cfg {
            nr: 4,
            sg: 1,
            tb: 0,
        },
        Spec50Cfg {
            nr: 8,
            sg: 1,
            tb: 0,
        },
        Spec50Cfg {
            nr: 16,
            sg: 1,
            tb: 0,
        },
        Spec50Cfg {
            nr: 32,
            sg: 1,
            tb: 0,
        },
        Spec50Cfg {
            nr: 4,
            sg: 4,
            tb: 0,
        },
        Spec50Cfg {
            nr: 4,
            sg: 8,
            tb: 0,
        },
        Spec50Cfg {
            nr: 4,
            sg: 1,
            tb: 16,
        },
        Spec50Cfg {
            nr: 8,
            sg: 1,
            tb: 16,
        },
        Spec50Cfg {
            nr: 16,
            sg: 1,
            tb: 16,
        },
        Spec50Cfg {
            nr: 32,
            sg: 1,
            tb: 16,
        },
        Spec50Cfg {
            nr: 4,
            sg: 2,
            tb: 16,
        },
        Spec50Cfg {
            nr: 4,
            sg: 4,
            tb: 16,
        },
        Spec50Cfg {
            nr: 4,
            sg: 8,
            tb: 16,
        },
        Spec50Cfg {
            nr: 8,
            sg: 4,
            tb: 16,
        },
        Spec50Cfg {
            nr: 4,
            sg: 16,
            tb: 16,
        },
        Spec50Cfg {
            nr: 8,
            sg: 8,
            tb: 16,
        },
        Spec50Cfg {
            nr: 4,
            sg: 8,
            tb: 8,
        },
        Spec50Cfg {
            nr: 4,
            sg: 8,
            tb: 32,
        },
    ];

    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn f32_pm1(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// A random Q4_0 tensor: `rows * blocks` 18-byte blocks. Scales are normal
    /// f16 in roughly 2^-5..2^2 so nothing degenerates to zero or inf.
    fn random_q4_0(rng: &mut Rng, rows: usize, blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; rows * blocks * Q4_BLOCK];
        for b in out.chunks_exact_mut(Q4_BLOCK) {
            let exp = 10 + (rng.next_u32() % 8); // 2^-5 .. 2^2
            let mant = rng.next_u32() % 1024;
            let sign = (rng.next_u32() & 1) << 15;
            let bits = (sign as u16) | ((exp as u16) << 10) | (mant as u16);
            b[0] = (bits & 0xFF) as u8;
            b[1] = (bits >> 8) as u8;
            for x in b[2..].iter_mut() {
                *x = (rng.next_u32() & 0xFF) as u8;
            }
        }
        out
    }

    fn random_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.f32_pm1()).collect()
    }

    fn buf_from<T: Copy>(device: &Device, data: &[T]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn zeros(device: &Device, n: usize) -> Buffer {
        let z = vec![0.0f32; n];
        buf_from(device, &z)
    }

    fn read(b: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
    }

    const TILE_GUARD_BITS: u32 = 0x7fc5_a55a;

    fn poisoned_f32(device: &Device, n: usize) -> Buffer {
        let poison = f32::from_bits(TILE_GUARD_BITS);
        buf_from(device, &vec![poison; n])
    }

    fn assert_poisoned_guards(name: &str, values: &[f32], prefix: usize, body: usize) {
        assert!(
            values[..prefix]
                .iter()
                .all(|value| value.to_bits() == TILE_GUARD_BITS),
            "{name}: prefix guard was modified"
        );
        assert!(
            values[prefix + body..]
                .iter()
                .all(|value| value.to_bits() == TILE_GUARD_BITS),
            "{name}: suffix guard was modified"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shipped_plain_k8_at_offsets(
        c: &Ctx,
        encoder: &metal::ComputeCommandEncoderRef,
        y: &Buffer,
        y_offset: u64,
        weight: &Buffer,
        output: &Buffer,
        output_offset: u64,
        rows: usize,
        blocks_per_row: usize,
    ) {
        let pipeline = c
            .old
            .q4_0_block_batch_k8_pipeline
            .as_ref()
            .expect("shipped fixed-K8 plain pipeline");
        let blocks_u32 = blocks_per_row as u32;
        let rows_u32 = rows as u32;
        let k_u32 = K8 as u32;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(y), y_offset);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(output), output_offset);
        encoder.set_bytes(4, 4, &blocks_u32 as *const u32 as *const _);
        encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: (rows as u64).div_ceil(4),
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

    struct Ctx {
        device: Device,
        queue: CommandQueue,
        old: &'static MetalLinearKernel,
    }

    fn ctx() -> Option<Ctx> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let old = metal_linear_kernel()?;
        Some(Ctx { device, queue, old })
    }

    /// Run one encoder's worth of work; returns (gpu_us, kernel_us).
    fn run<F: FnOnce(&metal::ComputeCommandEncoderRef)>(
        queue: &CommandQueue,
        f: F,
    ) -> (u128, u128) {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        f(e);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        command_buffer_gpu_times_us(cb)
    }

    fn bits_equal(name: &str, new: &[f32], old: &[f32]) -> usize {
        assert_eq!(new.len(), old.len());
        let mut bad = 0usize;
        let mut first = None;
        let mut max_ulp = 0i64;
        for (i, (n, o)) in new.iter().zip(old.iter()).enumerate() {
            if n.to_bits() != o.to_bits() {
                bad += 1;
                if first.is_none() {
                    first = Some((i, *n, *o));
                }
                let ulp = (n.to_bits() as i64 - o.to_bits() as i64).abs();
                max_ulp = max_ulp.max(ulp);
            }
        }
        if bad != 0 {
            let (i, n, o) = first.unwrap();
            eprintln!(
                "[spec50] {name}: {bad}/{} bits differ, max |ulp| {max_ulp}, \
                 first at {i}: new {n:e} (0x{:08x}) vs old {o:e} (0x{:08x})",
                new.len(),
                n.to_bits(),
                o.to_bits()
            );
        }
        bad
    }

    fn nonzero(name: &str, v: &[f32]) {
        assert!(
            v.iter().any(|x| x.is_finite() && x.abs() > 1e-6),
            "{name}: output is degenerate (all zero/NaN)"
        );
        assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite output");
    }

    fn build(c: &Ctx, cfg: Spec50Cfg) -> Spec50V4Kernels {
        Spec50V4Kernels::build(&c.device, cfg, cfg, cfg)
            .unwrap_or_else(|| panic!("spec50 library for {cfg:?}"))
    }

    // -- plain ---------------------------------------------------------------

    fn plain_new(
        c: &Ctx,
        ks: &Spec50V4Kernels,
        y: &Buffer,
        w: &Buffer,
        rows: usize,
        blocks: usize,
    ) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        run(&c.queue, |e| {
            spec50_encode_plain_with(ks, e, y, w, 0, &out, rows, blocks as u32);
        });
        read(&out, rows * K8)
    }

    fn plain_old(c: &Ctx, y: &Buffer, w: &Buffer, rows: usize, blocks: usize) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, K8 as u32, 0u32]);
        run(&c.queue, |e| {
            encode_gemma4_q4_0_matmul_batch_k(e, c.old, y, w, 0, &out, rows, &scalar, K8);
        });
        read(&out, rows * K8)
    }

    fn encode_plain_runtime_width(
        c: &Ctx,
        e: &metal::ComputeCommandEncoderRef,
        y: &Buffer,
        w: &Buffer,
        out: &Buffer,
        rows: usize,
        blocks: usize,
    ) {
        let pipeline = c
            .old
            .q4_0_block_batch_k_pipeline
            .as_ref()
            .expect("runtime-width plain pipeline");
        let blocks = blocks as u32;
        let rows_u32 = rows as u32;
        let k_u32 = K8 as u32;
        e.set_compute_pipeline_state(pipeline);
        e.set_buffer(0, Some(y), 0);
        e.set_buffer(2, Some(w), 0);
        e.set_buffer(3, Some(out), 0);
        e.set_bytes(4, 4, &blocks as *const u32 as *const _);
        e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
        e.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: (rows as u64).div_ceil(4),
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

    fn plain_runtime_width_reference(
        c: &Ctx,
        y: &Buffer,
        w: &Buffer,
        rows: usize,
        blocks: usize,
    ) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        run(&c.queue, |e| {
            encode_plain_runtime_width(c, e, y, w, &out, rows, blocks);
        });
        read(&out, rows * K8)
    }

    fn plain_exact_candidate(
        c: &Ctx,
        ks: &Spec50ExactDenseKernels,
        y: &Buffer,
        w: &Buffer,
        rows: usize,
        blocks: usize,
    ) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        run(&c.queue, |e| {
            spec50_encode_exact_plain_with(ks, e, y, w, 0, &out, rows, blocks as u32, K8 as u32);
        });
        read(&out, rows * K8)
    }

    /// Pins why exact speculative verification must not select the legacy
    /// fixed-K8 plain kernel: it compiles a different floating-point program
    /// from the runtime-width kernel used by K=1.
    #[test]
    fn legacy_fixed_k8_plain_is_not_partition_safe() {
        let Some(c) = ctx() else {
            return;
        };
        let mut rng = Rng(0x504c_4149_4e50_4152);
        let (rows, blocks) = (68usize, 11usize);
        let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
        let fixed = plain_old(&c, &y, &w, rows, blocks);
        let runtime = plain_runtime_width_reference(&c, &y, &w, rows, blocks);
        assert!(
            bits_equal("fixed/runtime plain", &fixed, &runtime) > 0,
            "legacy fixed K8 unexpectedly became partition-safe; re-evaluate its admission gate"
        );
    }

    /// Total bit mismatches for one config across the loop-carried shape set
    /// (blocks > 8 forces multi-iteration accumulation chains — the regime
    /// where compiled FMA contraction can diverge).
    fn plain_mismatches(c: &Ctx, cfg: Spec50Cfg) -> usize {
        let ks = build(c, cfg);
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut bad = 0usize;
        for &(rows, blocks) in &[(64usize, 8usize), (37, 11), (130, 16), (66, 22)] {
            let w = random_q4_0(&mut rng, rows, blocks);
            let wb = buf_from(&c.device, &w);
            let yv = random_f32(&mut rng, blocks * 32 * K8);
            let yb = buf_from(&c.device, &yv);
            let new = plain_new(&c, &ks, &yb, &wb, rows, blocks);
            let old = plain_old(c, &yb, &wb, rows, blocks);
            nonzero("plain", &new);
            bad += bits_equal(
                &format!("plain {cfg:?} rows={rows} blocks={blocks}"),
                &new,
                &old,
            );
        }
        bad
    }

    /// Informational sweep: prints which configs reproduce the reference bit
    /// for bit. Gate: the reference-geometry config (4,1,0) must pass — if it
    /// does not, cross-library equality is broken and nothing here can ship.
    #[test]
    fn spec50_plain_bitwise_sweep() {
        let Some(c) = ctx() else {
            eprintln!("[spec50] no Metal device, skipping");
            return;
        };
        for &cfg in &SWEEP {
            let bad = plain_mismatches(&c, cfg);
            println!(
                "[spec50] plain {cfg:?}: {}",
                if bad == 0 {
                    "BIT-EXACT".to_string()
                } else {
                    format!("{bad} mismatches")
                }
            );
            if cfg
                == (Spec50Cfg {
                    nr: 4,
                    sg: 1,
                    tb: 0,
                })
            {
                assert_eq!(bad, 0, "reference-geometry config must be bit-exact");
            }
        }
    }

    /// Hard parity gate on the shipped plain config: zero bit differences.
    #[test]
    fn spec50_plain_shipped_bitwise() {
        let Some(c) = ctx() else {
            return;
        };
        assert_eq!(
            plain_mismatches(&c, SPEC50_V4_PLAIN),
            0,
            "{SPEC50_V4_PLAIN:?}"
        );
    }

    /// Per-token batch independence at K=8: token t's outputs must not depend
    /// on the other seven tokens' activations.
    #[test]
    fn spec50_plain_batch_independent() {
        let Some(c) = ctx() else {
            return;
        };
        let ks = build(&c, SPEC50_V4_PLAIN);
        let mut rng = Rng(0xDEADBEEF12345678);
        let (rows, blocks) = (68usize, 12usize);
        let w = random_q4_0(&mut rng, rows, blocks);
        let wb = buf_from(&c.device, &w);
        let ya = random_f32(&mut rng, blocks * 32 * K8);
        let a = plain_new(&c, &ks, &buf_from(&c.device, &ya), &wb, rows, blocks);
        for t in 0..K8 {
            // batch B: every token replaced except t
            let mut ybv = random_f32(&mut rng, blocks * 32 * K8);
            ybv[t * blocks * 32..(t + 1) * blocks * 32]
                .copy_from_slice(&ya[t * blocks * 32..(t + 1) * blocks * 32]);
            let b = plain_new(&c, &ks, &buf_from(&c.device, &ybv), &wb, rows, blocks);
            let bad = bits_equal(
                &format!("plain independence t={t}"),
                &b[t * rows..(t + 1) * rows],
                &a[t * rows..(t + 1) * rows],
            );
            assert_eq!(bad, 0, "plain token {t} depends on its batch companions");
        }
    }

    // -- qkv -----------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn qkv_new(
        c: &Ctx,
        ks: &Spec50V4Kernels,
        y: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        (qr, kr, vr): (usize, usize, usize),
        blocks: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let oq = zeros(&c.device, qr * K8);
        let ok = zeros(&c.device, kr * K8);
        let ov = zeros(&c.device, vr * K8);
        run(&c.queue, |e| {
            spec50_encode_qkv_with(
                ks,
                e,
                y,
                wq,
                wk,
                wv,
                &oq,
                &ok,
                &ov,
                qr + kr + vr,
                (blocks as u32, qr as u32, kr as u32, vr as u32),
            );
        });
        (read(&oq, qr * K8), read(&ok, kr * K8), read(&ov, vr * K8))
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_old(
        c: &Ctx,
        y: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        (qr, kr, vr): (usize, usize, usize),
        blocks: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let oq = zeros(&c.device, qr * K8);
        let ok = zeros(&c.device, kr * K8);
        let ov = zeros(&c.device, vr * K8);
        let scalar = buf_from(
            &c.device,
            &[blocks as u32, qr as u32, kr as u32, vr as u32, K8 as u32],
        );
        run(&c.queue, |e| {
            encode_gemma4_q4_0_qkv_matmul_batch_k(
                e,
                c.old,
                y,
                wq,
                wk,
                wv,
                &oq,
                &ok,
                &ov,
                &scalar,
                qr + kr + vr,
                K8,
            );
        });
        (read(&oq, qr * K8), read(&ok, kr * K8), read(&ov, vr * K8))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qkv_runtime_width(
        c: &Ctx,
        e: &metal::ComputeCommandEncoderRef,
        y: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        oq: &Buffer,
        ok: &Buffer,
        ov: &Buffer,
        (qr, kr, vr): (usize, usize, usize),
        blocks: usize,
    ) {
        let pipeline = c
            .old
            .q4_0_qkv_block_batch_k_pipeline
            .as_ref()
            .expect("runtime-width QKV pipeline");
        let blocks = blocks as u32;
        let qr_u32 = qr as u32;
        let kr_u32 = kr as u32;
        let vr_u32 = vr as u32;
        let k_u32 = K8 as u32;
        e.set_compute_pipeline_state(pipeline);
        e.set_buffer(0, Some(y), 0);
        e.set_buffer(1, Some(wq), 0);
        e.set_buffer(2, Some(wk), 0);
        e.set_buffer(3, Some(wv), 0);
        e.set_buffer(4, Some(oq), 0);
        e.set_buffer(5, Some(ok), 0);
        e.set_buffer(6, Some(ov), 0);
        e.set_bytes(7, 4, &blocks as *const u32 as *const _);
        e.set_bytes(8, 4, &qr_u32 as *const u32 as *const _);
        e.set_bytes(9, 4, &kr_u32 as *const u32 as *const _);
        e.set_bytes(10, 4, &vr_u32 as *const u32 as *const _);
        e.set_bytes(11, 4, &k_u32 as *const u32 as *const _);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: ((qr + kr + vr) as u64).div_ceil(2),
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
    fn qkv_runtime_width_reference(
        c: &Ctx,
        y: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        (qr, kr, vr): (usize, usize, usize),
        blocks: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let oq = zeros(&c.device, qr * K8);
        let ok = zeros(&c.device, kr * K8);
        let ov = zeros(&c.device, vr * K8);
        run(&c.queue, |e| {
            encode_qkv_runtime_width(c, e, y, wq, wk, wv, &oq, &ok, &ov, (qr, kr, vr), blocks);
        });
        (read(&oq, qr * K8), read(&ok, kr * K8), read(&ov, vr * K8))
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_exact_candidate(
        c: &Ctx,
        ks: &Spec50ExactDenseKernels,
        y: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        (qr, kr, vr): (usize, usize, usize),
        blocks: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let oq = zeros(&c.device, qr * K8);
        let ok = zeros(&c.device, kr * K8);
        let ov = zeros(&c.device, vr * K8);
        run(&c.queue, |e| {
            spec50_encode_exact_qkv_with(
                ks,
                e,
                y,
                wq,
                wk,
                wv,
                &oq,
                &ok,
                &ov,
                qr + kr + vr,
                (blocks as u32, qr as u32, kr as u32, vr as u32),
                K8 as u32,
            );
        });
        (read(&oq, qr * K8), read(&ok, kr * K8), read(&ov, vr * K8))
    }

    const EXACT_CANDIDATES: [Spec50ExactCfg; 10] = [
        Spec50ExactCfg { sg: 1, tb: 0 },
        Spec50ExactCfg { sg: 2, tb: 0 },
        Spec50ExactCfg { sg: 4, tb: 0 },
        Spec50ExactCfg { sg: 8, tb: 0 },
        Spec50ExactCfg { sg: 2, tb: 8 },
        Spec50ExactCfg { sg: 2, tb: 16 },
        Spec50ExactCfg { sg: 4, tb: 8 },
        Spec50ExactCfg { sg: 4, tb: 16 },
        Spec50ExactCfg { sg: 4, tb: 32 },
        Spec50ExactCfg { sg: 8, tb: 16 },
    ];

    /// Direct Metal admission sweep against the runtime-width oracle.  This is
    /// intentionally not the legacy fixed-K8 oracle: exact speculative
    /// verification must preserve the K=1 partition's raw `f32` bits.
    #[test]
    fn spec50_exact_dense_k8_candidate_raw_bit_sweep() {
        let Some(c) = ctx() else {
            return;
        };
        for cfg in EXACT_CANDIDATES {
            let ks = Spec50ExactDenseKernels::build(&c.device, cfg)
                .unwrap_or_else(|| panic!("exact dense config {cfg:?}"));
            let mut rng = Rng(0x4558_4143_544b_3800 ^ ((cfg.sg as u64) << 8) ^ cfg.tb as u64);

            let (rows, blocks) = (68usize, 22usize);
            let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let plain = plain_exact_candidate(&c, &ks, &y, &w, rows, blocks);
            let plain_ref = plain_runtime_width_reference(&c, &y, &w, rows, blocks);
            let mut bad = bits_equal(&format!("exact plain {cfg:?}"), &plain, &plain_ref);

            let (qr, kr, vr, blocks) = (48usize, 16usize, 16usize, 22usize);
            let wq = buf_from(&c.device, &random_q4_0(&mut rng, qr, blocks));
            let wk = buf_from(&c.device, &random_q4_0(&mut rng, kr, blocks));
            let wv = buf_from(&c.device, &random_q4_0(&mut rng, vr, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let got = qkv_exact_candidate(&c, &ks, &y, &wq, &wk, &wv, (qr, kr, vr), blocks);
            let oracle = qkv_runtime_width_reference(&c, &y, &wq, &wk, &wv, (qr, kr, vr), blocks);
            bad += bits_equal(&format!("exact Q {cfg:?}"), &got.0, &oracle.0);
            bad += bits_equal(&format!("exact K {cfg:?}"), &got.1, &oracle.1);
            bad += bits_equal(&format!("exact V {cfg:?}"), &got.2, &oracle.2);
            println!(
                "[spec50 exact dense] {cfg:?}: {}",
                if bad == 0 {
                    "runtime-width RAW-BIT EXACT".to_string()
                } else {
                    format!("REJECTED ({bad} raw-bit mismatches)")
                }
            );
            if cfg == (Spec50ExactCfg { sg: 1, tb: 0 }) {
                assert_eq!(bad, 0, "literal runtime-width geometry must be exact");
            }
        }
    }

    /// The least-slow exact candidate is also pinned at every real Gemma 4
    /// dense shape before being classified as a performance no-win.  Keeping
    /// this separate from the small stress sweep makes the falsification
    /// reproducible if a future compiler or GPU changes the result.
    #[test]
    fn spec50_exact_dense_k8_best_candidate_real_shape_raw_bit_parity() {
        let Some(c) = ctx() else {
            return;
        };
        let cfg = Spec50ExactCfg { sg: 4, tb: 0 };
        let ks = Spec50ExactDenseKernels::build(&c.device, cfg)
            .expect("least-slow exact dense candidate");
        let mut rng = Rng(0x4558_4143_5452_4541);
        for (name, rows, blocks) in [
            ("local_o", 2816usize, 128usize),
            ("global_o", 2816, 256),
            ("global_q", 8192, 88),
            ("global_k", 1024, 88),
            ("shared_down", 2816, 66),
        ] {
            let weight = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let got = plain_exact_candidate(&c, &ks, &y, &weight, rows, blocks);
            let oracle = plain_runtime_width_reference(&c, &y, &weight, rows, blocks);
            assert_eq!(
                bits_equal(&format!("real {name} {cfg:?}"), &got, &oracle),
                0,
                "{name} must preserve runtime-width raw bits"
            );
        }

        let (q_rows, k_rows, v_rows, blocks) = (4096usize, 2048usize, 2048usize, 88usize);
        let q_weight = buf_from(&c.device, &random_q4_0(&mut rng, q_rows, blocks));
        let k_weight = buf_from(&c.device, &random_q4_0(&mut rng, k_rows, blocks));
        let v_weight = buf_from(&c.device, &random_q4_0(&mut rng, v_rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
        let got = qkv_exact_candidate(
            &c,
            &ks,
            &y,
            &q_weight,
            &k_weight,
            &v_weight,
            (q_rows, k_rows, v_rows),
            blocks,
        );
        let oracle = qkv_runtime_width_reference(
            &c,
            &y,
            &q_weight,
            &k_weight,
            &v_weight,
            (q_rows, k_rows, v_rows),
            blocks,
        );
        assert_eq!(bits_equal("real local Q", &got.0, &oracle.0), 0);
        assert_eq!(bits_equal("real local K", &got.1, &oracle.1), 0);
        assert_eq!(bits_equal("real local V", &got.2, &oracle.2), 0);
    }

    /// Existing v4 tests compare staged QKV to this legacy fixed-K8 kernel.
    /// This oracle instead compares that shared fixed program to the
    /// runtime-width K=1 family and pins the partition-parity defect directly.
    #[test]
    fn legacy_fixed_k8_qkv_is_not_partition_safe() {
        let Some(c) = ctx() else {
            return;
        };
        let mut rng = Rng(0x4b31_5041_5254_4954);
        let (qr, kr, vr, blocks) = (48usize, 16usize, 16usize, 11usize);
        let wq = buf_from(&c.device, &random_q4_0(&mut rng, qr, blocks));
        let wk = buf_from(&c.device, &random_q4_0(&mut rng, kr, blocks));
        let wv = buf_from(&c.device, &random_q4_0(&mut rng, vr, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
        let fixed = qkv_old(&c, &y, &wq, &wk, &wv, (qr, kr, vr), blocks);
        let runtime = qkv_runtime_width_reference(&c, &y, &wq, &wk, &wv, (qr, kr, vr), blocks);
        let bad = bits_equal("fixed/runtime Q", &fixed.0, &runtime.0)
            + bits_equal("fixed/runtime K", &fixed.1, &runtime.1)
            + bits_equal("fixed/runtime V", &fixed.2, &runtime.2);
        assert!(
            bad > 0,
            "legacy fixed K8 unexpectedly became partition-safe; re-evaluate its admission gate"
        );
    }

    fn qkv_mismatches(c: &Ctx, cfg: Spec50Cfg) -> usize {
        let ks = build(c, cfg);
        let mut rng = Rng(0x123456789ABCDEF);
        let mut bad = 0usize;
        // Splits are multiples of 4 and totals multiples of 4: the reference
        // _k8 kernel resolves the Q/K/V base once per 4-row group and stores
        // all 4 rows unconditionally, so it is only well defined there. The
        // real 26B fused-QKV splits (4096/2048/2048) satisfy this.
        for &(qr, kr, vr, blocks) in &[
            (16usize, 8usize, 8usize, 8usize),
            (48, 16, 16, 11),
            (24, 12, 12, 9),
            (128, 64, 64, 16),
        ] {
            let wq = buf_from(&c.device, &random_q4_0(&mut rng, qr, blocks));
            let wk = buf_from(&c.device, &random_q4_0(&mut rng, kr, blocks));
            let wv = buf_from(&c.device, &random_q4_0(&mut rng, vr, blocks));
            let yv = random_f32(&mut rng, blocks * 32 * K8);
            let yb = buf_from(&c.device, &yv);
            let (nq, nk, nv) = qkv_new(c, &ks, &yb, &wq, &wk, &wv, (qr, kr, vr), blocks);
            let (oq, ok, ov) = qkv_old(c, &yb, &wq, &wk, &wv, (qr, kr, vr), blocks);
            nonzero("qkv.q", &nq);
            let tag = format!("qkv {cfg:?} split={qr}/{kr}/{vr} blocks={blocks}");
            bad += bits_equal(&format!("{tag} Q"), &nq, &oq);
            bad += bits_equal(&format!("{tag} K"), &nk, &ok);
            bad += bits_equal(&format!("{tag} V"), &nv, &ov);
        }
        bad
    }

    #[test]
    fn spec50_qkv_bitwise_sweep() {
        let Some(c) = ctx() else {
            return;
        };
        for &cfg in &SWEEP {
            let bad = qkv_mismatches(&c, cfg);
            println!(
                "[spec50] qkv {cfg:?}: {}",
                if bad == 0 {
                    "BIT-EXACT".to_string()
                } else {
                    format!("{bad} mismatches")
                }
            );
            if cfg
                == (Spec50Cfg {
                    nr: 4,
                    sg: 1,
                    tb: 0,
                })
            {
                assert_eq!(bad, 0, "reference-geometry config must be bit-exact");
            }
        }
    }

    #[test]
    fn spec50_qkv_shipped_bitwise() {
        let Some(c) = ctx() else {
            return;
        };
        assert_eq!(qkv_mismatches(&c, SPEC50_V4_QKV), 0, "{SPEC50_V4_QKV:?}");
    }

    #[test]
    fn spec50_qkv_batch_independent() {
        let Some(c) = ctx() else {
            return;
        };
        let ks = build(&c, SPEC50_V4_QKV);
        let mut rng = Rng(0xFEEDFACECAFEBEEF);
        let (qr, kr, vr, blocks) = (32usize, 16usize, 16usize, 10usize);
        let wq = buf_from(&c.device, &random_q4_0(&mut rng, qr, blocks));
        let wk = buf_from(&c.device, &random_q4_0(&mut rng, kr, blocks));
        let wv = buf_from(&c.device, &random_q4_0(&mut rng, vr, blocks));
        let ya = random_f32(&mut rng, blocks * 32 * K8);
        let (aq, ak, av) = qkv_new(
            &c,
            &ks,
            &buf_from(&c.device, &ya),
            &wq,
            &wk,
            &wv,
            (qr, kr, vr),
            blocks,
        );
        for t in 0..K8 {
            let mut ybv = random_f32(&mut rng, blocks * 32 * K8);
            ybv[t * blocks * 32..(t + 1) * blocks * 32]
                .copy_from_slice(&ya[t * blocks * 32..(t + 1) * blocks * 32]);
            let (bq, bk, bv) = qkv_new(
                &c,
                &ks,
                &buf_from(&c.device, &ybv),
                &wq,
                &wk,
                &wv,
                (qr, kr, vr),
                blocks,
            );
            for (name, new, base, rows) in [
                ("Q", &bq, &aq, qr),
                ("K", &bk, &ak, kr),
                ("V", &bv, &av, vr),
            ] {
                let bad = bits_equal(
                    &format!("qkv independence {name} t={t}"),
                    &new[t * rows..(t + 1) * rows],
                    &base[t * rows..(t + 1) * rows],
                );
                assert_eq!(bad, 0, "qkv {name} token {t} depends on its companions");
            }
        }
    }

    /// A 16-token dense wave is two rebased invocations of the canonical K=8
    /// program, never a K=16 arithmetic flavor.  This covers the production
    /// fused-QKV kernel and the shipped plain K8 kernel used by O projection at
    /// real 26B local-layer shapes.  The guarded shared buffers make token 7/8
    /// adjacent in memory and catch any write outside either tiled output.
    #[test]
    fn spec50_dense_qkv_o_k16_is_two_bit_exact_k8_tiles() {
        let Some(c) = ctx() else {
            return;
        };
        let ks = build(&c, SPEC50_V4_QKV);
        const TILE_K: usize = 8;
        const WAVE_K: usize = 16;
        const GUARD: usize = 19;
        let poison = f32::from_bits(TILE_GUARD_BITS);
        let mut rng = Rng(0xd3e5_0008_0016_7e57);

        // Production local-layer fused QKV: 4096 Q rows, 2048 K rows, 2048 V
        // rows, over Gemma's 2,816-wide hidden vector (88 Q4 blocks).
        let (q_rows, k_rows, v_rows, qkv_blocks) = (4096usize, 2048usize, 2048usize, 88usize);
        let qkv_hidden = qkv_blocks * 32;
        let q_weight = buf_from(&c.device, &random_q4_0(&mut rng, q_rows, qkv_blocks));
        let k_weight = buf_from(&c.device, &random_q4_0(&mut rng, k_rows, qkv_blocks));
        let v_weight = buf_from(&c.device, &random_q4_0(&mut rng, v_rows, qkv_blocks));
        let qkv_y = random_f32(&mut rng, WAVE_K * qkv_hidden);

        let mut q_ref = Vec::with_capacity(WAVE_K * q_rows);
        let mut k_ref = Vec::with_capacity(WAVE_K * k_rows);
        let mut v_ref = Vec::with_capacity(WAVE_K * v_rows);
        for tile in 0..2 {
            let y0 = tile * TILE_K * qkv_hidden;
            let y = buf_from(&c.device, &qkv_y[y0..y0 + TILE_K * qkv_hidden]);
            let (q, k, v) = qkv_new(
                &c,
                &ks,
                &y,
                &q_weight,
                &k_weight,
                &v_weight,
                (q_rows, k_rows, v_rows),
                qkv_blocks,
            );
            q_ref.extend_from_slice(&q);
            k_ref.extend_from_slice(&k);
            v_ref.extend_from_slice(&v);
        }

        let mut guarded_qkv_y = vec![poison; GUARD];
        guarded_qkv_y.extend_from_slice(&qkv_y);
        guarded_qkv_y.extend_from_slice(&vec![poison; GUARD]);
        let qkv_y_buf = buf_from(&c.device, &guarded_qkv_y);
        let q_out = poisoned_f32(&c.device, GUARD + WAVE_K * q_rows + GUARD);
        let k_out = poisoned_f32(&c.device, GUARD + WAVE_K * k_rows + GUARD);
        let v_out = poisoned_f32(&c.device, GUARD + WAVE_K * v_rows + GUARD);
        run(&c.queue, |encoder| {
            for tile in 0..2 {
                spec50_encode_qkv_at_offsets(
                    &ks,
                    encoder,
                    &qkv_y_buf,
                    &q_weight,
                    &k_weight,
                    &v_weight,
                    &q_out,
                    &k_out,
                    &v_out,
                    q_rows + k_rows + v_rows,
                    (
                        qkv_blocks as u32,
                        q_rows as u32,
                        k_rows as u32,
                        v_rows as u32,
                    ),
                    Spec50QkvBufferOffsets {
                        y: ((GUARD + tile * TILE_K * qkv_hidden) * 4) as u64,
                        query_out: ((GUARD + tile * TILE_K * q_rows) * 4) as u64,
                        key_out: ((GUARD + tile * TILE_K * k_rows) * 4) as u64,
                        val_out: ((GUARD + tile * TILE_K * v_rows) * 4) as u64,
                    },
                );
            }
        });

        let q_all = read(&q_out, GUARD + WAVE_K * q_rows + GUARD);
        let k_all = read(&k_out, GUARD + WAVE_K * k_rows + GUARD);
        let v_all = read(&v_out, GUARD + WAVE_K * v_rows + GUARD);
        let q_got = &q_all[GUARD..GUARD + WAVE_K * q_rows];
        let k_got = &k_all[GUARD..GUARD + WAVE_K * k_rows];
        let v_got = &v_all[GUARD..GUARD + WAVE_K * v_rows];
        for (name, all, got, want, rows) in [
            ("QKV.Q", &q_all, q_got, &q_ref, q_rows),
            ("QKV.K", &k_all, k_got, &k_ref, k_rows),
            ("QKV.V", &v_all, v_got, &v_ref, v_rows),
        ] {
            assert_poisoned_guards(name, all, GUARD, WAVE_K * rows);
            assert_eq!(bits_equal(name, got, want), 0, "{name} K8 tiles diverged");
            for seam in [TILE_K * rows - 1, TILE_K * rows] {
                assert_eq!(
                    got[seam].to_bits(),
                    want[seam].to_bits(),
                    "{name}: token 7/8 seam differs at element {seam}"
                );
            }
        }
        let qkv_y_after = read(&qkv_y_buf, GUARD + WAVE_K * qkv_hidden + GUARD);
        assert_poisoned_guards("QKV input", &qkv_y_after, GUARD, WAVE_K * qkv_hidden);

        // Production local O projection: 2,816 output rows over the concatenated
        // 4,096-wide attention context (128 Q4 blocks).
        let (o_rows, o_blocks) = (2816usize, 128usize);
        let o_hidden = o_blocks * 32;
        let o_weight = buf_from(&c.device, &random_q4_0(&mut rng, o_rows, o_blocks));
        let o_y = random_f32(&mut rng, WAVE_K * o_hidden);
        let mut o_ref = Vec::with_capacity(WAVE_K * o_rows);
        for tile in 0..2 {
            let y0 = tile * TILE_K * o_hidden;
            let y = buf_from(&c.device, &o_y[y0..y0 + TILE_K * o_hidden]);
            let out = zeros(&c.device, TILE_K * o_rows);
            run(&c.queue, |encoder| {
                encode_shipped_plain_k8_at_offsets(
                    &c, encoder, &y, 0, &o_weight, &out, 0, o_rows, o_blocks,
                );
            });
            o_ref.extend_from_slice(&read(&out, TILE_K * o_rows));
        }

        let mut guarded_o_y = vec![poison; GUARD];
        guarded_o_y.extend_from_slice(&o_y);
        guarded_o_y.extend_from_slice(&vec![poison; GUARD]);
        let o_y_buf = buf_from(&c.device, &guarded_o_y);
        let o_out = poisoned_f32(&c.device, GUARD + WAVE_K * o_rows + GUARD);
        run(&c.queue, |encoder| {
            for tile in 0..2 {
                encode_shipped_plain_k8_at_offsets(
                    &c,
                    encoder,
                    &o_y_buf,
                    ((GUARD + tile * TILE_K * o_hidden) * 4) as u64,
                    &o_weight,
                    &o_out,
                    ((GUARD + tile * TILE_K * o_rows) * 4) as u64,
                    o_rows,
                    o_blocks,
                );
            }
        });
        let o_all = read(&o_out, GUARD + WAVE_K * o_rows + GUARD);
        let o_got = &o_all[GUARD..GUARD + WAVE_K * o_rows];
        assert_poisoned_guards("O", &o_all, GUARD, WAVE_K * o_rows);
        assert_eq!(bits_equal("O", o_got, &o_ref), 0, "O K8 tiles diverged");
        for seam in [TILE_K * o_rows - 1, TILE_K * o_rows] {
            assert_eq!(
                o_got[seam].to_bits(),
                o_ref[seam].to_bits(),
                "O: token 7/8 seam differs at element {seam}"
            );
        }
        let o_y_after = read(&o_y_buf, GUARD + WAVE_K * o_hidden + GUARD);
        assert_poisoned_guards("O input", &o_y_after, GUARD, WAVE_K * o_hidden);
    }

    // -- gateup ----------------------------------------------------------------

    fn gateup_new(
        c: &Ctx,
        ks: &Spec50V4Kernels,
        y: &Buffer,
        wg: &Buffer,
        wu: &Buffer,
        rows: usize,
        blocks: usize,
    ) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        run(&c.queue, |e| {
            spec50_encode_gateup_with(ks, e, y, wg, wu, &out, rows, blocks as u32);
        });
        read(&out, rows * K8)
    }

    fn gateup_old(
        c: &Ctx,
        y: &Buffer,
        wg: &Buffer,
        wu: &Buffer,
        rows: usize,
        blocks: usize,
    ) -> Vec<f32> {
        let out = zeros(&c.device, rows * K8);
        let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, K8 as u32, 0u32]);
        run(&c.queue, |e| {
            encode_gemma4_q4_0_gateup_matmul_batch_k(e, c.old, y, wg, wu, &out, &scalar, rows, K8);
        });
        read(&out, rows * K8)
    }

    fn gateup_mismatches(c: &Ctx, cfg: Spec50Cfg) -> usize {
        let ks = build(c, cfg);
        let mut rng = Rng(0x5EED5EED5EED5EED);
        let mut bad = 0usize;
        for &(rows, blocks) in &[(64usize, 8usize), (37, 11), (130, 16), (66, 22)] {
            let wg = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let wu = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let yv = random_f32(&mut rng, blocks * 32 * K8);
            let yb = buf_from(&c.device, &yv);
            let new = gateup_new(&c, &ks, &yb, &wg, &wu, rows, blocks);
            let old = gateup_old(c, &yb, &wg, &wu, rows, blocks);
            nonzero("gateup", &new);
            bad += bits_equal(
                &format!("gateup {cfg:?} rows={rows} blocks={blocks}"),
                &new,
                &old,
            );
        }
        bad
    }

    #[test]
    fn spec50_gateup_bitwise_sweep() {
        let Some(c) = ctx() else {
            return;
        };
        for &cfg in &SWEEP {
            let bad = gateup_mismatches(&c, cfg);
            println!(
                "[spec50] gateup {cfg:?}: {}",
                if bad == 0 {
                    "BIT-EXACT".to_string()
                } else {
                    format!("{bad} mismatches")
                }
            );
            if cfg
                == (Spec50Cfg {
                    nr: 4,
                    sg: 1,
                    tb: 0,
                })
            {
                assert_eq!(bad, 0, "reference-geometry config must be bit-exact");
            }
        }
    }

    #[test]
    fn spec50_gateup_shipped_bitwise() {
        let Some(c) = ctx() else {
            return;
        };
        assert_eq!(
            gateup_mismatches(&c, SPEC50_V4_GATEUP),
            0,
            "{SPEC50_V4_GATEUP:?}"
        );
    }

    #[test]
    fn spec50_gateup_batch_independent() {
        let Some(c) = ctx() else {
            return;
        };
        let ks = build(&c, SPEC50_V4_GATEUP);
        let mut rng = Rng(0x0BADC0DE0BADC0DE);
        let (rows, blocks) = (68usize, 12usize);
        let wg = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let wu = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let ya = random_f32(&mut rng, blocks * 32 * K8);
        let a = gateup_new(&c, &ks, &buf_from(&c.device, &ya), &wg, &wu, rows, blocks);
        for t in 0..K8 {
            let mut ybv = random_f32(&mut rng, blocks * 32 * K8);
            ybv[t * blocks * 32..(t + 1) * blocks * 32]
                .copy_from_slice(&ya[t * blocks * 32..(t + 1) * blocks * 32]);
            let b = gateup_new(&c, &ks, &buf_from(&c.device, &ybv), &wg, &wu, rows, blocks);
            let bad = bits_equal(
                &format!("gateup independence t={t}"),
                &b[t * rows..(t + 1) * rows],
                &a[t * rows..(t + 1) * rows],
            );
            assert_eq!(bad, 0, "gateup token {t} depends on its companions");
        }
    }

    // -- guard rails -----------------------------------------------------------

    #[test]
    fn spec50_v4_encode_gating() {
        let Some(c) = ctx() else {
            return;
        };
        let rows = 32usize;
        let blocks = 8usize;
        let mut rng = Rng(7);
        let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * 9));
        let out = zeros(&c.device, rows * 9);
        let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, 8u32, 0u32]);
        let qscalar = buf_from(&c.device, &[blocks as u32, 16u32, 8u32, 8u32, 8u32]);
        // Splits not multiples of 4: the group routing is undefined there.
        let qscalar_ragged = buf_from(&c.device, &[blocks as u32, 15u32, 9u32, 8u32, 8u32]);
        let cb = c.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        // QKV rejects every k_batch except 8 ...
        for bad_k in [0usize, 1, 2, 4, 6, 7, 9, 16] {
            assert!(!encode_gemma4_q4_0_qkv_matmul_batch_k_v4(
                e, c.old, &y, &w, &w, &w, &out, &out, &out, &qscalar, 32, bad_k
            ));
        }
        // ... and non-4-aligned splits even at k_batch 8.
        assert!(!encode_gemma4_q4_0_qkv_matmul_batch_k_v4(
            e,
            c.old,
            &y,
            &w,
            &w,
            &w,
            &out,
            &out,
            &out,
            &qscalar_ragged,
            32,
            8
        ));
        assert!(encode_gemma4_q4_0_qkv_matmul_batch_k_v4(
            e, c.old, &y, &w, &w, &w, &out, &out, &out, &qscalar, 32, 8
        ));
        // plain/gateup are measured no-wins: they must always fall back.
        for k in [0usize, 1, 4, 8, 16] {
            assert!(!encode_gemma4_q4_0_matmul_batch_k_v4(
                e, c.old, &y, &w, 0, &out, rows, &scalar, k
            ));
            assert!(!encode_gemma4_q4_0_gateup_matmul_batch_k_v4(
                e, c.old, &y, &w, &w, &out, &scalar, rows, k
            ));
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// The shipped configs must build and honor their declared widths.
    #[test]
    fn spec50_v4_shipped_configs_build() {
        let Some(c) = ctx() else {
            return;
        };
        let ks =
            Spec50V4Kernels::build(&c.device, SPEC50_V4_PLAIN, SPEC50_V4_QKV, SPEC50_V4_GATEUP)
                .expect("shipped spec50 configs must build");
        assert!(ks.plain.max_total_threads_per_threadgroup() >= SPEC50_V4_PLAIN.tg_width());
        assert!(ks.qkv.max_total_threads_per_threadgroup() >= SPEC50_V4_QKV.tg_width());
        assert!(ks.gateup.max_total_threads_per_threadgroup() >= SPEC50_V4_GATEUP.tg_width());
    }

    // -- benchmarks ------------------------------------------------------------
    //
    // Real 26B chained-lane geometry (read from the encode sites and the
    // strict-26B preflight): hidden 2816 (88 blocks), 30 layers = 25 local +
    // 5 global (layers 5/11/17/23/29).
    //   local:  fused QKV q=16*256=4096, k=v=8*256=2048 rows @ 88 blocks;
    //           o-proj 2816 rows @ 128 blocks (q_dim 4096).
    //   global: v_w is None -> separate plain q (16*512=8192 rows @ 88 blocks)
    //           and plain k (2*512=1024 rows @ 88 blocks); o-proj 2816 rows
    //           @ 256 blocks (q_dim 8192).
    //   shared: gate/up 2112 rows @ 88 blocks + down 2816 rows @ 66 blocks.
    // Run with: cam-lock.sh cargo test --release --lib spec50_bench -- \
    //   --ignored --nocapture --test-threads=1

    const BENCH_LAYERS: usize = 30;
    const BENCH_GLOBALS: [usize; 5] = [5, 11, 17, 23, 29];
    const BENCH_ROUNDS: usize = 6;

    fn bench<F: Fn(&metal::ComputeCommandEncoderRef)>(queue: &CommandQueue, f: F) -> f64 {
        for _ in 0..2 {
            run(queue, &f);
        }
        let mut best = u128::MAX;
        for _ in 0..BENCH_ROUNDS {
            let (gpu_us, _) = run(queue, &f);
            best = best.min(gpu_us);
        }
        best as f64 / 1000.0
    }

    fn gbs(bytes: f64, ms: f64) -> f64 {
        bytes / (ms * 1.0e-3) / 1.0e9
    }

    struct LocalLayer {
        wq: Buffer,
        wk: Buffer,
        wv: Buffer,
        wo: Buffer,
    }
    struct GlobalLayer {
        wq: Buffer,
        wk: Buffer,
        wo: Buffer,
    }

    /// Shape A: the 30-layer qkv + o-proj sweep at the real mixed local/global
    /// geometry. One dispatch pair (or triple) per layer, 30 layers per
    /// measurement.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_bench_shape_a_qkv_o() {
        let Some(c) = ctx() else {
            return;
        };
        let hidden = 2816usize;
        let blocks_h = hidden / 32; // 88
                                    // local layers
        let l_q = 4096usize;
        let l_kv = 2048usize;
        let l_qkv_rows = l_q + 2 * l_kv; // 8192
        let l_o_blocks = l_q / 32; // 128
                                   // global layers
        let g_q = 8192usize;
        let g_kv = 1024usize;
        let g_o_blocks = g_q / 32; // 256

        let mut rng = Rng(0xA11CE);
        let mut locals = Vec::new();
        let mut globals = Vec::new();
        for layer in 0..BENCH_LAYERS {
            if BENCH_GLOBALS.contains(&layer) {
                globals.push(GlobalLayer {
                    wq: buf_from(&c.device, &random_q4_0(&mut rng, g_q, blocks_h)),
                    wk: buf_from(&c.device, &random_q4_0(&mut rng, g_kv, blocks_h)),
                    wo: buf_from(&c.device, &random_q4_0(&mut rng, hidden, g_o_blocks)),
                });
            } else {
                locals.push(LocalLayer {
                    wq: buf_from(&c.device, &random_q4_0(&mut rng, l_q, blocks_h)),
                    wk: buf_from(&c.device, &random_q4_0(&mut rng, l_kv, blocks_h)),
                    wv: buf_from(&c.device, &random_q4_0(&mut rng, l_kv, blocks_h)),
                    wo: buf_from(&c.device, &random_q4_0(&mut rng, hidden, l_o_blocks)),
                });
            }
        }
        let y_norm = buf_from(&c.device, &random_f32(&mut rng, hidden * K8));
        let y_ctx = buf_from(&c.device, &random_f32(&mut rng, g_q * K8));
        let oq = zeros(&c.device, g_q * K8);
        let ok = zeros(&c.device, l_kv * K8);
        let ov = zeros(&c.device, l_kv * K8);
        let oo = zeros(&c.device, hidden * K8);

        let weight_bytes = (25 * ((l_qkv_rows * blocks_h) + (hidden * l_o_blocks))
            + 5 * (((g_q + g_kv) * blocks_h) + (hidden * g_o_blocks)))
            as f64
            * Q4_BLOCK as f64;
        println!(
            "\n[spec50] shape A qkv_o: 25 local (qkv {l_qkv_rows}x{blocks_h}b + o {hidden}x{l_o_blocks}b) \
             + 5 global (q {g_q}x{blocks_h}b + k {g_kv}x{blocks_h}b + o {hidden}x{g_o_blocks}b), \
             {:.1} MB weights/sweep",
            weight_bytes / 1.0e6
        );

        let qs = buf_from(
            &c.device,
            &[blocks_h as u32, l_q as u32, l_kv as u32, l_kv as u32, 8u32],
        );
        let scalar_q_g = buf_from(&c.device, &[blocks_h as u32, g_q as u32, 8u32, 0u32]);
        let scalar_k_g = buf_from(&c.device, &[blocks_h as u32, g_kv as u32, 8u32, 0u32]);
        let scalar_o_l = buf_from(&c.device, &[l_o_blocks as u32, hidden as u32, 8u32, 0u32]);
        let scalar_o_g = buf_from(&c.device, &[g_o_blocks as u32, hidden as u32, 8u32, 0u32]);

        let old_sweep = |e: &metal::ComputeCommandEncoderRef, k_batch: usize| {
            let mut li = 0usize;
            let mut gi = 0usize;
            for layer in 0..BENCH_LAYERS {
                if BENCH_GLOBALS.contains(&layer) {
                    let g = &globals[gi];
                    gi += 1;
                    encode_gemma4_q4_0_matmul_batch_k(
                        e,
                        c.old,
                        &y_norm,
                        &g.wq,
                        0,
                        &oq,
                        g_q,
                        &scalar_q_g,
                        k_batch,
                    );
                    encode_gemma4_q4_0_matmul_batch_k(
                        e,
                        c.old,
                        &y_norm,
                        &g.wk,
                        0,
                        &ok,
                        g_kv,
                        &scalar_k_g,
                        k_batch,
                    );
                    encode_gemma4_q4_0_matmul_batch_k(
                        e,
                        c.old,
                        &y_ctx,
                        &g.wo,
                        0,
                        &oo,
                        hidden,
                        &scalar_o_g,
                        k_batch,
                    );
                } else {
                    let l = &locals[li];
                    li += 1;
                    encode_gemma4_q4_0_qkv_matmul_batch_k(
                        e, c.old, &y_norm, &l.wq, &l.wk, &l.wv, &oq, &ok, &ov, &qs, l_qkv_rows,
                        k_batch,
                    );
                    encode_gemma4_q4_0_matmul_batch_k(
                        e,
                        c.old,
                        &y_ctx,
                        &l.wo,
                        0,
                        &oo,
                        hidden,
                        &scalar_o_l,
                        k_batch,
                    );
                }
            }
        };

        let old_ms = bench(&c.queue, |e| old_sweep(e, 8));
        println!(
            "  OLD K=8: {old_ms:7.3} ms ({:6.1} GB/s weights)",
            gbs(weight_bytes, old_ms)
        );
        let old_k1_ms = bench(&c.queue, |e| old_sweep(e, 1));
        println!(
            "  OLD K=1 (unchanged fallback): {old_k1_ms:7.3} ms ({:6.1} GB/s weights)",
            gbs(weight_bytes, old_k1_ms)
        );

        // Shipped composite: exactly the integrator call pattern — try _v4,
        // fall back to the existing encode on false.
        let shipped_ms = bench(&c.queue, |e| {
            let mut li = 0usize;
            let mut gi = 0usize;
            for layer in 0..BENCH_LAYERS {
                if BENCH_GLOBALS.contains(&layer) {
                    let g = &globals[gi];
                    gi += 1;
                    if !encode_gemma4_q4_0_matmul_batch_k_v4(
                        e,
                        c.old,
                        &y_norm,
                        &g.wq,
                        0,
                        &oq,
                        g_q,
                        &scalar_q_g,
                        8,
                    ) {
                        encode_gemma4_q4_0_matmul_batch_k(
                            e,
                            c.old,
                            &y_norm,
                            &g.wq,
                            0,
                            &oq,
                            g_q,
                            &scalar_q_g,
                            8,
                        );
                    }
                    if !encode_gemma4_q4_0_matmul_batch_k_v4(
                        e,
                        c.old,
                        &y_norm,
                        &g.wk,
                        0,
                        &ok,
                        g_kv,
                        &scalar_k_g,
                        8,
                    ) {
                        encode_gemma4_q4_0_matmul_batch_k(
                            e,
                            c.old,
                            &y_norm,
                            &g.wk,
                            0,
                            &ok,
                            g_kv,
                            &scalar_k_g,
                            8,
                        );
                    }
                    if !encode_gemma4_q4_0_matmul_batch_k_v4(
                        e,
                        c.old,
                        &y_ctx,
                        &g.wo,
                        0,
                        &oo,
                        hidden,
                        &scalar_o_g,
                        8,
                    ) {
                        encode_gemma4_q4_0_matmul_batch_k(
                            e,
                            c.old,
                            &y_ctx,
                            &g.wo,
                            0,
                            &oo,
                            hidden,
                            &scalar_o_g,
                            8,
                        );
                    }
                } else {
                    let l = &locals[li];
                    li += 1;
                    if !encode_gemma4_q4_0_qkv_matmul_batch_k_v4(
                        e, c.old, &y_norm, &l.wq, &l.wk, &l.wv, &oq, &ok, &ov, &qs, l_qkv_rows, 8,
                    ) {
                        encode_gemma4_q4_0_qkv_matmul_batch_k(
                            e, c.old, &y_norm, &l.wq, &l.wk, &l.wv, &oq, &ok, &ov, &qs, l_qkv_rows,
                            8,
                        );
                    }
                    if !encode_gemma4_q4_0_matmul_batch_k_v4(
                        e,
                        c.old,
                        &y_ctx,
                        &l.wo,
                        0,
                        &oo,
                        hidden,
                        &scalar_o_l,
                        8,
                    ) {
                        encode_gemma4_q4_0_matmul_batch_k(
                            e,
                            c.old,
                            &y_ctx,
                            &l.wo,
                            0,
                            &oo,
                            hidden,
                            &scalar_o_l,
                            8,
                        );
                    }
                }
            }
        });
        println!(
            "  SHIPPED K=8 (_v4 + fallback): {shipped_ms:7.3} ms ({:6.1} GB/s weights)  {:.2}x vs old",
            gbs(weight_bytes, shipped_ms),
            old_ms / shipped_ms
        );

        for &cfg in &SWEEP {
            let ks = build(&c, cfg);
            let new_ms = bench(&c.queue, |e| {
                let mut li = 0usize;
                let mut gi = 0usize;
                for layer in 0..BENCH_LAYERS {
                    if BENCH_GLOBALS.contains(&layer) {
                        let g = &globals[gi];
                        gi += 1;
                        spec50_encode_plain_with(
                            &ks,
                            e,
                            &y_norm,
                            &g.wq,
                            0,
                            &oq,
                            g_q,
                            blocks_h as u32,
                        );
                        spec50_encode_plain_with(
                            &ks,
                            e,
                            &y_norm,
                            &g.wk,
                            0,
                            &ok,
                            g_kv,
                            blocks_h as u32,
                        );
                        spec50_encode_plain_with(
                            &ks,
                            e,
                            &y_ctx,
                            &g.wo,
                            0,
                            &oo,
                            hidden,
                            g_o_blocks as u32,
                        );
                    } else {
                        let l = &locals[li];
                        li += 1;
                        spec50_encode_qkv_with(
                            &ks,
                            e,
                            &y_norm,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &oq,
                            &ok,
                            &ov,
                            l_qkv_rows,
                            (blocks_h as u32, l_q as u32, l_kv as u32, l_kv as u32),
                        );
                        spec50_encode_plain_with(
                            &ks,
                            e,
                            &y_ctx,
                            &l.wo,
                            0,
                            &oo,
                            hidden,
                            l_o_blocks as u32,
                        );
                    }
                }
            });
            println!(
                "  NEW K=8 {cfg:?}: {new_ms:7.3} ms ({:6.1} GB/s weights)  {:.2}x vs old",
                gbs(weight_bytes, new_ms),
                old_ms / new_ms
            );
        }
    }

    /// Per-dispatch-kind attribution: times each of the seven distinct 26B
    /// dispatch shapes in isolation (30 identical dispatches per measurement)
    /// so the win/loss can be pinned to shapes, not stages.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_bench_per_dispatch_kind() {
        let Some(c) = ctx() else {
            return;
        };
        let configs = [
            Spec50Cfg {
                nr: 4,
                sg: 2,
                tb: 16,
            },
            Spec50Cfg {
                nr: 4,
                sg: 4,
                tb: 8,
            },
            Spec50Cfg {
                nr: 4,
                sg: 4,
                tb: 16,
            },
            Spec50Cfg {
                nr: 4,
                sg: 4,
                tb: 32,
            },
        ];
        let mut rng = Rng(0xD15A);
        let reps = 30usize;

        // (label, rows, blocks) for plain shapes
        let plain_shapes = [
            ("o_local  2816x128b", 2816usize, 128usize),
            ("o_global 2816x256b", 2816, 256),
            ("q_global 8192x88b ", 8192, 88),
            ("k_global 1024x88b ", 1024, 88),
            ("down     2816x66b ", 2816, 66),
        ];
        for (label, rows, blocks) in plain_shapes {
            let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let out = zeros(&c.device, rows * K8);
            let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, 8u32, 0u32]);
            let bytes = (reps * rows * blocks * Q4_BLOCK) as f64;
            let old_ms = bench(&c.queue, |e| {
                for _ in 0..reps {
                    encode_gemma4_q4_0_matmul_batch_k(e, c.old, &y, &w, 0, &out, rows, &scalar, 8);
                }
            });
            print!(
                "  plain {label} x{reps}: old {old_ms:7.3} ms ({:5.1} GB/s w)",
                gbs(bytes, old_ms)
            );
            for cfg in configs {
                let ks = build(&c, cfg);
                let new_ms = bench(&c.queue, |e| {
                    for _ in 0..reps {
                        spec50_encode_plain_with(&ks, e, &y, &w, 0, &out, rows, blocks as u32);
                    }
                });
                print!(
                    " | sg{}tb{} {new_ms:7.3} ({:.2}x)",
                    cfg.sg,
                    cfg.tb,
                    old_ms / new_ms
                );
            }
            println!();
        }

        // fused QKV local
        {
            let (qr, kr, vr, blocks) = (4096usize, 2048usize, 2048usize, 88usize);
            let wq = buf_from(&c.device, &random_q4_0(&mut rng, qr, blocks));
            let wk = buf_from(&c.device, &random_q4_0(&mut rng, kr, blocks));
            let wv = buf_from(&c.device, &random_q4_0(&mut rng, vr, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let oq = zeros(&c.device, qr * K8);
            let ok = zeros(&c.device, kr * K8);
            let ov = zeros(&c.device, vr * K8);
            let scalar = buf_from(
                &c.device,
                &[blocks as u32, qr as u32, kr as u32, vr as u32, 8u32],
            );
            let bytes = (reps * (qr + kr + vr) * blocks * Q4_BLOCK) as f64;
            let old_ms = bench(&c.queue, |e| {
                for _ in 0..reps {
                    encode_gemma4_q4_0_qkv_matmul_batch_k(
                        e,
                        c.old,
                        &y,
                        &wq,
                        &wk,
                        &wv,
                        &oq,
                        &ok,
                        &ov,
                        &scalar,
                        qr + kr + vr,
                        8,
                    );
                }
            });
            print!(
                "  qkv_local 8192x88b  x{reps}: old {old_ms:7.3} ms ({:5.1} GB/s w)",
                gbs(bytes, old_ms)
            );
            for cfg in configs {
                let ks = build(&c, cfg);
                let new_ms = bench(&c.queue, |e| {
                    for _ in 0..reps {
                        spec50_encode_qkv_with(
                            &ks,
                            e,
                            &y,
                            &wq,
                            &wk,
                            &wv,
                            &oq,
                            &ok,
                            &ov,
                            qr + kr + vr,
                            (blocks as u32, qr as u32, kr as u32, vr as u32),
                        );
                    }
                });
                print!(
                    " | sg{}tb{} {new_ms:7.3} ({:.2}x)",
                    cfg.sg,
                    cfg.tb,
                    old_ms / new_ms
                );
            }
            println!();
        }

        // gateup
        {
            let (rows, blocks) = (2112usize, 88usize);
            let wg = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let wu = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let out = zeros(&c.device, rows * K8);
            let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, 8u32, 0u32]);
            let bytes = (reps * 2 * rows * blocks * Q4_BLOCK) as f64;
            let old_ms = bench(&c.queue, |e| {
                for _ in 0..reps {
                    encode_gemma4_q4_0_gateup_matmul_batch_k(
                        e, c.old, &y, &wg, &wu, &out, &scalar, rows, 8,
                    );
                }
            });
            print!(
                "  gateup   2112x88b  x{reps}: old {old_ms:7.3} ms ({:5.1} GB/s w)",
                gbs(bytes, old_ms)
            );
            for cfg in configs {
                let ks = build(&c, cfg);
                let new_ms = bench(&c.queue, |e| {
                    for _ in 0..reps {
                        spec50_encode_gateup_with(&ks, e, &y, &wg, &wu, &out, rows, blocks as u32);
                    }
                });
                print!(
                    " | sg{}tb{} {new_ms:7.3} ({:.2}x)",
                    cfg.sg,
                    cfg.tb,
                    old_ms / new_ms
                );
            }
            println!();
        }
    }

    /// Shape B: the 30-layer shared-MLP sweep (gate/up + GeGLU 2112x88b, down
    /// 2816x66b).
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_bench_shape_b_shared_mlp() {
        let Some(c) = ctx() else {
            return;
        };
        let hidden = 2816usize;
        let ffn = 2112usize;
        let blocks_h = hidden / 32; // 88
        let blocks_f = ffn / 32; // 66

        let mut rng = Rng(0xB0B);
        let mut wg = Vec::new();
        let mut wu = Vec::new();
        let mut wd = Vec::new();
        for _ in 0..BENCH_LAYERS {
            wg.push(buf_from(&c.device, &random_q4_0(&mut rng, ffn, blocks_h)));
            wu.push(buf_from(&c.device, &random_q4_0(&mut rng, ffn, blocks_h)));
            wd.push(buf_from(
                &c.device,
                &random_q4_0(&mut rng, hidden, blocks_f),
            ));
        }
        let y_norm = buf_from(&c.device, &random_f32(&mut rng, hidden * K8));
        let act = zeros(&c.device, ffn * K8);
        let down = zeros(&c.device, hidden * K8);

        let weight_bytes =
            (BENCH_LAYERS * ((2 * ffn * blocks_h) + (hidden * blocks_f))) as f64 * Q4_BLOCK as f64;
        println!(
            "\n[spec50] shape B shared MLP: gate/up {ffn}x{blocks_h}b + down {hidden}x{blocks_f}b, \
             {BENCH_LAYERS} layers, {:.1} MB weights/sweep",
            weight_bytes / 1.0e6
        );

        let gs = buf_from(&c.device, &[blocks_h as u32, ffn as u32, 8u32, 0u32]);
        let ds = buf_from(&c.device, &[blocks_f as u32, hidden as u32, 8u32, 0u32]);

        let old_sweep = |e: &metal::ComputeCommandEncoderRef, k_batch: usize| {
            for l in 0..BENCH_LAYERS {
                encode_gemma4_q4_0_gateup_matmul_batch_k(
                    e, c.old, &y_norm, &wg[l], &wu[l], &act, &gs, ffn, k_batch,
                );
                encode_gemma4_q4_0_matmul_batch_k(
                    e, c.old, &act, &wd[l], 0, &down, hidden, &ds, k_batch,
                );
            }
        };

        let old_ms = bench(&c.queue, |e| old_sweep(e, 8));
        println!(
            "  OLD K=8: {old_ms:7.3} ms ({:6.1} GB/s weights)",
            gbs(weight_bytes, old_ms)
        );
        let old_k1_ms = bench(&c.queue, |e| old_sweep(e, 1));
        println!(
            "  OLD K=1 (unchanged fallback): {old_k1_ms:7.3} ms ({:6.1} GB/s weights)",
            gbs(weight_bytes, old_k1_ms)
        );

        for &cfg in &SWEEP {
            let ks = build(&c, cfg);
            let new_ms = bench(&c.queue, |e| {
                for l in 0..BENCH_LAYERS {
                    spec50_encode_gateup_with(
                        &ks,
                        e,
                        &y_norm,
                        &wg[l],
                        &wu[l],
                        &act,
                        ffn,
                        blocks_h as u32,
                    );
                    spec50_encode_plain_with(
                        &ks,
                        e,
                        &act,
                        &wd[l],
                        0,
                        &down,
                        hidden,
                        blocks_f as u32,
                    );
                }
            });
            println!(
                "  NEW K=8 {cfg:?}: {new_ms:7.3} ms ({:6.1} GB/s weights)  {:.2}x vs old",
                gbs(weight_bytes, new_ms),
                old_ms / new_ms
            );
        }
    }

    /// Exact-partition 30-layer dense sweep.  The baseline is the dynamic
    /// runtime-width Metal oracle (never the faster but partition-drifting
    /// fixed-K8 kernels).  Each candidate runs the same 25 local + 5 global
    /// QKV/O schedule plus all 30 shared-down projections using distinct
    /// per-layer weights.  GateUp is deliberately excluded because its exact
    /// packed specialization is owned and benchmarked by `spec50_packed_gateup`.
    #[test]
    #[ignore = "30-layer GPU microbenchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_exact_dense_k8_bench_30_layer_runtime_oracle() {
        let Some(c) = ctx() else {
            return;
        };
        let hidden = 2816usize;
        let blocks_h = hidden / 32;
        let shared = 2112usize;
        let blocks_shared = shared / 32;
        let local_q = 4096usize;
        let local_kv = 2048usize;
        let local_qkv = local_q + 2 * local_kv;
        let local_o_blocks = local_q / 32;
        let global_q = 8192usize;
        let global_kv = 1024usize;
        let global_o_blocks = global_q / 32;

        let mut rng = Rng(0x4558_4143_5433_304c);
        let mut locals = Vec::with_capacity(25);
        let mut globals = Vec::with_capacity(5);
        let mut shared_down = Vec::with_capacity(BENCH_LAYERS);
        for layer in 0..BENCH_LAYERS {
            if BENCH_GLOBALS.contains(&layer) {
                globals.push(GlobalLayer {
                    wq: buf_from(&c.device, &random_q4_0(&mut rng, global_q, blocks_h)),
                    wk: buf_from(&c.device, &random_q4_0(&mut rng, global_kv, blocks_h)),
                    wo: buf_from(&c.device, &random_q4_0(&mut rng, hidden, global_o_blocks)),
                });
            } else {
                locals.push(LocalLayer {
                    wq: buf_from(&c.device, &random_q4_0(&mut rng, local_q, blocks_h)),
                    wk: buf_from(&c.device, &random_q4_0(&mut rng, local_kv, blocks_h)),
                    wv: buf_from(&c.device, &random_q4_0(&mut rng, local_kv, blocks_h)),
                    wo: buf_from(&c.device, &random_q4_0(&mut rng, hidden, local_o_blocks)),
                });
            }
            shared_down.push(buf_from(
                &c.device,
                &random_q4_0(&mut rng, hidden, blocks_shared),
            ));
        }

        let y_norm = buf_from(&c.device, &random_f32(&mut rng, hidden * K8));
        let y_context = buf_from(&c.device, &random_f32(&mut rng, global_q * K8));
        let y_shared = buf_from(&c.device, &random_f32(&mut rng, shared * K8));
        let query = zeros(&c.device, global_q * K8);
        let key = zeros(&c.device, local_kv * K8);
        let value = zeros(&c.device, local_kv * K8);
        let hidden_out = zeros(&c.device, hidden * K8);

        let qkv_o_weight_bytes = (25 * ((local_qkv * blocks_h) + (hidden * local_o_blocks))
            + 5 * (((global_q + global_kv) * blocks_h) + (hidden * global_o_blocks)))
            as f64
            * Q4_BLOCK as f64;
        let shared_down_weight_bytes = (BENCH_LAYERS * hidden * blocks_shared * Q4_BLOCK) as f64;
        let weight_bytes = qkv_o_weight_bytes + shared_down_weight_bytes;

        let oracle = |e: &metal::ComputeCommandEncoderRef| {
            let mut li = 0usize;
            let mut gi = 0usize;
            for layer in 0..BENCH_LAYERS {
                if BENCH_GLOBALS.contains(&layer) {
                    let weights = &globals[gi];
                    gi += 1;
                    encode_plain_runtime_width(
                        &c,
                        e,
                        &y_norm,
                        &weights.wq,
                        &query,
                        global_q,
                        blocks_h,
                    );
                    encode_plain_runtime_width(
                        &c,
                        e,
                        &y_norm,
                        &weights.wk,
                        &key,
                        global_kv,
                        blocks_h,
                    );
                    encode_plain_runtime_width(
                        &c,
                        e,
                        &y_context,
                        &weights.wo,
                        &hidden_out,
                        hidden,
                        global_o_blocks,
                    );
                } else {
                    let weights = &locals[li];
                    li += 1;
                    encode_qkv_runtime_width(
                        &c,
                        e,
                        &y_norm,
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &query,
                        &key,
                        &value,
                        (local_q, local_kv, local_kv),
                        blocks_h,
                    );
                    encode_plain_runtime_width(
                        &c,
                        e,
                        &y_context,
                        &weights.wo,
                        &hidden_out,
                        hidden,
                        local_o_blocks,
                    );
                }
                encode_plain_runtime_width(
                    &c,
                    e,
                    &y_shared,
                    &shared_down[layer],
                    &hidden_out,
                    hidden,
                    blocks_shared,
                );
            }
        };
        let oracle_ms = bench(&c.queue, oracle);
        println!(
            "\n[spec50 exact dense] 30-layer QKV/O/shared-down, {:.1} MB weights/sweep",
            weight_bytes / 1.0e6
        );
        println!(
            "  runtime-width oracle: {oracle_ms:8.3} ms ({:6.1} GB/s weights)",
            gbs(weight_bytes, oracle_ms)
        );

        for cfg in EXACT_CANDIDATES {
            let ks = Spec50ExactDenseKernels::build(&c.device, cfg)
                .unwrap_or_else(|| panic!("exact dense config {cfg:?}"));
            let candidate_ms = bench(&c.queue, |e| {
                let mut li = 0usize;
                let mut gi = 0usize;
                for layer in 0..BENCH_LAYERS {
                    if BENCH_GLOBALS.contains(&layer) {
                        let weights = &globals[gi];
                        gi += 1;
                        spec50_encode_exact_plain_with(
                            &ks,
                            e,
                            &y_norm,
                            &weights.wq,
                            0,
                            &query,
                            global_q,
                            blocks_h as u32,
                            K8 as u32,
                        );
                        spec50_encode_exact_plain_with(
                            &ks,
                            e,
                            &y_norm,
                            &weights.wk,
                            0,
                            &key,
                            global_kv,
                            blocks_h as u32,
                            K8 as u32,
                        );
                        spec50_encode_exact_plain_with(
                            &ks,
                            e,
                            &y_context,
                            &weights.wo,
                            0,
                            &hidden_out,
                            hidden,
                            global_o_blocks as u32,
                            K8 as u32,
                        );
                    } else {
                        let weights = &locals[li];
                        li += 1;
                        spec50_encode_exact_qkv_with(
                            &ks,
                            e,
                            &y_norm,
                            &weights.wq,
                            &weights.wk,
                            &weights.wv,
                            &query,
                            &key,
                            &value,
                            local_qkv,
                            (
                                blocks_h as u32,
                                local_q as u32,
                                local_kv as u32,
                                local_kv as u32,
                            ),
                            K8 as u32,
                        );
                        spec50_encode_exact_plain_with(
                            &ks,
                            e,
                            &y_context,
                            &weights.wo,
                            0,
                            &hidden_out,
                            hidden,
                            local_o_blocks as u32,
                            K8 as u32,
                        );
                    }
                    spec50_encode_exact_plain_with(
                        &ks,
                        e,
                        &y_shared,
                        &shared_down[layer],
                        0,
                        &hidden_out,
                        hidden,
                        blocks_shared as u32,
                        K8 as u32,
                    );
                }
            });
            println!(
                "  candidate {cfg:?}: {candidate_ms:8.3} ms ({:6.1} GB/s)  {:.3}x vs oracle",
                gbs(weight_bytes, candidate_ms),
                oracle_ms / candidate_ms
            );
        }
    }

    /// Stage attribution for the exact candidates.  Thirty repeated real-shape
    /// dispatches isolate whether a QKV-only win is hidden by a plain-kernel
    /// loss in the distinct-weight whole-model sweep above.
    #[test]
    #[ignore = "GPU attribution benchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_exact_dense_k8_bench_per_dispatch_kind() {
        let Some(c) = ctx() else {
            return;
        };
        let mut rng = Rng(0x4558_4143_5453_5447);
        let reps = 30usize;
        for (label, rows, blocks) in [
            ("o_local", 2816usize, 128usize),
            ("o_global", 2816, 256),
            ("q_global", 8192, 88),
            ("k_global", 1024, 88),
            ("shared_down", 2816, 66),
        ] {
            let weight = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
            let out = zeros(&c.device, rows * K8);
            let oracle_ms = bench(&c.queue, |e| {
                for _ in 0..reps {
                    encode_plain_runtime_width(&c, e, &y, &weight, &out, rows, blocks);
                }
            });
            print!("[spec50 exact dense] plain {label:11} oracle={oracle_ms:7.3} ms");
            for cfg in EXACT_CANDIDATES {
                let ks = Spec50ExactDenseKernels::build(&c.device, cfg)
                    .unwrap_or_else(|| panic!("exact dense config {cfg:?}"));
                let candidate_ms = bench(&c.queue, |e| {
                    for _ in 0..reps {
                        spec50_encode_exact_plain_with(
                            &ks,
                            e,
                            &y,
                            &weight,
                            0,
                            &out,
                            rows,
                            blocks as u32,
                            K8 as u32,
                        );
                    }
                });
                print!(
                    " | sg{}tb{} {:.3}x",
                    cfg.sg,
                    cfg.tb,
                    oracle_ms / candidate_ms
                );
            }
            println!();
        }

        let (q_rows, k_rows, v_rows, blocks) = (4096usize, 2048usize, 2048usize, 88usize);
        let q_weight = buf_from(&c.device, &random_q4_0(&mut rng, q_rows, blocks));
        let k_weight = buf_from(&c.device, &random_q4_0(&mut rng, k_rows, blocks));
        let v_weight = buf_from(&c.device, &random_q4_0(&mut rng, v_rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * K8));
        let query = zeros(&c.device, q_rows * K8);
        let key = zeros(&c.device, k_rows * K8);
        let value = zeros(&c.device, v_rows * K8);
        let oracle_ms = bench(&c.queue, |e| {
            for _ in 0..reps {
                encode_qkv_runtime_width(
                    &c,
                    e,
                    &y,
                    &q_weight,
                    &k_weight,
                    &v_weight,
                    &query,
                    &key,
                    &value,
                    (q_rows, k_rows, v_rows),
                    blocks,
                );
            }
        });
        print!("[spec50 exact dense] qkv local   oracle={oracle_ms:7.3} ms");
        for cfg in EXACT_CANDIDATES {
            let ks = Spec50ExactDenseKernels::build(&c.device, cfg)
                .unwrap_or_else(|| panic!("exact dense config {cfg:?}"));
            let candidate_ms = bench(&c.queue, |e| {
                for _ in 0..reps {
                    spec50_encode_exact_qkv_with(
                        &ks,
                        e,
                        &y,
                        &q_weight,
                        &k_weight,
                        &v_weight,
                        &query,
                        &key,
                        &value,
                        q_rows + k_rows + v_rows,
                        (blocks as u32, q_rows as u32, k_rows as u32, v_rows as u32),
                        K8 as u32,
                    );
                }
            });
            print!(
                " | sg{}tb{} {:.3}x",
                cfg.sg,
                cfg.tb,
                oracle_ms / candidate_ms
            );
        }
        println!();
    }

    /// Real-shape cost of making K=16 a scheduling decision (two canonical K8
    /// dispatches) rather than selecting the widened arithmetic flavor.  This
    /// is deliberately a hot single-layer stage breakdown.  The scaled totals
    /// below are directional estimates, not a substitute for a distinct-weight
    /// full sweep: cache residency and command-buffer amortization differ.
    #[test]
    #[ignore = "GPU microbenchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_bench_k16_two_k8_tiles_vs_widened() {
        let Some(c) = ctx() else {
            return;
        };
        let dense = build(&c, SPEC50_V4_QKV);
        let widen = spec50_widen_kernels().expect("widened K16 pipelines");
        let packed = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        let packed_direct = spec50_packed_gateup_direct_k16_kernel()
            .expect("experimental direct two-SG K16 pipeline");
        const TILE_K: usize = 8;
        const WAVE_K: usize = 16;
        let mut rng = Rng(0x4b31_3654_494c_4538);
        let mut qkv_o_model_tile_ms = 0.0f64;
        let mut qkv_o_model_wide_ms = 0.0f64;
        let mut shared_model_tile_ms = 0.0f64;
        let mut shared_model_wide_ms = 0.0f64;
        let mut shared_model_direct_ms = 0.0f64;
        let head_tile_ms;
        let head_wide_ms;

        println!("\n[tile8-width] real-shape K16 GPU ms: two fixed K8 tiles vs widened K16");
        println!("  stage              tile8x2    widened     ratio   raw mismatches");

        // Local fused QKV: Q/K/V = 4096/2048/2048, hidden = 2816.
        {
            let (q_rows, k_rows, v_rows, blocks) = (4096usize, 2048usize, 2048usize, 88usize);
            let hidden = blocks * 32;
            let total_rows = q_rows + k_rows + v_rows;
            let y = buf_from(&c.device, &random_f32(&mut rng, WAVE_K * hidden));
            let qw = buf_from(&c.device, &random_q4_0(&mut rng, q_rows, blocks));
            let kw = buf_from(&c.device, &random_q4_0(&mut rng, k_rows, blocks));
            let vw = buf_from(&c.device, &random_q4_0(&mut rng, v_rows, blocks));
            let tile_q = zeros(&c.device, WAVE_K * q_rows);
            let tile_k = zeros(&c.device, WAVE_K * k_rows);
            let tile_v = zeros(&c.device, WAVE_K * v_rows);
            let wide_q = zeros(&c.device, WAVE_K * q_rows);
            let wide_k = zeros(&c.device, WAVE_K * k_rows);
            let wide_v = zeros(&c.device, WAVE_K * v_rows);
            let scalars = (blocks as u32, q_rows as u32, k_rows as u32, v_rows as u32);
            let encode_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    spec50_encode_qkv_at_offsets(
                        &dense,
                        encoder,
                        &y,
                        &qw,
                        &kw,
                        &vw,
                        &tile_q,
                        &tile_k,
                        &tile_v,
                        total_rows,
                        scalars,
                        Spec50QkvBufferOffsets {
                            y: (tile * TILE_K * hidden * 4) as u64,
                            query_out: (tile * TILE_K * q_rows * 4) as u64,
                            key_out: (tile * TILE_K * k_rows * 4) as u64,
                            val_out: (tile * TILE_K * v_rows * 4) as u64,
                        },
                    );
                }
            };
            let encode_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_qkv(
                    encoder, widen, &y, &qw, &kw, &vw, &wide_q, &wide_k, &wide_v, scalars,
                    total_rows, WAVE_K,
                );
            };
            let tiled_ms = bench(&c.queue, encode_tiles);
            let wide_ms = bench(&c.queue, encode_wide);
            qkv_o_model_tile_ms += 25.0 * tiled_ms;
            qkv_o_model_wide_ms += 25.0 * wide_ms;
            run(&c.queue, encode_tiles);
            run(&c.queue, encode_wide);
            let bad = bits_equal(
                "bench QKV.Q",
                &read(&tile_q, WAVE_K * q_rows),
                &read(&wide_q, WAVE_K * q_rows),
            ) + bits_equal(
                "bench QKV.K",
                &read(&tile_k, WAVE_K * k_rows),
                &read(&wide_k, WAVE_K * k_rows),
            ) + bits_equal(
                "bench QKV.V",
                &read(&tile_v, WAVE_K * v_rows),
                &read(&wide_v, WAVE_K * v_rows),
            );
            println!(
                "  local QKV         {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad:16}",
                tiled_ms / wide_ms,
            );
        }

        // Local O projection: 2816 rows over 4096 input columns.
        {
            let (rows, blocks) = (2816usize, 128usize);
            let hidden = blocks * 32;
            let y = buf_from(&c.device, &random_f32(&mut rng, WAVE_K * hidden));
            let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let tiled = zeros(&c.device, WAVE_K * rows);
            let wide = zeros(&c.device, WAVE_K * rows);
            let encode_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    encode_shipped_plain_k8_at_offsets(
                        &c,
                        encoder,
                        &y,
                        (tile * TILE_K * hidden * 4) as u64,
                        &w,
                        &tiled,
                        (tile * TILE_K * rows * 4) as u64,
                        rows,
                        blocks,
                    );
                }
            };
            let encode_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_plain(
                    encoder,
                    widen,
                    &y,
                    &w,
                    0,
                    &wide,
                    blocks as u32,
                    rows,
                    WAVE_K,
                );
            };
            let tiled_ms = bench(&c.queue, encode_tiles);
            let wide_ms = bench(&c.queue, encode_wide);
            qkv_o_model_tile_ms += 25.0 * tiled_ms;
            qkv_o_model_wide_ms += 25.0 * wide_ms;
            run(&c.queue, encode_tiles);
            run(&c.queue, encode_wide);
            let bad = bits_equal(
                "bench O",
                &read(&tiled, WAVE_K * rows),
                &read(&wide, WAVE_K * rows),
            );
            println!(
                "  local O           {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad:16}",
                tiled_ms / wide_ms,
            );
        }

        // Global layers have no separate V tensor: Q and K are two plain
        // projections at 8192/1024 rows, followed by the 8192-wide O input.
        {
            let (hidden, q_rows, k_rows, blocks) = (2816usize, 8192usize, 1024usize, 88usize);
            let y = buf_from(&c.device, &random_f32(&mut rng, WAVE_K * hidden));
            let qw = buf_from(&c.device, &random_q4_0(&mut rng, q_rows, blocks));
            let kw = buf_from(&c.device, &random_q4_0(&mut rng, k_rows, blocks));
            let tiled_q = zeros(&c.device, WAVE_K * q_rows);
            let tiled_k = zeros(&c.device, WAVE_K * k_rows);
            let wide_q = zeros(&c.device, WAVE_K * q_rows);
            let wide_k = zeros(&c.device, WAVE_K * k_rows);
            let encode_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    encode_shipped_plain_k8_at_offsets(
                        &c,
                        encoder,
                        &y,
                        (tile * TILE_K * hidden * 4) as u64,
                        &qw,
                        &tiled_q,
                        (tile * TILE_K * q_rows * 4) as u64,
                        q_rows,
                        blocks,
                    );
                    encode_shipped_plain_k8_at_offsets(
                        &c,
                        encoder,
                        &y,
                        (tile * TILE_K * hidden * 4) as u64,
                        &kw,
                        &tiled_k,
                        (tile * TILE_K * k_rows * 4) as u64,
                        k_rows,
                        blocks,
                    );
                }
            };
            let encode_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_plain(
                    encoder,
                    widen,
                    &y,
                    &qw,
                    0,
                    &wide_q,
                    blocks as u32,
                    q_rows,
                    WAVE_K,
                );
                encode_spec50_widen_plain(
                    encoder,
                    widen,
                    &y,
                    &kw,
                    0,
                    &wide_k,
                    blocks as u32,
                    k_rows,
                    WAVE_K,
                );
            };
            let tiled_ms = bench(&c.queue, encode_tiles);
            let wide_ms = bench(&c.queue, encode_wide);
            qkv_o_model_tile_ms += 5.0 * tiled_ms;
            qkv_o_model_wide_ms += 5.0 * wide_ms;
            run(&c.queue, encode_tiles);
            run(&c.queue, encode_wide);
            let bad = bits_equal(
                "bench global Q",
                &read(&tiled_q, WAVE_K * q_rows),
                &read(&wide_q, WAVE_K * q_rows),
            ) + bits_equal(
                "bench global K",
                &read(&tiled_k, WAVE_K * k_rows),
                &read(&wide_k, WAVE_K * k_rows),
            );
            println!(
                "  global Q+K        {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad:16}",
                tiled_ms / wide_ms,
            );
        }

        {
            let (rows, blocks) = (2816usize, 256usize);
            let hidden = blocks * 32;
            let y = buf_from(&c.device, &random_f32(&mut rng, WAVE_K * hidden));
            let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let tiled = zeros(&c.device, WAVE_K * rows);
            let wide = zeros(&c.device, WAVE_K * rows);
            let encode_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    encode_shipped_plain_k8_at_offsets(
                        &c,
                        encoder,
                        &y,
                        (tile * TILE_K * hidden * 4) as u64,
                        &w,
                        &tiled,
                        (tile * TILE_K * rows * 4) as u64,
                        rows,
                        blocks,
                    );
                }
            };
            let encode_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_plain(
                    encoder,
                    widen,
                    &y,
                    &w,
                    0,
                    &wide,
                    blocks as u32,
                    rows,
                    WAVE_K,
                );
            };
            let tiled_ms = bench(&c.queue, encode_tiles);
            let wide_ms = bench(&c.queue, encode_wide);
            qkv_o_model_tile_ms += 5.0 * tiled_ms;
            qkv_o_model_wide_ms += 5.0 * wide_ms;
            run(&c.queue, encode_tiles);
            run(&c.queue, encode_wide);
            let bad = bits_equal(
                "bench global O",
                &read(&tiled, WAVE_K * rows),
                &read(&wide, WAVE_K * rows),
            );
            println!(
                "  global O          {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad:16}",
                tiled_ms / wide_ms,
            );
        }

        // Shared GateUp and Down, kept separate to expose which half benefits
        // from cross-tile weight reuse.
        {
            let (hidden, ffn) = (2816usize, 2112usize);
            let (gate_blocks, down_blocks) = (hidden / 32, ffn / 32);
            let y = buf_from(&c.device, &random_f32(&mut rng, WAVE_K * hidden));
            let gate = buf_from(&c.device, &random_q4_0(&mut rng, ffn, gate_blocks));
            let up = buf_from(&c.device, &random_q4_0(&mut rng, ffn, gate_blocks));
            let tile_act = zeros(&c.device, WAVE_K * ffn);
            let wide_act = zeros(&c.device, WAVE_K * ffn);
            let direct_act = zeros(&c.device, WAVE_K * ffn);
            let encode_gate_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    encode_spec50_packed_gateup_row_complete_at_offsets(
                        encoder,
                        packed,
                        &y,
                        &gate,
                        &up,
                        &tile_act,
                        gate_blocks as u32,
                        ffn,
                        TILE_K,
                        Spec50PackedGateupBufferOffsets {
                            y: (tile * TILE_K * hidden * 4) as u64,
                            act_output: (tile * TILE_K * ffn * 4) as u64,
                        },
                    );
                }
            };
            let encode_gate_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_gateup(
                    encoder,
                    widen,
                    &y,
                    &gate,
                    &up,
                    &wide_act,
                    gate_blocks as u32,
                    ffn,
                    WAVE_K,
                );
            };
            let encode_gate_direct = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_packed_gateup_k16_direct_two_sg(
                    encoder,
                    packed_direct,
                    &y,
                    &gate,
                    &up,
                    &direct_act,
                    gate_blocks as u32,
                    ffn,
                    Spec50PackedGateupBufferOffsets::default(),
                );
            };
            let tiled_ms = bench(&c.queue, encode_gate_tiles);
            let wide_ms = bench(&c.queue, encode_gate_wide);
            let direct_ms = bench(&c.queue, encode_gate_direct);
            shared_model_tile_ms += 30.0 * tiled_ms;
            shared_model_wide_ms += 30.0 * wide_ms;
            shared_model_direct_ms += 30.0 * direct_ms;
            run(&c.queue, encode_gate_tiles);
            run(&c.queue, encode_gate_wide);
            run(&c.queue, encode_gate_direct);
            let bad_wide = bits_equal(
                "bench shared GateUp widened",
                &read(&tile_act, WAVE_K * ffn),
                &read(&wide_act, WAVE_K * ffn),
            );
            let bad_direct = bits_equal(
                "bench shared GateUp direct two-SG",
                &read(&tile_act, WAVE_K * ffn),
                &read(&direct_act, WAVE_K * ffn),
            );
            println!(
                "  shared GateUp     {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad_wide:16}",
                tiled_ms / wide_ms,
            );
            println!(
                "    direct/tile     {direct_ms:9.3} {:10} {:9.3} {bad_direct:16}",
                "--",
                direct_ms / tiled_ms,
            );

            let down_w = buf_from(&c.device, &random_q4_0(&mut rng, hidden, down_blocks));
            let tile_down = zeros(&c.device, WAVE_K * hidden);
            let wide_down = zeros(&c.device, WAVE_K * hidden);
            let encode_down_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    encode_shipped_plain_k8_at_offsets(
                        &c,
                        encoder,
                        &tile_act,
                        (tile * TILE_K * ffn * 4) as u64,
                        &down_w,
                        &tile_down,
                        (tile * TILE_K * hidden * 4) as u64,
                        hidden,
                        down_blocks,
                    );
                }
            };
            let encode_down_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                encode_spec50_widen_plain(
                    encoder,
                    widen,
                    &tile_act,
                    &down_w,
                    0,
                    &wide_down,
                    down_blocks as u32,
                    hidden,
                    WAVE_K,
                );
            };
            let tiled_down_ms = bench(&c.queue, encode_down_tiles);
            let wide_down_ms = bench(&c.queue, encode_down_wide);
            shared_model_tile_ms += 30.0 * tiled_down_ms;
            shared_model_wide_ms += 30.0 * wide_down_ms;
            shared_model_direct_ms += 30.0 * tiled_down_ms;
            run(&c.queue, encode_down_tiles);
            run(&c.queue, encode_down_wide);
            let bad = bits_equal(
                "bench shared Down",
                &read(&tile_down, WAVE_K * hidden),
                &read(&wide_down, WAVE_K * hidden),
            );
            println!(
                "  shared Down       {tiled_down_ms:9.3} {wide_down_ms:10.3} {:9.3} {bad:16}",
                tiled_down_ms / wide_down_ms,
            );
        }

        // Full 26B tied head: 262,144 x 2,816 Q6_K (605.6 MB decimal).
        // Fill the mapped Metal buffer directly so the benchmark never holds a
        // second 605 MB host staging allocation.
        {
            const HEAD_ROWS: usize = 262_144;
            const HEAD_HIDDEN: usize = 2816;
            const HEAD_N_SB: usize = HEAD_HIDDEN / 256;
            const Q6K_BYTES: usize = 210;
            const SOFTCAP: f32 = 30.0;
            let head = spec50_head_kernels().expect("spec50 head pipelines");
            let weight_bytes = HEAD_ROWS * HEAD_N_SB * Q6K_BYTES;
            let weight = c
                .device
                .new_buffer(weight_bytes as u64, MTLResourceOptions::StorageModeShared);
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(weight.contents().cast::<u8>(), weight_bytes)
            };
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = index.wrapping_mul(37).wrapping_add(index >> 9) as u8;
            }
            for block in bytes.chunks_exact_mut(Q6K_BYTES) {
                // Small finite normal f16 super-scale.
                block[208] = 0x1f;
                block[209] = 0x21;
            }

            let scales: Vec<f32> = (0..WAVE_K * HEAD_N_SB)
                .map(|index| 0.002 + index as f32 * 1.0e-6)
                .collect();
            let quants: Vec<i8> = (0..WAVE_K * HEAD_HIDDEN)
                .map(|index| index.wrapping_mul(29).wrapping_add(7) as u8 as i8)
                .collect();
            let scales = buf_from(&c.device, &scales);
            let quants = buf_from(&c.device, &quants);
            let perm = c.device.new_buffer(
                spec50_activation_scratch_bytes(WAVE_K, HEAD_HIDDEN) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let tiled = zeros(&c.device, WAVE_K * HEAD_ROWS);
            let wide = zeros(&c.device, WAVE_K * HEAD_ROWS);
            let ids = c
                .device
                .new_buffer((WAVE_K * 4) as u64, MTLResourceOptions::StorageModeShared);
            let vals = c
                .device
                .new_buffer((WAVE_K * 4) as u64, MTLResourceOptions::StorageModeShared);
            let tile_perm_bytes = spec50_activation_scratch_bytes(TILE_K, HEAD_HIDDEN);
            let encode_tiles = |encoder: &metal::ComputeCommandEncoderRef| {
                for tile in 0..2 {
                    assert!(encode_q6k_spec50_batch_at_offsets(
                        encoder,
                        head,
                        &scales,
                        &quants,
                        &perm,
                        &weight,
                        0,
                        &tiled,
                        HEAD_N_SB,
                        HEAD_ROWS,
                        TILE_K,
                        HEAD_HIDDEN,
                        SOFTCAP,
                        Spec50HeadBufferOffsets {
                            input_scales: (tile * TILE_K * HEAD_N_SB * 4) as u64,
                            input_quants: (tile * TILE_K * HEAD_HIDDEN) as u64,
                            activation_perm: (tile * tile_perm_bytes) as u64,
                            output: (tile * TILE_K * HEAD_ROWS * 4) as u64,
                        },
                    ));
                }
                encode_q6k_spec50_argmax(encoder, head, &tiled, &ids, &vals, HEAD_ROWS, WAVE_K);
            };
            let encode_wide = |encoder: &metal::ComputeCommandEncoderRef| {
                assert!(encode_q6k_spec50_batch(
                    encoder,
                    head,
                    &scales,
                    &quants,
                    &perm,
                    &weight,
                    0,
                    &wide,
                    HEAD_N_SB,
                    HEAD_ROWS,
                    WAVE_K,
                    HEAD_HIDDEN,
                    SOFTCAP,
                ));
                encode_q6k_spec50_argmax(encoder, head, &wide, &ids, &vals, HEAD_ROWS, WAVE_K);
            };
            let tiled_ms = bench(&c.queue, encode_tiles);
            let wide_ms = bench(&c.queue, encode_wide);
            head_tile_ms = tiled_ms;
            head_wide_ms = wide_ms;
            run(&c.queue, encode_tiles);
            run(&c.queue, encode_wide);
            let bad = bits_equal(
                "bench tied head",
                &read(&tiled, WAVE_K * HEAD_ROWS),
                &read(&wide, WAVE_K * HEAD_ROWS),
            );
            println!(
                "  tied head         {tiled_ms:9.3} {wide_ms:10.3} {:9.3} {bad:16}",
                tiled_ms / wide_ms,
            );
            assert_eq!(bad, 0, "head widened K16 must retain canonical K8 bits");
        }

        println!(
            "\n  hot single-layer-scaled estimates (not a distinct-weight sweep; 25 local + 5 global):"
        );
        println!(
            "    QKV+O       tile8x2={qkv_o_model_tile_ms:8.3} ms widened={qkv_o_model_wide_ms:8.3} ms ({:.3}x)",
            qkv_o_model_tile_ms / qkv_o_model_wide_ms,
        );
        println!(
            "    shared MLP  tile8x2={shared_model_tile_ms:8.3} ms widened={shared_model_wide_ms:8.3} ms direct={shared_model_direct_ms:8.3} ms",
        );
        println!(
            "    tied head   tile8x2={head_tile_ms:8.3} ms widened={head_wide_ms:8.3} ms ({:.3}x)",
            head_tile_ms / head_wide_ms,
        );

        let [k8, direct_sg] = spec50_packed_gateup_pipeline_limits(packed, packed_direct);
        println!(
            "  GateUp limits: K8 max/TG={} SIMD={} tmem={} B; direct-K16 max/TG={} SIMD={} tmem={} B",
            k8.0, k8.1, k8.2,
            direct_sg.0, direct_sg.1, direct_sg.2,
        );
    }
}
