//! Speculative (K<=8) batch kernels for the Q6_K tied output head.
//!
//! The chained speculative round projects K candidate hidden states through the
//! 605.5 MB Q6_K `token_embd` table. The kernels this module replaces
//! (`q6k_linear_turbo_batch_k`, `q6k_linear_turbo_batch_k8`) reached only
//! 13.6 GB/s at K=8 against a 605.5 MB / ~90 GB/s = 6.7 ms byte floor. Two
//! separate defects, both measured rather than assumed:
//!
//!   1. **Quarter-rate int32 multiply.** They spend 64 int32 multiplies per
//!      weight-unit per candidate. Moving the inner product to full-rate f32 FMA
//!      is exact here, not approximate:
//!      * a weight is `c in [-32, 31]` and an activation is `y` with
//!        `|y| <= 128`, so `|c * y| <= 4096` -- every product is an integer f32
//!        represents exactly;
//!      * a group partial `A = sum_{l<16} c_l * y_l` has `|A| <= 65536`, and
//!        every prefix of it is an integer below `2^24`, so each `fma` rounds to
//!        the exact integer;
//!      * the Q6_K sub-scale is `s` with `|s| <= 128`, so `|s * A| <= 2^23`,
//!        again an exactly representable integer, and `int(s * A)` truncates a
//!        value that is already integral;
//!      * the four group terms are summed in **int32**, so the
//!        superblock-quarter `isum` is bit-for-bit the integer the reference
//!        kernels compute.
//!
//!   2. **Divergent activation fetch.** This turned out to be the larger of the
//!      two: ablating the activation load alone took K=8 from 35 ms to 9 ms. The
//!      32 lanes of a simdgroup work on 32 different superblock quarters, so in
//!      the natural Q8_K layout one vector load reaches 32 separate cache lines.
//!      `q6k_spec50_expand_f16` repacks the activations (exactly -- int8 to f16
//!      is lossless) so that consecutive lanes read consecutive spans, and the
//!      hot loop fetches one candidate-group with four `half4` loads.
//!
//! On top of that the lane mapping is flattened over (row group, unit): a row is
//! `n_sb * 4` = 44 work items, which over 32 lanes is two rounds at 68.75%
//! occupancy, and walking a flat index over 8 rows x 44 units fills 100% of the
//! lane slots instead.
//!
//! Everything that could perturb the result is held fixed: the same per-lane
//! unit partition (restored with one `simd_shuffle` where the flat walk rotates
//! it), the same `simd_sum` fold, the same
//! `(weight_scale * in_scale) * float(isum)` accumulate written character for
//! character, the same softcap, and the same `set_fast_math_enabled(false)`
//! compile options as `STRICT_Q8K_SHADER`, which owns the kernels replaced here.
//! `spec50_batch_is_bitwise_identical_to_reference` gates all of it.
#![allow(dead_code)]
use super::*;

/// Verbatim copies of the kernels this module replaces, used only by the
/// in-file exactness/benchmark harness so old and new can be driven from one
/// process without touching the production dispatch path. `spec50_reference_
/// copies_are_verbatim` asserts these are byte-identical to `src/metal.rs`.
#[cfg(test)]
const SPEC50_REFERENCE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q6k_linear_turbo(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const uchar* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n_sb [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (group * 4u + sgitg) * 4u;
    if (row0 >= rows) return;
    const uint batch = min(4u, rows - row0);
    const uint units = n_sb * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint u = lane; u < units; u += 32u) {
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;
        device const char* y = input_quants + sb * 256 + h * 128;
        const float in_scale = input_scales[sb];

        for (uint r = 0; r < batch; ++r) {
            device const uchar* block =
                weight_blocks + (ulong(row0 + r) * n_sb + sb) * 210ul;
            device const char* scales =
                reinterpret_cast<device const char*>(block + 192);
            const int s0 = int(scales[8 * h + s]);
            const int s1 = int(scales[8 * h + s + 2]);
            const int s2 = int(scales[8 * h + s + 4]);
            const int s3 = int(scales[8 * h + s + 6]);
            device const uchar* ql = block + h * 64;
            device const uchar* qh = block + 128 + h * 32;
            int isum = 0;
            for (uint l = 0; l < 16; ++l) {
                const uint j = s * 16 + l;
                const int qla = int(ql[j]);
                const int qlb = int(ql[32 + j]);
                const int qhv = int(qh[j]);
                const int c0 = ((qla & 0x0f) | ((qhv & 3) << 4)) - 32;
                const int c1 = ((qlb & 0x0f) | (((qhv >> 2) & 3) << 4)) - 32;
                const int c2 = ((qla >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
                const int c3 = ((qlb >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
                isum += s0 * int(y[j]) * c0;
                isum += s1 * int(y[j + 32]) * c1;
                isum += s2 * int(y[j + 64]) * c2;
                isum += s3 * int(y[j + 96]) * c3;
            }
            const float weight_scale =
                float(*reinterpret_cast<device const half*>(block + 208));
            const float term = (weight_scale * in_scale) * float(isum);
            if (r == 0) acc0 += term;
            else if (r == 1) acc1 += term;
            else if (r == 2) acc2 += term;
            else acc3 += term;
        }
    }
    const float s0 = simd_sum(acc0);
    const float s1 = simd_sum(acc1);
    const float s2 = simd_sum(acc2);
    const float s3 = simd_sum(acc3);
    if (lane == 0) {
        output[row0] = s0;
        if (batch > 1) output[row0 + 1] = s1;
        if (batch > 2) output[row0 + 2] = s2;
        if (batch > 3) output[row0 + 3] = s3;
    }
}

kernel void q6k_linear_turbo_batch_k(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const uchar* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n_sb [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    constant float& softcap [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (group * 4u + sgitg) * 4u;
    if (row0 >= rows) return;
    if (k_batch == 0u || k_batch > 8u) return;
    const uint batch = min(4u, rows - row0);
    const uint units = n_sb * 4;
    const uint hidden = n_sb * 256;

    float acc[4][8];
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        #pragma unroll
        for (uint t = 0; t < 8; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (uint u = lane; u < units; u += 32u) {
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;

        for (uint r = 0; r < batch; ++r) {
            device const uchar* block =
                weight_blocks + (ulong(row0 + r) * n_sb + sb) * 210ul;
            device const char* wscales =
                reinterpret_cast<device const char*>(block + 192);
            const int s0 = int(wscales[8 * h + s]);
            const int s1 = int(wscales[8 * h + s + 2]);
            const int s2 = int(wscales[8 * h + s + 4]);
            const int s3 = int(wscales[8 * h + s + 6]);
            device const uchar* ql = block + h * 64;
            device const uchar* qh = block + 128 + h * 32;
            const float weight_scale =
                float(*reinterpret_cast<device const half*>(block + 208));

            int4 w0_vec[4], w1_vec[4], w2_vec[4], w3_vec[4];
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                int4 v0, v1, v2, v3;
                #pragma unroll
                for (uint m = 0; m < 4; ++m) {
                    const uint l = k * 4 + m;
                    const uint j = s * 16 + l;
                    const int qla = int(ql[j]);
                    const int qlb = int(ql[32 + j]);
                    const int qhv = int(qh[j]);
                    const int c0 = ((qla & 0x0f) | ((qhv & 3) << 4)) - 32;
                    const int c1 = ((qlb & 0x0f) | (((qhv >> 2) & 3) << 4)) - 32;
                    const int c2 = ((qla >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
                    const int c3 = ((qlb >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
                    v0[m] = s0 * c0;
                    v1[m] = s1 * c1;
                    v2[m] = s2 * c2;
                    v3[m] = s3 * c3;
                }
                w0_vec[k] = v0;
                w1_vec[k] = v1;
                w2_vec[k] = v2;
                w3_vec[k] = v3;
            }

            for (uint t = 0; t < k_batch; ++t) {
                device const char* y_base =
                    input_quants + ulong(t) * hidden + sb * 256 + h * 128 + s * 16;
                device const char4* y0_ptr = reinterpret_cast<device const char4*>(y_base);
                device const char4* y1_ptr = reinterpret_cast<device const char4*>(y_base + 32);
                device const char4* y2_ptr = reinterpret_cast<device const char4*>(y_base + 64);
                device const char4* y3_ptr = reinterpret_cast<device const char4*>(y_base + 96);
                const float in_scale = input_scales[t * n_sb + sb];

                int4 sum = int4(0);
                #pragma unroll
                for (uint k = 0; k < 4; ++k) {
                    sum += w0_vec[k] * int4(y0_ptr[k])
                         + w1_vec[k] * int4(y1_ptr[k])
                         + w2_vec[k] * int4(y2_ptr[k])
                         + w3_vec[k] * int4(y3_ptr[k]);
                }
                const int isum = (sum.x + sum.y) + (sum.z + sum.w);
                acc[r][t] = acc[r][t] + (weight_scale * in_scale) * float(isum);
            }
        }
    }

    for (uint t = 0; t < k_batch; ++t) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            if (r >= batch) break;
            float s_val = simd_sum(acc[r][t]);
            if (lane == 0) {
                if (softcap > 0.0f) {
                    s_val = tanh(s_val / softcap) * softcap;
                }
                output[ulong(t) * rows + row0 + r] = s_val;
            }
        }
    }
}

kernel void q6k_linear_turbo_batch_k8(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const uchar* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n_sb [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    constant float& softcap [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)k_batch;
    constexpr uint KB = 8;
    const uint r = group * 4u + sgitg;
    if (r >= rows) return;
    const uint units = n_sb * 4;
    const uint hidden = n_sb * 256;

    float acc[KB];
    #pragma unroll
    for (uint t = 0; t < KB; ++t) {
        acc[t] = 0.0f;
    }

    for (uint u = lane; u < units; u += 32u) {
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;

        device const uchar* block =
            weight_blocks + (ulong(r) * n_sb + sb) * 210ul;
        device const char* wscales =
            reinterpret_cast<device const char*>(block + 192);
        const int s0 = int(wscales[8 * h + s]);
        const int s1 = int(wscales[8 * h + s + 2]);
        const int s2 = int(wscales[8 * h + s + 4]);
        const int s3 = int(wscales[8 * h + s + 6]);
        device const uchar* ql = block + h * 64;
        device const uchar* qh = block + 128 + h * 32;
        const float weight_scale =
            float(*reinterpret_cast<device const half*>(block + 208));

        int4 w0_vec[4], w1_vec[4], w2_vec[4], w3_vec[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            int4 v0, v1, v2, v3;
            #pragma unroll
            for (uint m = 0; m < 4; ++m) {
                const uint l = k * 4 + m;
                const uint j = s * 16 + l;
                const int qla = int(ql[j]);
                const int qlb = int(ql[32 + j]);
                const int qhv = int(qh[j]);
                const int c0 = ((qla & 0x0f) | ((qhv & 3) << 4)) - 32;
                const int c1 = ((qlb & 0x0f) | (((qhv >> 2) & 3) << 4)) - 32;
                const int c2 = ((qla >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
                const int c3 = ((qlb >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
                v0[m] = s0 * c0;
                v1[m] = s1 * c1;
                v2[m] = s2 * c2;
                v3[m] = s3 * c3;
            }
            w0_vec[k] = v0;
            w1_vec[k] = v1;
            w2_vec[k] = v2;
            w3_vec[k] = v3;
        }

        #pragma unroll
        for (uint t = 0; t < KB; ++t) {
            device const char* y_base =
                input_quants + ulong(t) * hidden + sb * 256 + h * 128 + s * 16;
            device const char4* y0_ptr = reinterpret_cast<device const char4*>(y_base);
            device const char4* y1_ptr = reinterpret_cast<device const char4*>(y_base + 32);
            device const char4* y2_ptr = reinterpret_cast<device const char4*>(y_base + 64);
            device const char4* y3_ptr = reinterpret_cast<device const char4*>(y_base + 96);
            const float in_scale = input_scales[t * n_sb + sb];

            int4 sum = int4(0);
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                sum += w0_vec[k] * int4(y0_ptr[k])
                     + w1_vec[k] * int4(y1_ptr[k])
                     + w2_vec[k] * int4(y2_ptr[k])
                     + w3_vec[k] * int4(y3_ptr[k]);
            }
            const int isum = (sum.x + sum.y) + (sum.z + sum.w);
            acc[t] += (weight_scale * in_scale) * float(isum);
        }
    }

    #pragma unroll
    for (uint t = 0; t < KB; ++t) {
        float s_val = simd_sum(acc[t]);
        if (lane == 0) {
            if (softcap > 0.0f) {
                s_val = tanh(s_val / softcap) * softcap;
            }
            output[ulong(t) * rows + r] = s_val;
        }
    }
}
"#;

/// Probe (NOT a verbatim copy): `q6k_linear_turbo` with the accumulator update
/// written in `q6k_linear_turbo_batch_k`'s form, i.e. the product inlined into
/// the add so the compiler may contract it to a single `fma`. Used only to
/// identify why the shipped single-token and batch heads disagree at K=1.
#[cfg(test)]
const SPEC50_PROBE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q6k_probe_turbo_inline_acc(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const uchar* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n_sb [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (group * 4u + sgitg) * 4u;
    if (row0 >= rows) return;
    const uint batch = min(4u, rows - row0);
    const uint units = n_sb * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint u = lane; u < units; u += 32u) {
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;
        device const char* y = input_quants + sb * 256 + h * 128;
        const float in_scale = input_scales[sb];

        for (uint r = 0; r < batch; ++r) {
            device const uchar* block =
                weight_blocks + (ulong(row0 + r) * n_sb + sb) * 210ul;
            device const char* scales =
                reinterpret_cast<device const char*>(block + 192);
            const int s0 = int(scales[8 * h + s]);
            const int s1 = int(scales[8 * h + s + 2]);
            const int s2 = int(scales[8 * h + s + 4]);
            const int s3 = int(scales[8 * h + s + 6]);
            device const uchar* ql = block + h * 64;
            device const uchar* qh = block + 128 + h * 32;
            int isum = 0;
            for (uint l = 0; l < 16; ++l) {
                const uint j = s * 16 + l;
                const int qla = int(ql[j]);
                const int qlb = int(ql[32 + j]);
                const int qhv = int(qh[j]);
                const int c0 = ((qla & 0x0f) | ((qhv & 3) << 4)) - 32;
                const int c1 = ((qlb & 0x0f) | (((qhv >> 2) & 3) << 4)) - 32;
                const int c2 = ((qla >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
                const int c3 = ((qlb >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
                isum += s0 * int(y[j]) * c0;
                isum += s1 * int(y[j + 32]) * c1;
                isum += s2 * int(y[j + 64]) * c2;
                isum += s3 * int(y[j + 96]) * c3;
            }
            const float weight_scale =
                float(*reinterpret_cast<device const half*>(block + 208));
            if (r == 0) acc0 = acc0 + (weight_scale * in_scale) * float(isum);
            else if (r == 1) acc1 = acc1 + (weight_scale * in_scale) * float(isum);
            else if (r == 2) acc2 = acc2 + (weight_scale * in_scale) * float(isum);
            else acc3 = acc3 + (weight_scale * in_scale) * float(isum);
        }
    }
    const float s0 = simd_sum(acc0);
    const float s1 = simd_sum(acc1);
    const float s2 = simd_sum(acc2);
    const float s3 = simd_sum(acc3);
    if (lane == 0) {
        output[row0] = s0;
        if (batch > 1) output[row0 + 1] = s1;
        if (batch > 2) output[row0 + 2] = s2;
        if (batch > 3) output[row0 + 3] = s3;
    }
}
"#;

/// The K<=8 speculative Q6_K tied-head shader.
const SPEC50_HEAD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Rows per simdgroup, and blocked vs flat lane mapping. Both are compile-time so
// the exactness harness can sweep them; neither changes the result (see below).
#ifndef SPEC50_ROWS_PER_SG
#define SPEC50_ROWS_PER_SG 1
#endif
#ifndef SPEC50_FLAT
#define SPEC50_FLAT 0
#endif
// Rows a single flat step handles together (they share one activation fetch).
// Must divide SPEC50_ROWS_PER_SG. RG == RB degenerates to the blocked mapping.
#ifndef SPEC50_ROWS_PER_STEP
#define SPEC50_ROWS_PER_STEP 1
#endif
// Simdgroups per threadgroup. Only affects scheduling; the per-row arithmetic is
// unchanged.
#ifndef SPEC50_SG_PER_TG
#define SPEC50_SG_PER_TG 4
#endif
// Diagnostic only: 1 = every candidate reads token 0 (same instruction count,
// 1/K the activation footprint); 2 = no activation load at all. The exactness
// gate rejects any non-zero value.
#ifndef SPEC50_ABLATE
#define SPEC50_ABLATE 0
#endif
// Storage format of the repacked activations: 0 = int8 (one 16-byte load per
// group, but sixteen byte-extract + convert pairs), 1 = f32 (no unpack at all,
// 4x the L1 traffic), 2 = f16 (half the traffic of f32, one convert per
// element). All three feed the FMA the identical exact integers; which one wins
// depends on K, so the encode picks per candidate count.
#ifndef SPEC50_YFMT
#define SPEC50_YFMT 1
#endif

// Repack the Q8_K activation quants so the inner loop's activation fetch
// coalesces. In the natural layout the 32 lanes of a simdgroup work on 32
// different superblock quarters, so one vector load reaches 32 separate cache
// lines and the load unit -- not the ALU and not DRAM -- becomes the wall
// (measured: 26 of 35 ms at K=8). Here element `l` of candidate `t`, unit `u`,
// superblock-quarter group `g` is stored at `((t*4 + g)*units + u)*16 + l`, so a
// single `uint4` load per (t, g) hands each lane its whole 16-byte group and
// consecutive lanes read consecutive 16-byte spans. Byte-for-byte the same
// int8s, only reordered.
kernel void q6k_spec50_expand(
    device const char* quants [[buffer(0)]],
    device char* out [[buffer(1)]],
    constant uint& n_sb [[buffer(2)]],
    constant uint& k_batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint units = n_sb * 4;
    const uint hidden = n_sb * 256;
    if (gid >= k_batch * hidden) return;
    const uint l = gid & 15u;
    const uint rest = gid >> 4;
    const uint u = rest % units;
    const uint rest2 = rest / units;
    const uint g = rest2 & 3u;
    const uint t = rest2 >> 2;
    const uint sb = u >> 2;
    const uint quarter = u & 3u;
    const uint h = quarter >> 1;
    const uint s = quarter & 1u;
    out[gid] = quants[t * hidden + sb * 256 + h * 128 + s * 16 + g * 32 + l];
}

// Same repack, widened to f16 (exact: |int8| <= 128 and f16 holds every integer
// up to 2048).
kernel void q6k_spec50_expand_f16(
    device const char* quants [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n_sb [[buffer(2)]],
    constant uint& k_batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint units = n_sb * 4;
    const uint hidden = n_sb * 256;
    if (gid >= k_batch * hidden) return;
    const uint l = gid & 15u;
    const uint rest = gid >> 4;
    const uint u = rest % units;
    const uint rest2 = rest / units;
    const uint g = rest2 & 3u;
    const uint t = rest2 >> 2;
    const uint sb = u >> 2;
    const uint quarter = u & 3u;
    const uint h = quarter >> 1;
    const uint s = quarter & 1u;
    out[gid] = half(quants[t * hidden + sb * 256 + h * 128 + s * 16 + g * 32 + l]);
}

// Same repack, widened to f32 (an exact int8 -> float widen).
kernel void q6k_spec50_expand_f32(
    device const char* quants [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n_sb [[buffer(2)]],
    constant uint& k_batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint units = n_sb * 4;
    const uint hidden = n_sb * 256;
    if (gid >= k_batch * hidden) return;
    const uint l = gid & 15u;
    const uint rest = gid >> 4;
    const uint u = rest % units;
    const uint rest2 = rest / units;
    const uint g = rest2 & 3u;
    const uint t = rest2 >> 2;
    const uint sb = u >> 2;
    const uint quarter = u & 3u;
    const uint h = quarter >> 1;
    const uint s = quarter & 1u;
    out[gid] = float(quants[t * hidden + sb * 256 + h * 128 + s * 16 + g * 32 + l]);
}

#define SPEC50_YT(tt) ((SPEC50_ABLATE == 1) ? 0u : (tt))

#if SPEC50_ABLATE == 2
#define SPEC50_YB_T float4
#define SPEC50_FETCH_Y(tt, gg)                                                \
    const float4 y0 = float4(1.0f, 2.0f, 3.0f, 4.0f);                         \
    const float4 y1 = y0; const float4 y2 = y0; const float4 y3 = y0;
#elif SPEC50_YFMT == 1
#define SPEC50_YB_T float4
#define SPEC50_FETCH_Y(tt, gg)                                                \
    device const float4* yp_ = yb                                             \
        + ((SPEC50_YT(tt) * 4u + (gg)) * units + u) * 4u;                     \
    const float4 y0 = yp_[0];                                                 \
    const float4 y1 = yp_[1];                                                 \
    const float4 y2 = yp_[2];                                                 \
    const float4 y3 = yp_[3];
#elif SPEC50_YFMT == 2
#define SPEC50_YB_T half4
#define SPEC50_FETCH_Y(tt, gg)                                                \
    device const half4* yp_ = yb                                              \
        + ((SPEC50_YT(tt) * 4u + (gg)) * units + u) * 4u;                     \
    const float4 y0 = float4(yp_[0]);                                         \
    const float4 y1 = float4(yp_[1]);                                         \
    const float4 y2 = float4(yp_[2]);                                         \
    const float4 y3 = float4(yp_[3]);
#else
#define SPEC50_YB_T uint4
#define SPEC50_FETCH_Y(tt, gg)                                                \
    const uint4 packed = yb[(SPEC50_YT(tt) * 4u + (gg)) * units + u];         \
    const float4 y0 = float4(as_type<char4>(packed.x));                       \
    const float4 y1 = float4(as_type<char4>(packed.y));                       \
    const float4 y2 = float4(as_type<char4>(packed.z));                       \
    const float4 y3 = float4(as_type<char4>(packed.w));
#endif

// One superblock-quarter group of one row against one candidate: sixteen f32
// FMAs in ascending element order, exactly the reference's term order.
#define SPEC50_GROUP_DOT(wv, out_part)                                        \
    {                                                                         \
        float p_ = 0.0f;                                                      \
        p_ = fma((wv)[0], y0.x, p_);  p_ = fma((wv)[1], y0.y, p_);            \
        p_ = fma((wv)[2], y0.z, p_);  p_ = fma((wv)[3], y0.w, p_);            \
        p_ = fma((wv)[4], y1.x, p_);  p_ = fma((wv)[5], y1.y, p_);            \
        p_ = fma((wv)[6], y1.z, p_);  p_ = fma((wv)[7], y1.w, p_);            \
        p_ = fma((wv)[8], y2.x, p_);  p_ = fma((wv)[9], y2.y, p_);            \
        p_ = fma((wv)[10], y2.z, p_); p_ = fma((wv)[11], y2.w, p_);           \
        p_ = fma((wv)[12], y3.x, p_); p_ = fma((wv)[13], y3.y, p_);           \
        p_ = fma((wv)[14], y3.z, p_); p_ = fma((wv)[15], y3.w, p_);           \
        out_part = p_;                                                        \
    }

// Decode the sixteen 6-bit weights of superblock-quarter group `g` of `block`.
#define SPEC50_DECODE_GROUP(block, g, h, s, wdst)                             \
    {                                                                         \
        device const uchar* ql_ = (block) + (h) * 64 + ((g) & 1u) * 32u        \
            + (s) * 16u;                                                      \
        device const uchar* qh_ = (block) + 128 + (h) * 32 + (s) * 16u;        \
        _Pragma("unroll")                                                     \
        for (uint l_ = 0; l_ < 16; ++l_) {                                    \
            const uint qlv_ = uint(ql_[l_]);                                  \
            const uint qhv_ = uint(qh_[l_]);                                  \
            const uint low_ = ((g) < 2u) ? (qlv_ & 0x0fu) : (qlv_ >> 4);      \
            (wdst)[l_] = float(int(low_                                       \
                | (((qhv_ >> (2u * (g))) & 3u) << 4)) - 32);                  \
        }                                                                     \
    }

// Blocked lane mapping, identical to q6k_linear_turbo / ..._batch_k: lane `l`
// owns units `l, l+32, ...` of every row the simdgroup holds, accumulates in
// that order, and the same simd_sum folds the 32 partials.
//
// The per-token inner product runs on f32 FMA, which is exact for this operand
// range (see the module comment), and the four superblock-quarter group terms
// are recombined in int32 so `isum` is the same integer the reference kernels
// produce.
template<uint KB>
static void q6k_spec50_body(
    device const float* input_scales,
    device const char* input_perm,
    device const uchar* weight_blocks,
    device float* output,
    uint n_sb,
    uint rows,
    float softcap,
    uint group,
    uint sgitg,
    uint lane
) {
    constexpr uint RB = SPEC50_ROWS_PER_SG;
    const uint row0 = (group * SPEC50_SG_PER_TG + sgitg) * RB;
    if (row0 >= rows) return;
    const uint batch = min(RB, rows - row0);
    const uint units = n_sb * 4;
    device const SPEC50_YB_T* yb =
        reinterpret_cast<device const SPEC50_YB_T*>(input_perm);

    float acc[RB][KB];
    #pragma unroll
    for (uint r = 0; r < RB; ++r) {
        #pragma unroll
        for (uint t = 0; t < KB; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (uint u = lane; u < units; u += 32u) {
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;

        device const uchar* blocks[RB];
        float weight_scale[RB];
        #pragma unroll
        for (uint r = 0; r < RB; ++r) {
            // Rows past the tail are clamped, computed and discarded so the
            // unrolled body stays uniform; only r < batch is ever stored.
            const uint rr = min(r, batch - 1u);
            device const uchar* block =
                weight_blocks + (ulong(row0 + rr) * n_sb + sb) * 210ul;
            blocks[r] = block;
            weight_scale[r] =
                float(*reinterpret_cast<device const half*>(block + 208));
        }

        int isum[RB][KB];
        #pragma unroll
        for (uint r = 0; r < RB; ++r) {
            #pragma unroll
            for (uint t = 0; t < KB; ++t) {
                isum[r][t] = 0;
            }
        }

        #pragma unroll
        for (uint g = 0; g < 4; ++g) {
            float wg[RB][16];
            float sub_scale[RB];
            #pragma unroll
            for (uint r = 0; r < RB; ++r) {
                device const char* wscales =
                    reinterpret_cast<device const char*>(blocks[r] + 192);
                sub_scale[r] = float(wscales[8 * h + s + 2 * g]);
                SPEC50_DECODE_GROUP(blocks[r], g, h, s, wg[r])
            }
            #pragma unroll
            for (uint t = 0; t < KB; ++t) {
                SPEC50_FETCH_Y(t, g)
                #pragma unroll
                for (uint r = 0; r < RB; ++r) {
                    float part;
                    SPEC50_GROUP_DOT(wg[r], part)
                    // |sub_scale| <= 128, |part| <= 65536: the product is an
                    // exactly representable integer, so the truncation is exact.
                    isum[r][t] += int(sub_scale[r] * part);
                }
            }
        }

        #pragma unroll
        for (uint r = 0; r < RB; ++r) {
            #pragma unroll
            for (uint t = 0; t < KB; ++t) {
                const float in_scale = input_scales[t * n_sb + sb];
                acc[r][t] = acc[r][t]
                    + (weight_scale[r] * in_scale) * float(isum[r][t]);
            }
        }
    }

    #pragma unroll
    for (uint t = 0; t < KB; ++t) {
        #pragma unroll
        for (uint r = 0; r < RB; ++r) {
            if (r >= batch) break;
            float s_val = simd_sum(acc[r][t]);
            if (lane == 0) {
                if (softcap > 0.0f) {
                    s_val = tanh(s_val / softcap) * softcap;
                }
                output[ulong(t) * rows + row0 + r] = s_val;
            }
        }
    }
}

// Flat lane mapping with row sharing. Two independent losses have to be paid at
// once: the blocked mapping leaves lanes idle (a row is `units = n_sb*4` = 44
// work items, which over 32 lanes is two rounds at 68.75% occupancy), while
// giving one lane several rows is the only way to amortise the activation fetch,
// which measurement showed is the real wall.
//
// So a simdgroup owns RB rows split into NG = RB/RG groups of RG rows. The lane
// walks a flat index over (group, unit) -- NG*units items, so 91.7% occupancy at
// NG = 2 or 4 and 100% at NG = 8 -- and each item does all RG rows of its group
// against one activation fetch.
//
// Bitwise identity survives. Lane `m` picks up flats `m, m+32, ...`, so for row
// group `gi` it accumulates exactly the units `u = (m - units*gi) mod 32, +32,
// ...` in ascending order -- the same terms, in the same order, that the
// reference's lane `j = (m - units*gi) mod 32` accumulates. Each group's
// partials are therefore a pure lane rotation of the reference's, and one
// `simd_shuffle` by `(lane + (units*gi) mod 32) mod 32` restores the reference
// lane assignment before the identical `simd_sum` runs.
template<uint KB>
static void q6k_spec50_body_flat(
    device const float* input_scales,
    device const char* input_perm,
    device const uchar* weight_blocks,
    device float* output,
    uint n_sb,
    uint rows,
    float softcap,
    uint group,
    uint sgitg,
    uint lane
) {
    constexpr uint RB = SPEC50_ROWS_PER_SG;
    constexpr uint RG = SPEC50_ROWS_PER_STEP;
    constexpr uint NG = RB / RG;
    const uint row0 = (group * SPEC50_SG_PER_TG + sgitg) * RB;
    if (row0 >= rows) return;
    const uint batch = min(RB, rows - row0);
    const uint units = n_sb * 4;
    const uint total = NG * units;
    device const SPEC50_YB_T* yb =
        reinterpret_cast<device const SPEC50_YB_T*>(input_perm);

    float acc[RB][KB];
    #pragma unroll
    for (uint r = 0; r < RB; ++r) {
        #pragma unroll
        for (uint t = 0; t < KB; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (uint f = lane; f < total; f += 32u) {
        const uint gi = f / units;
        const uint u = f - gi * units;
        const uint sb = u >> 2;
        const uint quarter = u & 3u;
        const uint h = quarter >> 1;
        const uint s = quarter & 1u;

        device const uchar* blocks[RG];
        float weight_scale[RG];
        #pragma unroll
        for (uint r = 0; r < RG; ++r) {
            // Rows past the tail are clamped, computed and discarded.
            const uint rr = min(gi * RG + r, batch - 1u);
            device const uchar* block =
                weight_blocks + (ulong(row0 + rr) * n_sb + sb) * 210ul;
            blocks[r] = block;
            weight_scale[r] =
                float(*reinterpret_cast<device const half*>(block + 208));
        }

        int isum[RG][KB];
        #pragma unroll
        for (uint r = 0; r < RG; ++r) {
            #pragma unroll
            for (uint t = 0; t < KB; ++t) {
                isum[r][t] = 0;
            }
        }

        #pragma unroll
        for (uint g = 0; g < 4; ++g) {
            float wg[RG][16];
            float sub_scale[RG];
            #pragma unroll
            for (uint r = 0; r < RG; ++r) {
                device const char* wscales =
                    reinterpret_cast<device const char*>(blocks[r] + 192);
                sub_scale[r] = float(wscales[8 * h + s + 2 * g]);
                SPEC50_DECODE_GROUP(blocks[r], g, h, s, wg[r])
            }
            #pragma unroll
            for (uint t = 0; t < KB; ++t) {
                SPEC50_FETCH_Y(t, g)
                #pragma unroll
                for (uint r = 0; r < RG; ++r) {
                    float part;
                    SPEC50_GROUP_DOT(wg[r], part)
                    // |sub_scale| <= 128, |part| <= 65536: the product is an
                    // exactly representable integer, so the truncation is exact.
                    isum[r][t] += int(sub_scale[r] * part);
                }
            }
        }

        // Predicated, never arithmetic: only the owning group's accumulators are
        // touched, so no zero addend is ever folded in.
        #pragma unroll
        for (uint gs = 0; gs < NG; ++gs) {
            if (gs == gi) {
                #pragma unroll
                for (uint r = 0; r < RG; ++r) {
                    #pragma unroll
                    for (uint t = 0; t < KB; ++t) {
                        const float in_scale = input_scales[t * n_sb + sb];
                        acc[gs * RG + r][t] = acc[gs * RG + r][t]
                            + (weight_scale[r] * in_scale) * float(isum[r][t]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint t = 0; t < KB; ++t) {
        #pragma unroll
        for (uint r = 0; r < RB; ++r) {
            if (r >= batch) break;
            const uint rot = (units * (r / RG)) & 31u;
            const float rotated =
                simd_shuffle(acc[r][t], ushort((lane + rot) & 31u));
            float s_val = simd_sum(rotated);
            if (lane == 0) {
                if (softcap > 0.0f) {
                    s_val = tanh(s_val / softcap) * softcap;
                }
                output[ulong(t) * rows + row0 + r] = s_val;
            }
        }
    }
}

#if SPEC50_FLAT
#define SPEC50_BODY q6k_spec50_body_flat
#else
#define SPEC50_BODY q6k_spec50_body
#endif

#define SPEC50_BATCH_ENTRY(N)                                                 \
kernel void q6k_spec50_batch_k##N(                                            \
    device const float* input_scales [[buffer(0)]],                           \
    device const char* input_perm [[buffer(1)]],                              \
    device const uchar* weight_blocks [[buffer(2)]],                          \
    device float* output [[buffer(3)]],                                       \
    constant uint& n_sb [[buffer(4)]],                                        \
    constant uint& rows [[buffer(5)]],                                        \
    constant float& softcap [[buffer(6)]],                                    \
    uint group [[threadgroup_position_in_grid]],                              \
    uint sgitg [[simdgroup_index_in_threadgroup]],                            \
    uint lane [[thread_index_in_simdgroup]]                                   \
) {                                                                           \
    SPEC50_BODY<N>(input_scales, input_perm, weight_blocks, output,           \
                   n_sb, rows, softcap, group, sgitg, lane);                  \
}

SPEC50_BATCH_ENTRY(1)
SPEC50_BATCH_ENTRY(2)
SPEC50_BATCH_ENTRY(3)
SPEC50_BATCH_ENTRY(4)
SPEC50_BATCH_ENTRY(5)
SPEC50_BATCH_ENTRY(6)
SPEC50_BATCH_ENTRY(7)
SPEC50_BATCH_ENTRY(8)
// K=9..16 capacity extension (spec50-widen). K is a compile-time template
// parameter and the per-row program is identical for every row regardless of
// K, so these instantiations change nothing about K<=8; the widened table is
// optional at build time and K<=8 selection never touches it.
SPEC50_BATCH_ENTRY(9)
SPEC50_BATCH_ENTRY(10)
SPEC50_BATCH_ENTRY(11)
SPEC50_BATCH_ENTRY(12)
SPEC50_BATCH_ENTRY(13)
SPEC50_BATCH_ENTRY(14)
SPEC50_BATCH_ENTRY(15)
SPEC50_BATCH_ENTRY(16)

// Per-candidate-row argmax with the same semantics as `argmax_f32_greedy`:
// a strictly-greater scan (so the first maximum in a thread's ascending stride
// wins) folded by a tree that breaks ties toward the lower index, i.e. the
// lowest index among all maxima. It reads the SOFTCAPPED logits the projection
// wrote. Softcap (tanh(x/c)*c) is monotonic, so it cannot reorder two distinct
// logits; it is applied first anyway because saturation can map two distinct
// pre-cap values onto one post-cap float, and the reference pipeline's argmax
// sees that collapsed tie too.
kernel void q6k_spec50_argmax_rows(
    device const float* logits [[buffer(0)]],
    device uint* out_id [[buffer(1)]],
    device float* out_val [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float sh_val[1024];
    threadgroup uint sh_idx[1024];
    device const float* base = logits + ulong(row) * ulong(count);
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < count; i += tg_size) {
        const float v = base[i];
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    sh_val[tid] = best;
    sh_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float ov = sh_val[tid + s];
            const uint oi = sh_idx[tid + s];
            if (ov > sh_val[tid] || (ov == sh_val[tid] && oi < sh_idx[tid])) {
                sh_val[tid] = ov;
                sh_idx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_id[row] = sh_idx[0];
        out_val[row] = sh_val[0];
    }
}
"#;

/// Vocab rows each simdgroup owns; mirrors `SPEC50_ROWS_PER_SG` in the shader.
/// One threadgroup covers `SPEC50_SG_PER_TG * this` rows. These five constants
/// are prepended to the shader source at compile time, so the host-side grid and
/// the kernel geometry cannot drift apart.
pub(crate) const SPEC50_ROWS_PER_SG: usize = 8;

/// Rows a single flat step handles together; mirrors `SPEC50_ROWS_PER_STEP`.
const SPEC50_ROWS_PER_STEP: usize = 1;

/// Mirrors `SPEC50_FLAT`: 1 = flat (row-group, unit) lane mapping.
const SPEC50_FLAT: u32 = 1;

/// Simdgroups per threadgroup; mirrors `SPEC50_SG_PER_TG`.
const SPEC50_SG_PER_TG: usize = 4;

/// Activation repack format, mirroring `SPEC50_YFMT`: 0 = int8, 1 = f32, 2 = f16.
const SPEC50_YFMT: u32 = 2;

/// Scratch bytes per activation element for `SPEC50_YFMT`.
const SPEC50_YFMT_STRIDE: usize = match SPEC50_YFMT {
    1 => 4,
    2 => 2,
    _ => 1,
};

/// Name of the repack kernel matching `SPEC50_YFMT`.
const SPEC50_EXPAND_KERNEL: &str = match SPEC50_YFMT {
    1 => "q6k_spec50_expand_f32",
    2 => "q6k_spec50_expand_f16",
    _ => "q6k_spec50_expand",
};

/// Compiled pipelines for the speculative Q6_K tied head.
pub(crate) struct Spec50HeadKernels {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    /// Index `k - 1` for `k` in `1..=8`.
    batch: [ComputePipelineState; 8],
    /// Index `k - 9` for `k` in `9..=16` (spec50-widen capacity extension).
    /// Optional so a failure here can never disable the proven K<=8 table;
    /// when absent, `encode_q6k_spec50_batch` refuses K>8 and the caller keeps
    /// its existing fallback.
    batch_wide: Option<[ComputePipelineState; 8]>,
    expand: ComputePipelineState,
    argmax: ComputePipelineState,
}

static SPEC50_HEAD_KERNELS: OnceLock<Option<Spec50HeadKernels>> = OnceLock::new();

/// Compile (once) the speculative Q6_K head library. Returns `None` and prints
/// the compiler diagnostic on failure so the caller can keep the existing lane.
pub(crate) fn spec50_head_kernels() -> Option<&'static Spec50HeadKernels> {
    SPEC50_HEAD_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            // Same options as STRICT_Q8K_SHADER, which owns the kernels this
            // replaces: fast math off, so the surviving float expressions and
            // the softcap `tanh` compile identically.
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let source = format!(
                "#define SPEC50_ROWS_PER_SG {SPEC50_ROWS_PER_SG}\n\
                 #define SPEC50_ROWS_PER_STEP {SPEC50_ROWS_PER_STEP}\n\
                 #define SPEC50_FLAT {SPEC50_FLAT}\n\
                 #define SPEC50_SG_PER_TG {SPEC50_SG_PER_TG}\n\
                 #define SPEC50_YFMT {SPEC50_YFMT}\n{SPEC50_HEAD_SHADER}"
            );
            let library = device
                .new_library_with_source(&source, &options)
                .map_err(|err| eprintln!("[metal] SPEC50_HEAD_SHADER compile failed: {err}"))
                .ok()?;
            let pipeline = |name: &str| -> Option<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] spec50 missing {name}: {err}"))
                    .ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] spec50 pipeline {name}: {err}"))
                    .ok()
            };
            let mut batch = Vec::with_capacity(8);
            for k in 1..=8 {
                batch.push(pipeline(&format!("q6k_spec50_batch_k{k}"))?);
            }
            let batch: [ComputePipelineState; 8] = batch.try_into().ok()?;
            // The lane -> unit partition this kernel inherits is only correct on
            // a 32-wide simdgroup, exactly like `admitted_32_lane_pipeline`.
            if batch.iter().any(|p| p.thread_execution_width() != 32)
                || batch
                    .iter()
                    .any(|p| p.max_total_threads_per_threadgroup() < 32 * SPEC50_SG_PER_TG as u64)
            {
                eprintln!("[metal] spec50 head: simd width or threadgroup size not admitted");
                return None;
            }
            // K=9..16 table (spec50-widen). Strictly optional: a compile or
            // admission failure here leaves the K<=8 table untouched and only
            // makes `encode_q6k_spec50_batch` refuse K>8.
            let mut wide = Vec::with_capacity(8);
            for k in 9..=16 {
                match pipeline(&format!("q6k_spec50_batch_k{k}")) {
                    Some(p) => wide.push(p),
                    None => break,
                }
            }
            let batch_wide: Option<[ComputePipelineState; 8]> = if wide.len() == 8
                && wide.iter().all(|p| {
                    p.thread_execution_width() == 32
                        && p.max_total_threads_per_threadgroup() >= 32 * SPEC50_SG_PER_TG as u64
                }) {
                wide.try_into().ok()
            } else {
                eprintln!("[metal] spec50 head: K=9..16 table unavailable; K<=8 unaffected");
                None
            };
            let expand = pipeline(SPEC50_EXPAND_KERNEL)?;
            let argmax = pipeline("q6k_spec50_argmax_rows")?;
            if argmax.max_total_threads_per_threadgroup() < 1024 {
                eprintln!("[metal] spec50 head: argmax threadgroup < 1024");
                return None;
            }
            let queue = device.new_command_queue();
            Some(Spec50HeadKernels {
                device,
                queue,
                batch,
                batch_wide,
                expand,
                argmax,
            })
        })
        .as_ref()
}

/// Bytes the integrator must allocate for the one new buffer, `activation_perm`:
/// a coalescing-friendly repack of the Q8_K activation quants, `max_k * hidden`
/// bytes (45 KB at the 26B row's `max_k = 16`, `hidden = 2816`). Shared storage,
/// written GPU-side by `q6k_spec50_expand` at the head of every encode.
pub(crate) fn spec50_activation_scratch_bytes(max_k: usize, hidden: usize) -> usize {
    max_k * hidden * SPEC50_YFMT_STRIDE
}

/// Encode the K<=16 speculative Q6_K tied-head projection.
///
/// Buffer order on `q6k_spec50_batch_k{K}`:
///   0 `input_scales`     `K * n_superblocks` f32 (candidate-major)
///   1 `activation_perm`  `K * hidden` bytes of scratch, filled by this call
///   2 `weight_blocks`    Q6_K table at `weight_offset`
///   3 `output`           `K * rows` f32, softcapped
///   4 `n_sb` u32, 5 `rows` u32, 6 `softcap` f32
///
/// Returns `false` (encoding nothing) when the pipelines are unavailable or
/// `k_batch` is outside `1..=16` (K in 9..=16 additionally requires the
/// optional widened table); the caller then keeps `encode_q6k_ordered_batch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q6k_spec50_batch(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50HeadKernels,
    input_scales: &Buffer,
    input_quants: &Buffer,
    activation_perm: &Buffer,
    weight: &Buffer,
    weight_offset: u64,
    output: &Buffer,
    n_superblocks: usize,
    rows: usize,
    k_batch: usize,
    hidden: usize,
    softcap: f32,
) -> bool {
    if k_batch == 0 || k_batch > 16 || rows == 0 || n_superblocks == 0 {
        return false;
    }
    if k_batch > 8 && kernels.batch_wide.is_none() {
        return false;
    }
    let count = (k_batch * hidden) as u32;
    let n_sb_expand = n_superblocks as u32;
    let k_expand = k_batch as u32;
    encoder.set_compute_pipeline_state(&kernels.expand);
    encoder.set_buffer(0, Some(input_quants), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_bytes(2, 4, &n_sb_expand as *const u32 as *const _);
    encoder.set_bytes(3, 4, &k_expand as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (count as u64).div_ceil(256),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    if k_batch > 8 {
        let wide = kernels.batch_wide.as_ref().expect("checked above");
        encoder.set_compute_pipeline_state(&wide[k_batch - 9]);
    } else {
        encoder.set_compute_pipeline_state(&kernels.batch[k_batch - 1]);
    }
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_buffer(2, Some(weight), weight_offset);
    encoder.set_buffer(3, Some(output), 0);
    let n_sb_u32 = n_superblocks as u32;
    let rows_u32 = rows as u32;
    encoder.set_bytes(4, 4, &n_sb_u32 as *const u32 as *const _);
    encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &softcap as *const f32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: rows.div_ceil(SPEC50_SG_PER_TG * SPEC50_ROWS_PER_SG) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32 * SPEC50_SG_PER_TG as u64,
            height: 1,
            depth: 1,
        },
    );
    true
}

/// Encode the per-candidate-row argmax over the softcapped logits written by
/// [`encode_q6k_spec50_batch`]. Must be encoded after it in the same encoder.
///
/// Buffer order: 0 `logits` (`k_batch * rows` f32), 1 `argmax_ids` (`k_batch`
/// u32), 2 `argmax_vals` (`k_batch` f32), 3 `rows` u32.
pub(crate) fn encode_q6k_spec50_argmax(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50HeadKernels,
    logits: &Buffer,
    argmax_ids: &Buffer,
    argmax_vals: &Buffer,
    rows: usize,
    k_batch: usize,
) {
    encoder.set_compute_pipeline_state(&kernels.argmax);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(argmax_ids), 0);
    encoder.set_buffer(2, Some(argmax_vals), 0);
    let rows_u32 = rows as u32;
    encoder.set_bytes(3, 4, &rows_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: k_batch as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::ComputePipelineState;

    const Q6K_WIRE: usize = 210;
    const HIDDEN: usize = 2816;
    const N_SB: usize = HIDDEN / 256;
    const SOFTCAP: f32 = 30.0;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
    }

    /// Write a plausible Q6_K row block: full-range ql/qh/int8 scales and a
    /// small normal f16 super-scale (never NaN/Inf, so bit comparison is total).
    fn fill_q6k_block(rng: &mut Rng, out: &mut [u8]) {
        for byte in out.iter_mut().take(208) {
            *byte = rng.byte();
        }
        let sign = (rng.next() & 1) as u16;
        let exp = 5 + (rng.next() % 8) as u16; // 2^-10 .. 2^-3 magnitudes
        let mant = (rng.next() % 1024) as u16;
        let bits = (sign << 15) | (exp << 10) | mant;
        out[208] = (bits & 0xff) as u8;
        out[209] = (bits >> 8) as u8;
    }

    fn build_weights(rng: &mut Rng, rows: usize) -> Vec<u8> {
        let mut w = vec![0u8; rows * N_SB * Q6K_WIRE];
        for block in w.chunks_exact_mut(Q6K_WIRE) {
            fill_q6k_block(rng, block);
        }
        w
    }

    fn build_activations(rng: &mut Rng, k: usize) -> (Vec<f32>, Vec<i8>) {
        let scales: Vec<f32> = (0..k * N_SB)
            .map(|_| 0.002 + (rng.next() % 4096) as f32 * 1.0e-6)
            .collect();
        let quants: Vec<i8> = (0..k * HIDDEN).map(|_| rng.byte() as i8).collect();
        (scales, quants)
    }

    struct RefKernels {
        device: Device,
        queue: CommandQueue,
        single: ComputePipelineState,
        batch_k: ComputePipelineState,
        batch_k8: ComputePipelineState,
        probe_inline_acc: ComputePipelineState,
    }

    fn reference_kernels() -> RefKernels {
        let device = Device::system_default().expect("no Metal device");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SPEC50_REFERENCE_SHADER, &options)
            .expect("reference shader compile");
        let pipe = |name: &str| {
            let f = library.get_function(name, None).expect(name);
            device
                .new_compute_pipeline_state_with_function(&f)
                .expect(name)
        };
        let probe_library = device
            .new_library_with_source(SPEC50_PROBE_SHADER, &options)
            .expect("probe shader compile");
        let probe_function = probe_library
            .get_function("q6k_probe_turbo_inline_acc", None)
            .expect("probe function");
        let probe_inline_acc = device
            .new_compute_pipeline_state_with_function(&probe_function)
            .expect("probe pipeline");
        let queue = device.new_command_queue();
        RefKernels {
            single: pipe("q6k_linear_turbo"),
            batch_k: pipe("q6k_linear_turbo_batch_k"),
            batch_k8: pipe("q6k_linear_turbo_batch_k8"),
            probe_inline_acc,
            device,
            queue,
        }
    }

    fn shared(device: &Device, bytes: usize) -> Buffer {
        device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_reference(
        encoder: &metal::ComputeCommandEncoderRef,
        refs: &RefKernels,
        scales: &Buffer,
        quants: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        k: usize,
        softcap: f32,
        force_batch_k: bool,
    ) {
        if k == 1 && !force_batch_k {
            encoder.set_compute_pipeline_state(&refs.single);
        } else if k == 8 && !force_batch_k {
            encoder.set_compute_pipeline_state(&refs.batch_k8);
        } else {
            encoder.set_compute_pipeline_state(&refs.batch_k);
        }
        encoder.set_buffer(0, Some(scales), 0);
        encoder.set_buffer(1, Some(quants), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(output), 0);
        let n_sb_u32 = N_SB as u32;
        let rows_u32 = rows as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(4, 4, &n_sb_u32 as *const u32 as *const _);
        encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
        encoder.set_bytes(7, 4, &softcap as *const f32 as *const _);
        if k == 8 && !force_batch_k {
            encoder.dispatch_thread_groups(
                metal::MTLSize {
                    width: rows.div_ceil(4) as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            encoder.dispatch_thread_groups(
                metal::MTLSize {
                    width: rows.div_ceil(16) as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    /// The in-file copies used by this harness must be byte-identical to the
    /// kernels they stand in for, or the exactness claim proves nothing.
    #[test]
    fn spec50_reference_copies_are_verbatim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal.rs");
        let source = std::fs::read_to_string(path).expect("read src/metal.rs");
        for name in [
            "q6k_linear_turbo",
            "q6k_linear_turbo_batch_k",
            "q6k_linear_turbo_batch_k8",
        ] {
            let needle = format!("\nkernel void {name}(\n");
            let start = source.find(&needle).unwrap_or_else(|| panic!("{name} not found"))
                + 1;
            let end = source[start..].find("\n}\n").expect("kernel end") + start + 3;
            let original = &source[start..end];
            let mine_start = SPEC50_REFERENCE_SHADER
                .find(&needle)
                .unwrap_or_else(|| panic!("{name} missing from copy"))
                + 1;
            let mine_end = SPEC50_REFERENCE_SHADER[mine_start..]
                .find("\n}\n")
                .expect("copy end")
                + mine_start
                + 3;
            let mine = &SPEC50_REFERENCE_SHADER[mine_start..mine_end];
            assert_eq!(original, mine, "{name} copy drifted from src/metal.rs");
        }
    }

    /// Diagnostic requested by the integrator: at k == 1 the speculative lane
    /// delegates to `forward()`, so `q6k_linear_turbo` and the batch kernel are
    /// both live for the same round and would be expected to agree. They do NOT
    /// (see the printed counts). The probe pins the cause: the only difference
    /// between the two sources is that `q6k_linear_turbo` names the product in a
    /// `term` local before adding it, while the batch kernel inlines it, so the
    /// batch kernel contracts the accumulator update to a single `fma` and the
    /// single-token kernel does not. Rewriting the accumulate in the batch
    /// kernel's form makes the single-token kernel reproduce the batch result
    /// bit for bit. This is a pre-existing inconsistency between the two engine
    /// lanes; it is NOT introduced here, and the new kernel deliberately follows
    /// the batch (oracle-verified) side.
    #[test]
    fn spec50_reference_batch_vs_single_at_k1_diagnostic() {
        let refs = reference_kernels();
        let mut rng = Rng(0x51ec_5000_1234_9abd);
        let rows = 2048usize;
        let weights = build_weights(&mut rng, rows);
        let (scales, quants) = build_activations(&mut rng, 1);

        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let out_single = shared(&refs.device, rows * 4);
        let out_batch = shared(&refs.device, rows * 4);
        let out_probe = shared(&refs.device, rows * 4);

        let cb = refs.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        encode_reference(e, &refs, &sbuf, &qbuf, &wbuf, &out_single, rows, 1, 0.0, false);
        encode_reference(e, &refs, &sbuf, &qbuf, &wbuf, &out_batch, rows, 1, 0.0, true);
        e.set_compute_pipeline_state(&refs.probe_inline_acc);
        e.set_buffer(0, Some(&sbuf), 0);
        e.set_buffer(1, Some(&qbuf), 0);
        e.set_buffer(2, Some(&wbuf), 0);
        e.set_buffer(3, Some(&out_probe), 0);
        let n_sb_u32 = N_SB as u32;
        let rows_u32 = rows as u32;
        e.set_bytes(4, 4, &n_sb_u32 as *const u32 as *const _);
        e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
        e.dispatch_thread_groups(
            metal::MTLSize { width: rows.div_ceil(16) as u64, height: 1, depth: 1 },
            metal::MTLSize { width: 128, height: 1, depth: 1 },
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let mut single = vec![0f32; rows];
        let mut batch = vec![0f32; rows];
        let mut probe = vec![0f32; rows];
        read_buffer_f32(&out_single, &mut single);
        read_buffer_f32(&out_batch, &mut batch);
        read_buffer_f32(&out_probe, &mut probe);

        let count = |a: &[f32], b: &[f32]| {
            let mut diff = 0usize;
            let mut worst = 0i64;
            for i in 0..a.len() {
                if a[i].to_bits() != b[i].to_bits() {
                    diff += 1;
                    worst =
                        worst.max((a[i].to_bits() as i64 - b[i].to_bits() as i64).abs());
                }
            }
            (diff, worst)
        };
        let (d_sb, u_sb) = count(&single, &batch);
        let (d_pb, u_pb) = count(&probe, &batch);
        let arg = |v: &[f32]| {
            let mut best = f32::NEG_INFINITY;
            let mut bi = 0usize;
            for (i, &x) in v.iter().enumerate() {
                if x > best {
                    best = x;
                    bi = i;
                }
            }
            bi
        };
        eprintln!(
            "[spec50] DIAGNOSTIC K=1: q6k_linear_turbo vs q6k_linear_turbo_batch_k: \
             {d_sb}/{rows} rows differ (worst {u_sb} ULP); argmax single={} batch={}",
            arg(&single),
            arg(&batch)
        );
        eprintln!(
            "[spec50] DIAGNOSTIC cause: turbo with the accumulate written in the batch \
             kernel's inline (fma-contractible) form differs from batch_k on \
             {d_pb}/{rows} rows (worst {u_pb} ULP)"
        );
        assert_ne!(d_sb, 0, "expected the shipped K=1 lanes to disagree");
        assert_eq!(
            d_pb, 0,
            "fma contraction of the accumulator update does not explain the K=1 split"
        );
    }

    /// The binding contract: the new kernel is bitwise identical to the kernel
    /// it replaces, for every row and every K in 1..=8.
    #[test]
    fn spec50_batch_is_bitwise_identical_to_reference() {
        let refs = reference_kernels();
        let kernels = spec50_head_kernels().expect("spec50 pipelines");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

        for &rows in &[2048usize, 2045usize] {
            let weights = build_weights(&mut rng, rows);
            let (scales, quants) = build_activations(&mut rng, 8);
            let wbuf = shared(&refs.device, weights.len());
            write_buffer_u8(&wbuf, &weights);
            let sbuf = shared(&refs.device, scales.len() * 4);
            write_buffer_f32(&sbuf, &scales);
            let qbuf = shared(&refs.device, quants.len());
            write_buffer_i8(&qbuf, &quants);
            let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, HIDDEN));

            for k in 1..=8usize {
                let out_ref = shared(&refs.device, k * rows * 4);
                let out_new = shared(&refs.device, k * rows * 4);
                let cb = refs.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                // Reference == the kernel the oracle-verified chained lane runs
                // for this K: batch_k8 at K=8, batch_k otherwise.
                encode_reference(
                    e, &refs, &sbuf, &qbuf, &wbuf, &out_ref, rows, k, SOFTCAP, k == 1,
                );
                assert!(encode_q6k_spec50_batch(
                    e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out_new, N_SB, rows, k,
                    HIDDEN, SOFTCAP,
                ));
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);

                let mut a = vec![0f32; k * rows];
                let mut b = vec![0f32; k * rows];
                read_buffer_f32(&out_ref, &mut a);
                read_buffer_f32(&out_new, &mut b);
                let mut diff = 0usize;
                let mut worst_ulp = 0i64;
                for i in 0..k * rows {
                    if a[i].to_bits() != b[i].to_bits() {
                        diff += 1;
                        worst_ulp = worst_ulp
                            .max((a[i].to_bits() as i64 - b[i].to_bits() as i64).abs());
                    }
                }
                assert_eq!(
                    diff, 0,
                    "rows={rows} K={k}: {diff} values differ, worst {worst_ulp} ULP"
                );
            }
            eprintln!("[spec50] bitwise identical to reference for K=1..=8 at rows={rows}");
        }
    }

    /// Token t's logits must not depend on how many candidates share the batch.
    #[test]
    fn spec50_batch_rows_are_independent_of_k() {
        let refs = reference_kernels();
        let kernels = spec50_head_kernels().expect("spec50 pipelines");
        let mut rng = Rng(0x1357_9bdf_0246_8ace);
        let rows = 1024usize;
        let weights = build_weights(&mut rng, rows);
        let (scales, quants) = build_activations(&mut rng, 8);
        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, HIDDEN));

        // Each token in isolation (K=1 over its own slice) then the K=8 batch.
        let mut solo = vec![0f32; 8 * rows];
        for t in 0..8usize {
            let sub_scales = &scales[t * N_SB..(t + 1) * N_SB];
            let sub_quants = &quants[t * HIDDEN..(t + 1) * HIDDEN];
            let ss = shared(&refs.device, N_SB * 4);
            write_buffer_f32(&ss, sub_scales);
            let sq = shared(&refs.device, HIDDEN);
            write_buffer_i8(&sq, sub_quants);
            let out = shared(&refs.device, rows * 4);
            let cb = refs.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            assert!(encode_q6k_spec50_batch(
                e, kernels, &ss, &sq, &fbuf, &wbuf, 0, &out, N_SB, rows, 1, HIDDEN, SOFTCAP,
            ));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            read_buffer_f32(&out, &mut solo[t * rows..(t + 1) * rows]);
        }

        for k in 2..=8usize {
            let out = shared(&refs.device, k * rows * 4);
            let cb = refs.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            assert!(encode_q6k_spec50_batch(
                e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out, N_SB, rows, k, HIDDEN, SOFTCAP,
            ));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let mut got = vec![0f32; k * rows];
            read_buffer_f32(&out, &mut got);
            for t in 0..k {
                for r in 0..rows {
                    assert_eq!(
                        got[t * rows + r].to_bits(),
                        solo[t * rows + r].to_bits(),
                        "K={k} token {t} row {r} depends on batch width"
                    );
                }
            }
        }
        eprintln!("[spec50] per-token batch independence holds for K=1..=8");
    }

    /// Per-row argmax must equal a CPU first-maximum scan of the same softcapped
    /// logits (lowest index wins ties).
    #[test]
    fn spec50_argmax_matches_cpu_first_maximum() {
        let refs = reference_kernels();
        let kernels = spec50_head_kernels().expect("spec50 pipelines");
        let mut rng = Rng(0x0bad_c0de_dead_beef);
        let rows = 4096usize;
        let k = 8usize;
        let weights = build_weights(&mut rng, rows);
        let (scales, quants) = build_activations(&mut rng, k);
        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(k, HIDDEN));
        let out = shared(&refs.device, k * rows * 4);
        let ids = shared(&refs.device, k * 4);
        let vals = shared(&refs.device, k * 4);

        let cb = refs.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        assert!(encode_q6k_spec50_batch(
            e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out, N_SB, rows, k, HIDDEN, SOFTCAP,
        ));
        encode_q6k_spec50_argmax(e, kernels, &out, &ids, &vals, rows, k);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let mut logits = vec![0f32; k * rows];
        read_buffer_f32(&out, &mut logits);
        let mut got_vals = vec![0f32; k];
        read_buffer_f32(&vals, &mut got_vals);
        let mut got_ids = vec![0u32; k];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids.contents().cast::<u32>(),
                got_ids.as_mut_ptr(),
                k,
            );
        }
        for t in 0..k {
            let row = &logits[t * rows..(t + 1) * rows];
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > best {
                    best = v;
                    best_i = i;
                }
            }
            assert_eq!(got_ids[t] as usize, best_i, "argmax id mismatch at token {t}");
            assert_eq!(
                got_vals[t].to_bits(),
                best.to_bits(),
                "argmax value mismatch at token {t}"
            );
        }
        // Force an exact tie at a higher index than an equal earlier one and
        // confirm the lower index still wins.
        let mut tied = logits[..rows].to_vec();
        tied[7] = 1.0e30;
        tied[9] = 1.0e30;
        let tie_buf = shared(&refs.device, rows * 4);
        write_buffer_f32(&tie_buf, &tied);
        let cb = refs.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        encode_q6k_spec50_argmax(e, kernels, &tie_buf, &ids, &vals, rows, 1);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let tie_id = unsafe { *(ids.contents() as *const u32) };
        assert_eq!(tie_id, 7, "tie must resolve to the lowest index");
        eprintln!("[spec50] per-row argmax matches CPU first-maximum, ties to lowest index");
    }

    /// Geometry sweep: rows-per-simdgroup x weight-decode shape, on the real
    /// 26B table. Each variant is first checked bitwise against the reference on
    /// a small table, so no configuration can win by being wrong.
    #[test]
    fn spec50_sweep_geometry() {
        const ROWS: usize = 262_144;
        const REPS: usize = 10;
        let refs = reference_kernels();
        let bytes = ROWS * N_SB * Q6K_WIRE;
        let wbuf = shared(&refs.device, bytes);
        {
            let mut rng = Rng(0x2718_2818_2845_9045);
            let dst =
                unsafe { std::slice::from_raw_parts_mut(wbuf.contents().cast::<u8>(), bytes) };
            for block in dst.chunks_exact_mut(Q6K_WIRE) {
                fill_q6k_block(&mut rng, block);
            }
        }
        let mut rng = Rng(0x3141_5926_5358_9793);
        let (scales, quants) = build_activations(&mut rng, 8);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, 8 * HIDDEN * 4);
        let out = shared(&refs.device, 8 * ROWS * 4);

        // Small table for the per-variant exactness gate.
        const SROWS: usize = 1024;
        let mut srng = Rng(0x0f0f_1234_5678_9abc);
        let sweights = build_weights(&mut srng, SROWS);
        let swbuf = shared(&refs.device, sweights.len());
        write_buffer_u8(&swbuf, &sweights);
        let sout_ref = shared(&refs.device, 8 * SROWS * 4);
        let sout_new = shared(&refs.device, 8 * SROWS * 4);

        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        // (rows/sg, rows/step, simdgroups/tg, flat, ablate, yfmt)
        #[rustfmt::skip]
        let configs: [(usize, usize, usize, u32, u32, u32); 10] = [
            (8, 1, 1, 1, 0, 2), (8, 1, 2, 1, 0, 2), (8, 1, 4, 1, 0, 2), (8, 1, 8, 1, 0, 2),
            (4, 1, 2, 1, 0, 2), (4, 1, 8, 1, 0, 2),
            (8, 2, 2, 1, 0, 2), (4, 2, 2, 1, 0, 2),
            (2, 1, 2, 1, 0, 2), (16, 1, 2, 1, 0, 2),
        ];
        for (rb, rg, sg, flat, ablate, yfmt) in configs {
            let src = format!(
                "#define SPEC50_ROWS_PER_SG {rb}\n#define SPEC50_ROWS_PER_STEP {rg}\n#define SPEC50_SG_PER_TG {sg}\n#define SPEC50_FLAT {flat}\n#define SPEC50_ABLATE {ablate}\n#define SPEC50_YFMT {yfmt}\n{SPEC50_HEAD_SHADER}"
            );
            let library = match refs.device.new_library_with_source(&src, &options) {
                Ok(l) => l,
                Err(err) => {
                    eprintln!("[spec50] sweep rb={rb} rg={rg} sg={sg} flat={flat} ablate={ablate} yfmt={yfmt}: compile failed: {err}");
                    continue;
                }
            };
            let expand = {
                let name = match yfmt {
                    1 => "q6k_spec50_expand_f32",
                    2 => "q6k_spec50_expand_f16",
                    _ => "q6k_spec50_expand",
                };
                let f = library.get_function(name, None).unwrap();
                refs.device
                    .new_compute_pipeline_state_with_function(&f)
                    .unwrap()
            };
            for &k in &[4usize, 8] {
                let f = library
                    .get_function(&format!("q6k_spec50_batch_k{k}"), None)
                    .unwrap();
                let pipe = refs
                    .device
                    .new_compute_pipeline_state_with_function(&f)
                    .unwrap();
                let encode = |e: &metal::ComputeCommandEncoderRef,
                              w: &Buffer,
                              o: &Buffer,
                              n_rows: usize| {
                    let count = (k * HIDDEN) as u32;
                    let n_sb_e = N_SB as u32;
                    let k_e = k as u32;
                    e.set_compute_pipeline_state(&expand);
                    e.set_buffer(0, Some(&qbuf), 0);
                    e.set_buffer(1, Some(&fbuf), 0);
                    e.set_bytes(2, 4, &n_sb_e as *const u32 as *const _);
                    e.set_bytes(3, 4, &k_e as *const u32 as *const _);
                    e.dispatch_thread_groups(
                        metal::MTLSize { width: (count as u64).div_ceil(256), height: 1, depth: 1 },
                        metal::MTLSize { width: 256, height: 1, depth: 1 },
                    );
                    e.set_compute_pipeline_state(&pipe);
                    e.set_buffer(0, Some(&sbuf), 0);
                    e.set_buffer(1, Some(&fbuf), 0);
                    e.set_buffer(2, Some(w), 0);
                    e.set_buffer(3, Some(o), 0);
                    let n_sb_u32 = N_SB as u32;
                    let rows_u32 = n_rows as u32;
                    let cap = SOFTCAP;
                    e.set_bytes(4, 4, &n_sb_u32 as *const u32 as *const _);
                    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
                    e.set_bytes(6, 4, &cap as *const f32 as *const _);
                    e.dispatch_thread_groups(
                        metal::MTLSize {
                            width: n_rows.div_ceil(sg * rb) as u64,
                            height: 1,
                            depth: 1,
                        },
                        metal::MTLSize { width: 32 * sg as u64, height: 1, depth: 1 },
                    );
                };
                // exactness gate
                let cb = refs.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                encode_reference(e, &refs, &sbuf, &qbuf, &swbuf, &sout_ref, SROWS, k, SOFTCAP, false);
                encode(e, &swbuf, &sout_new, SROWS);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let mut a = vec![0f32; k * SROWS];
                let mut b = vec![0f32; k * SROWS];
                read_buffer_f32(&sout_ref, &mut a);
                read_buffer_f32(&sout_new, &mut b);
                let bad = (0..k * SROWS).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
                if ablate == 0 {
                    assert_eq!(bad, 0, "rb={rb} rg={rg} sg={sg} flat={flat} yfmt={yfmt} K={k} not bitwise exact");
                }

                for pass in 0..2 {
                    let cb = refs.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    let reps = if pass == 0 { 1 } else { REPS };
                    for _ in 0..reps {
                        encode(e, &wbuf, &out, ROWS);
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    if pass == 1 {
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb);
                        let ms = gpu_us as f64 / 1000.0 / REPS as f64;
                        eprintln!(
                            "[spec50] sweep rb={rb} rg={rg} sg={sg} flat={flat} ablate={ablate} yfmt={yfmt} K={k}: {ms:7.2} ms  {:6.1} GB/s",
                            (bytes as f64 / 1.0e9) / (ms / 1000.0)
                        );
                    }
                }
            }
        }
    }

    /// 26B-shaped benchmark: the full 262144 x 2816 Q6_K table, 30 encodes of
    /// the head per measurement (a round's worth of layer time), reporting GPU
    /// ms and effective GB/s for the existing kernels and the new one.
    #[test]
    fn spec50_bench_full_26b_head() {
        const ROWS: usize = 262_144;
        const REPS: usize = 30;
        let refs = reference_kernels();
        let kernels = spec50_head_kernels().expect("spec50 pipelines");
        let bytes = ROWS * N_SB * Q6K_WIRE;
        let wbuf = shared(&refs.device, bytes);
        {
            let mut rng = Rng(0x2718_2818_2845_9045);
            // Fill in place: no 605 MB host-side staging copy.
            let dst = unsafe { std::slice::from_raw_parts_mut(wbuf.contents().cast::<u8>(), bytes) };
            for block in dst.chunks_exact_mut(Q6K_WIRE) {
                fill_q6k_block(&mut rng, block);
            }
        }
        let mut rng = Rng(0x3141_5926_5358_9793);
        let (scales, quants) = build_activations(&mut rng, 8);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, HIDDEN));
        let out = shared(&refs.device, 8 * ROWS * 4);
        let ids = shared(&refs.device, 8 * 4);
        let vals = shared(&refs.device, 8 * 4);

        let gb = bytes as f64 / 1.0e9;
        eprintln!(
            "[spec50] 26B head bench: {ROWS} rows x {HIDDEN} Q6_K = {:.1} MB, {REPS} encodes/measure",
            bytes as f64 / 1.0e6
        );
        for &k in &[1usize, 2, 4, 6, 8] {
            let run = |label: &str, new: bool, with_argmax: bool| {
                // Warm-up encode, then the measured command buffer.
                for pass in 0..2 {
                    let cb = refs.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    let reps = if pass == 0 { 1 } else { REPS };
                    for _ in 0..reps {
                        if new {
                            assert!(encode_q6k_spec50_batch(
                                e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out, N_SB, ROWS, k,
                                HIDDEN, SOFTCAP,
                            ));
                            if with_argmax {
                                encode_q6k_spec50_argmax(e, kernels, &out, &ids, &vals, ROWS, k);
                            }
                        } else {
                            encode_reference(
                                e, &refs, &sbuf, &qbuf, &wbuf, &out, ROWS, k, SOFTCAP, false,
                            );
                        }
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                    if pass == 1 {
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb);
                        let ms = gpu_us as f64 / 1000.0 / REPS as f64;
                        eprintln!(
                            "[spec50] K={k} {label:<26} {ms:7.2} ms  {:6.1} GB/s",
                            gb / (ms / 1000.0)
                        );
                    }
                }
            };
            let old_label = if k == 1 {
                "OLD q6k_linear_turbo"
            } else if k == 8 {
                "OLD ..._batch_k8"
            } else {
                "OLD ..._batch_k"
            };
            run(old_label, false, false);
            run("NEW q6k_spec50_batch", true, false);
            run("NEW spec50 + argmax", true, true);
        }
    }
}
