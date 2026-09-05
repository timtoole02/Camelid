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

// Per-candidate-row argmax with the established dense Gemma GPU semantics:
// Rust's `Iterator::max_by(|a, b| a.1.total_cmp(b.1))` keeps the later item on
// equality, so the HIGHEST vocabulary id wins an exact tie.  The signed key
// transform below is the one used by `f32::total_cmp`; retaining it also makes
// signed zero deterministic instead of treating +0 and -0 as interchangeable.
// It reads the SOFTCAPPED logits the projection wrote. Softcap can collapse
// distinct pre-cap values onto one float, which makes the tie rule observable.
inline int spec50_total_order_key(float value) {
    int key = as_type<int>(value);
    key ^= int(uint(key >> 31) >> 1);
    return key;
}

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
    int best_key = spec50_total_order_key(best);
    for (uint i = tid; i < count; i += tg_size) {
        const float v = base[i];
        const int key = spec50_total_order_key(v);
        if (best_i == 0xffffffffu || key > best_key ||
            (key == best_key && i > best_i)) {
            best = v;
            best_i = i;
            best_key = key;
        }
    }
    sh_val[tid] = best;
    sh_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float ov = sh_val[tid + s];
            const uint oi = sh_idx[tid + s];
            const int other_key = spec50_total_order_key(ov);
            const int this_key = spec50_total_order_key(sh_val[tid]);
            if (oi != 0xffffffffu &&
                (sh_idx[tid] == 0xffffffffu || other_key > this_key ||
                 (other_key == this_key && oi > sh_idx[tid]))) {
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

/// One lane geometry of the SPEC50 head. The five values are prepended to the
/// shader source as `#define`s at compile time AND drive the host-side grid, so
/// the two cannot drift apart. None of them changes the arithmetic (see the
/// module comment: the per-lane unit partition, the fold order and the
/// accumulate expression are all held fixed), which is exactly why they may be
/// swept and selected at run time; every geometry still has to pass the
/// bit-exact gates (`spec50_batch_is_bitwise_identical_*`) before it is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Spec50Geometry {
    pub(crate) name: &'static str,
    /// Vocab rows each simdgroup owns; mirrors `SPEC50_ROWS_PER_SG`. One
    /// threadgroup covers `sg_per_tg * rows_per_sg` rows.
    pub(crate) rows_per_sg: usize,
    /// Rows a single flat step handles together; mirrors `SPEC50_ROWS_PER_STEP`.
    pub(crate) rows_per_step: usize,
    /// Simdgroups per threadgroup; mirrors `SPEC50_SG_PER_TG`.
    pub(crate) sg_per_tg: usize,
    /// Mirrors `SPEC50_FLAT`: 1 = flat (row-group, unit) lane mapping.
    pub(crate) flat: u32,
    /// Activation repack format, mirroring `SPEC50_YFMT`: 0 = int8, 1 = f32,
    /// 2 = f16.
    pub(crate) yfmt: u32,
}

/// The established geometry, tuned on the 26B head (hidden 2816 = 44 units per
/// row): eight rows per simdgroup, one row per flat step, four simdgroups per
/// threadgroup, flat mapping, f16 activations. This is the default and the
/// only geometry the default path ever compiles.
pub(crate) const SPEC50_GEOMETRY_DEFAULT: Spec50Geometry = Spec50Geometry {
    name: "26b",
    rows_per_sg: 8,
    rows_per_step: 1,
    sg_per_tg: 4,
    flat: 1,
    yfmt: 2,
};

/// Winner of `spec50_sweep_geometry_12b` on the 12B head (hidden 3840 = 60
/// units per row, 15 superblocks): two rows per flat step share one activation
/// fetch. K=8 19.1 ms vs 21.5 ms for `26b` on the 826 MB table (mini2);
/// bit-exact against the reference on the full table at K=1,2,4,8.
pub(crate) const SPEC50_GEOMETRY_12B: Spec50Geometry = Spec50Geometry {
    name: "12b",
    rows_per_sg: 8,
    rows_per_step: 2,
    sg_per_tg: 4,
    flat: 1,
    yfmt: 2,
};

/// Geometries selectable by name through `CAMELID_GEMMA4_SPEC50_GEOMETRY`.
/// The generic form `rb<R>-rg<G>-sg<S>-flat<F>-y<Y>` (for example
/// `rb8-rg2-sg4-flat1-y2`) is accepted as well, so a sweep candidate can be
/// driven in situ without a rebuild.
const SPEC50_NAMED_GEOMETRIES: &[Spec50Geometry] = &[SPEC50_GEOMETRY_DEFAULT, SPEC50_GEOMETRY_12B];

/// The geometry the default path uses for a head of `hidden` width: `12b` for
/// the 3,840-wide Gemma 4 12B head, the established `26b` for everything else.
pub(crate) fn spec50_default_geometry_for(hidden: usize) -> Spec50Geometry {
    if hidden == 3840 {
        SPEC50_GEOMETRY_12B
    } else {
        SPEC50_GEOMETRY_DEFAULT
    }
}

impl Spec50Geometry {
    /// Vocab rows one threadgroup covers.
    pub(crate) fn rows_per_tg(&self) -> usize {
        self.sg_per_tg * self.rows_per_sg
    }

    /// Threads per threadgroup (32-lane simdgroups).
    pub(crate) fn threads_per_tg(&self) -> usize {
        32 * self.sg_per_tg
    }

    /// Name of the repack kernel matching `yfmt`.
    fn expand_kernel(&self) -> &'static str {
        match self.yfmt {
            1 => "q6k_spec50_expand_f32",
            2 => "q6k_spec50_expand_f16",
            _ => "q6k_spec50_expand",
        }
    }

    /// The `#define` prologue prepended to `SPEC50_HEAD_SHADER`.
    fn shader_defines(&self) -> String {
        format!(
            "#define SPEC50_ROWS_PER_SG {}\n\
             #define SPEC50_ROWS_PER_STEP {}\n\
             #define SPEC50_FLAT {}\n\
             #define SPEC50_SG_PER_TG {}\n\
             #define SPEC50_YFMT {}\n",
            self.rows_per_sg, self.rows_per_step, self.flat, self.sg_per_tg, self.yfmt
        )
    }

    /// Shapes the shader templates can be instantiated with: `rows_per_step`
    /// must divide `rows_per_sg`, a threadgroup must fit Metal's 1024 threads.
    fn admitted(&self) -> bool {
        (1..=32).contains(&self.rows_per_sg)
            && self.rows_per_step >= 1
            && self.rows_per_sg.is_multiple_of(self.rows_per_step)
            && (1..=32).contains(&self.sg_per_tg)
            && self.flat <= 1
            && self.yfmt <= 2
    }

    /// Parse a selector: a name from [`SPEC50_NAMED_GEOMETRIES`] or the generic
    /// `rb<R>-rg<G>-sg<S>-flat<F>-y<Y>` form (every field required).
    pub(crate) fn parse(spec: &str) -> Option<Self> {
        let spec = spec.trim();
        if let Some(named) = SPEC50_NAMED_GEOMETRIES
            .iter()
            .find(|geometry| geometry.name.eq_ignore_ascii_case(spec))
        {
            return Some(*named);
        }
        let mut geometry = Spec50Geometry {
            name: "custom",
            rows_per_sg: 0,
            rows_per_step: 0,
            sg_per_tg: 0,
            flat: u32::MAX,
            yfmt: u32::MAX,
        };
        for field in spec.split('-') {
            let (prefix, digits) = field.split_at(field.find(|c: char| c.is_ascii_digit())?);
            let value: usize = digits.parse().ok()?;
            match prefix {
                "rb" => geometry.rows_per_sg = value,
                "rg" => geometry.rows_per_step = value,
                "sg" => geometry.sg_per_tg = value,
                "flat" => geometry.flat = u32::try_from(value).ok()?,
                "y" => geometry.yfmt = u32::try_from(value).ok()?,
                _ => return None,
            }
        }
        (geometry.flat != u32::MAX && geometry.yfmt != u32::MAX && geometry.admitted())
            .then_some(geometry)
    }
}

/// `CAMELID_GEMMA4_SPEC50_GEOMETRY`, parsed once: `Some` when it names an
/// admitted geometry, which then overrides the width-keyed default everywhere
/// (`26b` restores the pre-sweep geometry on the 12B head for an A/B).
fn spec50_env_geometry() -> Option<Spec50Geometry> {
    static SELECTED: OnceLock<Option<Spec50Geometry>> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        let spec = std::env::var("CAMELID_GEMMA4_SPEC50_GEOMETRY")
            .ok()
            .filter(|spec| !spec.trim().is_empty())?;
        let parsed = Spec50Geometry::parse(&spec);
        if parsed.is_none() {
            eprintln!(
                "[metal] CAMELID_GEMMA4_SPEC50_GEOMETRY={spec:?} is not admitted; keeping the width default"
            );
        }
        parsed
    })
}

/// The geometry a head of `hidden` width runs with: the environment override
/// when set, else [`spec50_default_geometry_for`].
pub(crate) fn spec50_selected_geometry_for(hidden: usize) -> Spec50Geometry {
    spec50_env_geometry().unwrap_or_else(|| spec50_default_geometry_for(hidden))
}

/// Compiled pipelines for the speculative Q6_K tied head.
pub(crate) struct Spec50HeadKernels {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    /// Geometry these pipelines were compiled with; the encode grid follows it.
    pub(crate) geometry: Spec50Geometry,
    /// Index `k - 1` for `k` in `1..=8`.
    batch: [ComputePipelineState; 8],
    expand: ComputePipelineState,
    argmax: ComputePipelineState,
}

/// One compiled pipeline set per geometry, kept for the life of the process
/// (a failed compile is remembered too, so the diagnostic prints once).
static SPEC50_HEAD_KERNELS: OnceLock<Mutex<Vec<(Spec50Geometry, Option<&'static Spec50HeadKernels>)>>> =
    OnceLock::new();

/// The pipelines for a head of `hidden` width (see
/// [`spec50_selected_geometry_for`]). This is the production accessor.
pub(crate) fn spec50_head_kernels_for(hidden: usize) -> Option<&'static Spec50HeadKernels> {
    spec50_head_kernels_with(spec50_selected_geometry_for(hidden))
}

/// The pipelines of the established `26b` geometry (or the environment
/// override): the accessor for callers that do not know their head width.
pub(crate) fn spec50_head_kernels() -> Option<&'static Spec50HeadKernels> {
    spec50_head_kernels_with(spec50_env_geometry().unwrap_or(SPEC50_GEOMETRY_DEFAULT))
}

/// Compile (once per geometry) the speculative Q6_K head library. Returns
/// `None` and prints the compiler diagnostic on failure so the caller can keep
/// the existing lane.
pub(crate) fn spec50_head_kernels_with(geometry: Spec50Geometry) -> Option<&'static Spec50HeadKernels> {
    let cache = SPEC50_HEAD_KERNELS.get_or_init(|| Mutex::new(Vec::new()));
    let mut cache = cache.lock().ok()?;
    if let Some((_, kernels)) = cache.iter().find(|(cached, _)| *cached == geometry) {
        return *kernels;
    }
    let kernels = compile_spec50_head_kernels(geometry)
        .map(|kernels| &*Box::leak(Box::new(kernels)));
    cache.push((geometry, kernels));
    kernels
}

fn compile_spec50_head_kernels(geometry: Spec50Geometry) -> Option<Spec50HeadKernels> {
    let device = Device::system_default()?;
    eprintln!(
        "[metal] spec50 head geometry {:?}: rb{} rg{} sg{} flat{} y{}",
        geometry.name,
        geometry.rows_per_sg,
        geometry.rows_per_step,
        geometry.sg_per_tg,
        geometry.flat,
        geometry.yfmt,
    );
    // Same options as STRICT_Q8K_SHADER, which owns the kernels this
    // replaces: fast math off, so the surviving float expressions and
    // the softcap `tanh` compile identically.
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let source = format!("{}{SPEC50_HEAD_SHADER}", geometry.shader_defines());
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
            .any(|p| p.max_total_threads_per_threadgroup() < geometry.threads_per_tg() as u64)
    {
        eprintln!("[metal] spec50 head: simd width or threadgroup size not admitted");
        return None;
    }
    let expand = pipeline(geometry.expand_kernel())?;
    let argmax = pipeline("q6k_spec50_argmax_rows")?;
    if argmax.max_total_threads_per_threadgroup() < 1024 {
        eprintln!("[metal] spec50 head: argmax threadgroup < 1024");
        return None;
    }
    let queue = device.new_command_queue();
    Some(Spec50HeadKernels {
        device,
        queue,
        geometry,
        batch,
        expand,
        argmax,
    })
}

/// Bytes the integrator must allocate for the one new buffer, `activation_perm`:
/// a coalescing-friendly repack of the Q8_K activation quants. Sized for the
/// widest repack format (f32, 4 bytes per element: 180 KB at the 26B row's
/// `max_k = 16`, `hidden = 2816`) so one allocation serves every selectable
/// geometry. Shared storage, written GPU-side by the expand kernel at the head
/// of every encode.
pub(crate) fn spec50_activation_scratch_bytes(max_k: usize, hidden: usize) -> usize {
    max_k * hidden * 4
}


/// An exact matrix-unit K8 head, independently compiled only when requested.
/// K1/2/4 retain the existing kernels; the two-pass unit fold is admitted only
/// for the 12B head (15 superblocks / 60 units). The shader's final transpose
/// restores the original lane assignment before the same `simd_sum` runs.
struct Spec50MmaHeadKernels {
    expand: ComputePipelineState,
    batch: ComputePipelineState,
}

fn spec50_mma_requested() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("CAMELID_GEMMA4_SPEC50_MMA").is_ok_and(|value| value == "1"))
}

fn spec50_mma_admitted(hidden: usize, n_superblocks: usize, k_batch: usize) -> bool {
    hidden == 3840 && n_superblocks == 15 && k_batch == 8
}

fn spec50_mma_head_kernels() -> Option<&'static Spec50MmaHeadKernels> {
    static KERNELS: OnceLock<Option<Spec50MmaHeadKernels>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = device
                .new_library_with_source(include_str!("spec50_mma_head.metal"), &options)
                .map_err(|err| eprintln!("[metal] SPEC50 MMA head compile failed: {err}"))
                .ok()?;
            let pipeline = |name: &str| {
                let function = library.get_function(name, None).ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .ok()
            };
            let expand = pipeline("q6k_spec50_mma_expand_f16")?;
            let batch = pipeline("q6k_spec50_mma_k8")?;
            if expand.max_total_threads_per_threadgroup() < 256
                || batch.thread_execution_width() != 32
                || batch.max_total_threads_per_threadgroup() < 128
            {
                return None;
            }
            eprintln!("[metal] SPEC50 MMA head: K8, 3840-wide, 4 simdgroups, exact unit fold");
            Some(Spec50MmaHeadKernels { expand, batch })
        })
        .as_ref()
}

/// One dispatch form of the exact 12B/K8 MMA head, selected by
/// `CAMELID_GEMMA4_SPEC50_HEAD_FORM`. Only the schedule changes: which
/// simdgroup owns which `unit0` (`simdgroups`), whether the instruction-reduced
/// decode sibling runs (`lean`), and whether both unit passes' device loads are
/// hoisted above their arithmetic (`prefetch`). Every value folds the same
/// per-cell integers in the same order and runs the same `simd_sum` and
/// softcap, which is what `spec50_head_form_variants_are_bitwise_identical`
/// gates. Unset selector = [`SPEC50_FORM_BASE`], which is the established
/// `q6k_spec50_mma_k8` compiled from an unprefixed `spec50_mma_head.metal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Spec50HeadForm {
    pub(crate) name: &'static str,
    /// Simdgroups per threadgroup: `MMA_SG` for the established body, the `SG`
    /// template argument for the lean sibling. A threadgroup still owns exactly
    /// eight vocabulary rows, so this trades threadgroup count for the
    /// simdgroups resident per byte of the 8 KB `partials` transpose.
    pub(crate) simdgroups: usize,
    /// 0 = `q6k_spec50_mma_k8`, 1 = `q6k_spec50_mma_lean_k8_sg*_pf*`.
    pub(crate) lean: u32,
    /// Lean only: 1 hoists both passes' loads (double-buffered unit fetch).
    pub(crate) prefetch: u32,
}

/// The established form: today's kernel, four simdgroups, no prefetch.
pub(crate) const SPEC50_FORM_BASE: Spec50HeadForm = Spec50HeadForm {
    name: "base",
    simdgroups: 4,
    lean: 0,
    prefetch: 0,
};

/// Forms selectable by name; the generic `sg<N>-lean<L>-pf<P>` form is accepted
/// too, so a bench candidate can be driven in situ without a rebuild.
const SPEC50_NAMED_FORMS: &[Spec50HeadForm] = &[
    SPEC50_FORM_BASE,
    Spec50HeadForm { name: "sg1", simdgroups: 1, lean: 0, prefetch: 0 },
    Spec50HeadForm { name: "sg2", simdgroups: 2, lean: 0, prefetch: 0 },
    Spec50HeadForm { name: "sg8", simdgroups: 8, lean: 0, prefetch: 0 },
    Spec50HeadForm { name: "sg16", simdgroups: 16, lean: 0, prefetch: 0 },
    Spec50HeadForm { name: "lean", simdgroups: 4, lean: 1, prefetch: 0 },
    Spec50HeadForm { name: "lean-pf", simdgroups: 4, lean: 1, prefetch: 1 },
    Spec50HeadForm { name: "lean-sg2", simdgroups: 2, lean: 1, prefetch: 0 },
    Spec50HeadForm { name: "lean-sg8", simdgroups: 8, lean: 1, prefetch: 0 },
    Spec50HeadForm { name: "lean-sg16", simdgroups: 16, lean: 1, prefetch: 0 },
    Spec50HeadForm { name: "lean-pf-sg2", simdgroups: 2, lean: 1, prefetch: 1 },
    Spec50HeadForm { name: "lean-pf-sg8", simdgroups: 8, lean: 1, prefetch: 1 },
    Spec50HeadForm { name: "lean-pf-sg16", simdgroups: 16, lean: 1, prefetch: 1 },
];

impl Spec50HeadForm {
    /// Threads per threadgroup (32-lane simdgroups).
    pub(crate) fn threads_per_tg(&self) -> usize {
        32 * self.simdgroups
    }

    /// Shapes the shader has entry points for: only the compiled `SG` values,
    /// and `prefetch` exists on the lean sibling alone.
    fn admitted(&self) -> bool {
        matches!(self.simdgroups, 1 | 2 | 4 | 8 | 16)
            && self.lean <= 1
            && self.prefetch <= 1
            && (self.lean == 1 || self.prefetch == 0)
    }

    /// Name of the K8 entry point this form dispatches.
    fn batch_kernel(&self) -> String {
        if self.lean == 1 {
            format!(
                "q6k_spec50_mma_lean_k8_sg{}_pf{}",
                self.simdgroups, self.prefetch
            )
        } else {
            "q6k_spec50_mma_k8".to_string()
        }
    }

    /// The `#define` prologue prepended to `spec50_mma_head.metal`. The base
    /// form at four simdgroups prepends nothing at all, so the default library
    /// is compiled from the file byte-for-byte.
    fn shader_defines(&self) -> String {
        if self.lean == 0 && self.simdgroups != 4 {
            format!("#define MMA_SG {}\n", self.simdgroups)
        } else {
            String::new()
        }
    }

    /// Parse a selector: a name from [`SPEC50_NAMED_FORMS`] or the generic
    /// `sg<N>-lean<L>-pf<P>` form (every field required).
    pub(crate) fn parse(spec: &str) -> Option<Self> {
        let spec = spec.trim();
        if let Some(named) = SPEC50_NAMED_FORMS
            .iter()
            .find(|form| form.name.eq_ignore_ascii_case(spec))
        {
            return Some(*named);
        }
        let mut form = Spec50HeadForm {
            name: "custom",
            simdgroups: 0,
            lean: u32::MAX,
            prefetch: u32::MAX,
        };
        for field in spec.split('-') {
            let (prefix, digits) = field.split_at(field.find(|c: char| c.is_ascii_digit())?);
            let value: usize = digits.parse().ok()?;
            match prefix {
                "sg" => form.simdgroups = value,
                "lean" => form.lean = u32::try_from(value).ok()?,
                "pf" => form.prefetch = u32::try_from(value).ok()?,
                _ => return None,
            }
        }
        (form.lean != u32::MAX && form.prefetch != u32::MAX && form.admitted()).then_some(form)
    }
}

/// `CAMELID_GEMMA4_SPEC50_HEAD_FORM`, parsed once. Unset, empty or not admitted
/// keeps [`SPEC50_FORM_BASE`], i.e. today's dispatch byte-for-byte.
pub(crate) fn spec50_head_form() -> Spec50HeadForm {
    static SELECTED: OnceLock<Spec50HeadForm> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        let Some(spec) = std::env::var("CAMELID_GEMMA4_SPEC50_HEAD_FORM")
            .ok()
            .filter(|spec| !spec.trim().is_empty())
        else {
            return SPEC50_FORM_BASE;
        };
        match Spec50HeadForm::parse(&spec) {
            Some(form) => {
                eprintln!(
                    "[metal] SPEC50 head form {:?}: sg{} lean{} pf{}",
                    form.name, form.simdgroups, form.lean, form.prefetch
                );
                form
            }
            None => {
                eprintln!(
                    "[metal] CAMELID_GEMMA4_SPEC50_HEAD_FORM={spec:?} is not admitted; keeping the established head"
                );
                SPEC50_FORM_BASE
            }
        }
    })
}

/// One compiled pipeline set per form, kept for the life of the process (a
/// failed compile is remembered too, so the diagnostic prints once). The base
/// form is delegated to [`spec50_mma_head_kernels`] so the established lane
/// keeps its own single `OnceLock` and its unprefixed source.
static SPEC50_MMA_FORM_KERNELS: OnceLock<
    Mutex<Vec<(Spec50HeadForm, Option<&'static Spec50MmaHeadKernels>)>>,
> = OnceLock::new();

fn spec50_mma_head_kernels_for(form: Spec50HeadForm) -> Option<&'static Spec50MmaHeadKernels> {
    if form == SPEC50_FORM_BASE {
        return spec50_mma_head_kernels();
    }
    if !form.admitted() {
        return None;
    }
    let cache = SPEC50_MMA_FORM_KERNELS.get_or_init(|| Mutex::new(Vec::new()));
    let mut cache = cache.lock().ok()?;
    if let Some((_, kernels)) = cache.iter().find(|(cached, _)| *cached == form) {
        return *kernels;
    }
    let kernels = compile_spec50_mma_head_kernels(form).map(|k| &*Box::leak(Box::new(k)));
    cache.push((form, kernels));
    kernels
}

fn compile_spec50_mma_head_kernels(form: Spec50HeadForm) -> Option<Spec50MmaHeadKernels> {
    let device = Device::system_default()?;
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let source = format!(
        "{}{}",
        form.shader_defines(),
        include_str!("spec50_mma_head.metal")
    );
    let library = device
        .new_library_with_source(&source, &options)
        .map_err(|err| {
            eprintln!(
                "[metal] SPEC50 MMA head form {:?} compile failed: {err}",
                form.name
            )
        })
        .ok()?;
    let pipeline = |name: &str| {
        let function = library
            .get_function(name, None)
            .map_err(|err| eprintln!("[metal] SPEC50 MMA head missing {name}: {err}"))
            .ok()?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| eprintln!("[metal] SPEC50 MMA head pipeline {name}: {err}"))
            .ok()
    };
    let expand = pipeline("q6k_spec50_mma_expand_f16")?;
    let batch = pipeline(&form.batch_kernel())?;
    if expand.max_total_threads_per_threadgroup() < 256
        || batch.thread_execution_width() != 32
        || batch.max_total_threads_per_threadgroup() < form.threads_per_tg() as u64
    {
        eprintln!(
            "[metal] SPEC50 MMA head form {:?}: simd width or threadgroup size not admitted",
            form.name
        );
        return None;
    }
    Some(Spec50MmaHeadKernels { expand, batch })
}

/// W16 may opt into one dual-group head instead of two ordered K8 calls.
/// K8 and every other geometry retain their established path. A failed K16
/// compilation keeps W16 on its original two-chunk fallback before encoding.
struct Spec50Mma16HeadKernels {
    expand: ComputePipelineState,
    batch4: ComputePipelineState,
    batch8: ComputePipelineState,
}

fn spec50_mma16_simdgroups() -> usize {
    static SG: OnceLock<usize> = OnceLock::new();
    *SG.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_SPEC50_MMA_K16_SIMDGROUPS")
            .ok().and_then(|value| value.parse::<usize>().ok())
            .filter(|value| matches!(value, 4 | 8)).unwrap_or(8)
    })
}

pub(crate) fn spec50_mma16_available(hidden: usize) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    hidden == 3840 && *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_SPEC50_MMA_K16").is_ok_and(|value| value == "1")
    }) && spec50_mma16_head_kernels().is_some()
}

fn spec50_mma16_head_kernels() -> Option<&'static Spec50Mma16HeadKernels> {
    static KERNELS: OnceLock<Option<Spec50Mma16HeadKernels>> = OnceLock::new();
    KERNELS.get_or_init(|| {
        let device = Device::system_default()?;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device.new_library_with_source(include_str!("spec50_mma_head.metal"), &options)
            .map_err(|err| eprintln!("[metal] SPEC50 K16 MMA head compile failed: {err}")).ok()?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&function).ok()
        };
        let expand = pipeline("q6k_spec50_mma_expand16_f16")?;
        let batch4 = pipeline("q6k_spec50_mma_k16_sg4")?;
        let batch8 = pipeline("q6k_spec50_mma_k16_sg8")?;
        if expand.max_total_threads_per_threadgroup() < 256
            || batch4.thread_execution_width() != 32 || batch8.thread_execution_width() != 32
            || batch4.max_total_threads_per_threadgroup() < 128
            || batch8.max_total_threads_per_threadgroup() < 256 {
            return None;
        }
        eprintln!("[metal] SPEC50 K16 MMA head: 3840-wide, {} simdgroups, exact dual-group fold", spec50_mma16_simdgroups());
        Some(Spec50Mma16HeadKernels { expand, batch4, batch8 })
    }).as_ref()
}

/// Encode only after the exact 12B/K16 shape guard. Compiling failure encodes
/// nothing, allowing the established head to remain the fallback.
#[allow(clippy::too_many_arguments)]
fn encode_q6k_spec50_mma16(
    encoder: &metal::ComputeCommandEncoderRef,
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
    simdgroups: usize,
) -> bool {
    if !matches!(simdgroups, 4 | 8) || rows == 0 || !(hidden == 3840 && n_superblocks == 15 && k_batch == 16) {
        return false;
    }
    let Some(kernels) = spec50_mma16_head_kernels() else {
        return false;
    };
    let n_sb = n_superblocks as u32;
    let k = k_batch as u32;
    let rows_u32 = rows as u32;
    encoder.set_compute_pipeline_state(&kernels.expand);
    encoder.set_buffer(0, Some(input_quants), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_bytes(2, 4, &n_sb as *const u32 as *const _);
    encoder.set_bytes(3, 4, &k as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (k_batch * hidden).div_ceil(256) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.set_compute_pipeline_state(if simdgroups == 4 { &kernels.batch4 } else { &kernels.batch8 });
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_buffer(2, Some(weight), weight_offset);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(4, 4, &n_sb as *const u32 as *const _);
    encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &softcap as *const f32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: rows.div_ceil(8) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: (32 * simdgroups) as u64,
            height: 1,
            depth: 1,
        },
    );
    true
}

/// Encode only after the exact 12B/K8 shape guard, in the established form.
/// Compiling failure encodes nothing, allowing the established head to remain
/// the fallback.
#[allow(clippy::too_many_arguments)]
fn encode_q6k_spec50_mma8(
    encoder: &metal::ComputeCommandEncoderRef,
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
    encode_q6k_spec50_mma8_form(
        encoder,
        input_scales,
        input_quants,
        activation_perm,
        weight,
        weight_offset,
        output,
        n_superblocks,
        rows,
        k_batch,
        hidden,
        softcap,
        SPEC50_FORM_BASE,
    )
}

/// The same encode in an explicit [`Spec50HeadForm`]. Only the K8 batch
/// pipeline and its threads-per-threadgroup change; the grid still covers
/// `rows / 8` eight-row tiles, and the expand repack is form-independent.
#[allow(clippy::too_many_arguments)]
fn encode_q6k_spec50_mma8_form(
    encoder: &metal::ComputeCommandEncoderRef,
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
    form: Spec50HeadForm,
) -> bool {
    if rows == 0 || !spec50_mma_admitted(hidden, n_superblocks, k_batch) {
        return false;
    }
    let Some(kernels) = spec50_mma_head_kernels_for(form) else {
        return false;
    };
    let n_sb = n_superblocks as u32;
    let k = k_batch as u32;
    let rows_u32 = rows as u32;
    encoder.set_compute_pipeline_state(&kernels.expand);
    encoder.set_buffer(0, Some(input_quants), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_bytes(2, 4, &n_sb as *const u32 as *const _);
    encoder.set_bytes(3, 4, &k as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (k_batch * hidden).div_ceil(256) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.set_compute_pipeline_state(&kernels.batch);
    encoder.set_buffer(0, Some(input_scales), 0);
    encoder.set_buffer(1, Some(activation_perm), 0);
    encoder.set_buffer(2, Some(weight), weight_offset);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(4, 4, &n_sb as *const u32 as *const _);
    encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &softcap as *const f32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: rows.div_ceil(8) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: form.threads_per_tg() as u64,
            height: 1,
            depth: 1,
        },
    );
    true
}

/// Encode the K<=8 speculative Q6_K tied-head projection, or the independently
/// enabled exact 12B K16 MMA sibling.
///
/// Buffer order on `q6k_spec50_batch_k{K}`:
///   0 `input_scales`     `K * n_superblocks` f32 (candidate-major)
///   1 `activation_perm`  `K * hidden` bytes of scratch, filled by this call
///   2 `weight_blocks`    Q6_K table at `weight_offset`
///   3 `output`           `K * rows` f32, softcapped
///   4 `n_sb` u32, 5 `rows` u32, 6 `softcap` f32
///
/// Returns `false` (encoding nothing) when the pipelines are unavailable or
/// `k_batch` is outside `1..=8` and the opt-in 12B K16 sibling is unavailable;
/// the caller retains its established fallback.
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
    encode_q6k_spec50_batch_with_form(
        encoder,
        kernels,
        input_scales,
        input_quants,
        activation_perm,
        weight,
        weight_offset,
        output,
        n_superblocks,
        rows,
        k_batch,
        hidden,
        softcap,
        spec50_head_form(),
    )
}

/// [`encode_q6k_spec50_batch`] with the MMA form supplied explicitly instead of
/// read from the environment, so one process can drive several forms (the
/// GPU-vs-GPU equality test and the geometry bench both do).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q6k_spec50_batch_with_form(
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
    form: Spec50HeadForm,
) -> bool {
    if k_batch == 16 && spec50_mma16_available(hidden) {
        return encode_q6k_spec50_mma16(
            encoder, input_scales, input_quants, activation_perm, weight,
            weight_offset, output, n_superblocks, rows, k_batch, hidden, softcap,
            spec50_mma16_simdgroups(),
        );
    }
    if k_batch == 0 || k_batch > 8 || rows == 0 || n_superblocks == 0 {
        return false;
    }
    if spec50_mma_requested() && encode_q6k_spec50_mma8_form(
        encoder, input_scales, input_quants, activation_perm, weight,
        weight_offset, output, n_superblocks, rows, k_batch, hidden, softcap, form,
    ) {
        return true;
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

    encoder.set_compute_pipeline_state(&kernels.batch[k_batch - 1]);
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
            width: rows.div_ceil(kernels.geometry.rows_per_tg()) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: kernels.geometry.threads_per_tg() as u64,
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
        build_weights_for(rng, rows, N_SB)
    }

    fn build_weights_for(rng: &mut Rng, rows: usize, n_superblocks: usize) -> Vec<u8> {
        let mut w = vec![0u8; rows * n_superblocks * Q6K_WIRE];
        for block in w.chunks_exact_mut(Q6K_WIRE) {
            fill_q6k_block(rng, block);
        }
        w
    }

    fn build_activations(rng: &mut Rng, k: usize) -> (Vec<f32>, Vec<i8>) {
        build_activations_for(rng, k, HIDDEN, N_SB)
    }

    fn build_activations_for(
        rng: &mut Rng,
        k: usize,
        hidden: usize,
        n_superblocks: usize,
    ) -> (Vec<f32>, Vec<i8>) {
        let scales: Vec<f32> = (0..k * n_superblocks)
            .map(|_| 0.002 + (rng.next() % 4096) as f32 * 1.0e-6)
            .collect();
        let quants: Vec<i8> = (0..k * hidden).map(|_| rng.byte() as i8).collect();
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
        encode_reference_n_sb(
            encoder,
            refs,
            scales,
            quants,
            weight,
            output,
            rows,
            N_SB,
            k,
            softcap,
            force_batch_k,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_reference_n_sb(
        encoder: &metal::ComputeCommandEncoderRef,
        refs: &RefKernels,
        scales: &Buffer,
        quants: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        n_superblocks: usize,
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
        let n_sb_u32 = n_superblocks as u32;
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

    /// The production 12B target has a 3,840-wide tied head: fifteen Q6_K
    /// superblocks per vocabulary row. Keep that exact geometry in the gate;
    /// the original 26B campaign exercised only eleven superblocks.
    #[test]
    fn spec50_batch_is_bitwise_identical_at_12b_head_geometry() {
        const HIDDEN_12B: usize = 3840;
        const N_SB_12B: usize = HIDDEN_12B / 256;
        const ROWS: usize = 2051;

        let refs = reference_kernels();
        // The production accessor: the 12B width default (or the env override).
        let kernels = spec50_head_kernels_for(HIDDEN_12B).expect("spec50 pipelines");
        eprintln!("[spec50] 12B geometry under test: {:?}", kernels.geometry.name);
        let mut rng = Rng(0x12b0_3840_2621_4400);
        let weights = build_weights_for(&mut rng, ROWS, N_SB_12B);
        let (scales, quants) = build_activations_for(&mut rng, 8, HIDDEN_12B, N_SB_12B);
        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(
            &refs.device,
            spec50_activation_scratch_bytes(8, HIDDEN_12B),
        );

        for k in [1usize, 2, 4, 8] {
            let out_ref = shared(&refs.device, k * ROWS * 4);
            let out_new = shared(&refs.device, k * ROWS * 4);
            let cb = refs.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_reference_n_sb(
                encoder,
                &refs,
                &sbuf,
                &qbuf,
                &wbuf,
                &out_ref,
                ROWS,
                N_SB_12B,
                k,
                SOFTCAP,
                k == 1,
            );
            assert!(encode_q6k_spec50_batch(
                encoder,
                kernels,
                &sbuf,
                &qbuf,
                &fbuf,
                &wbuf,
                0,
                &out_new,
                N_SB_12B,
                ROWS,
                k,
                HIDDEN_12B,
                SOFTCAP,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);

            let mut expected = vec![0.0f32; k * ROWS];
            let mut actual = vec![0.0f32; k * ROWS];
            read_buffer_f32(&out_ref, &mut expected);
            read_buffer_f32(&out_new, &mut actual);
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "12B Q6_K head K={k} output index {index} diverged"
                );
            }
        }
    }

    /// Direct MMA gate: compare uncapped and capped values, including a partial
    /// row tile. Calling the MMA encoder directly prevents compile failure or
    /// a selector typo from silently validating the established fallback.
    #[test]
    fn spec50_mma8_is_bitwise_identical_at_12b_head_geometry() {
        const HIDDEN_12B: usize = 3840;
        const N_SB_12B: usize = HIDDEN_12B / 256;
        const ROWS: usize = 2051;

        let refs = reference_kernels();
        let mut rng = Rng(0x12b0_3840_2621_4400);
        let weights = build_weights_for(&mut rng, ROWS, N_SB_12B);
        let (scales, quants) = build_activations_for(&mut rng, 8, HIDDEN_12B, N_SB_12B);
        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, HIDDEN_12B));

        let k = 8usize;
        for softcap in [0.0, SOFTCAP] {
            let out_ref = shared(&refs.device, k * ROWS * 4);
            let out_new = shared(&refs.device, k * ROWS * 4);
            let cb = refs.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_reference_n_sb(
                encoder,
                &refs,
                &sbuf,
                &qbuf,
                &wbuf,
                &out_ref,
                ROWS,
                N_SB_12B,
                k,
                softcap,
                k == 1,
            );
            assert!(encode_q6k_spec50_mma8(
                encoder, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out_new, N_SB_12B, ROWS, k, HIDDEN_12B,
                softcap,
            ));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);

            let mut expected = vec![0.0f32; k * ROWS];
            let mut actual = vec![0.0f32; k * ROWS];
            read_buffer_f32(&out_ref, &mut expected);
            read_buffer_f32(&out_new, &mut actual);
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "12B Q6_K head K={k} output index {index} diverged"
                );
            }
        }
    }

    /// K16 reproduces two K8 calls, including cap ties and a ragged tile.
    #[test]
    fn spec50_mma16_is_bitwise_identical_to_two_mma8_calls() {
        const HIDDEN: usize = 3840;
        const N_SB: usize = 15;
        const ROWS: usize = 2051;
        let refs = reference_kernels();
        let mut rng = Rng(0x1616_3840_2051_7788);
        let weights = build_weights_for(&mut rng, ROWS, N_SB);
        let (scales, quants) = build_activations_for(&mut rng, 16, HIDDEN, N_SB);
        let wbuf = shared(&refs.device, weights.len());
        write_buffer_u8(&wbuf, &weights);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let perm = shared(&refs.device, spec50_activation_scratch_bytes(16, HIDDEN));
        let out = shared(&refs.device, 16 * ROWS * 4);
        for softcap in [0.0, SOFTCAP] {
            let mut expected = Vec::with_capacity(16 * ROWS);
            for group in 0..2 {
                let s8 = shared(&refs.device, 8 * N_SB * 4);
                write_buffer_f32(&s8, &scales[group * 8 * N_SB..(group + 1) * 8 * N_SB]);
                let q8 = shared(&refs.device, 8 * HIDDEN);
                write_buffer_i8(&q8, &quants[group * 8 * HIDDEN..(group + 1) * 8 * HIDDEN]);
                let out8 = shared(&refs.device, 8 * ROWS * 4);
                let cb = refs.queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                assert!(encode_q6k_spec50_mma8(encoder, &s8, &q8, &perm, &wbuf, 0,
                    &out8, N_SB, ROWS, 8, HIDDEN, softcap));
                encoder.end_encoding(); cb.commit(); cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let mut values = vec![0.0; 8 * ROWS];
                read_buffer_f32(&out8, &mut values);
                expected.extend(values);
            }
            for simdgroups in [4, 8] {
                let cb = refs.queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                assert!(encode_q6k_spec50_mma16(encoder, &sbuf, &qbuf, &perm, &wbuf, 0,
                    &out, N_SB, ROWS, 16, HIDDEN, softcap, simdgroups));
                encoder.end_encoding(); cb.commit(); cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let mut actual = vec![0.0; 16 * ROWS];
                read_buffer_f32(&out, &mut actual);
                for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    assert_eq!(actual.to_bits(), expected.to_bits(),
                        "K16 sg={simdgroups} cap={softcap} output={index}");
                }
            }
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

    /// Per-row argmax must equal the established dense Gemma GPU selector over
    /// the same softcapped logits (`max_by(total_cmp)`, highest id on ties).
    #[test]
    fn spec50_argmax_matches_established_total_cmp_maximum() {
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
            let (best_i, best) = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("non-empty logits row");
            assert_eq!(got_ids[t] as usize, best_i, "argmax id mismatch at token {t}");
            assert_eq!(
                got_vals[t].to_bits(),
                best.to_bits(),
                "argmax value mismatch at token {t}"
            );
        }
        // Softcap saturation can create exact ties.  Match Iterator::max_by by
        // selecting the later (higher vocabulary) id.
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
        assert_eq!(tie_id, 9, "exact ties must resolve to the highest index");

        // `total_cmp` distinguishes signed zero: +0 sorts above -0 even when
        // the -0 occurs at a later index.  This catches a tempting but wrong
        // numeric `>=` implementation of the established selector.
        let mut signed_zero = vec![-1.0f32; rows];
        signed_zero[7] = 0.0;
        signed_zero[9] = -0.0;
        write_buffer_f32(&tie_buf, &signed_zero);
        let cb = refs.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        encode_q6k_spec50_argmax(e, kernels, &tie_buf, &ids, &vals, rows, 1);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let zero_id = unsafe { *(ids.contents() as *const u32) };
        assert_eq!(zero_id, 7, "+0 must sort above a later -0");
        eprintln!(
            "[spec50] per-row argmax matches established total_cmp; exact ties select highest id"
        );
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
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
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

    /// Geometry sweep at an arbitrary head width. Every variant is first
    /// checked bitwise against the reference on a small table (a variant that
    /// is not exact is reported and NOT timed), then timed on a `rows`-row
    /// table. Prints GPU ms and effective GB/s per (geometry, K) for `ks`.
    /// `configs` entries are (rows/sg, rows/step, simdgroups/tg, flat, ablate,
    /// yfmt), the shader's `#define` order.
    fn sweep_geometry_at(
        hidden: usize,
        rows: usize,
        reps: usize,
        ks: &[usize],
        configs: &[(usize, usize, usize, u32, u32, u32)],
    ) {
        let n_sb = hidden / 256;
        let refs = reference_kernels();
        let bytes = rows * n_sb * Q6K_WIRE;
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
        let (scales, quants) = build_activations_for(&mut rng, 8, hidden, n_sb);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, hidden));
        let out = shared(&refs.device, 8 * rows * 4);

        // Small table for the per-variant exactness gate.
        const SROWS: usize = 1024;
        let mut srng = Rng(0x0f0f_1234_5678_9abc);
        let sweights = build_weights_for(&mut srng, SROWS, n_sb);
        let swbuf = shared(&refs.device, sweights.len());
        write_buffer_u8(&swbuf, &sweights);
        let sout_ref = shared(&refs.device, 8 * SROWS * 4);
        let sout_new = shared(&refs.device, 8 * SROWS * 4);

        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        eprintln!(
            "[spec50] sweep hidden={hidden} n_sb={n_sb} rows={rows} table={:.1} MB reps={reps}",
            bytes as f64 / 1.0e6
        );
        for &(rb, rg, sg, flat, ablate, yfmt) in configs {
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
            for &k in ks {
                let f = library
                    .get_function(&format!("q6k_spec50_batch_k{k}"), None)
                    .unwrap();
                let pipe = match refs.device.new_compute_pipeline_state_with_function(&f) {
                    Ok(pipe) => pipe,
                    Err(err) => {
                        eprintln!("[spec50] sweep rb={rb} rg={rg} sg={sg} flat={flat} yfmt={yfmt} K={k}: pipeline failed: {err}");
                        continue;
                    }
                };
                let encode = |e: &metal::ComputeCommandEncoderRef,
                              w: &Buffer,
                              o: &Buffer,
                              n_rows: usize| {
                    let count = (k * hidden) as u32;
                    let n_sb_e = n_sb as u32;
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
                    let n_sb_u32 = n_sb as u32;
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
                // Exactness gate: bitwise against the reference on the small table.
                let cb = refs.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                encode_reference_n_sb(
                    e, &refs, &sbuf, &qbuf, &swbuf, &sout_ref, SROWS, n_sb, k, SOFTCAP, false,
                );
                encode(e, &swbuf, &sout_new, SROWS);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let mut a = vec![0f32; k * SROWS];
                let mut b = vec![0f32; k * SROWS];
                read_buffer_f32(&sout_ref, &mut a);
                read_buffer_f32(&sout_new, &mut b);
                let bad = (0..k * SROWS).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
                if ablate == 0 && bad != 0 {
                    eprintln!("[spec50] sweep rb={rb} rg={rg} sg={sg} flat={flat} yfmt={yfmt} K={k}: NOT EXACT ({bad} values differ), not timed");
                    continue;
                }

                for pass in 0..2 {
                    let cb = refs.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    let n = if pass == 0 { 1 } else { reps };
                    for _ in 0..n {
                        encode(e, &wbuf, &out, rows);
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    if pass == 1 {
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
                        let ms = gpu_us as f64 / 1000.0 / reps as f64;
                        eprintln!(
                            "[spec50] sweep rb={rb} rg={rg} sg={sg} flat={flat} ablate={ablate} yfmt={yfmt} K={k}: {ms:7.2} ms  {:6.1} GB/s  exact",
                            (bytes as f64 / 1.0e9) / (ms / 1000.0)
                        );
                    }
                }
            }
        }
    }

    /// 12B geometry sweep: hidden 3840 = 60 units per row (fifteen
    /// superblocks), the full 262,144-row table (826 MB). Rows-per-simdgroup x
    /// rows-per-step x simdgroups-per-threadgroup x activation format, plus the
    /// blocked mapping. Run with `--ignored --nocapture`; the exact-and-fastest
    /// K=8 line is the `12b` entry of `SPEC50_NAMED_GEOMETRIES`.
    #[test]
    #[ignore]
    fn spec50_sweep_geometry_12b() {
        const HIDDEN_12B: usize = 3840;
        let mut configs = Vec::new();
        // Flat mapping, f16 activations: the main grid.
        for rb in [4usize, 8] {
            for rg in [1usize, 2, 4] {
                for sg in [2usize, 4, 8] {
                    configs.push((rb, rg, sg, 1u32, 0u32, 2u32));
                }
            }
        }
        // int8 / f32 activation formats on the middle of the grid.
        for yfmt in [0u32, 1] {
            for rb in [4usize, 8] {
                for rg in [1usize, 2] {
                    for sg in [2usize, 4] {
                        configs.push((rb, rg, sg, 1, 0, yfmt));
                    }
                }
            }
        }
        // Blocked mapping (60 units over 32 lanes is already 93.75% occupancy).
        for rb in [4usize, 8] {
            for sg in [2usize, 4, 8] {
                configs.push((rb, 1, sg, 0, 0, 2));
            }
        }
        // Wider and narrower row ownership per simdgroup.
        for rg in [1usize, 2, 4] {
            for sg in [2usize, 4] {
                configs.push((16, rg, sg, 1, 0, 2));
            }
        }
        for sg in [4usize, 8] {
            configs.push((2, 1, sg, 1, 0, 2));
        }
        sweep_geometry_at(HIDDEN_12B, 262_144, 10, &[4, 8], &configs);
    }

    /// 12B-shaped benchmark: the full 262144 x 3840 Q6_K table (826 MB)
    /// through the reference kernels and the SPEC50 kernels of the SELECTED
    /// geometry (honours `CAMELID_GEMMA4_SPEC50_GEOMETRY`), with and without
    /// the row argmax, 20 encodes per measurement. Every SPEC50 output is also
    /// checked bitwise against the reference on the full table.
    #[test]
    #[ignore]
    fn spec50_bench_full_12b_head() {
        const HIDDEN_12B: usize = 3840;
        const N_SB_12B: usize = HIDDEN_12B / 256;
        const ROWS: usize = 262_144;
        const REPS: usize = 20;
        let refs = reference_kernels();
        let kernels = spec50_head_kernels_for(HIDDEN_12B).expect("spec50 pipelines");
        let bytes = ROWS * N_SB_12B * Q6K_WIRE;
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
        let (scales, quants) = build_activations_for(&mut rng, 8, HIDDEN_12B, N_SB_12B);
        let sbuf = shared(&refs.device, scales.len() * 4);
        write_buffer_f32(&sbuf, &scales);
        let qbuf = shared(&refs.device, quants.len());
        write_buffer_i8(&qbuf, &quants);
        let fbuf = shared(&refs.device, spec50_activation_scratch_bytes(8, HIDDEN_12B));
        let out_ref = shared(&refs.device, 8 * ROWS * 4);
        let out = shared(&refs.device, 8 * ROWS * 4);
        let ids = shared(&refs.device, 8 * 4);
        let vals = shared(&refs.device, 8 * 4);

        let gb = bytes as f64 / 1.0e9;
        let g = kernels.geometry;
        eprintln!(
            "[spec50] 12B head bench: {ROWS} rows x {HIDDEN_12B} Q6_K = {:.1} MB, {REPS} encodes/measure, geometry {:?} rb{} rg{} sg{} flat{} y{}",
            bytes as f64 / 1.0e6, g.name, g.rows_per_sg, g.rows_per_step, g.sg_per_tg, g.flat, g.yfmt
        );
        for &k in &[1usize, 2, 4, 8] {
            // Full-table exactness of the selected geometry against the reference.
            {
                let cb = refs.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                encode_reference_n_sb(
                    e, &refs, &sbuf, &qbuf, &wbuf, &out_ref, ROWS, N_SB_12B, k, SOFTCAP, k == 1,
                );
                assert!(encode_q6k_spec50_batch(
                    e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out, N_SB_12B, ROWS, k,
                    HIDDEN_12B, SOFTCAP,
                ));
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let mut a = vec![0f32; k * ROWS];
                let mut b = vec![0f32; k * ROWS];
                read_buffer_f32(&out_ref, &mut a);
                read_buffer_f32(&out, &mut b);
                let bad = (0..k * ROWS).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
                assert_eq!(bad, 0, "geometry {:?} K={k}: {bad} logits differ from the reference", g.name);
            }
            let run = |label: &str, new: bool, with_argmax: bool| {
                for pass in 0..2 {
                    let cb = refs.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    let reps = if pass == 0 { 1 } else { REPS };
                    for _ in 0..reps {
                        if new {
                            assert!(encode_q6k_spec50_batch(
                                e, kernels, &sbuf, &qbuf, &fbuf, &wbuf, 0, &out, N_SB_12B, ROWS,
                                k, HIDDEN_12B, SOFTCAP,
                            ));
                            if with_argmax {
                                encode_q6k_spec50_argmax(e, kernels, &out, &ids, &vals, ROWS, k);
                            }
                        } else {
                            encode_reference_n_sb(
                                e, &refs, &sbuf, &qbuf, &wbuf, &out, ROWS, N_SB_12B, k, SOFTCAP,
                                false,
                            );
                        }
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                    if pass == 1 {
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
                        let ms = gpu_us as f64 / 1000.0 / REPS as f64;
                        eprintln!(
                            "[spec50] 12B K={k} {label:<26} {ms:7.2} ms  {:6.1} GB/s",
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

    /// The selector parser: names, the generic form, and refusals.
    #[test]
    fn spec50_geometry_selector_parses_names_and_generic_form() {
        assert_eq!(Spec50Geometry::parse("26b"), Some(SPEC50_GEOMETRY_DEFAULT));
        assert_eq!(Spec50Geometry::parse(" 26B "), Some(SPEC50_GEOMETRY_DEFAULT));
        let twelve = Spec50Geometry::parse("12b").expect("12b is named");
        assert!(twelve.admitted());
        assert_eq!(twelve, SPEC50_GEOMETRY_12B);
        // Width-keyed defaults: only the 3,840-wide 12B head moves to `12b`.
        assert_eq!(spec50_default_geometry_for(3840), SPEC50_GEOMETRY_12B);
        assert_eq!(spec50_default_geometry_for(2816), SPEC50_GEOMETRY_DEFAULT);
        assert_eq!(spec50_default_geometry_for(0), SPEC50_GEOMETRY_DEFAULT);
        let custom = Spec50Geometry::parse("rb8-rg2-sg4-flat1-y2").expect("generic form");
        assert_eq!(
            custom,
            Spec50Geometry { name: "custom", rows_per_sg: 8, rows_per_step: 2, sg_per_tg: 4, flat: 1, yfmt: 2 }
        );
        assert_eq!(custom.rows_per_tg(), 32);
        assert_eq!(custom.threads_per_tg(), 128);
        // rg must divide rb; every field is required; unknown fields refuse.
        assert!(Spec50Geometry::parse("rb8-rg3-sg4-flat1-y2").is_none());
        assert!(Spec50Geometry::parse("rb8-rg1-sg4-flat1").is_none());
        assert!(Spec50Geometry::parse("rb8-rg1-sg4-flat1-y3").is_none());
        assert!(Spec50Geometry::parse("rb8-rg1-sg4-flat2-y2").is_none());
        assert!(Spec50Geometry::parse("rb8-rg1-sg64-flat1-y2").is_none());
        assert!(Spec50Geometry::parse("zz8-rg1-sg4-flat1-y2").is_none());
        assert!(Spec50Geometry::parse("").is_none());
        // The default's defines are the text the established build prepended.
        assert_eq!(
            SPEC50_GEOMETRY_DEFAULT.shader_defines(),
            "#define SPEC50_ROWS_PER_SG 8\n#define SPEC50_ROWS_PER_STEP 1\n#define SPEC50_FLAT 1\n#define SPEC50_SG_PER_TG 4\n#define SPEC50_YFMT 2\n"
        );
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
                        let (gpu_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
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
