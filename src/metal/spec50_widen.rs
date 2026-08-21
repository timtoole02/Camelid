//! K=9..16 capacity widening for the speculative verify round (dense + MoE gateup).
//!
//! # Why
//!
//! The chained speculative round is capped at K=8 because the dense batch
//! kernels (`q4_0_block_linear_batch_k`, `q4_0_qkv_block_linear_batch_k`,
//! `q4_0_qkv_block_linear_batch_k_fused_rms`,
//! `q4_0_gateup_geglu_block_linear_batch_k`) hold fixed `float sums[4][8]`
//! (resp. `sum0[8]/sum1[8]`, `g_sums/u_sums[4][8]`) accumulators indexed by the
//! runtime `k_batch`; a wider chunk writes past the array and silently corrupts
//! the round. The routed-expert `gemma4_q4_multi_expert_fused_gateup_geglu_
//! quant_batch_k` has the same fixed `gate_acc[8]/up_acc[8]` shape (it refuses
//! `k_candidates > 8` instead of corrupting).
//!
//! # What this module adds
//!
//! `_k16` twins of four of those kernels (plain, QKV, GateUp, and the routed
//! MoE GateUp), selected ONLY when `k > 8`, so every K<=8 dispatch keeps
//! today's kernels byte for byte. The fused-RMS QKV kernel gets NO twin: its
//! verbatim `_k16` copy measured up to 492 ULP off the original at k_batch=8
//! (widening the accumulator array shifted the compiled FMA contraction of
//! the `(w * rms) * scale` chain), so it cannot satisfy the bitwise pinning
//! contract; `encode_gemma4_q4_0_qkv_matmul_batch_k_fused_rms` returns false
//! for K>8 instead, and callers use the separate-RMS + QKV path whose K>8
//! kernel IS verified. Each shipped twin's body is the generic kernel's body
//! VERBATIM with exactly these edits (asserted textually by
//! `spec50_widen_copies_are_verbatim`):
//!
//! * the kernel name gains a `k16` suffix;
//! * the fixed accumulator declarations widen from depth 8 to depth 16 (for
//!   the MoE kernel this includes its explicit zero-init loop bound);
//! * the MoE kernel's `k_candidates > 8u` refusal relaxes to `> 16u`.
//!
//! Nothing else changes: the runtime `k_batch` loops, the lane -> (block,
//! byte-quad) mapping, the accumulation order, the `simd_sum` folds and the
//! GeGLU expression are the reference kernels' own, so per-token arithmetic is
//! the already-oracle-verified program.
//!
//! # Compile options
//!
//! The library compiles with DEFAULT `CompileOptions`, matching
//! `LINEAR_ROW_SHADER` (which owns the kernels these widen) exactly — that
//! shader also uses `CompileOptions::new()` with no fast-math override.
//! Fast-math state changes reassociation (measured 16384 ULP in one kernel
//! family), so equivalence is not assumed: `spec50_widen_*_bitwise_at_k8`
//! asserts each `_k16` twin at `k_batch = 8` is bitwise identical
//! (`f32::to_bits`, 0 differences) to the existing generic pipeline at
//! `k_batch = 8` on random data, proving compile-option equivalence and body
//! fidelity in one shot. `spec50_widen_*_batch_independent` then pins K>8:
//! token `t` inside a K in {9,12,16} batch must be bit-identical to the same
//! token evaluated alone at K=1 through the EXISTING generic kernel.

#![allow(dead_code)]

use super::*;

const SPEC50_WIDEN_SHADER: &str = r#"
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
#define G4Q4_RECORD_BYTES 3345408ul
#define G4Q4_SLOT_STRIDE 3358720ul

struct Gemma4UniqueExpertWork {
    ulong candidate_mask;
    uint expert_weight_offset;
    uint slab_index;
};

kernel void q4_0_block_linear_batch_k16(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = tg * NR0;
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

    float sums[4][16] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 scaled_lo[4], scaled_hi[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const uint r = r0 + i;
                device const char* wb = weight_blocks + r * row_stride + ib * q4_block_bytes;
                const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                const uchar4 wq = *reinterpret_cast<device const uchar4*>(wb + 2 + ilb);
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
                if (lane == 0) {
                    output[k * rows + r0 + i] = tot;
                }
            }
        }
    }
}
kernel void q4_0_qkv_block_linear_batch_k16(
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
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint r0 = tg * NR0;
    if (r0 >= total_rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const bool has_row1 = (r0 + 1 < total_rows);

    device const char* w_base0;
    uint target_r0;
    uint target_kind0; // 0=Q, 1=K, 2=V
    if (r0 < q_rows) {
        w_base0 = q_weight;
        target_r0 = r0;
        target_kind0 = 0;
    } else if (r0 < q_rows + k_rows) {
        w_base0 = k_weight;
        target_r0 = r0 - q_rows;
        target_kind0 = 1;
    } else {
        w_base0 = v_weight;
        target_r0 = r0 - (q_rows + k_rows);
        target_kind0 = 2;
    }

    device const char* w_base1 = nullptr;
    uint target_r1 = 0;
    uint target_kind1 = 0;
    if (has_row1) {
        const uint r1 = r0 + 1;
        if (r1 < q_rows) {
            w_base1 = q_weight;
            target_r1 = r1;
            target_kind1 = 0;
        } else if (r1 < q_rows + k_rows) {
            w_base1 = k_weight;
            target_r1 = r1 - q_rows;
            target_kind1 = 1;
        } else {
            w_base1 = v_weight;
            target_r1 = r1 - (q_rows + k_rows);
            target_kind1 = 2;
        }
    }

    float sum0[16] = {0.0f};
    float sum1[16] = {0.0f};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        device const char* wb0 = w_base0 + target_r0 * row_stride + ib * q4_block_bytes;
        const float w_scale0 = float(*reinterpret_cast<device const half*>(wb0));
        const uchar4 wq0 = *reinterpret_cast<device const uchar4*>(wb0 + 2 + ilb);
        const float4 lo0 = float4(int4(wq0 & 0x0F) - 8);
        const float4 hi0 = float4(int4(wq0 >> 4) - 8);

        float w_scale1 = 0.0f;
        float4 lo1 = 0.0f;
        float4 hi1 = 0.0f;
        if (has_row1) {
            device const char* wb1 = w_base1 + target_r1 * row_stride + ib * q4_block_bytes;
            w_scale1 = float(*reinterpret_cast<device const half*>(wb1));
            const uchar4 wq1 = *reinterpret_cast<device const uchar4*>(wb1 + 2 + ilb);
            lo1 = float4(int4(wq1 & 0x0F) - 8);
            hi1 = float4(int4(wq1 >> 4) - 8);
        }

        for (uint k = 0; k < k_batch; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            sum0[k] += (dot(lo0, ylo) + dot(hi0, yhi)) * w_scale0;
            if (has_row1) {
                sum1[k] += (dot(lo1, ylo) + dot(hi1, yhi)) * w_scale1;
            }
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

kernel void q4_0_gateup_geglu_block_linear_batch_k16(
    device const float* y [[buffer(0)]],
    device const char* gate_weight [[buffer(1)]],
    device const char* up_weight [[buffer(2)]],
    device float* act_output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& k_batch [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint q4_block_bytes = 18;
    const uint r0 = tg * NR0;
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

    float g_sums[4][16] = {{0.0f}};
    float u_sums[4][16] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 g_scaled_lo[4], g_scaled_hi[4];
        float4 u_scaled_lo[4], u_scaled_hi[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const uint r = r0 + i;
                device const char* g_wb = gate_weight + r * row_stride + ib * q4_block_bytes;
                const float g_w_scale = float(*reinterpret_cast<device const half*>(g_wb));
                const uchar4 g_wq = *reinterpret_cast<device const uchar4*>(g_wb + 2 + ilb);
                const float4 g_lo = float4(int4(g_wq & 0x0F) - 8);
                const float4 g_hi = float4(int4(g_wq >> 4) - 8);
                g_scaled_lo[i] = g_lo * g_w_scale;
                g_scaled_hi[i] = g_hi * g_w_scale;

                device const char* u_wb = up_weight + r * row_stride + ib * q4_block_bytes;
                const float u_w_scale = float(*reinterpret_cast<device const half*>(u_wb));
                const uchar4 u_wq = *reinterpret_cast<device const uchar4*>(u_wb + 2 + ilb);
                const float4 u_lo = float4(int4(u_wq & 0x0F) - 8);
                const float4 u_hi = float4(int4(u_wq >> 4) - 8);
                u_scaled_lo[i] = u_lo * u_w_scale;
                u_scaled_hi[i] = u_hi * u_w_scale;
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
                    g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                    u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
                }
            }
        }
    }

    for (uint k = 0; k < k_batch; ++k) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            if (has_row[i]) {
                const float g_tot = simd_sum(g_sums[i][k]);
                const float u_tot = simd_sum(u_sums[i][k]);
                if (lane == 0) {
                    const float in_v = 0.7978845608f * (g_tot + 0.044715f * g_tot * g_tot * g_tot);
                    const float gelu = 0.5f * g_tot * (1.0f + tanh(clamp(in_v, -15.0f, 15.0f)));
                    act_output[k * rows + r0 + i] = gelu * u_tot;
                }
            }
        }
    }
}
kernel void gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k16(
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
    if (k_candidates == 0u || k_candidates > 16u) return;

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

        int4 wg_lo4[4], wg_hi4[4], wu_lo4[4], wu_hi4[4];
        #pragma unroll
        for (uint k = 0; k < 4; ++k) {
            int4 wg_lo, wg_hi, wu_lo, wu_hi;
            #pragma unroll
            for (uint m = 0; m < 4; ++m) {
                const uint l = k * 4 + m;
                const uchar bg = b_gate[2 + l];
                const uchar bu = b_up[2 + l];
                wg_lo[m] = int(bg & 0x0f) - 8;
                wg_hi[m] = int(bg >> 4) - 8;
                wu_lo[m] = int(bu & 0x0f) - 8;
                wu_hi[m] = int(bu >> 4) - 8;
            }
            wg_lo4[k] = wg_lo;
            wg_hi4[k] = wg_hi;
            wu_lo4[k] = wu_lo;
            wu_hi4[k] = wu_hi;
        }

        for (uint t = 0; t < k_candidates; ++t) {
            if ((mask & (1ULL << t)) == 0ULL) continue;
            device const char* x = input_quants + ulong(t) * G4Q4_HIDDEN + ulong(gb) * 32ul;
            device const char4* xlo4 = reinterpret_cast<device const char4*>(x);
            device const char4* xhi4 = reinterpret_cast<device const char4*>(x + 16);
            const float in_scale = input_scales[ulong(t) * G4Q4_GU_BLOCKS + gb];

            int isum_gate = 0;
            int isum_up = 0;
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const int4 xl = int4(xlo4[k]);
                const int4 xh = int4(xhi4[k]);
                const int4 pg = wg_lo4[k] * xl + wg_hi4[k] * xh;
                const int4 pu = wu_lo4[k] * xl + wu_hi4[k] * xh;
                isum_gate += (pg.x + pg.y) + (pg.z + pg.w);
                isum_up   += (pu.x + pu.y) + (pu.z + pu.w);
            }

            gate_acc[t] += (float(isum_gate) * w_scale_gate) * in_scale;
            up_acc[t]   += (float(isum_up) * w_scale_up) * in_scale;
        }
    }

    for (uint t = 0; t < k_candidates; ++t) {
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
            const ulong scale_idx = ulong(u) * ulong(k_candidates) * G4Q4_DOWN_BLOCKS + ulong(t) * G4Q4_DOWN_BLOCKS + ulong(b);
            output_scales[scale_idx] = stored_scale;
        }

        const int q = clamp(int(round(act_val * inverse)), -127, 127);
        const ulong quant_idx = ulong(u) * ulong(k_candidates) * G4Q4_FF + ulong(t) * G4Q4_FF + ulong(row);
        output_quants[quant_idx] = char(q);
    }
}

"#;

/// Compiled `_k16` pipelines. Only the `k > 8` branches of the encode
/// functions reach these; K<=8 never selects them.
pub(crate) struct Spec50WidenKernels {
    pub(crate) plain: ComputePipelineState,
    pub(crate) qkv: ComputePipelineState,
    pub(crate) gateup: ComputePipelineState,
    pub(crate) moe_gateup: ComputePipelineState,
}

static SPEC50_WIDEN_KERNELS: OnceLock<Option<Spec50WidenKernels>> = OnceLock::new();

/// Compile (once) the K=9..16 widened library. Returns `None` and prints the
/// compiler diagnostic on failure; callers must then refuse (or skip) the
/// K>8 dispatch rather than fall back to the fixed-depth kernels.
pub(crate) fn spec50_widen_kernels() -> Option<&'static Spec50WidenKernels> {
    SPEC50_WIDEN_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            // DEFAULT options: LINEAR_ROW_SHADER (the parent of the kernels
            // being widened) compiles with a bare CompileOptions::new() and no
            // fast-math override. The k8 bitwise guard tests assert the match.
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SPEC50_WIDEN_SHADER, &options)
                .map_err(|err| eprintln!("[metal] SPEC50_WIDEN shader compile failed: {err}"))
                .ok()?;
            let pipe = |name: &str| -> Option<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] spec50_widen missing {name}: {err}"))
                    .ok()?;
                let p = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] spec50_widen pipeline {name}: {err}"))
                    .ok()?;
                // Every kernel here is dispatched with 32-thread threadgroups
                // and uses simd_sum/simd_max over one simdgroup.
                if p.thread_execution_width() != 32 || p.max_total_threads_per_threadgroup() < 32 {
                    eprintln!("[metal] spec50_widen {name}: simd width not admitted");
                    return None;
                }
                Some(p)
            };
            Some(Spec50WidenKernels {
                plain: pipe("q4_0_block_linear_batch_k16")?,
                qkv: pipe("q4_0_qkv_block_linear_batch_k16")?,
                gateup: pipe("q4_0_gateup_geglu_block_linear_batch_k16")?,
                moe_gateup: pipe("gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k16")?,
            })
        })
        .as_ref()
}

fn tg32() -> metal::MTLSize {
    metal::MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    }
}

fn grid(width: u64) -> metal::MTLSize {
    metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// K=9..16 twin of the generic branch of `encode_gemma4_q4_0_matmul_batch_k`:
/// identical bindings and grid (4 rows per 32-thread threadgroup).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_widen_plain(
    e: &metal::ComputeCommandEncoderRef,
    k: &Spec50WidenKernels,
    y: &Buffer,
    weight: &Buffer,
    weight_offset: u64,
    output: &Buffer,
    blocks_per_row: u32,
    rows: usize,
    k_batch: usize,
) {
    debug_assert!((1..=16).contains(&k_batch));
    let rows_u32 = rows as u32;
    let k_batch_u32 = k_batch as u32;
    e.set_compute_pipeline_state(&k.plain);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), weight_offset);
    e.set_buffer(3, Some(output), 0);
    e.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    e.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(grid((rows as u64).div_ceil(4)), tg32());
}

/// K=9..16 twin of the generic branch of
/// `encode_gemma4_q4_0_qkv_matmul_batch_k`: identical bindings and grid
/// (2 concatenated Q+K+V rows per 32-thread threadgroup).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_widen_qkv(
    e: &metal::ComputeCommandEncoderRef,
    k: &Spec50WidenKernels,
    y: &Buffer,
    q_weight: &Buffer,
    k_weight: &Buffer,
    v_weight: &Buffer,
    query_out: &Buffer,
    key_out: &Buffer,
    val_out: &Buffer,
    scalars: (u32, u32, u32, u32),
    total_rows: usize,
    k_batch: usize,
) {
    debug_assert!((1..=16).contains(&k_batch));
    let (bpr_u32, q_rows_u32, k_rows_u32, v_rows_u32) = scalars;
    let k_batch_u32 = k_batch as u32;
    e.set_compute_pipeline_state(&k.qkv);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(1, Some(q_weight), 0);
    e.set_buffer(2, Some(k_weight), 0);
    e.set_buffer(3, Some(v_weight), 0);
    e.set_buffer(4, Some(query_out), 0);
    e.set_buffer(5, Some(key_out), 0);
    e.set_buffer(6, Some(val_out), 0);
    e.set_bytes(7, 4, &bpr_u32 as *const u32 as *const _);
    e.set_bytes(8, 4, &q_rows_u32 as *const u32 as *const _);
    e.set_bytes(9, 4, &k_rows_u32 as *const u32 as *const _);
    e.set_bytes(10, 4, &v_rows_u32 as *const u32 as *const _);
    e.set_bytes(11, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(grid((total_rows as u64).div_ceil(2)), tg32());
}

/// K=9..16 twin of the generic branch of
/// `encode_gemma4_q4_0_gateup_matmul_batch_k`: identical bindings and grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_widen_gateup(
    e: &metal::ComputeCommandEncoderRef,
    k: &Spec50WidenKernels,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    blocks_per_row: u32,
    rows: usize,
    k_batch: usize,
) {
    debug_assert!((1..=16).contains(&k_batch));
    let rows_u32 = rows as u32;
    let k_batch_u32 = k_batch as u32;
    e.set_compute_pipeline_state(&k.gateup);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(1, Some(gate_weight), 0);
    e.set_buffer(2, Some(up_weight), 0);
    e.set_buffer(3, Some(act_output), 0);
    e.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    e.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    e.dispatch_thread_groups(grid((rows as u64).div_ceil(4)), tg32());
}

/// K=9..16 twin of the fused routed-expert GateUp dispatch in
/// `encode_moe_topk_gateup_down`: identical bindings (including the optional
/// overflow slab on buffer 8) and grid (one 32-thread threadgroup per
/// (unique expert, 32-row FF block)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_widen_moe_gateup(
    e: &metal::ComputeCommandEncoderRef,
    k: &Spec50WidenKernels,
    input_scales: &Buffer,
    input_quants: &Buffer,
    slab: &Buffer,
    slab_byte_offset: u64,
    work_list: &Buffer,
    out_scales: &Buffer,
    out_quants: &Buffer,
    num_unique_experts: u32,
    k_candidates: u32,
    overflow_slab: Option<&Buffer>,
    ff_blocks_per_expert: usize,
) {
    debug_assert!((1..=16).contains(&(k_candidates as usize)));
    e.set_compute_pipeline_state(&k.moe_gateup);
    e.set_buffer(0, Some(input_scales), 0);
    e.set_buffer(1, Some(input_quants), 0);
    e.set_buffer(2, Some(slab), slab_byte_offset);
    e.set_buffer(3, Some(work_list), 0);
    e.set_buffer(4, Some(out_scales), 0);
    e.set_buffer(5, Some(out_quants), 0);
    e.set_bytes(6, 4, &num_unique_experts as *const u32 as *const _);
    e.set_bytes(7, 4, &k_candidates as *const u32 as *const _);
    e.set_buffer(8, overflow_slab.map(|v| &**v), 0);
    e.dispatch_thread_groups(
        grid((num_unique_experts as u64) * ff_blocks_per_expert as u64),
        tg32(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q4_BLOCK: usize = 18;

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

    struct Ctx {
        device: Device,
        queue: CommandQueue,
        old: &'static MetalLinearKernel,
        widen: &'static Spec50WidenKernels,
    }

    fn ctx() -> Option<Ctx> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let old = metal_linear_kernel()?;
        let widen = spec50_widen_kernels().expect("spec50_widen pipelines must compile");
        Some(Ctx {
            device,
            queue,
            old,
            widen,
        })
    }

    fn run<F: FnOnce(&metal::ComputeCommandEncoderRef)>(queue: &CommandQueue, f: F) {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        f(e);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
    }

    /// Count f32 bit mismatches; report the first and worst.
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
                max_ulp = max_ulp.max((n.to_bits() as i64 - o.to_bits() as i64).abs());
            }
        }
        if bad != 0 {
            let (i, n, o) = first.unwrap();
            eprintln!(
                "[spec50_widen] {name}: {bad}/{} bits differ, max |ulp| {max_ulp}, \
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

    /// Dispatch the EXISTING generic `q4_0_block_linear_batch_k` pipeline
    /// directly (never the k6/k8 specializations) at any k in 1..=8.
    #[allow(clippy::too_many_arguments)]
    fn plain_generic(
        c: &Ctx,
        y: &Buffer,
        y_off: u64,
        w: &Buffer,
        out: &Buffer,
        out_off: u64,
        rows: usize,
        blocks: usize,
        k: usize,
    ) {
        assert!(k <= 8);
        let pipe = c
            .old
            .q4_0_block_batch_k_pipeline
            .as_ref()
            .expect("generic batch_k pipeline");
        run(&c.queue, |e| {
            e.set_compute_pipeline_state(pipe);
            e.set_buffer(0, Some(y), y_off);
            e.set_buffer(2, Some(w), 0);
            e.set_buffer(3, Some(out), out_off);
            let bpr = blocks as u32;
            let rows_u32 = rows as u32;
            let k_u32 = k as u32;
            e.set_bytes(4, 4, &bpr as *const u32 as *const _);
            e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
            e.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
            e.dispatch_thread_groups(grid((rows as u64).div_ceil(4)), tg32());
        });
    }

    // -- plain ----------------------------------------------------------------

    /// (a) compile-option + body fidelity: the _k16 twin at k_batch=8 is
    /// bitwise identical to the generic pipeline at k_batch=8.
    #[test]
    fn spec50_widen_plain_bitwise_at_k8() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0xA5A5_1234_5678_9ABC);
        let mut bad = 0usize;
        for &(rows, blocks) in &[(64usize, 8usize), (37, 11), (130, 16), (66, 22)] {
            let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * 8));
            let out_new = zeros(&c.device, rows * 8);
            let out_old = zeros(&c.device, rows * 8);
            run(&c.queue, |e| {
                encode_spec50_widen_plain(e, c.widen, &y, &w, 0, &out_new, blocks as u32, rows, 8);
            });
            plain_generic(&c, &y, 0, &w, &out_old, 0, rows, blocks, 8);
            let new = read(&out_new, rows * 8);
            nonzero("plain k16@8", &new);
            bad += bits_equal(
                &format!("plain rows={rows} blocks={blocks}"),
                &new,
                &read(&out_old, rows * 8),
            );
        }
        assert_eq!(bad, 0, "plain _k16 at k=8 differs from the generic kernel");
    }

    /// (b) per-token independence: token t in a K in {9,12,16} batch equals
    /// the same token alone at K=1 through the EXISTING generic kernel.
    #[test]
    fn spec50_widen_plain_batch_independent() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0x1122_3344_5566_7788);
        let (rows, blocks) = (130usize, 22usize);
        let hidden = blocks * 32;
        let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let yv = random_f32(&mut rng, hidden * 16);
        let y = buf_from(&c.device, &yv);
        let mut solo = vec![0f32; 16 * rows];
        for t in 0..16usize {
            let out = zeros(&c.device, rows);
            plain_generic(
                &c,
                &y,
                (t * hidden * 4) as u64,
                &w,
                &out,
                0,
                rows,
                blocks,
                1,
            );
            solo[t * rows..(t + 1) * rows].copy_from_slice(&read(&out, rows));
        }
        nonzero("plain solo", &solo);
        for &k in &[9usize, 12, 16] {
            let out = zeros(&c.device, k * rows);
            run(&c.queue, |e| {
                encode_spec50_widen_plain(e, c.widen, &y, &w, 0, &out, blocks as u32, rows, k);
            });
            let got = read(&out, k * rows);
            let bad = bits_equal(
                &format!("plain independence K={k}"),
                &got,
                &solo[..k * rows],
            );
            assert_eq!(bad, 0, "plain K={k}: tokens depend on batch width");
        }
    }

    // -- qkv ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn qkv_generic(
        c: &Ctx,
        y: &Buffer,
        y_off: u64,
        qw: &Buffer,
        kw: &Buffer,
        vw: &Buffer,
        outs: (&Buffer, &Buffer, &Buffer),
        out_offs: (u64, u64, u64),
        splits: (usize, usize, usize),
        blocks: usize,
        k: usize,
    ) {
        assert!(k <= 8);
        let pipe = c
            .old
            .q4_0_qkv_block_batch_k_pipeline
            .as_ref()
            .expect("generic qkv batch_k pipeline");
        let total = splits.0 + splits.1 + splits.2;
        run(&c.queue, |e| {
            e.set_compute_pipeline_state(pipe);
            e.set_buffer(0, Some(y), y_off);
            e.set_buffer(1, Some(qw), 0);
            e.set_buffer(2, Some(kw), 0);
            e.set_buffer(3, Some(vw), 0);
            e.set_buffer(4, Some(outs.0), out_offs.0);
            e.set_buffer(5, Some(outs.1), out_offs.1);
            e.set_buffer(6, Some(outs.2), out_offs.2);
            let bpr = blocks as u32;
            let (q_r, k_r, v_r) = (splits.0 as u32, splits.1 as u32, splits.2 as u32);
            let k_u32 = k as u32;
            e.set_bytes(7, 4, &bpr as *const u32 as *const _);
            e.set_bytes(8, 4, &q_r as *const u32 as *const _);
            e.set_bytes(9, 4, &k_r as *const u32 as *const _);
            e.set_bytes(10, 4, &v_r as *const u32 as *const _);
            e.set_bytes(11, 4, &k_u32 as *const u32 as *const _);
            e.dispatch_thread_groups(grid((total as u64).div_ceil(2)), tg32());
        });
    }

    #[test]
    fn spec50_widen_qkv_bitwise_at_k8() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
        let splits = (64usize, 32usize, 32usize);
        let blocks = 11usize;
        let total = splits.0 + splits.1 + splits.2;
        let qw = buf_from(&c.device, &random_q4_0(&mut rng, splits.0, blocks));
        let kw = buf_from(&c.device, &random_q4_0(&mut rng, splits.1, blocks));
        let vw = buf_from(&c.device, &random_q4_0(&mut rng, splits.2, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * 8));
        let mk = |n: usize| (zeros(&c.device, n * 8), zeros(&c.device, n * 8));
        let (qo_new, qo_old) = mk(splits.0);
        let (ko_new, ko_old) = mk(splits.1);
        let (vo_new, vo_old) = mk(splits.2);
        run(&c.queue, |e| {
            encode_spec50_widen_qkv(
                e,
                c.widen,
                &y,
                &qw,
                &kw,
                &vw,
                &qo_new,
                &ko_new,
                &vo_new,
                (
                    blocks as u32,
                    splits.0 as u32,
                    splits.1 as u32,
                    splits.2 as u32,
                ),
                total,
                8,
            );
        });
        qkv_generic(
            &c,
            &y,
            0,
            &qw,
            &kw,
            &vw,
            (&qo_old, &ko_old, &vo_old),
            (0, 0, 0),
            splits,
            blocks,
            8,
        );
        let mut bad = 0usize;
        for (name, new, old, n) in [
            ("qkv Q", &qo_new, &qo_old, splits.0),
            ("qkv K", &ko_new, &ko_old, splits.1),
            ("qkv V", &vo_new, &vo_old, splits.2),
        ] {
            let a = read(new, n * 8);
            nonzero(name, &a);
            bad += bits_equal(name, &a, &read(old, n * 8));
        }
        assert_eq!(bad, 0, "qkv _k16 at k=8 differs from the generic kernel");
    }

    #[test]
    fn spec50_widen_qkv_batch_independent() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0x0F0F_F0F0_1234_4321);
        let splits = (64usize, 32usize, 32usize);
        let blocks = 11usize;
        let hidden = blocks * 32;
        let total = splits.0 + splits.1 + splits.2;
        let qw = buf_from(&c.device, &random_q4_0(&mut rng, splits.0, blocks));
        let kw = buf_from(&c.device, &random_q4_0(&mut rng, splits.1, blocks));
        let vw = buf_from(&c.device, &random_q4_0(&mut rng, splits.2, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, hidden * 16));
        let solo_q = zeros(&c.device, 16 * splits.0);
        let solo_k = zeros(&c.device, 16 * splits.1);
        let solo_v = zeros(&c.device, 16 * splits.2);
        for t in 0..16usize {
            qkv_generic(
                &c,
                &y,
                (t * hidden * 4) as u64,
                &qw,
                &kw,
                &vw,
                (&solo_q, &solo_k, &solo_v),
                (
                    (t * splits.0 * 4) as u64,
                    (t * splits.1 * 4) as u64,
                    (t * splits.2 * 4) as u64,
                ),
                splits,
                blocks,
                1,
            );
        }
        for &k in &[9usize, 12, 16] {
            let qo = zeros(&c.device, k * splits.0);
            let ko = zeros(&c.device, k * splits.1);
            let vo = zeros(&c.device, k * splits.2);
            run(&c.queue, |e| {
                encode_spec50_widen_qkv(
                    e,
                    c.widen,
                    &y,
                    &qw,
                    &kw,
                    &vw,
                    &qo,
                    &ko,
                    &vo,
                    (
                        blocks as u32,
                        splits.0 as u32,
                        splits.1 as u32,
                        splits.2 as u32,
                    ),
                    total,
                    k,
                );
            });
            let mut bad = 0usize;
            bad += bits_equal(
                &format!("qkv Q independence K={k}"),
                &read(&qo, k * splits.0),
                &read(&solo_q, 16 * splits.0)[..k * splits.0],
            );
            bad += bits_equal(
                &format!("qkv K independence K={k}"),
                &read(&ko, k * splits.1),
                &read(&solo_k, 16 * splits.1)[..k * splits.1],
            );
            bad += bits_equal(
                &format!("qkv V independence K={k}"),
                &read(&vo, k * splits.2),
                &read(&solo_v, 16 * splits.2)[..k * splits.2],
            );
            assert_eq!(bad, 0, "qkv K={k}: tokens depend on batch width");
        }
    }

    // -- gateup ---------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn gateup_generic(
        c: &Ctx,
        y: &Buffer,
        y_off: u64,
        gw: &Buffer,
        uw: &Buffer,
        out: &Buffer,
        out_off: u64,
        rows: usize,
        blocks: usize,
        k: usize,
    ) {
        assert!(k <= 8);
        let pipe = c
            .old
            .q4_0_gateup_geglu_block_batch_k_pipeline
            .as_ref()
            .expect("generic gateup batch_k pipeline");
        run(&c.queue, |e| {
            e.set_compute_pipeline_state(pipe);
            e.set_buffer(0, Some(y), y_off);
            e.set_buffer(1, Some(gw), 0);
            e.set_buffer(2, Some(uw), 0);
            e.set_buffer(3, Some(out), out_off);
            let bpr = blocks as u32;
            let rows_u32 = rows as u32;
            let k_u32 = k as u32;
            e.set_bytes(4, 4, &bpr as *const u32 as *const _);
            e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
            e.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
            e.dispatch_thread_groups(grid((rows as u64).div_ceil(4)), tg32());
        });
    }

    #[test]
    fn spec50_widen_gateup_bitwise_at_k8() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0xBBBB_CCCC_DDDD_EEEE);
        let mut bad = 0usize;
        for &(rows, blocks) in &[(66usize, 22usize), (37, 11)] {
            let gw = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let uw = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buf_from(&c.device, &random_f32(&mut rng, blocks * 32 * 8));
            let out_new = zeros(&c.device, rows * 8);
            let out_old = zeros(&c.device, rows * 8);
            run(&c.queue, |e| {
                encode_spec50_widen_gateup(
                    e,
                    c.widen,
                    &y,
                    &gw,
                    &uw,
                    &out_new,
                    blocks as u32,
                    rows,
                    8,
                );
            });
            gateup_generic(&c, &y, 0, &gw, &uw, &out_old, 0, rows, blocks, 8);
            let a = read(&out_new, rows * 8);
            nonzero("gateup k16@8", &a);
            bad += bits_equal(
                &format!("gateup rows={rows} blocks={blocks}"),
                &a,
                &read(&out_old, rows * 8),
            );
        }
        assert_eq!(bad, 0, "gateup _k16 at k=8 differs from the generic kernel");
    }

    #[test]
    fn spec50_widen_gateup_batch_independent() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        let (rows, blocks) = (66usize, 22usize);
        let hidden = blocks * 32;
        let gw = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let uw = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, hidden * 16));
        let solo = zeros(&c.device, 16 * rows);
        for t in 0..16usize {
            gateup_generic(
                &c,
                &y,
                (t * hidden * 4) as u64,
                &gw,
                &uw,
                &solo,
                (t * rows * 4) as u64,
                rows,
                blocks,
                1,
            );
        }
        for &k in &[9usize, 12, 16] {
            let out = zeros(&c.device, k * rows);
            run(&c.queue, |e| {
                encode_spec50_widen_gateup(e, c.widen, &y, &gw, &uw, &out, blocks as u32, rows, k);
            });
            let bad = bits_equal(
                &format!("gateup independence K={k}"),
                &read(&out, k * rows),
                &read(&solo, 16 * rows)[..k * rows],
            );
            assert_eq!(bad, 0, "gateup K={k}: tokens depend on batch width");
        }
    }

    // -- MoE gateup -------------------------------------------------------------

    /// Packed twin of the shader-side Gemma4UniqueExpertWork (16 bytes).
    fn work_list_bytes(entries: &[(u64, u32, u32)]) -> Vec<u8> {
        let mut out = Vec::with_capacity(entries.len() * 16);
        for &(mask, offset, slab) in entries {
            out.extend_from_slice(&mask.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&slab.to_le_bytes());
        }
        out
    }

    const MOE_FF: usize = 704; // G4Q4_FF
    const MOE_HIDDEN: usize = 2816; // G4Q4_HIDDEN
    const MOE_GU_BLOCKS: usize = 88; // G4Q4_GU_BLOCKS
    const MOE_DOWN_BLOCKS: usize = 22; // G4Q4_DOWN_BLOCKS
    const MOE_GU_ROW_BYTES: usize = 1584; // G4Q4_GU_ROW_BYTES
    const MOE_GATE_UP_BYTES: usize = 2_230_272; // G4Q4_GATE_UP_BYTES

    fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n).map(|_| (rng.next_u32() & 0xFF) as u8).collect()
    }

    /// Gate+Up wire bytes for `experts` slots packed back to back (only the
    /// gate/up region is read by the kernel; scales are made small normal f16
    /// like `random_q4_0` so nothing degenerates).
    fn random_gateup_slab(rng: &mut Rng, experts: usize) -> Vec<u8> {
        let mut out = random_bytes(rng, experts * MOE_GATE_UP_BYTES);
        for block in out.chunks_exact_mut(18) {
            let exp = 10 + (rng.next_u32() % 8);
            let mant = rng.next_u32() % 1024;
            let sign = (rng.next_u32() & 1) << 15;
            let bits = (sign as u16) | ((exp as u16) << 10) | (mant as u16);
            block[0] = (bits & 0xFF) as u8;
            block[1] = (bits >> 8) as u8;
        }
        out
    }

    fn random_i8(rng: &mut Rng, n: usize) -> Vec<i8> {
        (0..n)
            .map(|_| (rng.next_u32() & 0xFF) as u8 as i8)
            .collect()
    }

    fn random_scales(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| 0.002 + (rng.next_u32() % 4096) as f32 * 1.0e-6)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_gateup_reference(
        c: &Ctx,
        scales: &Buffer,
        scales_off: u64,
        quants: &Buffer,
        quants_off: u64,
        slab: &Buffer,
        work: &Buffer,
        out_scales: &Buffer,
        out_quants: &Buffer,
        unique: usize,
        k: usize,
        overflow: Option<&Buffer>,
    ) {
        assert!(k <= 8);
        let pipe = c
            .old
            .gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k_pipeline
            .as_ref()
            .expect("fused gateup batch_k pipeline");
        run(&c.queue, |e| {
            e.set_compute_pipeline_state(pipe);
            e.set_buffer(0, Some(scales), scales_off);
            e.set_buffer(1, Some(quants), quants_off);
            e.set_buffer(2, Some(slab), 0);
            e.set_buffer(3, Some(work), 0);
            e.set_buffer(4, Some(out_scales), 0);
            e.set_buffer(5, Some(out_quants), 0);
            let unique_u32 = unique as u32;
            let k_u32 = k as u32;
            e.set_bytes(6, 4, &unique_u32 as *const u32 as *const _);
            e.set_bytes(7, 4, &k_u32 as *const u32 as *const _);
            e.set_buffer(8, overflow.map(|v| &**v), 0);
            e.dispatch_thread_groups(grid((unique * (MOE_FF / 32)) as u64), tg32());
        });
    }

    fn read_i8(b: &Buffer, n: usize) -> Vec<i8> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const i8, n).to_vec() }
    }

    /// MoE gateup _k16: bitwise identical to the existing fused batch_k kernel
    /// at k=8 (guard), and per-token independent at K in {12,16} against the
    /// existing kernel at k_candidates=1 (including an overflow-slab expert).
    #[test]
    fn spec50_widen_moe_gateup_bitwise_and_independent() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
        let unique = 2usize;
        // Slot 0 lives in the primary slab, slot 1 in the overflow slab.
        let slab = buf_from(&c.device, &random_gateup_slab(&mut rng, 1));
        let overflow = buf_from(&c.device, &random_gateup_slab(&mut rng, 1));
        let scales_v = random_scales(&mut rng, 16 * MOE_GU_BLOCKS);
        let quants_v = random_i8(&mut rng, 16 * MOE_HIDDEN);
        let scales = buf_from(&c.device, &scales_v);
        let quants = buf_from(&c.device, &quants_v);

        // k=8 guard: full masks, both experts.
        let work8 = buf_from(&c.device, &work_list_bytes(&[(0xFF, 0, 0), (0xFF, 0, 1)]));
        let n_scale = unique * 8 * MOE_DOWN_BLOCKS;
        let n_quant = unique * 8 * MOE_FF;
        let os_new = zeros(&c.device, n_scale);
        let oq_new = zeros(&c.device, n_quant.div_ceil(4));
        let os_old = zeros(&c.device, n_scale);
        let oq_old = zeros(&c.device, n_quant.div_ceil(4));
        run(&c.queue, |e| {
            encode_spec50_widen_moe_gateup(
                e,
                c.widen,
                &scales,
                &quants,
                &slab,
                0,
                &work8,
                &os_new,
                &oq_new,
                unique as u32,
                8,
                Some(&overflow),
                MOE_FF / 32,
            );
        });
        moe_gateup_reference(
            &c,
            &scales,
            0,
            &quants,
            0,
            &slab,
            &work8,
            &os_old,
            &oq_old,
            unique,
            8,
            Some(&overflow),
        );
        let a = read(&os_new, n_scale);
        nonzero("moe gateup scales", &a);
        let mut bad = bits_equal("moe gateup scales k=8", &a, &read(&os_old, n_scale));
        let qa = read_i8(&oq_new, n_quant);
        let qb = read_i8(&oq_old, n_quant);
        let qbad = qa.iter().zip(qb.iter()).filter(|(x, y)| x != y).count();
        assert!(qa.iter().any(|&v| v != 0), "moe gateup quants degenerate");
        bad += qbad;
        assert_eq!(
            bad, 0,
            "moe gateup _k16 at k=8 differs from the fused batch_k kernel"
        );

        // Independence at K in {12,16}: token t vs the existing kernel at
        // k_candidates=1 over the same token's activation slice.
        for &k in &[12usize, 16] {
            let mask = (1u64 << k) - 1;
            let work = buf_from(&c.device, &work_list_bytes(&[(mask, 0, 0), (mask, 0, 1)]));
            let ns = unique * k * MOE_DOWN_BLOCKS;
            let nq = unique * k * MOE_FF;
            let os = zeros(&c.device, ns);
            let oq = zeros(&c.device, nq.div_ceil(4));
            run(&c.queue, |e| {
                encode_spec50_widen_moe_gateup(
                    e,
                    c.widen,
                    &scales,
                    &quants,
                    &slab,
                    0,
                    &work,
                    &os,
                    &oq,
                    unique as u32,
                    k as u32,
                    Some(&overflow),
                    MOE_FF / 32,
                );
            });
            let got_s = read(&os, ns);
            let got_q = read_i8(&oq, nq);
            let work1 = buf_from(&c.device, &work_list_bytes(&[(1, 0, 0), (1, 0, 1)]));
            for t in 0..k {
                let ns1 = unique * MOE_DOWN_BLOCKS;
                let nq1 = unique * MOE_FF;
                let os1 = zeros(&c.device, ns1);
                let oq1 = zeros(&c.device, nq1.div_ceil(4));
                moe_gateup_reference(
                    &c,
                    &scales,
                    (t * MOE_GU_BLOCKS * 4) as u64,
                    &quants,
                    (t * MOE_HIDDEN) as u64,
                    &slab,
                    &work1,
                    &os1,
                    &oq1,
                    unique,
                    1,
                    Some(&overflow),
                );
                let solo_s = read(&os1, ns1);
                let solo_q = read_i8(&oq1, nq1);
                let mut tbad = 0usize;
                for u in 0..unique {
                    let s_base = u * k * MOE_DOWN_BLOCKS + t * MOE_DOWN_BLOCKS;
                    tbad += bits_equal(
                        &format!("moe gateup scales K={k} t={t} u={u}"),
                        &got_s[s_base..s_base + MOE_DOWN_BLOCKS],
                        &solo_s[u * MOE_DOWN_BLOCKS..(u + 1) * MOE_DOWN_BLOCKS],
                    );
                    let q_base = u * k * MOE_FF + t * MOE_FF;
                    tbad += got_q[q_base..q_base + MOE_FF]
                        .iter()
                        .zip(&solo_q[u * MOE_FF..(u + 1) * MOE_FF])
                        .filter(|(x, y)| x != y)
                        .count();
                }
                assert_eq!(tbad, 0, "moe gateup K={k} token {t} depends on batch width");
            }
        }
    }

    // -- encode-fn routing --------------------------------------------------------

    /// (c) The production encode fns route k=12/16 to the _k16 pipelines
    /// (proved by output equality with the per-token generic reference — the
    /// fixed-depth kernels cannot produce these outputs, and before this change
    /// the encode fns asserted k<=8) and keep k<=8 on the originals.
    #[test]
    fn spec50_widen_encode_fns_route_wide_chunks() {
        let Some(c) = ctx() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        let (rows, blocks) = (130usize, 16usize);
        let hidden = blocks * 32;
        let w = buf_from(&c.device, &random_q4_0(&mut rng, rows, blocks));
        let y = buf_from(&c.device, &random_f32(&mut rng, hidden * 16));
        let scalar = buf_from(&c.device, &[blocks as u32, rows as u32, 16u32, 0u32]);

        let solo = zeros(&c.device, 16 * rows);
        for t in 0..16usize {
            plain_generic(
                &c,
                &y,
                (t * hidden * 4) as u64,
                &w,
                &solo,
                (t * rows * 4) as u64,
                rows,
                blocks,
                1,
            );
        }
        let solo_v = read(&solo, 16 * rows);

        for &k in &[12usize, 16] {
            let out = zeros(&c.device, k * rows);
            run(&c.queue, |e| {
                encode_gemma4_q4_0_matmul_batch_k(e, c.old, &y, &w, 0, &out, rows, &scalar, k);
            });
            let bad = bits_equal(
                &format!("encode routing K={k}"),
                &read(&out, k * rows),
                &solo_v[..k * rows],
            );
            assert_eq!(bad, 0, "encode_gemma4_q4_0_matmul_batch_k K={k} wrong");
        }

        // k<=8 must still match a direct dispatch of the shipped selection
        // (k=8 -> batch_k8 specialization when compiled, else generic).
        let out8 = zeros(&c.device, 8 * rows);
        run(&c.queue, |e| {
            encode_gemma4_q4_0_matmul_batch_k(e, c.old, &y, &w, 0, &out8, rows, &scalar, 8);
        });
        let direct8 = zeros(&c.device, 8 * rows);
        if let Some(pipe) = c.old.q4_0_block_batch_k8_pipeline.as_ref() {
            run(&c.queue, |e| {
                e.set_compute_pipeline_state(pipe);
                e.set_buffer(0, Some(&y), 0);
                e.set_buffer(2, Some(&w), 0);
                e.set_buffer(3, Some(&direct8), 0);
                let bpr = blocks as u32;
                let rows_u32 = rows as u32;
                let k_u32 = 8u32;
                e.set_bytes(4, 4, &bpr as *const u32 as *const _);
                e.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
                e.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
                e.dispatch_thread_groups(grid((rows as u64).div_ceil(4)), tg32());
            });
        } else {
            plain_generic(&c, &y, 0, &w, &direct8, 0, rows, blocks, 8);
        }
        let bad = bits_equal(
            "encode routing K=8 unchanged",
            &read(&out8, 8 * rows),
            &read(&direct8, 8 * rows),
        );
        assert_eq!(bad, 0, "k=8 selection changed");
    }

    // -- tied head K=9..16 ------------------------------------------------------

    const Q6K_WIRE: usize = 210;
    const HEAD_HIDDEN: usize = 2816;
    const HEAD_N_SB: usize = HEAD_HIDDEN / 256;
    const HEAD_SOFTCAP: f32 = 30.0;

    fn fill_q6k_block(rng: &mut Rng, out: &mut [u8]) {
        for byte in out.iter_mut().take(208) {
            *byte = (rng.next_u32() & 0xFF) as u8;
        }
        let sign = (rng.next_u32() & 1) as u16;
        let exp = 5 + (rng.next_u32() % 8) as u16; // 2^-10 .. 2^-3 magnitudes
        let mant = (rng.next_u32() % 1024) as u16;
        let bits = (sign << 15) | (exp << 10) | mant;
        out[208] = (bits & 0xff) as u8;
        out[209] = (bits >> 8) as u8;
    }

    fn head_weights(rng: &mut Rng, rows: usize) -> Vec<u8> {
        let mut w = vec![0u8; rows * HEAD_N_SB * Q6K_WIRE];
        for block in w.chunks_exact_mut(Q6K_WIRE) {
            fill_q6k_block(rng, block);
        }
        w
    }

    fn head_activations(rng: &mut Rng, k: usize) -> (Vec<f32>, Vec<i8>) {
        let scales = random_scales(rng, k * HEAD_N_SB);
        let quants = random_i8(rng, k * HEAD_HIDDEN);
        (scales, quants)
    }

    /// The K compile-time template makes every row's program independent of K
    /// by construction — still TEST it: row t at K in {9,12,16} must be
    /// bitwise identical to row t at K=1 (the k1 entry is itself asserted
    /// bitwise identical to the oracle-verified reference by the spec50_head
    /// suite). Also: k=17 must be refused.
    #[test]
    fn spec50_widen_head_batch_independent() {
        let Some(device) = Device::system_default() else {
            eprintln!("[spec50_widen] no Metal device, skipping");
            return;
        };
        let kernels = spec50_head_kernels().expect("spec50 head pipelines");
        let queue = device.new_command_queue();
        let mut rng = Rng(0x600D_C0DE_0000_0001);
        let rows = 1024usize;
        let weights = head_weights(&mut rng, rows);
        let (scales, quants) = head_activations(&mut rng, 16);
        let wbuf = buf_from(&device, &weights);
        let sbuf = buf_from(&device, &scales);
        let qbuf = buf_from(&device, &quants);
        let perm = device.new_buffer(
            spec50_activation_scratch_bytes(16, HEAD_HIDDEN).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let run_head = |sb: &Buffer, qb: &Buffer, out: &Buffer, k: usize| -> bool {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            let ok = encode_q6k_spec50_batch(
                e,
                kernels,
                sb,
                qb,
                &perm,
                &wbuf,
                0,
                out,
                HEAD_N_SB,
                rows,
                k,
                HEAD_HIDDEN,
                HEAD_SOFTCAP,
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
            ok
        };

        // Each token in isolation at K=1.
        let mut solo = vec![0f32; 16 * rows];
        for t in 0..16usize {
            let ss = buf_from(&device, &scales[t * HEAD_N_SB..(t + 1) * HEAD_N_SB]);
            let sq = buf_from(&device, &quants[t * HEAD_HIDDEN..(t + 1) * HEAD_HIDDEN]);
            let out = zeros(&device, rows);
            assert!(run_head(&ss, &sq, &out, 1), "head K=1 encode refused");
            solo[t * rows..(t + 1) * rows].copy_from_slice(&read(&out, rows));
        }
        nonzero("head solo", &solo);

        for &k in &[9usize, 12, 16] {
            let out = zeros(&device, k * rows);
            assert!(run_head(&sbuf, &qbuf, &out, k), "head K={k} encode refused");
            let bad = bits_equal(
                &format!("head independence K={k}"),
                &read(&out, k * rows),
                &solo[..k * rows],
            );
            assert_eq!(bad, 0, "head K={k}: rows depend on batch width");
        }

        // Out-of-range K must be refused, encoding nothing.
        let out = zeros(&device, rows);
        assert!(!run_head(&sbuf, &qbuf, &out, 17), "head K=17 must refuse");
    }

    // -- verbatim-copy guard --------------------------------------------------------

    /// The _k16 twins must be the src/metal.rs kernels verbatim, modulo the
    /// declared accumulator/name/guard substitutions — anything else is drift.
    #[test]
    fn spec50_widen_copies_are_verbatim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal.rs");
        let source = std::fs::read_to_string(path).expect("read src/metal.rs");
        let extract = |hay: &str, name: &str| -> String {
            let needle = format!("kernel void {name}(\n");
            let start = hay
                .find(&needle)
                .unwrap_or_else(|| panic!("{name} not found"));
            let end = hay[start..].find("\n}\n").expect("kernel end") + start + 2;
            hay[start..end].to_string()
        };
        let cases: [(&str, &str, &[(&str, &str)]); 4] = [
            (
                "q4_0_block_linear_batch_k",
                "q4_0_block_linear_batch_k16",
                &[(
                    "float sums[4][8] = {{0.0f}};",
                    "float sums[4][16] = {{0.0f}};",
                )],
            ),
            (
                "q4_0_qkv_block_linear_batch_k",
                "q4_0_qkv_block_linear_batch_k16",
                &[
                    ("float sum0[8] = {0.0f};", "float sum0[16] = {0.0f};"),
                    ("float sum1[8] = {0.0f};", "float sum1[16] = {0.0f};"),
                ],
            ),
            (
                "q4_0_gateup_geglu_block_linear_batch_k",
                "q4_0_gateup_geglu_block_linear_batch_k16",
                &[
                    (
                        "float g_sums[4][8] = {{0.0f}};",
                        "float g_sums[4][16] = {{0.0f}};",
                    ),
                    (
                        "float u_sums[4][8] = {{0.0f}};",
                        "float u_sums[4][16] = {{0.0f}};",
                    ),
                ],
            ),
            (
                "gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k",
                "gemma4_q4_multi_expert_fused_gateup_geglu_quant_batch_k16",
                &[
                    ("float gate_acc[8];", "float gate_acc[16];"),
                    ("float up_acc[8];", "float up_acc[16];"),
                    (
                        "for (uint t = 0; t < 8; ++t) {",
                        "for (uint t = 0; t < 16; ++t) {",
                    ),
                    (
                        "if (k_candidates == 0u || k_candidates > 8u) return;",
                        "if (k_candidates == 0u || k_candidates > 16u) return;",
                    ),
                ],
            ),
        ];
        for (orig_name, new_name, subs) in cases {
            let mut expected = extract(&source, orig_name);
            expected = expected.replacen(
                &format!("kernel void {orig_name}("),
                &format!("kernel void {new_name}("),
                1,
            );
            for (from, to) in subs {
                assert_eq!(
                    expected.matches(from).count(),
                    1,
                    "{orig_name}: substitution source `{from}` not unique"
                );
                expected = expected.replacen(from, to, 1);
            }
            let mine = extract(SPEC50_WIDEN_SHADER, new_name);
            assert_eq!(
                mine, expected,
                "{new_name} drifted from the src/metal.rs original"
            );
        }
    }
}
