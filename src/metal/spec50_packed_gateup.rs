//! Packed-Q4_0 row-complete GateUp experiment.
//!
//! The pre-unpacked shared-MLP experiment's fastest exact GateUp kernel did
//! two things at once: it changed the weight representation and removed the
//! per-row tail predicate from the four-row SIMD hot loop. Gemma 4's shared
//! GateUp has 2,112 rows, so every dispatched four-row group is complete. This
//! module copies the shipped packed-Q4_0 arithmetic and thread mapping, but
//! removes only that never-taken predicate. The raw-bit test and isolated
//! 30-layer benchmark below decide whether the control simplification alone is
//! useful. The measured K=8 row-complete shape is now selected by default;
//! other widths remain parity-only experiments until separately benchmarked.

#![allow(dead_code)]

use super::*;

const SPEC50_PACKED_GATEUP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Runtime-K twin for K=1..7. Apart from removing `has_row`, this is textually
// the shipped q4_0_gateup_geglu_block_linear_batch_k arithmetic.
kernel void s50p_gateup_row_complete_k(
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

    float g_sums[4][8] = {{0.0f}};
    float u_sums[4][8] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 g_scaled_lo[4], g_scaled_hi[4];
        float4 u_scaled_lo[4], u_scaled_hi[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            const uint r = r0 + i;
            device const char* g_wb = gate_weight + r * row_stride + ib * q4_block_bytes;
            const float g_w_scale = float(*reinterpret_cast<device const half*>(g_wb));
            const uchar4 g_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(g_wb + 2 + ilb));
            const float4 g_lo = float4(int4(g_wq & 0x0F) - 8);
            const float4 g_hi = float4(int4(g_wq >> 4) - 8);
            g_scaled_lo[i] = g_lo * g_w_scale;
            g_scaled_hi[i] = g_hi * g_w_scale;

            device const char* u_wb = up_weight + r * row_stride + ib * q4_block_bytes;
            const float u_w_scale = float(*reinterpret_cast<device const half*>(u_wb));
            const uchar4 u_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(u_wb + 2 + ilb));
            const float4 u_lo = float4(int4(u_wq & 0x0F) - 8);
            const float4 u_hi = float4(int4(u_wq >> 4) - 8);
            u_scaled_lo[i] = u_lo * u_w_scale;
            u_scaled_hi[i] = u_hi * u_w_scale;
        }

        #pragma unroll
        for (uint k = 0; k < k_batch; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
            }
        }
    }

    for (uint k = 0; k < k_batch; ++k) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
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

// Fixed-K=8 twin. Keeping this separate is part of the exactness contract:
// the shipped K=8 path has fixed loop bounds, and compiling it through the
// runtime-K flavor can change contraction/register allocation.
kernel void s50p_gateup_row_complete_k8(
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
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    (void)k_batch;
    const uint r0 = tg * NR0;
    if (r0 >= rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;

    float g_sums[4][8] = {{0.0f}};
    float u_sums[4][8] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 g_scaled_lo[4], g_scaled_hi[4];
        float4 u_scaled_lo[4], u_scaled_hi[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            const uint r = r0 + i;
            device const char* g_wb = gate_weight + r * row_stride + ib * q4_block_bytes;
            const float g_w_scale = float(*reinterpret_cast<device const half*>(g_wb));
            const uchar4 g_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(g_wb + 2 + ilb));
            const float4 g_lo = float4(int4(g_wq & 0x0F) - 8);
            const float4 g_hi = float4(int4(g_wq >> 4) - 8);
            g_scaled_lo[i] = g_lo * g_w_scale;
            g_scaled_hi[i] = g_hi * g_w_scale;

            device const char* u_wb = up_weight + r * row_stride + ib * q4_block_bytes;
            const float u_w_scale = float(*reinterpret_cast<device const half*>(u_wb));
            const uchar4 u_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(u_wb + 2 + ilb));
            const float4 u_lo = float4(int4(u_wq & 0x0F) - 8);
            const float4 u_hi = float4(int4(u_wq >> 4) - 8);
            u_scaled_lo[i] = u_lo * u_w_scale;
            u_scaled_hi[i] = u_hi * u_w_scale;
        }

        #pragma unroll
        for (uint k = 0; k < KB; ++k) {
            device const float* yb = y + k * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
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
"#;

// Kept in a separate runtime-compiled library so a compiler or pipeline
// rejection cannot disable the production row-complete K8 singleton above.
const SPEC50_PACKED_GATEUP_DIRECT_K16_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One 64-thread dispatch containing two independent canonical K8 SIMDgroups.
// Each SIMDgroup executes the literal fixed-K8 weight decode, accumulator and
// reduction program for its own eight-token tile.  There is no shared state,
// threadgroup staging, or barrier; concurrent identical weight loads are left
// to the GPU cache/coalescer.
[[max_total_threads_per_threadgroup(64)]]
kernel void s50p_gateup_k16_two_k8_direct_sg(
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
) {
    constexpr uint NR0 = 4;
    constexpr uint NQ = 8;
    constexpr uint NB = 4;
    constexpr uint KB = 8;
    constexpr uint q4_block_bytes = 18;
    (void)k_batch;
    const uint r0 = tg * NR0;
    if (r0 >= rows) return;
    const uint row_stride = blocks_per_row * q4_block_bytes;
    const uint hidden = blocks_per_row * 32;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB;
    const uint token0 = sg * KB;

    float g_sums[4][8] = {{0.0f}};
    float u_sums[4][8] = {{0.0f}};

    for (uint ib = ix; ib < blocks_per_row; ib += NQ) {
        float4 g_scaled_lo[4], g_scaled_hi[4];
        float4 u_scaled_lo[4], u_scaled_hi[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            const uint r = r0 + i;
            device const char* g_wb = gate_weight + r * row_stride + ib * q4_block_bytes;
            const float g_w_scale = float(*reinterpret_cast<device const half*>(g_wb));
            const uchar4 g_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(g_wb + 2 + ilb));
            const float4 g_lo = float4(int4(g_wq & 0x0F) - 8);
            const float4 g_hi = float4(int4(g_wq >> 4) - 8);
            g_scaled_lo[i] = g_lo * g_w_scale;
            g_scaled_hi[i] = g_hi * g_w_scale;

            device const char* u_wb = up_weight + r * row_stride + ib * q4_block_bytes;
            const float u_w_scale = float(*reinterpret_cast<device const half*>(u_wb));
            const uchar4 u_wq = uchar4(
                *reinterpret_cast<device const packed_uchar4*>(u_wb + 2 + ilb));
            const float4 u_lo = float4(int4(u_wq & 0x0F) - 8);
            const float4 u_hi = float4(int4(u_wq >> 4) - 8);
            u_scaled_lo[i] = u_lo * u_w_scale;
            u_scaled_hi[i] = u_hi * u_w_scale;
        }

        #pragma unroll
        for (uint k = 0; k < KB; ++k) {
            device const float* yb = y + (token0 + k) * hidden + ib * 32;
            const float4 ylo = *reinterpret_cast<device const float4*>(yb + ilb);
            const float4 yhi = *reinterpret_cast<device const float4*>(yb + 16 + ilb);

            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                g_sums[i][k] += dot(g_scaled_lo[i], ylo) + dot(g_scaled_hi[i], yhi);
                u_sums[i][k] += dot(u_scaled_lo[i], ylo) + dot(u_scaled_hi[i], yhi);
            }
        }
    }

    #pragma unroll
    for (uint k = 0; k < KB; ++k) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            const float g_tot = simd_sum(g_sums[i][k]);
            const float u_tot = simd_sum(u_sums[i][k]);
            if (lane == 0) {
                const float in_v = 0.7978845608f * (g_tot + 0.044715f * g_tot * g_tot * g_tot);
                const float gelu = 0.5f * g_tot * (1.0f + tanh(clamp(in_v, -15.0f, 15.0f)));
                act_output[(token0 + k) * rows + r0 + i] = gelu * u_tot;
            }
        }
    }
}
"#;

pub(crate) struct Spec50PackedGateupKernels {
    runtime_k: ComputePipelineState,
    fixed_k8: ComputePipelineState,
}

static SPEC50_PACKED_GATEUP_KERNELS: OnceLock<Option<Spec50PackedGateupKernels>> = OnceLock::new();

pub(crate) struct Spec50PackedGateupDirectK16Kernel {
    direct_sg: ComputePipelineState,
}

static SPEC50_PACKED_GATEUP_DIRECT_K16_KERNEL: OnceLock<Option<Spec50PackedGateupDirectK16Kernel>> =
    OnceLock::new();

/// Default-on timing lever. Set
/// `CAMELID_GEMMA4_PACKED_GATEUP_ROW_COMPLETE=0` (or `false`) to restore the
/// shipped guarded packed-Q4_0 kernels.
pub(crate) fn spec50_packed_gateup_row_complete_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        spec50_packed_gateup_row_complete_flag_from(
            std::env::var("CAMELID_GEMMA4_PACKED_GATEUP_ROW_COMPLETE")
                .ok()
                .as_deref(),
        )
    })
}

pub(crate) fn spec50_packed_gateup_row_complete_flag_from(value: Option<&str>) -> bool {
    !matches!(value, Some(v) if v == "0" || v.eq_ignore_ascii_case("false"))
}

/// Shape currently admitted by the runtime selector. Raw-bit parity covers
/// K=1..=8, but only K=8 has a performance receipt; all other widths retain
/// their existing specialized/runtime paths until separately benchmarked.
pub(crate) const fn spec50_packed_gateup_row_complete_eligible(
    rows: usize,
    k_batch: usize,
) -> bool {
    rows % 4 == 0 && k_batch == 8
}

pub(crate) fn spec50_packed_gateup_kernels() -> Option<&'static Spec50PackedGateupKernels> {
    SPEC50_PACKED_GATEUP_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            // LINEAR_ROW_SHADER uses the default fast-math compilation mode.
            options.set_fast_math_enabled(true);
            let library = device
                .new_library_with_source(SPEC50_PACKED_GATEUP_SHADER, &options)
                .map_err(|err| {
                    eprintln!("[metal] SPEC50_PACKED_GATEUP_SHADER compile failed: {err}")
                })
                .ok()?;
            let build = |name: &str| -> Option<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] packed GateUp {name} missing: {err}"))
                    .ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] packed GateUp {name} pipeline failed: {err}"))
                    .ok()
            };
            Some(Spec50PackedGateupKernels {
                runtime_k: build("s50p_gateup_row_complete_k")?,
                fixed_k8: build("s50p_gateup_row_complete_k8")?,
            })
        })
        .as_ref()
}

/// Experimental pipeline kept independent of the production K<=8 singleton.
/// A shader/pipeline failure here cannot disable row-complete GateUp.
pub(crate) fn spec50_packed_gateup_direct_k16_kernel(
) -> Option<&'static Spec50PackedGateupDirectK16Kernel> {
    SPEC50_PACKED_GATEUP_DIRECT_K16_KERNEL
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            options.set_fast_math_enabled(true);
            let library = device
                .new_library_with_source(SPEC50_PACKED_GATEUP_DIRECT_K16_SHADER, &options)
                .map_err(|err| {
                    eprintln!(
                        "[metal] packed GateUp direct-K16 shader compile failed: {err}"
                    )
                })
                .ok()?;
            let build = |name: &str, min_threads: u64| -> Option<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] packed GateUp {name} missing: {err}"))
                    .ok()?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] packed GateUp {name} pipeline failed: {err}"))
                    .ok()?;
                if pipeline.thread_execution_width() != 32
                    || pipeline.max_total_threads_per_threadgroup() < min_threads
                {
                    eprintln!(
                        "[metal] packed GateUp {name} rejected: SIMD={} max_threads/TG={} need={min_threads}",
                        pipeline.thread_execution_width(),
                        pipeline.max_total_threads_per_threadgroup(),
                    );
                    return None;
                }
                Some(pipeline)
            };
            Some(Spec50PackedGateupDirectK16Kernel {
                direct_sg: build("s50p_gateup_k16_two_k8_direct_sg", 64)?,
            })
        })
        .as_ref()
}

/// Byte offsets for one canonical, at-most-eight-token shared GateUp tile.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Spec50PackedGateupBufferOffsets {
    pub(crate) y: u64,
    pub(crate) act_output: u64,
}

/// Encode the packed-Q4_0 row-complete kernel. `rows % 4 == 0` is the fact
/// that permits the hot-loop predicate removal; refusing other shapes prevents
/// an OOB read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_packed_gateup_row_complete(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50PackedGateupKernels,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    blocks_per_row: u32,
    rows: usize,
    k_batch: usize,
) {
    encode_spec50_packed_gateup_row_complete_at_offsets(
        encoder,
        kernels,
        y,
        gate_weight,
        up_weight,
        act_output,
        blocks_per_row,
        rows,
        k_batch,
        Spec50PackedGateupBufferOffsets::default(),
    );
}

/// Offset-aware binding of the exact same GateUp pipeline.  Wider speculative
/// waves use this only to schedule multiple canonical K8 tiles; the shader's
/// token-local arithmetic and loop bounds are unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_packed_gateup_row_complete_at_offsets(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50PackedGateupKernels,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    blocks_per_row: u32,
    rows: usize,
    k_batch: usize,
    offsets: Spec50PackedGateupBufferOffsets,
) {
    assert!(
        rows.is_multiple_of(4),
        "row-complete GateUp requires rows % 4 == 0"
    );
    assert!(
        (1..=8).contains(&k_batch),
        "row-complete GateUp requires K in 1..=8"
    );
    let rows_u32 = rows as u32;
    let k_batch_u32 = k_batch as u32;
    let pipeline = if k_batch == 8 {
        &kernels.fixed_k8
    } else {
        &kernels.runtime_k
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(y), offsets.y);
    encoder.set_buffer(1, Some(gate_weight), 0);
    encoder.set_buffer(2, Some(up_weight), 0);
    encoder.set_buffer(3, Some(act_output), offsets.act_output);
    encoder.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows / 4) as u64,
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

/// Experimental K=16 dispatch containing two independent canonical K8
/// SIMDgroups with direct weight loads and no staging/barriers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_packed_gateup_k16_direct_two_sg(
    encoder: &metal::ComputeCommandEncoderRef,
    kernel: &Spec50PackedGateupDirectK16Kernel,
    y: &Buffer,
    gate_weight: &Buffer,
    up_weight: &Buffer,
    act_output: &Buffer,
    blocks_per_row: u32,
    rows: usize,
    offsets: Spec50PackedGateupBufferOffsets,
) {
    assert!(
        rows.is_multiple_of(4),
        "direct-two-SG K16 GateUp requires rows % 4 == 0"
    );
    let rows_u32 = rows as u32;
    let k_batch_u32 = 16u32;
    encoder.set_compute_pipeline_state(&kernel.direct_sg);
    encoder.set_buffer(0, Some(y), offsets.y);
    encoder.set_buffer(1, Some(gate_weight), 0);
    encoder.set_buffer(2, Some(up_weight), 0);
    encoder.set_buffer(3, Some(act_output), offsets.act_output);
    encoder.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
    encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows / 4) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
}

/// `(max threads/TG, SIMD width, static threadgroup bytes)` for canonical K8
/// and direct two-SG K16, respectively.
pub(crate) fn spec50_packed_gateup_pipeline_limits(
    kernels: &Spec50PackedGateupKernels,
    direct_k16: &Spec50PackedGateupDirectK16Kernel,
) -> [(u64, u64, u64); 2] {
    [&kernels.fixed_k8, &direct_k16.direct_sg].map(|pipeline| {
        (
            pipeline.max_total_threads_per_threadgroup(),
            pipeline.thread_execution_width(),
            pipeline.static_threadgroup_memory_length(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q4_BLOCK_BYTES: usize = 18;

    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
        }

        fn f32_pm1(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    fn random_q4_0(rng: &mut Rng, rows: usize, blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; rows * blocks * Q4_BLOCK_BYTES];
        for block in out.chunks_exact_mut(Q4_BLOCK_BYTES) {
            // Finite normal f16 scales spanning roughly 2^-5..2^2.
            let exponent = 10 + (rng.next_u32() % 8);
            let mantissa = rng.next_u32() % 1024;
            let sign = (rng.next_u32() & 1) << 15;
            let bits = sign as u16 | (exponent as u16) << 10 | mantissa as u16;
            block[0] = bits as u8;
            block[1] = (bits >> 8) as u8;
            for q in &mut block[2..] {
                *q = rng.next_u32() as u8;
            }
        }
        out
    }

    fn random_f32(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.f32_pm1()).collect()
    }

    fn buffer_from<T: Copy>(device: &Device, data: &[T]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn zero_f32(device: &Device, len: usize) -> Buffer {
        buffer_from(device, &vec![0.0f32; len])
    }

    fn read_f32(buffer: &Buffer, len: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len).to_vec() }
    }

    fn run(queue: &CommandQueue, body: impl FnOnce(&metal::ComputeCommandEncoderRef)) -> u128 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        body(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(
            command_buffer.status(),
            metal::MTLCommandBufferStatus::Completed,
            "packed GateUp command buffer failed"
        );
        command_buffer_gpu_times_us(command_buffer).0
    }

    fn mismatch_count(got: &[f32], want: &[f32]) -> usize {
        got.iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    }

    const TILE_GUARD_BITS: u32 = 0x7fc5_5aa5;

    fn poisoned_f32(device: &Device, len: usize) -> Buffer {
        buffer_from(device, &vec![f32::from_bits(TILE_GUARD_BITS); len])
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

    fn assert_body_written(name: &str, values: &[f32], prefix: usize, body: usize) {
        assert!(
            values[prefix..prefix + body]
                .iter()
                .all(|value| value.to_bits() != TILE_GUARD_BITS),
            "{name}: at least one output element retained the poison sentinel"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shipped_down_k8_at_offsets(
        encoder: &metal::ComputeCommandEncoderRef,
        shipped: &MetalLinearKernel,
        y: &Buffer,
        y_offset: u64,
        weight: &Buffer,
        output: &Buffer,
        output_offset: u64,
        blocks_per_row: usize,
        rows: usize,
    ) {
        let pipeline = shipped
            .q4_0_block_batch_k8_pipeline
            .as_ref()
            .expect("shipped fixed-K8 Down pipeline");
        let blocks_u32 = blocks_per_row as u32;
        let rows_u32 = rows as u32;
        let k_u32 = 8u32;
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

    #[test]
    fn spec50_packed_gateup_selection_is_default_on_and_fail_closed() {
        assert!(spec50_packed_gateup_row_complete_flag_from(None));
        assert!(spec50_packed_gateup_row_complete_flag_from(Some("1")));
        assert!(!spec50_packed_gateup_row_complete_flag_from(Some("0")));
        assert!(!spec50_packed_gateup_row_complete_flag_from(Some("FALSE")));

        assert!(spec50_packed_gateup_row_complete_eligible(2112, 8));
        assert!(!spec50_packed_gateup_row_complete_eligible(2112, 1));
        assert!(!spec50_packed_gateup_row_complete_eligible(2113, 8));
        assert!(!spec50_packed_gateup_row_complete_eligible(2112, 0));
        assert!(!spec50_packed_gateup_row_complete_eligible(2112, 9));
        assert!(!spec50_packed_gateup_row_complete_eligible(2112, 16));
    }

    /// Encode the actual guarded reference pipeline directly. The public
    /// Gemma dispatch now selects the row-complete kernel by default, so parity
    /// and A/B timing must bypass that selector to remain independent evidence.
    #[allow(clippy::too_many_arguments)]
    fn encode_shipped_guarded_gateup(
        encoder: &metal::ComputeCommandEncoderRef,
        shipped: &MetalLinearKernel,
        y: &Buffer,
        gate_weight: &Buffer,
        up_weight: &Buffer,
        act_output: &Buffer,
        blocks_per_row: u32,
        rows: usize,
        k_batch: usize,
    ) {
        let specialized = match k_batch {
            8 => shipped.q4_0_gateup_geglu_block_batch_k8_pipeline.as_ref(),
            6 => shipped.q4_0_gateup_geglu_block_batch_k6_pipeline.as_ref(),
            _ => None,
        };
        let pipeline = specialized
            .or(shipped.q4_0_gateup_geglu_block_batch_k_pipeline.as_ref())
            .expect("shipped guarded GateUp pipeline");
        let rows_u32 = rows as u32;
        let k_batch_u32 = k_batch as u32;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(y), 0);
        encoder.set_buffer(1, Some(gate_weight), 0);
        encoder.set_buffer(2, Some(up_weight), 0);
        encoder.set_buffer(3, Some(act_output), 0);
        encoder.set_bytes(4, 4, &blocks_per_row as *const u32 as *const _);
        encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k_batch_u32 as *const u32 as *const _);
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

    /// The experiment is admissible only if deleting the row-tail control has
    /// no numerical side effect for every supported wave width. The sweep
    /// includes the real shared GateUp geometry (2112 x 88 blocks).
    #[test]
    fn spec50_packed_gateup_row_complete_raw_bit_parity_k1_through_k8() {
        let Some(shipped) = metal_linear_kernel() else {
            eprintln!("[s50p] no Metal device; skipping");
            return;
        };
        let kernels = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        let mut rng = Rng(0x5041_434b_4544_4755);

        for &(rows, blocks) in &[(64usize, 8usize), (132, 16), (704, 88), (2112, 88)] {
            assert_eq!(rows % 4, 0);
            let gate = buffer_from(&shipped.device, &random_q4_0(&mut rng, rows, blocks));
            let up = buffer_from(&shipped.device, &random_q4_0(&mut rng, rows, blocks));
            let y = buffer_from(&shipped.device, &random_f32(&mut rng, blocks * 32 * 8));

            for k_batch in 1usize..=8 {
                let reference_out = zero_f32(&shipped.device, rows * k_batch);
                let candidate_out = zero_f32(&shipped.device, rows * k_batch);
                run(&shipped.queue, |encoder| {
                    encode_shipped_guarded_gateup(
                        encoder,
                        shipped,
                        &y,
                        &gate,
                        &up,
                        &reference_out,
                        blocks as u32,
                        rows,
                        k_batch,
                    );
                });
                run(&shipped.queue, |encoder| {
                    encode_spec50_packed_gateup_row_complete(
                        encoder,
                        kernels,
                        &y,
                        &gate,
                        &up,
                        &candidate_out,
                        blocks as u32,
                        rows,
                        k_batch,
                    );
                });

                let want = read_f32(&reference_out, rows * k_batch);
                let got = read_f32(&candidate_out, rows * k_batch);
                let bad = mismatch_count(&got, &want);
                if bad != 0 {
                    let first = got
                        .iter()
                        .zip(&want)
                        .position(|(a, b)| a.to_bits() != b.to_bits())
                        .expect("mismatch count was nonzero");
                    eprintln!(
                        "[s50p] rows={rows} blocks={blocks} K={k_batch}: {bad}/{} differ; first {first}: got {:e} ({:#010x}), want {:e} ({:#010x})",
                        got.len(),
                        got[first],
                        got[first].to_bits(),
                        want[first],
                        want[first].to_bits(),
                    );
                }
                assert_eq!(bad, 0, "packed row-complete GateUp lost raw-bit parity");
            }
        }
    }

    /// The direct two-SIMDgroup prototype is admissible only if both SIMDgroups
    /// keep the canonical K8 arithmetic flavor. Compare its one K16 dispatch to
    /// two separately scheduled fixed-K8 tiles at the real shared-MLP shape,
    /// with raw-bit guards around the contiguous token-major wave.
    #[test]
    fn spec50_packed_gateup_k16_direct_two_sg_is_bit_exact_two_k8_tiles() {
        let Some(shipped) = metal_linear_kernel() else {
            eprintln!("[s50p] no Metal device; skipping");
            return;
        };
        let kernels = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        let direct_k16 = spec50_packed_gateup_direct_k16_kernel()
            .expect("experimental direct two-SG K16 pipeline");
        const TILE_K: usize = 8;
        const WAVE_K: usize = 16;
        const GUARD: usize = 17;
        let poison = f32::from_bits(TILE_GUARD_BITS);
        let mut rng = Rng(0x5348_4152_4544_4b31);

        for &(rows, blocks) in &[(64usize, 16usize), (2112usize, 88usize)] {
            let hidden = blocks * 32;
            let gate = buffer_from(&shipped.device, &random_q4_0(&mut rng, rows, blocks));
            let up = buffer_from(&shipped.device, &random_q4_0(&mut rng, rows, blocks));
            let y_values = random_f32(&mut rng, WAVE_K * hidden);
            let mut guarded_y = vec![poison; GUARD];
            guarded_y.extend_from_slice(&y_values);
            guarded_y.extend_from_slice(&vec![poison; GUARD]);
            let y = buffer_from(&shipped.device, &guarded_y);
            let reference = poisoned_f32(&shipped.device, GUARD + WAVE_K * rows + GUARD);
            let candidate_direct_sg = poisoned_f32(&shipped.device, GUARD + WAVE_K * rows + GUARD);

            run(&shipped.queue, |encoder| {
                for tile in 0..2 {
                    encode_spec50_packed_gateup_row_complete_at_offsets(
                        encoder,
                        kernels,
                        &y,
                        &gate,
                        &up,
                        &reference,
                        blocks as u32,
                        rows,
                        TILE_K,
                        Spec50PackedGateupBufferOffsets {
                            y: ((GUARD + tile * TILE_K * hidden) * 4) as u64,
                            act_output: ((GUARD + tile * TILE_K * rows) * 4) as u64,
                        },
                    );
                }
            });
            run(&shipped.queue, |encoder| {
                encode_spec50_packed_gateup_k16_direct_two_sg(
                    encoder,
                    direct_k16,
                    &y,
                    &gate,
                    &up,
                    &candidate_direct_sg,
                    blocks as u32,
                    rows,
                    Spec50PackedGateupBufferOffsets {
                        y: (GUARD * 4) as u64,
                        act_output: (GUARD * 4) as u64,
                    },
                );
            });

            let ref_all = read_f32(&reference, GUARD + WAVE_K * rows + GUARD);
            let direct_sg_all = read_f32(&candidate_direct_sg, GUARD + WAVE_K * rows + GUARD);
            let y_all = read_f32(&y, GUARD + WAVE_K * hidden + GUARD);
            assert_poisoned_guards("direct input", &y_all, GUARD, WAVE_K * hidden);
            assert_poisoned_guards("canonical reference", &ref_all, GUARD, WAVE_K * rows);
            assert_body_written("canonical reference", &ref_all, GUARD, WAVE_K * rows);
            let want = &ref_all[GUARD..GUARD + WAVE_K * rows];
            let name = "direct two-SG";
            let all = direct_sg_all.as_slice();
            assert_poisoned_guards(name, all, GUARD, WAVE_K * rows);
            assert_body_written(name, all, GUARD, WAVE_K * rows);
            let got = &all[GUARD..GUARD + WAVE_K * rows];
            let bad = mismatch_count(got, want);
            if bad != 0 {
                let first = got
                    .iter()
                    .zip(want)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .expect("mismatch count was nonzero");
                eprintln!(
                    "[s50p] {name} rows={rows} blocks={blocks}: {bad}/{} differ; first token={} row={} got={:e} ({:#010x}) want={:e} ({:#010x})",
                    got.len(),
                    first / rows,
                    first % rows,
                    got[first],
                    got[first].to_bits(),
                    want[first],
                    want[first].to_bits(),
                );
            }
            assert_eq!(
                bad, 0,
                "{name} GateUp differs at rows={rows} blocks={blocks}"
            );
            for seam in [TILE_K * rows - 1, TILE_K * rows] {
                assert_eq!(
                    got[seam].to_bits(),
                    want[seam].to_bits(),
                    "{name} GateUp token 7/8 seam differs at {seam}"
                );
            }
        }

        let [k8, direct_sg] = spec50_packed_gateup_pipeline_limits(kernels, direct_k16);
        eprintln!(
            "[s50p] pipeline limits K8 max/TG={} SIMD={} tmem={} | direct two-SG max/TG={} SIMD={} tmem={}",
            k8.0, k8.1, k8.2,
            direct_sg.0, direct_sg.1, direct_sg.2,
        );
    }

    /// Compose a real-shape shared MLP K=16 wave from two invocations of the
    /// production K8 GateUp and Down pipelines.  GateUp's contiguous output is
    /// Down's contiguous input, so this checks both the intermediate and final
    /// token-major layouts across the token-7/8 boundary.
    #[test]
    fn spec50_shared_mlp_k16_is_two_bit_exact_k8_tiles() {
        let Some(shipped) = metal_linear_kernel() else {
            eprintln!("[s50p] no Metal device; skipping");
            return;
        };
        let kernels = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        const TILE_K: usize = 8;
        const WAVE_K: usize = 16;
        const HIDDEN: usize = 2816;
        const FFN: usize = 2112;
        const GATEUP_BLOCKS: usize = HIDDEN / 32;
        const DOWN_BLOCKS: usize = FFN / 32;
        const GUARD: usize = 23;
        let poison = f32::from_bits(TILE_GUARD_BITS);
        let mut rng = Rng(0x5348_4152_4544_4d4c);

        let gate = buffer_from(&shipped.device, &random_q4_0(&mut rng, FFN, GATEUP_BLOCKS));
        let up = buffer_from(&shipped.device, &random_q4_0(&mut rng, FFN, GATEUP_BLOCKS));
        let down = buffer_from(&shipped.device, &random_q4_0(&mut rng, HIDDEN, DOWN_BLOCKS));
        let wide_y = random_f32(&mut rng, WAVE_K * HIDDEN);

        let mut act_ref = Vec::with_capacity(WAVE_K * FFN);
        let mut down_ref = Vec::with_capacity(WAVE_K * HIDDEN);
        for tile in 0..2 {
            let y0 = tile * TILE_K * HIDDEN;
            let y = buffer_from(&shipped.device, &wide_y[y0..y0 + TILE_K * HIDDEN]);
            let act = zero_f32(&shipped.device, TILE_K * FFN);
            let out = zero_f32(&shipped.device, TILE_K * HIDDEN);
            run(&shipped.queue, |encoder| {
                encode_spec50_packed_gateup_row_complete(
                    encoder,
                    kernels,
                    &y,
                    &gate,
                    &up,
                    &act,
                    GATEUP_BLOCKS as u32,
                    FFN,
                    TILE_K,
                );
                encoder.memory_barrier_with_resources(&[&act]);
                encode_shipped_down_k8_at_offsets(
                    encoder,
                    shipped,
                    &act,
                    0,
                    &down,
                    &out,
                    0,
                    DOWN_BLOCKS,
                    HIDDEN,
                );
            });
            act_ref.extend_from_slice(&read_f32(&act, TILE_K * FFN));
            down_ref.extend_from_slice(&read_f32(&out, TILE_K * HIDDEN));
        }

        let mut guarded_y = vec![poison; GUARD];
        guarded_y.extend_from_slice(&wide_y);
        guarded_y.extend_from_slice(&vec![poison; GUARD]);
        let y = buffer_from(&shipped.device, &guarded_y);
        let act = poisoned_f32(&shipped.device, GUARD + WAVE_K * FFN + GUARD);
        let out = poisoned_f32(&shipped.device, GUARD + WAVE_K * HIDDEN + GUARD);
        run(&shipped.queue, |encoder| {
            for tile in 0..2 {
                encode_spec50_packed_gateup_row_complete_at_offsets(
                    encoder,
                    kernels,
                    &y,
                    &gate,
                    &up,
                    &act,
                    GATEUP_BLOCKS as u32,
                    FFN,
                    TILE_K,
                    Spec50PackedGateupBufferOffsets {
                        y: ((GUARD + tile * TILE_K * HIDDEN) * 4) as u64,
                        act_output: ((GUARD + tile * TILE_K * FFN) * 4) as u64,
                    },
                );
            }
            encoder.memory_barrier_with_resources(&[&act]);
            for tile in 0..2 {
                encode_shipped_down_k8_at_offsets(
                    encoder,
                    shipped,
                    &act,
                    ((GUARD + tile * TILE_K * FFN) * 4) as u64,
                    &down,
                    &out,
                    ((GUARD + tile * TILE_K * HIDDEN) * 4) as u64,
                    DOWN_BLOCKS,
                    HIDDEN,
                );
            }
        });

        let y_all = read_f32(&y, GUARD + WAVE_K * HIDDEN + GUARD);
        let act_all = read_f32(&act, GUARD + WAVE_K * FFN + GUARD);
        let out_all = read_f32(&out, GUARD + WAVE_K * HIDDEN + GUARD);
        let act_got = &act_all[GUARD..GUARD + WAVE_K * FFN];
        let down_got = &out_all[GUARD..GUARD + WAVE_K * HIDDEN];
        assert_poisoned_guards("shared input", &y_all, GUARD, WAVE_K * HIDDEN);
        assert_poisoned_guards("shared GateUp", &act_all, GUARD, WAVE_K * FFN);
        assert_poisoned_guards("shared Down", &out_all, GUARD, WAVE_K * HIDDEN);
        assert_eq!(
            mismatch_count(act_got, &act_ref),
            0,
            "shared GateUp K8 tiles diverged"
        );
        assert_eq!(
            mismatch_count(down_got, &down_ref),
            0,
            "shared Down K8 tiles diverged"
        );
        for (name, got, want, rows) in [
            ("shared GateUp", act_got, act_ref.as_slice(), FFN),
            ("shared Down", down_got, down_ref.as_slice(), HIDDEN),
        ] {
            for seam in [TILE_K * rows - 1, TILE_K * rows] {
                assert_eq!(
                    got[seam].to_bits(),
                    want[seam].to_bits(),
                    "{name}: token 7/8 seam differs at element {seam}"
                );
            }
        }
    }

    fn timed_sweep(queue: &CommandQueue, body: impl Fn(&metal::ComputeCommandEncoderRef)) -> f64 {
        let one = || run(queue, |encoder| body(encoder)) as f64 / 1000.0;
        for _ in 0..2 {
            one();
        }
        let mut samples: Vec<f64> = (0..9).map(|_| one()).collect();
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    #[derive(Clone, Copy, Debug)]
    struct SweepTiming {
        cold_ms: f64,
        warm_ms: f64,
        median_ms: f64,
    }

    fn timed_sweep_receipt(
        queue: &CommandQueue,
        body: impl Fn(&metal::ComputeCommandEncoderRef),
    ) -> SweepTiming {
        let one = || run(queue, |encoder| body(encoder)) as f64 / 1000.0;
        let cold_ms = one();
        let _first_warmup = one();
        let warm_ms = one();
        let mut samples: Vec<f64> = (0..9).map(|_| one()).collect();
        samples.sort_by(f64::total_cmp);
        SweepTiming {
            cold_ms,
            warm_ms,
            median_ms: samples[samples.len() / 2],
        }
    }

    /// Isolates the row-complete control change from pre-unpacking: candidate
    /// and reference read the same 30 pairs of packed Q4_0 matrices.
    #[test]
    #[ignore = "GPU microbenchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_packed_gateup_row_complete_bench_30_layers() {
        let Some(shipped) = metal_linear_kernel() else {
            eprintln!("[s50p] no Metal device; skipping");
            return;
        };
        let kernels = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        const LAYERS: usize = 30;
        const ROWS: usize = 2112;
        const BLOCKS: usize = 88;
        const K: usize = 8;
        let mut rng = Rng(0x524f_5758_5041_434b);
        let mut gates = Vec::with_capacity(LAYERS);
        let mut ups = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            gates.push(buffer_from(
                &shipped.device,
                &random_q4_0(&mut rng, ROWS, BLOCKS),
            ));
            ups.push(buffer_from(
                &shipped.device,
                &random_q4_0(&mut rng, ROWS, BLOCKS),
            ));
        }
        let y = buffer_from(&shipped.device, &random_f32(&mut rng, BLOCKS * 32 * K));
        let out = zero_f32(&shipped.device, ROWS * K);
        let old_ms = timed_sweep(&shipped.queue, |encoder| {
            for layer in 0..LAYERS {
                encode_shipped_guarded_gateup(
                    encoder,
                    shipped,
                    &y,
                    &gates[layer],
                    &ups[layer],
                    &out,
                    BLOCKS as u32,
                    ROWS,
                    K,
                );
            }
        });
        let candidate_ms = timed_sweep(&shipped.queue, |encoder| {
            for layer in 0..LAYERS {
                encode_spec50_packed_gateup_row_complete(
                    encoder,
                    kernels,
                    &y,
                    &gates[layer],
                    &ups[layer],
                    &out,
                    BLOCKS as u32,
                    ROWS,
                    K,
                );
            }
        });

        let bytes = (LAYERS * 2 * ROWS * BLOCKS * Q4_BLOCK_BYTES) as f64;
        println!(
            "\n[s50p] packed shared GateUp, {LAYERS} layers, K={K}, {:.1} MB weights/sweep",
            bytes / 1.0e6
        );
        println!(
            "  shipped guarded : {old_ms:8.3} ms ({:6.1} GB/s)",
            bytes / old_ms / 1.0e6
        );
        println!(
            "  packed row-complete: {candidate_ms:8.3} ms ({:6.1} GB/s), {:.3}x, {:+.3} ms",
            bytes / candidate_ms / 1.0e6,
            old_ms / candidate_ms,
            candidate_ms - old_ms,
        );
        println!(
            "  pipeline limits: guarded max/TG={} width={} | row-complete max/TG={} width={}",
            shipped
                .q4_0_gateup_geglu_block_batch_k8_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.max_total_threads_per_threadgroup()),
            shipped
                .q4_0_gateup_geglu_block_batch_k8_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.thread_execution_width()),
            kernels.fixed_k8.max_total_threads_per_threadgroup(),
            kernels.fixed_k8.thread_execution_width(),
        );
    }

    /// Distinct-weight sweep for the cross-tile reuse decision.  Unlike the
    /// hot one-layer stage table, 30 GateUp pairs exceed the cache hierarchy;
    /// each command buffer therefore reproduces the real layer-to-layer weight
    /// stream while keeping all tensors resident (no storage I/O in timing).
    #[test]
    #[ignore = "GPU microbenchmark; run explicitly with --ignored --test-threads=1"]
    fn spec50_packed_gateup_k16_direct_two_sg_bench_30_layers() {
        let Some(shipped) = metal_linear_kernel() else {
            eprintln!("[s50p] no Metal device; skipping");
            return;
        };
        let kernels = spec50_packed_gateup_kernels().expect("packed GateUp pipelines");
        let direct_k16 = spec50_packed_gateup_direct_k16_kernel()
            .expect("experimental direct two-SG K16 pipeline");
        let widened = spec50_widen_kernels().expect("existing widened K16 pipelines");
        const LAYERS: usize = 30;
        const ROWS: usize = 2112;
        const BLOCKS: usize = 88;
        const HIDDEN: usize = BLOCKS * 32;
        const TILE_K: usize = 8;
        const WAVE_K: usize = 16;
        let mut rng = Rng(0x4b31_3652_4555_5345);
        let mut gates = Vec::with_capacity(LAYERS);
        let mut ups = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            gates.push(buffer_from(
                &shipped.device,
                &random_q4_0(&mut rng, ROWS, BLOCKS),
            ));
            ups.push(buffer_from(
                &shipped.device,
                &random_q4_0(&mut rng, ROWS, BLOCKS),
            ));
        }
        let y = buffer_from(&shipped.device, &random_f32(&mut rng, WAVE_K * HIDDEN));
        let tiled = zero_f32(&shipped.device, WAVE_K * ROWS);
        let direct_sg = zero_f32(&shipped.device, WAVE_K * ROWS);
        let wide = zero_f32(&shipped.device, WAVE_K * ROWS);

        let tiled_body = |encoder: &metal::ComputeCommandEncoderRef| {
            for layer in 0..LAYERS {
                for tile in 0..2 {
                    encode_spec50_packed_gateup_row_complete_at_offsets(
                        encoder,
                        kernels,
                        &y,
                        &gates[layer],
                        &ups[layer],
                        &tiled,
                        BLOCKS as u32,
                        ROWS,
                        TILE_K,
                        Spec50PackedGateupBufferOffsets {
                            y: (tile * TILE_K * HIDDEN * 4) as u64,
                            act_output: (tile * TILE_K * ROWS * 4) as u64,
                        },
                    );
                }
            }
        };
        let direct_sg_body = |encoder: &metal::ComputeCommandEncoderRef| {
            for layer in 0..LAYERS {
                encode_spec50_packed_gateup_k16_direct_two_sg(
                    encoder,
                    direct_k16,
                    &y,
                    &gates[layer],
                    &ups[layer],
                    &direct_sg,
                    BLOCKS as u32,
                    ROWS,
                    Spec50PackedGateupBufferOffsets::default(),
                );
            }
        };
        let wide_body = |encoder: &metal::ComputeCommandEncoderRef| {
            for layer in 0..LAYERS {
                encode_spec50_widen_gateup(
                    encoder,
                    widened,
                    &y,
                    &gates[layer],
                    &ups[layer],
                    &wide,
                    BLOCKS as u32,
                    ROWS,
                    WAVE_K,
                );
            }
        };

        // AB then BA order makes the steady result robust to which pipeline
        // pays the first-use residency/driver cost.
        let tiled_a = timed_sweep_receipt(&shipped.queue, &tiled_body);
        let direct_sg_a = timed_sweep_receipt(&shipped.queue, &direct_sg_body);
        let wide_a = timed_sweep_receipt(&shipped.queue, &wide_body);
        let wide_b = timed_sweep_receipt(&shipped.queue, &wide_body);
        let direct_sg_b = timed_sweep_receipt(&shipped.queue, &direct_sg_body);
        let tiled_b = timed_sweep_receipt(&shipped.queue, &tiled_body);

        let tiled_values = read_f32(&tiled, WAVE_K * ROWS);
        for (name, values) in [
            ("direct two-SG", read_f32(&direct_sg, WAVE_K * ROWS)),
            ("existing widened", read_f32(&wide, WAVE_K * ROWS)),
        ] {
            assert_eq!(
                mismatch_count(&values, &tiled_values),
                0,
                "30-layer {name} sweep lost canonical K8 bits"
            );
        }
        let bytes = (LAYERS * 2 * ROWS * BLOCKS * Q4_BLOCK_BYTES) as f64;
        let [k8, direct_limits] = spec50_packed_gateup_pipeline_limits(kernels, direct_k16);
        println!(
            "\n[s50p] K16 cross-tile reuse, {LAYERS} distinct GateUp pairs, {:.1} MB weights/sweep",
            bytes / 1.0e6,
        );
        let print_receipt = |order: &str, name: &str, timing: SweepTiming, baseline: f64| {
            println!(
                "  {order} {name:18} cold={:8.3} warm={:8.3} median={:8.3} ms ratio/tile={:6.3}",
                timing.cold_ms,
                timing.warm_ms,
                timing.median_ms,
                timing.median_ms / baseline,
            );
        };
        print_receipt("A", "two canonical K8", tiled_a, tiled_a.median_ms);
        print_receipt("A", "direct two-SG", direct_sg_a, tiled_a.median_ms);
        print_receipt("A", "existing widened", wide_a, tiled_a.median_ms);
        print_receipt("B", "existing widened", wide_b, tiled_b.median_ms);
        print_receipt("B", "direct two-SG", direct_sg_b, tiled_b.median_ms);
        print_receipt("B", "two canonical K8", tiled_b, tiled_b.median_ms);
        println!(
            "  limits: K8 max/TG={} SIMD={} tmem={} B | direct two-SG max/TG={} SIMD={} tmem={} B | widened max/TG={} SIMD={} tmem={} B",
            k8.0, k8.1, k8.2,
            direct_limits.0, direct_limits.1, direct_limits.2,
            widened.gateup.max_total_threads_per_threadgroup(),
            widened.gateup.thread_execution_width(),
            widened.gateup.static_threadgroup_memory_length(),
        );
    }
}
