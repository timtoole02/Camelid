//! Optional CUDA GPU backend (additive, gated behind `--features cuda`).
//!
//! The CPU path remains the default build and the correctness reference. This
//! backend must reproduce the exact CPU/llama.cpp parity evidence — a CUDA lane
//! that is fast but diverges from the parity audit is a regression, not a
//! feature. To that end the Q8_0 dot kernel mirrors the CPU reference
//! (`dot_q8_0_encoded_row_with_scales`) operation-for-operation: one thread per
//! output row, an exact integer block dot, then a sequential f32 accumulation
//! of `(int_sum as f32) * weight_scale * input_scale` in block order, compiled
//! with `--fmad=false` so the GPU does not fuse the multiply/add the CPU keeps
//! separate. That yields bit-identical f32 logits and therefore identical
//! greedy argmax / token IDs.
//!
//! Mirrors `src/metal.rs`'s shape: the module is always present; the real
//! implementation is `#[cfg(feature = "cuda")]` and a stub returns "unavailable"
//! otherwise, so callers never need their own cfg gates.

// CUDA is part of the DEFAULT build on Windows (the primary CUDA dev host): `build.rs`
// injects the `cuda` cfg there, and `cudarc` is a non-optional Windows dependency.
// This fails the build loudly if that wiring ever regresses, so a Windows `cargo
// build` can never silently drop to the CPU-only path. (Linux/macOS keep CUDA opt-in.)
#[cfg(all(windows, not(feature = "cuda")))]
compile_error!(
    "CUDA must be enabled by default on Windows: build.rs should emit \
     `cargo:rustc-cfg=feature=\"cuda\"` for windows targets and Cargo.toml should \
     declare cudarc as a non-optional Windows dependency."
);

// MESA: the same default-on guarantee on x86_64 Linux. `build.rs` injects the `cuda`
// cfg and Cargo.toml declares cudarc non-optional for this target, so a bare `cargo
// build` can never silently drop to the CPU-only path here either. (aarch64 Linux / Pi
// and BSD keep CUDA opt-in and are intentionally not guarded.)
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda")))]
compile_error!(
    "CUDA must be enabled by default on x86_64 Linux (MESA): build.rs should emit \
     `cargo:rustc-cfg=feature=\"cuda\"` for this target and Cargo.toml should declare \
     cudarc as a non-optional x86_64-linux dependency."
);

/// Result of probing for a usable CUDA device at startup.
#[derive(Debug, Clone, Default)]
pub struct CudaDeviceInfo {
    pub available: bool,
    pub device_name: Option<String>,
    /// Why CUDA is unavailable (feature off, no device, init error), for logs.
    pub reason: Option<String>,
}

/// Lightweight device capability snapshot (no kernel compilation), used by the
/// startup hardware-profile probe so VRAM-driven tunables can size to the device.
#[derive(Debug, Clone, Default)]
pub struct CudaCapability {
    pub device_count: usize,
    pub device_name: String,
    /// (major, minor) compute capability; tensor cores require major >= 7.
    pub compute_capability: (u32, u32),
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
}

// Runtime GPU-enable switch, so the UI can toggle the CUDA decode path on/off
// without restarting. Seeded from `CAMELID_CUDA_Q8` on first read, then owned by
// `set_runtime_enabled`. 0 = uninitialised, 1 = disabled, 2 = enabled. This flag
// only *gates* the path; if no CUDA device is present the dispatch still falls
// back to the CPU reference, so enabling it on an unsupported host is harmless.
static RUNTIME_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn seed_runtime_from_env() -> bool {
    std::env::var("CAMELID_CUDA_Q8")
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Whether the CUDA Q8 decode path is currently enabled (UI/env switch). This is
/// the gate the inference dispatch reads; it is independent of whether a device
/// is actually present (see [`is_available`]).
pub fn runtime_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match RUNTIME_STATE.load(Ordering::Relaxed) {
        0 => {
            let enabled = seed_runtime_from_env();
            RUNTIME_STATE.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
            enabled
        }
        2 => true,
        _ => false,
    }
}

/// Turn the CUDA Q8 decode path on or off at runtime (the UI toggle calls this).
pub fn set_runtime_enabled(enabled: bool) {
    RUNTIME_STATE.store(
        if enabled { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Platform-neutral GPU capability selected for the user-facing acceleration control.
/// CUDA wins on a host that exposes both backends because the resident CUDA runtime is
/// the primary path there; Apple Metal is selected on macOS when CUDA is unavailable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAccelerationInfo {
    pub available: bool,
    pub device_name: Option<String>,
    pub backend: &'static str,
}

fn select_gpu_acceleration_info(
    cuda_available: bool,
    cuda_device_name: Option<String>,
    metal_available: bool,
    metal_device_name: Option<String>,
) -> GpuAccelerationInfo {
    if cuda_available {
        GpuAccelerationInfo {
            available: true,
            device_name: cuda_device_name,
            backend: "cuda",
        }
    } else if metal_available {
        GpuAccelerationInfo {
            available: true,
            device_name: metal_device_name,
            backend: "metal",
        }
    } else {
        GpuAccelerationInfo {
            available: false,
            device_name: None,
            backend: "none",
        }
    }
}

/// Detect the GPU backend represented by the single CLI/UI acceleration switch.
/// This is deliberately broader than [`is_available`], which remains the CUDA-only
/// capability predicate used by CUDA dispatch code.
pub fn gpu_acceleration_info() -> GpuAccelerationInfo {
    let cuda = detect_cuda_device();
    let metal = crate::metal::detect_metal_device();
    select_gpu_acceleration_info(
        cuda.available,
        cuda.device_name,
        metal.available,
        metal.device_name,
    )
}

// Master "GPU acceleration" switch as the user sees it in the UI. This gates both
// CUDA-resident decode and the opt-in Apple Metal paths. The legacy `RUNTIME_STATE`
// above only gates the CUDA hybrid Q8 *matmul* used on the CPU-forward fallback.
// Defaults ON whenever either supported GPU backend is present, so the app uses the
// accelerator out of the box. 0 = uninitialised, 1 = disabled, 2 = enabled.
static GPU_ACCEL_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the platform-neutral GPU acceleration control is enabled. On by default
/// when CUDA or Metal is present; flipped by the UI toggle. Independent of the CUDA
/// hybrid `runtime_enabled()` switch. Deterministic mode and backend-specific opt-outs
/// still force individual lanes off at their own call sites.
pub fn gpu_accel_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match GPU_ACCEL_STATE.load(Ordering::Relaxed) {
        0 => {
            let on = gpu_acceleration_info().available;
            GPU_ACCEL_STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        2 => true,
        _ => false,
    }
}

/// Turn GPU acceleration on or off at runtime — the CLI and UI share this switch.
/// Dispatch still checks backend capability, so enabling it on a host without CUDA
/// or Metal is harmless.
pub fn set_gpu_accel_enabled(enabled: bool) {
    GPU_ACCEL_STATE.store(
        if enabled { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Whether the operator has masked every CUDA device via `CUDA_VISIBLE_DEVICES`.
///
/// `-1` (or an empty value) is the standard "expose no devices" setting, and this
/// repo relies on it for CPU-pinned runs — the decode receipts and `alloc_gate`
/// both document `CUDA_VISIBLE_DEVICES=-1`. Nothing in the CUDA path used to read
/// it, so the driver was initialized anyway; on the WSL2 GPU-PV stack doing that
/// with every device masked aborts the process inside glibc ("free(): double free
/// detected"). That is a SIGABRT, NOT a Rust panic, so `catch_unwind` cannot
/// intercept it — the only reliable defense is to never make the call. Skipping
/// CUDA here is also exactly what the operator asked for, so this is honest on
/// every platform rather than a WSL-specific workaround.
// Only the CUDA implementation calls this, and the module compiles on hosts with
// CUDA off (macOS, aarch64 Linux), where it is legitimately unused rather than a
// mistake. Kept ungated so the parse below stays covered by the test on every
// platform, since the value's meaning does not vary by target.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn devices_masked_by_env() -> bool {
    devices_masked(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
}

/// Pure half of [`devices_masked_by_env`], split out so the parse is unit-testable
/// without mutating process env (same split as `fit_check_skipped`). Only an
/// explicit "no devices" value masks; an unset var, or any real device list,
/// leaves CUDA alone.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn devices_masked(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|v| v.is_empty() || v == "-1")
}

/// Whether a usable CUDA device is actually present (feature built + device + a
/// kernel that compiled). The UI uses this to decide whether to show the toggle.
pub fn is_available() -> bool {
    detect_cuda_device().available
}

/// The CUDA device name, if a device is present (for the UI label).
pub fn device_name() -> Option<String> {
    detect_cuda_device().device_name
}

/// Which CUDA device every GPU path binds to. Defaults to device 0 — on this
/// laptop the only CUDA device is the discrete NVIDIA RTX 3060 (the Intel iGPU
/// is not CUDA-capable and is never enumerated here). Override with
/// `CAMELID_CUDA_DEVICE=<index>` when a host genuinely has multiple NVIDIA GPUs
/// and the discrete one is not index 0; the chosen index is logged at startup.
pub fn selected_device_ordinal() -> usize {
    std::env::var("CAMELID_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
pub use backend::{
    detect_cuda_device, probe_capability, release_async_pool, try_bitnet_f16_head_matvec,
    try_bitnet_i2_s_linear_rows, try_q8_0_block_linear_row, try_q8_0_encoded_linear_row,
    try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
pub use stub::{
    detect_cuda_device, probe_capability, release_async_pool, try_bitnet_f16_head_matvec,
    try_bitnet_i2_s_linear_rows, try_q8_0_block_linear_row, try_q8_0_encoded_linear_row,
    try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
mod stub {
    use super::{CudaCapability, CudaDeviceInfo};

    pub fn probe_capability() -> Option<CudaCapability> {
        None
    }

    /// No-op without CUDA: there is no async memory pool to trim.
    pub fn release_async_pool() {}

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        CudaDeviceInfo {
            available: false,
            device_name: None,
            reason: Some("built without the `cuda` feature".to_string()),
        }
    }

    /// Decode-shaped Q8_0 linear (one input row × `rows` weight rows). Returns
    /// `false` so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Prefill-shaped Q8_0 linear (`input_rows` × `weight_rows`). Returns `false`
    /// so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _input_rows: usize,
        _weight_rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout
    /// (36 bytes/block: f32 scale + 32 i8 quants), matching the engine's
    /// retained block-dot path. Returns `false` (CPU fallback).
    pub fn try_q8_0_block_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_block_bytes: &[u8],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_bitnet_i2_s_linear_rows(
        _input: &[f32],
        _weight_bytes: &[u8],
        _input_rows: usize,
        _weight_rows: usize,
        _input_width: usize,
        _mode: u32,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Exact BitNet-b1.58-2B-4T's tied F16 output head. Non-CUDA builds
    /// decline it so the caller keeps using the portable row-dot fallback.
    pub fn try_bitnet_f16_head_matvec(
        _input: &[f32],
        _weight_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _weight_rows: usize,
        _input_width: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }
}

/// Number of Q8_0 matmuls the CUDA backend has completed on the GPU this
/// process. Zero means the GPU path never ran (e.g. CPU-only fallback). Used by
/// tests/diagnostics to prove the GPU lane was actually exercised.
#[cfg(feature = "cuda")]
pub fn cuda_q8_run_count() -> u64 {
    backend::run_count()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_q8_run_count() -> u64 {
    0
}

/// Number of cleanroom BitNet I2_S projections completed by CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_bitnet_run_count() -> u64 {
    backend::bitnet_run_count()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_bitnet_run_count() -> u64 {
    0
}

/// Number of exact tied BitNet F16 output-head matvecs completed by CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_bitnet_f16_head_run_count() -> u64 {
    backend::bitnet_f16_head_run_count()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_bitnet_f16_head_run_count() -> u64 {
    0
}

#[cfg(feature = "cuda")]
mod backend {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    use cudarc::driver::{
        result, sys, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::{CompileOptions, Ptx};
    use std::sync::Arc;

    use super::CudaDeviceInfo;

    static RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    static BITNET_RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    static BITNET_F16_HEAD_RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicBool = AtomicBool::new(false);
    static BITNET_LOGGED: AtomicBool = AtomicBool::new(false);
    static BITNET_F16_HEAD_LOGGED: AtomicBool = AtomicBool::new(false);
    static ENTRY_LOGGED: AtomicBool = AtomicBool::new(false);

    pub(super) fn run_count() -> u64 {
        RUN_COUNT.load(Ordering::Relaxed)
    }

    pub(super) fn bitnet_run_count() -> u64 {
        BITNET_RUN_COUNT.load(Ordering::Relaxed)
    }

    pub(super) fn bitnet_f16_head_run_count() -> u64 {
        BITNET_F16_HEAD_RUN_COUNT.load(Ordering::Relaxed)
    }

    /// CUDA C source for the Q8_0 encoded linear kernel. One thread computes one
    /// output row. The integer block dot is exact (i32); the f32 accumulation is
    /// sequential in block order and matches the CPU reference exactly. Built
    /// with `--fmad=false` (see [`compile_options`]) so the `a*b*c + sum` is not
    /// fused into an FMA — the CPU keeps those operations separate.
    const Q8_KERNEL_SRC: &str = r#"
extern "C" __global__ void q8_0_encoded_linear_rows(
    const float* __restrict__ input_scales,   // [input_rows * blocks_per_row]
    const signed char* __restrict__ input_quants, // [input_rows * blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [weight_rows * blocks_per_row * 34]
    const float* __restrict__ weight_scales,   // [weight_rows * blocks_per_row]
    const int input_rows,
    const int weight_rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [input_rows * weight_rows]
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)input_rows * weight_rows;
    if (idx >= total) return;
    int in_row = (int)(idx / weight_rows);
    int w_row = (int)(idx % weight_rows);

    const float* in_scales = input_scales + (long)in_row * blocks_per_row;
    const signed char* in_quants = input_quants + (long)in_row * blocks_per_row * 32;
    const unsigned char* w_bytes = weight_bytes + (long)w_row * blocks_per_row * 34;
    const float* w_scales = weight_scales + (long)w_row * blocks_per_row;

    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* wblk = w_bytes + (long)b * 34 + 2; // skip 2-byte f16 scale
        const signed char* iblk = in_quants + (long)b * 32;
        int int_sum = 0;
        for (int j = 0; j < 32; j++) {
            int_sum += (int)((signed char)wblk[j]) * (int)iblk[j];
        }
        sum += (float)int_sum * w_scales[b] * in_scales[b];
    }
    output[(long)in_row * weight_rows + w_row] = sum;
}

// Fast decode matvec over the in-memory Q8_0Block byte layout (36 bytes/block:
// f32 scale at offset 0, then 32 i8 quants). One *warp* per output row: the 32
// lanes stride over the row's blocks so consecutive lanes read consecutive
// blocks (coalesced global loads), each block's 32 i8*i8 products are summed
// exactly with `__dp4a` (4-wide integer dot), the per-block f32 terms are
// accumulated per lane, then a warp-shuffle reduction sums the lanes. The
// per-block integer dot is exact; the cross-block f32 reduction is reassociated
// vs the CPU's sequential sum, so this is token-identical (not bit-identical) —
// the same standard as the Metal GPU path. Verified by the parity audit.
// Spelled as the single PTX instruction `__dp4a` lowers to, so the kernel does
// not depend on NVRTC preincluding the sm_61 intrinsics header that declares it.
// `dp4a.s32.s32` needs PTX ISA 5.0 / sm_61, which `compile_options` already
// targets. NVRTC 12.9 does resolve `__dp4a` on its own (measured on sm_89), so
// this is portability insurance, not a fix for an observed failure.
__device__ __forceinline__ int camelid_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

extern "C" __global__ void q8_0_block_linear_row(
    const float* __restrict__ input_scales,   // [blocks_per_row]
    const signed char* __restrict__ input_quants, // [blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [rows * blocks_per_row * 36]
    const int rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [rows]
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gtid >> 5;          // one warp per output row
    int lane = gtid & 31;
    if (row >= rows) return;
    const unsigned char* wrow = weight_bytes + (long)row * blocks_per_row * 36;
    float partial = 0.0f;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const unsigned char* blk = wrow + (long)b * 36;
        float w_scale = __ldg(reinterpret_cast<const float*>(blk));
        // 32 i8 quants = 8 ints; blk+4 and input are 4-byte aligned.
        const int* wq = reinterpret_cast<const int*>(blk + 4);
        const int* iq = reinterpret_cast<const int*>(input_quants + (long)b * 32);
        int int_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int_sum = camelid_dp4a(wq[k], iq[k], int_sum);
        }
        partial += (float)int_sum * w_scale * input_scales[b];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    }
    if (lane == 0) output[row] = partial;
}

// Cleanroom BitNet kernel over the public canonical I2_S contract: four
// ternary values per byte, interleaved in 128-value tiles, plus a tensor-wide
// f32 scale at packed_bytes[weight_rows * input_width / 4]. `mode` selects
// direct decode (0), two-weight/9-entry lookup (1), or three-weight/27-entry
// lookup (2). One thread computes one output element in deterministic K order.
__device__ __forceinline__ int i2_s_ternary(
    const unsigned char* packed,
    const long long logical_index
) {
    const long long tile = logical_index / 128;
    const int within = (int)(logical_index % 128);
    const unsigned char byte = packed[tile * 32 + (within % 32)];
    const int code = (byte >> (6 - 2 * (within / 32))) & 3;
    return code == 0 ? -1 : (code == 2 ? 1 : 0);
}

extern "C" __global__ void bitnet_i2_s_linear_rows(
    const signed char* __restrict__ input,
    const unsigned char* __restrict__ weights,
    const int input_rows,
    const int weight_rows,
    const int input_width,
    const int mode,
    const float* __restrict__ activation_scales,
    float* __restrict__ output
) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)input_rows * weight_rows;
    if (idx >= total) return;
    const int input_row = (int)(idx / weight_rows);
    const int weight_row = (int)(idx % weight_rows);
    const signed char* x = input + (long long)input_row * input_width;
    const long long base = (long long)weight_row * input_width;
    int sum = 0;

    if (mode == 1) {
        int column = 0;
        for (; column + 1 < input_width; column += 2) {
            const int a = (int)x[column];
            const int b = (int)x[column + 1];
            const int table[9] = {-a-b, -a, -a+b, -b, 0, b, a-b, a, a+b};
            const int left = i2_s_ternary(weights, base + column) + 1;
            const int right = i2_s_ternary(weights, base + column + 1) + 1;
            sum += table[left * 3 + right];
        }
        for (; column < input_width; ++column) {
            sum += i2_s_ternary(weights, base + column) * (int)x[column];
        }
    } else if (mode == 2) {
        int column = 0;
        for (; column + 2 < input_width; column += 3) {
            int table[27];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int c = 0; c < 3; ++c) {
                        table[a * 9 + b * 3 + c] = (a - 1) * (int)x[column]
                            + (b - 1) * (int)x[column + 1]
                            + (c - 1) * (int)x[column + 2];
                    }
                }
            }
            const int d0 = i2_s_ternary(weights, base + column) + 1;
            const int d1 = i2_s_ternary(weights, base + column + 1) + 1;
            const int d2 = i2_s_ternary(weights, base + column + 2) + 1;
            sum += table[d0 * 9 + d1 * 3 + d2];
        }
        for (; column < input_width; ++column) {
            sum += i2_s_ternary(weights, base + column) * (int)x[column];
        }
    } else {
        for (int column = 0; column < input_width; ++column) {
            sum += i2_s_ternary(weights, base + column) * (int)x[column];
        }
    }

    const long long packed_len = (long long)weight_rows * input_width / 4;
    const float scale = *reinterpret_cast<const float*>(weights + packed_len);
    output[idx] = (float)sum * activation_scales[input_row] * scale;
}

// Header-free IEEE-754 binary16 conversion. The tied BitNet head stays in its
// canonical little-endian GGUF F16 wire form; converting in the kernel avoids a
// 1.25 GiB f32 expansion and lets the same resident bytes serve every token.
__device__ __forceinline__ float bitnet_f16_bits_to_f32(unsigned short bits) {
    const unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    const unsigned int exp = (bits & 0x7c00u) >> 10;
    const unsigned int frac = (unsigned int)(bits & 0x03ffu);
    unsigned int out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            unsigned int mant = frac;
            int e = -14;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ffu;
            out = sign | ((unsigned int)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((exp + (127u - 15u)) << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

// Exact BitNet-b1.58-2B-4T tied language-model head. One 256-thread block owns
// one vocabulary row and reduces its F16-row · f32-hidden dot in shared memory.
// Weight bytes are read explicitly as little-endian so this does not depend on
// host pointer alignment or CUDA's native half headers. The row base is 64-bit:
// the real matrix contains 328,335,360 elements, and future safe shape checks
// must not regress into Windows-host-sized indexing assumptions.
extern "C" __global__ void bitnet_f16_head_matvec(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_bytes,
    const int weight_rows,
    const int input_width,
    float* __restrict__ output
) {
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row >= weight_rows) return;

    const long long row_base = (long long)row * input_width;
    float partial = 0.0f;
    for (int column = lane; column < input_width; column += 256) {
        const long long byte_index = (row_base + column) * 2;
        const unsigned short bits = (unsigned short)(
            (unsigned short)weight_bytes[byte_index]
            | ((unsigned short)weight_bytes[byte_index + 1] << 8)
        );
        partial += bitnet_f16_bits_to_f32(bits) * input[column];
    }

    __shared__ float scratch[256];
    scratch[lane] = partial;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) output[row] = scratch[0];
}
"#;

    struct CudaBackend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel: CudaFunction,
        kernel_block: CudaFunction,
        kernel_bitnet: CudaFunction,
        kernel_bitnet_f16_head: CudaFunction,
        device_name: String,
        /// GPU-resident weight cache: each Q8_0 weight is uploaded to the GPU
        /// once (keyed by its stable host pointer + length) and reused across
        /// every token, instead of being re-uploaded each step. This is what
        /// makes decode compute-bound (fast) rather than PCIe-bound (slow). The
        /// model's `q8_0_blocks` live for the model's lifetime, so the pointer
        /// is a stable identity; distinct models map at distinct addresses.
        weight_cache: HashMap<(usize, usize), CudaSlice<u8>>,
        /// The official 2B BitNet model ties its F16 token embedding to the LM
        /// head. Keep that 626 MiB wire matrix on the GPU after its first use;
        /// it deliberately has a separate cache from packed I2_S/Q8 weights.
        bitnet_f16_head_weight_cache: HashMap<(usize, usize), BitNetF16HeadWeight>,
    }

    struct BitNetF16HeadWeight {
        /// Weak ownership keeps the allocation identity reserved without
        /// pinning the model's 626 MiB host table after unload. Expired entries
        /// are dropped before lookup, which also releases their device slice.
        host: std::sync::Weak<crate::wire_mmap::WirePages>,
        device: CudaSlice<u8>,
    }

    // SAFETY: cudarc's context/stream/function handles are Send + Sync; we
    // additionally serialize all access behind a Mutex.
    fn backend() -> Option<&'static Mutex<CudaBackend>> {
        static BACKEND: OnceLock<Option<Mutex<CudaBackend>>> = OnceLock::new();
        BACKEND
            .get_or_init(|| match init_backend() {
                Ok(b) => Some(Mutex::new(b)),
                Err(_) => None,
            })
            .as_ref()
    }

    fn compile_options() -> CompileOptions {
        CompileOptions {
            // Match the CPU reference's separate multiply/add: do not let the
            // compiler contract `a*b*c + sum` into a fused multiply-add, which
            // would round differently and could flip a near-tie token.
            fmad: Some(false),
            // Target a virtual arch that supports the `dp4a` 8-bit dot
            // instruction (compute_61, Pascal+). The PTX is forward-compatible, so
            // the driver JITs it for whatever newer GPU is present (e.g. sm_86).
            arch: Some("compute_61"),
            ..Default::default()
        }
    }

    fn init_backend() -> Result<CudaBackend, String> {
        // Honor an explicit device mask before touching the driver: see
        // `devices_masked_by_env`. Initializing CUDA with every device masked can
        // abort the process, and an abort is not catchable below.
        if super::devices_masked_by_env() {
            return Err("CUDA devices masked by CUDA_VISIBLE_DEVICES".to_string());
        }
        let ordinal = super::selected_device_ordinal();
        // cudarc panics (rather than returning Err) when the CUDA driver
        // library cannot be loaded — e.g. on a CI runner or any host with no
        // NVIDIA driver. Catch that so `--all-features` builds fall back to the
        // CPU path instead of aborting the process.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|_| "CUDA driver library not available".to_string())?
            .map_err(|e| format!("CudaContext::new({ordinal}) failed: {e}"))?;
        // After CudaContext::new (which runs cuInit) the driver can report the
        // device count.
        let device_count = result::device::get_count().unwrap_or(0);
        let stream = ctx.default_stream();
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string());
        // Log the exact device the GPU work binds to, so it is unambiguous which
        // physical GPU runs inference (the Intel iGPU is not a CUDA device and can
        // never appear here). Prints once, at first GPU init.
        let cc_major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        let (vram_free, vram_total) = result::mem_get_info().unwrap_or((0, 0));
        eprintln!(
            "[cuda] selected device {ordinal} of {device_count}: \"{device_name}\" \
             (compute capability {cc_major}.{cc_minor}) | VRAM {} MiB free / {} MiB total",
            vram_free / (1024 * 1024),
            vram_total / (1024 * 1024),
        );
        // Same failure class as the driver load above, and the same remedy: cudarc
        // panics from inside its lazy NVRTC loader when libnvrtc cannot be dlopen'd,
        // so the `.map_err` below never gets to run. This matters now that CUDA is
        // compiled into the DEFAULT x86_64 Linux build: installing the NVIDIA driver
        // does NOT install NVRTC (that ships with the CUDA toolkit), so the common
        // driver-only Linux host would abort the process at startup — including
        // under `--gpu off`, because capability detection reaches this path
        // regardless of the GPU switch. Degrade to the CPU path instead.
        //
        // Past the announcement above a usable device exists, so failing here
        // leaves a working GPU idle while inference silently runs on the CPU.
        // Say so once. Failures BEFORE the announcement mean "no CUDA device on
        // this host", which is ordinary and stays quiet.
        (|| -> Result<CudaBackend, String> {
            let ptx: Ptx = std::panic::catch_unwind(|| {
                cudarc::nvrtc::compile_ptx_with_opts(Q8_KERNEL_SRC, compile_options())
            })
            .map_err(|_| "CUDA NVRTC library not available".to_string())?
            .map_err(|e| format!("nvrtc compile failed: {e}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module failed: {e}"))?;
            let kernel = module
                .load_function("q8_0_encoded_linear_rows")
                .map_err(|e| format!("load_function failed: {e}"))?;
            let kernel_block = module
                .load_function("q8_0_block_linear_row")
                .map_err(|e| format!("load_function (block) failed: {e}"))?;
            let kernel_bitnet = module
                .load_function("bitnet_i2_s_linear_rows")
                .map_err(|e| format!("load_function (BitNet I2_S) failed: {e}"))?;
            let kernel_bitnet_f16_head = module
                .load_function("bitnet_f16_head_matvec")
                .map_err(|e| format!("load_function (BitNet F16 head) failed: {e}"))?;
            Ok(CudaBackend {
                ctx,
                stream,
                kernel,
                kernel_block,
                kernel_bitnet,
                kernel_bitnet_f16_head,
                device_name,
                weight_cache: HashMap::new(),
                bitnet_f16_head_weight_cache: HashMap::new(),
            })
        })()
        .inspect_err(|reason| {
            eprintln!(
                "[cuda] device found but the GPU lane could not start ({reason}); \
                 running on the CPU path instead"
            );
        })
    }

    /// Light device probe for the startup hardware profile: opens the CUDA context
    /// and reads device count / name / compute capability / VRAM, WITHOUT compiling
    /// kernels (so it is cheap and side-effect-free relative to full init). Returns
    /// `None` on any machine without a usable CUDA device.
    pub fn probe_capability() -> Option<super::CudaCapability> {
        // See `devices_masked_by_env`: never initialize the driver when the
        // operator has masked every device. This probe runs during the startup
        // hardware profile, so an abort here kills the process before it serves.
        if super::devices_masked_by_env() {
            return None;
        }
        let ordinal = super::selected_device_ordinal();
        // See init_backend: a missing CUDA driver library makes cudarc panic
        // rather than return Err, so guard the first call against it.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .ok()?
            .ok()?;
        let device_count = result::device::get_count().unwrap_or(0).max(0) as usize;
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string());
        let cc_major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0)
            .max(0) as u32;
        let cc_minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0)
            .max(0) as u32;
        let (vram_free, vram_total) = result::mem_get_info().unwrap_or((0, 0));
        Some(super::CudaCapability {
            device_count,
            device_name,
            compute_capability: (cc_major, cc_minor),
            vram_total_bytes: vram_total as u64,
            vram_free_bytes: vram_free as u64,
        })
    }

    /// Return memory cached in the device's default stream-ordered (async) memory
    /// pool to the driver. cudarc allocates device buffers via `cuMemAllocAsync`, so
    /// dropping a `CudaSlice` calls `cuMemFreeAsync`, which returns the bytes to this
    /// pool rather than to the OS — leaving `cuMemGetInfo` (the free-VRAM probe in
    /// `probe_capability`) still counting them as used. After dropping a resident
    /// decode engine we trim the pool to 0 so the freed VRAM becomes visible to the
    /// probe again; otherwise switching to a larger model under-counts free VRAM and
    /// wrongly falls back to the CPU decode path. Best-effort: any error (or a host
    /// without CUDA) is ignored — the caller only loses the reclaim, never correctness.
    pub fn release_async_pool() {
        // Nothing to reclaim when every device is masked, and initializing the
        // driver to find that out can abort — see `devices_masked_by_env`.
        if super::devices_masked_by_env() {
            return;
        }
        let ordinal = super::selected_device_ordinal();
        // Retain the primary context so the driver is initialized and the device
        // handle is valid; held until after the trim. Guard the first call against a
        // missing driver library (cudarc panics rather than returning Err there).
        let _ctx = match std::panic::catch_unwind(|| CudaContext::new(ordinal)) {
            Ok(Ok(ctx)) => ctx,
            _ => return,
        };
        let trace = std::env::var_os("CAMELID_RESIDENT_TRACE").is_some();
        let free_before = result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
        // The just-dropped engine released its weight/KV buffers with `cuMemFreeAsync`
        // (cudarc allocates via `cuMemAllocAsync`), which is STREAM-ORDERED: the pool
        // cannot hand that memory back — to the driver via trim OR to the next
        // allocation — until the device has actually retired the frees. Synchronize the
        // context FIRST, then trim. Without the sync the trim runs before the frees
        // retire and reclaims nothing (measured: free stays pinned at the old model's
        // footprint, so the next model's fit probe under-counts and falls back to CPU —
        // the exact bug this function exists to fix).
        let _ = result::ctx::synchronize();
        // SAFETY: `ordinal` indexes a device the driver just reported via the retained
        // context; the default pool handle is valid for the device's lifetime; and
        // `trim_to` only releases pool reservations that no live allocation is using,
        // so it cannot invalidate any outstanding `CudaSlice`.
        unsafe {
            let dev = match result::device::get(ordinal as core::ffi::c_int) {
                Ok(dev) => dev,
                Err(_) => return,
            };
            let pool = match result::device::get_default_mem_pool(dev) {
                Ok(pool) => pool,
                Err(_) => return,
            };
            let _ = result::mem_pool::trim_to(pool, 0);
        }
        if trace {
            let free_after = result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
            eprintln!(
                "[resident-cuda] release_async_pool: free {} MiB -> {} MiB (reclaimed {} MiB)",
                free_before / (1024 * 1024),
                free_after / (1024 * 1024),
                free_after.saturating_sub(free_before) / (1024 * 1024),
            );
        }
    }

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        match backend() {
            Some(b) => {
                let guard = b.lock().expect("cuda backend mutex poisoned");
                CudaDeviceInfo {
                    available: true,
                    device_name: Some(guard.device_name.clone()),
                    reason: None,
                }
            }
            None => CudaDeviceInfo {
                available: false,
                device_name: None,
                reason: Some("no usable CUDA device or initialization failed".to_string()),
            },
        }
    }

    /// Run the Q8_0 encoded linear on the GPU. `input_rows` input rows (each
    /// `blocks_per_row` blocks of 32 i8 quants + per-block f32 scale) are dotted
    /// against `weight_rows` weight rows (each `blocks_per_row` 34-byte blocks +
    /// per-block decoded f32 scale). Output is `input_rows * weight_rows` f32,
    /// laid out row-major by input row. Returns `false` (caller falls back to
    /// CPU) on any error or shape mismatch.
    #[allow(clippy::too_many_arguments)]
    fn run(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run() first call: input_rows={input_rows} weight_rows={weight_rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} w_scales={} out={} backend={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                weight_scales.len(),
                output.len(),
                backend().is_some(),
            );
        }
        if input_rows == 0 || weight_rows == 0 || blocks_per_row == 0 {
            return false;
        }
        // Shape guards: bail to CPU rather than risk an out-of-bounds GPU read.
        if input_scales.len() != input_rows * blocks_per_row
            || input_quants.len() != input_rows * blocks_per_row * 32
            || weight_bytes.len() != weight_rows * blocks_per_row * 34
            || weight_scales.len() != weight_rows * blocks_per_row
            || output.len() != input_rows * weight_rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let guard = b.lock().expect("cuda backend mutex poisoned");
        match run_inner(
            &guard,
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_inner(
        b: &CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let d_w_bytes = stream.clone_htod(weight_bytes)?;
        let d_w_scales = stream.clone_htod(weight_scales)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        let total = (input_rows * weight_rows) as u32;
        let block_dim = 256u32;
        let grid_dim = total.div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_rows_i = input_rows as i32;
        let weight_rows_i = weight_rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(&d_w_bytes)
            .arg(&d_w_scales)
            .arg(&input_rows_i)
            .arg(&weight_rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: the kernel reads the four input buffers and writes d_out, all
        // sized to match the launch dimensions per the shape guards in `run`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        // Decode: a single input row against `rows` weight rows.
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            1,
            rows,
            blocks_per_row,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        )
    }

    /// Decode matvec over the in-memory `Q8_0Block` byte layout. `weight_bytes`
    /// is `rows * blocks_per_row * 36` bytes (f32 scale + 32 i8 quants/block);
    /// the input row is given as separate `blocks_per_row` scales + quants.
    /// Returns `false` (CPU fallback) on any error or shape mismatch.
    fn run_block(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if rows == 0 || blocks_per_row == 0 {
            return false;
        }
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run_block() first call: rows={rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} out={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                output.len(),
            );
        }
        if input_scales.len() != blocks_per_row
            || input_quants.len() != blocks_per_row * 32
            || weight_bytes.len() != rows * blocks_per_row * 36
            || output.len() != rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let mut guard = b.lock().expect("cuda backend mutex poisoned");
        match run_block_inner(
            &mut guard,
            input_scales,
            input_quants,
            weight_bytes,
            rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    fn run_block_inner(
        b: &mut CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        // Upload this weight to the GPU once and keep it resident; reuse the
        // cached device buffer on every later token. The per-token traffic is
        // then just the small input vector and output vector, so decode becomes
        // GPU-compute-bound instead of PCIe-bound. On a failed upload (e.g. out
        // of VRAM) the `?` propagates and the caller falls back to the CPU dot.
        let key = (weight_bytes.as_ptr() as usize, weight_bytes.len());
        if !b.weight_cache.contains_key(&key) {
            let resident = b.stream.clone_htod(weight_bytes)?;
            b.weight_cache.insert(key, resident);
        }
        let d_w_bytes = b.weight_cache.get(&key).expect("weight just inserted");

        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        // One warp (32 threads) per output row.
        let block_dim = 256u32;
        let grid_dim = ((rows as u32) * 32).div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let rows_i = rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel_block);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(d_w_bytes)
            .arg(&rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: buffers are sized to the launch dimensions per the shape
        // guards in `run_block`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout.
    pub fn try_q8_0_block_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_block_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run_block(
            input_scales,
            input_quants,
            weight_block_bytes,
            rows,
            blocks_per_row,
            output,
        )
    }

    /// Execute canonical I2_S projections on CUDA. Shape or runtime failures
    /// return `false` so the caller can run the cleanroom CPU oracle.
    #[allow(clippy::too_many_arguments)]
    pub fn try_bitnet_i2_s_linear_rows(
        input: &[f32],
        weight_bytes: &[u8],
        input_rows: usize,
        weight_rows: usize,
        input_width: usize,
        mode: u32,
        output: &mut [f32],
    ) -> bool {
        let Some(elements) = input_width.checked_mul(weight_rows) else {
            return false;
        };
        let Some(input_elements) = input_width.checked_mul(input_rows) else {
            return false;
        };
        let Some(output_elements) = weight_rows.checked_mul(input_rows) else {
            return false;
        };
        if input_rows == 0
            || weight_rows == 0
            || input_width == 0
            || input_rows > i32::MAX as usize
            || weight_rows > i32::MAX as usize
            || input_width > i32::MAX as usize
            || !elements.is_multiple_of(128)
            || input.len() != input_elements
            || weight_bytes.len() != elements / 4 + 32
            || output.len() != output_elements
            || output_elements > u32::MAX as usize
            || mode > 2
        {
            return false;
        }
        let packed_len = elements / 4;
        let Ok(scale_bytes) = weight_bytes[packed_len..packed_len + 4].try_into() else {
            return false;
        };
        if !f32::from_le_bytes(scale_bytes).is_finite() {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let Ok((quantized_input, activation_scales)) =
            crate::bitnet_kernels::quantize_activation_rows(input, input_width)
        else {
            return false;
        };
        let mut guard = b.lock().expect("cuda backend mutex poisoned");
        match run_bitnet_inner(
            &mut guard,
            &quantized_input,
            &activation_scales,
            weight_bytes,
            input_rows,
            weight_rows,
            input_width,
            mode,
            output,
        ) {
            Ok(()) => {
                BITNET_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !BITNET_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[bitnet] CUDA I2_S cleanroom kernel active on {} (mode={})",
                        guard.device_name,
                        match mode {
                            1 => "tl1",
                            2 => "tl2",
                            _ => "i2_s",
                        }
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_bitnet_inner(
        b: &mut CudaBackend,
        input: &[i8],
        activation_scales: &[f32],
        weight_bytes: &[u8],
        input_rows: usize,
        weight_rows: usize,
        input_width: usize,
        mode: u32,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let key = (weight_bytes.as_ptr() as usize, weight_bytes.len());
        if !b.weight_cache.contains_key(&key) {
            let resident = b.stream.clone_htod(weight_bytes)?;
            b.weight_cache.insert(key, resident);
        }
        let d_weights = b.weight_cache.get(&key).expect("weight just inserted");
        let d_input = b.stream.clone_htod(input)?;
        let d_activation_scales = b.stream.clone_htod(activation_scales)?;
        let mut d_output = b.stream.alloc_zeros::<f32>(output.len())?;
        let total = output.len() as u32;
        let block_dim = 256_u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(block_dim), 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_rows = input_rows as i32;
        let weight_rows = weight_rows as i32;
        let input_width = input_width as i32;
        let mode = mode as i32;
        let mut builder = b.stream.launch_builder(&b.kernel_bitnet);
        builder
            .arg(&d_input)
            .arg(d_weights)
            .arg(&input_rows)
            .arg(&weight_rows)
            .arg(&input_width)
            .arg(&mode)
            .arg(&d_activation_scales)
            .arg(&mut d_output);
        // SAFETY: every buffer and scalar is shape-checked above; the grid is
        // bounded by output.len(), and the kernel guards its final partial block.
        unsafe { builder.launch(cfg)? };
        b.stream.memcpy_dtoh(&d_output, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    /// Execute the official 2B BitNet model's tied F16 output head on CUDA.
    /// The caller performs the model-identity gate; this boundary still rejects
    /// every malformed shape and non-finite activation, and validates every F16
    /// weight before its immutable host allocation is first cached on-device.
    pub fn try_bitnet_f16_head_matvec(
        input: &[f32],
        weight_pages: &Arc<crate::wire_mmap::WirePages>,
        weight_rows: usize,
        input_width: usize,
        output: &mut [f32],
    ) -> bool {
        let weight_bytes = weight_pages.bytes();
        let Some(weight_elements) = weight_rows.checked_mul(input_width) else {
            return false;
        };
        let Some(expected_weight_bytes) = weight_elements.checked_mul(2) else {
            return false;
        };
        if weight_rows == 0
            || input_width == 0
            || weight_rows > i32::MAX as usize
            || input_width > i32::MAX as usize
            || input.len() != input_width
            || weight_bytes.len() != expected_weight_bytes
            || output.len() != weight_rows
            || !input.iter().all(|value| value.is_finite())
        {
            return false;
        }

        let Some(backend) = backend() else {
            return false;
        };
        let mut guard = backend.lock().expect("cuda backend mutex poisoned");
        // Remove dead-model entries before pointer lookup. Keeping a Weak in
        // each entry prevents its allocation identity from being recycled
        // while stale device bytes still exist under the same key.
        guard
            .bitnet_f16_head_weight_cache
            .retain(|_, cached| cached.host.strong_count() != 0);
        let key = (Arc::as_ptr(weight_pages) as usize, weight_bytes.len());
        if !guard.bitnet_f16_head_weight_cache.contains_key(&key)
            && !weight_bytes.chunks_exact(2).all(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                bits & 0x7c00 != 0x7c00
            })
        {
            return false;
        }

        match run_bitnet_f16_head_inner(
            &mut guard,
            input,
            weight_pages,
            weight_rows,
            input_width,
            output,
        ) {
            Ok(()) if output.iter().all(|value| value.is_finite()) => {
                BITNET_F16_HEAD_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !BITNET_F16_HEAD_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[bitnet] CUDA tied F16 output-head kernel active on {}",
                        guard.device_name
                    );
                }
                true
            }
            Ok(()) | Err(_) => false,
        }
    }

    fn run_bitnet_f16_head_inner(
        b: &mut CudaBackend,
        input: &[f32],
        weight_pages: &Arc<crate::wire_mmap::WirePages>,
        weight_rows: usize,
        input_width: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let weight_bytes = weight_pages.bytes();
        let key = (Arc::as_ptr(weight_pages) as usize, weight_bytes.len());
        if !b.bitnet_f16_head_weight_cache.contains_key(&key) {
            let resident = b.stream.clone_htod(weight_bytes)?;
            b.bitnet_f16_head_weight_cache.insert(
                key,
                BitNetF16HeadWeight {
                    host: Arc::downgrade(weight_pages),
                    device: resident,
                },
            );
        }
        let d_weights = b
            .bitnet_f16_head_weight_cache
            .get(&key)
            .expect("BitNet F16 head weight just inserted")
            .device
            .as_view();
        let d_input = b.stream.clone_htod(input)?;
        let mut d_output = b.stream.alloc_zeros::<f32>(output.len())?;
        let config = LaunchConfig {
            grid_dim: (weight_rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let weight_rows = weight_rows as i32;
        let input_width = input_width as i32;
        let mut builder = b.stream.launch_builder(&b.kernel_bitnet_f16_head);
        builder
            .arg(&d_input)
            .arg(&d_weights)
            .arg(&weight_rows)
            .arg(&input_width)
            .arg(&mut d_output);
        // SAFETY: the public boundary checked all byte/element products, buffer
        // lengths and i32/grid conversions. Each block owns one validated row,
        // and this path always launches the kernel's required 256 threads.
        unsafe { builder.launch(config)? };
        b.stream.memcpy_dtoh(&d_output, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Reference dot, identical in operation order to the CPU engine's
        // `dot_q8_0_encoded_row_with_scales`: exact i32 block dot, then a
        // sequential f32 accumulation of `(int_sum as f32) * w_scale * i_scale`
        // in block order, no FMA. The GPU kernel must match this bit-for-bit.
        fn reference_row(
            input_scales: &[f32],
            input_quants: &[i8],
            weight_block_quants: &[&[i8]],
            weight_scales: &[f32],
            blocks_per_row: usize,
        ) -> f32 {
            let mut sum = 0.0f32;
            for b in 0..blocks_per_row {
                let mut int_sum = 0i32;
                for j in 0..32 {
                    int_sum +=
                        i32::from(weight_block_quants[b][j]) * i32::from(input_quants[b * 32 + j]);
                }
                sum += int_sum as f32 * weight_scales[b] * input_scales[b];
            }
            sum
        }

        // Tiny deterministic LCG so the test needs no rand dependency and is
        // reproducible across runs/platforms.
        struct Lcg(u64);
        impl Lcg {
            fn next_u32(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 32) as u32
            }
            fn next_i8(&mut self) -> i8 {
                (self.next_u32() & 0xff) as u8 as i8
            }
            fn next_scale(&mut self) -> f32 {
                // Small positive f16-ish scales, like real Q8_0 block scales.
                ((self.next_u32() % 1000) as f32 + 1.0) / 4096.0
            }
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_bitnet_i2_s_cleanroom_modes_match_cpu_oracle() {
            if !detect_cuda_device().available {
                if std::env::var("CAMELID_REQUIRE_CUDA_TESTS").as_deref() == Ok("1") {
                    panic!(
                        "CAMELID_REQUIRE_CUDA_TESTS=1 but the CUDA backend (including NVRTC \
                         compilation of bitnet_i2_s_linear_rows) is unavailable"
                    );
                }
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let input_rows = 3;
            let weight_rows = 3;
            let input_width = 128;
            let values = (0..weight_rows * input_width)
                .map(|index| match index % 3 {
                    0 => -1_i8,
                    1 => 0,
                    _ => 1,
                })
                .collect::<Vec<_>>();
            let mut wire = vec![0_u8; values.len() / 4];
            for (index, value) in values.iter().copied().enumerate() {
                let tile = index / 128;
                let within = index % 128;
                let code: u8 = match value {
                    -1 => 0,
                    // Both public zero encodings must decode identically.
                    0 if index.is_multiple_of(2) => 1,
                    0 => 3,
                    1 => 2,
                    _ => unreachable!(),
                };
                wire[tile * 32 + within % 32] |= code << (6 - 2 * (within / 32));
            }
            // A non-power-of-two scale exposes multiplication-association drift.
            wire.extend_from_slice(&0.000_581_790_6_f32.to_le_bytes());
            wire.extend_from_slice(&[0; 28]);
            let one_input = (0..input_width)
                .map(|index| ((index as f32 + 0.25) * 0.173).sin() * 3.75)
                .collect::<Vec<_>>();
            let second_input = (0..input_width)
                .map(|index| ((index as f32 + 0.5) * 0.119).cos() * 0.037 + 0.002)
                .collect::<Vec<_>>();
            let zero_input = vec![0.0_f32; input_width];
            let input_rows_data = [one_input, second_input, zero_input];
            let input = input_rows_data.concat();

            let mut invalid_wire = wire.clone();
            let scale_offset = weight_rows * input_width / 4;
            invalid_wire[scale_offset..scale_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            let mut invalid_output = vec![0.0_f32; input_rows * weight_rows];
            let before_invalid = bitnet_run_count();
            assert!(!try_bitnet_i2_s_linear_rows(
                &input,
                &invalid_wire,
                input_rows,
                weight_rows,
                input_width,
                0,
                &mut invalid_output,
            ));
            assert_eq!(bitnet_run_count(), before_invalid);

            for (mode, cpu_mode) in [
                (0, crate::bitnet_kernels::BitNetKernelMode::I2S),
                (1, crate::bitnet_kernels::BitNetKernelMode::Tl1),
                (2, crate::bitnet_kernels::BitNetKernelMode::Tl2),
            ] {
                let expected = crate::bitnet_kernels::i2_s_matmul(
                    &wire,
                    &input_rows_data,
                    weight_rows,
                    cpu_mode,
                )
                .expect("CPU oracle")
                .concat();
                let mut actual = vec![0.0_f32; input_rows * weight_rows];
                let before = bitnet_run_count();
                assert!(try_bitnet_i2_s_linear_rows(
                    &input,
                    &wire,
                    input_rows,
                    weight_rows,
                    input_width,
                    mode,
                    &mut actual,
                ));
                assert!(bitnet_run_count() > before);
                for (got, want) in actual.iter().zip(&expected) {
                    assert_eq!(got.to_bits(), want.to_bits(), "got={got} want={want}");
                }
            }
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_bitnet_f16_head_matches_cpu_oracle_and_reuses_weights() {
            use std::io::Write;

            if !detect_cuda_device().available {
                if std::env::var("CAMELID_REQUIRE_CUDA_TESTS").as_deref() == Ok("1") {
                    panic!(
                        "CAMELID_REQUIRE_CUDA_TESTS=1 but the CUDA backend (including NVRTC \
                         compilation of bitnet_f16_head_matvec) is unavailable"
                    );
                }
                eprintln!("skipping: no CUDA device available");
                return;
            }

            fn make_pages(bytes: &[u8]) -> Arc<crate::wire_mmap::WirePages> {
                let mut file = tempfile::tempfile().expect("temporary F16 wire file");
                file.write_all(bytes).expect("write F16 wire bytes");
                file.flush().expect("flush F16 wire bytes");
                crate::wire_mmap::WirePages::read_from_file(&file, 0, bytes.len())
                    .expect("page-backed F16 wire bytes")
            }

            fn reference_row(row: &[u8], input: &[f32]) -> f32 {
                let mut sum = 0.0_f32;
                for (bytes, value) in row.chunks_exact(2).zip(input) {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    sum += crate::tensor::f16_bits_to_f32(bits) * value;
                }
                sum
            }

            // Seven distinct rows and a width that is neither a warp nor block
            // multiple exercise both the kernel's final partial stride and its
            // 64-bit row addressing. Include signed zero, subnormals, ordinary
            // values, and the largest finite half without admitting NaN/Inf.
            let weight_rows = 7;
            let input_width = 513;
            let finite_bits = [
                0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3400, 0xb800, 0x3d00, 0xc000, 0x4300, 0x7bff,
            ];
            let mut wire = Vec::with_capacity(weight_rows * input_width * 2);
            for row in 0..weight_rows {
                for column in 0..input_width {
                    let bits = finite_bits[(row * 17 + column * 13) % finite_bits.len()];
                    wire.extend_from_slice(&bits.to_le_bytes());
                }
            }
            let pages = make_pages(&wire);
            let first_input = (0..input_width)
                .map(|index| ((index as f32 + 0.375) * 0.071).sin() * 0.000_23)
                .collect::<Vec<_>>();
            let second_input = (0..input_width)
                .map(|index| ((index as f32 + 0.625) * 0.053).cos() * -0.000_19 + 0.000_011)
                .collect::<Vec<_>>();

            let mut invalid_output = vec![0.0_f32; weight_rows];
            let before_invalid = bitnet_f16_head_run_count();
            assert!(!try_bitnet_f16_head_matvec(
                &first_input[..input_width - 1],
                &pages,
                weight_rows,
                input_width,
                &mut invalid_output,
            ));
            let mut nonfinite_input = first_input.clone();
            nonfinite_input[31] = f32::NAN;
            assert!(!try_bitnet_f16_head_matvec(
                &nonfinite_input,
                &pages,
                weight_rows,
                input_width,
                &mut invalid_output,
            ));
            let mut nonfinite_wire = wire.clone();
            nonfinite_wire[46..48].copy_from_slice(&0x7c00_u16.to_le_bytes());
            let nonfinite_pages = make_pages(&nonfinite_wire);
            assert!(!try_bitnet_f16_head_matvec(
                &first_input,
                &nonfinite_pages,
                weight_rows,
                input_width,
                &mut invalid_output,
            ));
            assert_eq!(bitnet_f16_head_run_count(), before_invalid);

            let cache_before = backend()
                .expect("CUDA backend")
                .lock()
                .expect("CUDA backend mutex")
                .bitnet_f16_head_weight_cache
                .len();
            for input in [&first_input, &second_input] {
                let expected = (0..weight_rows)
                    .map(|row| {
                        let start = row * input_width * 2;
                        reference_row(&wire[start..start + input_width * 2], input)
                    })
                    .collect::<Vec<_>>();
                let mut actual = vec![0.0_f32; weight_rows];
                let before = bitnet_f16_head_run_count();
                assert!(try_bitnet_f16_head_matvec(
                    input,
                    &pages,
                    weight_rows,
                    input_width,
                    &mut actual,
                ));
                assert_eq!(bitnet_f16_head_run_count(), before + 1);
                for (row, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                    let tolerance = 0.000_2 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tolerance,
                        "row={row} got={got} want={want} tolerance={tolerance}"
                    );
                }
            }
            let cache_after = backend()
                .expect("CUDA backend")
                .lock()
                .expect("CUDA backend mutex")
                .bitnet_f16_head_weight_cache
                .len();
            assert_eq!(
                cache_after,
                cache_before + 1,
                "weights uploaded more than once"
            );
        }

        // Requires a CUDA device; ignored by default so GPU-less CI (which has
        // no NVIDIA driver) compiles but does not run it. Run on a CUDA host
        // with `cargo test --features cuda -- --ignored`.
        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_q8_kernel_is_bit_identical_to_cpu_reference() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize; // TinyLlama hidden 2048 = 64 blocks of 32
            let weight_rows = 300usize; // not a multiple of the 256 block size
            let mut rng = Lcg(0x1234_5678_9abc_def0);

            // Input row.
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows: 34-byte blocks (2-byte scale header + 32 quants) plus
            // a separately decoded f32 scale per block (as the engine passes).
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 34];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 34;
                    // header bytes are ignored by the kernel; fill with noise.
                    weight_bytes[blk] = rng.next_i8() as u8;
                    weight_bytes[blk + 1] = rng.next_i8() as u8;
                    for j in 0..32 {
                        weight_bytes[blk + 2 + j] = rng.next_i8() as u8;
                    }
                    weight_scales[r * blocks_per_row + b] = rng.next_scale();
                }
            }

            // CPU reference.
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 34 + 2;
                        // reinterpret the u8 quants as i8
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            // GPU.
            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_encoded_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                &weight_scales,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU kernel did not run");

            let mut mismatches = 0;
            for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                if g.to_bits() != e.to_bits() {
                    if mismatches < 5 {
                        eprintln!(
                            "row {r}: gpu={g} ({:#010x}) cpu={e} ({:#010x})",
                            g.to_bits(),
                            e.to_bits()
                        );
                    }
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "{mismatches}/{weight_rows} rows differ bit-for-bit from the CPU reference"
            );
        }

        // The fast block kernel sums blocks across a warp (reassociated f32), so
        // it matches the CPU reference very closely but not bit-for-bit. Assert a
        // tight relative tolerance; end-to-end token identity is covered by the
        // TinyLlama parity audit.
        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_block_kernel_matches_cpu_reference_within_tolerance() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize;
            let weight_rows = 257usize;
            let mut rng = Lcg(0x0fed_cba9_8765_4321);

            // Single input row, as separate scales + quants (what the engine
            // passes via with_q8_0_block_scales_and_quants).
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows in the Q8_0Block byte layout: 36 bytes/block =
            // f32 scale (LE) + 32 i8 quants.
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 36];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 36;
                    let scale = rng.next_scale();
                    weight_scales[r * blocks_per_row + b] = scale;
                    weight_bytes[blk..blk + 4].copy_from_slice(&scale.to_le_bytes());
                    for j in 0..32 {
                        weight_bytes[blk + 4 + j] = rng.next_i8() as u8;
                    }
                }
            }

            // CPU reference (same op order as q8_0_dot_rows scalar path).
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 36 + 4;
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_block_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU block kernel did not run");

            let mut worst = 0.0f32;
            for (g, e) in got.iter().zip(expected.iter()) {
                let denom = e.abs().max(1.0);
                worst = worst.max((g - e).abs() / denom);
            }
            assert!(
                worst < 1e-4,
                "block-kernel worst relative error {worst} exceeds 1e-4 vs CPU reference"
            );
        }
    }
}

#[cfg(test)]
mod device_mask_tests {
    use super::{devices_masked, select_gpu_acceleration_info, GpuAccelerationInfo};

    #[test]
    fn only_an_explicit_no_devices_value_masks_cuda() {
        // The documented CPU-pinning setting, and the empty-list spelling.
        assert!(devices_masked(Some("-1")));
        assert!(devices_masked(Some("  -1  ")));
        assert!(devices_masked(Some("")));
        // Unset must never mask: that is every ordinary GPU host.
        assert!(!devices_masked(None));
        // A real device selection must never mask — masking here would silently
        // drop the GPU path for anyone pinning to a specific card.
        assert!(!devices_masked(Some("0")));
        assert!(!devices_masked(Some("1")));
        assert!(!devices_masked(Some("0,1")));
        // Not a mask value, and must not be parsed as one.
        assert!(!devices_masked(Some("-1,0")));
    }

    #[test]
    fn user_facing_gpu_selection_recognizes_metal_and_prefers_cuda() {
        assert_eq!(
            select_gpu_acceleration_info(false, None, true, Some("Apple M4 Max".to_string()),),
            GpuAccelerationInfo {
                available: true,
                device_name: Some("Apple M4 Max".to_string()),
                backend: "metal",
            }
        );
        assert_eq!(
            select_gpu_acceleration_info(
                true,
                Some("NVIDIA RTX".to_string()),
                true,
                Some("Apple GPU".to_string()),
            ),
            GpuAccelerationInfo {
                available: true,
                device_name: Some("NVIDIA RTX".to_string()),
                backend: "cuda",
            }
        );
        assert_eq!(
            select_gpu_acceleration_info(false, None, false, Some("ignored".to_string())),
            GpuAccelerationInfo {
                available: false,
                device_name: None,
                backend: "none",
            }
        );
    }
}
