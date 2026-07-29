#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
};

#[cfg(target_os = "macos")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub available: bool,
    pub device_name: Option<String>,
    pub low_power: Option<bool>,
    pub headless: Option<bool>,
    pub removable: Option<bool>,
    pub has_unified_memory: Option<bool>,
    pub registry_id: Option<u64>,
    pub max_threads_per_threadgroup: Option<(u64, u64, u64)>,
    pub note: Option<String>,
}

#[cfg(target_os = "macos")]
struct MetalLinearKernel {
    device: Device,
    queue: CommandQueue,
    descriptor_pipeline: ComputePipelineState,
    transposed_pipeline: ComputePipelineState,
    q8_0_encoded_pipeline: ComputePipelineState,
    q8_0_encoded_rows_pipeline: ComputePipelineState,
    q8_0_block_pipeline: ComputePipelineState,
    q8_0_block_simd_pipeline: ComputePipelineState,
    q8_0_block_simd_mr_pipeline: ComputePipelineState,
    q8_0_block_simd_qmv4_pipeline: ComputePipelineState,
    q8_0_block_ksplit_pipeline: ComputePipelineState,
    q8_0_block_ksplit_f32y_pipeline: ComputePipelineState,
    q8_0_block_ksplit_f32y_wire_pipeline: ComputePipelineState,
    q4_0_block_ksplit_f32y_wire_pipeline: ComputePipelineState,
    nvfp4_block_ksplit_f32y_wire_pipeline: ComputePipelineState,
    q8_0_block_ksplit_f32y_wire_nsg8_pipeline: ComputePipelineState,
    #[allow(dead_code)] // batched-column verify GEMV; exercised by the C0 unit test,
    // consumed by the speculative-verify lane in a later checkpoint
    q8_0_block_ksplit_f32y_wire_nsg8_verify_pipeline: ComputePipelineState,
    q8_0_block_ksplit_f32y_wire_gemm_pipeline: ComputePipelineState,
    q8_0_block_wire_mm_pipeline: ComputePipelineState,
    q8_0_block_wire_mm_f16o_pipeline: ComputePipelineState,
    rms_norm_pipeline: ComputePipelineState,
    rms_norm_per_head_pipeline: ComputePipelineState,
    residual_add_pipeline: ComputePipelineState,
    silu_mul_pipeline: ComputePipelineState,
    gelu_mul_pipeline: ComputePipelineState,
    soft_cap_pipeline: ComputePipelineState,
    scale_pipeline: ComputePipelineState,
    rope_rotate_pipeline: ComputePipelineState,
    attention_decode_pipeline: ComputePipelineState,
    attention_decode_kv16_pipeline: ComputePipelineState,
    attention_decode_v2_pipeline: ComputePipelineState,
    attention_decode_v2_kv16_pipeline: ComputePipelineState,
    quantize_q8_0_pipeline: ComputePipelineState,
    kv_scatter_pipeline: ComputePipelineState,
    kv_scatter_kv16_pipeline: ComputePipelineState,
    f32_to_f16_pipeline: ComputePipelineState,
    rms_norm_batch_pipeline: ComputePipelineState,
    rms_norm_batch_f16o_pipeline: ComputePipelineState,
    silu_mul_f16o_pipeline: ComputePipelineState,
    rope_rotate_batch_pipeline: ComputePipelineState,
    kv_scatter_batch_pipeline: ComputePipelineState,
    attention_prefill_v3_pipeline: ComputePipelineState,
    attention_prefill_flash_pipeline: ComputePipelineState,
    #[allow(dead_code)] // bit-exact reference variant; exercised by unit tests
    half_mm_batched_pipeline: ComputePipelineState,
    half_mm_batched_f16o_pipeline: ComputePipelineState,
    transpose_v16_pipeline: ComputePipelineState,
    rope_scatter_qh_pipeline: ComputePipelineState,
    rope_scatter_qh_h_pipeline: ComputePipelineState,
    rms_norm_batch_h_pipeline: ComputePipelineState,
    residual_add_h_pipeline: ComputePipelineState,
    silu_mul_h2_pipeline: ComputePipelineState,
    softmax_causal_rows_pipeline: ComputePipelineState,
    rms_norm_quantize_pipeline: ComputePipelineState,
    silu_mul_quantize_pipeline: ComputePipelineState,
    argmax_f32_greedy_pipeline: ComputePipelineState,
    sample_gumbel_f32_pipeline: ComputePipelineState,
    attention_decode_splitk_pipeline: ComputePipelineState,
    attention_decode_splitk_kv16_pipeline: ComputePipelineState,
    attention_decode_splitk_kv16_direct_pipeline: ComputePipelineState,
    #[allow(dead_code)] // stage-bandwidth probe variant; exercised by the depth-probe test
    attention_splitk_kv16_stageonly_pipeline: ComputePipelineState,
    attention_decode_splitk_merge_pipeline: ComputePipelineState,
    // WIN2METAL Phase 4 — TREE-verify attention clones (slot-indirected draft tail).
    attention_decode_f32_tree_pipeline: ComputePipelineState,
    attention_decode_v2_tree_pipeline: ComputePipelineState,
    attention_decode_splitk_kv16_tree_pipeline: ComputePipelineState,
    attention_decode_splitk_kv16_direct_tree_pipeline: ComputePipelineState,
    embed_row_gather_q8_wire_pipeline: ComputePipelineState,
    active_command_buffer: Mutex<Option<metal::CommandBuffer>>,
    /// Recycled per-token scratch buffers keyed by power-of-two byte class. The resident
    /// decode loop allocates hundreds of small scratch/scalar buffers per token; fresh
    /// MTLBuffer allocation crosses into IOGPU (a mach call per buffer, observed directly
    /// in time-profile samples of the decode loop), so settled tokens return their scratch
    /// here instead of dropping it.
    scratch_pool: Mutex<HashMap<u64, Vec<Buffer>>>,
}

#[cfg(target_os = "macos")]
struct DeferredRead {
    buffer: Buffer,
    dest_ptr: usize,
    dest_len: usize,
}

#[cfg(target_os = "macos")]
struct MetalLinearCache {
    // Permanent caches
    weight_buffers: HashMap<(usize, usize), Buffer>,
    q8_block_weight_buffers: HashMap<(usize, usize), Buffer>,
    /// Wire-format (34-byte f16-scale) conversions of 36-byte block slices, keyed by the
    /// SOURCE slice identity; the first source block is stored for the aliasing guard
    /// (the GPU contents are converted, so they cannot be probed against the source).
    q8_wire_weight_buffers: HashMap<(usize, usize), (Buffer, [u8; 36])>,
    /// Offset-0 NoCopy buffers wrapped over page-aligned wire-page allocations
    /// (fast-load). The Arc keeps each allocation alive for as long as the cache
    /// holds its buffer, so a dropped model can never leave the GPU pointing at
    /// freed memory.
    q8_wire_nocopy_buffers:
        HashMap<(usize, usize), (Buffer, std::sync::Arc<crate::wire_mmap::WirePages>)>,

    // Transient caches (activation buffers, scalars, deferred reads)
    activation_buffers: HashMap<(usize, usize), Buffer>,
    scalar_buffers: Vec<Buffer>,
    scalar_index: usize,
    deferred_reads: Vec<DeferredRead>,
}

#[cfg(target_os = "macos")]
impl MetalLinearCache {
    fn new() -> Self {
        Self {
            weight_buffers: HashMap::new(),
            q8_block_weight_buffers: HashMap::new(),
            q8_wire_weight_buffers: HashMap::new(),
            q8_wire_nocopy_buffers: HashMap::new(),
            activation_buffers: HashMap::new(),
            scalar_buffers: Vec::new(),
            scalar_index: 0,
            deferred_reads: Vec::new(),
        }
    }

    fn get_activation_buffer(&mut self, device: &Device, needed: usize, ptr: *const u8) -> Buffer {
        let key = (ptr as usize, needed);
        if let Some(buffer) = self.activation_buffers.get(&key) {
            return buffer.to_owned();
        }
        let buffer = device.new_buffer(needed as u64, MTLResourceOptions::StorageModeShared);
        self.activation_buffers.insert(key, buffer.to_owned());
        buffer
    }

    fn get_scalar_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        if self.scalar_buffers.len() <= self.scalar_index {
            let buffer = device.new_buffer(needed as u64, MTLResourceOptions::StorageModeShared);
            self.scalar_buffers.push(buffer);
        } else {
            let buffer = &self.scalar_buffers[self.scalar_index];
            if buffer.length() < needed as u64 {
                self.scalar_buffers[self.scalar_index] =
                    device.new_buffer(needed as u64, MTLResourceOptions::StorageModeShared);
            }
        }
        let buf = self.scalar_buffers[self.scalar_index].to_owned();
        self.scalar_index += 1;
        buf
    }

    fn input_buffer(&mut self, device: &Device, needed: usize, ptr: *const f32) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn output_buffer(&mut self, device: &Device, needed: usize, ptr: *mut f32) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn aux_output_buffer(&mut self, device: &Device, needed: usize, ptr: *mut f32) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn scalar_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        self.get_scalar_buffer(device, needed)
    }

    fn q8_input_scales_buffer(
        &mut self,
        device: &Device,
        needed: usize,
        ptr: *const f32,
    ) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn q8_input_quants_buffer(&mut self, device: &Device, needed: usize, ptr: *const i8) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn q8_encoded_rows_buffer(&mut self, device: &Device, needed: usize, ptr: *const u8) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn q8_weight_scales_buffer(
        &mut self,
        device: &Device,
        needed: usize,
        ptr: *const f32,
    ) -> Buffer {
        self.get_activation_buffer(device, needed, ptr.cast())
    }

    fn weight_buffer(&mut self, device: &Device, weights: &[f32]) -> Buffer {
        let key = (weights.as_ptr() as usize, weights.len());
        if let Some(buffer) = self.weight_buffers.get(&key) {
            if cached_weight_contents_match(buffer, weights.as_ptr().cast(), key.1 * 4) {
                return buffer.to_owned();
            }
            // (ptr,len) collided with a freed allocation — refresh the contents in place.
            write_buffer_f32(buffer, weights);
            return buffer.to_owned();
        }
        let buffer = device.new_buffer(
            std::mem::size_of_val(weights) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        write_buffer_f32(&buffer, weights);
        self.weight_buffers.insert(key, buffer.to_owned());
        buffer
    }

    fn q8_block_weight_buffer(&mut self, device: &Device, weight_blocks: &[u8]) -> Buffer {
        let key = (weight_blocks.as_ptr() as usize, weight_blocks.len());
        if let Some(buffer) = self.q8_block_weight_buffers.get(&key) {
            if cached_weight_contents_match(buffer, weight_blocks.as_ptr(), weight_blocks.len()) {
                return buffer.to_owned();
            }
            // (ptr,len) collided with a freed allocation — refresh the contents in place.
            write_buffer_u8(buffer, weight_blocks);
            return buffer.to_owned();
        }
        let buffer = device.new_buffer(
            weight_blocks.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        write_buffer_u8(&buffer, weight_blocks);
        self.q8_block_weight_buffers.insert(key, buffer.to_owned());
        buffer
    }

    /// Wire-format weight buffer: converts decoded 36-byte f32-scale Q8_0 blocks back to
    /// the raw GGUF 34-byte f16-scale layout at upload time (exact: the f32 scales are
    /// themselves f16 round-trips from the file), cutting the GPU's weight-byte traffic by
    /// ~5.9%. Cached permanently keyed on the source slice, with the first source block
    /// stored for the aliasing guard.
    fn q8_wire_weight_buffer(&mut self, device: &Device, weight_blocks_36: &[u8]) -> Buffer {
        let key = (weight_blocks_36.as_ptr() as usize, weight_blocks_36.len());
        let n_blocks = weight_blocks_36.len() / 36;
        let mut probe = [0u8; 36];
        let probe_len = weight_blocks_36.len().min(36);
        probe[..probe_len].copy_from_slice(&weight_blocks_36[..probe_len]);
        if let Some((buffer, cached_probe)) = self.q8_wire_weight_buffers.get(&key) {
            if *cached_probe == probe {
                return buffer.to_owned();
            }
        }
        let buffer = device.new_buffer(
            (n_blocks * 34) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let dst = buffer.contents() as *mut u8;
            for b in 0..n_blocks {
                let src = &weight_blocks_36[b * 36..b * 36 + 36];
                let scale = f32::from_le_bytes(src[..4].try_into().unwrap());
                let half_bits = f32_to_f16_bits(scale).to_le_bytes();
                std::ptr::copy_nonoverlapping(half_bits.as_ptr(), dst.add(b * 34), 2);
                std::ptr::copy_nonoverlapping(src[4..].as_ptr(), dst.add(b * 34 + 2), 32);
            }
        }
        self.q8_wire_weight_buffers
            .insert(key, (buffer.to_owned(), probe));
        buffer
    }

    /// Wrap a page-aligned wire-page allocation with an offset-0 NoCopy buffer:
    /// the GPU reads the allocation in place — no upload, no conversion. Keyed by
    /// (pointer, len); the stored Arc pins the allocation for the buffer's lifetime.
    fn q8_wire_nocopy_buffer(
        &mut self,
        device: &Device,
        pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
    ) -> Buffer {
        let key = (pages.base_ptr() as usize, pages.byte_len());
        if let Some((buffer, _)) = self.q8_wire_nocopy_buffers.get(&key) {
            return buffer.to_owned();
        }
        let buffer = device.new_buffer_with_bytes_no_copy(
            pages.base_ptr() as *const std::ffi::c_void,
            pages.alloc_len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        self.q8_wire_nocopy_buffers
            .insert(key, (buffer.to_owned(), std::sync::Arc::clone(pages)));
        buffer
    }
}

/// Hardware GPU timestamps from a completed command buffer: (GPU busy window µs,
/// kernel-prep window µs). Uses objc selectors not surfaced by the metal crate.
#[cfg(target_os = "macos")]
fn command_buffer_gpu_times_us(cb: &metal::CommandBuffer) -> (u128, u128) {
    use metal::foreign_types::ForeignType;
    use metal::objc::{msg_send, sel, sel_impl};
    unsafe {
        let p = cb.as_ptr();
        let gstart: f64 = msg_send![p, GPUStartTime];
        let gend: f64 = msg_send![p, GPUEndTime];
        let kstart: f64 = msg_send![p, kernelStartTime];
        let kend: f64 = msg_send![p, kernelEndTime];
        (
            ((gend - gstart) * 1e6) as u128,
            ((kend - kstart) * 1e6) as u128,
        )
    }
}

/// IEEE 754 binary16 bits -> f32. Exact in every case (f32 covers the whole f16 range,
/// subnormals included), so it is the exact inverse of [`f32_to_f16_bits`] on any value that
/// round-trips. Used by the KV readback when the resident cache is in half mode.
#[cfg(target_os = "macos")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = ((bits & 0x03ff) as u32) << 13;
    let rest = match exp {
        // Inf/NaN: max f32 exponent, mantissa carried through (quiet bit preserved by the shift).
        0x1f => 0x7f80_0000 | mant,
        // Zero / subnormal: renormalize into f32, which has the range to hold it exactly.
        0 => {
            if mant == 0 {
                0
            } else {
                let mut m = mant;
                let mut e: u32 = 127 - 15 + 1;
                while m & 0x0080_0000 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                ((e) << 23) | (m & 0x007f_ffff)
            }
        }
        // Normal: rebias the exponent (f16 bias 15 -> f32 bias 127).
        _ => ((exp + 127 - 15) << 23) | mant,
    };
    f32::from_bits(sign | rest)
}

/// f32 -> IEEE 754 binary16 bits, round-to-nearest-even (exact for values that started as
/// f16, which all GGUF Q8_0 scales did).
#[cfg(target_os = "macos")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf/NaN
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflow -> inf
    }
    if unbiased >= -14 {
        // Normal half. Round mantissa from 23 to 10 bits, nearest-even.
        let mut half_exp = (unbiased + 15) as u32;
        let mut half_mant = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
            half_mant += 1;
            if half_mant == 0x400 {
                half_mant = 0;
                half_exp += 1;
                if half_exp >= 31 {
                    return sign | 0x7c00;
                }
            }
        }
        return sign | ((half_exp as u16) << 10) | (half_mant as u16);
    }
    // Subnormal half (or zero).
    if unbiased < -25 {
        return sign;
    }
    let full_mant = mant | 0x0080_0000;
    // Subnormal path: shift with round-to-nearest-even.
    let total_shift = 13 + (-14 - unbiased) as u32;
    let shifted = full_mant >> total_shift;
    let rem_mask = (1u32 << total_shift) - 1;
    let rem = full_mant & rem_mask;
    let halfway = 1u32 << (total_shift - 1);
    let mut out = shifted;
    if rem > halfway || (rem == halfway && (out & 1) == 1) {
        out += 1;
    }
    sign | (out as u16)
}

/// Guard for the permanent weight-buffer caches, which key cached GPU copies by the CPU
/// slice's (pointer, length). That identity is unsound across an allocation's lifetime: when
/// a weight slice is freed and a different one of the same length is later allocated at the
/// same address (allocator reuse — routinely triggered by the test suite, and by reloading a
/// model in a long-lived process), a stale cache hit would silently serve the OLD weights.
/// Probe the first and last 32 bytes of the cached copy against the live slice (Q8_0 blocks
/// start with their f32 scale, so distinct real weights diverge immediately); on mismatch
/// the caller rewrites the buffer contents in place. Cost: a <=64-byte compare per cache
/// hit, noise next to the dispatch itself.
#[cfg(target_os = "macos")]
fn cached_weight_contents_match(buffer: &Buffer, bytes: *const u8, len: usize) -> bool {
    if buffer.length() as usize != len || len == 0 {
        return false;
    }
    let probe = len.min(32);
    unsafe {
        let gpu = buffer.contents() as *const u8;
        std::slice::from_raw_parts(gpu, probe) == std::slice::from_raw_parts(bytes, probe)
            && std::slice::from_raw_parts(gpu.add(len - probe), probe)
                == std::slice::from_raw_parts(bytes.add(len - probe), probe)
    }
}

#[cfg(target_os = "macos")]
static METAL_LINEAR_KERNEL: OnceLock<Option<MetalLinearKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_LINEAR_CACHE: OnceLock<Mutex<MetalLinearCache>> = OnceLock::new();

#[cfg(target_os = "macos")]
const LINEAR_ROW_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- NVFP4 decode primitives (GABBRO M3) --------------------------------------
// Bit-for-bit twins of the CPU oracle in src/tensor/mod.rs: KVALUES_MXFP4 (the
// E2M1 codebook, magnitudes doubled with the sign in nibble bit 3) and
// ue4m3_to_f32_const (the UE4M3 sub-block-scale decode, carrying the 0.5 pair-rule
// factor). Every op is an exact power-of-two scale of an exactly-representable
// mantissa, so ldexp cannot round differently from the Rust const path.
constant float NVFP4_KVALUES[16] = {
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0
};
inline float nvfp4_ue4m3_to_f32(uchar b) {
    if (b == 0x00 || b == 0x7F) {
        return 0.0f; // 0x7F is the NaN sentinel, FLUSHED (raw-byte check; 0xFF is not)
    }
    int e = (int(b) >> 3) & 0xF;
    float man = float(b & 0x7);
    float raw = (e == 0) ? ldexp(man, -9) : ldexp(1.0f + man * 0.125f, e - 7);
    return raw * 0.5f;
}

kernel void linear_row_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    float sum = 0.0;
    for (uint inner = 0; inner < rows; ++inner) {
        sum += input[inner] * weights[inner * cols + gid];
    }
    output[gid] += sum;
}

kernel void linear_row_transposed_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    float sum = 0.0;
    uint base = gid * rows;
    for (uint inner = 0; inner < rows; ++inner) {
        sum += input[inner] * weights[base + inner];
    }
    output[gid] = sum;
}

kernel void q8_0_encoded_linear_row(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* encoded_rows [[buffer(2)]],
    device const float* weight_scales [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& blocks_per_row [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint encoded_block_bytes = 34;
    float sum = 0.0;
    uint row_base = gid * blocks_per_row * encoded_block_bytes;
    uint scale_base = gid * blocks_per_row;
    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int int_sum = 0;
        uint encoded_base = row_base + block_idx * encoded_block_bytes + 2;
        uint input_base = block_idx * block_values;
        for (uint lane = 0; lane < block_values; ++lane) {
            int_sum += int(encoded_rows[encoded_base + lane]) * int(input_quants[input_base + lane]);
        }
        sum += float(int_sum) * weight_scales[scale_base + block_idx] * input_scales[block_idx];
    }
    output[gid] = sum;
}

kernel void q8_0_encoded_linear_rows(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* encoded_rows [[buffer(2)]],
    device const float* weight_scales [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& blocks_per_row [[buffer(5)]],
    constant uint& input_rows [[buffer(6)]],
    constant uint& weight_rows [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = input_rows * weight_rows;
    if (gid >= total) return;
    constexpr uint block_values = 32;
    constexpr uint encoded_block_bytes = 34;
    uint weight_row = gid / input_rows;
    uint input_row = gid - (weight_row * input_rows);
    float sum = 0.0;
    uint weight_base = weight_row * blocks_per_row * encoded_block_bytes;
    uint scale_base = weight_row * blocks_per_row;
    uint input_scale_base = input_row * blocks_per_row;
    uint input_quant_base = input_scale_base * block_values;
    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int int_sum = 0;
        uint encoded_base = weight_base + block_idx * encoded_block_bytes + 2;
        uint input_base = input_quant_base + block_idx * block_values;
        for (uint lane = 0; lane < block_values; ++lane) {
            int_sum += int(encoded_rows[encoded_base + lane]) * int(input_quants[input_base + lane]);
        }
        sum += float(int_sum) * weight_scales[scale_base + block_idx] * input_scales[input_scale_base + block_idx];
    }
    // Match inference.rs output_chunk_scratch layout: chunk_col * input_rows + input_row.
    output[gid] = sum;
}

kernel void q8_0_block_linear_row(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint q8_block_bytes = 36;
    float sum = 0.0;
    uint row_base = gid * blocks_per_row * q8_block_bytes;
    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int int_sum = 0;
        uint block_base = row_base + block_idx * q8_block_bytes;
        device const float* weight_scale = reinterpret_cast<device const float*>(weight_blocks + block_base);
        uint weight_quant_base = block_base + 4;
        uint input_base = block_idx * block_values;
        for (uint lane = 0; lane < block_values; ++lane) {
            int_sum += int(weight_blocks[weight_quant_base + lane]) * int(input_quants[input_base + lane]);
        }
        sum += float(int_sum) * (*weight_scale) * input_scales[block_idx];
    }
    output[gid] = sum;
}

// One SIMD-group (32 lanes) per output row: lanes stride over the row's blocks
// and a simd_sum reduces their partials. Parallelizes the contraction loop 32x
// versus the one-thread-per-row kernel above, which is bandwidth-bound on the
// long ffn_down rows. Dispatch with 32 threads/threadgroup, one group per row.
kernel void q8_0_block_linear_row_simd(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (row >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint q8_block_bytes = 36;
    uint row_base = row * blocks_per_row * q8_block_bytes;
    float partial = 0.0;
    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        uint block_base = row_base + block_idx * q8_block_bytes;
        device const float* weight_scale = reinterpret_cast<device const float*>(weight_blocks + block_base);
        uint weight_quant_base = block_base + 4;
        uint input_base = block_idx * block_values;
        int int_sum = 0;
        for (uint l = 0; l < block_values; ++l) {
            int_sum += int(weight_blocks[weight_quant_base + l]) * int(input_quants[input_base + l]);
        }
        partial += float(int_sum) * (*weight_scale) * input_scales[block_idx];
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        output[row] = total;
    }
}

// Multi-row variant: several SIMD groups per threadgroup, one output row per SIMD group.
// More rows' weight reads are in flight per threadgroup, which improves memory-latency
// hiding on the bandwidth-bound decode GEMV. Same per-row math as the single-row kernel.
kernel void q8_0_block_linear_row_simd_mr(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tg * sgs + sg;
    if (row >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint q8_block_bytes = 36;
    uint row_base = row * blocks_per_row * q8_block_bytes;
    float partial = 0.0;
    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        uint block_base = row_base + block_idx * q8_block_bytes;
        device const float* weight_scale = reinterpret_cast<device const float*>(weight_blocks + block_base);
        uint weight_quant_base = block_base + 4;
        uint input_base = block_idx * block_values;
        int int_sum = 0;
        for (uint l = 0; l < block_values; ++l) {
            int_sum += int(weight_blocks[weight_quant_base + l]) * int(input_quants[input_base + l]);
        }
        partial += float(int_sum) * (*weight_scale) * input_scales[block_idx];
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        output[row] = total;
    }
}

// MLX-layout Q8_0 GEMV: two simdgroups per threadgroup, FOUR output rows per simdgroup.
// Each lane caches its 32-value input block (quants + scale) in registers once per block
// iteration, then runs four independent dot-product chains against four weight-row streams:
// 4x the outstanding weight loads and four independent accumulators per lane for memory
// latency hiding (the layout MLX's qmv uses; results_per_simdgroup=4, no threadgroup
// memory, no pragmas). Same per-row arithmetic as q8_0_block_linear_row_simd.
// Requires rows % 4 == 0 (every decode projection satisfies this); callers fall back to
// the single-row kernel otherwise.
kernel void q8_0_block_linear_row_simd_qmv4(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint block_values = 32;
    constexpr uint q8_block_bytes = 36;
    uint row0 = (tg * sgs + sg) * 4;
    if (row0 >= rows) return;
    uint row_stride = blocks_per_row * q8_block_bytes;
    uint base0 = row0 * row_stride;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;
    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        char xq[32];
        uint input_base = block_idx * block_values;
        for (uint l = 0; l < block_values; ++l) {
            xq[l] = input_quants[input_base + l];
        }
        float x_scale = input_scales[block_idx];
        uint b0 = base0 + block_idx * q8_block_bytes;
        {
            device const float* wsc = reinterpret_cast<device const float*>(weight_blocks + b0);
            uint q = b0 + 4;
            int isum = 0;
            for (uint l = 0; l < block_values; ++l) {
                isum += int(weight_blocks[q + l]) * int(xq[l]);
            }
            acc0 += float(isum) * (*wsc) * x_scale;
        }
        {
            uint b = b0 + row_stride;
            device const float* wsc = reinterpret_cast<device const float*>(weight_blocks + b);
            uint q = b + 4;
            int isum = 0;
            for (uint l = 0; l < block_values; ++l) {
                isum += int(weight_blocks[q + l]) * int(xq[l]);
            }
            acc1 += float(isum) * (*wsc) * x_scale;
        }
        {
            uint b = b0 + 2u * row_stride;
            device const float* wsc = reinterpret_cast<device const float*>(weight_blocks + b);
            uint q = b + 4;
            int isum = 0;
            for (uint l = 0; l < block_values; ++l) {
                isum += int(weight_blocks[q + l]) * int(xq[l]);
            }
            acc2 += float(isum) * (*wsc) * x_scale;
        }
        {
            uint b = b0 + 3u * row_stride;
            device const float* wsc = reinterpret_cast<device const float*>(weight_blocks + b);
            uint q = b + 4;
            int isum = 0;
            for (uint l = 0; l < block_values; ++l) {
                isum += int(weight_blocks[q + l]) * int(xq[l]);
            }
            acc3 += float(isum) * (*wsc) * x_scale;
        }
    }
    float t0 = simd_sum(acc0);
    float t1 = simd_sum(acc1);
    float t2 = simd_sum(acc2);
    float t3 = simd_sum(acc3);
    if (lane == 0) {
        output[row0] = t0;
        output[row0 + 1u] = t1;
        output[row0 + 2u] = t2;
        output[row0 + 3u] = t3;
    }
}

// K-split cooperative GEMV: each threadgroup produces TWO output rows, with FOUR
// simdgroups partitioning the contraction dimension. Each thread owns an 8-quant slice
// of a block and strides 32 block-slots per threadgroup iteration, so one output row has
// 32 concurrent weight streams in flight (vs one sequential walk in the row-owned layouts
// above) — more memory-level parallelism per row at the cost of one threadgroup-memory
// reduction. Same int-dot arithmetic per block; partial order differs.
kernel void q8_0_block_linear_row_ksplit(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint q8_block_bytes = 36;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;

    const uint ix = lane / 4;        // 8 block slots per simdgroup
    const uint il = (lane % 4) * NQ; // this thread's 8-quant slice within the block

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        device const char* iq = input_quants + ib * 32 + il;
        const float in_scale = input_scales[ib];
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
            const float w_scale = *reinterpret_cast<device const float*>(wb);
            device const char* wq = wb + 4 + il;
            int isum = 0;
            for (uint i = 0; i < NQ; ++i) {
                isum += int(wq[i]) * int(iq[i]);
            }
            sumf[row] += float(isum) * w_scale * in_scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}

// K-split cooperative GEMV over an UNQUANTIZED f32 activation vector: the weights stay
// Q8_0 (the bandwidth that matters), but the input is read as float and the inner product
// runs on the float-FMA pipes (int8 weight -> float convert, fused multiply-add) with no
// input-quantize dispatch anywhere in the chain. Slightly more accurate than the
// quantized-activation path (no activation rounding), so it is parity-tested against an
// f32 reference rather than the int-dot reference.
kernel void q8_0_block_linear_row_ksplit_f32y(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint q8_block_bytes = 36;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;

    const uint ix = lane / 4;
    const uint il = (lane % 4) * NQ;

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        float yl[NQ];
        device const float* yb = y + ib * 32 + il;
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
            const float w_scale = *reinterpret_cast<device const float*>(wb);
            device const char* wq = wb + 4 + il;
            float sumq = 0.0f;
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(wq[i]) * yl[i];
            }
            sumf[row] += sumq * w_scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}

// Wire-format variant of the f32y K-split GEMV: weights are the raw GGUF Q8_0 wire layout
// (34-byte blocks: f16 scale + 32 int8 quants) instead of the decoded 36-byte f32-scale
// blocks — ~5.9% fewer weight bytes per token, which is the whole cost of a
// bandwidth-bound decode. Block offsets are multiples of 34, so the f16 scale is always
// 2-byte aligned.
kernel void q8_0_block_linear_row_ksplit_f32y_wire(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint q8_block_bytes = 34;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;

    const uint ix = lane / 4;
    const uint il = (lane % 4) * NQ;

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        float yl[NQ];
        device const float* yb = y + ib * 32 + il;
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
            const float w_scale = float(*reinterpret_cast<device const half*>(wb));
            device const char* wq = wb + 2 + il;
            float sumq = 0.0f;
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(wq[i]) * yl[i];
            }
            sumf[row] += sumq * w_scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}

// NSG=8 variant of the wire-format GEMV (256 threads/TG, 64 block slots in flight): weights are the raw GGUF Q8_0 wire layout
// (34-byte blocks: f16 scale + 32 int8 quants) instead of the decoded 36-byte f32-scale
// blocks — ~5.9% fewer weight bytes per token, which is the whole cost of a
// bandwidth-bound decode. Block offsets are multiples of 34, so the f16 scale is always
// 2-byte aligned.

// Q4_0 wire-format GEMV (f32 activation x inline-dequantized Q4_0 weights), the
// QAT-row counterpart of the Q8 wire kernel above. Q4_0 blocks are 18 bytes:
// an f16 scale + 16 nibble bytes packing 32 weights — byte j holds weight j in
// its low nibble and weight j+16 in its high nibble, both unsigned with a -8
// bias. So the 32 dequantized weights split into a low half (weights 0..16, low
// nibbles) and a high half (weights 16..32, high nibbles); each lane handles 4
// of the 16 nibble bytes (4 lanes x 4 = the full block), accumulating the low
// term against y[j] and the high term against y[j+16]. Same NSG=4/NR0=2 simd
// reduction as the Q8 kernel. Numerics: f32 activation x f32-dequantized weight
// (matches the CPU dequant reference to f32 rounding).
kernel void q4_0_block_linear_row_ksplit_f32y_wire(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;            // block slots in flight per simdgroup (matches Q8)
    constexpr uint NB = 4;            // nibble-bytes per lane (4 lanes * 4 = 16 bytes)
    constexpr uint q4_block_bytes = 18;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q4_block_bytes;

    const uint ix = lane / 4;
    const uint ilb = (lane % 4) * NB; // 0,4,8,12 byte offset within the 16 nibble bytes

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        float ylo[NB], yhi[NB];
        device const float* yb = y + ib * 32;
        for (uint i = 0; i < NB; ++i) {
            ylo[i] = yb[ilb + i];
            yhi[i] = yb[16 + ilb + i];
        }
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const char* wb = weight_blocks + rr * row_stride + ib * q4_block_bytes;
            const float w_scale = float(*reinterpret_cast<device const half*>(wb));
            device const uchar* wq = reinterpret_cast<device const uchar*>(wb + 2) + ilb;
            float sumq = 0.0f;
            for (uint i = 0; i < NB; ++i) {
                const uint b = uint(wq[i]);
                const float lo = float(int(b & 0x0F) - 8);
                const float hi = float(int(b >> 4) - 8);
                sumq += lo * ylo[i] + hi * yhi[i];
            }
            sumf[row] += sumq * w_scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}
// NVFP4 f32-activation GEMV (GABBRO M3). 64-value superblocks (36 bytes: d[4] UE4M3
// scales + qs[32] nibbles). Lane split: s = lane%4 owns one 16-value sub-block AND
// its matching UE4M3 scale d[s]; ix = lane/4 selects the superblock slot in flight.
// The 4 lanes sharing an ix cover a superblock's 4 sub-blocks, so simd_sum over the
// 32 lanes yields the full per-row dot — same reduction skeleton as the q4_0 kernel.
kernel void nvfp4_block_linear_row_ksplit_f32y_wire(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],   // 64-value NVFP4 superblocks per row
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;                 // superblock slots in flight per simdgroup
    constexpr uint nvfp4_block_bytes = 36;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * nvfp4_block_bytes;

    const uint s = lane % 4;               // sub-block index == UE4M3 scale index (0..3)
    const uint ix = lane / 4;              // which superblock slot in flight (0..7)

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        // this lane's 16 activations = sub-block s of superblock ib
        float ylo[8], yhi[8];
        device const float* yb = y + ib * 64 + s * 16;
        for (uint j = 0; j < 8; ++j) {
            ylo[j] = yb[j];       // element s*16 + j     (low nibble)
            yhi[j] = yb[8 + j];   // element s*16 + 8 + j (high nibble)
        }
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const uchar* wb = reinterpret_cast<device const uchar*>(
                weight_blocks + rr * row_stride + ib * nvfp4_block_bytes);
            const float scale = nvfp4_ue4m3_to_f32(wb[s]);   // d[s]
            device const uchar* wq = wb + 4 + s * 8;         // qs bytes for sub-block s
            float sumq = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const uint b = uint(wq[j]);
                const float lo = NVFP4_KVALUES[b & 0x0F];
                const float hi = NVFP4_KVALUES[b >> 4];
                sumq += lo * ylo[j] + hi * yhi[j];
            }
            sumf[row] += sumq * scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}
kernel void q8_0_block_linear_row_ksplit_f32y_wire_nsg8(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 8;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint q8_block_bytes = 34;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;

    const uint ix = lane / 4;
    const uint il = (lane % 4) * NQ;

    float sumf[NR0] = {0.0f, 0.0f};
    for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
        float yl[NQ];
        device const float* yb = y + ib * 32 + il;
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (uint row = 0; row < NR0; ++row) {
            const uint rr = r0 + row;
            if (rr >= rows) {
                break;
            }
            device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
            const float w_scale = float(*reinterpret_cast<device const half*>(wb));
            device const char* wq = wb + 2 + il;
            float sumq = 0.0f;
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(wq[i]) * yl[i];
            }
            sumf[row] += sumq * w_scale;
        }
    }
    for (uint row = 0; row < NR0; ++row) {
        if (sg == 0) {
            shmem[row * 32 + lane] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0; ++row) {
        if (lane == 0) {
            shmem[row * 32 + sg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
        const float tot = simd_sum(shmem[row * 32 + lane]);
        if (lane == 0 && sg == 0) {
            output[r0 + row] = tot;
        }
    }
}

// Wire-format K-split GEMM for prefill: identical layout to the GEMV above, but every
// weight block is dotted against ALL `n_rows_in` activation rows before moving on — the
// weights stream once per prefill (not once per token), so cost scales with the model,
// not the prompt. y is row-major [n_rows_in][k], out is [n_rows_in][rows].
kernel void q8_0_block_linear_ksplit_f32y_wire_gemm(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& n_rows_in [[buffer(6)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint MAX_T = 8; // activation rows processed per pass
    constexpr uint q8_block_bytes = 34;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;
    const uint ix = lane / 4;
    const uint il = (lane % 4) * NQ;

    for (uint t0 = 0; t0 < n_rows_in; t0 += MAX_T) {
        const uint tn = min(uint(MAX_T), n_rows_in - t0);
        float sumf[NR0][MAX_T];
        for (uint r = 0; r < NR0; ++r) {
            for (uint t = 0; t < MAX_T; ++t) {
                sumf[r][t] = 0.0f;
            }
        }
        for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
            float yl[MAX_T][NQ];
            for (uint t = 0; t < tn; ++t) {
                device const float* yb = y + (t0 + t) * blocks_per_row * 32 + ib * 32 + il;
                for (uint i = 0; i < NQ; ++i) {
                    yl[t][i] = yb[i];
                }
            }
            for (uint row = 0; row < NR0; ++row) {
                const uint rr = r0 + row;
                if (rr >= rows) {
                    break;
                }
                device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
                const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                device const char* wq = wb + 2 + il;
                float wv[NQ];
                for (uint i = 0; i < NQ; ++i) {
                    wv[i] = float(wq[i]) * w_scale;
                }
                for (uint t = 0; t < tn; ++t) {
                    float sumq = 0.0f;
                    for (uint i = 0; i < NQ; ++i) {
                        sumq += wv[i] * yl[t][i];
                    }
                    sumf[row][t] += sumq;
                }
            }
        }
        for (uint t = 0; t < tn; ++t) {
            for (uint row = 0; row < NR0; ++row) {
                if (sg == 0) {
                    shmem[row * 32 + lane] = 0.0f;
                }
                sumf[row][t] = simd_sum(sumf[row][t]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = 0; row < NR0; ++row) {
                if (lane == 0) {
                    shmem[row * 32 + sg] = sumf[row][t];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
                const float tot = simd_sum(shmem[row * 32 + lane]);
                if (lane == 0 && sg == 0) {
                    output[(t0 + t) * rows + r0 + row] = tot;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Byte-exact batched-column mirror of the single-token GEMV
// `q8_0_block_linear_row_ksplit_f32y_wire_nsg8` (above). It runs the SAME NSG=8
// reduction over `n_rows_in` INDEPENDENT activation columns (the speculative-verify
// rows, up to TREE_MAX_NODES=16 for the tree path — the outer `for t0 .. += MAX_T`
// loop tiles them MAX_T(8) columns per pass, so do NOT assume n_rows_in <= MAX_T),
// carrying a separate accumulator per column. For each weight block
// the per-block partial is computed as `sumq = Σ_i wq[i]*y[i]` and ONLY THEN scaled
// (`sumf += sumq * w_scale`) — w_scale is never folded onto the weight (that reorders
// the rounding and is what the prefill `wire_gemm` does), so column t equals the
// single-token GEMV bit-for-bit. Columns never interact: same block ownership
// (`ib = sg*NQ+ix`, step NSG*NQ=64), same dot order, same two-stage
// simd_sum -> shmem[row*32+sg] -> simd_sum finalize. y is row-major
// [token][blocks_per_row*32]; output is token-major [token][rows] so the next GEMM
// reads [token][dim]. A trailing per-column barrier guards the shared shmem against
// the next column's zero-out — it changes ordering only, not arithmetic.
kernel void q8_0_block_linear_ksplit_f32y_wire_nsg8_verify(
    device const float* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& n_rows_in [[buffer(6)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 8;
    constexpr uint NR0 = 2;
    constexpr uint NQ = 8;
    constexpr uint MAX_T = 8; // verify columns processed per pass
    constexpr uint q8_block_bytes = 34;
    const uint r0 = tg * NR0;
    const uint row_stride = blocks_per_row * q8_block_bytes;

    const uint ix = lane / 4;
    const uint il = (lane % 4) * NQ;

    for (uint t0 = 0; t0 < n_rows_in; t0 += MAX_T) {
        const uint tn = min(uint(MAX_T), n_rows_in - t0);
        float sumf[NR0][MAX_T];
        for (uint row = 0; row < NR0; ++row) {
            for (uint t = 0; t < MAX_T; ++t) {
                sumf[row][t] = 0.0f;
            }
        }
        for (uint ib = sg * NQ + ix; ib < blocks_per_row; ib += NSG * NQ) {
            float yl[MAX_T][NQ];
            for (uint t = 0; t < tn; ++t) {
                device const float* yb = y + (t0 + t) * blocks_per_row * 32 + ib * 32 + il;
                for (uint i = 0; i < NQ; ++i) {
                    yl[t][i] = yb[i];
                }
            }
            for (uint row = 0; row < NR0; ++row) {
                const uint rr = r0 + row;
                if (rr >= rows) {
                    break;
                }
                device const char* wb = weight_blocks + rr * row_stride + ib * q8_block_bytes;
                const float w_scale = float(*reinterpret_cast<device const half*>(wb));
                device const char* wq = wb + 2 + il;
                for (uint t = 0; t < tn; ++t) {
                    float sumq = 0.0f;
                    for (uint i = 0; i < NQ; ++i) {
                        sumq += float(wq[i]) * yl[t][i];
                    }
                    sumf[row][t] += sumq * w_scale;
                }
            }
        }
        for (uint t = 0; t < tn; ++t) {
            for (uint row = 0; row < NR0; ++row) {
                if (sg == 0) {
                    shmem[row * 32 + lane] = 0.0f;
                }
                sumf[row][t] = simd_sum(sumf[row][t]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = 0; row < NR0; ++row) {
                if (lane == 0) {
                    shmem[row * 32 + sg] = sumf[row][t];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = 0; row < NR0 && r0 + row < rows; ++row) {
                const float tot = simd_sum(shmem[row * 32 + lane]);
                if (lane == 0 && sg == 0) {
                    output[(t0 + t) * rows + r0 + row] = tot;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Simdgroup-matrix tiled GEMM over wire-format Q8_0 weights for prefill.
// out[t][r] = sum_k W[r][k] * Y[t][k].
//
// Long-prompt prefill GEMM is DEVICE-TRAFFIC bound, and the dominant term is the
// ACTIVATION tile: each row-tile re-reads the whole Y panel (weight streaming evicts
// it from L2 between threadgroups), so rows-per-tile is the first-order lever, with
// tokens-per-tile second (it divides weight re-streaming). Measured on the
// gate-projection shape: 64x64 tiles = 472MB of Y re-reads + 267MB of weights = ~3.3
// TFLOPS; this 128-row x 64-token tile halves the Y term. Both tiles stage in
// threadgroup memory swizzled into contiguous 8x8 blocks (A transposed to k-major
// during the dequant store, B token-major via one vectorized half2x4 store per
// thread), so every simdgroup_load is a dense 64-element block: no stride, no
// transpose (strided/transposed fragment loads measured ~2.6 TFLOPS). 256 threads = 8
// simdgroups in a 4-row x 2-token grid, each accumulating a 32x32 quadrant: 16 MMAs
// per 8 fragment loads. Activations arrive pre-converted to half (f32_to_f16 pass;
// same rounding point as staging f32 per-tile, so results are unchanged) and padded
// to a 64-token multiple so tail-tile loads stay in bounds; ragged tails store
// through per-simdgroup scratch reusing the A region. Accumulation order differs from
// the scalar/CPU path (tile MMA vs k-split dot), so this path is numerically
// equivalent but not byte-exact; it is gated by CAMELID_METAL_MM and requires
// rows % 128 == 0 (host checks).
//
// Threadgroup memory (12288 bytes): A 128x32 half (8192 B) | B 64x32 half (4096 B);
// the A region doubles as the 8 x 8x8 f32 tail-store scratch.
kernel void q8_0_block_wire_mm(
    device const half* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& n_rows_in [[buffer(6)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 64;  // weight rows per tile
    constexpr uint NR1 = 128; // tokens per tile (weights stream once per 128 tokens)
    constexpr uint NK = 32;   // k per step = one Q8_0 block
    constexpr uint q8_block_bytes = 34;
    // All position attributes must share one dimensionality; tg is uint2, so derive the
    // flat thread id from the (always-scalar) simdgroup/lane indices instead.
    const uint tid = sg * 32 + lane;

    threadgroup half* sa = shmem;        // A: 8 row-octets x 4 k-octets of 8x8 blocks
    threadgroup half* sb = shmem + 2048; // B: 16 token-octets x 4 k-octets of 8x8 blocks
    threadgroup float* scratch = reinterpret_cast<threadgroup float*>(shmem);
    const uint r0 = tg.x * NR0; // weight-row tile
    const uint t0 = tg.y * NR1; // token tile
    const uint row_stride = blocks_per_row * q8_block_bytes;
    const uint k_width = blocks_per_row * 32;

    const uint lr0 = tid / 4;      // weight row in tile (0..63)
    const uint il0 = (tid / 2) % 2; // which 16-value half of the Q8_0 block
    const uint lr1 = tid / 2;      // token in tile (0..127)

    device const char* x = weight_blocks + (r0 + lr0) * row_stride;
    device const half* yp = y + (t0 + lr1) * k_width;

    // This simdgroup's quadrant: 32 rows (sg % 2) x 32 tokens (sg / 2). A quadrant
    // entirely past n_rows_in only sees zero-padding — skip its loads and MMAs
    // (staging and barriers stay uniform across the threadgroup).
    const uint sg_row_oct = (sg % 2) * 4;
    const uint sg_tok_oct = (sg / 2) * 4;
    const bool sg_active = t0 + 8 * sg_tok_oct < n_rows_in;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (uint i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint ib = 0; ib < blocks_per_row; ++ib) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A: dequantize 8 values per thread (half of a 16-value half-block); store
        // TRANSPOSED (k-major inside each 8x8 block) so the A operand loads contiguously.
        {
            device const char* wb = x + ib * q8_block_bytes;
            const float w_scale = float(*reinterpret_cast<device const half*>(wb));
            const uint q8 = tid % 2; // which 8 of the 16-value half-block
            device const packed_char4* wq =
                reinterpret_cast<device const packed_char4*>(wb + 2 + il0 * 16 + q8 * 8);
            const uint sy = lr0 / 8;
            const uint lx = lr0 % 8;
            const uint sx = 2 * il0 + q8;
            for (uint i = 0; i < 8; ++i) {
                sa[64 * (8 * sx + sy) + 8 * i + lx] =
                    half(float(wq[i / 4][i % 4]) * w_scale);
            }
        }
        // B: one vectorized 8-half store per thread x 2 k-octets, token-major blocks.
        {
            const uint sy = lr1 / 8;
            const uint ly = lr1 % 8;
            for (uint s = 0; s < 2; ++s) {
                const uint sx = (tid % 2) * 2 + s;
                *reinterpret_cast<threadgroup half2x4*>(sb + 64 * (16 * sx + sy) + 8 * ly) =
                    *reinterpret_cast<device const half2x4*>(yp + ib * NK + 16 * (tid % 2) + 8 * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            threadgroup const half* lsma = sa + 64 * sg_row_oct;
            threadgroup const half* lsmb = sb + 64 * sg_tok_oct;
            for (uint ik = 0; ik < NK / 8; ++ik) {
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 16; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                }
                lsma += 64 * 8;
                lsmb += 64 * 16;
            }
        }
    }

    if (t0 + NR1 <= n_rows_in) {
        // Full tile: store fragments straight to device memory.
        device float* c = output + (r0 + 32 * (sg % 2)) + (t0 + 32 * (sg / 2)) * rows;
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], c + 8 * (i % 4) + 8 * rows * (i / 4), rows, 0, false);
        }
    } else {
        // Ragged token tail: stage each 8x8 fragment through per-simdgroup scratch
        // (reusing the consumed A region) and write guarded. Slow, but only the final
        // token tile takes this path.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], scratch + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint t_oct = t0 + 32 * (sg / 2) + 8 * (i / 4);
            const uint r_oct = r0 + 32 * (sg % 2) + 8 * (i % 4);
            for (uint e2 = lane; e2 < 64; e2 += 32) {
                const uint ft = e2 / 8;
                const uint fr = e2 % 8;
                if (t_oct + ft < n_rows_in) {
                    output[(t_oct + ft) * rows + r_oct + fr] = scratch[sg * 64 + ft * 8 + fr];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Half-output twin of q8_0_block_wire_mm for the all-half prefill
// activation stream (same accumulation; rounding at the store).
kernel void q8_0_block_wire_mm_f16o(
    device const half* y [[buffer(0)]],
    device const char* weight_blocks [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& n_rows_in [[buffer(6)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 64;  // weight rows per tile
    constexpr uint NR1 = 128; // tokens per tile (weights stream once per 128 tokens)
    constexpr uint NK = 32;   // k per step = one Q8_0 block
    constexpr uint q8_block_bytes = 34;
    // All position attributes must share one dimensionality; tg is uint2, so derive the
    // flat thread id from the (always-scalar) simdgroup/lane indices instead.
    const uint tid = sg * 32 + lane;

    threadgroup half* sa = shmem;        // A: 8 row-octets x 4 k-octets of 8x8 blocks
    threadgroup half* sb = shmem + 2048; // B: 16 token-octets x 4 k-octets of 8x8 blocks
    threadgroup float* scratch = reinterpret_cast<threadgroup float*>(shmem);
    const uint r0 = tg.x * NR0; // weight-row tile
    const uint t0 = tg.y * NR1; // token tile
    const uint row_stride = blocks_per_row * q8_block_bytes;
    const uint k_width = blocks_per_row * 32;

    const uint lr0 = tid / 4;      // weight row in tile (0..63)
    const uint il0 = (tid / 2) % 2; // which 16-value half of the Q8_0 block
    const uint lr1 = tid / 2;      // token in tile (0..127)

    device const char* x = weight_blocks + (r0 + lr0) * row_stride;
    device const half* yp = y + (t0 + lr1) * k_width;

    // This simdgroup's quadrant: 32 rows (sg % 2) x 32 tokens (sg / 2). A quadrant
    // entirely past n_rows_in only sees zero-padding — skip its loads and MMAs
    // (staging and barriers stay uniform across the threadgroup).
    const uint sg_row_oct = (sg % 2) * 4;
    const uint sg_tok_oct = (sg / 2) * 4;
    const bool sg_active = t0 + 8 * sg_tok_oct < n_rows_in;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (uint i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint ib = 0; ib < blocks_per_row; ++ib) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A: dequantize 8 values per thread (half of a 16-value half-block); store
        // TRANSPOSED (k-major inside each 8x8 block) so the A operand loads contiguously.
        {
            device const char* wb = x + ib * q8_block_bytes;
            const float w_scale = float(*reinterpret_cast<device const half*>(wb));
            const uint q8 = tid % 2; // which 8 of the 16-value half-block
            device const packed_char4* wq =
                reinterpret_cast<device const packed_char4*>(wb + 2 + il0 * 16 + q8 * 8);
            const uint sy = lr0 / 8;
            const uint lx = lr0 % 8;
            const uint sx = 2 * il0 + q8;
            for (uint i = 0; i < 8; ++i) {
                sa[64 * (8 * sx + sy) + 8 * i + lx] =
                    half(float(wq[i / 4][i % 4]) * w_scale);
            }
        }
        // B: one vectorized 8-half store per thread x 2 k-octets, token-major blocks.
        {
            const uint sy = lr1 / 8;
            const uint ly = lr1 % 8;
            for (uint s = 0; s < 2; ++s) {
                const uint sx = (tid % 2) * 2 + s;
                *reinterpret_cast<threadgroup half2x4*>(sb + 64 * (16 * sx + sy) + 8 * ly) =
                    *reinterpret_cast<device const half2x4*>(yp + ib * NK + 16 * (tid % 2) + 8 * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            threadgroup const half* lsma = sa + 64 * sg_row_oct;
            threadgroup const half* lsmb = sb + 64 * sg_tok_oct;
            for (uint ik = 0; ik < NK / 8; ++ik) {
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 16; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                }
                lsma += 64 * 8;
                lsmb += 64 * 16;
            }
        }
    }

    if (t0 + NR1 <= n_rows_in) {
        // Full tile, half output: per-lane element stores from the accumulators.
        device half* c = output + (r0 + 32 * (sg % 2)) + (t0 + 32 * (sg / 2)) * rows;
        const short qid = (short)(lane / 4);
        const short fm = (qid & 4) + (((short)lane / 2) % 4);
        const short fn = (qid & 2) * 2 + ((short)lane % 2) * 2;
        for (uint i = 0; i < 16; ++i) {
            device half* d2 =
                c + 8 * (i % 4) + 8 * rows * (i / 4) + (uint)fm * rows + (uint)fn;
            d2[0] = (half)mc[i].thread_elements()[0];
            d2[1] = (half)mc[i].thread_elements()[1];
        }
    } else {
        // Ragged token tail: stage each 8x8 fragment through per-simdgroup scratch
        // (reusing the consumed A region) and write guarded. Slow, but only the final
        // token tile takes this path.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], scratch + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint t_oct = t0 + 32 * (sg / 2) + 8 * (i / 4);
            const uint r_oct = r0 + 32 * (sg % 2) + 8 * (i % 4);
            for (uint e2 = lane; e2 < 64; e2 += 32) {
                const uint ft = e2 / 8;
                const uint fr = e2 % 8;
                if (t_oct + ft < n_rows_in) {
                    output[(t_oct + ft) * rows + r_oct + fr] =
                        half(scratch[sg * 64 + ft * 8 + fr]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

// Elementwise / norm building blocks for a GPU-resident forward pass. Each mirrors
// the CPU reference exactly (rms_norm: x / sqrt(mean(x^2) + eps) * w; silu_mul:
// (g / (1 + e^-g)) * u; residual: a + b) and is parity-checked in tests.
#[cfg(target_os = "macos")]
const ELEMENTWISE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup of 256 threads reduces the row's sum of squares, then scales.
kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    float local = 0.0;
    for (uint i = tid; i < width; i += tgsize) {
        float v = input[i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(width) + eps);
    for (uint i = tid; i < width; i += tgsize) {
        output[i] = input[i] * inv * weight[i];
    }
}

kernel void residual_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    output[gid] = a[gid] + b[gid];
}

// Per-head RMSNorm for Gemma's QK-norm (and weightless V-norm): one threadgroup
// per head independently normalizes that head's `head_dim` chunk, reusing the
// exact reduction + `1/sqrt(mean_sq + eps)` of rms_norm_f32. `use_weight == 0`
// skips the post-scale weight (weightless V-norm); otherwise weight is [head_dim]
// and shared across heads (q_norm / k_norm).
kernel void rms_norm_per_head_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& use_weight [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = head * head_dim;
    float local = 0.0;
    for (uint i = tid; i < head_dim; i += tgsize) {
        float v = input[base + i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tgsize) {
        float v = input[base + i] * inv;
        if (use_weight != 0) {
            v *= weight[i];
        }
        output[base + i] = v;
    }
}

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    output[gid] = (g / (1.0 + exp(-g))) * up[gid];
}

// GeGLU activation for Gemma's MLP: gelu_pytorch_tanh(gate) * up. Mirrors the CPU
// reference inference::gemma4::gelu_tanh exactly (same constants), so the GPU FFN
// stays parity-locked to llama.cpp.
kernel void gelu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float x = gate[gid];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    // tanh saturates to +/-1 well before |arg| = 15; clamp so the MSL implementation
    // (which can compute exp(2*arg) and overflow to inf/inf = NaN for large args)
    // matches the saturating CPU libm tanh on real-scale activations.
    float gelu = 0.5f * x * (1.0f + tanh(clamp(inner, -15.0f, 15.0f)));
    output[gid] = gelu * up[gid];
}

// Final-logit soft-cap: output = cap * tanh(input / cap). Mirrors the CPU
// reference inference::gemma4::soft_cap_in_place (cap = 30 for Gemma 4).
kernel void soft_cap_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& cap [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = input[gid];
    // Clamp like gelu_mul: tanh saturates well before |arg| = 15, and the MSL
    // implementation can overflow (inf/inf = NaN) for large arguments.
    output[gid] = cap * tanh(clamp(v / cap, -15.0f, 15.0f));
}

// Scale a vector by a constant: output = input * s. Used for Gemma's PLE per-layer
// output scale: h <- (h + ple_proj) * ple_output_scale.
kernel void scale_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& s [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    output[gid] = input[gid] * s;
}

// Forward RoPE rotation in-place across all heads. The per-pair cos/sin tables are
// computed on the CPU (which owns the freq/scaling math) and passed in, so this
// kernel only does the rotation. One thread per (head, pair).
// pairing: 0 = adjacent even/odd, 1 = split-half.
kernel void rope_rotate_f32(
    device float* data [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& half_rope [[buffer(5)]],
    constant uint& pairing [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = head_count * half_rope;
    if (gid >= total) return;
    uint head = gid / half_rope;
    uint pair = gid - head * half_rope;
    float c = cos_table[pair];
    float s = sin_table[pair];
    uint head_start = head * head_dim;
    uint dim0;
    uint dim1;
    if (pairing == 0u) {
        dim0 = head_start + pair * 2u;
        dim1 = dim0 + 1u;
    } else {
        dim0 = head_start + pair;
        dim1 = head_start + pair + half_rope;
    }
    float x0 = data[dim0];
    float x1 = data[dim1];
    data[dim0] = x0 * c - x1 * s;
    data[dim1] = x0 * s + x1 * c;
}

// Single-query (decode) causal attention over a contiguous KV cache, one thread per
// query head. Mirrors attention_context_for_head_into: score = dot(q, k_p) * scale,
// softmax over positions (max-shift), then out = sum_p prob_p * v_p. GQA maps query
// head -> kv head by integer group (group = n_heads / n_kv_heads). The K/V element at
// (kv_head, position, d) lives at kv_base_offset + kv_head*kv_head_stride +
// position*position_stride + d (strides in floats). The contiguous [kv_head][position]
// [head_dim] layout is kv_head_stride=position_count*head_dim, position_stride=head_dim,
// kv_base_offset=0; an interleaved per-layer slice of a [position][layer][kv_head]
// [head_dim] cache uses position_stride=layer_count*n_kv*head_dim, kv_head_stride=head_dim,
// kv_base_offset=layer*n_kv*head_dim. scores is scratch [n_heads*position_count].
// One threadgroup (a single 32-lane SIMD group) per query head. The lanes cooperate over
// positions for the score dot-products + online softmax reductions (simd_max / simd_sum),
// then split the output dimensions for the weighted-value sum so the per-token cost is
// parallelised across the SIMD group instead of running serially in one thread per head.
kernel void attention_decode_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (head >= n_heads) return;
    uint kv_head = head / group;
    uint q_base = head * head_dim;
    uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    uint score_base = head * position_count;

    // Phase 1: scaled q.k scores, lanes striding over positions; reduce the row max.
    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32) {
        uint k_base = kv_base + p * position_stride;
        float s = 0.0;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * keys[k_base + d];
        }
        s *= scale;
        scores[score_base + p] = s;
        local_max = max(local_max, s);
    }
    float max_score = simd_max(local_max);

    // Phase 2: exp(score - max) in place, reduce the denominator.
    float local_sum = 0.0;
    for (uint p = lane; p < position_count; p += 32) {
        float e = exp(scores[score_base + p] - max_score);
        scores[score_base + p] = e;
        local_sum += e;
    }
    float inv = 1.0 / simd_sum(local_sum);

    // Ensure every lane's score writes are visible before the weighted-value sum reads them.
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: out[d] = sum_p prob_p * v_p[d], lanes striding over the output dimensions so
    // each lane owns a disjoint set of dims and writes them directly (no cross-lane reduce).
    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0;
        for (uint p = 0; p < position_count; ++p) {
            acc += scores[score_base + p] * inv * values[kv_base + p * position_stride + d];
        }
        output[q_base + d] = acc;
    }
}

// f16-KV variant of attention_decode_f32: identical math, K/V read as half and converted
// per element (the scores/output stay f32).
kernel void attention_decode_kv16(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (head >= n_heads) return;
    uint kv_head = head / group;
    uint q_base = head * head_dim;
    uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    uint score_base = head * position_count;

    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32) {
        uint k_base = kv_base + p * position_stride;
        float s = 0.0;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * float(keys[k_base + d]);
        }
        s *= scale;
        scores[score_base + p] = s;
        local_max = max(local_max, s);
    }
    float max_score = simd_max(local_max);

    float local_sum = 0.0;
    for (uint p = lane; p < position_count; p += 32) {
        float e = exp(scores[score_base + p] - max_score);
        scores[score_base + p] = e;
        local_sum += e;
    }
    float inv = 1.0 / simd_sum(local_sum);

    threadgroup_barrier(mem_flags::mem_device);

    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0;
        for (uint p = 0; p < position_count; ++p) {
            acc += scores[score_base + p] * inv * float(values[kv_base + p * position_stride + d]);
        }
        output[q_base + d] = acc;
    }
}

// Tiled decode attention with online softmax. One threadgroup per query head, FOUR
// simdgroups; positions stride across simdgroups, lanes split head_dim, so K/V reads are
// coalesced across lanes (the v1 kernels read one full K row per lane). Each simdgroup
// keeps a flash-style running (max, denominator, weighted-V accumulator); the four partial
// states merge in threadgroup memory. No scores buffer, no device-memory barrier.
// Requires: head_dim % 32 == 0, head_dim <= 128.
kernel void attention_decode_v2_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint MAX_DPL = 4; // head_dim <= 128 -> at most 4 dims per lane
    if (head >= n_heads) return;
    const uint dpl = head_dim / 32;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    // Query slice for this lane's dims, scaled once.
    float q[MAX_DPL];
    for (uint i = 0; i < dpl; ++i) {
        q[i] = query[q_base + lane + i * 32] * scale;
    }

    // Flash-style running state for this simdgroup's positions.
    float m = -INFINITY;
    float l = 0.0;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint p = sg; p < position_count; p += NSG) {
        device const float* kr = keys + kv_base + p * position_stride;
        float s = 0.0;
        for (uint i = 0; i < dpl; ++i) {
            s += q[i] * kr[lane + i * 32];
        }
        s = simd_sum(s);
        float m_new = max(m, s);
        float w = exp(s - m_new);
        float corr = exp(m - m_new);
        device const float* vr = values + kv_base + p * position_stride;
        for (uint i = 0; i < dpl; ++i) {
            acc[i] = acc[i] * corr + w * vr[lane + i * 32];
        }
        l = l * corr + w;
        m = m_new;
    }

    // Merge the four simdgroup states: out = sum_i acc_i * exp(m_i - M) / sum_i l_i * exp(m_i - M).
    threadgroup float sh_m[NSG];
    threadgroup float sh_l[NSG];
    threadgroup float sh_acc[NSG * 128];
    if (lane == 0) {
        sh_m[sg] = m;
        sh_l[sg] = l;
    }
    for (uint i = 0; i < dpl; ++i) {
        sh_acc[sg * 128 + lane + i * 32] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m_tot = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float l_tot = 0.0;
        float w[NSG];
        for (uint i = 0; i < NSG; ++i) {
            w[i] = exp(sh_m[i] - m_tot);
            l_tot += sh_l[i] * w[i];
        }
        float inv = 1.0 / l_tot;
        for (uint i = 0; i < dpl; ++i) {
            uint d = lane + i * 32;
            float o = 0.0;
            for (uint g2 = 0; g2 < NSG; ++g2) {
                o += sh_acc[g2 * 128 + d] * w[g2];
            }
            output[q_base + d] = o * inv;
        }
    }
}

// Split-K flash decode attention (the depth-scaling path): grid (kv_head, split).
// Each threadgroup serves ALL query heads of one GQA group over one contiguous
// position chunk, staging K/V tiles once into threadgroup memory — K/V rows are
// read from device once per group instead of once per query head, and the
// (kv_heads x splits) grid keeps every GPU core busy at depth, unlike the
// one-threadgroup-per-head v2 kernel whose 24 threadgroups serialize long
// position walks. Partial flash states (acc, m, l) land in `partials` and a
// second kernel merges them exactly like v2 merges its per-simdgroup states.
// Requires group <= 4 (one simdgroup per query head); the host falls back to v2
// otherwise.
kernel void attention_decode_splitk_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* partials [[buffer(3)]], // [n_heads][n_splits][head_dim + 2]
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint PT = 8;      // staged positions per tile
    constexpr uint MAX_DPL = 4; // head_dim <= 128 -> at most 4 dims per lane
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint dpl = head_dim / 32;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);

    // One simdgroup per query head of this KV group.
    const uint qh = kvh * group + sg;
    const bool active = sg < group && qh < n_heads;
    float q[MAX_DPL];
    if (active) {
        for (uint i = 0; i < dpl; ++i) {
            q[i] = query[qh * head_dim + lane + i * 32] * scale;
        }
    }
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    threadgroup float k_s[PT * 128];
    threadgroup float v_s[PT * 128];
    const uint tid = sg * 32 + lane;
    for (uint pt = p0; pt < p1; pt += PT) {
        const uint count = min(PT, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < count * head_dim; idx += 128) {
            const uint p = idx / head_dim;
            const uint d = idx % head_dim;
            device const float* kr = keys + kv_base + (pt + p) * position_stride;
            device const float* vr = values + kv_base + (pt + p) * position_stride;
            k_s[p * head_dim + d] = kr[d];
            v_s[p * head_dim + d] = vr[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint j = 0; j < count; ++j) {
                float s = 0.0f;
                for (uint i = 0; i < dpl; ++i) {
                    s += q[i] * k_s[j * head_dim + lane + i * 32];
                }
                s = simd_sum(s);
                const float m_new = max(m, s);
                const float w = exp(s - m_new);
                const float corr = exp(m - m_new);
                for (uint i = 0; i < dpl; ++i) {
                    acc[i] = acc[i] * corr + w * v_s[j * head_dim + lane + i * 32];
                }
                l = l * corr + w;
                m = m_new;
            }
        }
    }
    if (active) {
        device float* dst = partials + ((ulong)qh * n_splits + split) * (head_dim + 2);
        for (uint i = 0; i < dpl; ++i) {
            dst[lane + i * 32] = acc[i];
        }
        if (lane == 0) {
            dst[head_dim] = m;
            dst[head_dim + 1] = l;
        }
    }
}

// Merge the per-split flash states: out = sum_s acc_s * exp(m_s - M) / sum_s l_s * exp(m_s - M).
kernel void attention_decode_splitk_merge_f32(
    device const float* partials [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& n_splits [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    device const float* base = partials + (ulong)head * n_splits * (head_dim + 2);
    float m_tot = -INFINITY;
    for (uint s2 = 0; s2 < n_splits; ++s2) {
        m_tot = max(m_tot, base[s2 * (head_dim + 2) + head_dim]);
    }
    float l_tot = 0.0f;
    for (uint s2 = 0; s2 < n_splits; ++s2) {
        const float mi = base[s2 * (head_dim + 2) + head_dim];
        if (mi != -INFINITY) {
            l_tot += base[s2 * (head_dim + 2) + head_dim + 1] * exp(mi - m_tot);
        }
    }
    const float inv = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    for (uint d = tid; d < head_dim; d += 128) {
        float o = 0.0f;
        for (uint s2 = 0; s2 < n_splits; ++s2) {
            const float mi = base[s2 * (head_dim + 2) + head_dim];
            if (mi != -INFINITY) {
                o += base[s2 * (head_dim + 2) + d] * exp(mi - m_tot);
            }
        }
        output[head * head_dim + d] = o * inv;
    }
}

// f16-KV twin of attention_decode_splitk_f32: reads the half K/V mirrors (half the
// device traffic of the f32 caches — the dominant cost at depth) and converts to f32
// in the staged tiles, so the flash math after staging is identical.
kernel void attention_decode_splitk_kv16(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]], // [n_heads][n_splits][head_dim + 2]
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // PT=16 half-staged: 8 KiB threadgroup memory total, so up to 4 threadgroups
    // stay resident per core (staging in one overlaps compute in another), with
    // half the barrier count of the PT=8 f32 twin per position walked.
    constexpr uint PT = 16;
    constexpr uint MAX_DPL = 4;
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint dpl = head_dim / 32;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);

    const uint qh = kvh * group + sg;
    const bool active = sg < group && qh < n_heads;
    float q[MAX_DPL];
    if (active) {
        for (uint i = 0; i < dpl; ++i) {
            q[i] = query[qh * head_dim + lane + i * 32] * scale;
        }
    }
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Half staging (half the threadgroup bytes of the f32 twin, no convert-on-store;
    // values convert to f32 in registers at use). Declared as half4 so the vector
    // stores below are alignment-guaranteed. NOTE: register-pipelined staging
    // (issuing tile t+1's loads before tile t's math) measured WORSE here
    // (21.4ms vs 14.3ms at 7.7k positions) — holding the in-flight tile across
    // the barrier costs registers/occupancy more than the overlap pays.
    threadgroup half4 k_s4[PT * 32];
    threadgroup half4 v_s4[PT * 32];
    threadgroup half* k_s = reinterpret_cast<threadgroup half*>(k_s4);
    threadgroup half* v_s = reinterpret_cast<threadgroup half*>(v_s4);
    const uint tid = sg * 32 + lane;
    for (uint pt = p0; pt < p1; pt += PT) {
        const uint count = min(PT, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Vectorized half4 loads: scalar 2-byte loads halve effective bus width and
        // erase the kv16 bandwidth win (head_dim is a multiple of 4 by the v2 gate).
        for (uint idx4 = tid; idx4 < (count * head_dim) / 4; idx4 += 128) {
            const uint e4 = idx4 * 4;
            const uint p = e4 / head_dim;
            const uint d = e4 % head_dim;
            const uint base = kv_base + (pt + p) * position_stride + d;
            k_s4[idx4] = *reinterpret_cast<device const half4*>(keys + base);
            v_s4[idx4] = *reinterpret_cast<device const half4*>(values + base);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            // Online softmax batched 4 positions per round: the 4 dots and simd_sums
            // are independent (their latencies overlap), and the running (m, l, acc)
            // state is rescaled once per round instead of once per position, cutting
            // the serial dependency chain ~4x. Tail slots pad with -INF scores
            // (w = 0), and their v_s reads are guarded: staged garbage could be NaN
            // and 0 * NaN would poison the accumulator.
            for (uint j0 = 0; j0 < count; j0 += 4) {
                float s4[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    const uint j = j0 + jj;
                    if (j < count) {
                        float s = 0.0f;
                        for (uint i = 0; i < dpl; ++i) {
                            s += q[i] * float(k_s[j * head_dim + lane + i * 32]);
                        }
                        s4[jj] = simd_sum(s);
                    } else {
                        s4[jj] = -INFINITY;
                    }
                }
                const float m4 = max(max(s4[0], s4[1]), max(s4[2], s4[3]));
                const float m_new = max(m, m4);
                const float corr = exp(m - m_new);
                float w4[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    w4[jj] = (s4[jj] == -INFINITY) ? 0.0f : exp(s4[jj] - m_new);
                }
                for (uint i = 0; i < dpl; ++i) {
                    float a = acc[i] * corr;
                    for (uint jj = 0; jj < 4; ++jj) {
                        if (j0 + jj < count) {
                            a += w4[jj] * float(v_s[(j0 + jj) * head_dim + lane + i * 32]);
                        }
                    }
                    acc[i] = a;
                }
                l = l * corr + w4[0] + w4[1] + w4[2] + w4[3];
                m = m_new;
            }
        }
    }
    if (active) {
        device float* dst = partials + ((ulong)qh * n_splits + split) * (head_dim + 2);
        for (uint i = 0; i < dpl; ++i) {
            dst[lane + i * 32] = acc[i];
        }
        if (lane == 0) {
            dst[head_dim] = m;
            dst[head_dim + 1] = l;
        }
    }
}

// f16-KV twin of attention_decode_v2_f32.
kernel void attention_decode_v2_kv16(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint MAX_DPL = 4;
    if (head >= n_heads) return;
    const uint dpl = head_dim / 32;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    float q[MAX_DPL];
    for (uint i = 0; i < dpl; ++i) {
        q[i] = query[q_base + lane + i * 32] * scale;
    }

    float m = -INFINITY;
    float l = 0.0;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint p = sg; p < position_count; p += NSG) {
        device const half* kr = keys + kv_base + p * position_stride;
        float s = 0.0;
        for (uint i = 0; i < dpl; ++i) {
            s += q[i] * float(kr[lane + i * 32]);
        }
        s = simd_sum(s);
        float m_new = max(m, s);
        float w = exp(s - m_new);
        float corr = exp(m - m_new);
        device const half* vr = values + kv_base + p * position_stride;
        for (uint i = 0; i < dpl; ++i) {
            acc[i] = acc[i] * corr + w * float(vr[lane + i * 32]);
        }
        l = l * corr + w;
        m = m_new;
    }

    threadgroup float sh_m[NSG];
    threadgroup float sh_l[NSG];
    threadgroup float sh_acc[NSG * 128];
    if (lane == 0) {
        sh_m[sg] = m;
        sh_l[sg] = l;
    }
    for (uint i = 0; i < dpl; ++i) {
        sh_acc[sg * 128 + lane + i * 32] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m_tot = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float l_tot = 0.0;
        float w[NSG];
        for (uint i = 0; i < NSG; ++i) {
            w[i] = exp(sh_m[i] - m_tot);
            l_tot += sh_l[i] * w[i];
        }
        float inv = 1.0 / l_tot;
        for (uint i = 0; i < dpl; ++i) {
            uint d = lane + i * 32;
            float o = 0.0;
            for (uint g2 = 0; g2 < NSG; ++g2) {
                o += sh_acc[g2 * 128 + d] * w[g2];
            }
            output[q_base + d] = o * inv;
        }
    }
}

// Direct-read split-K kv16 decode attention, specialized for head_dim == 128:
// each lane owns dims [lane*4, lane*4+4) so one half4 load per lane covers a
// position's K (or V) row coalesced, straight from device memory. No threadgroup
// staging, no barriers, minimal registers -> maximal resident threadgroups, so
// the GQA query heads' walks overlap and re-reads of a kv row land in the SLC
// instead of DRAM. Same batched online softmax as the staged kernel.
kernel void attention_decode_splitk_kv16_direct(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]], // [n_heads][n_splits][head_dim + 2]
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);

    const uint qh = kvh * group + sg;
    const bool active = sg < group && qh < n_heads;
    float4 q4 = float4(0.0f);
    if (active) {
        q4 = *reinterpret_cast<device const float4*>(query + qh * 128 + lane * 4) * scale;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = float4(0.0f);
    if (active) {
        for (uint j0 = p0; j0 < p1; j0 += 4) {
            float s4[4];
            for (uint jj = 0; jj < 4; ++jj) {
                const uint j = j0 + jj;
                if (j < p1) {
                    const half4 k4 = *reinterpret_cast<device const half4*>(
                        keys + kv_base + j * position_stride + lane * 4);
                    s4[jj] = simd_sum(dot(float4(k4), q4));
                } else {
                    s4[jj] = -INFINITY;
                }
            }
            const float m4 = max(max(s4[0], s4[1]), max(s4[2], s4[3]));
            const float m_new = max(m, m4);
            const float corr = exp(m - m_new);
            float w4[4];
            for (uint jj = 0; jj < 4; ++jj) {
                w4[jj] = (s4[jj] == -INFINITY) ? 0.0f : exp(s4[jj] - m_new);
            }
            acc *= corr;
            for (uint jj = 0; jj < 4; ++jj) {
                const uint j = j0 + jj;
                if (j < p1) {
                    const half4 v4 = *reinterpret_cast<device const half4*>(
                        values + kv_base + j * position_stride + lane * 4);
                    acc += w4[jj] * float4(v4);
                }
            }
            l = l * corr + w4[0] + w4[1] + w4[2] + w4[3];
            m = m_new;
        }
        device float* dst = partials + ((ulong)qh * n_splits + split) * (128 + 2);
        dst[lane * 4] = acc.x;
        dst[lane * 4 + 1] = acc.y;
        dst[lane * 4 + 2] = acc.z;
        dst[lane * 4 + 3] = acc.w;
        if (lane == 0) {
            dst[128] = m;
            dst[129] = l;
        }
    }
}

// Diagnostic twin of attention_decode_splitk_kv16 with the flash math stripped:
// stages the same tiles through the same barriers and consumes one staged value
// per tile (so the loads survive dead-code elimination), then writes a dummy
// partial. Measures the staging loop's standalone device-read ceiling.
kernel void attention_splitk_kv16_stageonly(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint PT = 16;
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);
    threadgroup half4 k_s4[PT * 32];
    threadgroup half4 v_s4[PT * 32];
    const uint tid = sg * 32 + lane;
    float acc = 0.0f;
    for (uint pt = p0; pt < p1; pt += PT) {
        const uint count = min(PT, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx4 = tid; idx4 < (count * head_dim) / 4; idx4 += 128) {
            const uint e4 = idx4 * 4;
            const uint p = e4 / head_dim;
            const uint d = e4 % head_dim;
            const uint base = kv_base + (pt + p) * position_stride + d;
            const uint s_idx = (p * head_dim + d) / 4;
            k_s4[s_idx] = *reinterpret_cast<device const half4*>(keys + base);
            v_s4[s_idx] = *reinterpret_cast<device const half4*>(values + base);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += float(k_s4[lane].x) + float(v_s4[lane].y);
    }
    if (tid == 0) {
        partials[(kvh * n_splits + split) % (n_heads * n_splits)] = acc + query[0] * scale;
    }
}

// Quantize an f32 activation row to Q8_0 (32-value blocks), one thread per block.
// Mirrors quantize_q8_0_block: scale = f16(max|v|/127) stored as f32, quant =
// round-ties-away(v / (max|v|/127)) clamped to [-127, 127]. Emits separate scale
// (f32) and quant (i8) arrays in the layout the Q8 block matmul consumes.
kernel void quantize_q8_0_f32(
    device const float* input [[buffer(0)]],
    device float* out_scales [[buffer(1)]],
    device char* out_quants [[buffer(2)]],
    constant uint& n_blocks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_blocks) return;
    uint base = gid * 32u;
    float max_abs = 0.0;
    for (uint i = 0; i < 32u; ++i) {
        max_abs = max(max_abs, fabs(input[base + i]));
    }
    float unrounded = max_abs / 127.0;
    float stored = float(half(unrounded));
    float inv = (unrounded == 0.0) ? 0.0 : 1.0 / unrounded;
    out_scales[gid] = stored;
    for (uint i = 0; i < 32u; ++i) {
        float scaled = input[base + i] * inv;
        int q = int(round(scaled));
        q = clamp(q, -127, 127);
        out_quants[base + i] = char(q);
    }
}

// Fused rms_norm + Q8_0 quantize: one threadgroup reduces the row's sum of squares, then
// each thread quantizes whole 32-value blocks of the normed row (value = input*inv*weight).
// Identical arithmetic to rms_norm_f32 followed by quantize_q8_0_f32, with no intermediate
// normed buffer round-trip and one dispatch instead of two.
kernel void rms_norm_quantize_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out_scales [[buffer(2)]],
    device char* out_quants [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    float local = 0.0;
    for (uint i = tid; i < width; i += tgsize) {
        float v = input[i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(width) + eps);
    uint n_blocks = width / 32u;
    for (uint b = tid; b < n_blocks; b += tgsize) {
        uint base = b * 32u;
        float v[32];
        float max_abs = 0.0;
        for (uint i = 0; i < 32u; ++i) {
            v[i] = input[base + i] * inv * weight[base + i];
            max_abs = max(max_abs, fabs(v[i]));
        }
        float unrounded = max_abs / 127.0;
        float stored = float(half(unrounded));
        float qinv = (unrounded == 0.0) ? 0.0 : 1.0 / unrounded;
        out_scales[b] = stored;
        for (uint i = 0; i < 32u; ++i) {
            int q = int(round(v[i] * qinv));
            q = clamp(q, -127, 127);
            out_quants[base + i] = char(q);
        }
    }
}

// Fused SiLU-mul + Q8_0 quantize, one thread per 32-value block: value =
// (g / (1 + e^-g)) * u. Identical arithmetic to silu_mul_f32 followed by
// quantize_q8_0_f32, with no intermediate activation buffer and one dispatch.
kernel void silu_mul_quantize_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out_scales [[buffer(2)]],
    device char* out_quants [[buffer(3)]],
    constant uint& n_blocks [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_blocks) return;
    uint base = gid * 32u;
    float v[32];
    float max_abs = 0.0;
    for (uint i = 0; i < 32u; ++i) {
        float g = gate[base + i];
        v[i] = (g / (1.0 + exp(-g))) * up[base + i];
        max_abs = max(max_abs, fabs(v[i]));
    }
    float unrounded = max_abs / 127.0;
    float stored = float(half(unrounded));
    float inv = (unrounded == 0.0) ? 0.0 : 1.0 / unrounded;
    out_scales[gid] = stored;
    for (uint i = 0; i < 32u; ++i) {
        int q = int(round(v[i] * inv));
        q = clamp(q, -127, 127);
        out_quants[base + i] = char(q);
    }
}

// Scatter the current token's K/V ([n_kv_heads * head_dim] contiguous) into the persistent
// cache buffers at slot `write_position` of a [kv_head][max_positions][head_dim] layout.
// Compute-encoder replacement for the previous blit, so a whole decode token can stay inside
// ONE compute command encoder.
kernel void kv_scatter_f32(
    device const float* src_k [[buffer(0)]],
    device const float* src_v [[buffer(1)]],
    device float* cache_k [[buffer(2)]],
    device float* cache_v [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& max_positions [[buffer(5)]],
    constant uint& write_position [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    device half* cache_k16 [[buffer(8)]],
    device half* cache_v16 [[buffer(9)]],
    constant uint& write_kv16 [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= total) return;
    uint h = i / head_dim;
    uint d = i % head_dim;
    uint dst = (h * max_positions + write_position) * head_dim + d;
    cache_k[dst] = src_k[i];
    cache_v[dst] = src_v[i];
    // Half mirrors feed the split-K kv16 decode attention; outside that mode the host
    // binds placeholders and the flag is 0 (see kv_scatter_batch_f32 for the OOB story).
    if (write_kv16 != 0) {
        cache_k16[dst] = half(src_k[i]);
        cache_v16[dst] = half(src_v[i]);
    }
}

// f16-KV variant: the cache stores half-precision K/V (half the bytes per token read back
// during attention, the dominant growing cost at long context).
kernel void kv_scatter_kv16(
    device const float* src_k [[buffer(0)]],
    device const float* src_v [[buffer(1)]],
    device half* cache_k [[buffer(2)]],
    device half* cache_v [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& max_positions [[buffer(5)]],
    constant uint& write_position [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= total) return;
    uint h = i / head_dim;
    uint d = i % head_dim;
    uint dst = (h * max_positions + write_position) * head_dim + d;
    cache_k[dst] = half(src_k[i]);
    cache_v[dst] = half(src_v[i]);
}

// ---- Batched (multi-token) prefill twins -------------------------------------------------
// One grid row per prompt token so a whole prompt's worth of elementwise work lands in a
// single dispatch instead of one dispatch per token. Each (token, unit) performs float ops
// in exactly the order of the per-token kernel it replaces, so outputs stay byte-exact with
// the per-position dispatch loop.

// f32 -> f16 copy (prefill GEMM inputs: the tiled-MM kernel reads activations as half
// directly from device memory; rounding here matches the f32->half tile staging it
// replaces, so results are unchanged).
kernel void f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}

// rms_norm_f32 over n_tokens contiguous rows: threadgroup = row.
kernel void rms_norm_batch_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    device const float* in_row = input + row * width;
    device float* out_row = output + row * width;
    float local = 0.0;
    for (uint i = tid; i < width; i += tgsize) {
        float v = in_row[i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(width) + eps);
    for (uint i = tid; i < width; i += tgsize) {
        out_row[i] = in_row[i] * inv * weight[i];
    }
}

// rms_norm_batch_f32 with a half output — feeds the simdgroup-MM prefill GEMM
// directly (same rounding point as the separate f32_to_f16 pass it replaces).
kernel void rms_norm_batch_f16o(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    device const float* in_row = input + row * width;
    device half* out_row = output + row * width;
    float local = 0.0;
    for (uint i = tid; i < width; i += tgsize) {
        float v = in_row[i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(width) + eps);
    for (uint i = tid; i < width; i += tgsize) {
        out_row[i] = half(in_row[i] * inv * weight[i]);
    }
}

// silu_mul_f32 with a half output — same rounding as silu then f32_to_f16.
kernel void silu_mul_f16o(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    output[gid] = half((g / (1.0 + exp(-g))) * up[gid]);
}

// All-half elementwise twins for the attention-matmul prefill stream.
kernel void rms_norm_batch_h(
    device const half* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    device const half* in_row = input + row * width;
    device half* out_row = output + row * width;
    float local = 0.0;
    for (uint i = tid; i < width; i += tgsize) {
        float v = float(in_row[i]);
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sqrt(partial[0] / float(width) + eps);
    for (uint i = tid; i < width; i += tgsize) {
        out_row[i] = half(float(in_row[i]) * inv * weight[i]);
    }
}

kernel void residual_add_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    output[gid] = half(float(a[gid]) + float(b[gid]));
}

kernel void silu_mul_h2(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = float(gate[gid]);
    output[gid] = half((g / (1.0 + exp(-g))) * float(up[gid]));
}

// rope_rotate_f32 over n_tokens rows: gid.y = token. The data row stride is
// head_count*head_dim (the packed Q or K row) and the cos/sin tables are flattened
// per-token (half_rope floats each).
kernel void rope_rotate_batch_f32(
    device float* data [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& half_rope [[buffer(5)]],
    constant uint& pairing [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint total = head_count * half_rope;
    if (gid.x >= total) return;
    uint t = gid.y;
    device float* row = data + t * head_count * head_dim;
    device const float* ct = cos_table + t * half_rope;
    device const float* st = sin_table + t * half_rope;
    uint head = gid.x / half_rope;
    uint pair = gid.x - head * half_rope;
    float c = ct[pair];
    float s = st[pair];
    uint head_start = head * head_dim;
    uint dim0;
    uint dim1;
    if (pairing == 0u) {
        dim0 = head_start + pair * 2u;
        dim1 = dim0 + 1u;
    } else {
        dim0 = head_start + pair;
        dim1 = head_start + pair + half_rope;
    }
    float x0 = row[dim0];
    float x1 = row[dim1];
    row[dim0] = x0 * c - x1 * s;
    row[dim1] = x0 * s + x1 * c;
}

// kv_scatter_f32 over n_tokens rows: gid.y = token, written at base_position + token.
// Also writes half copies (same layout) for the attention-as-matmul prefill path,
// whose batched GEMMs consume K and V as half operands.
kernel void kv_scatter_batch_f32(
    device const float* src_k [[buffer(0)]],
    device const float* src_v [[buffer(1)]],
    device float* cache_k [[buffer(2)]],
    device float* cache_v [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& max_positions [[buffer(5)]],
    constant uint& base_position [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    device half* cache_k16 [[buffer(8)]],
    device half* cache_v16 [[buffer(9)]],
    constant uint& write_kv16 [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= total) return;
    uint t = gid.y;
    uint i = gid.x;
    uint h = i / head_dim;
    uint d = i % head_dim;
    uint src = t * total + i;
    uint dst = (h * max_positions + base_position + t) * head_dim + d;
    cache_k[dst] = src_k[src];
    cache_v[dst] = src_v[src];
    // The half mirrors only exist on the attention-as-matmul path; outside it the host
    // binds 2-byte placeholders, and unguarded writes here sprayed out-of-bounds GPU
    // stores that corrupted neighboring buffers (non-finite logits on every fallback
    // prefill path).
    if (write_kv16 != 0) {
        cache_k16[dst] = half(src_k[src]);
        cache_v16[dst] = half(src_v[src]);
    }
}


// Fused per-layer RoPE + KV scatter + Q half-convert for the attention-as-matmul
// prefill path (requires full rotary coverage: half_rope * 2 == head_dim). One
// dispatch replaces rope(Q), rope(K), kv-scatter, and the Q f32->f16 convert: Q pairs
// rotate straight into the half Q panel the score matmul consumes (the f32 Q buffer
// is not written back), K pairs rotate straight into the f32 and f16 caches, and V
// elements scatter into both caches. Grid x lanes: [0, nq_pairs) Q pairs,
// [nq_pairs, nq_pairs + nk_pairs) K pairs, then kv_dim V elements; grid y = token.
kernel void rope_scatter_qh_batch(
    device const float* q_in [[buffer(0)]],
    device const float* k_in [[buffer(1)]],
    device const float* v_in [[buffer(2)]],
    device half* q_h [[buffer(3)]],
    device float* cache_k [[buffer(4)]],
    device float* cache_v [[buffer(5)]],
    device half* cache_k16 [[buffer(6)]],
    device half* cache_v16 [[buffer(7)]],
    device const float* cos_table [[buffer(8)]],
    device const float* sin_table [[buffer(9)]],
    constant uint& n_heads [[buffer(10)]],
    constant uint& n_kv_heads [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& half_rope [[buffer(13)]],
    constant uint& pairing [[buffer(14)]],
    constant uint& max_positions [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t = gid.y;
    const uint q_dim = n_heads * head_dim;
    const uint kv_dim = n_kv_heads * head_dim;
    const uint nq_pairs = n_heads * half_rope;
    const uint nk_pairs = n_kv_heads * half_rope;
    device const float* ct = cos_table + t * half_rope;
    device const float* st = sin_table + t * half_rope;
    uint x = gid.x;
    if (x < nq_pairs) {
        const uint head = x / half_rope;
        const uint pair = x - head * half_rope;
        const float c = ct[pair];
        const float s = st[pair];
        const uint head_start = head * head_dim;
        uint dim0;
        uint dim1;
        if (pairing == 0u) {
            dim0 = head_start + pair * 2u;
            dim1 = dim0 + 1u;
        } else {
            dim0 = head_start + pair;
            dim1 = head_start + pair + half_rope;
        }
        const float x0 = q_in[t * q_dim + dim0];
        const float x1 = q_in[t * q_dim + dim1];
        q_h[t * q_dim + dim0] = half(x0 * c - x1 * s);
        q_h[t * q_dim + dim1] = half(x0 * s + x1 * c);
        return;
    }
    x -= nq_pairs;
    if (x < nk_pairs) {
        const uint head = x / half_rope;
        const uint pair = x - head * half_rope;
        const float c = ct[pair];
        const float s = st[pair];
        const uint head_start = head * head_dim;
        uint dim0;
        uint dim1;
        if (pairing == 0u) {
            dim0 = head_start + pair * 2u;
            dim1 = dim0 + 1u;
        } else {
            dim0 = head_start + pair;
            dim1 = head_start + pair + half_rope;
        }
        const float x0 = k_in[t * kv_dim + dim0];
        const float x1 = k_in[t * kv_dim + dim1];
        const float r0 = x0 * c - x1 * s;
        const float r1 = x0 * s + x1 * c;
        const uint h0 = dim0 / head_dim;
        const uint d0 = dim0 % head_dim;
        const uint h1 = dim1 / head_dim;
        const uint d1 = dim1 % head_dim;
        const uint dst0 = (h0 * max_positions + t) * head_dim + d0;
        const uint dst1 = (h1 * max_positions + t) * head_dim + d1;
        cache_k[dst0] = r0;
        cache_k[dst1] = r1;
        cache_k16[dst0] = half(r0);
        cache_k16[dst1] = half(r1);
        return;
    }
    x -= nk_pairs;
    if (x < kv_dim) {
        const uint h = x / head_dim;
        const uint d = x % head_dim;
        const float v = v_in[t * kv_dim + x];
        const uint dst = (h * max_positions + t) * head_dim + d;
        cache_v[dst] = v;
        cache_v16[dst] = half(v);
    }
}

// Half-input twin of rope_scatter_qh_batch for the all-half stream.
kernel void rope_scatter_qh_batch_h(
    device const half* q_in [[buffer(0)]],
    device const half* k_in [[buffer(1)]],
    device const half* v_in [[buffer(2)]],
    device half* q_h [[buffer(3)]],
    device float* cache_k [[buffer(4)]],
    device float* cache_v [[buffer(5)]],
    device half* cache_k16 [[buffer(6)]],
    device half* cache_v16 [[buffer(7)]],
    device const float* cos_table [[buffer(8)]],
    device const float* sin_table [[buffer(9)]],
    constant uint& n_heads [[buffer(10)]],
    constant uint& n_kv_heads [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& half_rope [[buffer(13)]],
    constant uint& pairing [[buffer(14)]],
    constant uint& max_positions [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t = gid.y;
    const uint q_dim = n_heads * head_dim;
    const uint kv_dim = n_kv_heads * head_dim;
    const uint nq_pairs = n_heads * half_rope;
    const uint nk_pairs = n_kv_heads * half_rope;
    device const float* ct = cos_table + t * half_rope;
    device const float* st = sin_table + t * half_rope;
    uint x = gid.x;
    if (x < nq_pairs) {
        const uint head = x / half_rope;
        const uint pair = x - head * half_rope;
        const float c = ct[pair];
        const float s = st[pair];
        const uint head_start = head * head_dim;
        uint dim0;
        uint dim1;
        if (pairing == 0u) {
            dim0 = head_start + pair * 2u;
            dim1 = dim0 + 1u;
        } else {
            dim0 = head_start + pair;
            dim1 = head_start + pair + half_rope;
        }
        const float x0 = float(q_in[t * q_dim + dim0]);
        const float x1 = float(q_in[t * q_dim + dim1]);
        q_h[t * q_dim + dim0] = half(x0 * c - x1 * s);
        q_h[t * q_dim + dim1] = half(x0 * s + x1 * c);
        return;
    }
    x -= nq_pairs;
    if (x < nk_pairs) {
        const uint head = x / half_rope;
        const uint pair = x - head * half_rope;
        const float c = ct[pair];
        const float s = st[pair];
        const uint head_start = head * head_dim;
        uint dim0;
        uint dim1;
        if (pairing == 0u) {
            dim0 = head_start + pair * 2u;
            dim1 = dim0 + 1u;
        } else {
            dim0 = head_start + pair;
            dim1 = head_start + pair + half_rope;
        }
        const float x0 = float(k_in[t * kv_dim + dim0]);
        const float x1 = float(k_in[t * kv_dim + dim1]);
        const float r0 = x0 * c - x1 * s;
        const float r1 = x0 * s + x1 * c;
        const uint h0 = dim0 / head_dim;
        const uint d0 = dim0 % head_dim;
        const uint h1 = dim1 / head_dim;
        const uint d1 = dim1 % head_dim;
        const uint dst0 = (h0 * max_positions + t) * head_dim + d0;
        const uint dst1 = (h1 * max_positions + t) * head_dim + d1;
        cache_k[dst0] = r0;
        cache_k[dst1] = r1;
        cache_k16[dst0] = half(r0);
        cache_k16[dst1] = half(r1);
        return;
    }
    x -= nk_pairs;
    if (x < kv_dim) {
        const uint h = x / head_dim;
        const uint d = x % head_dim;
        const float v = float(v_in[t * kv_dim + x]);
        const uint dst = (h * max_positions + t) * head_dim + d;
        cache_v[dst] = v;
        cache_v16[dst] = half(v);
    }
}

// Transpose one layer's half V cache slice per KV head ([position][head_dim] ->
// [head_dim][position]) so the PV matmul's A-operand staging reads contiguously —
// strided V^T staging measured at roughly half the staged-GEMM rate. ~1MB per layer.
kernel void transpose_v16(
    device const half* v16 [[buffer(0)]],   // [kv_head][max_positions][head_dim]
    device half* vt [[buffer(1)]],          // [kv_head][head_dim][n_pad]
    constant uint& head_dim [[buffer(2)]],
    constant uint& max_positions [[buffer(3)]],
    constant uint& n_pad [[buffer(4)]],
    constant uint& n_tokens [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint p = gid.x; // position
    const uint d = gid.y; // head_dim element
    const uint h = gid.z; // kv head
    if (p >= n_pad || d >= head_dim) return;
    const half v = (p < n_tokens) ? v16[(h * max_positions + p) * head_dim + d] : half(0.0f);
    vt[((ulong)h * head_dim + d) * n_pad + p] = v;
}

// Batched half-precision GEMM for prefill attention-as-matmul:
//   C[z][t][r] = sum_k A[z][r][k] * B[z][t][k]
// with the same staged swizzled-8x8-tile structure as the Q8 prefill GEMM (A staged
// TRANSPOSED k-major in-block, B token-major, 64x64 tiles, 4 simdgroups of 32x32
// quadrants). A supports an element stride so V can be consumed as V^T without a
// repack (a_elem_stride = head_dim picks V columns), and A's batch index is z /
// group_a so GQA query heads share their KV head's K/V. causal_mode 1 (the S = Q K^T
// pass) skips tiles whose positions all exceed the tile's last query; causal_mode 2
// (the O = P V pass) clamps k to the tile's last query + 1 (P beyond is zero).
// Ragged column tails stage through the A region and store guarded.
kernel void half_mm_batched(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& kdim [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& a_batch_stride [[buffer(6)]],
    constant uint& b_batch_stride [[buffer(7)]],
    constant uint& c_batch_stride [[buffer(8)]],
    constant uint& a_row_stride [[buffer(9)]],
    constant uint& b_row_stride [[buffer(10)]],
    constant uint& c_row_stride [[buffer(11)]],
    constant uint& a_elem_stride [[buffer(12)]],
    constant uint& group_a [[buffer(13)]],
    constant uint& causal_mode [[buffer(14)]],
    constant uint& q_offset [[buffer(15)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 64; // A rows per tile
    constexpr uint NR1 = 64; // B rows (tokens/queries) per tile
    constexpr uint NK = 32;
    const uint tid = sg * 32 + lane;
    threadgroup half* sa = shmem;        // 64 x 32, swizzled, k-major in-block
    threadgroup half* sb = shmem + 2048; // 64 x 32, swizzled, token-major
    const uint r0 = tg.x * NR0;
    const uint t0 = tg.y * NR1;
    // q_offset maps this dispatch's query tile to absolute positions when the host
    // processes queries in blocks (S/P panels tiled by query block at long context).
    if (causal_mode == 1 && r0 > t0 + q_offset + NR1 - 1) {
        return; // S tile entirely above the causal diagonal: never read
    }
    const uint k_end = (causal_mode == 2) ? min(kdim, t0 + q_offset + NR1) : kdim;
    device const half* ab = a + (tg.z / group_a) * a_batch_stride;
    device const half* bb = b + tg.z * b_batch_stride;
    device float* cb = c + tg.z * c_batch_stride;

    const uint lr0 = tid / 2;
    const uint k0 = (tid % 2) * 16;
    const uint lr1 = tid / 2;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (uint i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_row_oct = (sg % 2) * 4;
    const uint sg_tok_oct = (sg / 2) * 4;
    // Skip quadrants that are entirely padding (tokens past cols) or, in the causal
    // score pass, entirely above the diagonal. Staging and barriers stay uniform.
    const uint quad_t0 = t0 + 8 * sg_tok_oct;
    const uint quad_r0 = r0 + 32 * (sg % 2);
    const bool sg_active = quad_t0 < cols
        && quad_r0 < rows
        && !(causal_mode == 1 && quad_r0 > quad_t0 + q_offset + 31);

    for (uint kk0 = 0; kk0 < k_end; kk0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A: 64 rows x 32 k staged TRANSPOSED (k-major in-block); zero-pad past kdim.
        // ROW TAIL: `rows` need not be a multiple of NR0, so the last row tile is
        // partial. A row past `rows` does not exist in A -- the attention PV pass hands
        // this kernel rows = head_dim over a slab allocated to exactly head_dim rows --
        // so reading it walks into the NEXT head's slab, or off the end of the buffer
        // for the last head. Zero-pad it, exactly as the k tail is already zero-padded.
        {
            const bool a_row_valid = (r0 + lr0) < rows;
            device const half* arow = a_row_valid
                ? ab + (r0 + lr0) * a_row_stride + (kk0 + k0) * a_elem_stride
                : ab;
            const uint sy = lr0 / 8;
            const uint lx = lr0 % 8;
            for (uint i = 0; i < 16; ++i) {
                const uint kg = k0 + i;
                sa[64 * (8 * (kg / 8) + sy) + 8 * (kg % 8) + lx] =
                    (a_row_valid && kk0 + kg < kdim) ? arow[i * a_elem_stride] : half(0.0f);
            }
        }
        // B: 64 rows x 32 k, token-major vector stores (B is padded past cols).
        {
            device const half* brow = bb + (t0 + lr1) * b_row_stride + kk0 + k0;
            const uint sy = lr1 / 8;
            const uint ly = lr1 % 8;
            for (uint s = 0; s < 2; ++s) {
                *reinterpret_cast<threadgroup half2x4*>(
                    sb + 64 * (8 * (k0 / 8 + s) + sy) + 8 * ly) =
                    *reinterpret_cast<device const half2x4*>(brow + 8 * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            threadgroup const half* lsma = sa + 64 * sg_row_oct;
            threadgroup const half* lsmb = sb + 64 * sg_tok_oct;
            for (uint ik = 0; ik < NK / 8; ++ik) {
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 16; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                }
                lsma += 64 * 8;
                lsmb += 64 * 8;
            }
        }
    }

    if (t0 + NR1 <= cols && r0 + NR0 <= rows) {
        device float* cq = cb + (r0 + 32 * (sg % 2)) + (t0 + 32 * (sg / 2)) * c_row_stride;
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], cq + 8 * (i % 4) + 8 * c_row_stride * (i / 4),
                            c_row_stride, 0, false);
        }
    } else {
        // Ragged token tail: stage each fragment through the A region, write guarded.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* scratch = reinterpret_cast<threadgroup float*>(shmem);
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], scratch + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint t_oct = t0 + 32 * (sg / 2) + 8 * (i / 4);
            const uint r_oct = r0 + 32 * (sg % 2) + 8 * (i % 4);
            for (uint e2 = lane; e2 < 64; e2 += 32) {
                const uint ft = e2 / 8;
                const uint fr = e2 % 8;
                if (t_oct + ft < cols && r_oct + fr < rows) {
                    cb[(t_oct + ft) * c_row_stride + r_oct + fr] =
                        scratch[sg * 64 + ft * 8 + fr];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void half_mm_batched_f16o(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint& kdim [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& a_batch_stride [[buffer(6)]],
    constant uint& b_batch_stride [[buffer(7)]],
    constant uint& c_batch_stride [[buffer(8)]],
    constant uint& a_row_stride [[buffer(9)]],
    constant uint& b_row_stride [[buffer(10)]],
    constant uint& c_row_stride [[buffer(11)]],
    constant uint& a_elem_stride [[buffer(12)]],
    constant uint& group_a [[buffer(13)]],
    constant uint& causal_mode [[buffer(14)]],
    constant uint& q_offset [[buffer(15)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 64; // A rows per tile
    constexpr uint NR1 = 64; // B rows (tokens/queries) per tile
    constexpr uint NK = 32;
    const uint tid = sg * 32 + lane;
    threadgroup half* sa = shmem;        // 64 x 32, swizzled, k-major in-block
    threadgroup half* sb = shmem + 2048; // 64 x 32, swizzled, token-major
    const uint r0 = tg.x * NR0;
    const uint t0 = tg.y * NR1;
    // q_offset maps this dispatch's query tile to absolute positions when the host
    // processes queries in blocks (S/P panels tiled by query block at long context).
    if (causal_mode == 1 && r0 > t0 + q_offset + NR1 - 1) {
        return; // S tile entirely above the causal diagonal: never read
    }
    const uint k_end = (causal_mode == 2) ? min(kdim, t0 + q_offset + NR1) : kdim;
    device const half* ab = a + (tg.z / group_a) * a_batch_stride;
    device const half* bb = b + tg.z * b_batch_stride;
    device half* cb = c + tg.z * c_batch_stride;

    const uint lr0 = tid / 2;
    const uint k0 = (tid % 2) * 16;
    const uint lr1 = tid / 2;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (uint i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_row_oct = (sg % 2) * 4;
    const uint sg_tok_oct = (sg / 2) * 4;
    // Skip quadrants that are entirely padding (tokens past cols) or, in the causal
    // score pass, entirely above the diagonal. Staging and barriers stay uniform.
    const uint quad_t0 = t0 + 8 * sg_tok_oct;
    const uint quad_r0 = r0 + 32 * (sg % 2);
    const bool sg_active = quad_t0 < cols
        && quad_r0 < rows
        && !(causal_mode == 1 && quad_r0 > quad_t0 + q_offset + 31);

    for (uint kk0 = 0; kk0 < k_end; kk0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A: 64 rows x 32 k staged TRANSPOSED (k-major in-block); zero-pad past kdim.
        // ROW TAIL: `rows` need not be a multiple of NR0, so the last row tile is
        // partial. A row past `rows` does not exist in A -- the attention PV pass hands
        // this kernel rows = head_dim over a slab allocated to exactly head_dim rows --
        // so reading it walks into the NEXT head's slab, or off the end of the buffer
        // for the last head. Zero-pad it, exactly as the k tail is already zero-padded.
        {
            const bool a_row_valid = (r0 + lr0) < rows;
            device const half* arow = a_row_valid
                ? ab + (r0 + lr0) * a_row_stride + (kk0 + k0) * a_elem_stride
                : ab;
            const uint sy = lr0 / 8;
            const uint lx = lr0 % 8;
            for (uint i = 0; i < 16; ++i) {
                const uint kg = k0 + i;
                sa[64 * (8 * (kg / 8) + sy) + 8 * (kg % 8) + lx] =
                    (a_row_valid && kk0 + kg < kdim) ? arow[i * a_elem_stride] : half(0.0f);
            }
        }
        // B: 64 rows x 32 k, token-major vector stores (B is padded past cols).
        {
            device const half* brow = bb + (t0 + lr1) * b_row_stride + kk0 + k0;
            const uint sy = lr1 / 8;
            const uint ly = lr1 % 8;
            for (uint s = 0; s < 2; ++s) {
                *reinterpret_cast<threadgroup half2x4*>(
                    sb + 64 * (8 * (k0 / 8 + s) + sy) + 8 * ly) =
                    *reinterpret_cast<device const half2x4*>(brow + 8 * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            threadgroup const half* lsma = sa + 64 * sg_row_oct;
            threadgroup const half* lsmb = sb + 64 * sg_tok_oct;
            for (uint ik = 0; ik < NK / 8; ++ik) {
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                for (uint i = 0; i < 16; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                }
                lsma += 64 * 8;
                lsmb += 64 * 8;
            }
        }
    }

    if (t0 + NR1 <= cols && r0 + NR0 <= rows) {
        // Half output: per-lane element stores from the accumulator fragments.
        device half* cq = cb + (r0 + 32 * (sg % 2)) + (t0 + 32 * (sg / 2)) * c_row_stride;
        const short qid = (short)(lane / 4);
        const short fm = (qid & 4) + (((short)lane / 2) % 4);
        const short fn = (qid & 2) * 2 + ((short)lane % 2) * 2;
        for (uint i = 0; i < 16; ++i) {
            device half* d2 = cq + 8 * (i % 4) + 8 * c_row_stride * (i / 4)
                + (uint)fm * c_row_stride + (uint)fn;
            d2[0] = (half)mc[i].thread_elements()[0];
            d2[1] = (half)mc[i].thread_elements()[1];
        }
    } else {
        // Ragged token tail: stage each fragment through the A region, write guarded.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* scratch = reinterpret_cast<threadgroup float*>(shmem);
        for (uint i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], scratch + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint t_oct = t0 + 32 * (sg / 2) + 8 * (i / 4);
            const uint r_oct = r0 + 32 * (sg % 2) + 8 * (i % 4);
            for (uint e2 = lane; e2 < 64; e2 += 32) {
                const uint ft = e2 / 8;
                const uint fr = e2 % 8;
                if (t_oct + ft < cols && r_oct + fr < rows) {
                    cb[(t_oct + ft) * c_row_stride + r_oct + fr] =
                        half(scratch[sg * 64 + ft * 8 + fr]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Causal softmax over one score row per simdgroup: P[q][p] = exp(scale*S - m) / l for
// p <= q (zero otherwise, including the whole row when q >= n_tokens, so the PV pass
// can consume padded rows safely). Grid: (n_heads, n_pad rows), 32 threads.
kernel void softmax_causal_rows(
    device const half* s [[buffer(0)]],
    device half* p_out [[buffer(1)]],
    constant uint& n_pad [[buffer(2)]],
    constant uint& n_tokens [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& q_offset [[buffer(5)]],
    constant uint& rows_per_block [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // 8 rows per 256-thread threadgroup, one simdgroup per row. Rows are block-local
    // (the host may tile queries into blocks of rows_per_block at long context);
    // q_offset maps them to absolute positions for the causal mask. P is written only
    // up to the 64-aligned causal boundary the PV pass reads (its k-range is clamped
    // to ((q_abs/64)+1)*64), except padded rows which zero their full span for safety.
    const uint q = tg.y * 8 + sg;
    if (q >= rows_per_block) return;
    const uint q_abs = q + q_offset;
    const ulong base = (ulong)tg.x * rows_per_block * n_pad + (ulong)q * n_pad;
    device const half* srow = s + base;
    device half* prow = p_out + base;
    if (q_abs >= n_tokens) {
        for (uint j = lane; j < n_pad; j += 32) {
            prow[j] = half(0.0f);
        }
        return;
    }
    const uint write_end = min(n_pad, ((q_abs / 64) + 1) * 64);
    float m = -INFINITY;
    for (uint j = lane; j <= q_abs; j += 32) {
        m = max(m, srow[j] * scale);
    }
    m = simd_max(m);
    float l = 0.0f;
    for (uint j = lane; j <= q_abs; j += 32) {
        l += exp(srow[j] * scale - m);
    }
    l = simd_sum(l);
    const float inv = 1.0f / l;
    for (uint j = lane; j < write_end; j += 32) {
        prow[j] = (j <= q_abs) ? half(exp(srow[j] * scale - m) * inv) : half(0.0f);
    }
}

// Flash-tiled causal prefill attention: threadgroup (q_head, 32-query tile), simdgroup
// matrices for BOTH score and value matmuls. The query-tiled scalar kernel below still
// re-reads K/V once per 4-query group (~30GB on a 600-token prompt = the dominant
// prefill attention cost); here each K/V tile stages ONCE per 32-query tile (~4GB) and
// the dot products ride the MMA pipeline. Each simdgroup owns 8 queries across the
// whole 32-position tile, so the online-softmax state (m/l) is simdgroup-private —
// no cross-simdgroup merge; softmax rows are reduced by quads (4 lanes per row, 8
// columns each, quad_shuffle_xor combines). Q fragments are loaded into registers
// once and their staging region is reused as the V tile, so a kv-tile iteration costs
// two threadgroup barriers (stage K+V | consume). Accumulator rescaling uses a
// diagonal-matrix multiply (simdgroup fragments cannot be scaled per-row directly).
// Not byte-exact with the per-query kernel (tile MMA + tile-at-a-time softmax order);
// numerically equivalent — gated with the same CAMELID_METAL_MM stack and verified by
// greedy-token parity. Requires head_dim % 8 == 0 and head_dim <= 128.
//
// Threadgroup memory (Q/V + K half tiles, S f32, P half, diag, inv-l; 23KB at
// head_dim=128).
kernel void attention_prefill_flash_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& position_stride [[buffer(9)]],
    constant uint& kv_head_stride [[buffer(10)]],
    constant uint& kv_base_offset [[buffer(11)]],
    constant uint& n_tokens [[buffer(12)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint QT = 32;       // queries per threadgroup
    constexpr uint PT = 32;       // kv positions per tile
    constexpr uint MAX_DOCT = 16; // head_dim <= 128 -> at most 16 d-octets
    const uint tid = sg * 32 + lane;
    const uint head = tg.x;
    if (head >= n_heads) return;
    const uint tq0 = tg.y * QT;
    if (tq0 >= n_tokens) return;
    const uint d_oct = head_dim / 8;
    const uint q_stride = n_heads * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    // Shmem layout (half units). The Q staging region is consumed into register
    // fragments before the kv loop and then reused as the V tile.
    threadgroup half* qv_s = shmem;                  // QT x head_dim: Q staging, then V
    threadgroup half* k_s = shmem + QT * head_dim;   // PT x head_dim K tile
    threadgroup float* s_s =
        reinterpret_cast<threadgroup float*>(shmem + 2 * QT * head_dim); // QT x PT scores
    threadgroup half* p_s = shmem + 2 * QT * head_dim + QT * PT * 2; // QT x PT half
    // Per-tile rescale diagonal in f32: a half-precision corr compounds its rounding
    // once per kv tile (dozens of times at depth) into the O accumulators, which is
    // what destroyed long-prompt recall on this kernel.
    threadgroup float* diag_f =
        reinterpret_cast<threadgroup float*>(p_s + QT * PT); // 4 x 64 f32
    threadgroup float* linv_s = diag_f + 4 * 64; // QT inv-l values

    // Stage Q (q-major 8x8 blocks, pre-scaled), then pull this simdgroup's q-octet
    // into register fragments.
    {
        const uint q = tid / 4;
        const uint dseg = (tid % 4) * 32;
        const uint qq = tq0 + q;
        const uint sy = q / 8;
        const uint ly = q % 8;
        for (uint d = dseg; d < min(dseg + 32u, head_dim); d += 8) {
            const uint sx = d / 8;
            threadgroup half* dst = qv_s + 64 * (4 * sx + sy) + 8 * ly;
            if (qq < n_tokens) {
                device const float* src = query + qq * q_stride + head * head_dim + d;
                for (uint i = 0; i < 8; ++i) {
                    dst[i] = half(src[i] * scale);
                }
            } else {
                for (uint i = 0; i < 8; ++i) {
                    dst[i] = half(0.0f);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 q_frag[MAX_DOCT];
    for (uint dx = 0; dx < d_oct; ++dx) {
        simdgroup_load(q_frag[dx], qv_s + 64 * (4 * dx + sg), 8, 0, false);
    }

    // Per-simdgroup flash state: this sg owns queries [tq0 + sg*8, +8). Each quad of
    // lanes carries one row's m/l (replicated across the quad's 4 lanes).
    float m_state = -INFINITY;
    float l_state = 0.0f;
    simdgroup_float8x8 o_acc[MAX_DOCT];
    for (uint i = 0; i < d_oct; ++i) {
        o_acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint p_end = min(tq0 + QT, n_tokens); // last query of the tile is causal limit
    for (uint kp0 = 0; kp0 < p_end; kp0 += PT) {
        // Stage K (transposed d-major blocks for the score MMA) and V (natural-order
        // blocks for the value MMA) together: one barrier pair per kv tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint p = tid / 4;
            const uint dseg = (tid % 4) * 32;
            const uint pp = kp0 + p;
            const uint v_sx = p / 8;
            const uint v_ly = p % 8;
            // Both tiles stage natural-order (vector-friendly); the score MMA loads K
            // fragments with transpose=true — cheap from threadgroup memory, unlike
            // the catastrophic device transpose loads.
            if (pp < n_tokens) {
                device const float* ks = keys + kv_base + pp * position_stride + dseg;
                device const float* vs = values + kv_base + pp * position_stride + dseg;
                for (uint d = dseg; d < min(dseg + 32u, head_dim); d += 8) {
                    threadgroup half* kd = k_s + 64 * (v_sx * d_oct + d / 8) + 8 * v_ly;
                    threadgroup half* vd = qv_s + 64 * (v_sx * d_oct + d / 8) + 8 * v_ly;
                    device const float* sk = ks + (d - dseg);
                    device const float* sv = vs + (d - dseg);
                    for (uint i = 0; i < 8; ++i) {
                        kd[i] = half(sk[i]);
                        vd[i] = half(sv[i]);
                    }
                }
            } else {
                for (uint d = dseg; d < min(dseg + 32u, head_dim); d += 8) {
                    threadgroup half* kd = k_s + 64 * (v_sx * d_oct + d / 8) + 8 * v_ly;
                    threadgroup half* vd = qv_s + 64 * (v_sx * d_oct + d / 8) + 8 * v_ly;
                    for (uint i = 0; i < 8; ++i) {
                        kd[i] = half(0.0f);
                        vd[i] = half(0.0f);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T for this sg's 8 queries x 32 positions: 4 p-octet fragments.
        {
            simdgroup_float8x8 s_frag[4];
            for (uint j = 0; j < 4; ++j) {
                s_frag[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint dx = 0; dx < d_oct; ++dx) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_half8x8 kf;
                    // K tile is [p][d] natural order; transpose-load gives K^T = [d][p].
                    simdgroup_load(kf, k_s + 64 * (j * d_oct + dx), 8, 0, true);
                    simdgroup_multiply_accumulate(s_frag[j], q_frag[dx], kf, s_frag[j]);
                }
            }
            for (uint j = 0; j < 4; ++j) {
                simdgroup_store(s_frag[j], s_s + (sg * 8) * PT + j * 8, PT, 0, false);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax, one quad per row (lane = 4*row_in_octet + quarter): each lane
        // scans 8 columns, quad_shuffle_xor combines; m/l replicate across the quad.
        {
            const uint row = lane / 4;
            const uint quarter = lane % 4;
            const uint q_global = tq0 + sg * 8 + row;
            threadgroup float* srow = s_s + (sg * 8 + row) * PT + quarter * 8;
            float m_new = m_state;
            for (uint j = 0; j < 8; ++j) {
                const uint p = kp0 + quarter * 8 + j;
                const float s = (p <= q_global && p < n_tokens) ? srow[j] : -INFINITY;
                srow[j] = s;
                m_new = max(m_new, s);
            }
            m_new = max(m_new, quad_shuffle_xor(m_new, 1));
            m_new = max(m_new, quad_shuffle_xor(m_new, 2));
            const float corr = (m_state == -INFINITY) ? 1.0f : exp(m_state - m_new);
            float l_add = 0.0f;
            threadgroup half* prow = p_s + (sg * 8 + row) * PT + quarter * 8;
            for (uint j = 0; j < 8; ++j) {
                const float w = (srow[j] == -INFINITY) ? 0.0f : exp(srow[j] - m_new);
                prow[j] = half(w);
                l_add += w;
            }
            l_add += quad_shuffle_xor(l_add, 1);
            l_add += quad_shuffle_xor(l_add, 2);
            l_state = l_state * corr + l_add;
            m_state = m_new;
            if (quarter == 0) {
                diag_f[sg * 64 + row * 8 + row] = corr;
            }
            // Zero off-diagonals once (they are never overwritten afterwards).
            if (kp0 == 0 && quarter != 0) {
                for (uint c2 = quarter - 1; c2 < 8; c2 += 3) {
                    if (c2 != row) {
                        diag_f[sg * 64 + row * 8 + c2] = 0.0f;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale O by diag(corr) in f32, then O += P V.
        {
            simdgroup_float8x8 dg;
            simdgroup_load(dg, diag_f + sg * 64, 8, 0, false);
            for (uint i = 0; i < d_oct; ++i) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, dg, o_acc[i]);
                o_acc[i] = tmp;
            }
        }
        for (uint px = 0; px < 4; ++px) {
            simdgroup_half8x8 pf;
            simdgroup_load(pf, p_s + (sg * 8) * PT + px * 8, PT, 0, false);
            for (uint i = 0; i < d_oct; ++i) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, qv_s + 64 * (px * d_oct + i), 8, 0, false);
                simdgroup_multiply_accumulate(o_acc[i], pf, vf, o_acc[i]);
            }
        }
    }

    // Final 1/l scaling via an f32 diagonal staged through the score region.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        linv_s[sg * 8 + lane] = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
        threadgroup float* fdiag = s_s + sg * 64;
        for (uint e2 = lane; e2 < 64; e2 += 32) {
            const uint r = e2 / 8;
            const uint c2 = e2 % 8;
            fdiag[e2] = (r == c2) ? linv_s[sg * 8 + r] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg;
        simdgroup_load(dg, fdiag, 8, 0, false);
        for (uint i = 0; i < d_oct; ++i) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, dg, o_acc[i]);
            o_acc[i] = tmp;
        }
    }
    const uint q_base = tq0 + sg * 8;
    if (q_base + 8 <= n_tokens) {
        device float* out = output + q_base * q_stride + head * head_dim;
        for (uint i = 0; i < d_oct; ++i) {
            simdgroup_store(o_acc[i], out + i * 8, q_stride, 0, false);
        }
    } else if (q_base < n_tokens) {
        // Ragged tail: stage each fragment through the score region, write guarded.
        for (uint i = 0; i < d_oct; ++i) {
            simdgroup_store(o_acc[i], s_s + sg * 64, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e2 = lane; e2 < 64; e2 += 32) {
                const uint r = e2 / 8;
                const uint c2 = e2 % 8;
                if (q_base + r < n_tokens) {
                    output[(q_base + r) * q_stride + head * head_dim + i * 8 + c2] =
                        s_s[sg * 64 + r * 8 + c2];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Query-TILED causal prefill attention: threadgroup (head, 8-query tile). The dominant
// prefill attention cost is K/V traffic — one threadgroup per query re-reads every K/V
// row, Sum(t) over 600 queries x 24 heads ~ 100+ GB. Here each K/V row is loaded ONCE
// per 8-query tile (traffic / 8); every (query, simdgroup) still walks positions in the
// same stride-NSG order with the same flash update and the same 4-way merge as
// attention_prefill_v2_f32, so outputs stay byte-exact with the untiled kernel.
kernel void attention_prefill_v3_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& position_stride [[buffer(9)]],
    constant uint& kv_head_stride [[buffer(10)]],
    constant uint& kv_base_offset [[buffer(11)]],
    constant uint& n_tokens [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint QT = 4;   // queries per threadgroup (8 spills registers: measured 493ms vs 300 untiled)
    constexpr uint MAX_DPL = 4; // head_dim <= 128 -> at most 4 dims per lane
    const uint head = tg.x;
    if (head >= n_heads) return;
    const uint tq0 = tg.y * QT;
    if (tq0 >= n_tokens) return;
    const uint qn = min(uint(QT), n_tokens - tq0);
    const uint dpl = head_dim / 32;
    const uint q_stride = n_heads * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    // Per-query scaled Q slices and flash state.
    float q[QT][MAX_DPL];
    float m[QT];
    float l[QT];
    float acc[QT][MAX_DPL];
    for (uint qi = 0; qi < qn; ++qi) {
        const uint q_base = (tq0 + qi) * q_stride + head * head_dim;
        for (uint i = 0; i < dpl; ++i) {
            q[qi][i] = query[q_base + lane + i * 32] * scale;
            acc[qi][i] = 0.0f;
        }
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    // Walk positions once; each K/V row feeds every in-range query of the tile.
    const uint p_max = tq0 + qn; // the tile's last query attends positions 0..=p_max-1
    for (uint p = sg; p < p_max; p += NSG) {
        device const float* kr = keys + kv_base + p * position_stride;
        device const float* vr = values + kv_base + p * position_stride;
        float kv[MAX_DPL];
        float vv[MAX_DPL];
        for (uint i = 0; i < dpl; ++i) {
            kv[i] = kr[lane + i * 32];
            vv[i] = vr[lane + i * 32];
        }
        // Causal: position p contributes to queries with token index >= p.
        const uint qi0 = p > tq0 ? p - tq0 : 0;
        for (uint qi = qi0; qi < qn; ++qi) {
            float s = 0.0;
            for (uint i = 0; i < dpl; ++i) {
                s += q[qi][i] * kv[i];
            }
            s = simd_sum(s);
            float m_new = max(m[qi], s);
            float w = exp(s - m_new);
            float corr = exp(m[qi] - m_new);
            for (uint i = 0; i < dpl; ++i) {
                acc[qi][i] = acc[qi][i] * corr + w * vv[i];
            }
            l[qi] = l[qi] * corr + w;
            m[qi] = m_new;
        }
    }

    // Merge the four simdgroup states per query — identical to attention_prefill_v2_f32.
    threadgroup float sh_m[NSG];
    threadgroup float sh_l[NSG];
    threadgroup float sh_acc[NSG * 128];
    for (uint qi = 0; qi < qn; ++qi) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            sh_m[sg] = m[qi];
            sh_l[sg] = l[qi];
        }
        for (uint i = 0; i < dpl; ++i) {
            sh_acc[sg * 128 + lane + i * 32] = acc[qi][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float m_tot = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
            float l_tot = 0.0;
            float w[NSG];
            for (uint i = 0; i < NSG; ++i) {
                w[i] = exp(sh_m[i] - m_tot);
                l_tot += sh_l[i] * w[i];
            }
            float inv = 1.0 / l_tot;
            const uint out_base = (tq0 + qi) * q_stride + head * head_dim;
            for (uint i = 0; i < dpl; ++i) {
                uint d = lane + i * 32;
                float o = 0.0;
                for (uint g2 = 0; g2 < NSG; ++g2) {
                    o += sh_acc[g2 * 128 + d] * w[g2];
                }
                output[out_base + d] = o * inv;
            }
        }
    }
}

// attention_decode_v2_f32 over n_tokens causal queries: threadgroup (head, token), each
// query attending to positions 0..=token. Identical per-(token, head) math and position
// order to the per-position decode dispatches, so logits stay byte-exact; the win is the
// n_heads*n_tokens-wide grid (full GPU occupancy) and ~600x fewer encoder commands.
kernel void attention_prefill_v2_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& position_stride [[buffer(9)]],
    constant uint& kv_head_stride [[buffer(10)]],
    constant uint& kv_base_offset [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint MAX_DPL = 4; // head_dim <= 128 -> at most 4 dims per lane
    const uint head = tg.x;
    if (head >= n_heads) return;
    const uint t = tg.y;
    const uint position_count = t + 1u;
    const uint dpl = head_dim / 32;
    const uint q_base = (t * n_heads + head) * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    // Query slice for this lane's dims, scaled once.
    float q[MAX_DPL];
    for (uint i = 0; i < dpl; ++i) {
        q[i] = query[q_base + lane + i * 32] * scale;
    }

    // Flash-style running state for this simdgroup's positions.
    float m = -INFINITY;
    float l = 0.0;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint p = sg; p < position_count; p += NSG) {
        device const float* kr = keys + kv_base + p * position_stride;
        float s = 0.0;
        for (uint i = 0; i < dpl; ++i) {
            s += q[i] * kr[lane + i * 32];
        }
        s = simd_sum(s);
        float m_new = max(m, s);
        float w = exp(s - m_new);
        float corr = exp(m - m_new);
        device const float* vr = values + kv_base + p * position_stride;
        for (uint i = 0; i < dpl; ++i) {
            acc[i] = acc[i] * corr + w * vr[lane + i * 32];
        }
        l = l * corr + w;
        m = m_new;
    }

    // Merge the four simdgroup states: out = sum_i acc_i * exp(m_i - M) / sum_i l_i * exp(m_i - M).
    threadgroup float sh_m[NSG];
    threadgroup float sh_l[NSG];
    threadgroup float sh_acc[NSG * 128];
    if (lane == 0) {
        sh_m[sg] = m;
        sh_l[sg] = l;
    }
    for (uint i = 0; i < dpl; ++i) {
        sh_acc[sg * 128 + lane + i * 32] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m_tot = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float l_tot = 0.0;
        float w[NSG];
        for (uint i = 0; i < NSG; ++i) {
            w[i] = exp(sh_m[i] - m_tot);
            l_tot += sh_l[i] * w[i];
        }
        float inv = 1.0 / l_tot;
        for (uint i = 0; i < dpl; ++i) {
            uint d = lane + i * 32;
            float o = 0.0;
            for (uint g2 = 0; g2 < NSG; ++g2) {
                o += sh_acc[g2 * 128 + d] * w[g2];
            }
            output[q_base + d] = o * inv;
        }
    }
}

// Greedy argmax over one logits row, matching the CPU greedy_sample exactly:
// strict greater-than with lowest-index tie-break. One threadgroup; each thread
// scans an ascending index stride (strict > keeps the lowest index within a
// thread), then the tree reduction prefers the lower index on equal values.
// NaN logits are never selected (the CPU path errors instead; the resident fast
// path only sees logits the model just produced).
kernel void argmax_f32_greedy(
    device const float* logits [[buffer(0)]],
    device uint* out_id [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float sh_val[1024];
    threadgroup uint sh_idx[1024];
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < count; i += tg_size) {
        const float v = logits[i];
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
        out_id[0] = sh_idx[0];
    }
}

// Temperature sampling via Gumbel-max. A categorical draw from
// softmax(logits / temperature) is argmax(logit / temperature + Gumbel(0, 1)),
// so one linear GPU pass replaces the full-logit host copy, softmax, and sort.
// The stateless SplitMix64 hash makes the result reproducible for a given seed.
inline float splitmix_uniform(ulong seed, uint idx) {
    ulong z = seed + 0x9E3779B97F4A7C15ul * ulong(idx + 1u);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z ^= z >> 31;
    uint mantissa = uint(z >> 40);
    return (float(mantissa) + 0.5f) / 16777216.0f;
}

kernel void sample_gumbel_f32(
    device const float* logits [[buffer(0)]],
    device uint* out_id [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& inv_temperature [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float sh_val[1024];
    threadgroup uint sh_idx[1024];
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < count; i += tg_size) {
        float u = splitmix_uniform(seed, i);
        float gumbel = -log(-log(u));
        float value = logits[i] * inv_temperature + gumbel;
        if (value > best) {
            best = value;
            best_i = i;
        }
    }
    sh_val[tid] = best;
    sh_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = sh_val[tid + stride];
            uint other_i = sh_idx[tid + stride];
            if (other > sh_val[tid] || (other == sh_val[tid] && other_i < sh_idx[tid])) {
                sh_val[tid] = other;
                sh_idx[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_id[0] = sh_idx[0];
    }
}

// Dequantize the sampled token's Q8_0 embedding row (wire 34-byte blocks) into the
// f32 input buffer the next pre-encoded token graph reads. Same dequant math as the
// CPU embedding_lookup (f16 scale -> f32, times the i8 quant), so the values are
// bit-identical to the CPU-written embedding.
kernel void embed_row_gather_q8_wire(
    device const char* emb [[buffer(0)]],
    device const uint* token_id [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& bpr [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= bpr * 32) return;
    device const char* wb = emb + ((ulong)token_id[0] * bpr + gid / 32) * 34;
    const float scale = float(*reinterpret_cast<device const half*>(wb));
    out[gid] = float(wb[2 + (gid % 32)]) * scale;
}

// ---------------------------------------------------------------------------
// WIN2METAL Phase 4 — TREE speculative-verify attention kernels.
//
// Each is a line-for-line clone of the decode-attention kernel the LINEAR verify
// path routes to, plus a transparent slot indirection on the draft tail. Two new
// params: `tail_slots` (per-node ancestor draft cache slots, increasing order,
// length tail_count) at buffer(14) and `base` (the contiguous prefix length) at
// buffer(15). Every cache-position deref `p`/`a` becomes
//     uint slot = (a < base) ? a : tail_slots[a - base];
// then indexes `kv_base + slot*position_stride + d`. position_count == count ==
// base + tail_count. The split-K chunk boundaries (n_splits over [0,count)) and
// every reduction stay IDENTICAL to the cloned kernel, so a LINEAR tree — where
// node i has ancestor slots [base..base+i] so slot==a for every a — produces the
// exact same instruction stream / float order / split boundaries as the linear
// kernel ⇒ bit-identical. Only the ≤TREE_MAX_NODES draft tail is gathered; the
// huge [0,base) prefix stays direct (no threadgroup-memory blow-up).
//
// NOTE: buffer(13) is already n_splits in the split-K kernels, so the tree params
// uniformly take buffer(14)/(15) across all four variants.
// ---------------------------------------------------------------------------

// Tree clone of attention_decode_f32 (universal one-simdgroup-per-head fallback).
kernel void attention_decode_f32_tree(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    device const uint* tail_slots [[buffer(14)]],
    constant uint& base [[buffer(15)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (head >= n_heads) return;
    uint kv_head = head / group;
    uint q_base = head * head_dim;
    uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    uint score_base = head * position_count;

    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32) {
        uint slot = (p < base) ? p : tail_slots[p - base];
        uint k_base = kv_base + slot * position_stride;
        float s = 0.0;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * keys[k_base + d];
        }
        s *= scale;
        scores[score_base + p] = s;
        local_max = max(local_max, s);
    }
    float max_score = simd_max(local_max);

    float local_sum = 0.0;
    for (uint p = lane; p < position_count; p += 32) {
        float e = exp(scores[score_base + p] - max_score);
        scores[score_base + p] = e;
        local_sum += e;
    }
    float inv = 1.0 / simd_sum(local_sum);

    threadgroup_barrier(mem_flags::mem_device);

    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0;
        for (uint p = 0; p < position_count; ++p) {
            uint slot = (p < base) ? p : tail_slots[p - base];
            acc += scores[score_base + p] * inv * values[kv_base + slot * position_stride + d];
        }
        output[q_base + d] = acc;
    }
}

// Tree clone of attention_decode_v2_f32 (the pc<128 tiled path).
kernel void attention_decode_v2_tree(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    device const uint* tail_slots [[buffer(14)]],
    constant uint& base [[buffer(15)]],
    uint head [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint MAX_DPL = 4; // head_dim <= 128 -> at most 4 dims per lane
    if (head >= n_heads) return;
    const uint dpl = head_dim / 32;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + (head / group) * kv_head_stride;

    float q[MAX_DPL];
    for (uint i = 0; i < dpl; ++i) {
        q[i] = query[q_base + lane + i * 32] * scale;
    }

    float m = -INFINITY;
    float l = 0.0;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint p = sg; p < position_count; p += NSG) {
        uint slot = (p < base) ? p : tail_slots[p - base];
        device const float* kr = keys + kv_base + slot * position_stride;
        float s = 0.0;
        for (uint i = 0; i < dpl; ++i) {
            s += q[i] * kr[lane + i * 32];
        }
        s = simd_sum(s);
        float m_new = max(m, s);
        float w = exp(s - m_new);
        float corr = exp(m - m_new);
        device const float* vr = values + kv_base + slot * position_stride;
        for (uint i = 0; i < dpl; ++i) {
            acc[i] = acc[i] * corr + w * vr[lane + i * 32];
        }
        l = l * corr + w;
        m = m_new;
    }

    threadgroup float sh_m[NSG];
    threadgroup float sh_l[NSG];
    threadgroup float sh_acc[NSG * 128];
    if (lane == 0) {
        sh_m[sg] = m;
        sh_l[sg] = l;
    }
    for (uint i = 0; i < dpl; ++i) {
        sh_acc[sg * 128 + lane + i * 32] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m_tot = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float l_tot = 0.0;
        float w[NSG];
        for (uint i = 0; i < NSG; ++i) {
            w[i] = exp(sh_m[i] - m_tot);
            l_tot += sh_l[i] * w[i];
        }
        float inv = 1.0 / l_tot;
        for (uint i = 0; i < dpl; ++i) {
            uint d = lane + i * 32;
            float o = 0.0;
            for (uint g2 = 0; g2 < NSG; ++g2) {
                o += sh_acc[g2 * 128 + d] * w[g2];
            }
            output[q_base + d] = o * inv;
        }
    }
}

// Tree clone of attention_decode_splitk_kv16 (the staged depth path, head_dim != 128).
// The original's local `base` (a K/V byte offset) is renamed `koff` to free the name
// for the prefix-length param.
kernel void attention_decode_splitk_kv16_tree(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]], // [n_heads][n_splits][head_dim + 2]
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    device const uint* tail_slots [[buffer(14)]],
    constant uint& base [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint PT = 16;
    constexpr uint MAX_DPL = 4;
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint dpl = head_dim / 32;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);

    const uint qh = kvh * group + sg;
    const bool active = sg < group && qh < n_heads;
    float q[MAX_DPL];
    if (active) {
        for (uint i = 0; i < dpl; ++i) {
            q[i] = query[qh * head_dim + lane + i * 32] * scale;
        }
    }
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DPL] = {0.0f, 0.0f, 0.0f, 0.0f};

    threadgroup half4 k_s4[PT * 32];
    threadgroup half4 v_s4[PT * 32];
    threadgroup half* k_s = reinterpret_cast<threadgroup half*>(k_s4);
    threadgroup half* v_s = reinterpret_cast<threadgroup half*>(v_s4);
    const uint tid = sg * 32 + lane;
    for (uint pt = p0; pt < p1; pt += PT) {
        const uint count = min(PT, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx4 = tid; idx4 < (count * head_dim) / 4; idx4 += 128) {
            const uint e4 = idx4 * 4;
            const uint p = e4 / head_dim;
            const uint d = e4 % head_dim;
            const uint pos = pt + p;
            const uint slot = (pos < base) ? pos : tail_slots[pos - base];
            const uint koff = kv_base + slot * position_stride + d;
            k_s4[idx4] = *reinterpret_cast<device const half4*>(keys + koff);
            v_s4[idx4] = *reinterpret_cast<device const half4*>(values + koff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint j0 = 0; j0 < count; j0 += 4) {
                float s4[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    const uint j = j0 + jj;
                    if (j < count) {
                        float s = 0.0f;
                        for (uint i = 0; i < dpl; ++i) {
                            s += q[i] * float(k_s[j * head_dim + lane + i * 32]);
                        }
                        s4[jj] = simd_sum(s);
                    } else {
                        s4[jj] = -INFINITY;
                    }
                }
                const float m4 = max(max(s4[0], s4[1]), max(s4[2], s4[3]));
                const float m_new = max(m, m4);
                const float corr = exp(m - m_new);
                float w4[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    w4[jj] = (s4[jj] == -INFINITY) ? 0.0f : exp(s4[jj] - m_new);
                }
                for (uint i = 0; i < dpl; ++i) {
                    float a = acc[i] * corr;
                    for (uint jj = 0; jj < 4; ++jj) {
                        if (j0 + jj < count) {
                            a += w4[jj] * float(v_s[(j0 + jj) * head_dim + lane + i * 32]);
                        }
                    }
                    acc[i] = a;
                }
                l = l * corr + w4[0] + w4[1] + w4[2] + w4[3];
                m = m_new;
            }
        }
    }
    if (active) {
        device float* dst = partials + ((ulong)qh * n_splits + split) * (head_dim + 2);
        for (uint i = 0; i < dpl; ++i) {
            dst[lane + i * 32] = acc[i];
        }
        if (lane == 0) {
            dst[head_dim] = m;
            dst[head_dim + 1] = l;
        }
    }
}

// Tree clone of attention_decode_splitk_kv16_direct (head_dim == 128).
kernel void attention_decode_splitk_kv16_direct_tree(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]], // [n_heads][n_splits][head_dim + 2]
    constant uint& n_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& position_count [[buffer(7)]],
    constant uint& group [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_stride [[buffer(10)]],
    constant uint& kv_head_stride [[buffer(11)]],
    constant uint& kv_base_offset [[buffer(12)]],
    constant uint& n_splits [[buffer(13)]],
    device const uint* tail_slots [[buffer(14)]],
    constant uint& base [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint kvh = tg.x;
    const uint split = tg.y;
    const uint kv_base = kv_base_offset + kvh * kv_head_stride;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);

    const uint qh = kvh * group + sg;
    const bool active = sg < group && qh < n_heads;
    float4 q4 = float4(0.0f);
    if (active) {
        q4 = *reinterpret_cast<device const float4*>(query + qh * 128 + lane * 4) * scale;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = float4(0.0f);
    if (active) {
        for (uint j0 = p0; j0 < p1; j0 += 4) {
            float s4[4];
            for (uint jj = 0; jj < 4; ++jj) {
                const uint j = j0 + jj;
                if (j < p1) {
                    const uint slot = (j < base) ? j : tail_slots[j - base];
                    const half4 k4 = *reinterpret_cast<device const half4*>(
                        keys + kv_base + slot * position_stride + lane * 4);
                    s4[jj] = simd_sum(dot(float4(k4), q4));
                } else {
                    s4[jj] = -INFINITY;
                }
            }
            const float m4 = max(max(s4[0], s4[1]), max(s4[2], s4[3]));
            const float m_new = max(m, m4);
            const float corr = exp(m - m_new);
            float w4[4];
            for (uint jj = 0; jj < 4; ++jj) {
                w4[jj] = (s4[jj] == -INFINITY) ? 0.0f : exp(s4[jj] - m_new);
            }
            acc *= corr;
            for (uint jj = 0; jj < 4; ++jj) {
                const uint j = j0 + jj;
                if (j < p1) {
                    const uint slot = (j < base) ? j : tail_slots[j - base];
                    const half4 v4 = *reinterpret_cast<device const half4*>(
                        values + kv_base + slot * position_stride + lane * 4);
                    acc += w4[jj] * float4(v4);
                }
            }
            l = l * corr + w4[0] + w4[1] + w4[2] + w4[3];
            m = m_new;
        }
        device float* dst = partials + ((ulong)qh * n_splits + split) * (128 + 2);
        dst[lane * 4] = acc.x;
        dst[lane * 4 + 1] = acc.y;
        dst[lane * 4 + 2] = acc.z;
        dst[lane * 4 + 3] = acc.w;
        if (lane == 0) {
            dst[128] = m;
            dst[129] = l;
        }
    }
}
"#;

#[cfg(target_os = "macos")]
fn metal_linear_kernel() -> Option<&'static MetalLinearKernel> {
    METAL_LINEAR_KERNEL
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            // A compile failure here silently disables the ENTIRE Metal stack (every
            // caller sees None and falls back to CPU paths) — always say why.
            let library = device
                .new_library_with_source(LINEAR_ROW_SHADER, &options)
                .map_err(|err| eprintln!("[metal] LINEAR_ROW_SHADER compile failed: {err}"))
                .ok()?;
            let elementwise_library = device
                .new_library_with_source(ELEMENTWISE_SHADER, &options)
                .map_err(|err| eprintln!("[metal] ELEMENTWISE_SHADER compile failed: {err}"))
                .ok()?;
            let rms_norm_function = elementwise_library
                .get_function("rms_norm_f32", None)
                .ok()?;
            let rms_norm_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_function)
                .ok()?;
            let rms_norm_per_head_function = elementwise_library
                .get_function("rms_norm_per_head_f32", None)
                .ok()?;
            let rms_norm_per_head_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_per_head_function)
                .ok()?;
            let residual_add_function = elementwise_library
                .get_function("residual_add_f32", None)
                .ok()?;
            let residual_add_pipeline = device
                .new_compute_pipeline_state_with_function(&residual_add_function)
                .ok()?;
            let silu_mul_function = elementwise_library
                .get_function("silu_mul_f32", None)
                .ok()?;
            let silu_mul_pipeline = device
                .new_compute_pipeline_state_with_function(&silu_mul_function)
                .ok()?;
            let gelu_mul_function = elementwise_library
                .get_function("gelu_mul_f32", None)
                .ok()?;
            let gelu_mul_pipeline = device
                .new_compute_pipeline_state_with_function(&gelu_mul_function)
                .ok()?;
            let soft_cap_function = elementwise_library
                .get_function("soft_cap_f32", None)
                .ok()?;
            let soft_cap_pipeline = device
                .new_compute_pipeline_state_with_function(&soft_cap_function)
                .ok()?;
            let scale_function = elementwise_library.get_function("scale_f32", None).ok()?;
            let scale_pipeline = device
                .new_compute_pipeline_state_with_function(&scale_function)
                .ok()?;
            let rope_rotate_function = elementwise_library
                .get_function("rope_rotate_f32", None)
                .ok()?;
            let rope_rotate_pipeline = device
                .new_compute_pipeline_state_with_function(&rope_rotate_function)
                .ok()?;
            let attention_decode_function = elementwise_library
                .get_function("attention_decode_f32", None)
                .ok()?;
            let attention_decode_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_function)
                .ok()?;
            let attention_decode_kv16_function = elementwise_library
                .get_function("attention_decode_kv16", None)
                .ok()?;
            let attention_decode_kv16_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_kv16_function)
                .ok()?;
            let attention_decode_v2_function = elementwise_library
                .get_function("attention_decode_v2_f32", None)
                .ok()?;
            let attention_decode_v2_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_v2_function)
                .ok()?;
            let attention_decode_v2_kv16_function = elementwise_library
                .get_function("attention_decode_v2_kv16", None)
                .ok()?;
            let attention_decode_v2_kv16_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_v2_kv16_function)
                .ok()?;
            let quantize_q8_0_function = elementwise_library
                .get_function("quantize_q8_0_f32", None)
                .ok()?;
            let quantize_q8_0_pipeline = device
                .new_compute_pipeline_state_with_function(&quantize_q8_0_function)
                .ok()?;
            let kv_scatter_function = elementwise_library
                .get_function("kv_scatter_f32", None)
                .ok()?;
            let kv_scatter_pipeline = device
                .new_compute_pipeline_state_with_function(&kv_scatter_function)
                .ok()?;
            let kv_scatter_kv16_function = elementwise_library
                .get_function("kv_scatter_kv16", None)
                .ok()?;
            let kv_scatter_kv16_pipeline = device
                .new_compute_pipeline_state_with_function(&kv_scatter_kv16_function)
                .ok()?;
            let f32_to_f16_function = elementwise_library.get_function("f32_to_f16", None).ok()?;
            let f32_to_f16_pipeline = device
                .new_compute_pipeline_state_with_function(&f32_to_f16_function)
                .ok()?;
            let rms_norm_batch_function = elementwise_library
                .get_function("rms_norm_batch_f32", None)
                .ok()?;
            let rms_norm_batch_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_batch_function)
                .ok()?;
            let rms_norm_batch_f16o_function = elementwise_library
                .get_function("rms_norm_batch_f16o", None)
                .ok()?;
            let rms_norm_batch_f16o_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_batch_f16o_function)
                .ok()?;
            let silu_mul_f16o_function = elementwise_library
                .get_function("silu_mul_f16o", None)
                .ok()?;
            let silu_mul_f16o_pipeline = device
                .new_compute_pipeline_state_with_function(&silu_mul_f16o_function)
                .ok()?;
            let rope_rotate_batch_function = elementwise_library
                .get_function("rope_rotate_batch_f32", None)
                .ok()?;
            let rope_rotate_batch_pipeline = device
                .new_compute_pipeline_state_with_function(&rope_rotate_batch_function)
                .ok()?;
            let kv_scatter_batch_function = elementwise_library
                .get_function("kv_scatter_batch_f32", None)
                .ok()?;
            let kv_scatter_batch_pipeline = device
                .new_compute_pipeline_state_with_function(&kv_scatter_batch_function)
                .ok()?;
            let attention_prefill_v3_function = elementwise_library
                .get_function("attention_prefill_v3_f32", None)
                .ok()?;
            let attention_prefill_v3_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_prefill_v3_function)
                .ok()?;
            let attention_prefill_flash_function = elementwise_library
                .get_function("attention_prefill_flash_f32", None)
                .ok()?;
            let attention_prefill_flash_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_prefill_flash_function)
                .ok()?;
            let half_mm_batched_function = elementwise_library
                .get_function("half_mm_batched", None)
                .ok()?;
            let half_mm_batched_pipeline = device
                .new_compute_pipeline_state_with_function(&half_mm_batched_function)
                .ok()?;
            let half_mm_batched_f16o_function = elementwise_library
                .get_function("half_mm_batched_f16o", None)
                .ok()?;
            let half_mm_batched_f16o_pipeline = device
                .new_compute_pipeline_state_with_function(&half_mm_batched_f16o_function)
                .ok()?;
            let transpose_v16_function = elementwise_library
                .get_function("transpose_v16", None)
                .ok()?;
            let transpose_v16_pipeline = device
                .new_compute_pipeline_state_with_function(&transpose_v16_function)
                .ok()?;
            let rope_scatter_qh_function = elementwise_library
                .get_function("rope_scatter_qh_batch", None)
                .ok()?;
            let rope_scatter_qh_pipeline = device
                .new_compute_pipeline_state_with_function(&rope_scatter_qh_function)
                .ok()?;
            let rope_scatter_qh_h_function = elementwise_library
                .get_function("rope_scatter_qh_batch_h", None)
                .ok()?;
            let rope_scatter_qh_h_pipeline = device
                .new_compute_pipeline_state_with_function(&rope_scatter_qh_h_function)
                .ok()?;
            let rms_norm_batch_h_function = elementwise_library
                .get_function("rms_norm_batch_h", None)
                .ok()?;
            let rms_norm_batch_h_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_batch_h_function)
                .ok()?;
            let residual_add_h_function = elementwise_library
                .get_function("residual_add_h", None)
                .ok()?;
            let residual_add_h_pipeline = device
                .new_compute_pipeline_state_with_function(&residual_add_h_function)
                .ok()?;
            let silu_mul_h2_function =
                elementwise_library.get_function("silu_mul_h2", None).ok()?;
            let silu_mul_h2_pipeline = device
                .new_compute_pipeline_state_with_function(&silu_mul_h2_function)
                .ok()?;
            let softmax_causal_rows_function = elementwise_library
                .get_function("softmax_causal_rows", None)
                .ok()?;
            let softmax_causal_rows_pipeline = device
                .new_compute_pipeline_state_with_function(&softmax_causal_rows_function)
                .ok()?;
            let rms_norm_quantize_function = elementwise_library
                .get_function("rms_norm_quantize_f32", None)
                .ok()?;
            let rms_norm_quantize_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_quantize_function)
                .ok()?;
            let silu_mul_quantize_function = elementwise_library
                .get_function("silu_mul_quantize_f32", None)
                .ok()?;
            let silu_mul_quantize_pipeline = device
                .new_compute_pipeline_state_with_function(&silu_mul_quantize_function)
                .ok()?;
            let attention_decode_splitk_function = elementwise_library
                .get_function("attention_decode_splitk_f32", None)
                .ok()?;
            let attention_decode_splitk_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_splitk_function)
                .ok()?;
            let attention_decode_splitk_kv16_function = elementwise_library
                .get_function("attention_decode_splitk_kv16", None)
                .ok()?;
            let attention_decode_splitk_kv16_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_splitk_kv16_function)
                .ok()?;
            let attention_decode_splitk_kv16_direct_function = elementwise_library
                .get_function("attention_decode_splitk_kv16_direct", None)
                .ok()?;
            let attention_decode_splitk_kv16_direct_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &attention_decode_splitk_kv16_direct_function,
                )
                .ok()?;
            let attention_splitk_kv16_stageonly_function = elementwise_library
                .get_function("attention_splitk_kv16_stageonly", None)
                .ok()?;
            let attention_splitk_kv16_stageonly_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_splitk_kv16_stageonly_function)
                .ok()?;
            let attention_decode_splitk_merge_function = elementwise_library
                .get_function("attention_decode_splitk_merge_f32", None)
                .ok()?;
            let attention_decode_splitk_merge_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_splitk_merge_function)
                .ok()?;
            // WIN2METAL Phase 4 — TREE-verify attention clones.
            let attention_decode_f32_tree_function = elementwise_library
                .get_function("attention_decode_f32_tree", None)
                .ok()?;
            let attention_decode_f32_tree_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_f32_tree_function)
                .ok()?;
            let attention_decode_v2_tree_function = elementwise_library
                .get_function("attention_decode_v2_tree", None)
                .ok()?;
            let attention_decode_v2_tree_pipeline = device
                .new_compute_pipeline_state_with_function(&attention_decode_v2_tree_function)
                .ok()?;
            let attention_decode_splitk_kv16_tree_function = elementwise_library
                .get_function("attention_decode_splitk_kv16_tree", None)
                .ok()?;
            let attention_decode_splitk_kv16_tree_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &attention_decode_splitk_kv16_tree_function,
                )
                .ok()?;
            let attention_decode_splitk_kv16_direct_tree_function = elementwise_library
                .get_function("attention_decode_splitk_kv16_direct_tree", None)
                .ok()?;
            let attention_decode_splitk_kv16_direct_tree_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &attention_decode_splitk_kv16_direct_tree_function,
                )
                .ok()?;
            let argmax_f32_greedy_function = elementwise_library
                .get_function("argmax_f32_greedy", None)
                .ok()?;
            let argmax_f32_greedy_pipeline = device
                .new_compute_pipeline_state_with_function(&argmax_f32_greedy_function)
                .ok()?;
            let sample_gumbel_f32_function = elementwise_library
                .get_function("sample_gumbel_f32", None)
                .ok()?;
            let sample_gumbel_f32_pipeline = device
                .new_compute_pipeline_state_with_function(&sample_gumbel_f32_function)
                .ok()?;
            let embed_row_gather_q8_wire_function = elementwise_library
                .get_function("embed_row_gather_q8_wire", None)
                .ok()?;
            let embed_row_gather_q8_wire_pipeline = device
                .new_compute_pipeline_state_with_function(&embed_row_gather_q8_wire_function)
                .ok()?;
            let descriptor_function = library.get_function("linear_row_f32", None).ok()?;
            let descriptor_pipeline = device
                .new_compute_pipeline_state_with_function(&descriptor_function)
                .ok()?;
            let transposed_function = library
                .get_function("linear_row_transposed_f32", None)
                .ok()?;
            let transposed_pipeline = device
                .new_compute_pipeline_state_with_function(&transposed_function)
                .ok()?;
            let q8_0_encoded_function =
                library.get_function("q8_0_encoded_linear_row", None).ok()?;
            let q8_0_encoded_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_encoded_function)
                .ok()?;
            let q8_0_encoded_rows_function = library
                .get_function("q8_0_encoded_linear_rows", None)
                .ok()?;
            let q8_0_encoded_rows_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_encoded_rows_function)
                .ok()?;
            let q8_0_block_function = library.get_function("q8_0_block_linear_row", None).ok()?;
            let q8_0_block_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_function)
                .ok()?;
            let q8_0_block_simd_function = library
                .get_function("q8_0_block_linear_row_simd", None)
                .ok()?;
            let q8_0_block_simd_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_simd_function)
                .ok()?;
            let q8_0_block_simd_mr_function = library
                .get_function("q8_0_block_linear_row_simd_mr", None)
                .ok()?;
            let q8_0_block_simd_mr_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_simd_mr_function)
                .ok()?;
            let q8_0_block_simd_qmv4_function = library
                .get_function("q8_0_block_linear_row_simd_qmv4", None)
                .ok()?;
            let q8_0_block_simd_qmv4_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_simd_qmv4_function)
                .ok()?;
            let q8_0_block_ksplit_function = library
                .get_function("q8_0_block_linear_row_ksplit", None)
                .ok()?;
            let q8_0_block_ksplit_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_ksplit_function)
                .ok()?;
            let q8_0_block_ksplit_f32y_function = library
                .get_function("q8_0_block_linear_row_ksplit_f32y", None)
                .ok()?;
            let q8_0_block_ksplit_f32y_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_ksplit_f32y_function)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_function = library
                .get_function("q8_0_block_linear_row_ksplit_f32y_wire", None)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_ksplit_f32y_wire_function)
                .ok()?;
            let q4_0_block_ksplit_f32y_wire_function = library
                .get_function("q4_0_block_linear_row_ksplit_f32y_wire", None)
                .ok()?;
            let q4_0_block_ksplit_f32y_wire_pipeline = device
                .new_compute_pipeline_state_with_function(&q4_0_block_ksplit_f32y_wire_function)
                .ok()?;
            let nvfp4_block_ksplit_f32y_wire_function = library
                .get_function("nvfp4_block_linear_row_ksplit_f32y_wire", None)
                .ok()?;
            let nvfp4_block_ksplit_f32y_wire_pipeline = device
                .new_compute_pipeline_state_with_function(&nvfp4_block_ksplit_f32y_wire_function)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_nsg8_function = library
                .get_function("q8_0_block_linear_row_ksplit_f32y_wire_nsg8", None)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_nsg8_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &q8_0_block_ksplit_f32y_wire_nsg8_function,
                )
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_nsg8_verify_function = library
                .get_function("q8_0_block_linear_ksplit_f32y_wire_nsg8_verify", None)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_nsg8_verify_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &q8_0_block_ksplit_f32y_wire_nsg8_verify_function,
                )
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_gemm_function = library
                .get_function("q8_0_block_linear_ksplit_f32y_wire_gemm", None)
                .ok()?;
            let q8_0_block_ksplit_f32y_wire_gemm_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &q8_0_block_ksplit_f32y_wire_gemm_function,
                )
                .ok()?;
            let q8_0_block_wire_mm_function =
                library.get_function("q8_0_block_wire_mm", None).ok()?;
            let q8_0_block_wire_mm_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_wire_mm_function)
                .ok()?;
            let q8_0_block_wire_mm_f16o_function =
                library.get_function("q8_0_block_wire_mm_f16o", None).ok()?;
            let q8_0_block_wire_mm_f16o_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_wire_mm_f16o_function)
                .ok()?;
            let queue = device.new_command_queue();
            Some(MetalLinearKernel {
                device,
                queue,
                descriptor_pipeline,
                transposed_pipeline,
                q8_0_encoded_pipeline,
                q8_0_encoded_rows_pipeline,
                q8_0_block_pipeline,
                q8_0_block_simd_pipeline,
                q8_0_block_simd_mr_pipeline,
                q8_0_block_simd_qmv4_pipeline,
                q8_0_block_ksplit_pipeline,
                q8_0_block_ksplit_f32y_pipeline,
                q8_0_block_ksplit_f32y_wire_pipeline,
                q4_0_block_ksplit_f32y_wire_pipeline,
                nvfp4_block_ksplit_f32y_wire_pipeline,
                q8_0_block_ksplit_f32y_wire_nsg8_pipeline,
                q8_0_block_ksplit_f32y_wire_nsg8_verify_pipeline,
                q8_0_block_ksplit_f32y_wire_gemm_pipeline,
                q8_0_block_wire_mm_pipeline,
                q8_0_block_wire_mm_f16o_pipeline,
                rms_norm_pipeline,
                rms_norm_per_head_pipeline,
                residual_add_pipeline,
                silu_mul_pipeline,
                gelu_mul_pipeline,
                soft_cap_pipeline,
                scale_pipeline,
                rope_rotate_pipeline,
                attention_decode_pipeline,
                attention_decode_kv16_pipeline,
                attention_decode_v2_pipeline,
                attention_decode_v2_kv16_pipeline,
                quantize_q8_0_pipeline,
                kv_scatter_pipeline,
                kv_scatter_kv16_pipeline,
                f32_to_f16_pipeline,
                rms_norm_batch_pipeline,
                rms_norm_batch_f16o_pipeline,
                silu_mul_f16o_pipeline,
                rope_rotate_batch_pipeline,
                kv_scatter_batch_pipeline,
                attention_prefill_v3_pipeline,
                attention_prefill_flash_pipeline,
                half_mm_batched_pipeline,
                half_mm_batched_f16o_pipeline,
                transpose_v16_pipeline,
                rope_scatter_qh_pipeline,
                rope_scatter_qh_h_pipeline,
                rms_norm_batch_h_pipeline,
                residual_add_h_pipeline,
                silu_mul_h2_pipeline,
                softmax_causal_rows_pipeline,
                rms_norm_quantize_pipeline,
                silu_mul_quantize_pipeline,
                argmax_f32_greedy_pipeline,
                sample_gumbel_f32_pipeline,
                attention_decode_splitk_pipeline,
                attention_decode_splitk_kv16_pipeline,
                attention_decode_splitk_kv16_direct_pipeline,
                attention_splitk_kv16_stageonly_pipeline,
                attention_decode_splitk_merge_pipeline,
                attention_decode_f32_tree_pipeline,
                attention_decode_v2_tree_pipeline,
                attention_decode_splitk_kv16_tree_pipeline,
                attention_decode_splitk_kv16_direct_tree_pipeline,
                embed_row_gather_q8_wire_pipeline,
                active_command_buffer: Mutex::new(None),
                scratch_pool: Mutex::new(HashMap::new()),
            })
        })
        .as_ref()
}

#[cfg(target_os = "macos")]
fn metal_linear_cache() -> &'static Mutex<MetalLinearCache> {
    METAL_LINEAR_CACHE.get_or_init(|| Mutex::new(MetalLinearCache::new()))
}

#[cfg(target_os = "macos")]
static SESSION_ACTIVE: Mutex<bool> = Mutex::new(false);

#[cfg(target_os = "macos")]
pub fn start_inference_session() {
    let mut active = SESSION_ACTIVE.lock().unwrap();
    *active = true;
}

#[cfg(target_os = "macos")]
pub fn end_inference_session() {
    synchronize_active_session();
    let mut active = SESSION_ACTIVE.lock().unwrap();
    *active = false;
}

#[cfg(target_os = "macos")]
pub fn synchronize_active_session() {
    let Some(kernel) = metal_linear_kernel() else {
        return;
    };
    let cb_opt = {
        let mut active_cb = kernel.active_command_buffer.lock().unwrap();
        active_cb.take()
    };
    if let Some(cb) = cb_opt {
        cb.commit();
        cb.wait_until_completed();
    }

    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let deferred = std::mem::take(&mut cache.deferred_reads);
    for read in deferred {
        unsafe {
            let dest_slice =
                std::slice::from_raw_parts_mut(read.dest_ptr as *mut f32, read.dest_len);
            read_buffer_f32(&read.buffer, dest_slice);
        }
    }
    cache.scalar_index = 0;
}

#[cfg(target_os = "macos")]
fn get_active_or_new_command_buffer(kernel: &MetalLinearKernel) -> (metal::CommandBuffer, bool) {
    let session_active = !cfg!(test) && *SESSION_ACTIVE.lock().unwrap();
    if session_active {
        let mut active = kernel.active_command_buffer.lock().unwrap();
        if active.is_none() {
            *active = Some(kernel.queue.new_command_buffer().to_owned());
        }
        (active.as_ref().unwrap().to_owned(), true)
    } else {
        (kernel.queue.new_command_buffer().to_owned(), false)
    }
}

#[cfg(target_os = "macos")]
fn write_buffer_f32(buffer: &Buffer, values: &[f32]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_u8(buffer: &Buffer, values: &[u8]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_i8(buffer: &Buffer, values: &[i8]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_bytes<T>(buffer: &Buffer, values: &[T]) {
    let len = std::mem::size_of_val(values);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            len,
        );
    }
}

#[cfg(target_os = "macos")]
fn read_buffer_f32(buffer: &Buffer, out: &mut [f32]) {
    let len = std::mem::size_of_val(out);
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().cast::<u8>(),
            out.as_mut_ptr().cast::<u8>(),
            len,
        );
    }
}

#[cfg(target_os = "macos")]
fn read_buffer_i8(buffer: &Buffer, out: &mut [i8]) {
    let len = std::mem::size_of_val(out);
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().cast::<u8>(),
            out.as_mut_ptr().cast::<u8>(),
            len,
        );
    }
}

#[cfg(target_os = "macos")]
pub fn try_linear_row_f32(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> bool {
    try_linear_row_impl(input_row, weights, rows, cols, output, false)
}

#[cfg(target_os = "macos")]
pub fn try_linear_row_transposed_f32(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> bool {
    try_linear_row_impl(input_row, weights, rows, cols, output, true)
}

#[cfg(target_os = "macos")]
fn try_linear_row_impl(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
    transposed: bool,
) -> bool {
    if rows != input_row.len() || cols != output.len() || weights.len() != rows.saturating_mul(cols)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_buffer = cache.input_buffer(
        &kernel.device,
        std::mem::size_of_val(input_row),
        input_row.as_ptr(),
    );
    let weight_buffer = cache.weight_buffer(&kernel.device, weights);
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_buffer, input_row);
    write_buffer_f32(&output_buffer, output);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = rows as u32;
        *scalars.add(1) = cols as u32;
    }

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(if transposed {
        &kernel.transposed_pipeline
    } else {
        &kernel.descriptor_pipeline
    });
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&weight_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_buffer(3, Some(&scalar_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let pipeline = if transposed {
        &kernel.transposed_pipeline
    } else {
        &kernel.descriptor_pipeline
    };
    let width = pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (cols as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();

    if is_session {
        cache.deferred_reads.push(DeferredRead {
            buffer: output_buffer.clone(),
            dest_ptr: output.as_mut_ptr() as usize,
            dest_len: output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&output_buffer, output);
    }
    true
}

#[cfg(target_os = "macos")]
pub fn try_q8_0_encoded_linear_row(
    input_scales: &[f32],
    input_quants: &[i8],
    encoded_rows: &[u8],
    weight_scales: &[f32],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_ENCODED_BLOCK_BYTES: usize = 34;
    if rows == 0 || blocks_per_row == 0 || output.len() != rows {
        return false;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || encoded_rows.len()
            != rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_ENCODED_BLOCK_BYTES)
        || weight_scales.len() != rows.saturating_mul(blocks_per_row)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let encoded_rows_buffer = cache.q8_encoded_rows_buffer(
        &kernel.device,
        std::mem::size_of_val(encoded_rows),
        encoded_rows.as_ptr(),
    );
    let weight_scales_buffer = cache.q8_weight_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(weight_scales),
        weight_scales.as_ptr(),
    );
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    write_buffer_u8(&encoded_rows_buffer, encoded_rows);
    write_buffer_f32(&weight_scales_buffer, weight_scales);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_encoded_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&encoded_rows_buffer), 0);
    encoder.set_buffer(3, Some(&weight_scales_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), 0);
    encoder.set_buffer(6, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let width = kernel.q8_0_encoded_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();

    if is_session {
        cache.deferred_reads.push(DeferredRead {
            buffer: output_buffer.clone(),
            dest_ptr: output.as_mut_ptr() as usize,
            dest_len: output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&output_buffer, output);
    }
    true
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_q8_0_encoded_linear_rows(
    input_scales: &[f32],
    input_quants: &[i8],
    encoded_rows: &[u8],
    weight_scales: &[f32],
    input_rows: usize,
    weight_rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_ENCODED_BLOCK_BYTES: usize = 34;
    if input_rows == 0 || weight_rows == 0 || blocks_per_row == 0 {
        return false;
    }
    if output.len() != input_rows.saturating_mul(weight_rows)
        || input_scales.len() != input_rows.saturating_mul(blocks_per_row)
        || input_quants.len()
            != input_rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_BLOCK_VALUES)
        || encoded_rows.len()
            != weight_rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_ENCODED_BLOCK_BYTES)
        || weight_scales.len() != weight_rows.saturating_mul(blocks_per_row)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let encoded_rows_buffer = cache.q8_encoded_rows_buffer(
        &kernel.device,
        std::mem::size_of_val(encoded_rows),
        encoded_rows.as_ptr(),
    );
    let weight_scales_buffer = cache.q8_weight_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(weight_scales),
        weight_scales.as_ptr(),
    );
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 3 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    write_buffer_u8(&encoded_rows_buffer, encoded_rows);
    write_buffer_f32(&weight_scales_buffer, weight_scales);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = input_rows as u32;
        *scalars.add(2) = weight_rows as u32;
    }

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_encoded_rows_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&encoded_rows_buffer), 0);
    encoder.set_buffer(3, Some(&weight_scales_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), 0);
    encoder.set_buffer(6, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);
    encoder.set_buffer(
        7,
        Some(&scalar_buffer),
        (2 * std::mem::size_of::<u32>()) as u64,
    );

    let total = input_rows.saturating_mul(weight_rows);
    let width = kernel
        .q8_0_encoded_rows_pipeline
        .thread_execution_width()
        .max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (total as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();

    if is_session {
        cache.deferred_reads.push(DeferredRead {
            buffer: output_buffer.clone(),
            dest_ptr: output.as_mut_ptr() as usize,
            dest_len: output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&output_buffer, output);
    }
    true
}

#[cfg(target_os = "macos")]
pub fn try_q8_0_block_linear_row(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 36;
    if rows == 0 || blocks_per_row == 0 || output.len() != rows {
        return false;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || weight_blocks.len()
            != rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_BLOCK_BYTES)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let weight_blocks_buffer = cache.q8_block_weight_buffer(&kernel.device, weight_blocks);
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_block_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let width = kernel.q8_0_block_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();

    if is_session {
        cache.deferred_reads.push(DeferredRead {
            buffer: output_buffer.clone(),
            dest_ptr: output.as_mut_ptr() as usize,
            dest_len: output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&output_buffer, output);
    }
    true
}

/// Encode `dispatches` back-to-back Q8_0 GEMV dispatches into a SINGLE command
/// buffer (one commit + one wait), with a memory barrier between each so they
/// serialize like a real dependent forward pass. `simd` selects the
/// SIMD-group-per-row kernel. Returns the wall-clock seconds for the whole
/// batch, or None if Metal is unavailable.
///
/// Diagnostic for the GPU-port decision: comparing per-dispatch cost here against
/// the per-call cost of `try_q8_0_block_linear_row` isolates fixed commit/wait
/// round-trip overhead from actual GPU work.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)] // bench helper: explicit tensor/shape params are clearer than a struct here
pub fn bench_q8_0_block_linear_row_batched(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
    dispatches: usize,
    simd: bool,
) -> Option<f64> {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 36;
    if rows == 0 || blocks_per_row == 0 || dispatches == 0 || output.len() != rows {
        return None;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row * Q8_0_BLOCK_VALUES
        || weight_blocks.len() != rows * blocks_per_row * Q8_0_BLOCK_BYTES
    {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let mut cache = metal_linear_cache().lock().ok()?;
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let weight_blocks_buffer = cache.q8_block_weight_buffer(&kernel.device, weight_blocks);
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let pipeline = if simd {
        &kernel.q8_0_block_simd_pipeline
    } else {
        &kernel.q8_0_block_pipeline
    };
    let (threads_per_group, threadgroups) = if simd {
        (
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
        )
    } else {
        let width = pipeline.thread_execution_width().max(1);
        (
            metal::MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: (rows as u64).div_ceil(width),
                height: 1,
                depth: 1,
            },
        )
    };

    let started = std::time::Instant::now();
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);
    for i in 0..dispatches {
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        if i + 1 < dispatches {
            encoder.memory_barrier_with_resources(&[&output_buffer]);
        }
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let elapsed = started.elapsed().as_secs_f64();
    drop(cache);
    read_buffer_f32(&output_buffer, output);
    Some(elapsed)
}

#[cfg(target_os = "macos")]
pub fn try_q8_0_block_linear_row_with_cpu<F>(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
    cpu_work: F,
) -> bool
where
    F: FnOnce(),
{
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 36;
    if rows == 0 || blocks_per_row == 0 || output.len() != rows {
        return false;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || weight_blocks.len()
            != rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_BLOCK_BYTES)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let weight_blocks_buffer = cache.q8_block_weight_buffer(&kernel.device, weight_blocks);
    let output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(output),
        output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_block_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let width = kernel.q8_0_block_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();

    if is_session {
        cpu_work();
        cache.deferred_reads.push(DeferredRead {
            buffer: output_buffer.clone(),
            dest_ptr: output.as_mut_ptr() as usize,
            dest_len: output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        cpu_work();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&output_buffer, output);
    }
    true
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_q8_0_block_two_linear_rows_with_cpu<F>(
    input_scales: &[f32],
    input_quants: &[i8],
    first_weight_blocks: &[u8],
    second_weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    first_output: &mut [f32],
    second_output: &mut [f32],
    cpu_work: F,
) -> bool
where
    F: FnOnce(),
{
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 36;
    if rows == 0 || blocks_per_row == 0 || first_output.len() != rows || second_output.len() != rows
    {
        return false;
    }
    let expected_weight_bytes = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(Q8_0_BLOCK_BYTES);
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || first_weight_blocks.len() != expected_weight_bytes
        || second_weight_blocks.len() != expected_weight_bytes
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer = cache.q8_input_scales_buffer(
        &kernel.device,
        std::mem::size_of_val(input_scales),
        input_scales.as_ptr(),
    );
    let input_quants_buffer = cache.q8_input_quants_buffer(
        &kernel.device,
        std::mem::size_of_val(input_quants),
        input_quants.as_ptr(),
    );
    let first_weight_blocks_buffer =
        cache.q8_block_weight_buffer(&kernel.device, first_weight_blocks);
    let second_weight_blocks_buffer =
        cache.q8_block_weight_buffer(&kernel.device, second_weight_blocks);
    let first_output_buffer = cache.output_buffer(
        &kernel.device,
        std::mem::size_of_val(first_output),
        first_output.as_mut_ptr(),
    );
    let second_output_buffer = cache.aux_output_buffer(
        &kernel.device,
        std::mem::size_of_val(second_output),
        second_output.as_mut_ptr(),
    );
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let width = kernel.q8_0_block_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };

    let (command_buffer, is_session) = get_active_or_new_command_buffer(kernel);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_block_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    encoder.set_buffer(2, Some(&first_weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&first_output_buffer), 0);
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    encoder.set_buffer(2, Some(&second_weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&second_output_buffer), 0);
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    encoder.end_encoding();

    if is_session {
        cpu_work();
        cache.deferred_reads.push(DeferredRead {
            buffer: first_output_buffer.clone(),
            dest_ptr: first_output.as_mut_ptr() as usize,
            dest_len: first_output.len(),
        });
        cache.deferred_reads.push(DeferredRead {
            buffer: second_output_buffer.clone(),
            dest_ptr: second_output.as_mut_ptr() as usize,
            dest_len: second_output.len(),
        });
        drop(cache);
    } else {
        command_buffer.commit();
        cpu_work();
        command_buffer.wait_until_completed();
        drop(cache);
        read_buffer_f32(&first_output_buffer, first_output);
        read_buffer_f32(&second_output_buffer, second_output);
    }
    true
}

/// GPU RMSNorm of a single row: output = input / sqrt(mean(input^2) + eps) * weight.
/// Returns None if Metal is unavailable (caller falls back to CPU). Building block for
/// the GPU-resident forward pass; parity-checked against the CPU rms_norm.
#[cfg(target_os = "macos")]
pub fn try_rms_norm_f32(input: &[f32], weight: &[f32], eps: f32) -> Option<Vec<f32>> {
    if input.is_empty() || input.len() != weight.len() {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let width = input.len();
    let byte_len = std::mem::size_of_val(input) as u64;
    let in_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let weight_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let out_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let scalar_buf = kernel
        .device
        .new_buffer(8, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&in_buf, input);
    write_buffer_f32(&weight_buf, weight);
    unsafe {
        let p = scalar_buf.contents() as *mut u8;
        *(p as *mut u32) = width as u32;
        *(p.add(4) as *mut f32) = eps;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.rms_norm_pipeline);
    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&weight_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_buffer(3, Some(&scalar_buf), 0);
    encoder.set_buffer(4, Some(&scalar_buf), 4);
    let threads = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let one_group = metal::MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(one_group, threads);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; width];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// GPU per-head RMSNorm for Gemma QK-norm / weightless V-norm. `input` is
/// `head_count * head_dim`; each head's chunk is normalized independently with the
/// same reduction as `try_rms_norm_f32`. `weight` is `head_dim` (shared across
/// heads, e.g. q_norm/k_norm) or None for the weightless V-norm. None if Metal
/// unavailable or shapes mismatch.
#[cfg(target_os = "macos")]
pub fn try_rms_norm_per_head_f32(
    input: &[f32],
    weight: Option<&[f32]>,
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Option<Vec<f32>> {
    if head_dim == 0 || head_count == 0 || input.len() != head_count * head_dim {
        return None;
    }
    if let Some(w) = weight {
        if w.len() != head_dim {
            return None;
        }
    }
    let kernel = metal_linear_kernel()?;
    let byte_len = std::mem::size_of_val(input) as u64;
    let in_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let out_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    // Weight buffer is always bound; a dummy when weightless (kernel won't read it).
    let weight_buf = kernel
        .device
        .new_buffer((head_dim * 4) as u64, MTLResourceOptions::StorageModeShared);
    if let Some(w) = weight {
        write_buffer_f32(&weight_buf, w);
    }
    let scalar_buf = kernel
        .device
        .new_buffer(12, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&in_buf, input);
    unsafe {
        let p = scalar_buf.contents() as *mut u8;
        *(p as *mut u32) = head_dim as u32;
        *(p.add(4) as *mut f32) = eps;
        *(p.add(8) as *mut u32) = u32::from(weight.is_some());
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.rms_norm_per_head_pipeline);
    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&weight_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_buffer(3, Some(&scalar_buf), 0);
    encoder.set_buffer(4, Some(&scalar_buf), 4);
    encoder.set_buffer(5, Some(&scalar_buf), 8);
    let threads = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let groups = metal::MTLSize {
        width: head_count as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(groups, threads);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; input.len()];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// Standalone gemma4 f32-activation × wire-Q8 GEMV (builds buffers, one dispatch,
/// reads back). `weight_wire` is row-major 34-byte Q8_0 wire blocks
/// (`rows * blocks_per_row` of them); `y` is the f32 activation
/// (`blocks_per_row * 32`). This is the workhorse the gemma resident decode graph
/// runs 8× per layer, validated here against a CPU f32×dequant reference. None if
/// Metal is unavailable or shapes are invalid.
#[cfg(target_os = "macos")]
pub fn try_gemma4_q8_matmul_f32y(
    y: &[f32],
    weight_wire: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Option<Vec<f32>> {
    const WIRE: usize = 34;
    if rows == 0
        || blocks_per_row == 0
        || y.len() != blocks_per_row * 32
        || weight_wire.len() != rows * blocks_per_row * WIRE
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let y_buf = k.device.new_buffer(
        std::mem::size_of_val(y) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = k.device.new_buffer(
        weight_wire.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = k
        .device
        .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
    let scalar_buf = k
        .device
        .new_buffer(8, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&y_buf, y);
    write_buffer_u8(&w_buf, weight_wire);
    unsafe {
        let p = scalar_buf.contents() as *mut u32;
        *p = blocks_per_row as u32;
        *p.add(1) = rows as u32;
    }
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_q8_matmul(e, k, &y_buf, &w_buf, &out_buf, &scalar_buf, rows);
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; rows];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// Q4_0 wire GEMV on the GPU — the QAT-row counterpart of
/// [`try_gemma4_q8_matmul_f32y`]. `weight_wire` is `rows * blocks_per_row`
/// 18-byte Q4_0 blocks; `y` is `blocks_per_row * 32` f32 activations. Returns
/// `rows` f32 outputs (f32 activation x inline-dequantized Q4_0 weight). For
/// validating the Q4_0 GPU kernel against the CPU dequant reference.
#[cfg(target_os = "macos")]
pub fn try_gemma4_q4_0_matmul_f32y(
    y: &[f32],
    weight_wire: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Option<Vec<f32>> {
    const WIRE: usize = 18;
    if rows == 0
        || blocks_per_row == 0
        || y.len() != blocks_per_row * 32
        || weight_wire.len() != rows * blocks_per_row * WIRE
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let y_buf = k.device.new_buffer(
        std::mem::size_of_val(y) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = k.device.new_buffer(
        weight_wire.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = k
        .device
        .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
    let scalar_buf = k
        .device
        .new_buffer(8, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&y_buf, y);
    write_buffer_u8(&w_buf, weight_wire);
    unsafe {
        let p = scalar_buf.contents() as *mut u32;
        *p = blocks_per_row as u32;
        *p.add(1) = rows as u32;
    }
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_q4_0_matmul(e, k, &y_buf, &w_buf, &out_buf, &scalar_buf, rows);
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; rows];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// NVFP4 wire GEMV on the GPU (GABBRO M3) — the NVFP4 counterpart of
/// [`try_gemma4_q4_0_matmul_f32y`]. `weight_wire` is `rows * blocks_per_row`
/// 36-byte NVFP4 superblocks; `y` is `blocks_per_row * 64` f32 activations (the
/// superblock is 64 values, unlike the 32-value Q8_0/Q4_0 blocks). Returns `rows`
/// f32 outputs (f32 activation × inline-decoded NVFP4 weight). For validating the
/// NVFP4 GPU kernel against the CPU `nvfp4_wire_block_dequant` reference.
#[cfg(target_os = "macos")]
pub fn try_gemma4_nvfp4_matmul_f32y(
    y: &[f32],
    weight_wire: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Option<Vec<f32>> {
    const WIRE: usize = 36;
    // The NVFP4 superblock is 64 values (block_elements), not 32 — the single
    // point where an NVFP4 row's activation stride differs from Q8_0/Q4_0.
    if rows == 0
        || blocks_per_row == 0
        || y.len() != blocks_per_row * GemmaWireFmt::Nvfp4.block_elements()
        || weight_wire.len() != rows * blocks_per_row * WIRE
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let y_buf = k.device.new_buffer(
        std::mem::size_of_val(y) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = k.device.new_buffer(
        weight_wire.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = k
        .device
        .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
    let scalar_buf = k
        .device
        .new_buffer(8, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&y_buf, y);
    write_buffer_u8(&w_buf, weight_wire);
    unsafe {
        let p = scalar_buf.contents() as *mut u32;
        *p = blocks_per_row as u32;
        *p.add(1) = rows as u32;
    }
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_nvfp4_matmul(e, k, &y_buf, &w_buf, &out_buf, &scalar_buf, rows);
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; rows];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// Standalone gemma4 FFN sub-block: builds buffers, runs the whole
/// [`encode_gemma4_ffn`] chain in one command buffer, reads back the hidden output.
/// `gate`/`up` are `ffn_dim` rows over `hidden` in; `down` is `hidden` rows over
/// `ffn_dim` in; weights are row-major 34-byte Q8 wire blocks. For validating the
/// FFN sub-graph against CPU. None if Metal unavailable or shapes are invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_ffn(
    fmt: GemmaWireFmt,
    h_in: &[f32],
    ffn_norm: &[f32],
    post_ffw_norm: &[f32],
    eps: f32,
    gate_wire: &[u8],
    up_wire: &[u8],
    down_wire: &[u8],
    ffn_dim: usize,
) -> Option<Vec<f32>> {
    let wire: usize = fmt.wire_bytes();
    let hidden = h_in.len();
    if hidden == 0
        || ffn_dim == 0
        || !hidden.is_multiple_of(32)
        || !ffn_dim.is_multiple_of(32)
        || ffn_norm.len() != hidden
        || post_ffw_norm.len() != hidden
    {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_ffn = ffn_dim / 32;
    if gate_wire.len() != ffn_dim * bpr_hidden * wire
        || up_wire.len() != ffn_dim * bpr_hidden * wire
        || down_wire.len() != hidden * bpr_ffn * wire
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let mkbuf = |bytes: usize| {
        k.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = mkbuf(hidden * 4);
    let out_buf = mkbuf(hidden * 4);
    let gate_buf = mkbuf(gate_wire.len());
    let up_buf = mkbuf(up_wire.len());
    let down_buf = mkbuf(down_wire.len());
    write_buffer_f32(&in_buf, h_in);
    write_buffer_u8(&gate_buf, gate_wire);
    write_buffer_u8(&up_buf, up_wire);
    write_buffer_u8(&down_buf, down_wire);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_ffn(
        fmt,
        e,
        k,
        &mut keep,
        &in_buf,
        &out_buf,
        ffn_norm,
        post_ffw_norm,
        eps,
        &gate_buf,
        &up_buf,
        &down_buf,
        ffn_dim,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    drop(keep);
    Some(out)
}

/// Standalone gemma4 attention sub-block: builds buffers (incl. a caller-prefilled
/// KV cache laid out `[kv_head][max_positions][head_dim]`), runs the whole
/// [`encode_gemma4_attention`] chain in one command buffer, reads back the hidden
/// output. For validating the attention sub-graph against CPU. The current token's
/// K/V are scattered into the cache at `write_position`; attention covers the window
/// `[window_start .. filled)`. None if Metal unavailable or shapes invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_attention(
    fmt: GemmaWireFmt,
    h_in: &[f32],
    attn_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    post_attn_norm: &[f32],
    eps: f32,
    q_wire: &[u8],
    k_wire: &[u8],
    v_wire: Option<&[u8]>,
    o_wire: &[u8],
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k_init: &[f32],
    cache_v_init: &[f32],
    max_positions: usize,
    write_position: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    filled: usize,
    window_start: usize,
    scale: f32,
    owns_kv: bool,
) -> Option<Vec<f32>> {
    let wire = fmt.wire_bytes();
    let hidden = h_in.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let cache_len = n_kv_heads * max_positions * head_dim;
    if hidden == 0
        || !hidden.is_multiple_of(32)
        || !head_dim.is_multiple_of(2)
        || n_kv_heads == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || attn_norm.len() != hidden
        || post_attn_norm.len() != hidden
        || q_norm.len() != head_dim
        || k_norm.len() != head_dim
        || cos_t.len() != head_dim / 2
        || sin_t.len() != head_dim / 2
        || write_position >= filled
        || filled > max_positions
        || window_start >= filled
        || cache_k_init.len() != cache_len
        || cache_v_init.len() != cache_len
    {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_q = q_dim / 32;
    if q_wire.len() != q_dim * bpr_hidden * wire
        || k_wire.len() != kv_dim * bpr_hidden * wire
        || v_wire.is_some_and(|w| w.len() != kv_dim * bpr_hidden * wire)
        || o_wire.len() != hidden * bpr_q * wire
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let mkbuf = |bytes: usize| {
        k.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = mkbuf(hidden * 4);
    let out_buf = mkbuf(hidden * 4);
    let q_buf = mkbuf(q_wire.len());
    let kw_buf = mkbuf(k_wire.len());
    let vw_buf = v_wire.map(|w| mkbuf(w.len()));
    let o_buf = mkbuf(o_wire.len());
    let cache_k = mkbuf(cache_len * 4);
    let cache_v = mkbuf(cache_len * 4);
    write_buffer_f32(&in_buf, h_in);
    write_buffer_u8(&q_buf, q_wire);
    write_buffer_u8(&kw_buf, k_wire);
    if let (Some(buf), Some(wire)) = (&vw_buf, v_wire) {
        write_buffer_u8(buf, wire);
    }
    write_buffer_u8(&o_buf, o_wire);
    write_buffer_f32(&cache_k, cache_k_init);
    write_buffer_f32(&cache_v, cache_v_init);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_attention(
        fmt,
        e,
        k,
        &mut keep,
        &in_buf,
        &out_buf,
        attn_norm,
        q_norm,
        k_norm,
        post_attn_norm,
        eps,
        &q_buf,
        &kw_buf,
        vw_buf.as_ref(),
        &o_buf,
        cos_t,
        sin_t,
        &cache_k,
        &cache_v,
        max_positions,
        write_position,
        n_heads,
        n_kv_heads,
        head_dim,
        filled,
        window_start,
        scale,
        owns_kv,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    drop(keep);
    Some(out)
}

/// Standalone gemma4 PLE per-layer injection: builds buffers, runs
/// [`encode_gemma4_ple`] in one command buffer, reads back the updated hidden.
/// `pli_l` is this layer's per-token Per-Layer-Embedding input (CPU-computed);
/// `ple_inp_gate` (output-major `[ple_dim][hidden]`) and `ple_proj`
/// (`[hidden][ple_dim]`) are the f32 PLE matrices. None if Metal unavailable or
/// shapes invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_ple(
    h_in: &[f32],
    pli_l: &[f32],
    ple_inp_gate: &[f32],
    ple_proj: &[f32],
    post_norm: &[f32],
    output_scale: f32,
    eps: f32,
    ple_dim: usize,
) -> Option<Vec<f32>> {
    let hidden = h_in.len();
    if hidden == 0
        || ple_dim == 0
        || pli_l.len() != ple_dim
        || ple_inp_gate.len() != ple_dim * hidden
        || ple_proj.len() != hidden * ple_dim
        || post_norm.len() != hidden
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let mkf = |v: &[f32]| {
        let b = k
            .device
            .new_buffer((v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        write_buffer_f32(&b, v);
        b
    };
    let h_buf = mkf(h_in);
    let pli_buf = mkf(pli_l);
    let ig = mkf(ple_inp_gate);
    let pj = mkf(ple_proj);
    let pn = mkf(post_norm);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_ple(
        e,
        k,
        &mut keep,
        &h_buf,
        &pli_buf,
        0,
        &ig,
        &pj,
        &pn,
        output_scale,
        eps,
        hidden,
        ple_dim,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&h_buf, &mut out);
    drop(keep);
    Some(out)
}

/// Standalone gemma4 per-token `pli` on the GPU (the folded
/// [`encode_gemma4_pli`] path). Takes the RAW gemma tensors and folds the constants
/// internally. `proj` is `[ple_total][hidden]` (per_layer_model_proj), `proj_norm`
/// is `[ple_dim]`, `ti` is the `[ple_total]` per_layer_token_embd row for the token.
/// Returns `pli` `[ple_total]`. For validating the GPU pli vs the CPU prep.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_pli(
    h0: &[f32],
    proj: &[f32],
    proj_norm: &[f32],
    ti: &[f32],
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
    eps: f32,
) -> Option<Vec<f32>> {
    let ple_total = n_layers * ple_dim;
    if h0.len() != hidden
        || proj.len() != ple_total * hidden
        || proj_norm.len() != ple_dim
        || ti.len() != ple_total
    {
        return None;
    }
    let frac = std::f32::consts::FRAC_1_SQRT_2;
    let proj_scale = (hidden as f32).powf(-0.5);
    let embed_scale = (ple_dim as f32).sqrt();
    // Fold the constants into the resident inputs.
    let proj_folded: Vec<f32> = proj.iter().map(|v| v * proj_scale).collect();
    let projnorm_folded: Vec<f32> = proj_norm.iter().map(|v| v * frac).collect();
    let ti_folded: Vec<f32> = ti.iter().map(|v| v * embed_scale * frac).collect();
    let k = metal_linear_kernel()?;
    let mkf = |v: &[f32]| {
        let b = k
            .device
            .new_buffer((v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        write_buffer_f32(&b, v);
        b
    };
    let h0_buf = mkf(h0);
    let proj_buf = mkf(&proj_folded);
    let projnorm_buf = mkf(&projnorm_folded);
    let ti_buf = mkf(&ti_folded);
    let z = vec![0.0f32; ple_total];
    let ctx_buf = mkf(&z);
    let ctx_n_buf = mkf(&z);
    let pli_buf = mkf(&z);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_pli(
        e,
        k,
        &mut keep,
        &h0_buf,
        &proj_buf,
        &projnorm_buf,
        &ti_buf,
        &ctx_buf,
        &ctx_n_buf,
        &pli_buf,
        hidden,
        ple_total,
        ple_dim,
        n_layers,
        eps,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; ple_total];
    read_buffer_f32(&pli_buf, &mut out);
    drop(keep);
    Some(out)
}

/// Standalone gemma4 logits head: builds buffers, runs [`encode_gemma4_head`] in one
/// command buffer, reads back the `vocab` soft-capped logits. `token_embd_wire` is
/// the vocab-major Q8 embedding table in 34-byte wire blocks (`vocab * hidden/32`
/// blocks). None if Metal unavailable or shapes invalid.
#[cfg(target_os = "macos")]
pub fn try_gemma4_head(
    h_in: &[f32],
    output_norm: &[f32],
    token_embd_wire: &[u8],
    vocab: usize,
    softcap: f32,
    eps: f32,
) -> Option<Vec<f32>> {
    const WIRE: usize = 34;
    let hidden = h_in.len();
    if hidden == 0 || !hidden.is_multiple_of(32) || vocab == 0 || output_norm.len() != hidden {
        return None;
    }
    let bpr_hidden = hidden / 32;
    if token_embd_wire.len() != vocab * bpr_hidden * WIRE {
        return None;
    }
    let k = metal_linear_kernel()?;
    let h_buf = k
        .device
        .new_buffer((hidden * 4) as u64, MTLResourceOptions::StorageModeShared);
    let logits_buf = k
        .device
        .new_buffer((vocab * 4) as u64, MTLResourceOptions::StorageModeShared);
    let embd_buf = k.device.new_buffer(
        token_embd_wire.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    write_buffer_f32(&h_buf, h_in);
    write_buffer_u8(&embd_buf, token_embd_wire);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_head(
        e,
        k,
        &mut keep,
        &h_buf,
        &logits_buf,
        output_norm,
        &embd_buf,
        vocab,
        softcap,
        eps,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; vocab];
    read_buffer_f32(&logits_buf, &mut out);
    drop(keep);
    Some(out)
}

/// A resident gemma4 model ready to decode: all weights (layers + tied `token_embd`)
/// stay GPU-resident, the per-owning-layer KV caches PERSIST across tokens (allocated
/// once, scattered one slot per token), and the hidden/logits buffers are reused. Each
/// [`Gemma4ResidentModel::forward_token`] runs the whole token graph in ONE command
/// buffer with no per-token weight copy — unlike [`try_gemma4_forward`] (a stateless
/// test helper that rebuilds the cache and re-copies the 0.7GB embedding every call).
/// Resident state for computing the per-token PLE input `pli` ON the GPU (replacing
/// the ~12ms/token CPU prep — a 110MB f32 matvec). `proj`/`projnorm` have the gemma
/// constants folded in (proj * hidden^-0.5, proj_norm * FRAC_1_SQRT_2); `ti` is
/// written per token (the per_layer_token_embd row * ple_dim^0.5 * FRAC_1_SQRT_2).
#[cfg(target_os = "macos")]
struct Gemma4PliResident {
    proj: Buffer,
    projnorm: Buffer,
    ti: Buffer,
    ctx: Buffer,
    ctx_n: Buffer,
    pli: Buffer,
    ple_total: usize,
    ple_dim: usize,
}

#[cfg(target_os = "macos")]
pub struct Gemma4ResidentModel {
    layers: Vec<Gemma4ResidentLayer>,
    ple: Vec<Option<Gemma4ResidentPle>>,
    /// Per-layer `layer_output_scale` (1.0 when the tensor is absent). E-series
    /// layers apply it inside the PLE encode (`Gemma4ResidentPle.output_scale`,
    /// same tensor); dense rows (12B-class, no PLE) apply it standalone at the
    /// end of the layer — the reference multiplies it UNCONDITIONALLY.
    layer_scales: Vec<f32>,
    /// Per-layer resident PLE matrix buffers (inp_gate, proj, post_norm), uploaded
    /// once so forward_token doesn't re-copy ~220MB of f32 matrices every token.
    ple_bufs: Vec<Option<(Buffer, Buffer, Buffer)>>,
    /// Resident PLE `pli` computation (set via `set_pli`); when present, forward_token
    /// computes `pli` on the GPU from the input embedding instead of on the CPU.
    pli_res: Option<Gemma4PliResident>,
    owns_kv: Vec<bool>,
    kv_source: Vec<usize>,
    caches: Vec<Option<(Buffer, Buffer)>>,
    token_embd: Buffer,
    output_norm: Vec<f32>,
    buf_a: Buffer,
    buf_b: Buffer,
    mid: Buffer,
    logits: Buffer,
    hidden: usize,
    vocab: usize,
    softcap: f32,
    eps: f32,
    max_positions: usize,
    scale: f32,
}

#[cfg(target_os = "macos")]
impl Gemma4ResidentModel {
    /// Build the resident model. `token_embd_wire` is the vocab-major Q8 table in
    /// 34-byte wire blocks for the GPU logits head (copied once into a resident buffer;
    /// in the production runtime pass a nocopy WirePages buffer instead). Pass an EMPTY
    /// slice on the QAT hybrid lane (Q6_K head runs on the CPU) — no head buffer is
    /// allocated and only `forward_token_hidden` is valid. KV caches are allocated for
    /// owning layers (`owns_kv[l]`) sized to that layer's head_dim. None on bad shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: Vec<Gemma4ResidentLayer>,
        ple: Vec<Option<Gemma4ResidentPle>>,
        layer_scales: Vec<f32>,
        owns_kv: Vec<bool>,
        kv_source: Vec<usize>,
        token_embd_wire: &[u8],
        output_norm: Vec<f32>,
        hidden: usize,
        vocab: usize,
        softcap: f32,
        eps: f32,
        max_positions: usize,
        scale: f32,
    ) -> Option<Self> {
        let n = layers.len();
        if n == 0
            || ple.len() != n
            || layer_scales.len() != n
            || owns_kv.len() != n
            || kv_source.len() != n
            || output_norm.len() != hidden
            || vocab == 0
            || max_positions == 0
        {
            return None;
        }
        let k = metal_linear_kernel()?;
        let mkbuf = |bytes: usize| {
            k.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let mut caches = Vec::with_capacity(n);
        for (l, layer) in layers.iter().enumerate() {
            if owns_kv[l] {
                let len = layer.n_kv_heads * max_positions * layer.head_dim;
                caches.push(Some((mkbuf(len * 4), mkbuf(len * 4))));
            } else {
                caches.push(None);
            }
        }
        // Empty wire = QAT hybrid lane: the Q6_K head runs on the CPU, so don't upload
        // the (large) tied table to the GPU. A 1-byte placeholder keeps the field valid;
        // forward_inner only reads it on the want_head path, which the hybrid never takes.
        let token_embd = mkbuf(token_embd_wire.len().max(1));
        if !token_embd_wire.is_empty() {
            write_buffer_u8(&token_embd, token_embd_wire);
        }
        // Upload each layer's PLE matrices once into resident buffers.
        let upload = |v: &[f32]| {
            let b = mkbuf(v.len() * 4);
            write_buffer_f32(&b, v);
            b
        };
        let ple_bufs: Vec<Option<(Buffer, Buffer, Buffer)>> = ple
            .iter()
            .map(|p| {
                p.as_ref()
                    .map(|p| (upload(&p.inp_gate), upload(&p.proj), upload(&p.post_norm)))
            })
            .collect();
        Some(Self {
            layers,
            ple,
            layer_scales,
            ple_bufs,
            pli_res: None,
            owns_kv,
            kv_source,
            caches,
            token_embd,
            output_norm,
            buf_a: mkbuf(hidden * 4),
            buf_b: mkbuf(hidden * 4),
            mid: mkbuf(hidden * 4),
            logits: mkbuf(vocab * 4),
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            scale,
        })
    }

    /// Enable on-GPU `pli` computation: uploads the (constant-folded) PLE projection
    /// matrices once. After this, `forward_token` computes `pli` from the input
    /// embedding on the GPU (passing `ti` per token) instead of relying on CPU prep.
    /// `proj` = per_layer_model_proj `[ple_total][hidden]`, `proj_norm` = `[ple_dim]`.
    pub fn set_pli(&mut self, proj: &[f32], proj_norm: &[f32], ple_dim: usize) -> bool {
        let n_layers = self.layers.len();
        let ple_total = n_layers * ple_dim;
        if proj.len() != ple_total * self.hidden || proj_norm.len() != ple_dim {
            return false;
        }
        let Some(k) = metal_linear_kernel() else {
            return false;
        };
        let frac = std::f32::consts::FRAC_1_SQRT_2;
        let proj_scale = (self.hidden as f32).powf(-0.5);
        let proj_folded: Vec<f32> = proj.iter().map(|v| v * proj_scale).collect();
        let projnorm_folded: Vec<f32> = proj_norm.iter().map(|v| v * frac).collect();
        let up = |v: &[f32]| {
            let b = k
                .device
                .new_buffer((v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
            write_buffer_f32(&b, v);
            b
        };
        let alloc = |n: usize| {
            k.device
                .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
        };
        self.pli_res = Some(Gemma4PliResident {
            proj: up(&proj_folded),
            projnorm: up(&projnorm_folded),
            ti: alloc(ple_total),
            ctx: alloc(ple_total),
            ctx_n: alloc(ple_total),
            pli: alloc(ple_total),
            ple_total,
            ple_dim,
        });
        true
    }

    /// Decode one token at absolute `position`: writes `h0` (the position's scaled
    /// input embedding) into the resident hidden buffer, runs all layers (each scatters
    /// its K/V into the PERSISTENT cache at `position`) + PLE + head in ONE command
    /// buffer, and returns the `vocab` soft-capped logits. `inputs[l]` carries this
    /// layer's RoPE tables, `pli`, and window start (CPU-computed for `position`).
    pub fn forward_token(
        &self,
        h0: &[f32],
        inputs: &[Gemma4TokenLayerInput],
        ti: &[f32],
        position: usize,
    ) -> Option<Vec<f32>> {
        self.forward_inner(h0, inputs, ti, position, true)
    }

    /// Like [`Self::forward_token`] but stops BEFORE the GPU logits head and returns the
    /// final decoder hidden state (the head's input). The QAT hybrid lane uses this: the
    /// tied head is Q6_K, which has no GPU kernel, so the head runs on the CPU instead.
    pub fn forward_token_hidden(
        &self,
        h0: &[f32],
        inputs: &[Gemma4TokenLayerInput],
        ti: &[f32],
        position: usize,
    ) -> Option<Vec<f32>> {
        self.forward_inner(h0, inputs, ti, position, false)
    }

    #[allow(clippy::needless_range_loop)] // layer index `l` indexes several parallel arrays
    fn forward_inner(
        &self,
        h0: &[f32],
        inputs: &[Gemma4TokenLayerInput],
        ti: &[f32],
        position: usize,
        want_head: bool,
    ) -> Option<Vec<f32>> {
        let n = self.layers.len();
        if h0.len() != self.hidden || inputs.len() != n || position >= self.max_positions {
            return None;
        }
        let k = metal_linear_kernel()?;
        let filled = position + 1;
        write_buffer_f32(&self.buf_a, h0);
        let mut keep = Vec::new();
        let cb = k.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        // Compute this token's pli on the GPU (folded constants) before the layers.
        if let Some(p) = &self.pli_res {
            if ti.len() != p.ple_total {
                return None;
            }
            write_buffer_f32(&p.ti, ti);
            encode_gemma4_pli(
                e,
                k,
                &mut keep,
                &self.buf_a,
                &p.proj,
                &p.projnorm,
                &p.ti,
                &p.ctx,
                &p.ctx_n,
                &p.pli,
                self.hidden,
                p.ple_total,
                p.ple_dim,
                n,
                self.eps,
            );
        }
        let mut from_a = true;
        for l in 0..n {
            let (in_buf, out_buf) = if from_a {
                (&self.buf_a, &self.buf_b)
            } else {
                (&self.buf_b, &self.buf_a)
            };
            let src = if self.owns_kv[l] {
                l
            } else {
                self.kv_source[l]
            };
            let (ck, cv) = self.caches[src].as_ref()?;
            let inp = &inputs[l];
            encode_gemma4_layer(
                e,
                k,
                &mut keep,
                &self.layers[l],
                in_buf,
                &self.mid,
                out_buf,
                &inp.cos_t,
                &inp.sin_t,
                ck,
                cv,
                self.max_positions,
                position,
                filled,
                inp.window_start,
                self.scale,
                self.owns_kv[l],
            );
            if let (Some(p), Some((ig, pj, pn)), Some(pr)) =
                (&self.ple[l], &self.ple_bufs[l], &self.pli_res)
            {
                encode_gemma4_ple(
                    e,
                    k,
                    &mut keep,
                    out_buf,
                    &pr.pli,
                    l * pr.ple_dim,
                    ig,
                    pj,
                    pn,
                    p.output_scale,
                    self.eps,
                    self.hidden,
                    pr.ple_dim,
                );
            } else if self.layer_scales[l] != 1.0 {
                // Dense rows (no PLE) still carry layer_output_scale — the
                // reference multiplies the layer output unconditionally.
                let sc = pool_get(k, 8);
                unsafe {
                    let p = sc.contents() as *mut u8;
                    *(p as *mut u32) = self.hidden as u32;
                    *(p.add(4) as *mut f32) = self.layer_scales[l];
                }
                encode_scale_f32(e, k, out_buf, out_buf, &sc, self.hidden);
                keep.push(sc);
            }
            from_a = !from_a;
        }
        let final_buf = if from_a { &self.buf_a } else { &self.buf_b };
        // QAT hybrid (`want_head == false`): skip the GPU head and read the final hidden
        // back so the CPU can run the Q6_K tied head. Otherwise encode the Q8 head and
        // read the soft-capped logits.
        if want_head {
            encode_gemma4_head(
                e,
                k,
                &mut keep,
                final_buf,
                &self.logits,
                &self.output_norm,
                &self.token_embd,
                self.vocab,
                self.softcap,
                self.eps,
            );
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let out = if want_head {
            let mut out = vec![0.0f32; self.vocab];
            read_buffer_f32(&self.logits, &mut out);
            out
        } else {
            let mut h = vec![0.0f32; self.hidden];
            read_buffer_f32(final_buf, &mut h);
            h
        };
        // Return the per-token scratch to the pool so the next token reuses it instead
        // of allocating ~hundreds of fresh Metal buffers (safe: the command buffer has
        // completed). Persistent weights/caches/token_embd are NOT in `keep`.
        pool_recycle(k, keep);
        Some(out)
    }
}

/// One layer's f32 PLE matrices (E-series only; `None` per layer for dense models).
/// Output-major like the CPU `f32_matvec`: `inp_gate` is `[ple_dim][hidden]`, `proj`
/// is `[hidden][ple_dim]`, `post_norm` is `[hidden]`.
pub struct Gemma4ResidentPle {
    pub inp_gate: Vec<f32>,
    pub proj: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub output_scale: f32,
}

/// Per-token CPU-computed inputs the GPU forward needs for one layer: the dual-θ
/// RoPE tables for this position, the PLE per-layer input `pli`, and the resolved
/// attention window start (`max(0, filled-window)` for sliding, 0 for global).
pub struct Gemma4TokenLayerInput {
    pub cos_t: Vec<f32>,
    pub sin_t: Vec<f32>,
    pub pli: Vec<f32>,
    pub window_start: usize,
}

/// Drive a full gemma4 token forward on the GPU in ONE command buffer:
/// `h0` (the position's scaled input embedding) → N decoder layers (each
/// [`encode_gemma4_layer`] + optional [`encode_gemma4_ple`]) → [`encode_gemma4_head`]
/// → soft-capped logits. Cross-layer KV sharing: owning layers (`owns_kv[l]`) hold
/// their own cache (built from `cache_k_init[l]`/`cache_v_init[l]`); shared layers
/// read `kv_source[l]`'s cache. The caller computes per-token data (embedding,
/// RoPE tables, `pli`, window starts) on the CPU — see [`Gemma4TokenLayerInput`].
/// Returns the `vocab` logits. None if Metal unavailable or shapes invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_forward(
    layers: &[Gemma4ResidentLayer],
    ple: &[Option<Gemma4ResidentPle>],
    owns_kv: &[bool],
    kv_source: &[usize],
    inputs: &[Gemma4TokenLayerInput],
    cache_k_init: &[Option<Vec<f32>>],
    cache_v_init: &[Option<Vec<f32>>],
    h0: &[f32],
    output_norm: &[f32],
    token_embd_wire: &[u8],
    vocab: usize,
    softcap: f32,
    eps: f32,
    max_positions: usize,
    write_position: usize,
    filled: usize,
    scale: f32,
) -> Option<Vec<f32>> {
    let n = layers.len();
    let hidden = h0.len();
    if n == 0
        || hidden == 0
        || ple.len() != n
        || owns_kv.len() != n
        || kv_source.len() != n
        || inputs.len() != n
        || cache_k_init.len() != n
        || cache_v_init.len() != n
        || output_norm.len() != hidden
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let mkbuf = |bytes: usize| {
        k.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    };
    // Build a cache buffer pair for each owning layer; shared layers reuse a source's.
    let mut caches: Vec<Option<(Buffer, Buffer)>> = Vec::with_capacity(n);
    for l in 0..n {
        if owns_kv[l] {
            let ck = cache_k_init[l].as_ref()?;
            let cv = cache_v_init[l].as_ref()?;
            let len = layers[l].n_kv_heads * max_positions * layers[l].head_dim;
            if ck.len() != len || cv.len() != len {
                return None;
            }
            let ckb = mkbuf(len * 4);
            let cvb = mkbuf(len * 4);
            write_buffer_f32(&ckb, ck);
            write_buffer_f32(&cvb, cv);
            caches.push(Some((ckb, cvb)));
        } else {
            caches.push(None);
        }
    }
    let buf_a = mkbuf(hidden * 4);
    let buf_b = mkbuf(hidden * 4);
    let mid = mkbuf(hidden * 4);
    let logits_buf = mkbuf(vocab * 4);
    let embd_buf = mkbuf(token_embd_wire.len());
    write_buffer_f32(&buf_a, h0);
    write_buffer_u8(&embd_buf, token_embd_wire);

    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    let mut from_a = true;
    for l in 0..n {
        let (in_buf, out_buf) = if from_a {
            (&buf_a, &buf_b)
        } else {
            (&buf_b, &buf_a)
        };
        let src = if owns_kv[l] { l } else { kv_source[l] };
        let (ck, cv) = caches[src].as_ref()?;
        let inp = &inputs[l];
        encode_gemma4_layer(
            e,
            k,
            &mut keep,
            &layers[l],
            in_buf,
            &mid,
            out_buf,
            &inp.cos_t,
            &inp.sin_t,
            ck,
            cv,
            max_positions,
            write_position,
            filled,
            inp.window_start,
            scale,
            owns_kv[l],
        );
        if let Some(p) = &ple[l] {
            let mkf = |v: &[f32]| {
                let b = k
                    .device
                    .new_buffer((v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
                write_buffer_f32(&b, v);
                b
            };
            let (ig, pj, pn) = (mkf(&p.inp_gate), mkf(&p.proj), mkf(&p.post_norm));
            let pli_buf = mkf(&inp.pli);
            encode_gemma4_ple(
                e,
                k,
                &mut keep,
                out_buf,
                &pli_buf,
                0,
                &ig,
                &pj,
                &pn,
                p.output_scale,
                eps,
                hidden,
                inp.pli.len(),
            );
            keep.push(pli_buf);
            keep.extend([ig, pj, pn]);
        }
        from_a = !from_a;
    }
    // After n layers the final hidden is the last out_buf (buf_b iff n is odd).
    let final_buf = if from_a { &buf_a } else { &buf_b };
    encode_gemma4_head(
        e,
        k,
        &mut keep,
        final_buf,
        &logits_buf,
        output_norm,
        &embd_buf,
        vocab,
        softcap,
        eps,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; vocab];
    read_buffer_f32(&logits_buf, &mut out);
    drop(keep);
    Some(out)
}

/// Standalone full gemma4 layer: builds buffers (incl. a prefilled cache), runs
/// [`encode_gemma4_layer`] (attention → FFN) in one command buffer, reads back the
/// hidden output. For validating the full-layer chain against CPU. None if Metal
/// unavailable or shapes invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_layer(
    layer: &Gemma4ResidentLayer,
    h_in: &[f32],
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k_init: &[f32],
    cache_v_init: &[f32],
    max_positions: usize,
    write_position: usize,
    filled: usize,
    window_start: usize,
    scale: f32,
    owns_kv: bool,
) -> Option<Vec<f32>> {
    let hidden = h_in.len();
    let cache_len = layer.n_kv_heads * max_positions * layer.head_dim;
    if hidden == 0 || cache_k_init.len() != cache_len || cache_v_init.len() != cache_len {
        return None;
    }
    let k = metal_linear_kernel()?;
    let mkbuf = |bytes: usize| {
        k.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = mkbuf(hidden * 4);
    let mid_buf = mkbuf(hidden * 4);
    let out_buf = mkbuf(hidden * 4);
    let cache_k = mkbuf(cache_len * 4);
    let cache_v = mkbuf(cache_len * 4);
    write_buffer_f32(&in_buf, h_in);
    write_buffer_f32(&cache_k, cache_k_init);
    write_buffer_f32(&cache_v, cache_v_init);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_gemma4_layer(
        e,
        k,
        &mut keep,
        layer,
        &in_buf,
        &mid_buf,
        &out_buf,
        cos_t,
        sin_t,
        &cache_k,
        &cache_v,
        max_positions,
        write_position,
        filled,
        window_start,
        scale,
        owns_kv,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    drop(keep);
    Some(out)
}

/// GPU elementwise binary op helper for residual add / silu-mul (same buffer shape).
#[cfg(target_os = "macos")]
fn try_binary_elementwise_f32(
    pipeline: &ComputePipelineState,
    lhs: &[f32],
    rhs: &[f32],
) -> Option<Vec<f32>> {
    if lhs.is_empty() || lhs.len() != rhs.len() {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let n = lhs.len();
    let byte_len = std::mem::size_of_val(lhs) as u64;
    let lhs_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let rhs_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let out_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let n_buf = kernel
        .device
        .new_buffer(4, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&lhs_buf, lhs);
    write_buffer_f32(&rhs_buf, rhs);
    unsafe {
        *(n_buf.contents() as *mut u32) = n as u32;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&rhs_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_buffer(3, Some(&n_buf), 0);
    let width = pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (n as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; n];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// GPU residual add: output = a + b. None if Metal unavailable.
#[cfg(target_os = "macos")]
pub fn try_residual_add_f32(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    let kernel = metal_linear_kernel()?;
    try_binary_elementwise_f32(&kernel.residual_add_pipeline, a, b)
}

/// GPU gated activation: output = (gate / (1 + e^-gate)) * up. None if Metal unavailable.
#[cfg(target_os = "macos")]
pub fn try_silu_mul_f32(gate: &[f32], up: &[f32]) -> Option<Vec<f32>> {
    let kernel = metal_linear_kernel()?;
    try_binary_elementwise_f32(&kernel.silu_mul_pipeline, gate, up)
}

/// GPU GeGLU activation (Gemma MLP): output = gelu_pytorch_tanh(gate) * up. Same
/// buffer shape as silu-mul. None if Metal unavailable.
#[cfg(target_os = "macos")]
pub fn try_gelu_mul_f32(gate: &[f32], up: &[f32]) -> Option<Vec<f32>> {
    let kernel = metal_linear_kernel()?;
    try_binary_elementwise_f32(&kernel.gelu_mul_pipeline, gate, up)
}

/// GPU final-logit soft-cap: output = cap * tanh(input / cap). A non-finite or
/// non-positive cap returns the input unchanged (matches the CPU reference). None
/// if Metal unavailable.
#[cfg(target_os = "macos")]
pub fn try_soft_cap_f32(input: &[f32], cap: f32) -> Option<Vec<f32>> {
    if input.is_empty() {
        return None;
    }
    if !cap.is_finite() || cap <= 0.0 {
        return Some(input.to_vec());
    }
    let kernel = metal_linear_kernel()?;
    let n = input.len();
    let byte_len = std::mem::size_of_val(input) as u64;
    let in_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let out_buf = kernel
        .device
        .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let n_buf = kernel
        .device
        .new_buffer(4, MTLResourceOptions::StorageModeShared);
    let cap_buf = kernel
        .device
        .new_buffer(4, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&in_buf, input);
    unsafe {
        *(n_buf.contents() as *mut u32) = n as u32;
        *(cap_buf.contents() as *mut f32) = cap;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.soft_cap_pipeline);
    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&out_buf), 0);
    encoder.set_buffer(2, Some(&n_buf), 0);
    encoder.set_buffer(3, Some(&cap_buf), 0);
    let width = kernel.soft_cap_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (n as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; n];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// GPU forward RoPE rotation across all heads, applied to a copy of `data`. The
/// cos/sin tables (one entry per rotated pair) come from the CPU's frequency/scaling
/// math; this only performs the rotation. `split_half_pairing` selects split-half vs
/// adjacent even/odd. None if Metal is unavailable.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_rope_rotate_f32(
    data: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_count: usize,
    head_dim: usize,
    half_rope: usize,
    split_half_pairing: bool,
) -> Option<Vec<f32>> {
    if head_count == 0
        || half_rope == 0
        || data.len() != head_count * head_dim
        || cos_table.len() != half_rope
        || sin_table.len() != half_rope
        || half_rope * 2 > head_dim
    {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let data_bytes = std::mem::size_of_val(data) as u64;
    let table_bytes = std::mem::size_of_val(cos_table) as u64;
    let data_buf = kernel
        .device
        .new_buffer(data_bytes, MTLResourceOptions::StorageModeShared);
    let cos_buf = kernel
        .device
        .new_buffer(table_bytes, MTLResourceOptions::StorageModeShared);
    let sin_buf = kernel
        .device
        .new_buffer(table_bytes, MTLResourceOptions::StorageModeShared);
    let scalar_buf = kernel
        .device
        .new_buffer(16, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&data_buf, data);
    write_buffer_f32(&cos_buf, cos_table);
    write_buffer_f32(&sin_buf, sin_table);
    unsafe {
        let p = scalar_buf.contents() as *mut u32;
        *p = head_count as u32;
        *p.add(1) = head_dim as u32;
        *p.add(2) = half_rope as u32;
        *p.add(3) = u32::from(split_half_pairing);
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.rope_rotate_pipeline);
    encoder.set_buffer(0, Some(&data_buf), 0);
    encoder.set_buffer(1, Some(&cos_buf), 0);
    encoder.set_buffer(2, Some(&sin_buf), 0);
    encoder.set_buffer(3, Some(&scalar_buf), 0);
    encoder.set_buffer(4, Some(&scalar_buf), 4);
    encoder.set_buffer(5, Some(&scalar_buf), 8);
    encoder.set_buffer(6, Some(&scalar_buf), 12);
    let total = (head_count * half_rope) as u64;
    let width = kernel.rope_rotate_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: total.div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; data.len()];
    read_buffer_f32(&data_buf, &mut out);
    Some(out)
}

/// GPU single-query (decode) causal attention over a contiguous KV cache laid out
/// [kv_head][position][head_dim]. GQA via `group = n_heads / n_kv_heads`. Returns the
/// per-head context [n_heads*head_dim], or None if Metal is unavailable. Mirrors
/// attention_context_for_head_into. Thin wrapper over `try_attention_decode_strided_f32`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_decode_f32(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    scale: f32,
) -> Option<Vec<f32>> {
    if head_dim == 0
        || position_count == 0
        || keys.len() != n_kv_heads * position_count * head_dim
        || values.len() != keys.len()
    {
        return None;
    }
    try_attention_decode_strided_f32(
        query,
        keys,
        values,
        n_heads,
        n_kv_heads,
        head_dim,
        position_count,
        scale,
        head_dim,
        position_count * head_dim,
        0,
    )
}

/// GPU single-query (decode) causal attention reading the KV cache with explicit strides:
/// the K/V element at (kv_head, position, d) is at `kv_base_offset + kv_head*kv_head_stride
/// + position*position_stride + d` (strides in floats) in `keys`/`values`. This lets the
/// kernel read a per-layer slice of an interleaved `[position][layer][kv_head][head_dim]`
/// cache directly, with no CPU repack. GQA via `group = n_heads / n_kv_heads`. Returns the
/// per-head context `[n_heads*head_dim]`, or None if Metal is unavailable / inputs invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_decode_strided_f32(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    scale: f32,
    position_stride: usize,
    kv_head_stride: usize,
    kv_base_offset: usize,
) -> Option<Vec<f32>> {
    // Largest float index the kernel will touch in keys/values (last head, last position).
    // saturating_sub keeps this panic-free for zero inputs, which the checks below reject.
    let max_index = kv_base_offset
        + n_kv_heads.saturating_sub(1) * kv_head_stride
        + position_count.saturating_sub(1) * position_stride
        + head_dim.saturating_sub(1);
    if n_heads == 0
        || n_kv_heads == 0
        || head_dim == 0
        || position_count == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || query.len() != n_heads * head_dim
        || values.len() != keys.len()
        || keys.len() <= max_index
    {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let group = (n_heads / n_kv_heads) as u32;
    let new = |bytes: u64| {
        kernel
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
    let query_buf = new(std::mem::size_of_val(query) as u64);
    let keys_buf = new(std::mem::size_of_val(keys) as u64);
    let values_buf = new(std::mem::size_of_val(values) as u64);
    let scores_buf = new((n_heads * position_count * 4) as u64);
    let output_buf = new((n_heads * head_dim * 4) as u64);
    let scalar_buf = new(32);
    write_buffer_f32(&query_buf, query);
    write_buffer_f32(&keys_buf, keys);
    write_buffer_f32(&values_buf, values);
    unsafe {
        let p = scalar_buf.contents() as *mut u8;
        *(p as *mut u32) = n_heads as u32;
        *(p.add(4) as *mut u32) = head_dim as u32;
        *(p.add(8) as *mut u32) = position_count as u32;
        *(p.add(12) as *mut u32) = group;
        *(p.add(16) as *mut f32) = scale;
        *(p.add(20) as *mut u32) = position_stride as u32;
        *(p.add(24) as *mut u32) = kv_head_stride as u32;
        *(p.add(28) as *mut u32) = kv_base_offset as u32;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.attention_decode_pipeline);
    encoder.set_buffer(0, Some(&query_buf), 0);
    encoder.set_buffer(1, Some(&keys_buf), 0);
    encoder.set_buffer(2, Some(&values_buf), 0);
    encoder.set_buffer(3, Some(&scores_buf), 0);
    encoder.set_buffer(4, Some(&output_buf), 0);
    encoder.set_buffer(5, Some(&scalar_buf), 0);
    encoder.set_buffer(6, Some(&scalar_buf), 4);
    encoder.set_buffer(7, Some(&scalar_buf), 8);
    encoder.set_buffer(8, Some(&scalar_buf), 12);
    encoder.set_buffer(9, Some(&scalar_buf), 16);
    encoder.set_buffer(10, Some(&scalar_buf), 20);
    encoder.set_buffer(11, Some(&scalar_buf), 24);
    encoder.set_buffer(12, Some(&scalar_buf), 28);
    // One threadgroup per head, a single 32-lane SIMD group cooperating within it.
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; n_heads * head_dim];
    read_buffer_f32(&output_buf, &mut out);
    Some(out)
}

/// GPU Q8_0 quantization of an f32 row (length a multiple of 32). Returns the per-block
/// scales (f32) and quants (i8) in the layout the Q8 block matmul consumes — the
/// on-GPU equivalent of the CPU activation quantizer, so activations produced on-GPU
/// can feed the matmul without a CPU round-trip. None if Metal is unavailable.
#[cfg(target_os = "macos")]
pub fn try_quantize_q8_0_f32(input: &[f32]) -> Option<(Vec<f32>, Vec<i8>)> {
    if input.is_empty() || !input.len().is_multiple_of(32) {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let n_blocks = input.len() / 32;
    let in_buf = kernel.device.new_buffer(
        std::mem::size_of_val(input) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = kernel
        .device
        .new_buffer((n_blocks * 4) as u64, MTLResourceOptions::StorageModeShared);
    let quants_buf = kernel
        .device
        .new_buffer(input.len() as u64, MTLResourceOptions::StorageModeShared);
    let n_buf = kernel
        .device
        .new_buffer(4, MTLResourceOptions::StorageModeShared);
    write_buffer_f32(&in_buf, input);
    unsafe {
        *(n_buf.contents() as *mut u32) = n_blocks as u32;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.quantize_q8_0_pipeline);
    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&scales_buf), 0);
    encoder.set_buffer(2, Some(&quants_buf), 0);
    encoder.set_buffer(3, Some(&n_buf), 0);
    let width = kernel
        .quantize_q8_0_pipeline
        .thread_execution_width()
        .max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (n_blocks as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut scales = vec![0.0f32; n_blocks];
    read_buffer_f32(&scales_buf, &mut scales);
    let mut quants = vec![0i8; input.len()];
    read_buffer_i8(&quants_buf, &mut quants);
    Some((scales, quants))
}

/// GPU-resident `quantize -> Q8 matmul` chain in a single command buffer: quantize the
/// f32 input activation and matmul it against a Q8_0 weight (36-byte blocks), passing
/// the quantized scales/quants between the two kernels via GPU buffers with no CPU
/// readback. This is the core decode primitive and the proof that resident buffer
/// chaining works; it is bit-identical to running the two standalone kernels. None if
/// Metal is unavailable. `weight_blocks` is [output_width * blocks_per_row * 36] bytes.
#[cfg(target_os = "macos")]
pub fn try_quantized_matmul_resident(
    input: &[f32],
    weight_blocks: &[u8],
    output_width: usize,
) -> Option<Vec<f32>> {
    if input.is_empty() || !input.len().is_multiple_of(32) || output_width == 0 {
        return None;
    }
    let blocks_per_row = input.len() / 32;
    if weight_blocks.len() != output_width * blocks_per_row * 36 {
        return None;
    }
    let kernel = metal_linear_kernel()?;
    let new = |bytes: u64| {
        kernel
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = new(std::mem::size_of_val(input) as u64);
    let scales_buf = new((blocks_per_row * 4) as u64);
    let quants_buf = new(input.len() as u64);
    let weight_buf = new(weight_blocks.len() as u64);
    let out_buf = new((output_width * 4) as u64);
    let n_blocks_buf = new(4);
    let mm_scalar_buf = new(8);
    write_buffer_f32(&in_buf, input);
    write_buffer_u8(&weight_buf, weight_blocks);
    unsafe {
        *(n_blocks_buf.contents() as *mut u32) = blocks_per_row as u32;
        let p = mm_scalar_buf.contents() as *mut u32;
        *p = blocks_per_row as u32;
        *p.add(1) = output_width as u32;
    }
    let command_buffer = kernel.queue.new_command_buffer();

    // Stage 1: quantize the activation into scales_buf / quants_buf.
    let q_enc = command_buffer.new_compute_command_encoder();
    q_enc.set_compute_pipeline_state(&kernel.quantize_q8_0_pipeline);
    q_enc.set_buffer(0, Some(&in_buf), 0);
    q_enc.set_buffer(1, Some(&scales_buf), 0);
    q_enc.set_buffer(2, Some(&quants_buf), 0);
    q_enc.set_buffer(3, Some(&n_blocks_buf), 0);
    let qw = kernel
        .quantize_q8_0_pipeline
        .thread_execution_width()
        .max(1);
    q_enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (blocks_per_row as u64).div_ceil(qw),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: qw,
            height: 1,
            depth: 1,
        },
    );
    q_enc.end_encoding();

    // Stage 2: matmul reads the quantized buffers stage 1 just wrote (same command
    // buffer; ordered, hazard-tracked for shared buffers).
    let m_enc = command_buffer.new_compute_command_encoder();
    m_enc.set_compute_pipeline_state(&kernel.q8_0_block_pipeline);
    m_enc.set_buffer(0, Some(&scales_buf), 0);
    m_enc.set_buffer(1, Some(&quants_buf), 0);
    m_enc.set_buffer(2, Some(&weight_buf), 0);
    m_enc.set_buffer(3, Some(&out_buf), 0);
    m_enc.set_buffer(4, Some(&mm_scalar_buf), 0);
    m_enc.set_buffer(5, Some(&mm_scalar_buf), 4);
    let mw = kernel.q8_0_block_pipeline.thread_execution_width().max(1);
    m_enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (output_width as u64).div_ceil(mw),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: mw,
            height: 1,
            depth: 1,
        },
    );
    m_enc.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    let mut out = vec![0.0f32; output_width];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

// --- Resident-buffer encode helpers: encode one kernel into a shared command buffer,
// operating on GPU buffers with no readback. Reusable building blocks for the
// GPU-resident forward pass (KV cache + per-token chaining build on these). ---

#[cfg(target_os = "macos")]
fn dispatch_1d(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    n: usize,
) {
    let w = pipeline.thread_execution_width().max(1);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (n as u64).div_ceil(w),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: w,
            height: 1,
            depth: 1,
        },
    );
}

/// Fused rms_norm + Q8_0 quantize: reads `input`, emits the quantized normed row directly
/// into `scales`/`quants` (one dispatch, no intermediate normed buffer). `scalar` holds
/// width (u32) then eps (f32), like `encode_rms_norm`.
#[cfg(target_os = "macos")]
fn encode_rms_norm_quantize(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    quants: &Buffer,
    scalar: &Buffer,
) {
    e.set_compute_pipeline_state(&k.rms_norm_quantize_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(weight), 0);
    e.set_buffer(2, Some(scales), 0);
    e.set_buffer(3, Some(quants), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Fused SiLU-mul + Q8_0 quantize: reads `gate`/`up`, emits the quantized activation row
/// directly into `scales`/`quants` (one dispatch, no intermediate activation buffer).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_silu_mul_quantize(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    gate: &Buffer,
    up: &Buffer,
    scales: &Buffer,
    quants: &Buffer,
    nblocks_buf: &Buffer,
    n_blocks: usize,
) {
    e.set_compute_pipeline_state(&k.silu_mul_quantize_pipeline);
    e.set_buffer(0, Some(gate), 0);
    e.set_buffer(1, Some(up), 0);
    e.set_buffer(2, Some(scales), 0);
    e.set_buffer(3, Some(quants), 0);
    e.set_buffer(4, Some(nblocks_buf), 0);
    dispatch_1d(e, &k.silu_mul_quantize_pipeline, n_blocks);
}

#[cfg(target_os = "macos")]
fn encode_quantize(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    scales: &Buffer,
    quants: &Buffer,
    nblocks_buf: &Buffer,
    n_blocks: usize,
) {
    e.set_compute_pipeline_state(&k.quantize_q8_0_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(scales), 0);
    e.set_buffer(2, Some(quants), 0);
    e.set_buffer(3, Some(nblocks_buf), 0);
    dispatch_1d(e, &k.quantize_q8_0_pipeline, n_blocks);
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_q8_matmul(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    scales: &Buffer,
    quants: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    // Three GEMV layouts, selected per dispatch:
    // - ksplit (opt-in via CAMELID_METAL_KSPLIT): two rows per threadgroup, four simdgroups
    //   partitioning the contraction with a threadgroup-memory reduction — maximizes
    //   concurrent weight streams per output row.
    // - qmv4 (rows % 4 == 0): two simdgroups per TG, four rows per simdgroup, input cached
    //   in registers, four independent accumulator chains.
    // - one-row-per-simdgroup fallback for other row counts.
    const SIMD_GROUPS_PER_TG: u64 = 2;
    let ksplit = ksplit_gemv_enabled();
    let qmv4 = rows.is_multiple_of(4);
    let (pipeline, rows_per_tg, threads_per_tg) = if ksplit {
        (&k.q8_0_block_ksplit_pipeline, 2, 128)
    } else if qmv4 {
        (
            &k.q8_0_block_simd_qmv4_pipeline,
            SIMD_GROUPS_PER_TG * 4,
            SIMD_GROUPS_PER_TG * 32,
        )
    } else {
        (
            &k.q8_0_block_simd_mr_pipeline,
            SIMD_GROUPS_PER_TG,
            SIMD_GROUPS_PER_TG * 32,
        )
    };
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(scales), 0);
    e.set_buffer(1, Some(quants), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    if ksplit {
        // Two rows x 32 lanes of f32 partials for the cross-simdgroup reduction.
        e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    }
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(rows_per_tg),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        },
    );
}

/// Opt-in experiment flag for the K-split cooperative GEMV layout in the resident decode.
#[cfg(target_os = "macos")]
fn ksplit_gemv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_KSPLIT")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Opt-in experiment flag: route the resident decode through the f32-activation GEMV
/// (no input-quantize dispatches; float-FMA inner product). See the kernel comment.
#[cfg(target_os = "macos")]
fn f32y_gemv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_F32Y")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Unfused RMSNorm to an f32 output buffer. Scalar layout: width (u32) then eps (f32).
#[cfg(target_os = "macos")]
fn encode_rms_norm_f32(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
) {
    e.set_compute_pipeline_state(&k.rms_norm_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(weight), 0);
    e.set_buffer(2, Some(output), 0);
    e.set_buffer(3, Some(scalar), 0);
    e.set_buffer(4, Some(scalar), 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Batched RMSNorm over `rows` contiguous f32 rows (`rms_norm_batch_f32`): one 256-thread
/// group per row, the SAME reduction + `1/sqrt(mean_sq+eps)` as the single-row
/// `rms_norm_f32`, so row `r`'s output bit-equals a single-token norm of `input[r*width..]`.
/// `scalar` holds width (u32) @0 then eps (f32) @4.
#[cfg(target_os = "macos")]
fn encode_rms_norm_batch(
    k: &MetalLinearKernel,
    e: &metal::ComputeCommandEncoderRef,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    e.set_compute_pipeline_state(&k.rms_norm_batch_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(weight), 0);
    e.set_buffer(2, Some(output), 0);
    e.set_buffer(3, Some(scalar), 0);
    e.set_buffer(4, Some(scalar), 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Opt-in: with CAMELID_METAL_F32Y also set, weights upload in the raw GGUF 34-byte
/// f16-scale wire layout (~5.9% fewer weight bytes per token).
#[cfg(target_os = "macos")]
fn wire_weights_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_WIRE")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Whether the resident stack consumes weights in the raw GGUF wire layout
/// (CAMELID_METAL_F32Y + CAMELID_METAL_WIRE, the default CLI stack). Wire-page
/// (fast-load) weights require this: their bytes only exist in wire form.
#[cfg(target_os = "macos")]
pub fn wire_mode_active() -> bool {
    f32y_gemv_enabled() && wire_weights_enabled()
}

/// Non-macOS stub: there is no Metal stack, so wire mode is never active.
#[cfg(not(target_os = "macos"))]
pub fn wire_mode_active() -> bool {
    false
}

/// Prefill GEMMs use the simdgroup-matrix tiled kernel (numerically equivalent to the
/// scalar k-split GEMM but not byte-exact: tile MMA accumulation order). Default on via
/// the fast stack; CAMELID_METAL_MM=0 restores the byte-exact scalar GEMM.
#[cfg(target_os = "macos")]
/// Scratch budget for the attention-as-matmul prefill's materialized S/P panels
/// (n_heads x n_pad^2, half each; the 4-byte factor in the gate covers both). The
/// budget directly sets how deep the fast prefill path reaches: 256 MiB admits
/// ~1.6k-token prompts at 24 heads, 2 GiB admits ~4.7k (anchored-recall checked
/// through 4k). Default is RAM-aware — an eighth of physical memory, capped at
/// 2 GiB — because the panels are transient prefill scratch in unified memory;
/// CAMELID_METAL_ATTN_MM_CAP_MB overrides explicitly. Prompts past the admitted
/// depth take the v3 attention path (correct, slower).
#[cfg(target_os = "macos")]
fn attn_mm_scratch_cap_bytes() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        if let Some(mb) = std::env::var("CAMELID_METAL_ATTN_MM_CAP_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return mb * 1024 * 1024;
        }
        let mut memsize: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = std::ffi::CString::new("hw.memsize").expect("static name");
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut memsize as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        let phys = if rc == 0 { memsize as usize } else { 0 };
        (phys / 8).clamp(256 * 1024 * 1024, 2048 * 1024 * 1024)
    })
}

#[cfg(target_os = "macos")]
fn mm_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_MM").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// K-split GEMV over an f32 activation vector (see q8_0_block_linear_row_ksplit_f32y).
/// The weight buffer must match the active format: decoded 36-byte blocks normally, wire
/// 34-byte blocks when CAMELID_METAL_WIRE is on (prepare_token resolves accordingly).
#[cfg(target_os = "macos")]
fn encode_q8_matmul_f32y(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    let nsg8 = wire_nsg8_enabled();
    let pipeline = if wire_weights_enabled() && nsg8 {
        &k.q8_0_block_ksplit_f32y_wire_nsg8_pipeline
    } else if wire_weights_enabled() {
        &k.q8_0_block_ksplit_f32y_wire_pipeline
    } else {
        &k.q8_0_block_ksplit_f32y_pipeline
    };
    let threads_per_tg = if wire_weights_enabled() && nsg8 {
        256
    } else {
        128
    };
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        },
    );
}

/// Batched-column mirror of [`encode_q8_matmul_f32y`]'s production NSG=8 wire GEMV.
/// Instead of dotting each weight block against one activation vector, it dots it
/// against all `n_rows_in` activation columns (the speculative-verify rows) before
/// moving on, so each weight is streamed once for the whole window rather than once
/// per token. `y` is row-major `[token][blocks_per_row*32]`; `out` is token-major
/// `[token][rows]` (matching the next GEMM's `[token][dim]` input). The bound kernel
/// keeps the single-token per-block `sumq` then `*w_scale` ordering and the exact
/// two-stage reduction, so each output column is BIT-IDENTICAL to the single-token
/// dispatch (proven by `metal_verify_gemv_batched_bit_identical`). `scalar` must be
/// at least 12 bytes with `blocks_per_row` @0 and `rows` @4 already written (same as
/// the single-token caller); the column count `n_rows_in` is written to @8 here.
// Consumed by the speculative-verify lane (a later checkpoint); for now exercised by
// the `metal_verify_gemv_batched_bit_identical` unit test, so it reads as dead in a
// non-test lib build.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn encode_q8_matmul_f32y_batched(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
    n_rows_in: usize,
) {
    unsafe {
        let p = scalar.contents() as *mut u8;
        *(p.add(8) as *mut u32) = n_rows_in as u32;
    }
    e.set_compute_pipeline_state(&k.q8_0_block_ksplit_f32y_wire_nsg8_verify_pipeline);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.set_buffer(6, Some(scalar), 8);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Wire quant format of a gemma4 weight tensor for the GPU GEMV dispatch.
/// `Q8_0` = 34-byte blocks (f16 scale + 32 i8); `Q4_0` = 18-byte blocks (f16
/// scale + 16 nibble bytes). Un-gated so it can appear in the public
/// `try_gemma4_ffn` signature on every target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmaWireFmt {
    Q8_0,
    Q4_0,
    /// NVFP4 (GABBRO M3): 64-element superblocks, 36 wire bytes (`d[4]` UE4M3
    /// sub-block scales + `qs[32]` packed E2M1 nibbles). Unlike Q8_0/Q4_0
    /// (32-value blocks) this is a 64-value block, so callers must size
    /// `blocks_per_row` by [`GemmaWireFmt::block_elements`], never a hardcoded 32.
    Nvfp4,
}

impl GemmaWireFmt {
    /// Bytes per wire block: 34/18 over 32 values (Q8_0/Q4_0), 36 over 64 values
    /// (NVFP4 superblock).
    pub fn wire_bytes(self) -> usize {
        match self {
            GemmaWireFmt::Q8_0 => 34,
            GemmaWireFmt::Q4_0 => 18,
            GemmaWireFmt::Nvfp4 => 36,
        }
    }

    /// Weight values per wire block: 32 for Q8_0/Q4_0, 64 for the NVFP4
    /// superblock. Row block count is `in_dim / block_elements()`, so this is the
    /// single source of truth that keeps `blocks_per_row`, `row_stride`, and the
    /// kernel's activation stride consistent across formats.
    pub fn block_elements(self) -> usize {
        match self {
            GemmaWireFmt::Q8_0 | GemmaWireFmt::Q4_0 => 32,
            GemmaWireFmt::Nvfp4 => 64,
        }
    }
}

/// Dispatch the gemma4 GPU GEMV for the weight's wire format. Q8_0 and Q4_0
/// share the same `(y, weight, out, scalar, rows)` contract — only the kernel
/// (and the per-block byte stride it reads) differs — so callers in the resident
/// graph stay format-agnostic.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_matmul(
    fmt: GemmaWireFmt,
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    match fmt {
        GemmaWireFmt::Q8_0 => encode_gemma4_q8_matmul(e, k, y, weight, out, scalar, rows),
        GemmaWireFmt::Q4_0 => encode_gemma4_q4_0_matmul(e, k, y, weight, out, scalar, rows),
        GemmaWireFmt::Nvfp4 => encode_gemma4_nvfp4_matmul(e, k, y, weight, out, scalar, rows),
    }
}

/// Encode one f32-activation × wire-Q8 GEMV into the shared encoder:
/// `out[r] = Σ_b w_scale[b] · Σ_j (w_i8[b][j] · y[b*32+j])`. The weight is the raw
/// 34-byte GGUF wire layout (f16 scale + 32 i8). Gemma's resident decode always
/// uses the nocopy wire weights, so — unlike [`encode_q8_matmul_f32y`] — this is
/// NOT gated on `CAMELID_METAL_WIRE`; it always binds the wire f32y K-split kernel.
/// `scalar` holds [blocks_per_row: u32 @0, rows: u32 @4].
#[cfg(target_os = "macos")]
fn encode_gemma4_q8_matmul(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    e.set_compute_pipeline_state(&k.q8_0_block_ksplit_f32y_wire_pipeline);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
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

/// Q4_0 wire GEMV — the QAT-row counterpart of [`encode_gemma4_q8_matmul`].
/// Identical dispatch (128 threads/TG, NR0=2 rows/TG, 2*32*4 threadgroup mem);
/// only the bound pipeline differs (it reads 18-byte Q4_0 wire blocks and
/// unpacks nibbles inline). `scalar` holds blocks_per_row at offset 0 and rows
/// at offset 4, exactly as the Q8 path.
#[cfg(target_os = "macos")]
fn encode_gemma4_q4_0_matmul(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    e.set_compute_pipeline_state(&k.q4_0_block_ksplit_f32y_wire_pipeline);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
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

/// Encode one f32-activation × wire-NVFP4 GEMV (GABBRO M3). Same dispatch shape as
/// the Q8/Q4_0 paths (128 threads/TG, NR0=2 rows/TG, 2*32*4 threadgroup mem); only
/// the bound pipeline differs — it reads 36-byte NVFP4 superblocks (64 values:
/// `d[4]` UE4M3 sub-block scales + `qs[32]` E2M1 nibbles) and reproduces the CPU
/// oracle `nvfp4_wire_block_dequant` bit-for-bit. `scalar` holds
/// [blocks_per_row: u32 @0, rows: u32 @4] where blocks_per_row counts 64-value
/// superblocks (in_dim / 64), so `row_stride = blocks_per_row * 36` is exact.
#[cfg(target_os = "macos")]
fn encode_gemma4_nvfp4_matmul(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    y: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    rows: usize,
) {
    e.set_compute_pipeline_state(&k.nvfp4_block_ksplit_f32y_wire_pipeline);
    e.set_buffer(0, Some(y), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
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

/// Encode the gemma4 FFN sub-block into the (serial) encoder with no commit/readback:
/// reads `in_buf` (hidden), writes the residual sum into `out_buf` (hidden):
///   normf = rms_norm(in_buf, ffn_norm)
///   gate  = normf · gate_w   (ffn_dim rows)
///   up    = normf · up_w     (ffn_dim rows)
///   act   = gelu_tanh(gate) * up            (GeGLU)
///   down  = act · down_w     (hidden rows)
///   dn    = rms_norm(down, post_ffw_norm)
///   out   = in_buf + dn
/// vs Llama's FFN this swaps SwiGLU→GeGLU and adds the extra `post_ffw_norm` before
/// the residual. Scratch buffers are pushed into `keep` so they outlive the command
/// buffer. The encoder is serial, so the dependent dispatches need no manual barriers.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_ffn(
    fmt: GemmaWireFmt,
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    in_buf: &Buffer,
    out_buf: &Buffer,
    ffn_norm: &[f32],
    post_ffw_norm: &[f32],
    eps: f32,
    gate_w: &Buffer,
    up_w: &Buffer,
    down_w: &Buffer,
    ffn_dim: usize,
) {
    let hidden = ffn_norm.len();
    // GABBRO M3-followup: blocks_per_row counts wire blocks, whose element width is
    // format-specific (32 for Q8_0/Q4_0, 64 for the NVFP4 superblock). The resident
    // NVFP4 GEMV reads row_stride = blocks_per_row * 36, so a hardcoded /32 would
    // double the stride and read past each row.
    let bpr_hidden = hidden / fmt.block_elements();
    let bpr_ffn = ffn_dim / fmt.block_elements();
    let nb = |bytes: u64| pool_get(k, bytes);
    let norm_w = nb((hidden * 4) as u64);
    let postnorm_w = nb((hidden * 4) as u64);
    let rms_scalar = nb(8);
    let gateup_scalar = nb(8);
    let down_scalar = nb(8);
    let geglu_n = nb(4);
    let resid_n = nb(4);
    let normf = nb((hidden * 4) as u64);
    let gate_buf = nb((ffn_dim * 4) as u64);
    let up_buf = nb((ffn_dim * 4) as u64);
    let act_buf = nb((ffn_dim * 4) as u64);
    let down_buf = nb((hidden * 4) as u64);
    let dn_buf = nb((hidden * 4) as u64);

    write_buffer_f32(&norm_w, ffn_norm);
    write_buffer_f32(&postnorm_w, post_ffw_norm);
    unsafe {
        let p = rms_scalar.contents() as *mut u8;
        *(p as *mut u32) = hidden as u32;
        *(p.add(4) as *mut f32) = eps;
        let g = gateup_scalar.contents() as *mut u32;
        *g = bpr_hidden as u32;
        *g.add(1) = ffn_dim as u32;
        let d = down_scalar.contents() as *mut u32;
        *d = bpr_ffn as u32;
        *d.add(1) = hidden as u32;
        *(geglu_n.contents() as *mut u32) = ffn_dim as u32;
        *(resid_n.contents() as *mut u32) = hidden as u32;
    }

    encode_rms_norm_f32(e, k, in_buf, &norm_w, &normf, &rms_scalar);
    encode_gemma4_matmul(
        fmt,
        e,
        k,
        &normf,
        gate_w,
        &gate_buf,
        &gateup_scalar,
        ffn_dim,
    );
    encode_gemma4_matmul(fmt, e, k, &normf, up_w, &up_buf, &gateup_scalar, ffn_dim);
    encode_binary(
        e,
        &k.gelu_mul_pipeline,
        &gate_buf,
        &up_buf,
        &act_buf,
        &geglu_n,
        ffn_dim,
    );
    encode_gemma4_matmul(fmt, e, k, &act_buf, down_w, &down_buf, &down_scalar, hidden);
    encode_rms_norm_f32(e, k, &down_buf, &postnorm_w, &dn_buf, &rms_scalar);
    encode_binary(
        e,
        &k.residual_add_pipeline,
        in_buf,
        &dn_buf,
        out_buf,
        &resid_n,
        hidden,
    );

    keep.extend([
        norm_w,
        postnorm_w,
        rms_scalar,
        gateup_scalar,
        down_scalar,
        geglu_n,
        resid_n,
        normf,
        gate_buf,
        up_buf,
        act_buf,
        down_buf,
        dn_buf,
    ]);
}

/// Encode a per-head RMSNorm (Gemma QK-norm / weightless V-norm) into the encoder:
/// one threadgroup per head normalizes that head's `head_dim` chunk. `scalar` is
/// [head_dim: u32 @0, eps: f32 @4, use_weight: u32 @8]; `weight` may be a dummy when
/// use_weight is 0.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_rms_norm_per_head(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    head_count: usize,
    // Byte offset applied to the (in-place) input/output buffer so the per-head norm
    // can run on an arbitrary verify row; the per-head `base = head*head_dim` indexing
    // is relative to the bound offset, so the arithmetic is unchanged. Default 0 for all
    // pre-verify callers (no behavior change).
    row_off: u64,
) {
    e.set_compute_pipeline_state(&k.rms_norm_per_head_pipeline);
    e.set_buffer(0, Some(input), row_off);
    e.set_buffer(1, Some(weight), 0);
    e.set_buffer(2, Some(output), row_off);
    e.set_buffer(3, Some(scalar), 0);
    e.set_buffer(4, Some(scalar), 4);
    e.set_buffer(5, Some(scalar), 8);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: head_count as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Encode the gemma4 attention sub-block into the (serial) encoder, no readback:
/// reads `in_buf` (hidden), writes the residual sum into `out_buf`:
///   normf = rms_norm(in_buf, attn_norm)
///   q/k/v = normf · {q,k,v}_w
///   per-head QK-norm (q_norm/k_norm) + weightless per-head V-norm
///   RoPE(q), RoPE(k)   (split-half pairing; cos/sin precomputed on CPU for this θ+pos)
///   scatter (roped) k + (normed) v into the cache at `write_position`
///   ctx = decode_attention(q, cache) over the window [window_start .. filled)
///   o   = ctx · o_w
///   out = in_buf + rms_norm(o, post_attn_norm)
/// vs Llama's attention this adds the QK/V per-head norms and the post-attn norm, is
/// always f32y, and windows via `window_start` (sliding layers; 0 = global). The
/// caller owns the weight + cache buffers; scratch goes into `keep`. The basic
/// one-simdgroup-per-head decode kernel is used (gemma head_dim 256/512 > the v2 cap).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_attention(
    fmt: GemmaWireFmt,
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    in_buf: &Buffer,
    out_buf: &Buffer,
    attn_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    post_attn_norm: &[f32],
    eps: f32,
    q_w: &Buffer,
    k_w: &Buffer,
    v_w: Option<&Buffer>,
    o_w: &Buffer,
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k_buf: &Buffer,
    cache_v_buf: &Buffer,
    max_positions: usize,
    write_position: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    filled: usize,
    window_start: usize,
    scale: f32,
    owns_kv: bool,
) {
    let hidden = attn_norm.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let half_rope = cos_t.len();
    // GABBRO M3-followup: format-aware block count (see encode_gemma4_ffn) — /64 for
    // the NVFP4 superblock, /32 for Q8_0/Q4_0 (block_elements()).
    let bpr_hidden = hidden / fmt.block_elements();
    let bpr_q = q_dim / fmt.block_elements();
    let group = (n_heads / n_kv_heads) as u32;
    let position_count = filled - window_start;
    let kv_base_offset = window_start * head_dim;

    let nb = |bytes: u64| pool_get(k, bytes);
    let f32b = |n: usize| nb((n * 4) as u64);
    let norm_w = f32b(hidden);
    let qnorm_w = f32b(head_dim);
    let knorm_w = f32b(head_dim);
    let postnorm_w = f32b(hidden);
    let rms_scalar = nb(8);
    let post_rms_scalar = nb(8);
    let perhead_q = nb(12);
    let perhead_k = nb(12);
    let perhead_v = nb(12);
    let q_mm = nb(8);
    let kv_mm = nb(8);
    let o_mm = nb(8);
    let rope_q = nb(16);
    let rope_k = nb(16);
    let attn_scalar = nb(32);
    let scatter_scalar = nb(16);
    let kv16_write = nb(4);
    let resid_n = nb(4);
    let cos_buf = f32b(half_rope);
    let sin_buf = f32b(half_rope);
    let normf = f32b(hidden);
    let query_buf = f32b(q_dim);
    let key_buf = f32b(kv_dim);
    let val_buf = f32b(kv_dim);
    let qn_buf = f32b(q_dim);
    let kn_buf = f32b(kv_dim);
    let vn_buf = f32b(kv_dim);
    let scores_buf = f32b(n_heads * position_count);
    let ctx_buf = f32b(q_dim);
    let o_buf = f32b(hidden);
    let on_buf = f32b(hidden);

    write_buffer_f32(&norm_w, attn_norm);
    write_buffer_f32(&qnorm_w, q_norm);
    write_buffer_f32(&knorm_w, k_norm);
    write_buffer_f32(&postnorm_w, post_attn_norm);
    write_buffer_f32(&cos_buf, cos_t);
    write_buffer_f32(&sin_buf, sin_t);
    unsafe {
        let set_rms = |buf: &Buffer| {
            let p = buf.contents() as *mut u8;
            *(p as *mut u32) = hidden as u32;
            *(p.add(4) as *mut f32) = eps;
        };
        set_rms(&rms_scalar);
        set_rms(&post_rms_scalar);
        let set_perhead = |buf: &Buffer, use_w: u32| {
            let p = buf.contents() as *mut u8;
            *(p as *mut u32) = head_dim as u32;
            *(p.add(4) as *mut f32) = eps;
            *(p.add(8) as *mut u32) = use_w;
        };
        set_perhead(&perhead_q, 1);
        set_perhead(&perhead_k, 1);
        set_perhead(&perhead_v, 0);
        let set_mm = |buf: &Buffer, bpr: usize, rows: usize| {
            let p = buf.contents() as *mut u32;
            *p = bpr as u32;
            *p.add(1) = rows as u32;
        };
        set_mm(&q_mm, bpr_hidden, q_dim);
        set_mm(&kv_mm, bpr_hidden, kv_dim);
        set_mm(&o_mm, bpr_q, hidden);
        let set_rope = |buf: &Buffer, hc: usize| {
            let r = buf.contents() as *mut u32;
            *r = hc as u32;
            *r.add(1) = head_dim as u32;
            *r.add(2) = half_rope as u32;
            *r.add(3) = 1; // gemma uses split-half pairing
        };
        set_rope(&rope_q, n_heads);
        set_rope(&rope_k, n_kv_heads);
        let a = attn_scalar.contents() as *mut u8;
        *(a as *mut u32) = n_heads as u32;
        *(a.add(4) as *mut u32) = head_dim as u32;
        *(a.add(8) as *mut u32) = position_count as u32;
        *(a.add(12) as *mut u32) = group;
        *(a.add(16) as *mut f32) = scale;
        *(a.add(20) as *mut u32) = head_dim as u32; // position stride
        *(a.add(24) as *mut u32) = (max_positions * head_dim) as u32; // kv_head stride
        *(a.add(28) as *mut u32) = kv_base_offset as u32; // window start offset
        let s = scatter_scalar.contents() as *mut u32;
        *s = head_dim as u32;
        *s.add(1) = max_positions as u32;
        *s.add(2) = write_position as u32;
        *s.add(3) = kv_dim as u32;
        *(kv16_write.contents() as *mut u32) = 0; // gemma keeps the f32 cache
        *(resid_n.contents() as *mut u32) = hidden as u32;
    }

    encode_rms_norm_f32(e, k, in_buf, &norm_w, &normf, &rms_scalar);
    encode_gemma4_matmul(fmt, e, k, &normf, q_w, &query_buf, &q_mm, q_dim);
    encode_rms_norm_per_head(e, k, &query_buf, &qnorm_w, &qn_buf, &perhead_q, n_heads, 0);
    encode_rope(
        e, k, &qn_buf, &cos_buf, &sin_buf, &rope_q, n_heads, half_rope, 0, 0,
    );
    // Owning layers project + cache their own K/V; the trailing cross-shared layers
    // skip all of that and run attention against the source layer's cache
    // (`cache_k_buf`/`cache_v_buf` are the source's, already holding this token).
    if owns_kv {
        encode_gemma4_matmul(fmt, e, k, &normf, k_w, &key_buf, &kv_mm, kv_dim);
        // V source: the layer's own V projection, or — on V-less layers (12B-class
        // full attention, no attn_v tensor) — the RAW K projection output, before
        // k_norm/RoPE touch it (reference: `if v_proj is not present, use Kcur as
        // Vcur`). `key_buf` is never mutated (norms/rope write to kn_buf), so
        // reading it as the V source is safe in the serial encoder.
        let v_src = if let Some(v_w) = v_w {
            encode_gemma4_matmul(fmt, e, k, &normf, v_w, &val_buf, &kv_mm, kv_dim);
            &val_buf
        } else {
            &key_buf
        };
        encode_rms_norm_per_head(e, k, &key_buf, &knorm_w, &kn_buf, &perhead_k, n_kv_heads, 0);
        // Weightless V-norm: qnorm_w is bound as a dummy (use_weight = 0, never read).
        encode_rms_norm_per_head(e, k, v_src, &qnorm_w, &vn_buf, &perhead_v, n_kv_heads, 0);
        encode_rope(
            e, k, &kn_buf, &cos_buf, &sin_buf, &rope_k, n_kv_heads, half_rope, 0, 0,
        );
        // Scatter the roped K and normed V into the cache at write_position. f32 cache
        // only: bind the kv16 mirror slots to a placeholder with the write flag at 0.
        e.set_compute_pipeline_state(&k.kv_scatter_pipeline);
        e.set_buffer(0, Some(&kn_buf), 0);
        e.set_buffer(1, Some(&vn_buf), 0);
        e.set_buffer(2, Some(cache_k_buf), 0);
        e.set_buffer(3, Some(cache_v_buf), 0);
        e.set_buffer(4, Some(&scatter_scalar), 0);
        e.set_buffer(5, Some(&scatter_scalar), 4);
        e.set_buffer(6, Some(&scatter_scalar), 8);
        e.set_buffer(7, Some(&scatter_scalar), 12);
        e.set_buffer(8, Some(&scatter_scalar), 0);
        e.set_buffer(9, Some(&scatter_scalar), 0);
        e.set_buffer(10, Some(&kv16_write), 0);
        dispatch_1d(e, &k.kv_scatter_pipeline, kv_dim);
    }
    encode_attention(
        e,
        k,
        keep,
        &qn_buf,
        cache_k_buf,
        cache_v_buf,
        None,
        &scores_buf,
        &ctx_buf,
        &attn_scalar,
        n_heads,
        n_kv_heads,
        head_dim,
        position_count,
        0,
        0,
    );
    encode_gemma4_matmul(fmt, e, k, &ctx_buf, o_w, &o_buf, &o_mm, hidden);
    encode_rms_norm_f32(e, k, &o_buf, &postnorm_w, &on_buf, &post_rms_scalar);
    encode_binary(
        e,
        &k.residual_add_pipeline,
        in_buf,
        &on_buf,
        out_buf,
        &resid_n,
        hidden,
    );

    keep.extend([
        norm_w,
        qnorm_w,
        knorm_w,
        postnorm_w,
        rms_scalar,
        post_rms_scalar,
        perhead_q,
        perhead_k,
        perhead_v,
        q_mm,
        kv_mm,
        o_mm,
        rope_q,
        rope_k,
        attn_scalar,
        scatter_scalar,
        kv16_write,
        resid_n,
        cos_buf,
        sin_buf,
        normf,
        query_buf,
        key_buf,
        val_buf,
        qn_buf,
        kn_buf,
        vn_buf,
        scores_buf,
        ctx_buf,
        o_buf,
        on_buf,
    ]);
}

/// One gemma4 layer's resident weights: the six f32 norms plus the seven Q8 wire
/// weight buffers (q/k/v/o/gate/up/down) and the layer's dims. Bundling them keeps
/// [`encode_gemma4_layer`] from being a 40-argument call. The weight buffers are
/// built once and reused across tokens. (`from_wire` copies the wire bytes into
/// shared buffers; the production runtime will swap in `q8_wire_nocopy_buffer` over
/// `WirePages` so the 8GB stays single-copy.)
#[cfg(target_os = "macos")]
pub struct Gemma4ResidentLayer {
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    q_w: Buffer,
    k_w: Buffer,
    /// `None` on V-less layers (12B-class full attention): V is the raw K
    /// projection output, weightless-normed — the reference's `if v_proj is
    /// not present, use Kcur as Vcur`.
    v_w: Option<Buffer>,
    o_w: Buffer,
    gate_w: Buffer,
    up_w: Buffer,
    down_w: Buffer,
    /// Wire format of the q/k/v/o/gate/up/down projection weights. Q8_0 for the
    /// supported dense rows; Q4_0 for QAT rows (the GPU GEMV dispatches on this).
    fmt: GemmaWireFmt,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

#[cfg(target_os = "macos")]
impl Gemma4ResidentLayer {
    /// Build a layer's resident weights from row-major wire bytes (34-byte Q8_0 or
    /// 18-byte Q4_0 blocks per `fmt`) and the f32 norms. None if Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn from_wire(
        fmt: GemmaWireFmt,
        attn_norm: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        post_attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        post_ffw_norm: Vec<f32>,
        q_wire: &[u8],
        k_wire: &[u8],
        v_wire: Option<&[u8]>,
        o_wire: &[u8],
        gate_wire: &[u8],
        up_wire: &[u8],
        down_wire: &[u8],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        eps: f32,
    ) -> Option<Self> {
        let k = metal_linear_kernel()?;
        let buf = |bytes: &[u8]| {
            let b = k
                .device
                .new_buffer(bytes.len() as u64, MTLResourceOptions::StorageModeShared);
            write_buffer_u8(&b, bytes);
            b
        };
        Some(Self {
            attn_norm,
            q_norm,
            k_norm,
            post_attn_norm,
            ffn_norm,
            post_ffw_norm,
            q_w: buf(q_wire),
            k_w: buf(k_wire),
            v_w: v_wire.map(buf),
            o_w: buf(o_wire),
            gate_w: buf(gate_wire),
            up_w: buf(up_wire),
            down_w: buf(down_wire),
            fmt,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        })
    }

    /// Build a layer's resident weights from per-tensor page-aligned `WirePages`: the
    /// GPU reads them in place via `new_buffer_with_bytes_no_copy` — NO 8GB copy. This
    /// is the production path on a 16GB box (the wire bytes are the single resident
    /// copy). The nocopy cache pins each `WirePages` Arc for the buffer's lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn from_wire_pages(
        fmt: GemmaWireFmt,
        attn_norm: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        post_attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        post_ffw_norm: Vec<f32>,
        q_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        k_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        v_pages: Option<&std::sync::Arc<crate::wire_mmap::WirePages>>,
        o_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        gate_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        up_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        down_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        eps: f32,
    ) -> Option<Self> {
        let k = metal_linear_kernel()?;
        let mut cache = metal_linear_cache().lock().ok()?;
        let mut buf = |pages: &std::sync::Arc<crate::wire_mmap::WirePages>| {
            cache.q8_wire_nocopy_buffer(&k.device, pages)
        };
        Some(Self {
            attn_norm,
            q_norm,
            k_norm,
            post_attn_norm,
            ffn_norm,
            post_ffw_norm,
            q_w: buf(q_pages),
            k_w: buf(k_pages),
            v_w: v_pages.map(&mut buf),
            o_w: buf(o_pages),
            gate_w: buf(gate_pages),
            up_w: buf(up_pages),
            down_w: buf(down_pages),
            fmt,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        })
    }
}

/// Encode one full gemma4 decoder layer into the (serial) encoder, no readback:
/// `in_buf` --attention--> `mid_buf` --FFN--> `out_buf` (each sub-block adds its own
/// residual). `cos_t`/`sin_t` are this layer's per-θ RoPE tables for the current
/// position; the cache + window params select owning vs (later) shared-KV behaviour.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_layer(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    layer: &Gemma4ResidentLayer,
    in_buf: &Buffer,
    mid_buf: &Buffer,
    out_buf: &Buffer,
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k: &Buffer,
    cache_v: &Buffer,
    max_positions: usize,
    write_position: usize,
    filled: usize,
    window_start: usize,
    scale: f32,
    owns_kv: bool,
) {
    encode_gemma4_attention(
        layer.fmt,
        e,
        k,
        keep,
        in_buf,
        mid_buf,
        &layer.attn_norm,
        &layer.q_norm,
        &layer.k_norm,
        &layer.post_attn_norm,
        layer.eps,
        &layer.q_w,
        &layer.k_w,
        layer.v_w.as_ref(),
        &layer.o_w,
        cos_t,
        sin_t,
        cache_k,
        cache_v,
        max_positions,
        write_position,
        layer.n_heads,
        layer.n_kv_heads,
        layer.head_dim,
        filled,
        window_start,
        scale,
        owns_kv,
    );
    encode_gemma4_ffn(
        layer.fmt,
        e,
        k,
        keep,
        mid_buf,
        out_buf,
        &layer.ffn_norm,
        &layer.post_ffw_norm,
        layer.eps,
        &layer.gate_w,
        &layer.up_w,
        &layer.down_w,
        layer.ffn_dim,
    );
}

/// f32 dense output-major GEMV into the encoder: `out[o] = Σ_i w[o*rows + i]·input[i]`
/// for `o in 0..cols`. Matches gemma's `f32_matvec(w, in_dim=rows, out_dim=cols, x)`
/// (the PLE matrices are stored output-major). `scalar` = [rows: u32 @0, cols: u32 @4].
#[cfg(target_os = "macos")]
fn encode_linear_transposed_f32(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    weights: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    cols: usize,
) {
    e.set_compute_pipeline_state(&k.transposed_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(weights), 0);
    e.set_buffer(2, Some(output), 0);
    e.set_buffer(3, Some(scalar), 0);
    e.set_buffer(4, Some(scalar), 4);
    dispatch_1d(e, &k.transposed_pipeline, cols);
}

/// `output = input * s` into the encoder. `scalar` = [n: u32 @0, s: f32 @4].
#[cfg(target_os = "macos")]
fn encode_scale_f32(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    n: usize,
) {
    e.set_compute_pipeline_state(&k.scale_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(output), 0);
    e.set_buffer(2, Some(scalar), 0);
    e.set_buffer(3, Some(scalar), 4);
    dispatch_1d(e, &k.scale_pipeline, n);
}

/// Encode Gemma's per-layer Per-Layer-Embedding injection into the (serial) encoder,
/// updating `h_buf` in place: `h <- (h + ple_proj·gelu(ple_inp_gate·h ⊙ pli)) * scale`.
/// Concretely: `gated = ple_inp_gate · h` (f32 GEMV, hidden→ple_dim); `gated =
/// gelu_tanh(gated) · pli`; `proj = ple_proj · gated` (f32 GEMV, ple_dim→hidden);
/// `pnv = rms_norm(proj, post_norm)`; `h = (h + pnv) * output_scale`. `pli` (the
/// per-layer input for this token) is computed on the CPU once per token and passed
/// in. The PLE matrices are f32. Scratch goes into `keep`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_ple(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    h_buf: &Buffer,
    pli_buf: &Buffer,
    pli_offset: usize,
    inp_gate_w: &Buffer,
    proj_w: &Buffer,
    postnorm_w: &Buffer,
    output_scale: f32,
    eps: f32,
    hidden: usize,
    ple_dim: usize,
) {
    let nb = |bytes: u64| pool_get(k, bytes);
    let f32b = |n: usize| nb((n * 4) as u64);
    // inp_gate_w / proj_w / postnorm_w are RESIDENT; `pli_buf` holds this token's pli
    // (computed on the GPU by encode_gemma4_pli), read at `pli_offset` for this layer.
    let gate_buf = f32b(ple_dim);
    let gated_buf = f32b(ple_dim);
    let proj_out = f32b(hidden);
    let pnv_buf = f32b(hidden);
    let summed_buf = f32b(hidden);
    let ig_scalar = nb(8);
    let pj_scalar = nb(8);
    let geglu_n = nb(4);
    let rms_scalar = nb(8);
    let resid_n = nb(4);
    let scale_scalar = nb(8);

    unsafe {
        let set2 = |buf: &Buffer, a: u32, b: u32| {
            let p = buf.contents() as *mut u32;
            *p = a;
            *p.add(1) = b;
        };
        set2(&ig_scalar, hidden as u32, ple_dim as u32); // rows, cols
        set2(&pj_scalar, ple_dim as u32, hidden as u32);
        *(geglu_n.contents() as *mut u32) = ple_dim as u32;
        let r = rms_scalar.contents() as *mut u8;
        *(r as *mut u32) = hidden as u32;
        *(r.add(4) as *mut f32) = eps;
        *(resid_n.contents() as *mut u32) = hidden as u32;
        let s = scale_scalar.contents() as *mut u8;
        *(s as *mut u32) = hidden as u32;
        *(s.add(4) as *mut f32) = output_scale;
    }

    encode_linear_transposed_f32(e, k, h_buf, inp_gate_w, &gate_buf, &ig_scalar, ple_dim);
    // gelu(gate) * pli — pli read from the resident pli_buf at this layer's offset.
    e.set_compute_pipeline_state(&k.gelu_mul_pipeline);
    e.set_buffer(0, Some(&gate_buf), 0);
    e.set_buffer(1, Some(pli_buf), (pli_offset * 4) as u64);
    e.set_buffer(2, Some(&gated_buf), 0);
    e.set_buffer(3, Some(&geglu_n), 0);
    dispatch_1d(e, &k.gelu_mul_pipeline, ple_dim);
    encode_linear_transposed_f32(e, k, &gated_buf, proj_w, &proj_out, &pj_scalar, hidden);
    encode_rms_norm_f32(e, k, &proj_out, postnorm_w, &pnv_buf, &rms_scalar);
    encode_binary(
        e,
        &k.residual_add_pipeline,
        h_buf,
        &pnv_buf,
        &summed_buf,
        &resid_n,
        hidden,
    );
    encode_scale_f32(e, k, &summed_buf, h_buf, &scale_scalar, hidden);

    keep.extend([
        gate_buf,
        gated_buf,
        proj_out,
        pnv_buf,
        summed_buf,
        ig_scalar,
        pj_scalar,
        geglu_n,
        rms_scalar,
        resid_n,
        scale_scalar,
    ]);
}

/// Encode the per-token Per-Layer-Embedding input `pli` ([n_layers][ple_dim],
/// flattened to `pli_buf`) into the encoder — the GPU version of the CPU pli prep,
/// which is the single biggest per-token CPU cost (a 110MB f32 matvec). The gemma
/// constants are folded into the resident inputs so no new kernel is needed:
///   `proj_buf`  = per_layer_model_proj * hidden^-0.5   (resident)
///   `projnorm_buf` = per_layer_proj_norm * FRAC_1_SQRT_2 (resident)
///   `ti_buf`    = per_layer_token_embd[token] * (ple_dim^0.5 * FRAC_1_SQRT_2) (per token)
/// then `pli = residual_add( rms_norm_per_head(proj_buf · h0, projnorm_buf), ti )`.
/// `ctx_buf`/`ctx_n_buf` are `ple_total`-sized scratch (reused across tokens).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_pli(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    h0_buf: &Buffer,
    proj_buf: &Buffer,
    projnorm_buf: &Buffer,
    ti_buf: &Buffer,
    ctx_buf: &Buffer,
    ctx_n_buf: &Buffer,
    pli_buf: &Buffer,
    hidden: usize,
    ple_total: usize,
    ple_dim: usize,
    n_layers: usize,
    eps: f32,
) {
    let nb = |bytes: u64| pool_get(k, bytes);
    let ctx_scalar = nb(8); // [rows=hidden, cols=ple_total] for the transposed GEMV
    let perhead = nb(12); // [head_dim=ple_dim, eps, use_weight=1]
    let resid_n = nb(4);
    unsafe {
        let p = ctx_scalar.contents() as *mut u32;
        *p = hidden as u32;
        *p.add(1) = ple_total as u32;
        let h = perhead.contents() as *mut u8;
        *(h as *mut u32) = ple_dim as u32;
        *(h.add(4) as *mut f32) = eps;
        *(h.add(8) as *mut u32) = 1;
        *(resid_n.contents() as *mut u32) = ple_total as u32;
    }
    encode_linear_transposed_f32(e, k, h0_buf, proj_buf, ctx_buf, &ctx_scalar, ple_total);
    encode_rms_norm_per_head(
        e,
        k,
        ctx_buf,
        projnorm_buf,
        ctx_n_buf,
        &perhead,
        n_layers,
        0,
    );
    encode_binary(
        e,
        &k.residual_add_pipeline,
        ctx_n_buf,
        ti_buf,
        pli_buf,
        &resid_n,
        ple_total,
    );
    keep.extend([ctx_scalar, perhead, resid_n]);
}

/// `output = cap * tanh(input / cap)` into the encoder (in-place safe). `scalar` =
/// [n: u32 @0, cap: f32 @4].
#[cfg(target_os = "macos")]
fn encode_soft_cap_f32(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    input: &Buffer,
    output: &Buffer,
    scalar: &Buffer,
    n: usize,
) {
    e.set_compute_pipeline_state(&k.soft_cap_pipeline);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(output), 0);
    e.set_buffer(2, Some(scalar), 0);
    e.set_buffer(3, Some(scalar), 4);
    dispatch_1d(e, &k.soft_cap_pipeline, n);
}

/// Encode the gemma4 logits head into the (serial) encoder, no readback: the final
/// hidden `h_buf` → `logits_buf` (`vocab`): `normf = rms_norm(h, output_norm)`;
/// `logits = token_embd · normf` (the tied embedding as the output projection — a
/// single Q8 wire GEMV over the vocab-major table); `logits = cap·tanh(logits/cap)`
/// in place when `softcap > 0`. Scratch goes into `keep`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_gemma4_head(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    h_buf: &Buffer,
    logits_buf: &Buffer,
    output_norm: &[f32],
    token_embd_w: &Buffer,
    vocab: usize,
    softcap: f32,
    eps: f32,
) {
    let hidden = output_norm.len();
    let bpr_hidden = hidden / 32;
    let nb = |bytes: u64| pool_get(k, bytes);
    let norm_w = nb((hidden * 4) as u64);
    let normf = nb((hidden * 4) as u64);
    let rms_scalar = nb(8);
    let mm_scalar = nb(8);
    let cap_scalar = nb(8);
    write_buffer_f32(&norm_w, output_norm);
    unsafe {
        let r = rms_scalar.contents() as *mut u8;
        *(r as *mut u32) = hidden as u32;
        *(r.add(4) as *mut f32) = eps;
        let m = mm_scalar.contents() as *mut u32;
        *m = bpr_hidden as u32;
        *m.add(1) = vocab as u32;
        let c = cap_scalar.contents() as *mut u8;
        *(c as *mut u32) = vocab as u32;
        *(c.add(4) as *mut f32) = softcap;
    }
    encode_rms_norm_f32(e, k, h_buf, &norm_w, &normf, &rms_scalar);
    encode_gemma4_q8_matmul(e, k, &normf, token_embd_w, logits_buf, &mm_scalar, vocab);
    if softcap.is_finite() && softcap > 0.0 {
        encode_soft_cap_f32(e, k, logits_buf, logits_buf, &cap_scalar, vocab);
    }
    keep.extend([norm_w, normf, rms_scalar, mm_scalar, cap_scalar]);
}

/// Opt-in: store the resident KV cache in f16 (half the KV bytes read per token at
/// context; K/V are converted on write and read back to f32 inside the attention kernel).
/// Changes numerics slightly vs the f32 cache.
#[cfg(target_os = "macos")]
fn kv16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_KV16")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Opt-in: tiled decode attention with online softmax (4 simdgroups per head, coalesced
/// K/V reads, no scores buffer). Requires head_dim % 32 == 0 and <= 128.
#[cfg(target_os = "macos")]
fn splitk_attention_enabled() -> bool {
    // Default ON; CAMELID_METAL_ATTN_SPLITK=0 falls back to the v2 kernel at all depths.
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("CAMELID_METAL_ATTN_SPLITK")
            .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
    })
}

#[cfg(target_os = "macos")]
fn attn2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_ATTN2")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Experiment flag: NSG=8 wire GEMV (256 threads/TG).
#[cfg(target_os = "macos")]
fn wire_nsg8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_METAL_WIRE_NSG8")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Test-only direct driver for the tiled (v2) decode attention pipeline.
#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn try_attention_v2_for_test(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    positions: usize,
    scale: f32,
) -> Option<Vec<f32>> {
    let kernel = metal_linear_kernel()?;
    let device = &kernel.device;
    let opts = MTLResourceOptions::StorageModeShared;
    let q = device.new_buffer(std::mem::size_of_val(query) as u64, opts);
    let k = device.new_buffer(std::mem::size_of_val(keys) as u64, opts);
    let v = device.new_buffer(std::mem::size_of_val(values) as u64, opts);
    let out = device.new_buffer((n_heads * head_dim * 4) as u64, opts);
    let scalar = device.new_buffer(32, opts);
    write_buffer_f32(&q, query);
    write_buffer_f32(&k, keys);
    write_buffer_f32(&v, values);
    unsafe {
        let p = scalar.contents() as *mut u32;
        *p = n_heads as u32;
        *p.add(1) = head_dim as u32;
        *p.add(2) = positions as u32;
        *p.add(3) = (n_heads / n_kv_heads) as u32;
        *(p.add(4) as *mut f32) = scale;
        *p.add(5) = head_dim as u32; // position_stride (contiguous)
        *p.add(6) = (positions * head_dim) as u32; // kv_head_stride
        *p.add(7) = 0; // kv_base_offset
    }
    let cb = kernel.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    e.set_compute_pipeline_state(&kernel.attention_decode_v2_pipeline);
    e.set_buffer(0, Some(&q), 0);
    e.set_buffer(1, Some(&k), 0);
    e.set_buffer(2, Some(&v), 0);
    e.set_buffer(4, Some(&out), 0);
    for i in 0..8u64 {
        e.set_buffer(5 + i, Some(&scalar), i * 4);
    }
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut result = vec![0.0f32; n_heads * head_dim];
    read_buffer_f32(&out, &mut result);
    Some(result)
}

/// Test-only driver for the split-K kv16 decode attention (split kernel + merge),
/// reading half K/V exactly as the resident decode path reads the mirrors at depth.
#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn try_attention_splitk_kv16_for_test(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    positions: usize,
    scale: f32,
    direct: bool,
) -> Option<Vec<f32>> {
    let kernel = metal_linear_kernel()?;
    let device = &kernel.device;
    let opts = MTLResourceOptions::StorageModeShared;
    let q = device.new_buffer(std::mem::size_of_val(query) as u64, opts);
    let k = device.new_buffer((keys.len() * 2) as u64, opts);
    let v = device.new_buffer((values.len() * 2) as u64, opts);
    let out = device.new_buffer((n_heads * head_dim * 4) as u64, opts);
    let scalar = device.new_buffer(32, opts);
    write_buffer_f32(&q, query);
    unsafe {
        let kp = k.contents() as *mut u16;
        let vp = v.contents() as *mut u16;
        for (i, x) in keys.iter().enumerate() {
            *kp.add(i) = f32_to_f16_bits(*x);
        }
        for (i, x) in values.iter().enumerate() {
            *vp.add(i) = f32_to_f16_bits(*x);
        }
    }
    let n_splits = positions.div_ceil(64).clamp(2, 64);
    let partials = device.new_buffer((n_heads * n_splits * (head_dim + 2) * 4) as u64, opts);
    let splits_scalar = device.new_buffer(4, opts);
    unsafe {
        *(splits_scalar.contents() as *mut u32) = n_splits as u32;
        let p = scalar.contents() as *mut u32;
        *p = n_heads as u32;
        *p.add(1) = head_dim as u32;
        *p.add(2) = positions as u32;
        *p.add(3) = (n_heads / n_kv_heads) as u32;
        *(p.add(4) as *mut f32) = scale;
        *p.add(5) = head_dim as u32; // position_stride (contiguous)
        *p.add(6) = (positions * head_dim) as u32; // kv_head_stride
        *p.add(7) = 0; // kv_base_offset
    }
    let cb = kernel.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    e.set_compute_pipeline_state(if direct {
        &kernel.attention_decode_splitk_kv16_direct_pipeline
    } else {
        &kernel.attention_decode_splitk_kv16_pipeline
    });
    e.set_buffer(0, Some(&q), 0);
    e.set_buffer(1, Some(&k), 0);
    e.set_buffer(2, Some(&v), 0);
    e.set_buffer(3, Some(&partials), 0);
    for i in 0..8u64 {
        e.set_buffer(5 + i, Some(&scalar), i * 4);
    }
    e.set_buffer(13, Some(&splits_scalar), 0);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: n_kv_heads as u64,
            height: n_splits as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    e.set_compute_pipeline_state(&kernel.attention_decode_splitk_merge_pipeline);
    e.set_buffer(0, Some(&partials), 0);
    e.set_buffer(1, Some(&out), 0);
    e.set_buffer(2, Some(&scalar), 4); // head_dim
    e.set_buffer(3, Some(&splits_scalar), 0);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut result = vec![0.0f32; n_heads * head_dim];
    read_buffer_f32(&out, &mut result);
    Some(result)
}

/// Test-only direct driver for the K-split GEMV pipeline (bypasses the env gate and the
/// weight-buffer cache so parity tests control exactly what runs).
#[cfg(all(test, target_os = "macos"))]
fn try_q8_0_ksplit_linear_for_test(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let device = &kernel.device;
    let opts = MTLResourceOptions::StorageModeShared;
    let scales = device.new_buffer(std::mem::size_of_val(input_scales) as u64, opts);
    let quants = device.new_buffer(input_quants.len() as u64, opts);
    let weight = device.new_buffer(weight_blocks.len() as u64, opts);
    let out = device.new_buffer(std::mem::size_of_val(&*output) as u64, opts);
    let scalar = device.new_buffer(8, opts);
    write_buffer_f32(&scales, input_scales);
    write_buffer_i8(&quants, input_quants);
    write_buffer_u8(&weight, weight_blocks);
    unsafe {
        let s = scalar.contents() as *mut u32;
        *s = blocks_per_row as u32;
        *s.add(1) = rows as u32;
    }
    let command_buffer = kernel.queue.new_command_buffer();
    let e = command_buffer.new_compute_command_encoder();
    e.set_compute_pipeline_state(&kernel.q8_0_block_ksplit_pipeline);
    e.set_buffer(0, Some(&scales), 0);
    e.set_buffer(1, Some(&quants), 0);
    e.set_buffer(2, Some(&weight), 0);
    e.set_buffer(3, Some(&out), 0);
    e.set_buffer(4, Some(&scalar), 0);
    e.set_buffer(5, Some(&scalar), 4);
    e.set_threadgroup_memory_length(0, 2 * 32 * 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(2),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    e.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_buffer_f32(&out, output);
    true
}

#[cfg(target_os = "macos")]
/// `encode_binary` with explicit element-count offset (for batched prefill buffers).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_binary_off(
    e: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    n_buf: &Buffer,
    n_off: u64,
    n: usize,
) {
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(a), 0);
    e.set_buffer(1, Some(b), 0);
    e.set_buffer(2, Some(out), 0);
    e.set_buffer(3, Some(n_buf), n_off);
    dispatch_1d(e, pipeline, n);
}

#[cfg(target_os = "macos")]
fn encode_binary(
    e: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    n_buf: &Buffer,
    n: usize,
) {
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(a), 0);
    e.set_buffer(1, Some(b), 0);
    e.set_buffer(2, Some(out), 0);
    e.set_buffer(3, Some(n_buf), 0);
    dispatch_1d(e, pipeline, n);
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_rope(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    data: &Buffer,
    cos_t: &Buffer,
    sin_t: &Buffer,
    scalar: &Buffer,
    head_count: usize,
    half_rope: usize,
    // Byte offset for the (in-place) data buffer, and a shared offset for the cos/sin
    // tables, so RoPE can run on a single verify row at position `base+i` (cos/sin are
    // position-major with stride half_rope). All in-kernel indexing is relative to the
    // bound offsets, so the math is unchanged. Default 0 for all pre-verify callers.
    data_off: u64,
    table_off: u64,
) {
    e.set_compute_pipeline_state(&k.rope_rotate_pipeline);
    e.set_buffer(0, Some(data), data_off);
    e.set_buffer(1, Some(cos_t), table_off);
    e.set_buffer(2, Some(sin_t), table_off);
    e.set_buffer(3, Some(scalar), 0); // head_count
    e.set_buffer(4, Some(scalar), 4); // head_dim
    e.set_buffer(5, Some(scalar), 8); // half_rope
    e.set_buffer(6, Some(scalar), 12); // pairing
    dispatch_1d(e, &k.rope_rotate_pipeline, head_count * half_rope);
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_attention(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    kv16_mirrors: Option<(&Buffer, &Buffer)>,
    scores: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    // Byte offsets for the query input and the context output buffers, so attention can
    // run on a single verify row (query/out at row `i`) while the K/V cache is shared.
    // The kernels index heads relative to the bound offset, so the routing/math are
    // unchanged. Default 0 for all pre-verify callers (single-token query/out at row 0).
    query_off: u64,
    out_off: u64,
) {
    // Tiled kernel (4 simdgroups/head, online softmax, no scores buffer) when enabled and
    // the head geometry allows; otherwise the one-simdgroup-per-head fallback.
    let v2 = attn2_enabled() && head_dim.is_multiple_of(32) && head_dim <= 128;
    // Split-K flash decode for deeper contexts: the v2 kernel's one-threadgroup-per-head
    // grid leaves the GPU mostly idle while each simdgroup walks a long position range
    // serially, and GQA re-reads every K/V row once per query head. The split-K kernel
    // covers (kv_heads x splits) threadgroups with the K/V tile staged once per group.
    // Below the threshold the fixed scratch/merge cost outweighs the win.
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(0);
    let splitk = v2
        && !kv16_enabled()
        && splitk_attention_enabled()
        && (1..=4).contains(&group)
        && position_count >= 128;
    if splitk {
        let n_splits = position_count.div_ceil(64).clamp(2, 64);
        let partials = pool_get(k, (n_heads * n_splits * (head_dim + 2) * 4) as u64);
        let splits_scalar = pool_get(k, 4);
        unsafe {
            *(splits_scalar.contents() as *mut u32) = n_splits as u32;
        }
        // Half-mirror reads halve the dominant KV traffic at depth; opt out with
        // CAMELID_METAL_ATTN_SPLITK_KV16=0 to keep the f32 split-K reads.
        let use_mirrors = kv16_mirrors.is_some()
            && !std::env::var("CAMELID_METAL_ATTN_SPLITK_KV16")
                .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"));
        if use_mirrors {
            let (mk, mv) = kv16_mirrors.expect("checked is_some above");
            // Direct-read variant for head_dim 128: no threadgroup staging or
            // barriers -> more resident threadgroups; GQA re-reads hit the SLC.
            // Probe @7.7k positions: 11.4ms vs 14.3ms staged (and better at
            // every shallower depth). Staged kernel remains the general fallback.
            if head_dim == 128 {
                e.set_compute_pipeline_state(&k.attention_decode_splitk_kv16_direct_pipeline);
            } else {
                e.set_compute_pipeline_state(&k.attention_decode_splitk_kv16_pipeline);
            }
            e.set_buffer(0, Some(query), query_off);
            e.set_buffer(1, Some(mk), 0);
            e.set_buffer(2, Some(mv), 0);
        } else {
            e.set_compute_pipeline_state(&k.attention_decode_splitk_pipeline);
            e.set_buffer(0, Some(query), query_off);
            e.set_buffer(1, Some(keys), 0);
            e.set_buffer(2, Some(values), 0);
        }
        e.set_buffer(3, Some(&partials), 0);
        e.set_buffer(5, Some(scalar), 0); // n_heads
        e.set_buffer(6, Some(scalar), 4); // head_dim
        e.set_buffer(7, Some(scalar), 8); // position_count
        e.set_buffer(8, Some(scalar), 12); // group
        e.set_buffer(9, Some(scalar), 16); // scale (f32)
        e.set_buffer(10, Some(scalar), 20); // position_stride
        e.set_buffer(11, Some(scalar), 24); // kv_head_stride
        e.set_buffer(12, Some(scalar), 28); // kv_base_offset
        e.set_buffer(13, Some(&splits_scalar), 0);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: n_kv_heads as u64,
                height: n_splits as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        e.set_compute_pipeline_state(&k.attention_decode_splitk_merge_pipeline);
        e.set_buffer(0, Some(&partials), 0);
        e.set_buffer(1, Some(out), out_off);
        e.set_buffer(2, Some(scalar), 4); // head_dim
        e.set_buffer(3, Some(&splits_scalar), 0);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: n_heads as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        keep.push(partials);
        keep.push(splits_scalar);
        return;
    }
    let attn_pipeline = match (v2, kv16_enabled()) {
        (true, true) => &k.attention_decode_v2_kv16_pipeline,
        (true, false) => &k.attention_decode_v2_pipeline,
        (false, true) => &k.attention_decode_kv16_pipeline,
        (false, false) => &k.attention_decode_pipeline,
    };
    e.set_compute_pipeline_state(attn_pipeline);
    e.set_buffer(0, Some(query), query_off);
    e.set_buffer(1, Some(keys), 0);
    e.set_buffer(2, Some(values), 0);
    if !v2 {
        e.set_buffer(3, Some(scores), 0);
    }
    e.set_buffer(4, Some(out), out_off);
    e.set_buffer(5, Some(scalar), 0); // n_heads
    e.set_buffer(6, Some(scalar), 4); // head_dim
    e.set_buffer(7, Some(scalar), 8); // position_count
    e.set_buffer(8, Some(scalar), 12); // group
    e.set_buffer(9, Some(scalar), 16); // scale (f32)
    e.set_buffer(10, Some(scalar), 20); // position_stride
    e.set_buffer(11, Some(scalar), 24); // kv_head_stride
    e.set_buffer(12, Some(scalar), 28); // kv_base_offset
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: if v2 { 128 } else { 32 },
            height: 1,
            depth: 1,
        },
    );
}

/// WIN2METAL Phase 4 — tree twin of `encode_attention`. Routes (v2 / split-K kv16 /
/// split-K kv16 direct / f32) EXACTLY like `encode_attention` at the same `position_count`,
/// but dispatches the `*_tree` pipelines and binds the per-node `tail_slots` buffer at
/// buffer(14) and the prefix-length `base` at buffer(15). For a LINEAR tree (`tail_slots ==
/// [base..base+i]`) the slot indirection is the identity, so this is byte-identical to
/// `encode_attention`.
///
/// `tail_slots_buf` holds this node's ancestor draft cache slots (increasing, length
/// `position_count - base`); `base_buf` holds `base` as a single u32. The split-K branch is
/// mirror-only (the f32 split-K opt-out kernel is not cloned): it always reads the f16
/// mirrors when present, matching the linear path under the default (mirror) config.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_attention_tree(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    kv16_mirrors: Option<(&Buffer, &Buffer)>,
    scores: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    query_off: u64,
    out_off: u64,
    base_buf: &Buffer,
    tail_slots_buf: &Buffer,
) {
    let v2 = attn2_enabled() && head_dim.is_multiple_of(32) && head_dim <= 128;
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(0);
    let splitk = v2
        && !kv16_enabled()
        && splitk_attention_enabled()
        && (1..=4).contains(&group)
        && position_count >= 128;
    if splitk {
        let n_splits = position_count.div_ceil(64).clamp(2, 64);
        let partials = pool_get(k, (n_heads * n_splits * (head_dim + 2) * 4) as u64);
        let splits_scalar = pool_get(k, 4);
        unsafe {
            *(splits_scalar.contents() as *mut u32) = n_splits as u32;
        }
        // Tree split-K is mirror-only: read the f16 mirrors whenever present (the f32
        // split-K kernel has no tree clone). Under the default config the linear path
        // also reads the mirrors, so the two stay byte-identical.
        let (mk, mv) = kv16_mirrors.expect("tree split-K requires f16 mirrors");
        if head_dim == 128 {
            e.set_compute_pipeline_state(&k.attention_decode_splitk_kv16_direct_tree_pipeline);
        } else {
            e.set_compute_pipeline_state(&k.attention_decode_splitk_kv16_tree_pipeline);
        }
        e.set_buffer(0, Some(query), query_off);
        e.set_buffer(1, Some(mk), 0);
        e.set_buffer(2, Some(mv), 0);
        e.set_buffer(3, Some(&partials), 0);
        e.set_buffer(5, Some(scalar), 0); // n_heads
        e.set_buffer(6, Some(scalar), 4); // head_dim
        e.set_buffer(7, Some(scalar), 8); // position_count
        e.set_buffer(8, Some(scalar), 12); // group
        e.set_buffer(9, Some(scalar), 16); // scale (f32)
        e.set_buffer(10, Some(scalar), 20); // position_stride
        e.set_buffer(11, Some(scalar), 24); // kv_head_stride
        e.set_buffer(12, Some(scalar), 28); // kv_base_offset
        e.set_buffer(13, Some(&splits_scalar), 0);
        e.set_buffer(14, Some(tail_slots_buf), 0);
        e.set_buffer(15, Some(base_buf), 0);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: n_kv_heads as u64,
                height: n_splits as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        // The merge kernel is slot-agnostic (operates on partials) — reuse it verbatim.
        e.set_compute_pipeline_state(&k.attention_decode_splitk_merge_pipeline);
        e.set_buffer(0, Some(&partials), 0);
        e.set_buffer(1, Some(out), out_off);
        e.set_buffer(2, Some(scalar), 4); // head_dim
        e.set_buffer(3, Some(&splits_scalar), 0);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: n_heads as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        keep.push(partials);
        keep.push(splits_scalar);
        return;
    }
    // Non-split path: v2 tiled (no scores buffer) or the f32 one-simdgroup fallback.
    let attn_pipeline = if v2 {
        &k.attention_decode_v2_tree_pipeline
    } else {
        &k.attention_decode_f32_tree_pipeline
    };
    e.set_compute_pipeline_state(attn_pipeline);
    e.set_buffer(0, Some(query), query_off);
    e.set_buffer(1, Some(keys), 0);
    e.set_buffer(2, Some(values), 0);
    if !v2 {
        e.set_buffer(3, Some(scores), 0);
    }
    e.set_buffer(4, Some(out), out_off);
    e.set_buffer(5, Some(scalar), 0); // n_heads
    e.set_buffer(6, Some(scalar), 4); // head_dim
    e.set_buffer(7, Some(scalar), 8); // position_count
    e.set_buffer(8, Some(scalar), 12); // group
    e.set_buffer(9, Some(scalar), 16); // scale (f32)
    e.set_buffer(10, Some(scalar), 20); // position_stride
    e.set_buffer(11, Some(scalar), 24); // kv_head_stride
    e.set_buffer(12, Some(scalar), 28); // kv_base_offset
    e.set_buffer(14, Some(tail_slots_buf), 0);
    e.set_buffer(15, Some(base_buf), 0);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: if v2 { 128 } else { 32 },
            height: 1,
            depth: 1,
        },
    );
}

/// Encode the FFN block op-chain into `cb` with no commit/readback: reads `in_buf`, writes
/// the residual sum into `out_buf` (rms_norm -> quantize -> gate & up matmul -> silu_mul ->
/// quantize -> down matmul -> residual add with `in_buf`). The `gate_w`/`up_w`/`down_w` Q8_0
/// weight buffers are caller-owned (so callers can keep them resident across tokens); this
/// allocates only its own scratch buffers and pushes them into `keep` so they outlive the
/// command buffer. Dimensions come from `ffn_norm.len()` and `ffn_dim`; callers must
/// pre-validate (see `try_ffn_block_resident`).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_ffn_block(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    in_buf: &Buffer,
    out_buf: &Buffer,
    ffn_norm: &[f32],
    eps: f32,
    gate_w: &Buffer,
    up_w: &Buffer,
    down_w: &Buffer,
    ffn_dim: usize,
) {
    let hidden = ffn_norm.len();
    let bpr_hidden = hidden / 32;
    let bpr_ffn = ffn_dim / 32;
    let nb = |bytes: u64| pool_get(k, bytes);
    let norm_w_buf = nb((hidden * 4) as u64);
    let rms_scalar = nb(8);
    let scales1 = nb((bpr_hidden * 4) as u64);
    let quants1 = nb(hidden as u64);
    let gate_buf = nb((ffn_dim * 4) as u64);
    let up_buf = nb((ffn_dim * 4) as u64);
    let gateup_scalar = nb(8);
    let scales2 = nb((bpr_ffn * 4) as u64);
    let quants2 = nb(ffn_dim as u64);
    let nblocks2 = nb(4);
    let down_buf = nb((hidden * 4) as u64);
    let down_scalar = nb(8);
    let resid_n = nb(4);

    write_buffer_f32(&norm_w_buf, ffn_norm);
    unsafe {
        let p = rms_scalar.contents() as *mut u8;
        *(p as *mut u32) = hidden as u32;
        *(p.add(4) as *mut f32) = eps;
        let g = gateup_scalar.contents() as *mut u32;
        *g = bpr_hidden as u32;
        *g.add(1) = ffn_dim as u32;
        *(nblocks2.contents() as *mut u32) = bpr_ffn as u32;
        let d = down_scalar.contents() as *mut u32;
        *d = bpr_ffn as u32;
        *d.add(1) = hidden as u32;
        *(resid_n.contents() as *mut u32) = hidden as u32;
    }

    if f32y_gemv_enabled() {
        // f32-activation chain: no quantize anywhere — norm and silu outputs feed the
        // float-FMA GEMV directly.
        let normf = nb((hidden * 4) as u64);
        let siluf = nb((ffn_dim * 4) as u64);
        let silu_n = nb(4);
        unsafe {
            *(silu_n.contents() as *mut u32) = ffn_dim as u32;
        }
        encode_rms_norm_f32(e, k, in_buf, &norm_w_buf, &normf, &rms_scalar);
        encode_q8_matmul_f32y(e, k, &normf, gate_w, &gate_buf, &gateup_scalar, ffn_dim);
        encode_q8_matmul_f32y(e, k, &normf, up_w, &up_buf, &gateup_scalar, ffn_dim);
        encode_binary(
            e,
            &k.silu_mul_pipeline,
            &gate_buf,
            &up_buf,
            &siluf,
            &silu_n,
            ffn_dim,
        );
        encode_q8_matmul_f32y(e, k, &siluf, down_w, &down_buf, &down_scalar, hidden);
        keep.extend([normf, siluf, silu_n]);
    } else {
        encode_rms_norm_quantize(e, k, in_buf, &norm_w_buf, &scales1, &quants1, &rms_scalar);
        encode_q8_matmul(
            e,
            k,
            &scales1,
            &quants1,
            gate_w,
            &gate_buf,
            &gateup_scalar,
            ffn_dim,
        );
        encode_q8_matmul(
            e,
            k,
            &scales1,
            &quants1,
            up_w,
            &up_buf,
            &gateup_scalar,
            ffn_dim,
        );
        encode_silu_mul_quantize(
            e, k, &gate_buf, &up_buf, &scales2, &quants2, &nblocks2, bpr_ffn,
        );
        encode_q8_matmul(
            e,
            k,
            &scales2,
            &quants2,
            down_w,
            &down_buf,
            &down_scalar,
            hidden,
        );
    }
    encode_binary(
        e,
        &k.residual_add_pipeline,
        in_buf,
        &down_buf,
        out_buf,
        &resid_n,
        hidden,
    );

    keep.extend([
        norm_w_buf,
        rms_scalar,
        scales1,
        quants1,
        gate_buf,
        up_buf,
        gateup_scalar,
        scales2,
        quants2,
        nblocks2,
        down_buf,
        down_scalar,
        resid_n,
    ]);
}

/// Encode the attention block op-chain into `cb` with no commit/readback: reads `in_buf`,
/// writes the residual sum into `out_buf` (rms_norm -> quantize -> q/k/v matmul -> RoPE(q,k)
/// -> blit current k/v into `cache_k_buf`/`cache_v_buf` at slot `write_position` -> decode
/// attention over the first `position_count` slots -> quantize -> o matmul -> residual add
/// with `in_buf`). The weight buffers AND the K/V cache buffers are caller-owned, so callers
/// can keep them resident across tokens (a persistent cache simply grows by one slot per
/// token). The cache buffers are laid out `[kv_head][max_positions][head_dim]`; this allocates
/// only its own scratch buffers and pushes them into `keep` so they outlive the command
/// buffer. Dimensions come from `attn_norm.len()` and the head params; callers must
/// pre-validate (including that `write_position < position_count <= max_positions`).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_attention_block(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    in_buf: &Buffer,
    out_buf: &Buffer,
    attn_norm: &[f32],
    eps: f32,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    q_w_buf: &Buffer,
    k_w_buf: &Buffer,
    v_w_buf: &Buffer,
    o_w_buf: &Buffer,
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k_buf: &Buffer,
    cache_v_buf: &Buffer,
    kv16_mirrors: Option<(&Buffer, &Buffer)>,
    max_positions: usize,
    write_position: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    scale: f32,
    split_half_pairing: bool,
) {
    let hidden = attn_norm.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let half_rope = cos_t.len();
    let bpr_hidden = hidden / 32;
    let bpr_q = q_dim / 32;
    let group = (n_heads / n_kv_heads) as u32;
    let nb = |bytes: u64| pool_get(k, bytes);
    let f32b = |n: usize| nb((n * 4) as u64);
    let norm_w_buf = f32b(hidden);
    let rms_scalar = nb(8);
    let scales_norm = f32b(bpr_hidden);
    let quants_norm = nb(hidden as u64);
    let query_buf = f32b(q_dim);
    let key_buf = f32b(kv_dim);
    let val_buf = f32b(kv_dim);
    let q_mm_scalar = nb(8);
    let kv_mm_scalar = nb(8);
    let cos_buf = f32b(half_rope);
    let sin_buf = f32b(half_rope);
    let rope_q_scalar = nb(16);
    let rope_k_scalar = nb(16);
    let scores_buf = f32b(n_heads * position_count);
    let ctx_buf = f32b(q_dim);
    let attn_scalar = nb(32);
    let scales_ctx = f32b(bpr_q);
    let quants_ctx = nb(q_dim as u64);
    let nblocks_ctx = nb(4);
    let o_buf = f32b(hidden);
    let o_mm_scalar = nb(8);
    let resid_n = nb(4);

    write_buffer_f32(&norm_w_buf, attn_norm);
    write_buffer_f32(&cos_buf, cos_t);
    write_buffer_f32(&sin_buf, sin_t);
    unsafe {
        let p = rms_scalar.contents() as *mut u8;
        *(p as *mut u32) = hidden as u32;
        *(p.add(4) as *mut f32) = eps;
        let q = q_mm_scalar.contents() as *mut u32;
        *q = bpr_hidden as u32;
        *q.add(1) = q_dim as u32;
        let kv = kv_mm_scalar.contents() as *mut u32;
        *kv = bpr_hidden as u32;
        *kv.add(1) = kv_dim as u32;
        let set_rope = |buf: &Buffer, hc: usize| {
            let r = buf.contents() as *mut u32;
            *r = hc as u32;
            *r.add(1) = head_dim as u32;
            *r.add(2) = half_rope as u32;
            *r.add(3) = u32::from(split_half_pairing);
        };
        set_rope(&rope_q_scalar, n_heads);
        set_rope(&rope_k_scalar, n_kv_heads);
        let a = attn_scalar.contents() as *mut u8;
        *(a as *mut u32) = n_heads as u32;
        *(a.add(4) as *mut u32) = head_dim as u32;
        *(a.add(8) as *mut u32) = position_count as u32;
        *(a.add(12) as *mut u32) = group;
        *(a.add(16) as *mut f32) = scale;
        // Per-layer cache [kv_head][max_positions][head_dim]: position stride is one head_dim,
        // head stride spans the full allocated position capacity, no base offset.
        *(a.add(20) as *mut u32) = head_dim as u32;
        *(a.add(24) as *mut u32) = (max_positions * head_dim) as u32;
        *(a.add(28) as *mut u32) = 0u32;
        *(nblocks_ctx.contents() as *mut u32) = bpr_q as u32;
        let o = o_mm_scalar.contents() as *mut u32;
        *o = bpr_q as u32;
        *o.add(1) = hidden as u32;
        *(resid_n.contents() as *mut u32) = hidden as u32;
    }

    let normf_attn = if f32y_gemv_enabled() {
        let normf = nb((hidden * 4) as u64);
        encode_rms_norm_f32(e, k, in_buf, &norm_w_buf, &normf, &rms_scalar);
        encode_q8_matmul_f32y(e, k, &normf, q_w_buf, &query_buf, &q_mm_scalar, q_dim);
        encode_q8_matmul_f32y(e, k, &normf, k_w_buf, &key_buf, &kv_mm_scalar, kv_dim);
        encode_q8_matmul_f32y(e, k, &normf, v_w_buf, &val_buf, &kv_mm_scalar, kv_dim);
        Some(normf)
    } else {
        encode_rms_norm_quantize(
            e,
            k,
            in_buf,
            &norm_w_buf,
            &scales_norm,
            &quants_norm,
            &rms_scalar,
        );
        encode_q8_matmul(
            e,
            k,
            &scales_norm,
            &quants_norm,
            q_w_buf,
            &query_buf,
            &q_mm_scalar,
            q_dim,
        );
        encode_q8_matmul(
            e,
            k,
            &scales_norm,
            &quants_norm,
            k_w_buf,
            &key_buf,
            &kv_mm_scalar,
            kv_dim,
        );
        encode_q8_matmul(
            e,
            k,
            &scales_norm,
            &quants_norm,
            v_w_buf,
            &val_buf,
            &kv_mm_scalar,
            kv_dim,
        );
        None
    };
    // Qwen3 QK-norm: per-head RMSNorm on Q/K after the projection and BEFORE RoPE.
    // In-place is safe — the per-head kernel reduces the sum-of-squares via
    // threadgroup memory before the (per-index) write. No-op for Llama rows.
    if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
        let qnorm_w_buf = f32b(head_dim);
        let knorm_w_buf = f32b(head_dim);
        let perhead_scalar = nb(12);
        write_buffer_f32(&qnorm_w_buf, qn);
        write_buffer_f32(&knorm_w_buf, kn);
        unsafe {
            let p = perhead_scalar.contents() as *mut u8;
            *(p as *mut u32) = head_dim as u32;
            *(p.add(4) as *mut f32) = eps;
            *(p.add(8) as *mut u32) = 1; // use_weight
        }
        encode_rms_norm_per_head(
            e,
            k,
            &query_buf,
            &qnorm_w_buf,
            &query_buf,
            &perhead_scalar,
            n_heads,
            0,
        );
        encode_rms_norm_per_head(
            e,
            k,
            &key_buf,
            &knorm_w_buf,
            &key_buf,
            &perhead_scalar,
            n_kv_heads,
            0,
        );
    }
    encode_rope(
        e,
        k,
        &query_buf,
        &cos_buf,
        &sin_buf,
        &rope_q_scalar,
        n_heads,
        half_rope,
        0,
        0,
    );
    encode_rope(
        e,
        k,
        &key_buf,
        &cos_buf,
        &sin_buf,
        &rope_k_scalar,
        n_kv_heads,
        half_rope,
        0,
        0,
    );
    // Write the current token's (roped) K and (raw) V into the cache at `write_position` via
    // a compute scatter, so the whole token stays inside ONE compute command encoder (encoder
    // boundaries force full GPU serialization; within one serial encoder Metal's hazard
    // tracking orders dependent dispatches and overlaps independent ones).
    let scatter_scalar = nb(16);
    unsafe {
        let p = scatter_scalar.contents() as *mut u32;
        *p = head_dim as u32;
        *p.add(1) = max_positions as u32;
        *p.add(2) = write_position as u32;
        *p.add(3) = kv_dim as u32;
    }
    let scatter_pipeline = if kv16_enabled() {
        &k.kv_scatter_kv16_pipeline
    } else {
        &k.kv_scatter_pipeline
    };
    e.set_compute_pipeline_state(scatter_pipeline);
    e.set_buffer(0, Some(&key_buf), 0);
    e.set_buffer(1, Some(&val_buf), 0);
    e.set_buffer(2, Some(cache_k_buf), 0);
    e.set_buffer(3, Some(cache_v_buf), 0);
    e.set_buffer(4, Some(&scatter_scalar), 0);
    e.set_buffer(5, Some(&scatter_scalar), 4);
    e.set_buffer(6, Some(&scatter_scalar), 8);
    e.set_buffer(7, Some(&scatter_scalar), 12);
    if !kv16_enabled() {
        // Dual-write the half mirrors when the session maintains them; otherwise bind
        // placeholders with the flag at 0 (the kernel never dereferences them).
        let kv16_write = nb(4);
        unsafe {
            *(kv16_write.contents() as *mut u32) = u32::from(kv16_mirrors.is_some());
        }
        let (mk, mv) = kv16_mirrors.unwrap_or((&scatter_scalar, &scatter_scalar));
        e.set_buffer(8, Some(mk), 0);
        e.set_buffer(9, Some(mv), 0);
        e.set_buffer(10, Some(&kv16_write), 0);
        keep.push(kv16_write);
    }
    dispatch_1d(e, scatter_pipeline, kv_dim);
    encode_attention(
        e,
        k,
        keep,
        &query_buf,
        cache_k_buf,
        cache_v_buf,
        kv16_mirrors,
        &scores_buf,
        &ctx_buf,
        &attn_scalar,
        n_heads,
        n_kv_heads,
        head_dim,
        position_count,
        0,
        0,
    );
    if f32y_gemv_enabled() {
        encode_q8_matmul_f32y(e, k, &ctx_buf, o_w_buf, &o_buf, &o_mm_scalar, hidden);
    } else {
        encode_quantize(
            e,
            k,
            &ctx_buf,
            &scales_ctx,
            &quants_ctx,
            &nblocks_ctx,
            bpr_q,
        );
        encode_q8_matmul(
            e,
            k,
            &scales_ctx,
            &quants_ctx,
            o_w_buf,
            &o_buf,
            &o_mm_scalar,
            hidden,
        );
    }
    if let Some(normf) = normf_attn {
        keep.push(normf);
    }
    encode_binary(
        e,
        &k.residual_add_pipeline,
        in_buf,
        &o_buf,
        out_buf,
        &resid_n,
        hidden,
    );

    keep.extend([
        norm_w_buf,
        rms_scalar,
        scales_norm,
        quants_norm,
        query_buf,
        key_buf,
        val_buf,
        q_mm_scalar,
        kv_mm_scalar,
        cos_buf,
        sin_buf,
        rope_q_scalar,
        rope_k_scalar,
        scores_buf,
        ctx_buf,
        attn_scalar,
        scales_ctx,
        quants_ctx,
        nblocks_ctx,
        o_buf,
        o_mm_scalar,
        resid_n,
        scatter_scalar,
    ]);
}

/// Hand out a scratch buffer of at least `bytes` from the recycle pool, or allocate a
/// fresh one at the pool's power-of-two class size. Pool-derived buffers are owned by the
/// per-token `keep` vec and MUST come back via `pool_recycle` only after the command
/// buffer that referenced them has completed.
#[cfg(target_os = "macos")]
fn pool_get(k: &MetalLinearKernel, bytes: u64) -> Buffer {
    let class = bytes.max(32).next_power_of_two();
    if let Some(buf) = k
        .scratch_pool
        .lock()
        .unwrap()
        .get_mut(&class)
        .and_then(|v| v.pop())
    {
        return buf;
    }
    k.device
        .new_buffer(class, MTLResourceOptions::StorageModeShared)
}

/// Return settled scratch buffers to the pool (keyed by their class-sized length).
#[cfg(target_os = "macos")]
fn pool_recycle<I: IntoIterator<Item = Buffer>>(k: &MetalLinearKernel, bufs: I) {
    let mut pool = k.scratch_pool.lock().unwrap();
    for b in bufs {
        pool.entry(b.length()).or_default().push(b);
    }
}

/// Allocate a fresh GPU buffer for a Q8_0 weight-block slice and upload it. Used by the
/// standalone block helpers, which (unlike the resident decode forward) do not keep weights
/// cached across calls.
#[cfg(target_os = "macos")]
fn upload_weight_buffer(k: &MetalLinearKernel, weight_blocks: &[u8]) -> Buffer {
    let buf = k.device.new_buffer(
        weight_blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    write_buffer_u8(&buf, weight_blocks);
    buf
}

/// Allocate a fresh f32 GPU buffer sized to `cache` and upload it. Used by the standalone
/// block helpers to hand `encode_attention_block` a transient per-call KV cache buffer
/// (`max_positions == position_count`); the resident decode session passes persistent
/// buffers instead.
#[cfg(target_os = "macos")]
fn upload_cache_buffer(k: &MetalLinearKernel, cache: &[f32]) -> Buffer {
    let buf = k.device.new_buffer(
        (cache.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    write_buffer_f32(&buf, cache);
    buf
}

/// GPU-resident FFN block in a single command buffer (no CPU readback between ops):
/// rms_norm -> quantize -> gate & up matmul -> silu_mul -> quantize -> down matmul ->
/// residual add with the input. Weights are Q8_0 36-byte blocks (gate/up are
/// [ffn_dim x hidden/32], down is [hidden x ffn_dim/32]). Returns [hidden]; None if
/// Metal is unavailable. Bit-identical to running the standalone kernels in sequence.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_ffn_block_resident(
    input: &[f32],
    ffn_norm: &[f32],
    eps: f32,
    gate_weight_blocks: &[u8],
    up_weight_blocks: &[u8],
    down_weight_blocks: &[u8],
    ffn_dim: usize,
) -> Option<Vec<f32>> {
    let hidden = input.len();
    if hidden == 0
        || !hidden.is_multiple_of(32)
        || ffn_dim == 0
        || !ffn_dim.is_multiple_of(32)
        || ffn_norm.len() != hidden
    {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_ffn = ffn_dim / 32;
    if gate_weight_blocks.len() != ffn_dim * bpr_hidden * 36
        || up_weight_blocks.len() != ffn_dim * bpr_hidden * 36
        || down_weight_blocks.len() != hidden * bpr_ffn * 36
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let nb = |n: usize| {
        k.device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = nb(hidden);
    let out_buf = nb(hidden);
    write_buffer_f32(&in_buf, input);
    let gate_w = upload_weight_buffer(k, gate_weight_blocks);
    let up_w = upload_weight_buffer(k, up_weight_blocks);
    let down_w = upload_weight_buffer(k, down_weight_blocks);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_ffn_block(
        e, k, &mut keep, &in_buf, &out_buf, ffn_norm, eps, &gate_w, &up_w, &down_w, ffn_dim,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// GPU-resident attention block in a single command buffer (no CPU readback): rms_norm
/// -> quantize -> q/k/v matmul -> RoPE(q,k) -> write current k/v into the KV cache
/// buffers at `position` (blit) -> decode attention over the cache -> quantize -> o
/// matmul -> residual add with the input. `cache_k`/`cache_v` are laid out as
/// `[n_kv * position_count * head_dim]`; positions 0..position-1 are caller-filled, this
/// writes position `position_count-1`. cos/sin tables come from the CPU's RoPE math.
/// Returns the `[hidden]` output, or None if Metal is unavailable. Bit-identical to
/// running the standalone kernels.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_block_resident(
    input: &[f32],
    attn_norm: &[f32],
    eps: f32,
    q_weight_blocks: &[u8],
    k_weight_blocks: &[u8],
    v_weight_blocks: &[u8],
    o_weight_blocks: &[u8],
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    scale: f32,
    split_half_pairing: bool,
) -> Option<Vec<f32>> {
    let hidden = input.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let half_rope = cos_t.len();
    if hidden == 0
        || !hidden.is_multiple_of(32)
        || !q_dim.is_multiple_of(32)
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || n_heads == 0
        || n_kv_heads == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || position_count == 0
        || attn_norm.len() != hidden
        || sin_t.len() != half_rope
        || half_rope * 2 > head_dim
        || cache_k.len() != kv_dim * position_count
        || cache_v.len() != cache_k.len()
    {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_q = q_dim / 32;
    if q_weight_blocks.len() != q_dim * bpr_hidden * 36
        || k_weight_blocks.len() != kv_dim * bpr_hidden * 36
        || v_weight_blocks.len() != kv_dim * bpr_hidden * 36
        || o_weight_blocks.len() != hidden * bpr_q * 36
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let nb = |n: usize| {
        k.device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = nb(hidden);
    let out_buf = nb(hidden);
    write_buffer_f32(&in_buf, input);
    let q_w = upload_weight_buffer(k, q_weight_blocks);
    let k_w = upload_weight_buffer(k, k_weight_blocks);
    let v_w = upload_weight_buffer(k, v_weight_blocks);
    let o_w = upload_weight_buffer(k, o_weight_blocks);
    let cache_k_buf = upload_cache_buffer(k, cache_k);
    let cache_v_buf = upload_cache_buffer(k, cache_v);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_attention_block(
        e,
        k,
        &mut keep,
        &in_buf,
        &out_buf,
        attn_norm,
        eps,
        None,
        None,
        &q_w,
        &k_w,
        &v_w,
        &o_w,
        cos_t,
        sin_t,
        &cache_k_buf,
        &cache_v_buf,
        None,
        position_count,
        position_count - 1,
        n_heads,
        n_kv_heads,
        head_dim,
        position_count,
        scale,
        split_half_pairing,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// GPU-resident decode layer for one token in a SINGLE command buffer: runs the attention
/// block then the FFN block with no CPU readback between them (the attention output stays
/// in a GPU buffer and feeds the FFN block directly), so there is one commit/wait per layer
/// instead of two. Bit-identical to `try_attention_block_resident` followed by
/// `try_ffn_block_resident`. Returns the `[hidden]` layer output, or None if Metal is
/// unavailable or the dimensions are invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_decode_layer_resident(
    input: &[f32],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    eps: f32,
    q_weight_blocks: &[u8],
    k_weight_blocks: &[u8],
    v_weight_blocks: &[u8],
    o_weight_blocks: &[u8],
    gate_weight_blocks: &[u8],
    up_weight_blocks: &[u8],
    down_weight_blocks: &[u8],
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    ffn_dim: usize,
    scale: f32,
    split_half_pairing: bool,
) -> Option<Vec<f32>> {
    let hidden = input.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let half_rope = cos_t.len();
    // Attention-block constraints.
    if hidden == 0
        || !hidden.is_multiple_of(32)
        || !q_dim.is_multiple_of(32)
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || n_heads == 0
        || n_kv_heads == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || position_count == 0
        || attn_norm.len() != hidden
        || sin_t.len() != half_rope
        || half_rope * 2 > head_dim
        || cache_k.len() != kv_dim * position_count
        || cache_v.len() != cache_k.len()
    {
        return None;
    }
    // FFN-block constraints.
    if ffn_dim == 0 || !ffn_dim.is_multiple_of(32) || ffn_norm.len() != hidden {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_q = q_dim / 32;
    let bpr_ffn = ffn_dim / 32;
    if q_weight_blocks.len() != q_dim * bpr_hidden * 36
        || k_weight_blocks.len() != kv_dim * bpr_hidden * 36
        || v_weight_blocks.len() != kv_dim * bpr_hidden * 36
        || o_weight_blocks.len() != hidden * bpr_q * 36
        || gate_weight_blocks.len() != ffn_dim * bpr_hidden * 36
        || up_weight_blocks.len() != ffn_dim * bpr_hidden * 36
        || down_weight_blocks.len() != hidden * bpr_ffn * 36
    {
        return None;
    }
    let k = metal_linear_kernel()?;
    let nb = |n: usize| {
        k.device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = nb(hidden);
    let attn_buf = nb(hidden);
    let out_buf = nb(hidden);
    write_buffer_f32(&in_buf, input);
    let q_w = upload_weight_buffer(k, q_weight_blocks);
    let k_w = upload_weight_buffer(k, k_weight_blocks);
    let v_w = upload_weight_buffer(k, v_weight_blocks);
    let o_w = upload_weight_buffer(k, o_weight_blocks);
    let gate_w = upload_weight_buffer(k, gate_weight_blocks);
    let up_w = upload_weight_buffer(k, up_weight_blocks);
    let down_w = upload_weight_buffer(k, down_weight_blocks);
    let cache_k_buf = upload_cache_buffer(k, cache_k);
    let cache_v_buf = upload_cache_buffer(k, cache_v);
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    encode_attention_block(
        e,
        k,
        &mut keep,
        &in_buf,
        &attn_buf,
        attn_norm,
        eps,
        None,
        None,
        &q_w,
        &k_w,
        &v_w,
        &o_w,
        cos_t,
        sin_t,
        &cache_k_buf,
        &cache_v_buf,
        None,
        position_count,
        position_count - 1,
        n_heads,
        n_kv_heads,
        head_dim,
        position_count,
        scale,
        split_half_pairing,
    );
    encode_ffn_block(
        e, k, &mut keep, &attn_buf, &out_buf, ffn_norm, eps, &gate_w, &up_w, &down_w, ffn_dim,
    );
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(&out_buf, &mut out);
    Some(out)
}

/// Borrowed per-layer inputs for `try_decode_forward_resident`: the layer's two RMSNorm
/// weights, its seven Q8_0 weight-block buffers (q/k/v/o gate/up/down), and its K/V caches
/// (each `[n_kv_heads * position_count * head_dim]`, positions `0..position_count-1`
/// caller-filled; the current token is written at the last position).
pub struct ResidentDecodeLayer<'a> {
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub q_weight_blocks: &'a [u8],
    pub k_weight_blocks: &'a [u8],
    pub v_weight_blocks: &'a [u8],
    pub o_weight_blocks: &'a [u8],
    pub gate_weight_blocks: &'a [u8],
    pub up_weight_blocks: &'a [u8],
    pub down_weight_blocks: &'a [u8],
    pub cache_k: &'a [f32],
    pub cache_v: &'a [f32],
}

/// GPU-resident per-token decode over ALL transformer layers in a SINGLE command buffer: for
/// each layer the attention block then the FFN block are encoded back-to-back with the hidden
/// state staying in GPU buffers, so the whole token costs exactly ONE commit/wait (vs one per
/// layer for `try_decode_layer_resident`, or one per op for the standalone kernels). RoPE
/// tables and the per-token attention `scale` are shared across layers. `embedding` is the
/// input hidden state `[hidden]`; the returned `[hidden]` is the post-final-layer hidden state
/// (the final norm + output projection are applied by the caller). Bit-identical to feeding
/// each layer's output into the next via `try_decode_layer_resident`. Returns None if Metal is
/// unavailable, `layers` is empty, or any layer's dimensions are invalid.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn try_decode_forward_resident(
    embedding: &[f32],
    layers: &[ResidentDecodeLayer],
    cos_t: &[f32],
    sin_t: &[f32],
    eps: f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    ffn_dim: usize,
    scale: f32,
    split_half_pairing: bool,
) -> Option<Vec<f32>> {
    let hidden = embedding.len();
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let half_rope = cos_t.len();
    if hidden == 0
        || !hidden.is_multiple_of(32)
        || !q_dim.is_multiple_of(32)
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || n_heads == 0
        || n_kv_heads == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || position_count == 0
        || sin_t.len() != half_rope
        || half_rope * 2 > head_dim
        || ffn_dim == 0
        || !ffn_dim.is_multiple_of(32)
        || layers.is_empty()
    {
        return None;
    }
    let bpr_hidden = hidden / 32;
    let bpr_q = q_dim / 32;
    let bpr_ffn = ffn_dim / 32;
    for l in layers {
        if l.attn_norm.len() != hidden
            || l.ffn_norm.len() != hidden
            || l.cache_k.len() != kv_dim * position_count
            || l.cache_v.len() != l.cache_k.len()
            || l.q_weight_blocks.len() != q_dim * bpr_hidden * 36
            || l.k_weight_blocks.len() != kv_dim * bpr_hidden * 36
            || l.v_weight_blocks.len() != kv_dim * bpr_hidden * 36
            || l.o_weight_blocks.len() != hidden * bpr_q * 36
            || l.gate_weight_blocks.len() != ffn_dim * bpr_hidden * 36
            || l.up_weight_blocks.len() != ffn_dim * bpr_hidden * 36
            || l.down_weight_blocks.len() != hidden * bpr_ffn * 36
        {
            return None;
        }
    }
    let k = metal_linear_kernel()?;
    let nb = |n: usize| {
        k.device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
    };
    // `buf_a`/`buf_b` ping-pong as each layer's in/out hidden; `mid` carries the attention
    // output into the FFN block within a layer. All three are reused across layers: compute
    // encoders execute in submission order with coherency between them, so the sequential
    // reuse is hazard-free (no encoder reads a buffer a later encoder in the same layer step
    // is still writing).
    let buf_a = nb(hidden);
    let buf_b = nb(hidden);
    let mid = nb(hidden);
    write_buffer_f32(&buf_a, embedding);
    // Resolve every layer's Q8_0 weights to cache-resident GPU buffers up front. They are
    // keyed by (pointer, len) in `MetalLinearCache`, so the first decode of a model uploads
    // them and every subsequent token reuses the same on-GPU buffers -- the upload-once win.
    let resident: Vec<[Buffer; 7]> = {
        let mut cache = metal_linear_cache().lock().ok()?;
        layers
            .iter()
            .map(|l| {
                [
                    cache.q8_block_weight_buffer(&k.device, l.q_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.k_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.v_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.o_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.gate_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.up_weight_blocks),
                    cache.q8_block_weight_buffer(&k.device, l.down_weight_blocks),
                ]
            })
            .collect()
    };
    let mut keep = Vec::new();
    let cb = k.queue.new_command_buffer();
    let e = cb.new_compute_command_encoder();
    let mut from_a = true;
    for (i, layer) in layers.iter().enumerate() {
        let (in_buf, out_buf) = if from_a {
            (&buf_a, &buf_b)
        } else {
            (&buf_b, &buf_a)
        };
        let w = &resident[i];
        let cache_k_buf = upload_cache_buffer(k, layer.cache_k);
        let cache_v_buf = upload_cache_buffer(k, layer.cache_v);
        encode_attention_block(
            e,
            k,
            &mut keep,
            in_buf,
            &mid,
            layer.attn_norm,
            eps,
            None,
            None,
            &w[0],
            &w[1],
            &w[2],
            &w[3],
            cos_t,
            sin_t,
            &cache_k_buf,
            &cache_v_buf,
            None,
            position_count,
            position_count - 1,
            n_heads,
            n_kv_heads,
            head_dim,
            position_count,
            scale,
            split_half_pairing,
        );
        encode_ffn_block(
            e,
            k,
            &mut keep,
            &mid,
            out_buf,
            layer.ffn_norm,
            eps,
            &w[4],
            &w[5],
            &w[6],
            ffn_dim,
        );
        // Keep the transient per-layer cache buffers alive until the command buffer completes.
        keep.push(cache_k_buf);
        keep.push(cache_v_buf);
        from_a = !from_a;
    }
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    // After N flips, the last layer's output sits in `buf_a` iff N is even (`from_a` true).
    let final_buf = if from_a { &buf_a } else { &buf_b };
    let mut out = vec![0.0f32; hidden];
    read_buffer_f32(final_buf, &mut out);
    Some(out)
}

/// Per-layer weights/norms for `ResidentDecodeState::forward_token`. Unlike
/// `ResidentDecodeLayer`, the K/V cache is NOT here -- it lives in the session and persists
/// across tokens. Holds the seven Q8_0 weight-block byte buffers and the two RMSNorm f32
/// weights.
pub struct ResidentLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    /// Per-head QK-norm weights (`[head_dim]`, F32), applied to Q/K after the
    /// projection and before RoPE (Qwen3). `None` for plain Llama-family rows.
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub q_weight_blocks: ResidentWeightBytes<'a>,
    pub k_weight_blocks: ResidentWeightBytes<'a>,
    pub v_weight_blocks: ResidentWeightBytes<'a>,
    pub o_weight_blocks: ResidentWeightBytes<'a>,
    pub gate_weight_blocks: ResidentWeightBytes<'a>,
    pub up_weight_blocks: ResidentWeightBytes<'a>,
    pub down_weight_blocks: ResidentWeightBytes<'a>,
}

/// Where a resident weight's bytes live on the CPU side.
#[derive(Clone, Copy)]
pub enum ResidentWeightBytes<'a> {
    /// 36-byte f32-scale CPU blocks; uploaded (and wire-converted when wire mode is
    /// on) through the buffer cache.
    Blocks36(&'a [u8]),
    /// Page-aligned 34-byte wire blocks (fast-load); wrapped in place by an offset-0
    /// NoCopy buffer — the GPU reads this allocation directly, no upload copy.
    WirePages(&'a std::sync::Arc<crate::wire_mmap::WirePages>),
}

impl ResidentWeightBytes<'_> {
    /// Number of Q8_0 blocks the bytes describe, independent of layout.
    pub fn block_count(&self) -> usize {
        match self {
            ResidentWeightBytes::Blocks36(bytes) => bytes.len() / 36,
            ResidentWeightBytes::WirePages(pages) => pages.byte_len() / 34,
        }
    }
}

/// Optional final stage for `forward_token`: when present, the session also runs the final
/// RMSNorm + output (vocab) projection on the GPU in the same command buffer and returns the
/// `[vocab_size]` logits instead of the hidden state — keeping the large output matmul off the
/// CPU. `output_weight_blocks` is the Q8_0 output/embedding projection.
pub struct LogitsStage<'a> {
    pub final_norm: &'a [f32],
    pub output_weight_blocks: ResidentWeightBytes<'a>,
    pub vocab_size: usize,
}

/// GPU-side sampling operation attached to a resident token graph.
#[derive(Clone, Copy)]
pub enum SampleMode {
    Greedy,
    Temperature {
        inv_temperature: f32,
        base_seed: u64,
    },
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleSignature {
    Greedy,
    Temperature {
        inv_temperature_bits: u32,
        token_seed: u64,
    },
}

impl SampleMode {
    #[cfg(target_os = "macos")]
    fn signature(self, position: usize) -> SampleSignature {
        match self {
            Self::Greedy => SampleSignature::Greedy,
            Self::Temperature {
                inv_temperature,
                base_seed,
            } => SampleSignature::Temperature {
                inv_temperature_bits: inv_temperature.to_bits(),
                token_seed: base_seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(position as u64 + 1),
            },
        }
    }
}

/// GPU-side sampling stage for the resident decode fast path: the token graph's
/// tail samples the logits and gathers the selected token's embedding row into
/// the next graph's input buffer, so the CPU never sits between two tokens.
pub struct SampleStage<'a> {
    /// Token-embedding Q8_0 bytes (36-byte CPU blocks wire-converted in the buffer
    /// cache, or page-aligned wire pages wrapped in place).
    pub embedding_blocks: ResidentWeightBytes<'a>,
    pub mode: SampleMode,
}

/// What a resident forward_token call produced: the raw logits/hidden vector (CPU
/// sampling path), or the GPU-sampled next token id (greedy fast path).
pub enum ResidentTokenOut {
    Data(Vec<f32>),
    Sampled(u32),
}

/// A resident decode session that owns the on-GPU KV cache (per layer, sized to
/// `max_positions`) and the reused hidden ping-pong buffers. A multi-token greedy decode runs
/// each token in ONE command buffer with the KV cache persisting on the GPU across tokens --
/// only the new token's K/V is blitted in each step, never re-uploaded -- and Q8 weights stay
/// resident via the global `MetalLinearCache`. For its sequence the session is the
/// authoritative KV store (the kernel computes and appends each token's K/V at `position`).
#[cfg(target_os = "macos")]
pub struct ResidentDecodeState {
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    ffn_dim: usize,
    /// Positions the KV cache is currently allocated for; grown on demand toward `cap`.
    max_positions: usize,
    /// Hard ceiling on `max_positions` (the model context length): growth never exceeds it.
    cap: usize,
    eps: f32,
    split_half_pairing: bool,
    cache_k: Vec<Buffer>,
    cache_v: Vec<Buffer>,
    /// Half-precision mirrors of cache_k/cache_v (empty in kv16-primary mode, where the
    /// primary caches are already half). Dual-written by the prefill and per-token
    /// scatters; read by the split-K kv16 decode attention, which halves the dominant
    /// KV traffic at depth. Zero-filled at allocation/growth so padded positions read 0.
    cache_k16: Vec<Buffer>,
    cache_v16: Vec<Buffer>,
    buf_a: Buffer,
    buf_b: Buffer,
    mid: Buffer,
    /// Number of KV positions currently materialized in the cache (seeded history + appended
    /// tokens). The caller uses this to detect a new sequence and reseed.
    filled: usize,
    /// Encode-ahead pipeline: the NEXT token's fully-encoded AND committed command buffer,
    /// built on the CPU while the previous token executed on the GPU. Execution is gated on
    /// `gate_event`, signaled only after the token's input embedding is written — so by
    /// signal time, scheduling cost is already paid. Invalidated whenever the KV cache
    /// reallocates (the encoded graph references the old buffers); a committed-but-ungated
    /// stale graph MUST be released via `release_stale` or it deadlocks the serial queue.
    pending: Option<PreparedToken>,
    /// True when `pending` was already event-signaled (GPU-sampling fast path): its input
    /// embedding comes from the previous graph's gather, so it may already be executing.
    pending_signaled: bool,
    /// Token id the previous graph's tail sampled (greedy fast path); the next call skips
    /// the CPU embedding write only when its input token matches.
    last_sampled: Option<u32>,
    /// Signaled BY the GPU as each token graph's last command; the fast lane polls this
    /// (a shared-memory read) instead of wait_until_completed, skipping the
    /// completion-handler wake-up latency on the per-token critical path.
    done_event: metal::SharedEvent,
    /// The previously returned fast-lane graph, kept alive until the next call calls
    /// wait_until_completed on it (its GPU work was already observed done via done_event;
    /// the deferred wait only settles command-buffer bookkeeping off the critical path).
    retiring: Option<PreparedToken>,
    /// Gates pre-committed token graphs; monotonically increasing values.
    gate_event: metal::SharedEvent,
    event_counter: u64,
    /// KV cache element width: f16 when CAMELID_METAL_KV16 is set, else f32.
    kv16: bool,
}

/// A fully-encoded, uncommitted per-token command buffer. The input embedding is written
/// into the session's input buffer just before commit (the graph reads it first), so the
/// expensive encode happens off the critical path.
#[cfg(target_os = "macos")]
struct PreparedToken {
    position: usize,
    has_logits: bool,
    has_sample: bool,
    sample_signature: Option<SampleSignature>,
    event_value: u64,
    cb: metal::CommandBuffer,
    logits_buf: Option<Buffer>,
    /// GPU-sampled token id (uint), present when the graph tail runs the greedy sampler.
    sampled_buf: Option<Buffer>,
    final_from_a: bool,
    /// Scratch buffers the encoded graph references; kept alive until completion.
    _keep: Vec<Buffer>,
    encode_us: u128,
}

/// WIN2METAL Phase 4 — tree-attention descriptor handed to the (otherwise shared)
/// `verify_batch_inner`. `None` ⇒ today's contiguous linear verify (byte-unchanged).
/// `Some` ⇒ each node attends `prefix [0, base)` ∪ its ancestor draft slots.
#[cfg(target_os = "macos")]
struct TreeAttn {
    /// Contiguous committed-history prefix length (`base_position`). Slots `[0, base)`
    /// are all-ancestor, so the kernels read them directly (no indirection).
    base: usize,
    /// Per-node ancestor draft cache slots (absolute, increasing, INCLUDING self), built
    /// from the ancestor bitset. `tail_slots[i].len()` is the node's `tail_count`; its
    /// attention `position_count` is `base + tail_count`.
    tail_slots: Vec<Vec<u32>>,
}

#[cfg(target_os = "macos")]
impl ResidentDecodeState {
    /// Allocate the session. `max_positions` is the initial KV-cache capacity (grown on demand
    /// up to `cap`, the model context length). Returns None if Metal is unavailable or the
    /// dimensions are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        ffn_dim: usize,
        max_positions: usize,
        cap: usize,
        eps: f32,
        split_half_pairing: bool,
    ) -> Option<Self> {
        let q_dim = n_heads * head_dim;
        if n_layers == 0
            || hidden == 0
            || !hidden.is_multiple_of(32)
            || !q_dim.is_multiple_of(32)
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || n_heads == 0
            || n_kv_heads == 0
            || !n_heads.is_multiple_of(n_kv_heads)
            || ffn_dim == 0
            || !ffn_dim.is_multiple_of(32)
            || max_positions == 0
            || cap < max_positions
        {
            return None;
        }
        let k = metal_linear_kernel()?;
        let nb = |n: usize| {
            k.device
                .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
        };
        let kv16 = kv16_enabled();
        let kv_elem = if kv16 { 2 } else { 4 };
        let kv_slots = n_kv_heads * max_positions * head_dim;
        let kvb = |slots: usize| {
            k.device.new_buffer(
                (slots * kv_elem) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let cache_k = (0..n_layers).map(|_| kvb(kv_slots)).collect();
        let cache_v = (0..n_layers).map(|_| kvb(kv_slots)).collect();
        // Half mirrors for the split-K kv16 decode attention (skip in kv16-primary
        // mode, where the primary caches are already half). Zero-filled so padded
        // positions read 0 in the attention kernels.
        let kvb16 = |slots: usize| {
            let b = k
                .device
                .new_buffer((slots * 2) as u64, MTLResourceOptions::StorageModeShared);
            unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, slots * 2) };
            b
        };
        let (cache_k16, cache_v16) = if kv16 {
            (Vec::new(), Vec::new())
        } else {
            (
                (0..n_layers).map(|_| kvb16(kv_slots)).collect(),
                (0..n_layers).map(|_| kvb16(kv_slots)).collect(),
            )
        };
        Some(Self {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden,
            ffn_dim,
            max_positions,
            cap,
            eps,
            split_half_pairing,
            cache_k,
            cache_v,
            cache_k16,
            cache_v16,
            buf_a: nb(hidden),
            buf_b: nb(hidden),
            mid: nb(hidden),
            filled: 0,
            pending: None,
            pending_signaled: false,
            last_sampled: None,
            done_event: k.device.new_shared_event(),
            retiring: None,
            gate_event: k.device.new_shared_event(),
            event_counter: 0,
            kv16,
        })
    }

    /// Grow the KV cache to at least `needed` positions (capped at `self.cap`) by allocating
    /// larger per-layer buffers and blitting the `filled` materialized slots across (the
    /// per-head position stride changes with `max_positions`). GPU-to-GPU, no CPU readback.
    /// Returns false if `needed` exceeds the cap or Metal is unavailable.
    fn ensure_capacity(&mut self, needed: usize) -> bool {
        if needed <= self.max_positions {
            return true;
        }
        if needed > self.cap {
            return false;
        }
        let new_max = (self.max_positions * 2).max(needed).min(self.cap);
        let k = match metal_linear_kernel() {
            Some(k) => k,
            None => return false,
        };
        let kv_elem = if self.kv16 { 2 } else { 4 };
        let kv_slots = self.n_kv_heads * new_max * self.head_dim;
        let new_k: Vec<Buffer> = (0..self.n_layers)
            .map(|_| {
                k.device.new_buffer(
                    (kv_slots * kv_elem) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let new_v: Vec<Buffer> = (0..self.n_layers)
            .map(|_| {
                k.device.new_buffer(
                    (kv_slots * kv_elem) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let mirror = |_| {
            let b = k
                .device
                .new_buffer((kv_slots * 2) as u64, MTLResourceOptions::StorageModeShared);
            unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, kv_slots * 2) };
            b
        };
        let new_k16: Vec<Buffer> = if self.kv16 {
            Vec::new()
        } else {
            (0..self.n_layers).map(mirror).collect()
        };
        let new_v16: Vec<Buffer> = if self.kv16 {
            Vec::new()
        } else {
            (0..self.n_layers).map(mirror).collect()
        };
        if self.filled > 0 {
            let run = (self.filled * self.head_dim * kv_elem) as u64;
            let cb = k.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for layer in 0..self.n_layers {
                for h in 0..self.n_kv_heads {
                    let src = (h * self.max_positions * self.head_dim * kv_elem) as u64;
                    let dst = (h * new_max * self.head_dim * kv_elem) as u64;
                    blit.copy_from_buffer(&self.cache_k[layer], src, &new_k[layer], dst, run);
                    blit.copy_from_buffer(&self.cache_v[layer], src, &new_v[layer], dst, run);
                    if !self.kv16 {
                        let run16 = (self.filled * self.head_dim * 2) as u64;
                        let src16 = (h * self.max_positions * self.head_dim * 2) as u64;
                        let dst16 = (h * new_max * self.head_dim * 2) as u64;
                        blit.copy_from_buffer(
                            &self.cache_k16[layer],
                            src16,
                            &new_k16[layer],
                            dst16,
                            run16,
                        );
                        blit.copy_from_buffer(
                            &self.cache_v16[layer],
                            src16,
                            &new_v16[layer],
                            dst16,
                            run16,
                        );
                    }
                }
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        self.cache_k = new_k;
        self.cache_v = new_v;
        self.cache_k16 = new_k16;
        self.cache_v16 = new_v16;
        self.max_positions = new_max;
        true
    }

    /// Positions currently materialized in the KV cache (seeded + appended).
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Mark `n` positions as materialized (called after seeding history from a CPU cache).
    pub fn set_filled(&mut self, n: usize) {
        self.filled = n;
    }

    /// Discard the KV positions at and after `position` (speculative rollback of rejected
    /// draft tokens). The cache is position-indexed and the next append overwrites the
    /// dropped slots, so nothing but the materialized count moves — no buffer work, and the
    /// still-valid history `[0, position)` stays on the GPU instead of being reseeded from
    /// the CPU. A no-op when the cache holds no more than `position` positions. The
    /// encode-ahead graph was built for the abandoned position, so it is released here
    /// rather than left for `forward_token` to discover (releasing is mandatory: it sits
    /// committed and gated on the serial queue).
    pub fn rollback_to_position(&mut self, position: usize) {
        if position >= self.filled {
            return;
        }
        if let Some(stale) = self.pending.take() {
            self.release_stale(stale);
        }
        self.pending_signaled = false;
        self.last_sampled = None;
        self.filled = position;
    }

    /// Decode one token at sequence position `position` (0-based): append this token's K/V to
    /// the persistent GPU cache and attend over positions `0..=position`, all layers in one
    /// command buffer. `cos_t`/`sin_t` are the RoPE tables for `position`; `scale` is the
    /// attention scale. Returns the post-final-layer hidden state `[hidden]` (the caller
    /// applies the final norm + output projection), or None on dimension mismatch.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &mut self,
        embedding: &[f32],
        layers: &[ResidentLayerWeights],
        cos_t: &[f32],
        sin_t: &[f32],
        position: usize,
        scale: f32,
        logits_stage: Option<LogitsStage>,
        sample_stage: Option<SampleStage>,
        input_token_id: u32,
        next_rope: Option<(&[f32], &[f32])>,
    ) -> Option<ResidentTokenOut> {
        let fn_started = std::time::Instant::now();
        let half_rope = cos_t.len();
        // Grow the on-GPU KV cache if this token's position is past the current capacity.
        // Growth reallocates the cache buffers, so any pre-encoded pending graph (which
        // references the old buffers) must be dropped.
        let before_growth = self.max_positions;
        if !self.ensure_capacity(position + 1) {
            return None;
        }
        if self.max_positions != before_growth {
            if let Some(stale) = self.pending.take() {
                self.release_stale(stale);
            }
            self.pending_signaled = false;
        }
        if embedding.len() != self.hidden
            || layers.len() != self.n_layers
            || sin_t.len() != half_rope
            || half_rope * 2 > self.head_dim
        {
            return None;
        }
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let bpr_hidden = self.hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = self.ffn_dim / 32;
        for l in layers {
            if l.attn_norm.len() != self.hidden
                || l.ffn_norm.len() != self.hidden
                || l.q_weight_blocks.block_count() != q_dim * bpr_hidden
                || l.k_weight_blocks.block_count() != kv_dim * bpr_hidden
                || l.v_weight_blocks.block_count() != kv_dim * bpr_hidden
                || l.o_weight_blocks.block_count() != self.hidden * bpr_q
                || l.gate_weight_blocks.block_count() != self.ffn_dim * bpr_hidden
                || l.up_weight_blocks.block_count() != self.ffn_dim * bpr_hidden
                || l.down_weight_blocks.block_count() != self.hidden * bpr_ffn
            {
                return None;
            }
        }
        if let Some(s) = &logits_stage {
            if s.vocab_size == 0
                || s.final_norm.len() != self.hidden
                || s.output_weight_blocks.block_count() != s.vocab_size * bpr_hidden
            {
                return None;
            }
        }
        let k = metal_linear_kernel()?;
        let trace = std::env::var_os("CAMELID_RESIDENT_TRACE").is_some();
        // Take the pre-encoded, pre-committed graph for this position (built and committed
        // while the PREVIOUS token was executing on the GPU, scheduling already paid), or
        // encode inline on the first token / after invalidation. A stale pending graph is
        // committed on the serial queue gated behind its event and MUST be released.
        // The sampling tail's embedding gather reads wire-format rows; outside wire mode
        // prepare_token would encode no tail, so drop the stage up front to keep the
        // pending-graph reuse check consistent with what actually gets encoded.
        let sample_stage = if f32y_gemv_enabled() && wire_weights_enabled() {
            sample_stage
        } else {
            None
        };
        let want_sample = sample_stage.is_some() && logits_stage.is_some();
        let requested_sample_signature = sample_stage
            .as_ref()
            .map(|stage| stage.mode.signature(position));
        let pending = self.pending.take();
        let pending_signaled = std::mem::take(&mut self.pending_signaled);
        let usable = matches!(
            &pending,
            Some(p) if p.position == position
                && p.has_logits == logits_stage.is_some()
                && p.has_sample == want_sample
                && p.sample_signature == requested_sample_signature
        );
        // Fast path: the pending graph was already signaled last call (its input embedding
        // comes from the previous graph's GPU-side gather). It only matches the caller's
        // intent when the input token IS the token that gather selected; a forced different
        // continuation re-runs the position with the CPU-written embedding (the KV scatter
        // overwrites the same slot, so the re-run is exact).
        let already_running =
            usable && pending_signaled && self.last_sampled == Some(input_token_id);
        let prepared = if usable && (already_running || !pending_signaled) {
            pending.unwrap()
        } else {
            if let Some(stale) = pending {
                self.release_stale(stale);
            }
            self.prepare_token(
                k,
                layers,
                cos_t,
                sin_t,
                position,
                scale,
                logits_stage.as_ref(),
                sample_stage.as_ref(),
            )?
        };
        let gpu_started = std::time::Instant::now();
        if !already_running {
            // The graph's first op reads the embedding from buf_a; it is gated on the shared
            // event, so writing the embedding and then signaling releases it instantly.
            write_buffer_f32(&self.buf_a, embedding);
            self.gate_event.set_signaled_value(prepared.event_value);
        }
        // Encode-ahead: build the NEXT token's command buffer on the CPU while this one
        // executes on the GPU (greedy decode advances one position at a time). Skipped at
        // the KV-capacity edge — growth would invalidate the encoded graph, so the next
        // call grows first and encodes inline once.
        let mut next_encode_us = 0u128;
        if let Some((cos_n, sin_n)) = next_rope {
            if position + 2 <= self.max_positions {
                if let Some(next) = self.prepare_token(
                    k,
                    layers,
                    cos_n,
                    sin_n,
                    position + 1,
                    scale,
                    logits_stage.as_ref(),
                    sample_stage.as_ref(),
                ) {
                    next_encode_us = next.encode_us;
                    // Greedy fast path: the next graph's input is produced by THIS graph's
                    // sampling tail on the GPU timeline (serial queue order), so it can be
                    // released now — the GPU runs token-to-token with no CPU on the path.
                    if want_sample && next.has_sample {
                        self.gate_event.set_signaled_value(next.event_value);
                        self.pending_signaled = true;
                    }
                    self.pending = Some(next);
                }
            }
        }
        if want_sample && prepared.sampled_buf.is_some() && !trace {
            // Settle the PREVIOUS fast-lane graph's bookkeeping (its GPU work finished at
            // least one token ago) while this token still executes. Its scratch goes back
            // to the pool instead of dropping: several hundred MTLBuffer creates/releases
            // per token otherwise cross into IOGPU on the decode loop's thread.
            if let Some(r) = self.retiring.take() {
                r.cb.wait_until_completed();
                pool_recycle(
                    k,
                    r._keep.into_iter().chain(r.sampled_buf).chain(r.logits_buf),
                );
            }
            // Observe THIS graph's completion via the GPU-stamped event: a shared-memory
            // poll, no kernel wake-up on the critical path.
            while self.done_event.signaled_value() < prepared.event_value {
                std::hint::spin_loop();
            }
        } else {
            if let Some(r) = self.retiring.take() {
                r.cb.wait_until_completed();
                pool_recycle(
                    k,
                    r._keep.into_iter().chain(r.sampled_buf).chain(r.logits_buf),
                );
            }
            prepared.cb.wait_until_completed();
        }
        if trace {
            let gpu_us = gpu_started.elapsed().as_micros();
            // True GPU-busy window from the command buffer's hardware timestamps: splits
            // "kernel executing" from submission/scheduling gaps inside commit_wait.
            let (gpu_busy_us, kernel_total_us) = command_buffer_gpu_times_us(&prepared.cb);
            let pre_us = fn_started.elapsed().as_micros() - gpu_us;
            eprintln!(
                "[resident] pos={position} layers={} pre={pre_us}us encode={}us next_encode={next_encode_us}us commit_wait={gpu_us}us gpu_busy={gpu_busy_us}us kernel_window={kernel_total_us}us",
                self.n_layers, prepared.encode_us,
            );
        }
        self.filled = position + 1;
        if let Some(sampled_buf) = &prepared.sampled_buf {
            let id = unsafe { *(sampled_buf.contents() as *const u32) };
            self.last_sampled = Some(id);
            // Keep the graph (and its scratch buffers) alive until the next call settles
            // it; its GPU work was observed complete above.
            self.retiring = Some(prepared);
            return Some(ResidentTokenOut::Sampled(id));
        }
        self.last_sampled = None;
        if let Some(logits_buf) = prepared.logits_buf {
            let vocab = logits_stage.as_ref().map(|s| s.vocab_size).unwrap_or(0);
            let mut out = vec![0.0f32; vocab];
            read_buffer_f32(&logits_buf, &mut out);
            return Some(ResidentTokenOut::Data(out));
        }
        let final_buf = if prepared.final_from_a {
            &self.buf_a
        } else {
            &self.buf_b
        };
        let mut out = vec![0.0f32; self.hidden];
        read_buffer_f32(final_buf, &mut out);
        Some(ResidentTokenOut::Data(out))
    }

    /// Encode one token's full graph into an uncommitted command buffer. Everything here is
    /// CPU work (buffer allocs + encoder calls) — callable while the GPU executes the
    /// previous token. The graph reads its input embedding from `buf_a` at execution time.
    #[allow(clippy::too_many_arguments)]
    fn prepare_token(
        &mut self,
        k: &'static MetalLinearKernel,
        layers: &[ResidentLayerWeights],
        cos_t: &[f32],
        sin_t: &[f32],
        position: usize,
        scale: f32,
        logits_stage: Option<&LogitsStage>,
        sample_stage: Option<&SampleStage>,
    ) -> Option<PreparedToken> {
        let bpr_hidden = self.hidden / 32;
        // Resolve all resident weight buffers (layer weights + optional output stage) under one
        // cache lock. They are keyed by (pointer, len), so they upload once and persist.
        let resident: Vec<[Buffer; 7]>;
        let stage_bufs: Option<(Buffer, Buffer)>;
        let emb_buf: Option<Buffer>;
        {
            let mut cache = metal_linear_cache().lock().ok()?;
            let wire = f32y_gemv_enabled() && wire_weights_enabled();
            let mut wb = |w: &ResidentWeightBytes| match w {
                ResidentWeightBytes::Blocks36(blocks) => Some(if wire {
                    cache.q8_wire_weight_buffer(&k.device, blocks)
                } else {
                    cache.q8_block_weight_buffer(&k.device, blocks)
                }),
                // Wire pages hold the 34-byte wire layout: only the wire kernels can
                // consume them, so outside wire mode this graph cannot be built.
                ResidentWeightBytes::WirePages(pages) => {
                    wire.then(|| cache.q8_wire_nocopy_buffer(&k.device, pages))
                }
            };
            resident = layers
                .iter()
                .map(|l| {
                    Some([
                        wb(&l.q_weight_blocks)?,
                        wb(&l.k_weight_blocks)?,
                        wb(&l.v_weight_blocks)?,
                        wb(&l.o_weight_blocks)?,
                        wb(&l.gate_weight_blocks)?,
                        wb(&l.up_weight_blocks)?,
                        wb(&l.down_weight_blocks)?,
                    ])
                })
                .collect::<Option<Vec<_>>>()?;
            stage_bufs = match logits_stage {
                Some(s) => {
                    let ow = wb(&s.output_weight_blocks)?;
                    Some((ow, cache.weight_buffer(&k.device, s.final_norm)))
                }
                None => None,
            };
            // The sampling tail needs the wire-format weight layout for the embedding
            // gather; outside wire mode the fast path is disabled by the caller.
            emb_buf = match sample_stage {
                Some(s) if wire => Some(match &s.embedding_blocks {
                    ResidentWeightBytes::Blocks36(blocks) => {
                        cache.q8_wire_weight_buffer(&k.device, blocks)
                    }
                    ResidentWeightBytes::WirePages(pages) => {
                        cache.q8_wire_nocopy_buffer(&k.device, pages)
                    }
                }),
                _ => None,
            };
        }
        let position_count = position + 1;
        let mut keep = Vec::new();
        let encode_started = std::time::Instant::now();
        self.event_counter += 1;
        let event_value = self.event_counter;
        let cb = k.queue.new_command_buffer();
        cb.encode_wait_for_event(&self.gate_event, event_value);
        let e = cb.new_compute_command_encoder();
        let mut from_a = true;
        for (i, layer) in layers.iter().enumerate() {
            let (in_buf, out_buf) = if from_a {
                (&self.buf_a, &self.buf_b)
            } else {
                (&self.buf_b, &self.buf_a)
            };
            let w = &resident[i];
            encode_attention_block(
                e,
                k,
                &mut keep,
                in_buf,
                &self.mid,
                layer.attn_norm,
                self.eps,
                layer.q_norm,
                layer.k_norm,
                &w[0],
                &w[1],
                &w[2],
                &w[3],
                cos_t,
                sin_t,
                &self.cache_k[i],
                &self.cache_v[i],
                if self.kv16 {
                    None
                } else {
                    Some((&self.cache_k16[i], &self.cache_v16[i]))
                },
                self.max_positions,
                position,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                position_count,
                scale,
                self.split_half_pairing,
            );
            encode_ffn_block(
                e,
                k,
                &mut keep,
                &self.mid,
                out_buf,
                layer.ffn_norm,
                self.eps,
                &w[4],
                &w[5],
                &w[6],
                self.ffn_dim,
            );
            from_a = !from_a;
        }
        let final_buf = if from_a { &self.buf_a } else { &self.buf_b };
        // Optional final stage: RMSNorm + output (vocab) projection in the SAME command buffer,
        // so the large logits matmul runs on the GPU instead of falling to the slow CPU path.
        let logits_buf = if let (Some(s), Some((ow_buf, fnorm_buf))) = (logits_stage, &stage_bufs) {
            let nb = |bytes: u64| pool_get(k, bytes);
            let rms_scalar = nb(8);
            let lscales = nb((bpr_hidden * 4) as u64);
            let lquants = nb(self.hidden as u64);

            let lmm_scalar = nb(8);
            let logits_buf = nb((s.vocab_size * 4) as u64);
            unsafe {
                let p = rms_scalar.contents() as *mut u8;
                *(p as *mut u32) = self.hidden as u32;
                *(p.add(4) as *mut f32) = self.eps;
                let m = lmm_scalar.contents() as *mut u32;
                *m = bpr_hidden as u32;
                *m.add(1) = s.vocab_size as u32;
            }
            if f32y_gemv_enabled() {
                let normf = nb((self.hidden * 4) as u64);
                encode_rms_norm_f32(e, k, final_buf, fnorm_buf, &normf, &rms_scalar);
                encode_q8_matmul_f32y(e, k, &normf, ow_buf, &logits_buf, &lmm_scalar, s.vocab_size);
                keep.push(normf);
            } else {
                encode_rms_norm_quantize(
                    e,
                    k,
                    final_buf,
                    fnorm_buf,
                    &lscales,
                    &lquants,
                    &rms_scalar,
                );
                encode_q8_matmul(
                    e,
                    k,
                    &lscales,
                    &lquants,
                    ow_buf,
                    &logits_buf,
                    &lmm_scalar,
                    s.vocab_size,
                );
            }
            keep.extend([rms_scalar, lscales, lquants, lmm_scalar]);
            Some(logits_buf)
        } else {
            None
        };
        // Sampling tail: select an id from the logits, then gather that token's
        // embedding row into buf_a — the input the NEXT pre-encoded graph reads.
        // Both operations are hazard-ordered after the logits matmul by Metal.
        let sampled_buf = match (&logits_buf, &emb_buf, logits_stage, sample_stage) {
            (Some(lb), Some(eb), Some(s), Some(sample)) => {
                let nb = |bytes: u64| pool_get(k, bytes);
                let id_buf = nb(4);
                let am_scalar = nb(4);
                let eg_scalar = nb(4);
                unsafe {
                    *(am_scalar.contents() as *mut u32) = s.vocab_size as u32;
                    *(eg_scalar.contents() as *mut u32) = bpr_hidden as u32;
                }
                match sample.mode {
                    SampleMode::Greedy => {
                        e.set_compute_pipeline_state(&k.argmax_f32_greedy_pipeline);
                        e.set_buffer(0, Some(lb), 0);
                        e.set_buffer(1, Some(&id_buf), 0);
                        e.set_buffer(2, Some(&am_scalar), 0);
                    }
                    SampleMode::Temperature {
                        inv_temperature,
                        base_seed,
                    } => {
                        let temperature_scalar = nb(4);
                        let seed_scalar = nb(8);
                        let token_seed =
                            base_seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(position as u64 + 1);
                        unsafe {
                            *(temperature_scalar.contents() as *mut f32) = inv_temperature;
                            *(seed_scalar.contents() as *mut u64) = token_seed;
                        }
                        e.set_compute_pipeline_state(&k.sample_gumbel_f32_pipeline);
                        e.set_buffer(0, Some(lb), 0);
                        e.set_buffer(1, Some(&id_buf), 0);
                        e.set_buffer(2, Some(&am_scalar), 0);
                        e.set_buffer(3, Some(&temperature_scalar), 0);
                        e.set_buffer(4, Some(&seed_scalar), 0);
                        keep.extend([temperature_scalar, seed_scalar]);
                    }
                }
                e.dispatch_thread_groups(
                    metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
                e.set_compute_pipeline_state(&k.embed_row_gather_q8_wire_pipeline);
                e.set_buffer(0, Some(eb), 0);
                e.set_buffer(1, Some(&id_buf), 0);
                e.set_buffer(2, Some(&self.buf_a), 0);
                e.set_buffer(3, Some(&eg_scalar), 0);
                dispatch_1d(e, &k.embed_row_gather_q8_wire_pipeline, self.hidden);
                keep.extend([am_scalar, eg_scalar]);
                Some(id_buf)
            }
            _ => None,
        };
        e.end_encoding();
        // The GPU stamps the done event as the graph's last command; the fast lane polls
        // it to observe completion without the completion-handler wake-up.
        cb.encode_signal_event(&self.done_event, event_value);
        // Commit NOW (still gated on the event): commandBuffer scheduling happens while the
        // previous token executes, so signaling the event later starts the GPU immediately.
        cb.commit();
        let encode_us = encode_started.elapsed().as_micros();
        Some(PreparedToken {
            position,
            has_logits: logits_stage.is_some(),
            has_sample: sampled_buf.is_some(),
            sample_signature: sample_stage.map(|stage| stage.mode.signature(position)),
            event_value,
            cb: cb.to_owned(),
            logits_buf,
            sampled_buf,
            final_from_a: from_a,
            _keep: keep,
            encode_us,
        })
    }

    /// One-command-buffer GPU prefill over `n` prompt tokens: rms_norm/RoPE/scatter/tiled
    /// attention run per token, but every matmul is the wire GEMM, so weights stream ONCE per
    /// prefill, not once per token. KV lands directly in the resident caches (positions
    /// 0..n); generation then continues with the resident decode, no CPU seeding. Returns
    /// the logits of the LAST prefilled token. Requirements mirror forward_token plus:
    /// f32 KV cache only, wire weights, head_dim % 32 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_tokens(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        layers: &[ResidentLayerWeights],
        cos_all: &[f32],
        sin_all: &[f32],
        scale: f32,
    ) -> Option<()> {
        if n_tokens == 0
            || self.kv16
            || !wire_weights_enabled()
            || !self.head_dim.is_multiple_of(32)
            || self.head_dim > 128
            || embeddings.len() != n_tokens * self.hidden
            || self.filled != 0
            || !self.ensure_capacity(n_tokens)
        {
            return None;
        }
        let half_rope = cos_all.len() / n_tokens;
        if sin_all.len() != cos_all.len() || half_rope * 2 > self.head_dim {
            return None;
        }
        let k = metal_linear_kernel()?;
        // Qwen3 per-head QK-norm: when the layers carry q_norm/k_norm we apply an
        // in-place per-head RMSNorm to Q and K (after the QKV GEMM, before RoPE).
        // The per-head norm kernel is f32, so we keep the tiled-MM GEMM (fast) but
        // force off use_attn_mm/use_h16: that makes the MM GEMM emit f32 Q/K
        // [token][head][head_dim] and routes attention through the validated
        // non-attn-mm path, so the per-head norm runs in the cpu_reference order.
        let has_qk_norm = layers.first().is_some_and(|l| l.q_norm.is_some());
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let bpr_hidden = self.hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = self.ffn_dim / 32;

        let resident: Vec<[Buffer; 7]>;
        {
            let mut cache = metal_linear_cache().lock().ok()?;
            let mut wb = |w: &ResidentWeightBytes| match w {
                ResidentWeightBytes::Blocks36(blocks) => {
                    cache.q8_wire_weight_buffer(&k.device, blocks)
                }
                ResidentWeightBytes::WirePages(pages) => {
                    cache.q8_wire_nocopy_buffer(&k.device, pages)
                }
            };
            resident = layers
                .iter()
                .map(|l| {
                    [
                        wb(&l.q_weight_blocks),
                        wb(&l.k_weight_blocks),
                        wb(&l.v_weight_blocks),
                        wb(&l.o_weight_blocks),
                        wb(&l.gate_weight_blocks),
                        wb(&l.up_weight_blocks),
                        wb(&l.down_weight_blocks),
                    ]
                })
                .collect();
        }

        let nb = |bytes: usize| {
            k.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        // The attention-as-matmul gate and the all-half activation-stream gate are
        // decided before buffer allocation: when active, the layer chain's working
        // buffers are half-precision end to end (GEMM outputs, residual stream,
        // norms, silu), halving activation traffic.
        let n_pad = n_tokens.next_multiple_of(128);
        let gqa_group = self.n_heads / self.n_kv_heads;
        let use_attn_mm = mm_prefill_enabled()
            && !has_qk_norm
            && q_dim.is_multiple_of(128)
            && kv_dim.is_multiple_of(128)
            && self.hidden.is_multiple_of(128)
            && self.ffn_dim.is_multiple_of(128)
            && self.head_dim.is_multiple_of(8)
            && self.head_dim <= 128
            && self.n_heads * n_pad * 256 * 4 <= attn_mm_scratch_cap_bytes();
        // Query-block width for the S/P panels: the panels are [head][qb][n_pad], so
        // memory is linear in the block width and the budget sets how many query
        // columns are materialized at once. Prompts that fit in one block reproduce
        // the untiled math bit-for-bit (q_offset = 0).
        let attn_qb = if use_attn_mm {
            let per_col = self.n_heads * n_pad * 4; // S + P bytes per query column
                                                    // max-then-min, NOT clamp: short prompts have n_pad < 256 and clamp(256,
                                                    // n_pad) panics on min > max; a single block covering n_pad is correct there.
            ((attn_mm_scratch_cap_bytes() / per_col) & !63)
                .max(256)
                .min(n_pad)
        } else {
            0
        };
        let use_h16 = use_attn_mm && half_rope * 2 == self.head_dim;
        let es = if use_h16 { 2 } else { 4 }; // element size of the activation stream
        let seq = nb(n_tokens * self.hidden * es);
        let seq_out = nb(n_tokens * self.hidden * es);
        let normf = nb(n_tokens * self.hidden * 4);
        let q_buf = nb(n_tokens * q_dim * es);
        let k_buf = nb(n_tokens * kv_dim * es);
        let v_buf = nb(n_tokens * kv_dim * es);
        let ctx_buf = nb(n_tokens * q_dim * 4);
        let o_buf = nb(n_tokens * self.hidden * es);
        let gate_buf = nb(n_tokens * self.ffn_dim * es);
        let up_buf = nb(n_tokens * self.ffn_dim * es);
        let silu_buf = nb(n_tokens * self.ffn_dim * 4);
        let down_buf = nb(n_tokens * self.hidden * es);
        let cos_buf = nb(cos_all.len() * 4);
        let sin_buf = nb(sin_all.len() * 4);
        // h16: memcpy raw f32 to a staging buffer; the command buffer's first dispatch
        // converts it to the half panel on the GPU (the per-element CPU conversion cost
        // ~3ms of wall time on a 601-token prompt).
        let seq_f32 = if use_h16 {
            let b = nb(n_tokens * self.hidden * 4);
            write_buffer_f32(&b, embeddings);
            Some(b)
        } else {
            write_buffer_f32(&seq, embeddings);
            None
        };
        write_buffer_f32(&cos_buf, cos_all);
        write_buffer_f32(&sin_buf, sin_all);

        // Constant scalars shared by every layer.
        let rms_scalar = nb(8);
        let qkv_scalar = nb(16);
        let ffn_scalar = nb(20);
        let o_scalar = nb(12);
        let n_elems = nb(12);
        unsafe {
            let p = rms_scalar.contents() as *mut u8;
            *(p as *mut u32) = self.hidden as u32;
            *(p.add(4) as *mut f32) = self.eps;
            let q = qkv_scalar.contents() as *mut u32;
            *q = bpr_hidden as u32; // blocks per row over hidden
            *q.add(1) = q_dim as u32;
            *q.add(2) = kv_dim as u32;
            *q.add(3) = n_tokens as u32;
            let f = ffn_scalar.contents() as *mut u32;
            *f = bpr_hidden as u32; // gate/up: blocks per row over hidden
            *f.add(1) = self.ffn_dim as u32; // gate/up rows
            *f.add(2) = bpr_ffn as u32; // down: blocks per row over ffn
            *f.add(3) = self.hidden as u32; // down rows
            *f.add(4) = n_tokens as u32;
            let o = o_scalar.contents() as *mut u32;
            *o = bpr_q as u32;
            *o.add(1) = self.hidden as u32;
            *o.add(2) = n_tokens as u32;
            let n = n_elems.contents() as *mut u32;
            *n = (n_tokens * self.hidden) as u32;
            *n.add(1) = (n_tokens * self.ffn_dim) as u32;
            *n.add(2) = (n_tokens * q_dim) as u32;
        }
        // Batched RoPE / scatter / attention scalars — one set shared by every token row
        // (the per-token position is recovered from the grid's y coordinate in-kernel).
        let rope_q_scalar = nb(16);
        let rope_k_scalar = nb(16);
        let scatter_scalar = nb(16);
        let attn_scalar = nb(32);
        unsafe {
            let r = rope_q_scalar.contents() as *mut u32;
            *r = self.n_heads as u32;
            *r.add(1) = self.head_dim as u32;
            *r.add(2) = half_rope as u32;
            *r.add(3) = u32::from(self.split_half_pairing);
            let r = rope_k_scalar.contents() as *mut u32;
            *r = self.n_kv_heads as u32;
            *r.add(1) = self.head_dim as u32;
            *r.add(2) = half_rope as u32;
            *r.add(3) = u32::from(self.split_half_pairing);
            let sc = scatter_scalar.contents() as *mut u32;
            *sc = self.head_dim as u32;
            *sc.add(1) = self.max_positions as u32;
            *sc.add(2) = 0u32; // base position: prefill always starts an empty cache
            *sc.add(3) = kv_dim as u32;
            let a = attn_scalar.contents() as *mut u8;
            *(a as *mut u32) = self.n_heads as u32;
            *(a.add(4) as *mut u32) = self.head_dim as u32;
            *(a.add(8) as *mut u32) = (self.n_heads / self.n_kv_heads) as u32;
            *(a.add(12) as *mut f32) = scale;
            *(a.add(16) as *mut u32) = self.head_dim as u32; // position stride
            *(a.add(20) as *mut u32) = (self.max_positions * self.head_dim) as u32; // kv-head stride
            *(a.add(24) as *mut u32) = 0u32; // kv base offset
            *(a.add(28) as *mut u32) = n_tokens as u32;
        }
        // Per-head QK-norm scalar (Qwen3): head_dim(u32) | eps(f32) | use_weight(u32).
        // Shared by every layer; the per-layer q_norm/k_norm weights vary, the geometry
        // does not. Built only when QK-norm is in play.
        let perhead_qk_scalar = nb(12);
        if has_qk_norm {
            unsafe {
                let p = perhead_qk_scalar.contents() as *mut u8;
                *(p as *mut u32) = self.head_dim as u32;
                *(p.add(4) as *mut f32) = self.eps;
                *(p.add(8) as *mut u32) = 1u32; // use_weight
            }
        }

        // CAMELID_PREFILL_TRACE=1: split each stage into its own command buffer and
        // report accumulated hardware GPU-busy time per stage (per-stage waits inflate
        // wall time; the split is for attribution, not for production).
        let trace = std::env::var_os("CAMELID_PREFILL_TRACE").is_some();
        let mut stage_us: std::collections::BTreeMap<&'static str, u128> =
            std::collections::BTreeMap::new();
        let mut cb: metal::CommandBuffer = k.queue.new_command_buffer().to_owned();
        let mut e: metal::ComputeCommandEncoder = cb.new_compute_command_encoder().to_owned();
        macro_rules! stage {
            ($label:expr) => {
                if trace {
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    let (busy_us, _) = command_buffer_gpu_times_us(&cb);
                    *stage_us.entry($label).or_insert(0) += busy_us;
                    cb = k.queue.new_command_buffer().to_owned();
                    e = cb.new_compute_command_encoder().to_owned();
                }
            };
        }
        // The tiled-MM prefill GEMM needs 32-multiple row counts, paired Q8_0 blocks
        // (BK=64 = 2 blocks/step), and half activations — decided once for the whole
        // prefill since the activation buffers are emitted in the matching precision.
        // QK-norm (Qwen3) keeps the tiled-MM GEMM (the dominant prefill cost: weights
        // stream once per 64 tokens instead of once per 8) — only use_attn_mm/use_h16
        // are forced off so Q/K stay f32 [token][head][head_dim] for the per-head norm
        // and the validated non-attn-mm attention path. The MM GEMM here outputs f32
        // (use_h16 is false), so the per-head norm runs in f32 against the existing
        // rms_norm_per_head_f32 kernel.
        let use_mm = mm_prefill_enabled()
            && q_dim.is_multiple_of(128)
            && kv_dim.is_multiple_of(128)
            && self.hidden.is_multiple_of(128)
            && self.ffn_dim.is_multiple_of(128);
        // Half-precision GEMM inputs, padded to a 64-token multiple so the MM kernel's
        // direct device B loads never run off the end (padding rows are garbage; they
        // only feed output columns past n_tokens, which are never stored).
        // Attention-as-matmul scratch (prompts short enough to materialize S):
        // half K/V copies, half Q, half S scores and P probabilities per head.
        // Half KV mirrors are persistent per-layer session buffers (zero-filled at
        // allocation/growth): the attention-as-matmul prefill consumes them, the fallback
        // path's scatter fills them, and the split-K kv16 decode attention reads them.
        // The half KV mirrors are always real session buffers on this path (resident
        // prefill requires !kv16-primary), so the scatter always fills them.
        let kv16_flag = nb(4);
        unsafe {
            *(kv16_flag.contents() as *mut u32) = 1u32;
        }

        let q_h = nb(if use_attn_mm { n_pad * q_dim * 2 } else { 2 });
        let s_big = nb(if use_attn_mm {
            self.n_heads * n_pad * attn_qb * 2
        } else {
            2
        });
        let p_big = nb(if use_attn_mm {
            self.n_heads * n_pad * attn_qb * 2
        } else {
            2
        });
        let vt_scratch = nb(if use_attn_mm {
            self.n_kv_heads * self.head_dim * n_pad * 2
        } else {
            2
        });
        let fused_rope_scalar = nb(24);
        let vt_scalar = nb(16);
        unsafe {
            let p = fused_rope_scalar.contents() as *mut u32;
            *p = self.n_heads as u32;
            *p.add(1) = self.n_kv_heads as u32;
            *p.add(2) = self.head_dim as u32;
            *p.add(3) = half_rope as u32;
            *p.add(4) = u32::from(self.split_half_pairing);
            *p.add(5) = self.max_positions as u32;
        }
        unsafe {
            let p = vt_scalar.contents() as *mut u32;
            *p = self.head_dim as u32;
            *p.add(1) = self.max_positions as u32;
            *p.add(2) = n_pad as u32;
            *p.add(3) = n_tokens as u32;
        }
        let attn_mm_scalar = nb(48);
        unsafe {
            let p = attn_mm_scalar.contents() as *mut u32;
            // S pass: kdim=head_dim, rows=n_pad (positions), cols=n_tokens (queries)
            *p = self.head_dim as u32;
            *p.add(1) = n_pad as u32;
            *p.add(2) = n_tokens as u32;
            // PV pass: kdim=n_pad, rows=head_dim, cols=n_tokens
            *p.add(3) = n_pad as u32;
            *p.add(4) = self.head_dim as u32;
            *p.add(5) = n_tokens as u32;
            // shared strides and modes filled per-dispatch below
            *p.add(6) = (self.max_positions * self.head_dim) as u32; // kv batch stride
            *p.add(7) = (n_pad * attn_qb) as u32; // S/P batch stride (per query block)
            *p.add(8) = self.head_dim as u32; // q batch stride / kv row stride
            *p.add(9) = q_dim as u32; // q row stride (also ctx row stride)
            *p.add(10) = 1u32; // group=1 placeholder (real group below)
            *p.add(11) = gqa_group as u32;
        }
        let softmax_scalar = nb(16);
        unsafe {
            let p = softmax_scalar.contents() as *mut u32;
            *p = n_pad as u32;
            *p.add(1) = n_tokens as u32;
            *(p.add(2) as *mut f32) = scale;
            *p.add(3) = attn_qb as u32; // rows per query block
        }
        let normf_h = nb(if use_mm { n_pad * self.hidden * 2 } else { 2 });
        let ctx_h = nb(if use_mm { n_pad * q_dim * 2 } else { 2 });
        let silu_h = nb(if use_mm { n_pad * self.ffn_dim * 2 } else { 2 });
        let convert = |e: &metal::ComputeCommandEncoderRef,
                       src: &Buffer,
                       dst: &Buffer,
                       count_buf: &Buffer,
                       count_off: u64,
                       count: usize| {
            e.set_compute_pipeline_state(&k.f32_to_f16_pipeline);
            e.set_buffer(0, Some(src), 0);
            e.set_buffer(1, Some(dst), 0);
            e.set_buffer(2, Some(count_buf), count_off);
            dispatch_1d(e, &k.f32_to_f16_pipeline, count);
        };
        let gemm = |e: &metal::ComputeCommandEncoderRef,
                    y: &Buffer,
                    w: &Buffer,
                    out: &Buffer,
                    scalar: &Buffer,
                    bpr_off: u64,
                    rows_off: u64,
                    n_off: u64,
                    rows: usize| {
            if use_mm {
                // Simdgroup-matrix tiles: 64-row x 64-token output per threadgroup, so
                // weights stream once per 64 tokens instead of once per 8.
                e.set_compute_pipeline_state(if use_h16 {
                    &k.q8_0_block_wire_mm_f16o_pipeline
                } else {
                    &k.q8_0_block_wire_mm_pipeline
                });
                e.set_buffer(0, Some(y), 0);
                e.set_buffer(2, Some(w), 0);
                e.set_buffer(3, Some(out), 0);
                e.set_buffer(4, Some(scalar), bpr_off);
                e.set_buffer(5, Some(scalar), rows_off);
                e.set_buffer(6, Some(scalar), n_off);
                // A 128x32 half | B 64x32 half (A region doubles as tail scratch).
                e.set_threadgroup_memory_length(0, 12288);
                e.dispatch_thread_groups(
                    metal::MTLSize {
                        width: (rows / 64) as u64,
                        height: (n_tokens as u64).div_ceil(128),
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                return;
            }
            e.set_compute_pipeline_state(&k.q8_0_block_ksplit_f32y_wire_gemm_pipeline);
            e.set_buffer(0, Some(y), 0);
            e.set_buffer(2, Some(w), 0);
            e.set_buffer(3, Some(out), 0);
            e.set_buffer(4, Some(scalar), bpr_off);
            e.set_buffer(5, Some(scalar), rows_off);
            e.set_buffer(6, Some(scalar), n_off);
            e.set_threadgroup_memory_length(0, 2 * 32 * 4);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (rows as u64).div_ceil(2),
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        };
        // One thread per (unit, token): elementwise grid with token rows on y.
        let dispatch_rows = |e: &metal::ComputeCommandEncoderRef,
                             pipeline: &ComputePipelineState,
                             n: usize,
                             rows: usize| {
            let w = pipeline.thread_execution_width().max(1);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (n as u64).div_ceil(w),
                    height: rows as u64,
                    depth: 1,
                },
                metal::MTLSize {
                    width: w,
                    height: 1,
                    depth: 1,
                },
            );
        };
        // rms_norm_batch over n_tokens contiguous rows: one 256-thread group per row
        // (same group size as the per-token kernel, so the reduction is byte-exact).
        let norm_rows = |e: &metal::ComputeCommandEncoderRef,
                         pipeline: &ComputePipelineState,
                         input: &Buffer,
                         weight: &Buffer,
                         output: &Buffer,
                         scalar: &Buffer,
                         rows: usize| {
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(weight), 0);
            e.set_buffer(2, Some(output), 0);
            e.set_buffer(3, Some(scalar), 0);
            e.set_buffer(4, Some(scalar), 4);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: rows as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        };
        if let Some(seq_f32) = &seq_f32 {
            convert(&e, seq_f32, &seq, &n_elems, 0, n_tokens * self.hidden);
        }
        let (cur, nxt) = (&seq, &seq_out);
        for (i, layer) in layers.iter().enumerate() {
            // The resident prefill exists to fill the KV cache: the caller discards the
            // hidden stream and the last prompt token re-runs through the resident
            // single-token path. So the final layer only needs norm -> K/V GEMMs ->
            // rope/scatter; its Q projection, attention, O, and FFN outputs are dead.
            let last_layer = i + 1 == layers.len();
            let w = &resident[i];
            let attn_norm_buf;
            let ffn_norm_buf;
            {
                let mut cache = metal_linear_cache().lock().ok()?;
                attn_norm_buf = cache.weight_buffer(&k.device, layer.attn_norm);
                ffn_norm_buf = cache.weight_buffer(&k.device, layer.ffn_norm);
            }
            // Attention half: batched norm -> batched QKV GEMM -> batched rope/scatter ->
            // causal prefill attention (one dispatch, grid (n_heads, n_tokens)) -> O GEMM
            // -> residual.
            let qkv_y: &Buffer = if use_mm {
                norm_rows(
                    &e,
                    if use_h16 {
                        &k.rms_norm_batch_h_pipeline
                    } else {
                        &k.rms_norm_batch_f16o_pipeline
                    },
                    cur,
                    &attn_norm_buf,
                    &normf_h,
                    &rms_scalar,
                    n_tokens,
                );
                &normf_h
            } else {
                norm_rows(
                    &e,
                    &k.rms_norm_batch_pipeline,
                    cur,
                    &attn_norm_buf,
                    &normf,
                    &rms_scalar,
                    n_tokens,
                );
                &normf
            };
            stage!("1:norm+cvt");
            if !last_layer {
                gemm(&e, qkv_y, &w[0], &q_buf, &qkv_scalar, 0, 4, 12, q_dim);
            }
            gemm(&e, qkv_y, &w[1], &k_buf, &qkv_scalar, 0, 8, 12, kv_dim);
            gemm(&e, qkv_y, &w[2], &v_buf, &qkv_scalar, 0, 8, 12, kv_dim);
            stage!("2:gemm_qkv");
            // Qwen3 per-head QK-norm: in-place RMSNorm over each head's head_dim slice
            // of the f32 Q/K buffers (layout [token][head][head_dim] -> n_tokens*heads
            // contiguous groups), applied after the QKV GEMM and before RoPE. has_qk_norm
            // forces the f32 path, so q_buf/k_buf are f32 here.
            if let (Some(qn), Some(kn)) = (layer.q_norm, layer.k_norm) {
                let qnorm_buf = nb(self.head_dim * 4);
                let knorm_buf = nb(self.head_dim * 4);
                write_buffer_f32(&qnorm_buf, qn);
                write_buffer_f32(&knorm_buf, kn);
                if !last_layer {
                    encode_rms_norm_per_head(
                        &e,
                        k,
                        &q_buf,
                        &qnorm_buf,
                        &q_buf,
                        &perhead_qk_scalar,
                        n_tokens * self.n_heads,
                        0,
                    );
                }
                encode_rms_norm_per_head(
                    &e,
                    k,
                    &k_buf,
                    &knorm_buf,
                    &k_buf,
                    &perhead_qk_scalar,
                    n_tokens * self.n_kv_heads,
                    0,
                );
                stage!("2b:qk_norm");
            }
            let use_fused_rope = use_attn_mm && half_rope * 2 == self.head_dim;
            if use_fused_rope {
                // Fused rope(Q)->half panel + rope(K)->caches + V scatter, one dispatch.
                e.set_compute_pipeline_state(if use_h16 {
                    &k.rope_scatter_qh_h_pipeline
                } else {
                    &k.rope_scatter_qh_pipeline
                });
                e.set_buffer(0, Some(&q_buf), 0);
                e.set_buffer(1, Some(&k_buf), 0);
                e.set_buffer(2, Some(&v_buf), 0);
                e.set_buffer(3, Some(&q_h), 0);
                e.set_buffer(4, Some(&self.cache_k[i]), 0);
                e.set_buffer(5, Some(&self.cache_v[i]), 0);
                e.set_buffer(6, Some(&self.cache_k16[i]), 0);
                e.set_buffer(7, Some(&self.cache_v16[i]), 0);
                e.set_buffer(8, Some(&cos_buf), 0);
                e.set_buffer(9, Some(&sin_buf), 0);
                for j in 0..6u64 {
                    e.set_buffer(10 + j, Some(&fused_rope_scalar), j * 4);
                }
                dispatch_rows(
                    &e,
                    if use_h16 {
                        &k.rope_scatter_qh_h_pipeline
                    } else {
                        &k.rope_scatter_qh_pipeline
                    },
                    (self.n_heads + self.n_kv_heads) * half_rope + kv_dim,
                    n_tokens,
                );
            } else {
                e.set_compute_pipeline_state(&k.rope_rotate_batch_pipeline);
                e.set_buffer(0, Some(&q_buf), 0);
                e.set_buffer(1, Some(&cos_buf), 0);
                e.set_buffer(2, Some(&sin_buf), 0);
                for j in 0..4u64 {
                    e.set_buffer(3 + j, Some(&rope_q_scalar), j * 4);
                }
                dispatch_rows(
                    &e,
                    &k.rope_rotate_batch_pipeline,
                    self.n_heads * half_rope,
                    n_tokens,
                );
                e.set_compute_pipeline_state(&k.rope_rotate_batch_pipeline);
                e.set_buffer(0, Some(&k_buf), 0);
                e.set_buffer(1, Some(&cos_buf), 0);
                e.set_buffer(2, Some(&sin_buf), 0);
                for j in 0..4u64 {
                    e.set_buffer(3 + j, Some(&rope_k_scalar), j * 4);
                }
                dispatch_rows(
                    &e,
                    &k.rope_rotate_batch_pipeline,
                    self.n_kv_heads * half_rope,
                    n_tokens,
                );
                e.set_compute_pipeline_state(&k.kv_scatter_batch_pipeline);
                e.set_buffer(0, Some(&k_buf), 0);
                e.set_buffer(1, Some(&v_buf), 0);
                e.set_buffer(2, Some(&self.cache_k[i]), 0);
                e.set_buffer(3, Some(&self.cache_v[i]), 0);
                for j in 0..4u64 {
                    e.set_buffer(4 + j, Some(&scatter_scalar), j * 4);
                }
                e.set_buffer(8, Some(&self.cache_k16[i]), 0);
                e.set_buffer(9, Some(&self.cache_v16[i]), 0);
                e.set_buffer(10, Some(&kv16_flag), 0);
                dispatch_rows(&e, &k.kv_scatter_batch_pipeline, kv_dim, n_tokens);
            }
            stage!("3:rope+scatter");
            if last_layer {
                // KV cache writes for this layer are complete; everything below only
                // feeds the discarded hidden stream.
                continue;
            }
            // The flash-tiled prefill attention stages K/V and the P weights as half;
            // past ~1.6k positions on low-entropy prompts that quantization measurably
            // degrades attention (anchored-recall probes fail while the v3 kernel and the
            // CPU path agree), so v3 is the default beyond the attention-as-matmul cap and
            // flash stays opt-in. Moving the per-tile rescale diagonal to f32 was not
            // sufficient (probes still fail at 2k/8k); the remaining sink is the half
            // K/Q score staging, and fixing it needs an f32 staging layout that exceeds
            // the current threadgroup-memory budget — a redesign, not a patch.
            let use_flash_attn = std::env::var_os("CAMELID_METAL_FLASH_PREFILL")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                && use_mm
                && self.head_dim.is_multiple_of(8)
                && self.head_dim <= 128;
            if use_attn_mm {
                // Attention as two batched half GEMMs + a causal row softmax: the
                // matmuls ride the same staged simdgroup-tile structure as the
                // weight GEMMs, with upper-triangle S tiles culled and the PV k-range
                // clamped to the causal limit.
                // q -> half (already produced by the fused rope pass when eligible)
                if !use_fused_rope {
                    convert(&e, &q_buf, &q_h, &n_elems, 8, n_tokens * q_dim);
                }
                // S = Q K^T (per query head; K shared per KV head)
                #[allow(clippy::too_many_arguments)]
                let smm = |e: &metal::ComputeCommandEncoderRef,
                           a: &Buffer,
                           b_buf2: &Buffer,
                           b_off: u64,
                           c_buf2: &Buffer,
                           c_off: u64,
                           kdim_off: u64,
                           a_bs: u32,
                           b_bs: u32,
                           c_bs: u32,
                           a_rs: u32,
                           b_rs: u32,
                           c_rs: u32,
                           a_es: u32,
                           grp: u32,
                           mode: u32,
                           q_off: u32,
                           rows: usize,
                           cols: usize| {
                    e.set_compute_pipeline_state(&k.half_mm_batched_f16o_pipeline);
                    e.set_buffer(0, Some(a), 0);
                    e.set_buffer(1, Some(b_buf2), b_off);
                    e.set_buffer(2, Some(c_buf2), c_off);
                    e.set_buffer(3, Some(&attn_mm_scalar), kdim_off);
                    e.set_buffer(4, Some(&attn_mm_scalar), kdim_off + 4);
                    // dynamic scalars in a transient buffer (cols and the absolute
                    // query offset vary per query block)
                    let dyn_buf = nb(44);
                    unsafe {
                        let p = dyn_buf.contents() as *mut u32;
                        *p = a_bs;
                        *p.add(1) = b_bs;
                        *p.add(2) = c_bs;
                        *p.add(3) = a_rs;
                        *p.add(4) = b_rs;
                        *p.add(5) = c_rs;
                        *p.add(6) = a_es;
                        *p.add(7) = grp;
                        *p.add(8) = mode;
                        *p.add(9) = cols as u32;
                        *p.add(10) = q_off;
                    }
                    e.set_buffer(5, Some(&dyn_buf), 36);
                    for j in 0..9u64 {
                        e.set_buffer(6 + j, Some(&dyn_buf), j * 4);
                    }
                    e.set_buffer(15, Some(&dyn_buf), 40);
                    e.set_threadgroup_memory_length(0, 8192);
                    e.dispatch_thread_groups(
                        metal::MTLSize {
                            width: (rows as u64).div_ceil(64),
                            height: (cols as u64).div_ceil(64),
                            depth: self.n_heads as u64,
                        },
                        metal::MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                };
                // Transpose this layer's V slice ONCE so every query block's PV
                // A-staging reads contiguously.
                e.set_compute_pipeline_state(&k.transpose_v16_pipeline);
                e.set_buffer(0, Some(&self.cache_v16[i]), 0);
                e.set_buffer(1, Some(&vt_scratch), 0);
                for j in 0..4u64 {
                    e.set_buffer(2 + j, Some(&vt_scalar), j * 4);
                }
                e.dispatch_threads(
                    metal::MTLSize {
                        width: n_pad as u64,
                        height: self.head_dim as u64,
                        depth: self.n_kv_heads as u64,
                    },
                    metal::MTLSize {
                        width: 32,
                        height: 4,
                        depth: 1,
                    },
                );
                // Query-block loop: S and P panels hold attn_qb query columns at a time
                // (memory linear in the block width); q_offset keeps the causal mask and
                // PV clamp on absolute positions. One block reproduces the untiled math.
                let mut qb = 0usize;
                while qb < n_tokens {
                    let cols_b = (n_tokens - qb).min(attn_qb);
                    let qh_off = (qb * q_dim * 2) as u64;
                    // S = Q K^T (per query head; K shared per KV head)
                    smm(
                        &e,
                        &self.cache_k16[i],
                        &q_h,
                        qh_off,
                        &s_big,
                        0,
                        0,
                        (self.max_positions * self.head_dim) as u32,
                        self.head_dim as u32,
                        (n_pad * attn_qb) as u32,
                        self.head_dim as u32,
                        q_dim as u32,
                        n_pad as u32,
                        1,
                        gqa_group as u32,
                        1,
                        qb as u32,
                        n_pad,
                        cols_b,
                    );
                    // causal softmax rows -> P
                    let qoff_buf = nb(4);
                    unsafe {
                        *(qoff_buf.contents() as *mut u32) = qb as u32;
                    }
                    e.set_compute_pipeline_state(&k.softmax_causal_rows_pipeline);
                    e.set_buffer(0, Some(&s_big), 0);
                    e.set_buffer(1, Some(&p_big), 0);
                    e.set_buffer(2, Some(&softmax_scalar), 0);
                    e.set_buffer(3, Some(&softmax_scalar), 4);
                    e.set_buffer(4, Some(&softmax_scalar), 8);
                    e.set_buffer(5, Some(&qoff_buf), 0);
                    e.set_buffer(6, Some(&softmax_scalar), 12);
                    e.dispatch_thread_groups(
                        metal::MTLSize {
                            width: self.n_heads as u64,
                            height: (attn_qb as u64).div_ceil(8),
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    // O = P V into ctx_h [token][q_dim] half (head offset via c batch
                    // stride; this block's queries land at qh_off)
                    smm(
                        &e,
                        &vt_scratch,
                        &p_big,
                        0,
                        &ctx_h,
                        qh_off,
                        12,
                        (self.head_dim * n_pad) as u32,
                        (n_pad * attn_qb) as u32,
                        self.head_dim as u32,
                        n_pad as u32,
                        n_pad as u32,
                        q_dim as u32,
                        1,
                        gqa_group as u32,
                        2,
                        qb as u32,
                        self.head_dim,
                        cols_b,
                    );
                    qb += attn_qb;
                }
            } else if use_flash_attn {
                // Flash-tiled attention: K/V tiles stage once per 32-query tile and the
                // score/value matmuls ride the simdgroup matrix units.
                e.set_compute_pipeline_state(&k.attention_prefill_flash_pipeline);
            } else {
                e.set_compute_pipeline_state(&k.attention_prefill_v3_pipeline);
            }
            if !use_attn_mm {
                e.set_buffer(0, Some(&q_buf), 0);
                e.set_buffer(1, Some(&self.cache_k[i]), 0);
                e.set_buffer(2, Some(&self.cache_v[i]), 0);
                e.set_buffer(4, Some(&ctx_buf), 0);
                for j in 0..8u64 {
                    e.set_buffer(5 + j, Some(&attn_scalar), j * 4);
                }
                if use_flash_attn {
                    // Q + K/V half tiles (32 x head_dim each) | S 32x32 f32 | P 32x32 half
                    // | 4 x 8x8 half diag | 32 f32 inv-l.
                    e.set_threadgroup_memory_length(0, (128 * self.head_dim + 7296) as u64);
                    e.dispatch_thread_groups(
                        metal::MTLSize {
                            width: self.n_heads as u64,
                            height: (n_tokens as u64).div_ceil(32),
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    e.dispatch_thread_groups(
                        metal::MTLSize {
                            width: self.n_heads as u64,
                            height: (n_tokens as u64).div_ceil(4),
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
            let o_y: &Buffer = if use_mm {
                if !use_attn_mm {
                    // flash/v3 write f32 ctx; the MM path's PV pass wrote ctx_h directly
                    convert(&e, &ctx_buf, &ctx_h, &n_elems, 8, n_tokens * q_dim);
                }
                &ctx_h
            } else {
                &ctx_buf
            };
            stage!("4:attention");
            gemm(&e, o_y, &w[3], &o_buf, &o_scalar, 0, 4, 8, self.hidden);
            stage!("5:gemm_o");
            encode_binary_off(
                &e,
                if use_h16 {
                    &k.residual_add_h_pipeline
                } else {
                    &k.residual_add_pipeline
                },
                cur,
                &o_buf,
                nxt,
                &n_elems,
                0,
                n_tokens * self.hidden,
            );
            // FFN half.
            let ffn_y: &Buffer = if use_mm {
                norm_rows(
                    &e,
                    if use_h16 {
                        &k.rms_norm_batch_h_pipeline
                    } else {
                        &k.rms_norm_batch_f16o_pipeline
                    },
                    nxt,
                    &ffn_norm_buf,
                    &normf_h,
                    &rms_scalar,
                    n_tokens,
                );
                &normf_h
            } else {
                norm_rows(
                    &e,
                    &k.rms_norm_batch_pipeline,
                    nxt,
                    &ffn_norm_buf,
                    &normf,
                    &rms_scalar,
                    n_tokens,
                );
                &normf
            };
            stage!("6:norm2+cvt");
            gemm(
                &e,
                ffn_y,
                &w[4],
                &gate_buf,
                &ffn_scalar,
                0,
                4,
                16,
                self.ffn_dim,
            );
            gemm(
                &e,
                ffn_y,
                &w[5],
                &up_buf,
                &ffn_scalar,
                0,
                4,
                16,
                self.ffn_dim,
            );
            stage!("7:gemm_gateup");
            if use_mm {
                encode_binary_off(
                    &e,
                    if use_h16 {
                        &k.silu_mul_h2_pipeline
                    } else {
                        &k.silu_mul_f16o_pipeline
                    },
                    &gate_buf,
                    &up_buf,
                    &silu_h,
                    &n_elems,
                    4,
                    n_tokens * self.ffn_dim,
                );
            } else {
                encode_binary_off(
                    &e,
                    &k.silu_mul_pipeline,
                    &gate_buf,
                    &up_buf,
                    &silu_buf,
                    &n_elems,
                    4,
                    n_tokens * self.ffn_dim,
                );
            }
            let down_y: &Buffer = if use_mm { &silu_h } else { &silu_buf };
            gemm(
                &e,
                down_y,
                &w[6],
                &down_buf,
                &ffn_scalar,
                8,
                12,
                16,
                self.hidden,
            );
            stage!("8:gemm_down");
            encode_binary_off(
                &e,
                if use_h16 {
                    &k.residual_add_h_pipeline
                } else {
                    &k.residual_add_pipeline
                },
                nxt,
                &down_buf,
                cur,
                &n_elems,
                0,
                n_tokens * self.hidden,
            );
            stage!("9:resid+silu");
            // Attention residual wrote cur -> nxt; FFN residual wrote nxt -> cur, so this
            // layer's output is back in `cur` for the next layer.

            // Commit the first layer as its own command buffer so the GPU starts (~35ms
            // of work) while the host encodes the remaining layers (~10ms): the encode
            // cost disappears behind GPU execution. The serial queue keeps ordering.
            if i == 0 && !trace {
                e.end_encoding();
                cb.commit();
                cb = k.queue.new_command_buffer().to_owned();
                e = cb.new_compute_command_encoder().to_owned();
            }
        }
        e.end_encoding();
        let time_edges = std::env::var_os("CAMELID_PREFILL_TIME").is_some();
        let commit_started = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        if time_edges && !trace {
            let (busy_us, window_us) = command_buffer_gpu_times_us(&cb);
            eprintln!(
                "[prefill-time] commit->complete {:.1}ms | gpu busy {:.1}ms | gpu window {:.1}ms",
                commit_started.elapsed().as_micros() as f64 / 1000.0,
                busy_us as f64 / 1000.0,
                window_us as f64 / 1000.0,
            );
        }
        if trace {
            let total: u128 = stage_us.values().sum();
            eprintln!(
                "[prefill-trace] n_tokens={n_tokens} total_gpu_busy={}ms",
                total / 1000
            );
            for (label, us) in &stage_us {
                eprintln!("[prefill-trace]   {label}: {}ms", us / 1000);
            }
        }
        self.filled = n_tokens;
        Some(())
    }

    /// WIN2METAL Phase 3: speculative-verify forward over `k` rows at sequence positions
    /// `[base_position, base_position + k)`. BIT-IDENTICAL to `k` independent
    /// `forward_token` decodes at those positions: the input RMSNorm/projections/FFN run as
    /// the proven byte-exact batched kernels (`rms_norm_batch_f32`, the C0 batched-column
    /// GEMV `encode_q8_matmul_f32y_batched`, elementwise residual/silu), while per-head
    /// QK-norm, RoPE, K/V scatter, attention and argmax run the EXACT single-token kernels
    /// once per row at a row byte-offset (so cos/sin pairing, scatter slot, attention
    /// v2/split-K routing and argmax tie-break are reproduced verbatim). All `k` K/V are
    /// scattered into `[base..base+k)` before any attention; row `i` attends over
    /// `position_count = base+i+1`, so it sees only `[0, base+i]` (its own slot + the prior
    /// rows), exactly as a standalone decode would. Does NOT advance `filled` — the host
    /// accept loop sets `filled = base + accepted.len()` and the rejected tail slots are
    /// overwritten by the next single-token decode. Returns the `k` greedy argmax ids, or
    /// `None` (lossless fallback) on any unsupported config.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Option<Vec<u32>> {
        self.verify_batch_inner(
            embeddings,
            cos_all,
            sin_all,
            layers,
            logits,
            base_position,
            k,
            scale,
            false,
            None,
        )
        .map(|(preds, _)| preds)
    }

    /// `verify_batch` that also reads back the `k * vocab` pre-argmax logits (the byte-exact
    /// gate compares logits by exact `to_bits`, which a logits bug that doesn't flip the
    /// argmax would slip past). Test-only: the production lane never pays the logits readback.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn verify_batch_logits(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Option<(Vec<u32>, Vec<f32>)> {
        self.verify_batch_inner(
            embeddings,
            cos_all,
            sin_all,
            layers,
            logits,
            base_position,
            k,
            scale,
            true,
            None,
        )
    }

    /// Maximum verify window (mirrors the CUDA host's `MAX_VERIFY_K`).
    const MAX_VERIFY_K: usize = 8;

    #[allow(clippy::too_many_arguments)]
    fn verify_batch_inner(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        base_position: usize,
        k: usize,
        scale: f32,
        read_logits: bool,
        tree: Option<&TreeAttn>,
    ) -> Option<(Vec<u32>, Vec<f32>)> {
        // ---- Eligibility gate (return None -> caller falls back, lossless) --------------
        // The tree path widens the node cap to TREE_MAX_NODES (a tree of N nodes has at most
        // depth N-1); the linear path keeps the MAX_VERIFY_K window. `None`-tree callers see
        // the original bound, so the linear path is byte-unchanged.
        let k_cap = if tree.is_some() {
            crate::inference::spec_tree::TREE_MAX_NODES
        } else {
            Self::MAX_VERIFY_K
        };
        if !(1..=k_cap).contains(&k) {
            return None;
        }
        if self.kv16 {
            return None; // f32 KV cache only for Phase 3 (kv16 is a follow-up)
        }
        // The tree split-K attention (encode_attention_tree) is mirror-only — there is no f32
        // split-K *tree* kernel. The CAMELID_METAL_ATTN_SPLITK_KV16=0 opt-out forces forward_token
        // and the LINEAR verify (encode_attention) onto the f32 split-K reads, so a tree verify
        // reading the f16 mirrors instead would diverge from plain greedy (NOT lossless). Fail
        // closed to the lossless single-token/CPU fallback rather than emit a non-lossless tree
        // verify. The linear path honors the env in encode_attention, so it is unaffected and
        // stays byte-exact; only the tree lane needs this guard. (Mirror predicate matches the
        // `use_mirrors` test in encode_attention.)
        if tree.is_some()
            && std::env::var("CAMELID_METAL_ATTN_SPLITK_KV16")
                .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        {
            return None;
        }
        // The new batched GEMV mirrors exactly the f32y + wire + NSG=8 production GEMV; any
        // other GEMV path would not match it bit-for-bit.
        if !(f32y_gemv_enabled() && wire_weights_enabled() && wire_nsg8_enabled()) {
            return None;
        }
        if !self.head_dim.is_multiple_of(32) || self.head_dim > 128 {
            return None; // v2 / split-K attention precondition
        }
        if base_position != self.filled() {
            return None;
        }
        if base_position + k > self.cap {
            return None;
        }
        if layers.len() != self.n_layers || embeddings.len() != k * self.hidden {
            return None;
        }
        if cos_all.len() != sin_all.len() || cos_all.is_empty() || !cos_all.len().is_multiple_of(k)
        {
            return None;
        }
        let half_rope = cos_all.len() / k;
        if half_rope * 2 > self.head_dim {
            return None;
        }
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let bpr_hidden = self.hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = self.ffn_dim / 32;
        for l in layers {
            if l.attn_norm.len() != self.hidden
                || l.ffn_norm.len() != self.hidden
                || l.q_weight_blocks.block_count() != q_dim * bpr_hidden
                || l.k_weight_blocks.block_count() != kv_dim * bpr_hidden
                || l.v_weight_blocks.block_count() != kv_dim * bpr_hidden
                || l.o_weight_blocks.block_count() != self.hidden * bpr_q
                || l.gate_weight_blocks.block_count() != self.ffn_dim * bpr_hidden
                || l.up_weight_blocks.block_count() != self.ffn_dim * bpr_hidden
                || l.down_weight_blocks.block_count() != self.hidden * bpr_ffn
            {
                return None;
            }
        }
        let vocab = logits.vocab_size;
        if vocab == 0
            || logits.final_norm.len() != self.hidden
            || logits.output_weight_blocks.block_count() != vocab * bpr_hidden
        {
            return None;
        }

        // A pre-committed pending graph (from forward_token's encode-ahead) sits gated on the
        // serial queue; release it so this command buffer is not ordered behind a graph that
        // never gets signaled (which would deadlock the wait below).
        if let Some(stale) = self.pending.take() {
            self.release_stale(stale);
        }
        self.pending_signaled = false;
        if !self.ensure_capacity(base_position + k) {
            return None;
        }

        let kern = metal_linear_kernel()?;
        let max_positions = self.max_positions;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let hidden = self.hidden;
        let ffn_dim = self.ffn_dim;
        let eps = self.eps;
        let pairing = u32::from(self.split_half_pairing);

        // ---- Resolve resident weights (wire format; gated on f32y+wire above) ------------
        let resident: Vec<[Buffer; 7]>;
        let attn_norm_bufs: Vec<Buffer>;
        let ffn_norm_bufs: Vec<Buffer>;
        let qk_norm_bufs: Vec<Option<(Buffer, Buffer)>>;
        let ow_buf: Buffer;
        let final_norm_buf: Buffer;
        {
            let mut cache = metal_linear_cache().lock().ok()?;
            let mut wb = |w: &ResidentWeightBytes| match w {
                ResidentWeightBytes::Blocks36(blocks) => {
                    Some(cache.q8_wire_weight_buffer(&kern.device, blocks))
                }
                ResidentWeightBytes::WirePages(pages) => {
                    Some(cache.q8_wire_nocopy_buffer(&kern.device, pages))
                }
            };
            resident = layers
                .iter()
                .map(|l| {
                    Some([
                        wb(&l.q_weight_blocks)?,
                        wb(&l.k_weight_blocks)?,
                        wb(&l.v_weight_blocks)?,
                        wb(&l.o_weight_blocks)?,
                        wb(&l.gate_weight_blocks)?,
                        wb(&l.up_weight_blocks)?,
                        wb(&l.down_weight_blocks)?,
                    ])
                })
                .collect::<Option<Vec<_>>>()?;
            ow_buf = wb(&logits.output_weight_blocks)?;
            attn_norm_bufs = layers
                .iter()
                .map(|l| cache.weight_buffer(&kern.device, l.attn_norm))
                .collect();
            ffn_norm_bufs = layers
                .iter()
                .map(|l| cache.weight_buffer(&kern.device, l.ffn_norm))
                .collect();
            qk_norm_bufs = layers
                .iter()
                .map(|l| match (l.q_norm, l.k_norm) {
                    (Some(qn), Some(kn)) => Some((
                        cache.weight_buffer(&kern.device, qn),
                        cache.weight_buffer(&kern.device, kn),
                    )),
                    _ => None,
                })
                .collect();
            final_norm_buf = cache.weight_buffer(&kern.device, logits.final_norm);
        }

        // ---- Buffers (row-major [token][dim]; token = verify row) -----------------------
        let nb = |bytes: usize| {
            kern.device
                .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
        };
        let act_a = nb(k * hidden * 4);
        let act_b = nb(k * hidden * 4);
        let mid = nb(k * hidden * 4);
        let norm_buf = nb(k * hidden * 4);
        let q_buf = nb(k * q_dim * 4);
        let k_buf = nb(k * kv_dim * 4);
        let v_buf = nb(k * kv_dim * 4);
        let ctx_buf = nb(k * q_dim * 4);
        let o_buf = nb(k * hidden * 4);
        let gate_buf = nb(k * ffn_dim * 4);
        let up_buf = nb(k * ffn_dim * 4);
        let silu_buf = nb(k * ffn_dim * 4);
        let down_buf = nb(k * hidden * 4);
        let fnorm_buf = nb(k * hidden * 4);
        let logits_buf = nb(k * vocab * 4);
        let pred_buf = nb(k * 4);
        let cos_buf = nb(cos_all.len() * 4);
        let sin_buf = nb(sin_all.len() * 4);
        // Attention scores scratch (only read by the non-v2 fallback kernel); size to the
        // deepest row's position_count = base+k.
        let scores_buf = nb(n_heads * (base_position + k) * 4);
        write_buffer_f32(&act_a, embeddings);
        write_buffer_f32(&cos_buf, cos_all);
        write_buffer_f32(&sin_buf, sin_all);

        // ---- Scalars --------------------------------------------------------------------
        let rms_scalar = nb(8); // width (hidden), eps — shared by attn/ffn/final norms
        let q_gemv = nb(12); // blocks_per_row, rows, [n_rows_in set by helper]
        let kv_gemv = nb(12);
        let o_gemv = nb(12);
        let gateup_gemv = nb(12);
        let down_gemv = nb(12);
        let out_gemv = nb(12);
        let rope_q_scalar = nb(16);
        let rope_k_scalar = nb(16);
        let silu_n = nb(4);
        let resid_n = nb(4);
        let kv16_write = nb(4);
        let argmax_count = nb(4);
        let perhead_qk_scalar = nb(12);
        unsafe {
            let p = rms_scalar.contents() as *mut u8;
            *(p as *mut u32) = hidden as u32;
            *(p.add(4) as *mut f32) = eps;
            let set_gemv = |buf: &Buffer, bpr: usize, rows: usize| {
                let q = buf.contents() as *mut u32;
                *q = bpr as u32;
                *q.add(1) = rows as u32;
            };
            set_gemv(&q_gemv, bpr_hidden, q_dim);
            set_gemv(&kv_gemv, bpr_hidden, kv_dim);
            set_gemv(&o_gemv, bpr_q, hidden);
            set_gemv(&gateup_gemv, bpr_hidden, ffn_dim);
            set_gemv(&down_gemv, bpr_ffn, hidden);
            set_gemv(&out_gemv, bpr_hidden, vocab);
            let set_rope = |buf: &Buffer, hc: usize| {
                let r = buf.contents() as *mut u32;
                *r = hc as u32;
                *r.add(1) = head_dim as u32;
                *r.add(2) = half_rope as u32;
                *r.add(3) = pairing;
            };
            set_rope(&rope_q_scalar, n_heads);
            set_rope(&rope_k_scalar, n_kv_heads);
            *(silu_n.contents() as *mut u32) = (k * ffn_dim) as u32;
            *(resid_n.contents() as *mut u32) = (k * hidden) as u32;
            *(kv16_write.contents() as *mut u32) = 1; // !kv16 -> mirrors are real, dual-write
            *(argmax_count.contents() as *mut u32) = vocab as u32;
            let p = perhead_qk_scalar.contents() as *mut u8;
            *(p as *mut u32) = head_dim as u32;
            *(p.add(4) as *mut f32) = eps;
            *(p.add(8) as *mut u32) = 1; // use_weight
        }
        // Per-row scatter scalars [head_dim, max_positions, base+i, kv_dim] and per-row
        // attention scalars (32 bytes) — write_position / position_count vary per row, so
        // each row gets its own storage (the command buffer reads them at execution time).
        let scatter_scalars = nb(16 * k);
        let mut attn_scalars: Vec<Buffer> = Vec::with_capacity(k);
        for i in 0..k {
            unsafe {
                let s = (scatter_scalars.contents() as *mut u8).add(i * 16) as *mut u32;
                *s = head_dim as u32;
                *s.add(1) = max_positions as u32;
                *s.add(2) = (base_position + i) as u32;
                *s.add(3) = kv_dim as u32;
            }
            // position_count: linear row i covers prefix+self = base+i+1; a tree node covers
            // prefix + its ancestor drafts (incl. self) = base + tail_count.
            let pc_i = match tree {
                Some(t) => t.base + t.tail_slots[i].len(),
                None => base_position + i + 1,
            };
            let a = nb(32);
            unsafe {
                let p = a.contents() as *mut u8;
                *(p as *mut u32) = n_heads as u32;
                *(p.add(4) as *mut u32) = head_dim as u32;
                *(p.add(8) as *mut u32) = pc_i as u32; // position_count
                *(p.add(12) as *mut u32) = (n_heads / n_kv_heads) as u32;
                *(p.add(16) as *mut f32) = scale;
                *(p.add(20) as *mut u32) = head_dim as u32; // position stride
                *(p.add(24) as *mut u32) = (max_positions * head_dim) as u32; // kv-head stride
                *(p.add(28) as *mut u32) = 0u32; // kv base offset
            }
            attn_scalars.push(a);
        }

        // ---- Per-node tree-attention buffers (tree path only) ---------------------------
        // `base` (the contiguous prefix length) is shared; each node gets its own
        // `tail_slots` buffer (its ancestor draft cache slots, increasing). The linear path
        // leaves these empty and dispatches the unchanged `encode_attention`.
        let (tree_base_buf, tree_tail_bufs): (Option<Buffer>, Vec<Buffer>) = if let Some(t) = tree {
            let bb = nb(4);
            unsafe {
                *(bb.contents() as *mut u32) = t.base as u32;
            }
            let bufs: Vec<Buffer> = (0..k)
                .map(|i| {
                    let slots = &t.tail_slots[i];
                    let buf = nb(slots.len().max(1) * 4);
                    unsafe {
                        let p = buf.contents() as *mut u32;
                        for (j, &s) in slots.iter().enumerate() {
                            *p.add(j) = s;
                        }
                    }
                    buf
                })
                .collect();
            (Some(bb), bufs)
        } else {
            (None, Vec::new())
        };

        // ---- Encode the whole verify window into one serial compute encoder -------------
        let mut keep: Vec<Buffer> = Vec::new();
        let cb = kern.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        let mut from_a = true;
        for l in 0..layers.len() {
            let (cur, nxt) = if from_a {
                (&act_a, &act_b)
            } else {
                (&act_b, &act_a)
            };
            let w = &resident[l];
            // --- Attention block ---
            // 1. input RMSNorm (batched, byte-exact vs single rms_norm_f32 per row)
            encode_rms_norm_batch(kern, e, cur, &attn_norm_bufs[l], &norm_buf, &rms_scalar, k);
            // 2. Q/K/V projections (batched-column GEMV; column t == single-token row t)
            encode_q8_matmul_f32y_batched(e, kern, &norm_buf, &w[0], &q_buf, &q_gemv, q_dim, k);
            encode_q8_matmul_f32y_batched(e, kern, &norm_buf, &w[1], &k_buf, &kv_gemv, kv_dim, k);
            encode_q8_matmul_f32y_batched(e, kern, &norm_buf, &w[2], &v_buf, &kv_gemv, kv_dim, k);
            // 3. per-head Q/K-norm (Qwen3) — per row, in place
            if let Some((qn_buf, kn_buf)) = &qk_norm_bufs[l] {
                for i in 0..k {
                    encode_rms_norm_per_head(
                        e,
                        kern,
                        &q_buf,
                        qn_buf,
                        &q_buf,
                        &perhead_qk_scalar,
                        n_heads,
                        (i * q_dim * 4) as u64,
                    );
                    encode_rms_norm_per_head(
                        e,
                        kern,
                        &k_buf,
                        kn_buf,
                        &k_buf,
                        &perhead_qk_scalar,
                        n_kv_heads,
                        (i * kv_dim * 4) as u64,
                    );
                }
            }
            // 4. RoPE — per row (position base+i; cos/sin position-major, stride half_rope)
            for i in 0..k {
                encode_rope(
                    e,
                    kern,
                    &q_buf,
                    &cos_buf,
                    &sin_buf,
                    &rope_q_scalar,
                    n_heads,
                    half_rope,
                    (i * q_dim * 4) as u64,
                    (i * half_rope * 4) as u64,
                );
                encode_rope(
                    e,
                    kern,
                    &k_buf,
                    &cos_buf,
                    &sin_buf,
                    &rope_k_scalar,
                    n_kv_heads,
                    half_rope,
                    (i * kv_dim * 4) as u64,
                    (i * half_rope * 4) as u64,
                );
            }
            // 5. K/V scatter — per row into slot base+i, dual-writing the f16 mirrors the
            //    split-K decode attention reads (ALL k before any attention reads).
            for i in 0..k {
                e.set_compute_pipeline_state(&kern.kv_scatter_pipeline);
                e.set_buffer(0, Some(&k_buf), (i * kv_dim * 4) as u64);
                e.set_buffer(1, Some(&v_buf), (i * kv_dim * 4) as u64);
                e.set_buffer(2, Some(&self.cache_k[l]), 0);
                e.set_buffer(3, Some(&self.cache_v[l]), 0);
                e.set_buffer(4, Some(&scatter_scalars), (i * 16) as u64);
                e.set_buffer(5, Some(&scatter_scalars), (i * 16 + 4) as u64);
                e.set_buffer(6, Some(&scatter_scalars), (i * 16 + 8) as u64);
                e.set_buffer(7, Some(&scatter_scalars), (i * 16 + 12) as u64);
                e.set_buffer(8, Some(&self.cache_k16[l]), 0);
                e.set_buffer(9, Some(&self.cache_v16[l]), 0);
                e.set_buffer(10, Some(&kv16_write), 0);
                dispatch_1d(e, &kern.kv_scatter_pipeline, kv_dim);
            }
            // 6. attention — per row. position_count auto-routes v2 vs split-K exactly like
            //    forward_token at that pc; reads the f16 mirrors. The None (linear) path
            //    dispatches the unchanged `encode_attention` (pc = base+i+1); the tree path
            //    dispatches the slot-indirected `encode_attention_tree` (pc = base+tail_count)
            //    with this node's ancestor draft slots.
            for (i, attn_scalar) in attn_scalars.iter().enumerate() {
                match tree {
                    None => encode_attention(
                        e,
                        kern,
                        &mut keep,
                        &q_buf,
                        &self.cache_k[l],
                        &self.cache_v[l],
                        Some((&self.cache_k16[l], &self.cache_v16[l])),
                        &scores_buf,
                        &ctx_buf,
                        attn_scalar,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        base_position + i + 1,
                        (i * q_dim * 4) as u64,
                        (i * q_dim * 4) as u64,
                    ),
                    Some(t) => encode_attention_tree(
                        e,
                        kern,
                        &mut keep,
                        &q_buf,
                        &self.cache_k[l],
                        &self.cache_v[l],
                        Some((&self.cache_k16[l], &self.cache_v16[l])),
                        &scores_buf,
                        &ctx_buf,
                        attn_scalar,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        t.base + t.tail_slots[i].len(),
                        (i * q_dim * 4) as u64,
                        (i * q_dim * 4) as u64,
                        tree_base_buf.as_ref().expect("tree base buffer present"),
                        &tree_tail_bufs[i],
                    ),
                }
            }
            // 7. O projection (batched), 8. attention residual (batched)
            encode_q8_matmul_f32y_batched(e, kern, &ctx_buf, &w[3], &o_buf, &o_gemv, hidden, k);
            encode_binary(
                e,
                &kern.residual_add_pipeline,
                cur,
                &o_buf,
                &mid,
                &resid_n,
                k * hidden,
            );
            // --- FFN block (all batched / elementwise) ---
            encode_rms_norm_batch(kern, e, &mid, &ffn_norm_bufs[l], &norm_buf, &rms_scalar, k);
            encode_q8_matmul_f32y_batched(
                e,
                kern,
                &norm_buf,
                &w[4],
                &gate_buf,
                &gateup_gemv,
                ffn_dim,
                k,
            );
            encode_q8_matmul_f32y_batched(
                e,
                kern,
                &norm_buf,
                &w[5],
                &up_buf,
                &gateup_gemv,
                ffn_dim,
                k,
            );
            encode_binary(
                e,
                &kern.silu_mul_pipeline,
                &gate_buf,
                &up_buf,
                &silu_buf,
                &silu_n,
                k * ffn_dim,
            );
            encode_q8_matmul_f32y_batched(
                e, kern, &silu_buf, &w[6], &down_buf, &down_gemv, hidden, k,
            );
            encode_binary(
                e,
                &kern.residual_add_pipeline,
                &mid,
                &down_buf,
                nxt,
                &resid_n,
                k * hidden,
            );
            from_a = !from_a;
        }
        let final_buf = if from_a { &act_a } else { &act_b };
        // ---- Final stage: norm -> output GEMV -> per-row argmax -------------------------
        encode_rms_norm_batch(
            kern,
            e,
            final_buf,
            &final_norm_buf,
            &fnorm_buf,
            &rms_scalar,
            k,
        );
        encode_q8_matmul_f32y_batched(
            e,
            kern,
            &fnorm_buf,
            &ow_buf,
            &logits_buf,
            &out_gemv,
            vocab,
            k,
        );
        for i in 0..k {
            e.set_compute_pipeline_state(&kern.argmax_f32_greedy_pipeline);
            e.set_buffer(0, Some(&logits_buf), (i * vocab * 4) as u64);
            e.set_buffer(1, Some(&pred_buf), (i * 4) as u64);
            e.set_buffer(2, Some(&argmax_count), 0);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
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
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        // Note: `filled` is intentionally NOT advanced — the host accept loop sets it.
        let preds: Vec<u32> = (0..k)
            .map(|i| unsafe { *(pred_buf.contents() as *const u32).add(i) })
            .collect();
        let logits_out = if read_logits {
            let mut out = vec![0.0f32; k * vocab];
            read_buffer_f32(&logits_buf, &mut out);
            out
        } else {
            Vec::new()
        };
        Some((preds, logits_out))
    }

    /// WIN2METAL Phase 4 — TREE speculative verify. Batched forward over the `n` BFS-ordered
    /// nodes of a draft tree: each node attends `prefix [0, base_position)` ∪ its ancestor
    /// draft slots (the rest of the pass — RoPE / norms / GEMV / K/V scatter / argmax — is
    /// reused unchanged from the linear `verify_batch`). Returns one greedy argmax per node
    /// (`predicted[i]` = the target's next-token after node `i`), or `None` on any
    /// unsupported config (caller falls back, lossless).
    ///
    /// A LINEAR tree drives the slot indirection to the identity, so this is byte-identical
    /// to `verify_batch` (proven by `metal_tree_verify_bit_identical`).
    ///
    /// - `embeddings`: `n*hidden`, BFS node order (already looked up).
    /// - `cos_all`/`sin_all`: per-NODE RoPE tables at position `base_position + depth[i]`.
    /// - `node_kvslot[i]` must equal `base_position + i` (the scatter target, BFS order).
    /// - `(ancestor_bits, words)` is `TokenTree::ancestor_bitset()`.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch_tree(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        node_kvslot: &[i32],
        ancestor_bits: &[u32],
        words: usize,
        base_position: usize,
        n: usize,
        scale: f32,
    ) -> Option<Vec<u32>> {
        let tree = self.build_tree_attn(node_kvslot, ancestor_bits, words, base_position, n)?;
        self.verify_batch_inner(
            embeddings,
            cos_all,
            sin_all,
            layers,
            logits,
            base_position,
            n,
            scale,
            false,
            Some(&tree),
        )
        .map(|(preds, _)| preds)
    }

    /// `verify_batch_tree` that also reads back the `n * vocab` pre-argmax logits (the gate
    /// compares logits by exact `to_bits`). Test-only.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn verify_batch_tree_logits(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        node_kvslot: &[i32],
        ancestor_bits: &[u32],
        words: usize,
        base_position: usize,
        n: usize,
        scale: f32,
    ) -> Option<(Vec<u32>, Vec<f32>)> {
        let tree = self.build_tree_attn(node_kvslot, ancestor_bits, words, base_position, n)?;
        self.verify_batch_inner(
            embeddings,
            cos_all,
            sin_all,
            layers,
            logits,
            base_position,
            n,
            scale,
            true,
            Some(&tree),
        )
    }

    /// Build the per-node `TreeAttn` descriptor from the ancestor bitset. For node `i`, scan
    /// bits `0..n`; a set bit `j` (an ancestor-or-self) contributes cache slot
    /// `base_position + j`, collected in increasing `j` so the kernel's split-K boundaries
    /// match the linear path. Validates the shared invariants; returns `None` on any mismatch.
    fn build_tree_attn(
        &self,
        node_kvslot: &[i32],
        ancestor_bits: &[u32],
        words: usize,
        base_position: usize,
        n: usize,
    ) -> Option<TreeAttn> {
        if n == 0 || words != n.div_ceil(32) || ancestor_bits.len() != n * words {
            return None;
        }
        if node_kvslot.len() != n {
            return None;
        }
        // Scatter target is reused unchanged: node i -> slot base+i (BFS order).
        for (i, &slot) in node_kvslot.iter().enumerate() {
            if slot != (base_position + i) as i32 {
                return None;
            }
        }
        let mut tail_slots: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut slots: Vec<u32> = Vec::new();
            for j in 0..n {
                let word = ancestor_bits[i * words + j / 32];
                if (word >> (j % 32)) & 1 == 1 {
                    slots.push((base_position + j) as u32);
                }
            }
            // A node must at least attend itself (bit i set); a malformed bitset is unsupported.
            if slots.is_empty() {
                return None;
            }
            tail_slots.push(slots);
        }
        Some(TreeAttn {
            base: base_position,
            tail_slots,
        })
    }

    /// WIN2METAL Phase 4 — compact the accepted tree path's K/V into the contiguous slots
    /// `[base, base + path.len())`, so subsequent linear decode/attention reads it as an
    /// ordinary history. `path` is `TokenTree::path_to(leaf)` (root-first BFS node indices,
    /// `path[r] >= r`). For each rank `r` with `path[r] != r`, copies the `head_dim` row of
    /// slot `base + path[r]` into slot `base + r` for EVERY layer and kv_head, in BOTH the
    /// f32 cache and the f16 mirrors (subsequent decode reads both). Exploits unified memory:
    /// a direct `contents()` memcpy, no device round-trip. A linear path `[0..L-1]` is a
    /// no-op (the byte-identity anchor — a linear tree leaves the cache exactly as a linear
    /// decode would).
    pub fn compact_tree_kv_path(&mut self, path: &[usize], base: usize) -> Result<(), String> {
        let head_dim = self.head_dim;
        let max_positions = self.max_positions;
        let n_kv = self.n_kv_heads;
        let row = head_dim; // elements per (kv_head, slot) row
        for (r, &src_node) in path.iter().enumerate() {
            if src_node == r {
                continue; // already in place (incl. every rank of a linear path)
            }
            // CUDA invariant: path[r] >= r, so the source slot is never an earlier write's
            // destination — the in-place gather is order-independent.
            let dst_slot = base + r;
            let src_slot = base + src_node;
            if dst_slot >= max_positions || src_slot >= max_positions {
                return Err(format!(
                    "compact_tree_kv_path: slot out of range (src {src_slot}, dst {dst_slot}, \
                     max_positions {max_positions})"
                ));
            }
            for l in 0..self.n_layers {
                unsafe {
                    let kp = self.cache_k[l].contents() as *mut f32;
                    let vp = self.cache_v[l].contents() as *mut f32;
                    for h in 0..n_kv {
                        let src_off = (h * max_positions + src_slot) * head_dim;
                        let dst_off = (h * max_positions + dst_slot) * head_dim;
                        std::ptr::copy_nonoverlapping(kp.add(src_off), kp.add(dst_off), row);
                        std::ptr::copy_nonoverlapping(vp.add(src_off), vp.add(dst_off), row);
                    }
                }
                // f16 mirrors (empty in kv16-primary mode, where the primary caches are half).
                if !self.cache_k16.is_empty() {
                    unsafe {
                        let kp = self.cache_k16[l].contents() as *mut u16;
                        let vp = self.cache_v16[l].contents() as *mut u16;
                        for h in 0..n_kv {
                            let src_off = (h * max_positions + src_slot) * head_dim;
                            let dst_off = (h * max_positions + dst_slot) * head_dim;
                            std::ptr::copy_nonoverlapping(kp.add(src_off), kp.add(dst_off), row);
                            std::ptr::copy_nonoverlapping(vp.add(src_off), vp.add(dst_off), row);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Release a stale pre-committed token graph: it sits on the serial queue gated behind
    /// its event, so signal it and let it run once against the old buffers (its outputs go
    /// to scratch that the next real token overwrites). Skipping this would deadlock the
    /// queue. Happens at most once per KV growth or sequence restart.
    fn release_stale(&mut self, stale: PreparedToken) {
        // A fast-path pending graph may already be signaled (pre-released); the shared
        // event is monotonic, so only raise it when it is actually still gated.
        if self.gate_event.signaled_value() < stale.event_value {
            self.gate_event.set_signaled_value(stale.event_value);
        }
        stale.cb.wait_until_completed();
    }

    /// Seed `seed_positions` history slots of layer `layer` from contiguous
    /// `[kv_head][seed_positions][head_dim]` (already-roped) K and (raw) V — e.g. the prompt's
    /// K/V copied out of a CPU KV cache after a batched prefill, so resident decode can take
    /// over with the history already on the GPU. Returns false on dimension mismatch.
    pub fn seed_layer(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        seed_positions: usize,
    ) -> bool {
        if layer >= self.n_layers
            || seed_positions > self.max_positions
            || keys.len() != self.n_kv_heads * seed_positions * self.head_dim
            || values.len() != keys.len()
        {
            return false;
        }
        Self::seed_into(
            &self.cache_k[layer],
            keys,
            self.n_kv_heads,
            self.max_positions,
            self.head_dim,
            seed_positions,
            self.kv16,
        );
        Self::seed_into(
            &self.cache_v[layer],
            values,
            self.n_kv_heads,
            self.max_positions,
            self.head_dim,
            seed_positions,
            self.kv16,
        );
        if !self.kv16 {
            Self::seed_into(
                &self.cache_k16[layer],
                keys,
                self.n_kv_heads,
                self.max_positions,
                self.head_dim,
                seed_positions,
                true,
            );
            Self::seed_into(
                &self.cache_v16[layer],
                values,
                self.n_kv_heads,
                self.max_positions,
                self.head_dim,
                seed_positions,
                true,
            );
        }
        true
    }

    /// Read layer `layer`'s first `positions` cached K and V back out as contiguous
    /// `[kv_head][positions][head_dim]` f32 — the exact inverse of [`seed_layer`](Self::seed_layer),
    /// and the Metal twin of the CUDA engine's `read_kv_layer`.
    ///
    /// This exists so a CPU fallback mid-sequence can recover the history the resident engine
    /// produced. Without it the CPU KV cache is simply empty for every GPU-produced position,
    /// and both the fallback step and any later GPU reseed silently run over a zeroed prompt.
    /// Metal buffers are shared storage, so this is a strided memcpy over unified memory — no
    /// device transfer, and it touches no kernel.
    ///
    /// Returns `None` on a dimension mismatch. Caller must ensure prior GPU work has completed.
    pub fn read_kv_layer(&self, layer: usize, positions: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if layer >= self.n_layers || positions > self.max_positions {
            return None;
        }
        Some((
            Self::read_from(
                &self.cache_k[layer],
                self.n_kv_heads,
                self.max_positions,
                self.head_dim,
                positions,
                self.kv16,
            ),
            Self::read_from(
                &self.cache_v[layer],
                self.n_kv_heads,
                self.max_positions,
                self.head_dim,
                positions,
                self.kv16,
            ),
        ))
    }

    /// Gather a `[kv_head][max_positions][head_dim]` cache buffer's first `positions` slots
    /// into contiguous `[kv_head][positions][head_dim]` f32. Inverse of [`Self::seed_into`].
    fn read_from(
        buf: &Buffer,
        n_kv_heads: usize,
        max_positions: usize,
        head_dim: usize,
        positions: usize,
        kv16: bool,
    ) -> Vec<f32> {
        let run = positions * head_dim;
        let mut out = vec![0.0f32; n_kv_heads * run];
        // SAFETY: shared-storage buffer of n_kv_heads*max_positions*head_dim elements (f32 or
        // f16); each head's `run` values are read from a disjoint slot well within that
        // capacity (`positions <= max_positions` checked by the caller).
        unsafe {
            if kv16 {
                let src = buf.contents() as *const u16;
                for h in 0..n_kv_heads {
                    let s = h * max_positions * head_dim;
                    let d = h * run;
                    for i in 0..run {
                        out[d + i] = f16_bits_to_f32(*src.add(s + i));
                    }
                }
            } else {
                let src = buf.contents() as *const f32;
                for h in 0..n_kv_heads {
                    let s = h * max_positions * head_dim;
                    let d = h * run;
                    std::ptr::copy_nonoverlapping(src.add(s), out[d..d + run].as_mut_ptr(), run);
                }
            }
        }
        out
    }

    /// Scatter contiguous `[kv_head][seed_positions][head_dim]` source data into a persistent
    /// `[kv_head][max_positions][head_dim]` cache buffer (the per-head position stride differs).
    fn seed_into(
        buf: &Buffer,
        src: &[f32],
        n_kv_heads: usize,
        max_positions: usize,
        head_dim: usize,
        seed_positions: usize,
        kv16: bool,
    ) {
        let run = seed_positions * head_dim;
        // SAFETY: shared-storage buffer of n_kv_heads*max_positions*head_dim elements (f32 or
        // f16); each head's `run` values land at a disjoint slot well within that capacity.
        unsafe {
            if kv16 {
                let dst = buf.contents() as *mut u16;
                for h in 0..n_kv_heads {
                    let s = h * seed_positions * head_dim;
                    let d = h * max_positions * head_dim;
                    for i in 0..run {
                        *dst.add(d + i) = f32_to_f16_bits(src[s + i]);
                    }
                }
            } else {
                let dst = buf.contents() as *mut f32;
                for h in 0..n_kv_heads {
                    let s = h * seed_positions * head_dim;
                    let d = h * max_positions * head_dim;
                    std::ptr::copy_nonoverlapping(src[s..s + run].as_ptr(), dst.add(d), run);
                }
            }
        }
    }

    /// Read back layer `layer`'s first `position_count` cached K positions as a contiguous
    /// `[kv_head][position_count][head_dim]` buffer (test-only; used to build a reference cache
    /// for the persistent-vs-full-upload parity check).
    #[cfg(test)]
    fn cache_k_contiguous(&self, layer: usize, position_count: usize) -> Vec<f32> {
        self.read_cache(&self.cache_k[layer], position_count)
    }

    #[cfg(test)]
    fn cache_v_contiguous(&self, layer: usize, position_count: usize) -> Vec<f32> {
        self.read_cache(&self.cache_v[layer], position_count)
    }

    #[cfg(test)]
    fn read_cache(&self, buf: &Buffer, position_count: usize) -> Vec<f32> {
        let mut full = vec![0.0f32; self.n_kv_heads * self.max_positions * self.head_dim];
        read_buffer_f32(buf, &mut full);
        let mut out = vec![0.0f32; self.n_kv_heads * position_count * self.head_dim];
        for h in 0..self.n_kv_heads {
            for p in 0..position_count {
                let src = (h * self.max_positions + p) * self.head_dim;
                let dst = (h * position_count + p) * self.head_dim;
                out[dst..dst + self.head_dim].copy_from_slice(&full[src..src + self.head_dim]);
            }
        }
        out
    }
}

#[cfg(target_os = "macos")]
impl Drop for ResidentDecodeState {
    fn drop(&mut self) {
        // A pre-committed pending graph left gated on the queue would block every future
        // commit on the shared serial queue; release it before the session goes away.
        if let Some(stale) = self.pending.take() {
            self.release_stale(stale);
        }
        if let Some(r) = self.retiring.take() {
            r.cb.wait_until_completed();
        }
    }
}

/// Gate for the gemma4 GPU-resident decode path (off by default until the full
/// graph is assembled and end-to-end parity is proven).
#[cfg(target_os = "macos")]
pub fn gemma4_gpu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_GPU")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// GPU-resident decode session state for gemma4 — allocation scaffolding. Holds the
/// per-layer KV cache (each sized to that layer's per-type head_dim, and allocated
/// ONLY for layers that own their K/V; the trailing cross-shared layers read an
/// earlier same-type layer's cache, so their slot is `None`), the ping-pong hidden
/// buffers, and the gate/done events. The per-token forward graph is layered on in a
/// later port step; until then most fields are intentionally unread.
#[cfg(target_os = "macos")]
#[allow(dead_code)] // fields consumed by the forward graph (later port step)
pub struct Gemma4ResidentState {
    plan: Vec<crate::model::Gemma4LayerPlan>,
    n_kv_heads: usize,
    hidden: usize,
    eps: f32,
    /// Positions the KV cache is currently allocated for; grown toward `cap` later.
    max_positions: usize,
    /// Hard ceiling on `max_positions` (the model context length).
    cap: usize,
    /// KV positions currently materialized (seeded history + appended tokens).
    filled: usize,
    /// Per layer: `Some` for owning layers, `None` for the shared (read-source) layers.
    cache_k: Vec<Option<Buffer>>,
    cache_v: Vec<Option<Buffer>>,
    buf_a: Buffer,
    buf_b: Buffer,
    mid: Buffer,
    gate_event: metal::SharedEvent,
    done_event: metal::SharedEvent,
    event_counter: u64,
}

#[cfg(target_os = "macos")]
impl Gemma4ResidentState {
    /// Allocate the resident KV cache + scratch for a gemma4 decode session.
    /// `max_positions` is the initial KV capacity; `cap` is the hard ceiling
    /// (context length). Returns None if Metal is unavailable or shapes are invalid.
    pub fn new(
        plan: Vec<crate::model::Gemma4LayerPlan>,
        n_kv_heads: usize,
        hidden: usize,
        eps: f32,
        max_positions: usize,
        cap: usize,
    ) -> Option<Self> {
        if plan.is_empty()
            || n_kv_heads == 0
            || hidden == 0
            || max_positions == 0
            || cap < max_positions
        {
            return None;
        }
        let k = metal_linear_kernel()?;
        let fbuf = |n: usize| {
            k.device
                .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
        };
        // Owning layers get a cache sized to their per-type head_dim; shared layers
        // hold None and read their `kv_source_layer`'s cache at attention time.
        let kv_buf = |p: &crate::model::Gemma4LayerPlan| {
            p.owns_kv
                .then(|| fbuf(n_kv_heads * max_positions * p.head_dim))
        };
        let cache_k: Vec<Option<Buffer>> = plan.iter().map(kv_buf).collect();
        let cache_v: Vec<Option<Buffer>> = plan.iter().map(kv_buf).collect();
        Some(Self {
            plan,
            n_kv_heads,
            hidden,
            eps,
            max_positions,
            cap,
            filled: 0,
            cache_k,
            cache_v,
            buf_a: fbuf(hidden),
            buf_b: fbuf(hidden),
            mid: fbuf(hidden),
            gate_event: k.device.new_shared_event(),
            done_event: k.device.new_shared_event(),
            event_counter: 0,
        })
    }

    /// Byte length of layer `l`'s K-cache buffer, or None for a shared layer.
    #[cfg(test)]
    fn cache_k_len(&self, l: usize) -> Option<u64> {
        self.cache_k[l].as_ref().map(|b| b.length())
    }
}

#[cfg(not(target_os = "macos"))]
pub fn gemma4_gpu_enabled() -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub struct Gemma4ResidentState;

#[cfg(not(target_os = "macos"))]
impl Gemma4ResidentState {
    pub fn new(
        _plan: Vec<crate::model::Gemma4LayerPlan>,
        _n_kv_heads: usize,
        _hidden: usize,
        _eps: f32,
        _max_positions: usize,
        _cap: usize,
    ) -> Option<Self> {
        None
    }
}

#[cfg(not(target_os = "macos"))]
pub struct Gemma4ResidentLayer;

#[cfg(not(target_os = "macos"))]
impl Gemma4ResidentLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_wire(
        _fmt: GemmaWireFmt,
        _attn_norm: Vec<f32>,
        _q_norm: Vec<f32>,
        _k_norm: Vec<f32>,
        _post_attn_norm: Vec<f32>,
        _ffn_norm: Vec<f32>,
        _post_ffw_norm: Vec<f32>,
        _q_wire: &[u8],
        _k_wire: &[u8],
        _v_wire: Option<&[u8]>,
        _o_wire: &[u8],
        _gate_wire: &[u8],
        _up_wire: &[u8],
        _down_wire: &[u8],
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _ffn_dim: usize,
        _eps: f32,
    ) -> Option<Self> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_wire_pages(
        _fmt: GemmaWireFmt,
        _attn_norm: Vec<f32>,
        _q_norm: Vec<f32>,
        _k_norm: Vec<f32>,
        _post_attn_norm: Vec<f32>,
        _ffn_norm: Vec<f32>,
        _post_ffw_norm: Vec<f32>,
        _q_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _k_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _v_pages: Option<&std::sync::Arc<crate::wire_mmap::WirePages>>,
        _o_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _gate_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _up_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _down_pages: &std::sync::Arc<crate::wire_mmap::WirePages>,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _ffn_dim: usize,
        _eps: f32,
    ) -> Option<Self> {
        None
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_layer(
    _layer: &Gemma4ResidentLayer,
    _h_in: &[f32],
    _cos_t: &[f32],
    _sin_t: &[f32],
    _cache_k_init: &[f32],
    _cache_v_init: &[f32],
    _max_positions: usize,
    _write_position: usize,
    _filled: usize,
    _window_start: usize,
    _scale: f32,
    _owns_kv: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_ple(
    _h_in: &[f32],
    _pli_l: &[f32],
    _ple_inp_gate: &[f32],
    _ple_proj: &[f32],
    _post_norm: &[f32],
    _output_scale: f32,
    _eps: f32,
    _ple_dim: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_pli(
    _h0: &[f32],
    _proj: &[f32],
    _proj_norm: &[f32],
    _ti: &[f32],
    _hidden: usize,
    _ple_dim: usize,
    _n_layers: usize,
    _eps: f32,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_gemma4_head(
    _h_in: &[f32],
    _output_norm: &[f32],
    _token_embd_wire: &[u8],
    _vocab: usize,
    _softcap: f32,
    _eps: f32,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_forward(
    _layers: &[Gemma4ResidentLayer],
    _ple: &[Option<Gemma4ResidentPle>],
    _owns_kv: &[bool],
    _kv_source: &[usize],
    _inputs: &[Gemma4TokenLayerInput],
    _cache_k_init: &[Option<Vec<f32>>],
    _cache_v_init: &[Option<Vec<f32>>],
    _h0: &[f32],
    _output_norm: &[f32],
    _token_embd_wire: &[u8],
    _vocab: usize,
    _softcap: f32,
    _eps: f32,
    _max_positions: usize,
    _write_position: usize,
    _filled: usize,
    _scale: f32,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub struct Gemma4ResidentModel;

#[cfg(not(target_os = "macos"))]
impl Gemma4ResidentModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _layers: Vec<Gemma4ResidentLayer>,
        _ple: Vec<Option<Gemma4ResidentPle>>,
        _layer_scales: Vec<f32>,
        _owns_kv: Vec<bool>,
        _kv_source: Vec<usize>,
        _token_embd_wire: &[u8],
        _output_norm: Vec<f32>,
        _hidden: usize,
        _vocab: usize,
        _softcap: f32,
        _eps: f32,
        _max_positions: usize,
        _scale: f32,
    ) -> Option<Self> {
        None
    }

    pub fn set_pli(&mut self, _proj: &[f32], _proj_norm: &[f32], _ple_dim: usize) -> bool {
        false
    }

    pub fn forward_token(
        &self,
        _h0: &[f32],
        _inputs: &[Gemma4TokenLayerInput],
        _ti: &[f32],
        _position: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    pub fn forward_token_hidden(
        &self,
        _h0: &[f32],
        _inputs: &[Gemma4TokenLayerInput],
        _ti: &[f32],
        _position: usize,
    ) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(not(target_os = "macos"))]
pub struct ResidentDecodeState;

#[cfg(not(target_os = "macos"))]
impl ResidentDecodeState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _n_layers: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _hidden: usize,
        _ffn_dim: usize,
        _max_positions: usize,
        _cap: usize,
        _eps: f32,
        _split_half_pairing: bool,
    ) -> Option<Self> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &mut self,
        _embedding: &[f32],
        _layers: &[ResidentLayerWeights],
        _cos_t: &[f32],
        _sin_t: &[f32],
        _position: usize,
        _scale: f32,
        _logits_stage: Option<LogitsStage>,
        _sample_stage: Option<SampleStage>,
        _input_token_id: u32,
        _next_rope: Option<(&[f32], &[f32])>,
    ) -> Option<ResidentTokenOut> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_tokens(
        &mut self,
        _embeddings: &[f32],
        _n_tokens: usize,
        _layers: &[ResidentLayerWeights],
        _cos_all: &[f32],
        _sin_all: &[f32],
        _scale: f32,
    ) -> Option<()> {
        None
    }

    pub fn seed_layer(
        &mut self,
        _layer: usize,
        _keys: &[f32],
        _values: &[f32],
        _seed_positions: usize,
    ) -> bool {
        false
    }

    pub fn read_kv_layer(&self, _layer: usize, _positions: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    pub fn filled(&self) -> usize {
        0
    }

    pub fn set_filled(&mut self, _n: usize) {}

    pub fn rollback_to_position(&mut self, _position: usize) {}
}

#[cfg(not(target_os = "macos"))]
pub fn try_quantized_matmul_resident(
    _input: &[f32],
    _weight_blocks: &[u8],
    _output_width: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_ffn_block_resident(
    _input: &[f32],
    _ffn_norm: &[f32],
    _eps: f32,
    _gate_weight_blocks: &[u8],
    _up_weight_blocks: &[u8],
    _down_weight_blocks: &[u8],
    _ffn_dim: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_block_resident(
    _input: &[f32],
    _attn_norm: &[f32],
    _eps: f32,
    _q_weight_blocks: &[u8],
    _k_weight_blocks: &[u8],
    _v_weight_blocks: &[u8],
    _o_weight_blocks: &[u8],
    _cos_t: &[f32],
    _sin_t: &[f32],
    _cache_k: &[f32],
    _cache_v: &[f32],
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _position_count: usize,
    _scale: f32,
    _split_half_pairing: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_decode_layer_resident(
    _input: &[f32],
    _attn_norm: &[f32],
    _ffn_norm: &[f32],
    _eps: f32,
    _q_weight_blocks: &[u8],
    _k_weight_blocks: &[u8],
    _v_weight_blocks: &[u8],
    _o_weight_blocks: &[u8],
    _gate_weight_blocks: &[u8],
    _up_weight_blocks: &[u8],
    _down_weight_blocks: &[u8],
    _cos_t: &[f32],
    _sin_t: &[f32],
    _cache_k: &[f32],
    _cache_v: &[f32],
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _position_count: usize,
    _ffn_dim: usize,
    _scale: f32,
    _split_half_pairing: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_decode_forward_resident(
    _embedding: &[f32],
    _layers: &[ResidentDecodeLayer],
    _cos_t: &[f32],
    _sin_t: &[f32],
    _eps: f32,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _position_count: usize,
    _ffn_dim: usize,
    _scale: f32,
    _split_half_pairing: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_quantize_q8_0_f32(_input: &[f32]) -> Option<(Vec<f32>, Vec<i8>)> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_rms_norm_f32(_input: &[f32], _weight: &[f32], _eps: f32) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_rms_norm_per_head_f32(
    _input: &[f32],
    _weight: Option<&[f32]>,
    _head_count: usize,
    _head_dim: usize,
    _eps: f32,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_gemma4_q8_matmul_f32y(
    _y: &[f32],
    _weight_wire: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_gemma4_q4_0_matmul_f32y(
    _y: &[f32],
    _weight_wire: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_ffn(
    _fmt: GemmaWireFmt,
    _h_in: &[f32],
    _ffn_norm: &[f32],
    _post_ffw_norm: &[f32],
    _eps: f32,
    _gate_wire: &[u8],
    _up_wire: &[u8],
    _down_wire: &[u8],
    _ffn_dim: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_gemma4_attention(
    _fmt: GemmaWireFmt,
    _h_in: &[f32],
    _attn_norm: &[f32],
    _q_norm: &[f32],
    _k_norm: &[f32],
    _post_attn_norm: &[f32],
    _eps: f32,
    _q_wire: &[u8],
    _k_wire: &[u8],
    _v_wire: Option<&[u8]>,
    _o_wire: &[u8],
    _cos_t: &[f32],
    _sin_t: &[f32],
    _cache_k_init: &[f32],
    _cache_v_init: &[f32],
    _max_positions: usize,
    _write_position: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _filled: usize,
    _window_start: usize,
    _scale: f32,
    _owns_kv: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_decode_f32(
    _query: &[f32],
    _keys: &[f32],
    _values: &[f32],
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _position_count: usize,
    _scale: f32,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_attention_decode_strided_f32(
    _query: &[f32],
    _keys: &[f32],
    _values: &[f32],
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _position_count: usize,
    _scale: f32,
    _position_stride: usize,
    _kv_head_stride: usize,
    _kv_base_offset: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_rope_rotate_f32(
    _data: &[f32],
    _cos_table: &[f32],
    _sin_table: &[f32],
    _head_count: usize,
    _head_dim: usize,
    _half_rope: usize,
    _split_half_pairing: bool,
) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_residual_add_f32(_a: &[f32], _b: &[f32]) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_silu_mul_f32(_gate: &[f32], _up: &[f32]) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_gelu_mul_f32(_gate: &[f32], _up: &[f32]) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn try_soft_cap_f32(_input: &[f32], _cap: f32) -> Option<Vec<f32>> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn start_inference_session() {}

#[cfg(not(target_os = "macos"))]
pub fn end_inference_session() {}

#[cfg(not(target_os = "macos"))]
pub fn synchronize_active_session() {}

#[cfg(not(target_os = "macos"))]
pub fn try_linear_row_f32(
    _input_row: &[f32],
    _weights: &[f32],
    _rows: usize,
    _cols: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_linear_row_transposed_f32(
    _input_row: &[f32],
    _weights: &[f32],
    _rows: usize,
    _cols: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_q8_0_encoded_linear_row(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _encoded_rows: &[u8],
    _weight_scales: &[f32],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_q8_0_encoded_linear_rows(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _encoded_rows: &[u8],
    _weight_scales: &[f32],
    _input_rows: usize,
    _weight_rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_q8_0_block_linear_row(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _weight_blocks: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_q8_0_block_linear_row_with_cpu<F>(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _weight_blocks: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
    _cpu_work: F,
) -> bool
where
    F: FnOnce(),
{
    false
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn try_q8_0_block_two_linear_rows_with_cpu<F>(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _first_weight_blocks: &[u8],
    _second_weight_blocks: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _first_output: &mut [f32],
    _second_output: &mut [f32],
    _cpu_work: F,
) -> bool
where
    F: FnOnce(),
{
    false
}

#[cfg(target_os = "macos")]
pub fn detect_metal_device() -> MetalDeviceInfo {
    match Device::system_default() {
        Some(device) => {
            let threadgroup = device.max_threads_per_threadgroup();
            MetalDeviceInfo {
                available: true,
                device_name: Some(device.name().to_string()),
                low_power: Some(device.is_low_power()),
                headless: Some(device.is_headless()),
                removable: Some(device.is_removable()),
                has_unified_memory: Some(device.has_unified_memory()),
                registry_id: Some(device.registry_id()),
                max_threads_per_threadgroup: Some((
                    threadgroup.width,
                    threadgroup.height,
                    threadgroup.depth,
                )),
                note: Some(
                    "Metal device detected. Camelid has an opt-in experimental dense linear-row kernel path on macOS; broader inference offload is still in progress.".to_string(),
                ),
            }
        }
        None => MetalDeviceInfo {
            available: false,
            device_name: None,
            low_power: None,
            headless: None,
            removable: None,
            has_unified_memory: None,
            registry_id: None,
            max_threads_per_threadgroup: None,
            note: Some("No Metal system device was reported by macOS.".to_string()),
        },
    }
}

#[cfg(not(target_os = "macos"))]
pub fn detect_metal_device() -> MetalDeviceInfo {
    MetalDeviceInfo {
        available: false,
        device_name: None,
        low_power: None,
        headless: None,
        removable: None,
        has_unified_memory: None,
        registry_id: None,
        max_threads_per_threadgroup: None,
        note: Some("Metal is only available on macOS builds.".to_string()),
    }
}

#[cfg(test)]
mod tests {

    /// Capability gate for the whole macOS Metal test lane.
    ///
    /// 56 device-gated tests in this file open with `if !detect_metal_device().available {
    /// return; }`, and `metal_resident_rollback_moves_filled_with_the_kv_position`
    /// (`src/inference/tests.rs`) — the only guard for the draft-rollback regression that needs
    /// no model file — skips the same way on `ResidentDecodeState::new(..) -> None`. If the host
    /// reports no device, or reports one whose driver cannot build the resident pipelines, every
    /// one of them prints `ok` while proving nothing. CI has never measured which case it is on
    /// `macos-latest`; the workflow comment merely *asserts* the tests self-skip.
    ///
    /// This test always PRINTS what the host can do (visible with `--nocapture`; CI runs it as a
    /// dedicated step) and, under `CAMELID_REQUIRE_METAL_TESTS=1`, turns "no lane" into a hard
    /// failure. Contributors never set that variable, so a Mac without a usable GPU session — or
    /// a non-macOS host, where this test does not exist — still passes locally.
    ///
    /// Two distinct capabilities are reported, because they fail independently:
    ///   `available`         — a Metal device exists.
    ///   `resident_kernels`  — `ResidentDecodeState::new` built its shaders and pipelines, i.e.
    ///                         the GPU-resident decode lane can actually run here.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_lane_capability_probe() {
        let info = super::detect_metal_device();
        // Same shape as `tiny_kv_budget_session` (1 layer / 1 head / 1 kv head, dims 32,
        // kv budget 64), so this reaches the kernel build with every dimension guard
        // satisfied: `Some(..)` proves device + shader libraries + compute pipelines.
        let resident = super::ResidentDecodeState::new(1, 1, 1, 32, 32, 32, 16, 64, 1.0e-5, false);
        let resident_kernels = resident.is_some();
        eprintln!(
            "CAMELID_METAL_CAPABILITY available={} resident_kernels={} device={:?} \
             headless={:?} unified={:?} low_power={:?} note={:?}",
            info.available,
            resident_kernels,
            info.device_name,
            info.headless,
            info.has_unified_memory,
            info.low_power,
            info.note
        );

        if std::env::var("CAMELID_REQUIRE_METAL_TESTS").as_deref() != Ok("1") {
            return;
        }
        assert!(
            info.available,
            "CAMELID_REQUIRE_METAL_TESTS=1 but no Metal device is reported — every \
             device-gated Metal test on this host is a silent no-op"
        );
        assert!(
            resident_kernels,
            "CAMELID_REQUIRE_METAL_TESTS=1 and a Metal device is present ({:?}), but \
             ResidentDecodeState::new returned None — the shaders/pipelines could not be \
             built here, so the GPU-resident lane and its regression tests cannot run",
            info.device_name
        );
    }

    /// End-to-end proof for the instant-start lane: Metal can wrap a page-aligned
    /// window of an mmap'd GGUF with newBufferWithBytesNoCopy and the GPU reads the
    /// file's own bytes at per-tensor offsets — no read loop, no conversion, no
    /// upload copy. Skips without a Metal device or a local model file.
    #[cfg(target_os = "macos")]
    #[test]
    fn wire_mmap_nocopy_buffer_gpu_reads_file_bytes() {
        use std::os::unix::fs::FileExt;

        if !super::detect_metal_device().available {
            return;
        }
        let model_path = std::env::var("CAMELID_TEST_GGUF").unwrap_or_else(|_| {
            "/Volumes/Untitled/models/Llama-3.2-3B-Instruct-Q8_0.gguf".to_string()
        });
        let model_path = std::path::Path::new(&model_path);
        if !model_path.exists() {
            return; // CI runners carry no model files.
        }

        let gguf = crate::gguf::read_metadata(model_path).unwrap();
        // Probe one early and one late Q8_0 tensor so both ends of the file are
        // exercised through the window math.
        let q8_tensors: Vec<_> = gguf
            .tensors
            .iter()
            .filter(|t| t.tensor_type == crate::gguf::GgufTensorType::Q8_0)
            .collect();
        assert!(q8_tensors.len() >= 2, "model should carry Q8_0 tensors");
        let probes = [q8_tensors[0], q8_tensors[q8_tensors.len() - 1]];

        let mapping = crate::wire_mmap::GgufWireMmap::map(model_path).unwrap();
        let ranges: Vec<(u64, usize)> = probes
            .iter()
            .map(|t| (t.absolute_offset, t.n_bytes as usize))
            .collect();
        let device = super::Device::system_default().unwrap();
        let max_window = (device.max_buffer_length() as usize).min(1 << 30);
        let plan = crate::wire_mmap::plan_wire_windows(&mapping, &ranges, max_window).unwrap();
        let (windows, placements) = (plan.windows, plan.placements);

        let queue = device.new_command_queue();
        for (probe_index, tensor) in probes.iter().enumerate() {
            let (window_index, in_window) = placements[probe_index];
            let window = windows[window_index];
            // SAFETY: window offsets are page-aligned multiples within the mapping,
            // so base + aligned_offset stays page-aligned and in bounds.
            let window_ptr = unsafe { mapping.base_ptr().add(window.aligned_offset as usize) };
            assert_eq!(
                window_ptr as usize % crate::wire_mmap::page_size(),
                0,
                "NoCopy pointer must be page-aligned"
            );
            let nocopy = device.new_buffer_with_bytes_no_copy(
                window_ptr as *const std::ffi::c_void,
                window.len as u64,
                super::MTLResourceOptions::StorageModeShared,
                None,
            );

            // GPU-copy the head and tail of the tensor out of the NoCopy buffer.
            let head_len = (tensor.n_bytes as usize).min(256 * 1024);
            let tail_len = (tensor.n_bytes as usize).min(4096);
            let tail_offset = tensor.n_bytes as usize - tail_len;
            let dst = device.new_buffer(
                (head_len + tail_len) as u64,
                super::MTLResourceOptions::StorageModeShared,
            );
            let cb = queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(&nocopy, in_window as u64, &dst, 0, head_len as u64);
            blit.copy_from_buffer(
                &nocopy,
                (in_window + tail_offset) as u64,
                &dst,
                head_len as u64,
                tail_len as u64,
            );
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Reference bytes straight from the file.
            let file = std::fs::File::open(model_path).unwrap();
            let mut expected_head = vec![0u8; head_len];
            file.read_exact_at(&mut expected_head, tensor.absolute_offset)
                .unwrap();
            let mut expected_tail = vec![0u8; tail_len];
            file.read_exact_at(
                &mut expected_tail,
                tensor.absolute_offset + tail_offset as u64,
            )
            .unwrap();

            // SAFETY: dst is StorageModeShared and the command buffer completed.
            let gpu = unsafe {
                std::slice::from_raw_parts(dst.contents() as *const u8, head_len + tail_len)
            };
            assert_eq!(&gpu[..head_len], &expected_head[..], "{}", tensor.name);
            assert_eq!(&gpu[head_len..], &expected_tail[..], "{}", tensor.name);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_ksplit_gemv_matches_cpu_reference() {
        if !detect_metal_device().available {
            return;
        }
        // 5 rows x 7 blocks: odd row count exercises the two-rows-per-threadgroup tail
        // guard, and 7 blocks exercises the K-split stride (4 simdgroups x 8 block slots
        // with most slots idle on the last pass).
        let rows = 5usize;
        let blocks_per_row = 7usize;
        let mut input_scales = Vec::new();
        let mut input_quants: Vec<i8> = Vec::new();
        for block in 0..blocks_per_row {
            input_scales.push(0.05 + block as f32 * 0.01);
            for lane in 0..32 {
                input_quants.push((((block * 37 + lane * 11) % 251) as i32 - 125) as i8);
            }
        }
        let mut weight_blocks = Vec::new();
        let mut weight_quants: Vec<Vec<i8>> = Vec::new();
        for row in 0..rows {
            let mut row_quants = Vec::new();
            for block in 0..blocks_per_row {
                let scale = 0.5 - row as f32 * 0.07 + block as f32 * 0.013;
                weight_blocks.extend_from_slice(&scale.to_le_bytes());
                for lane in 0..32 {
                    let q = (((row * 53 + block * 29 + lane * 7) % 255) as i32 - 127) as i8;
                    weight_blocks.push(q as u8);
                    row_quants.push(q);
                }
            }
            weight_quants.push(row_quants);
        }
        let mut expected = vec![0.0f32; rows];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let scale_bytes: [u8; 4] = weight_blocks
                    [(row * blocks_per_row + block) * 36..(row * blocks_per_row + block) * 36 + 4]
                    .try_into()
                    .unwrap();
                let w_scale = f32::from_le_bytes(scale_bytes);
                let isum: i32 = (0..32)
                    .map(|lane| {
                        i32::from(weight_quants[row][block * 32 + lane])
                            * i32::from(input_quants[block * 32 + lane])
                    })
                    .sum();
                expected[row] += isum as f32 * w_scale * input_scales[block];
            }
        }
        let mut output = vec![0.0f32; rows];
        assert!(try_q8_0_ksplit_linear_for_test(
            &input_scales,
            &input_quants,
            &weight_blocks,
            rows,
            blocks_per_row,
            &mut output,
        ));
        for (row, (actual, expected)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-2_f32.max(expected.abs() * 1.0e-5),
                "row {row}: {actual} != {expected}"
            );
        }
    }

    /// WIN2METAL Phase 3 C0 gate. The speculative-verify lane will run the production
    /// single-token GEMV (`q8_0_block_linear_row_ksplit_f32y_wire_nsg8`) as a batched
    /// column sweep over k verify rows. This proves the batched mirror
    /// (`encode_q8_matmul_f32y_batched`) is BIT-IDENTICAL — exact u32 bit-cast, not
    /// epsilon — to k independent single-token dispatches. Bit-exactness is the whole
    /// point: it gates the byte-exact-vs-`forward_token` claim for the lane.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_verify_gemv_batched_bit_identical() {
        if !detect_metal_device().available {
            return;
        }
        // The reference path is `encode_q8_matmul_f32y`, which only selects the NSG=8
        // wire kernel when the f32y/wire/nsg8 gates are on. Those are process-wide
        // OnceLocks; set the env BEFORE the first gate read so they latch on. If a
        // sibling test in this process already latched them off (e.g. a full --lib
        // run that read them earlier), SKIP rather than compare against a different
        // kernel. A targeted run (`--lib metal_verify_gemv_batched_bit_identical`) has
        // no such sibling, so the gates latch on here.
        std::env::set_var("CAMELID_METAL_F32Y", "1");
        std::env::set_var("CAMELID_METAL_WIRE", "1");
        std::env::set_var("CAMELID_METAL_WIRE_NSG8", "1");
        if !(f32y_gemv_enabled() && wire_weights_enabled() && wire_nsg8_enabled()) {
            eprintln!(
                "SKIP metal_verify_gemv_batched_bit_identical: f32y/wire/nsg8 gates \
                 inactive (OnceLock latched off earlier in this process); run targeted: \
                 cargo test --release --lib metal_verify_gemv_batched_bit_identical"
            );
            return;
        }
        let kernel = metal_linear_kernel().expect("metal kernel available");
        let device = &kernel.device;

        // 130 rows (prime: exercises the NR0=2 row tiling tail and the r0+row>=rows
        // guard) x 7 blocks (K-split stride NSG*NQ=64, so most block slots idle on the
        // last pass). cols = the projection input dimension.
        let rows = 130usize;
        let blocks_per_row = 7usize;
        let cols = blocks_per_row * 32;
        const WIRE: usize = 34; // f16 scale (2B) + 32 i8 quants

        // Deterministic wire-format Q8_0 weights (both paths read these exact bytes).
        let mut weight_wire = vec![0u8; rows * blocks_per_row * WIRE];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let off = (row * blocks_per_row + block) * WIRE;
                let scale = 0.5 - row as f32 * 0.003 + block as f32 * 0.013;
                weight_wire[off..off + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
                for lane in 0..32 {
                    let q = (((row * 53 + block * 29 + lane * 7) % 255) as i32 - 127) as i8;
                    weight_wire[off + 2 + lane] = q as u8;
                }
            }
        }
        let w_buf = device.new_buffer(
            weight_wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        write_buffer_u8(&w_buf, &weight_wire);

        // Deterministic activation columns (one per verify row), non-trivial + signed.
        let max_k = 8usize;
        let acts: Vec<Vec<f32>> = (0..max_k)
            .map(|t| {
                (0..cols)
                    .map(|i| {
                        ((((t * 131 + i * 17) % 251) as f32) - 125.0) * 0.013 + t as f32 * 0.0007
                    })
                    .collect()
            })
            .collect();

        for k in [1usize, 2, 7, 8] {
            // Reference: k single-token NSG=8 dispatches, one activation each.
            let mut reference: Vec<Vec<f32>> = Vec::with_capacity(k);
            for act in acts.iter().take(k) {
                let y_buf =
                    device.new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
                write_buffer_f32(&y_buf, act);
                let out_buf =
                    device.new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
                let scalar = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
                unsafe {
                    let p = scalar.contents() as *mut u32;
                    *p = blocks_per_row as u32;
                    *p.add(1) = rows as u32;
                }
                let cb = kernel.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                encode_q8_matmul_f32y(e, kernel, &y_buf, &w_buf, &out_buf, &scalar, rows);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let mut out = vec![0.0f32; rows];
                read_buffer_f32(&out_buf, &mut out);
                reference.push(out);
            }

            // Candidate: one batched dispatch over all k columns. y token-major
            // [k][cols]; out token-major [k][rows].
            let mut y_flat = vec![0.0f32; k * cols];
            for (t, act) in acts.iter().take(k).enumerate() {
                y_flat[t * cols..(t + 1) * cols].copy_from_slice(act);
            }
            let y_buf =
                device.new_buffer((k * cols * 4) as u64, MTLResourceOptions::StorageModeShared);
            write_buffer_f32(&y_buf, &y_flat);
            let out_buf =
                device.new_buffer((k * rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            // 12-byte scalar: blocks_per_row @0, rows @4 (n_rows_in @8 set by helper).
            let scalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
            unsafe {
                let p = scalar.contents() as *mut u32;
                *p = blocks_per_row as u32;
                *p.add(1) = rows as u32;
            }
            let cb = kernel.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            encode_q8_matmul_f32y_batched(e, kernel, &y_buf, &w_buf, &out_buf, &scalar, rows, k);
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let mut batched = vec![0.0f32; k * rows];
            read_buffer_f32(&out_buf, &mut batched);

            // Exact u32 bit identity for every column, every row.
            for (t, refrow) in reference.iter().enumerate().take(k) {
                for (r, &b) in refrow.iter().enumerate() {
                    let a = batched[t * rows + r];
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "k={k} column {t} row {r}: batched {a} ({:#010x}) != \
                         single-token {b} ({:#010x})",
                        a.to_bits(),
                        b.to_bits(),
                    );
                }
            }
            eprintln!(
                "metal_verify_gemv_batched_bit_identical: k={k} BIT-IDENTICAL ({rows} rows x {blocks_per_row} blocks)"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_decode_v2_matches_cpu_reference() {
        if !detect_metal_device().available {
            return;
        }
        // GQA (4 query heads over 2 KV heads), head_dim 32, 7 positions (prime, so all four
        // position-striding simdgroups see uneven work).
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 32usize;
        let positions = 7usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let query: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
            .collect();
        let keys: Vec<f32> = (0..n_kv_heads * positions * head_dim)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
            .collect();
        let values: Vec<f32> = (0..n_kv_heads * positions * head_dim)
            .map(|i| ((i as f32 % 4.0) - 1.0) * 0.5)
            .collect();
        let group = n_heads / n_kv_heads;
        let mut expected = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let qb = h * head_dim;
            let kvb = (h / group) * positions * head_dim;
            let mut scores = vec![0.0f32; positions];
            for (p, score) in scores.iter_mut().enumerate() {
                let kb = kvb + p * head_dim;
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += query[qb + d] * keys[kb + d];
                }
                *score = s * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut acc = 0.0;
                for (p, s) in scores.iter().enumerate() {
                    acc += (s / sum) * values[kvb + p * head_dim + d];
                }
                expected[qb + d] = acc;
            }
        }
        let got = try_attention_v2_for_test(
            &query, &keys, &values, n_heads, n_kv_heads, head_dim, positions, scale,
        )
        .expect("v2 attention");
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1.0e-4, "dim {i}: {a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_decode_splitk_kv16_matches_cpu_reference() {
        if !detect_metal_device().available {
            return;
        }
        // Production-shaped GQA (group of 3, so the fourth simdgroup is inactive) at
        // head_dim 128 (full half4 staging width). 131 positions -> 3 splits of chunk
        // 44 where the last covers 43: a 16/16/11 tile sequence whose tail round
        // exercises the -INF score padding and the guarded v_s reads.
        let n_heads = 6usize;
        let n_kv_heads = 2usize;
        let head_dim = 128usize;
        let positions = 131usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Steps of 0.25 are exactly representable in f16, so the f32 CPU reference
        // sees the same values the kernel reads back from the half mirrors.
        let query: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.25)
            .collect();
        let keys: Vec<f32> = (0..n_kv_heads * positions * head_dim)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.25)
            .collect();
        let values: Vec<f32> = (0..n_kv_heads * positions * head_dim)
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.25)
            .collect();
        let group = n_heads / n_kv_heads;
        let mut expected = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let qb = h * head_dim;
            let kvb = (h / group) * positions * head_dim;
            let mut scores = vec![0.0f32; positions];
            for (p, score) in scores.iter_mut().enumerate() {
                let kb = kvb + p * head_dim;
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += query[qb + d] * keys[kb + d];
                }
                *score = s * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut acc = 0.0;
                for (p, s) in scores.iter().enumerate() {
                    acc += (s / sum) * values[kvb + p * head_dim + d];
                }
                expected[qb + d] = acc;
            }
        }
        for direct in [false, true] {
            let got = try_attention_splitk_kv16_for_test(
                &query, &keys, &values, n_heads, n_kv_heads, head_dim, positions, scale, direct,
            )
            .expect("splitk kv16 attention");
            for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() < 1.0e-4,
                    "direct={direct} dim {i}: {a} != {b}"
                );
            }
        }
    }

    /// Probe: decode-attention kernel rate at depth, production-shaped (Llama-3.2-3B:
    /// 24q/8kv heads, head_dim 128, 28 layers' dispatches in one encoder, distinct
    /// per-layer KV buffers so the cache can't flatter the rate, [kv_head][position]
    /// [head_dim] strides as the resident session passes them).
    /// Run: cargo test --release --lib attention_decode_depth_probe -- --ignored --nocapture
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn attention_decode_depth_probe() {
        if !detect_metal_device().available {
            return;
        }
        let kernel = metal_linear_kernel().expect("metal kernel");
        let device = &kernel.device;
        let opts = MTLResourceOptions::StorageModeShared;
        let n_heads = 24usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let layers = 28usize;
        let group = (n_heads / n_kv_heads) as u32;
        for &positions in &[480usize, 2048, 7700] {
            let kv_slots = n_kv_heads * positions * head_dim;
            let zeroed = |bytes: usize| {
                let b = device.new_buffer(bytes as u64, opts);
                unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes) };
                b
            };
            let k16: Vec<Buffer> = (0..layers).map(|_| zeroed(kv_slots * 2)).collect();
            let v16: Vec<Buffer> = (0..layers).map(|_| zeroed(kv_slots * 2)).collect();
            let k32: Vec<Buffer> = (0..layers).map(|_| zeroed(kv_slots * 4)).collect();
            let v32: Vec<Buffer> = (0..layers).map(|_| zeroed(kv_slots * 4)).collect();
            let q = zeroed(n_heads * head_dim * 4);
            let out = zeroed(n_heads * head_dim * 4);
            let scalar = device.new_buffer(32, opts);
            unsafe {
                let p = scalar.contents() as *mut u32;
                *p = n_heads as u32;
                *p.add(1) = head_dim as u32;
                *p.add(2) = positions as u32;
                *p.add(3) = group;
                *(p.add(4) as *mut f32) = 1.0 / (head_dim as f32).sqrt();
                *p.add(5) = head_dim as u32; // position_stride
                *p.add(6) = (positions * head_dim) as u32; // kv_head_stride
                *p.add(7) = 0; // kv_base_offset
            }
            for &n_splits in &[32usize, 64, 128, 256] {
                let splits_scalar = device.new_buffer(4, opts);
                unsafe { *(splits_scalar.contents() as *mut u32) = n_splits as u32 };
                let partials: Vec<Buffer> = (0..layers)
                    .map(|_| {
                        device.new_buffer((n_heads * n_splits * (head_dim + 2) * 4) as u64, opts)
                    })
                    .collect();
                #[allow(clippy::type_complexity)]
                let variants: [(
                    &str,
                    &ComputePipelineState,
                    &[Buffer],
                    &[Buffer],
                    usize,
                ); 4] = [
                    (
                        "kv16 ",
                        &kernel.attention_decode_splitk_kv16_pipeline,
                        &k16,
                        &v16,
                        2,
                    ),
                    (
                        "direc",
                        &kernel.attention_decode_splitk_kv16_direct_pipeline,
                        &k16,
                        &v16,
                        2,
                    ),
                    (
                        "f32  ",
                        &kernel.attention_decode_splitk_pipeline,
                        &k32,
                        &v32,
                        4,
                    ),
                    (
                        "stage",
                        &kernel.attention_splitk_kv16_stageonly_pipeline,
                        &k16,
                        &v16,
                        2,
                    ),
                ];
                for (label, pipeline, keys, values, elem) in variants {
                    let bytes = layers * 2 * kv_slots * elem;
                    for round in 0..2 {
                        let cb = kernel.queue.new_command_buffer();
                        let e = cb.new_compute_command_encoder();
                        for l in 0..layers {
                            e.set_compute_pipeline_state(pipeline);
                            e.set_buffer(0, Some(&q), 0);
                            e.set_buffer(1, Some(&keys[l]), 0);
                            e.set_buffer(2, Some(&values[l]), 0);
                            e.set_buffer(3, Some(&partials[l]), 0);
                            for i in 0..8u64 {
                                e.set_buffer(5 + i, Some(&scalar), i * 4);
                            }
                            e.set_buffer(13, Some(&splits_scalar), 0);
                            e.dispatch_thread_groups(
                                metal::MTLSize {
                                    width: n_kv_heads as u64,
                                    height: n_splits as u64,
                                    depth: 1,
                                },
                                metal::MTLSize {
                                    width: 128,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            e.set_compute_pipeline_state(
                                &kernel.attention_decode_splitk_merge_pipeline,
                            );
                            e.set_buffer(0, Some(&partials[l]), 0);
                            e.set_buffer(1, Some(&out), 0);
                            e.set_buffer(2, Some(&scalar), 4);
                            e.set_buffer(3, Some(&splits_scalar), 0);
                            e.dispatch_thread_groups(
                                metal::MTLSize {
                                    width: n_heads as u64,
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
                        e.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                        let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
                        println!(
                        "attn probe {label} pos={positions} splits={n_splits}: round {round}: {:.2} ms/token, {:.1} GB/s",
                        busy_us as f64 / 1000.0,
                        bytes as f64 / (busy_us as f64 * 1e-6) / 1e9
                    );
                    }
                }
            }
        }
    }
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_rms_norm_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let width = 320usize;
        let input: Vec<f32> = (0..width)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.3)
            .collect();
        let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i as f32 % 7.0) * 0.1).collect();
        let eps = 1.0e-5f32;
        let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let expected: Vec<f32> = input
            .iter()
            .zip(&weight)
            .map(|(x, w)| x * inv * w)
            .collect();
        let got = try_rms_norm_f32(&input, &weight, eps).expect("metal rms_norm");
        assert_eq!(got.len(), width);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_silu_mul_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let n = 257usize; // non-multiple of the execution width
        let gate: Vec<f32> = (0..n).map(|i| ((i as f32 % 13.0) - 6.0) * 0.4).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i as f32 % 5.0) - 2.0) * 0.5).collect();
        let expected: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let got = try_silu_mul_f32(&gate, &up).expect("metal silu_mul");
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gelu_mul_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let n = 257usize; // non-multiple of the execution width
                          // Include large-magnitude gate values (±300): the x^3 term drives tanh's
                          // argument to ~1e6, where an unclamped MSL tanh overflows to NaN. The kernel
                          // clamps the tanh arg; this must still match the saturating CPU gelu.
        let gate: Vec<f32> = (0..n)
            .map(|i| ((i as f32 % 13.0) - 6.0) * if i % 7 == 0 { 50.0 } else { 0.4 })
            .collect();
        let up: Vec<f32> = (0..n).map(|i| ((i as f32 % 5.0) - 2.0) * 0.5).collect();
        let mut expected = vec![0.0f32; n];
        crate::inference::gemma4::geglu_into(&gate, &up, &mut expected);
        let got = try_gelu_mul_f32(&gate, &up).expect("metal gelu_mul");
        for (a, b) in got.iter().zip(&expected) {
            assert!(a.is_finite(), "gpu gelu produced non-finite {a}");
            assert!((a - b).abs() < 1.0e-2, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_soft_cap_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let cap = 30.0f32;
        let n = 263usize;
        // Include |logit| up to ~2000 (v/cap ~ 66): exp(2*v/cap) overflows f32, so an
        // unclamped tanh would NaN. The kernel clamps; result must still saturate to ±cap.
        let input: Vec<f32> = (0..n).map(|i| ((i as f32 % 91.0) - 45.0) * 45.0).collect();
        let mut expected = input.clone();
        crate::inference::gemma4::soft_cap_in_place(&mut expected, cap);
        let got = try_soft_cap_f32(&input, cap).expect("metal soft_cap");
        for (a, b) in got.iter().zip(&expected) {
            assert!(a.is_finite(), "gpu soft_cap produced non-finite {a}");
            assert!((a - b).abs() < 1.0e-2, "{a} != {b}");
        }
        // Disabled cap is a passthrough.
        let passthrough = try_soft_cap_f32(&input, 0.0).expect("metal soft_cap passthrough");
        assert_eq!(passthrough, input);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_rms_norm_per_head_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let eps = 1.0e-6f32;
        // Reference: standard RMSNorm applied independently per head_dim chunk.
        fn cpu_per_head(
            input: &[f32],
            weight: Option<&[f32]>,
            head_count: usize,
            head_dim: usize,
            eps: f32,
        ) -> Vec<f32> {
            let mut out = vec![0.0f32; input.len()];
            for h in 0..head_count {
                let base = h * head_dim;
                let mss: f32 = input[base..base + head_dim]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    / head_dim as f32;
                let inv = (mss + eps).powf(-0.5);
                for d in 0..head_dim {
                    let mut v = input[base + d] * inv;
                    if let Some(w) = weight {
                        v *= w[d];
                    }
                    out[base + d] = v;
                }
            }
            out
        }
        // Exercise both gemma layer-type shapes: sliding (8x256) and global (8x512).
        for (head_count, head_dim) in [(8usize, 256usize), (8, 512)] {
            let n = head_count * head_dim;
            let input: Vec<f32> = (0..n).map(|i| ((i as f32 % 17.0) - 8.0) * 0.3).collect();
            let weight: Vec<f32> = (0..head_dim)
                .map(|d| 0.5 + (d as f32 % 7.0) * 0.1)
                .collect();
            // Weighted (QK-norm).
            let want_w = cpu_per_head(&input, Some(&weight), head_count, head_dim, eps);
            let got_w = try_rms_norm_per_head_f32(&input, Some(&weight), head_count, head_dim, eps)
                .expect("metal per-head rms_norm (weighted)");
            for (a, b) in got_w.iter().zip(&want_w) {
                assert!(
                    (a - b).abs() < 1.0e-3,
                    "weighted {a} != {b} ({head_count}x{head_dim})"
                );
            }
            // Weightless (V-norm).
            let want_n = cpu_per_head(&input, None, head_count, head_dim, eps);
            let got_n = try_rms_norm_per_head_f32(&input, None, head_count, head_dim, eps)
                .expect("metal per-head rms_norm (weightless)");
            for (a, b) in got_n.iter().zip(&want_n) {
                assert!(
                    (a - b).abs() < 1.0e-3,
                    "weightless {a} != {b} ({head_count}x{head_dim})"
                );
            }
        }
    }

    // Gemma's sliding-window attention needs no dedicated kernel: attending to the
    // window [lo..=pos] is the existing decode kernel with kv_base_offset shifted by
    // lo*position_stride and position_count = window length. This locks that property
    // in so a future attention-kernel refactor can't silently break gemma windowing.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_sliding_window_attention_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        // CPU reference: per head, softmax over the contiguous window [lo..lo+win)
        // of a [n_kv_heads][total][head_dim] cache, then the prob-weighted V sum.
        #[allow(clippy::too_many_arguments)]
        fn cpu_windowed(
            query: &[f32],
            keys: &[f32],
            values: &[f32],
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            total: usize,
            lo: usize,
            win: usize,
            scale: f32,
        ) -> Vec<f32> {
            let group = n_heads / n_kv_heads;
            let kv_head_stride = total * head_dim;
            let mut out = vec![0.0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let kvh = h / group;
                let q = &query[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; win];
                let mut m = f32::NEG_INFINITY;
                for (i, s) in scores.iter_mut().enumerate() {
                    let p = lo + i;
                    let k = &keys[kvh * kv_head_stride + p * head_dim..][..head_dim];
                    *s = scale * q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>();
                    m = m.max(*s);
                }
                let mut den = 0.0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    den += *s;
                }
                let o = &mut out[h * head_dim..(h + 1) * head_dim];
                for (i, &sc) in scores.iter().enumerate() {
                    let p = lo + i;
                    let v = &values[kvh * kv_head_stride + p * head_dim..][..head_dim];
                    let w = sc / den;
                    for d in 0..head_dim {
                        o[d] += w * v[d];
                    }
                }
            }
            out
        }

        let n_heads = 8usize;
        let n_kv_heads = 2usize;
        let total = 10usize; // positions 0..9 in the cache
                             // (head_dim, lo, win): a true sliding window and a full range (global, lo=0).
        for (head_dim, lo, win) in [(256usize, 3usize, 7usize), (512, 0, total)] {
            let kv_head_stride = total * head_dim;
            let query: Vec<f32> = (0..n_heads * head_dim)
                .map(|i| ((i as f32 % 23.0) - 11.0) * 0.05)
                .collect();
            let keys: Vec<f32> = (0..n_kv_heads * kv_head_stride)
                .map(|i| ((i as f32 % 31.0) - 15.0) * 0.03)
                .collect();
            let values: Vec<f32> = (0..n_kv_heads * kv_head_stride)
                .map(|i| ((i as f32 % 19.0) - 9.0) * 0.04)
                .collect();
            let scale = 1.0 / (head_dim as f32).sqrt();
            let want = cpu_windowed(
                &query, &keys, &values, n_heads, n_kv_heads, head_dim, total, lo, win, scale,
            );
            // Window = shift base by lo*position_stride, count = win.
            let got = try_attention_decode_strided_f32(
                &query,
                &keys,
                &values,
                n_heads,
                n_kv_heads,
                head_dim,
                win, // position_count = window length
                scale,
                head_dim,       // position_stride
                kv_head_stride, // kv_head_stride
                lo * head_dim,  // kv_base_offset = lo * position_stride
            )
            .expect("metal windowed attention");
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a - b).abs() < 1.0e-3,
                    "windowed {a} != {b} (hd={head_dim} lo={lo})"
                );
            }
        }
    }

    // Gemma4ResidentState allocates a per-layer KV cache sized to each layer's
    // per-type head_dim, and only for layers that own their K/V — the trailing
    // cross-shared layers hold None and read an earlier layer's cache.
    #[cfg(target_os = "macos")]
    #[test]
    fn gemma4_resident_state_allocates_per_layer_kv() {
        if !detect_metal_device().available {
            return;
        }
        use crate::model::Gemma4LayerPlan;
        let mk = |sliding, head_dim, owns_kv, src| Gemma4LayerPlan {
            sliding,
            head_dim,
            q_dim: 2 * head_dim,
            kv_heads: 1,
            kv_dim: head_dim,
            theta: 1.0,
            window: if sliding { Some(16) } else { None },
            owns_kv,
            kv_source_layer: src,
        };
        // 2 owning (sliding hd=4, global hd=8), then 2 shared reading them.
        let plan = vec![
            mk(true, 4, true, 0),
            mk(false, 8, true, 1),
            mk(true, 4, false, 0),
            mk(false, 8, false, 1),
        ];
        let n_kv_heads = 1usize;
        let hidden = 16usize;
        let max_positions = 32usize;
        let st = Gemma4ResidentState::new(plan, n_kv_heads, hidden, 1.0e-6, max_positions, 64)
            .expect("resident state");
        let bytes = |head_dim: usize| (n_kv_heads * max_positions * head_dim * 4) as u64;
        assert_eq!(st.cache_k_len(0), Some(bytes(4))); // owning sliding
        assert_eq!(st.cache_k_len(1), Some(bytes(8))); // owning global
        assert_eq!(st.cache_k_len(2), None); // shared sliding
        assert_eq!(st.cache_k_len(3), None); // shared global
                                             // Bad shapes are rejected.
        assert!(Gemma4ResidentState::new(vec![], 1, 16, 1.0e-6, 8, 16).is_none());
    }

    // A DENSE (no-PLE) resident model must apply layer_output_scale standalone at
    // the end of each layer (the reference multiplies it unconditionally; E-series
    // applies it inside the PLE encode). The layer here is also V-LESS, so this
    // covers the full 12B-class global-layer path through forward_token: V from
    // the raw K projection + standalone output scale + head.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_dense_layer_scale_and_vless_forward_token() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 64usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 32usize;
        let ffn_dim = 96usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let vocab = 96usize;
        let max_positions = 8usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let attn_scale = 1.0f32;
        let layer_scale = 0.37f32;

        let mw = |rows: usize, in_dim: usize, seed: usize| -> Vec<u8> {
            let mut wire = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                for blk in quantize_q8_0_blocks(&row).iter() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for &q in blk.quants.iter() {
                        wire.push(q as u8);
                    }
                }
            }
            wire
        };
        let build_layer = || {
            let an: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
            let pa: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
            let fnw: Vec<f32> = (0..hidden)
                .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
                .collect();
            let pf: Vec<f32> = (0..hidden)
                .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
                .collect();
            let qn: Vec<f32> = (0..head_dim)
                .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
                .collect();
            let kn: Vec<f32> = (0..head_dim)
                .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
                .collect();
            Gemma4ResidentLayer::from_wire(
                GemmaWireFmt::Q8_0,
                an,
                qn,
                kn,
                pa,
                fnw,
                pf,
                &mw(q_dim, hidden, 1),
                &mw(kv_dim, hidden, 5),
                None, // V-less
                &mw(hidden, q_dim, 13),
                &mw(ffn_dim, hidden, 17),
                &mw(ffn_dim, hidden, 21),
                &mw(hidden, ffn_dim, 25),
                n_heads,
                n_kv_heads,
                head_dim,
                ffn_dim,
                eps,
            )
            .expect("layer")
        };
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let embd_wire = mw(vocab, hidden, 99);
        let cache_len = n_kv_heads * max_positions * head_dim;
        let zeros = vec![0.0f32; cache_len];

        // Expected: validated layer wrapper, CPU scale, validated head wrapper.
        let h1 = try_gemma4_layer(
            &build_layer(),
            &h0,
            &cos_t,
            &sin_t,
            &zeros,
            &zeros,
            max_positions,
            0,
            1,
            0,
            attn_scale,
            true,
        )
        .expect("layer fwd");
        let h1s: Vec<f32> = h1.iter().map(|v| v * layer_scale).collect();
        let want =
            try_gemma4_head(&h1s, &output_norm, &embd_wire, vocab, softcap, eps).expect("head");

        // Got: the resident model with layer_scales = [0.37] and no PLE.
        let model = Gemma4ResidentModel::new(
            vec![build_layer()],
            vec![None],
            vec![layer_scale],
            vec![true],
            vec![0],
            &embd_wire,
            output_norm.clone(),
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            attn_scale,
        )
        .expect("model");
        let inputs = vec![Gemma4TokenLayerInput {
            cos_t: cos_t.clone(),
            sin_t: sin_t.clone(),
            pli: Vec::new(),
            window_start: 0,
        }];
        let got = model.forward_token(&h0, &inputs, &[], 0).expect("forward");
        let argmax =
            |v: &[f32]| -> usize { (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() };
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
        assert_eq!(argmax(&got), argmax(&want));
    }

    // The gemma4 GPU GEMV workhorse (f32 activation × 34-byte wire Q8) must match a
    // CPU f32×dequant reference — this is the op the resident decode runs 8x/layer.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_q8_matmul_f32y_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        // Cover both the single-simdgroup case (<=8 blocks) AND realistic gemma sizes
        // (80 = hidden/32, 320 = ffn_dim/32) that exercise the 4-way k-split reduction.
        for blocks_per_row in [5usize, 80, 320] {
            let in_dim = blocks_per_row * 32;
            let rows = 7usize;
            let y: Vec<f32> = (0..in_dim)
                .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
                .collect();
            let mut wire = Vec::with_capacity(rows * blocks_per_row * 34);
            let mut want = vec![0.0f32; rows];
            for (r, w) in want.iter_mut().enumerate() {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| (((r * in_dim + i) as f32 % 23.0) - 11.0) * 0.05)
                    .collect();
                let mut acc = 0.0f32;
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        acc += blk.scale * q as f32 * y[b * 32 + j];
                    }
                }
                *w = acc;
            }
            let got = try_gemma4_q8_matmul_f32y(&y, &wire, rows, blocks_per_row)
                .expect("gemma4 q8 matmul");
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 2.0e-2, "bpr={blocks_per_row} {a} != {b}");
            }
        }
        // Shape guards.
        let blocks_per_row = 5usize;
        let in_dim = blocks_per_row * 32;
        let y: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
            .collect();
        let wire = vec![0u8; 7 * blocks_per_row * 34];
        let rows = 7usize;
        assert!(try_gemma4_q8_matmul_f32y(&y, &wire, 0, blocks_per_row).is_none());
        assert!(
            try_gemma4_q8_matmul_f32y(&y, &wire[..wire.len() - 1], rows, blocks_per_row).is_none()
        );
    }

    // The QAT-row GPU GEMV workhorse (f32 activation × 18-byte wire Q4_0, nibbles
    // unpacked inline) must match the CPU f32×dequant reference — the foundation
    // for running the Q4_0 attention/dense/expert matmuls on the GPU.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_q4_0_matmul_f32y_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::q4_0_wire_block_dequant;
        for blocks_per_row in [5usize, 80, 320] {
            let in_dim = blocks_per_row * 32;
            let rows = 7usize;
            let y: Vec<f32> = (0..in_dim)
                .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
                .collect();
            let mut wire = Vec::with_capacity(rows * blocks_per_row * 18);
            let mut want = vec![0.0f32; rows];
            for (r, w) in want.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let scale = (((r + b) as f32 % 7.0) + 1.0) * 0.02;
                    let mut blk = [0u8; 18];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
                    for (j, slot) in blk[2..].iter_mut().enumerate() {
                        *slot = ((r * 31 + b * 17 + j * 3) % 256) as u8;
                    }
                    let deq = q4_0_wire_block_dequant(&blk);
                    for (i, &d) in deq.iter().enumerate() {
                        acc += d * y[b * 32 + i];
                    }
                    wire.extend_from_slice(&blk);
                }
                *w = acc;
            }
            let got = try_gemma4_q4_0_matmul_f32y(&y, &wire, rows, blocks_per_row)
                .expect("gemma4 q4_0 matmul");
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 2.0e-2, "bpr={blocks_per_row} {a} != {b}");
            }
        }
        // Shape guards.
        let blocks_per_row = 5usize;
        let in_dim = blocks_per_row * 32;
        let y: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
            .collect();
        let wire = vec![0u8; 7 * blocks_per_row * 18];
        let rows = 7usize;
        assert!(try_gemma4_q4_0_matmul_f32y(&y, &wire, 0, blocks_per_row).is_none());
        assert!(
            try_gemma4_q4_0_matmul_f32y(&y, &wire[..wire.len() - 1], rows, blocks_per_row)
                .is_none()
        );
    }

    // GABBRO M3 parity gate: the Metal NVFP4 GEMV must match the CPU oracle
    // `nvfp4_wire_block_dequant` × f32 activation (self-parity, §3 leg 3). Reduction
    // order differs from the CPU dot, so the bar is a tolerance (like the q4_0 gate),
    // not bit-exact; decode bit-exactness itself is proven by the M1 golden vectors.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_nvfp4_matmul_f32y_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::nvfp4_wire_block_dequant;
        for blocks_per_row in [3usize, 40, 160] {
            // blocks_per_row counts 64-value NVFP4 superblocks.
            let in_dim = blocks_per_row * 64;
            let rows = 7usize;
            let y: Vec<f32> = (0..in_dim)
                .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
                .collect();
            let mut wire = Vec::with_capacity(rows * blocks_per_row * 36);
            let mut want = vec![0.0f32; rows];
            for (r, w) in want.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let mut blk = [0u8; 36];
                    // 4 UE4M3 sub-block scales in [8, 72) — non-zero, and never the
                    // 0x7F/0xFF NaN sentinels.
                    for (s, slot) in blk[0..4].iter_mut().enumerate() {
                        *slot = (((r * 5 + b * 3 + s) % 64) + 8) as u8;
                    }
                    // 32 packed E2M1 nibble bytes.
                    for (j, slot) in blk[4..].iter_mut().enumerate() {
                        *slot = ((r * 31 + b * 17 + j * 3) % 256) as u8;
                    }
                    let deq = nvfp4_wire_block_dequant(&blk);
                    for (i, &d) in deq.iter().enumerate() {
                        acc += d * y[b * 64 + i];
                    }
                    wire.extend_from_slice(&blk);
                }
                *w = acc;
            }
            let got = try_gemma4_nvfp4_matmul_f32y(&y, &wire, rows, blocks_per_row)
                .expect("gemma4 nvfp4 matmul");
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 2.0e-2, "bpr={blocks_per_row} {a} != {b}");
            }
        }
        // Shape guards.
        let blocks_per_row = 3usize;
        let in_dim = blocks_per_row * 64;
        let y: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
            .collect();
        let wire = vec![0u8; 7 * blocks_per_row * 36];
        let rows = 7usize;
        assert!(try_gemma4_nvfp4_matmul_f32y(&y, &wire, 0, blocks_per_row).is_none());
        assert!(
            try_gemma4_nvfp4_matmul_f32y(&y, &wire[..wire.len() - 1], rows, blocks_per_row)
                .is_none()
        );
    }

    // The gemma4 FFN sub-block run as one GPU command buffer (rms_norm -> gate/up
    // GEMV -> GeGLU -> down GEMV -> post_ffw_norm -> residual) must match the same
    // chain on CPU. Proves dependent dispatches compose correctly in one serial
    // encoder with no readback between ops.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_ffn_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize; // bpr_hidden = 4
        let ffn_dim = 256usize; // bpr_ffn = 8
        let eps = 1.0e-6f32;
        // Build a weight as (wire bytes, dequantized f32 rows) so the CPU reference
        // dots the SAME values the GPU's f32×dequant GEMV reads.
        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.04)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (gate_wire, gate_deq) = make_weight(ffn_dim, hidden, 1);
        let (up_wire, up_deq) = make_weight(ffn_dim, hidden, 7);
        let (down_wire, down_deq) = make_weight(hidden, ffn_dim, 3);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let ffn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();

        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            x.iter().zip(w).map(|(v, wv)| v * inv * wv).collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let normf = rms(&h_in, &ffn_norm);
        let gate = matmul(&gate_deq, &normf);
        let up = matmul(&up_deq, &normf);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| crate::inference::gemma4::gelu_tanh(*g) * u)
            .collect();
        let down = matmul(&down_deq, &act);
        let dn = rms(&down, &post_norm);
        let want: Vec<f32> = h_in.iter().zip(&dn).map(|(a, b)| a + b).collect();

        let got = try_gemma4_ffn(
            GemmaWireFmt::Q8_0,
            &h_in,
            &ffn_norm,
            &post_norm,
            eps,
            &gate_wire,
            &up_wire,
            &down_wire,
            ffn_dim,
        )
        .expect("gemma4 ffn");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1.0e-2, "{a} != {b}");
        }
    }

    // GPU QAT integration parity gate: the WHOLE gemma4 FFN sub-block (rms_norm ->
    // gate/up GEMV -> GeGLU -> down GEMV -> post_ffw_norm -> residual) run on the
    // GPU with Q4_0 weights must match the same chain on CPU. This proves the Q4_0
    // wire GEMV composes correctly inside the real gemma4 graph — the rung above
    // the standalone kernel test, and the integration gate for moving the QAT
    // rows' dominant cost (the FFN) onto the GPU.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_ffn_q4_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::{gemma4::gelu_tanh, q4_0_wire_block_dequant};
        let hidden = 128usize; // bpr_hidden = 4
        let ffn_dim = 256usize; // bpr_ffn = 8
        let eps = 1.0e-6f32;
        // Build each weight as (Q4_0 wire bytes, dequantized f32 rows) so the CPU
        // reference dots the SAME values the GPU's f32×dequant Q4_0 GEMV reads.
        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let mut drow = vec![0.0f32; in_dim];
                for b in 0..(in_dim / 32) {
                    let scale = (((r + b + seed) % 7) as f32 + 1.0) * 0.015;
                    let mut blk = [0u8; 18];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
                    for (j, slot) in blk[2..].iter_mut().enumerate() {
                        *slot = ((r * 13 + b * 7 + j * 5 + seed) % 256) as u8;
                    }
                    let d = q4_0_wire_block_dequant(&blk);
                    for (i, &v) in d.iter().enumerate() {
                        drow[b * 32 + i] = v;
                    }
                    wire.extend_from_slice(&blk);
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (gate_wire, gate_deq) = make_weight(ffn_dim, hidden, 1);
        let (up_wire, up_deq) = make_weight(ffn_dim, hidden, 7);
        let (down_wire, down_deq) = make_weight(hidden, ffn_dim, 3);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let ffn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            x.iter().zip(w).map(|(v, wv)| v * inv * wv).collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let normf = rms(&h_in, &ffn_norm);
        let gate = matmul(&gate_deq, &normf);
        let up = matmul(&up_deq, &normf);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * u)
            .collect();
        let down = matmul(&down_deq, &act);
        let dn = rms(&down, &post_norm);
        let want: Vec<f32> = h_in.iter().zip(&dn).map(|(a, b)| a + b).collect();

        let got = try_gemma4_ffn(
            GemmaWireFmt::Q4_0,
            &h_in,
            &ffn_norm,
            &post_norm,
            eps,
            &gate_wire,
            &up_wire,
            &down_wire,
            ffn_dim,
        )
        .expect("gemma4 q4 ffn");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1.0e-2, "{a} != {b}");
        }
    }

    // The gemma4 attention sub-block run as one GPU command buffer (rms_norm -> qkv
    // GEMV -> per-head QK/V norm -> RoPE -> KV scatter -> windowed decode attention ->
    // o GEMV -> post_attn_norm -> residual) must match the same chain on CPU. Uses
    // head_dim 256 (gemma's dim, forces the basic decode kernel) and a prefilled cache.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_attention_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let window_start = 0usize;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = make_weight(q_dim, hidden, 1);
        let (k_wire, k_deq) = make_weight(kv_dim, hidden, 5);
        let (v_wire, v_deq) = make_weight(kv_dim, hidden, 9);
        let (o_wire, o_deq) = make_weight(hidden, q_dim, 13);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        let cache_k_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // ---- CPU reference ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        let normf = rms(&h_in, Some(&attn_norm));
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let mut kk = per_head(&matmul(&k_deq, &normf), n_kv_heads, Some(&k_norm));
        let vv = per_head(&matmul(&v_deq, &normf), n_kv_heads, None);
        rope(&mut q, n_heads);
        rope(&mut kk, n_kv_heads);
        // Effective cache: prefilled positions, current token's k/v at write_position.
        let kv_at = |init: &[f32], cur: &[f32], kvh: usize, p: usize| -> Vec<f32> {
            if p == write_position {
                cur[kvh * head_dim..(kvh + 1) * head_dim].to_vec()
            } else {
                let base = kvh * max_positions * head_dim + p * head_dim;
                init[base..base + head_dim].to_vec()
            }
        };
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..filled {
                let kp = kv_at(&cache_k_init, &kk, kvh, p);
                scores.push(scale * qh.iter().zip(&kp).map(|(a, b)| a * b).sum::<f32>());
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                den += *s;
            }
            for (idx, p) in (window_start..filled).enumerate() {
                let vp = kv_at(&cache_v_init, &vv, kvh, p);
                let w = scores[idx] / den;
                for d in 0..head_dim {
                    ctx[h * head_dim + d] += w * vp[d];
                }
            }
        }
        let o = matmul(&o_deq, &ctx);
        let on = rms(&o, Some(&post_norm));
        let want: Vec<f32> = h_in.iter().zip(&on).map(|(a, b)| a + b).collect();

        let got = try_gemma4_attention(
            GemmaWireFmt::Q8_0,
            &h_in,
            &attn_norm,
            &q_norm,
            &k_norm,
            &post_norm,
            eps,
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &cos_t,
            &sin_t,
            &cache_k_init,
            &cache_v_init,
            max_positions,
            write_position,
            n_heads,
            n_kv_heads,
            head_dim,
            filled,
            window_start,
            scale,
            true, // owning layer
        )
        .expect("gemma4 attention");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
    }

    // QAT rung: the SAME gemma4 attention chain, but the q/k/v/o projections are
    // Q4_0 wire weights (18-byte blocks) dispatched through the GPU Q4_0 GEMV. Proves
    // GemmaWireFmt::Q4_0 routes attention end-to-end and matches the CPU f32×dequant
    // reference over the identical dequantized weights. Same geometry as the Q8 test.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_attention_q4_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::q4_0_wire_block_dequant;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let window_start = 0usize;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q4_0 wire weight + its dequantized rows (the CPU reference dots the SAME
        // values the GPU's f32×dequant Q4_0 GEMV reads).
        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let mut drow = vec![0.0f32; in_dim];
                for b in 0..(in_dim / 32) {
                    let scale = (((r + b + seed) % 7) as f32 + 1.0) * 0.015;
                    let mut blk = [0u8; 18];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
                    for (j, slot) in blk[2..].iter_mut().enumerate() {
                        *slot = ((r * 13 + b * 7 + j * 5 + seed) % 256) as u8;
                    }
                    let d = q4_0_wire_block_dequant(&blk);
                    for (i, &v) in d.iter().enumerate() {
                        drow[b * 32 + i] = v;
                    }
                    wire.extend_from_slice(&blk);
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = make_weight(q_dim, hidden, 1);
        let (k_wire, k_deq) = make_weight(kv_dim, hidden, 5);
        let (v_wire, v_deq) = make_weight(kv_dim, hidden, 9);
        let (o_wire, o_deq) = make_weight(hidden, q_dim, 13);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        let cache_k_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // ---- CPU reference ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        let normf = rms(&h_in, Some(&attn_norm));
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let mut kk = per_head(&matmul(&k_deq, &normf), n_kv_heads, Some(&k_norm));
        let vv = per_head(&matmul(&v_deq, &normf), n_kv_heads, None);
        rope(&mut q, n_heads);
        rope(&mut kk, n_kv_heads);
        let kv_at = |init: &[f32], cur: &[f32], kvh: usize, p: usize| -> Vec<f32> {
            if p == write_position {
                cur[kvh * head_dim..(kvh + 1) * head_dim].to_vec()
            } else {
                let base = kvh * max_positions * head_dim + p * head_dim;
                init[base..base + head_dim].to_vec()
            }
        };
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..filled {
                let kp = kv_at(&cache_k_init, &kk, kvh, p);
                scores.push(scale * qh.iter().zip(&kp).map(|(a, b)| a * b).sum::<f32>());
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                den += *s;
            }
            for (idx, p) in (window_start..filled).enumerate() {
                let vp = kv_at(&cache_v_init, &vv, kvh, p);
                let w = scores[idx] / den;
                for d in 0..head_dim {
                    ctx[h * head_dim + d] += w * vp[d];
                }
            }
        }
        let o = matmul(&o_deq, &ctx);
        let on = rms(&o, Some(&post_norm));
        let want: Vec<f32> = h_in.iter().zip(&on).map(|(a, b)| a + b).collect();

        let got = try_gemma4_attention(
            GemmaWireFmt::Q4_0,
            &h_in,
            &attn_norm,
            &q_norm,
            &k_norm,
            &post_norm,
            eps,
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &cos_t,
            &sin_t,
            &cache_k_init,
            &cache_v_init,
            max_positions,
            write_position,
            n_heads,
            n_kv_heads,
            head_dim,
            filled,
            window_start,
            scale,
            true, // owning layer
        )
        .expect("gemma4 q4 attention");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
    }

    // V-less attention (12B-class full-attention layers carry no attn_v): the GPU
    // graph must use the RAW K projection (pre-norm, pre-rope) as the V source —
    // reference: `if v_proj is not present, use Kcur as Vcur`. Geometry mirrors a
    // real 12B global layer: kv_heads = 1, head_dim = 512.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_vless_attention_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 256usize;
        let n_heads = 4usize;
        let n_kv_heads = 1usize;
        let head_dim = 512usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let window_start = 0usize;
        let eps = 1.0e-6f32;
        let scale = 1.0f32; // gemma folds the attention scale into the normed query

        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = make_weight(q_dim, hidden, 3);
        let (k_wire, k_deq) = make_weight(kv_dim, hidden, 7);
        let (o_wire, o_deq) = make_weight(hidden, q_dim, 11);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        let cache_k_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // ---- CPU reference (V = weightless-normed RAW K projection) ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        let normf = rms(&h_in, Some(&attn_norm));
        let k_raw = matmul(&k_deq, &normf);
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let mut kk = per_head(&k_raw, n_kv_heads, Some(&k_norm));
        // V from the RAW K projection, weightless norm, never roped.
        let vv = per_head(&k_raw, n_kv_heads, None);
        rope(&mut q, n_heads);
        rope(&mut kk, n_kv_heads);
        let kv_at = |init: &[f32], cur: &[f32], kvh: usize, p: usize| -> Vec<f32> {
            if p == write_position {
                cur[kvh * head_dim..(kvh + 1) * head_dim].to_vec()
            } else {
                let base = kvh * max_positions * head_dim + p * head_dim;
                init[base..base + head_dim].to_vec()
            }
        };
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..filled {
                let kp = kv_at(&cache_k_init, &kk, kvh, p);
                scores.push(scale * qh.iter().zip(&kp).map(|(a, b)| a * b).sum::<f32>());
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                den += *s;
            }
            for (idx, p) in (window_start..filled).enumerate() {
                let vp = kv_at(&cache_v_init, &vv, kvh, p);
                let w = scores[idx] / den;
                for d in 0..head_dim {
                    ctx[h * head_dim + d] += w * vp[d];
                }
            }
        }
        let o = matmul(&o_deq, &ctx);
        let on = rms(&o, Some(&post_norm));
        let want: Vec<f32> = h_in.iter().zip(&on).map(|(a, b)| a + b).collect();

        let got = try_gemma4_attention(
            GemmaWireFmt::Q8_0,
            &h_in,
            &attn_norm,
            &q_norm,
            &k_norm,
            &post_norm,
            eps,
            &q_wire,
            &k_wire,
            None, // V-less: no attn_v weight
            &o_wire,
            &cos_t,
            &sin_t,
            &cache_k_init,
            &cache_v_init,
            max_positions,
            write_position,
            n_heads,
            n_kv_heads,
            head_dim,
            filled,
            window_start,
            scale,
            true, // owning layer
        )
        .expect("gemma4 v-less attention");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
    }

    // A cross-shared gemma4 layer (owns_kv = false) must skip K/V projection + scatter
    // and run q-only attention against the source layer's cache (here a fully-prefilled
    // cache covering all `filled` positions). Validates the owns_kv=false branch.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_attention_shared_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let window_start = 0usize;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = make_weight(q_dim, hidden, 1);
        // K/V weights are required by the wrapper's shape checks but unused when shared.
        let (k_wire, _) = make_weight(kv_dim, hidden, 5);
        let (v_wire, _) = make_weight(kv_dim, hidden, 9);
        let (o_wire, o_deq) = make_weight(hidden, q_dim, 13);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        // Fully prefilled: the source layer already wrote every position incl. write_position.
        let cache_k_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // CPU reference: q only, attention over the source cache directly.
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let normf = rms(&h_in, Some(&attn_norm));
        let mut q = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let qh = rms(
                &matmul(&q_deq, &normf)[h * head_dim..(h + 1) * head_dim],
                Some(&q_norm),
            );
            q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&qh);
        }
        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..half {
                let (x0, x1) = (q[base + i], q[base + half + i]);
                q[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                q[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
            }
        }
        let cache_at = |c: &[f32], kvh: usize, p: usize| -> Vec<f32> {
            let base = kvh * max_positions * head_dim + p * head_dim;
            c[base..base + head_dim].to_vec()
        };
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..filled {
                let kp = cache_at(&cache_k_init, kvh, p);
                scores.push(scale * qh.iter().zip(&kp).map(|(a, b)| a * b).sum::<f32>());
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                den += *s;
            }
            for (idx, p) in (window_start..filled).enumerate() {
                let vp = cache_at(&cache_v_init, kvh, p);
                let w = scores[idx] / den;
                for d in 0..head_dim {
                    ctx[h * head_dim + d] += w * vp[d];
                }
            }
        }
        let on = rms(&matmul(&o_deq, &ctx), Some(&post_norm));
        let want: Vec<f32> = h_in.iter().zip(&on).map(|(a, b)| a + b).collect();

        let got = try_gemma4_attention(
            GemmaWireFmt::Q8_0,
            &h_in,
            &attn_norm,
            &q_norm,
            &k_norm,
            &post_norm,
            eps,
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &cos_t,
            &sin_t,
            &cache_k_init,
            &cache_v_init,
            max_positions,
            write_position,
            n_heads,
            n_kv_heads,
            head_dim,
            filled,
            window_start,
            scale,
            false, // shared layer: read the source cache, no K/V projection
        )
        .expect("gemma4 shared attention");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
    }

    // A full gemma4 layer (attention sub-block then FFN sub-block, chained in one GPU
    // command buffer via Gemma4ResidentLayer) must match the same two-stage chain on
    // CPU: h_mid = h_in + attention(h_in); h_out = h_mid + ffn(h_mid).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_layer_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 2560usize;
        let n_heads = 8usize;
        let n_kv_heads = 2usize;
        let head_dim = 256usize;
        let ffn_dim = 10240usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let window_start = 0usize;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = make_weight(q_dim, hidden, 1);
        let (k_wire, k_deq) = make_weight(kv_dim, hidden, 5);
        let (v_wire, v_deq) = make_weight(kv_dim, hidden, 9);
        let (o_wire, o_deq) = make_weight(hidden, q_dim, 13);
        let (gate_wire, gate_deq) = make_weight(ffn_dim, hidden, 17);
        let (up_wire, up_deq) = make_weight(ffn_dim, hidden, 21);
        let (down_wire, down_deq) = make_weight(hidden, ffn_dim, 25);
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_attn: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let ffn_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
            .collect();
        let post_ffw: Vec<f32> = (0..hidden)
            .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
            .collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        let cache_k_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v_init: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // ---- CPU reference: attention(h_in) -> h_mid, then ffn(h_mid) -> h_out ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        // attention
        let normf = rms(&h_in, Some(&attn_norm));
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let mut kk = per_head(&matmul(&k_deq, &normf), n_kv_heads, Some(&k_norm));
        let vv = per_head(&matmul(&v_deq, &normf), n_kv_heads, None);
        rope(&mut q, n_heads);
        rope(&mut kk, n_kv_heads);
        let kv_at = |init: &[f32], cur: &[f32], kvh: usize, p: usize| -> Vec<f32> {
            if p == write_position {
                cur[kvh * head_dim..(kvh + 1) * head_dim].to_vec()
            } else {
                let base = kvh * max_positions * head_dim + p * head_dim;
                init[base..base + head_dim].to_vec()
            }
        };
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..filled {
                let kp = kv_at(&cache_k_init, &kk, kvh, p);
                scores.push(scale * qh.iter().zip(&kp).map(|(a, b)| a * b).sum::<f32>());
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                den += *s;
            }
            for (idx, p) in (window_start..filled).enumerate() {
                let vp = kv_at(&cache_v_init, &vv, kvh, p);
                let w = scores[idx] / den;
                for d in 0..head_dim {
                    ctx[h * head_dim + d] += w * vp[d];
                }
            }
        }
        let attn_out = rms(&matmul(&o_deq, &ctx), Some(&post_attn));
        let h_mid: Vec<f32> = h_in.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
        // ffn
        let fnormf = rms(&h_mid, Some(&ffn_norm));
        let gate = matmul(&gate_deq, &fnormf);
        let up = matmul(&up_deq, &fnormf);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| crate::inference::gemma4::gelu_tanh(*g) * u)
            .collect();
        let dn = rms(&matmul(&down_deq, &act), Some(&post_ffw));
        let want: Vec<f32> = h_mid.iter().zip(&dn).map(|(a, b)| a + b).collect();

        let layer = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Q8_0,
            attn_norm.clone(),
            q_norm.clone(),
            k_norm.clone(),
            post_attn.clone(),
            ffn_norm.clone(),
            post_ffw.clone(),
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &gate_wire,
            &up_wire,
            &down_wire,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("layer weights");
        let got = try_gemma4_layer(
            &layer,
            &h_in,
            &cos_t,
            &sin_t,
            &cache_k_init,
            &cache_v_init,
            max_positions,
            write_position,
            filled,
            window_start,
            scale,
            true, // owning layer
        )
        .expect("gemma4 layer");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 3.0e-2, "{a} != {b}");
        }
    }

    // Two gemma4 layers chained in ONE command buffer with a persistent shared cache:
    // layer 0 owns its K/V and scatters this token; layer 1 (cross-shared) reads
    // layer 0's cache (incl. the just-scattered token) with no K/V projection. Must
    // match the same two-layer chain on CPU. Validates the multi-layer driver +
    // cross-layer KV sharing end-to-end.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_two_layers_shared_kv_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let group = n_heads / n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_weight = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                let mut drow = vec![0.0f32; in_dim];
                for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for (j, &q) in blk.quants.iter().enumerate() {
                        wire.push(q as u8);
                        drow[b * 32 + j] = blk.scale * q as f32;
                    }
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        // Distinct weights per layer (seed offset 100 for layer 1).
        let mk_layer_w = |off: usize| {
            (
                make_weight(q_dim, hidden, off + 1),
                make_weight(kv_dim, hidden, off + 5),
                make_weight(kv_dim, hidden, off + 9),
                make_weight(hidden, q_dim, off + 13),
                make_weight(ffn_dim, hidden, off + 17),
                make_weight(ffn_dim, hidden, off + 21),
                make_weight(hidden, ffn_dim, off + 25),
            )
        };
        let w0 = mk_layer_w(0);
        let w1 = mk_layer_w(100);
        let mk_norms = |off: usize| {
            (
                (0..hidden)
                    .map(|i| 0.8 + ((i + off) as f32 % 5.0) * 0.05)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.9 + ((i + off) as f32 % 3.0) * 0.03)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.85 + ((i + off) as f32 % 4.0) * 0.04)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.95 + ((i + off) as f32 % 6.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.7 + ((i + off) as f32 % 11.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.6 + ((i + off) as f32 % 7.0) * 0.03)
                    .collect::<Vec<f32>>(),
            )
        };
        let (an0, pa0, fn0, pf0, qn0, kn0) = mk_norms(0);
        let (an1, pa1, fn1, pf1, qn1, kn1) = mk_norms(3);
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let cache_len = n_kv_heads * max_positions * head_dim;
        let cache_k0: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cache_v0: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();

        // ---- CPU reference ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        // attention over a flat cache; returns ctx [q_dim].
        let attend = |q: &[f32], ck: &[f32], cv: &[f32]| -> Vec<f32> {
            let mut ctx = vec![0.0f32; q_dim];
            for h in 0..n_heads {
                let kvh = h / group;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut sc = Vec::new();
                for p in 0..filled {
                    let base = kvh * max_positions * head_dim + p * head_dim;
                    sc.push(
                        scale
                            * qh.iter()
                                .zip(&ck[base..base + head_dim])
                                .map(|(a, b)| a * b)
                                .sum::<f32>(),
                    );
                }
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0f32;
                for s in &mut sc {
                    *s = (*s - m).exp();
                    den += *s;
                }
                for (idx, p) in (0..filled).enumerate() {
                    let base = kvh * max_positions * head_dim + p * head_dim;
                    let wgt = sc[idx] / den;
                    for d in 0..head_dim {
                        ctx[h * head_dim + d] += wgt * cv[base + d];
                    }
                }
            }
            ctx
        };
        let ffn = |h: &[f32],
                   gate: &[Vec<f32>],
                   up: &[Vec<f32>],
                   down: &[Vec<f32>],
                   fnw: &[f32],
                   pfw: &[f32]|
         -> Vec<f32> {
            let nf = rms(h, Some(fnw));
            let g = matmul(gate, &nf);
            let u = matmul(up, &nf);
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(a, b)| crate::inference::gemma4::gelu_tanh(*a) * b)
                .collect();
            rms(&matmul(down, &act), Some(pfw))
        };

        // layer 0 (owning): compute k/v, inject into a copy of cache0, attention, ffn.
        let nf0 = rms(&h_in, Some(&an0));
        let mut q0 = per_head(&matmul(&w0.0 .1, &nf0), n_heads, Some(&qn0));
        let mut k0 = per_head(&matmul(&w0.1 .1, &nf0), n_kv_heads, Some(&kn0));
        let v0 = per_head(&matmul(&w0.2 .1, &nf0), n_kv_heads, None);
        rope(&mut q0, n_heads);
        rope(&mut k0, n_kv_heads);
        let mut eff_k = cache_k0.clone();
        let mut eff_v = cache_v0.clone();
        for kvh in 0..n_kv_heads {
            let base = kvh * max_positions * head_dim + write_position * head_dim;
            eff_k[base..base + head_dim].copy_from_slice(&k0[kvh * head_dim..(kvh + 1) * head_dim]);
            eff_v[base..base + head_dim].copy_from_slice(&v0[kvh * head_dim..(kvh + 1) * head_dim]);
        }
        let ctx0 = attend(&q0, &eff_k, &eff_v);
        let attn0 = rms(&matmul(&w0.3 .1, &ctx0), Some(&pa0));
        let h_mid0: Vec<f32> = h_in.iter().zip(&attn0).map(|(a, b)| a + b).collect();
        let dn0 = ffn(&h_mid0, &w0.4 .1, &w0.5 .1, &w0.6 .1, &fn0, &pf0);
        let h_out0: Vec<f32> = h_mid0.iter().zip(&dn0).map(|(a, b)| a + b).collect();

        // layer 1 (shared, reads eff_k/eff_v from layer 0): q only.
        let nf1 = rms(&h_out0, Some(&an1));
        let mut q1 = per_head(&matmul(&w1.0 .1, &nf1), n_heads, Some(&qn1));
        rope(&mut q1, n_heads);
        let ctx1 = attend(&q1, &eff_k, &eff_v);
        let attn1 = rms(&matmul(&w1.3 .1, &ctx1), Some(&pa1));
        let h_mid1: Vec<f32> = h_out0.iter().zip(&attn1).map(|(a, b)| a + b).collect();
        let dn1 = ffn(&h_mid1, &w1.4 .1, &w1.5 .1, &w1.6 .1, &fn1, &pf1);
        let want: Vec<f32> = h_mid1.iter().zip(&dn1).map(|(a, b)| a + b).collect();

        // ---- GPU: two layers, one command buffer, persistent shared cache ----
        let layer0 = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Q8_0,
            an0,
            qn0,
            kn0,
            pa0,
            fn0,
            pf0,
            &w0.0 .0,
            &w0.1 .0,
            Some(&w0.2 .0),
            &w0.3 .0,
            &w0.4 .0,
            &w0.5 .0,
            &w0.6 .0,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("layer0");
        let layer1 = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Q8_0,
            an1,
            qn1,
            kn1,
            pa1,
            fn1,
            pf1,
            &w1.0 .0,
            &w1.1 .0,
            Some(&w1.2 .0),
            &w1.3 .0,
            &w1.4 .0,
            &w1.5 .0,
            &w1.6 .0,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("layer1");
        let kk = metal_linear_kernel().expect("metal");
        let mk = |bytes: usize| {
            kk.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let buf_a = mk(hidden * 4);
        let buf_b = mk(hidden * 4);
        let mid = mk(hidden * 4);
        let ck = mk(cache_len * 4);
        let cv = mk(cache_len * 4);
        write_buffer_f32(&buf_a, &h_in);
        write_buffer_f32(&ck, &cache_k0);
        write_buffer_f32(&cv, &cache_v0);
        let mut keep = Vec::new();
        let cb = kk.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        // layer 0 owns: buf_a -> buf_b, scatters into ck/cv.
        encode_gemma4_layer(
            e,
            kk,
            &mut keep,
            &layer0,
            &buf_a,
            &mid,
            &buf_b,
            &cos_t,
            &sin_t,
            &ck,
            &cv,
            max_positions,
            write_position,
            filled,
            0,
            scale,
            true,
        );
        // layer 1 shared: buf_b -> buf_a, reads ck/cv (now holding layer 0's token).
        encode_gemma4_layer(
            e,
            kk,
            &mut keep,
            &layer1,
            &buf_b,
            &mid,
            &buf_a,
            &cos_t,
            &sin_t,
            &ck,
            &cv,
            max_positions,
            write_position,
            filled,
            0,
            scale,
            false,
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let mut got = vec![0.0f32; hidden];
        read_buffer_f32(&buf_a, &mut got);
        drop(keep);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 4.0e-2, "{a} != {b}");
        }
    }

    // Gemma's PLE per-layer injection on GPU (ple_inp_gate GEMV -> gelu*pli ->
    // ple_proj GEMV -> rms_norm(post_norm) -> residual -> output_scale) must match the
    // CPU reference (gemma4_runtime's per-layer PLE step).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_ple_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let hidden = 128usize;
        let ple_dim = 64usize;
        let eps = 1.0e-6f32;
        let output_scale = 0.061f32;
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let pli: Vec<f32> = (0..ple_dim)
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.2)
            .collect();
        // Output-major: inp_gate[o*hidden + i] (o in 0..ple_dim), proj[o*ple_dim + i].
        let inp_gate: Vec<f32> = (0..ple_dim * hidden)
            .map(|n| (((n % 29) as f32) - 14.0) * 0.02)
            .collect();
        let proj_w: Vec<f32> = (0..hidden * ple_dim)
            .map(|n| (((n % 23) as f32) - 11.0) * 0.03)
            .collect();
        let post_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 5.0) * 0.04).collect();

        // CPU reference.
        let mut gated = vec![0.0f32; ple_dim];
        for (o, g) in gated.iter_mut().enumerate() {
            let dot: f32 = (0..hidden)
                .map(|i| inp_gate[o * hidden + i] * h_in[i])
                .sum();
            *g = crate::inference::gemma4::gelu_tanh(dot) * pli[o];
        }
        let mut proj = vec![0.0f32; hidden];
        for (o, pr) in proj.iter_mut().enumerate() {
            *pr = (0..ple_dim)
                .map(|i| proj_w[o * ple_dim + i] * gated[i])
                .sum();
        }
        let mss = proj.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mss + eps).powf(-0.5);
        let want: Vec<f32> = (0..hidden)
            .map(|i| (h_in[i] + proj[i] * inv * post_norm[i]) * output_scale)
            .collect();

        let got = try_gemma4_ple(
            &h_in,
            &pli,
            &inp_gate,
            &proj_w,
            &post_norm,
            output_scale,
            eps,
            ple_dim,
        )
        .expect("gemma4 ple");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    // GPU pli (the folded encode_gemma4_pli) must match the CPU pli prep
    // (Gemma4Runtime::step / Gemma4GpuRuntime::forward).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_pli_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let hidden = 128usize;
        let ple_dim = 64usize;
        let n_layers = 3usize;
        let ple_total = n_layers * ple_dim;
        let eps = 1.0e-6f32;
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let proj: Vec<f32> = (0..ple_total * hidden)
            .map(|n| (((n % 29) as f32) - 14.0) * 0.01)
            .collect();
        let proj_norm: Vec<f32> = (0..ple_dim)
            .map(|i| 0.8 + (i as f32 % 5.0) * 0.05)
            .collect();
        let ti: Vec<f32> = (0..ple_total)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
            .collect();

        // CPU reference (step() math).
        let proj_scale = (hidden as f32).powf(-0.5);
        let embed_scale = (ple_dim as f32).sqrt();
        let frac = std::f32::consts::FRAC_1_SQRT_2;
        let mut want = vec![0.0f32; ple_total];
        let ctx: Vec<f32> = (0..ple_total)
            .map(|o| {
                (0..hidden)
                    .map(|i| proj[o * hidden + i] * h0[i])
                    .sum::<f32>()
            })
            .collect();
        for l in 0..n_layers {
            let ctx_l: Vec<f32> = (0..ple_dim)
                .map(|d| ctx[l * ple_dim + d] * proj_scale)
                .collect();
            let mss = ctx_l.iter().map(|v| v * v).sum::<f32>() / ple_dim as f32;
            let inv = (mss + eps).powf(-0.5);
            for d in 0..ple_dim {
                want[l * ple_dim + d] =
                    (ctx_l[d] * inv * proj_norm[d] + ti[l * ple_dim + d] * embed_scale) * frac;
            }
        }
        let got = try_gemma4_pli(&h0, &proj, &proj_norm, &ti, hidden, ple_dim, n_layers, eps)
            .expect("gemma4 pli");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    // Gemma's logits head on GPU (rms_norm(output_norm) -> tied token_embd GEMV ->
    // soft_cap) must match the CPU reference, and the argmax (greedy next token) must
    // agree.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_head_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let vocab = 160usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.12)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i as f32 % 5.0) * 0.05)
            .collect();
        // Vocab-major Q8 embedding table: row v is token v's embedding.
        let mut wire = Vec::new();
        let mut deq = Vec::new();
        for v in 0..vocab {
            let row: Vec<f32> = (0..hidden)
                .map(|i| ((((v * hidden + i) % 31) as f32) - 15.0) * 0.05)
                .collect();
            let mut drow = vec![0.0f32; hidden];
            for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                for (j, &q) in blk.quants.iter().enumerate() {
                    wire.push(q as u8);
                    drow[b * 32 + j] = blk.scale * q as f32;
                }
            }
            deq.push(drow);
        }
        // CPU reference: rms_norm -> matvec -> soft_cap.
        let mss = h_in.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mss + eps).powf(-0.5);
        let normf: Vec<f32> = (0..hidden)
            .map(|i| h_in[i] * inv * output_norm[i])
            .collect();
        let want: Vec<f32> = deq
            .iter()
            .map(|row| {
                let logit: f32 = row.iter().zip(&normf).map(|(a, b)| a * b).sum();
                softcap * (logit / softcap).tanh()
            })
            .collect();

        let got =
            try_gemma4_head(&h_in, &output_norm, &wire, vocab, softcap, eps).expect("gemma4 head");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 5.0e-3, "{a} != {b}");
        }
        // Greedy argmax must agree.
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(amax(&got), amax(&want));
    }

    // The full GPU forward (try_gemma4_forward: N layers + per-layer PLE + head, one
    // command buffer) must match the SAME pipeline assembled from the individually
    // CPU-validated try_* pieces in sequence. Two owning layers + PLE + head.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_forward_matches_composed_pieces() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let ple_dim = 64usize;
        let vocab = 160usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let write_position = 3usize;
        let filled = 4usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mw = |rows: usize, in_dim: usize, seed: usize| -> Vec<u8> {
            let mut wire = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                for blk in quantize_q8_0_blocks(&row).iter() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for &q in blk.quants.iter() {
                        wire.push(q as u8);
                    }
                }
            }
            wire
        };
        let mk_norms = |off: usize| {
            (
                (0..hidden)
                    .map(|i| 0.8 + ((i + off) as f32 % 5.0) * 0.05)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.9 + ((i + off) as f32 % 3.0) * 0.03)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.85 + ((i + off) as f32 % 4.0) * 0.04)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.95 + ((i + off) as f32 % 6.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.7 + ((i + off) as f32 % 11.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.6 + ((i + off) as f32 % 7.0) * 0.03)
                    .collect::<Vec<f32>>(),
            )
        };
        let build_layer = |off: usize| {
            let (an, pa, fnw, pf, qn, kn) = mk_norms(off);
            Gemma4ResidentLayer::from_wire(
                GemmaWireFmt::Q8_0,
                an,
                qn,
                kn,
                pa,
                fnw,
                pf,
                &mw(q_dim, hidden, off + 1),
                &mw(kv_dim, hidden, off + 5),
                Some(&mw(kv_dim, hidden, off + 9)),
                &mw(hidden, q_dim, off + 13),
                &mw(ffn_dim, hidden, off + 17),
                &mw(ffn_dim, hidden, off + 21),
                &mw(hidden, ffn_dim, off + 25),
                n_heads,
                n_kv_heads,
                head_dim,
                ffn_dim,
                eps,
            )
            .expect("layer")
        };
        let mk_ple = |off: usize| Gemma4ResidentPle {
            inp_gate: (0..ple_dim * hidden)
                .map(|n| (((n + off) % 29) as f32 - 14.0) * 0.02)
                .collect(),
            proj: (0..hidden * ple_dim)
                .map(|n| (((n + off) % 23) as f32 - 11.0) * 0.03)
                .collect(),
            post_norm: (0..hidden)
                .map(|i| 0.9 + ((i + off) as f32 % 5.0) * 0.04)
                .collect(),
            output_scale: 0.061,
        };
        let mk_input = |off: usize| Gemma4TokenLayerInput {
            cos_t: (0..half)
                .map(|i| (0.3 + (i + off) as f32 * 0.01).cos())
                .collect(),
            sin_t: (0..half)
                .map(|i| (0.3 + (i + off) as f32 * 0.01).sin())
                .collect(),
            pli: (0..ple_dim)
                .map(|i| ((i + off) as f32 % 9.0 - 4.0) * 0.2)
                .collect(),
            window_start: 0,
        };
        let cache_len = n_kv_heads * max_positions * head_dim;
        let mk_cache = |off: usize| -> Vec<f32> {
            (0..cache_len)
                .map(|i| (((i + off) % 17) as f32 - 8.0) * 0.05)
                .collect()
        };

        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let embd_wire = mw(vocab, hidden, 99);

        let l0 = build_layer(0);
        let l1 = build_layer(50);
        let ple0 = mk_ple(2);
        let ple1 = mk_ple(7);
        let in0 = mk_input(0);
        let in1 = mk_input(4);
        let ck0 = mk_cache(0);
        let cv0 = mk_cache(3);
        let ck1 = mk_cache(6);
        let cv1 = mk_cache(11);

        // ---- Oracle: chain the individually CPU-validated try_* pieces. ----
        let h1 = try_gemma4_layer(
            &l0,
            &h0,
            &in0.cos_t,
            &in0.sin_t,
            &ck0,
            &cv0,
            max_positions,
            write_position,
            filled,
            0,
            scale,
            true,
        )
        .expect("l0");
        let h1p = try_gemma4_ple(
            &h1,
            &in0.pli,
            &ple0.inp_gate,
            &ple0.proj,
            &ple0.post_norm,
            ple0.output_scale,
            eps,
            ple_dim,
        )
        .expect("ple0");
        let h2 = try_gemma4_layer(
            &l1,
            &h1p,
            &in1.cos_t,
            &in1.sin_t,
            &ck1,
            &cv1,
            max_positions,
            write_position,
            filled,
            0,
            scale,
            true,
        )
        .expect("l1");
        let h2p = try_gemma4_ple(
            &h2,
            &in1.pli,
            &ple1.inp_gate,
            &ple1.proj,
            &ple1.post_norm,
            ple1.output_scale,
            eps,
            ple_dim,
        )
        .expect("ple1");
        let want =
            try_gemma4_head(&h2p, &output_norm, &embd_wire, vocab, softcap, eps).expect("head");

        // ---- Integrated: one command buffer for everything. ----
        let got = try_gemma4_forward(
            &[l0, l1],
            &[Some(ple0), Some(ple1)],
            &[true, true],
            &[0, 1],
            &[in0, in1],
            &[Some(ck0), Some(ck1)],
            &[Some(cv0), Some(cv1)],
            &h0,
            &output_norm,
            &embd_wire,
            vocab,
            softcap,
            eps,
            max_positions,
            write_position,
            filled,
            scale,
        )
        .expect("forward");

        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 5.0e-3, "{a} != {b}");
        }
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(amax(&got), amax(&want));
    }

    // from_wire_pages (nocopy, the production 16GB-fit path) must produce a layer
    // identical to from_wire (copy) on the same wire bytes: the GPU reads the same
    // bytes either way. Writes the 7 weights to a temp file and maps each as WirePages.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_layer_from_wire_pages_matches_copy() {
        use crate::inference::quantize_q8_0_blocks;
        use crate::wire_mmap::WirePages;
        use std::fs::File;
        use std::io::Write;
        if !detect_metal_device().available {
            return;
        }
        let hidden = 64usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 64usize;
        let ffn_dim = 64usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = 1.0e-6f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mw = |rows: usize, in_dim: usize, seed: usize| -> Vec<u8> {
            let mut wire = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                for blk in quantize_q8_0_blocks(&row).iter() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for &q in blk.quants.iter() {
                        wire.push(q as u8);
                    }
                }
            }
            wire
        };
        let weights = [
            mw(q_dim, hidden, 1),
            mw(kv_dim, hidden, 5),
            mw(kv_dim, hidden, 9),
            mw(hidden, q_dim, 13),
            mw(ffn_dim, hidden, 17),
            mw(ffn_dim, hidden, 21),
            mw(hidden, ffn_dim, 25),
        ];
        // Concatenate into a temp file; record each tensor's (offset, len).
        let mut blob = Vec::new();
        let mut spans = Vec::new();
        for w in &weights {
            spans.push((blob.len() as u64, w.len()));
            blob.extend_from_slice(w);
        }
        let path = std::env::temp_dir().join(format!("gemma4_wp_test_{}.bin", std::process::id()));
        File::create(&path).unwrap().write_all(&blob).unwrap();
        let file = File::open(&path).unwrap();
        let pages: Vec<_> = spans
            .iter()
            .map(|&(off, len)| WirePages::read_from_file(&file, off, len).unwrap())
            .collect();

        let norms = || {
            (
                (0..hidden)
                    .map(|i| 0.8 + (i as f32 % 5.0) * 0.05)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.9 + (i as f32 % 3.0) * 0.03)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
                    .collect::<Vec<f32>>(),
            )
        };
        let (an, pa, fnw, pf, qn, kn) = norms();
        let copy = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Q8_0,
            an.clone(),
            qn.clone(),
            kn.clone(),
            pa.clone(),
            fnw.clone(),
            pf.clone(),
            &weights[0],
            &weights[1],
            Some(&weights[2]),
            &weights[3],
            &weights[4],
            &weights[5],
            &weights[6],
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("copy");
        let nocopy = Gemma4ResidentLayer::from_wire_pages(
            GemmaWireFmt::Q8_0,
            an,
            qn,
            kn,
            pa,
            fnw,
            pf,
            &pages[0],
            &pages[1],
            Some(&pages[2]),
            &pages[3],
            &pages[4],
            &pages[5],
            &pages[6],
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("nocopy");

        let h_in: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let half = head_dim / 2;
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();
        let cache_len = n_kv_heads * 8 * head_dim;
        let ck: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let cv: Vec<f32> = (0..cache_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        let run = |layer: &Gemma4ResidentLayer| {
            try_gemma4_layer(
                layer, &h_in, &cos_t, &sin_t, &ck, &cv, 8, 3, 4, 0, scale, true,
            )
            .expect("layer")
        };
        let a = run(&copy);
        let b = run(&nocopy);
        let _ = std::fs::remove_file(&path);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1.0e-4, "{x} != {y}");
        }
    }

    // The stateful Gemma4ResidentModel::forward_token (resident weights + token_embd +
    // persistent caches) must match the stateless try_gemma4_forward for one token at
    // position 0 — same layers/PLE/head, so the only difference is statefulness.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_resident_model_forward_token_matches_stateless() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let ple_dim = 64usize;
        let vocab = 160usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mw = |rows: usize, in_dim: usize, seed: usize| -> Vec<u8> {
            let mut wire = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                for blk in quantize_q8_0_blocks(&row).iter() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for &q in blk.quants.iter() {
                        wire.push(q as u8);
                    }
                }
            }
            wire
        };
        let mk_norms = |off: usize| {
            (
                (0..hidden)
                    .map(|i| 0.8 + ((i + off) as f32 % 5.0) * 0.05)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.9 + ((i + off) as f32 % 3.0) * 0.03)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.85 + ((i + off) as f32 % 4.0) * 0.04)
                    .collect::<Vec<f32>>(),
                (0..hidden)
                    .map(|i| 0.95 + ((i + off) as f32 % 6.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.7 + ((i + off) as f32 % 11.0) * 0.02)
                    .collect::<Vec<f32>>(),
                (0..head_dim)
                    .map(|i| 0.6 + ((i + off) as f32 % 7.0) * 0.03)
                    .collect::<Vec<f32>>(),
            )
        };
        let build_layer = |off: usize| {
            let (an, pa, fnw, pf, qn, kn) = mk_norms(off);
            Gemma4ResidentLayer::from_wire(
                GemmaWireFmt::Q8_0,
                an,
                qn,
                kn,
                pa,
                fnw,
                pf,
                &mw(q_dim, hidden, off + 1),
                &mw(kv_dim, hidden, off + 5),
                Some(&mw(kv_dim, hidden, off + 9)),
                &mw(hidden, q_dim, off + 13),
                &mw(ffn_dim, hidden, off + 17),
                &mw(ffn_dim, hidden, off + 21),
                &mw(hidden, ffn_dim, off + 25),
                n_heads,
                n_kv_heads,
                head_dim,
                ffn_dim,
                eps,
            )
            .expect("layer")
        };
        let mk_ple = |off: usize| Gemma4ResidentPle {
            inp_gate: (0..ple_dim * hidden)
                .map(|n| (((n + off) % 29) as f32 - 14.0) * 0.02)
                .collect(),
            proj: (0..hidden * ple_dim)
                .map(|n| (((n + off) % 23) as f32 - 11.0) * 0.03)
                .collect(),
            post_norm: (0..hidden)
                .map(|i| 0.9 + ((i + off) as f32 % 5.0) * 0.04)
                .collect(),
            output_scale: 0.061,
        };
        let mk_input = |off: usize| Gemma4TokenLayerInput {
            cos_t: (0..half)
                .map(|i| (0.3 + (i + off) as f32 * 0.01).cos())
                .collect(),
            sin_t: (0..half)
                .map(|i| (0.3 + (i + off) as f32 * 0.01).sin())
                .collect(),
            pli: (0..ple_dim)
                .map(|i| ((i + off) as f32 % 9.0 - 4.0) * 0.2)
                .collect(),
            window_start: 0,
        };
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let embd_wire = mw(vocab, hidden, 99);
        let cache_len = n_kv_heads * max_positions * head_dim;
        let zeros = vec![0.0f32; cache_len];

        // Stateless oracle (validated): position 0, filled 1, fresh cache.
        // PLE is exercised separately (metal_gemma4_ple/_pli); here both paths run
        // PLE-free so forward_token (GPU pli) and try_gemma4_forward stay comparable.
        let _ = mk_ple(0);
        let want = try_gemma4_forward(
            &[build_layer(0), build_layer(50)],
            &[None, None],
            &[true, true],
            &[0, 1],
            &[mk_input(0), mk_input(4)],
            &[Some(zeros.clone()), Some(zeros.clone())],
            &[Some(zeros.clone()), Some(zeros.clone())],
            &h0,
            &output_norm,
            &embd_wire,
            vocab,
            softcap,
            eps,
            max_positions,
            0,
            1,
            scale,
        )
        .expect("stateless");

        // Stateful model (fresh layers from the same bytes), token at position 0.
        let model = Gemma4ResidentModel::new(
            vec![build_layer(0), build_layer(50)],
            vec![None, None],
            vec![1.0, 1.0],
            vec![true, true],
            vec![0, 1],
            &embd_wire,
            output_norm.clone(),
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            scale,
        )
        .expect("model");
        let got = model
            .forward_token(&h0, &[mk_input(0), mk_input(4)], &[], 0)
            .expect("forward_token");

        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 5.0e-3, "{a} != {b}");
        }
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(amax(&got), amax(&want));
    }

    // QAT integration gate (visible): a Gemma4ResidentModel built with Q4_0 projection
    // weights must decode a token on the GPU matching the CPU f32×dequant reference of
    // the SAME model. Exercises the resident path end-to-end with GemmaWireFmt::Q4_0
    // routed through encode_gemma4_layer (attention + FFN Q4_0 GEMV) plus the Q8 head.
    // One owning V-ful layer, no PLE, position 0 (fresh cache). The supported dense
    // rows are untouched (they pass Q8_0); this proves the Q4_0 layer routing is real.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_resident_q4_forward_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::{gemma4::gelu_tanh, q4_0_wire_block_dequant, quantize_q8_0_blocks};
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let vocab = 160usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q4_0 projection weight -> (wire bytes, dequantized rows). The CPU reference
        // dots the SAME values the GPU's f32×dequant Q4_0 GEMV reads.
        let mw4 = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let mut drow = vec![0.0f32; in_dim];
                for b in 0..(in_dim / 32) {
                    let sc = (((r + b + seed) % 7) as f32 + 1.0) * 0.015;
                    let mut blk = [0u8; 18];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(sc).to_le_bytes());
                    for (j, slot) in blk[2..].iter_mut().enumerate() {
                        *slot = ((r * 13 + b * 7 + j * 5 + seed) % 256) as u8;
                    }
                    let d = q4_0_wire_block_dequant(&blk);
                    for (i, &val) in d.iter().enumerate() {
                        drow[b * 32 + i] = val;
                    }
                    wire.extend_from_slice(&blk);
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = mw4(q_dim, hidden, 1);
        let (k_wire, k_deq) = mw4(kv_dim, hidden, 5);
        let (v_wire, v_deq) = mw4(kv_dim, hidden, 9);
        let (o_wire, o_deq) = mw4(hidden, q_dim, 13);
        let (gate_wire, gate_deq) = mw4(ffn_dim, hidden, 17);
        let (up_wire, up_deq) = mw4(ffn_dim, hidden, 21);
        let (down_wire, down_deq) = mw4(hidden, ffn_dim, 25);

        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_attn_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let ffn_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
            .collect();
        let post_ffw_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
            .collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();

        // Q8 vocab-major embedding table (the head is unchanged Q8) -> wire + dequant.
        let mut embd_wire = Vec::new();
        let mut embd_deq = Vec::new();
        for v in 0..vocab {
            let row: Vec<f32> = (0..hidden)
                .map(|i| ((((v * hidden + i) % 31) as f32) - 15.0) * 0.05)
                .collect();
            let mut drow = vec![0.0f32; hidden];
            for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                embd_wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                for (j, &qq) in blk.quants.iter().enumerate() {
                    embd_wire.push(qq as u8);
                    drow[b * 32 + j] = blk.scale * qq as f32;
                }
            }
            embd_deq.push(drow);
        }

        // ---- CPU reference over the dequantized weights ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        // Attention at position 0 (fresh cache): only the current token's K/V, so the
        // single-element softmax weight is 1 and ctx is just V.
        let normf = rms(&h0, Some(&attn_norm));
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let _kk = per_head(&matmul(&k_deq, &normf), n_kv_heads, Some(&k_norm));
        let vv = per_head(&matmul(&v_deq, &normf), n_kv_heads, None);
        rope(&mut q, n_heads);
        let group = n_heads / n_kv_heads;
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            for d in 0..head_dim {
                ctx[h * head_dim + d] = vv[kvh * head_dim + d];
            }
        }
        let attn_out = matmul(&o_deq, &ctx);
        let mid: Vec<f32> = h0
            .iter()
            .zip(rms(&attn_out, Some(&post_attn_norm)))
            .map(|(a, b)| a + b)
            .collect();
        // FFN.
        let normf2 = rms(&mid, Some(&ffn_norm));
        let gate = matmul(&gate_deq, &normf2);
        let up = matmul(&up_deq, &normf2);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * u)
            .collect();
        let down = matmul(&down_deq, &act);
        let h_final: Vec<f32> = mid
            .iter()
            .zip(rms(&down, Some(&post_ffw_norm)))
            .map(|(a, b)| a + b)
            .collect();
        // Head: rms -> logits -> soft_cap.
        let normh = rms(&h_final, Some(&output_norm));
        let want: Vec<f32> = embd_deq
            .iter()
            .map(|row| {
                let logit: f32 = row.iter().zip(&normh).map(|(a, b)| a * b).sum();
                softcap * (logit / softcap).tanh()
            })
            .collect();

        // ---- GPU resident model (Q4_0 layer) ----
        let layer = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Q4_0,
            attn_norm,
            q_norm,
            k_norm,
            post_attn_norm,
            ffn_norm,
            post_ffw_norm,
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &gate_wire,
            &up_wire,
            &down_wire,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("q4 layer");
        let model = Gemma4ResidentModel::new(
            vec![layer],
            vec![None],
            vec![1.0],
            vec![true],
            vec![0],
            &embd_wire,
            output_norm,
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            scale,
        )
        .expect("q4 model");
        let input = Gemma4TokenLayerInput {
            cos_t,
            sin_t,
            pli: Vec::new(),
            window_start: 0,
        };
        let got = model
            .forward_token(&h0, &[input], &[], 0)
            .expect("forward_token q4");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(amax(&got), amax(&want), "greedy argmax must agree");
    }

    // GABBRO M3-followup: the SAME full resident forward (attention + FFN + head) with
    // NVFP4 layer projections must match the CPU dequant oracle — proving the NVFP4
    // GEMV runs correctly IN CONTEXT (not just the standalone gate), and that the
    // format-aware blocks_per_row (block_elements()=64) threads through encode_gemma4_ffn
    // /encode_gemma4_attention. Built directly (no Gemma4GpuRuntime::load), so it needs
    // no model file and does not touch the load-path admission gate.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_resident_nvfp4_forward_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::{gemma4::gelu_tanh, nvfp4_wire_block_dequant, quantize_q8_0_blocks};
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let vocab = 160usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // NVFP4 projection weight -> (36-byte superblocks, dequantized rows). The CPU
        // reference dots the SAME values the GPU's f32×decode NVFP4 GEMV reads.
        let mw_nvfp4 = |rows: usize, in_dim: usize, seed: usize| -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut wire = Vec::new();
            let mut deq = Vec::new();
            for r in 0..rows {
                let mut drow = vec![0.0f32; in_dim];
                for b in 0..(in_dim / 64) {
                    let mut blk = [0u8; 36];
                    // 4 UE4M3 sub-block scales in [8,72): non-zero, never 0x7F/0xFF sentinels.
                    for (s, slot) in blk[0..4].iter_mut().enumerate() {
                        *slot = (((r * 5 + b * 3 + s + seed) % 64) + 8) as u8;
                    }
                    // 32 packed E2M1 nibble bytes.
                    for (j, slot) in blk[4..].iter_mut().enumerate() {
                        *slot = ((r * 13 + b * 7 + j * 5 + seed) % 256) as u8;
                    }
                    let d = nvfp4_wire_block_dequant(&blk);
                    for (i, &val) in d.iter().enumerate() {
                        drow[b * 64 + i] = val;
                    }
                    wire.extend_from_slice(&blk);
                }
                deq.push(drow);
            }
            (wire, deq)
        };
        let (q_wire, q_deq) = mw_nvfp4(q_dim, hidden, 1);
        let (k_wire, k_deq) = mw_nvfp4(kv_dim, hidden, 5);
        let (v_wire, v_deq) = mw_nvfp4(kv_dim, hidden, 9);
        let (o_wire, o_deq) = mw_nvfp4(hidden, q_dim, 13);
        let (gate_wire, gate_deq) = mw_nvfp4(ffn_dim, hidden, 17);
        let (up_wire, up_deq) = mw_nvfp4(ffn_dim, hidden, 21);
        let (down_wire, down_deq) = mw_nvfp4(hidden, ffn_dim, 25);

        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let post_attn_norm: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let ffn_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
            .collect();
        let post_ffw_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
            .collect();
        let q_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let k_norm: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect();

        // Q8 vocab-major embedding table (the head is unchanged Q8) -> wire + dequant.
        let mut embd_wire = Vec::new();
        let mut embd_deq = Vec::new();
        for v in 0..vocab {
            let row: Vec<f32> = (0..hidden)
                .map(|i| ((((v * hidden + i) % 31) as f32) - 15.0) * 0.05)
                .collect();
            let mut drow = vec![0.0f32; hidden];
            for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                embd_wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                for (j, &qq) in blk.quants.iter().enumerate() {
                    embd_wire.push(qq as u8);
                    drow[b * 32 + j] = blk.scale * qq as f32;
                }
            }
            embd_deq.push(drow);
        }

        // ---- CPU reference over the dequantized weights ----
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = (mss + eps).powf(-0.5);
            (0..x.len())
                .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
                .collect()
        };
        let matmul = |deq: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
            deq.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let per_head = |x: &[f32], heads: usize, w: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len()];
            for h in 0..heads {
                let n = rms(&x[h * head_dim..(h + 1) * head_dim], w);
                out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
            }
            out
        };
        let rope = |x: &mut [f32], heads: usize| {
            for h in 0..heads {
                let base = h * head_dim;
                for i in 0..half {
                    let (x0, x1) = (x[base + i], x[base + half + i]);
                    x[base + i] = x0 * cos_t[i] - x1 * sin_t[i];
                    x[base + half + i] = x0 * sin_t[i] + x1 * cos_t[i];
                }
            }
        };
        let normf = rms(&h0, Some(&attn_norm));
        let mut q = per_head(&matmul(&q_deq, &normf), n_heads, Some(&q_norm));
        let _kk = per_head(&matmul(&k_deq, &normf), n_kv_heads, Some(&k_norm));
        let vv = per_head(&matmul(&v_deq, &normf), n_kv_heads, None);
        rope(&mut q, n_heads);
        let group = n_heads / n_kv_heads;
        let mut ctx = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            for d in 0..head_dim {
                ctx[h * head_dim + d] = vv[kvh * head_dim + d];
            }
        }
        let attn_out = matmul(&o_deq, &ctx);
        let mid: Vec<f32> = h0
            .iter()
            .zip(rms(&attn_out, Some(&post_attn_norm)))
            .map(|(a, b)| a + b)
            .collect();
        let normf2 = rms(&mid, Some(&ffn_norm));
        let gate = matmul(&gate_deq, &normf2);
        let up = matmul(&up_deq, &normf2);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * u)
            .collect();
        let down = matmul(&down_deq, &act);
        let h_final: Vec<f32> = mid
            .iter()
            .zip(rms(&down, Some(&post_ffw_norm)))
            .map(|(a, b)| a + b)
            .collect();
        let normh = rms(&h_final, Some(&output_norm));
        let want: Vec<f32> = embd_deq
            .iter()
            .map(|row| {
                let logit: f32 = row.iter().zip(&normh).map(|(a, b)| a * b).sum();
                softcap * (logit / softcap).tanh()
            })
            .collect();

        // ---- GPU resident model (NVFP4 layer projections, Q8 head) ----
        let layer = Gemma4ResidentLayer::from_wire(
            GemmaWireFmt::Nvfp4,
            attn_norm,
            q_norm,
            k_norm,
            post_attn_norm,
            ffn_norm,
            post_ffw_norm,
            &q_wire,
            &k_wire,
            Some(&v_wire),
            &o_wire,
            &gate_wire,
            &up_wire,
            &down_wire,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            eps,
        )
        .expect("nvfp4 layer");
        let model = Gemma4ResidentModel::new(
            vec![layer],
            vec![None],
            vec![1.0],
            vec![true],
            vec![0],
            &embd_wire,
            output_norm,
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            scale,
        )
        .expect("nvfp4 model");
        let input = Gemma4TokenLayerInput {
            cos_t,
            sin_t,
            pli: Vec::new(),
            window_start: 0,
        };
        let got = model
            .forward_token(&h0, &[input], &[], 0)
            .expect("forward_token nvfp4");
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 2.0e-2, "{a} != {b}");
        }
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(amax(&got), amax(&want), "greedy argmax must agree");
    }

    // The QAT hybrid split point: forward_token_hidden (GPU layers, no head) followed by
    // the CPU head (rms_norm -> token_embd matvec -> soft_cap) must reproduce exactly
    // what forward_token computes with the head encoded on the GPU. Proves the runtime's
    // "GPU layers + CPU Q6_K head" lane is byte-faithful to the all-GPU path; here the
    // head is Q8 so the same dequantized table drives both sides.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gemma4_forward_token_hidden_plus_cpu_head_matches_forward_token() {
        if !detect_metal_device().available {
            return;
        }
        use crate::inference::quantize_q8_0_blocks;
        let hidden = 128usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 256usize;
        let ffn_dim = 256usize;
        let vocab = 160usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let half = head_dim / 2;
        let max_positions = 8usize;
        let eps = 1.0e-6f32;
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mw = |rows: usize, in_dim: usize, seed: usize| -> Vec<u8> {
            let mut wire = Vec::new();
            for r in 0..rows {
                let row: Vec<f32> = (0..in_dim)
                    .map(|i| ((((r * in_dim + i + seed) % 29) as f32) - 14.0) * 0.03)
                    .collect();
                for blk in quantize_q8_0_blocks(&row).iter() {
                    wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                    for &q in blk.quants.iter() {
                        wire.push(q as u8);
                    }
                }
            }
            wire
        };
        let an: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32 % 5.0) * 0.05).collect();
        let pa: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 % 3.0) * 0.03).collect();
        let fnw: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i as f32 % 4.0) * 0.04)
            .collect();
        let pf: Vec<f32> = (0..hidden)
            .map(|i| 0.95 + (i as f32 % 6.0) * 0.02)
            .collect();
        let qn: Vec<f32> = (0..head_dim)
            .map(|i| 0.7 + (i as f32 % 11.0) * 0.02)
            .collect();
        let kn: Vec<f32> = (0..head_dim)
            .map(|i| 0.6 + (i as f32 % 7.0) * 0.03)
            .collect();
        let output_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.88 + (i as f32 % 5.0) * 0.03)
            .collect();
        let h0: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();

        // Q8 vocab-major head table -> wire + dequant (drives the CPU head reference).
        let mut embd_wire = Vec::new();
        let mut embd_deq = Vec::new();
        for v in 0..vocab {
            let row: Vec<f32> = (0..hidden)
                .map(|i| ((((v * hidden + i) % 31) as f32) - 15.0) * 0.05)
                .collect();
            let mut drow = vec![0.0f32; hidden];
            for (b, blk) in quantize_q8_0_blocks(&row).iter().enumerate() {
                embd_wire.extend_from_slice(&f32_to_f16_bits(blk.scale).to_le_bytes());
                for (j, &qq) in blk.quants.iter().enumerate() {
                    embd_wire.push(qq as u8);
                    drow[b * 32 + j] = blk.scale * qq as f32;
                }
            }
            embd_deq.push(drow);
        }

        let build = || {
            let layer = Gemma4ResidentLayer::from_wire(
                GemmaWireFmt::Q8_0,
                an.clone(),
                qn.clone(),
                kn.clone(),
                pa.clone(),
                fnw.clone(),
                pf.clone(),
                &mw(q_dim, hidden, 1),
                &mw(kv_dim, hidden, 5),
                Some(&mw(kv_dim, hidden, 9)),
                &mw(hidden, q_dim, 13),
                &mw(ffn_dim, hidden, 17),
                &mw(ffn_dim, hidden, 21),
                &mw(hidden, ffn_dim, 25),
                n_heads,
                n_kv_heads,
                head_dim,
                ffn_dim,
                eps,
            )
            .expect("layer");
            Gemma4ResidentModel::new(
                vec![layer],
                vec![None],
                vec![1.0],
                vec![true],
                vec![0],
                &embd_wire,
                output_norm.clone(),
                hidden,
                vocab,
                softcap,
                eps,
                max_positions,
                scale,
            )
            .expect("model")
        };
        let input = || Gemma4TokenLayerInput {
            cos_t: (0..half).map(|i| (0.3 + i as f32 * 0.01).cos()).collect(),
            sin_t: (0..half).map(|i| (0.3 + i as f32 * 0.01).sin()).collect(),
            pli: Vec::new(),
            window_start: 0,
        };

        let gpu_head = build();
        let logits_gpu = gpu_head
            .forward_token(&h0, &[input()], &[], 0)
            .expect("forward_token");
        let cpu_head = build();
        let hidden_only = cpu_head
            .forward_token_hidden(&h0, &[input()], &[], 0)
            .expect("forward_token_hidden");

        // CPU head over the returned hidden state.
        let mss = hidden_only.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mss + eps).powf(-0.5);
        let last: Vec<f32> = (0..hidden)
            .map(|i| hidden_only[i] * inv * output_norm[i])
            .collect();
        let logits_cpu: Vec<f32> = embd_deq
            .iter()
            .map(|row| {
                let logit: f32 = row.iter().zip(&last).map(|(a, b)| a * b).sum();
                softcap * (logit / softcap).tanh()
            })
            .collect();

        for (a, b) in logits_gpu.iter().zip(&logits_cpu) {
            assert!((a - b).abs() < 5.0e-3, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_block_resident_matches_standalone() {
        if !detect_metal_device().available {
            return;
        }
        let n_heads = 2usize;
        let n_kv = 2usize;
        let head_dim = 32usize;
        let hidden = 64usize;
        let position_count = 3usize;
        let pos = position_count - 1;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.2)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32 % 3.0) * 0.1).collect();
        let cos_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).sin()).collect();
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        let q_w = mkw(q_dim, bpr_hidden, 1);
        let k_w = mkw(kv_dim, bpr_hidden, 2);
        let v_w = mkw(kv_dim, bpr_hidden, 3);
        let o_w = mkw(hidden, bpr_q, 4);
        let cache_k: Vec<f32> = (0..kv_dim * position_count)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let cache_v: Vec<f32> = (0..kv_dim * position_count)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.15)
            .collect();

        // Reference: standalone (CPU-verified) kernels in sequence.
        let norm = try_rms_norm_f32(&input, &attn_norm, eps).unwrap();
        let (sn, qn) = try_quantize_q8_0_f32(&norm).unwrap();
        let mut query = vec![0.0f32; q_dim];
        assert!(try_q8_0_block_linear_row(
            &sn, &qn, &q_w, q_dim, bpr_hidden, &mut query
        ));
        let mut key = vec![0.0f32; kv_dim];
        assert!(try_q8_0_block_linear_row(
            &sn, &qn, &k_w, kv_dim, bpr_hidden, &mut key
        ));
        let mut val = vec![0.0f32; kv_dim];
        assert!(try_q8_0_block_linear_row(
            &sn, &qn, &v_w, kv_dim, bpr_hidden, &mut val
        ));
        let query_r =
            try_rope_rotate_f32(&query, &cos_t, &sin_t, n_heads, head_dim, half, false).unwrap();
        let key_r = try_rope_rotate_f32(&key, &cos_t, &sin_t, n_kv, head_dim, half, false).unwrap();
        let mut ref_k = cache_k.clone();
        let mut ref_v = cache_v.clone();
        for h in 0..n_kv {
            let dst = (h * position_count + pos) * head_dim;
            ref_k[dst..dst + head_dim].copy_from_slice(&key_r[h * head_dim..(h + 1) * head_dim]);
            ref_v[dst..dst + head_dim].copy_from_slice(&val[h * head_dim..(h + 1) * head_dim]);
        }
        let ctx = try_attention_decode_f32(
            &query_r,
            &ref_k,
            &ref_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            scale,
        )
        .unwrap();
        let (sc, qc) = try_quantize_q8_0_f32(&ctx).unwrap();
        let mut o = vec![0.0f32; hidden];
        assert!(try_q8_0_block_linear_row(
            &sc, &qc, &o_w, hidden, bpr_q, &mut o
        ));
        let expected = try_residual_add_f32(&input, &o).unwrap();

        let got = try_attention_block_resident(
            &input,
            &attn_norm,
            eps,
            &q_w,
            &k_w,
            &v_w,
            &o_w,
            &cos_t,
            &sin_t,
            &cache_k,
            &cache_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            scale,
            false,
        )
        .unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ffn_block_resident_matches_standalone() {
        if !detect_metal_device().available {
            return;
        }
        let hidden = 64usize;
        let ffn = 128usize;
        let eps = 1.0e-5f32;
        let bpr_hidden = hidden / 32;
        let bpr_ffn = ffn / 32;
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.2)
            .collect();
        let ffn_norm: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32 % 3.0) * 0.1).collect();
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let scale = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&scale.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        let gate_w = mkw(ffn, bpr_hidden, 1);
        let up_w = mkw(ffn, bpr_hidden, 2);
        let down_w = mkw(hidden, bpr_ffn, 3);

        // Reference: the standalone (CPU-verified) kernels run in sequence.
        let norm = try_rms_norm_f32(&input, &ffn_norm, eps).unwrap();
        let (s1, q1) = try_quantize_q8_0_f32(&norm).unwrap();
        let mut gate = vec![0.0f32; ffn];
        assert!(try_q8_0_block_linear_row(
            &s1, &q1, &gate_w, ffn, bpr_hidden, &mut gate
        ));
        let mut up = vec![0.0f32; ffn];
        assert!(try_q8_0_block_linear_row(
            &s1, &q1, &up_w, ffn, bpr_hidden, &mut up
        ));
        let act = try_silu_mul_f32(&gate, &up).unwrap();
        let (s2, q2) = try_quantize_q8_0_f32(&act).unwrap();
        let mut down = vec![0.0f32; hidden];
        assert!(try_q8_0_block_linear_row(
            &s2, &q2, &down_w, hidden, bpr_ffn, &mut down
        ));
        let expected = try_residual_add_f32(&input, &down).unwrap();

        let got =
            try_ffn_block_resident(&input, &ffn_norm, eps, &gate_w, &up_w, &down_w, ffn).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-3, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_layer_resident_matches_blocks() {
        if !detect_metal_device().available {
            return;
        }
        // The fused decode layer (attention block + FFN block in ONE command buffer, with the
        // attention output staying GPU-resident and feeding the FFN block) must equal running
        // the two standalone resident blocks separately. Same kernels + same inputs, so this
        // should be bit-identical; it guards the cross-block buffer handoff and lifetimes.
        let n_heads = 2usize;
        let n_kv = 2usize;
        let head_dim = 32usize;
        let hidden = 64usize;
        let ffn = 128usize;
        let position_count = 3usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = ffn / 32;
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.2)
            .collect();
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32 % 3.0) * 0.1).collect();
        let ffn_norm: Vec<f32> = (0..hidden).map(|i| 0.4 + (i as f32 % 5.0) * 0.07).collect();
        let cos_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).sin()).collect();
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        let q_w = mkw(q_dim, bpr_hidden, 1);
        let k_w = mkw(kv_dim, bpr_hidden, 2);
        let v_w = mkw(kv_dim, bpr_hidden, 3);
        let o_w = mkw(hidden, bpr_q, 4);
        let gate_w = mkw(ffn, bpr_hidden, 5);
        let up_w = mkw(ffn, bpr_hidden, 6);
        let down_w = mkw(hidden, bpr_ffn, 7);
        let cache_k: Vec<f32> = (0..kv_dim * position_count)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let cache_v: Vec<f32> = (0..kv_dim * position_count)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.15)
            .collect();

        // Reference: the two standalone resident blocks run separately, attention -> FFN.
        let attn_out = try_attention_block_resident(
            &input,
            &attn_norm,
            eps,
            &q_w,
            &k_w,
            &v_w,
            &o_w,
            &cos_t,
            &sin_t,
            &cache_k,
            &cache_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            scale,
            false,
        )
        .unwrap();
        let expected =
            try_ffn_block_resident(&attn_out, &ffn_norm, eps, &gate_w, &up_w, &down_w, ffn)
                .unwrap();

        let got = try_decode_layer_resident(
            &input,
            &attn_norm,
            &ffn_norm,
            eps,
            &q_w,
            &k_w,
            &v_w,
            &o_w,
            &gate_w,
            &up_w,
            &down_w,
            &cos_t,
            &sin_t,
            &cache_k,
            &cache_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            ffn,
            scale,
            false,
        )
        .unwrap();
        assert_eq!(got.len(), hidden);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_forward_resident_matches_per_layer() {
        if !detect_metal_device().available {
            return;
        }
        // Running all layers in ONE command buffer must equal feeding each layer's output
        // into the next via the single-layer fused path. Same kernels + inputs, so it should
        // be bit-identical; this guards the cross-LAYER ping-pong buffering and the
        // single-commit lifetime over many encoders.
        let n_heads = 2usize;
        let n_kv = 2usize;
        let head_dim = 32usize;
        let hidden = 64usize;
        let ffn = 128usize;
        let position_count = 3usize;
        let n_layers = 3usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = ffn / 32;
        let embedding: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.2)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|p| (0.2 + p as f32 * 0.1).sin()).collect();
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };

        struct LayerData {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            q: Vec<u8>,
            k: Vec<u8>,
            v: Vec<u8>,
            o: Vec<u8>,
            gate: Vec<u8>,
            up: Vec<u8>,
            down: Vec<u8>,
            cache_k: Vec<f32>,
            cache_v: Vec<f32>,
        }
        // Seeds vary by layer so each layer has distinct weights/norms/cache.
        let data: Vec<LayerData> = (0..n_layers)
            .map(|li| {
                let s = li * 100;
                LayerData {
                    attn_norm: (0..hidden)
                        .map(|i| 0.5 + ((i + li) as f32 % 3.0) * 0.1)
                        .collect(),
                    ffn_norm: (0..hidden)
                        .map(|i| 0.4 + ((i + li) as f32 % 5.0) * 0.07)
                        .collect(),
                    q: mkw(q_dim, bpr_hidden, s + 1),
                    k: mkw(kv_dim, bpr_hidden, s + 2),
                    v: mkw(kv_dim, bpr_hidden, s + 3),
                    o: mkw(hidden, bpr_q, s + 4),
                    gate: mkw(ffn, bpr_hidden, s + 5),
                    up: mkw(ffn, bpr_hidden, s + 6),
                    down: mkw(hidden, bpr_ffn, s + 7),
                    cache_k: (0..kv_dim * position_count)
                        .map(|i| (((i + li) as f32 % 13.0) - 6.0) * 0.1)
                        .collect(),
                    cache_v: (0..kv_dim * position_count)
                        .map(|i| (((i + li) as f32 % 7.0) - 3.0) * 0.15)
                        .collect(),
                }
            })
            .collect();

        // Reference: loop the single-layer fused path, feeding each output into the next.
        let mut hidden_state = embedding.clone();
        for d in &data {
            hidden_state = try_decode_layer_resident(
                &hidden_state,
                &d.attn_norm,
                &d.ffn_norm,
                eps,
                &d.q,
                &d.k,
                &d.v,
                &d.o,
                &d.gate,
                &d.up,
                &d.down,
                &cos_t,
                &sin_t,
                &d.cache_k,
                &d.cache_v,
                n_heads,
                n_kv,
                head_dim,
                position_count,
                ffn,
                scale,
                false,
            )
            .unwrap();
        }
        let expected = hidden_state;

        let layers: Vec<ResidentDecodeLayer> = data
            .iter()
            .map(|d| ResidentDecodeLayer {
                attn_norm: &d.attn_norm,
                ffn_norm: &d.ffn_norm,
                q_weight_blocks: &d.q,
                k_weight_blocks: &d.k,
                v_weight_blocks: &d.v,
                o_weight_blocks: &d.o,
                gate_weight_blocks: &d.gate,
                up_weight_blocks: &d.up,
                down_weight_blocks: &d.down,
                cache_k: &d.cache_k,
                cache_v: &d.cache_v,
            })
            .collect();
        let got = try_decode_forward_resident(
            &embedding,
            &layers,
            &cos_t,
            &sin_t,
            eps,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            ffn,
            scale,
            false,
        )
        .unwrap();
        assert_eq!(got.len(), hidden);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        }

        // Upload-once: after that decode every layer's Q8 weights are resident in the
        // process-global cache (keyed by pointer+len), so subsequent tokens reuse the same
        // on-GPU buffers rather than re-uploading. Checked on this test's own pointers, so
        // it is race-free under parallel test execution.
        {
            let cache = metal_linear_cache().lock().unwrap();
            let resident = |b: &[u8]| {
                cache
                    .q8_block_weight_buffers
                    .contains_key(&(b.as_ptr() as usize, b.len()))
            };
            for d in &data {
                assert!(resident(&d.q) && resident(&d.k) && resident(&d.v) && resident(&d.o));
                assert!(resident(&d.gate) && resident(&d.up) && resident(&d.down));
            }
        }
        // A second decode reusing the same weight slices (now cache hits) must be identical.
        let got2 = try_decode_forward_resident(
            &embedding,
            &layers,
            &cos_t,
            &sin_t,
            eps,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            ffn,
            scale,
            false,
        )
        .unwrap();
        assert_eq!(got, got2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_decode_strided_matches_contiguous() {
        if !detect_metal_device().available {
            return;
        }
        // Reading a per-layer slice of an interleaved [position][layer][kv_head][head_dim]
        // cache via explicit strides must match the contiguous [kv_head][position][head_dim]
        // result for the same logical K/V. Surrounding layers are filled with noise, so a
        // wrong stride or base offset would read the wrong layer and corrupt the output.
        let n_heads = 4usize;
        let n_kv = 2usize;
        let head_dim = 8usize;
        let position_count = 5usize;
        let layer_count = 3usize;
        let target_layer = 1usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;

        let query: Vec<f32> = (0..q_dim).map(|i| ((i as f32 % 7.0) - 3.0) * 0.2).collect();
        let logical = |kvh: usize, p: usize, d: usize, base: f32| {
            (((kvh * 131 + p * 17 + d) as f32 % 19.0) - 9.0) * 0.1 + base
        };

        // Contiguous [kv_head][position][head_dim] reference.
        let mut contig_k = vec![0.0f32; kv_dim * position_count];
        let mut contig_v = vec![0.0f32; kv_dim * position_count];
        for kvh in 0..n_kv {
            for p in 0..position_count {
                for d in 0..head_dim {
                    let idx = (kvh * position_count + p) * head_dim + d;
                    contig_k[idx] = logical(kvh, p, d, 0.0);
                    contig_v[idx] = logical(kvh, p, d, 0.5);
                }
            }
        }

        // Interleaved [position][layer][kv_head][head_dim]; only `target_layer` holds the
        // logical values, the other layers hold noise.
        let mut inter_k = vec![0.0f32; position_count * layer_count * kv_dim];
        let mut inter_v = vec![0.0f32; position_count * layer_count * kv_dim];
        for p in 0..position_count {
            for l in 0..layer_count {
                for kvh in 0..n_kv {
                    for d in 0..head_dim {
                        let idx = ((p * layer_count + l) * n_kv + kvh) * head_dim + d;
                        if l == target_layer {
                            inter_k[idx] = logical(kvh, p, d, 0.0);
                            inter_v[idx] = logical(kvh, p, d, 0.5);
                        } else {
                            inter_k[idx] = 7.0 + idx as f32;
                            inter_v[idx] = -3.0 - idx as f32;
                        }
                    }
                }
            }
        }

        let expected = try_attention_decode_f32(
            &query,
            &contig_k,
            &contig_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            scale,
        )
        .unwrap();
        let got = try_attention_decode_strided_f32(
            &query,
            &inter_k,
            &inter_v,
            n_heads,
            n_kv,
            head_dim,
            position_count,
            scale,
            layer_count * n_kv * head_dim,  // position_stride
            head_dim,                       // kv_head_stride
            target_layer * n_kv * head_dim, // kv_base_offset
        )
        .unwrap();
        assert_eq!(got.len(), q_dim);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_resident_decode_state_matches_full_upload() {
        if !detect_metal_device().available {
            return;
        }
        // The persistent on-GPU KV session (append one slot per token, never re-upload) must,
        // at each token, equal the full-upload try_decode_forward_resident fed the same
        // accumulated history. Same kernels + inputs => identical; this guards the persistent
        // cache append/stride bookkeeping across many tokens.
        let n_layers = 2usize;
        let n_heads = 2usize;
        let n_kv = 2usize;
        let head_dim = 32usize;
        let hidden = 64usize;
        let ffn = 128usize;
        let max_positions = 8usize;
        let tokens = 4usize;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = ffn / 32;
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        struct LW {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            q: Vec<u8>,
            k: Vec<u8>,
            v: Vec<u8>,
            o: Vec<u8>,
            gate: Vec<u8>,
            up: Vec<u8>,
            down: Vec<u8>,
        }
        let data: Vec<LW> = (0..n_layers)
            .map(|li| {
                let s = li * 100;
                LW {
                    attn_norm: (0..hidden)
                        .map(|i| 0.5 + ((i + li) as f32 % 3.0) * 0.1)
                        .collect(),
                    ffn_norm: (0..hidden)
                        .map(|i| 0.4 + ((i + li) as f32 % 5.0) * 0.07)
                        .collect(),
                    q: mkw(q_dim, bpr_hidden, s + 1),
                    k: mkw(kv_dim, bpr_hidden, s + 2),
                    v: mkw(kv_dim, bpr_hidden, s + 3),
                    o: mkw(hidden, bpr_q, s + 4),
                    gate: mkw(ffn, bpr_hidden, s + 5),
                    up: mkw(ffn, bpr_hidden, s + 6),
                    down: mkw(hidden, bpr_ffn, s + 7),
                }
            })
            .collect();
        let weights: Vec<ResidentLayerWeights> = data
            .iter()
            .map(|d| ResidentLayerWeights {
                attn_norm: &d.attn_norm,
                ffn_norm: &d.ffn_norm,
                q_weight_blocks: ResidentWeightBytes::Blocks36(&d.q),
                k_weight_blocks: ResidentWeightBytes::Blocks36(&d.k),
                v_weight_blocks: ResidentWeightBytes::Blocks36(&d.v),
                o_weight_blocks: ResidentWeightBytes::Blocks36(&d.o),
                gate_weight_blocks: ResidentWeightBytes::Blocks36(&d.gate),
                up_weight_blocks: ResidentWeightBytes::Blocks36(&d.up),
                down_weight_blocks: ResidentWeightBytes::Blocks36(&d.down),
                q_norm: None,
                k_norm: None,
            })
            .collect();

        let mut session = ResidentDecodeState::new(
            n_layers,
            n_heads,
            n_kv,
            head_dim,
            hidden,
            ffn,
            // Tiny initial capacity with a larger cap so the multi-token loop exercises the
            // on-demand growth (GPU->GPU blit of materialized slots) as well as appends.
            2,
            max_positions,
            eps,
            false,
        )
        .unwrap();

        for t in 0..tokens {
            let emb: Vec<f32> = (0..hidden)
                .map(|i| (((i + t) as f32 % 11.0) - 5.0) * 0.2)
                .collect();
            let cos_t: Vec<f32> = (0..half)
                .map(|p| (0.2 + (p + t) as f32 * 0.1).cos())
                .collect();
            let sin_t: Vec<f32> = (0..half)
                .map(|p| (0.2 + (p + t) as f32 * 0.1).sin())
                .collect();
            let position_count = t + 1;

            // Reference history = the session's accumulated slots 0..t-1, re-laid into a
            // [kv_head][position_count][head_dim] buffer with slot t left for the kernel.
            let ref_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
                .map(|layer| {
                    let hist_k = session.cache_k_contiguous(layer, t);
                    let hist_v = session.cache_v_contiguous(layer, t);
                    let mut ck = vec![0.0f32; kv_dim * position_count];
                    let mut cv = vec![0.0f32; kv_dim * position_count];
                    for h in 0..n_kv {
                        for p in 0..t {
                            let src = (h * t + p) * head_dim;
                            let dst = (h * position_count + p) * head_dim;
                            ck[dst..dst + head_dim].copy_from_slice(&hist_k[src..src + head_dim]);
                            cv[dst..dst + head_dim].copy_from_slice(&hist_v[src..src + head_dim]);
                        }
                    }
                    (ck, cv)
                })
                .collect();
            let ref_layers: Vec<ResidentDecodeLayer> = data
                .iter()
                .zip(&ref_caches)
                .map(|(d, (ck, cv))| ResidentDecodeLayer {
                    attn_norm: &d.attn_norm,
                    ffn_norm: &d.ffn_norm,
                    q_weight_blocks: &d.q,
                    k_weight_blocks: &d.k,
                    v_weight_blocks: &d.v,
                    o_weight_blocks: &d.o,
                    gate_weight_blocks: &d.gate,
                    up_weight_blocks: &d.up,
                    down_weight_blocks: &d.down,
                    cache_k: ck,
                    cache_v: cv,
                })
                .collect();
            let expected = try_decode_forward_resident(
                &emb,
                &ref_layers,
                &cos_t,
                &sin_t,
                eps,
                n_heads,
                n_kv,
                head_dim,
                position_count,
                ffn,
                scale,
                false,
            )
            .unwrap();

            let got = match session
                .forward_token(
                    &emb, &weights, &cos_t, &sin_t, t, scale, None, None, 0, None,
                )
                .unwrap()
            {
                ResidentTokenOut::Data(v) => v,
                ResidentTokenOut::Sampled(_) => panic!("unexpected sampled output"),
            };
            assert_eq!(got.len(), hidden);
            for (a, b) in got.iter().zip(&expected) {
                assert!((a - b).abs() < 1.0e-4, "token {t}: {a} != {b}");
            }
        }
    }

    /// WIN2METAL Phase 3 C2/C3 gate. `verify_batch` over `k` rows at positions
    /// `[base, base+k)` must be BIT-IDENTICAL — exact u32 `to_bits`, not epsilon — to `k`
    /// independent `forward_token` decodes seeded with the same KV history. The straddle
    /// windows cross the split-K routing boundaries: `base=126,k=6` (rows pc 127..132 — the
    /// first row takes the v2 f32-cache path, the rest the split-K f16-mirror path) and
    /// `base=510,k=6` (rows pc 511..516, deep split-K with a varying split count). Logits are
    /// compared by exact bits (an offset/stride bug that doesn't flip the argmax would slip
    /// past an argmax-only check); the argmax ids are asserted too.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_spec_verify_bit_identical() {
        if !detect_metal_device().available {
            return;
        }
        // Production GEMV stack + tiled/split-K attention. Latch the OnceLock gates ON
        // before the first read; if a sibling test already latched f32y/wire/nsg8 OFF (a
        // full --lib run read them earlier), SKIP rather than compare against a different
        // kernel. A targeted run has no such sibling, so the gates latch on here (and ATTN2
        // engages the split-K path the straddle windows are designed to exercise).
        std::env::set_var("CAMELID_METAL_F32Y", "1");
        std::env::set_var("CAMELID_METAL_WIRE", "1");
        std::env::set_var("CAMELID_METAL_WIRE_NSG8", "1");
        std::env::set_var("CAMELID_METAL_ATTN2", "1");
        // Require the split-K attention path (ATTN2 + split-K) too, so the straddle-126 / deep-510
        // windows genuinely exercise the split-K verify kernels instead of silently falling to the
        // v2 path when a sibling latched attn2/split-K OFF first. SKIP loudly otherwise — the
        // targeted run (qa/speed/verify-gates.sh, --test-threads=1) is the enforced proof lane; a
        // green --all-targets does NOT by itself exercise these byte-exact gates.
        if !(f32y_gemv_enabled()
            && wire_weights_enabled()
            && wire_nsg8_enabled()
            && attn2_enabled()
            && splitk_attention_enabled())
        {
            eprintln!(
                "SKIP metal_spec_verify_bit_identical: f32y/wire/nsg8/attn2/split-K gates inactive \
                 (OnceLock latched off earlier in this process); run targeted: \
                 qa/speed/verify-gates.sh (or cargo test --release --lib \
                 metal_spec_verify_bit_identical)"
            );
            return;
        }

        let n_layers = 2usize;
        let n_heads = 4usize;
        let n_kv = 2usize; // group = 2
        let head_dim = 32usize;
        let hidden = 128usize;
        let ffn = 256usize;
        let vocab = 256usize;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = ffn / 32;
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        struct LW {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            q: Vec<u8>,
            k: Vec<u8>,
            v: Vec<u8>,
            o: Vec<u8>,
            gate: Vec<u8>,
            up: Vec<u8>,
            down: Vec<u8>,
        }
        let data: Vec<LW> = (0..n_layers)
            .map(|li| {
                let s = li * 100;
                LW {
                    attn_norm: (0..hidden)
                        .map(|i| 0.5 + ((i + li) as f32 % 3.0) * 0.1)
                        .collect(),
                    ffn_norm: (0..hidden)
                        .map(|i| 0.4 + ((i + li) as f32 % 5.0) * 0.07)
                        .collect(),
                    q: mkw(q_dim, bpr_hidden, s + 1),
                    k: mkw(kv_dim, bpr_hidden, s + 2),
                    v: mkw(kv_dim, bpr_hidden, s + 3),
                    o: mkw(hidden, bpr_q, s + 4),
                    gate: mkw(ffn, bpr_hidden, s + 5),
                    up: mkw(ffn, bpr_hidden, s + 6),
                    down: mkw(hidden, bpr_ffn, s + 7),
                }
            })
            .collect();
        let weights: Vec<ResidentLayerWeights> = data
            .iter()
            .map(|d| ResidentLayerWeights {
                attn_norm: &d.attn_norm,
                ffn_norm: &d.ffn_norm,
                q_weight_blocks: ResidentWeightBytes::Blocks36(&d.q),
                k_weight_blocks: ResidentWeightBytes::Blocks36(&d.k),
                v_weight_blocks: ResidentWeightBytes::Blocks36(&d.v),
                o_weight_blocks: ResidentWeightBytes::Blocks36(&d.o),
                gate_weight_blocks: ResidentWeightBytes::Blocks36(&d.gate),
                up_weight_blocks: ResidentWeightBytes::Blocks36(&d.up),
                down_weight_blocks: ResidentWeightBytes::Blocks36(&d.down),
                q_norm: None,
                k_norm: None,
            })
            .collect();
        // Output projection + final norm (the LogitsStage).
        let final_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.45 + (i as f32 % 4.0) * 0.05)
            .collect();
        let out_w = mkw(vocab, bpr_hidden, 777);
        let make_stage = || LogitsStage {
            final_norm: &final_norm,
            output_weight_blocks: ResidentWeightBytes::Blocks36(&out_w),
            vocab_size: vocab,
        };

        // Deterministic synthetic K/V seeding the `base` history positions (identical for
        // both sessions). Layout per layer: [kv_head][base][head_dim].
        let synth_kv = |base: usize, layer: usize, is_v: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; n_kv * base * head_dim];
            for h in 0..n_kv {
                for p in 0..base {
                    for d in 0..head_dim {
                        let idx = (h * base + p) * head_dim + d;
                        let t = (layer * 7 + h * 13 + p * 3 + d + usize::from(is_v) * 5) as f32;
                        out[idx] = ((t % 19.0) - 9.0) * 0.05;
                    }
                }
            }
            out
        };

        let run = |base: usize, k: usize| {
            let cap = base + k + 16;
            let max_positions = cap; // no growth: identical strides in both sessions
            let mut cos_all = vec![0.0f32; k * half];
            let mut sin_all = vec![0.0f32; k * half];
            let mut emb_all = vec![0.0f32; k * hidden];
            for i in 0..k {
                for p in 0..half {
                    let theta = 0.2 + ((base + i) as f32) * 0.05 + p as f32 * 0.1;
                    cos_all[i * half + p] = theta.cos();
                    sin_all[i * half + p] = theta.sin();
                }
                for j in 0..hidden {
                    emb_all[i * hidden + j] = (((i + j + base) as f32 % 11.0) - 5.0) * 0.2;
                }
            }

            let mk_session = || {
                let mut s = ResidentDecodeState::new(
                    n_layers,
                    n_heads,
                    n_kv,
                    head_dim,
                    hidden,
                    ffn,
                    max_positions,
                    cap,
                    eps,
                    false,
                )
                .unwrap();
                for layer in 0..n_layers {
                    let ck = synth_kv(base, layer, false);
                    let cv = synth_kv(base, layer, true);
                    assert!(s.seed_layer(layer, &ck, &cv, base), "seed_layer");
                }
                s.set_filled(base);
                s
            };

            // Reference: k sequential forward_token decodes (logits path -> full per-row
            // logits + CPU argmax with the sampler's first-max tie-break).
            let mut ref_session = mk_session();
            let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
            for i in 0..k {
                let emb = &emb_all[i * hidden..(i + 1) * hidden];
                let cos_i = &cos_all[i * half..(i + 1) * half];
                let sin_i = &sin_all[i * half..(i + 1) * half];
                let out = ref_session
                    .forward_token(
                        emb,
                        &weights,
                        cos_i,
                        sin_i,
                        base + i,
                        scale,
                        Some(make_stage()),
                        None,
                        0,
                        None,
                    )
                    .expect("reference forward_token");
                match out {
                    ResidentTokenOut::Data(v) => {
                        assert_eq!(v.len(), vocab);
                        ref_logits.push(v);
                    }
                    ResidentTokenOut::Sampled(_) => panic!("unexpected sampled output"),
                }
            }
            let ref_argmax: Vec<u32> = ref_logits
                .iter()
                .map(|row| {
                    let mut best = f32::NEG_INFINITY;
                    let mut best_i = 0u32;
                    for (j, &v) in row.iter().enumerate() {
                        if v > best {
                            best = v;
                            best_i = j as u32;
                        }
                    }
                    best_i
                })
                .collect();

            // Candidate: verify_batch on the identically-seeded session.
            let mut cand_session = mk_session();
            let stage = make_stage();
            let (preds, cand_logits) = cand_session
                .verify_batch_logits(
                    &emb_all, &cos_all, &sin_all, &weights, &stage, base, k, scale,
                )
                .expect("verify_batch eligible");
            assert_eq!(preds.len(), k);
            assert_eq!(cand_logits.len(), k * vocab);

            // Hard gate: exact u32 bit identity for every row/element + argmax id equality.
            for i in 0..k {
                let pc = base + i + 1;
                for v in 0..vocab {
                    let a = cand_logits[i * vocab + v];
                    let b = ref_logits[i][v];
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "base={base} k={k} row {i} (pc={pc}) vocab {v}: verify {a} ({:#010x}) \
                         != forward {b} ({:#010x})",
                        a.to_bits(),
                        b.to_bits(),
                    );
                }
                assert_eq!(
                    preds[i], ref_argmax[i],
                    "base={base} k={k} row {i} (pc={pc}): argmax verify {} != forward {}",
                    preds[i], ref_argmax[i]
                );
            }
            eprintln!(
                "metal_spec_verify_bit_identical: base={base} k={k} BIT-IDENTICAL \
                 (split_k_engaged={})",
                splitk_attention_enabled() && attn2_enabled()
            );
        };

        // C2: straddle the 128 split-K boundary (rows pc in [127..132]).
        run(126, 6);
        // C3: deep split-K (rows pc in [511..516], n_splits varies).
        run(510, 6);
    }

    /// WIN2METAL Phase 4 gate. The slot-indirected TREE verify path must reproduce the
    /// Phase-3 linear verify byte-for-byte on a single-branch tree (the ANCHOR), and on a
    /// genuinely branching tree the accepted-path KV after compaction must equal an
    /// independent linear `forward_token` decode of that path (byte-identical), with every
    /// emitted token == the reference greedy argmax (Parity-class A).
    ///
    /// Same synthetic 2-layer / 4-head / group-2 / head_dim-32 / vocab-256 model and seeded
    /// `base` history as `metal_spec_verify_bit_identical`. The LINEAR anchor straddles the
    /// split-K boundary at base 126 AND deep split-K at base 510 (both mandatory).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tree_verify_bit_identical() {
        use crate::inference::spec_tree::TokenTree;
        if !detect_metal_device().available {
            return;
        }
        // Latch the production GEMV + tiled/split-K attention gates ON before the first read
        // (identical to the linear gate); SKIP if a sibling latched them off earlier.
        std::env::set_var("CAMELID_METAL_F32Y", "1");
        std::env::set_var("CAMELID_METAL_WIRE", "1");
        std::env::set_var("CAMELID_METAL_WIRE_NSG8", "1");
        std::env::set_var("CAMELID_METAL_ATTN2", "1");
        // Require the split-K attention path too (see metal_spec_verify_bit_identical) so the
        // linear-anchor straddle-126 / deep-510 windows really exercise split-K; SKIP loudly
        // otherwise. Targeted proof lane: qa/speed/verify-gates.sh (--test-threads=1).
        if !(f32y_gemv_enabled()
            && wire_weights_enabled()
            && wire_nsg8_enabled()
            && attn2_enabled()
            && splitk_attention_enabled())
        {
            eprintln!(
                "SKIP metal_tree_verify_bit_identical: f32y/wire/nsg8/attn2/split-K gates inactive \
                 (OnceLock latched off earlier in this process); run targeted: \
                 qa/speed/verify-gates.sh (or cargo test --release --lib \
                 metal_tree_verify_bit_identical)"
            );
            return;
        }

        let n_layers = 2usize;
        let n_heads = 4usize;
        let n_kv = 2usize; // group = 2
        let head_dim = 32usize;
        let hidden = 128usize;
        let ffn = 256usize;
        let vocab = 256usize;
        let eps = 1.0e-5f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let half = head_dim / 2;
        let bpr_hidden = hidden / 32;
        let bpr_q = q_dim / 32;
        let bpr_ffn = ffn / 32;
        let mkw = |rows: usize, bpr: usize, seed: usize| {
            let mut w: Vec<u8> = Vec::new();
            for r in 0..rows {
                for b in 0..bpr {
                    let s = 0.05 + ((r * bpr + b + seed) as f32 % 7.0) * 0.01;
                    w.extend_from_slice(&s.to_le_bytes());
                    for l in 0..32 {
                        w.push((((r * 5 + b * 3 + l + seed) as i32 % 17) - 8) as i8 as u8);
                    }
                }
            }
            w
        };
        struct LW {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            q: Vec<u8>,
            k: Vec<u8>,
            v: Vec<u8>,
            o: Vec<u8>,
            gate: Vec<u8>,
            up: Vec<u8>,
            down: Vec<u8>,
        }
        let data: Vec<LW> = (0..n_layers)
            .map(|li| {
                let s = li * 100;
                LW {
                    attn_norm: (0..hidden)
                        .map(|i| 0.5 + ((i + li) as f32 % 3.0) * 0.1)
                        .collect(),
                    ffn_norm: (0..hidden)
                        .map(|i| 0.4 + ((i + li) as f32 % 5.0) * 0.07)
                        .collect(),
                    q: mkw(q_dim, bpr_hidden, s + 1),
                    k: mkw(kv_dim, bpr_hidden, s + 2),
                    v: mkw(kv_dim, bpr_hidden, s + 3),
                    o: mkw(hidden, bpr_q, s + 4),
                    gate: mkw(ffn, bpr_hidden, s + 5),
                    up: mkw(ffn, bpr_hidden, s + 6),
                    down: mkw(hidden, bpr_ffn, s + 7),
                }
            })
            .collect();
        let weights: Vec<ResidentLayerWeights> = data
            .iter()
            .map(|d| ResidentLayerWeights {
                attn_norm: &d.attn_norm,
                ffn_norm: &d.ffn_norm,
                q_weight_blocks: ResidentWeightBytes::Blocks36(&d.q),
                k_weight_blocks: ResidentWeightBytes::Blocks36(&d.k),
                v_weight_blocks: ResidentWeightBytes::Blocks36(&d.v),
                o_weight_blocks: ResidentWeightBytes::Blocks36(&d.o),
                gate_weight_blocks: ResidentWeightBytes::Blocks36(&d.gate),
                up_weight_blocks: ResidentWeightBytes::Blocks36(&d.up),
                down_weight_blocks: ResidentWeightBytes::Blocks36(&d.down),
                q_norm: None,
                k_norm: None,
            })
            .collect();
        let final_norm: Vec<f32> = (0..hidden)
            .map(|i| 0.45 + (i as f32 % 4.0) * 0.05)
            .collect();
        let out_w = mkw(vocab, bpr_hidden, 777);
        let make_stage = || LogitsStage {
            final_norm: &final_norm,
            output_weight_blocks: ResidentWeightBytes::Blocks36(&out_w),
            vocab_size: vocab,
        };
        let synth_kv = |base: usize, layer: usize, is_v: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; n_kv * base * head_dim];
            for h in 0..n_kv {
                for p in 0..base {
                    for d in 0..head_dim {
                        let idx = (h * base + p) * head_dim + d;
                        let t = (layer * 7 + h * 13 + p * 3 + d + usize::from(is_v) * 5) as f32;
                        out[idx] = ((t % 19.0) - 9.0) * 0.05;
                    }
                }
            }
            out
        };
        // Absolute-position RoPE table entry and per-node embedding — shared by the tree pass
        // and the reference so node `x` at position `pos` is identical in both.
        let cos_at =
            |pos: usize, p: usize| -> f32 { (0.2 + (pos as f32) * 0.05 + p as f32 * 0.1).cos() };
        let sin_at =
            |pos: usize, p: usize| -> f32 { (0.2 + (pos as f32) * 0.05 + p as f32 * 0.1).sin() };
        let emb_of = |node: usize, j: usize| -> f32 { (((node + j) as f32 % 11.0) - 5.0) * 0.2 };

        // ---------- LINEAR ANCHOR: tree verify == linear verify, bit-for-bit ----------
        let linear_anchor = |base: usize| {
            let k = 6usize; // 6 nodes (anchor + 5 drafts) — straddles the split-K boundary
            let cap = base + k + 16;
            let max_positions = cap;
            // Per-row tables at position base+i (a linear tree has depth[i] == i).
            let mut cos_all = vec![0.0f32; k * half];
            let mut sin_all = vec![0.0f32; k * half];
            let mut emb_all = vec![0.0f32; k * hidden];
            for i in 0..k {
                for p in 0..half {
                    cos_all[i * half + p] = cos_at(base + i, p);
                    sin_all[i * half + p] = sin_at(base + i, p);
                }
                for j in 0..hidden {
                    emb_all[i * hidden + j] = emb_of(i, j);
                }
            }
            let mk_session = || {
                let mut s = ResidentDecodeState::new(
                    n_layers,
                    n_heads,
                    n_kv,
                    head_dim,
                    hidden,
                    ffn,
                    max_positions,
                    cap,
                    eps,
                    false,
                )
                .unwrap();
                for layer in 0..n_layers {
                    let ck = synth_kv(base, layer, false);
                    let cv = synth_kv(base, layer, true);
                    assert!(s.seed_layer(layer, &ck, &cv, base), "seed_layer");
                }
                s.set_filled(base);
                s
            };

            // Reference: the proven Phase-3 linear verify.
            let mut lin_session = mk_session();
            let stage = make_stage();
            let (lin_preds, lin_logits) = lin_session
                .verify_batch_logits(
                    &emb_all, &cos_all, &sin_all, &weights, &stage, base, k, scale,
                )
                .expect("verify_batch eligible");

            // Candidate: the TREE verify on a single-branch (linear) tree. Token ids are
            // irrelevant to the forward pass (only the structure drives ancestor_bits /
            // node_kvslot), so any chain of k nodes works.
            let drafts: Vec<u32> = (1..k as u32).collect();
            let tree = TokenTree::linear(0, &drafts);
            assert_eq!(tree.nodes(), k);
            let node_kvslot = tree.node_kvslot(base);
            let (ancestor_bits, words) = tree.ancestor_bitset();
            let mut tree_session = mk_session();
            let stage = make_stage();
            let (tree_preds, tree_logits) = tree_session
                .verify_batch_tree_logits(
                    &emb_all,
                    &cos_all,
                    &sin_all,
                    &weights,
                    &stage,
                    &node_kvslot,
                    &ancestor_bits,
                    words,
                    base,
                    k,
                    scale,
                )
                .expect("verify_batch_tree eligible");

            assert_eq!(tree_preds.len(), k);
            assert_eq!(tree_logits.len(), k * vocab);
            for i in 0..k {
                let pc = base + i + 1;
                for v in 0..vocab {
                    let a = tree_logits[i * vocab + v];
                    let b = lin_logits[i * vocab + v];
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "LINEAR base={base} node {i} (pc={pc}) vocab {v}: tree {a} ({:#010x}) != \
                         linear {b} ({:#010x})",
                        a.to_bits(),
                        b.to_bits(),
                    );
                }
                assert_eq!(
                    tree_preds[i], lin_preds[i],
                    "LINEAR base={base} node {i} (pc={pc}): argmax tree {} != linear {}",
                    tree_preds[i], lin_preds[i]
                );
            }
            eprintln!("metal_tree_verify_bit_identical: LINEAR base={base} k={k} BIT-IDENTICAL");
        };
        linear_anchor(126); // straddle the split-K boundary (pc 127..132)
        linear_anchor(510); // deep split-K (pc 511..516)

        // ---------- BRANCHING: compaction == linear decode + Parity-class A ----------
        // Tree (BFS):     0
        //                / \
        //               1   2
        //              /   / \
        //             3   4   5
        let branching = |base: usize| {
            let parent = [-1i32, 0, 0, 1, 2, 2];
            let depth = [0u16, 1, 1, 2, 2, 2];
            let n = 6usize;
            let cap = base + n + 16;
            let max_positions = cap;
            // Per-NODE tables at position base+depth[i]; per-node embeddings.
            let node_pos: Vec<usize> = (0..n).map(|i| base + depth[i] as usize).collect();
            let mut cos_all = vec![0.0f32; n * half];
            let mut sin_all = vec![0.0f32; n * half];
            let mut emb_node = vec![0.0f32; n * hidden];
            for i in 0..n {
                for p in 0..half {
                    cos_all[i * half + p] = cos_at(node_pos[i], p);
                    sin_all[i * half + p] = sin_at(node_pos[i], p);
                }
                for j in 0..hidden {
                    emb_node[i * hidden + j] = emb_of(i, j);
                }
            }
            let mk_session = || {
                let mut s = ResidentDecodeState::new(
                    n_layers,
                    n_heads,
                    n_kv,
                    head_dim,
                    hidden,
                    ffn,
                    max_positions,
                    cap,
                    eps,
                    false,
                )
                .unwrap();
                for layer in 0..n_layers {
                    let ck = synth_kv(base, layer, false);
                    let cv = synth_kv(base, layer, true);
                    assert!(s.seed_layer(layer, &ck, &cv, base), "seed_layer");
                }
                s.set_filled(base);
                s
            };

            // Provisional tree (tokens set after we see `predicted`, to force a real branch).
            let provisional = TokenTree {
                tokens: vec![0, 0, 0, 0, 0, 0],
                parent: parent.to_vec(),
                depth: depth.to_vec(),
            };
            let node_kvslot = provisional.node_kvslot(base);
            let (ancestor_bits, words) = provisional.ancestor_bitset();

            let mut tree_session = mk_session();
            let stage = make_stage();
            let predicted = tree_session
                .verify_batch_tree(
                    &emb_node,
                    &cos_all,
                    &sin_all,
                    &weights,
                    &stage,
                    &node_kvslot,
                    &ancestor_bits,
                    words,
                    base,
                    n,
                    scale,
                )
                .expect("verify_batch_tree eligible");
            assert_eq!(predicted.len(), n);

            // Craft tokens so accept_longest_path follows the right branch 0 -> 2 -> 5 (a
            // genuine reorder: path[1]=2!=1 and path[2]=5!=2, so compaction does real work).
            // accept picks the FIRST child (index order) whose token == predicted[parent], so
            // make node 1 / node 4 mismatch and node 2 / node 5 match.
            let mut tokens = vec![7u32, 0, 0, 3, 0, 0];
            tokens[1] = (predicted[0] + 1) % vocab as u32; // sibling of node 2: must NOT match
            tokens[2] = predicted[0]; // accepted child of root
            tokens[4] = (predicted[2] + 1) % vocab as u32; // sibling of node 5: must NOT match
            tokens[5] = predicted[2]; // accepted child of node 2
            let tree = TokenTree {
                tokens,
                parent: parent.to_vec(),
                depth: depth.to_vec(),
            };
            let (emitted, leaf) = tree.accept_longest_path(&predicted);
            assert_eq!(leaf, 5, "branch should accept down to node 5");
            let path = tree.path_to(leaf);
            assert_eq!(path, vec![0, 2, 5], "accepted path");
            assert_eq!(
                emitted,
                vec![predicted[0], predicted[2], predicted[5]],
                "emitted == target argmax along the accepted path"
            );

            // Compact the accepted path into contiguous slots [base, base+3).
            tree_session
                .compact_tree_kv_path(&path, base)
                .expect("compact");

            // Independent reference: a fresh session that linearly decodes the path tokens
            // (nodes 0, 2, 5) at positions base, base+1, base+2.
            let mut ref_session = mk_session();
            let mut ref_argmax = vec![0u32; path.len()];
            for (r, &node) in path.iter().enumerate() {
                let pos = base + r; // == node_pos[node] since depth[node] == r
                let emb = &emb_node[node * hidden..(node + 1) * hidden];
                let cos_i = &cos_all[node * half..(node + 1) * half];
                let sin_i = &sin_all[node * half..(node + 1) * half];
                let out = ref_session
                    .forward_token(
                        emb,
                        &weights,
                        cos_i,
                        sin_i,
                        pos,
                        scale,
                        Some(make_stage()),
                        None,
                        0,
                        None,
                    )
                    .expect("reference forward_token");
                let v = match out {
                    ResidentTokenOut::Data(v) => v,
                    ResidentTokenOut::Sampled(_) => panic!("unexpected sampled output"),
                };
                assert_eq!(v.len(), vocab);
                let mut best = f32::NEG_INFINITY;
                let mut best_i = 0u32;
                for (j, &x) in v.iter().enumerate() {
                    if x > best {
                        best = x;
                        best_i = j as u32;
                    }
                }
                ref_argmax[r] = best_i;
            }

            // (a) Compacted accepted-path KV == the linear decode's KV, byte-for-byte, in both
            //     the f32 caches and the f16 mirrors.
            let read_row =
                |s: &ResidentDecodeState, kbuf: bool, f16: bool, l: usize, slot: usize| {
                    let mut k_out = vec![0u32; n_kv * head_dim];
                    unsafe {
                        if f16 {
                            let buf = if kbuf {
                                &s.cache_k16[l]
                            } else {
                                &s.cache_v16[l]
                            };
                            let p = buf.contents() as *const u16;
                            for h in 0..n_kv {
                                let off = (h * max_positions + slot) * head_dim;
                                for d in 0..head_dim {
                                    k_out[h * head_dim + d] = *p.add(off + d) as u32;
                                }
                            }
                        } else {
                            let buf = if kbuf { &s.cache_k[l] } else { &s.cache_v[l] };
                            let p = buf.contents() as *const f32;
                            for h in 0..n_kv {
                                let off = (h * max_positions + slot) * head_dim;
                                for d in 0..head_dim {
                                    k_out[h * head_dim + d] = (*p.add(off + d)).to_bits();
                                }
                            }
                        }
                    }
                    k_out
                };
            for l in 0..n_layers {
                for r in 0..path.len() {
                    let slot = base + r;
                    for kbuf in [true, false] {
                        for f16 in [false, true] {
                            let a = read_row(&tree_session, kbuf, f16, l, slot);
                            let b = read_row(&ref_session, kbuf, f16, l, slot);
                            assert_eq!(
                                a, b,
                                "BRANCH base={base} layer {l} slot {slot} (rank {r}) \
                                 kbuf={kbuf} f16={f16}: compacted KV != linear-decode KV"
                            );
                        }
                    }
                }
            }

            // (b) Parity-class A: the tree's argmax at each accepted node == the reference
            //     greedy argmax along the path.
            for (r, &node) in path.iter().enumerate() {
                assert_eq!(
                    predicted[node], ref_argmax[r],
                    "BRANCH base={base} class-A: tree predicted[node {node}] {} != reference \
                     argmax[rank {r}] {}",
                    predicted[node], ref_argmax[r]
                );
            }
            eprintln!(
                "metal_tree_verify_bit_identical: BRANCH base={base} path={path:?} \
                 compaction==linear-decode + class-A PASS"
            );
        };
        branching(126);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_argmax_greedy_matches_cpu_first_max() {
        if !detect_metal_device().available {
            return;
        }
        let k = metal_linear_kernel().expect("metal");
        // Shapes around the real vocab plus tie/edge cases; CPU reference is the
        // sampler's exact semantics: ascending scan, strict greater-than.
        let cases: Vec<Vec<f32>> = vec![
            (0..128256)
                .map(|i| ((i * 2654435761u64 as usize) % 9973) as f32 * 0.001)
                .collect(),
            vec![1.0, 5.0, 5.0, 3.0, 5.0], // tie -> lowest index (1)
            vec![7.0, 1.0, 2.0],           // max at 0
            vec![-3.0, -1.5, -2.0],        // all negative
            (0..1000)
                .map(|i| if i == 999 { 10.0 } else { 0.0 })
                .collect(), // max at end
            vec![42.0],                    // single element
        ];
        for logits in cases {
            let mut best_idx = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best {
                    best = v;
                    best_idx = i;
                }
            }
            let lb = k.device.new_buffer_with_data(
                logits.as_ptr() as *const _,
                (logits.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let ib = k
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
            let cb_scalar = k
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
            unsafe { *(cb_scalar.contents() as *mut u32) = logits.len() as u32 };
            let cb = k.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&k.argmax_f32_greedy_pipeline);
            e.set_buffer(0, Some(&lb), 0);
            e.set_buffer(1, Some(&ib), 0);
            e.set_buffer(2, Some(&cb_scalar), 0);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { *(ib.contents() as *const u32) };
            assert_eq!(got as usize, best_idx, "len {}", logits.len());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_gumbel_sampler_matches_stateless_reference() {
        if !detect_metal_device().available {
            return;
        }
        let k = metal_linear_kernel().expect("metal");
        let logits = vec![0.0_f32; 4096];

        let reference = |seed: u64| {
            let mut best_uniform = f32::NEG_INFINITY;
            let mut best_idx = 0_u32;
            for idx in 0..logits.len() {
                let mut z =
                    seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(idx as u64 + 1));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                // Gumbel is monotonic in the uniform draw. Comparing the
                // exact 24-bit uniforms avoids any CPU/GPU log approximation
                // becoming part of this stateless-RNG and reduction test.
                let uniform = ((z >> 40) as u32) as f32;
                if uniform > best_uniform {
                    best_uniform = uniform;
                    best_idx = idx as u32;
                }
            }
            best_idx
        };

        let logits_buf = k.device.new_buffer_with_data(
            logits.as_ptr() as *const _,
            (logits.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = k
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        let count_buf = k
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        let temperature_buf = k
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        let seed_buf = k
            .device
            .new_buffer(8, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(count_buf.contents() as *mut u32) = logits.len() as u32;
            *(temperature_buf.contents() as *mut f32) = 1.0;
        }

        let mut sampled = Vec::new();
        for seed in [0_u64, 1, 7, 42, u32::MAX as u64, u64::MAX] {
            unsafe {
                *(seed_buf.contents() as *mut u64) = seed;
            }
            let cb = k.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&k.sample_gumbel_f32_pipeline);
            e.set_buffer(0, Some(&logits_buf), 0);
            e.set_buffer(1, Some(&out_buf), 0);
            e.set_buffer(2, Some(&count_buf), 0);
            e.set_buffer(3, Some(&temperature_buf), 0);
            e.set_buffer(4, Some(&seed_buf), 0);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { *(out_buf.contents() as *const u32) };
            assert_eq!(got, reference(seed), "seed {seed}");
            sampled.push(got);
        }
        sampled.sort_unstable();
        sampled.dedup();
        assert!(
            sampled.len() > 1,
            "different seeds should not collapse to one sampled token"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_embed_gather_matches_cpu_dequant() {
        if !detect_metal_device().available {
            return;
        }
        let k = metal_linear_kernel().expect("metal");
        let vocab = 7usize;
        let hidden = 96usize; // 3 blocks per row
        let bpr = hidden / 32;
        // Wire blocks: f16 scale + 32 i8 quants, 34 bytes per block.
        let mut wire = vec![0u8; vocab * bpr * 34];
        for (b, chunk) in wire.chunks_mut(34).enumerate() {
            let scale = f32_to_f16_bits(0.013 + b as f32 * 0.0007);
            chunk[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, q) in chunk[2..].iter_mut().enumerate() {
                *q = (((b * 31 + j * 7) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let token: u32 = 5;
        // CPU reference: f16 scale -> f32, times the i8 quant.
        let mut expected = vec![0.0f32; hidden];
        for blk in 0..bpr {
            let base = (token as usize * bpr + blk) * 34;
            let scale_bits = u16::from_le_bytes([wire[base], wire[base + 1]]);
            let scale = f32::from_bits({
                // same conversion the GPU's float(half) performs
                let h = scale_bits as u32;
                let sign = (h & 0x8000) << 16;
                let exp = (h >> 10) & 0x1f;
                let man = h & 0x3ff;
                if exp == 0 && man == 0 {
                    sign
                } else if exp == 0 {
                    let mut e = 127 - 15 + 1;
                    let mut m = man;
                    while m & 0x400 == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
                } else {
                    sign | ((exp + 127 - 15) << 23) | (man << 13)
                }
            });
            for j in 0..32 {
                expected[blk * 32 + j] = (wire[base + 2 + j] as i8) as f32 * scale;
            }
        }
        let eb = k.device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ib = k
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(ib.contents() as *mut u32) = token };
        let ob = k
            .device
            .new_buffer((hidden * 4) as u64, MTLResourceOptions::StorageModeShared);
        let sc = k
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(sc.contents() as *mut u32) = bpr as u32 };
        let cb = k.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&k.embed_row_gather_q8_wire_pipeline);
        e.set_buffer(0, Some(&eb), 0);
        e.set_buffer(1, Some(&ib), 0);
        e.set_buffer(2, Some(&ob), 0);
        e.set_buffer(3, Some(&sc), 0);
        dispatch_1d(e, &k.embed_row_gather_q8_wire_pipeline, hidden);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let mut got = vec![0.0f32; hidden];
        read_buffer_f32(&ob, &mut got);
        assert_eq!(got, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_quantized_matmul_resident_matches_two_step() {
        if !detect_metal_device().available {
            return;
        }
        // The resident quantize->matmul chain (one command buffer, GPU buffers passed
        // between kernels) must equal running the two standalone kernels separately
        // (each already parity-checked vs CPU). This validates resident buffer chaining.
        let input_width = 64usize; // 2 blocks
        let output_width = 5usize;
        let blocks_per_row = input_width / 32;
        let input: Vec<f32> = (0..input_width)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.3)
            .collect();
        let mut weight_blocks: Vec<u8> = Vec::new();
        for r in 0..output_width {
            for b in 0..blocks_per_row {
                let scale = 0.1 + (r * blocks_per_row + b) as f32 * 0.02;
                weight_blocks.extend_from_slice(&scale.to_le_bytes());
                for l in 0..32 {
                    weight_blocks.push((((r * 7 + b * 3 + l) as i32 % 19) - 9) as i8 as u8);
                }
            }
        }
        let (scales, quants) = try_quantize_q8_0_f32(&input).expect("quantize");
        let mut expected = vec![0.0f32; output_width];
        assert!(try_q8_0_block_linear_row(
            &scales,
            &quants,
            &weight_blocks,
            output_width,
            blocks_per_row,
            &mut expected,
        ));
        let got = try_quantized_matmul_resident(&input, &weight_blocks, output_width)
            .expect("resident chain");
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_quantize_q8_0_matches_reference() {
        if !detect_metal_device().available {
            return;
        }
        // Two blocks chosen so max|v|/127 is f16-exact (127 -> 1.0, 254 -> 2.0),
        // isolating the kernel's max/round/clamp logic from f16-rounding subtleties.
        let mut input = vec![0.0f32; 64];
        for (i, v) in input.iter_mut().enumerate().take(32) {
            *v = ((i as i32 % 9) - 4) as f32; // small ints
        }
        input[7] = 127.0; // sets block-0 max_abs to 127 -> scale 1.0
        for (i, v) in input.iter_mut().enumerate().skip(32) {
            *v = (((i as i32) % 5) * 2 - 4) as f32; // even ints
        }
        input[40] = 254.0; // block-1 max_abs 254 -> scale 2.0
                           // Reference
        let mut exp_scales = [0.0f32; 2];
        let mut exp_quants = [0i8; 64];
        for b in 0..2 {
            let blk = &input[b * 32..b * 32 + 32];
            let max_abs = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let unrounded = max_abs / 127.0; // f16-exact here (1.0 / 2.0)
            exp_scales[b] = unrounded;
            let inv = if unrounded == 0.0 {
                0.0
            } else {
                1.0 / unrounded
            };
            for (i, v) in blk.iter().enumerate() {
                let q = (v * inv).round() as i32;
                exp_quants[b * 32 + i] = q.clamp(-127, 127) as i8;
            }
        }
        let (scales, quants) = try_quantize_q8_0_f32(&input).expect("metal quantize");
        for (a, b) in scales.iter().zip(&exp_scales) {
            assert!((a - b).abs() < 1.0e-6, "scale {a} != {b}");
        }
        assert_eq!(quants.as_slice(), &exp_quants[..], "quants mismatch");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_decode_matches_reference() {
        if !detect_metal_device().available {
            return;
        }
        // MHA (n_kv_heads == n_heads), contiguous KV cache.
        let n_heads = 2usize;
        let head_dim = 4usize;
        let positions = 3usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let query: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
            .collect();
        let keys: Vec<f32> = (0..n_heads * positions * head_dim)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
            .collect();
        let values: Vec<f32> = (0..n_heads * positions * head_dim)
            .map(|i| ((i as f32 % 4.0) - 1.0) * 0.5)
            .collect();
        // Reference: per head, softmax(dot(q,k)*scale) then weighted V sum.
        let mut expected = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let qb = h * head_dim;
            let kvb = h * positions * head_dim;
            let mut scores = vec![0.0f32; positions];
            for (p, score) in scores.iter_mut().enumerate() {
                let kb = kvb + p * head_dim;
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += query[qb + d] * keys[kb + d];
                }
                *score = s * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for (p, s) in scores.iter().enumerate() {
                let prob = s / sum;
                let vb = kvb + p * head_dim;
                for d in 0..head_dim {
                    expected[qb + d] += prob * values[vb + d];
                }
            }
        }
        let got = try_attention_decode_f32(
            &query, &keys, &values, n_heads, n_heads, head_dim, positions, scale,
        )
        .expect("metal attention");
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_rope_rotate_matches_reference() {
        if !detect_metal_device().available {
            return;
        }
        let head_count = 4usize;
        let head_dim = 8usize;
        let half = head_dim / 2;
        let data: Vec<f32> = (0..head_count * head_dim)
            .map(|i| (i as f32 % 9.0) - 4.0)
            .collect();
        let cos_t: Vec<f32> = (0..half).map(|p| (0.3 + p as f32 * 0.2).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|p| (0.3 + p as f32 * 0.2).sin()).collect();
        // Reference: adjacent even/odd forward rotation, same as apply_rope_to_row.
        let mut expected = data.clone();
        for head in 0..head_count {
            let hs = head * head_dim;
            for pair in 0..half {
                let d0 = hs + pair * 2;
                let d1 = d0 + 1;
                let (x0, x1) = (data[d0], data[d1]);
                expected[d0] = x0 * cos_t[pair] - x1 * sin_t[pair];
                expected[d1] = x0 * sin_t[pair] + x1 * cos_t[pair];
            }
        }
        let got = try_rope_rotate_f32(&data, &cos_t, &sin_t, head_count, head_dim, half, false)
            .expect("metal rope");
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_residual_add_matches_cpu() {
        if !detect_metal_device().available {
            return;
        }
        let n = 300usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * -0.1).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let got = try_residual_add_f32(&a, &b).expect("metal residual_add");
        for (x, y) in got.iter().zip(&expected) {
            assert!((x - y).abs() < 1.0e-4, "{x} != {y}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_linear_row_matches_cpu_for_small_dense_accumulation() {
        if !detect_metal_device().available {
            return;
        }

        let input = [2.0_f32, -1.0, 0.5];
        let weights = [
            1.0_f32, 2.0, -3.0, 4.0, // row 0
            -2.0, 0.5, 1.5, -1.0, // row 1
            0.25, -4.0, 2.0, 0.0, // row 2
        ];
        let mut output = [1.0_f32, -2.0, 0.5, 3.0];
        let mut expected = output;
        for col in 0..expected.len() {
            for row in 0..input.len() {
                expected[col] += input[row] * weights[row * expected.len() + col];
            }
        }

        assert!(try_linear_row_f32(
            &input,
            &weights,
            input.len(),
            output.len(),
            &mut output
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_linear_row_transposed_matches_cpu_for_small_dense_dot_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input = [2.0_f32, -1.0, 0.5];
        let weights = [
            1.0_f32, 2.0, -3.0, 4.0, -2.0, 0.5, 1.5, -1.0, 0.25, -4.0, 2.0, 0.0,
        ];
        let mut output = [0.0_f32; 4];
        let mut expected = [0.0_f32; 4];
        for col in 0..expected.len() {
            for row in 0..input.len() {
                expected[col] += input[row] * weights[col * input.len() + row];
            }
        }

        assert!(try_linear_row_transposed_f32(
            &input,
            &weights,
            input.len(),
            expected.len(),
            &mut output
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_encoded_linear_row_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ];
        let row0 = [
            -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
            -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
        ];
        let row1 = [
            2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7, -6,
            8, 7, -8, -7, 9, 8, -9, -8,
        ];
        let mut encoded_rows = Vec::new();
        for row in [&row0, &row1] {
            encoded_rows.extend_from_slice(&[0, 0]);
            encoded_rows.extend(row.iter().map(|value| *value as u8));
        }
        let weight_scales = [0.5_f32, 0.125];
        let mut output = [0.0_f32; 2];
        let expected = [
            input_quants
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[0],
            input_quants
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[1],
        ];

        assert!(try_q8_0_encoded_linear_row(
            &input_scales,
            &input_quants,
            &encoded_rows,
            &weight_scales,
            2,
            1,
            &mut output,
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_encoded_linear_rows_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32, 0.5];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32, -3_i8, 4, -5, 6, -7, 8, -9, 10, -11,
            12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30,
            -31, 32, -33, 34,
        ];
        let row0 = [
            -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
            -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
        ];
        let row1 = [
            2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7, -6,
            8, 7, -8, -7, 9, 8, -9, -8,
        ];
        let mut encoded_rows = Vec::new();
        for row in [&row0, &row1] {
            encoded_rows.extend_from_slice(&[0, 0]);
            encoded_rows.extend(row.iter().map(|value| *value as u8));
        }
        let weight_scales = [0.5_f32, 0.125];
        let input_rows = 2;
        let weight_rows = 2;
        let blocks_per_row = 1;
        let mut output = [0.0_f32; 4];
        let input_row = |idx: usize| &input_quants[idx * 32..(idx + 1) * 32];
        let expected = [
            input_row(0)
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[0],
            input_row(1)
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[1]
                * weight_scales[0],
            input_row(0)
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[1],
            input_row(1)
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[1]
                * weight_scales[1],
        ];

        assert!(try_q8_0_encoded_linear_rows(
            &input_scales,
            &input_quants,
            &encoded_rows,
            &weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            &mut output,
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_block_linear_row_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ];
        let row0 = [
            -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
            -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
        ];
        let row1 = [
            2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7, -6,
            8, 7, -8, -7, 9, 8, -9, -8,
        ];
        let weight_scales = [0.5_f32, 0.125];
        let mut weight_blocks = Vec::new();
        for (scale, row) in weight_scales.iter().zip([&row0, &row1]) {
            weight_blocks.extend_from_slice(&scale.to_le_bytes());
            weight_blocks.extend(row.iter().map(|value| *value as u8));
        }
        let mut output = [0.0_f32; 2];
        let expected = [
            input_quants
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[0],
            input_quants
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[1],
        ];

        assert!(try_q8_0_block_linear_row(
            &input_scales,
            &input_quants,
            &weight_blocks,
            2,
            1,
            &mut output,
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_block_linear_row_simd_matches_cpu_multi_block() {
        if !detect_metal_device().available {
            return;
        }
        // Multi-block rows exercise the SIMD kernel's strided lane loop + simd_sum.
        let blocks_per_row = 4_usize;
        let rows = 8_usize;
        let input_scales: Vec<f32> = (0..blocks_per_row).map(|b| 0.1 + b as f32 * 0.05).collect();
        let input_quants: Vec<i8> = (0..blocks_per_row * 32)
            .map(|j| ((j as i32 % 17) - 8) as i8)
            .collect();
        let mut weight_blocks = Vec::new();
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let scale = 0.2_f32 + (r * blocks_per_row + b) as f32 * 0.01;
                weight_blocks.extend_from_slice(&scale.to_le_bytes());
                for l in 0..32 {
                    weight_blocks.push((((r * 7 + b * 3 + l) as i32 % 19) - 9) as i8 as u8);
                }
            }
        }
        let mut expected = vec![0.0_f32; rows];
        for (r, slot) in expected.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let base = (r * blocks_per_row + b) * 36;
                let scale = f32::from_le_bytes([
                    weight_blocks[base],
                    weight_blocks[base + 1],
                    weight_blocks[base + 2],
                    weight_blocks[base + 3],
                ]);
                let mut isum = 0i32;
                for l in 0..32 {
                    isum += (weight_blocks[base + 4 + l] as i8 as i32)
                        * input_quants[b * 32 + l] as i32;
                }
                sum += isum as f32 * scale * input_scales[b];
            }
            *slot = sum;
        }
        let mut out = vec![0.0_f32; rows];
        let elapsed = bench_q8_0_block_linear_row_batched(
            &input_scales,
            &input_quants,
            &weight_blocks,
            rows,
            blocks_per_row,
            &mut out,
            1,
            true,
        );
        assert!(elapsed.is_some(), "SIMD kernel unavailable");
        for (actual, expected) in out.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-3, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_block_two_linear_rows_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ];
        let rows = [
            [
                -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
                -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
            ],
            [
                2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7,
                -6, 8, 7, -8, -7, 9, 8, -9, -8,
            ],
        ];
        let first_weight_scales = [0.5_f32, 0.125];
        let second_weight_scales = [0.25_f32, 0.75];
        let encode_weight_blocks = |scales: &[f32; 2]| {
            let mut weight_blocks = Vec::new();
            for (scale, row) in scales.iter().zip(rows) {
                weight_blocks.extend_from_slice(&scale.to_le_bytes());
                weight_blocks.extend(row.iter().map(|value| *value as u8));
            }
            weight_blocks
        };
        let first_weight_blocks = encode_weight_blocks(&first_weight_scales);
        let second_weight_blocks = encode_weight_blocks(&second_weight_scales);
        let expected_for = |scales: &[f32; 2]| -> [f32; 2] {
            [
                input_quants
                    .iter()
                    .zip(rows[0])
                    .map(|(a, b)| i32::from(*a) * i32::from(b))
                    .sum::<i32>() as f32
                    * input_scales[0]
                    * scales[0],
                input_quants
                    .iter()
                    .zip(rows[1])
                    .map(|(a, b)| i32::from(*a) * i32::from(b))
                    .sum::<i32>() as f32
                    * input_scales[0]
                    * scales[1],
            ]
        };

        let mut first_output = [0.0_f32; 2];
        let mut second_output = [0.0_f32; 2];
        let mut cpu_work_ran = false;
        assert!(try_q8_0_block_two_linear_rows_with_cpu(
            &input_scales,
            &input_quants,
            &first_weight_blocks,
            &second_weight_blocks,
            2,
            1,
            &mut first_output,
            &mut second_output,
            || cpu_work_ran = true,
        ));
        assert!(cpu_work_ran);
        for (actual, expected) in first_output
            .into_iter()
            .zip(expected_for(&first_weight_scales))
        {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
        for (actual, expected) in second_output
            .into_iter()
            .zip(expected_for(&second_weight_scales))
        {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_linear_row_stub_returns_false() {
        let input = [1.0_f32, 2.0];
        let weights = [3.0_f32, 4.0, 5.0, 6.0];
        let mut output = [0.0_f32, 0.0];
        assert!(!try_linear_row_f32(&input, &weights, 2, 2, &mut output));
        assert_eq!(output, [0.0, 0.0]);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod mma_probe {
    use super::*;

    /// Pure simdgroup-MMA throughput probe: register fragments only, no memory traffic.
    /// Prints the practical fp16-in/f32-acc MMA ceiling. Run with:
    /// cargo test --release mma_throughput_probe -- --nocapture --ignored
    #[test]
    #[ignore]
    fn mma_throughput_probe() {
        let device = Device::system_default().expect("no Metal device");
        let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mma_probe(
    device float* out [[buffer(0)]],
    constant uint& iters [[buffer(1)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // Stage 4KB A + 2KB B once; then run the EXACT mul_mm inner loop (4 a-loads,
    // 2 b-loads, 8 MMAs per 8-k step, 4 steps per 32-k block) against it. Data is
    // L1/shmem resident, so this is the practical ceiling of the MMA pipeline.
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + 2048;
    for (uint i = ltid; i < 3072; i += 128) {
        shmem[i] = half(float(i % 17) * 0.0625f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint it = 0; it < iters; ++it) {
        // Rotate the base address over 128 distinct offsets so the loads can never be
        // hoisted (a 2-way rotation still unrolls+hoists and fakes pure-MMA numbers).
        threadgroup const half* lsma = sa + ((it * 13) % 128);
        threadgroup const half* lsmb = sb + ((it * 7) % 128);
        for (uint ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup float sink[64];
    simdgroup_store(mc[0], sink, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { out[0] = sink[lane % 64]; }
}
"#;
        let options = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .expect("probe compile");
        let f = lib.get_function("mma_probe", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let queue = device.new_command_queue();
        let out = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let iters_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let iters: u32 = 2048;
        unsafe { *(iters_buf.contents() as *mut u32) = iters };
        let tgs: u64 = 4096;
        for round in 0..3 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&out), 0);
            e.set_buffer(1, Some(&iters_buf), 0);
            e.set_threadgroup_memory_length(0, 3072 * 2);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: tgs,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
            // 4 simdgroups/tg x 8 mma/iter x 1024 flop/mma
            let flops = tgs as f64 * 4.0 * iters as f64 * 32.0 * 1024.0;
            eprintln!(
                "[mma-probe] round {round}: {:.0}us -> {:.2} TFLOPS fp16-mma",
                busy_us as f64,
                flops / (busy_us as f64 * 1e-6) / 1e12
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod gemm_probe {
    use super::*;

    /// Standalone microbench of the prefill MM kernel on the gate-projection shape
    /// (8192 rows x k=3072, 601 tokens). Reports per-run GPU-busy, effective weight
    /// bandwidth, and TFLOPS so kernel variants can be evaluated in seconds.
    /// cargo test --release gemm_probe_gate -- --nocapture --ignored
    #[test]
    #[ignore]
    fn gemm_probe_gate() {
        for (label, rows, k) in [
            ("gate/up 8192x3072", 8192usize, 3072usize),
            ("down   3072x8192", 3072, 8192),
            ("q/o    3072x3072", 3072, 3072),
            ("k/v    1024x3072", 1024, 3072),
        ] {
            eprintln!("[gemm-probe] ==== {label} ====");
            gemm_probe_shape(rows, k);
        }
    }

    fn gemm_probe_shape(rows: usize, k: usize) {
        let n_tokens: usize = 601;
        let n_pad: usize = n_tokens.next_multiple_of(128);
        let bpr = k / 32;
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;

        // Synthetic wire blocks: scale half + 32 i8.
        let mut wire = vec![0u8; rows * bpr * 34];
        for (i, chunk) in wire.chunks_mut(34).enumerate() {
            let scale = f32_to_f16_bits(0.01 + (i % 7) as f32 * 0.001);
            chunk[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, q) in chunk[2..].iter_mut().enumerate() {
                *q = (((i + j * 3) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let w_buf = device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut y = vec![0u16; n_pad * k];
        for (i, v) in y.iter_mut().enumerate() {
            *v = f32_to_f16_bits(((i % 31) as f32 - 15.0) * 0.05);
        }
        let y_buf = device.new_buffer_with_data(
            y.as_ptr() as *const _,
            (y.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (n_tokens * rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u32;
            *p = bpr as u32;
            *p.add(1) = rows as u32;
            *p.add(2) = n_tokens as u32;
        }
        let weight_gb = (rows * bpr * 34) as f64 / 1e9;
        let flops = 2.0 * rows as f64 * k as f64 * n_tokens as f64;
        for round in 0..5 {
            let cb = k_ref.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&k_ref.q8_0_block_wire_mm_pipeline);
            e.set_buffer(0, Some(&y_buf), 0);
            e.set_buffer(2, Some(&w_buf), 0);
            e.set_buffer(3, Some(&out_buf), 0);
            e.set_buffer(4, Some(&scalar), 0);
            e.set_buffer(5, Some(&scalar), 4);
            e.set_buffer(6, Some(&scalar), 8);
            e.set_threadgroup_memory_length(0, 12288);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (rows / 64) as u64,
                    height: (n_tokens as u64).div_ceil(128),
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
            let tiles = (n_tokens as f64 / 64.0).ceil();
            eprintln!(
                "[gemm-probe] round {round}: {busy_us}us  weights {:.1} GB/s (x{tiles} tiles: {:.1} GB/s streamed)  {:.2} TFLOPS",
                weight_gb / (busy_us as f64 * 1e-6),
                weight_gb * tiles / (busy_us as f64 * 1e-6),
                flops / (busy_us as f64 * 1e-6) / 1e12,
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod ggml_mm_probe {
    use super::*;

    /// ggml's legacy kernel_mul_mm (q8_0 x f32 instantiation), hand-monomorphized,
    /// run on the gate-projection shape for a ground-truth rate comparison.
    /// cargo test --release ggml_mul_mm_probe -- --nocapture --ignored
    #[test]
    #[ignore]
    fn ggml_mul_mm_probe() {
        let rows: usize = 8192; // ne0
        let k: usize = 3072; // ne00
        let n_tokens: usize = 601; // ne1
        let bpr = k / 32;
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;
        let src = r#"
#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
typedef struct { half d; int8_t qs[32]; } block_q8_0;
typedef struct {
    int ne00; ulong nb01; int ne0; int ne1;
} kargs;
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread half4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (half4x4) reg_f;
}
kernel void kernel_mul_mm_q8_0_f32(
        constant kargs & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;
    const short offset1 = il0/nl;

    device const block_q8_0 * x = (device const block_q8_0 *)(src0 + args.nb01*(r0 + lr0)) + offset1;

    const short iy = 8*(tiitg % NL1);
    const ulong nb11 = (ulong)args.ne00 * 4;
    device const float * y = (device const float *)(src1 + nb11*(r1 + lr1) + 4*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q8_0(x, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0;
                device float4 * D4 = (device float4 *) D;
                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;
                int i = 0;
                for (; i < 16; i++) {
                    *(D4 + i) = *(C4 + i);
                }
            }
        }
    }
}
"#;
        let options = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .map_err(|e| panic!("ggml probe compile: {e}"))
            .unwrap();
        let f = lib.get_function("kernel_mul_mm_q8_0_f32", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();

        let mut wire = vec![0u8; rows * bpr * 34];
        for (i, chunk) in wire.chunks_mut(34).enumerate() {
            let scale = f32_to_f16_bits(0.01 + (i % 7) as f32 * 0.001);
            chunk[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, q) in chunk[2..].iter_mut().enumerate() {
                *q = (((i + j * 3) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let w_buf = device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y: Vec<f32> = (0..n_tokens * k)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
            .collect();
        let y_buf = device.new_buffer_with_data(
            y.as_ptr() as *const _,
            (y.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (n_tokens * rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scalar = device.new_buffer(24, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u8;
            *(p as *mut i32) = k as i32; // ne00
            *(p.add(8) as *mut u64) = (bpr * 34) as u64; // nb01
            *(p.add(16) as *mut i32) = rows as i32; // ne0
            *(p.add(20) as *mut i32) = n_tokens as i32; // ne1
        }
        let weight_gb = (rows * bpr * 34) as f64 / 1e9;
        let flops = 2.0 * rows as f64 * k as f64 * n_tokens as f64;
        for round in 0..5 {
            let cb = k_ref.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&scalar), 0);
            e.set_buffer(1, Some(&w_buf), 0);
            e.set_buffer(2, Some(&y_buf), 0);
            e.set_buffer(3, Some(&out_buf), 0);
            e.set_threadgroup_memory_length(0, 8192);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (n_tokens as u64).div_ceil(32),
                    height: (rows as u64).div_ceil(64),
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
            let tiles = (n_tokens as f64 / 32.0).ceil();
            eprintln!(
                "[ggml-mm-probe] round {round}: {busy_us}us  weights x{tiles} tiles: {:.1} GB/s streamed  {:.2} TFLOPS",
                weight_gb * tiles / (busy_us as f64 * 1e-6),
                flops / (busy_us as f64 * 1e-6) / 1e12,
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod attn_mm_probe {
    use super::*;

    /// Batched f16 GEMM probe on the prefill-attention shapes (S = Q K^T and O = P V,
    /// 24 heads, 601 tokens, head_dim 128) using the staged swizzled-tile structure.
    /// cargo test --release attn_mm_probe_shapes -- --nocapture --ignored
    #[test]
    #[ignore]
    fn attn_mm_probe_shapes() {
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;
        let src = r#"
#include <metal_stdlib>
using namespace metal;
// C[b][t][r] = sum_k A[b][r][k] * B[b][t][k] — both operands half, staged in swizzled
// 8x8 blocks like the Q8 prefill GEMM (A k-major transposed in-block, B token-major).
// 64-row x 64-token tiles, 128 threads, 4 simdgroups of 32x32 quadrants.
kernel void half_mm_batched(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& kdim [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],      // tokens
    constant uint& a_batch_stride [[buffer(6)]],
    constant uint& b_batch_stride [[buffer(7)]],
    constant uint& c_batch_stride [[buffer(8)]],
    constant uint& a_row_stride [[buffer(9)]],
    constant uint& b_row_stride [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR0 = 64;
    constexpr uint NR1 = 64;
    constexpr uint NK = 32;
    const uint tid = sg * 32 + lane;
    threadgroup half sa[64 * 32];
    threadgroup half sb[64 * 32];
    const uint r0 = tg.x * NR0;
    const uint t0 = tg.y * NR1;
    device const half* ab = a + tg.z * a_batch_stride;
    device const half* bb = b + tg.z * b_batch_stride;
    device float* cb = c + tg.z * c_batch_stride;

    const uint lr0 = tid / 2;
    const uint k0 = (tid % 2) * 16;
    const uint lr1 = tid / 2;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (uint i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_row_oct = (sg % 2) * 4;
    const uint sg_tok_oct = (sg / 2) * 4;

    for (uint kk0 = 0; kk0 < kdim; kk0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A: 64 rows x 32 k staged TRANSPOSED (k-major in-block).
        {
            device const half* arow = ab + (r0 + lr0) * a_row_stride + kk0 + k0;
            const uint sy = lr0 / 8;
            const uint lx = lr0 % 8;
            for (uint i = 0; i < 16; ++i) {
                const uint kg = k0 + i;
                sa[64 * (8 * (kg / 8) + sy) + 8 * (kg % 8) + lx] = arow[i];
            }
        }
        // B: 64 tokens x 32 k, token-major vector stores.
        {
            device const half* brow = bb + (t0 + lr1) * b_row_stride + kk0 + k0;
            const uint sy = lr1 / 8;
            const uint ly = lr1 % 8;
            for (uint s = 0; s < 2; ++s) {
                *reinterpret_cast<threadgroup half2x4*>(sb + 64 * (8 * (k0 / 8 + s) + sy) + 8 * ly) =
                    *reinterpret_cast<device const half2x4*>(brow + 8 * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 64 * sg_row_oct;
        threadgroup const half* lsmb = sb + 64 * sg_tok_oct;
        for (uint ik = 0; ik < NK / 8; ++ik) {
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            for (uint i = 0; i < 16; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 64 * 8;
            lsmb += 64 * 8;
        }
    }
    device float* cq = cb + (r0 + 32 * (sg % 2)) + (t0 + 32 * (sg / 2)) * rows;
    for (uint i = 0; i < 16; ++i) {
        simdgroup_store(mc[i], cq + 8 * (i % 4) + 8 * rows * (i / 4), rows, 0, false);
    }
}
"#;
        let options = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .map_err(|e| panic!("attn mm compile: {e}"))
            .unwrap();
        let f = lib.get_function("half_mm_batched", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();

        let heads: usize = 24;
        let n_pad: usize = 640;
        let hd: usize = 128;
        // S shape: rows = positions(n_pad), cols = queries(n_pad), k = 128
        // PV shape: rows = head_dim(128), cols = queries(n_pad), k = n_pad
        let big = heads * n_pad * hd.max(n_pad);
        let a_buf = device.new_buffer(
            (heads * n_pad * n_pad.max(hd) * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer(
            (heads * n_pad * n_pad.max(hd) * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = device.new_buffer(
            (heads * n_pad * n_pad * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let _ = big;
        let scalar = device.new_buffer(44, MTLResourceOptions::StorageModeShared);
        let run = |label: &str, kdim: usize, rows: usize, cols: usize, a_rs: usize, b_rs: usize| {
            unsafe {
                let s = scalar.contents() as *mut u32;
                *s = kdim as u32;
                *s.add(1) = rows as u32;
                *s.add(2) = cols as u32;
                *s.add(3) = (n_pad * a_rs.max(1)) as u32; // a batch stride (elems)
                *s.add(4) = (n_pad * b_rs.max(1)) as u32;
                *s.add(5) = (rows * n_pad) as u32; // c batch stride
                *s.add(6) = a_rs as u32;
                *s.add(7) = b_rs as u32;
            }
            for round in 0..3 {
                let cb = k_ref.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&p);
                e.set_buffer(0, Some(&a_buf), 0);
                e.set_buffer(1, Some(&b_buf), 0);
                e.set_buffer(2, Some(&c_buf), 0);
                for j in 0..8u64 {
                    e.set_buffer(3 + j, Some(&scalar), j * 4);
                }
                e.dispatch_thread_groups(
                    metal::MTLSize {
                        width: (rows as u64).div_ceil(64),
                        height: (cols as u64).div_ceil(64),
                        depth: heads as u64,
                    },
                    metal::MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
                let flops = 2.0 * heads as f64 * rows as f64 * cols as f64 * kdim as f64;
                eprintln!(
                    "[attn-mm] {label} round {round}: {busy_us}us  {:.2} TFLOPS",
                    flops / (busy_us as f64 * 1e-6) / 1e12
                );
            }
        };
        // S = Q K^T: A = K [n_pad x 128], B = Q [n_pad x 128], C [n_pad x n_pad]
        run("S(QK^T)", hd, n_pad, n_pad, hd, hd);
        // O = P V: A = V^T-ish [128 x n_pad] (row stride n_pad), B = P [n_pad x n_pad]
        run("O(PV)", n_pad, hd, n_pad, n_pad, n_pad);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod attn_mm_parity {
    use super::*;

    fn h(x: f32) -> u16 {
        f32_to_f16_bits(x)
    }
    fn fh(b: u16) -> f32 {
        // f16 bits -> f32 (sufficient for test values)
        let s = ((b >> 15) & 1) as i32;
        let e = ((b >> 10) & 0x1f) as i32;
        let m = (b & 0x3ff) as i32;
        let v = if e == 0 {
            (m as f32) * 2f32.powi(-24)
        } else {
            (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15)
        };
        if s == 1 {
            -v
        } else {
            v
        }
    }

    /// half_mm_batched S-shape parity vs CPU: C[z][t][r] = sum_k A[z/g][r][k]*B[z][t][k].
    /// cargo test --release attn_mm_small_parity -- --nocapture --ignored
    #[test]
    #[ignore]
    fn attn_mm_small_parity() {
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;
        let heads = 4usize;
        let group = 2usize;
        let kv_heads = heads / group;
        let n = 70usize; // ragged
        let n_pad = 128usize;
        let hd = 64usize; // PV rows must be 64-aligned (model head_dim = 128)
        let max_pos = 256usize;

        // A = "K": [kv_head][max_pos][hd]
        let mut a = vec![0u16; kv_heads * max_pos * hd];
        for (i, v) in a.iter_mut().enumerate() {
            *v = h(((i % 23) as f32 - 11.0) * 0.07);
        }
        // B = "Q": [token][heads*hd]
        let qdim = heads * hd;
        let mut b = vec![0u16; n_pad * qdim];
        for (i, v) in b.iter_mut().enumerate() {
            *v = h(((i % 19) as f32 - 9.0) * 0.05);
        }
        let a_buf = device.new_buffer_with_data(
            a.as_ptr() as *const _,
            (a.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b.as_ptr() as *const _,
            (b.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = device.new_buffer(
            (heads * n_pad * n_pad * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scalar = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u32;
            *p = hd as u32; // kdim
            *p.add(1) = n_pad as u32; // rows
            *p.add(2) = n_pad as u32; // cols
            *p.add(3) = (max_pos * hd) as u32; // a batch stride
            *p.add(4) = hd as u32; // b batch stride
            *p.add(5) = (n_pad * n_pad) as u32; // c batch stride
            *p.add(6) = hd as u32; // a row stride
            *p.add(7) = qdim as u32; // b row stride
            *p.add(8) = n_pad as u32; // c row stride
            *p.add(9) = 1u32; // a elem stride
            *p.add(10) = group as u32;
            *p.add(11) = 0u32; // no causal culling for the parity check
        }
        let cb = k_ref.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&k_ref.half_mm_batched_pipeline);
        e.set_buffer(0, Some(&a_buf), 0);
        e.set_buffer(1, Some(&b_buf), 0);
        e.set_buffer(2, Some(&c_buf), 0);
        for j in 0..12u64 {
            e.set_buffer(3 + j, Some(&scalar), j * 4);
        }
        e.set_threadgroup_memory_length(0, 8192);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: (n_pad as u64).div_ceil(64),
                height: (n_pad as u64).div_ceil(64),
                depth: heads as u64,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let c = unsafe {
            std::slice::from_raw_parts(c_buf.contents() as *const f32, heads * n_pad * n_pad)
        };
        let mut max_err = 0f32;
        let mut worst = (0, 0, 0, 0f32, 0f32);
        for z in 0..heads {
            for t in (0..n).step_by(7) {
                for r in (0..n).step_by(11) {
                    let mut acc = 0f32;
                    for kk in 0..hd {
                        let av = fh(a[(z / group) * max_pos * hd + r * hd + kk]);
                        let bv = fh(b[t * qdim + z * hd + kk]);
                        acc += av * bv;
                    }
                    let got = c[z * n_pad * n_pad + t * n_pad + r];
                    let err = (got - acc).abs();
                    if err > max_err {
                        max_err = err;
                        worst = (z, t, r, got, acc);
                    }
                }
            }
        }
        eprintln!("[attn-mm-parity] max_err={max_err} worst={worst:?}");
        assert!(max_err < 1e-2, "half_mm_batched mismatch: {worst:?}");

        // ---- full chain: causal softmax + PV vs CPU attention reference ----
        let scale = 0.25f32;
        let p_buf = device.new_buffer(
            (heads * n_pad * n_pad * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let sm_scalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = sm_scalar.contents() as *mut u32;
            *p = n_pad as u32;
            *p.add(1) = n as u32;
            *(p.add(2) as *mut f32) = scale;
        }
        let ctx_buf =
            device.new_buffer((n * qdim * 2) as u64, MTLResourceOptions::StorageModeShared);
        // S through the half-output production variant for the chain.
        let s16_buf = device.new_buffer(
            (heads * n_pad * n_pad * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        {
            let cb = k_ref.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&k_ref.half_mm_batched_f16o_pipeline);
            e.set_buffer(0, Some(&a_buf), 0);
            e.set_buffer(1, Some(&b_buf), 0);
            e.set_buffer(2, Some(&s16_buf), 0);
            for j in 0..12u64 {
                e.set_buffer(3 + j, Some(&scalar), j * 4);
            }
            e.set_threadgroup_memory_length(0, 8192);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (n_pad as u64).div_ceil(64),
                    height: (n_pad as u64).div_ceil(64),
                    depth: heads as u64,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let pv_scalar = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = pv_scalar.contents() as *mut u32;
            *p = n_pad as u32; // kdim
            *p.add(1) = hd as u32; // rows
            *p.add(2) = n as u32; // cols
            *p.add(3) = (max_pos * hd) as u32; // a batch (V)
            *p.add(4) = (n_pad * n_pad) as u32; // b batch (P)
            *p.add(5) = hd as u32; // c batch (ctx head offset)
            *p.add(6) = 1u32; // a row stride (V^T)
            *p.add(7) = n_pad as u32; // b row stride
            *p.add(8) = qdim as u32; // c row stride
            *p.add(9) = hd as u32; // a elem stride (V^T)
            *p.add(10) = group as u32;
            *p.add(11) = 2u32; // causal k-clamp
        }
        // V: reuse `a` pattern as V too (separate buffer to be explicit)
        let v = a.clone();
        let v_buf = device.new_buffer_with_data(
            v.as_ptr() as *const _,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cb = k_ref.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&k_ref.softmax_causal_rows_pipeline);
        e.set_buffer(0, Some(&s16_buf), 0);
        e.set_buffer(1, Some(&p_buf), 0);
        e.set_buffer(2, Some(&sm_scalar), 0);
        e.set_buffer(3, Some(&sm_scalar), 4);
        e.set_buffer(4, Some(&sm_scalar), 8);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: heads as u64,
                height: (n_pad as u64).div_ceil(8),
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        e.set_compute_pipeline_state(&k_ref.half_mm_batched_f16o_pipeline);
        e.set_buffer(0, Some(&v_buf), 0);
        e.set_buffer(1, Some(&p_buf), 0);
        e.set_buffer(2, Some(&ctx_buf), 0);
        for j in 0..12u64 {
            e.set_buffer(3 + j, Some(&pv_scalar), j * 4);
        }
        e.set_threadgroup_memory_length(0, 8192);
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: (hd as u64).div_ceil(64),
                height: (n as u64).div_ceil(64),
                depth: heads as u64,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let ctx16 =
            unsafe { std::slice::from_raw_parts(ctx_buf.contents() as *const u16, n * qdim) };
        let ctx: Vec<f32> = ctx16.iter().map(|&b| fh(b)).collect();
        let mut max_err2 = 0f32;
        let mut worst2 = (0, 0, 0, 0f32, 0f32);
        for z in 0..heads {
            for q in (0..n).step_by(5) {
                // CPU reference attention for row q
                let mut scores = vec![0f32; q + 1];
                let mut m = f32::NEG_INFINITY;
                for p2 in 0..=q {
                    let mut acc = 0f32;
                    for kk in 0..hd {
                        let kv = fh(a[(z / group) * max_pos * hd + p2 * hd + kk]);
                        let qv = fh(b[q * qdim + z * hd + kk]);
                        acc += kv * qv;
                    }
                    scores[p2] = acc * scale;
                    m = m.max(scores[p2]);
                }
                let l: f32 = scores.iter().map(|s| (s - m).exp()).sum();
                for d in (0..hd).step_by(3) {
                    let mut o = 0f32;
                    for p2 in 0..=q {
                        let pw = ((scores[p2] - m).exp() / l) as f32;
                        // kernel rounds P to half
                        let pw = fh(h(pw));
                        o += pw * fh(v[(z / group) * max_pos * hd + p2 * hd + d]);
                    }
                    let got = ctx[q * qdim + z * hd + d];
                    let err = (got - o).abs();
                    if err > max_err2 {
                        max_err2 = err;
                        worst2 = (z, q, d, got, o);
                    }
                }
            }
        }
        eprintln!("[attn-chain-parity] max_err={max_err2} worst={worst2:?}");
        assert!(max_err2 < 5e-2, "chain mismatch: {worst2:?}");
    }
}

#[cfg(all(test, target_os = "macos"))]
mod steel_probe {
    use super::*;

    /// Steel-architecture wire-Q8 GEMM probe: 32x32x32 tiles, 2x2 simdgroups, padded
    /// row-major threadgroup tiles, per-LANE fragment element loads into
    /// simdgroup_matrix thread storage (no simdgroup_load), per-lane stores.
    /// cargo test --release steel_mm_probe_gate -- --nocapture --ignored
    #[test]
    #[ignore]
    fn steel_mm_probe_gate() {
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;
        let src = r#"
#include <metal_stdlib>
using namespace metal;

// Per-lane coordinate inside an 8x8 simdgroup fragment (hardware mapping).
static inline short2 frag_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2(fn, fm);
}

kernel void steel_q8_mm(
    device const half* x [[buffer(0)]],          // [M tokens][K] half
    device const char* w [[buffer(1)]],          // wire Q8: [N rows][K/32 blocks * 34B]
    device float* y [[buffer(2)]],               // [M][N] f32
    constant uint& kdim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    constant uint& m_tokens [[buffer(5)]],
    uint3 tid3 [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 32;       // tokens
    constexpr uint BN = 32;       // weight rows
    constexpr uint BK = 32;       // one Q8 block per row per step
    constexpr uint LD = BK + 8;   // padded leading dim
    const uint tid = sg * 32 + lane;

    threadgroup half Xs[BM * LD];
    threadgroup half Ws[BN * LD];

    const uint n0 = tid3.x * BN;
    const uint m0 = tid3.y * BM;
    const uint row_stride = (kdim / 32) * 34;

    // Loader mapping: thread -> (row bi, col bj), one 8-wide segment each.
    const uint bi = tid / 4;
    const uint bj = (tid % 4) * 8;
    device const half* xs_src = x + (m0 + bi) * kdim + bj;
    device const char* w_row = w + (ulong)(n0 + bi) * row_stride;

    // MMA thread placement (2x2 warp grid, interleaved fragment tiling).
    const short tm = 8 * (short)(sg / 2);
    const short tn = 8 * (short)(sg % 2);
    const short2 c = frag_coord((ushort)lane);
    const short sm = c.y;
    const short sn = c.x;
    // A (X tile): str_m = LD, str_k = 1 ; B (W tile, transposed use): str_k = 1, str_n = LD
    const uint a_off = (uint)(tm + sm) * LD + (uint)sn;
    const uint b_off = (uint)sm + (uint)(tn + sn) * LD;

    simdgroup_half8x8 a_frag[2];
    simdgroup_half8x8 b_frag[2];
    simdgroup_float8x8 c_frag[2][2];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            c_frag[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint kb = 0; kb < kdim; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // X: one vectorized 16B copy per thread.
        *reinterpret_cast<threadgroup half4*>(&Xs[bi * LD + bj]) =
            *reinterpret_cast<device const half4*>(xs_src + kb);
        *reinterpret_cast<threadgroup half4*>(&Xs[bi * LD + bj + 4]) =
            *reinterpret_cast<device const half4*>(xs_src + kb + 4);
        // W: dequantize 8 values of this row's block kb/32, fully vectorized
        // (char4 -> float4 -> half4, two 8-byte stores into the padded row).
        {
            device const char* wb = w_row + (kb / 32) * 34;
            const float scale = float(*reinterpret_cast<device const half*>(wb));
            device const packed_char4* q =
                reinterpret_cast<device const packed_char4*>(wb + 2 + bj);
            const char4 q0 = q[0];
            const char4 q1 = q[1];
            *reinterpret_cast<threadgroup half4*>(&Ws[bi * LD + bj]) =
                half4(float4(q0) * scale);
            *reinterpret_cast<threadgroup half4*>(&Ws[bi * LD + bj + 4]) =
                half4(float4(q1) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            // Per-lane fragment element loads (vec<half,2> per frag per lane).
            {
                threadgroup const half* a0 = &Xs[a_off + kk];
                a_frag[0].thread_elements()[0] = a0[0];
                a_frag[0].thread_elements()[1] = a0[1];
                threadgroup const half* a1 = a0 + 16 * LD;
                a_frag[1].thread_elements()[0] = a1[0];
                a_frag[1].thread_elements()[1] = a1[1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            {
                threadgroup const half* b0 = &Ws[b_off + kk];
                b_frag[0].thread_elements()[0] = b0[0];
                b_frag[0].thread_elements()[1] = b0[LD];
                threadgroup const half* b1 = b0 + 16 * LD;
                b_frag[1].thread_elements()[0] = b1[0];
                b_frag[1].thread_elements()[1] = b1[LD];
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
            simdgroup_multiply_accumulate(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
            simdgroup_multiply_accumulate(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
            simdgroup_multiply_accumulate(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
        }
    }

    // Per-lane element stores: C[m][n], m = m0+tm+sm(+16i), n = n0+tn+sn(+16j).
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            const uint m = m0 + (uint)(tm + sm) + 16 * i;
            const uint n = n0 + (uint)(tn + sn) + 16 * j;
            if (m < m_tokens) {
                device float* dst = y + (ulong)m * n_rows + n;
                dst[0] = c_frag[i][j].thread_elements()[0];
                dst[1] = c_frag[i][j].thread_elements()[1];
            }
        }
    }
}
"#;
        let options = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .map_err(|e| panic!("steel probe compile: {e}"))
            .unwrap();
        let f = lib.get_function("steel_q8_mm", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();

        let rows: usize = 8192;
        let k: usize = 3072;
        let m: usize = 601;
        let m_pad: usize = m.next_multiple_of(32);
        let bpr = k / 32;
        let mut wire = vec![0u8; rows * bpr * 34];
        for (i, chunk) in wire.chunks_mut(34).enumerate() {
            let scale = f32_to_f16_bits(0.01 + (i % 7) as f32 * 0.001);
            chunk[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, q) in chunk[2..].iter_mut().enumerate() {
                *q = (((i + j * 3) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let w_buf = device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut xv = vec![0u16; m_pad * k];
        for (i, v) in xv.iter_mut().enumerate() {
            *v = f32_to_f16_bits(((i % 31) as f32 - 15.0) * 0.05);
        }
        let x_buf = device.new_buffer_with_data(
            xv.as_ptr() as *const _,
            (xv.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer(
            (m_pad * rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        unsafe {
            let s = scalar.contents() as *mut u32;
            *s = k as u32;
            *s.add(1) = rows as u32;
            *s.add(2) = m as u32;
        }
        // correctness spot-check once
        let run = |label: &str| {
            for round in 0..4 {
                let cb = k_ref.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&p);
                e.set_buffer(0, Some(&x_buf), 0);
                e.set_buffer(1, Some(&w_buf), 0);
                e.set_buffer(2, Some(&y_buf), 0);
                e.set_buffer(3, Some(&scalar), 0);
                e.set_buffer(4, Some(&scalar), 4);
                e.set_buffer(5, Some(&scalar), 8);
                e.dispatch_thread_groups(
                    metal::MTLSize {
                        width: (rows / 32) as u64,
                        height: (m_pad / 32) as u64,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
                let flops = 2.0 * rows as f64 * k as f64 * m as f64;
                eprintln!(
                    "[steel-probe] {label} round {round}: {busy_us}us  {:.2} TFLOPS",
                    flops / (busy_us as f64 * 1e-6) / 1e12
                );
            }
        };
        run("gate-shape");
        // CPU spot check a few outputs
        let y = unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, m_pad * rows) };
        let fh = |b: u16| -> f32 {
            let e = ((b >> 10) & 0x1f) as i32;
            let mant = (b & 0x3ff) as i32;
            let v = if e == 0 {
                (mant as f32) * 2f32.powi(-24)
            } else {
                (1.0 + mant as f32 / 1024.0) * 2f32.powi(e - 15)
            };
            if b >> 15 == 1 {
                -v
            } else {
                v
            }
        };
        let mut max_err = 0f32;
        for mi in (0..m).step_by(97) {
            for n in (0..rows).step_by(513) {
                let mut acc = 0f32;
                for kk in 0..k {
                    let blk = &wire[(n * bpr + kk / 32) * 34..];
                    let scale = fh(u16::from_le_bytes([blk[0], blk[1]]));
                    let q = blk[2 + kk % 32] as i8;
                    acc += fh(xv[mi * k + kk]) * (q as f32 * scale);
                }
                let err = (y[mi * rows + n] - acc).abs() / acc.abs().max(1.0);
                max_err = max_err.max(err);
            }
        }
        eprintln!("[steel-probe] rel max_err vs CPU: {max_err}");
        assert!(max_err < 2e-2, "steel mm mismatch");
    }
}

#[cfg(all(test, target_os = "macos"))]
mod steel_dual_probe {
    use super::*;

    /// Dual interleaved steel-shape GEMM: two weight matrices (gate+up) against ONE
    /// shared activation tile per staging round — doubles MMA work per barrier.
    /// cargo test --release steel_dual_probe_gate -- --nocapture --ignored
    #[test]
    #[ignore]
    fn steel_dual_probe_gate() {
        let k_ref = metal_linear_kernel().expect("metal");
        let device = &k_ref.device;
        let src = r#"
#include <metal_stdlib>
using namespace metal;
static inline short2 frag_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2(fn, fm);
}
kernel void steel_q8_mm_dual(
    device const half* x [[buffer(0)]],
    device const char* w0 [[buffer(1)]],
    device const char* w1 [[buffer(2)]],
    device float* y0 [[buffer(3)]],
    device float* y1 [[buffer(4)]],
    constant uint& kdim [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& m_tokens [[buffer(7)]],
    uint3 tid3 [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 32;
    constexpr uint BN = 32;
    constexpr uint BK = 32;
    constexpr uint LD = BK + 8;
    const uint tid = sg * 32 + lane;

    threadgroup half Xs[BM * LD];
    threadgroup half Ws0[BN * LD];
    threadgroup half Ws1[BN * LD];

    const uint n0 = tid3.x * BN;
    const uint m0 = tid3.y * BM;
    const uint row_stride = (kdim / 32) * 34;

    const uint bi = tid / 4;
    const uint bj = (tid % 4) * 8;
    device const half* xs_src = x + (m0 + bi) * kdim + bj;
    device const char* w0_row = w0 + (ulong)(n0 + bi) * row_stride;
    device const char* w1_row = w1 + (ulong)(n0 + bi) * row_stride;

    const short tm = 8 * (short)(sg / 2);
    const short tn = 8 * (short)(sg % 2);
    const short2 c = frag_coord((ushort)lane);
    const short sm = c.y;
    const short sn = c.x;
    const uint a_off = (uint)(tm + sm) * LD + (uint)sn;
    const uint b_off = (uint)sm + (uint)(tn + sn) * LD;

    simdgroup_half8x8 a_frag[2];
    simdgroup_half8x8 b_frag0[2];
    simdgroup_half8x8 b_frag1[2];
    simdgroup_float8x8 c0[2][2];
    simdgroup_float8x8 c1[2][2];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            c0[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            c1[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint kb = 0; kb < kdim; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup half4*>(&Xs[bi * LD + bj]) =
            *reinterpret_cast<device const half4*>(xs_src + kb);
        *reinterpret_cast<threadgroup half4*>(&Xs[bi * LD + bj + 4]) =
            *reinterpret_cast<device const half4*>(xs_src + kb + 4);
        {
            device const char* wb = w0_row + (kb / 32) * 34;
            const float scale = float(*reinterpret_cast<device const half*>(wb));
            device const packed_char4* q =
                reinterpret_cast<device const packed_char4*>(wb + 2 + bj);
            *reinterpret_cast<threadgroup half4*>(&Ws0[bi * LD + bj]) =
                half4(float4(q[0]) * scale);
            *reinterpret_cast<threadgroup half4*>(&Ws0[bi * LD + bj + 4]) =
                half4(float4(q[1]) * scale);
        }
        {
            device const char* wb = w1_row + (kb / 32) * 34;
            const float scale = float(*reinterpret_cast<device const half*>(wb));
            device const packed_char4* q =
                reinterpret_cast<device const packed_char4*>(wb + 2 + bj);
            *reinterpret_cast<threadgroup half4*>(&Ws1[bi * LD + bj]) =
                half4(float4(q[0]) * scale);
            *reinterpret_cast<threadgroup half4*>(&Ws1[bi * LD + bj + 4]) =
                half4(float4(q[1]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            {
                threadgroup const half* a0 = &Xs[a_off + kk];
                a_frag[0].thread_elements()[0] = a0[0];
                a_frag[0].thread_elements()[1] = a0[1];
                threadgroup const half* a1 = a0 + 16 * LD;
                a_frag[1].thread_elements()[0] = a1[0];
                a_frag[1].thread_elements()[1] = a1[1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            {
                threadgroup const half* p0 = &Ws0[b_off + kk];
                b_frag0[0].thread_elements()[0] = p0[0];
                b_frag0[0].thread_elements()[1] = p0[LD];
                threadgroup const half* p1 = p0 + 16 * LD;
                b_frag0[1].thread_elements()[0] = p1[0];
                b_frag0[1].thread_elements()[1] = p1[LD];
                threadgroup const half* q0 = &Ws1[b_off + kk];
                b_frag1[0].thread_elements()[0] = q0[0];
                b_frag1[0].thread_elements()[1] = q0[LD];
                threadgroup const half* q1 = q0 + 16 * LD;
                b_frag1[1].thread_elements()[0] = q1[0];
                b_frag1[1].thread_elements()[1] = q1[LD];
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c0[0][0], a_frag[0], b_frag0[0], c0[0][0]);
            simdgroup_multiply_accumulate(c0[0][1], a_frag[0], b_frag0[1], c0[0][1]);
            simdgroup_multiply_accumulate(c0[1][0], a_frag[1], b_frag0[0], c0[1][0]);
            simdgroup_multiply_accumulate(c0[1][1], a_frag[1], b_frag0[1], c0[1][1]);
            simdgroup_multiply_accumulate(c1[0][0], a_frag[0], b_frag1[0], c1[0][0]);
            simdgroup_multiply_accumulate(c1[0][1], a_frag[0], b_frag1[1], c1[0][1]);
            simdgroup_multiply_accumulate(c1[1][0], a_frag[1], b_frag1[0], c1[1][0]);
            simdgroup_multiply_accumulate(c1[1][1], a_frag[1], b_frag1[1], c1[1][1]);
        }
    }

    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            const uint m = m0 + (uint)(tm + sm) + 16 * i;
            const uint n = n0 + (uint)(tn + sn) + 16 * j;
            if (m < m_tokens) {
                device float* d0 = y0 + (ulong)m * n_rows + n;
                d0[0] = c0[i][j].thread_elements()[0];
                d0[1] = c0[i][j].thread_elements()[1];
                device float* d1 = y1 + (ulong)m * n_rows + n;
                d1[0] = c1[i][j].thread_elements()[0];
                d1[1] = c1[i][j].thread_elements()[1];
            }
        }
    }
}
"#;
        let options = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .map_err(|e| panic!("dual probe compile: {e}"))
            .unwrap();
        let f = lib.get_function("steel_q8_mm_dual", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();

        let rows: usize = 8192;
        let k: usize = 3072;
        let m: usize = 601;
        let m_pad: usize = m.next_multiple_of(32);
        let bpr = k / 32;
        let mut wire = vec![0u8; rows * bpr * 34];
        for (i, chunk) in wire.chunks_mut(34).enumerate() {
            let scale = f32_to_f16_bits(0.01 + (i % 7) as f32 * 0.001);
            chunk[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, q) in chunk[2..].iter_mut().enumerate() {
                *q = (((i + j * 3) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let w0 = device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w1 = device.new_buffer_with_data(
            wire.as_ptr() as *const _,
            wire.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut xv = vec![0u16; m_pad * k];
        for (i, v) in xv.iter_mut().enumerate() {
            *v = f32_to_f16_bits(((i % 31) as f32 - 15.0) * 0.05);
        }
        let x_buf = device.new_buffer_with_data(
            xv.as_ptr() as *const _,
            (xv.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y0 = device.new_buffer(
            (m_pad * rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y1 = device.new_buffer(
            (m_pad * rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        unsafe {
            let s = scalar.contents() as *mut u32;
            *s = k as u32;
            *s.add(1) = rows as u32;
            *s.add(2) = m as u32;
        }
        for round in 0..4 {
            let cb = k_ref.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&x_buf), 0);
            e.set_buffer(1, Some(&w0), 0);
            e.set_buffer(2, Some(&w1), 0);
            e.set_buffer(3, Some(&y0), 0);
            e.set_buffer(4, Some(&y1), 0);
            e.set_buffer(5, Some(&scalar), 0);
            e.set_buffer(6, Some(&scalar), 4);
            e.set_buffer(7, Some(&scalar), 8);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: (rows / 32) as u64,
                    height: (m_pad / 32) as u64,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let (busy_us, _) = command_buffer_gpu_times_us(&cb.to_owned());
            // flops for BOTH matmuls
            let flops = 2.0 * 2.0 * rows as f64 * k as f64 * m as f64;
            eprintln!(
                "[dual-probe] round {round}: {busy_us}us  {:.2} TFLOPS combined ({:.1}ms vs 2x single ~17.7ms)",
                flops / (busy_us as f64 * 1e-6) / 1e12,
                busy_us as f64 / 1000.0
            );
        }
    }
}
