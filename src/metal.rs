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
    rms_norm_pipeline: ComputePipelineState,
    residual_add_pipeline: ComputePipelineState,
    silu_mul_pipeline: ComputePipelineState,
    rope_rotate_pipeline: ComputePipelineState,
    attention_decode_pipeline: ComputePipelineState,
    quantize_q8_0_pipeline: ComputePipelineState,
    kv_scatter_pipeline: ComputePipelineState,
    rms_norm_quantize_pipeline: ComputePipelineState,
    silu_mul_quantize_pipeline: ComputePipelineState,
    active_command_buffer: Mutex<Option<metal::CommandBuffer>>,
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
}

#[cfg(target_os = "macos")]
static METAL_LINEAR_KERNEL: OnceLock<Option<MetalLinearKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_LINEAR_CACHE: OnceLock<Mutex<MetalLinearCache>> = OnceLock::new();

#[cfg(target_os = "macos")]
const LINEAR_ROW_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
    uint i [[thread_position_in_grid]]
) {
    if (i >= total) return;
    uint h = i / head_dim;
    uint d = i % head_dim;
    uint dst = (h * max_positions + write_position) * head_dim + d;
    cache_k[dst] = src_k[i];
    cache_v[dst] = src_v[i];
}
"#;

#[cfg(target_os = "macos")]
fn metal_linear_kernel() -> Option<&'static MetalLinearKernel> {
    METAL_LINEAR_KERNEL
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(LINEAR_ROW_SHADER, &options)
                .ok()?;
            let elementwise_library = device
                .new_library_with_source(ELEMENTWISE_SHADER, &options)
                .ok()?;
            let rms_norm_function = elementwise_library
                .get_function("rms_norm_f32", None)
                .ok()?;
            let rms_norm_pipeline = device
                .new_compute_pipeline_state_with_function(&rms_norm_function)
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
                rms_norm_pipeline,
                residual_add_pipeline,
                silu_mul_pipeline,
                rope_rotate_pipeline,
                attention_decode_pipeline,
                quantize_q8_0_pipeline,
                kv_scatter_pipeline,
                rms_norm_quantize_pipeline,
                silu_mul_quantize_pipeline,
                active_command_buffer: Mutex::new(None),
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
    // SIMD-group GEMV with several rows per threadgroup (one row per SIMD group): keeps more
    // weight reads in flight per threadgroup for better memory-latency hiding on the
    // bandwidth-bound decode path.
    const SIMD_GROUPS_PER_TG: u64 = 2;
    e.set_compute_pipeline_state(&k.q8_0_block_simd_mr_pipeline);
    e.set_buffer(0, Some(scales), 0);
    e.set_buffer(1, Some(quants), 0);
    e.set_buffer(2, Some(weight), 0);
    e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(scalar), 0);
    e.set_buffer(5, Some(scalar), 4);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows as u64).div_ceil(SIMD_GROUPS_PER_TG),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: SIMD_GROUPS_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
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
) {
    e.set_compute_pipeline_state(&k.rope_rotate_pipeline);
    e.set_buffer(0, Some(data), 0);
    e.set_buffer(1, Some(cos_t), 0);
    e.set_buffer(2, Some(sin_t), 0);
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
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    scores: &Buffer,
    out: &Buffer,
    scalar: &Buffer,
    n_heads: usize,
) {
    e.set_compute_pipeline_state(&k.attention_decode_pipeline);
    e.set_buffer(0, Some(query), 0);
    e.set_buffer(1, Some(keys), 0);
    e.set_buffer(2, Some(values), 0);
    e.set_buffer(3, Some(scores), 0);
    e.set_buffer(4, Some(out), 0);
    e.set_buffer(5, Some(scalar), 0); // n_heads
    e.set_buffer(6, Some(scalar), 4); // head_dim
    e.set_buffer(7, Some(scalar), 8); // position_count
    e.set_buffer(8, Some(scalar), 12); // group
    e.set_buffer(9, Some(scalar), 16); // scale (f32)
    e.set_buffer(10, Some(scalar), 20); // position_stride
    e.set_buffer(11, Some(scalar), 24); // kv_head_stride
    e.set_buffer(12, Some(scalar), 28); // kv_base_offset
                                        // One threadgroup per head, a single 32-lane SIMD group cooperating within it.
    e.dispatch_thread_groups(
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
    let nb = |bytes: u64| {
        k.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
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
    q_w_buf: &Buffer,
    k_w_buf: &Buffer,
    v_w_buf: &Buffer,
    o_w_buf: &Buffer,
    cos_t: &[f32],
    sin_t: &[f32],
    cache_k_buf: &Buffer,
    cache_v_buf: &Buffer,
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
    let nb = |bytes: u64| {
        k.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
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
    encode_rope(
        e,
        k,
        &query_buf,
        &cos_buf,
        &sin_buf,
        &rope_q_scalar,
        n_heads,
        half_rope,
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
    e.set_compute_pipeline_state(&k.kv_scatter_pipeline);
    e.set_buffer(0, Some(&key_buf), 0);
    e.set_buffer(1, Some(&val_buf), 0);
    e.set_buffer(2, Some(cache_k_buf), 0);
    e.set_buffer(3, Some(cache_v_buf), 0);
    e.set_buffer(4, Some(&scatter_scalar), 0);
    e.set_buffer(5, Some(&scatter_scalar), 4);
    e.set_buffer(6, Some(&scatter_scalar), 8);
    e.set_buffer(7, Some(&scatter_scalar), 12);
    dispatch_1d(e, &k.kv_scatter_pipeline, kv_dim);
    encode_attention(
        e,
        k,
        &query_buf,
        cache_k_buf,
        cache_v_buf,
        &scores_buf,
        &ctx_buf,
        &attn_scalar,
        n_heads,
    );
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
        &q_w,
        &k_w,
        &v_w,
        &o_w,
        cos_t,
        sin_t,
        &cache_k_buf,
        &cache_v_buf,
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
        &q_w,
        &k_w,
        &v_w,
        &o_w,
        cos_t,
        sin_t,
        &cache_k_buf,
        &cache_v_buf,
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
            &w[0],
            &w[1],
            &w[2],
            &w[3],
            cos_t,
            sin_t,
            &cache_k_buf,
            &cache_v_buf,
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
    pub q_weight_blocks: &'a [u8],
    pub k_weight_blocks: &'a [u8],
    pub v_weight_blocks: &'a [u8],
    pub o_weight_blocks: &'a [u8],
    pub gate_weight_blocks: &'a [u8],
    pub up_weight_blocks: &'a [u8],
    pub down_weight_blocks: &'a [u8],
}

/// Optional final stage for `forward_token`: when present, the session also runs the final
/// RMSNorm + output (vocab) projection on the GPU in the same command buffer and returns the
/// `[vocab_size]` logits instead of the hidden state — keeping the large output matmul off the
/// CPU. `output_weight_blocks` is the Q8_0 output/embedding projection.
pub struct LogitsStage<'a> {
    pub final_norm: &'a [f32],
    pub output_weight_blocks: &'a [u8],
    pub vocab_size: usize,
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
    buf_a: Buffer,
    buf_b: Buffer,
    mid: Buffer,
    /// Number of KV positions currently materialized in the cache (seeded history + appended
    /// tokens). The caller uses this to detect a new sequence and reseed.
    filled: usize,
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
        let kv_slots = n_kv_heads * max_positions * head_dim;
        let cache_k = (0..n_layers).map(|_| nb(kv_slots)).collect();
        let cache_v = (0..n_layers).map(|_| nb(kv_slots)).collect();
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
            buf_a: nb(hidden),
            buf_b: nb(hidden),
            mid: nb(hidden),
            filled: 0,
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
        let kv_slots = self.n_kv_heads * new_max * self.head_dim;
        let new_k: Vec<Buffer> = (0..self.n_layers)
            .map(|_| {
                k.device
                    .new_buffer((kv_slots * 4) as u64, MTLResourceOptions::StorageModeShared)
            })
            .collect();
        let new_v: Vec<Buffer> = (0..self.n_layers)
            .map(|_| {
                k.device
                    .new_buffer((kv_slots * 4) as u64, MTLResourceOptions::StorageModeShared)
            })
            .collect();
        if self.filled > 0 {
            let run = (self.filled * self.head_dim * 4) as u64;
            let cb = k.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for layer in 0..self.n_layers {
                for h in 0..self.n_kv_heads {
                    let src = (h * self.max_positions * self.head_dim * 4) as u64;
                    let dst = (h * new_max * self.head_dim * 4) as u64;
                    blit.copy_from_buffer(&self.cache_k[layer], src, &new_k[layer], dst, run);
                    blit.copy_from_buffer(&self.cache_v[layer], src, &new_v[layer], dst, run);
                }
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        self.cache_k = new_k;
        self.cache_v = new_v;
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

    /// Decode one token at sequence position `position` (0-based): append this token's K/V to
    /// the persistent GPU cache and attend over positions `0..=position`, all layers in one
    /// command buffer. `cos_t`/`sin_t` are the RoPE tables for `position`; `scale` is the
    /// attention scale. Returns the post-final-layer hidden state `[hidden]` (the caller
    /// applies the final norm + output projection), or None on dimension mismatch.
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
    ) -> Option<Vec<f32>> {
        let half_rope = cos_t.len();
        // Grow the on-GPU KV cache if this token's position is past the current capacity.
        if !self.ensure_capacity(position + 1) {
            return None;
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
                || l.q_weight_blocks.len() != q_dim * bpr_hidden * 36
                || l.k_weight_blocks.len() != kv_dim * bpr_hidden * 36
                || l.v_weight_blocks.len() != kv_dim * bpr_hidden * 36
                || l.o_weight_blocks.len() != self.hidden * bpr_q * 36
                || l.gate_weight_blocks.len() != self.ffn_dim * bpr_hidden * 36
                || l.up_weight_blocks.len() != self.ffn_dim * bpr_hidden * 36
                || l.down_weight_blocks.len() != self.hidden * bpr_ffn * 36
            {
                return None;
            }
        }
        if let Some(s) = &logits_stage {
            if s.vocab_size == 0
                || s.final_norm.len() != self.hidden
                || s.output_weight_blocks.len() != s.vocab_size * bpr_hidden * 36
            {
                return None;
            }
        }
        let k = metal_linear_kernel()?;
        // Resolve all resident weight buffers (layer weights + optional output stage) under one
        // cache lock. They are keyed by (pointer, len), so they upload once and persist.
        let resident: Vec<[Buffer; 7]>;
        let stage_bufs: Option<(Buffer, Buffer)>;
        {
            let mut cache = metal_linear_cache().lock().ok()?;
            resident = layers
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
                .collect();
            stage_bufs = logits_stage.as_ref().map(|s| {
                (
                    cache.q8_block_weight_buffer(&k.device, s.output_weight_blocks),
                    cache.weight_buffer(&k.device, s.final_norm),
                )
            });
        }
        let position_count = position + 1;
        write_buffer_f32(&self.buf_a, embedding);
        let mut keep = Vec::new();
        let trace = std::env::var_os("CAMELID_RESIDENT_TRACE").is_some();
        let encode_started = std::time::Instant::now();
        let cb = k.queue.new_command_buffer();
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
                &w[0],
                &w[1],
                &w[2],
                &w[3],
                cos_t,
                sin_t,
                &self.cache_k[i],
                &self.cache_v[i],
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
        let logits_buf = if let (Some(s), Some((ow_buf, fnorm_buf))) = (&logits_stage, &stage_bufs)
        {
            let nb = |bytes: u64| {
                k.device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
            };
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
            encode_rms_norm_quantize(e, k, final_buf, fnorm_buf, &lscales, &lquants, &rms_scalar);
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
            keep.extend([rms_scalar, lscales, lquants, lmm_scalar]);
            Some(logits_buf)
        } else {
            None
        };
        let encode_us = encode_started.elapsed().as_micros();
        let gpu_started = std::time::Instant::now();
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if trace {
            let gpu_us = gpu_started.elapsed().as_micros();
            eprintln!(
                "[resident] pos={position} layers={} encode={encode_us}us commit_wait={gpu_us}us",
                self.n_layers
            );
        }
        self.filled = position + 1;
        if let Some(logits_buf) = logits_buf {
            let vocab = logits_stage.as_ref().map(|s| s.vocab_size).unwrap_or(0);
            let mut out = vec![0.0f32; vocab];
            read_buffer_f32(&logits_buf, &mut out);
            return Some(out);
        }
        let mut out = vec![0.0f32; self.hidden];
        read_buffer_f32(final_buf, &mut out);
        Some(out)
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
        );
        Self::seed_into(
            &self.cache_v[layer],
            values,
            self.n_kv_heads,
            self.max_positions,
            self.head_dim,
            seed_positions,
        );
        true
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
    ) {
        let run = seed_positions * head_dim;
        // SAFETY: shared-storage buffer of n_kv_heads*max_positions*head_dim f32; each head's
        // `run` floats land at a disjoint slot well within that capacity.
        unsafe {
            let dst = buf.contents() as *mut f32;
            for h in 0..n_kv_heads {
                let s = h * seed_positions * head_dim;
                let d = h * max_positions * head_dim;
                std::ptr::copy_nonoverlapping(src[s..s + run].as_ptr(), dst.add(d), run);
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
    pub fn forward_token(
        &mut self,
        _embedding: &[f32],
        _layers: &[ResidentLayerWeights],
        _cos_t: &[f32],
        _sin_t: &[f32],
        _position: usize,
        _scale: f32,
        _logits_stage: Option<LogitsStage>,
    ) -> Option<Vec<f32>> {
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

    pub fn filled(&self) -> usize {
        0
    }

    pub fn set_filled(&mut self, _n: usize) {}
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
                q_weight_blocks: &d.q,
                k_weight_blocks: &d.k,
                v_weight_blocks: &d.v,
                o_weight_blocks: &d.o,
                gate_weight_blocks: &d.gate,
                up_weight_blocks: &d.up,
                down_weight_blocks: &d.down,
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

            let got = session
                .forward_token(&emb, &weights, &cos_t, &sin_t, t, scale, None)
                .unwrap();
            assert_eq!(got.len(), hidden);
            for (a, b) in got.iter().zip(&expected) {
                assert!((a - b).abs() < 1.0e-4, "token {t}: {a} != {b}");
            }
        }
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
