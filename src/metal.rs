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
    rms_norm_pipeline: ComputePipelineState,
    residual_add_pipeline: ComputePipelineState,
    silu_mul_pipeline: ComputePipelineState,
    rope_rotate_pipeline: ComputePipelineState,
    attention_decode_pipeline: ComputePipelineState,
    quantize_q8_0_pipeline: ComputePipelineState,
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
// head -> kv head by integer group (group = n_heads / n_kv_heads). Keys/values are
// laid out [kv_head][position][head_dim]; scores is scratch [n_heads*position_count].
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
    uint head [[thread_position_in_grid]]
) {
    if (head >= n_heads) return;
    uint kv_head = head / group;
    uint q_base = head * head_dim;
    uint kv_base = kv_head * position_count * head_dim;
    uint score_base = head * position_count;

    float max_score = -INFINITY;
    for (uint p = 0; p < position_count; ++p) {
        float s = 0.0;
        uint k_base = kv_base + p * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * keys[k_base + d];
        }
        s *= scale;
        scores[score_base + p] = s;
        max_score = max(max_score, s);
    }
    float sum = 0.0;
    for (uint p = 0; p < position_count; ++p) {
        float e = exp(scores[score_base + p] - max_score);
        scores[score_base + p] = e;
        sum += e;
    }
    float inv = 1.0 / sum;
    for (uint d = 0; d < head_dim; ++d) {
        output[q_base + d] = 0.0;
    }
    for (uint p = 0; p < position_count; ++p) {
        float prob = scores[score_base + p] * inv;
        uint v_base = kv_base + p * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            output[q_base + d] += prob * values[v_base + d];
        }
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
                rms_norm_pipeline,
                residual_add_pipeline,
                silu_mul_pipeline,
                rope_rotate_pipeline,
                attention_decode_pipeline,
                quantize_q8_0_pipeline,
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
/// attention_context_for_head_into.
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
    if n_heads == 0
        || n_kv_heads == 0
        || head_dim == 0
        || position_count == 0
        || !n_heads.is_multiple_of(n_kv_heads)
        || query.len() != n_heads * head_dim
        || keys.len() != n_kv_heads * position_count * head_dim
        || values.len() != keys.len()
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
    let scalar_buf = new(20);
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
    let width = kernel
        .attention_decode_pipeline
        .thread_execution_width()
        .max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (n_heads as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
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

#[cfg(not(target_os = "macos"))]
pub fn try_quantized_matmul_resident(
    _input: &[f32],
    _weight_blocks: &[u8],
    _output_width: usize,
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
