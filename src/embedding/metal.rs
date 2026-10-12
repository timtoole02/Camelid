//! Bounded Metal layer stack for Knowledge Library document embeddings.
//!
//! Q8 block dots preserve the CPU's integer sum and block accumulation order.
//! RoPE uses host-computed tables. Metal has no f64 arithmetic: residual
//! LayerNorm runs through the shared CPU f64 implementation over completed
//! unified-memory buffers. Tokenization, embedding rows, initial LayerNorm
//! and pooling also stay shared with the CPU implementation.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};

use metal::foreign_types::ForeignType;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::NomicBertRuntime;
use crate::tensor::CpuTensor;
use crate::Result;

// At the pinned geometry a group needs under 160 MiB of activation/scratch
// storage. Larger inputs use the CPU; a long query cannot permanently disable
// the GPU for normal chunks. Document batches are split without padding.
const MAX_ROWS: usize = 1024;

const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void encoder_quantize_q8(
    device const float* input [[buffer(0)]],
    device float* scales [[buffer(1)]],
    device char* quants [[buffer(2)]],
    constant uint& blocks [[buffer(3)]],
    uint b [[thread_position_in_grid]]
) {
    if (b >= blocks) return;
    uint base = b * 32u;
    float maximum = 0.0f;
    for (uint i = 0; i < 32u; i++) maximum = max(maximum, fabs(input[base + i]));
    float unrounded = maximum / 127.0f;
    scales[b] = float(half(unrounded));
    float inv = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    for (uint i = 0; i < 32u; i++) {
        quants[base + i] = char(clamp(int(round(input[base + i] * inv)), -128, 127));
    }
}

// One thread per output; no floating reductions across lanes. Integer dots
// are exact, followed by the same two multiplies and block-ordered sum as CPU.
kernel void encoder_q8_gemm(
    device const float* xs [[buffer(0)]],
    device const char* xq [[buffer(1)]],
    device const float* ws [[buffer(2)]],
    device const char* wq [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& outs [[buffer(6)]],
    constant uint& blocks [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= rows * outs) return;
    uint row = idx / outs, o = idx % outs;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        uint xb = (row * blocks + b) * 32u;
        uint wb = (o * blocks + b) * 32u;
        int dot = 0;
        for (uint i = 0; i < 32u; i++) dot += int(xq[xb + i]) * int(wq[wb + i]);
        acc = acc + (float(dot) * ws[o * blocks + b]) * xs[row * blocks + b];
    }
    out[idx] = acc;
}

kernel void encoder_rope_split(
    device const float* qkv [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device const float* ct [[buffer(2)]],
    device const float* st [[buffer(3)]],
    device float* q [[buffer(4)]], device float* k [[buffer(5)]], device float* v [[buffer(6)]],
    constant uint& rows [[buffer(7)]], constant uint& heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]], uint idx [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2u;
    if (idx >= rows * heads * half_dim) return;
    uint pair = idx % half_dim, head = (idx / half_dim) % heads;
    uint row = idx / (half_dim * heads), width = heads * head_dim;
    uint cl = head * head_dim + pair, cr = cl + half_dim;
    uint left = row * width + cl, right = row * width + cr;
    uint t = positions[row] * half_dim + pair, src = row * width * 3u;
    float c = ct[t], s = st[t];
    float ql = qkv[src + cl], qr = qkv[src + cr];
    float kl = qkv[src + width + cl], kr = qkv[src + width + cr];
    q[left] = ql * c - qr * s; q[right] = ql * s + qr * c;
    k[left] = kl * c - kr * s; k[right] = kl * s + kr * c;
    v[left] = qkv[src + width * 2u + cl];
    v[right] = qkv[src + width * 2u + cr];
}

// QK score dots on the GPU, in the CPU's scalar reduction order.
kernel void encoder_attention_scores(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const uint* starts [[buffer(2)]], device const uint* lengths [[buffer(3)]],
    device float* scores [[buffer(4)]], constant uint& rows [[buffer(5)]],
    constant uint& heads [[buffer(6)]], constant uint& head_dim [[buffer(7)]],
    constant uint& max_len [[buffer(8)]], constant float& scale [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= rows * heads) return;
    uint row = idx / heads, head = idx % heads, width = heads * head_dim;
    uint start = starts[row], len = lengths[row], offset = row * width + head * head_dim;
    device float* ss = scores + idx * max_len;
    for (uint key = 0; key < len; key++) {
        uint ko = (start + key) * width + head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot = dot + q[offset + d] * k[ko + d];
        ss[key] = dot * scale;
    }
}

// PV after the shared CPU softmax. Keep the key-order f32 accumulation.
kernel void encoder_attention_values(
    device const float* scores [[buffer(0)]], device const float* v [[buffer(1)]],
    device const uint* starts [[buffer(2)]], device const uint* lengths [[buffer(3)]],
    device float* out [[buffer(4)]], constant uint& rows [[buffer(5)]],
    constant uint& heads [[buffer(6)]], constant uint& head_dim [[buffer(7)]],
    constant uint& max_len [[buffer(8)]], uint idx [[thread_position_in_grid]]
) {
    if (idx >= rows * heads) return;
    uint row = idx / heads, head = idx % heads, width = heads * head_dim;
    uint start = starts[row], len = lengths[row], offset = row * width + head * head_dim;
    device const float* ss = scores + idx * max_len;
    for (uint d0 = 0; d0 < head_dim; d0 += 16u) {
        float acc[16];
        for (uint j = 0; j < 16u; j++) acc[j] = 0.0f;
        for (uint key = 0; key < len; key++) {
            uint vo = (start + key) * width + head * head_dim + d0;
            for (uint j = 0; j < 16u; j++)
                if (d0 + j < head_dim) acc[j] = acc[j] + ss[key] * v[vo + j];
        }
        for (uint j = 0; j < 16u; j++) if (d0 + j < head_dim) out[offset + d0 + j] = acc[j];
    }
}

"#;

#[derive(Default)]
pub(super) struct Slot {
    encoder: OnceLock<Option<Mutex<Option<MetalEncoder>>>>,
    failed: AtomicBool,
}

impl Slot {
    pub(super) fn embed(
        &self,
        runtime: &NomicBertRuntime,
        inputs: &[String],
        wait: bool,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        if !crate::cuda::gpu_accel_enabled()
            || disabled_by_env()
            || self.failed.load(Ordering::Acquire)
        {
            return Ok(None);
        }
        let Some(encoder) = self
            .encoder
            .get_or_init(|| match MetalEncoder::build(runtime) {
                Ok(encoder) => Some(Mutex::new(Some(encoder))),
                Err(reason) => {
                    tracing::warn!(%reason, "Metal document encoder unavailable; using the CPU");
                    None
                }
            })
        else {
            return Ok(None);
        };
        let mut slot = match if wait {
            encoder.lock().map_err(TryLockError::Poisoned)
        } else {
            encoder.try_lock()
        } {
            Ok(encoder) => encoder,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                self.failed.store(true, Ordering::Release);
                return Ok(None);
            }
        };
        // A caller may have queued behind a batch that disabled this lane.
        if self.failed.load(Ordering::Acquire) {
            return Ok(None);
        }
        let Some(encoder) = slot.as_mut() else {
            return Ok(None);
        };
        let prepared = super::embedding_batch_pool().install(|| {
            use rayon::prelude::*;
            inputs
                .par_iter()
                .map(|input| runtime.embed_input(input))
                .collect::<Result<Vec<_>>>()
        })?;
        if prepared.iter().any(|(ids, _)| ids.len() > MAX_ROWS) {
            return Ok(None);
        }
        let hidden = match encoder.forward(&prepared) {
            Ok(hidden) => hidden,
            Err(reason) => {
                // Publish the failure while still owning the queue; a query
                // cannot acquire it in the gap before the lane is disabled.
                self.failed.store(true, Ordering::Release);
                // Release weights and scratch from unified RAM before the
                // caller retries on the CPU, including allocation failures.
                slot.take();
                tracing::warn!(%reason, "Metal document encoder failed; using the CPU");
                return Ok(None);
            }
        };
        let result = prepared
            .into_iter()
            .zip(hidden)
            .map(|((ids, _), hidden)| {
                runtime
                    .output(ids, hidden)
                    .pool(runtime.config.pooling)
                    .map_err(|error| error.to_string())
            })
            .collect();
        match result {
            Ok(vectors) => Ok(Some(vectors)),
            Err(reason) => {
                self.failed.store(true, Ordering::Release);
                slot.take();
                tracing::warn!(%reason, "Metal document encoder failed; using the CPU");
                Ok(None)
            }
        }
    }
}

fn disabled_by_env() -> bool {
    disabled_by(
        std::env::var("CAMELID_DOCUMENT_GPU_ENCODER")
            .ok()
            .as_deref(),
    )
}

fn disabled_by(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

struct DeviceLinear {
    scales: Buffer,
    quants: Buffer,
    outs: usize,
    blocks: usize,
}

struct DeviceLayer {
    qkv: DeviceLinear,
    attention_output: DeviceLinear,
    gate: DeviceLinear,
    up: DeviceLinear,
    down: DeviceLinear,
    attention_norm_weight: Buffer,
    attention_norm_bias: Buffer,
    output_norm_weight: Buffer,
    output_norm_bias: Buffer,
}

struct Kernels {
    quantize: ComputePipelineState,
    gemm: ComputePipelineState,
    rope: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_values: ComputePipelineState,
}

impl Kernels {
    fn build(device: &Device) -> std::result::Result<Self, String> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device.new_library_with_source(KERNELS, &options)?;
        let pipeline = |name| {
            let function = library
                .get_function(name, None)
                .map_err(|error| error.to_string())?;
            device.new_compute_pipeline_state_with_function(&function)
        };
        Ok(Self {
            quantize: pipeline("encoder_quantize_q8")?,
            gemm: pipeline("encoder_q8_gemm")?,
            rope: pipeline("encoder_rope_split")?,
            attention_scores: pipeline("encoder_attention_scores")?,
            attention_values: pipeline("encoder_attention_values")?,
        })
    }
}

pub(super) struct MetalEncoder {
    device: Device,
    queue: CommandQueue,
    kernels: Kernels,
    layers: Vec<DeviceLayer>,
    rope_cos: Buffer,
    rope_sin: Buffer,
    width: usize,
    ffn: usize,
    heads: usize,
    head_dim: usize,
    epsilon: f32,
    buffers: Option<Buffers>,
    #[cfg(test)]
    trace: Option<Mutex<Vec<Vec<f32>>>>,
}

struct Buffers {
    rows: usize,
    max_len: usize,
    hidden: Buffer,
    positions: Buffer,
    starts: Buffer,
    lengths: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attended: Buffer,
    projected: Buffer,
    gate: Buffer,
    up: Buffer,
    xs: Buffer,
    xq: Buffer,
    scores: Buffer,
}

fn upload<T>(device: &Device, values: &[T]) -> std::result::Result<Buffer, String> {
    let bytes = std::mem::size_of_val(values);
    if bytes == 0 || bytes as u64 > device.max_buffer_length() {
        return Err("Metal encoder buffer exceeds the device limit".into());
    }
    let device: &metal::DeviceRef = device;
    let data = values.as_ptr().cast::<c_void>();
    let length = bytes as u64;
    let options = MTLResourceOptions::StorageModeShared;
    // SAFETY: device is valid and the byte slice stays alive until Metal has
    // copied it. Unlike metal 0.31's Buffer-returning wrapper, retain the raw
    // nullable result so real memory pressure follows the CPU fallback path.
    let pointer =
        unsafe { msg_send![device, newBufferWithBytes:data length:length options:options] };
    // SAFETY: the newBuffer selector returns null or a +1 retained MTLBuffer.
    unsafe { buffer_from_allocation(pointer) }
}

fn allocate_buffer(device: &Device, length: u64) -> std::result::Result<Buffer, String> {
    if length == 0 || length > device.max_buffer_length() {
        return Err("Metal encoder allocation exceeds device limit".into());
    }
    let device: &metal::DeviceRef = device;
    let options = MTLResourceOptions::StorageModeShared;
    // SAFETY: device is valid, size is bounded, and the nullable raw pointer
    // is checked before constructing an owned foreign object.
    let pointer = unsafe { msg_send![device, newBufferWithLength:length options:options] };
    // SAFETY: the newBuffer selector returns null or a +1 retained MTLBuffer.
    unsafe { buffer_from_allocation(pointer) }
}

/// `pointer` must be null or a valid +1 retained MTLBuffer from a newBuffer
/// selector. Ownership of a non-null allocation transfers to the result.
unsafe fn buffer_from_allocation(
    pointer: *mut metal::MTLBuffer,
) -> std::result::Result<Buffer, String> {
    if pointer.is_null() {
        return Err("Metal encoder buffer allocation failed".into());
    }
    // SAFETY: the caller provides a valid retained allocation, checked above.
    Ok(unsafe { Buffer::from_ptr(pointer) })
}

fn write<T>(buffer: &Buffer, values: &[T]) {
    assert!(std::mem::size_of_val(values) as u64 <= buffer.length());
    // SAFETY: the private shared buffers have room for the slice, the lane's
    // mutex excludes concurrent CPU writes, and the previous command finished.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast::<T>(), values.len())
    };
}

fn read<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
    assert!(count * std::mem::size_of::<T>() <= buffer.length() as usize);
    // SAFETY: the command completed before a read, this shared buffer contains
    // `count` initialized T values, and callers choose its original scalar type.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), count).to_vec() }
}

fn scalar<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        (value as *const T).cast::<c_void>(),
    );
}

fn dispatch(encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState, count: usize) {
    encoder.set_compute_pipeline_state(pipeline);
    let threads = pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(
        MTLSize::new(count as u64, 1, 1),
        MTLSize::new(threads, 1, 1),
    );
}

fn bind(encoder: &ComputeCommandEncoderRef, buffers: &[&Buffer]) {
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
}

fn encode_linear(
    encoder: &ComputeCommandEncoderRef,
    kernels: &Kernels,
    input: &Buffer,
    weight: &DeviceLinear,
    output: &Buffer,
    buffers: &Buffers,
    rows: usize,
) {
    bind(encoder, &[input, &buffers.xs, &buffers.xq]);
    scalar(encoder, 3, &((rows * weight.blocks) as u32));
    dispatch(encoder, &kernels.quantize, rows * weight.blocks);
    encoder.memory_barrier_with_resources(&[&buffers.xs, &buffers.xq]);
    bind(
        encoder,
        &[
            &buffers.xs,
            &buffers.xq,
            &weight.scales,
            &weight.quants,
            output,
        ],
    );
    scalar(encoder, 5, &(rows as u32));
    scalar(encoder, 6, &(weight.outs as u32));
    scalar(encoder, 7, &(weight.blocks as u32));
    dispatch(encoder, &kernels.gemm, rows * weight.outs);
    // Also order the quantizer's scratch reuse after this GEMM's reads.
    encoder.memory_barrier_with_resources(&[output, &buffers.xs, &buffers.xq]);
}

fn finish(
    command: &metal::CommandBufferRef,
    encoder: &ComputeCommandEncoderRef,
) -> std::result::Result<(), String> {
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "Metal encoder command failed: {:?}",
            command.status()
        ));
    }
    Ok(())
}

fn upload_linear(
    device: &Device,
    tensor: &CpuTensor,
    inputs: usize,
    outs: usize,
) -> std::result::Result<DeviceLinear, String> {
    let blocks = tensor
        .q8_0_blocks
        .as_ref()
        .ok_or("encoder matrix has no Q8_0 blocks")?;
    if !inputs.is_multiple_of(32) || blocks.len() != outs * (inputs / 32) {
        return Err("Metal encoder Q8 matrix shape mismatch".into());
    }
    let scales: Vec<f32> = blocks.iter().map(|block| block.scale).collect();
    let quants: Vec<i8> = blocks.iter().flat_map(|block| block.quants).collect();
    Ok(DeviceLinear {
        scales: upload(device, &scales)?,
        quants: upload(device, &quants)?,
        outs,
        blocks: inputs / 32,
    })
}

fn rope_tables(tokens: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let (mut cos, mut sin) = (
        Vec::with_capacity(tokens * half),
        Vec::with_capacity(tokens * half),
    );
    for position in 0..tokens {
        for pair in 0..half {
            let frequency = 1.0 / base.powf((2 * pair) as f32 / head_dim as f32);
            let (s, c) = (position as f32 * frequency).sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    (cos, sin)
}

impl MetalEncoder {
    pub(super) fn build(runtime: &NomicBertRuntime) -> std::result::Result<Self, String> {
        metal::objc::rc::autoreleasepool(|| Self::build_inner(runtime))
    }

    fn build_inner(runtime: &NomicBertRuntime) -> std::result::Result<Self, String> {
        if !cfg!(target_arch = "aarch64") {
            return Err("Metal document encoder requires Apple silicon".into());
        }
        let device = Device::system_default().ok_or("no Metal device")?;
        // The shared-memory lane is deliberately Apple-silicon scoped. Intel
        // discrete GPUs retain the CPU path until they have parity evidence.
        if !device.has_unified_memory() {
            return Err("Metal document encoder needs unified memory".into());
        }
        let config = &runtime.config;
        let (width, ffn) = (config.embedding_length, config.feed_forward_length);
        if width == 0
            || ffn < width
            || !width.is_multiple_of(32)
            || !ffn.is_multiple_of(32)
            || !config.head_dim.is_multiple_of(2)
        {
            return Err("Metal encoder geometry unsupported".into());
        }
        // Only the pinned Nomic Q8 geometry is admitted to this internal lane.
        if (
            width,
            ffn,
            config.head_count,
            config.head_dim,
            config.block_count,
        ) != (768, 3072, 12, 64, 12)
        {
            return Err("Metal document encoder needs the pinned Nomic geometry".into());
        }
        let kernels = Kernels::build(&device)?;
        let queue = device.new_command_queue();
        let mut layers = Vec::with_capacity(runtime.layers.len());
        for layer in &runtime.layers {
            layers.push(DeviceLayer {
                qkv: upload_linear(&device, &layer.qkv, width, width * 3)?,
                attention_output: upload_linear(&device, &layer.attention_output, width, width)?,
                gate: upload_linear(&device, &layer.ffn_gate, width, ffn)?,
                up: upload_linear(&device, &layer.ffn_up, width, ffn)?,
                down: upload_linear(&device, &layer.ffn_down, ffn, width)?,
                attention_norm_weight: upload(&device, &layer.attention_norm_weight)?,
                attention_norm_bias: upload(&device, &layer.attention_norm_bias)?,
                output_norm_weight: upload(&device, &layer.output_norm_weight)?,
                output_norm_bias: upload(&device, &layer.output_norm_bias)?,
            });
        }
        let (cos, sin) = rope_tables(
            config.context_length.min(MAX_ROWS),
            config.head_dim,
            config.rope_frequency_base,
        );
        let rope_cos = upload(&device, &cos)?;
        let rope_sin = upload(&device, &sin)?;
        tracing::info!(device = %device.name(), "document encoder layers run on Metal");
        Ok(Self {
            device,
            queue,
            kernels,
            layers,
            rope_cos,
            rope_sin,
            width,
            ffn,
            heads: config.head_count,
            head_dim: config.head_dim,
            epsilon: config.layer_norm_epsilon,
            buffers: None,
            #[cfg(test)]
            trace: None,
        })
    }

    fn allocate(&self, rows: usize, max_len: usize) -> std::result::Result<Buffers, String> {
        let alloc = |count: usize, bytes: usize| {
            let length = count
                .checked_mul(bytes)
                .ok_or("Metal encoder allocation overflow")?;
            if length == 0 || length as u64 > self.device.max_buffer_length() {
                return Err("Metal encoder allocation exceeds device limit".to_string());
            }
            allocate_buffer(&self.device, length as u64)
        };
        Ok(Buffers {
            rows,
            max_len,
            hidden: alloc(rows * self.width, 4)?,
            positions: alloc(rows, 4)?,
            starts: alloc(rows, 4)?,
            lengths: alloc(rows, 4)?,
            qkv: alloc(rows * self.width * 3, 4)?,
            q: alloc(rows * self.width, 4)?,
            k: alloc(rows * self.width, 4)?,
            v: alloc(rows * self.width, 4)?,
            attended: alloc(rows * self.width, 4)?,
            projected: alloc(rows * self.width, 4)?,
            gate: alloc(rows * self.ffn, 4)?,
            up: alloc(rows * self.ffn, 4)?,
            xs: alloc(rows * self.ffn / 32, 4)?,
            xq: alloc(rows * self.ffn, 1)?,
            scores: alloc(rows * self.heads * max_len, 4)?,
        })
    }

    pub(super) fn forward(
        &mut self,
        prepared: &[(Vec<u32>, Vec<f32>)],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut output = Vec::with_capacity(prepared.len());
        let (mut group, mut rows) = (Vec::new(), 0);
        for item in prepared {
            if item.0.is_empty()
                || item.0.len() > MAX_ROWS
                || item.1.len() != item.0.len() * self.width
            {
                return Err("Metal encoder input shape unsupported".into());
            }
            if rows + item.0.len() > MAX_ROWS {
                output.extend(self.forward_group(&group)?);
                group.clear();
                rows = 0;
            }
            rows += item.0.len();
            group.push(item);
        }
        if !group.is_empty() {
            output.extend(self.forward_group(&group)?);
        }
        Ok(output)
    }

    fn forward_group(
        &mut self,
        group: &[&(Vec<u32>, Vec<f32>)],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let rows: usize = group.iter().map(|item| item.0.len()).sum();
        let max_len = group.iter().map(|item| item.0.len()).max().unwrap_or(0);
        let buffers = match self.buffers.take() {
            Some(buffers) if buffers.rows >= rows && buffers.max_len >= max_len => buffers,
            Some(buffers) => {
                let (rows, max_len) = (rows.max(buffers.rows), max_len.max(buffers.max_len));
                drop(buffers);
                self.allocate(rows, max_len)?
            }
            None => self.allocate(rows, max_len)?,
        };
        // Indexing runs on a Rust worker, without a Cocoa run-loop pool.
        // Drain each group's autoreleased command buffer and encoder after
        // completion so a large library does not retain them indefinitely.
        let result =
            metal::objc::rc::autoreleasepool(|| self.run_group(group, rows, max_len, &buffers));
        self.buffers = Some(buffers);
        result
    }

    fn run_group(
        &self,
        group: &[&(Vec<u32>, Vec<f32>)],
        rows: usize,
        max_len: usize,
        b: &Buffers,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let (mut hidden, mut positions, mut starts, mut lengths) = (
            Vec::with_capacity(rows * self.width),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        );
        for (ids, values) in group.iter().copied() {
            let start = positions.len() as u32;
            for position in 0..ids.len() {
                positions.push(position as u32);
                starts.push(start);
                lengths.push(ids.len() as u32);
            }
            hidden.extend_from_slice(values);
        }
        write(&b.hidden, &hidden);
        write(&b.positions, &positions);
        write(&b.starts, &starts);
        write(&b.lengths, &lengths);
        // Initialize padding as well: the host normalizes only live scores,
        // but copies the complete active rectangular scratch buffer.
        write(&b.scores, &vec![0.0_f32; rows * self.heads * max_len]);
        for layer in &self.layers {
            let command = self.queue.new_command_buffer();
            let e = command.new_compute_command_encoder();
            encode_linear(e, &self.kernels, &b.hidden, &layer.qkv, &b.qkv, b, rows);
            bind(
                e,
                &[
                    &b.qkv,
                    &b.positions,
                    &self.rope_cos,
                    &self.rope_sin,
                    &b.q,
                    &b.k,
                    &b.v,
                ],
            );
            scalar(e, 7, &(rows as u32));
            scalar(e, 8, &(self.heads as u32));
            scalar(e, 9, &(self.head_dim as u32));
            dispatch(
                e,
                &self.kernels.rope,
                rows * self.heads * (self.head_dim / 2),
            );
            e.memory_barrier_with_resources(&[&b.q, &b.k, &b.v]);
            bind(e, &[&b.q, &b.k, &b.starts, &b.lengths, &b.scores]);
            scalar(e, 5, &(rows as u32));
            scalar(e, 6, &(self.heads as u32));
            scalar(e, 7, &(self.head_dim as u32));
            scalar(e, 8, &(max_len as u32));
            scalar(e, 9, &(1.0 / (self.head_dim as f32).sqrt()));
            dispatch(e, &self.kernels.attention_scores, rows * self.heads);
            finish(command, e)?;
            self.softmax(&b.scores, &lengths, max_len)?;
            let command = self.queue.new_command_buffer();
            let e = command.new_compute_command_encoder();
            bind(e, &[&b.scores, &b.v, &b.starts, &b.lengths, &b.attended]);
            scalar(e, 5, &(rows as u32));
            scalar(e, 6, &(self.heads as u32));
            scalar(e, 7, &(self.head_dim as u32));
            scalar(e, 8, &(max_len as u32));
            dispatch(e, &self.kernels.attention_values, rows * self.heads);
            e.memory_barrier_with_resources(&[&b.attended]);
            encode_linear(
                e,
                &self.kernels,
                &b.attended,
                &layer.attention_output,
                &b.projected,
                b,
                rows,
            );
            finish(command, e)?;
            self.norm(
                &b.projected,
                &b.hidden,
                &layer.attention_norm_weight,
                &layer.attention_norm_bias,
                rows,
            )?;
            let command = self.queue.new_command_buffer();
            let e = command.new_compute_command_encoder();
            encode_linear(
                e,
                &self.kernels,
                &b.projected,
                &layer.gate,
                &b.gate,
                b,
                rows,
            );
            encode_linear(e, &self.kernels, &b.projected, &layer.up, &b.up, b, rows);
            finish(command, e)?;
            self.silu(&b.gate, &b.up, rows)?;
            let command = self.queue.new_command_buffer();
            let e = command.new_compute_command_encoder();
            encode_linear(e, &self.kernels, &b.gate, &layer.down, &b.hidden, b, rows);
            finish(command, e)?;
            self.norm(
                &b.hidden,
                &b.projected,
                &layer.output_norm_weight,
                &layer.output_norm_bias,
                rows,
            )?;
            #[cfg(test)]
            if let Some(trace) = &self.trace {
                trace
                    .lock()
                    .unwrap()
                    .push(read(&b.hidden, rows * self.width));
            }
        }
        let hidden = read::<f32>(&b.hidden, rows * self.width);
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err("Metal encoder returned non-finite rows".into());
        }
        let (mut output, mut start) = (Vec::with_capacity(group.len()), 0);
        for (ids, _) in group {
            let len = ids.len() * self.width;
            output.push(hidden[start..start + len].to_vec());
            start += len;
        }
        Ok(output)
    }

    fn softmax(
        &self,
        scores: &Buffer,
        lengths: &[u32],
        max_len: usize,
    ) -> std::result::Result<(), String> {
        use rayon::prelude::*;
        let mut values = read::<f32>(scores, lengths.len() * self.heads * max_len);
        values
            .par_chunks_mut(max_len)
            .enumerate()
            .try_for_each(|(index, row)| {
                super::softmax_in_place(&mut row[..lengths[index / self.heads] as usize])
                    .map_err(|error| error.to_string())
            })?;
        write(scores, &values);
        Ok(())
    }

    fn silu(&self, gate: &Buffer, up: &Buffer, rows: usize) -> std::result::Result<(), String> {
        let mut values = read::<f32>(gate, rows * self.ffn);
        let up = read::<f32>(up, rows * self.ffn);
        for (gate, up) in values.iter_mut().zip(up) {
            *gate = super::silu(*gate) * up;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("Metal encoder activation returned non-finite values".into());
        }
        write(gate, &values);
        Ok(())
    }

    fn norm(
        &self,
        input: &Buffer,
        residual: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) -> std::result::Result<(), String> {
        // The preceding command has completed. Metal cannot reproduce the
        // CPU's f64 mean, variance and normalized term with an f32 reduction;
        // even sub-ulp errors can cross a later Q8 quantization boundary.
        // Keep the shared oracle here while all five projections and full
        // attention remain on the GPU. StorageModeShared needs no GPU copy.
        let mut values = read::<f32>(input, rows * self.width);
        let residual = read::<f32>(residual, rows * self.width);
        super::add_in_place(&mut values, &residual).map_err(|error| error.to_string())?;
        super::layer_norm_in_place(
            &mut values,
            rows,
            self.width,
            &read::<f32>(weight, self.width),
            &read::<f32>(bias, self.width),
            self.epsilon,
        )
        .map_err(|error| error.to_string())?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("Metal encoder normalization returned non-finite rows".into());
        }
        write(input, &values);
        Ok(())
    }

    #[cfg(all(test, target_arch = "aarch64"))]
    fn qkv_for_test(
        &self,
        runtime: &NomicBertRuntime,
        input: &[f32],
        rows: usize,
    ) -> std::result::Result<Vec<f32>, String> {
        let buffers = self.allocate(rows, 1)?;
        write(&buffers.hidden, input);
        let weight = upload_linear(
            &self.device,
            &runtime.layers[0].qkv,
            self.width,
            self.width * 3,
        )?;
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_linear(
            encoder,
            &self.kernels,
            &buffers.hidden,
            &weight,
            &buffers.qkv,
            &buffers,
            rows,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Metal QKV command failed".into());
        }
        Ok(read(&buffers.qkv, rows * self.width * 3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_metal_buffer_allocation_returns_an_error() {
        // A null allocation satisfies the helper's safety contract and must
        // never reach Buffer::from_ptr, panic, or dereference a foreign object.
        let result = unsafe { buffer_from_allocation(std::ptr::null_mut()) };
        assert!(result.is_err());
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    #[cfg(target_arch = "aarch64")]
    struct CpuOracle {
        model_sha256: String,
        texts: Vec<String>,
        vectors: Vec<Vec<u32>>,
        probe_layers: Vec<Vec<u32>>,
    }

    #[cfg(target_arch = "aarch64")]
    fn cpu_oracle(runtime: &NomicBertRuntime, model: &str, texts: &[String]) -> CpuOracle {
        use sha2::{Digest, Sha256};
        use std::io::Read;

        let mut file = std::fs::File::open(model).unwrap();
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 65_536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let model_sha256 = format!("{:x}", hasher.finalize());
        // Opt-in scratch cache for GPU-only iterations. Refresh it whenever
        // the CPU oracle changes; f32 bits avoid JSON float round-trip drift.
        let path = std::env::var_os("CAMELID_METAL_CPU_ORACLE").map(std::path::PathBuf::from);
        if let Some(path) = &path {
            if let Ok(bytes) = std::fs::read(path) {
                let cached: CpuOracle =
                    serde_json::from_slice(&bytes).expect("valid CPU oracle cache");
                if cached.model_sha256 == model_sha256
                    && cached.texts == texts
                    && cached.vectors.len() == texts.len()
                    && cached.probe_layers.len() == runtime.layers.len()
                {
                    eprintln!("CPU oracle loaded from scratch cache");
                    return cached;
                }
            }
        }
        let started = std::time::Instant::now();
        let (ids, mut hidden) = runtime.embed_input(&texts[0]).unwrap();
        let mut probe_layers = Vec::new();
        for (index, layer) in runtime.layers.iter().enumerate() {
            hidden = runtime
                .forward_layer(hidden, ids.len(), index, layer)
                .unwrap();
            probe_layers.push(hidden.iter().map(|value| value.to_bits()).collect());
        }
        let first = runtime
            .output(ids, hidden)
            .pool(runtime.config.pooling)
            .unwrap();
        // Duplicate passages exercise separate GPU groups but need only one
        // deterministic CPU result. Other inputs use the bounded worker pool.
        let mut unique = Vec::new();
        for text in &texts[1..] {
            if text != &texts[0] && !unique.contains(text) {
                unique.push(text.clone());
            }
        }
        let other = runtime.embed_batch(&unique, None).unwrap();
        let vectors = texts
            .iter()
            .map(|text| {
                let vector = if text == &texts[0] {
                    &first
                } else {
                    &other[unique.iter().position(|item| item == text).unwrap()]
                };
                vector.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        let oracle = CpuOracle {
            model_sha256,
            texts: texts.to_vec(),
            vectors,
            probe_layers,
        };
        eprintln!(
            "CPU oracle computed in {:.3}s (debug build measurement)",
            started.elapsed().as_secs_f64()
        );
        // Save before comparisons, so a rejected GPU path never loses an
        // expensive CPU oracle and later attempts can retain the same gate.
        if let Some(path) = path {
            if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, serde_json::to_vec(&oracle).unwrap()).unwrap();
        }
        oracle
    }

    #[test]
    fn document_gpu_opt_out_values() {
        for value in ["0", "OFF", " no ", "False"] {
            assert!(disabled_by(Some(value)));
        }
        for value in ["1", "on", "", "other"] {
            assert!(!disabled_by(Some(value)));
        }
        assert!(!disabled_by(None));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn metal_q8_linears_match_cpu_bits_on_ragged_rows() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal kernel test skipped: no Metal device");
            return;
        };
        let kernels = Kernels::build(&device).expect("Metal encoder kernels");
        let (rows, width, outs) = (3, 64, 67);
        let input: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.031)
            .collect();
        let weights: Vec<f32> = (0..outs * width)
            .map(|i| ((i * 37 % 113) as f32 - 56.0) * 0.017)
            .collect();
        let blocks = crate::inference::quantize_q8_0_blocks(&weights);
        let tensor = CpuTensor::from_q8_0_blocks(
            "weights",
            crate::tensor::TensorShape {
                dims: vec![outs, width],
            },
            blocks,
        )
        .unwrap();
        let input_tensor = CpuTensor::from_f32("input", vec![rows, width], input.clone()).unwrap();
        let expected = crate::inference::linear_for_role_runtime(
            &input_tensor,
            &tensor,
            "expected",
            "attention_q",
            false,
        )
        .unwrap()
        .data;
        let (xs, xq) = (
            upload(&device, &vec![0.0_f32; rows * width / 32]).unwrap(),
            upload(&device, &vec![0_i8; rows * width]).unwrap(),
        );
        let input = upload(&device, &input).unwrap();
        let output = upload(&device, &vec![0.0_f32; rows * outs]).unwrap();
        let weight = upload_linear(&device, &tensor, width, outs).unwrap();
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let e = command.new_compute_command_encoder();
        bind(e, &[&input, &xs, &xq]);
        scalar(e, 3, &((rows * width / 32) as u32));
        dispatch(e, &kernels.quantize, rows * width / 32);
        e.memory_barrier_with_resources(&[&xs, &xq]);
        bind(e, &[&xs, &xq, &weight.scales, &weight.quants, &output]);
        scalar(e, 5, &(rows as u32));
        scalar(e, 6, &(outs as u32));
        scalar(e, 7, &((width / 32) as u32));
        dispatch(e, &kernels.gemm, rows * outs);
        e.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let actual: Vec<f32> = read(&output, rows * outs);
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "needs CAMELID_NOMIC_GGUF and an Apple GPU"]
    fn metal_document_vectors_match_the_cpu() {
        let model = std::env::var("CAMELID_NOMIC_GGUF").expect("CAMELID_NOMIC_GGUF");
        let runtime = NomicBertRuntime::load(&model).expect("load encoder");
        let mut gpu = MetalEncoder::build(&runtime).expect("Metal encoder");
        let texts: Vec<String> = [
            "search_document: Enterprise customers may request a refund within 60 days of the invoice date. Refunds go to the original payment method.",
            "search_document: Trial accounts are not eligible for refunds. Document exports are fulfilled within thirty days.",
            "search_query: How many days do enterprise customers have to request a refund?",
            "search_document: The western district allocated 937,700 dollars. The eastern district allocated 707,600 dollars.",
        ].into_iter().map(str::to_string).collect();
        let mut texts = texts;
        texts.push(format!(
            "search_document: {}",
            "A long passage about invoices, exports, account eligibility and library retrieval. "
                .repeat(45)
        ));
        texts.push(texts.last().unwrap().clone());
        let prepared = texts
            .iter()
            .map(|text| runtime.embed_input(text).unwrap())
            .collect::<Vec<_>>();
        assert!(
            prepared.iter().map(|item| item.0.len()).sum::<usize>() > MAX_ROWS,
            "test must split the batch"
        );
        assert!(prepared.iter().all(|item| item.0.len() <= MAX_ROWS));
        let oracle = cpu_oracle(&runtime, &model, &texts);
        let rows = prepared[0].0.len();
        let cpu_input = CpuTensor::from_f32(
            "probe",
            vec![rows, runtime.config.embedding_length],
            prepared[0].1.clone(),
        )
        .unwrap();
        let cpu_qkv = crate::inference::linear_for_role_runtime(
            &cpu_input,
            &runtime.layers[0].qkv,
            "qkv",
            "attention_q",
            false,
        )
        .unwrap()
        .data;
        let gpu_qkv = gpu.qkv_for_test(&runtime, &prepared[0].1, rows).unwrap();
        assert_eq!(
            cpu_qkv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            gpu_qkv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "Q8 projection"
        );
        gpu.trace = Some(Mutex::new(Vec::new()));
        gpu.forward(&prepared[..1]).expect("Metal layer probe");
        let trace = gpu.trace.take().unwrap().into_inner().unwrap();
        for (index, (cpu, metal)) in oracle.probe_layers.iter().zip(&trace).enumerate() {
            let cpu = cpu.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
            let maximum = cpu
                .iter()
                .zip(metal)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            let cosine = crate::embedding::cosine_similarity(&cpu, metal).unwrap();
            eprintln!(
                "layer {}: cosine {cosine:.8}, max absolute difference {maximum:.3e}",
                index + 1
            );
        }
        let started = std::time::Instant::now();
        let hidden = gpu.forward(&prepared).expect("Metal forward");
        eprintln!(
            "Metal layer stack: {} chunks in {:.3}s",
            texts.len(),
            started.elapsed().as_secs_f64()
        );
        // Reuse a grown group's storage for ragged smaller inputs; attention
        // strides must use the active group, never the allocation's capacity.
        let reused = gpu.forward(&prepared[..2]).expect("reuse buffers");
        assert_eq!(reused, hidden[..2]);
        let mut minimum = 1.0_f32;
        for (index, ((ids, _), hidden)) in prepared.iter().zip(hidden).enumerate() {
            let gpu = runtime
                .output(ids.clone(), hidden)
                .pool(runtime.config.pooling)
                .unwrap();
            let cpu = oracle.vectors[index]
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            let cosine = crate::embedding::cosine_similarity(&cpu, &gpu).unwrap();
            eprintln!(
                "sample {} ({} tokens): cosine {cosine:.8}",
                index + 1,
                ids.len()
            );
            minimum = minimum.min(cosine);
            assert!(gpu.iter().all(|value| value.is_finite()));
            let norm: f64 = gpu.iter().map(|value| f64::from(*value).powi(2)).sum();
            assert!((norm - 1.0).abs() < 1e-6, "L2 normalization: {norm}");
        }
        eprintln!("Metal whole-vector minimum cosine: {minimum:.8}");
        assert!(minimum >= 0.9999, "minimum cosine {minimum}");
        // Exercise the same dispatch as the indexer and search, and the
        // explicit opt-out. The oracle above calls the actual GPU directly.
        let (vectors, lane) = runtime.embed_documents(&texts).unwrap();
        assert_eq!(lane, super::super::EncoderLane::Metal);
        assert_eq!(vectors.len(), texts.len());
        assert_eq!(
            runtime.embed_query(&texts[2]).unwrap().len(),
            runtime.config.embedding_length
        );
        let old_opt_out = std::env::var_os("CAMELID_DOCUMENT_GPU_ENCODER");
        std::env::set_var("CAMELID_DOCUMENT_GPU_ENCODER", "0");
        let (fallback, lane) = runtime.embed_documents(&texts[..1]).unwrap();
        match old_opt_out {
            Some(value) => std::env::set_var("CAMELID_DOCUMENT_GPU_ENCODER", value),
            None => std::env::remove_var("CAMELID_DOCUMENT_GPU_ENCODER"),
        }
        assert_eq!(lane, super::super::EncoderLane::Cpu);
        assert_eq!(fallback[0], runtime.embed(&texts[0], None).unwrap());
        // Searches fall back immediately while indexing owns the lane.
        let occupied = runtime
            .metal
            .encoder
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(
            runtime.embed_query(&texts[2]).unwrap(),
            runtime.embed(&texts[2], None).unwrap()
        );
        drop(occupied);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "needs CAMELID_NOMIC_GGUF and an Apple GPU"]
    fn metal_failed_document_lane_releases_memory_and_retries_cpu() {
        let model = std::env::var("CAMELID_NOMIC_GGUF").expect("CAMELID_NOMIC_GGUF");
        let runtime = NomicBertRuntime::load(&model).expect("load encoder");
        let texts =
            ["search_document: Customers may request refunds within sixty days.".to_string()];
        let expected = runtime.embed(&texts[0], None).unwrap();
        let (_, lane) = runtime.embed_documents(&texts).unwrap();
        assert_eq!(lane, super::super::EncoderLane::Metal);
        // A GPU result failure retries on the oracle, disables the lane, and
        // releases its allocations. Corrupt only a resident GPU norm vector;
        // the independent CPU model remains a valid fallback oracle.
        {
            let mut slot = runtime
                .metal
                .encoder
                .get()
                .unwrap()
                .as_ref()
                .unwrap()
                .lock()
                .unwrap();
            let encoder = slot.as_mut().unwrap();
            write(
                &encoder.layers[0].attention_norm_weight,
                &vec![f32::NAN; runtime.config.embedding_length],
            );
        }
        let (fallback, lane) = runtime.embed_documents(&texts[..1]).unwrap();
        assert_eq!(lane, super::super::EncoderLane::Cpu);
        assert_eq!(fallback[0], expected);
        assert!(runtime.metal.failed.load(Ordering::Acquire));
        assert!(runtime
            .metal
            .encoder
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .is_none());
        let (fallback, lane) = runtime.embed_documents(&texts[..1]).unwrap();
        assert_eq!(lane, super::super::EncoderLane::Cpu);
        assert_eq!(fallback[0], expected);
    }
}
