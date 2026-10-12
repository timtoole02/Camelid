//! CUDA layer stack for the Nomic-BERT document encoder (#801).
//!
//! Indexing a large library for search by meaning runs every chunk through the
//! encoder, which takes hours to days on a CPU. This runs the same layers on
//! the GPU, many chunks per forward. Tokenizing, the embedding rows, the first
//! layer norm and pooling stay on the CPU and are shared with
//! [`NomicBertRuntime::encode`].
//!
//! Every kernel follows its CPU counterpart operation for operation, and the
//! module is built with `--fmad=false` so nothing is fused into an FMA:
//! - the Q8 linears quantize each activation block like `quantize_q8_0_block`,
//!   take the exact integer block dot, and add `int as f32 * weight_scale *
//!   input_scale` in block order, so they return the CPU's bits;
//! - layer norm runs in f64 in the CPU's order, and RoPE uses tables computed
//!   by the CPU's own formula;
//! - attention and the gated FFN keep the CPU's loop order; `exp` is taken in
//!   double precision and rounded once, which matches the host's correctly
//!   rounded `expf` in nearly every case but is not guaranteed to. So the
//!   vectors are very close to the CPU's, not promised bit-identical; the
//!   ignored test `cuda_document_vectors_match_the_cpu` measures how close.

use std::sync::{Arc, Mutex, OnceLock, TryLockError};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::CompileOptions;

use super::NomicBertRuntime;
use crate::tensor::CpuTensor;
use crate::Result;

const KERNELS: &str = r#"
__device__ __forceinline__ int camelid_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Port of inference.rs `f32_to_f16_bits` (round to nearest even).
__device__ unsigned short camelid_f32_to_f16_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000);
    int exp = (int)((bits >> 23) & 0xff);
    unsigned int mant = bits & 0x007fffff;
    if (exp == 0xff) {
        return sign | (mant == 0 ? 0x7c00 : 0x7e00);
    }
    int half_exp = exp - 127 + 15;
    if (half_exp >= 0x1f) {
        return sign | 0x7c00;
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return sign;
        }
        unsigned int mantissa = mant | 0x00800000;
        int shift = 14 - half_exp;
        unsigned short half_mant = (unsigned short)(mantissa >> shift);
        unsigned int round_bit = 1u << (shift - 1);
        if ((mantissa & round_bit) != 0
            && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)) {
            half_mant = (unsigned short)(half_mant + 1);
        }
        return sign | half_mant;
    }
    unsigned short half = sign | (unsigned short)(half_exp << 10) | (unsigned short)(mant >> 13);
    if ((mant & 0x00001000) != 0 && ((mant & 0x00000fff) != 0 || (half & 1) != 0)) {
        half = (unsigned short)(half + 1);
    }
    return half;
}

// Port of inference.rs `f16_bits_to_f32`.
__device__ float camelid_f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000)) << 16;
    unsigned int exp = (bits & 0x7c00) >> 10;
    unsigned int frac = bits & 0x03ff;
    unsigned int out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            unsigned int m = frac;
            int e = -14;
            while ((m & 0x0400) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            out = sign | ((unsigned int)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000 | (frac << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

// The host's `f32::exp` (glibc expf) is correctly rounded in nearly every case; CUDA's
// expf can be 2 ulp off. Rounding the double-precision exp once matches the host.
__device__ __forceinline__ float camelid_expf(float x) {
    return (float)exp((double)x);
}

// One thread per 32-value block: `quantize_q8_0_block`.
extern "C" __global__ void encoder_quantize_q8(
    const float* __restrict__ input, long long blocks,
    float* __restrict__ scales, signed char* __restrict__ quants
) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float* v = input + b * 32;
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        max_abs = fmaxf(max_abs, fabsf(v[i]));
    }
    float unrounded = max_abs / 127.0f;
    scales[b] = camelid_f16_bits_to_f32(camelid_f32_to_f16_bits(unrounded));
    float inv = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    signed char* q = quants + b * 32;
    for (int i = 0; i < 32; i++) {
        float r = roundf(v[i] * inv);
        r = fminf(fmaxf(r, -128.0f), 127.0f);
        q[i] = (signed char)r;
    }
}

// out[row][o] = the sum over blocks b, in order, of int_dot * w_scale * x_scale.
// 64x64 output tile per 256-thread block; each thread owns 4 rows x 4 outputs.
#define TILE 64
#define PAD 9
extern "C" __global__ void encoder_q8_gemm(
    const float* __restrict__ x_scales, const signed char* __restrict__ x_quants, int rows,
    const float* __restrict__ w_scales, const signed char* __restrict__ w_quants, int outs,
    int blocks_per_row, float* __restrict__ out
) {
    __shared__ int sx[TILE * PAD];
    __shared__ int sw[TILE * PAD];
    __shared__ float sxs[TILE];
    __shared__ float sws[TILE];
    int tid = threadIdx.x;
    int tr = tid / 16;
    int to = tid % 16;
    int row0 = blockIdx.y * TILE;
    int out0 = blockIdx.x * TILE;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        for (int i = tid; i < TILE * 8; i += 256) {
            int r = i / 8;
            int k = i % 8;
            int gr = row0 + r;
            sx[r * PAD + k] = gr < rows
                ? reinterpret_cast<const int*>(x_quants + ((long long)gr * blocks_per_row + b) * 32)[k]
                : 0;
            int go = out0 + r;
            sw[r * PAD + k] = go < outs
                ? reinterpret_cast<const int*>(w_quants + ((long long)go * blocks_per_row + b) * 32)[k]
                : 0;
        }
        if (tid < TILE) {
            int gr = row0 + tid;
            sxs[tid] = gr < rows ? x_scales[(long long)gr * blocks_per_row + b] : 0.0f;
            int go = out0 + tid;
            sws[tid] = go < outs ? w_scales[(long long)go * blocks_per_row + b] : 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < 4; i++) {
            int r = tr + 16 * i;
            for (int j = 0; j < 4; j++) {
                int o = to + 16 * j;
                int s = 0;
                for (int k = 0; k < 8; k++) {
                    s = camelid_dp4a(sx[r * PAD + k], sw[o * PAD + k], s);
                }
                acc[i][j] = acc[i][j] + ((float)s * sws[o]) * sxs[r];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        int gr = row0 + tr + 16 * i;
        if (gr >= rows) continue;
        for (int j = 0; j < 4; j++) {
            int go = out0 + to + 16 * j;
            if (go < outs) out[(long long)gr * outs + go] = acc[i][j];
        }
    }
}

// Split qkv rows into q, k, v and rotate q and k: `apply_neox_rope` with the
// CPU's cos/sin tables. One thread per (row, head, pair).
extern "C" __global__ void encoder_rope_split(
    const float* __restrict__ qkv, const int* __restrict__ positions, int rows, int heads,
    int head_dim, const float* __restrict__ cos_table, const float* __restrict__ sin_table,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v
) {
    int half = head_dim / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * heads * half;
    if (idx >= total) return;
    int pair = (int)(idx % half);
    int head = (int)((idx / half) % heads);
    long long row = idx / ((long long)half * heads);
    int width = heads * head_dim;
    const float* src = qkv + row * 3 * width;
    int col_left = head * head_dim + pair;
    int col_right = col_left + half;
    long long left = row * width + col_left;
    long long right = row * width + col_right;
    long long t = (long long)positions[row] * half + pair;
    float c = cos_table[t];
    float s = sin_table[t];
    float ql = src[col_left], qr = src[col_right];
    q[left] = ql * c - qr * s;
    q[right] = ql * s + qr * c;
    float kl = src[width + col_left], kr = src[width + col_right];
    k[left] = kl * c - kr * s;
    k[right] = kl * s + kr * c;
    v[left] = src[2 * width + col_left];
    v[right] = src[2 * width + col_right];
}

// `full_attention` for one (row, head): scores against every key of the row's
// own chunk, softmax, then the value sum, all in the CPU's order.
extern "C" __global__ void encoder_attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const int* __restrict__ seg_start, const int* __restrict__ seg_len, int rows, int heads,
    int head_dim, float scale, int max_len, float* __restrict__ scratch, float* __restrict__ out,
    int* __restrict__ failed
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * heads) return;
    int head = (int)(idx % heads);
    long long row = idx / heads;
    int width = heads * head_dim;
    int start = seg_start[row];
    int len = seg_len[row];
    const float* qr = q + row * width + head * head_dim;
    float* scores = scratch + idx * max_len;
    for (int key = 0; key < len; key++) {
        const float* kr = k + (long long)(start + key) * width + head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot = dot + qr[d] * kr[d];
        scores[key] = dot * scale;
    }
    float maximum = __int_as_float(0xff800000); // -inf
    for (int key = 0; key < len; key++) maximum = fmaxf(maximum, scores[key]);
    float sum = 0.0f;
    for (int key = 0; key < len; key++) {
        scores[key] = camelid_expf(scores[key] - maximum);
        sum = sum + scores[key];
    }
    // Valid only when 0 < sum <= FLT_MAX; NaN fails both comparisons.
    if (!(sum > 0.0f && sum <= 3.402823466e+38f)) {
        atomicExch(failed, 1);
        return;
    }
    for (int key = 0; key < len; key++) scores[key] = scores[key] / sum;
    // Each output value is 0 + p0*v0 + p1*v1 + ... in key order, as on the CPU;
    // 16 of them are accumulated in registers at a time and written once.
    float* o = out + row * width + head * head_dim;
    for (int d0 = 0; d0 < head_dim; d0 += 16) {
        float acc[16];
        for (int j = 0; j < 16; j++) acc[j] = 0.0f;
        for (int key = 0; key < len; key++) {
            float p = scores[key];
            const float* vr = v + (long long)(start + key) * width + head * head_dim + d0;
            for (int j = 0; j < 16; j++) {
                if (d0 + j < head_dim) acc[j] = acc[j] + p * vr[j];
            }
        }
        for (int j = 0; j < 16; j++) {
            if (d0 + j < head_dim) o[d0 + j] = acc[j];
        }
    }
}

// x += residual, then `layer_norm_in_place` (f64, CPU order). One thread per row.
extern "C" __global__ void encoder_residual_layer_norm(
    float* __restrict__ x, const float* __restrict__ residual, int rows, int cols,
    const float* __restrict__ weight, const float* __restrict__ bias, float epsilon
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float* r = x + row * cols;
    const float* res = residual + row * cols;
    for (int c = 0; c < cols; c++) r[c] = r[c] + res[c];
    double sum = 0.0;
    for (int c = 0; c < cols; c++) sum = sum + (double)r[c];
    double mean = sum / (double)cols;
    double var = 0.0;
    for (int c = 0; c < cols; c++) {
        double delta = (double)r[c] - mean;
        var = var + delta * delta;
    }
    var = var / (double)cols;
    double inverse_stddev = 1.0 / sqrt(var + (double)epsilon);
    for (int c = 0; c < cols; c++) {
        r[c] = (float)(((double)r[c] - mean) * inverse_stddev) * weight[c] + bias[c];
    }
}

// gate = silu(gate) * up.
extern "C" __global__ void encoder_silu_mul(
    float* __restrict__ gate, const float* __restrict__ up, long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + camelid_expf(-g))) * up[i];
}
"#;

/// Rows (tokens) per GPU forward; bounds activation memory at roughly 60 KB a row.
const MAX_ROWS_PER_FORWARD: usize = 8192;

struct DeviceLinear {
    scales: CudaSlice<f32>,
    quants: CudaSlice<i8>,
    outs: usize,
    blocks_per_row: usize,
}

struct DeviceLayer {
    qkv: DeviceLinear,
    attention_output: DeviceLinear,
    ffn_gate: DeviceLinear,
    ffn_up: DeviceLinear,
    ffn_down: DeviceLinear,
    attention_norm_weight: CudaSlice<f32>,
    attention_norm_bias: CudaSlice<f32>,
    output_norm_weight: CudaSlice<f32>,
    output_norm_bias: CudaSlice<f32>,
}

struct Kernels {
    quantize: CudaFunction,
    gemm: CudaFunction,
    rope_split: CudaFunction,
    attention: CudaFunction,
    residual_layer_norm: CudaFunction,
    silu_mul: CudaFunction,
}

pub(super) struct CudaEncoder {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    layers: Vec<DeviceLayer>,
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
    width: usize,
    ffn: usize,
    heads: usize,
    head_dim: usize,
    epsilon: f32,
    device_name: String,
    /// Activation buffers, reused while the next group fits. A replacement
    /// uses the new group's dimensions so unrelated high-water marks do not
    /// multiply into an unnecessarily large attention scratch buffer.
    buffers: Option<Buffers>,
}

/// The lazily built CUDA encoder of one [`NomicBertRuntime`].
#[derive(Default)]
pub(super) struct Slot(OnceLock<Option<Mutex<Option<CudaEncoder>>>>);

/// Set once a CUDA forward fails: every later batch in this process uses the CPU.
static FAILED: OnceLock<()> = OnceLock::new();

impl Slot {
    /// `Ok(None)` means the CUDA lane is off, unavailable, or (with `wait`
    /// false) busy: use the CPU.
    pub(super) fn embed(
        &self,
        runtime: &NomicBertRuntime,
        inputs: &[String],
        wait: bool,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        if !crate::cuda::gpu_accel_enabled() || disabled_by_env() || FAILED.get().is_some() {
            return Ok(None);
        }
        let Some(encoder) = self
            .0
            .get_or_init(|| match CudaEncoder::build(runtime) {
                Ok(encoder) => {
                    eprintln!(
                        "[embedding] document encoder layers run on CUDA ({})",
                        encoder.device_name
                    );
                    Some(Mutex::new(Some(encoder)))
                }
                Err(reason) => {
                    tracing::warn!(%reason, "CUDA document encoder unavailable; using the CPU");
                    None
                }
            })
            .as_ref()
        else {
            return Ok(None);
        };
        let prepared = super::embedding_batch_pool().install(|| {
            use rayon::prelude::*;
            inputs
                .par_iter()
                .map(|input| runtime.embed_input(input))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut slot = match if wait {
            encoder.lock().map_err(TryLockError::Poisoned)
        } else {
            encoder.try_lock()
        } {
            Ok(slot) => slot,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => {
                let _ = FAILED.set(());
                let mut slot = poisoned.into_inner();
                if let Some(encoder) = slot.take() {
                    drop(encoder);
                    crate::cuda::release_async_pool();
                }
                tracing::warn!("CUDA document encoder lock poisoned; using the CPU from now on");
                return Ok(None);
            }
        };
        // Keep the queue locked until pooling has validated the output. Failure
        // must disable the lane and release its allocations before a waiting
        // query can enter or the caller retries on the CPU.
        let had_encoder = slot.is_some();
        let result = run_encoder(&mut slot, &FAILED, |encoder| {
            let hidden = encoder.forward(&prepared)?;
            prepared
                .into_iter()
                .zip(hidden)
                .map(|((token_ids, _), hidden)| {
                    runtime
                        .output(token_ids, hidden)
                        .pool(runtime.config.pooling)
                        .map_err(|error| error.to_string())
                })
                .collect()
        });
        if had_encoder && slot.is_none() {
            // CudaSlice drops enqueue async frees. Drain and trim the selected
            // device's pool before CPU fallback so fit probes see reclaimed VRAM.
            crate::cuda::release_async_pool();
        }
        match result {
            Ok(vectors) => Ok(vectors),
            Err(reason) => {
                tracing::warn!(%reason, "CUDA document encoder failed; using the CPU from now on");
                Ok(None)
            }
        }
    }
}

/// Called with the slot's mutex held, including the failure check and cleanup.
fn run_encoder<T, R>(
    slot: &mut Option<T>,
    failed: &OnceLock<()>,
    run: impl FnOnce(&mut T) -> std::result::Result<R, String>,
) -> std::result::Result<Option<R>, String> {
    // A caller may have queued before another batch disabled the CUDA lane.
    if failed.get().is_some() {
        slot.take();
        return Ok(None);
    }
    let Some(encoder) = slot.as_mut() else {
        return Ok(None);
    };
    match run(encoder) {
        Ok(output) => Ok(Some(output)),
        Err(reason) => {
            let _ = failed.set(());
            slot.take();
            Err(reason)
        }
    }
}

fn buffer_shape_for_request(
    cached: Option<(usize, usize)>,
    rows: usize,
    max_len: usize,
) -> (usize, usize) {
    cached
        .filter(|&(cached_rows, cached_max_len)| cached_rows >= rows && cached_max_len >= max_len)
        .unwrap_or((rows, max_len))
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

fn launch_1d(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn driver(error: cudarc::driver::DriverError) -> String {
    format!("cuda: {error}")
}

struct Buffers {
    rows: usize,
    max_len: usize,
    hidden: CudaSlice<f32>,
    positions: CudaSlice<i32>,
    seg_start: CudaSlice<i32>,
    seg_len: CudaSlice<i32>,
    t: Scratch,
}

struct Scratch {
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attended: CudaSlice<f32>,
    projected: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    x_scales: CudaSlice<f32>,
    x_quants: CudaSlice<i8>,
    scores: CudaSlice<f32>,
    failed: CudaSlice<i32>,
}

impl CudaEncoder {
    fn build(runtime: &NomicBertRuntime) -> std::result::Result<Self, String> {
        if crate::cuda::devices_masked_by_env() {
            return Err("CUDA devices masked by CUDA_VISIBLE_DEVICES".into());
        }
        let ordinal = crate::cuda::selected_device_ordinal();
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|_| "CUDA driver library not available".to_string())?
            .map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
        let device_name = ctx.name().unwrap_or_else(|_| "CUDA device".into());
        let stream = ctx.new_stream().map_err(driver)?;
        let options = CompileOptions {
            fmad: Some(false),
            arch: Some("compute_61"),
            ..Default::default()
        };
        let ptx =
            std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx_with_opts(KERNELS, options))
                .map_err(|_| "CUDA NVRTC library not available".to_string())?
                .map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let module = ctx.load_module(ptx).map_err(driver)?;
        let function = |name: &str| module.load_function(name).map_err(driver);
        let kernels = Kernels {
            quantize: function("encoder_quantize_q8")?,
            gemm: function("encoder_q8_gemm")?,
            rope_split: function("encoder_rope_split")?,
            attention: function("encoder_attention")?,
            residual_layer_norm: function("encoder_residual_layer_norm")?,
            silu_mul: function("encoder_silu_mul")?,
        };
        let config = &runtime.config;
        let width = config.embedding_length;
        let ffn = config.feed_forward_length;
        let upload_linear = |tensor: &CpuTensor, inputs: usize, outs: usize| {
            let blocks = tensor
                .q8_0_blocks
                .as_ref()
                .ok_or_else(|| "encoder matrix has no Q8_0 blocks".to_string())?;
            let blocks_per_row = inputs / 32;
            if !inputs.is_multiple_of(32) || blocks.len() != outs * blocks_per_row {
                return Err(format!(
                    "encoder matrix has {} blocks, expected {outs} x {blocks_per_row}",
                    blocks.len()
                ));
            }
            let scales: Vec<f32> = blocks.iter().map(|block| block.scale).collect();
            let quants: Vec<i8> = blocks.iter().flat_map(|block| block.quants).collect();
            Ok::<_, String>(DeviceLinear {
                scales: stream.clone_htod(&scales).map_err(driver)?,
                quants: stream.clone_htod(&quants).map_err(driver)?,
                outs,
                blocks_per_row,
            })
        };
        let vector = |values: &[f32]| stream.clone_htod(values).map_err(driver);
        let mut layers = Vec::with_capacity(runtime.layers.len());
        for layer in &runtime.layers {
            layers.push(DeviceLayer {
                qkv: upload_linear(&layer.qkv, width, width * 3)?,
                attention_output: upload_linear(&layer.attention_output, width, width)?,
                ffn_gate: upload_linear(&layer.ffn_gate, width, ffn)?,
                ffn_up: upload_linear(&layer.ffn_up, width, ffn)?,
                ffn_down: upload_linear(&layer.ffn_down, ffn, width)?,
                attention_norm_weight: vector(&layer.attention_norm_weight)?,
                attention_norm_bias: vector(&layer.attention_norm_bias)?,
                output_norm_weight: vector(&layer.output_norm_weight)?,
                output_norm_bias: vector(&layer.output_norm_bias)?,
            });
        }
        let (cos, sin) = rope_tables(
            config.context_length,
            config.head_dim,
            config.rope_frequency_base,
        );
        let rope_cos = vector(&cos)?;
        let rope_sin = vector(&sin)?;
        stream.synchronize().map_err(driver)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            kernels,
            layers,
            rope_cos,
            rope_sin,
            width,
            ffn,
            heads: config.head_count,
            head_dim: config.head_dim,
            epsilon: config.layer_norm_epsilon,
            device_name,
            buffers: None,
        })
    }

    /// Runs the layer stack over every prepared input (the token ids and input
    /// rows from `embed_input`) and returns each input's final rows.
    fn forward(
        &mut self,
        prepared: &[(Vec<u32>, Vec<f32>)],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut outputs = Vec::with_capacity(prepared.len());
        let mut group = Vec::new();
        let mut rows = 0;
        for item in prepared {
            if !group.is_empty() && rows + item.0.len() > MAX_ROWS_PER_FORWARD {
                outputs.extend(self.forward_group(&group)?);
                group.clear();
                rows = 0;
            }
            rows += item.0.len();
            group.push(item);
        }
        if !group.is_empty() {
            outputs.extend(self.forward_group(&group)?);
        }
        Ok(outputs)
    }

    /// Buffers for at least `rows` rows and chunks of up to `max_len` tokens.
    fn take_buffers(
        &mut self,
        rows: usize,
        max_len: usize,
    ) -> std::result::Result<Buffers, String> {
        let (rows, max_len) = buffer_shape_for_request(
            self.buffers
                .as_ref()
                .map(|buffers| (buffers.rows, buffers.max_len)),
            rows,
            max_len,
        );
        if let Some(buffers) = self.buffers.take() {
            if (buffers.rows, buffers.max_len) == (rows, max_len) {
                return Ok(buffers);
            }
            drop(buffers);
        }
        self.allocate(rows, max_len)
    }

    fn allocate(&self, rows: usize, max_len: usize) -> std::result::Result<Buffers, String> {
        let s = &self.stream;
        Ok(Buffers {
            rows,
            max_len,
            hidden: s.alloc_zeros(rows * self.width).map_err(driver)?,
            positions: s.alloc_zeros(rows).map_err(driver)?,
            seg_start: s.alloc_zeros(rows).map_err(driver)?,
            seg_len: s.alloc_zeros(rows).map_err(driver)?,
            t: self.scratch(rows, max_len)?,
        })
    }

    fn scratch(&self, rows: usize, max_len: usize) -> std::result::Result<Scratch, String> {
        let s = &self.stream;
        let (width, ffn) = (self.width, self.ffn);
        Ok(Scratch {
            qkv: s.alloc_zeros(rows * width * 3).map_err(driver)?,
            q: s.alloc_zeros(rows * width).map_err(driver)?,
            k: s.alloc_zeros(rows * width).map_err(driver)?,
            v: s.alloc_zeros(rows * width).map_err(driver)?,
            attended: s.alloc_zeros(rows * width).map_err(driver)?,
            projected: s.alloc_zeros(rows * width).map_err(driver)?,
            gate: s.alloc_zeros(rows * ffn).map_err(driver)?,
            up: s.alloc_zeros(rows * ffn).map_err(driver)?,
            x_scales: s.alloc_zeros(rows * ffn / 32).map_err(driver)?,
            x_quants: s.alloc_zeros(rows * ffn).map_err(driver)?,
            scores: s.alloc_zeros(rows * self.heads * max_len).map_err(driver)?,
            failed: s.alloc_zeros(1).map_err(driver)?,
        })
    }

    fn forward_group(
        &mut self,
        group: &[&(Vec<u32>, Vec<f32>)],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let rows: usize = group.iter().map(|item| item.0.len()).sum();
        let max_len = group.iter().map(|item| item.0.len()).max().unwrap_or(0);
        let mut buffers = self.take_buffers(rows, max_len)?;
        let result = self.run_group(group, rows, max_len, &mut buffers);
        self.buffers = Some(buffers);
        result
    }

    fn run_group(
        &self,
        group: &[&(Vec<u32>, Vec<f32>)],
        rows: usize,
        max_len: usize,
        buffers: &mut Buffers,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let s = &self.stream;
        let (width, ffn, heads, head_dim) = (self.width, self.ffn, self.heads, self.head_dim);
        let mut host_hidden = Vec::with_capacity(rows * width);
        let mut positions = Vec::with_capacity(rows);
        let mut seg_start = Vec::with_capacity(rows);
        let mut seg_len = Vec::with_capacity(rows);
        for (token_ids, hidden) in group.iter().copied() {
            let start = positions.len() as i32;
            for position in 0..token_ids.len() {
                positions.push(position as i32);
                seg_start.push(start);
                seg_len.push(token_ids.len() as i32);
            }
            host_hidden.extend_from_slice(hidden);
        }
        let Buffers {
            ref mut hidden,
            positions: ref mut device_positions,
            seg_start: ref mut device_seg_start,
            seg_len: ref mut device_seg_len,
            ref mut t,
            ..
        } = *buffers;
        s.memcpy_htod(&host_hidden, &mut hidden.slice_mut(..rows * width))
            .map_err(driver)?;
        s.memcpy_htod(&positions, &mut device_positions.slice_mut(..rows))
            .map_err(driver)?;
        s.memcpy_htod(&seg_start, &mut device_seg_start.slice_mut(..rows))
            .map_err(driver)?;
        s.memcpy_htod(&seg_len, &mut device_seg_len.slice_mut(..rows))
            .map_err(driver)?;
        s.memcpy_htod(&[0_i32], &mut t.failed).map_err(driver)?;
        let (positions, seg_start, seg_len) =
            (&*device_positions, &*device_seg_start, &*device_seg_len);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (rows_i, heads_i, head_dim_i, max_len_i, width_i) = (
            rows as i32,
            heads as i32,
            head_dim as i32,
            max_len as i32,
            width as i32,
        );

        for layer in &self.layers {
            self.linear(
                hidden,
                rows,
                &layer.qkv,
                &mut t.x_scales,
                &mut t.x_quants,
                &mut t.qkv,
            )?;
            let mut b = s.launch_builder(&self.kernels.rope_split);
            b.arg(&t.qkv)
                .arg(positions)
                .arg(&rows_i)
                .arg(&heads_i)
                .arg(&head_dim_i)
                .arg(&self.rope_cos)
                .arg(&self.rope_sin)
                .arg(&mut t.q)
                .arg(&mut t.k)
                .arg(&mut t.v);
            // SAFETY: q/k/v hold rows*width values and qkv rows*3*width; every
            // position is below the context length the tables cover.
            unsafe { b.launch(launch_1d(rows * heads * (head_dim / 2))) }.map_err(driver)?;
            let mut b = s.launch_builder(&self.kernels.attention);
            b.arg(&t.q)
                .arg(&t.k)
                .arg(&t.v)
                .arg(seg_start)
                .arg(seg_len)
                .arg(&rows_i)
                .arg(&heads_i)
                .arg(&head_dim_i)
                .arg(&scale)
                .arg(&max_len_i)
                .arg(&mut t.scores)
                .arg(&mut t.attended)
                .arg(&mut t.failed);
            // SAFETY: scores holds max_len values per (row, head); each segment lies inside the batch.
            unsafe { b.launch(launch_1d(rows * heads)) }.map_err(driver)?;
            self.linear(
                &t.attended,
                rows,
                &layer.attention_output,
                &mut t.x_scales,
                &mut t.x_quants,
                &mut t.projected,
            )?;
            self.residual_layer_norm(
                &mut t.projected,
                hidden,
                rows_i,
                width_i,
                &layer.attention_norm_weight,
                &layer.attention_norm_bias,
            )?;
            self.linear(
                &t.projected,
                rows,
                &layer.ffn_gate,
                &mut t.x_scales,
                &mut t.x_quants,
                &mut t.gate,
            )?;
            self.linear(
                &t.projected,
                rows,
                &layer.ffn_up,
                &mut t.x_scales,
                &mut t.x_quants,
                &mut t.up,
            )?;
            let n = (rows * ffn) as i64;
            let mut b = s.launch_builder(&self.kernels.silu_mul);
            b.arg(&mut t.gate).arg(&t.up).arg(&n);
            // SAFETY: gate and up hold rows*ffn values.
            unsafe { b.launch(launch_1d(rows * ffn)) }.map_err(driver)?;
            self.linear(
                &t.gate,
                rows,
                &layer.ffn_down,
                &mut t.x_scales,
                &mut t.x_quants,
                hidden,
            )?;
            self.residual_layer_norm(
                hidden,
                &t.projected,
                rows_i,
                width_i,
                &layer.output_norm_weight,
                &layer.output_norm_bias,
            )?;
        }
        let host = s
            .clone_dtoh(&hidden.slice(..rows * width))
            .map_err(driver)?;
        let failed = s.clone_dtoh(&t.failed).map_err(driver)?;
        s.synchronize().map_err(driver)?;
        if failed[0] != 0 {
            return Err("encoder attention softmax produced an invalid normalization sum".into());
        }
        let mut outputs = Vec::with_capacity(group.len());
        let mut offset = 0;
        for (token_ids, _) in group {
            let len = token_ids.len() * width;
            outputs.push(host[offset..offset + len].to_vec());
            offset += len;
        }
        Ok(outputs)
    }

    /// `output = input @ weight^T` with the input quantized to Q8 first, as the
    /// CPU block-dot linear does.
    fn linear(
        &self,
        input: &CudaSlice<f32>,
        rows: usize,
        weight: &DeviceLinear,
        x_scales: &mut CudaSlice<f32>,
        x_quants: &mut CudaSlice<i8>,
        output: &mut CudaSlice<f32>,
    ) -> std::result::Result<(), String> {
        let s = &self.stream;
        let blocks = (rows * weight.blocks_per_row) as i64;
        let mut b = s.launch_builder(&self.kernels.quantize);
        b.arg(input)
            .arg(&blocks)
            .arg(&mut *x_scales)
            .arg(&mut *x_quants);
        // SAFETY: input holds rows*blocks_per_row*32 values; the scratch fits rows*ffn.
        unsafe { b.launch(launch_1d(blocks as usize)) }.map_err(driver)?;
        let (rows_i, outs_i, blocks_i) = (
            rows as i32,
            weight.outs as i32,
            weight.blocks_per_row as i32,
        );
        let config = LaunchConfig {
            grid_dim: (weight.outs.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&self.kernels.gemm);
        b.arg(&*x_scales)
            .arg(&*x_quants)
            .arg(&rows_i)
            .arg(&weight.scales)
            .arg(&weight.quants)
            .arg(&outs_i)
            .arg(&blocks_i)
            .arg(output);
        // SAFETY: output holds rows*outs values; tiles past either edge are masked.
        unsafe { b.launch(config) }.map_err(driver)?;
        Ok(())
    }

    fn residual_layer_norm(
        &self,
        x: &mut CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        rows: i32,
        cols: i32,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
    ) -> std::result::Result<(), String> {
        let mut b = self
            .stream
            .launch_builder(&self.kernels.residual_layer_norm);
        b.arg(x)
            .arg(residual)
            .arg(&rows)
            .arg(&cols)
            .arg(weight)
            .arg(bias)
            .arg(&self.epsilon);
        // SAFETY: x and residual hold rows*cols values; weight and bias hold cols.
        unsafe { b.launch(launch_1d(rows as usize)) }.map_err(driver)?;
        Ok(())
    }
}

/// The cos/sin of every (position, pair), computed as `apply_neox_rope` does.
fn rope_tables(positions: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(positions * half);
    let mut sin = Vec::with_capacity(positions * half);
    for position in 0..positions {
        for pair in 0..half {
            let frequency = 1.0 / base.powf((2 * pair) as f32 / head_dim as f32);
            let angle = position as f32 * frequency;
            let (s, c) = angle.sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    (cos, sin)
}

#[cfg(test)]
impl CudaEncoder {
    pub(super) fn build_for_test(runtime: &NomicBertRuntime) -> std::result::Result<Self, String> {
        Self::build(runtime)
    }

    pub(super) fn forward_for_test(
        &mut self,
        prepared: &[(Vec<u32>, Vec<f32>)],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        self.forward(prepared)
    }

    /// Layer `layer`'s qkv projection on the GPU, to compare with the CPU's bits.
    pub(super) fn qkv_for_test(
        &self,
        layer: usize,
        input: &[f32],
        rows: usize,
    ) -> std::result::Result<Vec<f32>, String> {
        let s = &self.stream;
        let weight = &self.layers[layer].qkv;
        let input = s.clone_htod(input).map_err(driver)?;
        let mut x_scales = s.alloc_zeros(rows * self.width / 32).map_err(driver)?;
        let mut x_quants = s.alloc_zeros(rows * self.width).map_err(driver)?;
        let mut output = s.alloc_zeros(rows * weight.outs).map_err(driver)?;
        self.linear(
            &input,
            rows,
            weight,
            &mut x_scales,
            &mut x_quants,
            &mut output,
        )?;
        let host = s.clone_dtoh(&output).map_err(driver)?;
        s.synchronize().map_err(driver)?;
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn failed_encoder_is_dropped_before_a_waiting_caller_can_run() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = Mutex::new(Some(DropProbe(Arc::clone(&drops))));
        let failed = OnceLock::new();
        std::thread::scope(|scope| {
            let mut active = slot.lock().unwrap();
            let waiting = scope.spawn(|| {
                let mut queued = slot.lock().unwrap();
                run_encoder(
                    &mut queued,
                    &failed,
                    |_| -> std::result::Result<(), String> {
                        panic!("a queued caller must use the CPU after CUDA fails")
                    },
                )
            });
            let result = run_encoder(&mut active, &failed, |_| {
                Err::<(), _>("injected CUDA or pooling failure".to_string())
            });
            assert!(result.is_err());
            assert!(failed.get().is_some());
            assert!(active.is_none());
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            drop(active);
            assert_eq!(waiting.join().unwrap().unwrap(), None);
        });
    }

    #[test]
    fn process_failure_discards_another_waiting_encoders_allocations() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut slot = Some(DropProbe(Arc::clone(&drops)));
        let failed = OnceLock::from(());
        assert_eq!(
            run_encoder(&mut slot, &failed, |_| -> std::result::Result<(), String> {
                panic!("an already disabled lane must not run")
            })
            .unwrap(),
            None
        );
        assert!(slot.is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn successful_encoder_retains_its_allocations() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut slot = Some(DropProbe(Arc::clone(&drops)));
        let failed = OnceLock::new();
        assert_eq!(run_encoder(&mut slot, &failed, |_| Ok(42)), Ok(Some(42)));
        assert!(slot.is_some());
        assert!(failed.get().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn buffer_replacement_does_not_multiply_unrelated_high_water_marks() {
        // A full indexing batch followed by one long query needs the same
        // attention scratch capacity, not eight times as much.
        let indexing = buffer_shape_for_request(None, 8192, 128);
        let query = buffer_shape_for_request(Some(indexing), 1024, 1024);
        assert_eq!(query, (1024, 1024));
        assert_eq!(query.0 * query.1, indexing.0 * indexing.1);
        assert_eq!(buffer_shape_for_request(Some(query), 8192, 128), indexing);
        assert_eq!(buffer_shape_for_request(Some(indexing), 512, 64), indexing);
    }

    #[test]
    #[ignore = "needs CAMELID_NOMIC_GGUF and a CUDA device"]
    fn cuda_buffer_replacement_uses_the_requested_shape() {
        let model = std::env::var("CAMELID_NOMIC_GGUF").expect("CAMELID_NOMIC_GGUF");
        let runtime = super::super::EmbeddingRuntime::load(&model).expect("load encoder");
        let super::super::EmbeddingRuntime::Nomic(nomic) = runtime else {
            panic!("not the Nomic encoder");
        };
        let mut encoder = CudaEncoder::build(&nomic).expect("CUDA encoder");
        encoder.buffers = Some(encoder.take_buffers(64, 2).unwrap());
        let buffers = encoder.take_buffers(8, 16).unwrap();
        assert_eq!((buffers.rows, buffers.max_len), (8, 16));
        assert_eq!(buffers.t.scores.len(), 8 * encoder.heads * 16);
        encoder.buffers = Some(buffers);
        let buffers = encoder.take_buffers(4, 8).unwrap();
        assert_eq!((buffers.rows, buffers.max_len), (8, 16));
    }

    #[test]
    fn only_explicit_off_values_disable_the_cuda_lane() {
        for off in ["0", "false", "OFF", " no "] {
            assert!(disabled_by(Some(off)), "{off}");
        }
        for on in [None, Some("1"), Some("on"), Some("")] {
            assert!(!disabled_by(on), "{on:?}");
        }
    }

    /// The GPU rotates q and k with these tables; they must give exactly what
    /// `apply_neox_rope` computes on the CPU.
    #[test]
    fn rope_tables_reproduce_the_cpu_rotation_bit_for_bit() {
        let (tokens, heads, head_dim, base) = (37, 3, 8, 1000.0_f32);
        let width = heads * head_dim;
        let original: Vec<f32> = (0..tokens * width)
            .map(|i| ((i * 7919 % 1013) as f32 - 506.0) * 0.003)
            .collect();
        let mut cpu = original.clone();
        super::super::apply_neox_rope(&mut cpu, tokens, heads, head_dim, base);
        let (cos, sin) = rope_tables(tokens, head_dim, base);
        let half = head_dim / 2;
        let mut gpu_formula = original.clone();
        for position in 0..tokens {
            for head in 0..heads {
                for pair in 0..half {
                    let left = position * width + head * head_dim + pair;
                    let right = left + half;
                    let (c, s) = (cos[position * half + pair], sin[position * half + pair]);
                    let (l, r) = (original[left], original[right]);
                    gpu_formula[left] = l * c - r * s;
                    gpu_formula[right] = l * s + r * c;
                }
            }
        }
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&gpu_formula), bits(&cpu));
    }
}
