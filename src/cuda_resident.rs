//! GPU-resident decode kernels for the CUDA backend (`--features cuda`).
//!
//! This module holds the CUDA kernels that, together, run a full Llama decode
//! step on the GPU with weights resident and one sync per token — the analog of
//! `metal.rs`'s resident decode path. Each kernel mirrors the exact math of the
//! CPU reference (RMSNorm, Q8_0 quantize + dot, RoPE adjacent-even-odd,
//! GQA attention with f16-rounded KV, SwiGLU, residual, greedy argmax) so the
//! produced tokens are identical. The kernels are validated against small CPU
//! references in this file before being assembled into the per-token forward.
//!
//! The whole module is behind `#[cfg(feature = "cuda")]` (applied to the `mod`
//! declaration in `lib.rs`); nothing here compiles into the default build.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaStream, CudaView};
use cudarc::nvrtc::{CompileOptions, Ptx};

/// CUDA C source for every resident-decode kernel. Compiled once via NVRTC with
/// `--fmad=false` and `arch=compute_61` (for `__dp4a`).
const KERNELS: &str = r#"
// ---- f16 round-trip (header-free) ------------------------------------------
// Bit-exact port of inference.rs f32_to_f16_bits / f16_bits_to_f32 (IEEE-754
// round-half-to-even). Used wherever the CPU reference rounds a value through
// f16 (Q8_0 block scales, KV cache writes). Pure integer/float bit ops via the
// always-available __float_as_uint / __uint_as_float builtins, so NVRTC needs
// no cuda_fp16.h (whose __float2half/__half2float are not always defined).
__device__ __forceinline__ unsigned short f32_to_f16_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    int exp = (int)((bits >> 23) & 0xffu);
    unsigned int mant = bits & 0x007fffffu;
    if (exp == 0xff) {
        return (unsigned short)(sign | (mant == 0u ? 0x7c00u : 0x7e00u));
    }
    int half_exp = exp - 127 + 15;
    if (half_exp >= 0x1f) {
        return (unsigned short)(sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        unsigned int mantissa = mant | 0x00800000u;
        int shift = 14 - half_exp;
        unsigned short half_mant = (unsigned short)(mantissa >> shift);
        unsigned int round_bit = 1u << (shift - 1);
        if ((mantissa & round_bit) != 0u &&
            ((mantissa & (round_bit - 1u)) != 0u || (half_mant & 1u) != 0u)) {
            half_mant = (unsigned short)(half_mant + 1);
        }
        return (unsigned short)(sign | half_mant);
    }
    unsigned short half = (unsigned short)(sign
        | ((unsigned short)half_exp << 10) | (unsigned short)(mant >> 13));
    if ((mant & 0x00001000u) != 0u && ((mant & 0x00000fffu) != 0u || (half & 1u) != 0u)) {
        half = (unsigned short)(half + 1);
    }
    return half;
}
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    unsigned int exp = (bits & 0x7c00u) >> 10;
    unsigned int frac = (unsigned int)(bits & 0x03ffu);
    unsigned int out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            unsigned int mant = frac;
            int e = -14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; e -= 1; }
            mant &= 0x03ffu;
            unsigned int exp32 = (unsigned int)(e + 127);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        unsigned int exp32 = exp + (127u - 15u);
        out = sign | (exp32 << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}
__device__ __forceinline__ float f16_round(float x) {
    return f16_bits_to_f32(f32_to_f16_bits(x));
}

// ---- RMSNorm: out[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i] -------------
// One block, blockDim threads, shared-memory sum of squares.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, int n, float eps
) {
    // The sum-of-squares must stay in CPU order (i = 0,1,2,...): a parallel tree
    // reduction reassociates it and was measured to change greedy tokens (a parity
    // regression). But running that serial scan in one thread off global memory
    // left 255 threads idle and the SM stalled on load latency (~31us). Instead all
    // threads cooperatively stage the row into shared memory (coalesced), then
    // thread 0 sums it in order from on-chip shared (~few us) -- identical
    // arithmetic, identical order, just no global-latency stall. The per-element
    // apply is parallel and order-independent.
    extern __shared__ float xs[]; // n floats
    __shared__ float s_scale;
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += xs[i] * xs[i];
        float mean_sq = sum / (float)n;
        s_scale = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = tid; i < n; i += blockDim.x) out[i] = xs[i] * scale * weight[i];
}

// ---- Per-head RMSNorm (Qwen3 QK-norm): one block per head, serial sum ------
// Applies RMSNorm in-place to each head's head_dim slice. Weight is [head_dim]
// and shared across all heads. The sum-of-squares uses the same serial-in-shared-
// memory strategy as rms_norm_f32 to match CPU ordering (thread 0 sums, all apply).
// In-place safe: reads to shared memory before writing back.
extern "C" __global__ void rms_norm_per_head_f32(
    float* __restrict__ buf,
    const float* __restrict__ weight,
    int head_dim, float eps, int use_weight
) {
    extern __shared__ float xs[];
    __shared__ float s_scale;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int base = head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) xs[i] = buf[base + i];
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++) sum += xs[i] * xs[i];
        s_scale = 1.0f / sqrtf(sum / (float)head_dim + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = xs[i] * scale;
        if (use_weight != 0) v *= weight[i];
        buf[base + i] = v;
    }
}

// ---- Quantize f32 row to Q8_0 blocks (matches quantize_q8_0_block) ---------
// One thread per 32-value block. scale is f16-rounded; quants use the unrounded
// inverse and round-half-to-even, clamped to [-128, 127].
extern "C" __global__ void quantize_q8_0(
    const float* __restrict__ x, signed char* __restrict__ quants,
    float* __restrict__ scales, int n_blocks
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + (long)b * 32;
    float max_abs = 0.0f;
    for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > max_abs) max_abs = a; }
    float unrounded = max_abs / 127.0f;
    scales[b] = f16_round(unrounded); // f16-rounded block scale
    float inv = (unrounded == 0.0f) ? 0.0f : 1.0f / unrounded;
    signed char* qb = quants + (long)b * 32;
    for (int j = 0; j < 32; j++) {
        float v = rintf(xb[j] * inv);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qb[j] = (signed char)v;
    }
}

// ---- Fused RMS-norm + Q8_0 quantize (F1) -----------------------------------
// One block stages the row in shared, thread 0 does the in-order sum-of-squares
// (bit-identical to rms_norm_f32), every thread applies norm*weight back into shared,
// then quantizes 32-wide blocks straight from shared (bit-identical to quantize_q8_0).
// Fuses two kernels + drops the f32 `normed` global round-trip — same arithmetic.
extern "C" __global__ void rms_norm_quantize(
    const float* __restrict__ x, const float* __restrict__ weight,
    signed char* __restrict__ quants, float* __restrict__ scales, int n, float eps
) {
    extern __shared__ float xs[]; // n floats
    __shared__ float s_scale;
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += xs[i] * xs[i]; // CPU-order serial sum
        s_scale = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = tid; i < n; i += blockDim.x) xs[i] = xs[i] * scale * weight[i];
    __syncthreads();
    int n_blocks = n >> 5; // n / 32
    for (int b = tid; b < n_blocks; b += blockDim.x) {
        const float* xb = xs + ((long)b << 5);
        float max_abs = 0.0f;
        for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > max_abs) max_abs = a; }
        float unrounded = max_abs / 127.0f;
        scales[b] = f16_round(unrounded); // f16-rounded block scale
        float inv = (unrounded == 0.0f) ? 0.0f : 1.0f / unrounded;
        signed char* qb = quants + (long)b * 32;
        for (int j = 0; j < 32; j++) {
            float v = rintf(xb[j] * inv);
            if (v > 127.0f) v = 127.0f;
            if (v < -128.0f) v = -128.0f;
            qb[j] = (signed char)v;
        }
    }
}

// ---- Q8_0 GEMV: one warp per output row, __dp4a dot, ordered float sum -------
// weight_bytes is the repacked SoA layout (see repack_q8_soa): all quants first
// (rows*blocks_per_row*32 i8, 16-byte aligned), then all scales (rows*blocks_per_row
// f32). Quants-first means each block's 32 i8 are read as two aligned int4 loads
// instead of eight scalar int loads off a 36-byte stride, which lifts the kernel
// off ~52% of memory bandwidth. The math is unchanged: the integer block dot
// (__dp4a) is exact regardless of order, and the per-block float terms are still
// summed sequentially by lane 0 in block order (acc += int_sum * w_scale *
// x_scale), reproducing the CPU reference's summation order so the decode stays
// token-identical. Only the load instructions change, not the arithmetic.
// The input activation (quants + scales) is the SAME for every output row, so
// instead of each of the block's 8 warps re-reading it from global for its row,
// the block stages the whole input vector into shared memory once and every warp
// reads it from on-chip shared. That removes a chunk of memory traffic roughly
// equal to the weight traffic for the larger projections (down/gate/up), where
// the input is as big as one weight row. Shared layout: input quants, then input
// scales, then the per-warp ordered-sum scratch.
extern "C" __global__ void q8_gemv(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem[];
    signed char* s_iq = (signed char*)smem;                          // blocks_per_row*32 i8
    float* s_is = (float*)(smem + (long)blocks_per_row * 32);         // blocks_per_row f32
    float* terms = (float*)(smem + (long)blocks_per_row * 36);        // warps*blocks_per_row f32
    int tid = threadIdx.x;
    // Stage the shared input vector cooperatively (coalesced), once per block.
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i]; // blocks_per_row*32 bytes as ints
    for (int i = tid; i < blocks_per_row; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    if (row < rows) {
        long total_blocks = (long)rows * blocks_per_row;
        const signed char* quants = reinterpret_cast<const signed char*>(weight_bytes);
        const float* scales =
            reinterpret_cast<const float*>(weight_bytes + total_blocks * 32);
        long row_block0 = (long)row * blocks_per_row;
        const int4* siq = reinterpret_cast<const int4*>(s_iq);
        // Process U blocks per lane-iteration: issue all U weight loads FIRST, then do the
        // dp4a math — so ~U weight loads are in flight at once instead of ~1, hiding DRAM
        // latency (the batch-1 GEMV is latency-bound, ~60% of peak DRAM otherwise). Each
        // per-u load is still coalesced across the warp (lanes read consecutive blocks), and
        // every term lands in myterms[b] by block index, so the lane-0 ordered sum below is
        // unchanged — bit-identical to the one-block-at-a-time loop.
        const int U = 4;
        for (int base = lane; base < blocks_per_row; base += 32 * U) {
            int4 w0[U], w1[U];
            float ws[U];
            int present = 0;
            #pragma unroll
            for (int u = 0; u < U; u++) {
                int b = base + u * 32;
                if (b < blocks_per_row) {
                    const int4* wq =
                        reinterpret_cast<const int4*>(quants + (row_block0 + b) * 32);
                    w0[u] = wq[0];
                    w1[u] = wq[1];
                    ws[u] = scales[row_block0 + b];
                    present |= (1 << u);
                }
            }
            #pragma unroll
            for (int u = 0; u < U; u++) {
                if (present & (1 << u)) {
                    int b = base + u * 32;
                    int4 i0 = siq[b * 2], i1 = siq[b * 2 + 1];
                    int int_sum = 0;
                    int_sum = __dp4a(w0[u].x, i0.x, int_sum);
                    int_sum = __dp4a(w0[u].y, i0.y, int_sum);
                    int_sum = __dp4a(w0[u].z, i0.z, int_sum);
                    int_sum = __dp4a(w0[u].w, i0.w, int_sum);
                    int_sum = __dp4a(w1[u].x, i1.x, int_sum);
                    int_sum = __dp4a(w1[u].y, i1.y, int_sum);
                    int_sum = __dp4a(w1[u].z, i1.z, int_sum);
                    int_sum = __dp4a(w1[u].w, i1.w, int_sum);
                    myterms[b] = (float)int_sum * ws[u] * s_is[b];
                }
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) acc += myterms[b];
        // residual!=0 fuses the post-projection residual add (output += acc), saving a
        // separate residual_add launch + the f32 projection round-trip. Bit-identical:
        // output[row] (old hidden) + acc == hidden + projection, the same f32 sum.
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- Fused SiLU-gate * up + Q8_0 quantize (F3) ------------------------------
// One thread per 32-block: compute silu(gate)*up for the block's 32 elements (bit-
// identical to silu_mul) and quantize them (bit-identical to quantize_q8_0), straight
// to the down-projection's input — no f32 `ffn_act` round-trip, one fewer launch. No
// shared memory, so it is not bounded by the FFN width.
extern "C" __global__ void silu_mul_quantize(
    const float* __restrict__ gate, const float* __restrict__ up,
    signed char* __restrict__ quants, float* __restrict__ scales, int n_blocks
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    float vals[32];
    float max_abs = 0.0f;
    for (int j = 0; j < 32; j++) {
        float g = gate[(long)b * 32 + j];
        float v = (g / (1.0f + expf(-g))) * up[(long)b * 32 + j];
        vals[j] = v;
        float a = fabsf(v);
        if (a > max_abs) max_abs = a;
    }
    float unrounded = max_abs / 127.0f;
    scales[b] = f16_round(unrounded);
    float inv = (unrounded == 0.0f) ? 0.0f : 1.0f / unrounded;
    signed char* qb = quants + (long)b * 32;
    for (int j = 0; j < 32; j++) {
        float v = rintf(vals[j] * inv);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qb[j] = (signed char)v;
    }
}

// ---- Batched Q8 GEMM: K token-inputs against M weight rows ------------------
// The speculative-decode verify runs K tokens through the model in one pass; the
// win is that each weight block is read from global ONCE and reused for all K
// tokens (vs K separate GEMVs reading the weights K times). One warp per output
// row; for each block the weight is loaded once and dotted against all K inputs.
// The per-block float terms are summed by lane 0 in block order (per token), the
// SAME ordered sum as the single-token q8_gemv, so verify_batch is bit-identical
// to K sequential forward_token calls — which makes speculative decode losslessly
// reproduce greedy decode (not just token-identical-modulo-near-ties). Shared
// holds [warp][token][block] terms; K is bounded by MAX_VERIFY_K so this fits.
extern "C" __global__ void q8_gemm_batched(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
    extern __shared__ float terms[]; // warps_per_block * k_tokens * blocks_per_row
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * k_tokens * blocks_per_row; // [token][block]
    if (row < rows) {
        long total_blocks = (long)rows * blocks_per_row;
        const signed char* quants = reinterpret_cast<const signed char*>(weight_bytes);
        const float* scales =
            reinterpret_cast<const float*>(weight_bytes + total_blocks * 32);
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            float w_scale = scales[row_block0 + b];
            const int4* wq = reinterpret_cast<const int4*>(quants + (row_block0 + b) * 32);
            int4 w0 = wq[0], w1 = wq[1]; // weight block read once, reused for all K
            for (int t = 0; t < k_tokens; t++) {
                const int4* iq = reinterpret_cast<const int4*>(
                    input_quants + ((long)t * blocks_per_row + b) * 32);
                int4 i0 = iq[0], i1 = iq[1];
                int s = 0;
                s = __dp4a(w0.x, i0.x, s);
                s = __dp4a(w0.y, i0.y, s);
                s = __dp4a(w0.z, i0.z, s);
                s = __dp4a(w0.w, i0.w, s);
                s = __dp4a(w1.x, i1.x, s);
                s = __dp4a(w1.y, i1.y, s);
                s = __dp4a(w1.z, i1.z, s);
                s = __dp4a(w1.w, i1.w, s);
                myterms[t * blocks_per_row + b] =
                    (float)s * w_scale * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) acc += myterms[t * blocks_per_row + b];
            output[(long)t * rows + row] = acc;
        }
    }
}

// ---- RoPE: supports adjacent-even-odd (pairing=0) and split-half/NEOX (pairing=1).
// cos/sin are per-pair (rope_dim/2). ---
extern "C" __global__ void rope_rotate(
    float* __restrict__ vec, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    int n_heads, int head_dim, int rope_dim, int pairing
) {
    int pairs = rope_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * pairs) return;
    int head = idx / pairs;
    int pair = idx % pairs;
    float c = cos_t[pair], s = sin_t[pair];
    float* h = vec + (long)head * head_dim;
    int d0, d1;
    if (pairing == 0) {
        d0 = 2 * pair; d1 = d0 + 1;
    } else {
        d0 = pair; d1 = pair + pairs;
    }
    float x0 = h[d0], x1 = h[d1];
    h[d0] = x0 * c - x1 * s;
    h[d1] = x0 * s + x1 * c;
}

// ---- KV scatter: write current position's K (or V) with f16 round-trip -----
// cache layout [kv_head][position][head_dim].
extern "C" __global__ void kv_scatter(
    const float* __restrict__ src, unsigned short* __restrict__ cache,
    const int* __restrict__ position_ptr, int n_kv_heads, int head_dim, int max_pos
) {
    int position = position_ptr[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_kv_heads * head_dim) return;
    int kv_head = idx / head_dim;
    int d = idx % head_dim;
    // KV stored as f16 bits (half the VRAM). The value is f16-rounded either way, so this is
    // bit-identical to storing f16_round(src) in f32 — the attention kernels read it back via
    // f16_bits_to_f32 and feed the same f32 into the dot product.
    cache[((long)kv_head * max_pos + position) * head_dim + d] =
        f32_to_f16_bits(src[(long)kv_head * head_dim + d]);
}

// ---- Attention decode: per query head, GQA, scale, softmax, weighted V -----
// One block per query head. cache_k/v layout [kv_head][position][head_dim].
extern "C" __global__ void attention_decode(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale
) {
    // position_count = current position + 1 (keys [0..=position] including this token).
    int position_count = position_ptr[0] + 1;
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    // KV is stored as f16 bits; read back to f32 (exact for the f16-rounded values written).
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int G = blockDim.x / head_dim;       // weighted-V groups per dim (blockDim is a multiple of head_dim)
    float* qsh = shared;                 // head_dim
    float* vpart = shared + head_dim;    // G * head_dim (per-dim partials, fixed-order combine)
    float* scores = shared + head_dim + (long)G * head_dim;   // position_count
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    // scores
    for (int p = tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        scores[p] = dot * scale;
    }
    __syncthreads();

    // max (single-thread reduce — position_count is modest; keep it simple/correct)
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = scores[0];
        for (int p = 1; p < position_count; p++) if (scores[p] > m) m = scores[p];
        s_max = m;
    }
    __syncthreads();
    // exp + sum
    for (int p = tid; p < position_count; p += blockDim.x) scores[p] = expf(scores[p] - s_max);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = 0; p < position_count; p++) sum += scores[p];
        s_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;
    // weighted V (parallelized; TOKEN-PARITY, *not* bit-identical to CPU): G threads
    // cooperate per output dim. Thread (gid,did) sums the CONTIGUOUS key range
    // [gid*pc/G, (gid+1)*pc/G) in p-order into vpart[did*G+gid]; group gid==0 then sums
    // the G partials in g-order into out[did]. Same math as the sequential p=0..pc-1
    // sum but FP-REASSOCIATED (each partial restarts at 0), so logits differ in the low
    // bits. This is the lever that fixes the O(context) weighted-V collapse at depth
    // (the sequential reduction caps parallelism at head_dim threads). Greedy tokens
    // are verified identical (parity gate first_divergent==-1 vs llama.cpp acd79d603).
    // CAVEAT: attention_batched (spec-decode verify) is still the sequential reduction;
    // for spec-decode losslessness it must get the identical reorder so decode==batched
    // stays exact. Greedy single-token decode (this path / the benchmark) is unaffected.
    int gid = tid / head_dim;            // 0..G-1
    int did = tid % head_dim;            // 0..head_dim-1
    int p_lo = (int)((long)gid * position_count / G);
    int p_hi = (int)((long)(gid + 1) * position_count / G);
    float acc = 0.0f;
    for (int p = p_lo; p < p_hi; p++)
        acc += (scores[p] * inv) * f16_bits_to_f32(vbase[(long)p * head_dim + did]);
    vpart[(long)did * G + gid] = acc;
    __syncthreads();
    if (gid == 0) {
        float sum = 0.0f;
        for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
        out[(long)head * head_dim + did] = sum;
    }
}

// ---- SwiGLU: out[i] = silu(gate[i]) * up[i], silu(x)=x/(1+exp(-x)) ---------
extern "C" __global__ void silu_mul(
    const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ---- Residual add: acc[i] += add[i] ---------------------------------------
extern "C" __global__ void residual_add(float* __restrict__ acc, const float* __restrict__ add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += add[i];
}

// ---- Greedy argmax (strict >, first index wins ties) ----------------------
// Single block. Each thread scans a stride; reduce in shared keeping lower idx
// on ties to match the CPU `>` scan.
extern "C" __global__ void argmax_f32(
    const float* __restrict__ logits, int n, unsigned int* __restrict__ out_idx
) {
    extern __shared__ float sh[];
    float* sval = sh;                                  // blockDim
    int* sidx = (int*)(sh + blockDim.x);               // blockDim
    int tid = threadIdx.x;
    float best = -3.4e38f; int besti = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; besti = i; }
    }
    sval[tid] = best; sidx[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s]; int oi = sidx[tid + s];
            // strict >: take the other only if strictly greater, else keep lower index
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov; sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = (unsigned int)sidx[0];
}

// ---- Temperature sampling via the Gumbel-max trick --------------------------
// A draw from softmax(logits/temp) equals argmax_i(logits[i]/temp + g_i) with
// g_i ~ Gumbel(0,1) = -log(-log(u_i)), u_i ~ Uniform(0,1). One pass over the
// vocab (same shape as argmax) — no softmax, no sort, no host logits copy. The
// per-element uniform is a stateless splitmix64 hash of (seed, index), so the
// whole draw is reproducible from `seed` (varied per token by the host). As
// temp -> 0, inv_temp -> inf and the (bounded) Gumbel term is dominated by the
// logits, so this collapses to the exact greedy argmax — matching the greedy
// gate. Strict-greater tie-break to the lower index, as in argmax_f32.
__device__ __forceinline__ float splitmix_uniform(unsigned long long seed, unsigned int idx) {
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(idx + 1u);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    unsigned int m = (unsigned int)(z >> 40); // 24 random bits
    return ((float)m + 0.5f) / 16777216.0f;   // in (0,1), excludes 0 and 1
}
extern "C" __global__ void sample_gumbel(
    const float* __restrict__ logits, int n, float inv_temp,
    unsigned long long seed, unsigned int* __restrict__ out_idx
) {
    extern __shared__ float sh[];
    float* sval = sh;
    int* sidx = (int*)(sh + blockDim.x);
    int tid = threadIdx.x;
    float best = -3.4e38f; int besti = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float u = splitmix_uniform(seed, (unsigned int)i);
        float g = -logf(-logf(u));
        float v = logits[i] * inv_temp + g;
        if (v > best) { best = v; besti = i; }
    }
    sval[tid] = best; sidx[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s]; int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov; sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = (unsigned int)sidx[0];
}

// ---- Batched (K-token) variants for the speculative-verify forward ----------
// Each processes K tokens laid out [token][...] in one launch. Elementwise
// kernels (quantize/silu/residual) are batched just by launching over K x the
// work, so only the per-token ops below need dedicated variants.

// One block per token; staged-shared serial sum (matches rms_norm_f32 order).
extern "C" __global__ void rms_norm_batched(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, int n, float eps, int k_tokens
) {
    int t = blockIdx.x;
    if (t >= k_tokens) return;
    const float* xt = x + (long)t * n;
    float* outt = out + (long)t * n;
    extern __shared__ float xs[];
    __shared__ float s_scale;
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) xs[i] = xt[i];
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += xs[i] * xs[i];
        s_scale = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    float sc = s_scale;
    for (int i = tid; i < n; i += blockDim.x) outt[i] = xs[i] * sc * weight[i];
}

// RoPE for K tokens; cos/sin are per-token tables [token][rope_dim/2].
extern "C" __global__ void rope_batched(
    float* __restrict__ vec, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    int n_heads, int head_dim, int rope_dim, int per_token_dim, int half, int k_tokens, int pairing
) {
    int pairs = rope_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_tokens * n_heads * pairs;
    if (idx >= total) return;
    int t = idx / (n_heads * pairs);
    int rem = idx % (n_heads * pairs);
    int head = rem / pairs;
    int pair = rem % pairs;
    float c = cos_t[(long)t * half + pair], s = sin_t[(long)t * half + pair];
    float* h = vec + (long)t * per_token_dim + (long)head * head_dim;
    int d0, d1;
    if (pairing == 0) {
        d0 = 2 * pair; d1 = d0 + 1;
    } else {
        d0 = pair; d1 = pair + pairs;
    }
    float x0 = h[d0], x1 = h[d1];
    h[d0] = x0 * c - x1 * s;
    h[d1] = x0 * s + x1 * c;
}

// Scatter K tokens' K/V into the cache at consecutive positions base..base+K-1.
extern "C" __global__ void kv_scatter_batched(
    const float* __restrict__ src, unsigned short* __restrict__ cache, int base_position,
    int n_kv_heads, int head_dim, int max_pos, int per_token_dim, int k_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_tokens * n_kv_heads * head_dim;
    if (idx >= total) return;
    int t = idx / (n_kv_heads * head_dim);
    int rem = idx % (n_kv_heads * head_dim);
    int kv_head = rem / head_dim;
    int d = rem % head_dim;
    int position = base_position + t;
    // f16-bit KV store (see kv_scatter): bit-identical to f16_round into f32.
    cache[((long)kv_head * max_pos + position) * head_dim + d] =
        f32_to_f16_bits(src[(long)t * per_token_dim + (long)kv_head * head_dim + d]);
}

// Causal attention for K tokens: token t (at position base+t) attends [0, base+t].
// One block per (token, query head). Shared sized for the longest prefix (base+K).
extern "C" __global__ void attention_batched(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens
) {
    int t = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int position_count = base_position + t + 1;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    // f16-bit KV (see attention_decode): read back to f32 for the dot product.
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ float shared[];
    float* qsh = shared;               // head_dim
    float* scores = shared + head_dim; // position_count
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();
    for (int p = tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        scores[p] = dot * scale;
    }
    __syncthreads();
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = scores[0];
        for (int p = 1; p < position_count; p++) if (scores[p] > m) m = scores[p];
        s_max = m;
    }
    __syncthreads();
    for (int p = tid; p < position_count; p += blockDim.x) scores[p] = expf(scores[p] - s_max);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = 0; p < position_count; p++) sum += scores[p];
        s_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < position_count; p++)
            acc += (scores[p] * inv) * f16_bits_to_f32(vbase[(long)p * head_dim + d]);
        out[(long)t * q_per_token + (long)head * head_dim + d] = acc;
    }
}

// Argmax of each of K logit rows (one block per token). Strict-greater, lowest
// index — the greedy choice used to verify drafts.
extern "C" __global__ void argmax_batched(
    const float* __restrict__ logits, int n, int k_tokens, unsigned int* __restrict__ out
) {
    int t = blockIdx.x;
    if (t >= k_tokens) return;
    const float* lt = logits + (long)t * n;
    extern __shared__ float sh[];
    float* sval = sh;
    int* sidx = (int*)(sh + blockDim.x);
    int tid = threadIdx.x;
    float best = -3.4e38f; int besti = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (lt[i] > best) { best = lt[i]; besti = i; }
    }
    sval[tid] = best; sidx[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s]; int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov; sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[t] = (unsigned int)sidx[0];
}

// ---- Split-K decode attention (fills SMs at depth) --------------------------
// One block per (head, split) instead of one block per head, so grid = n_heads x
// n_splits covers all 30 SMs even though there are only 32 heads. TOKEN-PARITY, not
// bit-identical: the per-position dot and exp use the EXACT sequential order and the
// EXACT global max (so those are bit-identical), but the exp-sum and weighted-V are
// split into contiguous chunks and recombined in chunk order — re-associating the
// position sum exactly as the (parity-passing) Stage-2 weighted-V split does. True
// bit-identity is impossible for a split sequential reduction. Verified token-identical.
//
// Pass 1: per (head, split) compute the chunk's scores (sequential d-order dot ->
// bit-identical) into scores_buf and the chunk's max into chunkmax_buf.
extern "C" __global__ void attn_sk_scores(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale, int n_splits
) {
    int position_count = position_ptr[0] + 1;
    int head = blockIdx.x;
    int sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);

    extern __shared__ float qsh[];      // head_dim
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    float local_max = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        float sc = dot * scale;
        scores_buf[(long)head * max_pos + p] = sc;
        local_max = fmaxf(local_max, sc);
    }
    __shared__ float red[1024];
    red[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) chunkmax_buf[(long)head * n_splits + sp] = red[0];
}

// Pass 2: per (head, split) read the EXACT global max over all splits, exp the chunk in
// place (per-position, no reassociation), then write the chunk's sequential exp-sum
// (lsum_buf) and UNNORMALIZED weighted-V (acc_buf, sequential p per dim).
extern "C" __global__ void attn_sk_partial(
    const unsigned short* __restrict__ cache_v, float* __restrict__ scores_buf,
    const float* __restrict__ chunkmax_buf, float* __restrict__ lsum_buf,
    float* __restrict__ acc_buf, int n_heads, int n_kv_heads, int head_dim,
    const int* __restrict__ position_ptr, int max_pos, int n_splits
) {
    int position_count = position_ptr[0] + 1;
    int head = blockIdx.x;
    int sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    float* sc_head = scores_buf + (long)head * max_pos;
    int tid = threadIdx.x;

    float gmax = -3.4e38f;
    for (int i = 0; i < n_splits; i++) gmax = fmaxf(gmax, chunkmax_buf[(long)head * n_splits + i]);

    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) sc_head[p] = expf(sc_head[p] - gmax);
    __syncthreads();
    if (tid == 0) {
        float ls = 0.0f;
        for (int p = p_lo; p < p_hi; p++) ls += sc_head[p];
        lsum_buf[(long)head * n_splits + sp] = ls;
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float a = 0.0f;
        for (int p = p_lo; p < p_hi; p++) a += sc_head[p] * f16_bits_to_f32(vbase[(long)p * head_dim + d]);
        acc_buf[(((long)head * n_splits + sp) * head_dim) + d] = a;
    }
}

// Pass 3: per head, combine the splits in order: s = sum_sp lsum (ordered) and
// out[d] = (sum_sp acc[sp][d]) / s (ordered). Chunk order == position order.
extern "C" __global__ void attn_sk_combine(
    const float* __restrict__ lsum_buf, const float* __restrict__ acc_buf,
    float* __restrict__ out, int n_heads, int head_dim, int n_splits
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;
    __shared__ float s_inv;
    if (tid == 0) {
        float s = 0.0f;
        for (int sp = 0; sp < n_splits; sp++) s += lsum_buf[(long)head * n_splits + sp];
        s_inv = 1.0f / s;
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float a = 0.0f;
        for (int sp = 0; sp < n_splits; sp++) a += acc_buf[(((long)head * n_splits + sp) * head_dim) + d];
        out[(long)head * head_dim + d] = a * s_inv;
    }
}
"#;

/// Compiled kernel set + a CUDA context/stream, used to run resident-decode
/// kernels. (The full per-token `forward_token` orchestration is assembled on
/// top of these once every kernel passes its parity check.)
// Some kernel handles are only exercised by the per-kernel parity tests until
// `forward_token` (next step) drives the whole sequence.
#[allow(dead_code)]
pub struct CudaResidentKernels {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) rms_norm_per_head: CudaFunction,
    pub(crate) quantize: CudaFunction,
    pub(crate) rms_norm_quantize: CudaFunction,
    pub(crate) gemv: CudaFunction,
    pub(crate) rope: CudaFunction,
    pub(crate) kv_scatter: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) silu_mul: CudaFunction,
    pub(crate) silu_mul_quantize: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    pub(crate) argmax: CudaFunction,
    pub(crate) sample_gumbel: CudaFunction,
    pub(crate) gemm_batched: CudaFunction,
    pub(crate) rms_norm_batched: CudaFunction,
    pub(crate) rope_batched: CudaFunction,
    pub(crate) kv_scatter_batched: CudaFunction,
    pub(crate) attention_batched: CudaFunction,
    pub(crate) argmax_batched: CudaFunction,
    pub(crate) attn_sk_scores: CudaFunction,
    pub(crate) attn_sk_partial: CudaFunction,
    pub(crate) attn_sk_combine: CudaFunction,
}

impl CudaResidentKernels {
    pub fn new() -> Result<Self, String> {
        let ordinal = crate::cuda::selected_device_ordinal();
        let ctx =
            CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            fmad: Some(false),
            arch: Some("compute_61"),
            ..Default::default()
        };
        let ptx: Ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNELS, opts)
            .map_err(|e| format!("nvrtc: {e}"))?;
        let m = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module: {e}"))?;
        let f = |name: &str| {
            m.load_function(name)
                .map_err(|e| format!("load {name}: {e}"))
        };
        Ok(Self {
            rms_norm: f("rms_norm_f32")?,
            rms_norm_per_head: f("rms_norm_per_head_f32")?,
            quantize: f("quantize_q8_0")?,
            rms_norm_quantize: f("rms_norm_quantize")?,
            gemv: f("q8_gemv")?,
            rope: f("rope_rotate")?,
            kv_scatter: f("kv_scatter")?,
            attention: f("attention_decode")?,
            silu_mul: f("silu_mul")?,
            silu_mul_quantize: f("silu_mul_quantize")?,
            residual_add: f("residual_add")?,
            argmax: f("argmax_f32")?,
            sample_gumbel: f("sample_gumbel")?,
            gemm_batched: f("q8_gemm_batched")?,
            rms_norm_batched: f("rms_norm_batched")?,
            rope_batched: f("rope_batched")?,
            kv_scatter_batched: f("kv_scatter_batched")?,
            attention_batched: f("attention_batched")?,
            argmax_batched: f("argmax_batched")?,
            attn_sk_scores: f("attn_sk_scores")?,
            attn_sk_partial: f("attn_sk_partial")?,
            attn_sk_combine: f("attn_sk_combine")?,
            ctx,
            stream,
        })
    }
}

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

// ---- Free launch helpers (take explicit refs so callers can pass disjoint
// fields of the resident state without the `&self` whole-struct borrow). ----

#[allow(clippy::too_many_arguments)]
/// Repack Q8_0 weight bytes from the on-disk AoS block layout (interleaved
/// 36-byte blocks: f32 scale + 32 i8) into the GPU SoA layout the resident
/// `q8_gemv` reads: all quants first (`n_blocks * 32` i8), then all scales
/// (`n_blocks` f32). Quants-first keeps every block's 32 i8 16-byte aligned so
/// the kernel can issue `int4` loads. Done once per weight at upload; the values
/// are unchanged, only their arrangement.
fn repack_q8_soa(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 36;
    let mut out = vec![0u8; n * 32 + n * 4];
    let (quants, scales) = out.split_at_mut(n * 32);
    for b in 0..n {
        let blk = &bytes[b * 36..b * 36 + 36];
        scales[b * 4..b * 4 + 4].copy_from_slice(&blk[0..4]);
        quants[b * 32..b * 32 + 32].copy_from_slice(&blk[4..36]);
    }
    out
}

fn launch_rmsnorm(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        // Stage the whole n-element row in shared memory for the in-order sum.
        shared_mem_bytes: (n as u32) * 4,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(w).arg(out).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_quantize(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_blocks: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb = n_blocks as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(quants).arg(scales).arg(&nb);
    unsafe { b.launch(cfg) }.map(|_| ())
}

// Fused RMS-norm + Q8_0 quantize (F1): one block stages the `n`-element row in shared
// for the in-order sum (same as rms_norm), then quantizes from shared — replacing a
// launch_rmsnorm + launch_quantize pair and the f32 `normed` round-trip.
#[allow(clippy::too_many_arguments)]
fn launch_rmsnorm_quantize(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (n as u32) * 4, // stage the whole row for the in-order sum
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(w).arg(quants).arg(scales).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    // 8 warps/block, one warp per output row. Shared holds the staged input
    // vector (quants `bpr*32` + scales `bpr*4`) shared by all warps, then each
    // warp's per-block float terms for the in-order lane-0 reduction. (A block-size
    // sweep — 64/128/256/512 — left decode tok/s flat within noise: the batch-1 GEMV
    // is memory-latency-bound and the decode CUDA graph already cuts launch overhead,
    // so occupancy is not the limiter. Kept at the profiled default.)
    let block = 256u32;
    let warps_per_block = block / 32;
    let bpr_u = blocks_per_row as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: bpr_u * 36 + warps_per_block * bpr_u * 4,
    };
    let (r, bpr) = (rows as i32, blocks_per_row as i32);
    let residual = 0i32;
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q8 GEMV that fuses the post-projection residual add: writes `out[row] += acc` instead of
/// `= acc`, so `out` must be the residual (hidden) buffer. Saves a separate residual_add launch
/// and the projection's f32 round-trip. Bit-identical to gemv-then-residual_add (F2).
#[allow(clippy::too_many_arguments)]
fn launch_gemv_residual(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let bpr_u = blocks_per_row as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: bpr_u * 36 + warps_per_block * bpr_u * 4,
    };
    let (r, bpr) = (rows as i32, blocks_per_row as i32);
    let residual = 1i32;
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

// Fused SiLU*up + Q8_0 quantize (F3): one thread per 32-block, no shared memory.
fn launch_silu_mul_quantize(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_blocks: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb = n_blocks as i32;
    let mut b = s.launch_builder(f);
    b.arg(gate).arg(up).arg(quants).arg(scales).arg(&nb);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Batched Q8 GEMM: `k_tokens` inputs (`[token][block]`) against `rows` weight
/// rows, output `[token][row]`. Weights are read once and reused across tokens.
// Driven by the batched speculative-verify forward (next stage); the parity test
// exercises it today.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn launch_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    blocks_per_row: usize,
    k_tokens: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    // Each warp computes one output row; warps_per_block only sets how many rows a
    // block handles, so it never changes the per-row block-order reduction (the
    // result is bit-identical for any warps_per_block). Cap it so the
    // [warp][token][block] ordered-sum scratch fits the 48 KiB default shared-mem
    // limit — necessary once K grows (e.g. K=8, blocks_per_row=256 needs 6 warps,
    // not the historic 8). Use a 46 KiB budget for headroom. The K=4 / small-row
    // cases keep the full 8 warps/block (unchanged from before).
    const SHARED_BUDGET: u32 = 46 * 1024;
    let per_warp_bytes = (k_tokens as u32) * (blocks_per_row as u32) * 4;
    let warps_per_block = (SHARED_BUDGET / per_warp_bytes.max(1)).clamp(1, 8);
    let block = warps_per_block * 32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // [warp][token][block] ordered-sum scratch.
        shared_mem_bytes: warps_per_block * per_warp_bytes,
    };
    let (r, bpr, kt) = (rows as i32, blocks_per_row as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(&kt)
        .arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_rms_norm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (k as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (n as u32) * 4,
    };
    let (n_i, k_i) = (n as i32, k as i32);
    let mut b = s.launch_builder(f);
    b.arg(x).arg(w).arg(out).arg(&n_i).arg(&eps).arg(&k_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_rope_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    vec: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    per_token_dim: usize,
    k: usize,
    pairing: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let half = rope_dim / 2;
    let total = (k * n_heads * half) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, rd, ptd, hf, ki) = (
        n_heads as i32,
        head_dim as i32,
        rope_dim as i32,
        per_token_dim as i32,
        half as i32,
        k as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(vec)
        .arg(cos)
        .arg(sin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd)
        .arg(&ptd)
        .arg(&hf)
        .arg(&ki)
        .arg(&pairing);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_scatter_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u16>,
    base_position: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
    per_token_dim: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (k * n_kv_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bp, nkv, hd, mp, ptd, ki) = (
        base_position as i32,
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
        per_token_dim as i32,
        k as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(src)
        .arg(cache)
        .arg(&bp)
        .arg(&nkv)
        .arg(&hd)
        .arg(&mp)
        .arg(&ptd)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u16>,
    cache_v: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_position: usize,
    max_pos: usize,
    scale: f32,
    q_per_token: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    // Shared = query (head_dim) + scores (longest prefix = base + k).
    let shared = ((head_dim + base_position + k) as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((k * n_heads) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: shared,
    };
    let (nh, nkv, hd, bp, mp, qpt, ki) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        base_position as i32,
        max_pos as i32,
        q_per_token as i32,
        k as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(q)
        .arg(cache_k)
        .arg(cache_v)
        .arg(out)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&bp)
        .arg(&mp)
        .arg(&scale)
        .arg(&qpt)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_argmax_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    logits: &CudaSlice<f32>,
    n: usize,
    k: usize,
    out: &mut CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (k as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8,
    };
    let (n_i, k_i) = (n as i32, k as i32);
    let mut b = s.launch_builder(f);
    b.arg(logits).arg(&n_i).arg(&k_i).arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_rope(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    vec: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pairing: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (n_heads * (rope_dim / 2)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, rd) = (n_heads as i32, head_dim as i32, rope_dim as i32);
    let mut b = s.launch_builder(f);
    b.arg(vec)
        .arg(cos)
        .arg(sin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd)
        .arg(&pairing);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_rms_norm_per_head(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    buf: &mut CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (head_count as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (head_dim as u32) * 4,
    };
    let (hd, uw) = (head_dim as i32, 1i32);
    let mut b = s.launch_builder(f);
    b.arg(buf).arg(weight).arg(&hd).arg(&eps).arg(&uw);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_scatter(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u16>,
    position: &CudaSlice<i32>,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (n_kv_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nkv, hd, mp) = (n_kv_heads as i32, head_dim as i32, max_pos as i32);
    let mut b = s.launch_builder(f);
    b.arg(src)
        .arg(cache)
        .arg(position)
        .arg(&nkv)
        .arg(&hd)
        .arg(&mp);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
// Max splits the split-K decode attention may use (scratch in CudaResidentDecode is
// sized to this), and the context length above which it is used. Below the threshold the
// one-block-per-head `launch_attention` is cheaper (one launch, no scratch round-trip);
// above it, split-K's n_heads x n_splits grid is needed to fill the SMs.
const SPLITK_MAX: usize = 16;
const SPLITK_THRESHOLD: usize = 512;

/// Split-K decode attention: grid = n_heads x n_splits (vs one block per head), so the
/// 30 SMs fill even with 32 heads. Three passes: (1) chunk scores + chunk max, (2) exp
/// with the EXACT global max + chunk exp-sum + chunk unnormalized weighted-V, (3) ordered
/// combine. TOKEN-PARITY: dot and global max are bit-identical; the cross-split sum
/// re-associates exactly as the (parity-passing) Stage-2 weighted-V split. Verified.
#[allow(clippy::too_many_arguments)]
fn launch_attention_splitk(
    s: &Arc<CudaStream>,
    k: &CudaResidentKernels,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u16>,
    cache_v: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    scores_buf: &mut CudaSlice<f32>,
    chunkmax_buf: &mut CudaSlice<f32>,
    lsum_buf: &mut CudaSlice<f32>,
    acc_buf: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: &CudaSlice<i32>,
    position_count: usize,
    max_pos: usize,
    scale: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let n_splits = position_count.div_ceil(256).clamp(2, SPLITK_MAX);
    let (nh, nkv, hd, mp, ns) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
        n_splits as i32,
    );
    let block: u32 = 256;
    // Pass 1: scores + per-chunk max. shared = qsh[head_dim].
    {
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_splits as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (head_dim as u32) * 4,
        };
        let mut b = s.launch_builder(&k.attn_sk_scores);
        b.arg(q)
            .arg(cache_k)
            .arg(&mut *scores_buf)
            .arg(&mut *chunkmax_buf)
            .arg(&nh)
            .arg(&nkv)
            .arg(&hd)
            .arg(position)
            .arg(&mp)
            .arg(&scale)
            .arg(&ns);
        unsafe { b.launch(cfg) }?;
    }
    // Pass 2: exp(global max) + chunk exp-sum + chunk unnormalized weighted-V.
    {
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_splits as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&k.attn_sk_partial);
        b.arg(cache_v)
            .arg(&mut *scores_buf)
            .arg(&mut *chunkmax_buf)
            .arg(&mut *lsum_buf)
            .arg(&mut *acc_buf)
            .arg(&nh)
            .arg(&nkv)
            .arg(&hd)
            .arg(position)
            .arg(&mp)
            .arg(&ns);
        unsafe { b.launch(cfg) }?;
    }
    // Pass 3: ordered combine -> out. One block per head, head_dim threads.
    {
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&k.attn_sk_combine);
        b.arg(&mut *lsum_buf).arg(&mut *acc_buf).arg(out).arg(&nh).arg(&hd).arg(&ns);
        unsafe { b.launch(cfg) }?;
    }
    Ok(())
}

fn launch_attention(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u16>,
    cache_v: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: &CudaSlice<i32>,
    // Positions to size the shared `scores[]` array for. The non-graph path passes
    // the exact current count (tight, best occupancy); the graph-capture path passes
    // `max_pos` so the captured launch config holds for every replayed position.
    shared_positions: usize,
    max_pos: usize,
    scale: f32,
) -> Result<(), cudarc::driver::DriverError> {
    // Adaptive launch (occupancy/latency fix). attention_decode was starved at
    // batch-1 (ncu @ block 64: 4.4% occupancy, 0.07 waves/SM, 0.44% DRAM) — too few
    // warps to hide the K/V f16 read latency, and its cost is O(context) so decode
    // collapses at depth. Size the block to the key count in units of head_dim (G
    // weighted-V groups), capped at 1024 threads. G = block/head_dim is passed
    // implicitly via blockDim so the kernel parallelizes the weighted-V across G
    // contiguous key ranges. The strided score/exp loops and the tid==0 softmax
    // reductions stay bit-identical; the weighted-V is FP-reassociated for parallelism
    // (token-parity, not bit-identical to CPU — see the kernel body). Verified token-id.
    let max_groups = (1024 / head_dim as u32).max(1);
    let groups = (shared_positions.max(1) as u32)
        .div_ceil(head_dim as u32)
        .clamp(1, max_groups);
    let block = groups * head_dim as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block, 1, 1),
        // qsh[head_dim] + vpart[groups*head_dim] + scores[shared_positions]
        shared_mem_bytes: ((head_dim as u32 * (1 + groups)) + shared_positions as u32) * 4,
    };
    let (nh, nkv, hd, mp) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(q)
        .arg(cache_k)
        .arg(cache_v)
        .arg(out)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(position)
        .arg(&mp)
        .arg(&scale);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_silu_mul(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(gate).arg(up).arg(out).arg(&n_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_residual(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    acc: &mut CudaSlice<f32>,
    add: &CudaSlice<f32>,
    n: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(acc).arg(add).arg(&n_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_argmax(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    logits: &CudaSlice<f32>,
    n: usize,
    out_idx: &mut CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(logits).arg(&n_i).arg(out_idx);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_sample_gumbel(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    logits: &CudaSlice<f32>,
    n: usize,
    inv_temp: f32,
    seed: u64,
    out_idx: &mut CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(logits)
        .arg(&n_i)
        .arg(&inv_temp)
        .arg(&seed)
        .arg(out_idx);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// One layer's GPU-resident Q8_0 weights + norm vectors.
/// One layer's projection weight: resident in VRAM, or offloaded to host RAM and
/// streamed into the shared scratch buffer before the layer computes. The bytes
/// are the repacked Q8_0 SoA layout in both cases — where they live never changes
/// the math (offloading is a capacity feature, parity is unaffected).
/// Page-locked (pinned) host memory allocated with DEFAULT (cacheable) flags rather
/// than write-combined. `CudaContext::alloc_pinned` hardcodes WRITE_COMBINED, but on
/// this platform's PCIe link cacheable pinned memory reads ~18% FASTER for host->device
/// DMA (measured 9.4 vs 7.9 GB/s back-to-back). Offloaded weights stream H2D every
/// forward, so that 18% is a direct decode-throughput win. The driver auto-detects the
/// pinned pointer, so a plain `&[u8]` view drives the fast async DMA path.
struct CacheablePinned {
    ptr: *mut u8,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: `ptr` is a pinned host allocation owned solely by this struct (freed on drop).
// The resident engine is only ever accessed under the process-global resident-cache
// mutex — the same discipline that lets its `CudaGraph` be `Send` — so the pointer is
// never touched from two threads at once.
unsafe impl Send for CacheablePinned {}

impl CacheablePinned {
    fn from_bytes(ctx: &Arc<CudaContext>, bytes: &[u8]) -> Result<Self, String> {
        use cudarc::driver::result;
        ctx.bind_to_thread().map_err(|e| format!("bind: {e}"))?;
        // flags = 0 → cacheable (NOT write-combined). max(1) avoids a zero-size alloc.
        let ptr = unsafe { result::malloc_host(bytes.len().max(1), 0) }
            .map_err(|e| format!("malloc_host: {e}"))? as *mut u8;
        assert!(!ptr.is_null());
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        Ok(Self {
            ptr,
            len: bytes.len(),
            ctx: ctx.clone(),
        })
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for CacheablePinned {
    fn drop(&mut self) {
        use cudarc::driver::result;
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = result::free_host(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// The seven projection weights of one offloaded layer, packed CONTIGUOUSLY in one
/// pinned host buffer so the per-forward host->device stream is a SINGLE transfer.
/// Splitting it into seven separate `memcpy_htod` calls (one per projection) added a
/// little DMA ramp-up per sub-transfer; one contiguous copy is marginally faster and
/// simpler. `off[i]..off[i+1]` is projection i's byte range (order q,k,v,o,gate,up,
/// down); `off[7]` is the total.
struct OffloadedLayer {
    host: CacheablePinned,
    off: [usize; 8],
}

/// Multi-buffered offload streaming state. The weights of the next `N-1` offloaded
/// layers are prefetched into idle scratch buffers on `copy_stream` while the compute
/// stream runs the current layer, so the PCIe transfers overlap useful work and the
/// copy stream stays saturated near the link's peak (a single look-ahead buffer left
/// the link idle in the bubbles between transfers). `N` = `scratch.len()`
/// (`CAMELID_OFFLOAD_BUFFERS`, default 4). Each `scratch[b]` is ONE contiguous buffer
/// sized to the largest layer's total weight bytes; a layer's seven projections are
/// sub-views (`scratch[b].slice(off[i]..off[i+1])`) into it.
struct OffloadState {
    scratch: Vec<CudaSlice<u8>>,
    copy_stream: std::sync::Arc<CudaStream>,
    /// `copy_done[b]`: prefetch into buffer b finished — the compute stream waits on
    /// it before reading buffer b. `compute_done[b]`: the compute that last read
    /// buffer b finished — the copy stream waits on it before overwriting buffer b
    /// (write-after-read). A fresh event reads as already-occurred, so the first use
    /// of each buffer doesn't block. Both indexed by buffer (length = `scratch.len()`).
    copy_done: Vec<CudaEvent>,
    compute_done: Vec<CudaEvent>,
}

struct ResidentLayer {
    // Resident VRAM projections. For an OFFLOADED layer (`offloaded.is_some()`) these
    // are 1-byte placeholders that are never read — the real bytes live in `offloaded`
    // and stream into scratch each forward.
    q: CudaSlice<u8>,
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    o: CudaSlice<u8>,
    gate: CudaSlice<u8>,
    up: CudaSlice<u8>,
    down: CudaSlice<u8>,
    attn_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    offloaded: Option<OffloadedLayer>,
}

/// A captured CUDA graph, wrapped to be `Send`. cudarc does not mark `CudaGraph`
/// Send because graphs are not internally synchronized; the resident engine is only
/// ever accessed under the process-global resident-cache `Mutex`, which serializes
/// all use, so moving the graph across threads with the engine is sound (the same
/// justification the rest of the engine's cudarc handles rely on). Every `launch`
/// binds the context to the calling thread first.
struct SendCudaGraph(CudaGraph);
// SAFETY: see the type doc — all access is serialized behind the resident-cache Mutex.
unsafe impl Send for SendCudaGraph {}

/// GPU-resident Llama decode engine. Weights and KV cache live on the GPU; one
/// `forward_token` call runs the whole per-token forward with a single sync.
pub struct CudaResidentDecode {
    k: CudaResidentKernels,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    ffn_dim: usize,
    rope_dim: usize,
    max_pos: usize,
    vocab: usize,
    eps: f32,
    q_width: usize,
    kv_width: usize,
    split_half_pairing: bool,
    layers: Vec<ResidentLayer>,
    final_norm: CudaSlice<f32>,
    output_weight: CudaSlice<u8>,
    // KV cache stored as f16 bits (u16) — half the VRAM of f32, bit-identical because the
    // stored values are f16-rounded either way (see the kv_scatter / attention kernels).
    cache_k: Vec<CudaSlice<u16>>,
    cache_v: Vec<CudaSlice<u16>>,
    /// Number of KV positions materialized on the GPU (so the driver knows
    /// whether the session needs (re)seeding from the CPU history).
    filled: usize,
    // per-token scratch (reused)
    d_hidden: CudaSlice<f32>,
    d_normed: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    // Split-K decode-attention scratch (sized for up to SPLITK_MAX splits). Used only
    // when the context is long enough (see SPLITK_THRESHOLD) to need more than the
    // one-block-per-head launch to fill the SMs.
    d_sk_scores: CudaSlice<f32>,   // n_heads * max_pos
    d_sk_chunkmax: CudaSlice<f32>, // n_heads * SPLITK_MAX
    d_sk_lsum: CudaSlice<f32>,     // n_heads * SPLITK_MAX
    d_sk_acc: CudaSlice<f32>,      // n_heads * SPLITK_MAX * head_dim
    d_proj: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_ffn_act: CudaSlice<f32>,
    d_in_scales: CudaSlice<f32>,
    d_in_quants: CudaSlice<i8>,
    d_logits: CudaSlice<f32>,
    d_sampled: CudaSlice<u32>,
    d_cos: CudaSlice<f32>,
    d_sin: CudaSlice<f32>,
    /// Current decode position, held on the device so `kv_scatter` / `attention`
    /// read it from memory rather than a launch-time scalar. This is what lets the
    /// per-token kernel chain be captured once into a CUDA graph and replayed: the
    /// graph's kernel args are frozen, so the only thing that varies per token
    /// (position, embedding, RoPE) must arrive through device buffers it reads.
    d_position: CudaSlice<i32>,
    /// Captured CUDA graph of the greedy decode forward (layer stack + output proj +
    /// argmax). Recorded once, then replayed per token with one `launch()` instead of
    /// ~600 individual kernel launches. The per-token inputs (embedding / RoPE /
    /// position) are written to device buffers BEFORE replay, so the frozen graph
    /// reads fresh values each step. Captured at the engine's `eps`/`scale`/`max_pos`.
    decode_graph: Option<SendCudaGraph>,
    /// Lazily-allocated K-batched scratch for the speculative-verify forward.
    verify_scratch: Option<VerifyScratch>,
    /// Shared GPU scratch for offloaded layers (None when every layer is resident).
    /// Allocated by `enable_offload_scratch` when the build decides to offload.
    offload: Option<OffloadState>,
}

/// Max tokens verified per speculative round. The batched GEMM keeps the ordered
/// per-(token,block) sum in shared memory (`k * blocks_per_row * warps_per_block *
/// 4` bytes). At K=8 the 3B FFN (blocks_per_row=256) would need 64 KiB at the
/// historic 8 warps/block, past the 48 KiB default shared-mem limit, so
/// `launch_gemm_batched` now caps warps/block to fit the budget (warps map to
/// output rows, so fewer-warps-per-block changes only the grid shape, never the
/// per-row block-order reduction — the result stays bit-identical). A larger K
/// lets each weight read verify more drafts per round, raising the ceiling on
/// repetitive/structured output where n-gram acceptance is high.
pub(crate) const MAX_VERIFY_K: usize = 8;

/// K-batched scratch buffers for `verify_batch`, sized `MAX_VERIFY_K * dim`.
struct VerifyScratch {
    vh: CudaSlice<f32>,
    vn: CudaSlice<f32>,
    viq: CudaSlice<i8>,
    vis: CudaSlice<f32>,
    vq: CudaSlice<f32>,
    vk: CudaSlice<f32>,
    vv: CudaSlice<f32>,
    vattn: CudaSlice<f32>,
    vproj: CudaSlice<f32>,
    vgate: CudaSlice<f32>,
    vup: CudaSlice<f32>,
    vact: CudaSlice<f32>,
    vlogits: CudaSlice<f32>,
    vsamp: CudaSlice<u32>,
    vcos: CudaSlice<f32>,
    vsin: CudaSlice<f32>,
}

/// Whether greedy decode replays a captured CUDA graph. **Default off**: measured on
/// an RTX 3060, single-token decode is GPU-bandwidth-bound (the dominant q8_gemv runs
/// at ~76% of peak DRAM), so the ~600 per-token kernel launches enqueue *ahead* of the
/// GPU and their host overhead is already hidden — replaying them as one graph saved
/// nothing and cost a small fixed overhead (3B 53.2→52.5, TinyLlama 129→124 tok/s),
/// at identical tokens. The path is kept (correct + parity-clean) because it pays off
/// where decode becomes launch-bound: a much faster GPU, or after kernel fusion cuts
/// GPU time below the launch cost. Opt in with `CAMELID_CUDA_GRAPHS=1`.
fn cuda_graphs_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_CUDA_GRAPHS").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// Whether the resident decode uses the fused kernels (rms-norm+quantize, etc.). Default ON:
/// the fused kernels are bit-identical to the unfused chain (validated by the cuda_resident
/// parity tests) and cut the per-token kernel count, which is the dominant cost for small
/// models (the speculative draft). Set `CAMELID_RESIDENT_NO_FUSION=1` to fall back to the
/// separate kernels (A/B comparison, debugging).
fn resident_fusion_enabled() -> bool {
    !matches!(
        std::env::var("CAMELID_RESIDENT_NO_FUSION").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

impl CudaResidentDecode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        ffn_dim: usize,
        rope_dim: usize,
        max_pos: usize,
        vocab: usize,
        eps: f32,
        split_half_pairing: bool,
    ) -> Result<Self, String> {
        let k = CudaResidentKernels::new()?;
        let s = &k.stream;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let max_in = hidden.max(ffn_dim).max(q_width); // widest quantize input
        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("alloc: {e}"));
        let cache_k = (0..n_layers)
            .map(|_| s.alloc_zeros::<u16>(kv_width * max_pos))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("kv alloc: {e}"))?;
        let cache_v = (0..n_layers)
            .map(|_| s.alloc_zeros::<u16>(kv_width * max_pos))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("kv alloc: {e}"))?;
        Ok(Self {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden,
            ffn_dim,
            rope_dim,
            max_pos,
            vocab,
            eps,
            q_width,
            kv_width,
            split_half_pairing,
            layers: Vec::with_capacity(n_layers),
            final_norm: alloc_f(hidden)?,
            output_weight: s.alloc_zeros::<u8>(1).map_err(|e| format!("alloc: {e}"))?,
            cache_k,
            cache_v,
            filled: 0,
            d_hidden: alloc_f(hidden)?,
            d_normed: alloc_f(max_in)?,
            d_q: alloc_f(q_width)?,
            d_k: alloc_f(kv_width)?,
            d_v: alloc_f(kv_width)?,
            d_attn: alloc_f(q_width)?,
            d_sk_scores: alloc_f(n_heads * max_pos)?,
            d_sk_chunkmax: alloc_f(n_heads * SPLITK_MAX)?,
            d_sk_lsum: alloc_f(n_heads * SPLITK_MAX)?,
            d_sk_acc: alloc_f(n_heads * SPLITK_MAX * head_dim)?,
            d_proj: alloc_f(hidden)?,
            d_gate: alloc_f(ffn_dim)?,
            d_up: alloc_f(ffn_dim)?,
            d_ffn_act: alloc_f(ffn_dim)?,
            d_in_scales: alloc_f(max_in / 32)?,
            d_in_quants: s
                .alloc_zeros::<i8>(max_in)
                .map_err(|e| format!("alloc: {e}"))?,
            d_logits: alloc_f(vocab)?,
            d_sampled: s.alloc_zeros::<u32>(1).map_err(|e| format!("alloc: {e}"))?,
            d_cos: alloc_f(rope_dim / 2)?,
            d_sin: alloc_f(rope_dim / 2)?,
            d_position: s.alloc_zeros::<i32>(1).map_err(|e| format!("alloc: {e}"))?,
            decode_graph: None,
            verify_scratch: None,
            offload: None,
            k,
        })
    }

    /// Upload one layer's resident weights (Q8_0 36-byte block bytes) + norms.
    #[allow(clippy::too_many_arguments)]
    pub fn set_layer(
        &mut self,
        q: &[u8],
        kk: &[u8],
        v: &[u8],
        o: &[u8],
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
    ) -> Result<(), String> {
        // Default: every layer resident in VRAM (unchanged behavior).
        self.set_layer_located(
            q, kk, v, o, gate, up, down, attn_norm, ffn_norm, None, None, true,
        )
    }

    /// As `set_layer`, but `resident` chooses where the projection weights live:
    /// VRAM (resident, uploaded once) or host RAM (offloaded, streamed to scratch
    /// each forward). The repacked SoA bytes are identical either way. The small
    /// norms always stay resident.
    #[allow(clippy::too_many_arguments)]
    pub fn set_layer_located(
        &mut self,
        q: &[u8],
        kk: &[u8],
        v: &[u8],
        o: &[u8],
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        resident: bool,
    ) -> Result<(), String> {
        let ctx = &self.k.ctx;
        let s = &self.k.stream;
        let up_f = |b: &[f32]| s.clone_htod(b).map_err(|e| format!("htod: {e}"));
        let projections = [q, kk, v, o, gate, up, down];
        let (attn_norm, ffn_norm) = (up_f(attn_norm)?, up_f(ffn_norm)?);
        let q_norm_gpu = q_norm.map(up_f).transpose()?;
        let k_norm_gpu = k_norm.map(up_f).transpose()?;

        if resident {
            // Resident: each projection uploaded once to its own VRAM slice; no
            // offload metadata.
            let vram = |b: &[u8]| -> Result<CudaSlice<u8>, String> {
                s.clone_htod(&repack_q8_soa(b))
                    .map_err(|e| format!("htod: {e}"))
            };
            self.layers.push(ResidentLayer {
                q: vram(projections[0])?,
                k: vram(projections[1])?,
                v: vram(projections[2])?,
                o: vram(projections[3])?,
                gate: vram(projections[4])?,
                up: vram(projections[5])?,
                down: vram(projections[6])?,
                attn_norm,
                ffn_norm,
                q_norm: q_norm_gpu,
                k_norm: k_norm_gpu,
                offloaded: None,
            });
            return Ok(());
        }

        // Offloaded: repack all seven projections and lay them out contiguously in
        // one pinned host buffer so the per-forward transfer is a single memcpy.
        let repacked: Vec<Vec<u8>> = projections.iter().map(|b| repack_q8_soa(b)).collect();
        let mut off = [0usize; 8];
        for (i, r) in repacked.iter().enumerate() {
            off[i + 1] = off[i] + r.len();
        }
        let total = off[7];
        // Cacheable pinned host buffer (faster H2D than write-combined here), filled
        // with the seven projections laid out back-to-back.
        let mut packed = vec![0u8; total];
        for (i, r) in repacked.iter().enumerate() {
            packed[off[i]..off[i + 1]].copy_from_slice(r);
        }
        let pinned = CacheablePinned::from_bytes(ctx, &packed)?;
        // 1-byte placeholders for the resident-projection fields (never read while
        // offloaded — the forward resolves weights from the streamed scratch).
        let ph = || s.clone_htod(&[0u8]).map_err(|e| format!("htod: {e}"));
        self.layers.push(ResidentLayer {
            q: ph()?,
            k: ph()?,
            v: ph()?,
            o: ph()?,
            gate: ph()?,
            up: ph()?,
            down: ph()?,
            attn_norm,
            ffn_norm,
            q_norm: q_norm_gpu,
            k_norm: k_norm_gpu,
            offloaded: Some(OffloadedLayer { host: pinned, off }),
        });
        Ok(())
    }

    /// Allocate the multi-buffered offload state: `N` scratch buffers (each sized to
    /// the largest offloaded layer's total weight bytes), a dedicated copy stream, and
    /// `N` copy-done + `N` compute-done events. `N` is `CAMELID_OFFLOAD_BUFFERS`
    /// (default 2, clamped to >=2). More buffers let the copy stream run further ahead,
    /// but on this hardware throughput is flat past 2: during generation the H2D link is
    /// slower than its idle peak because the compute kernels contend for the memory
    /// controller, so offload is link-bound, not buffer-bound — the extra buffers only
    /// cost VRAM. The knob stays for hardware where the loaded link has more headroom.
    /// Call after all layers are set, only when at least one layer is offloaded.
    pub fn enable_offload_scratch(&mut self) -> Result<(), String> {
        if self.offload.is_some() {
            return Ok(());
        }
        let n_buffers = std::env::var("CAMELID_OFFLOAD_BUFFERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(2);
        // Each scratch buffer is sized to the largest offloaded layer's TOTAL weight
        // bytes (all seven projections contiguous).
        let max_total = self
            .layers
            .iter()
            .filter_map(|l| l.offloaded.as_ref().map(|ol| ol.off[7]))
            .max()
            .unwrap_or(0);
        let copy_stream = self
            .k
            .ctx
            .new_stream()
            .map_err(|e| format!("offload copy stream: {e}"))?;
        let ev = || {
            self.k
                .ctx
                .new_event(None)
                .map_err(|e| format!("offload event: {e}"))
        };
        let s = self.k.stream.clone();
        let mut scratch = Vec::with_capacity(n_buffers);
        let mut copy_done = Vec::with_capacity(n_buffers);
        let mut compute_done = Vec::with_capacity(n_buffers);
        for _ in 0..n_buffers {
            scratch.push(
                s.alloc_zeros::<u8>(max_total)
                    .map_err(|e| format!("scratch alloc: {e}"))?,
            );
            copy_done.push(ev()?);
            compute_done.push(ev()?);
        }
        self.offload = Some(OffloadState {
            scratch,
            copy_stream,
            copy_done,
            compute_done,
        });
        Ok(())
    }

    /// Stream layer `li`'s offloaded weights into scratch buffer `buf` on the copy
    /// stream (asynchronous; the caller records an event and the compute stream waits
    /// on it before reading the buffer).
    fn prefetch_offloaded(
        &mut self,
        li: usize,
        buf: usize,
        copy_stream: &std::sync::Arc<CudaStream>,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("offload prefetch: {e}");
        // Write-after-read: don't overwrite buffer `buf` until the compute that last
        // read it has finished (a no-op the first time the buffer is used).
        copy_stream
            .wait(&self.offload.as_ref().expect("offload state").compute_done[buf])
            .map_err(map)?;
        // ONE contiguous host->device transfer for all seven projections. (The scratch
        // buffer is sized to the largest layer; memcpy_htod copies only host.len()
        // bytes into the front, exactly the range the gemv sub-views read.)
        if self.layers[li].offloaded.is_some() {
            // Borrow the host buffer and the scratch separately (disjoint fields).
            let offloaded = self.offload.as_mut().expect("offload state");
            let sc = &mut offloaded.scratch[buf];
            // SAFETY of the index: `li` is an offloaded layer (checked above). The
            // `&[u8]` view points at pinned memory, so this is the fast async DMA.
            let host = self.layers[li].offloaded.as_ref().unwrap().host.as_bytes();
            copy_stream.memcpy_htod(host, sc).map_err(map)?;
        }
        // Signal that buffer `buf` now holds this layer's weights; the compute
        // stream waits on this before reading the scratch.
        self.offload.as_ref().expect("offload state").copy_done[buf]
            .record(copy_stream)
            .map_err(map)?;
        Ok(())
    }

    pub fn set_output(&mut self, final_norm: &[f32], output_weight: &[u8]) -> Result<(), String> {
        let s = &self.k.stream;
        self.final_norm = s.clone_htod(final_norm).map_err(|e| format!("htod: {e}"))?;
        self.output_weight = s
            .clone_htod(&repack_q8_soa(output_weight))
            .map_err(|e| format!("htod: {e}"))?;
        Ok(())
    }

    /// Diagnostic: time `iters` back-to-back host->device transfers of the largest
    /// offloaded layer on the copy stream, with NO interleaved compute, and return
    /// (bytes_per_transfer, peak_GiB_per_s). This isolates the copy stream's saturated
    /// throughput from the per-forward pipeline's average (which includes compute and
    /// sync gaps), so we can tell whether offload is link-bound or pipeline-bound.
    /// Returns None if nothing is offloaded.
    pub fn probe_offload_pcie(&mut self, iters: usize) -> Option<(usize, f64)> {
        let bytes = self
            .layers
            .iter()
            .filter_map(|l| l.offloaded.as_ref().map(|o| o.off[7]))
            .max()?;
        // Index of the largest offloaded layer (to read its pinned host buffer).
        let li = (0..self.n_layers)
            .filter(|&i| self.layers[i].offloaded.is_some())
            .max_by_key(|&i| self.layers[i].offloaded.as_ref().unwrap().off[7])?;
        let cs = self.offload.as_ref()?.copy_stream.clone();
        // Warmup (ramp the link / first-touch the buffers), then timed loop.
        for _ in 0..3.min(iters) {
            let sc = &mut self.offload.as_mut().unwrap().scratch[0];
            let host = self.layers[li].offloaded.as_ref().unwrap().host.as_bytes();
            cs.memcpy_htod(host, sc).ok()?;
        }
        cs.synchronize().ok()?;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let sc = &mut self.offload.as_mut().unwrap().scratch[0];
            let host = self.layers[li].offloaded.as_ref().unwrap().host.as_bytes();
            cs.memcpy_htod(host, sc).ok()?;
        }
        cs.synchronize().ok()?;
        let secs = start.elapsed().as_secs_f64();
        let gibs = (bytes as f64 * iters as f64) / secs / (1024.0 * 1024.0 * 1024.0);
        Some((bytes, gibs))
    }

    /// Whether `set_layer` has been called for every layer + the output stage.
    pub fn weights_ready(&self) -> bool {
        self.layers.len() == self.n_layers && self.output_weight.len() > 1
    }

    pub fn filled(&self) -> usize {
        self.filled
    }

    pub fn set_filled(&mut self, filled: usize) {
        self.filled = filled;
    }

    /// True when any layer's weights live in host RAM and stream to a GPU scratch buffer
    /// each forward (the capacity split for models too big to fit fully resident, e.g.
    /// 8B on a 6 GiB card). Only `forward_pass` implements that streaming; the batched
    /// layer stack reads VRAM slices directly, so batched prefill must defer to the
    /// serial path when this is true.
    pub fn is_offloaded(&self) -> bool {
        self.offload.is_some() || self.layers.iter().any(|l| l.offloaded.is_some())
    }

    /// Resident KV capacity (positions) this engine was built for. Sized from free
    /// VRAM at build time, so it is the authoritative cap the decode/prefill seams
    /// guard against (a position at or beyond it falls back to the CPU path).
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Seed one layer's KV cache from CPU history. `ck`/`cv` hold positions
    /// `[0, position)` laid out `[kv_head][position'][head_dim]` (stride
    /// `position`); they are written into the existing GPU cache buffers (stride
    /// `max_pos`) in place. For each KV head, positions `[0, position)` are
    /// contiguous in both layouts, so this is one host->device copy of
    /// `position * head_dim` floats per head — `position * kv_width` total, not
    /// the whole `max_pos`-sized buffer. (Re-uploading the full buffer made
    /// seeding a 14-token prompt cost ~160 ms of pointless PCIe traffic.) The CPU
    /// history is already f16-rounded, so it is copied as-is.
    pub fn seed_layer(
        &mut self,
        layer: usize,
        ck: &[f32],
        cv: &[f32],
        position: usize,
    ) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err("seed_layer: layer out of range".into());
        }
        if position == 0 {
            return Ok(());
        }
        let (hd, max_pos, n_kv) = (self.head_dim, self.max_pos, self.n_kv_heads);
        let span = position * hd;
        let s = self.k.stream.clone();
        for h in 0..n_kv {
            let hsrc = h * span; // host: head h's [0,position) block
            let gdst = h * max_pos * hd; // gpu: head h's base (positions 0..)
                                         // The GPU KV cache holds f16 bits; convert host f32 (already f16-rounded) before
                                         // upload so the bytes match what kv_scatter writes.
            let kbits: Vec<u16> = ck[hsrc..hsrc + span]
                .iter()
                .map(|&x| crate::inference::f32_to_f16_bits(x))
                .collect();
            let mut vk = self.cache_k[layer].slice_mut(gdst..gdst + span);
            s.memcpy_htod(&kbits, &mut vk)
                .map_err(|e| format!("seed htod k: {e}"))?;
            let vbits: Vec<u16> = cv[hsrc..hsrc + span]
                .iter()
                .map(|&x| crate::inference::f32_to_f16_bits(x))
                .collect();
            let mut vv = self.cache_v[layer].slice_mut(gdst..gdst + span);
            s.memcpy_htod(&vbits, &mut vv)
                .map_err(|e| format!("seed htod v: {e}"))?;
        }
        Ok(())
    }

    /// Read back the stored K and V for `layer`, positions `[0, n_positions)`, all KV
    /// heads, into `[head][position][head_dim]` host order. Used to make the CPU-side
    /// KV cache authoritative after a GPU prefill so any later CPU-path forward
    /// (diagnostics, fallback) reads the same history the GPU holds.
    pub fn read_kv_layer(
        &self,
        layer: usize,
        n_positions: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (hd, max_pos, n_kv) = (self.head_dim, self.max_pos, self.n_kv_heads);
        let s = self.k.stream.clone();
        let span = n_positions * hd;
        // KV is stored as f16 bits on the GPU; download to u16 then convert back to f32.
        let mut k_bits = vec![0u16; n_kv * span];
        let mut v_bits = vec![0u16; n_kv * span];
        for h in 0..n_kv {
            let gsrc = h * max_pos * hd;
            s.memcpy_dtoh(
                &self.cache_k[layer].slice(gsrc..gsrc + span),
                &mut k_bits[h * span..(h + 1) * span],
            )
            .map_err(|e| format!("read_kv_layer K dtoh: {e}"))?;
            s.memcpy_dtoh(
                &self.cache_v[layer].slice(gsrc..gsrc + span),
                &mut v_bits[h * span..(h + 1) * span],
            )
            .map_err(|e| format!("read_kv_layer V dtoh: {e}"))?;
        }
        self.k
            .ctx
            .synchronize()
            .map_err(|e| format!("read_kv_layer sync: {e}"))?;
        let k_out = k_bits
            .iter()
            .map(|&b| crate::inference::f16_bits_to_f32(b))
            .collect();
        let v_out = v_bits
            .iter()
            .map(|&b| crate::inference::f16_bits_to_f32(b))
            .collect();
        Ok((k_out, v_out))
    }

    /// Run one decode step on the GPU. `embedding` is the current token's f32
    /// embedding; `cos`/`sin` are the per-pair RoPE tables for `position`;
    /// `scale` = 1/sqrt(head_dim). With `compute_logits`, also runs the final
    /// norm + output projection + greedy argmax and returns the sampled token.
    /// One device sync at the end.
    /// Run the full per-token forward on the GPU, leaving the final logits in
    /// `d_logits` when `compute_logits`. Does NOT sample or sync — the public
    /// wrappers (`forward_token` greedy, `forward_token_logits` sampling) add the
    /// tail they need so the forward body is shared.
    #[allow(clippy::too_many_arguments)]
    fn forward_pass(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        compute_logits: bool,
        // When true the kernel chain is being recorded into a CUDA graph: the
        // per-token inputs are NOT uploaded here (the replay does that just before
        // launch) and attention's shared `scores[]` is sized to `max_pos` so the
        // captured launch config holds for every replayed position.
        graph_capture: bool,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        let fused = resident_fusion_enabled();
        let hb = self.hidden / 32; // hidden blocks
        let fb = self.ffn_dim / 32; // ffn blocks
        let qb = self.q_width / 32; // q_width blocks
        let attn_shared = if graph_capture {
            self.max_pos
        } else {
            position + 1
        };

        if !graph_capture {
            s.memcpy_htod(embedding, &mut self.d_hidden).map_err(map)?;
            s.memcpy_htod(cos, &mut self.d_cos).map_err(map)?;
            s.memcpy_htod(sin, &mut self.d_sin).map_err(map)?;
            // Publish the position on the device so kv_scatter/attention read it from
            // memory (graph-replayable) rather than as a frozen launch scalar.
            s.memcpy_htod(&[position as i32], &mut self.d_position)
                .map_err(map)?;
        }

        // Multi-buffered offload streaming (Phase 3c): the weights of the next N-1
        // offloaded layers are prefetched on a separate copy stream so the copy engine
        // stays saturated while the compute stream runs the current layer. `off_idx` is
        // the ordered list of offloaded layer indices; the offloaded layer at sequence
        // position `seq` reads scratch buffer `seq % N` (and that buffer is reused for
        // `seq + N`). Priming fills all N buffers up front so every in-loop wait already
        // has a copy in flight. Where the bytes came from never changes the math.
        let copy_stream = self.offload.as_ref().map(|o| o.copy_stream.clone());
        let n_buf = self.offload.as_ref().map(|o| o.scratch.len()).unwrap_or(0);
        let off_idx: Vec<usize> = if n_buf > 0 {
            (0..self.n_layers)
                .filter(|&i| self.layers[i].offloaded.is_some())
                .collect()
        } else {
            Vec::new()
        };
        if let Some(cs) = &copy_stream {
            for (seq, &li) in off_idx.iter().enumerate().take(n_buf) {
                self.prefetch_offloaded(li, seq % n_buf, cs)?;
            }
        }
        let mut off_seq = 0usize;

        for li in 0..self.n_layers {
            // Resolve this layer's seven projection weights to GPU slices. An
            // offloaded layer (weights in host RAM) reads from the scratch buffer its
            // prefetch streamed into; a resident layer uses its VRAM slice. The math
            // is identical regardless of where the bytes came from — parity holds.
            let offloaded = self.layers[li].offloaded.is_some();
            let cur_buf = if n_buf > 0 { off_seq % n_buf } else { 0 };
            if offloaded && copy_stream.is_some() {
                // Wait for THIS layer's prefetch to land in scratch[cur_buf] before the
                // compute stream reads it. (The look-ahead prefetch that refills this
                // buffer is issued at the END of the layer, AFTER compute_done is
                // recorded — issuing it here would let the copy clobber the buffer this
                // layer is about to read, since its write-after-read event is not yet
                // recorded.)
                s.wait(&self.offload.as_ref().expect("offload state").copy_done[cur_buf])
                    .map_err(map)?;
            }
            // Seven projection weights as GPU views. Offloaded: sub-views into the
            // single streamed scratch buffer at each projection's byte range. Resident:
            // a full-buffer view of each VRAM slice. (Views unify both into the same
            // type for `launch_gemv`; they are zero-copy handles, not allocations.)
            type W<'a> = CudaView<'a, u8>;
            let (wq, wk, wv, wo, wgate, wup, wdown): (W, W, W, W, W, W, W) = if offloaded {
                let off = self.layers[li].offloaded.as_ref().expect("offloaded").off;
                let sc = &self.offload.as_ref().expect("offload state").scratch[cur_buf];
                (
                    sc.slice(off[0]..off[1]),
                    sc.slice(off[1]..off[2]),
                    sc.slice(off[2]..off[3]),
                    sc.slice(off[3]..off[4]),
                    sc.slice(off[4]..off[5]),
                    sc.slice(off[5]..off[6]),
                    sc.slice(off[6]..off[7]),
                )
            } else {
                let l = &self.layers[li];
                (
                    l.q.as_view(),
                    l.k.as_view(),
                    l.v.as_view(),
                    l.o.as_view(),
                    l.gate.as_view(),
                    l.up.as_view(),
                    l.down.as_view(),
                )
            };
            // attention norm + quantize
            if fused {
                launch_rmsnorm_quantize(
                    &s,
                    &self.k.rms_norm_quantize,
                    &self.d_hidden,
                    &self.layers[li].attn_norm,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
            } else {
                launch_rmsnorm(
                    &s,
                    &self.k.rms_norm,
                    &self.d_hidden,
                    &self.layers[li].attn_norm,
                    &mut self.d_normed,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
                launch_quantize(
                    &s,
                    &self.k.quantize,
                    &self.d_normed,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    hb,
                )
                .map_err(map)?;
            }
            // Q,K,V
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &wq,
                self.q_width,
                hb,
                &mut self.d_q,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &wk,
                self.kv_width,
                hb,
                &mut self.d_k,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &wv,
                self.kv_width,
                hb,
                &mut self.d_v,
            )
            .map_err(map)?;
            // Qwen3 QK-norm: per-head RMSNorm on Q and K after projection, before RoPE
            if let (Some(ref qn), Some(ref kn)) = (&self.layers[li].q_norm, &self.layers[li].k_norm)
            {
                launch_rms_norm_per_head(
                    &s,
                    &self.k.rms_norm_per_head,
                    &mut self.d_q,
                    qn,
                    self.n_heads,
                    self.head_dim,
                    self.eps,
                )
                .map_err(map)?;
                launch_rms_norm_per_head(
                    &s,
                    &self.k.rms_norm_per_head,
                    &mut self.d_k,
                    kn,
                    self.n_kv_heads,
                    self.head_dim,
                    self.eps,
                )
                .map_err(map)?;
            }
            // RoPE on Q and K
            let pairing = if self.split_half_pairing { 1i32 } else { 0i32 };
            launch_rope(
                &s,
                &self.k.rope,
                &mut self.d_q,
                &self.d_cos,
                &self.d_sin,
                self.n_heads,
                self.head_dim,
                self.rope_dim,
                pairing,
            )
            .map_err(map)?;
            launch_rope(
                &s,
                &self.k.rope,
                &mut self.d_k,
                &self.d_cos,
                &self.d_sin,
                self.n_kv_heads,
                self.head_dim,
                self.rope_dim,
                pairing,
            )
            .map_err(map)?;
            // KV write
            launch_kv_scatter(
                &s,
                &self.k.kv_scatter,
                &self.d_k,
                &mut self.cache_k[li],
                &self.d_position,
                self.n_kv_heads,
                self.head_dim,
                self.max_pos,
            )
            .map_err(map)?;
            launch_kv_scatter(
                &s,
                &self.k.kv_scatter,
                &self.d_v,
                &mut self.cache_v[li],
                &self.d_position,
                self.n_kv_heads,
                self.head_dim,
                self.max_pos,
            )
            .map_err(map)?;
            // attention. At depth, split-K (grid n_heads x n_splits) fills the SMs that
            // the one-block-per-head launch leaves idle; below SPLITK_THRESHOLD the single
            // kernel is cheaper (one launch, no scratch). Both are token-parity to the same
            // reference. Split-K is skipped during graph capture (split count is ctx-dependent).
            if !graph_capture && attn_shared > SPLITK_THRESHOLD {
                launch_attention_splitk(
                    &s,
                    &self.k,
                    &self.d_q,
                    &self.cache_k[li],
                    &self.cache_v[li],
                    &mut self.d_attn,
                    &mut self.d_sk_scores,
                    &mut self.d_sk_chunkmax,
                    &mut self.d_sk_lsum,
                    &mut self.d_sk_acc,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    &self.d_position,
                    attn_shared,
                    self.max_pos,
                    scale,
                )
                .map_err(map)?;
            } else {
                launch_attention(
                    &s,
                    &self.k.attention,
                    &self.d_q,
                    &self.cache_k[li],
                    &self.cache_v[li],
                    &mut self.d_attn,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    &self.d_position,
                    attn_shared,
                    self.max_pos,
                    scale,
                )
                .map_err(map)?;
            }
            // O projection + residual
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_attn,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                qb,
            )
            .map_err(map)?;
            if fused {
                launch_gemv_residual(
                    &s,
                    &self.k.gemv,
                    &self.d_in_scales,
                    &self.d_in_quants,
                    &wo,
                    self.hidden,
                    qb,
                    &mut self.d_hidden,
                )
                .map_err(map)?;
            } else {
                launch_gemv(
                    &s,
                    &self.k.gemv,
                    &self.d_in_scales,
                    &self.d_in_quants,
                    &wo,
                    self.hidden,
                    qb,
                    &mut self.d_proj,
                )
                .map_err(map)?;
                launch_residual(
                    &s,
                    &self.k.residual_add,
                    &mut self.d_hidden,
                    &self.d_proj,
                    self.hidden,
                )
                .map_err(map)?;
            }
            // ffn norm + gate/up + silu + down + residual
            if fused {
                launch_rmsnorm_quantize(
                    &s,
                    &self.k.rms_norm_quantize,
                    &self.d_hidden,
                    &self.layers[li].ffn_norm,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
            } else {
                launch_rmsnorm(
                    &s,
                    &self.k.rms_norm,
                    &self.d_hidden,
                    &self.layers[li].ffn_norm,
                    &mut self.d_normed,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
                launch_quantize(
                    &s,
                    &self.k.quantize,
                    &self.d_normed,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    hb,
                )
                .map_err(map)?;
            }
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &wgate,
                self.ffn_dim,
                hb,
                &mut self.d_gate,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &wup,
                self.ffn_dim,
                hb,
                &mut self.d_up,
            )
            .map_err(map)?;
            if fused {
                launch_silu_mul_quantize(
                    &s,
                    &self.k.silu_mul_quantize,
                    &self.d_gate,
                    &self.d_up,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    fb,
                )
                .map_err(map)?;
            } else {
                launch_silu_mul(
                    &s,
                    &self.k.silu_mul,
                    &self.d_gate,
                    &self.d_up,
                    &mut self.d_ffn_act,
                    self.ffn_dim,
                )
                .map_err(map)?;
                launch_quantize(
                    &s,
                    &self.k.quantize,
                    &self.d_ffn_act,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    fb,
                )
                .map_err(map)?;
            }
            if fused {
                launch_gemv_residual(
                    &s,
                    &self.k.gemv,
                    &self.d_in_scales,
                    &self.d_in_quants,
                    &wdown,
                    self.hidden,
                    fb,
                    &mut self.d_hidden,
                )
                .map_err(map)?;
            } else {
                launch_gemv(
                    &s,
                    &self.k.gemv,
                    &self.d_in_scales,
                    &self.d_in_quants,
                    &wdown,
                    self.hidden,
                    fb,
                    &mut self.d_proj,
                )
                .map_err(map)?;
                launch_residual(
                    &s,
                    &self.k.residual_add,
                    &mut self.d_hidden,
                    &self.d_proj,
                    self.hidden,
                )
                .map_err(map)?;
            }
            if offloaded {
                if let Some(cs) = &copy_stream {
                    // This layer is done reading scratch[cur_buf]: record compute_done so
                    // the copy stream may reuse the buffer, THEN issue the look-ahead
                    // prefetch of the layer N positions ahead into it. Doing it here (not
                    // at the layer's start) makes the prefetch's write-after-read wait on
                    // a compute_done that is actually recorded, so it never overwrites a
                    // buffer still being read. N-1 transfers stay in flight ahead.
                    self.offload.as_ref().expect("offload state").compute_done[cur_buf]
                        .record(&s)
                        .map_err(map)?;
                    if let Some(&li_ahead) = off_idx.get(off_seq + n_buf) {
                        self.prefetch_offloaded(li_ahead, cur_buf, cs)?;
                    }
                }
                off_seq += 1;
            }
        }

        if !compute_logits {
            return Ok(());
        }
        // final norm + output projection -> d_logits (no argmax / no sync here).
        if fused {
            launch_rmsnorm_quantize(
                &s,
                &self.k.rms_norm_quantize,
                &self.d_hidden,
                &self.final_norm,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                self.hidden,
                self.eps,
            )
            .map_err(map)?;
        } else {
            launch_rmsnorm(
                &s,
                &self.k.rms_norm,
                &self.d_hidden,
                &self.final_norm,
                &mut self.d_normed,
                self.hidden,
                self.eps,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_normed,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                hb,
            )
            .map_err(map)?;
        }
        launch_gemv(
            &s,
            &self.k.gemv,
            &self.d_in_scales,
            &self.d_in_quants,
            &self.output_weight.as_view(),
            self.vocab,
            hb,
            &mut self.d_logits,
        )
        .map_err(map)?;
        Ok(())
    }

    /// Greedy decode: full forward + GPU argmax. Returns the sampled token id, or
    /// `None` when logits were not requested. One device sync at the end.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        if !compute_logits {
            self.forward_pass(embedding, cos, sin, position, scale, false, false)?;
            self.k.ctx.synchronize().map_err(map)?;
            return Ok(None);
        }
        // Greedy decode: replay the captured CUDA graph when enabled (one launch for
        // the whole ~600-kernel token), else the per-launch path. Both are byte-exact:
        // the graph records the identical kernels reading the same device buffers.
        if cuda_graphs_enabled() {
            return self
                .forward_token_greedy_graphed(embedding, cos, sin, position, scale)
                .map(Some);
        }
        self.forward_pass(embedding, cos, sin, position, scale, true, false)?;
        launch_argmax(
            &s,
            &self.k.argmax,
            &self.d_logits,
            self.vocab,
            &mut self.d_sampled,
        )
        .map_err(map)?;
        let mut out = [0u32; 1];
        s.memcpy_dtoh(&self.d_sampled, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(Some(out[0]))
    }

    /// Greedy decode via CUDA graph: upload this token's inputs to device buffers,
    /// then replay the captured forward (lazily recorded on the first call). One
    /// `graph.launch()` replaces the ~600 individual kernel launches, cutting the
    /// host-side launch overhead the profiler flagged. Byte-identical to the
    /// per-launch path: the same kernels read the same buffers; only `position`,
    /// the embedding, and the RoPE tables change, and those arrive through the
    /// device buffers the graph reads.
    fn forward_token_greedy_graphed(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
    ) -> Result<u32, String> {
        use cudarc::driver::sys;
        let map = |e: cudarc::driver::DriverError| format!("cuda graph: {e}");
        let s = self.k.stream.clone();
        // Per-token inputs live in device buffers the (frozen) graph reads on replay.
        s.memcpy_htod(embedding, &mut self.d_hidden).map_err(map)?;
        s.memcpy_htod(cos, &mut self.d_cos).map_err(map)?;
        s.memcpy_htod(sin, &mut self.d_sin).map_err(map)?;
        s.memcpy_htod(&[position as i32], &mut self.d_position)
            .map_err(map)?;

        if self.decode_graph.is_none() {
            // Record the greedy forward (layer stack + output projection + argmax)
            // once. Stream capture records without executing, so this does not write
            // KV; the first real execution is the `launch()` below.
            s.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(map)?;
            let recorded = (|| -> Result<(), String> {
                self.forward_pass(embedding, cos, sin, position, scale, true, true)?;
                launch_argmax(
                    &s,
                    &self.k.argmax,
                    &self.d_logits,
                    self.vocab,
                    &mut self.d_sampled,
                )
                .map_err(map)?;
                Ok(())
            })();
            // Always end capture to leave the stream clean, then surface a record error.
            // flags = 0 (no special instantiation flags); the repr(u32) enum has no
            // zero variant, and cudarc consumes it via `as u32`, so the 0 bits pass.
            let flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0) };
            let captured = s.end_capture(flags);
            recorded?;
            match captured.map_err(map)? {
                Some(graph) => {
                    graph.upload().map_err(map)?;
                    self.decode_graph = Some(SendCudaGraph(graph));
                }
                None => return Err("decode graph capture produced no graph".into()),
            }
        }

        self.decode_graph
            .as_ref()
            .expect("decode graph present")
            .0
            .launch()
            .map_err(map)?;
        let mut out = [0u32; 1];
        s.memcpy_dtoh(&self.d_sampled, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(out[0])
    }

    /// Sampling decode: full forward on the GPU, returns the full f32 logits row
    /// so the CPU sampler can apply temperature / top-p / top-k. This keeps the
    /// whole layer stack on the GPU for non-greedy generation (the chat UI's
    /// default), instead of falling back to the CPU layer loop. One device sync.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_logits(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        self.forward_pass(embedding, cos, sin, position, scale, true, false)?;
        let mut logits = vec![0f32; self.vocab];
        s.memcpy_dtoh(&self.d_logits, &mut logits).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(logits)
    }

    /// Temperature sampling decode entirely on the GPU: full forward, then a
    /// Gumbel-max draw over the logits (one pass, no softmax/sort/host copy).
    /// Returns the sampled token id. `inv_temp` is `1.0 / temperature`; `seed`
    /// varies the draw per token. One device sync. Used for the default chat
    /// case (temperature only); top-k / top-p / penalties stay on the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_sample(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        inv_temp: f32,
        seed: u64,
    ) -> Result<u32, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        self.forward_pass(embedding, cos, sin, position, scale, true, false)?;
        launch_sample_gumbel(
            &s,
            &self.k.sample_gumbel,
            &self.d_logits,
            self.vocab,
            inv_temp,
            seed,
            &mut self.d_sampled,
        )
        .map_err(map)?;
        let mut out = [0u32; 1];
        s.memcpy_dtoh(&self.d_sampled, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(out[0])
    }

    /// GPU prefill of `n` prompt tokens. Each token's forward runs at its own
    /// position (writing its KV); because position i's attention reads the cache
    /// over `[0, i]` only, the sequence of single-token forwards IS the causal
    /// prefill — no separate batched kernels needed. All `n` forwards are enqueued
    /// on the stream back to back with a single device sync at the end (no logits,
    /// no per-token sync), so the whole prompt is processed in one GPU burst
    /// instead of on the CPU. `embeddings` is `n * hidden` f32; `cos_all`/`sin_all`
    /// are the per-position RoPE tables flattened (`n * rope_dim/2`). Leaves the
    /// GPU KV cache holding positions `0..n`.
    pub fn prefill(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        n: usize,
        scale: f32,
    ) -> Result<(), String> {
        let half = self.rope_dim / 2;
        let hidden = self.hidden;
        if embeddings.len() < n * hidden || cos_all.len() < n * half || sin_all.len() < n * half {
            return Err("prefill: input slices too short".into());
        }
        for i in 0..n {
            let emb = &embeddings[i * hidden..(i + 1) * hidden];
            let cos = &cos_all[i * half..(i + 1) * half];
            let sin = &sin_all[i * half..(i + 1) * half];
            self.forward_pass(emb, cos, sin, i, scale, false, false)?;
        }
        self.k
            .ctx
            .synchronize()
            .map_err(|e| format!("cuda prefill sync: {e}"))?;
        Ok(())
    }

    /// Speculative-verify forward: run `k` tokens at consecutive positions
    /// `[base_position, base_position+k)` through the whole model in one batched
    /// pass and return the greedy argmax at each position. Each weight is read
    /// once and reused across the `k` tokens (via `q8_gemm_batched`), so this is
    /// much cheaper than `k` separate `forward_token` calls. The K tokens' K/V are
    /// written to the cache at their positions; the caller decides how many to
    /// keep (accepted prefix) and rewinds `filled`/position accordingly. One sync.
    /// `embeddings` is `k*hidden`; `cos_all`/`sin_all` are per-token RoPE tables
    /// (`k*rope_dim/2`). `k` must be in `1..=MAX_VERIFY_K`.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Result<Vec<u32>, String> {
        if k == 0 || k > MAX_VERIFY_K {
            return Err(format!("verify_batch: k={k} out of 1..={MAX_VERIFY_K}"));
        }
        let map = |e: cudarc::driver::DriverError| format!("cuda verify: {e}");
        // The per-layer batched stack lives in `run_batched_layer_stack` (shared with
        // batched prefill); here we only need the dims for input staging and the final
        // logits projection.
        let (hidden, vocab, eps) = (self.hidden, self.vocab, self.eps);
        let half = self.rope_dim / 2;
        let hb = hidden / 32;
        if embeddings.len() < k * hidden || cos_all.len() < k * half || sin_all.len() < k * half {
            return Err("verify_batch: input slices too short".into());
        }
        self.ensure_verify_scratch()?;
        let s = self.k.stream.clone();
        let mut sc = self.verify_scratch.take().expect("allocated above");

        s.memcpy_htod(
            &embeddings[..k * hidden],
            &mut sc.vh.slice_mut(0..k * hidden),
        )
        .map_err(map)?;
        s.memcpy_htod(&cos_all[..k * half], &mut sc.vcos.slice_mut(0..k * half))
            .map_err(map)?;
        s.memcpy_htod(&sin_all[..k * half], &mut sc.vsin.slice_mut(0..k * half))
            .map_err(map)?;

        self.run_batched_layer_stack(&mut sc, &s, base_position, k, scale)?;
        launch_rms_norm_batched(
            &s,
            &self.k.rms_norm_batched,
            &sc.vh,
            &self.final_norm,
            &mut sc.vn,
            hidden,
            eps,
            k,
        )
        .map_err(map)?;
        launch_quantize(
            &s,
            &self.k.quantize,
            &sc.vn,
            &mut sc.viq,
            &mut sc.vis,
            k * hb,
        )
        .map_err(map)?;
        launch_gemm_batched(
            &s,
            &self.k.gemm_batched,
            &sc.vis,
            &sc.viq,
            &self.output_weight,
            vocab,
            hb,
            k,
            &mut sc.vlogits,
        )
        .map_err(map)?;
        launch_argmax_batched(
            &s,
            &self.k.argmax_batched,
            &sc.vlogits,
            vocab,
            k,
            &mut sc.vsamp,
        )
        .map_err(map)?;
        let mut out = vec![0u32; MAX_VERIFY_K];
        s.memcpy_dtoh(&sc.vsamp, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        out.truncate(k);
        self.verify_scratch = Some(sc);
        Ok(out)
    }

    /// Allocate the K-batched scratch (`verify_scratch`) if not already present.
    /// Sized to `MAX_VERIFY_K * dim` and shared by `verify_batch` and `prefill_batched`.
    /// Idempotent — a no-op once the buffers exist.
    fn ensure_verify_scratch(&mut self) -> Result<(), String> {
        if self.verify_scratch.is_some() {
            return Ok(());
        }
        let (hidden, q_width, kv_width, ffn_dim, vocab) = (
            self.hidden,
            self.q_width,
            self.kv_width,
            self.ffn_dim,
            self.vocab,
        );
        let half = self.rope_dim / 2;
        let st = &self.k.stream;
        let mk = MAX_VERIFY_K;
        let max_in = hidden.max(q_width).max(ffn_dim);
        let af = |n: usize| {
            st.alloc_zeros::<f32>(n)
                .map_err(|e| format!("verify alloc: {e}"))
        };
        self.verify_scratch = Some(VerifyScratch {
            vh: af(mk * hidden)?,
            vn: af(mk * hidden)?,
            viq: st
                .alloc_zeros::<i8>(mk * max_in)
                .map_err(|e| format!("verify alloc: {e}"))?,
            vis: af(mk * (max_in / 32))?,
            vq: af(mk * q_width)?,
            vk: af(mk * kv_width)?,
            vv: af(mk * kv_width)?,
            vattn: af(mk * q_width)?,
            vproj: af(mk * hidden)?,
            vgate: af(mk * ffn_dim)?,
            vup: af(mk * ffn_dim)?,
            vact: af(mk * ffn_dim)?,
            vlogits: af(mk * vocab)?,
            vsamp: st
                .alloc_zeros::<u32>(mk)
                .map_err(|e| format!("verify alloc: {e}"))?,
            vcos: af(mk * half)?,
            vsin: af(mk * half)?,
        });
        Ok(())
    }

    /// Run the batched layer stack for `k` tokens (`1..=MAX_VERIFY_K`) at consecutive
    /// positions `[base_position, base_position+k)`. Reads the staged per-token input
    /// from `sc.vh` / `sc.vcos` / `sc.vsin`, writes each token's K/V into the cache,
    /// and leaves the post-final-layer hidden state in `sc.vh`. The caller stages the
    /// inputs and (for `verify_batch`) projects logits afterward.
    ///
    /// This is the single source of truth for the batched forward, shared by
    /// `verify_batch` (speculative decode) and `prefill_batched`. It is bit-identical
    /// to the serial `forward_pass` per token: `q8_gemm_batched` reproduces the same
    /// per-block integer dot and block-ordered fp32 sum as the decode `q8_gemv`, and
    /// the batched norm/RoPE/scatter/attention kernels match their serial counterparts.
    /// All K/V of the current chunk are scattered before attention reads them, so a
    /// token attends to every earlier position (prior chunks + earlier tokens in this
    /// chunk) exactly as sequential decoding would.
    fn run_batched_layer_stack(
        &mut self,
        sc: &mut VerifyScratch,
        s: &Arc<CudaStream>,
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda batched layers: {e}");
        // Own the Arc locally so each per-launch `&s` is `&Arc<CudaStream>` (what the
        // launch helpers take), not `&&Arc` — the launch calls below are copied verbatim
        // from the original inline loop. Arc::clone is a cheap refcount bump.
        let s = s.clone();
        let (hidden, q_width, kv_width, ffn_dim) =
            (self.hidden, self.q_width, self.kv_width, self.ffn_dim);
        let (head_dim, n_heads, n_kv, rope_dim, max_pos, eps) = (
            self.head_dim,
            self.n_heads,
            self.n_kv_heads,
            self.rope_dim,
            self.max_pos,
            self.eps,
        );
        let (hb, qb, fb) = (hidden / 32, q_width / 32, ffn_dim / 32);
        for li in 0..self.n_layers {
            let layer = &self.layers[li];
            launch_rms_norm_batched(
                &s,
                &self.k.rms_norm_batched,
                &sc.vh,
                &layer.attn_norm,
                &mut sc.vn,
                hidden,
                eps,
                k,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &sc.vn,
                &mut sc.viq,
                &mut sc.vis,
                k * hb,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.q,
                q_width,
                hb,
                k,
                &mut sc.vq,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.k,
                kv_width,
                hb,
                k,
                &mut sc.vk,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.v,
                kv_width,
                hb,
                k,
                &mut sc.vv,
            )
            .map_err(map)?;
            // Qwen3 QK-norm (batched): per-head RMSNorm on Q and K
            if let (Some(ref qn), Some(ref kn)) = (&self.layers[li].q_norm, &self.layers[li].k_norm)
            {
                launch_rms_norm_per_head(
                    &s,
                    &self.k.rms_norm_per_head,
                    &mut sc.vq,
                    qn,
                    k * n_heads,
                    head_dim,
                    eps,
                )
                .map_err(map)?;
                launch_rms_norm_per_head(
                    &s,
                    &self.k.rms_norm_per_head,
                    &mut sc.vk,
                    kn,
                    k * n_kv,
                    head_dim,
                    eps,
                )
                .map_err(map)?;
            }
            let pairing = if self.split_half_pairing { 1i32 } else { 0i32 };
            launch_rope_batched(
                &s,
                &self.k.rope_batched,
                &mut sc.vq,
                &sc.vcos,
                &sc.vsin,
                n_heads,
                head_dim,
                rope_dim,
                q_width,
                k,
                pairing,
            )
            .map_err(map)?;
            launch_rope_batched(
                &s,
                &self.k.rope_batched,
                &mut sc.vk,
                &sc.vcos,
                &sc.vsin,
                n_kv,
                head_dim,
                rope_dim,
                kv_width,
                k,
                pairing,
            )
            .map_err(map)?;
            launch_kv_scatter_batched(
                &s,
                &self.k.kv_scatter_batched,
                &sc.vk,
                &mut self.cache_k[li],
                base_position,
                n_kv,
                head_dim,
                max_pos,
                kv_width,
                k,
            )
            .map_err(map)?;
            launch_kv_scatter_batched(
                &s,
                &self.k.kv_scatter_batched,
                &sc.vv,
                &mut self.cache_v[li],
                base_position,
                n_kv,
                head_dim,
                max_pos,
                kv_width,
                k,
            )
            .map_err(map)?;
            launch_attention_batched(
                &s,
                &self.k.attention_batched,
                &sc.vq,
                &self.cache_k[li],
                &self.cache_v[li],
                &mut sc.vattn,
                n_heads,
                n_kv,
                head_dim,
                base_position,
                max_pos,
                scale,
                q_width,
                k,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &sc.vattn,
                &mut sc.viq,
                &mut sc.vis,
                k * qb,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.o,
                hidden,
                qb,
                k,
                &mut sc.vproj,
            )
            .map_err(map)?;
            launch_residual(&s, &self.k.residual_add, &mut sc.vh, &sc.vproj, k * hidden)
                .map_err(map)?;
            launch_rms_norm_batched(
                &s,
                &self.k.rms_norm_batched,
                &sc.vh,
                &layer.ffn_norm,
                &mut sc.vn,
                hidden,
                eps,
                k,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &sc.vn,
                &mut sc.viq,
                &mut sc.vis,
                k * hb,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.gate,
                ffn_dim,
                hb,
                k,
                &mut sc.vgate,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.up,
                ffn_dim,
                hb,
                k,
                &mut sc.vup,
            )
            .map_err(map)?;
            launch_silu_mul(
                &s,
                &self.k.silu_mul,
                &sc.vgate,
                &sc.vup,
                &mut sc.vact,
                k * ffn_dim,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &sc.vact,
                &mut sc.viq,
                &mut sc.vis,
                k * fb,
            )
            .map_err(map)?;
            launch_gemm_batched(
                &s,
                &self.k.gemm_batched,
                &sc.vis,
                &sc.viq,
                &layer.down,
                hidden,
                fb,
                k,
                &mut sc.vproj,
            )
            .map_err(map)?;
            launch_residual(&s, &self.k.residual_add, &mut sc.vh, &sc.vproj, k * hidden)
                .map_err(map)?;
        }
        Ok(())
    }

    /// Batched GPU prefill: ingest `n` prompt tokens at positions `[0, n)` through the
    /// batched layer stack in chunks of `MAX_VERIFY_K`, reading each weight once per
    /// chunk instead of once per prompt token. The serial `prefill` re-streams every
    /// weight from VRAM once per token (a memory-bound, device-under-filling GEMV per
    /// token); batching turns each weight read into a GEMM amortized over the chunk's
    /// tokens. Writes the KV cache identically to the serial path (same per-block dot
    /// and block-ordered sum), so decode after prefill stays token-identical. Skips the
    /// output projection entirely — prefill needs no logits — saving the large vocab
    /// GEMM the serial path also skips per token.
    pub fn prefill_batched(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        n: usize,
        scale: f32,
    ) -> Result<(), String> {
        // The batched layer stack reads each layer's VRAM weight slice directly and has
        // no offload-streaming path (unlike forward_pass), so for an offloaded model
        // (e.g. 8B on a 6 GiB card) it would read placeholder bytes. Fall back to the
        // serial prefill, which streams offloaded weights correctly. Batching is a
        // resident-only fast path.
        if self.is_offloaded() {
            return self.prefill(embeddings, cos_all, sin_all, n, scale);
        }
        let map = |e: cudarc::driver::DriverError| format!("cuda prefill: {e}");
        let hidden = self.hidden;
        let half = self.rope_dim / 2;
        if embeddings.len() < n * hidden || cos_all.len() < n * half || sin_all.len() < n * half {
            return Err("prefill_batched: input slices too short".into());
        }
        self.ensure_verify_scratch()?;
        let s = self.k.stream.clone();
        let mut sc = self.verify_scratch.take().expect("allocated above");
        let mut base = 0usize;
        while base < n {
            let kk = (n - base).min(MAX_VERIFY_K);
            // Stage this chunk's embeddings + RoPE tables into the shared scratch at
            // offset 0; the layer stack reads [0, kk) and scatters K/V at [base, base+kk).
            s.memcpy_htod(
                &embeddings[base * hidden..(base + kk) * hidden],
                &mut sc.vh.slice_mut(0..kk * hidden),
            )
            .map_err(map)?;
            s.memcpy_htod(
                &cos_all[base * half..(base + kk) * half],
                &mut sc.vcos.slice_mut(0..kk * half),
            )
            .map_err(map)?;
            s.memcpy_htod(
                &sin_all[base * half..(base + kk) * half],
                &mut sc.vsin.slice_mut(0..kk * half),
            )
            .map_err(map)?;
            // Same stream → the next chunk's stage waits for this chunk's reads; no
            // explicit per-chunk sync needed (matches the serial prefill's one-sync-at-end).
            self.run_batched_layer_stack(&mut sc, &s, base, kk, scale)?;
            base += kk;
        }
        self.k
            .ctx
            .synchronize()
            .map_err(|e| format!("cuda prefill sync: {e}"))?;
        self.verify_scratch = Some(sc);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
