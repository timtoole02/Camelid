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
// (rows*blocks_per_row*32 i8, 16-byte aligned), then the original f16 scale bits
// (rows*blocks_per_row u16). Quants-first means each block's 32 i8 are read as two aligned int4 loads
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
        const unsigned short* scales =
            reinterpret_cast<const unsigned short*>(weight_bytes + total_blocks * 32);
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
                    ws[u] = f16_bits_to_f32(scales[row_block0 + b]);
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

// ---- Q4_0 GEMV: one warp per output row, raw 18-byte wire, Q8_0 activation ----
// Bit-identical reproduction of the validated CPU oracle `q4_0_wire_row_dot_scalar`
// (the gemma4 QAT linear lane). Per 18-byte block: scale = f16(blk[0..2]); for
// j in 0..16, lo = (byte & 0xF) - 8, hi = (byte >> 4) - 8; isum += lo*y[j] +
// hi*y[j+16]; term = (float)isum * w_scale * x_scale[b]. Lane 0 sums the per-block
// terms IN ORDER — the exact same ordered-f32 contract as q8_gemv, so the result is
// bit-identical to the CPU. Weights are read RAW (nibbles packed) to keep the 4-bit
// footprint; the activation is Q8_0 (input_scales[bpr] + input_quants[bpr*32] i8),
// staged once in shared like q8_gemv. The -8 bias precludes a clean __dp4a, so the
// per-block integer dot is the scalar nibble unpack the oracle uses (parity-first).
extern "C" __global__ void q4_0_gemv(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem40[];
    signed char* s_iq = (signed char*)smem40;                        // blocks_per_row*32 i8
    float* s_is = (float*)(smem40 + (long)blocks_per_row * 32);       // blocks_per_row f32
    float* terms = (float*)(smem40 + (long)blocks_per_row * 36);      // warps*blocks_per_row f32
    int tid = threadIdx.x;
    // Stage the shared Q8_0 input vector cooperatively (coalesced), once per block.
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];             // blocks_per_row*32 bytes as ints
    for (int i = tid; i < blocks_per_row; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    const int WIRE = 18;
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_block0 + b) * WIRE;
            float w_scale = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            const signed char* y = s_iq + (long)b * 32;
            int isum = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char byte = blk[2 + j];
                int lo = (int)(byte & 0xF) - 8;
                int hi = (int)(byte >> 4) - 8;
                isum += lo * (int)y[j];
                isum += hi * (int)y[j + 16];
            }
            myterms[b] = (float)isum * w_scale * s_is[b];
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) acc += myterms[b];
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- Q4_1 GEMV: one warp per output row, raw 20-byte wire, Q8_0 activation -----
// Bit-identical to the CPU oracle `q4_1_wire_row_dot`. Q4_1 block = 20 bytes: d =
// f16(blk[0..2]), m = f16(blk[2..4]), then 16 nibble bytes. The nibble is UNSIGNED
// (no -8 bias); dequant = q*d + m. Factored exactly like the oracle: per block
// isum = Σ q*y, asum = Σ y; term = (d*isum + m*asum) * x_scale[b]. Lane 0 sums the
// per-block terms IN ORDER (same ordered-f32 contract as q4_0/q8). The activation is
// Q8_0 (input_scales[bpr] + input_quants[bpr*32] i8), staged once in shared.
extern "C" __global__ void q4_1_gemv(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem41[];
    signed char* s_iq = (signed char*)smem41;                        // blocks_per_row*32 i8
    float* s_is = (float*)(smem41 + (long)blocks_per_row * 32);       // blocks_per_row f32
    float* terms = (float*)(smem41 + (long)blocks_per_row * 36);      // warps*blocks_per_row f32
    int tid = threadIdx.x;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
    for (int i = tid; i < blocks_per_row; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    const int WIRE = 20;
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_block0 + b) * WIRE;
            float w_d = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            float w_m = f16_bits_to_f32((unsigned short)(blk[2] | (blk[3] << 8)));
            const signed char* y = s_iq + (long)b * 32;
            int isum = 0;
            int asum = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char byte = blk[4 + j];
                int lo = (int)(byte & 0xF);
                int hi = (int)(byte >> 4);
                int ylo = (int)y[j];
                int yhi = (int)y[j + 16];
                isum += lo * ylo + hi * yhi;
                asum += ylo + yhi;
            }
            myterms[b] = (w_d * (float)isum + w_m * (float)asum) * s_is[b];
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) acc += myterms[b];
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- UE4M3 sub-block scale decode (header-free, exact) ---------------------
// Bit-for-bit port of tensor::ue4m3_to_f32_const (which is itself pin-CPU-bitwise
// vs ggml_ue4m3_to_fp32, ggml-impl.h): raw bytes 0x00 and 0x7F flush to 0.0 (the
// NaN sentinel is checked on the RAW byte, so 0xFF is NOT flushed and decodes to
// 240.0 — pin-CPU semantics; sentinel-bearing files are refused whole at load in
// both lanes, so 0xFF only ever appears in crafted below-the-refusal-seam tests).
// exp = bits 6..3 (bias 7), man = bits 2..0; exp==0 is subnormal man*2^-9, else
// (1 + man/8) * 2^(exp-7); the extra 0.5 is the doubled-LUT pair-rule factor.
// Every step scales an exact value by a power of two built directly from its
// biased exponent via __uint_as_float, so with --fmad=false the result is
// bit-equal to the Rust const table by construction (no libm, no rounding slack).
__device__ __forceinline__ float ue4m3_to_f32(unsigned char byte) {
    if (byte == 0x00 || byte == 0x7F) return 0.0f;
    int e = (byte >> 3) & 0xF;
    float man = (float)(byte & 0x7);
    float raw;
    if (e == 0) {
        raw = man * __uint_as_float((unsigned int)(127 - 9) << 23); // man * 2^-9
    } else {
        raw = (1.0f + man / 8.0f) * __uint_as_float((unsigned int)(e - 7 + 127) << 23);
    }
    return raw * 0.5f;
}

// ---- NVFP4 nibble -> signed-int8 codebook expansion via __byte_perm ----------
// Ported EXACTLY from the pin's get_int_from_table_16 (ggml-cuda/vecdotq.cuh:34-80,
// CUDA branch). `q4` packs 8 nibbles across 4 little-endian bytes; the returned
// pair holds the codebook value at each nibble index as four packed int8s ready
// for __dp4a: `.x` = the four EVEN indices (the low nibble of each byte), `.y` =
// the four ODD indices (the high nibble of each byte). CUDA has no 4-bit-index
// byte select, so __byte_perm (3-bit index) is used twice per half with a
// fourth-bit low/high merge. Crucially it selects whole BYTES from `table`, so a
// signed codebook entry (e.g. -12 stored as 0xF4) survives intact and __dp4a
// reads it back as a signed int8. `table` is the 16-entry signed E2M1 codebook.
__device__ __forceinline__ int2 nvfp4_table_lookup_16(int q4, const signed char* table) {
    const unsigned int* table32 = (const unsigned int*) table;
    // __byte_perm selects bytes from the low 3 bits of each index nibble; the 4th
    // (sign/high-half) bit picks between the low and high 8 table entries.
    const unsigned int low_high_selection_indices = (0x32103210u | ((q4 & 0x88888888) >> 1));
    unsigned int tmp[2];
    #pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
        const unsigned int shift = 16 * i;
        const unsigned int low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const unsigned int high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    // tmp holds the bytes in nibble order; regroup into all-even / all-odd ints.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---- NVFP4 GEMV: one warp per output row, raw 36-byte wire, Q8_0 activation ----
// Bit-identical reproduction of the validated CPU oracle `nvfp4_wire_row_dot`
// (inference.rs), i.e. the pin's `ggml_vec_dot_nvfp4_q8_0_generic` numeric shape.
// Each 64-value NVFP4 superblock is 36 wire bytes: d[4] UE4M3 sub-block scales
// then qs[32] packed E2M1 nibbles. One superblock spans TWO 32-value Q8_0
// activation blocks: sub-blocks s=0,1 dot input[2*ib], s=2,3 dot input[2*ib+1],
// at offset (s%2)*16 within that block; the low nibble of qs[s*8+j] is element
// (s*16+j), the high nibble is element (s*16+8+j). Per sub-block the integer
// accumulation is an EXACT i32 scalar-LUT nibble unpack (the q4_0_gemv precedent —
// KV[] is tensor::KVALUES_MXFP4_I8), then the term is (x_scale * ue4m3(d[s])) *
// (float)(sumi_lo + sumi_hi) with the SAME left-to-right association as the Rust,
// and lane 0 sums the per-sub-block terms IN superblock-major / sub-block-minor
// order == the CPU loop order, so the reduction stays token-identical. Weights are
// read RAW (4.5 bpw preserved, no host expansion); the activation is the shared
// Q8_0 buffer staged once like q8_gemv. `blocks_per_row` is the Q8_0 block count
// (in_dim/32); one superblock spans two, so n_sb = blocks_per_row/2 (the launcher
// refuses an odd count). PHASE 4b (Option B): the per-sub-block integer dot is now
// the pin's get_int_from_table_16 __byte_perm expansion + __dp4a inner loop (see
// the loop body). The v1 scalar KV-LUT nibble unpack was correct (46/46 bit-
// identical) but COMPUTE-BOUND on this box (13.3% of the 336 GB/s DRAM roofline vs
// Q8_0's 39.8%), so its 1.70x byte reduction did not become speed; dp4a yields the
// IDENTICAL i32 sumi (recon §3.2), so it cannot move parity, only the instructions.
extern "C" __global__ void nvfp4_gemv(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smemN4[];
    signed char* s_iq = (signed char*)smemN4;                        // blocks_per_row*32 i8
    float* s_is = (float*)(smemN4 + (long)blocks_per_row * 32);       // blocks_per_row f32
    float* terms = (float*)(smemN4 + (long)blocks_per_row * 36);      // warps*2*blocks_per_row f32
    int tid = threadIdx.x;
    // Stage the shared Q8_0 input vector cooperatively (coalesced), once per block.
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];             // blocks_per_row*32 bytes as ints
    for (int i = tid; i < blocks_per_row; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int n_sb = blocks_per_row >> 1;                                  // NVFP4 superblocks per row (bpr even)
    float* myterms = terms + (long)warp * 2 * blocks_per_row;        // 2*bpr sub-block terms per warp
    const int WIRE = 36;
    // Signed E2M1 codebook (== tensor::KVALUES_MXFP4_I8), stored as int8 bytes so
    // the __byte_perm table lookup (nvfp4_table_lookup_16) selects the correct
    // signed value for __dp4a (e.g. -12 == 0xF4 survives as a signed int8).
    const signed char KV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    if (row < rows) {
        long row_sb0 = (long)row * n_sb;
        for (int b = lane; b < n_sb; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                float d = ue4m3_to_f32(blk[s]);
                int act_blk = 2 * b + (s >> 1);         // input[2*ib + s/2]
                int off = (s & 1) * 16;                 // (s%2)*16 within the activation block
                const signed char* y = s_iq + (long)act_blk * 32;
                const unsigned char* qs = blk + 4 + s * 8;
                // v1 (scalar, kept for the receipt) accumulated, over j=0..8:
                //   sumi_lo += KV[qs[j] & 0xF] * y[off + j];
                //   sumi_hi += KV[qs[j] >> 4] * y[off + 8 + j];
                // then used (float)(sumi_lo + sumi_hi). That was compute-bound
                // (13.3% of roofline on this box), so the nibble unpack + KV
                // multiply is replaced by nvfp4_table_lookup_16 (__byte_perm) +
                // __dp4a: t0/t1 expand the 8 low + 8 high nibbles to signed int8
                // codebook values (.x = low nibbles, .y = high nibbles), and four
                // __dp4a calls form the SAME 16 low + 16 high signed-int8 products.
                // Integer add is exact and order-free, so the accumulated i32
                // equals sumi_lo + sumi_hi exactly — parity is untouched (§3.2);
                // the sub-block UE4M3 scale is still applied ONCE here, and the
                // lane-0 ordered f32 sum below is UNCHANGED.
                int q0 = *reinterpret_cast<const int*>(qs);       // qs[0..4]
                int q1 = *reinterpret_cast<const int*>(qs + 4);   // qs[4..8]
                int2 t0 = nvfp4_table_lookup_16(q0, KV);          // .x low, .y high nibbles
                int2 t1 = nvfp4_table_lookup_16(q1, KV);
                const int* ylo = reinterpret_cast<const int*>(y + off);      // y[off+0..8]
                const int* yhi = reinterpret_cast<const int*>(y + off + 8);  // y[off+8..16]
                int sumi = __dp4a(t0.x, ylo[0], 0);   // KV[qs0..3 & F] . y[off+0..4]
                sumi = __dp4a(t1.x, ylo[1], sumi);    // KV[qs4..7 & F] . y[off+4..8]
                sumi = __dp4a(t0.y, yhi[0], sumi);    // KV[qs0..3 >> 4] . y[off+8..12]
                sumi = __dp4a(t1.y, yhi[1], sumi);    // KV[qs4..7 >> 4] . y[off+12..16]
                myterms[4 * b + s] = (s_is[act_blk] * d) * (float)sumi;
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int t = 0; t < 4 * n_sb; t++) acc += myterms[t];  // superblock-major, sub-block-minor
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- Q4_K_M GEMV: one warp per output row, fused dequant + integer dot -------
// Bit-identical reproduction of the validated CPU oracle `q4_k_wire_row_dot`
// (ggml_vec_dot_q4_K_q8_K_generic shape). The activation is Q8_K (256-wide blocks
// WITH per-16 bsums), NOT Q8_0. Weights are the repacked SoA layout (see
// repack_q4k_soa): first all expanded 4-bit quants (rows*n_sb*256 i8, the oracle's
// `a[256]` — nibbles already expanded low-then-high in 64-value groups), then
// per-superblock metadata: d & dmin (f32 each, the f16 super-scales already
// widened) and the 8 unpacked 6-bit scales + 8 unpacked mins (u8 each, the kmask
// `utmp` unpack already done on the host). The per-superblock integer dot is kept
// scalar (correctness-first, matching the oracle's "no SIMD" doc) because the
// oracle's 8-lane f32 split (below) cannot be reproduced by a 4-wide __dp4a, which
// would collapse four distinct lanes into one accumulator.
//
// PARITY ANCHOR: the oracle keeps 8 f32 main-lane accumulators sums[0..8] plus a
// scalar mins accumulator sumf, both summed over superblocks IN ORDER, with final
// `sumf + sums[0] + ... + sums[7]` (left-to-right). The per-superblock integer
// work (aux32[l] for l in 0..8, and the mins integer sumi) is exact regardless of
// order, so the lanes compute those integers per superblock and stash them in
// shared (the analog of q8_gemv's myterms[b]); lane 0 then replays the EXACT f32
// accumulation order. The 8-lane split is load-bearing: dd*aux32[l] is rounded to
// f32 per lane before summing, so collapsing the lanes would change the f32 result.
//
// aux32[l] = Σ_{j=0..8} scale[j] * Σ_{k=0..4} q8[j*32+k*8+l] * a[j*32+k*8+l]
//   (lane l owns the 8th element of every 8-stride within each 32-group; folding
//    the per-group scale into the integer accumulator matches the oracle exactly)
// sumi      = Σ_{j=0..16} mins[j/2] * Σ_{l=0..16} q8[j*16+l]   (per-16 bsums)
// term_main += dd * aux32[l]   (dd = d * d_act),  per superblock, per lane
// term_min  -= dmin * d_act * sumi               per superblock
extern "C" __global__ void q4k_gemv(
    const float* __restrict__ input_scales,         // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,   // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_bytes, // RAW 144-byte Q4_K wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem4[];
    signed char* s_iq = (signed char*)smem4;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem4 + (long)n_sb * 256);        // n_sb f32 staged scales
    // per-warp scratch: 8 aux32 lanes + 1 sumi = 9 ints, per superblock.
    int* aux = (int*)(smem4 + (long)n_sb * 256 + (long)n_sb * 4); // warps*n_sb*9 int
    int tid = threadIdx.x;
    // Stage the input vector SWIZZLED to match the upload-time weight swizzle
    // (swz_q4k_blocks): within each 32-wide scale group jg, byte l+k*8 lands at
    // l*4+k, so an aux lane's four stride-8 activations sit in ONE aligned i32
    // and pair with the weight word for __dp4a. A pure byte permutation —
    // group sums (bsums) and per-lane products are the same integers.
    for (int i = tid; i < n_sb * 256; i += blockDim.x) {
        int jg = i >> 5;      // 32-wide group index
        int p = i & 31;       // linear position within the group
        int l = p & 7;
        int kk = p >> 3;
        s_iq[(long)jg * 32 + l * 4 + kk] = input_quants[i];
    }
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 144;
    const unsigned int KMASK1 = 0x3f3f3f3fu, KMASK2 = 0x0f0f0f0fu, KMASK3 = 0x03030303u;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myaux = aux + (long)warp * n_sb * 9;
    // Phase 1 — integer partials with ALL 32 lanes cooperating. Work unit
    // u = (super-block b = u>>2, byte-group g = u&3): 32 quant bytes = two
    // consecutive uint4 loads, so the four units of one super-block read
    // consecutive 32 B chunks and the warp's loads coalesce (the old layout
    // gave each lane a whole 144 B-strided super-block, leaving half the warp
    // idle whenever n_sb <= 16 — every 4096-col projection — and defeating
    // coalescing; measured ~60 GB/s achieved vs ~20-25% of that fixed here).
    // NEGATIVE RESULT (measured 2026-07-03, do not retry): a warp-sequential
    // remap (whole warp walks SBs in order, lane n reads quant word n — one
    // perfectly coalesced 128 B transaction per SB) REGRESSED ~8% end-to-end
    // (Q4_K_M 18.7 -> 17.1 tok/s, parity intact). The per-SB combine needs ~7
    // dependent shuffles + a shared store, serializing the loop, while THIS
    // unit-spread shape keeps 8 independent dp4a chains in flight and L1
    // already absorbs the scattered-sector cost. ~135 GB/s is the plateau of
    // this shape; the llama.cpp mmvq gap (~190 GB/s) is not reachable by
    // load-coalescing alone.
    // Each unit computes its two scale-groups' contributions to the 8 aux
    // lanes plus the mins-side sumi over ITS activation quarter (per-16
    // groups 4g..4g+4, mins index gg>>1 = 2g/2g+1 — an exact partition of the
    // oracle's loops). Integer addition is associative, so this split is
    // BIT-EXACT; partials combine across the 4 lanes of a super-block with
    // two shuffle steps and the g==0 lane stores the 9 totals. The ordered
    // f32 tail below is byte-for-byte the oracle's — parity is unchanged.
    long row_sb0 = (long)row * n_sb;
    int units = n_sb * 4;
    for (int u0 = 0; u0 < units; u0 += 32) {
        int u = u0 + lane;
        bool active = (u < units) && (row < rows);
        int b = u >> 2;
        int g = u & 3;
        int aux32[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) aux32[l] = 0;
        int sumi = 0;
        if (active) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            const signed char* y256 = s_iq + (long)b * 256;  // staged activation
            // Unpack the packed 6-bit (scale, min) pairs via the kmask scheme
            // (oracle order). The 12 scale/min bytes are header bytes 4..16.
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);  // bytes 0..16
            unsigned int u0w = hdr.y;
            unsigned int u1 = hdr.z;
            unsigned int u2 = hdr.w;
            unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            unsigned int uaux = u1 & KMASK1;
            u1 = (u2 & KMASK2) | (((u0w >> 6) & KMASK3) << 4);
            u2 = uaux;
            u0w &= KMASK1;
            unsigned char sc[8], mn[8];
            sc[0] = u0w & 0xff; sc[1] = (u0w >> 8) & 0xff; sc[2] = (u0w >> 16) & 0xff; sc[3] = (u0w >> 24) & 0xff;
            sc[4] = u1 & 0xff; sc[5] = (u1 >> 8) & 0xff; sc[6] = (u1 >> 16) & 0xff; sc[7] = (u1 >> 24) & 0xff;
            mn[0] = u2 & 0xff; mn[1] = (u2 >> 8) & 0xff; mn[2] = (u2 >> 16) & 0xff; mn[3] = (u2 >> 24) & 0xff;
            mn[4] = u3 & 0xff; mn[5] = (u3 >> 8) & 0xff; mn[6] = (u3 >> 16) & 0xff; mn[7] = (u3 >> 24) & 0xff;
            int slo = (int)sc[2 * g];
            int shi = (int)sc[2 * g + 1];
            // dp4a form over the SWIZZLED layouts: weight word l holds the four
            // stride-8 quant bytes of aux lane l (swz_q4k_blocks); the staged
            // activation words are permuted identically, and the low/high
            // nibble halves of the SAME weight word serve scale groups 2g and
            // 2g+1. aux32[l] = slo*Σ(y_lo·q_lo) + shi*Σ(y_hi·q_hi) — the
            // distributed form of the oracle's per-element products: identical
            // integers. The mins side collapses because the two per-16 bsums of
            // a 32-group share one min: sumi += mn[j]*(whole-group activation
            // sum), computed as packed dp4a against 0x01010101.
            const int* qw = reinterpret_cast<const int*>(blk + 16 + g * 32);
            const int* ylo = reinterpret_cast<const int*>(y256 + (2 * g) * 32);
            const int* yhi = reinterpret_cast<const int*>(y256 + (2 * g + 1) * 32);
            int sum_lo = 0, sum_hi = 0;
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                int q = qw[l];
                int yl = ylo[l];
                int yh = yhi[l];
                int qlo = q & 0x0F0F0F0F;
                int qhi = (q >> 4) & 0x0F0F0F0F;
                aux32[l] += slo * __dp4a(qlo, yl, 0) + shi * __dp4a(qhi, yh, 0);
                sum_lo = __dp4a(yl, 0x01010101, sum_lo);
                sum_hi = __dp4a(yh, 0x01010101, sum_hi);
            }
            sumi += (int)mn[2 * g] * sum_lo + (int)mn[2 * g + 1] * sum_hi;
        }
        // Combine the 4 same-super-block lanes (g==0 collects g=1..3). Lanes
        // whose shuffle source crosses a group boundary read garbage, but only
        // g==0 lanes store, so it is discarded. Integer sums — order-free.
        #pragma unroll
        for (int off = 2; off >= 1; off >>= 1) {
            #pragma unroll
            for (int l = 0; l < 8; l++)
                aux32[l] += __shfl_down_sync(0xffffffffu, aux32[l], off);
            sumi += __shfl_down_sync(0xffffffffu, sumi, off);
        }
        if (active && g == 0) {
            int* ax = myaux + (long)b * 9;
            #pragma unroll
            for (int l = 0; l < 8; l++) ax[l] = aux32[l];
            ax[8] = sumi;
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        long row_sb0 = (long)row * n_sb;
        float sums[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) sums[l] = 0.0f;
        float sumf = 0.0f;
        for (int b = 0; b < n_sb; b++) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            int* ax = myaux + (long)b * 9;
            float d = f16_bits_to_f32((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
            float dmin = f16_bits_to_f32((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
            float dact = s_is[b];
            float dd = d * dact;
            #pragma unroll
            for (int l = 0; l < 8; l++) sums[l] += dd * (float)ax[l];
            sumf -= dmin * dact * (float)ax[8];
        }
        // Final reduction in the oracle's EXACT order: it returns
        // `sumf + sums.iter().sum()`, i.e. the 8 main lanes are summed FIRST
        // (left-to-right from 0.0) and only then added to the mins accumulator sumf.
        // `(((sumf+s0)+s1)+...)` would be a different f32 association — keep this split.
        float smain = 0.0f;
        #pragma unroll
        for (int l = 0; l < 8; l++) smain += sums[l];
        float acc = sumf + smain;
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- Q5_K_M GEMV: one warp per output row, fused dequant + integer dot -------
// Bit-identical reproduction of the validated CPU oracle `q5_k_wire_row_dot`.
// Q5_K is Q4_K PLUS a fifth bit per weight, so this is `q4k_gemv` with two
// differences: (1) the super-block is 176 bytes (WIRE) instead of 144 — the
// layout is d(f16), dmin(f16), scales[12], qh[32], qs[128] (the 32-byte high-bit
// plane sits BETWEEN the packed scales and the low nibbles); and (2) the weight
// rebuild folds the qh high bit in, so codes are 0..31 not 0..15. EVERYTHING else
// — the kmask 6-bit scale/min unpack, the per-16 mins subtraction, and the 8-lane
// `d_w·d_act` ordered f32 accumulation (the parity anchor) — is IDENTICAL to
// q4k_gemv. See that kernel's header for the full anchor rationale.
//
// The qh plane is 32 bytes indexed by the position p (0..32) WITHIN each 32-value
// byte-group, and the SAME 32 bytes are reused for all four byte-groups g (0..4);
// only the selected bit changes: low nibble uses bit (1<<(2g)), high nibble bit
// (1<<(2g+1)). This exactly mirrors the oracle's `u1 = 1<<(2j)`, `u2 = 1<<(2j+1)`
// per j-group with `a[j*64+l] = low + (qh[l]&u1?16:0)` etc. (qh[l] reused per j).
extern "C" __global__ void q5k_gemv(
    const float* __restrict__ input_scales,         // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,   // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_bytes, // RAW 176-byte Q5_K wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem5[];
    signed char* s_iq = (signed char*)smem5;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem5 + (long)n_sb * 256);        // n_sb f32 staged scales
    // per-warp scratch: 8 aux32 lanes + 1 sumi = 9 ints, per superblock.
    int* aux = (int*)(smem5 + (long)n_sb * 256 + (long)n_sb * 4); // warps*n_sb*9 int
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i]; // n_sb*256 bytes as ints
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 176;
    const unsigned int KMASK1 = 0x3f3f3f3fu, KMASK2 = 0x0f0f0f0fu, KMASK3 = 0x03030303u;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myaux = aux + (long)warp * n_sb * 9;
    if (row < rows) {
        long row_sb0 = (long)row * n_sb;
        for (int b = lane; b < n_sb; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            const signed char* y256 = s_iq + (long)b * 256;  // staged activation
            int* ax = myaux + (long)b * 9;
            // Packed 6-bit (scale, min) unpack via the kmask scheme (bytes 4..16).
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);  // bytes 0..16
            unsigned int u0 = hdr.y;
            unsigned int u1 = hdr.z;
            unsigned int u2 = hdr.w;
            unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            unsigned int uaux = u1 & KMASK1;
            u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
            u2 = uaux;
            u0 &= KMASK1;
            unsigned char sc[8], mn[8];
            sc[0] = u0 & 0xff; sc[1] = (u0 >> 8) & 0xff; sc[2] = (u0 >> 16) & 0xff; sc[3] = (u0 >> 24) & 0xff;
            sc[4] = u1 & 0xff; sc[5] = (u1 >> 8) & 0xff; sc[6] = (u1 >> 16) & 0xff; sc[7] = (u1 >> 24) & 0xff;
            mn[0] = u2 & 0xff; mn[1] = (u2 >> 8) & 0xff; mn[2] = (u2 >> 16) & 0xff; mn[3] = (u2 >> 24) & 0xff;
            mn[4] = u3 & 0xff; mn[5] = (u3 >> 8) & 0xff; mn[6] = (u3 >> 16) & 0xff; mn[7] = (u3 >> 24) & 0xff;
            // qh: 32 high bits, bytes 16..48. Loaded as 2 uint4 (8 uint32 words); the
            // SAME 32 bytes serve every byte-group (only the selected bit varies).
            const uint4* qhv = reinterpret_cast<const uint4*>(blk + 16);
            uint4 qh_a = qhv[0];
            uint4 qh_b = qhv[1];
            const unsigned int* qhd = reinterpret_cast<const unsigned int*>(&qh_a);
            const unsigned int* qhd2 = reinterpret_cast<const unsigned int*>(&qh_b);
            const uint4* q5v = reinterpret_cast<const uint4*>(blk + 48);  // 128 low-nibble bytes
            int aux32[8];
            #pragma unroll
            for (int l = 0; l < 8; l++) aux32[l] = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int slo = (int)sc[2 * g];
                int shi = (int)sc[2 * g + 1];
                int lobase = g * 64;          // a-index of low-nibble scale-group 2g
                int hibase = g * 64 + 32;     // a-index of high-nibble scale-group 2g+1
                unsigned int mlo = 1u << (2 * g);      // qh bit for the low nibble
                unsigned int mhi = 1u << (2 * g + 1);  // qh bit for the high nibble
                uint4 wlo = q5v[g * 2];
                uint4 whi = q5v[g * 2 + 1];
                const unsigned int* wd = reinterpret_cast<const unsigned int*>(&wlo);
                const unsigned int* wd2 = reinterpret_cast<const unsigned int*>(&whi);
                #pragma unroll
                for (int w = 0; w < 8; w++) {
                    unsigned int word = (w < 4) ? wd[w] : wd2[w - 4];       // 4 low-nibble bytes
                    unsigned int qhword = (w < 4) ? qhd[w] : qhd2[w - 4];   // 4 matching qh bytes
                    #pragma unroll
                    for (int t = 0; t < 4; t++) {
                        int p = w * 4 + t;             // 0..32 position in the group
                        unsigned int byte = (word >> (t * 8)) & 0xff;
                        unsigned int qhb = (qhword >> (t * 8)) & 0xff;
                        int lo = (int)(byte & 0xF) + ((qhb & mlo) ? 16 : 0);
                        int hi = (int)(byte >> 4) + ((qhb & mhi) ? 16 : 0);
                        int l = p & 7;
                        aux32[l] += slo * (int)y256[lobase + p] * lo;
                        aux32[l] += shi * (int)y256[hibase + p] * hi;
                    }
                }
            }
            #pragma unroll
            for (int l = 0; l < 8; l++) ax[l] = aux32[l];
            // Mins side: identical to q4k — per-16 activation sums times mins[g/2].
            int sumi = 0;
            #pragma unroll
            for (int g = 0; g < 16; g++) {
                int bsum = 0;
                #pragma unroll
                for (int l = 0; l < 16; l++) bsum += (int)y256[g * 16 + l];
                sumi += bsum * (int)mn[g >> 1];
            }
            ax[8] = sumi;
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        long row_sb0 = (long)row * n_sb;
        float sums[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) sums[l] = 0.0f;
        float sumf = 0.0f;
        for (int b = 0; b < n_sb; b++) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            int* ax = myaux + (long)b * 9;
            float d = f16_bits_to_f32((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
            float dmin = f16_bits_to_f32((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
            float dact = s_is[b];
            float dd = d * dact;
            #pragma unroll
            for (int l = 0; l < 8; l++) sums[l] += dd * (float)ax[l];
            sumf -= dmin * dact * (float)ax[8];
        }
        // Same ordered reduction as q4k: 8 main lanes summed first, then + sumf.
        float smain = 0.0f;
        #pragma unroll
        for (int l = 0; l < 8; l++) smain += sums[l];
        float acc = sumf + smain;
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- Q6_K GEMV: one warp per output row, fused dequant + integer dot ---------
// Bit-identical reproduction of the validated CPU oracle `q6_k_wire_row_dot`.
// The activation is Q8_K (256-wide blocks). Weights are read STRAIGHT from the
// 210-byte GGUF wire super-block (ql[128] + qh[64] + scales(i8)[16] + d(f16)) —
// no SoA repack is needed: the oracle reads the same byte layout, and each warp
// stages the shared Q8_K activation once, so the per-row weight read is already
// the dominant DRAM stream. The 8-lane main-side split is the SAME parity anchor
// as q4k_gemv: the oracle keeps 8 f32 accumulators sums[0..8] summed over
// superblocks IN ORDER, then returns sums[0]+...+sums[7] (left-to-right). The
// weights are pre-subtracted by 32 (the oracle bakes `- 32` into the rebuilt
// signed 6-bit value), so there is NO mins term (unlike the diffusion_gemma
// kernel, which keeps weights unsigned and subtracts 32*isum_mins — a DIFFERENT
// f32 association; we must match THIS oracle, not that one).
//
// Per superblock, the oracle's main side is:
//   aux32[l] += scale[j] * y.qs[off+l] * a[off+l]   for j in 0..16, off=j*16,
//                                                    l in 0..8 then l in 8..16
// where a[256] are the rebuilt signed-6-bit weights (recombination order from
// q6_k_wire_block_dequant). Lane l (0..8) owns its own aux32 lane; lane 0 then
// replays sums[l] += (d_w * d_act) * aux32[l] per superblock, in order.
extern "C" __global__ void q6k_gemv(
    const float* __restrict__ input_scales,         // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,   // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_bytes, // raw 210-byte Q6_K wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem6[];
    signed char* s_iq = (signed char*)smem6;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem6 + (long)n_sb * 256);        // n_sb f32 staged scales
    // per-warp scratch: 8 aux32 lanes per superblock (the main-side integers).
    int* aux = (int*)(smem6 + (long)n_sb * 256 + (long)n_sb * 4); // warps*n_sb*8 int
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i]; // n_sb*256 bytes as ints
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myaux = aux + (long)warp * n_sb * 8;
    // Blocks are PADDED 210->224 at upload (pad_q6k_blocks) so ql(+0)/qh(+128)/
    // scales(+192)/d(+208) are all 16-aligned for uint4 loads.
    const int WIRE = 224;
    // Phase 1 — integer partials with ALL 32 lanes cooperating. Work unit
    // u = (super-block b = u>>2, quarter = u&3 with h = quarter>>1, s =
    // quarter&1): each unit covers l in [s*16, s*16+16) of half h, i.e. the 64
    // weights at indices h*128 + s*16 + l' + {0,32,64,96} — exactly four whole
    // 16-element scale groups (j = 8h+s+{0,2,4,6}), an exact partition of the
    // oracle's loops. Three uint4 loads per unit (ql lo/hi 16 B chunks + qh
    // 16 B chunk) replace the old per-lane a[256] local-memory rebuild with
    // scalar byte loads off the unaligned 210 B wire (half the warp also sat
    // idle whenever n_sb <= 16). Integer sums are order-free, so this split is
    // BIT-EXACT; the ordered f32 tail below is unchanged.
    long row_sb0 = (long)row * n_sb;
    int units = n_sb * 4;
    for (int u0 = 0; u0 < units; u0 += 32) {
        int u = u0 + lane;
        bool active = (u < units) && (row < rows);
        int b = u >> 2;
        int quarter = u & 3;
        int aux32[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) aux32[l] = 0;
        if (active) {
            const unsigned char* block = weight_bytes + (long)(row_sb0 + b) * WIRE;
            const signed char* y256 = s_iq + (long)b * 256;
            int h = quarter >> 1;
            int s = quarter & 1;
            int qlb = h * 64;
            int qhb = 128 + h * 32;
            int wbase = h * 128;
            // All 16 scales in one uint4 (block+192 is 16-aligned).
            uint4 scv = *reinterpret_cast<const uint4*>(block + 192);
            const signed char* sc = (const signed char*)&scv;
            uint4 qlo = *reinterpret_cast<const uint4*>(block + qlb + s * 16);
            uint4 qhiv = *reinterpret_cast<const uint4*>(block + qlb + 32 + s * 16);
            uint4 qhv = *reinterpret_cast<const uint4*>(block + qhb + s * 16);
            const unsigned char* ql_lo = (const unsigned char*)&qlo;
            const unsigned char* ql_hi = (const unsigned char*)&qhiv;
            const unsigned char* qh = (const unsigned char*)&qhv;
            int base = wbase + s * 16;
            int j0 = 8 * h + s; // scale group of the {+0} sub-range
            int s0 = (int)sc[j0];
            int s1 = (int)sc[j0 + 2];
            int s2 = (int)sc[j0 + 4];
            int s3 = (int)sc[j0 + 6];
            #pragma unroll
            for (int l = 0; l < 16; l++) {
                int albyte = (int)ql_lo[l];
                int ahbyte = (int)ql_hi[l];
                int hbyte = (int)qh[l];
                int a0 = ((albyte & 0xF) | ((hbyte & 3) << 4)) - 32;
                int a1 = ((ahbyte & 0xF) | (((hbyte >> 2) & 3) << 4)) - 32;
                int a2 = ((albyte >> 4) | (((hbyte >> 4) & 3) << 4)) - 32;
                int a3 = ((ahbyte >> 4) | (((hbyte >> 6) & 3) << 4)) - 32;
                int al = l & 7;
                aux32[al] += s0 * (int)y256[base + l] * a0;
                aux32[al] += s1 * (int)y256[base + l + 32] * a1;
                aux32[al] += s2 * (int)y256[base + l + 64] * a2;
                aux32[al] += s3 * (int)y256[base + l + 96] * a3;
            }
        }
        // Combine the 4 same-super-block lanes (quarter==0 collects 1..3);
        // cross-group shuffle reads are discarded. Integer sums — order-free.
        #pragma unroll
        for (int off = 2; off >= 1; off >>= 1) {
            #pragma unroll
            for (int l = 0; l < 8; l++)
                aux32[l] += __shfl_down_sync(0xffffffffu, aux32[l], off);
        }
        if (active && quarter == 0) {
            int* ax = myaux + (long)b * 8;
            #pragma unroll
            for (int l = 0; l < 8; l++) ax[l] = aux32[l];
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        long row_sb0 = (long)row * n_sb;
        float sums[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) sums[l] = 0.0f;
        for (int b = 0; b < n_sb; b++) {
            const unsigned char* block = weight_bytes + (long)(row_sb0 + b) * WIRE;
            unsigned short d_bits = (unsigned short)block[208]
                | ((unsigned short)block[209] << 8);
            float d = f16_bits_to_f32(d_bits) * s_is[b];
            int* ax = myaux + (long)b * 8;
            #pragma unroll
            for (int l = 0; l < 8; l++) sums[l] += d * (float)ax[l];
        }
        float acc = 0.0f;
        #pragma unroll
        for (int l = 0; l < 8; l++) acc += sums[l];
        output[row] = residual ? (output[row] + acc) : acc;
    }
}

// ---- IQ4_XS GEMV: one warp per output row, fused codebook dequant + integer dot
// Numerically matches the CPU oracle `iq4_xs_wire_row_dot` (validated to 1e-4,
// the same tolerance gate as the other resident K-quant lanes). The activation is
// Q8_K (256-wide blocks). Weights are read STRAIGHT from the 136-byte GGUF wire
// super-block: d(f16 @0) + scales_h(u16 @2) + scales_l[4] @4 + qs[128] @8. The
// 16-entry non-linear codebook `kvalues_iq4nl` maps each 4-bit index to a signed
// weight; each of the 8 sub-blocks has a 6-bit scale `ls` (low nibble in scales_l,
// high 2 bits in scales_h) biased by -32. Per super-block b: d4d8 = d_w * d_act;
// per sub-block ib: sumi = Σ kv[qs nibble]·q8 (exact integer), then the f32 term
// `d4d8 * ls * sumi` (matching the oracle's per-ib association). Each lane owns a
// strided set of super-blocks; the final warp-reduce over f32 partials is what the
// 1e-4 tolerance absorbs (integer sumi/ls are exact).
extern "C" __global__ void iq4xs_gemv(
    const float* __restrict__ input_scales,         // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,   // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_bytes, // raw 136-byte IQ4_XS wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem_iq4[];
    signed char* s_iq = (signed char*)smem_iq4;              // n_sb*256 i8 staged input
    float* s_is = (float*)(smem_iq4 + (long)n_sb * 256);     // n_sb f32 staged scales
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];     // n_sb*256 bytes as ints
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    const int KV[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };
    const int WIRE = 136;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    long row_sb0 = (long)row * n_sb;
    float acc = 0.0f;
    for (int b = lane; b < n_sb; b += 32) {
        const unsigned char* block = weight_bytes + (long)(row_sb0 + b) * WIRE;
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        float d4d8 = f16_bits_to_f32(d_bits) * s_is[b];
        unsigned short sh = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        const unsigned char* sl = block + 4;
        const unsigned char* qs = block + 8;
        const signed char* y256 = s_iq + (long)b * 256;
        #pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            int low = (sl[ib >> 1] >> (4 * (ib & 1))) & 0xF;
            int high = (sh >> (2 * ib)) & 0x3;
            int ls = (low | (high << 4)) - 32;
            const unsigned char* q = qs + ib * 16;
            const signed char* yy = y256 + ib * 32;
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int byte = q[j];
                sumi += KV[byte & 0xF] * (int)yy[j];
                sumi += KV[byte >> 4] * (int)yy[j + 16];
            }
            acc += d4d8 * (float)ls * (float)sumi;
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        output[row] = residual ? (output[row] + acc) : acc;
}

// ---- Q2_K GEMV: one warp per output row, fused dequant + integer dot ---------
// Bit-identical reproduction of the CPU oracle `q2_k_wire_row_dot`
// (ggml_vec_dot_q2_K_q8_K generic shape). The activation is Q8_K (256-wide
// blocks). Weights are read STRAIGHT from the 84-byte GGUF wire super-block:
// scales[16] (each byte: low nibble = quant scale, high nibble = min scale),
// qs[64] (2-bit quants), d(f16), dmin(f16). Unlike q4k/q6k, the reference keeps a
// SINGLE integer accumulator `isum` per super-block (no 8-lane f32 split), so the
// per-super-block term is just `dall*isum - dmin*summs`, summed over super-blocks
// IN ORDER. Each lane owns whole super-blocks and stashes (isum, summs) in shared;
// lane 0 replays the ordered f32 reduction so the result matches the oracle.
//
// PARITY ANCHOR: isum and summs are integers (exact, order-free). Lane 0 forms
// `dall*(float)isum - dmin*(float)summs` per super-block (subtraction first, the
// oracle's `dall*isum - dmin*summs`) and accumulates left-to-right — the only
// f32-order-sensitive step, reproduced exactly.
extern "C" __global__ void q2k_gemv(
    const float* __restrict__ input_scales,         // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,   // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_bytes, // RAW 84-byte Q2_K wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem2[];
    signed char* s_iq = (signed char*)smem2;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem2 + (long)n_sb * 256);        // n_sb f32 staged scales
    // per-warp scratch: 2 ints (isum, summs) per superblock.
    int* acc = (int*)(smem2 + (long)n_sb * 256 + (long)n_sb * 4); // warps*n_sb*2 int
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i]; // n_sb*256 bytes as ints
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 84;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myacc = acc + (long)warp * n_sb * 2;
    if (row < rows) {
        long row_sb0 = (long)row * n_sb;
        for (int b = lane; b < n_sb; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            const signed char* y256 = s_iq + (long)b * 256;  // staged activation
            const unsigned char* sc = blk;          // scales[16]
            const unsigned char* qs = blk + 16;     // qs[64] (2-bit quants)
            // Mins side: per-16 activation sums (bsums) times the high-nibble min
            // scale of each sub-block, summed over the 16 sub-blocks (oracle order).
            int summs = 0;
            for (int j = 0; j < 16; j++) {
                int bsum = 0;
                for (int l = 0; l < 16; l++) bsum += (int)y256[j * 16 + l];
                summs += bsum * (int)(sc[j] >> 4);
            }
            // Main side: 2 halves x 4 groups; each group reuses the same 32 qs bytes
            // at shift 2*group, split into a low (l<16) and high (l>=16) sub-block,
            // each with its own low-nibble quant scale. q8 advances 32 per group.
            int isum = 0;
            int is = 0;
            for (int k = 0; k < 2; k++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int dlo = (int)(sc[is++] & 0xF);
                    int isuml = 0;
                    for (int l = 0; l < 16; l++)
                        isuml += (int)y256[k * 128 + j * 32 + l]
                               * (int)((qs[k * 32 + l] >> shift) & 3);
                    isum += dlo * isuml;
                    int dhi = (int)(sc[is++] & 0xF);
                    isuml = 0;
                    for (int l = 0; l < 16; l++)
                        isuml += (int)y256[k * 128 + j * 32 + 16 + l]
                               * (int)((qs[k * 32 + 16 + l] >> shift) & 3);
                    isum += dhi * isuml;
                    shift += 2;
                }
            }
            myacc[b * 2 + 0] = isum;
            myacc[b * 2 + 1] = summs;
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        long row_sb0 = (long)row * n_sb;
        float sumf = 0.0f;
        for (int b = 0; b < n_sb; b++) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            float d = f16_bits_to_f32((unsigned short)blk[80] | ((unsigned short)blk[81] << 8));
            float dmin = f16_bits_to_f32((unsigned short)blk[82] | ((unsigned short)blk[83] << 8));
            float dact = s_is[b];
            float dall = d * dact;
            float dminx = dmin * dact;
            sumf += dall * (float)myacc[b * 2 + 0] - dminx * (float)myacc[b * 2 + 1];
        }
        output[row] = residual ? (output[row] + sumf) : sumf;
    }
}

// ---- Q3_K GEMV: one warp per output row, fused dequant + integer dot ---------
// Bit-identical reproduction of the CPU oracle `q3_k_wire_row_dot`
// (ggml_vec_dot_q3_K_q8_K). 110-byte wire: hmask[32] (high bit of each 3-bit
// quant), qs[64] (low 2 bits), scales[12] (16 signed 6-bit scales, kmask-packed),
// d(f16). NO mins: the 3-bit quant is `((qs>>shift)&3) - (hmask_bit ? 0 : 4)` and
// dequant = d*(scale-32)*value. Each super-block contributes a single d*isum
// (isum = Σ_sb (scale-32)*Σ q8*value), summed IN ORDER. Each lane owns whole
// super-blocks and stashes isum (1 int); lane 0 replays the ordered f32 reduction.
extern "C" __global__ void q3k_gemv(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, // RAW 110-byte Q3_K wire, row-major
    int rows, int n_sb, float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem3[];
    signed char* s_iq = (signed char*)smem3;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem3 + (long)n_sb * 256);        // n_sb f32 staged scales
    int* acc = (int*)(smem3 + (long)n_sb * 256 + (long)n_sb * 4); // warps*n_sb*1 int
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
    for (int i = tid; i < n_sb; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 110;
    const unsigned int KMASK1 = 0x03030303u, KMASK2 = 0x0f0f0f0fu;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myacc = acc + (long)warp * n_sb;
    if (row < rows) {
        long row_sb0 = (long)row * n_sb;
        for (int b = lane; b < n_sb; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            const signed char* y256 = s_iq + (long)b * 256;
            const unsigned char* hmask = blk;          // hmask[32]
            const unsigned char* qs = blk + 32;        // qs[64]
            const unsigned char* sr = blk + 96;        // scales[12]
            // Expand the 16 signed 6-bit scales (kmask scheme).
            unsigned int a0 = (unsigned int)sr[0] | ((unsigned int)sr[1] << 8) | ((unsigned int)sr[2] << 16) | ((unsigned int)sr[3] << 24);
            unsigned int a1 = (unsigned int)sr[4] | ((unsigned int)sr[5] << 8) | ((unsigned int)sr[6] << 16) | ((unsigned int)sr[7] << 24);
            unsigned int a2 = (unsigned int)sr[8] | ((unsigned int)sr[9] << 8) | ((unsigned int)sr[10] << 16) | ((unsigned int)sr[11] << 24);
            unsigned int tmp = a2;
            unsigned int e2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            unsigned int e3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            unsigned int e0 = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
            unsigned int e1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            signed char sc[16];
            sc[0]=(signed char)(e0&0xff); sc[1]=(signed char)((e0>>8)&0xff); sc[2]=(signed char)((e0>>16)&0xff); sc[3]=(signed char)((e0>>24)&0xff);
            sc[4]=(signed char)(e1&0xff); sc[5]=(signed char)((e1>>8)&0xff); sc[6]=(signed char)((e1>>16)&0xff); sc[7]=(signed char)((e1>>24)&0xff);
            sc[8]=(signed char)(e2&0xff); sc[9]=(signed char)((e2>>8)&0xff); sc[10]=(signed char)((e2>>16)&0xff); sc[11]=(signed char)((e2>>24)&0xff);
            sc[12]=(signed char)(e3&0xff); sc[13]=(signed char)((e3>>8)&0xff); sc[14]=(signed char)((e3>>16)&0xff); sc[15]=(signed char)((e3>>24)&0xff);
            int isum = 0;
            int sb = 0;
            unsigned int high_mask = 1u;
            for (int half = 0; half < 2; half++) {
                int value_base = half * 32;
                int q8_base = half * 128;
                int shift = 0;
                for (int g = 0; g < 4; g++) {
                    int sc_lo = (int)sc[sb] - 32; sb++;
                    int dot = 0;
                    for (int l = 0; l < 16; l++) {
                        int hb = (hmask[l] & high_mask) ? 0 : 4;
                        int v = (int)((qs[value_base + l] >> shift) & 3) - hb;
                        dot += (int)y256[q8_base + g * 32 + l] * v;
                    }
                    isum += sc_lo * dot;
                    int sc_hi = (int)sc[sb] - 32; sb++;
                    int dot2 = 0;
                    for (int l = 0; l < 16; l++) {
                        int hb = (hmask[16 + l] & high_mask) ? 0 : 4;
                        int v = (int)((qs[value_base + 16 + l] >> shift) & 3) - hb;
                        dot2 += (int)y256[q8_base + g * 32 + 16 + l] * v;
                    }
                    isum += sc_hi * dot2;
                    shift += 2;
                    high_mask <<= 1;
                }
            }
            myacc[b] = isum;
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        long row_sb0 = (long)row * n_sb;
        float sumf = 0.0f;
        for (int b = 0; b < n_sb; b++) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + b) * WIRE;
            float d = f16_bits_to_f32((unsigned short)blk[108] | ((unsigned short)blk[109] << 8));
            float dact = s_is[b];
            sumf += d * dact * (float)myacc[b];
        }
        output[row] = residual ? (output[row] + sumf) : sumf;
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

// ---- Q8_K activation quantize (256-wide blocks; K-quant input format) -------
// Bit-exact port of inference.rs `quantize_q8_k_blocks` + `nearest_int_reference`:
//   amax over abs but `max` is the SIGNED value at the abs-max position; iscale =
//   -127/max; q = nearest_int(iscale*v) clamped to <=127 (no low clamp — matches
//   the reference, which only `.min(127)`s); d = 1/iscale. The reference's
//   nearest_int adds 1.5*2^23 and masks the mantissa (round-to-nearest-EVEN), not
//   rintf — reproduced here bit-for-bit so the resident Q4_K/Q6_K dot matches the
//   CPU oracle token-for-token. One thread per 256-block; `n_sb` super-blocks.
__device__ __forceinline__ int nearest_int_ref(float fval) {
    float v = fval + 12582912.0f; // 1.5 * 2^23
    return (int)(__float_as_uint(v) & 0x007fffffu) - 0x00400000;
}
__device__ __forceinline__ void quant_q8k_block(
    const float* xb, signed char* qb, float* scale_out
) {
    float amax = 0.0f, maxv = 0.0f;
    for (int j = 0; j < 256; j++) {
        float a = fabsf(xb[j]);
        if (a > amax) { amax = a; maxv = xb[j]; }
    }
    if (amax == 0.0f) {
        *scale_out = 0.0f;
        for (int j = 0; j < 256; j++) qb[j] = 0;
        return;
    }
    float iscale = -127.0f / maxv;
    for (int j = 0; j < 256; j++) {
        int q = nearest_int_ref(iscale * xb[j]);
        if (q > 127) q = 127;
        qb[j] = (signed char)q;
    }
    *scale_out = 1.0f / iscale;
}
// Parallel Q8_K super-block quantize: one BLOCK of 256 threads per super-block,
// one element per thread. The amax scan is a first-max-wins tree reduction
// (larger |v| wins; on an exact |v| tie the SMALLER index wins), which
// reproduces the serial quant_q8k_block scan BIT-EXACTLY — fabsf and the
// comparisons involve no rounding, so the reduction order is free. Replaces
// the one-thread-per-super-block shape whose serial 256-element loops (and,
// in silu_mul's case, a 1 KB local vals[] spill) left the kernels ~10-20x off
// their roofline and latency-bound at ~46-78 us per call.
__device__ __forceinline__ void quant_q8k_block_parallel(
    float v, int tid, float* r_amax, float* r_maxv, int* r_idx,
    signed char* __restrict__ qb, float* __restrict__ scale_out
) {
    r_amax[tid] = fabsf(v);
    r_maxv[tid] = v;
    r_idx[tid] = tid;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            float oa = r_amax[tid + off];
            if (oa > r_amax[tid] || (oa == r_amax[tid] && r_idx[tid + off] < r_idx[tid])) {
                r_amax[tid] = oa;
                r_maxv[tid] = r_maxv[tid + off];
                r_idx[tid] = r_idx[tid + off];
            }
        }
        __syncthreads();
    }
    float amax = r_amax[0];
    if (amax == 0.0f) {
        qb[tid] = 0;
        if (tid == 0) *scale_out = 0.0f;
        return;
    }
    float iscale = -127.0f / r_maxv[0];
    int q = nearest_int_ref(iscale * v);
    if (q > 127) q = 127;
    qb[tid] = (signed char)q;
    if (tid == 0) *scale_out = 1.0f / iscale;
}
// Standalone Q8_K quantize (used for the attention-output activation before the
// O projection of a K-quant layer). One block per 256-block, parallel quantize.
extern "C" __global__ void quantize_q8k(
    const float* __restrict__ x, signed char* __restrict__ quants,
    float* __restrict__ scales, int n_sb
) {
    __shared__ float r_amax[256];
    __shared__ float r_maxv[256];
    __shared__ int r_idx[256];
    int b = blockIdx.x;
    if (b >= n_sb) return;
    int tid = threadIdx.x;
    float v = x[(long)b * 256 + tid];
    quant_q8k_block_parallel(v, tid, r_amax, r_maxv, r_idx,
                             quants + (long)b * 256, scales + b);
}
// Fused RMS-norm + Q8_K quantize (K-quant analog of rms_norm_quantize). One block
// stages the row in shared, thread 0 does the in-order sum-of-squares (bit-identical
// to rms_norm_f32), every thread applies norm*weight back into shared, then each
// thread quantizes 256-wide blocks straight from shared.
extern "C" __global__ void rms_norm_quantize_q8k(
    const float* __restrict__ x, const float* __restrict__ weight,
    signed char* __restrict__ quants, float* __restrict__ scales, int n, float eps
) {
    // One block per 256-element super-block. The x·x sum must stay the CPU's
    // serial f32 chain (the parity anchor — a tree reduce would reassociate),
    // so EVERY block redundantly recomputes it on its thread 0: the ~n serial
    // FMAs run concurrently across resident blocks, so wall time is one chain
    // (~12 us at n=4096) instead of one chain PLUS a single block serially
    // quantizing every super-block alone (measured 46 us at grid(1)).
    extern __shared__ float xsk[]; // n floats
    __shared__ float s_scale;
    __shared__ float r_amax[256];
    __shared__ float r_maxv[256];
    __shared__ int r_idx[256];
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) xsk[i] = x[i];
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += xsk[i] * xsk[i]; // CPU-order serial sum
        s_scale = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    int b = blockIdx.x;
    long base = (long)b << 8;
    float v = xsk[base + tid] * s_scale * weight[base + tid];
    quant_q8k_block_parallel(v, tid, r_amax, r_maxv, r_idx, quants + base, scales + b);
}
// Fused SiLU(gate)*up + Q8_K quantize (K-quant analog of silu_mul_quantize). One
// thread per 256-block: compute silu*up for the block's 256 elements into a local
// buffer (bit-identical to silu_mul), then quantize them to Q8_K straight to the
// down-projection's K-quant input.
extern "C" __global__ void silu_mul_quantize_q8k(
    const float* __restrict__ gate, const float* __restrict__ up,
    signed char* __restrict__ quants, float* __restrict__ scales, int n_sb
) {
    __shared__ float r_amax[256];
    __shared__ float r_maxv[256];
    __shared__ int r_idx[256];
    int b = blockIdx.x;
    if (b >= n_sb) return;
    int tid = threadIdx.x;
    long i = (long)b * 256 + tid;
    float g = gate[i];
    float v = (g / (1.0f + expf(-g))) * up[i]; // per-element — bit-identical
    quant_q8k_block_parallel(v, tid, r_amax, r_maxv, r_idx,
                             quants + (long)b * 256, scales + b);
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
        const unsigned short* scales =
            reinterpret_cast<const unsigned short*>(weight_bytes + total_blocks * 32);
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            float w_scale = f16_bits_to_f32(scales[row_block0 + b]);
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

// ---- Batched Q4_K GEMM: K Q8_K inputs against M weight rows ----------------
// Mirrors q4k_gemv's integer decomposition and ordered f32 tail, but keeps each
// 144-byte weight super-block in registers while applying it to every token in
// the chunk. Four full Q8_K activation rows are staged once per thread block:
// for the 3B FFN contraction this is 32.5 KiB, leaving enough shared memory for
// three warps' ordered integer partials under the portable 46 KiB budget.
extern "C" __global__ void q4k_gemm_batched(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int n_sb, int k_tokens, float* __restrict__ output
) {
    extern __shared__ unsigned char smem4b[];
    signed char* s_iq = (signed char*)smem4b;
    float* s_is = (float*)(smem4b + (long)k_tokens * n_sb * 256);
    int* aux = (int*)(smem4b + (long)k_tokens * n_sb * 260);
    int tid = threadIdx.x;
    // Stage each token's activation in the same stride-8 swizzle used by the
    // resident Q4_K weights, so the hot dot loop stays at two dp4a operations
    // per aux lane rather than scattered scalar global loads.
    for (int i = tid; i < k_tokens * n_sb * 256; i += blockDim.x) {
        int block = i >> 8;
        int p = i & 255;
        int group = p >> 5;
        int pg = p & 31;
        int l = pg & 7;
        int kk = pg >> 3;
        s_iq[(long)block * 256 + group * 32 + l * 4 + kk] = input_quants[i];
    }
    for (int i = tid; i < k_tokens * n_sb; i += blockDim.x)
        s_is[i] = input_scales[i];
    __syncthreads();

    const unsigned int KMASK1 = 0x3f3f3f3fu;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    const unsigned int KMASK3 = 0x03030303u;
    const int WIRE = 144;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myaux = aux + (long)warp * k_tokens * n_sb * 9;
    long row_sb0 = (long)row * n_sb;
    int units = n_sb * 4;

    for (int u0 = 0; u0 < units; u0 += 32) {
        int u = u0 + lane;
        bool active = (u < units) && (row < rows);
        int sb = u >> 2;
        int g = u & 3;
        int slo = 0, shi = 0, mnlo = 0, mnhi = 0;
        int qwords[8];
        #pragma unroll
        for (int l = 0; l < 8; l++) qwords[l] = 0;
        if (active) {
            const unsigned char* blk = weight_bytes + (long)(row_sb0 + sb) * WIRE;
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            unsigned int u0w = hdr.y;
            unsigned int u1 = hdr.z;
            unsigned int u2 = hdr.w;
            unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            unsigned int uaux = u1 & KMASK1;
            u1 = (u2 & KMASK2) | (((u0w >> 6) & KMASK3) << 4);
            u2 = uaux;
            u0w &= KMASK1;
            unsigned char sc[8], mn[8];
            sc[0] = u0w & 0xff; sc[1] = (u0w >> 8) & 0xff;
            sc[2] = (u0w >> 16) & 0xff; sc[3] = (u0w >> 24) & 0xff;
            sc[4] = u1 & 0xff; sc[5] = (u1 >> 8) & 0xff;
            sc[6] = (u1 >> 16) & 0xff; sc[7] = (u1 >> 24) & 0xff;
            mn[0] = u2 & 0xff; mn[1] = (u2 >> 8) & 0xff;
            mn[2] = (u2 >> 16) & 0xff; mn[3] = (u2 >> 24) & 0xff;
            mn[4] = u3 & 0xff; mn[5] = (u3 >> 8) & 0xff;
            mn[6] = (u3 >> 16) & 0xff; mn[7] = (u3 >> 24) & 0xff;
            slo = (int)sc[2 * g];
            shi = (int)sc[2 * g + 1];
            mnlo = (int)mn[2 * g];
            mnhi = (int)mn[2 * g + 1];
            const int* qw = reinterpret_cast<const int*>(blk + 16 + g * 32);
            #pragma unroll
            for (int l = 0; l < 8; l++) qwords[l] = qw[l];
        }

        for (int t = 0; t < k_tokens; t++) {
            int partial[8];
            #pragma unroll
            for (int l = 0; l < 8; l++) partial[l] = 0;
            int sumi = 0;
            if (active) {
                const signed char* y256 =
                    s_iq + ((long)t * n_sb + sb) * 256;
                const int* ylo =
                    reinterpret_cast<const int*>(y256 + (2 * g) * 32);
                const int* yhi =
                    reinterpret_cast<const int*>(y256 + (2 * g + 1) * 32);
                int sum_lo = 0, sum_hi = 0;
                #pragma unroll
                for (int l = 0; l < 8; l++) {
                    int q = qwords[l];
                    int yl = ylo[l];
                    int yh = yhi[l];
                    partial[l] +=
                        slo * __dp4a(q & 0x0F0F0F0F, yl, 0)
                        + shi * __dp4a((q >> 4) & 0x0F0F0F0F, yh, 0);
                    sum_lo = __dp4a(yl, 0x01010101, sum_lo);
                    sum_hi = __dp4a(yh, 0x01010101, sum_hi);
                }
                sumi = mnlo * sum_lo + mnhi * sum_hi;
            }
            #pragma unroll
            for (int off = 2; off >= 1; off >>= 1) {
                #pragma unroll
                for (int l = 0; l < 8; l++)
                    partial[l] += __shfl_down_sync(0xffffffffu, partial[l], off);
                sumi += __shfl_down_sync(0xffffffffu, sumi, off);
            }
            if (active && g == 0) {
                int* dst = myaux + ((long)t * n_sb + sb) * 9;
                #pragma unroll
                for (int l = 0; l < 8; l++) dst[l] = partial[l];
                dst[8] = sumi;
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float sums[8];
            #pragma unroll
            for (int l = 0; l < 8; l++) sums[l] = 0.0f;
            float sumf = 0.0f;
            for (int sb = 0; sb < n_sb; sb++) {
                const unsigned char* blk =
                    weight_bytes + (long)(row_sb0 + sb) * WIRE;
                float d = f16_bits_to_f32(
                    (unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
                float dmin = f16_bits_to_f32(
                    (unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
                float dact = s_is[(long)t * n_sb + sb];
                float dd = d * dact;
                int* src = myaux + ((long)t * n_sb + sb) * 9;
                #pragma unroll
                for (int l = 0; l < 8; l++) sums[l] += dd * (float)src[l];
                sumf -= dmin * dact * (float)src[8];
            }
            float smain = 0.0f;
            #pragma unroll
            for (int l = 0; l < 8; l++) smain += sums[l];
            output[(long)t * rows + row] = sumf + smain;
        }
    }
}

// ---- Batched Q6_K GEMM: K Q8_K inputs against M weight rows ----------------
// Same strategy and parity contract as q4k_gemm_batched. Q8_K activations are
// staged in natural order; the padded 224-byte weight block is loaded once per
// row/super-block work unit and reused across every token.
extern "C" __global__ void q6k_gemm_batched(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int n_sb, int k_tokens, float* __restrict__ output
) {
    extern __shared__ unsigned char smem6b[];
    signed char* s_iq = (signed char*)smem6b;
    float* s_is = (float*)(smem6b + (long)k_tokens * n_sb * 256);
    int* aux = (int*)(smem6b + (long)k_tokens * n_sb * 260);
    int tid = threadIdx.x;
    for (int i = tid; i < k_tokens * n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
    for (int i = tid; i < k_tokens * n_sb; i += blockDim.x)
        s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 224;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int* myaux = aux + (long)warp * k_tokens * n_sb * 8;
    long row_sb0 = (long)row * n_sb;
    int units = n_sb * 4;

    for (int u0 = 0; u0 < units; u0 += 32) {
        int u = u0 + lane;
        bool active = (u < units) && (row < rows);
        int sb = u >> 2;
        int quarter = u & 3;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int base = 0;
        uint4 qlo = make_uint4(0, 0, 0, 0);
        uint4 qhi = make_uint4(0, 0, 0, 0);
        uint4 qhv = make_uint4(0, 0, 0, 0);
        if (active) {
            const unsigned char* blk =
                weight_bytes + (long)(row_sb0 + sb) * WIRE;
            int h = quarter >> 1;
            int ss = quarter & 1;
            int qlb = h * 64;
            int qhb = 128 + h * 32;
            int wbase = h * 128;
            uint4 scv = *reinterpret_cast<const uint4*>(blk + 192);
            const signed char* sc = (const signed char*)&scv;
            qlo = *reinterpret_cast<const uint4*>(blk + qlb + ss * 16);
            qhi = *reinterpret_cast<const uint4*>(blk + qlb + 32 + ss * 16);
            qhv = *reinterpret_cast<const uint4*>(blk + qhb + ss * 16);
            base = wbase + ss * 16;
            int j0 = 8 * h + ss;
            s0 = (int)sc[j0];
            s1 = (int)sc[j0 + 2];
            s2 = (int)sc[j0 + 4];
            s3 = (int)sc[j0 + 6];
        }
        const unsigned char* ql_lo = (const unsigned char*)&qlo;
        const unsigned char* ql_hi = (const unsigned char*)&qhi;
        const unsigned char* qh = (const unsigned char*)&qhv;

        for (int t = 0; t < k_tokens; t++) {
            int partial[8];
            #pragma unroll
            for (int l = 0; l < 8; l++) partial[l] = 0;
            if (active) {
                const signed char* y256 =
                    s_iq + ((long)t * n_sb + sb) * 256;
                #pragma unroll
                for (int l = 0; l < 16; l++) {
                    int albyte = (int)ql_lo[l];
                    int ahbyte = (int)ql_hi[l];
                    int hbyte = (int)qh[l];
                    int a0 = ((albyte & 0xF) | ((hbyte & 3) << 4)) - 32;
                    int a1 = ((ahbyte & 0xF) | (((hbyte >> 2) & 3) << 4)) - 32;
                    int a2 = ((albyte >> 4) | (((hbyte >> 4) & 3) << 4)) - 32;
                    int a3 = ((ahbyte >> 4) | (((hbyte >> 6) & 3) << 4)) - 32;
                    int al = l & 7;
                    partial[al] += s0 * (int)y256[base + l] * a0;
                    partial[al] += s1 * (int)y256[base + l + 32] * a1;
                    partial[al] += s2 * (int)y256[base + l + 64] * a2;
                    partial[al] += s3 * (int)y256[base + l + 96] * a3;
                }
            }
            #pragma unroll
            for (int off = 2; off >= 1; off >>= 1) {
                #pragma unroll
                for (int l = 0; l < 8; l++)
                    partial[l] += __shfl_down_sync(0xffffffffu, partial[l], off);
            }
            if (active && quarter == 0) {
                int* dst = myaux + ((long)t * n_sb + sb) * 8;
                #pragma unroll
                for (int l = 0; l < 8; l++) dst[l] = partial[l];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float sums[8];
            #pragma unroll
            for (int l = 0; l < 8; l++) sums[l] = 0.0f;
            for (int sb = 0; sb < n_sb; sb++) {
                const unsigned char* blk =
                    weight_bytes + (long)(row_sb0 + sb) * WIRE;
                unsigned short d_bits = (unsigned short)blk[208]
                    | ((unsigned short)blk[209] << 8);
                float d = f16_bits_to_f32(d_bits)
                    * s_is[(long)t * n_sb + sb];
                int* src = myaux + ((long)t * n_sb + sb) * 8;
                #pragma unroll
                for (int l = 0; l < 8; l++) sums[l] += d * (float)src[l];
            }
            float acc = 0.0f;
            #pragma unroll
            for (int l = 0; l < 8; l++) acc += sums[l];
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

    // scores. SIROCCO Lane K: widen the f16 K read to uint4 (128-bit = 8 keys/load) when the
    // per-position row is 16B-aligned (head_dim % 8 == 0 => kp = kbase + p*head_dim is aligned
    // for all p). The dot is accumulated in the SAME d-order as the scalar loop, so the f32 sum
    // is BYTE-IDENTICAL — decode stays bit-exact vs the spec-verify kernels (attention_batched/
    // tree) and the splitk_spec_verify_bit_identical gate. Non-8 head_dim falls back to scalar.
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    for (int p = tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d + 0] * f16_bits_to_f32(k8[0]);
            dot += qsh[d + 1] * f16_bits_to_f32(k8[1]);
            dot += qsh[d + 2] * f16_bits_to_f32(k8[2]);
            dot += qsh[d + 3] * f16_bits_to_f32(k8[3]);
            dot += qsh[d + 4] * f16_bits_to_f32(k8[4]);
            dot += qsh[d + 5] * f16_bits_to_f32(k8[5]);
            dot += qsh[d + 6] * f16_bits_to_f32(k8[6]);
            dot += qsh[d + 7] * f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
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
    // NOTE: attention_batched / attention_tree_batched (spec-decode verify) now use the IDENTICAL
    // G-group reorder at or below SPLITK_THRESHOLD, and emulate the split-K reduction above it
    // (gated by splitk_active), so decode==verify stays bit-exact across the threshold for greedy spec.
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

// ---- Sliding-window attention decode (gemma4 sliding layers) ---------------
// Identical to attention_decode but attends only the last `window` keys:
//   start = (window > 0 && position_count > window) ? position_count - window : 0
// then keys [start, position_count). window <= 0 reproduces full-causal
// attention_decode exactly (so the non-sliding gemma4 layers / any caller can
// share this kernel). Same online softmax + FP-reassociated weighted-V
// (token-parity, not bit-identical) shape as attention_decode.
extern "C" __global__ void attention_decode_sw(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale, int window
) {
    int position_count = position_ptr[0] + 1;
    int start = (window > 0 && position_count > window) ? (position_count - window) : 0;
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ float shared_sw[];
    int tid = threadIdx.x;
    int G = blockDim.x / head_dim;
    float* qsh = shared_sw;
    float* vpart = shared_sw + head_dim;
    float* scores = shared_sw + head_dim + (long)G * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;  // uint4 K-read when row is 16B-aligned
    for (int p = start + tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d + 0] * f16_bits_to_f32(k8[0]); dot += qsh[d + 1] * f16_bits_to_f32(k8[1]);
            dot += qsh[d + 2] * f16_bits_to_f32(k8[2]); dot += qsh[d + 3] * f16_bits_to_f32(k8[3]);
            dot += qsh[d + 4] * f16_bits_to_f32(k8[4]); dot += qsh[d + 5] * f16_bits_to_f32(k8[5]);
            dot += qsh[d + 6] * f16_bits_to_f32(k8[6]); dot += qsh[d + 7] * f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        scores[p] = dot * scale;
    }
    __syncthreads();

    __shared__ float s_max_sw, s_sum_sw;
    if (tid == 0) {
        float m = scores[start];
        for (int p = start + 1; p < position_count; p++) if (scores[p] > m) m = scores[p];
        s_max_sw = m;
    }
    __syncthreads();
    for (int p = start + tid; p < position_count; p += blockDim.x) scores[p] = expf(scores[p] - s_max_sw);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = start; p < position_count; p++) sum += scores[p];
        s_sum_sw = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum_sw;
    int gid = tid / head_dim;
    int did = tid % head_dim;
    int active = position_count - start;
    int p_lo = start + (int)((long)gid * active / G);
    int p_hi = start + (int)((long)(gid + 1) * active / G);
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

// ---- Gemma GeGLU: out[i] = gelu_tanh(gate[i]) * up[i] ---------------------
// gelu_pytorch_tanh: 0.5*x*(1 + tanh(0.79788456*(x + 0.044715*x^3))). Same
// constants and left-to-right f32 order as the CPU oracle
// inference::gemma4::gelu_tanh; only tanhf's transcendental last-bit rounding
// differs (validated to tolerance, not bit-exact). --fmad=false keeps the
// polynomial unfused so the non-transcendental part matches.
extern "C" __global__ void geglu_mul(
    const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = gate[i];
    float inner = 0.79788456f * (x + 0.044715f * x * x * x);
    float gv = 0.5f * x * (1.0f + tanhf(inner));
    out[i] = gv * up[i];
}

// ---- Gemma final-logit soft-cap (in place): x = cap*tanh(x/cap) -----------
// Mirrors inference::gemma4::soft_cap_in_place (cap = 30 for Gemma 4). The
// caller passes a finite, positive cap (disabled-cap is handled host-side).
extern "C" __global__ void soft_cap(float* __restrict__ x, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = cap * tanhf(x[i] / cap);
}

// ---- f32 GEMV: out[o] = sum_i W[o*in_dim + i] * x[i] (row-major, out-major) ----
// For gemma4's small f32 PLE matrices (ple_inp_gate, ple_proj). One thread per
// output row, sequential per-row sum — bit-identical to the CPU f32_matvec order.
extern "C" __global__ void f32_gemv(
    const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ out,
    int in_dim, int out_dim
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    const float* row = w + (long)o * in_dim;
    float acc = 0.0f;
    for (int i = 0; i < in_dim; i++) acc += row[i] * x[i];
    out[o] = acc;
}

// ---- Scalar scale (in place): x[i] *= s (gemma4 PLE ple_output_scale) --------
extern "C" __global__ void scale_f32(float* __restrict__ x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// ---- Residual add: acc[i] += add[i] ---------------------------------------
extern "C" __global__ void residual_add(float* __restrict__ acc, const float* __restrict__ add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += add[i];
}

// ---- Scaled axpy: acc[i] += y[i] * scale ----------------------------------
// SSER (M3) on-device MoE accumulation: each cached expert's down-GEMV output
// `y` is folded into the layer's device `moe_acc` scaled by its routing weight,
// so a k-hit layer costs one dtoh + one sync total instead of k of each. The
// f32 mul-then-add matches the CPU host accumulate `*a += yv * scale` (with
// --fmad=false the compiler cannot fuse it, so the rounding is identical).
extern "C" __global__ void scaled_axpy(
    float* __restrict__ acc, const float* __restrict__ y, float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += y[i] * scale;
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
    int q_per_token, int k_tokens, int splitk_active
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
    // SIROCCO prefill lever: uint4 (128-bit, 8 keys/load) f16 K read, same d-order accumulation ->
    // BYTE-IDENTICAL (fmad=false), mirroring attention_decode / attn_sk_scores (win #1). This is the
    // resident-prefill + spec-verify attention (the O(n^2) dominant term of long-context prompt
    // processing); the scalar read never got #1's treatment. splitk_spec_verify_bit_identical still
    // passes (uint4 == scalar bitwise, already proven by #1). Non-mult-of-8 head_dim -> scalar tail.
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    for (int p = tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d + 0] * f16_bits_to_f32(k8[0]); dot += qsh[d + 1] * f16_bits_to_f32(k8[1]);
            dot += qsh[d + 2] * f16_bits_to_f32(k8[2]); dot += qsh[d + 3] * f16_bits_to_f32(k8[3]);
            dot += qsh[d + 4] * f16_bits_to_f32(k8[4]); dot += qsh[d + 5] * f16_bits_to_f32(k8[5]);
            dot += qsh[d + 6] * f16_bits_to_f32(k8[6]); dot += qsh[d + 7] * f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
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
    // Denominator + weighted-V. Spec-verify must match WHICHEVER reduction plain greedy decode
    // would use for this token's position_count, or a near-tie argmax can flip (spec != greedy).
    // Above SPLITK_THRESHOLD (512) on the LIVE (non-graph-captured) decode path, plain decode uses
    // the split-K reduction (launch_attention_splitk); at/below it (and under graph capture) it
    // uses the G-group attention_decode. `splitk_active` (host = !cuda_graphs_enabled) selects the
    // regime; the per-token threshold is tested HERE so a verify batch straddling 512 picks the
    // right path per token (each (t,head) block has its own position_count). Both branches are
    // block-uniform -> no warp divergence.
    if (splitk_active && position_count > 512) {   // 512 == SPLITK_THRESHOLD
        // Split-K EMULATION: bit-identical to attn_sk_partial + attn_sk_combine. n_splits MUST equal
        // Rust's position_count.div_ceil(256).clamp(2, SPLITK_MAX) (launch_attention_splitk).
        int n_splits = (position_count + 255) / 256;
        if (n_splits < 2) n_splits = 2;
        if (n_splits > 32) n_splits = 32;          // SPLITK_MAX (mirror src const)
        if (tid == 0) {
            float total = 0.0f;                    // denom: per-chunk fresh-0 p-sum, then sp-order combine
            for (int sp = 0; sp < n_splits; sp++) {
                int p_lo = (int)((long)sp * position_count / n_splits);
                int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
                float ls = 0.0f;
                for (int p = p_lo; p < p_hi; p++) ls += scores[p];
                total += ls;
            }
            s_sum = total;
        }
        __syncthreads();
        float inv = 1.0f / s_sum;
        for (int did = tid; did < head_dim; did += blockDim.x) {
            float acc = 0.0f;                      // weighted-V: per-chunk UNNORMALIZED p-sum, sp-order, /s once
            for (int sp = 0; sp < n_splits; sp++) {
                int p_lo = (int)((long)sp * position_count / n_splits);
                int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
                float a = 0.0f;
                for (int p = p_lo; p < p_hi; p++)
                    a += scores[p] * f16_bits_to_f32(vbase[(long)p * head_dim + did]);
                acc += a;
            }
            out[(long)t * q_per_token + (long)head * head_dim + did] = acc * inv;
        }
    } else {
        // G-group (<= SPLITK_THRESHOLD, or graph-captured decode): IDENTICAL reorder to
        // attention_decode (launch_attention: G = clamp(ceil(pc/head_dim), 1, 1024/head_dim));
        // contiguous p-split + g-order combine match exactly.
        if (tid == 0) {
            float sum = 0.0f;
            for (int p = 0; p < position_count; p++) sum += scores[p];
            s_sum = sum;
        }
        __syncthreads();
        float inv = 1.0f / s_sum;
        int max_groups = 1024 / head_dim; if (max_groups < 1) max_groups = 1;
        int G = (position_count + head_dim - 1) / head_dim;
        if (G < 1) G = 1; if (G > max_groups) G = max_groups;
        float* vpart = shared + head_dim + base_position + k_tokens; // [max_groups * head_dim]
        for (int idx = tid; idx < G * head_dim; idx += blockDim.x) {
            int gid = idx / head_dim, did = idx % head_dim;
            int p_lo = (int)((long)gid * position_count / G);
            int p_hi = (int)((long)(gid + 1) * position_count / G);
            float acc = 0.0f;
            for (int p = p_lo; p < p_hi; p++)
                acc += (scores[p] * inv) * f16_bits_to_f32(vbase[(long)p * head_dim + did]);
            vpart[(long)did * G + gid] = acc;
        }
        __syncthreads();
        for (int did = tid; did < head_dim; did += blockDim.x) {
            float sum = 0.0f;
            for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
            out[(long)t * q_per_token + (long)head * head_dim + did] = sum;
        }
    }
}

// ---- Tree-verify kernels (lossless GPU tree speculation, Lane A) -----------
// Generalize the linear batched verify to a draft TREE: the N nodes no longer
// occupy consecutive positions on one branch. Each node t lives at its own KV
// slot `node_kvslot[t]` (= base + BFS index t) and at RoPE position
// `base + node_depth[t]`. A node attends the DENSE committed prefix [0, base)
// PLUS only the in-chunk slots on its own root-to-node path (its ancestors).
// On a LINEAR (single-branch) tree these reduce EXACTLY to kv_scatter_batched /
// attention_batched (slots base..base+t, ancestors 0..t), so the tree path is
// bit-identical to the linear verify there — the losslessness anchor.

// Scatter each node t's K/V into its own cache slot node_kvslot[t]. RoPE is
// already baked into src (the host stages per-node cos/sin at base+depth[t]),
// so this only relocates the per-node write target vs kv_scatter_batched
// (which writes base+t). On a linear tree node_kvslot[t] == base+t ⇒ identical.
extern "C" __global__ void kv_scatter_tree_batched(
    const float* __restrict__ src, unsigned short* __restrict__ cache,
    const int* __restrict__ node_kvslot,
    int n_kv_heads, int head_dim, int max_pos, int per_token_dim, int k_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_tokens * n_kv_heads * head_dim;
    if (idx >= total) return;
    int t = idx / (n_kv_heads * head_dim);
    int rem = idx % (n_kv_heads * head_dim);
    int kv_head = rem / head_dim;
    int d = rem % head_dim;
    int position = node_kvslot[t];
    cache[((long)kv_head * max_pos + position) * head_dim + d] =
        f32_to_f16_bits(src[(long)t * per_token_dim + (long)kv_head * head_dim + d]);
}

// Tree attention: node t (query) attends (a) the dense committed prefix
// [0, base) EXACTLY as attention_batched, then (b) the in-chunk node slots on
// its root-to-node path, in DEPTH order, so the exp-sum order matches a linear
// decode. The path is the set of nodes j whose ancestor bit is set for t
// (ancestor_bits[t*words + j/32] >> (j%32)); node j's K/V is at slot base+j.
// We append ancestors in BFS-index order (== depth order along a single path,
// since parent index < child index), giving the same sequential score / max /
// exp-sum / weighted-V order the linear kernel uses. Masked (non-ancestor)
// slots are SKIPPED, never scored. One block per (node, query head).
extern "C" __global__ void attention_tree_batched(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    const unsigned int* __restrict__ ancestor_bits, int words,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens, int splitk_active
) {
    int t = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;
    const unsigned int* anc = ancestor_bits + (long)t * words;

    extern __shared__ float shared[];
    float* qsh = shared;               // head_dim
    float* scores = shared + head_dim;  // base + (#in-chunk ancestors)
    int* slots = (int*)(scores + base_position + k_tokens); // absolute KV slot per score
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();
    // Build the ordered list of KV slots this node attends: the dense prefix
    // [0, base) then the in-chunk ancestor slots base+j (BFS / depth order).
    // Thread 0 builds it (small N); the dot products parallelize over it.
    __shared__ int s_count;
    if (tid == 0) {
        int n = 0;
        for (int p = 0; p < base_position; p++) slots[n++] = p;
        for (int j = 0; j < k_tokens; j++) {
            if ((anc[j >> 5] >> (j & 31)) & 1u) slots[n++] = base_position + j;
        }
        s_count = n;
    }
    __syncthreads();
    int count = s_count;
    // SIROCCO prefill lever: uint4 f16 K read, same d-order accumulation -> BYTE-IDENTICAL (mirrors
    // attention_batched above and win #1). Gathered row slots[i]*head_dim stays 16B-aligned (head_dim=64).
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    for (int i = tid; i < count; i += blockDim.x) {
        const unsigned short* kp = kbase + (long)slots[i] * head_dim;
        float dot = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d + 0] * f16_bits_to_f32(k8[0]); dot += qsh[d + 1] * f16_bits_to_f32(k8[1]);
            dot += qsh[d + 2] * f16_bits_to_f32(k8[2]); dot += qsh[d + 3] * f16_bits_to_f32(k8[3]);
            dot += qsh[d + 4] * f16_bits_to_f32(k8[4]); dot += qsh[d + 5] * f16_bits_to_f32(k8[5]);
            dot += qsh[d + 6] * f16_bits_to_f32(k8[6]); dot += qsh[d + 7] * f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        scores[i] = dot * scale;
    }
    __syncthreads();
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = scores[0];
        for (int i = 1; i < count; i++) if (scores[i] > m) m = scores[i];
        s_max = m;
    }
    __syncthreads();
    for (int i = tid; i < count; i += blockDim.x) scores[i] = expf(scores[i] - s_max);
    __syncthreads();
    // Denominator + weighted-V, keyed on the attended-key count `count` (not absolute position).
    // On a LINEAR tree count==position_count and slots[i]==i, so this matches attention_batched and
    // single-token decode for the committed path (the only path held to losslessness; branching
    // nodes are discarded). Split-K emulation above SPLITK_THRESHOLD mirrors plain decode there.
    if (splitk_active && count > 512) {            // 512 == SPLITK_THRESHOLD
        int n_splits = (count + 255) / 256;        // == count.div_ceil(256).clamp(2, SPLITK_MAX)
        if (n_splits < 2) n_splits = 2;
        if (n_splits > 32) n_splits = 32;          // SPLITK_MAX (mirror src const)
        if (tid == 0) {
            float total = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int i_lo = (int)((long)sp * count / n_splits);
                int i_hi = (int)((long)(sp + 1) * count / n_splits);
                float ls = 0.0f;
                for (int i = i_lo; i < i_hi; i++) ls += scores[i];
                total += ls;
            }
            s_sum = total;
        }
        __syncthreads();
        float inv = 1.0f / s_sum;
        for (int did = tid; did < head_dim; did += blockDim.x) {
            float acc = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int i_lo = (int)((long)sp * count / n_splits);
                int i_hi = (int)((long)(sp + 1) * count / n_splits);
                float a = 0.0f;
                for (int i = i_lo; i < i_hi; i++)
                    a += scores[i] * f16_bits_to_f32(vbase[(long)slots[i] * head_dim + did]);
                acc += a;
            }
            out[(long)t * q_per_token + (long)head * head_dim + did] = acc * inv;
        }
    } else {
        // G-group: IDENTICAL reorder to attention_decode (G from `count`; contiguous split +
        // g-order combine). The losslessness anchor for the committed linear path.
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) sum += scores[i];
            s_sum = sum;
        }
        __syncthreads();
        float inv = 1.0f / s_sum;
        int max_groups = 1024 / head_dim; if (max_groups < 1) max_groups = 1;
        int G = (count + head_dim - 1) / head_dim;
        if (G < 1) G = 1; if (G > max_groups) G = max_groups;
        float* vpart = (float*)(slots + base_position + k_tokens); // [max_groups * head_dim]
        for (int idx = tid; idx < G * head_dim; idx += blockDim.x) {
            int gid = idx / head_dim, did = idx % head_dim;
            int i_lo = (int)((long)gid * count / G);
            int i_hi = (int)((long)(gid + 1) * count / G);
            float acc = 0.0f;
            for (int i = i_lo; i < i_hi; i++)
                acc += (scores[i] * inv) * f16_bits_to_f32(vbase[(long)slots[i] * head_dim + did]);
            vpart[(long)did * G + gid] = acc;
        }
        __syncthreads();
        for (int did = tid; did < head_dim; did += blockDim.x) {
            float sum = 0.0f;
            for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
            out[(long)t * q_per_token + (long)head * head_dim + did] = sum;
        }
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

    // SIROCCO Lane K: uint4 (128-bit, 8 keys/load) f16 K read, same d-order accumulation ->
    // BYTE-IDENTICAL (fmad=false), so the split-K bit-identity vs spec-verify (attention_batched/
    // tree) and splitk_spec_verify_bit_identical gate are preserved. Non-8 head_dim -> scalar.
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    float local_max = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d + 0] * f16_bits_to_f32(k8[0]); dot += qsh[d + 1] * f16_bits_to_f32(k8[1]);
            dot += qsh[d + 2] * f16_bits_to_f32(k8[2]); dot += qsh[d + 3] * f16_bits_to_f32(k8[3]);
            dot += qsh[d + 4] * f16_bits_to_f32(k8[4]); dot += qsh[d + 5] * f16_bits_to_f32(k8[5]);
            dot += qsh[d + 6] * f16_bits_to_f32(k8[6]); dot += qsh[d + 7] * f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d] * f16_bits_to_f32(kp[d]);
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

// Pass 1 (COALESCED variant, env-gated CAMELID_ATTN_COALESCED): identical math/IO to
// attn_sk_scores but assigns ONE WARP (32 lanes) per key position so the warp's loads of
// kp[L..L+31] are 32 consecutive f16 = 64 contiguous bytes = coalesced (vs the scalar
// kernel where adjacent threads scatter 256 bytes apart). head_dim=128 -> lane L sums
// d=L,L+32,L+64,L+96; a __shfl_down_sync warp-tree reduces to the position dot. This
// re-associates the head_dim sum (warp-tree vs sequential) -> parity-sensitive. scale,
// scores_buf layout and chunkmax_buf semantics are IDENTICAL so passes 2/3 are unchanged.
// block_dim must be a multiple of 32 (launched at 256 = 8 warps); warp w strides positions.
extern "C" __global__ void attn_sk_scores_coalesced(
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

    int n_warps = blockDim.x >> 5;          // 32 lanes per warp
    int warp_id = tid >> 5;
    int lane = tid & 31;

    float local_max = -3.4e38f;
    // warp `warp_id` processes positions p_lo+warp_id, +n_warps, ...
    for (int p = p_lo + warp_id; p < p_hi; p += n_warps) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        // lane L owns d = L, L+32, ... -> warp's simultaneous kp[L..L+31] loads coalesce.
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) dot += qsh[d] * f16_bits_to_f32(kp[d]);
        // warp-tree reduce the partial dots to lane 0.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) {
            float sc = dot * scale;
            scores_buf[(long)head * max_pos + p] = sc;
            local_max = fmaxf(local_max, sc);
        }
    }
    // per-warp max lives in lane 0 of each warp; reduce the n_warps lane-0 maxes.
    __shared__ float wmax[32];              // up to 32 warps (block <= 1024)
    if (lane == 0) wmax[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -3.4e38f;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, wmax[w]);
        chunkmax_buf[(long)head * n_splits + sp] = m;
    }
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

// ---- SIROCCO Phase P M1: flash prefill attention (opt-in CAMELID_FLASH_PREFILL) -------------
// The prefill bottleneck (99% of long-ctx wall) is attention_batched running ONE block per
// (query-token, head), each re-streaming the full prefix K/V -> a k_tokens x GQA re-read. These two
// kernels are the split-K decode passes (attn_sk_scores/partial) EXTENDED to the whole k_tokens query
// chunk: for each key p, K[p]/V[p] is loaded and dequantized ONCE and reused across all k_tokens dots
// (the query-reuse axis, ~k_tokens x less K/V DRAM traffic). Buffers are laid out flat by
// (t*n_heads+head) so attn_sk_combine is reused UNCHANGED (called with n_heads = k_tokens*n_heads).
// Per-token causal mask: token t (at position base+t) attends [0, base+t]; masked score = -inf ->
// exp = 0. Reduction is the split-K reassociation (n_splits from the chunk's full length base+k),
// so this is TOKEN-PARITY (gated by the multi-prompt oracle), not byte-identical. Prefill-only:
// verify_batch stays on attention_batched and its bit-identity contract. MAX_BQ bounds the per-thread
// token arrays (k_tokens <= MAX_VERIFY_K = 8).
#define FLASH_MAX_BQ 16
// Pass 1: per (head, split), compute all k_tokens scores for the split (K[p] dequant reused across
// tokens) into scores_buf[(t*n_heads+head)*max_pos + p] and per-token chunk max into chunkmax_buf.
extern "C" __global__ void flash_pref_scores(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int k_tokens,
    int q_per_token, int max_pos, float scale, int n_splits, int position_count
) {
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    extern __shared__ float qsh[];                 // k_tokens * head_dim (all query rows of this head)
    int tid = threadIdx.x;
    for (int i = tid; i < k_tokens * head_dim; i += blockDim.x)
        qsh[i] = q[(long)(i / head_dim) * q_per_token + (long)head * head_dim + (i % head_dim)];
    __syncthreads();
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    float local_max[FLASH_MAX_BQ];
    for (int t = 0; t < k_tokens; t++) local_max[t] = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot[FLASH_MAX_BQ];
        for (int t = 0; t < k_tokens; t++) dot[t] = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {                  // load + dequant K[p] chunk ONCE, reuse across tokens
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            float kf0 = f16_bits_to_f32(k8[0]), kf1 = f16_bits_to_f32(k8[1]);
            float kf2 = f16_bits_to_f32(k8[2]), kf3 = f16_bits_to_f32(k8[3]);
            float kf4 = f16_bits_to_f32(k8[4]), kf5 = f16_bits_to_f32(k8[5]);
            float kf6 = f16_bits_to_f32(k8[6]), kf7 = f16_bits_to_f32(k8[7]);
            for (int t = 0; t < k_tokens; t++) {
                const float* qt = qsh + (long)t * head_dim;
                dot[t] += qt[d + 0] * kf0; dot[t] += qt[d + 1] * kf1;
                dot[t] += qt[d + 2] * kf2; dot[t] += qt[d + 3] * kf3;
                dot[t] += qt[d + 4] * kf4; dot[t] += qt[d + 5] * kf5;
                dot[t] += qt[d + 6] * kf6; dot[t] += qt[d + 7] * kf7;
            }
        }
        for (; d < head_dim; d++) {
            float kf = f16_bits_to_f32(kp[d]);
            for (int t = 0; t < k_tokens; t++) dot[t] += qsh[(long)t * head_dim + d] * kf;
        }
        for (int t = 0; t < k_tokens; t++) {
            float sc = (p <= base_position + t) ? dot[t] * scale : -3.4e38f;   // causal mask
            scores_buf[((long)t * n_heads + head) * max_pos + p] = sc;
            local_max[t] = fmaxf(local_max[t], sc);
        }
    }
    __shared__ float red[256];                     // block_dim == 256
    for (int t = 0; t < k_tokens; t++) {
        red[tid] = local_max[t];
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        if (tid == 0) chunkmax_buf[((long)t * n_heads + head) * n_splits + sp] = red[0];
        __syncthreads();
    }
}

// Pass 2: per (head, split), for each token read the exact global max over splits, exp its split in
// place, write the split's exp-sum (lsum) and unnormalized weighted-V (acc). V[p] dequant reused
// across tokens. Masked positions have score -inf -> exp 0 -> contribute nothing.
extern "C" __global__ void flash_pref_partial(
    const unsigned short* __restrict__ cache_v, float* __restrict__ scores_buf,
    const float* __restrict__ chunkmax_buf, float* __restrict__ lsum_buf, float* __restrict__ acc_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int k_tokens,
    int max_pos, int n_splits, int position_count
) {
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    int tid = threadIdx.x;
    for (int t = 0; t < k_tokens; t++) {
        float* sc_t = scores_buf + ((long)t * n_heads + head) * max_pos;
        float gmax = -3.4e38f;
        for (int i = 0; i < n_splits; i++) gmax = fmaxf(gmax, chunkmax_buf[((long)t * n_heads + head) * n_splits + i]);
        for (int p = p_lo + tid; p < p_hi; p += blockDim.x) sc_t[p] = expf(sc_t[p] - gmax);
        __syncthreads();
        if (tid == 0) {
            float ls = 0.0f;
            for (int p = p_lo; p < p_hi; p++) ls += sc_t[p];
            lsum_buf[((long)t * n_heads + head) * n_splits + sp] = ls;
        }
        __syncthreads();
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float a[FLASH_MAX_BQ];
        for (int t = 0; t < k_tokens; t++) a[t] = 0.0f;
        for (int p = p_lo; p < p_hi; p++) {
            float vv = f16_bits_to_f32(vbase[(long)p * head_dim + d]);   // V[p][d] read once, reuse across tokens
            for (int t = 0; t < k_tokens; t++)
                a[t] += scores_buf[((long)t * n_heads + head) * max_pos + p] * vv;
        }
        for (int t = 0; t < k_tokens; t++)
            acc_buf[((((long)t * n_heads + head) * n_splits + sp) * head_dim) + d] = a[t];
    }
}

// SIROCCO Compute M-C3: byte-identical const-k=8 twins of the two flash kernels for the common
// k_tokens==8 prefill chunk. Same ops in the same order (bit-identical output), but the token loops
// are compile-time-bounded (FLASH_KT8) + #pragma unroll so ptxas promotes the per-row dot/local_max/a
// accumulators from LOCAL memory (128B/64B stack frames -> ptxas -v) to REGISTERS (0B). The runtime
// k_tokens loop bound (flash_pref_scores/partial) blocks this unroll, forcing local-mem traffic that
// gates occupancy -- confirmed the binder (M2b: FLASH_MAX_BQ 16->32 regressed even k=8). Dispatched by
// launch_attention_flash_prefill only when k_tokens==8; other chunk sizes use the runtime kernels.
#define FLASH_KT8 8
extern "C" __global__ void flash_pref_scores_k8(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int k_tokens,
    int q_per_token, int max_pos, float scale, int n_splits, int position_count
) {
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    extern __shared__ float qsh[];
    int tid = threadIdx.x;
    for (int i = tid; i < FLASH_KT8 * head_dim; i += blockDim.x)
        qsh[i] = q[(long)(i / head_dim) * q_per_token + (long)head * head_dim + (i % head_dim)];
    __syncthreads();
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    float local_max[FLASH_KT8];
    #pragma unroll
    for (int t = 0; t < FLASH_KT8; t++) local_max[t] = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot[FLASH_KT8];
        #pragma unroll
        for (int t = 0; t < FLASH_KT8; t++) dot[t] = 0.0f;
        int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            float kf0 = f16_bits_to_f32(k8[0]), kf1 = f16_bits_to_f32(k8[1]);
            float kf2 = f16_bits_to_f32(k8[2]), kf3 = f16_bits_to_f32(k8[3]);
            float kf4 = f16_bits_to_f32(k8[4]), kf5 = f16_bits_to_f32(k8[5]);
            float kf6 = f16_bits_to_f32(k8[6]), kf7 = f16_bits_to_f32(k8[7]);
            #pragma unroll
            for (int t = 0; t < FLASH_KT8; t++) {
                const float* qt = qsh + (long)t * head_dim;
                dot[t] += qt[d + 0] * kf0; dot[t] += qt[d + 1] * kf1;
                dot[t] += qt[d + 2] * kf2; dot[t] += qt[d + 3] * kf3;
                dot[t] += qt[d + 4] * kf4; dot[t] += qt[d + 5] * kf5;
                dot[t] += qt[d + 6] * kf6; dot[t] += qt[d + 7] * kf7;
            }
        }
        for (; d < head_dim; d++) {
            float kf = f16_bits_to_f32(kp[d]);
            #pragma unroll
            for (int t = 0; t < FLASH_KT8; t++) dot[t] += qsh[(long)t * head_dim + d] * kf;
        }
        #pragma unroll
        for (int t = 0; t < FLASH_KT8; t++) {
            float sc = (p <= base_position + t) ? dot[t] * scale : -3.4e38f;
            scores_buf[((long)t * n_heads + head) * max_pos + p] = sc;
            local_max[t] = fmaxf(local_max[t], sc);
        }
    }
    __shared__ float red[256];
    #pragma unroll 1
    for (int t = 0; t < FLASH_KT8; t++) {
        red[tid] = local_max[t];
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        if (tid == 0) chunkmax_buf[((long)t * n_heads + head) * n_splits + sp] = red[0];
        __syncthreads();
    }
}

extern "C" __global__ void flash_pref_partial_k8(
    const unsigned short* __restrict__ cache_v, float* __restrict__ scores_buf,
    const float* __restrict__ chunkmax_buf, float* __restrict__ lsum_buf, float* __restrict__ acc_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int k_tokens,
    int max_pos, int n_splits, int position_count
) {
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    int tid = threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < FLASH_KT8; t++) {
        float* sc_t = scores_buf + ((long)t * n_heads + head) * max_pos;
        float gmax = -3.4e38f;
        for (int i = 0; i < n_splits; i++) gmax = fmaxf(gmax, chunkmax_buf[((long)t * n_heads + head) * n_splits + i]);
        for (int p = p_lo + tid; p < p_hi; p += blockDim.x) sc_t[p] = expf(sc_t[p] - gmax);
        __syncthreads();
        if (tid == 0) {
            float ls = 0.0f;
            for (int p = p_lo; p < p_hi; p++) ls += sc_t[p];
            lsum_buf[((long)t * n_heads + head) * n_splits + sp] = ls;
        }
        __syncthreads();
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float a[FLASH_KT8];
        #pragma unroll
        for (int t = 0; t < FLASH_KT8; t++) a[t] = 0.0f;
        for (int p = p_lo; p < p_hi; p++) {
            float vv = f16_bits_to_f32(vbase[(long)p * head_dim + d]);
            #pragma unroll
            for (int t = 0; t < FLASH_KT8; t++)
                a[t] += scores_buf[((long)t * n_heads + head) * max_pos + p] * vv;
        }
        #pragma unroll
        for (int t = 0; t < FLASH_KT8; t++)
            acc_buf[((((long)t * n_heads + head) * n_splits + sp) * head_dim) + d] = a[t];
    }
}

// =====================================================================
// qwen35 (Ornith) hybrid gated-delta-net (SSM) kernels.
// Mirror the CPU reference in src/runnable/model.rs::qwen35_ssm_compute,
// preserving its arithmetic AND summation order so greedy-token parity holds.
// =====================================================================

// ---- Per-head L2 normalize (q/k in SSM layers) -----------------------------
// In-place: buf[head*head_dim + i] /= max(sqrt(sum x^2), eps). One block per head.
// Matches CPU l2_norm_inplace: DOUBLE-precision sum of squares, cast to f32, sqrt,
// then fmax(eps). The per-element apply is parallel.
extern "C" __global__ void ssm_l2_norm_per_head(
    float* __restrict__ buf, int head_dim, float eps
) {
    extern __shared__ float xs[];
    __shared__ float s_scale;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    long base = (long)head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) xs[i] = buf[base + i];
    __syncthreads();
    if (tid == 0) {
        double ss = 0.0;
        for (int i = 0; i < head_dim; i++) { double v = (double)xs[i]; ss += v * v; }
        float denom = sqrtf((float)ss);
        if (denom < eps) denom = eps;
        s_scale = 1.0f / denom;
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = tid; i < head_dim; i += blockDim.x) buf[base + i] = xs[i] * scale;
}

// ---- Causal depthwise conv1d (kernel d_conv) + SiLU ------------------------
// One thread per channel c. window = [ring_state(oldest..newest), current x[c]];
// acc = sum_t w[c,t]*state[c,t] + w[c,d_conv-1]*x[c]; out = silu(acc); then the
// ring buffer shifts left and appends x[c]. Bit-identical to the CPU conv loop.
extern "C" __global__ void ssm_conv1d(
    const float* __restrict__ conv_w,    // [conv_dim * d_conv]
    const float* __restrict__ x,         // [conv_dim] current input
    float* __restrict__ conv_state,      // [conv_dim * (d_conv-1)] ring, updated
    float* __restrict__ conv_out,        // [conv_dim] output (post-SiLU)
    int conv_dim, int d_conv
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int cm1 = d_conv - 1;
    const float* w = conv_w + (long)c * d_conv;
    float* st = conv_state + (long)c * cm1;
    float xc = x[c];
    float acc = 0.0f;
    for (int t = 0; t < cm1; t++) acc += w[t] * st[t];
    acc += w[cm1] * xc;
    conv_out[c] = acc / (1.0f + expf(-acc));   // SiLU
    for (int t = 0; t < cm1 - 1; t++) st[t] = st[t + 1];
    st[cm1 - 1] = xc;
}

// ---- Gated delta-rule recurrence (one decode step) + gated RMSNorm ---------
// One block per value-head (gridDim.x = nv), blockDim.x = d_state (== head_v_dim).
// Thread j owns column j of the [d_state x d_state] state S (row-major [i*d_state+j]).
//   decay:  S[i,j] *= exp(glog[h])
//   sk[j]   = sum_i S[i,j]*k[i]                 (fused with decay, i-order)
//   d[j]    = (v[j] - sk[j]) * beta[h]
//   S[i,j] += k[i]*d[j];  o[j] = sum_i S[i,j]*(q[i]*qscale)   (fused, i-order)
//   gated RMSNorm: out = RMSNorm(o, ssm_norm) * SiLU(z)       (j-order serial sum)
// k_conv/q_conv are L2-normed; GQA tile-repeat key head = h % nk. beta is already
// sigmoid'd; glog is pre-exp. Bit-identical order to the CPU reference.
extern "C" __global__ void ssm_delta_rule(
    float* __restrict__ state,           // [nv*d_state*d_state], updated in place
    const float* __restrict__ k_conv,    // [nk*d_state]
    const float* __restrict__ q_conv,    // [nk*d_state]
    const float* __restrict__ v_conv,    // [nv*d_state]
    const float* __restrict__ z,         // [nv*d_state]
    const float* __restrict__ beta,      // [nv] (post-sigmoid)
    const float* __restrict__ glog,      // [nv] (pre-exp)
    const float* __restrict__ ssm_norm,  // [d_state]
    float* __restrict__ out,             // [nv*d_state]
    int d_state, int nk, float eps
) {
    int h = blockIdx.x;
    int j = threadIdx.x;                 // 0..d_state-1
    int hk = h % nk;
    extern __shared__ float sh[];
    float* sk_ = sh;                     // [d_state] k head
    float* sq_ = sh + d_state;           // [d_state] q head
    float* so_ = sh + 2 * d_state;       // [d_state] o scratch
    sk_[j] = k_conv[(long)hk * d_state + j];
    sq_[j] = q_conv[(long)hk * d_state + j];
    __syncthreads();
    float* St = state + (long)h * d_state * d_state;
    float g = expf(glog[h]);
    float bh = beta[h];
    float qscale = 1.0f / sqrtf((float)d_state);
    float skj = 0.0f;
    for (int i = 0; i < d_state; i++) {
        float s = St[(long)i * d_state + j] * g;
        St[(long)i * d_state + j] = s;
        skj += s * sk_[i];
    }
    float dj = (v_conv[(long)h * d_state + j] - skj) * bh;
    float oj = 0.0f;
    for (int i = 0; i < d_state; i++) {
        float s = St[(long)i * d_state + j] + sk_[i] * dj;
        St[(long)i * d_state + j] = s;
        oj += s * (sq_[i] * qscale);
    }
    so_[j] = oj;
    __syncthreads();
    __shared__ float s_scale;
    if (j == 0) {
        float sum = 0.0f;
        for (int t = 0; t < d_state; t++) sum += so_[t] * so_[t];
        s_scale = 1.0f / sqrtf(sum / (float)d_state + eps);
    }
    __syncthreads();
    float normed = so_[j] * s_scale * ssm_norm[j];
    float zj = z[(long)h * d_state + j];
    float silu_z = zj / (1.0f + expf(-zj));
    out[(long)h * d_state + j] = normed * silu_z;
}

// ---- Sigmoid output gate (qwen35 full-attention) ---------------------------
// out[i] *= sigmoid(gate[i]). One thread per element.
extern "C" __global__ void sigmoid_mul(
    float* __restrict__ out, const float* __restrict__ gate, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = gate[i];
    out[i] *= 1.0f / (1.0f + expf(-gv));
}

// ---- Device-side decode-loop helpers ----------------------------------------
// embed_gather_*: dequantize ONE embedding row (the token id is read from a
// DEVICE u32 pointer, e.g. the previous step's argmax output) straight into the
// f32 hidden buffer — removing the per-token CPU round-trip (argmax D2H ->
// host dequant_row -> 16 KB H2D) from the decode loop. Each kernel is the
// elementwise-exact mirror of the CPU block dequantize for its family
// (tensor/mod.rs Q*Block::dequantize / dequant.rs): identical formulas,
// identical f32 association per element, so the hidden vector is bit-identical
// to the host-fed path. One thread per element.
extern "C" __global__ void embed_gather_q4k(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= dim) return;
    const unsigned char* row = table + (long)(*token) * (long)(dim / 256) * 144;
    const unsigned char* blk = row + (long)(e / 256) * 144;
    int i = e & 255;
    int pair = i >> 6;
    int r = i & 63;
    int half = r >> 5;
    int l = r & 31;
    float d = f16_bits_to_f32((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    float dmin = f16_bits_to_f32((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
    const unsigned char* sc12 = blk + 4;
    int idx = pair * 2 + half;
    unsigned char sc, mn;
    if (idx < 4) {
        sc = sc12[idx] & 63;
        mn = sc12[idx + 4] & 63;
    } else {
        sc = (sc12[idx + 4] & 0x0f) | ((sc12[idx - 4] >> 6) << 4);
        mn = (sc12[idx + 4] >> 4) | ((sc12[idx] >> 6) << 4);
    }
    unsigned char byte = blk[16 + pair * 32 + l];
    unsigned char q = half ? (byte >> 4) : (byte & 0x0f);
    // CPU order: (d*sc) * q - (dmin*mn)
    out[e] = (d * (float)sc) * (float)q - (dmin * (float)mn);
}
// Q6_K rows use the 224 B PADDED wire (pad_q6k_blocks), same as the gemv lane.
extern "C" __global__ void embed_gather_q6k(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= dim) return;
    const unsigned char* row = table + (long)(*token) * (long)(dim / 256) * 224;
    const unsigned char* blk = row + (long)(e / 256) * 224;
    int i = e & 255;
    int n128 = i >> 7;
    int r = i & 127;
    int quarter = r >> 5;
    int l = r & 31;
    int is = l >> 4;
    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const signed char* sc = (const signed char*)(blk + 192);
    float d = f16_bits_to_f32((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
    int ql_off = n128 * 64;
    int qh_off = n128 * 32;
    int sc_off = n128 * 8;
    unsigned char h = qh[qh_off + l];
    int q;
    if (quarter == 0) {
        q = (int)((ql[ql_off + l] & 0x0f) | ((h & 3) << 4)) - 32;
    } else if (quarter == 1) {
        q = (int)((ql[ql_off + l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) - 32;
    } else if (quarter == 2) {
        q = (int)((ql[ql_off + l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
    } else {
        q = (int)((ql[ql_off + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
    }
    // CPU order: d * sc * q (left-assoc)
    out[e] = d * (float)sc[sc_off + is + 2 * quarter] * (float)q;
}
extern "C" __global__ void embed_gather_q3k(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    const unsigned int KMASK1 = 0x03030303u, KMASK2 = 0x0f0f0f0fu;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= dim) return;
    const unsigned char* row = table + (long)(*token) * (long)(dim / 256) * 110;
    const unsigned char* blk = row + (long)(e / 256) * 110;
    int i = e & 255;
    int sup = i >> 7;
    int r = i & 127;
    int group = r >> 5;
    int k = r & 31;
    int half16 = k >> 4;
    int l = k & 15;
    const unsigned char* hb = blk;          // high_bits[32]
    const unsigned char* vals = blk + 32;   // values[64]
    const unsigned char* sr = blk + 96;     // scales[12]
    float d = f16_bits_to_f32((unsigned short)blk[108] | ((unsigned short)blk[109] << 8));
    // kmask 6-bit scale expansion — identical to Q3KBlock::expanded_scales.
    unsigned int a0 = (unsigned int)sr[0] | ((unsigned int)sr[1] << 8) | ((unsigned int)sr[2] << 16) | ((unsigned int)sr[3] << 24);
    unsigned int a1 = (unsigned int)sr[4] | ((unsigned int)sr[5] << 8) | ((unsigned int)sr[6] << 16) | ((unsigned int)sr[7] << 24);
    unsigned int a2 = (unsigned int)sr[8] | ((unsigned int)sr[9] << 8) | ((unsigned int)sr[10] << 16) | ((unsigned int)sr[11] << 24);
    unsigned int e2w = ((a0 >> 4) & KMASK2) | (((a2 >> 4) & KMASK1) << 4);
    unsigned int e3w = ((a1 >> 4) & KMASK2) | (((a2 >> 6) & KMASK1) << 4);
    unsigned int e0w = (a0 & KMASK2) | ((a2 & KMASK1) << 4);
    unsigned int e1w = (a1 & KMASK2) | (((a2 >> 2) & KMASK1) << 4);
    int scale_idx = sup * 8 + group * 2 + half16;
    unsigned int word = (scale_idx < 4) ? e0w : (scale_idx < 8) ? e1w : (scale_idx < 12) ? e2w : e3w;
    signed char scv = (signed char)((word >> ((scale_idx & 3) * 8)) & 0xff);
    int shift = group * 2;
    unsigned int mask = 1u << (sup * 4 + group);
    int vidx = sup * 32 + half16 * 16 + l;
    int hidx = half16 * 16 + l;
    int high = (hb[hidx] & mask) ? 0 : 4;
    int val = (int)((vals[vidx] >> shift) & 3) - high;
    // CPU order: (d * (sc - 32)) * val
    out[e] = (d * (float)((int)scv - 32)) * (float)val;
}
extern "C" __global__ void embed_gather_q8_0(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= dim) return;
    const unsigned char* row = table + (long)(*token) * (long)(dim / 32) * 34;
    const unsigned char* blk = row + (long)(e / 32) * 34;
    int j = e & 31;
    float d = f16_bits_to_f32((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    out[e] = d * (float)((signed char)blk[2 + j]);
}
// rope_select: copy position `pos`'s precomputed cos/sin row (half = rope_dim/2
// values each) out of the resident all-positions tables into the small per-token
// buffers the rope kernel reads — the tables are built once on the host with the
// VERBATIM qwen35_rope_tables math, so the rope inputs are bit-identical to the
// host-uploaded path.
extern "C" __global__ void rope_select(
    const float* __restrict__ cos_all, const float* __restrict__ sin_all,
    int pos, int half, float* __restrict__ cos_out, float* __restrict__ sin_out
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= half) return;
    cos_out[i] = cos_all[(long)pos * half + i];
    sin_out[i] = sin_all[(long)pos * half + i];
}

// ---- SSM gates: beta = sigmoid(beta_raw); glog = softplus(alpha_raw+dt_bias)*a -
// One thread per value-head (nv). Feeds ssm_delta_rule (beta post-sigmoid, glog
// pre-exp). softplus matches the CPU's (1+exp(x)).ln() with the x>20 passthrough.
extern "C" __global__ void ssm_gates(
    const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw,
    const float* __restrict__ dt_bias, const float* __restrict__ a,
    float* __restrict__ beta_out, float* __restrict__ glog_out, int nv
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    beta_out[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
    float x = alpha_raw[h] + dt_bias[h];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    glog_out[h] = sp * a[h];
}

// ---- Deinterleave fused per-head [query(hd) | gate(hd)] x n_heads ----------
// qg is [n_heads * 2 * head_dim]; split into contiguous q_out / gate_out
// [n_heads*head_dim] (qwen35 full-attention fused query+output-gate projection).
extern "C" __global__ void deinterleave_qgate(
    const float* __restrict__ qg, float* __restrict__ q_out,
    float* __restrict__ gate_out, int n_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * head_dim) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    long src = (long)head * 2 * head_dim;
    q_out[idx] = qg[src + d];
    gate_out[idx] = qg[src + head_dim + d];
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
    pub(crate) q4_0_gemv: CudaFunction,
    pub(crate) q4_1_gemv: CudaFunction,
    pub(crate) nvfp4_gemv: CudaFunction,
    pub(crate) q4k_gemv: CudaFunction,
    pub(crate) q5k_gemv: CudaFunction,
    pub(crate) q6k_gemv: CudaFunction,
    pub(crate) q2k_gemv: CudaFunction,
    pub(crate) q3k_gemv: CudaFunction,
    pub(crate) iq4xs_gemv: CudaFunction,
    pub(crate) q4k_gemm_batched: CudaFunction,
    pub(crate) q6k_gemm_batched: CudaFunction,
    pub(crate) quantize_q8k: CudaFunction,
    pub(crate) rms_norm_quantize_q8k: CudaFunction,
    pub(crate) silu_mul_quantize_q8k: CudaFunction,
    pub(crate) rope: CudaFunction,
    pub(crate) kv_scatter: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) attention_sw: CudaFunction,
    pub(crate) silu_mul: CudaFunction,
    pub(crate) silu_mul_quantize: CudaFunction,
    pub(crate) geglu_mul: CudaFunction,
    pub(crate) soft_cap: CudaFunction,
    pub(crate) f32_gemv: CudaFunction,
    pub(crate) scale_f32: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    pub(crate) scaled_axpy: CudaFunction,
    pub(crate) argmax: CudaFunction,
    pub(crate) sample_gumbel: CudaFunction,
    pub(crate) gemm_batched: CudaFunction,
    pub(crate) rms_norm_batched: CudaFunction,
    pub(crate) rope_batched: CudaFunction,
    pub(crate) kv_scatter_batched: CudaFunction,
    pub(crate) attention_batched: CudaFunction,
    pub(crate) kv_scatter_tree_batched: CudaFunction,
    pub(crate) attention_tree_batched: CudaFunction,
    pub(crate) argmax_batched: CudaFunction,
    pub(crate) attn_sk_scores: CudaFunction,
    pub(crate) attn_sk_scores_coalesced: CudaFunction,
    pub(crate) attn_sk_partial: CudaFunction,
    pub(crate) attn_sk_combine: CudaFunction,
    pub(crate) flash_pref_scores: CudaFunction,
    pub(crate) flash_pref_partial: CudaFunction,
    pub(crate) flash_pref_scores_k8: CudaFunction,
    pub(crate) flash_pref_partial_k8: CudaFunction,
    // qwen35 (Ornith) gated-delta-net SSM kernels.
    pub(crate) ssm_l2_norm_per_head: CudaFunction,
    pub(crate) ssm_conv1d: CudaFunction,
    pub(crate) ssm_delta_rule: CudaFunction,
    pub(crate) sigmoid_mul: CudaFunction,
    pub(crate) embed_gather_q4k: CudaFunction,
    pub(crate) embed_gather_q6k: CudaFunction,
    pub(crate) embed_gather_q3k: CudaFunction,
    pub(crate) embed_gather_q8_0: CudaFunction,
    pub(crate) rope_select: CudaFunction,
    pub(crate) ssm_gates: CudaFunction,
    pub(crate) deinterleave_qgate: CudaFunction,
    /// Env-gated (CAMELID_ATTN_COALESCED) dispatch of the coalesced K-dot in
    /// split-K pass 1. Read ONCE at construction; default OFF so the shipped
    /// path stays byte-identical.
    pub(crate) attn_coalesced: bool,
}

impl CudaResidentKernels {
    pub fn new() -> Result<Self, String> {
        let ordinal = crate::cuda::selected_device_ordinal();
        let ctx =
            CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
        // A dedicated (non-default) stream: CUDA forbids stream capture on the
        // legacy default stream (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED at
        // begin_capture), so the graphed decode path requires this. Ordering is
        // unaffected: all engine work runs on THIS one stream, the offload copy
        // stream already synchronizes via explicit events, and host-visible
        // results go through ctx.synchronize() (device-wide).
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("engine stream: {e}"))?;
        let opts = CompileOptions {
            fmad: Some(false),
            arch: Some("compute_61"),
            ..Default::default()
        };
        // cudarc panics from inside its lazy NVRTC loader when libnvrtc is absent,
        // so `.map_err` never runs — catch it and report a normal Err so the caller
        // falls back to CPU instead of the process aborting. See the same guard in
        // `cuda::init_backend`; a driver-only Linux host (NVRTC ships with the CUDA
        // toolkit, not the driver) reaches this path once CUDA is in the default build.
        let ptx: Ptx =
            std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx_with_opts(KERNELS, opts))
                .map_err(|_| "CUDA NVRTC library not available".to_string())?
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
            q4_0_gemv: f("q4_0_gemv")?,
            q4_1_gemv: f("q4_1_gemv")?,
            nvfp4_gemv: f("nvfp4_gemv")?,
            q4k_gemv: f("q4k_gemv")?,
            q5k_gemv: f("q5k_gemv")?,
            q6k_gemv: f("q6k_gemv")?,
            q2k_gemv: f("q2k_gemv")?,
            iq4xs_gemv: f("iq4xs_gemv")?,
            q3k_gemv: f("q3k_gemv")?,
            q4k_gemm_batched: f("q4k_gemm_batched")?,
            q6k_gemm_batched: f("q6k_gemm_batched")?,
            quantize_q8k: f("quantize_q8k")?,
            rms_norm_quantize_q8k: f("rms_norm_quantize_q8k")?,
            silu_mul_quantize_q8k: f("silu_mul_quantize_q8k")?,
            rope: f("rope_rotate")?,
            kv_scatter: f("kv_scatter")?,
            attention: f("attention_decode")?,
            attention_sw: f("attention_decode_sw")?,
            silu_mul: f("silu_mul")?,
            silu_mul_quantize: f("silu_mul_quantize")?,
            geglu_mul: f("geglu_mul")?,
            soft_cap: f("soft_cap")?,
            f32_gemv: f("f32_gemv")?,
            scale_f32: f("scale_f32")?,
            residual_add: f("residual_add")?,
            scaled_axpy: f("scaled_axpy")?,
            argmax: f("argmax_f32")?,
            sample_gumbel: f("sample_gumbel")?,
            gemm_batched: f("q8_gemm_batched")?,
            rms_norm_batched: f("rms_norm_batched")?,
            rope_batched: f("rope_batched")?,
            kv_scatter_batched: f("kv_scatter_batched")?,
            attention_batched: f("attention_batched")?,
            kv_scatter_tree_batched: f("kv_scatter_tree_batched")?,
            attention_tree_batched: f("attention_tree_batched")?,
            argmax_batched: f("argmax_batched")?,
            attn_sk_scores: f("attn_sk_scores")?,
            attn_sk_scores_coalesced: f("attn_sk_scores_coalesced")?,
            attn_sk_partial: f("attn_sk_partial")?,
            attn_sk_combine: f("attn_sk_combine")?,
            flash_pref_scores: f("flash_pref_scores")?,
            flash_pref_partial: f("flash_pref_partial")?,
            flash_pref_scores_k8: f("flash_pref_scores_k8")?,
            flash_pref_partial_k8: f("flash_pref_partial_k8")?,
            ssm_l2_norm_per_head: f("ssm_l2_norm_per_head")?,
            ssm_conv1d: f("ssm_conv1d")?,
            ssm_delta_rule: f("ssm_delta_rule")?,
            sigmoid_mul: f("sigmoid_mul")?,
            embed_gather_q4k: f("embed_gather_q4k")?,
            embed_gather_q6k: f("embed_gather_q6k")?,
            embed_gather_q3k: f("embed_gather_q3k")?,
            embed_gather_q8_0: f("embed_gather_q8_0")?,
            rope_select: f("rope_select")?,
            ssm_gates: f("ssm_gates")?,
            deinterleave_qgate: f("deinterleave_qgate")?,
            attn_coalesced: std::env::var("CAMELID_ATTN_COALESCED")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            ctx,
            stream,
        })
    }
}

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

// ---- Free launch helpers (take explicit refs so callers can pass disjoint
// fields of the resident state without the `&self` whole-struct borrow). ----

/// Widen Q8_0 GGUF wire blocks (34 bytes: f16 scale + 32 i8) to the 36-byte layout
/// (f32 scale + 32 i8) used at the existing CPU-facing resident upload seam.
/// [`repack_q8_soa`] restores the original f16-scale footprint before GPU upload.
// Used by build_qwen35_resident (model.rs) — dead in a lib-only build until the M4
// generate_qwen35_cuda driver calls the builder from lib code (next).
#[allow(dead_code)]
pub(crate) fn widen_q8(bytes: &[u8]) -> Vec<u8> {
    let nb = bytes.len() / 34;
    let mut out = Vec::with_capacity(nb * 36);
    for b in 0..nb {
        let base = b * 34;
        let scale =
            crate::tensor::f16_bits_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&bytes[base + 2..base + 34]);
    }
    out
}

/// Repack CPU Q8_0 weight bytes (36-byte blocks: f32 scale + 32 i8) into the
/// compact GPU SoA layout the resident `q8_gemv` reads: all quants first
/// (`n_blocks * 32` i8), then all scales (`n_blocks` f16 bit patterns).
/// Quants-first keeps every block's 32 i8 16-byte aligned for `int4` loads;
/// compact scales restore the GGUF block's 34-byte footprint in VRAM.
pub(crate) fn repack_q8_soa(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 36;
    let mut out = vec![0u8; n * 32 + n * 2];
    let (quants, scales) = out.split_at_mut(n * 32);
    for b in 0..n {
        let blk = &bytes[b * 36..b * 36 + 36];
        let scale = f32::from_le_bytes(blk[0..4].try_into().expect("four-byte scale"));
        scales[b * 2..b * 2 + 2]
            .copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        quants[b * 32..b * 32 + 32].copy_from_slice(&blk[4..36]);
    }
    out
}

/// Repack one projection's wire bytes into the GPU layout its lane reads. Q8_0 is
/// repacked to the SoA layout `q8_gemv` reads; the K-quant lanes pass the RAW GGUF
/// super-block wire bytes straight through — `q4k_gemv` (144 B/sb) and `q6k_gemv`
/// (210 B/sb) expand the nibbles / unpack the packed scales on the fly. Keeping the
/// nibbles PACKED in VRAM is what lets 8B-Q4_K_M fit a 6 GB card: a host-side nibble
/// expansion to i8 would near-double the Q4_K footprint (256 vs 128 bytes/sb).
fn repack_for_lane(bytes: &[u8], q: ProjQuant) -> Vec<u8> {
    match q {
        ProjQuant::Q8_0 => repack_q8_soa(bytes),
        ProjQuant::Q6K => pad_q6k_blocks(bytes),
        ProjQuant::Q4K => swz_q4k_blocks(bytes),
        ProjQuant::Q5K | ProjQuant::Q2K | ProjQuant::Q3K => bytes.to_vec(),
        // IQ4_XS is read straight from the 136-byte GGUF wire (raw passthrough, like
        // Q5_K/Q2_K/Q3_K): the kernel unpacks nibbles/codebook on the fly.
        ProjQuant::IQ4XS => bytes.to_vec(),
    }
}

/// GPU-side Q4_K quant-byte swizzle (VRAM size unchanged, 144 B/sb): within each
/// 32-byte nibble group g, the four stride-8 bytes an aux lane consumes
/// (qs[g*32 + l], +8, +16, +24 for l in 0..8) are made CONTIGUOUS at
/// swz[g*32 + l*4 ..], so `q4k_gemv` reads one aligned i32 per lane and feeds it
/// straight to __dp4a (both nibble halves of the same word serve the group
/// pair 2g / 2g+1). A pure byte permutation — every product still forms from
/// the same operands into the same integer aux lane, so the gemv result is
/// BIT-IDENTICAL. The 16-byte header (d, dmin, packed scales) is untouched.
/// NOTE: this is the GEMV lane's layout only; `embed_gather_q4k` reads the
/// stock wire (the embedding table is uploaded raw, not via repack_for_lane).
pub(crate) fn swz_q4k_blocks(bytes: &[u8]) -> Vec<u8> {
    const WIRE: usize = 144;
    let blocks = bytes.len() / WIRE;
    let mut out = bytes.to_vec();
    for b in 0..blocks {
        let src = &bytes[b * WIRE + 16..(b + 1) * WIRE];
        let dst = &mut out[b * WIRE + 16..(b + 1) * WIRE];
        for g in 0..4 {
            for l in 0..8 {
                for k in 0..4 {
                    dst[g * 32 + l * 4 + k] = src[g * 32 + l + k * 8];
                }
            }
        }
    }
    out
}

/// Q6_K super-blocks are 210 B on the GGUF wire — not 16-byte aligned, which
/// forced the q6k GEMV into scalar byte loads (measured ~60-70 GB/s achieved).
/// Pad each block to 224 B at upload so every block — and its ql(+0)/qh(+128)/
/// scales(+192)/d(+208) sections — sits 16-aligned for vectorized uint4 loads.
/// The 210 payload bytes are untouched (pad is trailing zeros), so the gemv
/// result is bit-identical; cost is +6.7% VRAM on q6_K tensors only.
pub(crate) fn pad_q6k_blocks(bytes: &[u8]) -> Vec<u8> {
    const WIRE: usize = 210;
    const PADDED: usize = 224;
    let blocks = bytes.len() / WIRE;
    let mut out = vec![0u8; blocks * PADDED];
    for b in 0..blocks {
        out[b * PADDED..b * PADDED + WIRE].copy_from_slice(&bytes[b * WIRE..(b + 1) * WIRE]);
    }
    out
}

pub(crate) fn launch_rmsnorm(
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

pub(crate) fn launch_quantize(
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
pub(crate) fn launch_rmsnorm_quantize(
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
pub(crate) fn launch_gemv(
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
pub(crate) fn launch_gemv_residual(
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

/// Q4_K_M GEMV launch: same warp-per-row geometry as `launch_gemv`, but the input
/// is Q8_K (256-wide super-blocks: `n_sb` f32 scales + `n_sb*256` i8 quants) and the
/// weight is `repack_q4k_soa` bytes. Shared holds the staged Q8_K input vector
/// (`n_sb*256` i8 + `n_sb` f32) shared by all warps, then each warp's per-super-block
/// 9-int scratch (8 main lanes + 1 mins) for lane 0's ordered f32 reduction.
// As repack_q4k_soa: exercised by the bit-parity test; the production per-tensor
// dispatch into this launcher is the deferred end-to-end follow-up.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4k_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: n_sb*256 i8 + n_sb*4 f32; per-warp scratch: n_sb*9 i32.
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4 + warps_per_block * n_sb_u * 9 * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// IQ4_XS GEMV launch: same warp-per-row geometry as `launch_q6k_gemv`, but the
/// kernel accumulates f32 partials in registers (no per-warp integer scratch), so
/// shared memory is just the staged Q8_K activation. Weight is the RAW 136-byte
/// IQ4_XS wire (raw passthrough upload; the kernel unpacks the codebook on the fly).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_iq4xs_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input only: n_sb*256 i8 + n_sb*4 f32 (partials live in registers).
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q5_K_M GEMV launch: identical warp-per-row geometry + shared-memory shape to
/// `launch_q4k_gemv` (the per-warp scratch is the same 9 ints per super-block: 8
/// main lanes + 1 mins). Input is Q8_K (`n_sb` f32 scales + `n_sb*256` i8 quants);
/// weight is the RAW 176-byte Q5_K wire bytes (no SoA repack — the kernel expands
/// the low nibbles + folds in the qh fifth bit on the fly).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q5k_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: n_sb*256 i8 + n_sb*4 f32; per-warp scratch: n_sb*9 i32.
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4 + warps_per_block * n_sb_u * 9 * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q6_K GEMV launch: same warp-per-row geometry as `launch_q4k_gemv`. Input is
/// Q8_K (`n_sb` f32 scales + `n_sb*256` i8 quants); weight is the RAW 210-byte
/// Q6_K wire bytes (no SoA repack). Shared holds the staged Q8_K input vector
/// (`n_sb*256` i8 + `n_sb` f32) shared by all warps, then each warp's per-super-block
/// 8-int main-lane scratch for lane 0's ordered f32 reduction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_q6k_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: n_sb*256 i8 + n_sb*4 f32; per-warp scratch: n_sb*8 i32.
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4 + warps_per_block * n_sb_u * 8 * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q2_K GEMV launch: same warp-per-row geometry as `launch_q6k_gemv`. Input is
/// Q8_K (`n_sb` f32 scales + `n_sb*256` i8 quants); weight is the RAW 84-byte
/// Q2_K wire bytes (no SoA repack). Shared holds the staged Q8_K input vector
/// (`n_sb*256` i8 + `n_sb` f32) shared by all warps, then each warp's per-super-block
/// 2-int scratch (isum, summs) for lane 0's ordered f32 reduction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_q2k_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: n_sb*256 i8 + n_sb*4 f32; per-warp scratch: n_sb*2 i32.
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4 + warps_per_block * n_sb_u * 2 * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q3_K GEMV launch: same warp-per-row geometry as `launch_q2k_gemv`. Input is
/// Q8_K (`n_sb` f32 scales + `n_sb*256` i8 quants); weight is the RAW 110-byte
/// Q3_K wire bytes (no SoA repack). Shared holds the staged Q8_K input vector plus
/// each warp's per-super-block 1-int scratch (isum) for lane 0's ordered f32
/// reduction (Q3_K has no mins term).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_q3k_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    n_sb: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let n_sb_u = n_sb as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: n_sb*256 i8 + n_sb*4 f32; per-warp scratch: n_sb*1 i32.
        shared_mem_bytes: n_sb_u * 256 + n_sb_u * 4 + warps_per_block * n_sb_u * 4,
    };
    let (r, nb) = (rows as i32, n_sb as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q4_0 GEMV launch: same warp-per-row geometry as `launch_gemv` (q8). Input is
/// Q8_0 (`blocks_per_row` f32 scales + `blocks_per_row*32` i8 quants); weight is the
/// RAW 18-byte Q4_0 wire bytes (no SoA repack). Shared holds the staged Q8_0 input
/// (`bpr*32` i8 + `bpr` f32) shared by all warps, then each warp's per-block f32 term
/// scratch for lane 0's ordered reduction (mirrors q8_gemv).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let bpr = blocks_per_row as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged input: bpr*32 i8 + bpr*4 f32; per-warp scratch: bpr f32 terms.
        shared_mem_bytes: bpr * 32 + bpr * 4 + warps_per_block * bpr * 4,
    };
    let (r, nb) = (rows as i32, blocks_per_row as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Q4_1 GEMV launch: identical geometry + shared layout to `launch_q4_0_gemv` (Q8_0
/// activation, raw 20-byte Q4_1 wire, no SoA repack); only the kernel `f` differs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_q4_1_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let bpr = blocks_per_row as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: bpr * 32 + bpr * 4 + warps_per_block * bpr * 4,
    };
    let (r, nb) = (rows as i32, blocks_per_row as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Typed failure for [`launch_nvfp4_gemv`]. `OddBlocksPerRow` is the lane-native
/// I-k-div guard — an odd Q8_0-block count (`in_dim % 64 != 0`) cannot form whole
/// 64-value NVFP4 superblocks, so the launcher refuses it rather than mis-index;
/// `Driver` wraps an ordinary CUDA launch error. Kept distinct from the bare
/// `DriverError` the sibling launchers return so the divisibility refusal is a
/// distinguishable, directly-testable variant (`nvfp4_gemv_requires_even_q8_blocks`).
#[derive(Debug)]
pub(crate) enum Nvfp4LaunchError {
    OddBlocksPerRow(usize),
    Driver(cudarc::driver::DriverError),
}

impl std::fmt::Display for Nvfp4LaunchError {
    fn fmt(&self, fmtr: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Nvfp4LaunchError::OddBlocksPerRow(bpr) => write!(
                fmtr,
                "NVFP4 GEMV blocks_per_row {bpr} is odd (in_dim % 64 != 0); one 64-value \
                 superblock spans two 32-value Q8_0 activation blocks"
            ),
            Nvfp4LaunchError::Driver(e) => write!(fmtr, "NVFP4 GEMV launch: {e}"),
        }
    }
}

impl std::error::Error for Nvfp4LaunchError {}

/// NVFP4 GEMV launch: warp-per-row geometry like `launch_q4_0_gemv` (Q8_0
/// activation), but the weight is RAW 36-byte NVFP4 superblock wire and one
/// superblock spans TWO Q8_0 activation blocks, so the per-warp ordered-sum scratch
/// holds `2*blocks_per_row` f32 sub-block terms (4 per superblock) instead of
/// `blocks_per_row`. `blocks_per_row` is the Q8_0 activation block count
/// (`in_dim/32`) and MUST be even (`in_dim % 64 == 0`): a lone 32-value activation
/// block cannot pair into a 64-value superblock, so an odd count refuses TYPED
/// (I-k-div lane-native guard; the file-parse boundary already refuses non-%64
/// NVFP4 tensors, so this cannot fire in production — it exists for defense in depth
/// and the direct refusal test). `residual != 0` fuses the post-projection add.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_nvfp4_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), Nvfp4LaunchError> {
    // Fail-closed in EVERY build profile (not a debug_assert, which would panic
    // before this typed refusal could be observed): an odd Q8_0-block count cannot
    // form whole 64-value superblocks. This is the directly-tested I-k-div guard
    // (nvfp4_gemv_requires_even_q8_blocks); in production gemma_proj_gemv proves the
    // odd case impossible (parse refuses non-%64 NVFP4 first-dims) and maps it to
    // unreachable!, which is the loud developer-facing failure for that path.
    if !blocks_per_row.is_multiple_of(2) {
        return Err(Nvfp4LaunchError::OddBlocksPerRow(blocks_per_row));
    }
    let block = 256u32;
    let warps_per_block = block / 32;
    let bpr = blocks_per_row as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        // staged Q8_0 input: bpr*32 i8 + bpr*4 f32; per-warp scratch: 2*bpr f32
        // sub-block terms (4 per NVFP4 superblock == 2 per activation block).
        shared_mem_bytes: bpr * 32 + bpr * 4 + warps_per_block * 2 * bpr * 4,
    };
    let (r, nb) = (rows as i32, blocks_per_row as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&nb)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(Nvfp4LaunchError::Driver)
}

/// Per-projection GEMV dispatch: picks the kernel + activation buffers + contraction
/// unit by the projection's quant lane. `cols` is the contraction dimension (input
/// width); Q8_0 reads `cols/32` compact 34-byte blocks from `q8_0_*`, the K-quant lanes read
/// `cols/256` super-blocks from `q8k_*`. `residual != 0` fuses the post-projection
/// residual add into the GEMV (only valid when `out` is the residual/hidden buffer).
#[allow(clippy::too_many_arguments)]
fn dispatch_gemv(
    s: &Arc<CudaStream>,
    kern: &CudaResidentKernels,
    lane: ProjQuant,
    q8_0_scales: &CudaSlice<f32>,
    q8_0_quants: &CudaSlice<i8>,
    q8k_scales: &CudaSlice<f32>,
    q8k_quants: &CudaSlice<i8>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    match lane {
        ProjQuant::Q8_0 => {
            if residual != 0 {
                launch_gemv_residual(
                    s,
                    &kern.gemv,
                    q8_0_scales,
                    q8_0_quants,
                    weight,
                    rows,
                    cols / 32,
                    out,
                )
            } else {
                launch_gemv(
                    s,
                    &kern.gemv,
                    q8_0_scales,
                    q8_0_quants,
                    weight,
                    rows,
                    cols / 32,
                    out,
                )
            }
        }
        ProjQuant::Q4K => launch_q4k_gemv(
            s,
            &kern.q4k_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
        ProjQuant::Q5K => launch_q5k_gemv(
            s,
            &kern.q5k_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
        ProjQuant::Q6K => launch_q6k_gemv(
            s,
            &kern.q6k_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
        ProjQuant::Q2K => launch_q2k_gemv(
            s,
            &kern.q2k_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
        ProjQuant::Q3K => launch_q3k_gemv(
            s,
            &kern.q3k_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
        ProjQuant::IQ4XS => launch_iq4xs_gemv(
            s,
            &kern.iq4xs_gemv,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            out,
            residual,
        ),
    }
}

/// Per-projection batched GEMM dispatch. The activation buffers are token-major:
/// Q8_0 uses 32-value blocks, while Q4_K/Q6_K consume Q8_K super-blocks.
#[allow(clippy::too_many_arguments)]
fn dispatch_gemm_batched(
    s: &Arc<CudaStream>,
    kern: &CudaResidentKernels,
    lane: ProjQuant,
    q8_0_scales: &CudaSlice<f32>,
    q8_0_quants: &CudaSlice<i8>,
    q8k_scales: &CudaSlice<f32>,
    q8k_quants: &CudaSlice<i8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    match lane {
        ProjQuant::Q8_0 => launch_gemm_batched(
            s,
            &kern.gemm_batched,
            q8_0_scales,
            q8_0_quants,
            weight,
            rows,
            cols / 32,
            k_tokens,
            out,
        ),
        ProjQuant::Q4K => launch_kquant_gemm_batched(
            s,
            &kern.q4k_gemm_batched,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            k_tokens,
            9,
            out,
        ),
        ProjQuant::Q6K => launch_kquant_gemm_batched(
            s,
            &kern.q6k_gemm_batched,
            q8k_scales,
            q8k_quants,
            weight,
            rows,
            cols / 256,
            k_tokens,
            8,
            out,
        ),
        _ => unreachable!("unsupported batched projection lane: {lane:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_batched_for_lanes(
    s: &Arc<CudaStream>,
    kern: &CudaResidentKernels,
    x: &CudaSlice<f32>,
    q8_0_quants: &mut CudaSlice<i8>,
    q8_0_scales: &mut CudaSlice<f32>,
    q8k_quants: &mut CudaSlice<i8>,
    q8k_scales: &mut CudaSlice<f32>,
    cols: usize,
    k_tokens: usize,
    lanes: &[ProjQuant],
) -> Result<(), cudarc::driver::DriverError> {
    if lanes.contains(&ProjQuant::Q8_0) {
        launch_quantize(
            s,
            &kern.quantize,
            x,
            q8_0_quants,
            q8_0_scales,
            k_tokens * (cols / 32),
        )?;
    }
    if lanes.iter().any(|q| q.needs_q8k()) {
        launch_quantize_q8k(
            s,
            &kern.quantize_q8k,
            x,
            q8k_quants,
            q8k_scales,
            k_tokens * (cols / 256),
        )?;
    }
    Ok(())
}

/// Standalone Q8_K activation quantize: f32 row `[n_sb*256]` -> `n_sb` Q8_K blocks
/// (scales `[n_sb]`, quants `[n_sb*256]` i8). One thread per 256-block.
pub(crate) fn launch_quantize_q8k(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_sb: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32; // one block per super-block, one thread per element
    let cfg = LaunchConfig {
        grid_dim: (n_sb as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb = n_sb as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(quants).arg(scales).arg(&nb);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Fused RMS-norm + Q8_K quantize: stages the `n`-element row in shared for the
/// in-order sum, then quantizes 256-wide blocks straight from shared. K-quant
/// analog of `launch_rmsnorm_quantize`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rmsnorm_quantize_q8k(
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
        grid_dim: ((n as u32) / 256, 1, 1), // one block per Q8_K super-block
        block_dim: (block, 1, 1),
        shared_mem_bytes: (n as u32) * 4,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(w).arg(quants).arg(scales).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Fused SiLU(gate)*up + Q8_K quantize: one thread per 256-block. K-quant analog
/// of `launch_silu_mul_quantize`.
pub(crate) fn launch_silu_mul_quantize_q8k(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_sb: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32; // one block per super-block, one thread per element
    let cfg = LaunchConfig {
        grid_dim: (n_sb as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb = n_sb as i32;
    let mut b = s.launch_builder(f);
    b.arg(gate).arg(up).arg(quants).arg(scales).arg(&nb);
    unsafe { b.launch(cfg) }.map(|_| ())
}

// Fused SiLU*up + Q8_0 quantize (F3): one thread per 32-block, no shared memory.
pub(crate) fn launch_silu_mul_quantize(
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
pub(crate) fn launch_gemm_batched(
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

/// Batched Q4_K/Q6_K GEMM. Each warp owns one output row and keeps the
/// per-(token,super-block) integer partials in shared memory so lane 0 can
/// reproduce the corresponding GEMV's ordered f32 accumulation exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kquant_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    n_sb: usize,
    k_tokens: usize,
    aux_lanes: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    const SHARED_BUDGET: u32 = 46 * 1024;
    let input_bytes = (k_tokens as u32) * (n_sb as u32) * 260;
    let per_warp_bytes = (k_tokens as u32) * (n_sb as u32) * (aux_lanes as u32) * 4;
    assert!(
        input_bytes + per_warp_bytes <= SHARED_BUDGET,
        "K-quant batch does not fit the shared-memory budget"
    );
    let warps_per_block = ((SHARED_BUDGET - input_bytes) / per_warp_bytes.max(1)).clamp(1, 8);
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (warps_per_block * 32, 1, 1),
        shared_mem_bytes: input_bytes + warps_per_block * per_warp_bytes,
    };
    let (r, ns, kt) = (rows as i32, n_sb as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&ns)
        .arg(&kt)
        .arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rms_norm_batched(
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
pub(crate) fn launch_rope_batched(
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
pub(crate) fn launch_kv_scatter_batched(
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
pub(crate) fn launch_attention_batched(
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
    splitk_active: i32,
) -> Result<(), cudarc::driver::DriverError> {
    // Shared = query (head_dim) + scores (longest prefix = base + k) + weighted-V partials
    // (max_groups * head_dim, the decode-parity G-group reduction scratch).
    let max_groups = (1024 / head_dim).max(1);
    let shared = ((head_dim + base_position + k + max_groups * head_dim) as u32) * 4;
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
        .arg(&ki)
        .arg(&splitk_active);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kv_scatter_tree_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u16>,
    node_kvslot: &CudaSlice<i32>,
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
    let (nkv, hd, mp, ptd, ki) = (
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
        per_token_dim as i32,
        k as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(src)
        .arg(cache)
        .arg(node_kvslot)
        .arg(&nkv)
        .arg(&hd)
        .arg(&mp)
        .arg(&ptd)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_tree_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u16>,
    cache_v: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    ancestor_bits: &CudaSlice<u32>,
    words: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_position: usize,
    max_pos: usize,
    scale: f32,
    q_per_token: usize,
    k: usize,
    splitk_active: i32,
) -> Result<(), cudarc::driver::DriverError> {
    // Shared = query (head_dim) + scores (<= base + k) + slot indices (<= base + k) + weighted-V
    // partials (max_groups * head_dim, the decode-parity G-group reduction scratch).
    // scores are f32 and slots are i32, both 4 bytes ⇒ 2*(base+k) words past head_dim.
    let max_groups = (1024 / head_dim).max(1);
    let shared = ((head_dim + 2 * (base_position + k) + max_groups * head_dim) as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((k * n_heads) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: shared,
    };
    let (wd, nh, nkv, hd, bp, mp, qpt, ki) = (
        words as i32,
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
        .arg(ancestor_bits)
        .arg(&wd)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&bp)
        .arg(&mp)
        .arg(&scale)
        .arg(&qpt)
        .arg(&ki)
        .arg(&splitk_active);
    unsafe { b.launch(cfg) }.map(|_| ())
}

pub(crate) fn launch_argmax_batched(
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
pub(crate) fn launch_rope(
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
pub(crate) fn launch_rms_norm_per_head(
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

// ---- qwen35 (Ornith) SSM + full-attn launch helpers ------------------------
// Single source of truth for the qwen35-specific kernel launches; the parity tests
// in runnable/model.rs proved these exact configs (bit-identical SSM layer, 6e-3
// full-attn layer), and forward_pass calls the SAME helpers.

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_gates(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    beta_raw: &CudaSlice<f32>,
    alpha_raw: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    glog_out: &mut CudaSlice<f32>,
    nv: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (nv as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nvi = nv as i32;
    let mut b = s.launch_builder(f);
    b.arg(beta_raw)
        .arg(alpha_raw)
        .arg(dt_bias)
        .arg(a)
        .arg(beta_out)
        .arg(glog_out)
        .arg(&nvi);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_conv1d(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    conv_w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    conv_state: &mut CudaSlice<f32>,
    conv_out: &mut CudaSlice<f32>,
    conv_dim: usize,
    d_conv: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (cdi, dci) = (conv_dim as i32, d_conv as i32);
    let mut b = s.launch_builder(f);
    b.arg(conv_w)
        .arg(x)
        .arg(conv_state)
        .arg(conv_out)
        .arg(&cdi)
        .arg(&dci);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Per-head L2 norm in place over `buf[offset .. offset + head_count*head_dim]`.
pub(crate) fn launch_ssm_l2_norm_per_head(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    buf: &mut CudaSlice<f32>,
    offset: usize,
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (head_count as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: (head_dim as u32) * 4,
    };
    let dsi = head_dim as i32;
    let mut view = buf.slice_mut(offset..offset + head_count * head_dim);
    let mut b = s.launch_builder(f);
    b.arg(&mut view).arg(&dsi).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Gated delta-rule + gated RMSNorm. `conv_out` holds [q(key_dim)|k(key_dim)|v(value_dim)]
/// (q/k already L2-normed); the kernel arg order is `state, k_conv, q_conv, v_conv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_delta_rule(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    state: &mut CudaSlice<f32>,
    conv_out: &CudaSlice<f32>,
    key_dim: usize,
    value_dim: usize,
    z: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    glog: &CudaSlice<f32>,
    ssm_norm: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    d_state: usize,
    nk: usize,
    nv: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 1),
        block_dim: (d_state as u32, 1, 1),
        shared_mem_bytes: (3 * d_state as u32) * 4,
    };
    let (dsi, nki) = (d_state as i32, nk as i32);
    let qv = conv_out.slice(0..key_dim);
    let kv = conv_out.slice(key_dim..2 * key_dim);
    let vv = conv_out.slice(2 * key_dim..2 * key_dim + value_dim);
    let mut b = s.launch_builder(f);
    b.arg(state)
        .arg(&kv)
        .arg(&qv)
        .arg(&vv)
        .arg(z)
        .arg(beta)
        .arg(glog)
        .arg(ssm_norm)
        .arg(out)
        .arg(&dsi)
        .arg(&nki)
        .arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Split the fused per-head [query(head_dim)|gate(head_dim)] x n_heads into q / gate.
pub(crate) fn launch_deinterleave_qgate(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    qg: &CudaSlice<f32>,
    q_out: &mut CudaSlice<f32>,
    gate_out: &mut CudaSlice<f32>,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (((n_heads * head_dim) as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hdi) = (n_heads as i32, head_dim as i32);
    let mut b = s.launch_builder(f);
    b.arg(qg).arg(q_out).arg(gate_out).arg(&nh).arg(&hdi);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// `out[i] *= sigmoid(gate[i])` (qwen35 full-attention output gate).
pub(crate) fn launch_sigmoid_mul(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    out: &mut CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    n: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let ni = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(out).arg(gate).arg(&ni);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kv_scatter(
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

// Max splits the split-K decode attention may use (scratch in CudaResidentDecode is
// sized to this), and the context length above which it is used. Below the threshold the
// one-block-per-head `launch_attention` is cheaper (one launch, no scratch round-trip);
// above it, split-K's n_heads x n_splits grid is needed to fill the SMs.
// SIROCCO Lane K: raised 16 -> 32. The split-K V read (attn_sk_partial) uses only head_dim
// of 256 block threads, so it is occupancy-limited by the split count at long context; n_splits=32
// lifts the V-read bandwidth ~+19% (microbench) where the cap of 16 pinned it. MUST stay in lockstep
// with the two CUDA verify emulations (attention_batched, attention_tree_batched) or decode != verify.
const SPLITK_MAX: usize = 32;
const SPLITK_THRESHOLD: usize = 512;

/// Whether spec-verify must EMULATE the split-K attention reduction to stay token-identical to
/// plain greedy decode. Mirrors the plain-decode dispatch (`!graph_capture && attn_shared >
/// SPLITK_THRESHOLD`): on the LIVE (non-graph-captured) decode path, `position_count >
/// SPLITK_THRESHOLD` runs split-K, so the verify kernels must reproduce its exact chunked reduction
/// above the threshold. A graph-captured greedy decode skips split-K (uses the G-group
/// attention_decode), so when CUDA graphs drive greedy decode the verify must STAY G-group. Passed
/// to the verify kernels as `splitk_active`; the per-token `pc > SPLITK_THRESHOLD` test is done
/// in-kernel so a verify batch straddling the boundary picks the right path per token.
///
/// SCOPE: `CAMELID_ATTN_COALESCED` additionally re-associates split-K's per-position dot (Pass 1
/// warp-shuffle), which this emulation does NOT reproduce — so the > SPLITK_THRESHOLD lossless
/// guarantee holds only on the default non-coalesced path. The kernel-parity test asserts that.
fn splitk_verify_active() -> bool {
    !cuda_graphs_enabled()
}

/// Split-K decode attention: grid = n_heads x n_splits (vs one block per head), so the
/// 30 SMs fill even with 32 heads. Three passes: (1) chunk scores + chunk max, (2) exp
/// with the EXACT global max + chunk exp-sum + chunk unnormalized weighted-V, (3) ordered
/// combine. TOKEN-PARITY: dot and global max are bit-identical; the cross-split sum
/// re-associates exactly as the (parity-passing) Stage-2 weighted-V split. Verified.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_splitk(
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
        // Env-gated coalesced K-dot (CAMELID_ATTN_COALESCED). Identical signature,
        // shared-mem and grid; only the K access pattern differs. Default OFF.
        let scores_fn = if k.attn_coalesced {
            &k.attn_sk_scores_coalesced
        } else {
            &k.attn_sk_scores
        };
        let mut b = s.launch_builder(scores_fn);
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
        b.arg(&mut *lsum_buf)
            .arg(&mut *acc_buf)
            .arg(out)
            .arg(&nh)
            .arg(&hd)
            .arg(&ns);
        unsafe { b.launch(cfg) }?;
    }
    Ok(())
}

fn flash_prefill_enabled() -> bool {
    std::env::var("CAMELID_FLASH_PREFILL").is_ok_and(|v| v != "0")
}

/// SIROCCO Phase P M1: flash prefill attention (opt-in). The split-K decode reduction extended to
/// the whole `k_tokens` query chunk so each key's K/V is loaded/dequantized ONCE and reused across
/// all query rows of the head (~k_tokens x less K/V DRAM traffic than attention_batched, which uses
/// one block per (query-token, head)). Buffers are flat by (t*n_heads+head) so Pass 3 reuses
/// attn_sk_combine unchanged with n_heads = k_tokens*n_heads. TOKEN-PARITY (split-K reassociation
/// over the chunk length) -- gated by the multi-prompt oracle; prefill-only (verify untouched).
/// Requires q_per_token == n_heads*head_dim (the combine writes out[(t*n_heads+head)*head_dim+d]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_flash_prefill(
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
    base_position: usize,
    k_tokens: usize,
    q_per_token: usize,
    max_pos: usize,
    scale: f32,
) -> Result<(), cudarc::driver::DriverError> {
    // Chunk length = full range every token could attend; per-token causal mask lives in the kernel.
    let position_count = base_position + k_tokens;
    let n_splits = position_count.div_ceil(256).clamp(2, SPLITK_MAX);
    let (nh, nkv, hd, bp, kt, qpt, mp, ns) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        base_position as i32,
        k_tokens as i32,
        q_per_token as i32,
        max_pos as i32,
        n_splits as i32,
    );
    let pc = position_count as i32;
    let block: u32 = 256;
    // Pass 1: per (head, split), all k_tokens scores (K[p] dequant reused). shared = k_tokens*head_dim.
    {
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_splits as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (k_tokens * head_dim) as u32 * 4,
        };
        // M-C3: byte-identical const-k=8 unroll for the common chunk (registers, not local mem).
        let scores_fn = if k_tokens == 8 {
            &k.flash_pref_scores_k8
        } else {
            &k.flash_pref_scores
        };
        let mut b = s.launch_builder(scores_fn);
        b.arg(q)
            .arg(cache_k)
            .arg(&mut *scores_buf)
            .arg(&mut *chunkmax_buf)
            .arg(&nh)
            .arg(&nkv)
            .arg(&hd)
            .arg(&bp)
            .arg(&kt)
            .arg(&qpt)
            .arg(&mp)
            .arg(&scale)
            .arg(&ns)
            .arg(&pc);
        unsafe { b.launch(cfg) }?;
    }
    // Pass 2: per (head, split), per-token exp(global max) + exp-sum + unnormalized weighted-V (V reuse).
    {
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_splits as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_fn = if k_tokens == 8 {
            &k.flash_pref_partial_k8
        } else {
            &k.flash_pref_partial
        };
        let mut b = s.launch_builder(partial_fn);
        b.arg(cache_v)
            .arg(&mut *scores_buf)
            .arg(&mut *chunkmax_buf)
            .arg(&mut *lsum_buf)
            .arg(&mut *acc_buf)
            .arg(&nh)
            .arg(&nkv)
            .arg(&hd)
            .arg(&bp)
            .arg(&kt)
            .arg(&mp)
            .arg(&ns)
            .arg(&pc);
        unsafe { b.launch(cfg) }?;
    }
    // Pass 3: ordered combine (REUSED attn_sk_combine), one block per (token,head) = k_tokens*n_heads.
    {
        let fh = (k_tokens * n_heads) as i32;
        let cfg = LaunchConfig {
            grid_dim: ((k_tokens * n_heads) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&k.attn_sk_combine);
        b.arg(&mut *lsum_buf)
            .arg(&mut *acc_buf)
            .arg(out)
            .arg(&fh)
            .arg(&hd)
            .arg(&ns);
        unsafe { b.launch(cfg) }?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention(
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

pub(crate) fn launch_silu_mul(
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

pub(crate) fn launch_residual(
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_f32_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((out_dim as u32).div_ceil(128).max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (i, o) = (in_dim as i32, out_dim as i32);
    let mut b = s.launch_builder(f);
    b.arg(w).arg(x).arg(out).arg(&i).arg(&o);
    unsafe { b.launch(cfg) }.map(|_| ())
}

pub(crate) fn launch_scale(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &mut CudaSlice<f32>,
    n: usize,
    factor: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(&n_i).arg(&factor);
    unsafe { b.launch(cfg) }.map(|_| ())
}

pub(crate) fn launch_argmax(
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

/// `launch_argmax` writing into a VIEW (one u32 slot of the device-side decode
/// loop's d_out_tokens ring) — same kernel, same math.
pub(crate) fn launch_argmax_at(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    logits: &CudaSlice<f32>,
    n: usize,
    out_idx: &mut cudarc::driver::CudaViewMut<u32>,
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

/// Dequantize ONE embedding row on-device: the token id is read from `token`
/// (a device u32 — the previous argmax slot or the host-fed prefill id) and the
/// f32 row lands in `out`. `f` selects the quant family's kernel.
pub(crate) fn launch_embed_gather(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    table: &CudaSlice<u8>,
    token: &cudarc::driver::CudaView<u32>,
    dim: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: ((dim as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let dim_i = dim as i32;
    let mut b = s.launch_builder(f);
    b.arg(table).arg(token).arg(&dim_i).arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Copy position `pos`'s cos/sin rows out of the resident all-positions rope
/// tables into the per-token buffers the rope kernel reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rope_select(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    cos_all: &CudaSlice<f32>,
    sin_all: &CudaSlice<f32>,
    pos: usize,
    half: usize,
    cos_out: &mut CudaSlice<f32>,
    sin_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((half as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let (p, h) = (pos as i32, half as i32);
    let mut b = s.launch_builder(f);
    b.arg(cos_all)
        .arg(sin_all)
        .arg(&p)
        .arg(&h)
        .arg(cos_out)
        .arg(sin_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_sample_gumbel(
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

/// STAMPEDE Phase 6 multi-stream overlap state (`CAMELID_CUDA_STREAMS`, default off).
/// Two side streams run the independent K chain (`side_a`: K gemv → k-norm → rope-K →
/// scatter, reused for FFN-up) and V chain (`side_b`: V gemv → scatter) of each Full
/// layer, joined back to the main stream by events before every dependent read. Every
/// kernel launches unchanged (same grid, same reduction order) — only the stream an
/// existing launch is enqueued on changes, so the math is bitwise-identical to the
/// single-stream path. Constructed in `new` ONLY when the flag is on: with the flag
/// off, no side stream or event exists and forward_pass enqueues the byte-identical
/// launch sequence it always has. (NOTE: the engine's context is ALREADY in cudarc
/// multi-stream mode either way — `CudaResidentKernels::new` makes the main stream
/// via `ctx.new_stream()` — so lazy construction buys provable flag-off inertness,
/// not a mode switch. What `disable_event_tracking` in `new` removes is cudarc's
/// per-slice-arg event bookkeeping on this context's launches, for the flag-on
/// engine only.) The five events are re-recorded every layer; `cuEventRecord`
/// overwrites and each wait is enqueued (host order) after the record it consumes,
/// so reuse across layers and tokens is correct — no per-token churn.
struct StreamOverlap {
    side_a: std::sync::Arc<CudaStream>,
    side_b: std::sync::Arc<CudaStream>,
    /// attn norm+quantize done on main → side gemvs may read `d_in_*`/`d_q8k_*`.
    ev_act: CudaEvent,
    /// K chain done on side_a → attention on main may read `cache_k[li]`.
    ev_k: CudaEvent,
    /// V chain done on side_b → attention on main may read `cache_v[li]`.
    ev_v: CudaEvent,
    /// ffn norm+quantize done on main → up gemv on side_a may read `d_in_*`/`d_q8k_*`.
    ev_ffn: CudaEvent,
    /// up gemv done on side_a → silu on main may read `d_up` (and overwrite `d_in_*`).
    ev_up: CudaEvent,
}

/// Per-projection quantization lane the resident decode dispatches on. Q8_0 is the
/// historical default (byte-identical to before); Q4K and Q6K are the K-quant lanes
/// added for Q4_K_M models (mixed quant — Q4_K projections plus Q6_K attn_v/ffn_down
/// and the Q6_K lm_head). The activation a projection consumes is Q8_0 for `Q8_0` and
/// Q8_K for `Q4K`/`Q6K`, so `needs_q8k()` lets the per-norm-point quantizer pick.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjQuant {
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Q2K,
    Q3K,
    IQ4XS,
}

impl ProjQuant {
    /// Whether the GEMV reads a Q8_K activation (true for the K-quant lanes).
    fn needs_q8k(self) -> bool {
        matches!(
            self,
            ProjQuant::Q4K
                | ProjQuant::Q5K
                | ProjQuant::Q6K
                | ProjQuant::Q2K
                | ProjQuant::Q3K
                | ProjQuant::IQ4XS
        )
    }

    /// Whether this lane has a resident K-token GEMM. Other K-quant families
    /// retain the proven serial prefill path until they receive parity-checked
    /// batched kernels of their own.
    fn supports_batched(self) -> bool {
        matches!(self, ProjQuant::Q8_0 | ProjQuant::Q4K | ProjQuant::Q6K)
    }

    /// Whether a device-side `embed_gather_*` kernel exists for this family. The
    /// qwen35 device-decode loop gathers the embedding row on the GPU, so a family
    /// without a gather kernel (Q5_K / Q2_K / IQ4_XS) must NOT be installed for
    /// device decode — the caller falls back to the host-fed loop (CPU dequant)
    /// instead. Kept in lockstep with the `embed_gather_*` dispatch in
    /// `forward_token_device`.
    pub(crate) fn has_device_embed_gather(self) -> bool {
        matches!(
            self,
            ProjQuant::Q8_0 | ProjQuant::Q4K | ProjQuant::Q6K | ProjQuant::Q3K
        )
    }
}

/// The seven projection quant types of one layer, in q,k,v,o,gate,up,down order.
type LayerQuants = [ProjQuant; 7];

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
    /// Per-projection quant lane (q,k,v,o,gate,up,down), so the forward picks the
    /// right GEMV kernel + activation quantizer per tensor. All `Q8_0` for a plain
    /// Q8_0 model (path stays byte-identical).
    quants: LayerQuants,
    /// Which qwen35 (Ornith) mixer this layer runs. `Full` for every non-qwen35 model
    /// (the existing attention path, byte-identical) and for qwen35's sparse
    /// full-attention layers; `Ssm` for qwen35's gated-delta-net layers, whose mixer
    /// replaces attention with the recurrent SSM compute. See [`forward_pass`].
    #[allow(dead_code)] // read by the SSM forward branch (wired next).
    kind: LayerKind,
}

/// qwen35 layer mixer discriminant on [`ResidentLayer`]. Default `Full` keeps every
/// existing architecture on the unchanged attention path.
#[allow(dead_code)] // `Ssm` is constructed by the qwen35 builder + read by `forward_pass` (wired next).
enum LayerKind {
    Full,
    Ssm(Box<SsmResident>),
}

/// Resident VRAM weights + persistent runtime state for one qwen35 gated-delta-net
/// (SSM) layer. Mirrors `Qwen35Kind::Ssm` / `Qwen35Cache` (src/runnable/model.rs): the
/// 5 projections are repacked per their quant lane like the dense path, the 4 small
/// tensors stay f32, and `conv_state` + `state` persist across tokens (never reset).
#[allow(dead_code)] // fields read by the SSM forward branch (wired next).
struct SsmResident {
    wqkv: CudaSlice<u8>,      // hidden -> conv_dim
    wqkv_gate: CudaSlice<u8>, // hidden -> value_dim (z gate)
    beta: CudaSlice<u8>,      // hidden -> num_v_heads
    alpha: CudaSlice<u8>,     // hidden -> num_v_heads
    ssm_out: CudaSlice<u8>,   // value_dim -> hidden
    /// Per-projection quant lane (wqkv, wqkv_gate, beta, alpha, ssm_out).
    quants: [ProjQuant; 5],
    conv1d: CudaSlice<f32>, // [conv_dim * d_conv], channel-major [c*d_conv + tap]
    dt_bias: CudaSlice<f32>, // [num_v_heads]
    a: CudaSlice<f32>,      // [num_v_heads]
    ssm_norm: CudaSlice<f32>, // [d_state]
                            // NOTE: the persistent conv ring buffer + recurrent state live in engine-level
                            // `ssm_conv_state` / `ssm_state` Vecs (indexed by layer), NOT here — so the SSM
                            // forward can borrow the state mutably while borrowing this layer's `ssm_norm`
                            // immutably (disjoint engine fields). See `CudaResidentDecode`.
}

/// qwen35 gated-delta-net dimensions + the per-token SSM/full scratch, allocated by the
/// qwen35 builder. `None` on `CudaResidentDecode` for every other architecture, so no
/// extra VRAM and no path change for non-qwen35 models.
#[allow(dead_code)] // read by the SSM forward branch + builder (wired next).
struct Qwen35Gpu {
    d_state: usize,
    d_conv: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    // per-token scratch (reused across layers)
    d_qkv: CudaSlice<f32>,       // conv_dim  (wqkv output / conv input)
    d_conv_out: CudaSlice<f32>,  // conv_dim  (post-conv, sliced into q|k|v)
    d_z: CudaSlice<f32>,         // value_dim (wqkv_gate output)
    d_beta: CudaSlice<f32>,      // num_v_heads (post-sigmoid)
    d_glog: CudaSlice<f32>,      // num_v_heads (pre-exp)
    d_ssm_mix: CudaSlice<f32>,   // value_dim (delta-rule output, before ssm_out)
    d_qgate: CudaSlice<f32>,     // 2*q_width (full-attn fused query+gate projection)
    d_gate_attn: CudaSlice<f32>, // q_width (deinterleaved attention gate)
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
    /// Quant lane of the output (lm_head) projection. Q6_K for Q4_K_M models.
    output_quant: ProjQuant,
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
    // SIROCCO Phase P M1 flash prefill scratch: k_tokens-major (flat (t*n_heads+head)), sized for
    // MAX_VERIFY_K query tokens so attn_sk_combine reuses these with n_heads = k*n_heads. Only
    // allocated/used when CAMELID_FLASH_PREFILL is on.
    d_flash_scores: CudaSlice<f32>, // MAX_VERIFY_K * n_heads * max_pos
    d_flash_chunkmax: CudaSlice<f32>, // MAX_VERIFY_K * n_heads * SPLITK_MAX
    d_flash_lsum: CudaSlice<f32>,   // MAX_VERIFY_K * n_heads * SPLITK_MAX
    d_flash_acc: CudaSlice<f32>,    // MAX_VERIFY_K * n_heads * SPLITK_MAX * head_dim
    d_proj: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_ffn_act: CudaSlice<f32>,
    d_in_scales: CudaSlice<f32>,
    d_in_quants: CudaSlice<i8>,
    /// Q8_K activation scratch (K-quant lanes): `max_in/256` f32 scales + `max_in` i8
    /// quants. Separate from the Q8_0 `d_in_*` so the Q8_0 path stays byte-identical.
    d_q8k_scales: CudaSlice<f32>,
    d_q8k_quants: CudaSlice<i8>,
    d_logits: CudaSlice<f32>,
    d_sampled: CudaSlice<u32>,
    d_cos: CudaSlice<f32>,
    d_sin: CudaSlice<f32>,
    /// Device-side decode loop (qwen35): resident quantized embedding table
    /// (wire bytes + quant family; q6_K rows 224 B-padded), the precomputed
    /// all-positions rope tables (max_pos x rope_dim/2 cos + sin, built once on
    /// the host with the VERBATIM qwen35_rope_tables math), the on-device
    /// generated-token ring (argmax writes d_out_tokens[step]; the NEXT step's
    /// embed_gather reads it directly — no per-token D2H/H2D round-trip), and a
    /// 1-slot buffer for host-fed (prefill) token ids.
    embd_table: Option<(CudaSlice<u8>, ProjQuant)>,
    d_rope_cos_all: Option<CudaSlice<f32>>,
    d_rope_sin_all: Option<CudaSlice<f32>>,
    d_out_tokens: Option<CudaSlice<u32>>,
    d_token_in: Option<CudaSlice<u32>>,
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
    /// Phase 6 multi-stream overlap (side streams + join events). `None` unless
    /// `CAMELID_CUDA_STREAMS` was on at build — the flag-off path constructs
    /// nothing and stays in cudarc's single-stream mode.
    overlap: Option<StreamOverlap>,
    /// Lazily-allocated K-batched scratch for the speculative-verify forward.
    verify_scratch: Option<VerifyScratch>,
    /// Lazily-allocated TREE-verify scratch (sized to `TREE_MAX_NODES`, wider than
    /// the linear `verify_scratch`) plus the per-node KV-slot / ancestor-bitset
    /// device buffers the tree kernels read. Allocated by `ensure_tree_scratch`.
    tree_scratch: Option<TreeScratch>,
    /// Shared GPU scratch for offloaded layers (None when every layer is resident).
    /// Allocated by `enable_offload_scratch` when the build decides to offload.
    offload: Option<OffloadState>,
    /// qwen35 (Ornith) gated-delta-net dims + SSM/full per-token scratch. `None` for
    /// every other architecture (no extra VRAM, no path change). Set by the qwen35
    /// resident builder.
    qwen35: Option<Qwen35Gpu>,
    /// qwen35 SSM causal-conv ring buffers, per layer (`[conv_dim*(d_conv-1)]`); a
    /// 1-element placeholder for Full layers. Empty for non-qwen35 models. Persists
    /// across tokens — zeroed by `reset_qwen35_state` at the start of each generation.
    ssm_conv_state: Vec<CudaSlice<f32>>,
    /// qwen35 SSM recurrent per-head state, per layer (`[num_v_heads*d_state*d_state]`);
    /// 1-element placeholder for Full layers. Empty for non-qwen35. Persists across
    /// tokens — zeroed by `reset_qwen35_state`.
    ssm_state: Vec<CudaSlice<f32>>,
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
    viqk: CudaSlice<i8>,
    visk: CudaSlice<f32>,
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

/// Tree-verify scratch: a `VerifyScratch` widened to `TREE_MAX_NODES` plus the
/// per-node KV-slot and ancestor-bitset device buffers the two tree kernels
/// read. Sized once for the maximum tree (`TREE_MAX_NODES` nodes, `words =
/// ceil(N/32)` ancestor words per node).
struct TreeScratch {
    sc: VerifyScratch,
    /// Per-node KV slot (absolute position) = base + BFS index. Re-uploaded per round.
    node_kvslot: CudaSlice<i32>,
    /// Flat ancestor bitset `[node][words]` (causal tree mask). Re-uploaded per round.
    ancestor_bits: CudaSlice<u32>,
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

/// Whether decode overlaps the independent K/V and FFN-up GEMV chains of each Full
/// layer on side CUDA streams (STAMPEDE Phase 6). **Default off** while the win is
/// unproven on this driver: the side streams flip cudarc into multi-stream event
/// tracking and WDDM may not co-schedule sub-100µs kernels, so the overlap is opt-in
/// until the +8% low-ctx gate passes. Bitwise-neutral either way — every kernel
/// launches unchanged; only the enqueue stream differs. Opt in with
/// `CAMELID_CUDA_STREAMS=1`.
fn cuda_streams_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_CUDA_STREAMS").ok().as_deref(),
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
        // Phase 6 side streams + join events, ONLY when opted in — keeps the default
        // path provably inert: nothing constructed, byte-identical launch sequence
        // (see `StreamOverlap` for why this is NOT a multi-stream-mode switch).
        let overlap = if cuda_streams_enabled() {
            // Rung B: without this, cudarc auto-records and drops a CudaEvent per
            // slice argument on every launch (~600 launches × ~7 args per token; the
            // context is in multi-stream mode regardless of this flag, so flag-off
            // engines pay it too). Measured effect of removing it was ~nil (Rung A
            // −9.5%/−2.5% low-ctx vs Rung B −9.0%/−4.2%) — the regression is WDDM
            // scheduling, not bookkeeping — but tracking-off stays: it is strictly
            // less host work and the ordering is owned anyway. With tracking off WE
            // own all cross-stream ordering: the per-layer ev_act/ev_k/ev_v/ev_ffn/
            // ev_up graph in forward_pass (side streams only run downstream of an
            // ev_act wait recorded after all prior main-stream work, so load-time
            // uploads and memsets are ordered transitively), plus the one-time drain
            // in `enable_offload_scratch` covering the offload copy stream's first
            // read of main-stream-zeroed scratch. Scope: THIS engine's CudaContext
            // only (gemma4/dg/runnable lanes build their own contexts), permanent
            // for the engine's lifetime — unlike gemma4's load-scoped disable.
            unsafe { k.ctx.disable_event_tracking() };
            let side = || {
                k.ctx
                    .new_stream()
                    .map_err(|e| format!("cuda-streams side stream: {e}"))
            };
            let ev = || {
                k.ctx
                    .new_event(None)
                    .map_err(|e| format!("cuda-streams event: {e}"))
            };
            let o = StreamOverlap {
                side_a: side()?,
                side_b: side()?,
                ev_act: ev()?,
                ev_k: ev()?,
                ev_v: ev()?,
                ev_ffn: ev()?,
                ev_up: ev()?,
            };
            if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
                eprintln!("[cuda-streams] armed: 2 side streams + 5 join events constructed");
            }
            Some(o)
        } else {
            if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
                eprintln!("[cuda-streams] off: single stream, no side streams constructed");
            }
            None
        };
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
            output_quant: ProjQuant::Q8_0,
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
            // Flash-prefill scratch: k_tokens-major, allocated only when opt-in (else 1-elem stub).
            d_flash_scores: alloc_f(if flash_prefill_enabled() {
                MAX_VERIFY_K * n_heads * max_pos
            } else {
                1
            })?,
            d_flash_chunkmax: alloc_f(if flash_prefill_enabled() {
                MAX_VERIFY_K * n_heads * SPLITK_MAX
            } else {
                1
            })?,
            d_flash_lsum: alloc_f(if flash_prefill_enabled() {
                MAX_VERIFY_K * n_heads * SPLITK_MAX
            } else {
                1
            })?,
            d_flash_acc: alloc_f(if flash_prefill_enabled() {
                MAX_VERIFY_K * n_heads * SPLITK_MAX * head_dim
            } else {
                1
            })?,
            d_proj: alloc_f(hidden)?,
            d_gate: alloc_f(ffn_dim)?,
            d_up: alloc_f(ffn_dim)?,
            d_ffn_act: alloc_f(ffn_dim)?,
            d_in_scales: alloc_f(max_in / 32)?,
            d_in_quants: s
                .alloc_zeros::<i8>(max_in)
                .map_err(|e| format!("alloc: {e}"))?,
            d_q8k_scales: alloc_f(max_in.div_ceil(256).max(1))?,
            d_q8k_quants: s
                .alloc_zeros::<i8>(max_in)
                .map_err(|e| format!("alloc: {e}"))?,
            d_logits: alloc_f(vocab)?,
            d_sampled: s.alloc_zeros::<u32>(1).map_err(|e| format!("alloc: {e}"))?,
            d_cos: alloc_f(rope_dim / 2)?,
            d_sin: alloc_f(rope_dim / 2)?,
            d_position: s.alloc_zeros::<i32>(1).map_err(|e| format!("alloc: {e}"))?,
            decode_graph: None,
            overlap,
            verify_scratch: None,
            tree_scratch: None,
            offload: None,
            qwen35: None,
            ssm_conv_state: Vec::new(),
            ssm_state: Vec::new(),
            embd_table: None,
            d_rope_cos_all: None,
            d_rope_sin_all: None,
            d_out_tokens: None,
            d_token_in: None,
            k,
        })
    }

    /// Upload one layer's weights (CPU Q8_0 36-byte blocks, compacted on upload) + norms.
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
        // Default: every layer resident in VRAM, all Q8_0 (unchanged behavior).
        self.set_layer_located(
            q,
            kk,
            v,
            o,
            gate,
            up,
            down,
            attn_norm,
            ffn_norm,
            None,
            None,
            true,
            [ProjQuant::Q8_0; 7],
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
        quants: LayerQuants,
    ) -> Result<(), String> {
        let ctx = &self.k.ctx;
        let s = &self.k.stream;
        let up_f = |b: &[f32]| s.clone_htod(b).map_err(|e| format!("htod: {e}"));
        let projections = [q, kk, v, o, gate, up, down];
        let (attn_norm, ffn_norm) = (up_f(attn_norm)?, up_f(ffn_norm)?);
        let q_norm_gpu = q_norm.map(up_f).transpose()?;
        let k_norm_gpu = k_norm.map(up_f).transpose()?;

        if resident {
            // Resident: each projection uploaded once to its own VRAM slice (repacked
            // into the layout its quant lane reads); no offload metadata.
            let vram = |i: usize| -> Result<CudaSlice<u8>, String> {
                s.clone_htod(&repack_for_lane(projections[i], quants[i]))
                    .map_err(|e| format!("htod: {e}"))
            };
            self.layers.push(ResidentLayer {
                q: vram(0)?,
                k: vram(1)?,
                v: vram(2)?,
                o: vram(3)?,
                gate: vram(4)?,
                up: vram(5)?,
                down: vram(6)?,
                attn_norm,
                ffn_norm,
                q_norm: q_norm_gpu,
                k_norm: k_norm_gpu,
                offloaded: None,
                quants,
                kind: LayerKind::Full,
            });
            return Ok(());
        }

        // Offloaded: repack all seven projections (each into its lane's layout) and lay
        // them out contiguously in one pinned host buffer so the per-forward transfer is
        // a single memcpy.
        let repacked: Vec<Vec<u8>> = projections
            .iter()
            .enumerate()
            .map(|(i, b)| repack_for_lane(b, quants[i]))
            .collect();
        // 16-byte-align each projection start so the resident GEMV kernels' wide
        // (uint4) wire loads are legal off any projection's view base (the q4k_gemv
        // super-block is 144 B = 9*16, so every block in a row stays 16-aligned once
        // the row base is; q6k uses byte loads so it is alignment-agnostic). Resident
        // tensors are separate 256-aligned device allocations, so this only matters for
        // the packed offload scratch path. Padding is at most 15 B per projection.
        let mut off = [0usize; 8];
        for (i, r) in repacked.iter().enumerate() {
            off[i + 1] = (off[i] + r.len() + 15) & !15;
        }
        let total = off[7];
        // Cacheable pinned host buffer (faster H2D than write-combined here), filled
        // with the seven projections laid out back-to-back.
        let mut packed = vec![0u8; total];
        for (i, r) in repacked.iter().enumerate() {
            packed[off[i]..off[i] + r.len()].copy_from_slice(r);
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
            quants,
            kind: LayerKind::Full,
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
        // One-time load-time drain BEFORE installing the state: the scratch buffers
        // were zeroed on the main stream, and the FIRST prefetch into each buffer
        // waits only on a fresh compute_done event (which reads as already-occurred)
        // — nothing else orders the copy stream's first write after the memset.
        // cudarc's auto event tracking used to insert that ordering; with
        // CAMELID_CUDA_STREAMS on, tracking is disabled for this engine's context
        // (see `new`), so make it explicit here. Draining first means a failed drain
        // leaves `self.offload` None (no half-armed state). Unconditional: one sync
        // at load time costs nothing per token.
        self.k
            .ctx
            .synchronize()
            .map_err(|e| format!("offload scratch drain: {e}"))?;
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

    pub fn set_output(
        &mut self,
        final_norm: &[f32],
        output_weight: &[u8],
        output_quant: ProjQuant,
    ) -> Result<(), String> {
        let s = &self.k.stream;
        self.final_norm = s.clone_htod(final_norm).map_err(|e| format!("htod: {e}"))?;
        self.output_weight = s
            .clone_htod(&repack_for_lane(output_weight, output_quant))
            .map_err(|e| format!("htod: {e}"))?;
        self.output_quant = output_quant;
        Ok(())
    }

    /// qwen35 (Ornith): attach the gated-delta-net dims and allocate the per-token
    /// SSM/full scratch. `head_v_dim` == `d_state` for Ornith.
    #[allow(clippy::too_many_arguments)]
    pub fn set_qwen35(
        &mut self,
        d_state: usize,
        d_conv: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
    ) -> Result<(), String> {
        let s = self.k.stream.clone();
        let af = |n: usize| -> Result<CudaSlice<f32>, String> {
            s.alloc_zeros::<f32>(n).map_err(|e| format!("alloc: {e}"))
        };
        self.qwen35 = Some(Qwen35Gpu {
            d_state,
            d_conv,
            num_k_heads,
            num_v_heads,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            d_qkv: af(conv_dim)?,
            d_conv_out: af(conv_dim)?,
            d_z: af(value_dim)?,
            d_beta: af(num_v_heads)?,
            d_glog: af(num_v_heads)?,
            d_ssm_mix: af(value_dim)?,
            d_qgate: af(2 * self.q_width)?,
            d_gate_attn: af(self.q_width)?,
        });
        Ok(())
    }

    /// Keep the per-layer `ssm_conv_state`/`ssm_state` Vecs length-synced with `layers`
    /// for a Full (non-SSM) qwen35 layer: a never-read 1-element placeholder.
    pub fn push_ssm_placeholders(&mut self) -> Result<(), String> {
        let s = self.k.stream.clone();
        self.ssm_conv_state
            .push(s.alloc_zeros::<f32>(1).map_err(|e| format!("alloc: {e}"))?);
        self.ssm_state
            .push(s.alloc_zeros::<f32>(1).map_err(|e| format!("alloc: {e}"))?);
        Ok(())
    }

    /// qwen35 SPARSE KV: free the KV cache buffer of every NON-attending (SSM) layer,
    /// replacing it with a 1-element placeholder. Only the full-attention layers
    /// (`keep_full[li] == true`) keep a real `kv_width*max_pos` buffer. `new()` allocated
    /// dense KV for all `n_layers` (shared with every arch); this is called by the qwen35
    /// builder AFTER `new()` but BEFORE the weight uploads — while no weights are resident
    /// yet — so the freed dense KV (the caller then `release_async_pool()`s it back to the
    /// device) is reused by the 5+ GB of weights. With only 8/32 layers attending, KV
    /// shrinks ~4x, letting `max_pos` go ~4x higher in the same VRAM. The SSM placeholders
    /// are never read: the SSM forward arm skips kv_scatter/attention, and the qwen35
    /// driver only uses `forward_token` (never seed_layer/read_kv_layer). Absolute `li`
    /// indexing is preserved (the Vecs stay length `n_layers`).
    pub fn sparsify_kv(&mut self, keep_full: &[bool]) -> Result<(), String> {
        let s = self.k.stream.clone();
        for li in 0..self.n_layers {
            if !keep_full.get(li).copied().unwrap_or(false) {
                self.cache_k[li] = s
                    .alloc_zeros::<u16>(1)
                    .map_err(|e| format!("kv placeholder: {e}"))?;
                self.cache_v[li] = s
                    .alloc_zeros::<u16>(1)
                    .map_err(|e| format!("kv placeholder: {e}"))?;
            }
        }
        Ok(())
    }

    /// Zero every qwen35 SSM recurrent state + conv ring buffer and reset `filled`.
    /// MUST be called at the start of each generation — the SSM state persists across
    /// tokens AND across generate calls, so skipping this decodes turn 2 on stale state.
    pub fn reset_qwen35_state(&mut self) -> Result<(), String> {
        let s = self.k.stream.clone();
        for buf in self
            .ssm_conv_state
            .iter_mut()
            .chain(self.ssm_state.iter_mut())
        {
            if buf.len() > 1 {
                let zeros = vec![0.0f32; buf.len()];
                s.memcpy_htod(&zeros, buf)
                    .map_err(|e| format!("reset htod: {e}"))?;
            }
        }
        self.filled = 0;
        Ok(())
    }

    /// Upload one qwen35 gated-delta-net (SSM) layer plus its persistent conv ring and
    /// recurrent state. Q8_0 source bytes use the 36-byte CPU layout (`widen_q8`)
    /// and are compacted back to f16-scale SoA at upload; K-quant bytes are raw
    /// GGUF super-blocks. `quants` order is
    /// wqkv, wqkv_gate, beta, alpha, ssm_out; `ffn_quants` is gate, up, down.
    #[allow(clippy::too_many_arguments)]
    pub fn set_layer_ssm_qwen35(
        &mut self,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        wqkv: &[u8],
        wqkv_gate: &[u8],
        beta: &[u8],
        alpha: &[u8],
        ssm_out: &[u8],
        conv1d: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        ssm_norm: &[f32],
        ssm_quants: [ProjQuant; 5],
        ffn_quants: [ProjQuant; 3],
        conv_dim: usize,
        d_conv: usize,
        nv: usize,
        d_state: usize,
    ) -> Result<(), String> {
        let s = self.k.stream.clone();
        let up_u8 = |b: &[u8], q: ProjQuant| -> Result<CudaSlice<u8>, String> {
            s.clone_htod(&repack_for_lane(b, q))
                .map_err(|e| format!("htod: {e}"))
        };
        let up_f = |b: &[f32]| -> Result<CudaSlice<f32>, String> {
            s.clone_htod(b).map_err(|e| format!("htod: {e}"))
        };
        let ph = || s.clone_htod(&[0u8]).map_err(|e| format!("htod: {e}"));
        let ssm = SsmResident {
            wqkv: up_u8(wqkv, ssm_quants[0])?,
            wqkv_gate: up_u8(wqkv_gate, ssm_quants[1])?,
            beta: up_u8(beta, ssm_quants[2])?,
            alpha: up_u8(alpha, ssm_quants[3])?,
            ssm_out: up_u8(ssm_out, ssm_quants[4])?,
            quants: ssm_quants,
            conv1d: up_f(conv1d)?,
            dt_bias: up_f(dt_bias)?,
            a: up_f(a)?,
            ssm_norm: up_f(ssm_norm)?,
        };
        let gate_g = up_u8(gate, ffn_quants[0])?;
        let up_g = up_u8(up, ffn_quants[1])?;
        let down_g = up_u8(down, ffn_quants[2])?;
        let attn_norm_g = up_f(attn_norm)?;
        let ffn_norm_g = up_f(ffn_norm)?;
        // layer.quants drives the shared attn-norm+quantize (cols 0..3 = the SSM
        // activation consumers) and the FFN (cols 4..6 = gate/up/down).
        let quants = [
            ssm_quants[0],
            ssm_quants[1],
            ssm_quants[2],
            ssm_quants[3],
            ffn_quants[0],
            ffn_quants[1],
            ffn_quants[2],
        ];
        self.layers.push(ResidentLayer {
            q: ph()?,
            k: ph()?,
            v: ph()?,
            o: ph()?,
            gate: gate_g,
            up: up_g,
            down: down_g,
            attn_norm: attn_norm_g,
            ffn_norm: ffn_norm_g,
            q_norm: None,
            k_norm: None,
            offloaded: None,
            quants,
            kind: LayerKind::Ssm(Box::new(ssm)),
        });
        self.ssm_conv_state.push(
            s.alloc_zeros::<f32>(conv_dim * (d_conv - 1))
                .map_err(|e| format!("alloc: {e}"))?,
        );
        self.ssm_state.push(
            s.alloc_zeros::<f32>(nv * d_state * d_state)
                .map_err(|e| format!("alloc: {e}"))?,
        );
        Ok(())
    }

    fn supports_batched_layer_stack(&self) -> bool {
        !self.is_offloaded()
            && self.layers.iter().all(|layer| {
                matches!(&layer.kind, LayerKind::Full)
                    && layer.quants.iter().all(|q| q.supports_batched())
            })
    }

    fn batched_layer_token_cap(&self) -> usize {
        let has_kquant = self
            .layers
            .iter()
            .any(|layer| layer.quants.iter().any(|q| *q != ProjQuant::Q8_0));
        if !has_kquant {
            return MAX_VERIFY_K;
        }
        // Two tokens leave enough of the portable 46 KiB shared-memory
        // budget for eight warps on the 8192-wide 3B FFN. Four-token tiles
        // remain available for same-binary diagnostics where the dimensions
        // fit; larger models are clamped down instead of panicking at launch.
        let requested = std::env::var("CAMELID_CUDA_KQUANT_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 4);
        let max_cols = self.hidden.max(self.q_width).max(self.ffn_dim);
        let n_sb = max_cols / 256;
        let aux_lanes = if self
            .layers
            .iter()
            .any(|layer| layer.quants.contains(&ProjQuant::Q4K))
        {
            9
        } else {
            8
        };
        (1..=requested)
            .rev()
            .find(|&k| k * n_sb * (260 + aux_lanes * 4) <= 46 * 1024)
            .unwrap_or(1)
    }

    fn batched_verify_token_cap(&self) -> usize {
        let layer_cap = self.batched_layer_token_cap();
        if self.output_quant == ProjQuant::Q8_0 {
            return layer_cap;
        }
        let aux_lanes = if self.output_quant == ProjQuant::Q4K {
            9
        } else {
            8
        };
        let n_sb = self.hidden / 256;
        let output_cap = (1..=4)
            .rev()
            .find(|&k| k * n_sb * (260 + aux_lanes * 4) <= 46 * 1024)
            .unwrap_or(1);
        layer_cap.min(output_cap)
    }

    /// Whether prompt prefill can use the resident K-token layer stack. Q8_0,
    /// Q4_K, and Q6_K projections are supported; offloaded, SSM, and other
    /// quant families safely retain serial prefill.
    pub fn supports_batched_prefill(&self) -> bool {
        self.supports_batched_layer_stack()
    }

    /// Default policy after same-host Windows/WDDM benchmarking. Q8_0 batching
    /// remains a clear win; the parity-correct Q4_K/Q6_K tiles stay opt-in until
    /// a device shows a sustained gain over the mature serial GEMVs.
    pub fn prefers_batched_prefill(&self) -> bool {
        self.supports_batched_layer_stack()
            && self
                .layers
                .iter()
                .all(|layer| layer.quants.iter().all(|q| *q == ProjQuant::Q8_0))
    }

    /// Whether linear speculative verification can use the batched stack,
    /// including its final vocabulary projection.
    pub fn supports_batched_verify(&self) -> bool {
        self.supports_batched_layer_stack() && self.output_quant.supports_batched()
    }

    /// Tree verification still uses the legacy Q8-only layer stack.
    pub fn supports_tree_verify(&self) -> bool {
        !self.is_offloaded()
            && self.output_quant == ProjQuant::Q8_0
            && self.layers.iter().all(|layer| {
                matches!(&layer.kind, LayerKind::Full)
                    && layer.quants.iter().all(|q| *q == ProjQuant::Q8_0)
            })
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

    /// Layer count this engine was built for — the session's OWNED layer range, so a caller
    /// mapping engine slots to absolute layer ids can confirm the shard shapes agree before
    /// trusting the global engine's KV (see `recover_cpu_kv_from_cuda_resident`).
    pub fn n_layers(&self) -> usize {
        self.n_layers
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
        // Bounds first, as `seed_layer` does: the slices below would otherwise panic on an
        // out-of-range layer or an over-long read. Callers are seams that already fall back on
        // Err, so refusing is strictly better than aborting the process.
        if layer >= self.n_layers {
            return Err("read_kv_layer: layer out of range".into());
        }
        if n_positions > self.max_pos {
            return Err(format!(
                "read_kv_layer: {n_positions} positions exceeds resident capacity {}",
                self.max_pos
            ));
        }
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
    ///
    /// Error containment for the Phase 6 overlap: a `?` failure inside the body can
    /// return with side-stream kernels enqueued but not yet joined to main (the
    /// joins live after the dispatches). Callers may drop or rebuild the engine on
    /// error, and with event tracking disabled nothing else orders those in-flight
    /// side launches against later frees — so on an Err with the overlap armed,
    /// drain the context (best-effort) before propagating. Zero cost on Ok and on
    /// the flag-off path.
    #[allow(clippy::too_many_arguments)]
    fn forward_pass(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        compute_logits: bool,
        graph_capture: bool,
        device_inputs: bool,
    ) -> Result<(), String> {
        let r = self.forward_pass_inner(
            embedding,
            cos,
            sin,
            position,
            scale,
            compute_logits,
            graph_capture,
            device_inputs,
        );
        if r.is_err() && self.overlap.is_some() {
            let _ = self.k.ctx.synchronize();
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_pass_inner(
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
        // When true the per-token inputs (hidden/cos/sin/position) are ALREADY in
        // the device buffers — written by embed_gather/rope_select plus a 4-byte
        // async position upload (the device-side decode loop) — so the host
        // uploads are skipped. Unlike graph_capture, attention shared stays sized
        // to position+1 (each token still launches with live scalars).
        device_inputs: bool,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        let fused = resident_fusion_enabled();
        // STAMPEDE Phase 6: resolve the overlap side streams once per forward. `None`
        // keeps every launch on `s` exactly as today (flag off, graph capture, or
        // offload active — a third stream would contend with the copy engine). Cloned
        // Arcs rather than a held `&StreamOverlap` because `prefetch_offloaded`
        // (&mut self) is called inside the layer loop; join events are borrowed
        // per-statement instead, like the offload events.
        let ov_streams = match self.overlap.as_ref() {
            Some(o) if !graph_capture && self.offload.is_none() => {
                Some((o.side_a.clone(), o.side_b.clone()))
            }
            _ => None,
        };
        if ov_streams.is_some() {
            // Engaged-check for receipts: prove the ON leg actually ran the overlap
            // (a receipt without this line measured the single-stream path).
            static ENGAGED: std::sync::Once = std::sync::Once::new();
            ENGAGED.call_once(|| {
                if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
                    eprintln!(
                        "[cuda-streams] overlap ENGAGED: K chain + FFN-up on side_a, \
                         V chain on side_b, event-joined per Full layer"
                    );
                }
            });
        }
        let hb = self.hidden / 32; // hidden blocks
        let fb = self.ffn_dim / 32; // ffn blocks
        let qb = self.q_width / 32; // q_width blocks
        let attn_shared = if graph_capture {
            self.max_pos
        } else {
            position + 1
        };

        if !graph_capture && !device_inputs {
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
            // Per-projection quant lanes for this layer (q,k,v,o,gate,up,down).
            let lq = self.layers[li].quants;
            // attention norm + quantize. Produce the Q8_0 activation (existing fused/
            // unfused path, byte-identical) when any consumer is Q8_0, and the Q8_K
            // activation when any consumer is a K-quant lane. For an all-Q8_0 layer only
            // the Q8_0 branch runs, so the legacy path is unchanged.
            // The attn-norm activation feeds 3 consumers for a Full layer (q/k/v) but 4
            // for a qwen35 SSM layer (wqkv/wqkv_gate/beta/alpha — all read d_in_*/d_q8k_*),
            // so widen the consumer span to lq[0..4] for SSM. (O-proj lq[3] of a Full
            // layer consumes the attention output, quantized separately — not here.)
            let n_attn_consumers = if matches!(&self.layers[li].kind, LayerKind::Ssm(_)) {
                4
            } else {
                3
            };
            let attn_need_q8_0 = lq[..n_attn_consumers].contains(&ProjQuant::Q8_0);
            let attn_need_q8k = lq[..n_attn_consumers].iter().any(|q| q.needs_q8k());
            if attn_need_q8_0 {
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
            }
            if attn_need_q8k {
                launch_rmsnorm_quantize_q8k(
                    &s,
                    &self.k.rms_norm_quantize_q8k,
                    &self.d_hidden,
                    &self.layers[li].attn_norm,
                    &mut self.d_q8k_quants,
                    &mut self.d_q8k_scales,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
            }
            // qwen35 hybrid: SSM layers replace the whole attention mixer; full-attn
            // layers run the existing path plus a fused query+gate split and a sigmoid
            // output gate. Both kinds share the attn-norm+quantize above and the FFN below.
            match &self.layers[li].kind {
                LayerKind::Full => {
                    // Phase 6 overlap: publish the attn activation (`ev_act` on main,
                    // recorded after the norm+quantize above) to the side streams,
                    // then run the K chain on `side_a` and the V chain on `side_b`.
                    // Q stays on main — largest output, keeps the critical path warm.
                    // With overlap off both handles alias `s`: every launch below is
                    // enqueued exactly as today.
                    let (s_k, s_v): (&Arc<CudaStream>, &Arc<CudaStream>) =
                        if let Some((a, b)) = ov_streams.as_ref() {
                            let o = self.overlap.as_ref().expect("overlap state");
                            o.ev_act.record(&s).map_err(map)?;
                            a.wait(&o.ev_act).map_err(map)?;
                            b.wait(&o.ev_act).map_err(map)?;
                            (a, b)
                        } else {
                            (&s, &s)
                        };
                    // Q,K,V  (qwen35 full-attn: wq is the fused [query|gate] projection -> 2*q_width)
                    if let Some(q) = self.qwen35.as_mut() {
                        dispatch_gemv(
                            &s,
                            &self.k,
                            lq[0],
                            &self.d_in_scales,
                            &self.d_in_quants,
                            &self.d_q8k_scales,
                            &self.d_q8k_quants,
                            &wq,
                            2 * self.q_width,
                            self.hidden,
                            &mut q.d_qgate,
                            0,
                        )
                        .map_err(map)?;
                        launch_deinterleave_qgate(
                            &s,
                            &self.k.deinterleave_qgate,
                            &q.d_qgate,
                            &mut self.d_q,
                            &mut q.d_gate_attn,
                            self.n_heads,
                            self.head_dim,
                        )
                        .map_err(map)?;
                    } else {
                        dispatch_gemv(
                            &s,
                            &self.k,
                            lq[0],
                            &self.d_in_scales,
                            &self.d_in_quants,
                            &self.d_q8k_scales,
                            &self.d_q8k_quants,
                            &wq,
                            self.q_width,
                            self.hidden,
                            &mut self.d_q,
                            0,
                        )
                        .map_err(map)?;
                    }
                    dispatch_gemv(
                        s_k,
                        &self.k,
                        lq[1],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &wk,
                        self.kv_width,
                        self.hidden,
                        &mut self.d_k,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemv(
                        s_v,
                        &self.k,
                        lq[2],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &wv,
                        self.kv_width,
                        self.hidden,
                        &mut self.d_v,
                        0,
                    )
                    .map_err(map)?;
                    // Qwen3 QK-norm: per-head RMSNorm on Q and K after projection, before RoPE
                    if let (Some(ref qn), Some(ref kn)) =
                        (&self.layers[li].q_norm, &self.layers[li].k_norm)
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
                            s_k,
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
                        s_k,
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
                        s_k,
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
                        s_v,
                        &self.k.kv_scatter,
                        &self.d_v,
                        &mut self.cache_v[li],
                        &self.d_position,
                        self.n_kv_heads,
                        self.head_dim,
                        self.max_pos,
                    )
                    .map_err(map)?;
                    // Join: K and V (including their cache scatters) are now published;
                    // attention on main reads both caches, so it waits on the side
                    // chains here. These event waits also transitively order every
                    // later main-stream write of the shared activation scratch
                    // (`d_in_*`/`d_q8k_*`, next written by the O-proj quantize) after
                    // the side gemvs' reads — the WAR hazard the single stream hides.
                    if ov_streams.is_some() {
                        let o = self.overlap.as_ref().expect("overlap state");
                        o.ev_k.record(s_k).map_err(map)?;
                        o.ev_v.record(s_v).map_err(map)?;
                        s.wait(&o.ev_k).map_err(map)?;
                        s.wait(&o.ev_v).map_err(map)?;
                    }
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
                    // qwen35 full-attention sigmoid output gate: attn[i] *= sigmoid(gate[i]).
                    if let Some(q) = self.qwen35.as_ref() {
                        launch_sigmoid_mul(
                            &s,
                            &self.k.sigmoid_mul,
                            &mut self.d_attn,
                            &q.d_gate_attn,
                            self.q_width,
                        )
                        .map_err(map)?;
                    }
                    // O projection + residual. Input is the attention output (q_width wide):
                    // quantize it to the format the O lane reads, then project + add residual.
                    if lq[3] == ProjQuant::Q8_0 {
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
                    } else {
                        // K-quant O lane: Q8_K activation, fused-residual GEMV (bit-identical
                        // to gemv + residual_add — the kernel's residual arg adds onto d_hidden).
                        launch_quantize_q8k(
                            &s,
                            &self.k.quantize_q8k,
                            &self.d_attn,
                            &mut self.d_q8k_quants,
                            &mut self.d_q8k_scales,
                            self.q_width / 256,
                        )
                        .map_err(map)?;
                        dispatch_gemv(
                            &s,
                            &self.k,
                            lq[3],
                            &self.d_in_scales,
                            &self.d_in_quants,
                            &self.d_q8k_scales,
                            &self.d_q8k_quants,
                            &wo,
                            self.hidden,
                            self.q_width,
                            &mut self.d_hidden,
                            1,
                        )
                        .map_err(map)?;
                    }
                }
                LayerKind::Ssm(ssm) => {
                    // The shared attn-norm+quantize above produced the Q8 activation in
                    // d_in_*. Run the proven SSM mixer (== runnable qwen35_ssm_compute):
                    // wqkv/wqkv_gate/beta/alpha gemv -> gates -> conv1d -> l2norm q/k ->
                    // delta-rule -> ssm_out gemv (+= residual). beta_raw/alpha_raw reuse the
                    // idle attention scratch d_k/d_v (kv_width >= num_v_heads).
                    let (ds, nk, nv, key_dim, value_dim, conv_dim, d_conv) = {
                        let g = self.qwen35.as_ref().unwrap();
                        (
                            g.d_state,
                            g.num_k_heads,
                            g.num_v_heads,
                            g.key_dim,
                            g.value_dim,
                            g.conv_dim,
                            g.d_conv,
                        )
                    };
                    let sq = ssm.quants;
                    let q = self.qwen35.as_mut().unwrap();
                    dispatch_gemv(
                        &s,
                        &self.k,
                        sq[0],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &ssm.wqkv.as_view(),
                        conv_dim,
                        self.hidden,
                        &mut q.d_qkv,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemv(
                        &s,
                        &self.k,
                        sq[1],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &ssm.wqkv_gate.as_view(),
                        value_dim,
                        self.hidden,
                        &mut q.d_z,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemv(
                        &s,
                        &self.k,
                        sq[2],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &ssm.beta.as_view(),
                        nv,
                        self.hidden,
                        &mut self.d_k,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemv(
                        &s,
                        &self.k,
                        sq[3],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &ssm.alpha.as_view(),
                        nv,
                        self.hidden,
                        &mut self.d_v,
                        0,
                    )
                    .map_err(map)?;
                    launch_ssm_gates(
                        &s,
                        &self.k.ssm_gates,
                        &self.d_k,
                        &self.d_v,
                        &ssm.dt_bias,
                        &ssm.a,
                        &mut q.d_beta,
                        &mut q.d_glog,
                        nv,
                    )
                    .map_err(map)?;
                    launch_ssm_conv1d(
                        &s,
                        &self.k.ssm_conv1d,
                        &ssm.conv1d,
                        &q.d_qkv,
                        &mut self.ssm_conv_state[li],
                        &mut q.d_conv_out,
                        conv_dim,
                        d_conv,
                    )
                    .map_err(map)?;
                    launch_ssm_l2_norm_per_head(
                        &s,
                        &self.k.ssm_l2_norm_per_head,
                        &mut q.d_conv_out,
                        0,
                        nk,
                        ds,
                        self.eps,
                    )
                    .map_err(map)?;
                    launch_ssm_l2_norm_per_head(
                        &s,
                        &self.k.ssm_l2_norm_per_head,
                        &mut q.d_conv_out,
                        key_dim,
                        nk,
                        ds,
                        self.eps,
                    )
                    .map_err(map)?;
                    launch_ssm_delta_rule(
                        &s,
                        &self.k.ssm_delta_rule,
                        &mut self.ssm_state[li],
                        &q.d_conv_out,
                        key_dim,
                        value_dim,
                        &q.d_z,
                        &q.d_beta,
                        &q.d_glog,
                        &ssm.ssm_norm,
                        &mut q.d_ssm_mix,
                        ds,
                        nk,
                        nv,
                        self.eps,
                    )
                    .map_err(map)?;
                    // ssm_out projection + residual into d_hidden. Quantize the SSM mix to
                    // the activation format ssm_out's lane reads: Q8_K for a K-quant
                    // (Q4_K/Q6_K) ssm_out, Q8_0 otherwise.
                    if sq[4].needs_q8k() {
                        launch_quantize_q8k(
                            &s,
                            &self.k.quantize_q8k,
                            &q.d_ssm_mix,
                            &mut self.d_q8k_quants,
                            &mut self.d_q8k_scales,
                            value_dim / 256,
                        )
                        .map_err(map)?;
                    } else {
                        launch_quantize(
                            &s,
                            &self.k.quantize,
                            &q.d_ssm_mix,
                            &mut self.d_in_quants,
                            &mut self.d_in_scales,
                            value_dim / 32,
                        )
                        .map_err(map)?;
                    }
                    dispatch_gemv(
                        &s,
                        &self.k,
                        sq[4],
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &ssm.ssm_out.as_view(),
                        self.hidden,
                        value_dim,
                        &mut self.d_hidden,
                        1,
                    )
                    .map_err(map)?;
                }
            }
            // ffn norm + gate/up + silu + down + residual. gate/up consume the ffn-norm
            // activation; down consumes the silu(gate)*up activation. Each is produced in
            // the format its consumers read (Q8_0 path byte-identical for an all-Q8_0 layer).
            let ffn_need_q8_0 = lq[4] == ProjQuant::Q8_0 || lq[5] == ProjQuant::Q8_0;
            let ffn_need_q8k = lq[4].needs_q8k() || lq[5].needs_q8k();
            if ffn_need_q8_0 {
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
            }
            if ffn_need_q8k {
                launch_rmsnorm_quantize_q8k(
                    &s,
                    &self.k.rms_norm_quantize_q8k,
                    &self.d_hidden,
                    &self.layers[li].ffn_norm,
                    &mut self.d_q8k_quants,
                    &mut self.d_q8k_scales,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
            }
            // Phase 6 overlap (FFN): publish the ffn activation to side_a and run the
            // up gemv there while the gate gemv keeps main warm. Full layers only —
            // the SSM mixer is out of scope in v1, so an SSM layer's FFN stays serial.
            let ffn_overlap =
                ov_streams.is_some() && matches!(&self.layers[li].kind, LayerKind::Full);
            let s_up: &Arc<CudaStream> = if ffn_overlap {
                let (a, _) = ov_streams.as_ref().expect("overlap streams");
                let o = self.overlap.as_ref().expect("overlap state");
                o.ev_ffn.record(&s).map_err(map)?;
                a.wait(&o.ev_ffn).map_err(map)?;
                a
            } else {
                &s
            };
            dispatch_gemv(
                &s,
                &self.k,
                lq[4],
                &self.d_in_scales,
                &self.d_in_quants,
                &self.d_q8k_scales,
                &self.d_q8k_quants,
                &wgate,
                self.ffn_dim,
                self.hidden,
                &mut self.d_gate,
                0,
            )
            .map_err(map)?;
            dispatch_gemv(
                s_up,
                &self.k,
                lq[5],
                &self.d_in_scales,
                &self.d_in_quants,
                &self.d_q8k_scales,
                &self.d_q8k_quants,
                &wup,
                self.ffn_dim,
                self.hidden,
                &mut self.d_up,
                0,
            )
            .map_err(map)?;
            // Join: silu on main reads d_up, so it waits on the side up gemv. The
            // wait also transitively orders silu's write of the shared d_in_*
            // (/d_q8k_*) scratch after the up gemv's read of it (WAR), and the next
            // layer's attn norm write after that.
            if ffn_overlap {
                let o = self.overlap.as_ref().expect("overlap state");
                o.ev_up.record(s_up).map_err(map)?;
                s.wait(&o.ev_up).map_err(map)?;
            }
            // SiLU(gate)*up -> down's activation, in down's format.
            if lq[6] == ProjQuant::Q8_0 {
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
            } else {
                launch_silu_mul_quantize_q8k(
                    &s,
                    &self.k.silu_mul_quantize_q8k,
                    &self.d_gate,
                    &self.d_up,
                    &mut self.d_q8k_quants,
                    &mut self.d_q8k_scales,
                    self.ffn_dim / 256,
                )
                .map_err(map)?;
                dispatch_gemv(
                    &s,
                    &self.k,
                    lq[6],
                    &self.d_in_scales,
                    &self.d_in_quants,
                    &self.d_q8k_scales,
                    &self.d_q8k_quants,
                    &wdown,
                    self.hidden,
                    self.ffn_dim,
                    &mut self.d_hidden,
                    1,
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
        // final norm + output (lm_head) projection -> d_logits (no argmax / no sync).
        // Produce the activation in the lm_head lane's format (Q6_K for Q4_K_M).
        if self.output_quant == ProjQuant::Q8_0 {
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
        } else {
            launch_rmsnorm_quantize_q8k(
                &s,
                &self.k.rms_norm_quantize_q8k,
                &self.d_hidden,
                &self.final_norm,
                &mut self.d_q8k_quants,
                &mut self.d_q8k_scales,
                self.hidden,
                self.eps,
            )
            .map_err(map)?;
        }
        let out_w = self.output_weight.as_view();
        dispatch_gemv(
            &s,
            &self.k,
            self.output_quant,
            &self.d_in_scales,
            &self.d_in_quants,
            &self.d_q8k_scales,
            &self.d_q8k_quants,
            &out_w,
            self.vocab,
            self.hidden,
            &mut self.d_logits,
            0,
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
            self.forward_pass(embedding, cos, sin, position, scale, false, false, false)?;
            self.k.ctx.synchronize().map_err(map)?;
            return Ok(None);
        }
        // Greedy decode: replay the captured CUDA graph when enabled (one launch for
        // the whole ~600-kernel token), else the per-launch path. STATUS 2026-07-03:
        // capture is broken on this Windows/WDDM driver (576.83, CUDA 12.9) for BOTH
        // the llama and qwen35 arches — begin_capture on the default stream returns
        // CAPTURE_UNSUPPORTED (fixed by the dedicated engine stream), and recording
        // then dies with CAPTURE_ISOLATION ("dependency created on uncaptured work in
        // another stream") inside the common layer loop even after a pre-capture
        // stream drain, in release builds deterministically (llama probe:
        // full_forward_token_matches_cpu fails identically). The qwen35 guard below is
        // kept so an env opt-in cannot silently fall back to the CPU lane on serve;
        // the real launch-overhead fix is the device-side decode loop (resident
        // embedding gather + resident rope tables reading d_sampled/d_position), which
        // needs no capture at all. qwen35's SSM state itself is graph-compatible
        // (stable engine-level buffers, no position launch scalars).
        if cuda_graphs_enabled() && self.qwen35.is_none() {
            return self
                .forward_token_greedy_graphed(embedding, cos, sin, position, scale)
                .map(Some);
        }
        self.forward_pass(embedding, cos, sin, position, scale, true, false, false)?;
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

    /// Install the device-side decode-loop tables (qwen35): the quantized
    /// embedding table wire bytes (q6_K rows pre-padded 210->224 by the caller),
    /// the all-positions rope tables (each max_pos * rope_dim/2 f32, host-built
    /// with the VERBATIM qwen35_rope_tables math), the on-device generated-token
    /// ring, and the 1-slot host-fed token buffer. On failure (e.g. VRAM) the
    /// engine simply stays on the host-fed loop.
    pub(crate) fn set_device_decode_tables(
        &mut self,
        embd_wire: &[u8],
        family: ProjQuant,
        cos_all: &[f32],
        sin_all: &[f32],
    ) -> Result<(), String> {
        // The device-decode loop gathers the embedding row on the GPU, so it needs an
        // `embed_gather_*` kernel for `family`. Families without one (Q5_K/Q2_K/IQ4_XS)
        // MUST fail here so the caller falls back to the host-fed loop (CPU dequant)
        // instead of installing tables that then error mid-`forward_token_device`.
        if !family.has_device_embed_gather() {
            return Err(format!(
                "device-decode embedding gather not implemented for {family:?}"
            ));
        }
        let map = |e: cudarc::driver::DriverError| format!("device decode tables: {e}");
        let s = self.k.stream.clone();
        let table = s.clone_htod(embd_wire).map_err(map)?;
        let cos = s.clone_htod(cos_all).map_err(map)?;
        let sin = s.clone_htod(sin_all).map_err(map)?;
        let out_tokens = s.alloc_zeros::<u32>(self.max_pos).map_err(map)?;
        let token_in = s.alloc_zeros::<u32>(1).map_err(map)?;
        self.embd_table = Some((table, family));
        self.d_rope_cos_all = Some(cos);
        self.d_rope_sin_all = Some(sin);
        self.d_out_tokens = Some(out_tokens);
        self.d_token_in = Some(token_in);
        Ok(())
    }

    pub(crate) fn device_decode_ready(&self) -> bool {
        self.embd_table.is_some()
    }

    /// One fully device-side forward: the input token id comes from
    /// `prev_step` of the on-device output ring (None => the host-fed
    /// d_token_in slot, uploaded here from `host_token`), the rope row is
    /// selected on-device, the embedding row is gathered/dequantized on-device
    /// (bit-identical elementwise mirror of the CPU dequant), then the standard
    /// per-launch forward runs; when `out_step` is Some the greedy argmax lands
    /// in d_out_tokens[out_step] — which the NEXT call can consume directly.
    /// NO host synchronization: the caller syncs once per readback chunk.
    pub(crate) fn forward_token_device(
        &mut self,
        prev_step: Option<usize>,
        host_token: Option<u32>,
        position: usize,
        scale: f32,
        out_step: Option<usize>,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda device-decode: {e}");
        let s = self.k.stream.clone();
        s.memcpy_htod(&[position as i32], &mut self.d_position)
            .map_err(map)?;
        if let Some(tok) = host_token {
            let tin = self
                .d_token_in
                .as_mut()
                .ok_or("device decode tables not installed")?;
            s.memcpy_htod(&[tok], tin).map_err(map)?;
        }
        {
            let this = &mut *self;
            let cos_all = this
                .d_rope_cos_all
                .as_ref()
                .ok_or("device decode tables not installed")?;
            let sin_all = this
                .d_rope_sin_all
                .as_ref()
                .ok_or("device decode tables not installed")?;
            let half = this.d_cos.len();
            launch_rope_select(
                &s,
                &this.k.rope_select,
                cos_all,
                sin_all,
                position,
                half,
                &mut this.d_cos,
                &mut this.d_sin,
            )
            .map_err(map)?;
            let (table, family) = this
                .embd_table
                .as_ref()
                .ok_or("device decode tables not installed")?;
            let tok_view = match prev_step {
                Some(st) => this
                    .d_out_tokens
                    .as_ref()
                    .ok_or("device decode tables not installed")?
                    .slice(st..st + 1),
                None => this
                    .d_token_in
                    .as_ref()
                    .ok_or("device decode tables not installed")?
                    .slice(0..1),
            };
            let f = match family {
                ProjQuant::Q4K => &this.k.embed_gather_q4k,
                ProjQuant::Q6K => &this.k.embed_gather_q6k,
                ProjQuant::Q3K => &this.k.embed_gather_q3k,
                ProjQuant::Q8_0 => &this.k.embed_gather_q8_0,
                ProjQuant::Q5K => return Err("q5_K embedding gather not implemented".into()),
                ProjQuant::Q2K => return Err("q2_K embedding gather not implemented".into()),
                ProjQuant::IQ4XS => return Err("iq4_xs embedding gather not implemented".into()),
            };
            let dim = this.hidden;
            launch_embed_gather(&s, f, table, &tok_view, dim, &mut this.d_hidden).map_err(map)?;
        }
        self.forward_pass(
            &[],
            &[],
            &[],
            position,
            scale,
            out_step.is_some(),
            false,
            true,
        )?;
        if let Some(st) = out_step {
            let this = &mut *self;
            let mut view = this
                .d_out_tokens
                .as_mut()
                .ok_or("device decode tables not installed")?
                .slice_mut(st..st + 1);
            launch_argmax_at(&s, &this.k.argmax, &this.d_logits, this.vocab, &mut view)
                .map_err(map)?;
        }
        Ok(())
    }

    /// Sync once and read `len` generated ids starting at ring slot `start`.
    pub(crate) fn read_out_tokens(&mut self, start: usize, len: usize) -> Result<Vec<u32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda device-decode read: {e}");
        let ring = self
            .d_out_tokens
            .as_ref()
            .ok_or("device decode tables not installed")?;
        let view = ring.slice(start..start + len);
        let mut out = vec![0u32; len];
        self.k.stream.memcpy_dtoh(&view, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(out)
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
        let map = |step: &'static str| {
            move |e: cudarc::driver::DriverError| format!("cuda graph {step}: {e}")
        };
        let s = self.k.stream.clone();
        // Per-token inputs live in device buffers the (frozen) graph reads on replay.
        s.memcpy_htod(embedding, &mut self.d_hidden)
            .map_err(map("htod-emb"))?;
        s.memcpy_htod(cos, &mut self.d_cos)
            .map_err(map("htod-cos"))?;
        s.memcpy_htod(sin, &mut self.d_sin)
            .map_err(map("htod-sin"))?;
        s.memcpy_htod(&[position as i32], &mut self.d_position)
            .map_err(map("htod-pos"))?;

        if self.decode_graph.is_none() {
            // Record the greedy forward (layer stack + output projection + argmax)
            // once. Stream capture records without executing, so this does not write
            // KV; the first real execution is the `launch()` below.
            //
            // Drain the stream first: the input uploads above may still be in
            // flight (WDDM stages pageable H2D copies through driver-internal
            // work), and capture beginning while they are pending records a
            // dependency on uncaptured work — CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
            // (Debug builds masked this: the slower host let the copies complete
            // before begin_capture.) One-time cost — replays skip this branch.
            s.synchronize().map_err(map("pre-capture sync"))?;
            s.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(map("begin"))?;
            let recorded = (|| -> Result<(), String> {
                self.forward_pass(embedding, cos, sin, position, scale, true, true, false)?;
                launch_argmax(
                    &s,
                    &self.k.argmax,
                    &self.d_logits,
                    self.vocab,
                    &mut self.d_sampled,
                )
                .map_err(map("argmax"))?;
                Ok(())
            })();
            // Always end capture to leave the stream clean, then surface a record error.
            // flags = 0 (no special instantiation flags); the repr(u32) enum has no
            // zero variant, and cudarc consumes it via `as u32`, so the 0 bits pass.
            let flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0) };
            let captured = s.end_capture(flags);
            recorded?;
            match captured.map_err(map("end"))? {
                Some(graph) => {
                    graph.upload().map_err(map("upload"))?;
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
            .map_err(map("replay"))?;
        let mut out = [0u32; 1];
        s.memcpy_dtoh(&self.d_sampled, &mut out)
            .map_err(map("dtoh"))?;
        self.k.ctx.synchronize().map_err(map("sync"))?;
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
        self.forward_pass(embedding, cos, sin, position, scale, true, false, false)?;
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
        self.forward_pass(embedding, cos, sin, position, scale, true, false, false)?;
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
            self.forward_pass(emb, cos, sin, i, scale, false, false, false)?;
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
    /// once and reused across the `k` tokens (via the quant lane's batched GEMM),
    /// so this is much cheaper than `k` separate `forward_token` calls. The K tokens' K/V are
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
        if !self.supports_batched_verify() {
            return Err("verify_batch: model has no resident batched projection path".into());
        }
        if k > self.batched_verify_token_cap() {
            return Err(format!(
                "verify_batch: k={k} exceeds this quant lane's shared-memory cap of {}",
                self.batched_verify_token_cap()
            ));
        }
        let map = |e: cudarc::driver::DriverError| format!("cuda verify: {e}");
        // The per-layer batched stack lives in `run_batched_layer_stack` (shared with
        // batched prefill); here we only need the dims for input staging and the final
        // logits projection.
        let (hidden, vocab, eps) = (self.hidden, self.vocab, self.eps);
        let half = self.rope_dim / 2;
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

        self.run_batched_layer_stack(&mut sc, &s, base_position, k, scale, false)?;
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
        quantize_batched_for_lanes(
            &s,
            &self.k,
            &sc.vn,
            &mut sc.viq,
            &mut sc.vis,
            &mut sc.viqk,
            &mut sc.visk,
            hidden,
            k,
            std::slice::from_ref(&self.output_quant),
        )
        .map_err(map)?;
        dispatch_gemm_batched(
            &s,
            &self.k,
            self.output_quant,
            &sc.vis,
            &sc.viq,
            &sc.visk,
            &sc.viqk,
            &self.output_weight,
            vocab,
            hidden,
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
        self.verify_scratch = Some(self.alloc_verify_scratch(MAX_VERIFY_K)?);
        Ok(())
    }

    /// Allocate a `VerifyScratch` sized to `cap` rows (`cap * dim`). Used by the
    /// linear verify (`cap = MAX_VERIFY_K`) and the tree verify (`cap =
    /// TREE_MAX_NODES`); the buffers are dimensionally identical, only wider.
    fn alloc_verify_scratch(&self, cap: usize) -> Result<VerifyScratch, String> {
        let (hidden, q_width, kv_width, ffn_dim, vocab) = (
            self.hidden,
            self.q_width,
            self.kv_width,
            self.ffn_dim,
            self.vocab,
        );
        let half = self.rope_dim / 2;
        let st = &self.k.stream;
        let mk = cap;
        let max_in = hidden.max(q_width).max(ffn_dim);
        let af = |n: usize| {
            st.alloc_zeros::<f32>(n)
                .map_err(|e| format!("verify alloc: {e}"))
        };
        Ok(VerifyScratch {
            vh: af(mk * hidden)?,
            vn: af(mk * hidden)?,
            viq: st
                .alloc_zeros::<i8>(mk * max_in)
                .map_err(|e| format!("verify alloc: {e}"))?,
            vis: af(mk * (max_in / 32))?,
            viqk: st
                .alloc_zeros::<i8>(mk * max_in)
                .map_err(|e| format!("verify alloc: {e}"))?,
            visk: af(mk * max_in.div_ceil(256))?,
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
        })
    }

    /// Run the batched layer stack for `k` tokens (`1..=MAX_VERIFY_K`) at consecutive
    /// positions `[base_position, base_position+k)`. Reads the staged per-token input
    /// from `sc.vh` / `sc.vcos` / `sc.vsin`, writes each token's K/V into the cache,
    /// and leaves the post-final-layer hidden state in `sc.vh`. The caller stages the
    /// inputs and (for `verify_batch`) projects logits afterward.
    ///
    /// This is the single source of truth for the batched forward, shared by
    /// `verify_batch` (speculative decode) and `prefill_batched`. It is bit-identical
    /// to the serial `forward_pass` per token: each supported batched projection
    /// reproduces its decode GEMV's integer decomposition and ordered fp32 sum, and the
    /// batched norm/RoPE/scatter/attention kernels match their serial counterparts.
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
        flash_ok: bool, // SIROCCO Phase P M1: prefill passes true (opt-in flash); verify passes false.
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
        for li in 0..self.n_layers {
            let layer = &self.layers[li];
            let lq = layer.quants;
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
            quantize_batched_for_lanes(
                &s,
                &self.k,
                &sc.vn,
                &mut sc.viq,
                &mut sc.vis,
                &mut sc.viqk,
                &mut sc.visk,
                hidden,
                k,
                &lq[..3],
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s, &self.k, lq[0], &sc.vis, &sc.viq, &sc.visk, &sc.viqk, &layer.q, q_width,
                hidden, k, &mut sc.vq,
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s, &self.k, lq[1], &sc.vis, &sc.viq, &sc.visk, &sc.viqk, &layer.k, kv_width,
                hidden, k, &mut sc.vk,
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s, &self.k, lq[2], &sc.vis, &sc.viq, &sc.visk, &sc.viqk, &layer.v, kv_width,
                hidden, k, &mut sc.vv,
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
            if flash_ok
                && flash_prefill_enabled()
                && q_width == n_heads * head_dim
                && k <= MAX_VERIFY_K
                && base_position + k > SPLITK_THRESHOLD
            {
                // SIROCCO Phase P M1: flash prefill attention (opt-in). K/V read once per key, reused
                // across the k query rows of the head. Token-parity (gated by the oracle); prefill-only.
                launch_attention_flash_prefill(
                    &s,
                    &self.k,
                    &sc.vq,
                    &self.cache_k[li],
                    &self.cache_v[li],
                    &mut sc.vattn,
                    &mut self.d_flash_scores,
                    &mut self.d_flash_chunkmax,
                    &mut self.d_flash_lsum,
                    &mut self.d_flash_acc,
                    n_heads,
                    n_kv,
                    head_dim,
                    base_position,
                    k,
                    q_width,
                    max_pos,
                    scale,
                )
                .map_err(map)?;
            } else {
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
                    if splitk_verify_active() { 1 } else { 0 },
                )
                .map_err(map)?;
            }
            quantize_batched_for_lanes(
                &s,
                &self.k,
                &sc.vattn,
                &mut sc.viq,
                &mut sc.vis,
                &mut sc.viqk,
                &mut sc.visk,
                q_width,
                k,
                &lq[3..4],
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[3],
                &sc.vis,
                &sc.viq,
                &sc.visk,
                &sc.viqk,
                &layer.o,
                hidden,
                q_width,
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
            quantize_batched_for_lanes(
                &s,
                &self.k,
                &sc.vn,
                &mut sc.viq,
                &mut sc.vis,
                &mut sc.viqk,
                &mut sc.visk,
                hidden,
                k,
                &lq[4..6],
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[4],
                &sc.vis,
                &sc.viq,
                &sc.visk,
                &sc.viqk,
                &layer.gate,
                ffn_dim,
                hidden,
                k,
                &mut sc.vgate,
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[5],
                &sc.vis,
                &sc.viq,
                &sc.visk,
                &sc.viqk,
                &layer.up,
                ffn_dim,
                hidden,
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
            quantize_batched_for_lanes(
                &s,
                &self.k,
                &sc.vact,
                &mut sc.viq,
                &mut sc.vis,
                &mut sc.viqk,
                &mut sc.visk,
                ffn_dim,
                k,
                &lq[6..7],
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[6],
                &sc.vis,
                &sc.viq,
                &sc.visk,
                &sc.viqk,
                &layer.down,
                hidden,
                ffn_dim,
                k,
                &mut sc.vproj,
            )
            .map_err(map)?;
            launch_residual(&s, &self.k.residual_add, &mut sc.vh, &sc.vproj, k * hidden)
                .map_err(map)?;
        }
        Ok(())
    }

    /// Allocate the tree-verify scratch (sized to `TREE_MAX_NODES`) and the per-node
    /// index device buffers if not already present. Idempotent.
    fn ensure_tree_scratch(&mut self) -> Result<(), String> {
        if self.tree_scratch.is_some() {
            return Ok(());
        }
        let cap = crate::inference::spec_tree::TREE_MAX_NODES;
        let words = cap.div_ceil(32);
        let sc = self.alloc_verify_scratch(cap)?;
        let st = &self.k.stream;
        let node_kvslot = st
            .alloc_zeros::<i32>(cap)
            .map_err(|e| format!("tree alloc: {e}"))?;
        let ancestor_bits = st
            .alloc_zeros::<u32>(cap * words)
            .map_err(|e| format!("tree alloc: {e}"))?;
        self.tree_scratch = Some(TreeScratch {
            sc,
            node_kvslot,
            ancestor_bits,
        });
        Ok(())
    }

    /// Run the batched layer stack for an N-node draft TREE. Identical to
    /// `run_batched_layer_stack` except the two position-aware kernels are swapped
    /// for their tree variants: `kv_scatter_tree_batched` writes node `t` to its
    /// own slot `node_kvslot[t]` (not `base+t`), and `attention_tree_batched`
    /// scores the dense committed prefix `[0, base)` plus only the in-chunk slots
    /// on each node's ancestor path (the causal tree mask). RoPE per node is baked
    /// into the staged `sc.vcos`/`sc.vsin` (position `base+depth[t]`), so
    /// `rope_batched` is unchanged. On a LINEAR tree this reduces bit-identically
    /// to the batched stack (proven in tests). `node_kvslot` / `ancestor_bits`
    /// must already hold this tree's per-node data (`words` ancestor words/node).
    #[allow(clippy::too_many_arguments)]
    fn run_tree_layer_stack(
        &mut self,
        sc: &mut VerifyScratch,
        node_kvslot: &CudaSlice<i32>,
        ancestor_bits: &CudaSlice<u32>,
        words: usize,
        s: &Arc<CudaStream>,
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda tree layers: {e}");
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
            // Tree scatter: each node to its own slot node_kvslot[t].
            launch_kv_scatter_tree_batched(
                &s,
                &self.k.kv_scatter_tree_batched,
                &sc.vk,
                &mut self.cache_k[li],
                node_kvslot,
                n_kv,
                head_dim,
                max_pos,
                kv_width,
                k,
            )
            .map_err(map)?;
            launch_kv_scatter_tree_batched(
                &s,
                &self.k.kv_scatter_tree_batched,
                &sc.vv,
                &mut self.cache_v[li],
                node_kvslot,
                n_kv,
                head_dim,
                max_pos,
                kv_width,
                k,
            )
            .map_err(map)?;
            // Tree attention: dense prefix [0, base) + ancestor slots only.
            launch_attention_tree_batched(
                &s,
                &self.k.attention_tree_batched,
                &sc.vq,
                &self.cache_k[li],
                &self.cache_v[li],
                &mut sc.vattn,
                ancestor_bits,
                words,
                n_heads,
                n_kv,
                head_dim,
                base_position,
                max_pos,
                scale,
                q_width,
                k,
                if splitk_verify_active() { 1 } else { 0 },
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

    /// Tree-verify forward: run an N-node draft TREE through the model in one batched
    /// pass and return the greedy argmax for each node (`predicted[i]` = the model's
    /// next token after node `i` along its path). Lossless tree speculation: the caller
    /// feeds `predicted` to [`TokenTree::accept_longest_path`] to pick the accepted path.
    ///
    /// `node_kvslot[i]` = base + BFS index `i` (each node's unique cache slot);
    /// `node_depth[i]` = depth (RoPE position = base + depth); `ancestor_bits` is the
    /// flat `[node][words]` causal tree mask (`words = ceil(N/32)`). `embeddings` is
    /// `N*hidden`, staged in BFS order; `cos_all`/`sin_all` are per-NODE RoPE tables
    /// (`N*rope_dim/2`) at position `base + node_depth[i]`. `n` must be `1..=TREE_MAX_NODES`.
    ///
    /// After argmax, the caller's accepted path may be a strict subset of the scattered
    /// nodes. The KV slots of the accepted path are then COMPACTED by rescatter into the
    /// contiguous slots `base..base+L-1` (path order) via [`compact_tree_kv`], leaving the
    /// cache exactly as a linear decode of the accepted path would — so the next round's
    /// committed prefix is correct. For a single-branch tree this is a no-op.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_tree(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        node_kvslot: &[i32],
        ancestor_bits: &[u32],
        words: usize,
        base_position: usize,
        n: usize,
        scale: f32,
    ) -> Result<Vec<u32>, String> {
        let cap = crate::inference::spec_tree::TREE_MAX_NODES;
        if n == 0 || n > cap {
            return Err(format!("verify_tree: n={n} out of 1..={cap}"));
        }
        if !self.supports_tree_verify() {
            return Err("verify_tree: model is not eligible for the Q8-only tree path".into());
        }
        if node_kvslot.len() < n || ancestor_bits.len() < n * words {
            return Err("verify_tree: index slices too short".into());
        }
        let map = |e: cudarc::driver::DriverError| format!("cuda verify_tree: {e}");
        let (hidden, vocab, eps) = (self.hidden, self.vocab, self.eps);
        let half = self.rope_dim / 2;
        let hb = hidden / 32;
        if embeddings.len() < n * hidden || cos_all.len() < n * half || sin_all.len() < n * half {
            return Err("verify_tree: input slices too short".into());
        }
        self.ensure_tree_scratch()?;
        let s = self.k.stream.clone();
        let mut ts = self.tree_scratch.take().expect("allocated above");

        s.memcpy_htod(
            &embeddings[..n * hidden],
            &mut ts.sc.vh.slice_mut(0..n * hidden),
        )
        .map_err(map)?;
        s.memcpy_htod(&cos_all[..n * half], &mut ts.sc.vcos.slice_mut(0..n * half))
            .map_err(map)?;
        s.memcpy_htod(&sin_all[..n * half], &mut ts.sc.vsin.slice_mut(0..n * half))
            .map_err(map)?;
        s.memcpy_htod(&node_kvslot[..n], &mut ts.node_kvslot.slice_mut(0..n))
            .map_err(map)?;
        s.memcpy_htod(
            &ancestor_bits[..n * words],
            &mut ts.ancestor_bits.slice_mut(0..n * words),
        )
        .map_err(map)?;

        // Move sc/index buffers out so run_tree_layer_stack can borrow &mut self.
        let TreeScratch {
            mut sc,
            node_kvslot: d_slot,
            ancestor_bits: d_anc,
        } = ts;
        self.run_tree_layer_stack(&mut sc, &d_slot, &d_anc, words, &s, base_position, n, scale)?;
        launch_rms_norm_batched(
            &s,
            &self.k.rms_norm_batched,
            &sc.vh,
            &self.final_norm,
            &mut sc.vn,
            hidden,
            eps,
            n,
        )
        .map_err(map)?;
        launch_quantize(
            &s,
            &self.k.quantize,
            &sc.vn,
            &mut sc.viq,
            &mut sc.vis,
            n * hb,
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
            n,
            &mut sc.vlogits,
        )
        .map_err(map)?;
        launch_argmax_batched(
            &s,
            &self.k.argmax_batched,
            &sc.vlogits,
            vocab,
            n,
            &mut sc.vsamp,
        )
        .map_err(map)?;
        let mut out = vec![0u32; cap];
        s.memcpy_dtoh(&sc.vsamp, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        out.truncate(n);
        // Put the scratch back for reuse.
        self.tree_scratch = Some(TreeScratch {
            sc,
            node_kvslot: d_slot,
            ancestor_bits: d_anc,
        });
        Ok(out)
    }

    /// COMPACT-BY-RESCATTER the accepted path's KV into the contiguous slots
    /// `base..base+L-1` (path order), per layer, so the cache after a tree round is
    /// byte-for-byte what a linear decode of the accepted path would leave. `path`
    /// is the accepted node indices INCLUDING the root anchor (node 0), root first —
    /// exactly `tree.path_to(leaf)`. Slot of node `j` is `base + j` (its
    /// `node_kvslot`). We copy, for each accepted node at path rank `r`, the K/V row
    /// from source slot `base + path[r]` to destination slot `base + r`.
    ///
    /// CRITICAL off-by-one note: `path[0]` is the anchor (node 0, already at slot
    /// `base + 0 = base`), so its copy is the identity and `r=0` is correct to
    /// include. For a single-branch (linear) tree `path == [0,1,..,L-1]` so every
    /// copy is slot→same slot — a NO-OP — which is why a linear tree needs no
    /// compaction. Copies run front-to-back; since `path[r] >= r` always (the path
    /// is a strictly increasing subsequence of BFS indices starting at 0), the source
    /// slot is never below the destination, so a forward copy never clobbers a source
    /// it still needs. After compaction the caller sets position/filled = base + L.
    pub fn compact_tree_kv_path(&mut self, path: &[usize], base: usize) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda compact: {e}");
        let s = self.k.stream.clone();
        let (n_kv, head_dim, max_pos) = (self.n_kv_heads, self.head_dim, self.max_pos);
        // A copy within one CudaSlice can't borrow it &mut and & at once (and the
        // dst slot may equal another node's src), so route each row through host.
        // Rows are tiny (head_dim u16) and compaction only fires when a branch
        // diverges, so the round-trip is negligible. `path[r] >= r` always, so the
        // gather-then-scatter is order-independent anyway.
        let mut row = vec![0u16; head_dim];
        for (r, &node) in path.iter().enumerate() {
            if node == r {
                continue; // identity (the whole linear case) — slot already correct
            }
            let src_pos = base + node;
            let dst_pos = base + r;
            for li in 0..self.n_layers {
                for kv_head in 0..n_kv {
                    let row_base = kv_head * max_pos * head_dim;
                    let src = row_base + src_pos * head_dim;
                    let dst = row_base + dst_pos * head_dim;
                    // K
                    s.memcpy_dtoh(&self.cache_k[li].slice(src..src + head_dim), &mut row)
                        .map_err(map)?;
                    s.memcpy_htod(&row, &mut self.cache_k[li].slice_mut(dst..dst + head_dim))
                        .map_err(map)?;
                    // V
                    s.memcpy_dtoh(&self.cache_v[li].slice(src..src + head_dim), &mut row)
                        .map_err(map)?;
                    s.memcpy_htod(&row, &mut self.cache_v[li].slice_mut(dst..dst + head_dim))
                        .map_err(map)?;
                }
            }
        }
        self.k.ctx.synchronize().map_err(map)?;
        Ok(())
    }

    /// Batched GPU prefill: ingest `n` prompt tokens at positions `[0, n)` through the
    /// batched layer stack in quant-aware chunks, reading each weight once per chunk
    /// instead of once per prompt token. The serial `prefill` re-streams every
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
        if !self.supports_batched_prefill() {
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
        let batch_cap = self.batched_layer_token_cap();
        while base < n {
            let kk = (n - base).min(batch_cap);
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
            self.run_batched_layer_stack(&mut sc, &s, base, kk, scale, true)?;
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
