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
#ifdef CAMELID_HAS_WMMA
#include <mma.h>
#endif

// ---- Hardware intrinsics (header-free) -------------------------------------
// NVRTC runs without <cuda_runtime.h> or <sm_61_intrinsics.h>; provide direct
// inline PTX assembly for dp4a (SM61+) and byte_perm (SM32+) so NVRTC does not
// generate unresolved extern function calls to _Z6__dp4aiii.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
#endif

static __device__ __forceinline__ unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int s) {
    unsigned int d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

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

// Lossless bit-sliced view of the existing Q8/32 activation. One warp owns one
// chunk; input scales stay in their authoritative f32 buffer.
extern "C" __global__ void prism_q8_32_bitplanes_qsum(
    const signed char* __restrict__ quants,
    unsigned int* __restrict__ bitplanes,
    int* __restrict__ qsums,
    int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int chunk = blockIdx.x * (blockDim.x >> 5) + warp;
    if (chunk >= n_chunks) return;
    int q = (int)quants[(long)chunk * 32 + lane];
    unsigned int uq = (unsigned int)q & 0xffu;
    unsigned int mask = 0xffffffffu;
    #pragma unroll
    for (int plane = 0; plane < 8; plane++) {
        unsigned int bits = __ballot_sync(mask, ((uq >> plane) & 1u) != 0u);
        if (lane == 0) bitplanes[(long)chunk * 8 + plane] = bits;
    }
    q += __shfl_down_sync(mask, q, 16);
    q += __shfl_down_sync(mask, q, 8);
    q += __shfl_down_sync(mask, q, 4);
    q += __shfl_down_sync(mask, q, 2);
    q += __shfl_down_sync(mask, q, 1);
    if (lane == 0) qsums[chunk] = q;
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

// ---- Host-RMS-inverse norm + Windows-parity Q8_0 quantize -----------------
// Gemma's CPU router already scans the post-attention row. The host supplies the
// exact sequential Rust `(mss + eps).powf(-0.5)` result from that scan, while this
// kernel applies `x * rms_inv * weight` directly to the resident device row. One
// warp owns each 32-value Q8 block. Lane 0 visits the normalized values in index
// order, preserving the CPU's serial max scan, then every lane quantizes one value.
// `roundf` is intentional: CUDA roundf and Rust f32::round both break exact halves
// away from zero. The generic quantizers above retain their existing rintf semantics.
extern "C" __global__ void rms_inv_norm_quantize_q8_0(
    const float* __restrict__ x, const float* __restrict__ weight,
    signed char* __restrict__ quants, float* __restrict__ scales,
    int n, float rms_inv
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_cta = blockDim.x >> 5;
    int qblock = blockIdx.x * warps_per_cta + warp;
    int n_blocks = n >> 5;
    if (qblock >= n_blocks) return;

    long i = ((long)qblock << 5) + lane;
    // Keep the same left-associated f32 operations as
    // `rms_norm`: `(x[i] * rms_inv) * weight[i]`.
    float value = x[i] * rms_inv * weight[i];
    unsigned int mask = 0xffffffffu;
    float max_abs = 0.0f;
    #pragma unroll
    for (int source_lane = 0; source_lane < 32; source_lane++) {
        float candidate = __shfl_sync(mask, value, source_lane);
        if (lane == 0) {
            float a = fabsf(candidate);
            if (a > max_abs) max_abs = a;
        }
    }
    max_abs = __shfl_sync(mask, max_abs, 0);

    float unrounded = max_abs / 127.0f;
    if (lane == 0) scales[qblock] = f16_round(unrounded);
    float inv_scale = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    float q = roundf(value * inv_scale); // Rust f32::round: ties away from zero
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    quants[i] = (signed char)q;
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

// Resolve one logical Q1 row/K-block in either stock row-major wire or the
// same-size Q1T128 upload layout. Q1T128 groups every <=128-row tile/K-block as
// [nr*16 signs][nr*2 scales]; only the final row tile may have nr < 128.
__device__ __forceinline__ const unsigned char* prism_q1_sign_ptr(
    const unsigned char* weight_bytes, int rows, int blocks_per_row,
    int row, int weight_block, int q1_tiled
) {
    if (!q1_tiled)
        return weight_bytes + ((long)row * blocks_per_row + weight_block) * 18 + 2;
    int tile = row >> 7;
    int row0 = tile << 7;
    int nr = rows - row0;
    if (nr > 128) nr = 128;
    long tile_base = (long)tile * 128 * blocks_per_row * 18;
    long group = tile_base + (long)weight_block * nr * 18;
    return weight_bytes + group + (row - row0) * 16;
}

__device__ __forceinline__ const unsigned char* prism_q1_scale_ptr(
    const unsigned char* weight_bytes, int rows, int blocks_per_row,
    int row, int weight_block, int q1_tiled
) {
    if (!q1_tiled)
        return weight_bytes + ((long)row * blocks_per_row + weight_block) * 18;
    int tile = row >> 7;
    int row0 = tile << 7;
    int nr = rows - row0;
    if (nr > 128) nr = 128;
    long tile_base = (long)tile * 128 * blocks_per_row * 18;
    long group = tile_base + (long)weight_block * nr * 18;
    return weight_bytes + group + (long)nr * 16 + (row - row0) * 2;
}

// Prism decode GEMV over the original f32 activation. This mirrors the three
// parity-locked Metal kernels in metal.rs rather than quantizing the activation
// to Q8_0 first. The lane/block association is intentional: ultra-low-bit rows
// can change greedy logits when an algebraically equivalent reduction order is
// substituted.
extern "C" __global__ void prism_low_bit_f32_gemv(
    const float* __restrict__ input, const unsigned char* __restrict__ weight_bytes,
    int rows, int blocks_per_row, int bits, int weight_block_elements, int q1_tiled,
    float* __restrict__ output, int residual
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int work = blockIdx.x * warps_per_block + warp;
    unsigned mask = 0xffffffffu;

    if (bits == 1) {
        // Q1_0: one warp evaluates eight rows; four 128-value blocks are in
        // flight and eight lanes own the sixteen-value slices of each block.
        // All eight warps in the CTA consume the SAME activation blocks. Stage
        // each four-block window cooperatively once, with fully coalesced global
        // loads, instead of having every warp issue the same strided loads. The
        // per-lane slice and b=lane_group+4*k order stay unchanged, preserving
        // the parity-locked accumulation and reduction exactly.
        int first_row = work * 8;
        int block_lane = lane >> 3;
        int slice = (lane & 7) << 4;
        float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        extern __shared__ float staged_input[];
        for (int chunk = 0; chunk < blocks_per_row; chunk += 4) {
            int remaining = blocks_per_row - chunk;
            int staged_values = (remaining < 4 ? remaining : 4) * 128;
            for (int i = threadIdx.x; i < staged_values; i += blockDim.x)
                staged_input[i] = input[chunk * 128 + i];
            __syncthreads();
            int b = chunk + block_lane;
            if (first_row < rows && b < blocks_per_row) {
                int input_base = block_lane * 128 + slice;
                float values[16];
                float input_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    values[i] = staged_input[input_base + i];
                    input_sum += values[i];
                }
                #pragma unroll
                for (int ro = 0; ro < 8; ro++) {
                    int row = first_row + ro;
                    if (row < rows) {
                        const unsigned char* scale_ptr = prism_q1_scale_ptr(
                            weight_bytes, rows, blocks_per_row, row, b, q1_tiled);
                        float d = f16_bits_to_f32(
                            (unsigned short)scale_ptr[0]
                            | ((unsigned short)scale_ptr[1] << 8));
                        const unsigned char* qs = prism_q1_sign_ptr(
                            weight_bytes, rows, blocks_per_row, row, b, q1_tiled)
                            + (slice >> 3);
                        unsigned char b0 = qs[0], b1 = qs[1];
                        float selected = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            selected += (b0 & (1u << i)) ? values[i] : 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            selected += (b1 & (1u << i)) ? values[i + 8] : 0.0f;
                        sums[ro] += d * (2.0f * selected - input_sum);
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int ro = 0; ro < 8; ro++) {
            float total = sums[ro];
            total += __shfl_down_sync(mask, total, 16);
            total += __shfl_down_sync(mask, total, 8);
            total += __shfl_down_sync(mask, total, 4);
            total += __shfl_down_sync(mask, total, 2);
            total += __shfl_down_sync(mask, total, 1);
            int row = first_row + ro;
            if (lane == 0 && row < rows)
                output[row] = residual ? output[row] + total : total;
        }
        return;
    }

    // Both Q2 geometries reuse one activation slice across eight output rows,
    // matching the Q1 launch geometry. Each row still accumulates and reduces
    // in exactly the same lane/block order as the former one-row-per-warp path.
    int first_row = work * 8;
    if (first_row >= rows) return;
    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    if (weight_block_elements == 64) {
        // Q2_0-G64: each lane owns indices lane and lane+32 in every block.
        for (int b = 0; b < blocks_per_row; b++) {
            int input_base = b * 64;
            float values[2] = { input[input_base + lane], input[input_base + lane + 32] };
            #pragma unroll
            for (int ro = 0; ro < 8; ro++) {
                int row = first_row + ro;
                if (row < rows) {
                    const unsigned char* block = weight_bytes
                        + ((long)row * blocks_per_row + b) * 18;
                    float d = f16_bits_to_f32(
                        (unsigned short)block[0] | ((unsigned short)block[1] << 8));
                    const unsigned char* qs = block + 2;
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        int i = lane + half * 32;
                        int q = (qs[i >> 2] >> ((i & 3) << 1)) & 3;
                        sums[ro] += values[half] * (float)(q - 1) * d;
                    }
                }
            }
        }
    } else {
        // Q2_0-G128: four blocks in flight, eight lanes per 16-value slice.
        int block_lane = lane >> 3;
        int slice = (lane & 7) << 4;
        for (int b = block_lane; b < blocks_per_row; b += 4) {
            int input_base = b * 128 + slice;
            float values[16];
            float input_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                values[i] = input[input_base + i];
                input_sum += values[i];
            }
            #pragma unroll
            for (int ro = 0; ro < 8; ro++) {
                int row = first_row + ro;
                if (row < rows) {
                    const unsigned char* block = weight_bytes
                        + ((long)row * blocks_per_row + b) * 34;
                    float d = f16_bits_to_f32(
                        (unsigned short)block[0] | ((unsigned short)block[1] << 8));
                    const unsigned char* qs = block + 2 + (slice >> 2);
                    float acc_lo = 0.0f;
                    float acc_hi = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        unsigned char packed = qs[i >> 2];
                        acc_lo += (packed & (1u << ((i & 3) * 2))) ? values[i] : 0.0f;
                        acc_hi += (packed & (1u << ((i & 3) * 2 + 1))) ? values[i] : 0.0f;
                    }
                    sums[ro] += d * (acc_lo + 2.0f * acc_hi - input_sum);
                }
            }
        }
    }
    #pragma unroll
    for (int ro = 0; ro < 8; ro++) {
        float total = sums[ro];
        total += __shfl_down_sync(mask, total, 16);
        total += __shfl_down_sync(mask, total, 8);
        total += __shfl_down_sync(mask, total, 4);
        total += __shfl_down_sync(mask, total, 2);
        total += __shfl_down_sync(mask, total, 1);
        int row = first_row + ro;
        if (lane == 0 && row < rows)
            output[row] = residual ? output[row] + total : total;
    }
}

// Fast Prism Q1 decode: one warp owns eight output rows and reuses each Q8_0
// activation chunk across all eight. Packed signs are expanded four bytes at a
// time with byte_perm and contracted by DP4A. The current exact-f32 kernel uses
// the same eight-row geometry, but spends scalar float work selecting 128
// activation values; this lane trades activation quantization for the integer
// dot product used by the proven CUDA Q1 implementations.
extern "C" __global__ void prism_q1_q8_gemv(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int first_row = (blockIdx.x * warps_per_block + warp) * 8;
    if (first_row >= rows) return;

    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f };
    int chunks_per_row = blocks_per_row * 4;
    for (int chunk_index = lane; chunk_index < chunks_per_row; chunk_index += 32) {
        int weight_block = chunk_index >> 2;
        int chunk = chunk_index & 3;
        const int* aq = (const int*)(input_quants + (long)chunk_index * 32);
        float da = input_scales[chunk_index];

        #pragma unroll
        for (int ro = 0; ro < 8; ro++) {
            int row = first_row + ro;
            if (row < rows) {
                const unsigned char* block = weight_bytes
                    + ((long)row * blocks_per_row + weight_block) * 18;
                float dw = f16_bits_to_f32((unsigned short)block[0]
                    | ((unsigned short)block[1] << 8));
                const unsigned short* qs =
                    (const unsigned short*)(block + 2 + chunk * 4);
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int q = (int)qs[j];
                    int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
                    int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
                    int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
                    int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
                    int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
                    int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
                    sumi = __dp4a((int)__byte_perm(s0, s1, 0x5410), aq[j * 4 + 0], sumi);
                    sumi = __dp4a((int)__byte_perm(s0, s1, 0x7632), aq[j * 4 + 1], sumi);
                    sumi = __dp4a((int)__byte_perm(s2, s3, 0x5410), aq[j * 4 + 2], sumi);
                    sumi = __dp4a((int)__byte_perm(s2, s3, 0x7632), aq[j * 4 + 3], sumi);
                }
                sums[ro] += (float)sumi * dw * da;
            }
        }
    }

    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int ro = 0; ro < 8; ro++) {
        float total = sums[ro];
        total += __shfl_down_sync(mask, total, 16);
        total += __shfl_down_sync(mask, total, 8);
        total += __shfl_down_sync(mask, total, 4);
        total += __shfl_down_sync(mask, total, 2);
        total += __shfl_down_sync(mask, total, 1);
        int row = first_row + ro;
        if (lane == 0 && row < rows)
            output[row] = residual ? output[row] + total : total;
    }
}

// Q1T128 decode: one four-lane subgroup owns one output row. For every K=128
// block the warp's 32 lanes consume the eight rows' contiguous 128-byte sign
// slab; each subgroup lane contracts one K=32 quarter and the four scaled terms
// reduce within the subgroup. This replaces eight per-thread row accumulators
// and the stock row-major 18-byte gathers without expanding the model.
extern "C" __global__ void prism_q1t128_q8_gemv(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_in_warp = lane >> 2;
    int quarter = lane & 3;
    int first_row = (blockIdx.x * warps_per_block + warp) * 8;
    int row = first_row + row_in_warp;
    if (first_row >= rows) return;

    float sum = 0.0f;
    for (int weight_block = 0; weight_block < blocks_per_row; weight_block++) {
        int chunk_index = weight_block * 4 + quarter;
        const int* aq = (const int*)(input_quants + (long)chunk_index * 32);
        const unsigned char* signs_ptr = row < rows
            ? prism_q1_sign_ptr(weight_bytes, rows, blocks_per_row, row, weight_block, 1)
                + quarter * 4
            : weight_bytes;
        unsigned int signs = row < rows
            ? ((unsigned int)signs_ptr[0]
                | ((unsigned int)signs_ptr[1] << 8)
                | ((unsigned int)signs_ptr[2] << 16)
                | ((unsigned int)signs_ptr[3] << 24))
            : 0u;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int q = (int)((signs >> (j * 16)) & 0xffffu);
            int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
            int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
            int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
            int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
            int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
            int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
            sumi = __dp4a((int)__byte_perm(s0, s1, 0x5410), aq[j * 4 + 0], sumi);
            sumi = __dp4a((int)__byte_perm(s0, s1, 0x7632), aq[j * 4 + 1], sumi);
            sumi = __dp4a((int)__byte_perm(s2, s3, 0x5410), aq[j * 4 + 2], sumi);
            sumi = __dp4a((int)__byte_perm(s2, s3, 0x7632), aq[j * 4 + 3], sumi);
        }
        unsigned int scale_bits = 0u;
        if (quarter == 0 && row < rows) {
            const unsigned char* scale_ptr = prism_q1_scale_ptr(
                weight_bytes, rows, blocks_per_row, row, weight_block, 1);
            scale_bits = (unsigned int)scale_ptr[0] | ((unsigned int)scale_ptr[1] << 8);
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0, 4);
        float dw = f16_bits_to_f32((unsigned short)scale_bits);
        sum += ((float)sumi * dw) * input_scales[chunk_index];
    }
    sum += __shfl_down_sync(0xffffffffu, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffu, sum, 1, 4);
    if (quarter == 0 && row < rows)
        output[row] = residual ? output[row] + sum : sum;
}

__device__ __forceinline__ unsigned int prism_q1_popc_load_word(
    const unsigned char* p
) {
    return (unsigned int)p[0]
        | ((unsigned int)p[1] << 8)
        | ((unsigned int)p[2] << 16)
        | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ int prism_q1_q8_popc_dot32(
    unsigned int signs, const unsigned int planes[8], int qsum
) {
    int selected = -__popc(signs & planes[7]);
    #pragma unroll
    for (int plane = 6; plane >= 0; plane--)
        selected = 2 * selected + __popc(signs & planes[plane]);
    return 2 * selected - qsum;
}

// Two rows per four-lane subgroup; eight warps cover one complete Q1T tile.
// Direct activation loads intentionally rely on the warp's 8-way multicast.
extern "C" __global__ void prism_q1t128_q8_popc_gemv_m16(
    const unsigned int* __restrict__ input_bitplanes,
    const int* __restrict__ input_qsums,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = lane >> 2;
    int quarter = lane & 3;
    int row_base = blockIdx.x * 128 + warp * 16;
    if (row_base >= rows) return;
    int row0 = row_base + group;
    int row1 = row0 + 8;
    unsigned int mask = 0xffffffffu;
    float sum0 = 0.0f, sum1 = 0.0f;

    for (int weight_block = 0; weight_block < blocks_per_row; weight_block++) {
        int chunk = weight_block * 4 + quarter;
        unsigned int planes[8];
        #pragma unroll
        for (int plane = 0; plane < 8; plane++)
            planes[plane] = input_bitplanes[(long)chunk * 8 + plane];
        int qsum = input_qsums[chunk];
        float da = input_scales[chunk];

        unsigned int signs0 = 0u, signs1 = 0u;
        if (row0 < rows) {
            const unsigned char* p = prism_q1_sign_ptr(
                weight_bytes, rows, blocks_per_row, row0, weight_block, 1)
                + quarter * 4;
            signs0 = prism_q1_popc_load_word(p);
        }
        if (row1 < rows) {
            const unsigned char* p = prism_q1_sign_ptr(
                weight_bytes, rows, blocks_per_row, row1, weight_block, 1)
                + quarter * 4;
            signs1 = prism_q1_popc_load_word(p);
        }
        int dot0 = prism_q1_q8_popc_dot32(signs0, planes, qsum);
        int dot1 = prism_q1_q8_popc_dot32(signs1, planes, qsum);
        unsigned int scale0_bits = 0u, scale1_bits = 0u;
        if (quarter == 0 && row0 < rows) {
            const unsigned char* p = prism_q1_scale_ptr(
                weight_bytes, rows, blocks_per_row, row0, weight_block, 1);
            scale0_bits = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
        }
        if (quarter == 0 && row1 < rows) {
            const unsigned char* p = prism_q1_scale_ptr(
                weight_bytes, rows, blocks_per_row, row1, weight_block, 1);
            scale1_bits = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
        }
        scale0_bits = __shfl_sync(mask, scale0_bits, 0, 4);
        scale1_bits = __shfl_sync(mask, scale1_bits, 0, 4);
        sum0 += ((float)dot0 * f16_bits_to_f32((unsigned short)scale0_bits)) * da;
        sum1 += ((float)dot1 * f16_bits_to_f32((unsigned short)scale1_bits)) * da;
    }
    sum0 += __shfl_down_sync(mask, sum0, 2, 4);
    sum0 += __shfl_down_sync(mask, sum0, 1, 4);
    sum1 += __shfl_down_sync(mask, sum1, 2, 4);
    sum1 += __shfl_down_sync(mask, sum1, 1, 4);
    if (quarter == 0) {
        if (row0 < rows) output[row0] = residual ? output[row0] + sum0 : sum0;
        if (row1 < rows) output[row1] = residual ? output[row1] + sum1 : sum1;
    }
}

// Exact Bonsai-27B gate+up fusion: 17,408 rows, 5,120 columns, forty K=128
// blocks. The activation bitplanes/qsum/scale are loaded once and applied to
// both Q1T weights; all per-row block and quarter accumulation stays unchanged.
extern "C" __global__ void prism_q1t128_q8_popc_fused_ffn_bonsai27b(
    const unsigned int* __restrict__ input_bitplanes,
    const int* __restrict__ input_qsums,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ up_weight,
    float* __restrict__ gate_out,
    float* __restrict__ up_out
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = lane >> 2;
    int quarter = lane & 3;
    int row0 = blockIdx.x * 128 + warp * 16 + group;
    int row1 = row0 + 8;
    unsigned int mask = 0xffffffffu;
    float gate0 = 0.0f, gate1 = 0.0f;
    float up0 = 0.0f, up1 = 0.0f;

    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        int chunk = block * 4 + quarter;
        unsigned int planes[8];
        #pragma unroll
        for (int plane = 0; plane < 8; plane++)
            planes[plane] = input_bitplanes[(long)chunk * 8 + plane];
        int qsum = input_qsums[chunk];
        float da = input_scales[chunk];

        long group0 = ((long)(row0 >> 7) * 40 + block) * 2304;
        long group1 = ((long)(row1 >> 7) * 40 + block) * 2304;
        const unsigned int* gs0 = (const unsigned int*)(gate_weight + group0
            + (row0 & 127) * 16 + quarter * 4);
        const unsigned int* gs1 = (const unsigned int*)(gate_weight + group1
            + (row1 & 127) * 16 + quarter * 4);
        const unsigned int* us0 = (const unsigned int*)(up_weight + group0
            + (row0 & 127) * 16 + quarter * 4);
        const unsigned int* us1 = (const unsigned int*)(up_weight + group1
            + (row1 & 127) * 16 + quarter * 4);
        int gate_dot0 = prism_q1_q8_popc_dot32(*gs0, planes, qsum);
        int gate_dot1 = prism_q1_q8_popc_dot32(*gs1, planes, qsum);
        int up_dot0 = prism_q1_q8_popc_dot32(*us0, planes, qsum);
        int up_dot1 = prism_q1_q8_popc_dot32(*us1, planes, qsum);

        unsigned int gate_scale0 = 0u, gate_scale1 = 0u;
        unsigned int up_scale0 = 0u, up_scale1 = 0u;
        if (quarter == 0) {
            const unsigned char* g0 = gate_weight + group0 + 2048 + (row0 & 127) * 2;
            const unsigned char* g1 = gate_weight + group1 + 2048 + (row1 & 127) * 2;
            const unsigned char* u0 = up_weight + group0 + 2048 + (row0 & 127) * 2;
            const unsigned char* u1 = up_weight + group1 + 2048 + (row1 & 127) * 2;
            gate_scale0 = (unsigned int)g0[0] | ((unsigned int)g0[1] << 8);
            gate_scale1 = (unsigned int)g1[0] | ((unsigned int)g1[1] << 8);
            up_scale0 = (unsigned int)u0[0] | ((unsigned int)u0[1] << 8);
            up_scale1 = (unsigned int)u1[0] | ((unsigned int)u1[1] << 8);
        }
        gate_scale0 = __shfl_sync(mask, gate_scale0, 0, 4);
        gate_scale1 = __shfl_sync(mask, gate_scale1, 0, 4);
        up_scale0 = __shfl_sync(mask, up_scale0, 0, 4);
        up_scale1 = __shfl_sync(mask, up_scale1, 0, 4);
        gate0 += ((float)gate_dot0 * f16_bits_to_f32((unsigned short)gate_scale0)) * da;
        gate1 += ((float)gate_dot1 * f16_bits_to_f32((unsigned short)gate_scale1)) * da;
        up0 += ((float)up_dot0 * f16_bits_to_f32((unsigned short)up_scale0)) * da;
        up1 += ((float)up_dot1 * f16_bits_to_f32((unsigned short)up_scale1)) * da;
    }
    gate0 += __shfl_down_sync(mask, gate0, 2, 4);
    gate0 += __shfl_down_sync(mask, gate0, 1, 4);
    gate1 += __shfl_down_sync(mask, gate1, 2, 4);
    gate1 += __shfl_down_sync(mask, gate1, 1, 4);
    up0 += __shfl_down_sync(mask, up0, 2, 4);
    up0 += __shfl_down_sync(mask, up0, 1, 4);
    up1 += __shfl_down_sync(mask, up1, 2, 4);
    up1 += __shfl_down_sync(mask, up1, 1, 4);
    if (quarter == 0) {
        gate_out[row0] = gate0;
        gate_out[row1] = gate1;
        up_out[row0] = up0;
        up_out[row1] = up1;
    }
}

// Exact POPC full-attention fusion. One CTA owns one complete 128-row Q1T tile:
// 96 qgate tiles followed by eight K and eight V tiles. The M16 mapping doubles
// the rows produced by each four-lane subgroup while preserving the exact
// per-quarter and per-K-block accumulation order of the production DP4A lane.
extern "C" __global__ void prism_q1t128_q8_popc_fused_full_bonsai27b(
    const unsigned int* __restrict__ input_bitplanes,
    const int* __restrict__ input_qsums,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ qgate_weight,
    const unsigned char* __restrict__ k_weight,
    const unsigned char* __restrict__ v_weight,
    float* __restrict__ qgate_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    int task = (int)blockIdx.x;
    const unsigned char* weight;
    float* output;
    int tile;
    if (task < 96) {
        weight = qgate_weight;
        output = qgate_out;
        tile = task;
    } else if (task < 104) {
        weight = k_weight;
        output = k_out;
        tile = task - 96;
    } else {
        weight = v_weight;
        output = v_out;
        tile = task - 104;
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = lane >> 2;
    int quarter = lane & 3;
    int row0 = tile * 128 + warp * 16 + group;
    int row1 = row0 + 8;
    unsigned int mask = 0xffffffffu;
    float sum0 = 0.0f, sum1 = 0.0f;

    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        int chunk = block * 4 + quarter;
        unsigned int planes[8];
        #pragma unroll
        for (int plane = 0; plane < 8; plane++)
            planes[plane] = input_bitplanes[(long)chunk * 8 + plane];
        int qsum = input_qsums[chunk];
        float da = input_scales[chunk];
        long group0 = ((long)(row0 >> 7) * 40 + block) * 2304;
        long group1 = ((long)(row1 >> 7) * 40 + block) * 2304;
        const unsigned int* signs0 = (const unsigned int*)(weight + group0
            + (row0 & 127) * 16 + quarter * 4);
        const unsigned int* signs1 = (const unsigned int*)(weight + group1
            + (row1 & 127) * 16 + quarter * 4);
        int dot0 = prism_q1_q8_popc_dot32(*signs0, planes, qsum);
        int dot1 = prism_q1_q8_popc_dot32(*signs1, planes, qsum);
        unsigned int scale0 = 0u, scale1 = 0u;
        if (quarter == 0) {
            const unsigned char* s0 = weight + group0 + 2048 + (row0 & 127) * 2;
            const unsigned char* s1 = weight + group1 + 2048 + (row1 & 127) * 2;
            scale0 = (unsigned int)s0[0] | ((unsigned int)s0[1] << 8);
            scale1 = (unsigned int)s1[0] | ((unsigned int)s1[1] << 8);
        }
        scale0 = __shfl_sync(mask, scale0, 0, 4);
        scale1 = __shfl_sync(mask, scale1, 0, 4);
        sum0 += ((float)dot0 * f16_bits_to_f32((unsigned short)scale0)) * da;
        sum1 += ((float)dot1 * f16_bits_to_f32((unsigned short)scale1)) * da;
    }
    sum0 += __shfl_down_sync(mask, sum0, 2, 4);
    sum0 += __shfl_down_sync(mask, sum0, 1, 4);
    sum1 += __shfl_down_sync(mask, sum1, 2, 4);
    sum1 += __shfl_down_sync(mask, sum1, 1, 4);
    if (quarter == 0) {
        output[row0] = sum0;
        output[row1] = sum1;
    }
}

// Exact POPC SSM-input fusion. The first 128 CTAs use M16 over the 80 wqkv and
// 48 z Q1T tiles; the final two CTAs use the tail-safe M8 mapping for the 48-row
// beta and alpha tiles. Keeping the short tails separate avoids speculative
// reads past their compact 864-byte-per-K-block layout.
extern "C" __global__ void prism_q1t128_q8_popc_fused_ssm_bonsai27b(
    const unsigned int* __restrict__ input_bitplanes,
    const int* __restrict__ input_qsums,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ wqkv_weight,
    const unsigned char* __restrict__ z_weight,
    const unsigned char* __restrict__ beta_weight,
    const unsigned char* __restrict__ alpha_weight,
    float* __restrict__ wqkv_out,
    float* __restrict__ z_out,
    float* __restrict__ beta_out,
    float* __restrict__ alpha_out
) {
    int task = (int)blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = lane >> 2;
    int quarter = lane & 3;
    unsigned int mask = 0xffffffffu;

    if (task < 128) {
        const unsigned char* weight = task < 80 ? wqkv_weight : z_weight;
        float* output = task < 80 ? wqkv_out : z_out;
        int tile = task < 80 ? task : task - 80;
        int row0 = tile * 128 + warp * 16 + group;
        int row1 = row0 + 8;
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll 1
        for (int block = 0; block < 40; block++) {
            int chunk = block * 4 + quarter;
            unsigned int planes[8];
            #pragma unroll
            for (int plane = 0; plane < 8; plane++)
                planes[plane] = input_bitplanes[(long)chunk * 8 + plane];
            int qsum = input_qsums[chunk];
            float da = input_scales[chunk];
            long group0 = ((long)(row0 >> 7) * 40 + block) * 2304;
            long group1 = ((long)(row1 >> 7) * 40 + block) * 2304;
            const unsigned int* signs0 = (const unsigned int*)(weight + group0
                + (row0 & 127) * 16 + quarter * 4);
            const unsigned int* signs1 = (const unsigned int*)(weight + group1
                + (row1 & 127) * 16 + quarter * 4);
            int dot0 = prism_q1_q8_popc_dot32(*signs0, planes, qsum);
            int dot1 = prism_q1_q8_popc_dot32(*signs1, planes, qsum);
            unsigned int scale0 = 0u, scale1 = 0u;
            if (quarter == 0) {
                const unsigned char* s0 = weight + group0 + 2048 + (row0 & 127) * 2;
                const unsigned char* s1 = weight + group1 + 2048 + (row1 & 127) * 2;
                scale0 = (unsigned int)s0[0] | ((unsigned int)s0[1] << 8);
                scale1 = (unsigned int)s1[0] | ((unsigned int)s1[1] << 8);
            }
            scale0 = __shfl_sync(mask, scale0, 0, 4);
            scale1 = __shfl_sync(mask, scale1, 0, 4);
            sum0 += ((float)dot0 * f16_bits_to_f32((unsigned short)scale0)) * da;
            sum1 += ((float)dot1 * f16_bits_to_f32((unsigned short)scale1)) * da;
        }
        sum0 += __shfl_down_sync(mask, sum0, 2, 4);
        sum0 += __shfl_down_sync(mask, sum0, 1, 4);
        sum1 += __shfl_down_sync(mask, sum1, 2, 4);
        sum1 += __shfl_down_sync(mask, sum1, 1, 4);
        if (quarter == 0) {
            output[row0] = sum0;
            output[row1] = sum1;
        }
        return;
    }

    const unsigned char* weight = task == 128 ? beta_weight : alpha_weight;
    float* output = task == 128 ? beta_out : alpha_out;
    int row = warp * 8 + group;
    if (row >= 48) return;
    float sum = 0.0f;
    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        int chunk = block * 4 + quarter;
        unsigned int planes[8];
        #pragma unroll
        for (int plane = 0; plane < 8; plane++)
            planes[plane] = input_bitplanes[(long)chunk * 8 + plane];
        int qsum = input_qsums[chunk];
        float da = input_scales[chunk];
        const unsigned int* signs = (const unsigned int*)(weight
            + (long)block * 864 + row * 16 + quarter * 4);
        int dot = prism_q1_q8_popc_dot32(*signs, planes, qsum);
        unsigned int scale = 0u;
        if (quarter == 0) {
            const unsigned char* s = weight + (long)block * 864 + 768 + row * 2;
            scale = (unsigned int)s[0] | ((unsigned int)s[1] << 8);
        }
        scale = __shfl_sync(mask, scale, 0, 4);
        sum += ((float)dot * f16_bits_to_f32((unsigned short)scale)) * da;
    }
    sum += __shfl_down_sync(mask, sum, 2, 4);
    sum += __shfl_down_sync(mask, sum, 1, 4);
    if (quarter == 0) output[row] = sum;
}

// Shape-specialized Bonsai-27B projection fusion. These kernels only merge
// read-only projections that consume the same Q8 activation; every output keeps
// the Q1T128 decode kernel's 4-lane row ownership, 40-block accumulation order,
// DP4A contraction, scale multiplication, and subgroup reduction. Epilogues
// deliberately remain separate.
__device__ __forceinline__ int prism_q1t_bonsai_dp4a_16(
    unsigned int q, int a0, int a1, int a2, int a3, int acc
) {
    int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
    int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
    int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
    int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
    int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
    int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
    acc = __dp4a((int)__byte_perm(s0, s1, 0x5410), a0, acc);
    acc = __dp4a((int)__byte_perm(s0, s1, 0x7632), a1, acc);
    acc = __dp4a((int)__byte_perm(s2, s3, 0x5410), a2, acc);
    acc = __dp4a((int)__byte_perm(s2, s3, 0x7632), a3, acc);
    return acc;
}

__device__ __forceinline__ unsigned int prism_q1t_bonsai_signs(
    const unsigned char* weight, int row, int block, int quarter
) {
    long group = ((long)(row >> 7) * 40 + block) * 2304;
    const unsigned int* signs = (const unsigned int*)(weight + group
        + (row & 127) * 16 + quarter * 4);
    return *signs;
}

__device__ __forceinline__ float prism_q1t_bonsai_scale(
    const unsigned char* weight, int row, int block, int quarter
) {
    unsigned int bits = 0u;
    if (quarter == 0) {
        long group = ((long)(row >> 7) * 40 + block) * 2304;
        const unsigned char* scale = weight + group + 2048 + (row & 127) * 2;
        bits = (unsigned int)scale[0] | ((unsigned int)scale[1] << 8);
    }
    bits = __shfl_sync(0xffffffffu, bits, 0, 4);
    return f16_bits_to_f32((unsigned short)bits);
}

// The 48-row beta/alpha tensors have one short Q1T row tile per K block:
// [48*16 signs][48*2 scales] == 864 bytes.
__device__ __forceinline__ unsigned int prism_q1t_bonsai_tail48_signs(
    const unsigned char* weight, int row, int block, int quarter
) {
    const unsigned int* signs = (const unsigned int*)(weight
        + (long)block * 864 + row * 16 + quarter * 4);
    return *signs;
}

__device__ __forceinline__ float prism_q1t_bonsai_tail48_scale(
    const unsigned char* weight, int row, int block, int quarter
) {
    unsigned int bits = 0u;
    if (quarter == 0) {
        const unsigned char* scale = weight + (long)block * 864 + 768 + row * 2;
        bits = (unsigned int)scale[0] | ((unsigned int)scale[1] << 8);
    }
    bits = __shfl_sync(0xffffffffu, bits, 0, 4);
    return f16_bits_to_f32((unsigned short)bits);
}

// Full-attention projections: all 192 CTAs produce one 64-row qgate tile.
// Every sixth CTA also produces one of sixteen K or sixteen V tiles, spreading
// the secondary work across the anchor grid instead of front-loading it.
extern "C" __global__ void prism_q1t128_fused_full_bonsai27b(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ qgate_weight,
    const unsigned char* __restrict__ k_weight,
    const unsigned char* __restrict__ v_weight,
    float* __restrict__ qgate_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_in_warp = lane >> 2;
    int quarter = lane & 3;
    int local_row = warp * 8 + row_in_warp;
    int qgate_row = (int)blockIdx.x * 64 + local_row;

    bool paired = ((int)blockIdx.x % 6) == 0;
    int paired_rank = (int)blockIdx.x / 6;
    bool paired_k = paired_rank < 16;
    int secondary_row = (paired_rank & 15) * 64 + local_row;
    const unsigned char* secondary_weight = paired_k ? k_weight : v_weight;
    float* secondary_out = paired_k ? k_out : v_out;

    float qgate_sum = 0.0f;
    float secondary_sum = 0.0f;
    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        const int* aq = (const int*)(input_quants + (long)(block * 4 + quarter) * 32);
        unsigned int qgate_signs = prism_q1t_bonsai_signs(
            qgate_weight, qgate_row, block, quarter);
        unsigned int secondary_signs = paired
            ? prism_q1t_bonsai_signs(secondary_weight, secondary_row, block, quarter)
            : 0u;
        int qgate_dot = 0;
        int secondary_dot = 0;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int a0 = aq[half * 4 + 0];
            int a1 = aq[half * 4 + 1];
            int a2 = aq[half * 4 + 2];
            int a3 = aq[half * 4 + 3];
            qgate_dot = prism_q1t_bonsai_dp4a_16(
                qgate_signs >> (half * 16), a0, a1, a2, a3, qgate_dot);
            if (paired) {
                secondary_dot = prism_q1t_bonsai_dp4a_16(
                    secondary_signs >> (half * 16), a0, a1, a2, a3, secondary_dot);
            }
        }
        float da = input_scales[block * 4 + quarter];
        float qgate_scale = prism_q1t_bonsai_scale(
            qgate_weight, qgate_row, block, quarter);
        qgate_sum += ((float)qgate_dot * qgate_scale) * da;
        if (paired) {
            float secondary_scale = prism_q1t_bonsai_scale(
                secondary_weight, secondary_row, block, quarter);
            secondary_sum += ((float)secondary_dot * secondary_scale) * da;
        }
    }
    qgate_sum += __shfl_down_sync(0xffffffffu, qgate_sum, 2, 4);
    qgate_sum += __shfl_down_sync(0xffffffffu, qgate_sum, 1, 4);
    if (paired) {
        secondary_sum += __shfl_down_sync(0xffffffffu, secondary_sum, 2, 4);
        secondary_sum += __shfl_down_sync(0xffffffffu, secondary_sum, 1, 4);
    }
    if (quarter == 0) {
        qgate_out[qgate_row] = qgate_sum;
        if (paired) secondary_out[secondary_row] = secondary_sum;
    }
}

// SSM input projections: all 160 CTAs produce wqkv. In each five-CTA group,
// three also produce consecutive z tiles (96 total); the last CTA additionally
// contracts both 48-row beta/alpha tails.
extern "C" __global__ void prism_q1t128_fused_ssm_bonsai27b(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ wqkv_weight,
    const unsigned char* __restrict__ z_weight,
    const unsigned char* __restrict__ beta_weight,
    const unsigned char* __restrict__ alpha_weight,
    float* __restrict__ wqkv_out,
    float* __restrict__ z_out,
    float* __restrict__ beta_out,
    float* __restrict__ alpha_out
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_in_warp = lane >> 2;
    int quarter = lane & 3;
    int local_row = warp * 8 + row_in_warp;
    int cta = (int)blockIdx.x;
    int wqkv_row = cta * 64 + local_row;

    int group_slot = cta % 5;
    bool paired_z = group_slot < 3;
    int z_row = ((cta / 5) * 3 + group_slot) * 64 + local_row;
    bool paired_tail = cta == 159 && warp < 6;
    int tail_row = local_row;

    float wqkv_sum = 0.0f;
    float z_sum = 0.0f;
    float beta_sum = 0.0f;
    float alpha_sum = 0.0f;
    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        const int* aq = (const int*)(input_quants + (long)(block * 4 + quarter) * 32);
        unsigned int wqkv_signs = prism_q1t_bonsai_signs(
            wqkv_weight, wqkv_row, block, quarter);
        unsigned int z_signs = paired_z
            ? prism_q1t_bonsai_signs(z_weight, z_row, block, quarter)
            : 0u;
        unsigned int beta_signs = paired_tail
            ? prism_q1t_bonsai_tail48_signs(beta_weight, tail_row, block, quarter)
            : 0u;
        unsigned int alpha_signs = paired_tail
            ? prism_q1t_bonsai_tail48_signs(alpha_weight, tail_row, block, quarter)
            : 0u;
        int wqkv_dot = 0;
        int z_dot = 0;
        int beta_dot = 0;
        int alpha_dot = 0;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int a0 = aq[half * 4 + 0];
            int a1 = aq[half * 4 + 1];
            int a2 = aq[half * 4 + 2];
            int a3 = aq[half * 4 + 3];
            wqkv_dot = prism_q1t_bonsai_dp4a_16(
                wqkv_signs >> (half * 16), a0, a1, a2, a3, wqkv_dot);
            if (paired_z) {
                z_dot = prism_q1t_bonsai_dp4a_16(
                    z_signs >> (half * 16), a0, a1, a2, a3, z_dot);
            }
            if (paired_tail) {
                beta_dot = prism_q1t_bonsai_dp4a_16(
                    beta_signs >> (half * 16), a0, a1, a2, a3, beta_dot);
                alpha_dot = prism_q1t_bonsai_dp4a_16(
                    alpha_signs >> (half * 16), a0, a1, a2, a3, alpha_dot);
            }
        }
        float da = input_scales[block * 4 + quarter];
        float wqkv_scale = prism_q1t_bonsai_scale(
            wqkv_weight, wqkv_row, block, quarter);
        wqkv_sum += ((float)wqkv_dot * wqkv_scale) * da;
        if (paired_z) {
            float z_scale = prism_q1t_bonsai_scale(z_weight, z_row, block, quarter);
            z_sum += ((float)z_dot * z_scale) * da;
        }
        if (paired_tail) {
            float beta_scale = prism_q1t_bonsai_tail48_scale(
                beta_weight, tail_row, block, quarter);
            float alpha_scale = prism_q1t_bonsai_tail48_scale(
                alpha_weight, tail_row, block, quarter);
            beta_sum += ((float)beta_dot * beta_scale) * da;
            alpha_sum += ((float)alpha_dot * alpha_scale) * da;
        }
    }
    wqkv_sum += __shfl_down_sync(0xffffffffu, wqkv_sum, 2, 4);
    wqkv_sum += __shfl_down_sync(0xffffffffu, wqkv_sum, 1, 4);
    if (paired_z) {
        z_sum += __shfl_down_sync(0xffffffffu, z_sum, 2, 4);
        z_sum += __shfl_down_sync(0xffffffffu, z_sum, 1, 4);
    }
    if (paired_tail) {
        beta_sum += __shfl_down_sync(0xffffffffu, beta_sum, 2, 4);
        beta_sum += __shfl_down_sync(0xffffffffu, beta_sum, 1, 4);
        alpha_sum += __shfl_down_sync(0xffffffffu, alpha_sum, 2, 4);
        alpha_sum += __shfl_down_sync(0xffffffffu, alpha_sum, 1, 4);
    }
    if (quarter == 0) {
        wqkv_out[wqkv_row] = wqkv_sum;
        if (paired_z) z_out[z_row] = z_sum;
        if (paired_tail) {
            beta_out[tail_row] = beta_sum;
            alpha_out[tail_row] = alpha_sum;
        }
    }
}

// FFN gate/up have identical 17,408-row geometry. Each subgroup owns the same
// row in both matrices and reuses each of the eight Q8 words loaded per K block.
extern "C" __global__ void prism_q1t128_fused_ffn_bonsai27b(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ up_weight,
    float* __restrict__ gate_out,
    float* __restrict__ up_out
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_in_warp = lane >> 2;
    int quarter = lane & 3;
    int row = (int)blockIdx.x * 64 + warp * 8 + row_in_warp;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    #pragma unroll 1
    for (int block = 0; block < 40; block++) {
        const int* aq = (const int*)(input_quants + (long)(block * 4 + quarter) * 32);
        unsigned int gate_signs = prism_q1t_bonsai_signs(
            gate_weight, row, block, quarter);
        unsigned int up_signs = prism_q1t_bonsai_signs(
            up_weight, row, block, quarter);
        int gate_dot = 0;
        int up_dot = 0;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int a0 = aq[half * 4 + 0];
            int a1 = aq[half * 4 + 1];
            int a2 = aq[half * 4 + 2];
            int a3 = aq[half * 4 + 3];
            gate_dot = prism_q1t_bonsai_dp4a_16(
                gate_signs >> (half * 16), a0, a1, a2, a3, gate_dot);
            up_dot = prism_q1t_bonsai_dp4a_16(
                up_signs >> (half * 16), a0, a1, a2, a3, up_dot);
        }
        float da = input_scales[block * 4 + quarter];
        float gate_scale = prism_q1t_bonsai_scale(gate_weight, row, block, quarter);
        float up_scale = prism_q1t_bonsai_scale(up_weight, row, block, quarter);
        gate_sum += ((float)gate_dot * gate_scale) * da;
        up_sum += ((float)up_dot * up_scale) * da;
    }
    gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, 2, 4);
    gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, 1, 4);
    up_sum += __shfl_down_sync(0xffffffffu, up_sum, 2, 4);
    up_sum += __shfl_down_sync(0xffffffffu, up_sum, 1, 4);
    if (quarter == 0) {
        gate_out[row] = gate_sum;
        up_out[row] = up_sum;
    }
}

// Q1_0 prompt GEMM. One warp owns four output rows and evaluates one or two
// token-major activation rows while each packed weight block is hot in
// registers. This preserves the decode kernel's per-token lane/block reduction
// exactly, but amortizes the dominant weight read over the prompt tile. A CTA
// stages the same four-block activation window for all eight row warps.
extern "C" __global__ void prism_q1_f32_gemm_batched(
    const float* __restrict__ input, const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row, int k_tokens, int q1_tiled,
    float* __restrict__ output
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int first_row = (blockIdx.x * warps_per_block + warp) * 4;
    int block_lane = lane >> 3;
    int slice = (lane & 7) << 4;
    unsigned mask = 0xffffffffu;
    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    extern __shared__ float staged_input[]; // [k_tokens<=2][4 * 128]

    for (int chunk = 0; chunk < blocks_per_row; chunk += 4) {
        int remaining = blocks_per_row - chunk;
        int staged_values = (remaining < 4 ? remaining : 4) * 128;
        int all_values = k_tokens * staged_values;
        for (int i = threadIdx.x; i < all_values; i += blockDim.x) {
            int token = i / staged_values;
            int within = i - token * staged_values;
            staged_input[token * 512 + within] =
                input[(long)token * cols + chunk * 128 + within];
        }
        __syncthreads();

        int b = chunk + block_lane;
        if (first_row < rows && b < blocks_per_row) {
            int input_base = block_lane * 128 + slice;
            float values[2][16];
            float input_sums[2] = { 0.0f, 0.0f };
            #pragma unroll
            for (int token = 0; token < 2; token++) {
                if (token < k_tokens) {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        float value = staged_input[token * 512 + input_base + i];
                        values[token][i] = value;
                        input_sums[token] += value;
                    }
                }
            }
            #pragma unroll
            for (int ro = 0; ro < 4; ro++) {
                int row = first_row + ro;
                if (row < rows) {
                    const unsigned char* scale_ptr = prism_q1_scale_ptr(
                        weight_bytes, rows, blocks_per_row, row, b, q1_tiled);
                    float d = f16_bits_to_f32(
                        (unsigned short)scale_ptr[0]
                        | ((unsigned short)scale_ptr[1] << 8));
                    const unsigned char* qs = prism_q1_sign_ptr(
                        weight_bytes, rows, blocks_per_row, row, b, q1_tiled)
                        + (slice >> 3);
                    unsigned char b0 = qs[0], b1 = qs[1];
                    #pragma unroll
                    for (int token = 0; token < 2; token++) {
                        if (token < k_tokens) {
                            float selected = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < 8; i++)
                                selected += (b0 & (1u << i)) ? values[token][i] : 0.0f;
                            #pragma unroll
                            for (int i = 0; i < 8; i++)
                                selected += (b1 & (1u << i)) ? values[token][i + 8] : 0.0f;
                            sums[token * 4 + ro] +=
                                d * (2.0f * selected - input_sums[token]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int token = 0; token < 2; token++) {
        if (token < k_tokens) {
            #pragma unroll
            for (int ro = 0; ro < 4; ro++) {
                float total = sums[token * 4 + ro];
                total += __shfl_down_sync(mask, total, 16);
                total += __shfl_down_sync(mask, total, 8);
                total += __shfl_down_sync(mask, total, 4);
                total += __shfl_down_sync(mask, total, 2);
                total += __shfl_down_sync(mask, total, 1);
                int row = first_row + ro;
                if (lane == 0 && row < rows)
                    output[(long)token * rows + row] = total;
            }
        }
    }
}

// Ampere prompt MMQ prototype: packed Q1_0 weights times Q8_0 activations.
// One warp owns one output row and an eight-token tile. Each lane walks a
// disjoint subset of the row's 32-element chunks, expands the 1-bit signs into
// four packed int8 lanes with __byte_perm, and evaluates them with __dp4a.
// Weight decode is amortized across all eight prompt tokens without expanding
// the resident model. This deliberately follows llama.cpp's Q1_0 x Q8_1 MMQ
// arithmetic; unlike the f32 parity lane above it is a fast, quantized-activation
// prompt path and therefore has a separate correctness/performance gate.
extern "C" __global__ void prism_q1_q8_gemm_batched(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row, int k_tokens, int q1_tiled,
    float* __restrict__ output, int residual
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token_base = blockIdx.y * 8;
    if (row >= rows || token_base >= k_tokens) return;

    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int chunks_per_row = blocks_per_row * 4;
    int activation_blocks_per_row = cols >> 5;

    for (int chunk_index = lane; chunk_index < chunks_per_row; chunk_index += 32) {
        int weight_block = chunk_index >> 2;
        int chunk = chunk_index & 3;
        const unsigned char* scale_ptr = prism_q1_scale_ptr(
            weight_bytes, rows, blocks_per_row, row, weight_block, q1_tiled);
        float dw = f16_bits_to_f32(
            (unsigned short)scale_ptr[0] | ((unsigned short)scale_ptr[1] << 8));
        const unsigned short* qs = (const unsigned short*)(prism_q1_sign_ptr(
            weight_bytes, rows, blocks_per_row, row, weight_block, q1_tiled)
            + chunk * 4);

        int packed_weights[8];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int q = (int)qs[j];
            int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
            int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
            int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
            int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
            int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
            int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
            packed_weights[j * 4 + 0] = __byte_perm(s0, s1, 0x5410);
            packed_weights[j * 4 + 1] = __byte_perm(s0, s1, 0x7632);
            packed_weights[j * 4 + 2] = __byte_perm(s2, s3, 0x5410);
            packed_weights[j * 4 + 3] = __byte_perm(s2, s3, 0x7632);
        }

        #pragma unroll
        for (int token_offset = 0; token_offset < 8; token_offset++) {
            int token = token_base + token_offset;
            if (token < k_tokens) {
                const int* aq = (const int*)(input_quants
                    + (long)token * cols + (long)chunk_index * 32);
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    sumi = __dp4a(packed_weights[j], aq[j], sumi);
                float da = input_scales[(long)token * activation_blocks_per_row + chunk_index];
                sums[token_offset] += dw * da * (float)sumi;
            }
        }
    }

    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int token_offset = 0; token_offset < 8; token_offset++) {
        int token = token_base + token_offset;
        if (token < k_tokens) {
            float total = sums[token_offset];
            total += __shfl_down_sync(mask, total, 16);
            total += __shfl_down_sync(mask, total, 8);
            total += __shfl_down_sync(mask, total, 4);
            total += __shfl_down_sync(mask, total, 2);
            total += __shfl_down_sync(mask, total, 1);
            if (lane == 0) {
                long oi = (long)token * rows + row;
                output[oi] = residual ? output[oi] + total : total;
            }
        }
    }
}

// Ampere/Turing tensor-core prompt MMQ. A 256-thread CTA owns 128 output rows
// by as many as 128 prompt tokens. Warp w owns its 16-token column tile and
// walks the eight 16-row tiles; all warps reuse the same expanded 128x32 Q1 A
// tile, while all row tiles reuse the same 32x128 Q8 B tile. This I=128/J=128
// geometry is deliberately large: prompt performance is governed by reading
// the 27B weights and activations once per broad tile, not by issuing many tiny
// GEMVs. Two signed-int8 WMMA operations cover each K=32 activation block;
// scaling remains outside MMA to preserve the Q1_0 x Q8_0 formula.
extern "C" __global__ void prism_q1_q8_wmma_gemm_batched(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row, int k_tokens, int q1_tiled,
    float* __restrict__ output, int residual
) {
#if defined(CAMELID_HAS_WMMA) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_base = blockIdx.x * 128;
    int token_base = blockIdx.y * 128 + warp * 16;
    int activation_blocks_per_row = cols >> 5;
    bool token_active = token_base < k_tokens;

    __shared__ __align__(32) signed char tile_a[128 * 32];
    __shared__ __align__(32) signed char tile_b[8 * 16 * 32];
    __shared__ float weight_scales[128];
    __shared__ float activation_scales[128];
    float sums[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) sums[i] = 0.0f;

    for (int weight_block = 0; weight_block < blocks_per_row; weight_block++) {
        if (threadIdx.x < 128) {
            int row = row_base + threadIdx.x;
            float scale = 0.0f;
            if (row < rows) {
                const unsigned char* scale_ptr = prism_q1_scale_ptr(
                    weight_bytes, rows, blocks_per_row, row, weight_block, q1_tiled);
                scale = f16_bits_to_f32((unsigned short)scale_ptr[0]
                    | ((unsigned short)scale_ptr[1] << 8));
            }
            weight_scales[threadIdx.x] = scale;
        }
        for (int chunk = 0; chunk < 4; chunk++) {
            // Cooperatively expand 4096 Q1 signs. Each warp walks contiguous
            // K=32 rows, so the compact bytes stay hot despite the row stride.
            for (int ai = threadIdx.x; ai < 128 * 32; ai += blockDim.x) {
                int ar = ai >> 5;
                int ak = ai & 31;
                int row = row_base + ar;
                signed char av = 0;
                if (row < rows) {
                    int within = chunk * 32 + ak;
                    const unsigned char* signs = prism_q1_sign_ptr(
                        weight_bytes, rows, blocks_per_row, row, weight_block, q1_tiled);
                    av = (signs[within >> 3] & (1u << (within & 7))) ? 1 : -1;
                }
                tile_a[ai] = av;
            }

            // Each warp stages one K=32 by N=16 B tile in column-major order.
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int bi = lane + j * 32;
                int token_col = bi >> 5;
                int bk = bi & 31;
                int token = token_base + token_col;
                signed char bv = 0;
                if (token < k_tokens) {
                    bv = input_quants[(long)token * cols
                        + weight_block * 128 + chunk * 32 + bk];
                }
                tile_b[warp * 512 + bi] = bv;
            }
            if (threadIdx.x < 128) {
                int token = blockIdx.y * 128 + threadIdx.x;
                activation_scales[threadIdx.x] = token < k_tokens
                    ? input_scales[(long)token * activation_blocks_per_row
                        + weight_block * 4 + chunk]
                    : 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int row_group = 0; row_group < 8; row_group++) {
                bool active = token_active && row_base + row_group * 16 < rows;
                if (active) {
                    // ldmatrix loads the 16x32 row-major A tile into the exact
                    // four-register layout consumed by Ampere's m16n8k32 IMMA.
                    // This avoids two WMMA k16 instructions and, critically,
                    // keeps the four int32 outputs in registers instead of
                    // store-to-shared/read-back round trips.
                    int a0, a1, a2, a3;
                    const int* a_row = (const int*)(tile_a + row_group * 16 * 32);
                    const int* a_src = a_row
                        + (lane % 16) * 8 + (lane / 16) * 4;
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                        : "l"(a_src));

                    const int* b_row = (const int*)(tile_b + warp * 512);
                    #pragma unroll
                    for (int token_half = 0; token_half < 2; token_half++) {
                        const int* b_half = b_row + token_half * 8 * 8;
                        int b0 = b_half[(lane / 4) * 8 + (lane % 4)];
                        int b1 = b_half[(lane / 4) * 8 + (lane % 4) + 4];
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                            "{%0, %1, %2, %3};"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0), "r"(b1));
                        int cv[4] = { c0, c1, c2, c3 };
                        #pragma unroll
                        for (int l = 0; l < 4; l++) {
                            int cr = (l / 2) * 8 + lane / 4;
                            int token_col = token_half * 8
                                + (lane % 4) * 2 + (l % 2);
                            int row_in_tile = row_group * 16 + cr;
                            int row = row_base + row_in_tile;
                            int token = token_base + token_col;
                            if (row < rows && token < k_tokens) {
                                float dw = weight_scales[row_in_tile];
                                float da = activation_scales[warp * 16 + token_col];
                                sums[row_group * 8 + token_half * 4 + l] +=
                                    (float)cv[l] * dw * da;
                            }
                        }
                    }
                }
            }
            // The next K=32 chunk overwrites CTA-wide A/B. This is the sole
            // block-wide fence needed after the independent row-group sweep.
            __syncthreads();
        }
    }

    if (token_active) {
        #pragma unroll
        for (int row_group = 0; row_group < 8; row_group++) {
            #pragma unroll
            for (int token_half = 0; token_half < 2; token_half++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int cr = (l / 2) * 8 + lane / 4;
                    int token_col = token_half * 8
                        + (lane % 4) * 2 + (l % 2);
                    int row = row_base + row_group * 16 + cr;
                    int token = token_base + token_col;
                    if (row < rows && token < k_tokens)
                    {
                        long oi = (long)token * rows + row;
                        float value = sums[row_group * 8 + token_half * 4 + l];
                        output[oi] = residual ? output[oi] + value : value;
                    }
                }
            }
        }
    }
#endif
}

// Experimental Ampere binary-MMA prompt lane. Activations are quantized in
// 128-value blocks and stored as eight two's-complement bitplanes. This makes
// one Q1_0 weight block and one activation block line up exactly with
// m16n8k128.b1.b1.xor.popc. Production dispatch admits this lane only on SM80+
// after strict-mode, force-disable, shape, and measured-crossover gates pass.
//
// Packed activation layout (u32 words):
//   [k_block][bit_plane][token][word_in_128]
// A token/block occupies 8 * 4 words == 128 bytes, the same payload as int8.
extern "C" __global__ void prism_q8_b128_bitpack(
    const float* __restrict__ input,
    unsigned int* __restrict__ bitplanes,
    float* __restrict__ scales,
    int cols, int k_tokens
) {
    int k_block = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int within = warp * 32 + lane;
    if (token >= k_tokens || within >= 128) return;

    float value = input[(long)token * cols + k_block * 128 + within];
    float amax = fabsf(value);
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, offset));

    __shared__ float warp_max[4];
    __shared__ float unrounded_scale;
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float block_max = lane < 4 ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max, offset));
        if (lane == 0) {
            unrounded_scale = block_max / 127.0f;
            scales[(long)k_block * k_tokens + token] = f16_round(unrounded_scale);
        }
    }
    __syncthreads();

    float scale = unrounded_scale;
    float qf = scale == 0.0f ? 0.0f : rintf(value / scale);
    if (qf > 127.0f) qf = 127.0f;
    if (qf < -127.0f) qf = -127.0f;
    int q = (int)qf;
    unsigned int uq = (unsigned int)q & 0xffu;
    #pragma unroll
    for (int plane = 0; plane < 8; plane++) {
        unsigned int bits = __ballot_sync(mask, ((uq >> plane) & 1u) != 0u);
        if (lane == 0) {
            long dst = (((long)k_block * 8 + plane) * k_tokens + token) * 4 + warp;
            bitplanes[dst] = bits;
        }
    }
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned int prism_load_u32_le(const unsigned char* p) {
    return (unsigned int)p[0]
        | ((unsigned int)p[1] << 8)
        | ((unsigned int)p[2] << 16)
        | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ void prism_bmma_xor_m16n8k128(
    unsigned int a0, unsigned int a1, unsigned int b,
    int& d0, int& d1, int& d2, int& d3
) {
    int zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b),
          "r"(zero), "r"(zero), "r"(zero), "r"(zero));
}
#endif

// One 256-thread CTA computes a 128-row by 128-token output tile. Warp w owns
// rows [16*w, 16*w+16) and walks sixteen N=8 BMMA fragments. For each K=128
// block, the CTA stages compact A bits and B bitplanes once; each thread keeps
// its 64 output accumulators in registers across the complete K loop.
extern "C" __global__ void prism_q1_q8_b128_bmma_gemm_batched(
    const unsigned int* __restrict__ input_bitplanes,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int cols, int blocks_per_row, int k_tokens, int q1_tiled,
    float* __restrict__ output, int residual
) {
#if __CUDA_ARCH__ >= 800
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int group = lane >> 2;
    int thread_in_group = lane & 3;
    int row_base = blockIdx.x * 128;
    int token_base = blockIdx.y * 128;
    int token_count = k_tokens - token_base;
    if (token_count > 128) token_count = 128;
    int n_tiles = (token_count + 7) >> 3;
    int warp_row = warp * 16;
    unsigned mask = 0xffffffffu;

    extern __shared__ unsigned int tile_a[];
    __shared__ __align__(16) unsigned int tile_b[8 * 128 * 4];
    __shared__ float weight_scales[128];
    __shared__ float activation_scales[128];

    float sums[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) sums[i] = 0.0f;

    for (int k_block = 0; k_block < blocks_per_row; k_block++) {
        // Raw Q1 needs an aligned shared staging tile. Q1T128 already stores the
        // complete 128-row sign slab contiguously, so its fragments load direct
        // from global and tile_a is untouched.
        if (tid < 128) {
            int row = row_base + tid;
            float dw = 0.0f;
            if (!q1_tiled) {
                #pragma unroll
                for (int word = 0; word < 4; word++) tile_a[tid * 4 + word] = 0u;
            }
            if (row < rows) {
                const unsigned char* scale_ptr = prism_q1_scale_ptr(
                    weight_bytes, rows, blocks_per_row, row, k_block, q1_tiled);
                dw = f16_bits_to_f32((unsigned short)scale_ptr[0]
                    | ((unsigned short)scale_ptr[1] << 8));
                if (!q1_tiled) {
                    const unsigned char* signs = prism_q1_sign_ptr(
                        weight_bytes, rows, blocks_per_row, row, k_block, 0);
                    #pragma unroll
                    for (int word = 0; word < 4; word++)
                        tile_a[tid * 4 + word] = prism_load_u32_le(signs + word * 4);
                }
            }
            weight_scales[tid] = dw;

            int token = token_base + tid;
            activation_scales[tid] = token < k_tokens
                ? input_scales[(long)k_block * k_tokens + token]
                : 0.0f;
        }

        // Only stage live columns for a tail prompt tile. The shared stride
        // remains 128 so the BMMA fragment address math is identical for full
        // and partial tiles, but N=8/16 no longer pays the N=128 copy cost.
        int live_plane_words = 8 * token_count * 4;
        for (int index = tid; index < live_plane_words; index += blockDim.x) {
            int word = index & 3;
            int plane_token = index >> 2;
            int token_in_tile = plane_token % token_count;
            int plane = plane_token / token_count;
            int token = token_base + token_in_tile;
            tile_b[(plane * 128 + token_in_tile) * 4 + word] =
                input_bitplanes[(((long)k_block * 8 + plane) * k_tokens + token) * 4 + word];
        }
        __syncthreads();

        int fragment_row0 = row_base + warp_row + group;
        int fragment_row1 = fragment_row0 + 8;
        unsigned int a0 = 0u, a1 = 0u;
        if (q1_tiled) {
            if (fragment_row0 < rows) {
                const unsigned char* signs0 = prism_q1_sign_ptr(
                    weight_bytes, rows, blocks_per_row, fragment_row0, k_block, 1);
                a0 = prism_load_u32_le(signs0 + thread_in_group * 4);
            }
            if (fragment_row1 < rows) {
                const unsigned char* signs1 = prism_q1_sign_ptr(
                    weight_bytes, rows, blocks_per_row, fragment_row1, k_block, 1);
                a1 = prism_load_u32_le(signs1 + thread_in_group * 4);
            }
        } else {
            a0 = tile_a[(warp_row + group) * 4 + thread_in_group];
            a1 = tile_a[(warp_row + group + 8) * 4 + thread_in_group];
        }
        int pop0 = __popc(a0);
        int pop1 = __popc(a1);
        pop0 += __shfl_xor_sync(mask, pop0, 1, 4);
        pop1 += __shfl_xor_sync(mask, pop1, 1, 4);
        pop0 += __shfl_xor_sync(mask, pop0, 2, 4);
        pop1 += __shfl_xor_sync(mask, pop1, 2, 4);

        // For two's-complement q = -128*b7 + sum(2^p*bp), and Q1 sign
        // s=2*w-1, Xp=popc(w xor bp) gives
        //   dot(s,q) = 128*X7 - 64*X6 - ... - X0 - popc(w).
        // Horner evaluation needs only four transient integer registers.
        #pragma unroll
        for (int n_tile = 0; n_tile < n_tiles; n_tile++) {
            int d0, d1, d2, d3;
            unsigned int b = tile_b[(7 * 128 + n_tile * 8 + group) * 4
                + thread_in_group];
            prism_bmma_xor_m16n8k128(a0, a1, b, d0, d1, d2, d3);
            int v0 = d0, v1 = d1, v2 = d2, v3 = d3;
            #pragma unroll
            for (int plane = 6; plane >= 0; plane--) {
                b = tile_b[(plane * 128 + n_tile * 8 + group) * 4
                    + thread_in_group];
                prism_bmma_xor_m16n8k128(a0, a1, b, d0, d1, d2, d3);
                v0 = 2 * v0 - d0;
                v1 = 2 * v1 - d1;
                v2 = 2 * v2 - d2;
                v3 = 2 * v3 - d3;
            }
            v0 -= pop0;
            v1 -= pop0;
            v2 -= pop1;
            v3 -= pop1;

            int token0 = n_tile * 8 + thread_in_group * 2;
            int token1 = token0 + 1;
            float dw0 = weight_scales[warp_row + group];
            float dw1 = weight_scales[warp_row + group + 8];
            float da0 = activation_scales[token0];
            float da1 = activation_scales[token1];
            sums[n_tile * 4 + 0] += ((float)v0 * dw0) * da0;
            sums[n_tile * 4 + 1] += ((float)v1 * dw0) * da1;
            sums[n_tile * 4 + 2] += ((float)v2 * dw1) * da0;
            sums[n_tile * 4 + 3] += ((float)v3 * dw1) * da1;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < n_tiles; n_tile++) {
        int token0 = token_base + n_tile * 8 + thread_in_group * 2;
        int token1 = token0 + 1;
        int row0 = row_base + warp_row + group;
        int row1 = row0 + 8;
        if (row0 < rows && token0 < k_tokens) {
            long oi = (long)token0 * rows + row0;
            float value = sums[n_tile * 4 + 0];
            output[oi] = residual ? output[oi] + value : value;
        }
        if (row0 < rows && token1 < k_tokens) {
            long oi = (long)token1 * rows + row0;
            float value = sums[n_tile * 4 + 1];
            output[oi] = residual ? output[oi] + value : value;
        }
        if (row1 < rows && token0 < k_tokens) {
            long oi = (long)token0 * rows + row1;
            float value = sums[n_tile * 4 + 2];
            output[oi] = residual ? output[oi] + value : value;
        }
        if (row1 < rows && token1 < k_tokens) {
            long oi = (long)token1 * rows + row1;
            float value = sums[n_tile * 4 + 3];
            output[oi] = residual ? output[oi] + value : value;
        }
    }
#endif
}

// Decode one Q4_0 block into packed signed bytes and contract with Q8_0 via
// DP4A. `__vsub4` performs four independent byte subtractions, so subtracting
// 0x08 from each nibble is exactly the scalar `(q & 15) - 8` operation without
// cross-byte borrow. Integer addition order cannot affect the exact i32 result.
__device__ __forceinline__ int q4_0_dot32_dp4a(
    const unsigned char* __restrict__ qs,
    const signed char* __restrict__ y
) {
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < 16; j += 4) {
        unsigned int packed = (unsigned int)qs[j]
            | ((unsigned int)qs[j + 1] << 8)
            | ((unsigned int)qs[j + 2] << 16)
            | ((unsigned int)qs[j + 3] << 24);
        int lo = (int)__vsub4(packed & 0x0f0f0f0fu, 0x08080808u);
        int hi = (int)__vsub4((packed >> 4) & 0x0f0f0f0fu, 0x08080808u);
        int ylo = *((const int*)(y + j));
        int yhi = *((const int*)(y + 16 + j));
        sum = __dp4a(lo, ylo, sum);
        sum = __dp4a(hi, yhi, sum);
    }
    return sum;
}

// Same computation as `q4_0_dot32_dp4a` above, for callers that already hold the
// block's 16 nibble bytes in a register (one aligned `uint4` load out of an SoA
// quant plane) instead of pointing at an unaligned 18-byte wire block. Identical
// low/high nibble split, identical activation pairing, identical __dp4a chain --
// so the integer result is the same integer, and any kernel can be swapped between
// the two forms without moving a single output bit.
//
// Defined here, beside its sibling, because NVRTC needs it before its first use
// and `q4_0_gemv_soa` follows shortly.
__device__ __forceinline__ int q4_0_dot32_dp4a_packed(
    uint4 packed, const signed char* __restrict__ y
) {
    unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned int q = words[j];
        int lo = (int)__vsub4(q & 0x0f0f0f0fu, 0x08080808u);
        int hi = (int)__vsub4((q >> 4) & 0x0f0f0f0fu, 0x08080808u);
        int ylo = *((const int*)(y + j * 4));
        int yhi = *((const int*)(y + 16 + j * 4));
        sum = __dp4a(lo, ylo, sum);
        sum = __dp4a(hi, yhi, sum);
    }
    return sum;
}

// ---- Q4_0 GEMV: one warp per output row, raw 18-byte wire, Q8_0 activation ----
// Bit-identical reproduction of the validated CPU oracle `q4_0_wire_row_dot_scalar`
// (the gemma4 QAT linear lane). Per 18-byte block: scale = f16(blk[0..2]); for
// j in 0..16, lo = (byte & 0xF) - 8, hi = (byte >> 4) - 8; isum += lo*y[j] +
// hi*y[j+16]; term = (float)isum * w_scale * x_scale[b]. Lane 0 sums the per-block
// terms IN ORDER — the exact same ordered-f32 contract as q8_gemv, so the result is
// bit-identical to the CPU. Weights are read RAW (nibbles packed) to keep the 4-bit
// footprint; the activation is Q8_0 (input_scales[bpr] + input_quants[bpr*32] i8),
// staged once in shared like q8_gemv. Packed byte-wise subtract removes the -8
// nibble bias exactly, allowing the integer dot to use DP4A without changing math.
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
            int isum = q4_0_dot32_dp4a(blk + 2, y);
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

// ---- Q4_0 GEMV, quants-first SoA weights ------------------------------------
// Identical arithmetic to `q4_0_gemv` above; the ONLY difference is how the same
// 18 bytes per block reach the registers.
//
// The raw-wire kernel reads its 16 nibble bytes off an 18-byte stride, so the
// block never lands on a 4-byte boundary and `q4_0_dot32_dp4a` has to assemble
// each 32-bit word from four scalar byte loads -- 16 scalar loads per block,
// spanning up to five cache lines, plus two more for the f16 scale. That is the
// same defect that held the sibling q8_gemv at ~52% of memory bandwidth until it
// got its SoA repack (+12% decode on this GPU); Q4_0 never received the fix.
//
// `q4_0_wire_to_soa` splits the tensor into a quants plane and a scales plane:
//   [rows*blocks_per_row*16 nibble bytes][rows*blocks_per_row*2 f16 scale bits]
// Block b of row r therefore starts at a 16-byte aligned offset, so its nibbles
// are ONE `uint4` load, and the scales are a coalesced u16 read. Total bytes are
// unchanged at 18/block, so VRAM residency and every slot budget stay identical.
//
// BIT-IDENTICAL BY CONSTRUCTION, not by observation:
//   * `q4_0_dot32_dp4a_packed` consumes exactly the bytes `q4_0_dot32_dp4a` did,
//     with the same low/high nibble split and the same activation pairing
//     (y[j] with the low nibble, y[16+j] with the high nibble).
//   * The integer `__dp4a` chain is exact regardless of grouping, so `isum` is
//     the same integer.
//   * The per-block float term `(float)isum * w_scale * s_is[b]` is unchanged.
//   * The tail fold is still lane 0 summing `myterms` in increasing block order.
// Only the load instructions change. `q4_0_gemv_soa_matches_wire` pins this.
extern "C" __global__ void q4_0_gemv_soa(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ unsigned char smem40s[];
    signed char* s_iq = (signed char*)smem40s;                        // blocks_per_row*32 i8
    float* s_is = (float*)(smem40s + (long)blocks_per_row * 32);      // blocks_per_row f32
    float* terms = (float*)(smem40s + (long)blocks_per_row * 36);     // warps*blocks_per_row f32
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
    // Plane bases. The scale plane begins after every row's nibbles.
    const uint4* quant_plane = (const uint4*)weight_bytes;
    const unsigned short* scale_plane =
        (const unsigned short*)(weight_bytes + (long)rows * blocks_per_row * 16);
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            long idx = row_block0 + b;
            uint4 packed = quant_plane[idx];            // one aligned 16-byte load
            float w_scale = f16_bits_to_f32(scale_plane[idx]);
            const signed char* y = s_iq + (long)b * 32;
            int isum = q4_0_dot32_dp4a_packed(packed, y);
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

// ---- Q4_0 GEMV, quants-first SoA weights + native f32 activation ----------
//
// The Gemma 4 MTP assistant is packed from BF16 directly to Q4_0. Quantizing
// its activation to Q8_0 as well would introduce a second approximation that
// the established full-Q4 Metal assistant does not make. This lane therefore
// consumes the f32 activation directly while retaining the 18-byte Q4_0 weight
// footprint and aligned SoA weight reads.
//
// The reduction order is explicit and deterministic:
//   1. one lane owns each 32-value Q4_0 block;
//   2. that lane accumulates columns 0..15, then 16..31, in order;
//   3. the f16 block scale is applied once to that ordered block dot;
//   4. lane 0 folds block terms in increasing block order.
//
// NVRTC is compiled with --fmad=false, so the CPU oracle can reproduce every
// multiply and add bit for bit. The input row is staged once per CTA; at the
// widest official assistant contraction (8192 columns), eight warps consume
// 40 KiB of shared memory and stay below the launcher's 46 KiB budget.
extern "C" __global__ void q4_0_f32_gemv_soa(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int blocks_per_row,
    float* __restrict__ output, int residual
) {
    extern __shared__ float smem40f[];
    float* staged_input = smem40f;                                  // blocks_per_row*32 f32
    float* terms = staged_input + (long)blocks_per_row * 32;         // warps*blocks_per_row f32
    int tid = threadIdx.x;
    int cols = blocks_per_row * 32;
    for (int i = tid; i < cols; i += blockDim.x) staged_input[i] = input[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    const uint4* quant_plane = (const uint4*)weight_bytes;
    const unsigned short* scale_plane =
        (const unsigned short*)(weight_bytes + (long)rows * blocks_per_row * 16);
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            long idx = row_block0 + b;
            uint4 packed = quant_plane[idx];
            unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
            const float* x = staged_input + (long)b * 32;
            float block_dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned int byte = (words[j >> 2] >> ((j & 3) * 8)) & 0xffu;
                float product = (float)((int)(byte & 0x0fu) - 8) * x[j];
                block_dot += product;
            }
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned int byte = (words[j >> 2] >> ((j & 3) * 8)) & 0xffu;
                float product = (float)((int)(byte >> 4) - 8) * x[16 + j];
                block_dot += product;
            }
            myterms[b] = block_dot * f16_bits_to_f32(scale_plane[idx]);
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

// Pack four bytes from an arbitrary (and, for Q4_0's 18-byte stride, commonly
// unaligned) wire address without issuing an unaligned uint load. Keeping the
// packed words in registers lets the K-batched kernels decode one weight block
// once and reuse it for every speculative token.
__device__ __forceinline__ unsigned int q4_pack4_le(const unsigned char* p) {
    return (unsigned int)p[0]
        | ((unsigned int)p[1] << 8)
        | ((unsigned int)p[2] << 16)
        | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ int2 q4_1_dot32_dp4a_packed(
    uint4 packed, const signed char* __restrict__ y
) {
    unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    int isum = 0;
    int asum = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned int q = words[j];
        int lo = (int)(q & 0x0f0f0f0fu);
        int hi = (int)((q >> 4) & 0x0f0f0f0fu);
        int ylo = *((const int*)(y + j * 4));
        int yhi = *((const int*)(y + 16 + j * 4));
        isum = __dp4a(lo, ylo, isum);
        isum = __dp4a(hi, yhi, isum);
        asum = __dp4a(0x01010101, ylo, asum);
        asum = __dp4a(0x01010101, yhi, asum);
    }
    return make_int2(isum, asum);
}

// ---- Batched raw-wire Q4 GEMMs: K token-inputs against M weight rows --------
// These are the Gemma-4 QAT counterparts of q8_gemm_batched. One warp owns an
// output row. Each lane loads one 18/20-byte weight block exactly once, keeps its
// packed nibbles and scale(s) in registers, and contracts it with all K Q8_0
// activation rows. Lane 0 then replays the scalar GEMV's block-order f32 sum for
// every token, preserving the exact per-block association and greedy parity.
extern "C" __global__ void q4_0_gemm_batched(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
    extern __shared__ float terms[]; // [warp][token][block]
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * k_tokens * blocks_per_row;
    const int WIRE = 18;
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_block0 + b) * WIRE;
            float w_scale = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 2), q4_pack4_le(blk + 6),
                q4_pack4_le(blk + 10), q4_pack4_le(blk + 14));
            for (int t = 0; t < k_tokens; t++) {
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int isum = q4_0_dot32_dp4a_packed(packed, y);
                myterms[(long)t * blocks_per_row + b] =
                    (float)isum * w_scale * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++)
                acc += myterms[(long)t * blocks_per_row + b];
            output[(long)t * rows + row] = acc;
        }
    }
}

// Shared-scratch SoA twin of q4_0_gemm_batched. This deliberately preserves the
// raw-wire kernel's block-owner work assignment, [warp][token][block] term
// addresses, and lane-0 left-to-right f32 fold; only the two weight-plane reads
// change to match q4_0_wire_to_soa. It is the measured default for Gemma 4 MTP.
extern "C" __global__ void q4_0_gemm_batched_soa_shared(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
    extern __shared__ float terms[]; // [warp][token][block]
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * k_tokens * blocks_per_row;
    const uint4* quant_plane = (const uint4*)weight_bytes;
    const unsigned short* scale_plane =
        (const unsigned short*)(weight_bytes + (long)rows * blocks_per_row * 16);
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            long idx = row_block0 + b;
            uint4 packed = quant_plane[idx];
            float w_scale = f16_bits_to_f32(scale_plane[idx]);
            for (int t = 0; t < k_tokens; t++) {
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int isum = q4_0_dot32_dp4a_packed(packed, y);
                myterms[(long)t * blocks_per_row + b] =
                    (float)isum * w_scale * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++)
                acc += myterms[(long)t * blocks_per_row + b];
            output[(long)t * rows + row] = acc;
        }
    }
}

// Quants-first SoA twin of q4_0_gemm_batched. Gemma 4 uploads every resident
// common Q4_0 projection in this layout so scalar decode can use q4_0_gemv_soa:
//   [rows*blocks_per_row*16 nibble bytes][rows*blocks_per_row*2 f16 scales].
//
// Only the weight addresses differ from the scalar SoA kernel above. One lane
// owns each weight block and keeps that block in registers while contracting it
// with all K rows. Lanes 0..K-1 own the token accumulators; block-owner terms are
// shuffled to them in group-then-lane order, which is exactly increasing block
// order. This removes the former [warp][token][block] shared array without
// changing a multiply or add. The production verifier is eligible only for
// 1 <= K <= 14, leaving every token accumulator in a distinct warp lane.
// q4_0_gemm_batched_soa_variants_match_scalar_soa_bitwise pins that exactness at
// K=1, K=7, and K=14 with Gemma 4's production hidden=2816 geometry.
extern "C" __global__ void q4_0_gemm_batched_soa(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    const uint4* quant_plane = (const uint4*)weight_bytes;
    const unsigned short* scale_plane =
        (const unsigned short*)(weight_bytes + (long)rows * blocks_per_row * 16);

    // `token_acc` is meaningful only in token-owner lanes [0, k_tokens), but
    // keeping one scalar in every lane avoids a dynamically indexed local array.
    // Every warp executes every shuffle, including the final out-of-range row.
    float token_acc = 0.0f;
    long row_block0 = (long)row * blocks_per_row;
    for (int block0 = 0; block0 < blocks_per_row; block0 += 32) {
        int b = block0 + lane;
        int valid_lanes = blocks_per_row - block0;
        if (valid_lanes > 32) valid_lanes = 32;
        int valid_block = row < rows && lane < valid_lanes;
        uint4 packed = make_uint4(0u, 0u, 0u, 0u);
        float w_scale = 0.0f;
        if (valid_block) {
            long idx = row_block0 + b;
            packed = quant_plane[idx];
            w_scale = f16_bits_to_f32(scale_plane[idx]);
        }
        for (int t = 0; t < k_tokens; t++) {
            float term = 0.0f;
            if (valid_block) {
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int isum = q4_0_dot32_dp4a_packed(packed, y);
                term = (float)isum * w_scale
                    * input_scales[(long)t * blocks_per_row + b];
            }
            // The token owner performs `acc += term[b]` for b=block0.. in
            // strictly increasing order. Do not tree-reduce: exact verification
            // depends on matching q4_0_gemv_soa's left-to-right f32 fold.
            for (int owner = 0; owner < valid_lanes; owner++) {
                float ordered_term = __shfl_sync(0xffffffffu, term, owner);
                if (lane == t) token_acc += ordered_term;
            }
        }
    }
    if (row < rows && lane < k_tokens) {
        output[(long)lane * rows + row] = token_acc;
    }
}

// Ampere SM80+ signed-int8 tensor-core twin of q4_0_gemm_batched_soa_shared.
// One 256-thread CTA owns 128 output rows and the complete verifier width
// (1..=14 tokens). For every Q4_0 block it expands the 128x32 signed weights
// once, stages a shared 32x16 Q8 activation tile, and issues one m16n8k32 MMA
// per warp/token half. The integer result is exact; critically, every result is
// converted/scaled and accumulated immediately, one block at a time, with the
// exact scalar expression and increasing-block f32 fold used by q4_0_gemv_soa:
//
//     acc += ((float)isum * weight_scale) * activation_scale
//
// Do not accumulate multiple blocks in the s32 MMA accumulator: Q4_0 and Q8_0
// each have an independent scale at every block, and doing so would also move a
// scalar rounding point. --fmad=false keeps the final add separate. The Rust
// dispatch keeps this experimental kernel behind a strict SM86-only env gate.
extern "C" __global__ void q4_0_gemm_batched_soa_imma(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_base = blockIdx.x * 128;

    // A is row-major [128,32]. B is stored token-major [16,32], which is the
    // register mapping consumed as the column-major B operand below (the same
    // proven mapping as prism_q1_q8_wmma_gemm_batched).
    __shared__ __align__(32) signed char tile_a[128 * 32];
    __shared__ __align__(32) signed char tile_b[16 * 32];
    __shared__ float weight_scales[128];
    __shared__ float activation_scales[16];

    float sums0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float sums1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const unsigned char* quant_plane = weight_bytes;
    const unsigned short* scale_plane =
        (const unsigned short*)(weight_bytes + (long)rows * blocks_per_row * 16);

    for (int b = 0; b < blocks_per_row; b++) {
        // Sixteen threads decode the sixteen packed bytes of each row. Low
        // nibbles are columns 0..15 and high nibbles columns 16..31, exactly as
        // q4_0_dot32_dp4a_packed consumes them.
        for (int qi = threadIdx.x; qi < 128 * 16; qi += blockDim.x) {
            int ar = qi >> 4;
            int j = qi & 15;
            int row = row_base + ar;
            signed char low = 0;
            signed char high = 0;
            if (row < rows) {
                long idx = (long)row * blocks_per_row + b;
                unsigned char packed = quant_plane[idx * 16 + j];
                low = (signed char)((int)(packed & 0x0fu) - 8);
                high = (signed char)((int)(packed >> 4) - 8);
            }
            tile_a[ar * 32 + j] = low;
            tile_a[ar * 32 + 16 + j] = high;
        }
        if (threadIdx.x < 128) {
            int row = row_base + threadIdx.x;
            weight_scales[threadIdx.x] = row < rows
                ? f16_bits_to_f32(scale_plane[(long)row * blocks_per_row + b])
                : 0.0f;
        }
        for (int bi = threadIdx.x; bi < 16 * 32; bi += blockDim.x) {
            int token = bi >> 5;
            int bk = bi & 31;
            tile_b[bi] = token < k_tokens
                ? input_quants[((long)token * blocks_per_row + b) * 32 + bk]
                : 0;
        }
        if (threadIdx.x < 16) {
            int token = threadIdx.x;
            activation_scales[token] = token < k_tokens
                ? input_scales[(long)token * blocks_per_row + b]
                : 0.0f;
        }
        __syncthreads();

        int a0, a1, a2, a3;
        const int* a_row = (const int*)(tile_a + warp * 16 * 32);
        const int* a_src = a_row + (lane % 16) * 8 + (lane / 16) * 4;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_src));

        const int* b_row = (const int*)tile_b;
        #pragma unroll
        for (int token_half = 0; token_half < 2; token_half++) {
            // K<=8 has no active columns in the second N=8 fragment. This is a
            // warp-uniform condition, so skipping the inactive MMA cannot
            // disturb fragment participation or the per-output block fold.
            if (token_half * 8 >= k_tokens) continue;
            const int* b_half = b_row + token_half * 8 * 8;
            int b0 = b_half[(lane / 4) * 8 + (lane % 4)];
            int b1 = b_half[(lane / 4) * 8 + (lane % 4) + 4];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1));
            int cv[4] = { c0, c1, c2, c3 };
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int cr = (l / 2) * 8 + lane / 4;
                int token = token_half * 8 + (lane % 4) * 2 + (l % 2);
                int row_in_tile = warp * 16 + cr;
                int row = row_base + row_in_tile;
                if (row < rows && token < k_tokens) {
                    float term = (float)cv[l] * weight_scales[row_in_tile];
                    term = term * activation_scales[token];
                    if (token_half == 0) sums0[l] += term;
                    else sums1[l] += term;
                }
            }
        }
        // All warps consume the CTA-wide A/B/scales before the next block
        // overwrites them. This also orders each thread's scalar f32 fold.
        __syncthreads();
    }

    #pragma unroll
    for (int token_half = 0; token_half < 2; token_half++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int cr = (l / 2) * 8 + lane / 4;
            int token = token_half * 8 + (lane % 4) * 2 + (l % 2);
            int row = row_base + warp * 16 + cr;
            if (row < rows && token < k_tokens) {
                float value = token_half == 0 ? sums0[l] : sums1[l];
                output[(long)token * rows + row] = value;
            }
        }
    }
#endif
}

// ---- Routed Q4_0 GEMM: one expert's weights against ITS tokens -------------
// This is the prefill counterpart of `q4_0_gemv_routed`, and it exists because the
// GEMV form re-reads a weight for every token that routes to it. On a 362-token
// prompt each expert is wanted by ~22.6 tokens, so the per-layer weight traffic is
// 362 x 8 x 3.19 MiB = 9.02 GiB instead of the 408 MiB the layer actually contains
// — 271 GiB across a whole prefill against 12 GiB here. The FLOPs are identical;
// only the redundancy goes away, which turns a memory-bound GEMV into a
// compute-bound GEMM.
//
// Routing is RAGGED — expert e is wanted by a variable subset of the chunk's
// tokens — so the assignment is passed CSR-style: `token_offsets[e]..token_offsets[e+1]`
// indexes into `token_ids`. Output is written per ASSIGNMENT, not per token, so the
// caller's existing route-order weighted sum keeps its exact accumulation order.
//
// BIT-IDENTICAL to `q4_0_gemv_routed` for every (expert, token) pair it covers: the
// same 16 weight bytes meet the same activation row through the same
// `q4_0_dot32_dp4a_packed`, the per-block float term is the same product in the same
// order, and each row is still folded by ONE lane over increasing b. Only the loop
// nesting changes — the weight is hoisted into registers and reused across tokens
// instead of being re-fetched. Pinned by `q4_0_gemm_routed_matches_gemv`.
//
// Shared scratch is [warp][tile][blocks_per_row] floats, so the caller tiles the
// assignment list: at blocks_per_row 88 and 8 warps that is 2.75 KiB per tile row.
extern "C" __global__ void q4_0_gemm_routed(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ token_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int tile
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int first = token_offsets[expert];
    int count = token_offsets[expert + 1] - first;
    int tile_base = blockIdx.z * tile;
    if (tile_base >= count) return;
    int n_tok = count - tile_base;
    if (n_tok > tile) n_tok = tile;

    const unsigned char* expert_weights = weight_arena + (long)slot_ids[expert] * weight_stride;

    extern __shared__ float smem40gr[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = smem40gr + (long)warp * tile * blocks_per_row;
    const int WIRE = 18;

    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            // Load this block's weight ONCE and hold it in registers across all
            // n_tok activation rows. This reuse is the entire point of the kernel.
            const unsigned char* blk = expert_weights + (row_block0 + b) * WIRE;
            float w_scale = f16_bits_to_f32(
                (unsigned short)(blk[0] | (blk[1] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 2), q4_pack4_le(blk + 6),
                q4_pack4_le(blk + 10), q4_pack4_le(blk + 14));
            for (int j = 0; j < n_tok; j++) {
                int t = token_ids[first + tile_base + j];
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int isum = q4_0_dot32_dp4a_packed(packed, y);
                myterms[(long)j * blocks_per_row + b] =
                    (float)isum * w_scale * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int j = 0; j < n_tok; j++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++)
                acc += myterms[(long)j * blocks_per_row + b];
            // Per-assignment output: (first + tile_base + j) is this expert's slot in
            // the caller's flat route list, so route order is preserved downstream.
            output[(long)(first + tile_base + j) * rows + row] = acc;
        }
    }
}

// Low-shared opt-in twin of q4_0_gemm_routed. The 32 block-owner lanes still
// load one weight block and reuse it across every token assigned to the expert.
// Terms are staged only for one 32-block chunk at a time; lanes 0..n_tok-1 each
// carry one token's accumulator across chunks and fold that token's terms in
// strictly increasing block order. This preserves the exact scalar association
// without the shuffle traffic of a zero-shared owner broadcast.
#define G4_ROUTED_GEMM_CHUNK 32
extern "C" __global__ void q4_0_gemm_routed_chunked(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ token_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int tile
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int first = token_offsets[expert];
    int count = token_offsets[expert + 1] - first;
    int tile_base = blockIdx.z * tile;
    if (tile_base >= count) return;
    int n_tok = count - tile_base;
    if (n_tok > tile) n_tok = tile;

    const unsigned char* expert_weights = weight_arena + (long)slot_ids[expert] * weight_stride;
    extern __shared__ float smem40gc[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int chunk_stride = blocks_per_row < G4_ROUTED_GEMM_CHUNK
        ? blocks_per_row : G4_ROUTED_GEMM_CHUNK;
    float* myterms = smem40gc + (long)warp * tile * chunk_stride;
    const int WIRE = 18;
    float token_acc = 0.0f;

    for (int block0 = 0; block0 < blocks_per_row; block0 += G4_ROUTED_GEMM_CHUNK) {
        int chunk_blocks = blocks_per_row - block0;
        if (chunk_blocks > G4_ROUTED_GEMM_CHUNK) chunk_blocks = G4_ROUTED_GEMM_CHUNK;
        int b = block0 + lane;
        if (row < rows && lane < chunk_blocks) {
            long idx = (long)row * blocks_per_row + b;
            const unsigned char* blk = expert_weights + idx * WIRE;
            float w_scale = f16_bits_to_f32(
                (unsigned short)(blk[0] | (blk[1] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 2), q4_pack4_le(blk + 6),
                q4_pack4_le(blk + 10), q4_pack4_le(blk + 14));
            for (int j = 0; j < n_tok; j++) {
                int t = token_ids[first + tile_base + j];
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int isum = q4_0_dot32_dp4a_packed(packed, y);
                myterms[(long)j * chunk_stride + lane] =
                    (float)isum * w_scale * input_scales[(long)t * blocks_per_row + b];
            }
        }
        __syncwarp();
        if (row < rows && lane < n_tok) {
            const float* token_terms = myterms + (long)lane * chunk_stride;
            for (int owner = 0; owner < chunk_blocks; owner++)
                token_acc += token_terms[owner];
        }
        // Token owners must finish reading this chunk before block owners reuse
        // the same bounded scratch for the next 32 weight blocks.
        __syncwarp();
    }
    if (row < rows && lane < n_tok) {
        output[(long)(first + tile_base + lane) * rows + row] = token_acc;
    }
}

// ---- Routed Q4_1 GEMM: the mixed-format half of the same lever -------------
// The 26B-A4B `.cghost` is MIXED — `down_exps` is Q4_1 in layers 0..=6 and Q4_0 in
// 7..=29 — so batching only the Q4_0 form would leave a seventh of the stack on the
// redundant GEMV path. A speculative verify pays that on every round, which is why
// this twin exists rather than a fallback.
//
// Identical to `q4_0_gemm_routed` in every structural respect: same CSR assignment
// (`token_offsets[e]..token_offsets[e+1]` indexes `token_ids`), same weight-in-registers
// reuse across an expert's tokens, same per-ASSIGNMENT output so the caller's route-order
// weighted sum keeps its accumulation order, same one-lane fold over increasing b. Only
// the block decode differs: 20 wire bytes, two f16 scales, and the
// `(w_d*isum + w_m*asum) * input_scale` term.
//
// BIT-IDENTICAL to `q4_1_gemv_routed`. That kernel spells the block out as a scalar
// 16-iteration nibble loop; `q4_1_dot32_dp4a_packed` consumes the same bytes with the same
// low/high nibble split and the same activation pairing (y[j] with lo, y[16+j] with hi),
// and integer sums are exact regardless of grouping, so `isum` and `asum` are the same
// integers. The float term is then the same expression on the same values. Pinned by
// `q4_1_gemm_routed_matches_gemv`.
extern "C" __global__ void q4_1_gemm_routed(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ token_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int tile
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int first = token_offsets[expert];
    int count = token_offsets[expert + 1] - first;
    int tile_base = blockIdx.z * tile;
    if (tile_base >= count) return;
    int n_tok = count - tile_base;
    if (n_tok > tile) n_tok = tile;

    const unsigned char* expert_weights = weight_arena + (long)slot_ids[expert] * weight_stride;

    extern __shared__ float smem41gr[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = smem41gr + (long)warp * tile * blocks_per_row;
    const int WIRE = 20;

    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = expert_weights + (row_block0 + b) * WIRE;
            float w_d = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            float w_m = f16_bits_to_f32((unsigned short)(blk[2] | (blk[3] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 4), q4_pack4_le(blk + 8),
                q4_pack4_le(blk + 12), q4_pack4_le(blk + 16));
            for (int j = 0; j < n_tok; j++) {
                int t = token_ids[first + tile_base + j];
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int2 sums = q4_1_dot32_dp4a_packed(packed, y);
                myterms[(long)j * blocks_per_row + b] =
                    (w_d * (float)sums.x + w_m * (float)sums.y)
                    * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int j = 0; j < n_tok; j++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++)
                acc += myterms[(long)j * blocks_per_row + b];
            output[(long)(first + tile_base + j) * rows + row] = acc;
        }
    }
}

// Q4_1 twin of q4_0_gemm_routed_chunked. Only the packed block decode and
// two-scale term differ; scratch lifetime and the per-token ordered fold match.
extern "C" __global__ void q4_1_gemm_routed_chunked(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ token_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int tile
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int first = token_offsets[expert];
    int count = token_offsets[expert + 1] - first;
    int tile_base = blockIdx.z * tile;
    if (tile_base >= count) return;
    int n_tok = count - tile_base;
    if (n_tok > tile) n_tok = tile;

    const unsigned char* expert_weights = weight_arena + (long)slot_ids[expert] * weight_stride;
    extern __shared__ float smem41gc[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int chunk_stride = blocks_per_row < G4_ROUTED_GEMM_CHUNK
        ? blocks_per_row : G4_ROUTED_GEMM_CHUNK;
    float* myterms = smem41gc + (long)warp * tile * chunk_stride;
    const int WIRE = 20;
    float token_acc = 0.0f;

    for (int block0 = 0; block0 < blocks_per_row; block0 += G4_ROUTED_GEMM_CHUNK) {
        int chunk_blocks = blocks_per_row - block0;
        if (chunk_blocks > G4_ROUTED_GEMM_CHUNK) chunk_blocks = G4_ROUTED_GEMM_CHUNK;
        int b = block0 + lane;
        if (row < rows && lane < chunk_blocks) {
            long idx = (long)row * blocks_per_row + b;
            const unsigned char* blk = expert_weights + idx * WIRE;
            float w_d = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            float w_m = f16_bits_to_f32((unsigned short)(blk[2] | (blk[3] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 4), q4_pack4_le(blk + 8),
                q4_pack4_le(blk + 12), q4_pack4_le(blk + 16));
            for (int j = 0; j < n_tok; j++) {
                int t = token_ids[first + tile_base + j];
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int2 sums = q4_1_dot32_dp4a_packed(packed, y);
                myterms[(long)j * chunk_stride + lane] =
                    (w_d * (float)sums.x + w_m * (float)sums.y)
                    * input_scales[(long)t * blocks_per_row + b];
            }
        }
        __syncwarp();
        if (row < rows && lane < n_tok) {
            const float* token_terms = myterms + (long)lane * chunk_stride;
            for (int owner = 0; owner < chunk_blocks; owner++)
                token_acc += token_terms[owner];
        }
        __syncwarp();
    }
    if (row < rows && lane < n_tok) {
        output[(long)(first + tile_base + lane) * rows + row] = token_acc;
    }
}

extern "C" __global__ void q4_1_gemm_batched(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    int k_tokens, float* __restrict__ output
) {
    extern __shared__ float terms[]; // [warp][token][block]
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * k_tokens * blocks_per_row;
    const int WIRE = 20;
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = weight_bytes + (long)(row_block0 + b) * WIRE;
            float w_d = f16_bits_to_f32((unsigned short)(blk[0] | (blk[1] << 8)));
            float w_m = f16_bits_to_f32((unsigned short)(blk[2] | (blk[3] << 8)));
            uint4 packed = make_uint4(
                q4_pack4_le(blk + 4), q4_pack4_le(blk + 8),
                q4_pack4_le(blk + 12), q4_pack4_le(blk + 16));
            for (int t = 0; t < k_tokens; t++) {
                const signed char* y = input_quants
                    + ((long)t * blocks_per_row + b) * 32;
                int2 sums = q4_1_dot32_dp4a_packed(packed, y);
                myterms[(long)t * blocks_per_row + b] =
                    (w_d * (float)sums.x + w_m * (float)sums.y)
                    * input_scales[(long)t * blocks_per_row + b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        for (int t = 0; t < k_tokens; t++) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; b++)
                acc += myterms[(long)t * blocks_per_row + b];
            output[(long)t * rows + row] = acc;
        }
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
    const unsigned char* __restrict__ weight_bytes, // SWIZZLED 144-byte Q4_K blocks, row-major
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
    const unsigned char* __restrict__ weight_bytes, // 224-byte PADDED Q6_K blocks, row-major
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

// Host supplies Gemma's exact sequential Rust RMS inverse; CUDA applies the
// weighted norm to the resident row and feeds the existing parity-locked Q8_K
// reducer. `quant_q8k_block_parallel` retains the serial reference's first
// absolute-maximum signed value and `nearest_int_ref` byte quantization.
extern "C" __global__ void rms_inv_norm_quantize_q8k(
    const float* __restrict__ x, const float* __restrict__ weight,
    signed char* __restrict__ quants, float* __restrict__ scales,
    int n, float rms_inv
) {
    __shared__ float r_amax[256];
    __shared__ float r_maxv[256];
    __shared__ int r_idx[256];
    int b = blockIdx.x;
    int n_sb = n >> 8;
    if (b >= n_sb) return;
    int tid = threadIdx.x;
    long i = ((long)b << 8) + tid;
    float value = x[i] * rms_inv * weight[i];
    quant_q8k_block_parallel(value, tid, r_amax, r_maxv, r_idx,
                             quants + ((long)b << 8), scales + b);
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

// Exact anchor-major DP4A twin of q6k_gemm_batched. The established Q6_K
// oracle owns eight integer position lanes per super-block, scales each lane
// into its own ordered f32 accumulator, then left-folds those eight f32 lanes.
// This kernel preserves that shape while removing the former
// [warp][token][super-block][8] shared scratch and lane-0 replay:
//
//   * lane = anchor + 8*quarter, where anchor is the oracle lane (0..7);
//   * each quarter owns four of the sixteen Q6_K scale groups;
//   * the two values belonging to one (scale-group, anchor) are contracted by
//     one signed DP4A with two zero bytes;
//   * lanes anchor+{0,8,16,24} reduce in exact integer arithmetic;
//   * lanes 0..7 retain the eight ordered f32 accumulators across super-blocks;
//   * lane 0 left-folds anchors 0..7 exactly like q6_k_wire_row_dot.
//
// Weight bytes remain in the existing 224-byte padded Q6_K upload layout. The
// only extra shared memory is the K-row Q8_K activation tile, so verifier-width
// K=14 at hidden=2816 fits with eight warps per CTA.
extern "C" __global__ void q6k_gemm_batched_anchor_dp4a(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes,
    int rows, int n_sb, int k_tokens, float* __restrict__ output
) {
    extern __shared__ unsigned char smem6a[];
    signed char* s_iq = (signed char*)smem6a;
    float* s_is = (float*)(smem6a + (long)k_tokens * n_sb * 256);
    int tid = threadIdx.x;
    for (int i = tid; i < k_tokens * n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
    for (int i = tid; i < k_tokens * n_sb; i += blockDim.x)
        s_is[i] = input_scales[i];
    __syncthreads();

    const int WIRE = 224;
    const int MAX_TOKENS = 14;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int anchor = lane & 7;
    int quarter = lane >> 3;
    float sums[MAX_TOKENS];
    #pragma unroll
    for (int t = 0; t < MAX_TOKENS; t++) sums[t] = 0.0f;

    for (int sb = 0; sb < n_sb; sb++) {
        const unsigned char* block = weight_bytes;
        if (row < rows)
            block += ((long)row * n_sb + sb) * WIRE;

        int packed_weights[4];
        int group_scales[4];
        int group_offsets[4];
        int half = quarter >> 1;
        int segment = quarter & 1;
        int ql_base = half * 64 + segment * 16 + anchor;
        int qh_base = 128 + half * 32 + segment * 16 + anchor;
        int qlo0 = 0, qlo1 = 0, qhi0 = 0, qhi1 = 0, qh0 = 0, qh1 = 0;
        if (row < rows) {
            qlo0 = (int)block[ql_base];
            qlo1 = (int)block[ql_base + 8];
            qhi0 = (int)block[ql_base + 32];
            qhi1 = (int)block[ql_base + 40];
            qh0 = (int)block[qh_base];
            qh1 = (int)block[qh_base + 8];
        }
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int group = half * 8 + segment + 2 * g;
            int ql0 = (g & 1) ? qhi0 : qlo0;
            int ql1 = (g & 1) ? qhi1 : qlo1;
            int nibble_shift = (g >> 1) * 4;
            int high_shift = g * 2;
            int w0 = 0;
            int w1 = 0;
            int scale = 0;
            if (row < rows) {
                int low0 = (ql0 >> nibble_shift) & 0xf;
                int low1 = (ql1 >> nibble_shift) & 0xf;
                int high0 = (qh0 >> high_shift) & 3;
                int high1 = (qh1 >> high_shift) & 3;
                w0 = (low0 | (high0 << 4)) - 32;
                w1 = (low1 | (high1 << 4)) - 32;
                scale = (int)((const signed char*)block)[192 + group];
            }
            packed_weights[g] = (int)((unsigned int)(unsigned char)w0
                | ((unsigned int)(unsigned char)w1 << 8));
            group_scales[g] = scale;
            group_offsets[g] = group * 16 + anchor;
        }

        float weight_scale = 0.0f;
        if (row < rows) {
            unsigned short d_bits = (unsigned short)block[208]
                | ((unsigned short)block[209] << 8);
            weight_scale = f16_bits_to_f32(d_bits);
        }

        #pragma unroll
        for (int t = 0; t < MAX_TOKENS; t++) {
            if (t < k_tokens) {
                const signed char* y256 =
                    s_iq + ((long)t * n_sb + sb) * 256;
                int partial = 0;
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int off = group_offsets[g];
                    int packed_input = (int)((unsigned int)(unsigned char)y256[off]
                        | ((unsigned int)(unsigned char)y256[off + 8] << 8));
                    int pair_dot = __dp4a(packed_weights[g], packed_input, 0);
                    partial += group_scales[g] * pair_dot;
                }
                partial += __shfl_down_sync(0xffffffffu, partial, 16);
                partial += __shfl_down_sync(0xffffffffu, partial, 8);
                if (lane < 8) {
                    float d = weight_scale * s_is[(long)t * n_sb + sb];
                    sums[t] += d * (float)partial;
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MAX_TOKENS; t++) {
        if (t < k_tokens) {
            float acc = 0.0f;
            #pragma unroll
            for (int owner = 0; owner < 8; owner++) {
                float value = __shfl_sync(0xffffffffu, sums[t], owner);
                if (lane == 0) acc += value;
            }
            if (row < rows && lane == 0)
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
// cache layout [kv_head][slot][head_dim], slot = position % max_pos: `max_pos` is the
// cache's per-head position CAPACITY and positions wrap onto it as a ring. Full-context
// callers never wrap (position < capacity, so slot == position); the gemma4 lane sizes
// sliding-layer caches at window plus verifier slack and relies on the wrap -- the
// matching ring read is in attention_decode_sw.
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
    cache[((long)kv_head * max_pos + (position % max_pos)) * head_dim + d] =
        f32_to_f16_bits(src[(long)kv_head * head_dim + d]);
}

// ---- Attention decode: per query head, GQA, scale, softmax, weighted V -----
// One block per query head. cache_k/v layout [kv_head][position][head_dim].
extern "C" __global__ void attention_decode(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale
, float* __restrict__ global_scores) {
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
    float* scores = global_scores + (long)head * max_pos;
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
//
// K/V slots ring on `max_pos` (slot = p % max_pos, matching kv_scatter's write):
// a sliding-layer cache holds only the active window plus bounded verifier slack, so every p in
// [start, position_count) still maps to a distinct live slot. Full-capacity
// callers (gemma3's sliding layers, and every window <= 0 caller — full-causal
// needs all positions) never wrap. scores[] is indexed relative to `start`, so
// each head needs only `max_pos` entries even after the ring wraps.
extern "C" __global__ void attention_decode_sw(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale, int window
, float* __restrict__ global_scores) {
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
    float* scores = global_scores + (long)head * max_pos;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;  // uint4 K-read when row is 16B-aligned
    for (int p = start + tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)(p % max_pos) * head_dim;
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
        scores[p - start] = dot * scale;
    }
    __syncthreads();

    __shared__ float s_max_sw, s_sum_sw;
    if (tid == 0) {
        float m = scores[0];
        for (int p = start + 1; p < position_count; p++) if (scores[p - start] > m) m = scores[p - start];
        s_max_sw = m;
    }
    __syncthreads();
    for (int p = start + tid; p < position_count; p += blockDim.x)
        scores[p - start] = expf(scores[p - start] - s_max_sw);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = start; p < position_count; p++) sum += scores[p - start];
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
        acc += (scores[p - start] * inv) * f16_bits_to_f32(vbase[(long)(p % max_pos) * head_dim + did]);
    vpart[(long)did * G + gid] = acc;
    __syncthreads();
    if (gid == 0) {
        float sum = 0.0f;
        for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
        out[(long)head * head_dim + did] = sum;
    }
}

// ---- Quantized Q8_0 KV-cache block structure (34 bytes per 32 elements) ----
struct __align__(2) block_q8_0 {
    unsigned short scale; // f16 scale
    signed char qs[32];   // 32 int8 quantized values
};

// ---- KV scatter (Q8_0): quantize and store current position's K (or V) -----
extern "C" __global__ void kv_scatter_q8_0(
    const float* __restrict__ src, block_q8_0* __restrict__ cache,
    const int* __restrict__ position_ptr, int n_kv_heads, int head_dim, int max_pos
) {
    int position = position_ptr[0];
    int slot = position % max_pos;
    int blocks_per_head = head_dim / 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_kv_heads * blocks_per_head) return;
    int kv_head = idx / blocks_per_head;
    int b = idx % blocks_per_head;

    const float* chunk = src + ((long)kv_head * head_dim + b * 32);
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = fabsf(chunk[i]);
        if (v > amax) amax = v;
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    block_q8_0 blk;
    blk.scale = f32_to_f16_bits(d);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float val = roundf(chunk[i] * id);
        if (val < -127.0f) val = -127.0f;
        if (val > 127.0f) val = 127.0f;
        blk.qs[i] = (signed char)val;
    }
    cache[((long)kv_head * max_pos + slot) * blocks_per_head + b] = blk;
}

// ---- Attention decode (Q8_0): per query head, GQA, scale, softmax, weighted V -----
extern "C" __global__ void attention_decode_q8_0(
    const float* __restrict__ q, const block_q8_0* __restrict__ cache_k,
    const block_q8_0* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale
, float* __restrict__ global_scores) {
    int position_count = position_ptr[0] + 1;
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    int blocks_per_head = head_dim / 32;
    const block_q8_0* kbase = cache_k + (long)kv_head * max_pos * blocks_per_head;
    const block_q8_0* vbase = cache_v + (long)kv_head * max_pos * blocks_per_head;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int G = blockDim.x / head_dim;
    float* qsh = shared;
    float* vpart = shared + head_dim;
    float* scores = global_scores + (long)head * max_pos;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    for (int p = tid; p < position_count; p += blockDim.x) {
        const block_q8_0* kp = kbase + (long)p * blocks_per_head;
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            float d = f16_bits_to_f32(kp[b].scale);
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum += qsh[b * 32 + i] * (float)kp[b].qs[i];
            }
            dot += sum * d;
        }
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

    int gid = tid / head_dim;
    int did = tid % head_dim;
    int b = did / 32;
    int bi = did % 32;
    int p_lo = (int)((long)gid * position_count / G);
    int p_hi = (int)((long)(gid + 1) * position_count / G);
    float acc = 0.0f;
    for (int p = p_lo; p < p_hi; p++) {
        const block_q8_0* vp = vbase + ((long)p * blocks_per_head + b);
        float d = f16_bits_to_f32(vp->scale);
        acc += (scores[p] * inv) * (d * (float)vp->qs[bi]);
    }
    vpart[(long)did * G + gid] = acc;
    __syncthreads();
    if (gid == 0) {
        float sum = 0.0f;
        for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
        out[(long)head * head_dim + did] = sum;
    }
}

// ---- Sliding-window attention decode (Q8_0) ---------------------------------
extern "C" __global__ void attention_decode_sw_q8_0(
    const float* __restrict__ q, const block_q8_0* __restrict__ cache_k,
    const block_q8_0* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale, int window
, float* __restrict__ global_scores) {
    int position_count = position_ptr[0] + 1;
    int start = (window > 0 && position_count > window) ? (position_count - window) : 0;
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    int blocks_per_head = head_dim / 32;
    const block_q8_0* kbase = cache_k + (long)kv_head * max_pos * blocks_per_head;
    const block_q8_0* vbase = cache_v + (long)kv_head * max_pos * blocks_per_head;

    extern __shared__ float shared_sw[];
    int tid = threadIdx.x;
    int G = blockDim.x / head_dim;
    float* qsh = shared_sw;
    float* vpart = shared_sw + head_dim;
    float* scores = global_scores + (long)head * max_pos;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    for (int p = start + tid; p < position_count; p += blockDim.x) {
        const block_q8_0* kp = kbase + (long)(p % max_pos) * blocks_per_head;
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            float d = f16_bits_to_f32(kp[b].scale);
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum += qsh[b * 32 + i] * (float)kp[b].qs[i];
            }
            dot += sum * d;
        }
        scores[p - start] = dot * scale;
    }
    __syncthreads();

    __shared__ float s_max_sw, s_sum_sw;
    if (tid == 0) {
        float m = scores[0];
        for (int p = start + 1; p < position_count; p++) if (scores[p - start] > m) m = scores[p - start];
        s_max_sw = m;
    }
    __syncthreads();
    for (int p = start + tid; p < position_count; p += blockDim.x)
        scores[p - start] = expf(scores[p - start] - s_max_sw);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = start; p < position_count; p++) sum += scores[p - start];
        s_sum_sw = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum_sw;
    int gid = tid / head_dim;
    int did = tid % head_dim;
    int b = did / 32;
    int bi = did % 32;
    int active = position_count - start;
    int p_lo = start + (int)((long)gid * active / G);
    int p_hi = start + (int)((long)(gid + 1) * active / G);
    float acc = 0.0f;
    for (int p = p_lo; p < p_hi; p++) {
        const block_q8_0* vp = vbase + ((long)(p % max_pos) * blocks_per_head + b);
        float d = f16_bits_to_f32(vp->scale);
        acc += (scores[p - start] * inv) * (d * (float)vp->qs[bi]);
    }
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

// ---- Routed batched Q4_0 GEMV -------------------------------------------
// Gemma 4 selects several experts that share one input activation. Launching
// q4_0_gemv once per expert makes batch-1 decode submission-bound on WDDM. This
// 2-D variant keeps the exact warp-per-row arithmetic above, but grid.y selects
// an expert arena slot and writes the result into its original router position.
// `batched_input=0` shares one activation (gate/up); `1` selects the activation
// at route*blocks_per_row (down). The ordered per-row sum is unchanged.
extern "C" __global__ void q4_0_gemv_routed(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ route_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int batched_input
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int route = route_ids[expert];
    int slot = slot_ids[expert];
    const float* expert_scales = input_scales
        + (batched_input ? (long)route * blocks_per_row : 0);
    const signed char* expert_quants = input_quants
        + (batched_input ? (long)route * blocks_per_row * 32 : 0);
    const unsigned char* expert_weights = weight_arena + (long)slot * weight_stride;

    extern __shared__ unsigned char smem40r[];
    signed char* s_iq = (signed char*)smem40r;
    float* s_is = (float*)(smem40r + (long)blocks_per_row * 32);
    float* terms = (float*)(smem40r + (long)blocks_per_row * 36);
    int tid = threadIdx.x;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)expert_quants)[i];
    for (int i = tid; i < blocks_per_row; i += blockDim.x)
        s_is[i] = expert_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    if (row < rows) {
        long row_block0 = (long)row * blocks_per_row;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = expert_weights + (row_block0 + b) * 18;
            float w_scale = f16_bits_to_f32(
                (unsigned short)(blk[0] | (blk[1] << 8)));
            const signed char* y = s_iq + (long)b * 32;
            int isum = q4_0_dot32_dp4a(blk + 2, y);
            myterms[b] = (float)isum * w_scale * s_is[b];
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) acc += myterms[b];
        output[(long)route * rows + row] = acc;
    }
}

// ---- Routed Q4_0 GEMV, R output rows per warp ------------------------------
// Same arithmetic as `q4_0_gemv_routed`; the only change is WHO does the tail fold.
//
// The one-row-per-warp shape ends with `if (lane == 0) for (b...) acc += myterms[b]`.
// For the routed gate_up geometry that is 88 serially dependent f32 adds -- roughly
// 350 cycles of pure dependent-FADD latency -- issued by a single lane while the other
// 31 sit predicated off, against about 24 dp4a instructions of actual work. The fold,
// not the dot product, is the kernel.
//
// Here each warp owns G4_ROUTED_ROWS consecutive output rows. Phase 1 computes every
// (row, block) term exactly as before, so the weight reads keep their coalescing: at a
// given row the 32 lanes still walk 32 consecutive blocks. Phase 2 hands row r to lane
// r, and each lane sums ITS OWN row across b = 0,1,2,... in the same increasing-block
// order the single lane used. Every row's f32 association is therefore byte-identical
// and the greedy token stream cannot move -- this is the parity-SAFE alternative to a
// warp-shuffle tree, which commit a03fca63 tried on the sibling q8_gemv for +4%, found
// it flipped a greedy near-tie, and reverted.
//
// Costs R x more shared scratch per warp and divides the grid by R, so R trades fold
// parallelism against occupancy. R=8 keeps 176 blocks for the 1408-row gate_up shape,
// still comfortably above this box's 30 SMs. `q4_0_gemv_routed_rows_matches_scalar`
// pins the bitwise equality.
// Blocks processed per pass. 32 keeps every lane busy in the term phase (the inner
// stride is 32) while bounding the scratch; the launcher must size shared memory with
// the same constant.
#define G4_ROUTED_CHUNK 32
extern "C" __global__ void q4_0_gemv_routed_rows(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ route_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int batched_input,
    int rows_per_warp
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int route = route_ids[expert];
    int slot = slot_ids[expert];
    const float* expert_scales = input_scales
        + (batched_input ? (long)route * blocks_per_row : 0);
    const signed char* expert_quants = input_quants
        + (batched_input ? (long)route * blocks_per_row * 32 : 0);
    const unsigned char* expert_weights = weight_arena + (long)slot * weight_stride;

    extern __shared__ unsigned char smem40rr[];
    signed char* s_iq = (signed char*)smem40rr;
    float* s_is = (float*)(smem40rr + (long)blocks_per_row * 32);
    float* terms = (float*)(smem40rr + (long)blocks_per_row * 36);
    int tid = threadIdx.x;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)expert_quants)[i];
    for (int i = tid; i < blocks_per_row; i += blockDim.x)
        s_is[i] = expert_scales[i];
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row0 = (blockIdx.x * warps_per_block + warp) * rows_per_warp;
    // Scratch is R x CHUNK per warp, NOT R x blocks_per_row: holding every row's full
    // term vector would take 8 x 88 x 4 = 22.5 KB per block on the gate_up shape and
    // halve occupancy, which measurably ate most of the fold win. Walking the blocks in
    // CHUNK-sized passes keeps the scratch at 8 x 32 x 4 = 1 KB per warp and full
    // occupancy, and each lane simply carries its running accumulator across passes --
    // so the per-row summation is still strictly increasing in b, and still done by one
    // lane. Bit-exactness is unchanged; only the residency is better.
    float* myterms = terms + (long)warp * rows_per_warp * G4_ROUTED_CHUNK;

    int myrow = row0 + lane;
    bool folder = (lane < rows_per_warp) && (myrow < rows);
    float acc = 0.0f;

    for (int c0 = 0; c0 < blocks_per_row; c0 += G4_ROUTED_CHUNK) {
        int cn = blocks_per_row - c0;
        if (cn > G4_ROUTED_CHUNK) cn = G4_ROUTED_CHUNK;
        for (int r = 0; r < rows_per_warp; r++) {
            int row = row0 + r;
            if (row >= rows) break;
            long row_block0 = (long)row * blocks_per_row;
            float* rowterms = myterms + (long)r * G4_ROUTED_CHUNK;
            for (int b = lane; b < cn; b += 32) {
                const unsigned char* blk = expert_weights + (row_block0 + c0 + b) * 18;
                float w_scale = f16_bits_to_f32(
                    (unsigned short)(blk[0] | (blk[1] << 8)));
                const signed char* y = s_iq + (long)(c0 + b) * 32;
                int isum = q4_0_dot32_dp4a(blk + 2, y);
                rowterms[b] = (float)isum * w_scale * s_is[c0 + b];
            }
        }
        __syncwarp();
        if (folder) {
            const float* rowterms = myterms + (long)lane * G4_ROUTED_CHUNK;
            for (int b = 0; b < cn; b++) acc += rowterms[b];
        }
        __syncwarp();
    }
    if (folder) output[(long)route * rows + myrow] = acc;
}

// ---- Routed batched Q4_1 GEMV -------------------------------------------
extern "C" __global__ void q4_1_gemv_routed(
    const float* __restrict__ input_scales,
    const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ route_ids,
    unsigned long long weight_stride, int rows, int blocks_per_row,
    float* __restrict__ output, int expert_count, int batched_input
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int route = route_ids[expert];
    int slot = slot_ids[expert];
    const float* expert_scales = input_scales
        + (batched_input ? (long)route * blocks_per_row : 0);
    const signed char* expert_quants = input_quants
        + (batched_input ? (long)route * blocks_per_row * 32 : 0);
    const unsigned char* expert_weights = weight_arena + (long)slot * weight_stride;

    extern __shared__ unsigned char smem41r[];
    signed char* s_iq = (signed char*)smem41r;
    float* s_is = (float*)(smem41r + (long)blocks_per_row * 32);
    float* terms = (float*)(smem41r + (long)blocks_per_row * 36);
    int tid = threadIdx.x;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)expert_quants)[i];
    for (int i = tid; i < blocks_per_row; i += blockDim.x)
        s_is[i] = expert_scales[i];
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
            const unsigned char* blk = expert_weights + (long)(row_block0 + b) * WIRE;
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
        output[(long)route * rows + row] = acc;
    }
}

// Routed twin of q2k_gemv for the Ghost-MoE expert arenas: blockIdx.y selects
// one routed expert; slot_ids locate its Q2_K gate_up slab in the fixed-stride
// VRAM arena and route_ids place the output row block in router order. Every
// expert dots the SAME shared Q8_K activation (the layer input), so there is no
// batched-input variant — the only Q2_K projection is gate_up, whose input is
// the pre-FFN residual; the per-expert down GEMV stays Q4_0. The per-super-block
// integer core and the lane-0 ordered f32 fold are copied VERBATIM from
// q2k_gemv, so each expert's output row is bit-identical to a dense q2k_gemv
// call over the same wire bytes — the dense kernel's oracle-parity contract
// (matches CPU q2_k_wire_row_dot) carries over unchanged.
extern "C" __global__ void q2k_gemv_routed(
    const float* __restrict__ input_scales,        // n_sb f32 (Q8_K d per superblock)
    const signed char* __restrict__ input_quants,  // n_sb*256 i8 (Q8_K quants)
    const unsigned char* __restrict__ weight_arena,
    const int* __restrict__ slot_ids,
    const int* __restrict__ route_ids,
    unsigned long long weight_stride, int rows, int n_sb,
    float* __restrict__ output, int expert_count
) {
    int expert = blockIdx.y;
    if (expert >= expert_count) return;
    int route = route_ids[expert];
    int slot = slot_ids[expert];
    const unsigned char* weight_bytes = weight_arena + (long)slot * weight_stride;

    extern __shared__ unsigned char smem2r[];
    signed char* s_iq = (signed char*)smem2r;                 // n_sb*256 i8 staged input
    float* s_is = (float*)(smem2r + (long)n_sb * 256);        // n_sb f32 staged scales
    // per-warp scratch: 2 ints (isum, summs) per superblock.
    int* acc = (int*)(smem2r + (long)n_sb * 256 + (long)n_sb * 4);
    int tid = threadIdx.x;
    for (int i = tid; i < n_sb * 64; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
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
            const signed char* y256 = s_iq + (long)b * 256;
            const unsigned char* sc = blk;          // scales[16]
            const unsigned char* qs = blk + 16;     // qs[64] (2-bit quants)
            int summs = 0;
            for (int j = 0; j < 16; j++) {
                int bsum = 0;
                for (int l = 0; l < 16; l++) bsum += (int)y256[j * 16 + l];
                summs += bsum * (int)(sc[j] >> 4);
            }
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
        output[(long)route * rows + row] = sumf;
    }
}

// Fuse each routed expert's GeGLU with its Q8_0 quantization. One WARP owns a
// 32-value Q8 block, so every lane keeps one GeGLU value in a register while a
// comparison-only shuffle reduction finds the block maximum. Max is independent
// of reduction order; initializing each lane from zero also preserves the former
// serial loop's behavior of ignoring NaNs in the maximum scan. This replaces 32
// serial tanhf/quantize iterations in one thread with one iteration per lane and
// eliminates the former 32-float per-thread local array.
extern "C" __global__ void geglu_quantize_routed(
    const float* __restrict__ gate_up,
    const int* __restrict__ route_ids,
    signed char* __restrict__ quants,
    float* __restrict__ scales,
    int nff, int blocks_per_expert, int expert_count
) {
    int expert = blockIdx.y;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_cta = blockDim.x >> 5;
    int block = blockIdx.x * warps_per_cta + warp;
    if (expert >= expert_count || block >= blocks_per_expert) return;
    int route = route_ids[expert];
    long value_index = (long)block * 32 + lane;
    const float* route_values = gate_up + (long)route * (2 * nff);
    float x = route_values[value_index];
    float up = route_values[nff + value_index];
    float inner = 0.79788456f * (x + 0.044715f * x * x * x);
    float value = (0.5f * x * (1.0f + tanhf(inner))) * up;

    // Match `max_abs = 0; if (fabsf(value) > max_abs) ...`: in particular,
    // a NaN candidate remains zero and cannot poison the warp reduction.
    float max_abs = 0.0f;
    float a = fabsf(value);
    if (a > max_abs) max_abs = a;
    unsigned int mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, max_abs, offset);
        if (other > max_abs) max_abs = other;
    }
    max_abs = __shfl_sync(mask, max_abs, 0);

    float unrounded = max_abs / 127.0f;
    if (lane == 0)
        scales[(long)route * blocks_per_expert + block] = f16_round(unrounded);
    float inv = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    float q = rintf(value * inv);
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    quants[((long)route * blocks_per_expert + block) * 32 + lane] = (signed char)q;
}

// Strict router-order weighted sum. One thread owns one hidden coordinate and
// accumulates expert 0..top_k exactly as the former scaled_axpy launch sequence.
extern "C" __global__ void moe_weighted_sum_routed(
    const float* __restrict__ expert_y,
    const float* __restrict__ route_scales,
    float* __restrict__ output, int hidden, int expert_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float acc = 0.0f;
    for (int expert = 0; expert < expert_count; expert++)
        acc += expert_y[(long)expert * hidden + i] * route_scales[expert];
    output[i] = acc;
}

// K-token counterpart of moe_weighted_sum_routed. Routed GEMMs write one row
// per CSR assignment (expert-major), while strict Gemma accumulation is ordered
// by [token][router-rank]. route_to_assignment bridges those layouts without
// reordering the floating-point fold: every output coordinate walks rank
// 0..route_count-1 and performs exactly the former mul-then-add sequence.
//
// The bounds check protects the arena read if a malformed plan reaches CUDA;
// production plans are total permutations over their assignment rows. NVRTC is
// compiled with --fmad=false, so the multiply and add remain separately rounded.
extern "C" __global__ void moe_weighted_sum_batched(
    const float* __restrict__ expert_y,
    const int* __restrict__ route_to_assignment,
    const float* __restrict__ route_scales,
    float* __restrict__ output,
    int assignment_count, int hidden, int k_tokens, int route_count
) {
    int token = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= k_tokens || i >= hidden) return;
    long route_base = (long)token * route_count;
    float acc = 0.0f;
    for (int route = 0; route < route_count; route++) {
        int assignment = route_to_assignment[route_base + route];
        if ((unsigned int)assignment < (unsigned int)assignment_count) {
            acc += expert_y[(long)assignment * hidden + i]
                * route_scales[route_base + route];
        }
    }
    output[(long)token * hidden + i] = acc;
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

// Prism/Bonsai fast RMSNorm -> Q8 activation. One CTA owns one token. The
// reduction is parallel (the fast Q1 lane already has a quantized-activation
// contract), and each warp quantizes a 32-value block directly from x/weight.
// No normalized f32 row is written to global memory.
extern "C" __global__ void prism_rms_norm_q8_batched(
    const float* __restrict__ x, const float* __restrict__ weight,
    signed char* __restrict__ quants, float* __restrict__ scales,
    int n, float eps, int k_tokens
) {
    int token = blockIdx.x;
    if (token >= k_tokens) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const float* xt = x + (long)token * n;
    __shared__ float warp_sums[8];
    __shared__ float norm_scale;
    float sum = 0.0f;
    for (int i = tid; i < n; i += 256) {
        float v = xt[i];
        sum += v * v;
    }
    sum += __shfl_down_sync(0xffffffffu, sum, 16);
    sum += __shfl_down_sync(0xffffffffu, sum, 8);
    sum += __shfl_down_sync(0xffffffffu, sum, 4);
    sum += __shfl_down_sync(0xffffffffu, sum, 2);
    sum += __shfl_down_sync(0xffffffffu, sum, 1);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0f;
        total += __shfl_down_sync(0xffffffffu, total, 16);
        total += __shfl_down_sync(0xffffffffu, total, 8);
        total += __shfl_down_sync(0xffffffffu, total, 4);
        total += __shfl_down_sync(0xffffffffu, total, 2);
        total += __shfl_down_sync(0xffffffffu, total, 1);
        if (lane == 0) norm_scale = 1.0f / sqrtf(total / (float)n + eps);
    }
    __syncthreads();

    int n_blocks = n >> 5;
    long token_q = (long)token * n;
    long token_s = (long)token * n_blocks;
    for (int qb = warp; qb < n_blocks; qb += 8) {
        int i = qb * 32 + lane;
        float v = xt[i] * norm_scale * weight[i];
        float max_abs = fabsf(v);
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 16));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 8));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 4));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 2));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 1));
        max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
        float unrounded = max_abs / 127.0f;
        if (lane == 0) scales[token_s + qb] = f16_round(unrounded);
        float inv = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
        float qv = rintf(v * inv);
        if (qv > 127.0f) qv = 127.0f;
        if (qv < -128.0f) qv = -128.0f;
        quants[token_q + i] = (signed char)qv;
    }
}

// Prism/Bonsai fast SwiGLU -> Q8 activation. Eight warps quantize eight Q8
// blocks per CTA; SiLU, amax and quantization are all lane-parallel.
extern "C" __global__ void prism_silu_mul_q8_batched(
    const float* __restrict__ gate, const float* __restrict__ up,
    signed char* __restrict__ quants, float* __restrict__ scales, int n_blocks
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int qb = blockIdx.x * 8 + warp;
    if (qb >= n_blocks) return;
    long i = (long)qb * 32 + lane;
    float g = gate[i];
    float v = (g / (1.0f + expf(-g))) * up[i];
    float max_abs = fabsf(v);
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 16));
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 8));
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 4));
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 2));
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 1));
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
    float unrounded = max_abs / 127.0f;
    if (lane == 0) scales[qb] = f16_round(unrounded);
    float inv = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    float qv = rintf(v * inv);
    if (qv > 127.0f) qv = 127.0f;
    if (qv < -128.0f) qv = -128.0f;
    quants[i] = (signed char)qv;
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

// Scatter K tokens' K/V into consecutive logical positions base..base+K-1.
// Physical slots wrap modulo max_pos, matching scalar kv_scatter and every ring
// attention read. Sliding callers must provide enough capacity that later rows
// in this launch cannot replace history still needed by an earlier row.
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
    int slot = position % max_pos;
    // f16-bit KV store (see kv_scatter): bit-identical to f16_round into f32.
    cache[((long)kv_head * max_pos + slot) * head_dim + d] =
        f32_to_f16_bits(src[(long)t * per_token_dim + (long)kv_head * head_dim + d]);
}

// Causal attention for K tokens: token t (at position base+t) attends [0, base+t].
// One block per (token, query head). Shared sized for the longest prefix (base+K).
extern "C" __global__ void attention_batched(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens, int splitk_active
, float* __restrict__ global_scores) {
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
    float* scores = global_scores + (long)blockIdx.x * max_pos;
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
        float* vpart = shared + head_dim; // [max_groups * head_dim]
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

// Sliding-window causal attention for K tokens: token t (at position base+t) attends
// [start_t, base+t], where start_t crops the window exactly as attention_decode_sw does.
//
// This is the batched counterpart of `attention_decode_sw`, and it exists because
// `attention_batched` above has no window: gemma4 runs 5 sliding layers to every full
// one, so a speculative verify batch cannot use `attention_batched` for 5/6 of the stack.
//
// BIT-EXACTNESS AGAINST DECODE, AND ITS PRECONDITION. `attention_decode_sw` reduces with
// G = blockDim.x / head_dim groups, so the reduction it performs depends on the width its
// launcher chose. There are two different callers in this tree and only one of them is
// this kernel's reference:
//
//   * the gemma4 runtime launches `attention_decode_sw` directly at blockDim.x == head_dim,
//     so G == 1 there unconditionally -- one ascending accumulation per output dimension,
//     no group split. THAT is what this kernel reproduces.
//   * `launch_attention_sw` (the gemma3 path) sizes G from the position count, so its G
//     varies with context length. This kernel is NOT bit-identical to that launcher and
//     must not be substituted for it.
//
// `launch_attention_sw_batched` therefore fixes blockDim.x at head_dim rather than taking
// it as a parameter: a wider block would silently reorder the weighted-V sum and let a
// verified token's argmax disagree with the same token's greedy argmax.
//
// `scores` is indexed RELATIVE to the window start, not by absolute position. That is
// pure addressing -- the loop bounds, visit order and accumulation order are unchanged --
// and it bounds shared memory by the window rather than by the whole context.
extern "C" __global__ void attention_sw_batched(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int window, int q_per_token, int k_tokens
) {
    int t = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int position_count = base_position + t + 1;
    int start = (window > 0 && position_count > window) ? (position_count - window) : 0;
    int active = position_count - start;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ float shared_swb[];
    float* qsh = shared_swb;              // head_dim
    float* scores = shared_swb + head_dim; // active, indexed p - start
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    // Same uint4 (8 keys/load) f16 read and same d-order accumulation as
    // attention_decode_sw, so the dot product is byte-identical under fmad=false.
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    for (int p = start + tid; p < position_count; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)(p % max_pos) * head_dim;
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
        scores[p - start] = dot * scale;
    }
    __syncthreads();

    __shared__ float s_max_swb, s_sum_swb;
    if (tid == 0) {
        float m = scores[0];
        for (int i = 1; i < active; i++) if (scores[i] > m) m = scores[i];
        s_max_swb = m;
    }
    __syncthreads();
    for (int i = tid; i < active; i += blockDim.x) scores[i] = expf(scores[i] - s_max_swb);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < active; i++) sum += scores[i];
        s_sum_swb = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum_swb;
    // G == 1: one thread per output dimension walks the whole window in ascending order,
    // which is the p_lo..p_hi loop attention_decode_sw runs when blockDim.x == head_dim.
    for (int did = tid; did < head_dim; did += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active; i++) {
            int p = start + i;
            acc += (scores[i] * inv) * f16_bits_to_f32(vbase[(long)(p % max_pos) * head_dim + did]);
        }
        out[(long)t * q_per_token + (long)head * head_dim + did] = acc;
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
, float* __restrict__ global_scores) {
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
    int* slots = (int*)(shared + head_dim); // absolute KV slot per score
    float* scores = global_scores + (long)blockIdx.x * max_pos;
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

// Scatter K tokens' K/V into the cache in Q8_0 format at consecutive positions base..base+K-1.
extern "C" __global__ void kv_scatter_batched_q8_0(
    const float* __restrict__ src, block_q8_0* __restrict__ cache, int base_position,
    int n_kv_heads, int head_dim, int max_pos, int per_token_dim, int k_tokens
) {
    int blocks_per_head = head_dim / 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_tokens * n_kv_heads * blocks_per_head;
    if (idx >= total) return;
    int t = idx / (n_kv_heads * blocks_per_head);
    int rem = idx % (n_kv_heads * blocks_per_head);
    int kv_head = rem / blocks_per_head;
    int b = rem % blocks_per_head;
    int position = base_position + t;

    const float* chunk = src + ((long)t * per_token_dim + (long)kv_head * head_dim + b * 32);
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = fabsf(chunk[i]);
        if (v > amax) amax = v;
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    block_q8_0 blk;
    blk.scale = f32_to_f16_bits(d);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float val = roundf(chunk[i] * id);
        if (val < -127.0f) val = -127.0f;
        if (val > 127.0f) val = 127.0f;
        blk.qs[i] = (signed char)val;
    }
    cache[((long)kv_head * max_pos + position) * blocks_per_head + b] = blk;
}

extern "C" __global__ void kv_scatter_tree_batched_q8_0(
    const float* __restrict__ src, block_q8_0* __restrict__ cache,
    const int* __restrict__ node_kvslot,
    int n_kv_heads, int head_dim, int max_pos, int per_token_dim, int k_tokens
) {
    int blocks_per_head = head_dim / 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_tokens * n_kv_heads * blocks_per_head;
    if (idx >= total) return;
    int t = idx / (n_kv_heads * blocks_per_head);
    int rem = idx % (n_kv_heads * blocks_per_head);
    int kv_head = rem / blocks_per_head;
    int b = rem % blocks_per_head;
    int position = node_kvslot[t];

    const float* chunk = src + ((long)t * per_token_dim + (long)kv_head * head_dim + b * 32);
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = fabsf(chunk[i]);
        if (v > amax) amax = v;
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    block_q8_0 blk;
    blk.scale = f32_to_f16_bits(d);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float val = roundf(chunk[i] * id);
        if (val < -127.0f) val = -127.0f;
        if (val > 127.0f) val = 127.0f;
        blk.qs[i] = (signed char)val;
    }
    cache[((long)kv_head * max_pos + position) * blocks_per_head + b] = blk;
}

extern "C" __global__ void attention_batched_q8_0(
    const float* __restrict__ q, const block_q8_0* __restrict__ cache_k,
    const block_q8_0* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens, int splitk_active
, float* __restrict__ global_scores) {
    int t = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int position_count = base_position + t + 1;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    int blocks_per_head = head_dim / 32;
    const block_q8_0* kbase = cache_k + (long)kv_head * max_pos * blocks_per_head;
    const block_q8_0* vbase = cache_v + (long)kv_head * max_pos * blocks_per_head;

    extern __shared__ float shared[];
    float* qsh = shared;
    float* scores = global_scores + (long)blockIdx.x * max_pos;
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    for (int p = tid; p < position_count; p += blockDim.x) {
        const block_q8_0* kp = kbase + (long)p * blocks_per_head;
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            float d = f16_bits_to_f32(kp[b].scale);
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum += qsh[b * 32 + i] * (float)kp[b].qs[i];
            }
            dot += sum * d;
        }
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

    if (splitk_active && position_count > 512) {
        int n_splits = (position_count + 255) / 256;
        if (n_splits < 2) n_splits = 2;
        if (n_splits > 32) n_splits = 32;
        if (tid == 0) {
            float total = 0.0f;
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
            int b = did / 32;
            int bi = did % 32;
            float acc = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int p_lo = (int)((long)sp * position_count / n_splits);
                int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
                float a = 0.0f;
                for (int p = p_lo; p < p_hi; p++) {
                    const block_q8_0* vp = vbase + ((long)p * blocks_per_head + b);
                    float d = f16_bits_to_f32(vp->scale);
                    a += scores[p] * (d * (float)vp->qs[bi]);
                }
                acc += a;
            }
            out[(long)t * q_per_token + (long)head * head_dim + did] = acc * inv;
        }
    } else {
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
        float* vpart = shared + head_dim;
        int gid = tid / head_dim;
        int did = tid % head_dim;
        int b = did / 32;
        int bi = did % 32;
        int p_lo = (int)((long)gid * position_count / G);
        int p_hi = (int)((long)(gid + 1) * position_count / G);
        float acc = 0.0f;
        for (int p = p_lo; p < p_hi; p++) {
            const block_q8_0* vp = vbase + ((long)p * blocks_per_head + b);
            float d = f16_bits_to_f32(vp->scale);
            acc += (scores[p] * inv) * (d * (float)vp->qs[bi]);
        }
        vpart[(long)did * G + gid] = acc;
        __syncthreads();
        if (gid == 0) {
            float sum = 0.0f;
            for (int g = 0; g < G; g++) sum += vpart[(long)did * G + g];
            out[(long)t * q_per_token + (long)head * head_dim + did] = sum;
        }
    }
}

extern "C" __global__ void attention_tree_batched_q8_0(
    const float* __restrict__ q, const block_q8_0* __restrict__ cache_k,
    const block_q8_0* __restrict__ cache_v, float* __restrict__ out,
    const unsigned int* __restrict__ ancestor_bits, int words,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens, int splitk_active
, float* __restrict__ global_scores) {
    int t = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    int blocks_per_head = head_dim / 32;
    const block_q8_0* kbase = cache_k + (long)kv_head * max_pos * blocks_per_head;
    const block_q8_0* vbase = cache_v + (long)kv_head * max_pos * blocks_per_head;
    const unsigned int* anc = ancestor_bits + (long)t * words;

    extern __shared__ float shared[];
    float* qsh = shared;
    int* slots = (int*)(shared + head_dim);
    float* scores = global_scores + (long)blockIdx.x * max_pos;
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

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

    for (int i = tid; i < count; i += blockDim.x) {
        const block_q8_0* kp = kbase + (long)slots[i] * blocks_per_head;
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            float d = f16_bits_to_f32(kp[b].scale);
            float sum = 0.0f;
            #pragma unroll
            for (int idx = 0; idx < 32; idx++) {
                sum += qsh[b * 32 + idx] * (float)kp[b].qs[idx];
            }
            dot += sum * d;
        }
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

    if (splitk_active && count > 512) {
        int n_splits = (count + 255) / 256;
        if (n_splits < 2) n_splits = 2;
        if (n_splits > 32) n_splits = 32;
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
            int b = did / 32;
            int bi = did % 32;
            float acc = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int i_lo = (int)((long)sp * count / n_splits);
                int i_hi = (int)((long)(sp + 1) * count / n_splits);
                float a = 0.0f;
                for (int i = i_lo; i < i_hi; i++) {
                    const block_q8_0* vp = vbase + ((long)slots[i] * blocks_per_head + b);
                    float d = f16_bits_to_f32(vp->scale);
                    a += scores[i] * (d * (float)vp->qs[bi]);
                }
                acc += a;
            }
            out[(long)t * q_per_token + (long)head * head_dim + did] = acc * inv;
        }
    } else {
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
        float* vpart = (float*)(slots + base_position + k_tokens);
        for (int idx = tid; idx < G * head_dim; idx += blockDim.x) {
            int gid = idx / head_dim, did = idx % head_dim;
            int b = did / 32;
            int bi = did % 32;
            int i_lo = (int)((long)gid * count / G);
            int i_hi = (int)((long)(gid + 1) * count / G);
            float acc = 0.0f;
            for (int i = i_lo; i < i_hi; i++) {
                const block_q8_0* vp = vbase + ((long)slots[i] * blocks_per_head + b);
                float d = f16_bits_to_f32(vp->scale);
                acc += (scores[i] * inv) * (d * (float)vp->qs[bi]);
            }
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

extern "C" __global__ void attn_sk_scores_q8_0(
    const float* __restrict__ q, const block_q8_0* __restrict__ cache_k,
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
    int blocks_per_head = head_dim / 32;
    const block_q8_0* kbase = cache_k + (long)kv_head * max_pos * blocks_per_head;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);

    extern __shared__ float qsh[];
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    float local_max = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const block_q8_0* kp = kbase + (long)p * blocks_per_head;
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            float d = f16_bits_to_f32(kp[b].scale);
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum += qsh[b * 32 + i] * (float)kp[b].qs[i];
            }
            dot += sum * d;
        }
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

extern "C" __global__ void attn_sk_partial_q8_0(
    const block_q8_0* __restrict__ cache_v, float* __restrict__ scores_buf,
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
    int blocks_per_head = head_dim / 32;
    const block_q8_0* vbase = cache_v + (long)kv_head * max_pos * blocks_per_head;
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
        int b = d / 32;
        int bi = d % 32;
        float a = 0.0f;
        for (int p = p_lo; p < p_hi; p++) {
            const block_q8_0* vp = vbase + ((long)p * blocks_per_head + b);
            float scale_val = f16_bits_to_f32(vp->scale);
            a += sc_head[p] * (scale_val * (float)vp->qs[bi]);
        }
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

// ---- Fused Tiled Flash Prefill Attention with Online Softmax ----------------
// Dynamic chunked flash prefill for prompt ingestion (1..=256+ tokens).
// Tiled across query blocks (B_Q = 16) and key blocks (B_K = 32) in shared memory and registers.
// Maintains running online softmax max (m) and sum-exp (l) with per-tile rescaling,
// computing causal attention in a single fused kernel pass with 0 bytes intermediate DRAM scratch
// and fully unrolled register accumulation for standard head dimensions (64, 128, 256).
#define FLASH_PREFILL_BQ 16
#define FLASH_PREFILL_BK 32

template<int HEAD_DIM>
__device__ void flash_attention_prefill_tiled_impl(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    constexpr int D_STEPS = HEAD_DIM / 32;

    int head = blockIdx.x;
    int q_tile = blockIdx.y;
    if (head >= n_heads) return;

    int t_start = q_tile * FLASH_PREFILL_BQ;
    if (t_start >= k_tokens) return;
    int t_count = k_tokens - t_start;
    if (t_count > FLASH_PREFILL_BQ) t_count = FLASH_PREFILL_BQ;

    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * HEAD_DIM;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * HEAD_DIM;

    extern __shared__ char smem_raw[];
    float* q_smem = reinterpret_cast<float*>(smem_raw);
    unsigned short* k_smem = reinterpret_cast<unsigned short*>(q_smem + FLASH_PREFILL_BQ * HEAD_DIM);
    unsigned short* v_smem = k_smem + FLASH_PREFILL_BK * HEAD_DIM;

    int tid = threadIdx.x;
    int num_threads = blockDim.x; // 256 threads = 8 warps

    // 1. Collaboratively load Q tile into shared memory
    for (int i = tid; i < t_count * HEAD_DIM; i += num_threads) {
        int t = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        q_smem[t * HEAD_DIM + d] = q[(long)(t_start + t) * q_per_token + (long)head * HEAD_DIM + d];
    }
    __syncthreads();

    // 2. Warp-level query assignment:
    // With 8 warps (blockDim.x = 256):
    // Warp w in [0, 8) handles queries qi = 2*w and qi = 2*w + 1
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int q_local[2] = { warp_id * 2, warp_id * 2 + 1 };

    float m_state[2] = { -3.4e38f, -3.4e38f };
    float l_state[2] = { 0.0f, 0.0f };

    float o_acc[2][D_STEPS];
    #pragma unroll
    for (int qi = 0; qi < 2; qi++) {
        #pragma unroll
        for (int di = 0; di < D_STEPS; di++) {
            o_acc[qi][di] = 0.0f;
        }
    }

    int max_p_tile = base_position + t_start + t_count;

    // 3. Tile over keys and values in blocks of FLASH_PREFILL_BK (32)
    for (int kp0 = 0; kp0 < max_p_tile; kp0 += FLASH_PREFILL_BK) {
        int k_count = max_p_tile - kp0;
        if (k_count > FLASH_PREFILL_BK) k_count = FLASH_PREFILL_BK;

        for (int i = tid; i < k_count * HEAD_DIM; i += num_threads) {
            int p_idx = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int global_p = kp0 + p_idx;
            k_smem[p_idx * HEAD_DIM + d] = kbase[(long)global_p * HEAD_DIM + d];
            v_smem[p_idx * HEAD_DIM + d] = vbase[(long)global_p * HEAD_DIM + d];
        }
        __syncthreads();

        #pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            int q_idx = q_local[qi];
            if (q_idx < t_count) {
                int global_q_pos = base_position + t_start + q_idx;
                if (kp0 <= global_q_pos) {
                    float tile_scores[FLASH_PREFILL_BK];
                    float tile_m = -3.4e38f;

                    #pragma unroll
                    for (int p_idx = 0; p_idx < FLASH_PREFILL_BK; p_idx++) {
                        if (p_idx < k_count && (kp0 + p_idx) <= global_q_pos) {
                            float pdot = 0.0f;
                            #pragma unroll
                            for (int di = 0; di < D_STEPS; di++) {
                                int d = lane_id + di * 32;
                                float q_val = q_smem[q_idx * HEAD_DIM + d];
                                float k_val = f16_bits_to_f32(k_smem[p_idx * HEAD_DIM + d]);
                                pdot += q_val * k_val;
                            }

                            #pragma unroll
                            for (int mask = 16; mask > 0; mask >>= 1) {
                                pdot += __shfl_xor_sync(0xffffffffu, pdot, mask);
                            }

                            float score = pdot * scale;
                            tile_scores[p_idx] = score;
                            tile_m = fmaxf(tile_m, score);
                        } else {
                            tile_scores[p_idx] = -3.4e38f;
                        }
                    }

                    if (tile_m > -3.0e38f) {
                        float m_prev = m_state[qi];
                        float m_curr = fmaxf(m_prev, tile_m);
                        float alpha = expf(m_prev - m_curr);

                        l_state[qi] = l_state[qi] * alpha;
                        #pragma unroll
                        for (int di = 0; di < D_STEPS; di++) {
                            o_acc[qi][di] = o_acc[qi][di] * alpha;
                        }

                        float tile_l = 0.0f;
                        #pragma unroll
                        for (int p_idx = 0; p_idx < FLASH_PREFILL_BK; p_idx++) {
                            if (p_idx < k_count && (kp0 + p_idx) <= global_q_pos) {
                                float weight = expf(tile_scores[p_idx] - m_curr);
                                tile_l += weight;
                                #pragma unroll
                                for (int di = 0; di < D_STEPS; di++) {
                                    int d = lane_id + di * 32;
                                    float v_val = f16_bits_to_f32(v_smem[p_idx * HEAD_DIM + d]);
                                    o_acc[qi][di] += weight * v_val;
                                }
                            }
                        }

                        l_state[qi] += tile_l;
                        m_state[qi] = m_curr;
                    }
                }
            }
        }
        __syncthreads();
    }

    // 4. Write normalized output to global memory
    #pragma unroll
    for (int qi = 0; qi < 2; qi++) {
        int q_idx = q_local[qi];
        if (q_idx < t_count) {
            float inv_l = (l_state[qi] > 0.0f) ? (1.0f / l_state[qi]) : 0.0f;
            int global_t = t_start + q_idx;
            #pragma unroll
            for (int di = 0; di < D_STEPS; di++) {
                int d = lane_id + di * 32;
                out[(long)global_t * q_per_token + (long)head * HEAD_DIM + d] = o_acc[qi][di] * inv_l;
            }
        }
    }
}

template<int MAX_D_STEPS = 8>
__device__ void flash_attention_prefill_tiled_dynamic(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    int d_steps = (head_dim + 31) / 32;
    if (d_steps > MAX_D_STEPS) d_steps = MAX_D_STEPS;

    int head = blockIdx.x;
    int q_tile = blockIdx.y;
    if (head >= n_heads) return;

    int t_start = q_tile * FLASH_PREFILL_BQ;
    if (t_start >= k_tokens) return;
    int t_count = k_tokens - t_start;
    if (t_count > FLASH_PREFILL_BQ) t_count = FLASH_PREFILL_BQ;

    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ char smem_raw[];
    float* q_smem = reinterpret_cast<float*>(smem_raw);
    unsigned short* k_smem = reinterpret_cast<unsigned short*>(q_smem + FLASH_PREFILL_BQ * head_dim);
    unsigned short* v_smem = k_smem + FLASH_PREFILL_BK * head_dim;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = tid; i < t_count * head_dim; i += num_threads) {
        int t = i / head_dim;
        int d = i % head_dim;
        q_smem[t * head_dim + d] = q[(long)(t_start + t) * q_per_token + (long)head * head_dim + d];
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int q_local[2] = { warp_id * 2, warp_id * 2 + 1 };

    float m_state[2] = { -3.4e38f, -3.4e38f };
    float l_state[2] = { 0.0f, 0.0f };

    float o_acc[2][MAX_D_STEPS];
    #pragma unroll
    for (int qi = 0; qi < 2; qi++) {
        #pragma unroll
        for (int di = 0; di < MAX_D_STEPS; di++) {
            o_acc[qi][di] = 0.0f;
        }
    }

    int max_p_tile = base_position + t_start + t_count;

    for (int kp0 = 0; kp0 < max_p_tile; kp0 += FLASH_PREFILL_BK) {
        int k_count = max_p_tile - kp0;
        if (k_count > FLASH_PREFILL_BK) k_count = FLASH_PREFILL_BK;

        for (int i = tid; i < k_count * head_dim; i += num_threads) {
            int p_idx = i / head_dim;
            int d = i % head_dim;
            int global_p = kp0 + p_idx;
            k_smem[p_idx * head_dim + d] = kbase[(long)global_p * head_dim + d];
            v_smem[p_idx * head_dim + d] = vbase[(long)global_p * head_dim + d];
        }
        __syncthreads();

        for (int qi = 0; qi < 2; qi++) {
            int q_idx = q_local[qi];
            if (q_idx < t_count) {
                int global_q_pos = base_position + t_start + q_idx;
                if (kp0 <= global_q_pos) {
                    float tile_scores[FLASH_PREFILL_BK];
                    float tile_m = -3.4e38f;

                    for (int p_idx = 0; p_idx < FLASH_PREFILL_BK; p_idx++) {
                        if (p_idx < k_count && (kp0 + p_idx) <= global_q_pos) {
                            float pdot = 0.0f;
                            for (int di = 0; di < d_steps; di++) {
                                int d = lane_id + di * 32;
                                if (d < head_dim) {
                                    float q_val = q_smem[q_idx * head_dim + d];
                                    float k_val = f16_bits_to_f32(k_smem[p_idx * head_dim + d]);
                                    pdot += q_val * k_val;
                                }
                            }

                            #pragma unroll
                            for (int mask = 16; mask > 0; mask >>= 1) {
                                pdot += __shfl_xor_sync(0xffffffffu, pdot, mask);
                            }

                            float score = pdot * scale;
                            tile_scores[p_idx] = score;
                            tile_m = fmaxf(tile_m, score);
                        } else {
                            tile_scores[p_idx] = -3.4e38f;
                        }
                    }

                    if (tile_m > -3.0e38f) {
                        float m_prev = m_state[qi];
                        float m_curr = fmaxf(m_prev, tile_m);
                        float alpha = expf(m_prev - m_curr);

                        l_state[qi] = l_state[qi] * alpha;
                        for (int di = 0; di < d_steps; di++) {
                            o_acc[qi][di] = o_acc[qi][di] * alpha;
                        }

                        float tile_l = 0.0f;
                        for (int p_idx = 0; p_idx < FLASH_PREFILL_BK; p_idx++) {
                            if (p_idx < k_count && (kp0 + p_idx) <= global_q_pos) {
                                float weight = expf(tile_scores[p_idx] - m_curr);
                                tile_l += weight;
                                for (int di = 0; di < d_steps; di++) {
                                    int d = lane_id + di * 32;
                                    if (d < head_dim) {
                                        float v_val = f16_bits_to_f32(v_smem[p_idx * head_dim + d]);
                                        o_acc[qi][di] += weight * v_val;
                                    }
                                }
                            }
                        }

                        l_state[qi] += tile_l;
                        m_state[qi] = m_curr;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int qi = 0; qi < 2; qi++) {
        int q_idx = q_local[qi];
        if (q_idx < t_count) {
            float inv_l = (l_state[qi] > 0.0f) ? (1.0f / l_state[qi]) : 0.0f;
            int global_t = t_start + q_idx;
            for (int di = 0; di < d_steps; di++) {
                int d = lane_id + di * 32;
                if (d < head_dim) {
                    out[(long)global_t * q_per_token + (long)head * head_dim + d] = o_acc[qi][di] * inv_l;
                }
            }
        }
    }
}

// One entry point per supported head_dim. They are NOT folded into a single kernel with a runtime
// branch: ptxas sizes registers and the stack frame for the WHOLE function, so the worst branch is
// charged to every launch. Measured on sm_89 with all four folded into one entry point: num_regs=71
// and a 224-byte per-thread stack frame on EVERY launch. Split, the frame is zero on all three
// specialized kernels (64/128/256 -> 63/89/96 registers), which is what actually makes `o_acc`
// register-resident rather than local-memory backed.
extern "C" __global__ void flash_attention_prefill_tiled_d64(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    (void)head_dim;
    flash_attention_prefill_tiled_impl<64>(q, cache_k, cache_v, out, n_heads, n_kv_heads, base_position, k_tokens, q_per_token, max_pos, scale);
}

extern "C" __global__ void flash_attention_prefill_tiled_d128(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    (void)head_dim;
    flash_attention_prefill_tiled_impl<128>(q, cache_k, cache_v, out, n_heads, n_kv_heads, base_position, k_tokens, q_per_token, max_pos, scale);
}

extern "C" __global__ void flash_attention_prefill_tiled_d256(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    (void)head_dim;
    flash_attention_prefill_tiled_impl<256>(q, cache_k, cache_v, out, n_heads, n_kv_heads, base_position, k_tokens, q_per_token, max_pos, scale);
}

// Runtime-head_dim twin for every other head_dim. Its loops are bounded by the runtime `head_dim`,
// so `o_acc` is dynamically indexed and genuinely lands in local memory (measured: 224 bytes). It
// stays a separate entry point so that frame is never charged to the specialized kernels above.
extern "C" __global__ void flash_attention_prefill_tiled_dyn(
    const float* __restrict__ q,
    const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v,
    float* __restrict__ out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int base_position,
    int k_tokens,
    int q_per_token,
    int max_pos,
    float scale
) {
    flash_attention_prefill_tiled_dynamic(q, cache_k, cache_v, out, n_heads, n_kv_heads, head_dim, base_position, k_tokens, q_per_token, max_pos, scale);
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
//   decay:  S[i,j] *= decay[h]
//   sk[j]   = sum_i S[i,j]*k[i]                 (fused with decay, i-order)
//   d[j]    = (v[j] - sk[j]) * beta[h]
//   S[i,j] += k[i]*d[j];  o[j] = sum_i S[i,j]*(q[i]*qscale)   (fused, i-order)
//   gated RMSNorm: out = RMSNorm(o, ssm_norm) * SiLU(z)       (j-order serial sum)
// k_conv/q_conv are L2-normed; GQA tile-repeat key head = h % nk. beta is already
// sigmoid'd; decay is already exponentiated. Bit-identical update order to the
// CPU reference.
extern "C" __global__ void ssm_delta_rule(
    float* __restrict__ state,           // [nv*d_state*d_state], updated in place
    const float* __restrict__ k_conv,    // [nk*d_state]
    const float* __restrict__ q_conv,    // [nk*d_state]
    const float* __restrict__ v_conv,    // [nv*d_state]
    const float* __restrict__ z,         // [nv*d_state]
    const float* __restrict__ beta,      // [nv] (post-sigmoid)
    const float* __restrict__ decay,     // [nv] (post-exp)
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
    float g = decay[h];
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
__device__ __forceinline__ float prism_embed_value(
    const unsigned char* table, unsigned int token, int dim, int e,
    int bits, int block_elements
) {
    int block_bytes = 2 + ((block_elements * bits) >> 3);
    int row_bytes = (dim / block_elements) * block_bytes;
    const unsigned char* block = table + (long)token * row_bytes
        + (long)(e / block_elements) * block_bytes;
    int j = e % block_elements;
    float d = f16_bits_to_f32((unsigned short)block[0] | ((unsigned short)block[1] << 8));
    if (bits == 1) {
        int bit = (block[2 + (j >> 3)] >> (j & 7)) & 1;
        return bit ? d : -d;
    }
    int q = (block[2 + (j >> 2)] >> ((j & 3) << 1)) & 3;
    return (float)(q - 1) * d;
}
extern "C" __global__ void embed_gather_q1_0(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < dim) out[e] = prism_embed_value(table, *token, dim, e, 1, 128);
}
extern "C" __global__ void embed_gather_q2_0_g64(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < dim) out[e] = prism_embed_value(table, *token, dim, e, 2, 64);
}
extern "C" __global__ void embed_gather_q2_0_g128(
    const unsigned char* __restrict__ table, const unsigned int* __restrict__ token,
    int dim, float* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < dim) out[e] = prism_embed_value(table, *token, dim, e, 2, 128);
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

// ---- SSM gates: beta = sigmoid(beta_raw); decay = exp(softplus(alpha+bias)*a)
// One thread per value-head (nv). Feeds ssm_delta_rule with both values ready.
// `decay_out` is the already exponentiated recurrence multiplier.  Computing it
// here is important for qwen35: the delta-rule kernel has one thread per state
// column, so exponentiating there would repeat the exact same transcendental
// operation `d_state` times for every token and value head.
// softplus matches the CPU's (1+exp(x)).ln() with the x>20 passthrough.
extern "C" __global__ void ssm_gates(
    const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw,
    const float* __restrict__ dt_bias, const float* __restrict__ a,
    float* __restrict__ beta_out, float* __restrict__ decay_out, int nv
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    beta_out[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
    float x = alpha_raw[h] + dt_bias[h];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    decay_out[h] = expf(sp * a[h]);
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

// Batched qwen35 helpers. Dense projections stay token-major; only the causal
// conv and delta-rule recurrence loop over tokens inside one launch so state is
// updated in exactly the same order as tokenwise prefill.
extern "C" __global__ void deinterleave_qgate_batched(
    const float* __restrict__ qg, float* __restrict__ q_out,
    float* __restrict__ gate_out, int n_heads, int head_dim, int k_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_width = n_heads * head_dim;
    if (idx >= k_tokens * q_width) return;
    int token = idx / q_width;
    int within = idx - token * q_width;
    int head = within / head_dim;
    int d = within - head * head_dim;
    long src = (long)token * 2 * q_width + (long)head * 2 * head_dim;
    q_out[idx] = qg[src + d];
    gate_out[idx] = qg[src + head_dim + d];
}

extern "C" __global__ void ssm_gates_batched(
    const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw,
    const float* __restrict__ dt_bias, const float* __restrict__ a,
    float* __restrict__ beta_out, float* __restrict__ decay_out,
    int nv, int k_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k_tokens * nv) return;
    int h = idx % nv;
    beta_out[idx] = 1.0f / (1.0f + expf(-beta_raw[idx]));
    float x = alpha_raw[idx] + dt_bias[h];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    decay_out[idx] = expf(sp * a[h]);
}

extern "C" __global__ void ssm_conv1d_batched(
    const float* __restrict__ conv_w, const float* __restrict__ x,
    float* __restrict__ conv_state, float* __restrict__ conv_out,
    int conv_dim, int d_conv, int k_tokens
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int cm1 = d_conv - 1;
    const float* w = conv_w + (long)c * d_conv;
    float* st = conv_state + (long)c * cm1;
    for (int token = 0; token < k_tokens; token++) {
        float xc = x[(long)token * conv_dim + c];
        float acc = 0.0f;
        for (int t = 0; t < cm1; t++) acc += w[t] * st[t];
        acc += w[cm1] * xc;
        conv_out[(long)token * conv_dim + c] = acc / (1.0f + expf(-acc));
        for (int t = 0; t < cm1 - 1; t++) st[t] = st[t + 1];
        st[cm1 - 1] = xc;
    }
}

extern "C" __global__ void ssm_l2_norm_per_head_batched(
    float* __restrict__ buf, int token_stride, int offset,
    int n_heads, int head_dim, int k_tokens, float eps
) {
    int token_head = blockIdx.x;
    int token = token_head / n_heads;
    int head = token_head - token * n_heads;
    if (token >= k_tokens) return;
    int tid = threadIdx.x;
    extern __shared__ float xs[];
    __shared__ float s_scale;
    long base = (long)token * token_stride + offset + (long)head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) xs[i] = buf[base + i];
    __syncthreads();
    if (tid == 0) {
        double ss = 0.0;
        for (int i = 0; i < head_dim; i++) {
            double v = (double)xs[i];
            ss += v * v;
        }
        float denom = sqrtf((float)ss);
        if (denom < eps) denom = eps;
        s_scale = 1.0f / denom;
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = tid; i < head_dim; i += blockDim.x) buf[base + i] = xs[i] * scale;
}

extern "C" __global__ void ssm_delta_rule_batched(
    float* __restrict__ state, const float* __restrict__ conv,
    const float* __restrict__ z, const float* __restrict__ beta,
    const float* __restrict__ decay, const float* __restrict__ ssm_norm,
    float* __restrict__ out, int d_state, int nk, int nv,
    int key_dim, int value_dim, int conv_dim, int k_tokens, float eps
) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    int hk = h % nk;
    extern __shared__ float sh[];
    float* sk_ = sh;
    float* sq_ = sh + d_state;
    float* so_ = sh + 2 * d_state;
    float* St = state + (long)h * d_state * d_state;
    float qscale = 1.0f / sqrtf((float)d_state);
    __shared__ float s_scale;

    for (int token = 0; token < k_tokens; token++) {
        const float* token_conv = conv + (long)token * conv_dim;
        sk_[j] = token_conv[key_dim + (long)hk * d_state + j];
        sq_[j] = token_conv[(long)hk * d_state + j];
        __syncthreads();
        float g = decay[(long)token * nv + h];
        float bh = beta[(long)token * nv + h];
        float skj = 0.0f;
        for (int i = 0; i < d_state; i++) {
            float sv = St[(long)i * d_state + j] * g;
            St[(long)i * d_state + j] = sv;
            skj += sv * sk_[i];
        }
        float dj = (token_conv[2 * key_dim + (long)h * d_state + j] - skj) * bh;
        float oj = 0.0f;
        for (int i = 0; i < d_state; i++) {
            float sv = St[(long)i * d_state + j] + sk_[i] * dj;
            St[(long)i * d_state + j] = sv;
            oj += sv * (sq_[i] * qscale);
        }
        so_[j] = oj;
        __syncthreads();
        if (j == 0) {
            float sum = 0.0f;
            for (int i = 0; i < d_state; i++) sum += so_[i] * so_[i];
            s_scale = 1.0f / sqrtf(sum / (float)d_state + eps);
        }
        __syncthreads();
        float normed = so_[j] * s_scale * ssm_norm[j];
        float zj = z[(long)token * value_dim + (long)h * d_state + j];
        float silu_z = zj / (1.0f + expf(-zj));
        out[(long)token * value_dim + (long)h * d_state + j] = normed * silu_z;
        __syncthreads();
    }
}

// ---- Qwen3.5/Bonsai D=128 SSM kernels ------------------------------------
// These are not generic compatibility kernels.  They encode the geometry used
// by the Bonsai family (128 state channels and a four-tap causal convolution)
// so the compiler can keep the recurrent state and convolution history in
// registers across the complete prompt chunk.

// Four-tap causal convolution with register-resident history.  The generic
// implementation updates global conv_state after every token; this version
// loads the three history values and four weights once per channel and writes
// the final history once after the complete chunk.
extern "C" __global__ void qwen35_ssm_conv1d_d4_batched(
    const float* __restrict__ conv_w, const float* __restrict__ x,
    float* __restrict__ conv_state, float* __restrict__ conv_out,
    int conv_dim, int k_tokens
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float* w = conv_w + (long)c * 4;
    float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    float* history = conv_state + (long)c * 3;
    float s0 = history[0], s1 = history[1], s2 = history[2];
    for (int token = 0; token < k_tokens; token++) {
        float xc = x[(long)token * conv_dim + c];
        float acc = 0.0f;
        acc += w0 * s0;
        acc += w1 * s1;
        acc += w2 * s2;
        acc += w3 * xc;
        conv_out[(long)token * conv_dim + c] = acc / (1.0f + expf(-acc));
        s0 = s1;
        s1 = s2;
        s2 = xc;
    }
    history[0] = s0;
    history[1] = s1;
    history[2] = s2;
}

// Normalize the paired Q/K vectors in one launch.  One block owns one
// (token,key-head), and the serial double-precision sums intentionally retain
// the reference kernel's arithmetic while eliminating the second launch.
extern "C" __global__ void qwen35_ssm_qk_l2_norm_d128_batched(
    float* __restrict__ conv, int conv_dim, int key_dim,
    int n_key_heads, int k_tokens, float eps
) {
    int token_head = blockIdx.x;
    int token = token_head / n_key_heads;
    int head = token_head - token * n_key_heads;
    if (token >= k_tokens) return;
    int j = threadIdx.x;
    __shared__ float qv[128];
    __shared__ float kv[128];
    __shared__ float q_scale;
    __shared__ float k_scale;
    long q_base = (long)token * conv_dim + (long)head * 128;
    long k_base = q_base + key_dim;
    qv[j] = conv[q_base + j];
    kv[j] = conv[k_base + j];
    __syncthreads();
    if (j == 0) {
        double qss = 0.0;
        double kss = 0.0;
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            double q = (double)qv[i];
            double k = (double)kv[i];
            qss += q * q;
            kss += k * k;
        }
        float qden = sqrtf((float)qss);
        float kden = sqrtf((float)kss);
        if (qden < eps) qden = eps;
        if (kden < eps) kden = eps;
        q_scale = 1.0f / qden;
        k_scale = 1.0f / kden;
    }
    __syncthreads();
    conv[q_base + j] = qv[j] * q_scale;
    conv[k_base + j] = kv[j] * k_scale;
}

__device__ __forceinline__ float qwen35_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

// Register-sharded gated delta rule for D=128. Fast-mode state is persistently
// transposed as [head][column][row]. Each of eight warps owns four columns and
// each lane keeps four rows from every owned column in registers, so a block
// advances 32 columns through the complete prompt chunk. Four blocks cover a
// value head. This avoids the 128-register/thread cliff of one-thread/column,
// retains high occupancy, and moves each state element exactly once in and once
// out for the whole chunk.
extern "C" __global__ __launch_bounds__(256, 2)
void qwen35_ssm_delta_rule_d128_batched(
    float* __restrict__ state_t, const float* __restrict__ conv,
    const float* __restrict__ beta, const float* __restrict__ decay,
    float* __restrict__ raw_out, int nk, int nv, int key_dim,
    int value_dim, int conv_dim, int k_tokens
) {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int flat = warp * 32 + lane;
    int h = blockIdx.x;
    if (h >= nv) return;
    int hk = h % nk;
    int col0 = blockIdx.z * 32 + warp * 4;
    __shared__ float sk_[128];
    __shared__ float sq_[128];
    __shared__ float s_decay;
    __shared__ float s_beta;
    float st[4][4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = col0 + c;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = lane + 32 * r;
            st[c][r] = state_t[((long)h * 128 + col) * 128 + row];
        }
    }
    const float qscale = 0.08838834764831845f; // 1/sqrt(128)

    for (int token = 0; token < k_tokens; token++) {
        const float* token_conv = conv + (long)token * conv_dim;
        if (flat < 128) {
            sk_[flat] = token_conv[key_dim + (long)hk * 128 + flat];
            sq_[flat] = token_conv[(long)hk * 128 + flat];
        }
        if (flat == 0) {
            s_decay = decay[(long)token * nv + h];
            s_beta = beta[(long)token * nv + h];
        }
        __syncthreads();

        float kr[4];
        float qr[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = lane + 32 * r;
            kr[r] = sk_[row];
            qr[r] = sq_[row] * qscale;
        }

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int col = col0 + c;
            float sk_partial = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                st[c][r] *= s_decay;
                sk_partial += st[c][r] * kr[r];
            }
            float sk = qwen35_warp_sum(sk_partial);
            float dj = 0.0f;
            if (lane == 0) {
                float v = token_conv[2 * key_dim + (long)h * 128 + col];
                dj = (v - sk) * s_beta;
            }
            dj = __shfl_sync(0xffffffffu, dj, 0);
            float o_partial = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                st[c][r] += kr[r] * dj;
                o_partial += st[c][r] * qr[r];
            }
            float o = qwen35_warp_sum(o_partial);
            if (lane == 0)
                raw_out[(long)token * value_dim + (long)h * 128 + col] = o;
        }
        // No warp may replace the shared Q/K tile while another warp still
        // consumes it for the current token.
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = col0 + c;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = lane + 32 * r;
            state_t[((long)h * 128 + col) * 128 + row] = st[c][r];
        }
    }
}

// Complete the register recurrence with Bonsai's gated RMSNorm and write the
// Q8 activation consumed by the Q1 ssm_out projection. Fusing quantization here
// avoids materializing the gated f32 mix and removes the following generic
// quantize launch entirely.
extern "C" __global__ void qwen35_ssm_rmsnorm_gate_q8_d128_batched(
    const float* __restrict__ raw, const float* __restrict__ z,
    const float* __restrict__ ssm_norm, signed char* __restrict__ quants,
    float* __restrict__ scales, int nv, int value_dim, int k_tokens, float eps
) {
    int token_head = blockIdx.x;
    int token = token_head / nv;
    int h = token_head - token * nv;
    if (token >= k_tokens) return;
    int j = threadIdx.x;
    long base = (long)token * value_dim + (long)h * 128;
    __shared__ float values[128];
    __shared__ float s_scale;
    __shared__ float warp_sums[4];
    float raw_j = raw[base + j];
    values[j] = raw_j;
    float ss = raw_j * raw_j;
    ss = qwen35_warp_sum(ss);
    int lane = j & 31;
    int warp = j >> 5;
    if (lane == 0) warp_sums[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < 4 ? warp_sums[lane] : 0.0f;
        sum = qwen35_warp_sum(sum);
        if (lane == 0)
            s_scale = 1.0f / sqrtf(sum * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    float zj = z[base + j];
    float silu_z = zj / (1.0f + expf(-zj));
    values[j] = values[j] * s_scale * ssm_norm[j] * silu_z;
    __syncthreads();

    {
        int qb = warp;
        float v = values[qb * 32 + lane];
        float max_abs = fabsf(v);
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 16));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 8));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 4));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 2));
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, 1));
        max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
        float unrounded = max_abs / 127.0f;
        if (lane == 0) scales[(base >> 5) + qb] = f16_round(unrounded);
        float inv = (unrounded == 0.0f) ? 0.0f : 1.0f / unrounded;
        float qv = rintf(v * inv);
        if (qv > 127.0f) qv = 127.0f;
        if (qv < -128.0f) qv = -128.0f;
        quants[base + qb * 32 + lane] = (signed char)qv;
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
    /// POPC decode kernels are promoted only on the benchmarked Ampere SM86 lane.
    pub(crate) sm86: bool,
    /// Immutable fast/strict arithmetic contract for this kernel/engine instance.
    /// Read once at construction so a later model reload can choose independently.
    pub(crate) fast_q1: bool,
    /// Immutable Q1 projection layout contract for this kernel/engine instance.
    /// Standalone kernel users default to raw GGUF; resident model construction
    /// selects this before any weight upload.
    pub(crate) q1_tiled: bool,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) rms_norm_per_head: CudaFunction,
    pub(crate) quantize: CudaFunction,
    pub(crate) rms_norm_quantize: CudaFunction,
    pub(crate) rms_inv_norm_quantize_q8_0: CudaFunction,
    pub(crate) gemv: CudaFunction,
    pub(crate) prism_low_bit_f32_gemv: CudaFunction,
    pub(crate) prism_q1_q8_gemv: CudaFunction,
    pub(crate) prism_q1t128_q8_gemv: CudaFunction,
    pub(crate) prism_q8_32_bitplanes_qsum: CudaFunction,
    pub(crate) prism_q1t128_q8_popc_gemv_m16: CudaFunction,
    pub(crate) prism_q1t128_q8_popc_fused_ffn_bonsai27b: CudaFunction,
    pub(crate) prism_q1t128_q8_popc_fused_full_bonsai27b: CudaFunction,
    pub(crate) prism_q1t128_q8_popc_fused_ssm_bonsai27b: CudaFunction,
    pub(crate) prism_q1t128_fused_full_bonsai27b: CudaFunction,
    pub(crate) prism_q1t128_fused_ssm_bonsai27b: CudaFunction,
    pub(crate) prism_q1t128_fused_ffn_bonsai27b: CudaFunction,
    pub(crate) prism_q1_f32_gemm_batched: CudaFunction,
    pub(crate) prism_q1_q8_gemm_batched: CudaFunction,
    pub(crate) prism_q1_q8_wmma_gemm_batched: Option<CudaFunction>,
    /// Optional SM80+ Q8/128 bit-slice packer and binary-MMA prompt lane.
    /// Production dispatch uses them only after the strict/env/shape gates pass.
    pub(crate) prism_q8_b128_bitpack: Option<CudaFunction>,
    pub(crate) prism_q1_q8_b128_bmma_gemm_batched: Option<CudaFunction>,
    pub(crate) q4_0_gemv: CudaFunction,
    /// Quants-first SoA variant of `q4_0_gemv`, bit-identical and used for every
    /// gemma4 dense/attention/FFN projection (see `q4_0_wire_to_soa`).
    pub(crate) q4_0_gemv_soa: CudaFunction,
    /// Native-f32 activation lane for the full-Q4 Gemma 4 MTP assistant.
    pub(crate) q4_0_f32_gemv_soa: CudaFunction,
    pub(crate) q4_1_gemv: CudaFunction,
    pub(crate) q4_0_gemm_batched: CudaFunction,
    /// Shared-scratch K-row twin of `q4_0_gemv_soa`; the measured default.
    pub(crate) q4_0_gemm_batched_soa_shared: CudaFunction,
    /// Zero-shared K-row twin retained behind an exact A/B policy gate.
    pub(crate) q4_0_gemm_batched_soa: CudaFunction,
    /// Optional Ampere signed-int8 tensor-core twin. Production dispatch remains
    /// behind the strict SM86-only environment gate below.
    pub(crate) q4_0_gemm_batched_soa_imma: Option<CudaFunction>,
    pub(crate) q4_1_gemm_batched: CudaFunction,
    pub(crate) nvfp4_gemv: CudaFunction,
    pub(crate) q4k_gemv: CudaFunction,
    pub(crate) q5k_gemv: CudaFunction,
    pub(crate) q6k_gemv: CudaFunction,
    pub(crate) q2k_gemv: CudaFunction,
    pub(crate) q3k_gemv: CudaFunction,
    pub(crate) iq4xs_gemv: CudaFunction,
    pub(crate) q4k_gemm_batched: CudaFunction,
    pub(crate) q6k_gemm_batched: CudaFunction,
    pub(crate) q6k_gemm_batched_anchor_dp4a: CudaFunction,
    pub(crate) quantize_q8k: CudaFunction,
    pub(crate) rms_norm_quantize_q8k: CudaFunction,
    pub(crate) rms_inv_norm_quantize_q8k: CudaFunction,
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
    pub(crate) q4_0_gemv_routed: CudaFunction,
    /// R-rows-per-warp variant of `q4_0_gemv_routed`; bitwise-identical, opt-in.
    pub(crate) q4_0_gemv_routed_rows: CudaFunction,
    /// Prefill counterpart of `q4_0_gemv_routed`: one expert against its CSR token list.
    pub(crate) q4_0_gemm_routed: CudaFunction,
    /// 32-block low-shared A/B twin; never selected unless the strict opt-in is `1`.
    pub(crate) q4_0_gemm_routed_chunked: CudaFunction,
    pub(crate) q4_1_gemm_routed: CudaFunction,
    pub(crate) q4_1_gemm_routed_chunked: CudaFunction,
    pub(crate) q4_1_gemv_routed: CudaFunction,
    pub(crate) q2k_gemv_routed: CudaFunction,
    pub(crate) geglu_quantize_routed: CudaFunction,
    pub(crate) moe_weighted_sum_routed: CudaFunction,
    /// Strict [token][router-rank] fold over expert-major CSR assignment rows.
    pub(crate) moe_weighted_sum_batched: CudaFunction,
    pub(crate) argmax: CudaFunction,
    pub(crate) sample_gumbel: CudaFunction,
    pub(crate) gemm_batched: CudaFunction,
    pub(crate) rms_norm_batched: CudaFunction,
    pub(crate) prism_rms_norm_q8_batched: CudaFunction,
    pub(crate) prism_silu_mul_q8_batched: CudaFunction,
    pub(crate) rope_batched: CudaFunction,
    pub(crate) kv_scatter_batched: CudaFunction,
    pub(crate) attention_batched: CudaFunction,
    pub(crate) attention_sw_batched: CudaFunction,
    pub(crate) kv_scatter_tree_batched: CudaFunction,
    pub(crate) attention_tree_batched: CudaFunction,
    pub(crate) kv_scatter_q8_0: CudaFunction,
    pub(crate) attention_q8_0: CudaFunction,
    pub(crate) attention_sw_q8_0: CudaFunction,
    pub(crate) kv_scatter_batched_q8_0: CudaFunction,
    pub(crate) kv_scatter_tree_batched_q8_0: CudaFunction,
    pub(crate) attention_batched_q8_0: CudaFunction,
    pub(crate) attention_tree_batched_q8_0: CudaFunction,
    pub(crate) attn_sk_scores_q8_0: CudaFunction,
    pub(crate) attn_sk_partial_q8_0: CudaFunction,
    pub(crate) argmax_batched: CudaFunction,
    pub(crate) attn_sk_scores: CudaFunction,
    pub(crate) attn_sk_scores_coalesced: CudaFunction,
    pub(crate) attn_sk_partial: CudaFunction,
    pub(crate) attn_sk_combine: CudaFunction,
    pub(crate) flash_attention_prefill_tiled_d64: CudaFunction,
    pub(crate) flash_attention_prefill_tiled_d128: CudaFunction,
    pub(crate) flash_attention_prefill_tiled_d256: CudaFunction,
    pub(crate) flash_attention_prefill_tiled_dyn: CudaFunction,
    // qwen35 (Ornith) gated-delta-net SSM kernels.
    pub(crate) ssm_l2_norm_per_head: CudaFunction,
    pub(crate) ssm_l2_norm_per_head_batched: CudaFunction,
    pub(crate) qwen35_ssm_qk_l2_norm_d128_batched: CudaFunction,
    pub(crate) ssm_conv1d: CudaFunction,
    pub(crate) ssm_conv1d_batched: CudaFunction,
    pub(crate) qwen35_ssm_conv1d_d4_batched: CudaFunction,
    pub(crate) ssm_delta_rule: CudaFunction,
    pub(crate) ssm_delta_rule_batched: CudaFunction,
    pub(crate) qwen35_ssm_delta_rule_d128_batched: CudaFunction,
    pub(crate) qwen35_ssm_rmsnorm_gate_q8_d128_batched: CudaFunction,
    pub(crate) sigmoid_mul: CudaFunction,
    pub(crate) embed_gather_q4k: CudaFunction,
    pub(crate) embed_gather_q6k: CudaFunction,
    pub(crate) embed_gather_q3k: CudaFunction,
    pub(crate) embed_gather_q8_0: CudaFunction,
    pub(crate) embed_gather_q1_0: CudaFunction,
    pub(crate) embed_gather_q2_0_g64: CudaFunction,
    pub(crate) embed_gather_q2_0_g128: CudaFunction,
    pub(crate) rope_select: CudaFunction,
    pub(crate) ssm_gates: CudaFunction,
    pub(crate) ssm_gates_batched: CudaFunction,
    pub(crate) deinterleave_qgate: CudaFunction,
    pub(crate) deinterleave_qgate_batched: CudaFunction,
    /// Process-stable strict opt-in for the zero-shared dense Q4_0 A/B.
    pub(crate) gemma4_mtp_dense_q4_zero_shared: bool,
    /// Process-stable strict opt-in for the exact SM86 IMMA dense Q4_0 A/B.
    pub(crate) gemma4_mtp_dense_q4_imma: bool,
    /// Process-stable strict opt-in for the exact anchor-major Q6_K DP4A A/B.
    pub(crate) gemma4_mtp_dense_q6_anchor_dp4a: bool,
    /// Process-stable strict opt-in for the low-shared routed Q4 A/B.
    pub(crate) gemma4_mtp_routed_q4_chunked: bool,
    /// Env-gated (CAMELID_ATTN_COALESCED) dispatch of the coalesced K-dot in
    /// split-K pass 1. Read ONCE at construction; default OFF so the shipped
    /// path stays byte-identical.
    pub(crate) attn_coalesced: bool,
}

impl CudaResidentKernels {
    pub fn new() -> Result<Self, String> {
        Self::new_with_q1_policy(false, prism_cuda_fast_from_env())
    }

    fn new_with_q1_policy(q1_tiled: bool, fast_q1: bool) -> Result<Self, String> {
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
        // The resident engine owns this context's stream ordering. On Windows the
        // dedicated stream otherwise flips cudarc into automatic multi-stream mode,
        // making every safe kernel argument enqueue redundant wait/record events on
        // the SAME stream. Besides multiplying WDDM submission work, those external
        // event dependencies make stream capture fail with CAPTURE_ISOLATION. No
        // resident allocation exists yet, so disabling here also avoids allocating
        // per-slice tracking events. Offload and overlap use explicit CUDA events.
        // CAMELID_CUDA_SAFE_EVENTS=1 is the diagnostic escape hatch.
        if cuda_manual_stream_order_enabled() {
            // SAFETY: all same-stream dependencies are ordered by `stream`; the only
            // side streams (offload / overlap) are explicitly event-joined below.
            unsafe { ctx.disable_event_tracking() };
        }
        let cc_major = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        // Compile the resident module for the selected device generation. Int8
        // WMMA needs sm_75+; older CUDA devices retain the portable compute_61
        // lane and simply do not load the optional tensor-core function.
        let arch = if cc_major >= 8 {
            "compute_80"
        } else if cc_major == 7 && cc_minor >= 5 {
            "compute_75"
        } else {
            "compute_61"
        };
        let cuda_include = ["CUDA_PATH", "CUDA_HOME"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(std::path::PathBuf::from)
            .map(|root| root.join("include"))
            .chain([std::path::PathBuf::from("/usr/local/cuda/include")])
            .find(|include| include.join("mma.h").is_file());
        let tensor_core_q1 =
            (cc_major > 7 || (cc_major == 7 && cc_minor >= 5)) && cuda_include.is_some();
        let binary_tensor_core_q1 = cc_major >= 8;
        let mut include_paths = Vec::new();
        let mut options = Vec::new();
        if let Some(include) = cuda_include {
            include_paths.push(include.to_string_lossy().into_owned());
            options.push("-DCAMELID_HAS_WMMA=1".to_string());
        }
        let opts = CompileOptions {
            fmad: Some(false),
            arch: Some(arch),
            include_paths,
            options,
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
        let sm86 = cc_major == 8 && cc_minor == 6;
        let gemma4_mtp_dense_q4_zero_shared = gemma4_mtp_dense_q4_zero_shared_from_env()?;
        let gemma4_mtp_dense_q4_imma = gemma4_mtp_dense_q4_imma_from_env()?;
        let gemma4_mtp_dense_q6_anchor_dp4a = gemma4_mtp_dense_q6_anchor_dp4a_from_env()?;
        let gemma4_mtp_routed_q4_chunked = gemma4_mtp_routed_q4_chunked_from_env()?;
        if gemma4_mtp_dense_q4_imma && !sm86 {
            return Err(format!(
                "{GEMMA4_MTP_DENSE_Q4_IMMA_ENV}=1 requires an Ampere SM86 CUDA device; selected device is SM{cc_major}{cc_minor}"
            ));
        }
        if gemma4_mtp_dense_q4_imma && gemma4_mtp_dense_q4_zero_shared {
            return Err(format!(
                "{GEMMA4_MTP_DENSE_Q4_IMMA_ENV}=1 and {GEMMA4_MTP_DENSE_Q4_ZERO_SHARED_ENV}=1 are mutually exclusive A/B lanes"
            ));
        }
        if gemma4_mtp_dense_q4_imma && !crate::gemma4_runtime::gemma4_q4_0_soa_enabled() {
            return Err(format!(
                "{GEMMA4_MTP_DENSE_Q4_IMMA_ENV}=1 requires CAMELID_GEMMA4_Q4_0_SOA to remain enabled"
            ));
        }
        if gemma4_mtp_dense_q4_zero_shared {
            eprintln!(
                "[gemma4-mtp-cuda] dense_q4_kernel=soa-zero-shared \
                 env=CAMELID_GEMMA4_MTP_DENSE_Q4_ZERO_SHARED"
            );
        }
        if gemma4_mtp_dense_q4_imma {
            eprintln!(
                "[gemma4-mtp-cuda] dense_q4_kernel=soa-imma-m16n8k32 \
                 env=CAMELID_GEMMA4_MTP_DENSE_Q4_IMMA"
            );
        }
        if gemma4_mtp_dense_q6_anchor_dp4a {
            eprintln!(
                "[gemma4-mtp-cuda] dense_q6_kernel=anchor-dp4a \
                 env=CAMELID_GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A"
            );
        }
        if gemma4_mtp_routed_q4_chunked {
            eprintln!(
                "[gemma4-mtp-cuda] routed_q4_kernel=chunked32 \
                 env=CAMELID_GEMMA4_MTP_ROUTED_Q4_CHUNKED"
            );
        }
        Ok(Self {
            sm86,
            fast_q1,
            q1_tiled,
            rms_norm: f("rms_norm_f32")?,
            rms_norm_per_head: f("rms_norm_per_head_f32")?,
            quantize: f("quantize_q8_0")?,
            rms_norm_quantize: f("rms_norm_quantize")?,
            rms_inv_norm_quantize_q8_0: f("rms_inv_norm_quantize_q8_0")?,
            gemv: f("q8_gemv")?,
            prism_low_bit_f32_gemv: f("prism_low_bit_f32_gemv")?,
            prism_q1_q8_gemv: f("prism_q1_q8_gemv")?,
            prism_q1t128_q8_gemv: f("prism_q1t128_q8_gemv")?,
            prism_q8_32_bitplanes_qsum: f("prism_q8_32_bitplanes_qsum")?,
            prism_q1t128_q8_popc_gemv_m16: f("prism_q1t128_q8_popc_gemv_m16")?,
            prism_q1t128_q8_popc_fused_ffn_bonsai27b: f(
                "prism_q1t128_q8_popc_fused_ffn_bonsai27b",
            )?,
            prism_q1t128_q8_popc_fused_full_bonsai27b: f(
                "prism_q1t128_q8_popc_fused_full_bonsai27b",
            )?,
            prism_q1t128_q8_popc_fused_ssm_bonsai27b: f(
                "prism_q1t128_q8_popc_fused_ssm_bonsai27b",
            )?,
            prism_q1t128_fused_full_bonsai27b: f("prism_q1t128_fused_full_bonsai27b")?,
            prism_q1t128_fused_ssm_bonsai27b: f("prism_q1t128_fused_ssm_bonsai27b")?,
            prism_q1t128_fused_ffn_bonsai27b: f("prism_q1t128_fused_ffn_bonsai27b")?,
            prism_q1_f32_gemm_batched: f("prism_q1_f32_gemm_batched")?,
            prism_q1_q8_gemm_batched: f("prism_q1_q8_gemm_batched")?,
            prism_q1_q8_wmma_gemm_batched: tensor_core_q1
                .then(|| f("prism_q1_q8_wmma_gemm_batched"))
                .transpose()?,
            prism_q8_b128_bitpack: binary_tensor_core_q1
                .then(|| f("prism_q8_b128_bitpack"))
                .transpose()?,
            prism_q1_q8_b128_bmma_gemm_batched: binary_tensor_core_q1
                .then(|| f("prism_q1_q8_b128_bmma_gemm_batched"))
                .transpose()?,
            q4_0_gemv: f("q4_0_gemv")?,
            q4_0_gemv_soa: f("q4_0_gemv_soa")?,
            q4_0_f32_gemv_soa: f("q4_0_f32_gemv_soa")?,
            q4_1_gemv: f("q4_1_gemv")?,
            q4_0_gemm_batched: f("q4_0_gemm_batched")?,
            q4_0_gemm_batched_soa_shared: f("q4_0_gemm_batched_soa_shared")?,
            q4_0_gemm_batched_soa: f("q4_0_gemm_batched_soa")?,
            q4_0_gemm_batched_soa_imma: (cc_major >= 8)
                .then(|| f("q4_0_gemm_batched_soa_imma"))
                .transpose()?,
            q4_1_gemm_batched: f("q4_1_gemm_batched")?,
            nvfp4_gemv: f("nvfp4_gemv")?,
            q4k_gemv: f("q4k_gemv")?,
            q5k_gemv: f("q5k_gemv")?,
            q6k_gemv: f("q6k_gemv")?,
            q2k_gemv: f("q2k_gemv")?,
            iq4xs_gemv: f("iq4xs_gemv")?,
            q3k_gemv: f("q3k_gemv")?,
            q4k_gemm_batched: f("q4k_gemm_batched")?,
            q6k_gemm_batched: f("q6k_gemm_batched")?,
            q6k_gemm_batched_anchor_dp4a: f("q6k_gemm_batched_anchor_dp4a")?,
            quantize_q8k: f("quantize_q8k")?,
            rms_norm_quantize_q8k: f("rms_norm_quantize_q8k")?,
            rms_inv_norm_quantize_q8k: f("rms_inv_norm_quantize_q8k")?,
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
            q4_0_gemv_routed: f("q4_0_gemv_routed")?,
            q4_0_gemv_routed_rows: f("q4_0_gemv_routed_rows")?,
            q4_0_gemm_routed: f("q4_0_gemm_routed")?,
            q4_0_gemm_routed_chunked: f("q4_0_gemm_routed_chunked")?,
            q4_1_gemm_routed: f("q4_1_gemm_routed")?,
            q4_1_gemm_routed_chunked: f("q4_1_gemm_routed_chunked")?,
            q4_1_gemv_routed: f("q4_1_gemv_routed")?,
            q2k_gemv_routed: f("q2k_gemv_routed")?,
            geglu_quantize_routed: f("geglu_quantize_routed")?,
            moe_weighted_sum_routed: f("moe_weighted_sum_routed")?,
            moe_weighted_sum_batched: f("moe_weighted_sum_batched")?,
            argmax: f("argmax_f32")?,
            sample_gumbel: f("sample_gumbel")?,
            gemm_batched: f("q8_gemm_batched")?,
            rms_norm_batched: f("rms_norm_batched")?,
            prism_rms_norm_q8_batched: f("prism_rms_norm_q8_batched")?,
            prism_silu_mul_q8_batched: f("prism_silu_mul_q8_batched")?,
            rope_batched: f("rope_batched")?,
            kv_scatter_batched: f("kv_scatter_batched")?,
            attention_batched: f("attention_batched")?,
            attention_sw_batched: f("attention_sw_batched")?,
            kv_scatter_tree_batched: f("kv_scatter_tree_batched")?,
            attention_tree_batched: f("attention_tree_batched")?,
            kv_scatter_q8_0: f("kv_scatter_q8_0")?,
            attention_q8_0: f("attention_decode_q8_0")?,
            attention_sw_q8_0: f("attention_decode_sw_q8_0")?,
            kv_scatter_batched_q8_0: f("kv_scatter_batched_q8_0")?,
            kv_scatter_tree_batched_q8_0: f("kv_scatter_tree_batched_q8_0")?,
            attention_batched_q8_0: f("attention_batched_q8_0")?,
            attention_tree_batched_q8_0: f("attention_tree_batched_q8_0")?,
            attn_sk_scores_q8_0: f("attn_sk_scores_q8_0")?,
            attn_sk_partial_q8_0: f("attn_sk_partial_q8_0")?,
            argmax_batched: f("argmax_batched")?,
            attn_sk_scores: f("attn_sk_scores")?,
            attn_sk_scores_coalesced: f("attn_sk_scores_coalesced")?,
            attn_sk_partial: f("attn_sk_partial")?,
            attn_sk_combine: f("attn_sk_combine")?,
            flash_attention_prefill_tiled_d64: f("flash_attention_prefill_tiled_d64")?,
            flash_attention_prefill_tiled_d128: f("flash_attention_prefill_tiled_d128")?,
            flash_attention_prefill_tiled_d256: f("flash_attention_prefill_tiled_d256")?,
            flash_attention_prefill_tiled_dyn: f("flash_attention_prefill_tiled_dyn")?,
            ssm_l2_norm_per_head: f("ssm_l2_norm_per_head")?,
            ssm_l2_norm_per_head_batched: f("ssm_l2_norm_per_head_batched")?,
            qwen35_ssm_qk_l2_norm_d128_batched: f("qwen35_ssm_qk_l2_norm_d128_batched")?,
            ssm_conv1d: f("ssm_conv1d")?,
            ssm_conv1d_batched: f("ssm_conv1d_batched")?,
            qwen35_ssm_conv1d_d4_batched: f("qwen35_ssm_conv1d_d4_batched")?,
            ssm_delta_rule: f("ssm_delta_rule")?,
            ssm_delta_rule_batched: f("ssm_delta_rule_batched")?,
            qwen35_ssm_delta_rule_d128_batched: f("qwen35_ssm_delta_rule_d128_batched")?,
            qwen35_ssm_rmsnorm_gate_q8_d128_batched: f("qwen35_ssm_rmsnorm_gate_q8_d128_batched")?,
            sigmoid_mul: f("sigmoid_mul")?,
            embed_gather_q4k: f("embed_gather_q4k")?,
            embed_gather_q6k: f("embed_gather_q6k")?,
            embed_gather_q3k: f("embed_gather_q3k")?,
            embed_gather_q8_0: f("embed_gather_q8_0")?,
            embed_gather_q1_0: f("embed_gather_q1_0")?,
            embed_gather_q2_0_g64: f("embed_gather_q2_0_g64")?,
            embed_gather_q2_0_g128: f("embed_gather_q2_0_g128")?,
            rope_select: f("rope_select")?,
            ssm_gates: f("ssm_gates")?,
            ssm_gates_batched: f("ssm_gates_batched")?,
            deinterleave_qgate: f("deinterleave_qgate")?,
            deinterleave_qgate_batched: f("deinterleave_qgate_batched")?,
            gemma4_mtp_dense_q4_zero_shared,
            gemma4_mtp_dense_q4_imma,
            gemma4_mtp_dense_q6_anchor_dp4a,
            gemma4_mtp_routed_q4_chunked,
            attn_coalesced: std::env::var("CAMELID_ATTN_COALESCED")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            ctx,
            stream,
        })
    }

    pub(crate) fn gemma4_mtp_q4_0_gemm_routed_kernel(&self) -> (&CudaFunction, bool) {
        if self.gemma4_mtp_routed_q4_chunked {
            (&self.q4_0_gemm_routed_chunked, true)
        } else {
            (&self.q4_0_gemm_routed, false)
        }
    }

    pub(crate) fn gemma4_mtp_dense_q4_0_gemm_kernel(&self) -> (&CudaFunction, bool) {
        if self.gemma4_mtp_dense_q4_zero_shared {
            (&self.q4_0_gemm_batched_soa, true)
        } else {
            (&self.q4_0_gemm_batched_soa_shared, false)
        }
    }

    pub(crate) fn gemma4_mtp_dense_q4_0_imma_kernel(&self) -> Option<&CudaFunction> {
        self.gemma4_mtp_dense_q4_imma.then(|| {
            self.q4_0_gemm_batched_soa_imma
                .as_ref()
                .expect("SM86 IMMA gate validated kernel availability")
        })
    }

    pub(crate) fn gemma4_mtp_dense_q6_gemm_kernel(&self) -> (&CudaFunction, usize) {
        if self.gemma4_mtp_dense_q6_anchor_dp4a {
            (&self.q6k_gemm_batched_anchor_dp4a, 0)
        } else {
            (&self.q6k_gemm_batched, 8)
        }
    }

    pub(crate) fn gemma4_mtp_q4_1_gemm_routed_kernel(&self) -> (&CudaFunction, bool) {
        if self.gemma4_mtp_routed_q4_chunked {
            (&self.q4_1_gemm_routed_chunked, true)
        } else {
            (&self.q4_1_gemm_routed, false)
        }
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
/// Q4_0 wire (18 B/block: `[2 B f16 scale][16 B nibbles]`) into the quants-first
/// SoA layout `q4_0_gemv_soa` reads: `[n*16 nibble bytes][n*2 f16 scale bits]`.
///
/// Same total size — 18 bytes per block either way — so VRAM residency, every
/// slot budget and every DMA byte count are unchanged. The point is alignment:
/// block b's nibbles now begin at a 16-byte boundary, so the GEMV issues one
/// `uint4` load instead of assembling the identical 16 bytes from 16 scalar byte
/// loads off an 18-byte stride.
///
/// Pure byte permutation, bit-identical by construction — see the kernel comment
/// on `q4_0_gemv_soa` for the four-step argument, and
/// `q4_0_gemv_soa_matches_wire` for the test that pins it. Mirrors the same
/// contract as `repack_q8_soa`, `swz_q4k_blocks` and `repack_q1_t128`.
pub(crate) fn q4_0_wire_to_soa(bytes: &[u8]) -> Vec<u8> {
    const WIRE: usize = 18;
    let n = bytes.len() / WIRE;
    let mut out = vec![0u8; n * WIRE];
    let (quants, scales) = out.split_at_mut(n * 16);
    for b in 0..n {
        let blk = &bytes[b * WIRE..b * WIRE + WIRE];
        scales[b * 2..b * 2 + 2].copy_from_slice(&blk[0..2]);
        quants[b * 16..b * 16 + 16].copy_from_slice(&blk[2..WIRE]);
    }
    out
}

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

/// Tile Q1_0 into a same-size row-supertile layout. For every <=128-row tile and
/// K=128 block the bytes are `[nr * 16 signs][nr * 2 scales]`. Full tiles make
/// every eight-row decode sign slab one aligned/coalesced 128-byte transaction;
/// tail tiles retain only their live rows, so the tensor stays exactly 18 bytes
/// per logical row/block with no padding or duplicate VRAM copy.
pub(crate) fn repack_q1_t128(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if cols == 0 || !cols.is_multiple_of(128) {
        return Err(format!(
            "Q1T128 cols {cols} must be a non-zero multiple of 128"
        ));
    }
    let blocks_per_row = cols / 128;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(18))
        .ok_or_else(|| "Q1T128 shape byte count overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "Q1T128 wire length {} != rows {rows} * blocks {blocks_per_row} * 18 = {expected}",
            bytes.len()
        ));
    }
    let mut out = vec![0u8; expected];
    let mut dst_group = 0usize;
    for row0 in (0..rows).step_by(128) {
        let nr = (rows - row0).min(128);
        for block in 0..blocks_per_row {
            let signs = dst_group;
            let scales = signs + nr * 16;
            for row_in_tile in 0..nr {
                let src = ((row0 + row_in_tile) * blocks_per_row + block) * 18;
                out[signs + row_in_tile * 16..signs + (row_in_tile + 1) * 16]
                    .copy_from_slice(&bytes[src + 2..src + 18]);
                out[scales + row_in_tile * 2..scales + (row_in_tile + 1) * 2]
                    .copy_from_slice(&bytes[src..src + 2]);
            }
            dst_group += nr * 18;
        }
    }
    debug_assert_eq!(dst_group, expected);
    Ok(out)
}

#[cfg(test)]
pub(crate) fn unpack_q1_t128(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if cols == 0 || !cols.is_multiple_of(128) {
        return Err(format!(
            "Q1T128 cols {cols} must be a non-zero multiple of 128"
        ));
    }
    let blocks_per_row = cols / 128;
    let expected = rows * blocks_per_row * 18;
    if bytes.len() != expected {
        return Err(format!("Q1T128 tiled length {} != {expected}", bytes.len()));
    }
    let mut out = vec![0u8; expected];
    let mut src_group = 0usize;
    for row0 in (0..rows).step_by(128) {
        let nr = (rows - row0).min(128);
        for block in 0..blocks_per_row {
            let signs = src_group;
            let scales = signs + nr * 16;
            for row_in_tile in 0..nr {
                let dst = ((row0 + row_in_tile) * blocks_per_row + block) * 18;
                out[dst..dst + 2].copy_from_slice(
                    &bytes[scales + row_in_tile * 2..scales + (row_in_tile + 1) * 2],
                );
                out[dst + 2..dst + 18].copy_from_slice(
                    &bytes[signs + row_in_tile * 16..signs + (row_in_tile + 1) * 16],
                );
            }
            src_group += nr * 18;
        }
    }
    Ok(out)
}

/// Repack one projection's wire bytes into the GPU layout its lane reads. Q8_0 is
/// repacked to the SoA layout `q8_gemv` reads; the K-quant lanes pass the RAW GGUF
/// super-block wire bytes straight through — `q4k_gemv` (144 B/sb) and `q6k_gemv`
/// (210 B/sb) expand the nibbles / unpack the packed scales on the fly. Keeping the
/// nibbles PACKED in VRAM is what lets 8B-Q4_K_M fit a 6 GB card: a host-side nibble
/// expansion to i8 would near-double the Q4_K footprint (256 vs 128 bytes/sb).
fn repack_for_lane(
    bytes: &[u8],
    q: ProjQuant,
    rows: usize,
    cols: usize,
    q1_tiled: bool,
) -> Result<Vec<u8>, String> {
    let repacked = match q {
        ProjQuant::Q8_0 => repack_q8_soa(bytes),
        ProjQuant::Q6K => pad_q6k_blocks(bytes),
        ProjQuant::Q4K => swz_q4k_blocks(bytes),
        ProjQuant::Q5K | ProjQuant::Q2K | ProjQuant::Q3K => bytes.to_vec(),
        ProjQuant::Q1_0 if q1_tiled => {
            return repack_q1_t128(bytes, rows, cols);
        }
        // IQ4_XS is read straight from the 136-byte GGUF wire (raw passthrough, like
        // Q5_K/Q2_K/Q3_K): the kernel unpacks nibbles/codebook on the fly.
        ProjQuant::IQ4XS | ProjQuant::Q1_0 | ProjQuant::Q2_0G64 | ProjQuant::Q2_0G128 => {
            bytes.to_vec()
        }
    };
    Ok(repacked)
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

const GEMMA4_EXPERT_Q8_DEFAULT_WARPS: u32 = 2;

fn parse_gemma4_expert_q8_warps(value: Option<&str>) -> u32 {
    value
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .filter(|warps| matches!(*warps, 1 | 2 | 4 | 8))
        .unwrap_or(GEMMA4_EXPERT_Q8_DEFAULT_WARPS)
}

/// Process-stable launch policy so a receipt cannot change geometry halfway
/// through generation. Only whole power-of-two warp counts up to the former
/// 256-thread shape are valid; malformed/unsupported values retain that default.
fn gemma4_expert_q8_warps() -> u32 {
    static WARPS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WARPS.get_or_init(|| {
        let value = std::env::var("CAMELID_GEMMA4_CUDA_EXPERT_Q8_WARPS").ok();
        parse_gemma4_expert_q8_warps(value.as_deref())
    })
}

const GEMMA4_MTP_DENSE_Q4_ZERO_SHARED_ENV: &str = "CAMELID_GEMMA4_MTP_DENSE_Q4_ZERO_SHARED";

fn parse_gemma4_mtp_dense_q4_zero_shared(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q4_ZERO_SHARED_ENV} must be exactly 0 or 1, got {other:?}"
        )),
    }
}

fn gemma4_mtp_dense_q4_zero_shared_from_env() -> Result<bool, String> {
    match std::env::var(GEMMA4_MTP_DENSE_Q4_ZERO_SHARED_ENV) {
        Ok(value) => parse_gemma4_mtp_dense_q4_zero_shared(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gemma4_mtp_dense_q4_zero_shared(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q4_ZERO_SHARED_ENV} is not valid Unicode"
        )),
    }
}

const GEMMA4_MTP_DENSE_Q4_IMMA_ENV: &str = "CAMELID_GEMMA4_MTP_DENSE_Q4_IMMA";

fn parse_gemma4_mtp_dense_q4_imma(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q4_IMMA_ENV} must be exactly 0 or 1, got {other:?}"
        )),
    }
}

fn gemma4_mtp_dense_q4_imma_from_env() -> Result<bool, String> {
    match std::env::var(GEMMA4_MTP_DENSE_Q4_IMMA_ENV) {
        Ok(value) => parse_gemma4_mtp_dense_q4_imma(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gemma4_mtp_dense_q4_imma(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q4_IMMA_ENV} is not valid Unicode"
        )),
    }
}

const GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A_ENV: &str = "CAMELID_GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A";

fn parse_gemma4_mtp_dense_q6_anchor_dp4a(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A_ENV} must be exactly 0 or 1, got {other:?}"
        )),
    }
}

fn gemma4_mtp_dense_q6_anchor_dp4a_from_env() -> Result<bool, String> {
    match std::env::var(GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A_ENV) {
        Ok(value) => parse_gemma4_mtp_dense_q6_anchor_dp4a(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gemma4_mtp_dense_q6_anchor_dp4a(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{GEMMA4_MTP_DENSE_Q6_ANCHOR_DP4A_ENV} is not valid Unicode"
        )),
    }
}

const GEMMA4_MTP_ROUTED_Q4_CHUNKED_ENV: &str = "CAMELID_GEMMA4_MTP_ROUTED_Q4_CHUNKED";

fn parse_gemma4_mtp_routed_q4_chunked(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{GEMMA4_MTP_ROUTED_Q4_CHUNKED_ENV} must be exactly 0 or 1, got {other:?}"
        )),
    }
}

fn gemma4_mtp_routed_q4_chunked_from_env() -> Result<bool, String> {
    match std::env::var(GEMMA4_MTP_ROUTED_Q4_CHUNKED_ENV) {
        Ok(value) => parse_gemma4_mtp_routed_q4_chunked(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gemma4_mtp_routed_q4_chunked(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{GEMMA4_MTP_ROUTED_Q4_CHUNKED_ENV} is not valid Unicode"
        )),
    }
}

/// Apply a host-computed RMS inverse to a resident row and quantize it to Q8_0
/// with Windows/Rust byte semantics. The caller must compute `rms_inv` through
/// the CPU reference's sequential `(mss + eps).powf(-0.5)` path. Set
/// `CAMELID_GEMMA4_CUDA_EXPERT_Q8_WARPS` to 1, 2, 4, or 8 before the first launch
/// to A/B the process-cached CTA geometry; invalid values retain the tuned default.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rms_inv_norm_quantize_q8_0(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n: usize,
    rms_inv: f32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(n.is_multiple_of(32));
    let warps_per_cta = gemma4_expert_q8_warps();
    let block = warps_per_cta * 32;
    let n_blocks = (n / 32) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks.div_ceil(warps_per_cta), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x)
        .arg(w)
        .arg(quants)
        .arg(scales)
        .arg(&n_i)
        .arg(&rms_inv);
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

/// Packed Prism Q1_0/Q2_0 GEMV over the original f32 activation. Grid geometry
/// follows the Metal parity kernels, with each warp reusing one activation slice
/// across eight Q1 or Q2 output rows without changing per-row reduction order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_low_bit_f32_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    bits: usize,
    weight_block_elements: usize,
    q1_tiled: bool,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let work_items = rows.div_ceil(8) as u32;
    let cfg = LaunchConfig {
        grid_dim: (work_items.div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: if bits == 1 { 4 * 128 * 4 } else { 0 },
    };
    let (rows, blocks_per_row, bits, elements, q1_tiled) = (
        rows as i32,
        (cols / weight_block_elements) as i32,
        bits as i32,
        weight_block_elements as i32,
        i32::from(q1_tiled),
    );
    let mut b = s.launch_builder(f);
    b.arg(input)
        .arg(weight)
        .arg(&rows)
        .arg(&blocks_per_row)
        .arg(&bits)
        .arg(&elements)
        .arg(&q1_tiled)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Packed Prism Q1_0 by Q8_0 decode GEMV. One warp computes eight output rows
/// while reusing every activation chunk across them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1_q8_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let warps_per_block = block / 32;
    let work_items = rows.div_ceil(8) as u32;
    let cfg = LaunchConfig {
        grid_dim: (work_items.div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows, cols, blocks_per_row) = (rows as i32, cols as i32, (cols / 128) as i32);
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(weight)
        .arg(&rows)
        .arg(&cols)
        .arg(&blocks_per_row)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code)]
pub(crate) fn launch_prism_q8_32_bitplanes_qsum(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    bitplanes: &mut CudaSlice<u32>,
    qsums: &mut CudaSlice<i32>,
    n_chunks: usize,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_quants.len() >= n_chunks * 32);
    debug_assert!(bitplanes.len() >= n_chunks * 8);
    debug_assert!(qsums.len() >= n_chunks);
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_chunks as u32).div_ceil(block / 32), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_chunks = n_chunks as i32;
    let mut b = s.launch_builder(f);
    b.arg(input_quants).arg(bitplanes).arg(qsums).arg(&n_chunks);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_q8_popc_gemv_m16(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_bitplanes: &CudaSlice<u32>,
    input_qsums: &CudaSlice<i32>,
    input_scales: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert_eq!(cols % 128, 0);
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(128), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows, blocks_per_row) = (rows as i32, (cols / 128) as i32);
    let mut b = s.launch_builder(f);
    b.arg(input_bitplanes)
        .arg(input_qsums)
        .arg(input_scales)
        .arg(weight)
        .arg(&rows)
        .arg(&blocks_per_row)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_q8_popc_fused_ffn_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_bitplanes: &CudaSlice<u32>,
    input_qsums: &CudaSlice<i32>,
    input_scales: &CudaSlice<f32>,
    gate_weight: &CudaView<u8>,
    up_weight: &CudaView<u8>,
    gate_out: &mut CudaSlice<f32>,
    up_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_bitplanes.len() >= 5_120 / 4);
    debug_assert!(input_qsums.len() >= 5_120 / 32);
    debug_assert!(input_scales.len() >= 5_120 / 32);
    debug_assert!(gate_weight.len() >= 17_408 * 40 * 18);
    debug_assert!(up_weight.len() >= 17_408 * 40 * 18);
    debug_assert!(gate_out.len() >= 17_408);
    debug_assert!(up_out.len() >= 17_408);
    let cfg = LaunchConfig {
        grid_dim: (17_408u32 / 128, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_bitplanes)
        .arg(input_qsums)
        .arg(input_scales)
        .arg(gate_weight)
        .arg(up_weight)
        .arg(gate_out)
        .arg(up_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_q8_popc_fused_full_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_bitplanes: &CudaSlice<u32>,
    input_qsums: &CudaSlice<i32>,
    input_scales: &CudaSlice<f32>,
    qgate_weight: &CudaView<u8>,
    k_weight: &CudaView<u8>,
    v_weight: &CudaView<u8>,
    qgate_out: &mut CudaSlice<f32>,
    k_out: &mut CudaSlice<f32>,
    v_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_bitplanes.len() >= 5_120 / 4);
    debug_assert!(input_qsums.len() >= 5_120 / 32);
    debug_assert!(input_scales.len() >= 5_120 / 32);
    debug_assert!(qgate_weight.len() >= 12_288 * 40 * 18);
    debug_assert!(k_weight.len() >= 1_024 * 40 * 18);
    debug_assert!(v_weight.len() >= 1_024 * 40 * 18);
    debug_assert!(qgate_out.len() >= 12_288);
    debug_assert!(k_out.len() >= 1_024);
    debug_assert!(v_out.len() >= 1_024);
    let cfg = LaunchConfig {
        grid_dim: (112, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_bitplanes)
        .arg(input_qsums)
        .arg(input_scales)
        .arg(qgate_weight)
        .arg(k_weight)
        .arg(v_weight)
        .arg(qgate_out)
        .arg(k_out)
        .arg(v_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_q8_popc_fused_ssm_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_bitplanes: &CudaSlice<u32>,
    input_qsums: &CudaSlice<i32>,
    input_scales: &CudaSlice<f32>,
    wqkv_weight: &CudaView<u8>,
    z_weight: &CudaView<u8>,
    beta_weight: &CudaView<u8>,
    alpha_weight: &CudaView<u8>,
    wqkv_out: &mut CudaSlice<f32>,
    z_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    alpha_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_bitplanes.len() >= 5_120 / 4);
    debug_assert!(input_qsums.len() >= 5_120 / 32);
    debug_assert!(input_scales.len() >= 5_120 / 32);
    debug_assert!(wqkv_weight.len() >= 10_240 * 40 * 18);
    debug_assert!(z_weight.len() >= 6_144 * 40 * 18);
    debug_assert!(beta_weight.len() >= 48 * 40 * 18);
    debug_assert!(alpha_weight.len() >= 48 * 40 * 18);
    debug_assert!(wqkv_out.len() >= 10_240);
    debug_assert!(z_out.len() >= 6_144);
    debug_assert!(beta_out.len() >= 48);
    debug_assert!(alpha_out.len() >= 48);
    let cfg = LaunchConfig {
        grid_dim: (130, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_bitplanes)
        .arg(input_qsums)
        .arg(input_scales)
        .arg(wqkv_weight)
        .arg(z_weight)
        .arg(beta_weight)
        .arg(alpha_weight)
        .arg(wqkv_out)
        .arg(z_out)
        .arg(beta_out)
        .arg(alpha_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Decode-only projection fusion for the exact Bonsai-27B full-attention
/// geometry: qgate=12,288 rows and K/V=1,024 rows, all with K=5,120 Q1T128.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_fused_full_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    qgate_weight: &CudaView<u8>,
    k_weight: &CudaView<u8>,
    v_weight: &CudaView<u8>,
    qgate_out: &mut CudaSlice<f32>,
    k_out: &mut CudaSlice<f32>,
    v_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_quants.len() >= 5_120);
    debug_assert!(input_scales.len() >= 160);
    debug_assert!(qgate_weight.len() >= 12_288 * 40 * 18);
    debug_assert!(k_weight.len() >= 1_024 * 40 * 18);
    debug_assert!(v_weight.len() >= 1_024 * 40 * 18);
    debug_assert!(qgate_out.len() >= 12_288);
    debug_assert!(k_out.len() >= 1_024);
    debug_assert!(v_out.len() >= 1_024);
    let cfg = LaunchConfig {
        grid_dim: (192, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(qgate_weight)
        .arg(k_weight)
        .arg(v_weight)
        .arg(qgate_out)
        .arg(k_out)
        .arg(v_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Decode-only projection fusion for the exact Bonsai-27B SSM input geometry:
/// wqkv=10,240, z=6,144, beta=48, alpha=48 rows, all with K=5,120 Q1T128.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_fused_ssm_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    wqkv_weight: &CudaView<u8>,
    z_weight: &CudaView<u8>,
    beta_weight: &CudaView<u8>,
    alpha_weight: &CudaView<u8>,
    wqkv_out: &mut CudaSlice<f32>,
    z_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    alpha_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_quants.len() >= 5_120);
    debug_assert!(input_scales.len() >= 160);
    debug_assert!(wqkv_weight.len() >= 10_240 * 40 * 18);
    debug_assert!(z_weight.len() >= 6_144 * 40 * 18);
    debug_assert!(beta_weight.len() >= 48 * 40 * 18);
    debug_assert!(alpha_weight.len() >= 48 * 40 * 18);
    debug_assert!(wqkv_out.len() >= 10_240);
    debug_assert!(z_out.len() >= 6_144);
    debug_assert!(beta_out.len() >= 48);
    debug_assert!(alpha_out.len() >= 48);
    let cfg = LaunchConfig {
        grid_dim: (160, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(wqkv_weight)
        .arg(z_weight)
        .arg(beta_weight)
        .arg(alpha_weight)
        .arg(wqkv_out)
        .arg(z_out)
        .arg(beta_out)
        .arg(alpha_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Decode-only projection fusion for the exact Bonsai-27B FFN geometry:
/// gate/up=17,408 rows with K=5,120 Q1T128.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1t128_fused_ffn_bonsai27b(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    gate_weight: &CudaView<u8>,
    up_weight: &CudaView<u8>,
    gate_out: &mut CudaSlice<f32>,
    up_out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(input_quants.len() >= 5_120);
    debug_assert!(input_scales.len() >= 160);
    debug_assert!(gate_weight.len() >= 17_408 * 40 * 18);
    debug_assert!(up_weight.len() >= 17_408 * 40 * 18);
    debug_assert!(gate_out.len() >= 17_408);
    debug_assert!(up_out.len() >= 17_408);
    let cfg = LaunchConfig {
        grid_dim: (272, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(gate_weight)
        .arg(up_weight)
        .arg(gate_out)
        .arg(up_out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Packed Prism Q1_0 prompt GEMM over `k_tokens` token-major f32 rows. The
/// kernel supports one or two tokens and writes token-major output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1_f32_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    q1_tiled: bool,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!((1..=2).contains(&k_tokens));
    let block = 256u32;
    let warps_per_block = block / 32;
    let cfg = LaunchConfig {
        grid_dim: ((rows.div_ceil(4) as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (k_tokens * 4 * 128 * std::mem::size_of::<f32>()) as u32,
    };
    let (rows, cols, blocks_per_row, k_tokens, q1_tiled) = (
        rows as i32,
        cols as i32,
        (cols / 128) as i32,
        k_tokens as i32,
        i32::from(q1_tiled),
    );
    let mut b = s.launch_builder(f);
    b.arg(input)
        .arg(weight)
        .arg(&rows)
        .arg(&cols)
        .arg(&blocks_per_row)
        .arg(&k_tokens)
        .arg(&q1_tiled)
        .arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Packed Prism Q1_0 by Q8_0 activation MMQ over up to eight token-major rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1_q8_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    q1_tiled: bool,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!((1..=MAX_PRISM_PREFILL_K).contains(&k_tokens));
    let block = 256u32;
    let warps_per_block = block / 32;
    let cfg = LaunchConfig {
        grid_dim: (
            (rows as u32).div_ceil(warps_per_block),
            (k_tokens as u32).div_ceil(8),
            1,
        ),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows, cols, blocks_per_row, k_tokens, q1_tiled) = (
        rows as i32,
        cols as i32,
        (cols / 128) as i32,
        k_tokens as i32,
        i32::from(q1_tiled),
    );
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(weight)
        .arg(&rows)
        .arg(&cols)
        .arg(&blocks_per_row)
        .arg(&k_tokens)
        .arg(&q1_tiled)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Packed Prism Q1_0 by Q8_0 prompt MMQ using signed-int8 tensor cores. A CTA
/// evaluates 16 output rows by as many as 128 tokens, so the packed Q1 tile is
/// fetched and decoded once for eight WMMA warps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1_q8_wmma_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_quants: &CudaSlice<i8>,
    input_scales: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    q1_tiled: bool,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!((1..=128).contains(&k_tokens));
    // A short multimodal tail uses only one 16-token warp.  Halving the CTA
    // keeps the same 128-row weight tile and fragment mapping while allowing
    // twice as many independent row CTAs to reside on an SM; the four inactive
    // prompt warps in the 256-thread shape only consumed registers.
    let block_threads = if k_tokens <= 16 { 128 } else { 256 };
    let cfg = LaunchConfig {
        grid_dim: (
            (rows as u32).div_ceil(128),
            (k_tokens as u32).div_ceil(128),
            1,
        ),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows, cols, blocks_per_row, k_tokens, q1_tiled) = (
        rows as i32,
        cols as i32,
        (cols / 128) as i32,
        k_tokens as i32,
        i32::from(q1_tiled),
    );
    let mut b = s.launch_builder(f);
    b.arg(input_quants)
        .arg(input_scales)
        .arg(weight)
        .arg(&rows)
        .arg(&cols)
        .arg(&blocks_per_row)
        .arg(&k_tokens)
        .arg(&q1_tiled)
        .arg(out)
        .arg(&residual);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Production f32 -> Q8/128 two's-complement bit-slice packer for the SM80
/// BMMA prompt lane. `bitplanes` holds exactly one byte per
/// activation value, arranged `[k_block][plane][token][u32_word]`; scales are
/// f16-rounded f32 values arranged `[k_block][token]`.
pub(crate) fn launch_prism_q8_b128_bitpack(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    bitplanes: &mut CudaSlice<u32>,
    scales: &mut CudaSlice<f32>,
    cols: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert_eq!(cols % 128, 0);
    debug_assert!(k_tokens > 0);
    debug_assert!(bitplanes.len() >= k_tokens * cols / std::mem::size_of::<u32>());
    debug_assert!(scales.len() >= k_tokens * (cols / 128));
    let cfg = LaunchConfig {
        grid_dim: ((cols / 128) as u32, k_tokens as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (cols, k_tokens) = (cols as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(input)
        .arg(bitplanes)
        .arg(scales)
        .arg(&cols)
        .arg(&k_tokens);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// SM80+ Q1_0 x Q8/128 binary-MMA prompt kernel. Dispatch is guarded by the
/// measured token crossover and retains DP4A/IMMA fallbacks for every other
/// device, shape, or A/B configuration.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_q1_q8_b128_bmma_gemm_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input_bitplanes: &CudaSlice<u32>,
    input_scales: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    q1_tiled: bool,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert_eq!(cols % 128, 0);
    debug_assert!((1..=MAX_PRISM_PREFILL_K).contains(&k_tokens));
    let cfg = LaunchConfig {
        grid_dim: (
            (rows as u32).div_ceil(128),
            (k_tokens as u32).div_ceil(128),
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: if q1_tiled { 0 } else { 128 * 16 },
    };
    let (rows, cols, blocks_per_row, k_tokens, q1_tiled) = (
        rows as i32,
        cols as i32,
        (cols / 128) as i32,
        k_tokens as i32,
        i32::from(q1_tiled),
    );
    let mut b = s.launch_builder(f);
    b.arg(input_bitplanes)
        .arg(input_scales)
        .arg(weight)
        .arg(&rows)
        .arg(&cols)
        .arg(&blocks_per_row)
        .arg(&k_tokens)
        .arg(&q1_tiled)
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

/// Q4_0 SoA GEMV over a native f32 activation, for the full-Q4 Gemma 4 MTP
/// assistant. Each lane forms an ordered 32-value block dot and lane 0 folds
/// those scaled block terms in block order. The largest official input width is
/// 8192, which fits eight warps plus the staged f32 row under 46 KiB.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_f32_gemv_soa(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    weight: &CudaView<u8>,
    rows: usize,
    cols: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    const SHARED_BUDGET: usize = 46 * 1024;
    assert!(rows > 0, "Q4_0 f32 GEMV has zero rows");
    assert!(
        cols > 0 && cols.is_multiple_of(32),
        "Q4_0 f32 GEMV columns must be a positive multiple of 32"
    );
    assert!(
        input.len() >= cols,
        "Q4_0 f32 GEMV activation is shorter than its contraction width"
    );
    let blocks_per_row = cols / 32;
    let expected_weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(18))
        .expect("Q4_0 f32 GEMV weight size overflowed");
    assert!(
        weight.len() >= expected_weight_bytes,
        "Q4_0 f32 GEMV weight view is shorter than its matrix"
    );
    assert!(
        out.len() >= rows,
        "Q4_0 f32 GEMV output is shorter than its row count"
    );

    let staged_input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .expect("Q4_0 f32 GEMV input scratch size overflowed");
    let per_warp_bytes = blocks_per_row
        .checked_mul(std::mem::size_of::<f32>())
        .expect("Q4_0 f32 GEMV term scratch size overflowed");
    assert!(
        staged_input_bytes
            .checked_add(per_warp_bytes)
            .is_some_and(|bytes| bytes <= SHARED_BUDGET),
        "Q4_0 f32 GEMV does not fit one warp under the shared-memory budget"
    );
    let warps_per_block = ((SHARED_BUDGET - staged_input_bytes) / per_warp_bytes).clamp(1, 8);
    let shared_mem_bytes = staged_input_bytes + warps_per_block * per_warp_bytes;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block as u32), 1, 1),
        block_dim: ((warps_per_block * 32) as u32, 1, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    };
    let (rows_i, blocks_i) = (rows as i32, blocks_per_row as i32);
    let mut builder = s.launch_builder(f);
    builder
        .arg(input)
        .arg(weight)
        .arg(&rows_i)
        .arg(&blocks_i)
        .arg(out)
        .arg(&residual);
    unsafe { builder.launch(cfg) }.map(|_| ())
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
    input_f32: &CudaSlice<f32>,
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
        ProjQuant::Q1_0 if kern.fast_q1 => {
            let function = if kern.q1_tiled {
                &kern.prism_q1t128_q8_gemv
            } else {
                &kern.prism_q1_q8_gemv
            };
            launch_prism_q1_q8_gemv(
                s,
                function,
                q8_0_quants,
                q8_0_scales,
                weight,
                rows,
                cols,
                out,
                residual,
            )
        }
        ProjQuant::Q1_0 | ProjQuant::Q2_0G64 | ProjQuant::Q2_0G128 => {
            let (bits, block_elements) = lane
                .prism_layout()
                .expect("Prism projection lane has a wire layout");
            launch_prism_low_bit_f32_gemv(
                s,
                &kern.prism_low_bit_f32_gemv,
                input_f32,
                weight,
                rows,
                cols,
                bits,
                block_elements,
                lane == ProjQuant::Q1_0 && kern.q1_tiled,
                out,
                residual,
            )
        }
    }
}

/// Per-projection batched GEMM dispatch. The activation buffers are token-major:
/// Q8_0 uses 32-value blocks, while Q4_K/Q6_K consume Q8_K super-blocks.
#[allow(clippy::too_many_arguments)]
fn dispatch_gemm_batched(
    s: &Arc<CudaStream>,
    kern: &CudaResidentKernels,
    lane: ProjQuant,
    input_f32: &CudaSlice<f32>,
    q8_0_scales: &CudaSlice<f32>,
    q8_0_quants: &CudaSlice<i8>,
    q8_b128_bitplanes: &CudaSlice<u32>,
    q8_b128_scales: &CudaSlice<f32>,
    bmma_ready: bool,
    q8k_scales: &CudaSlice<f32>,
    q8k_quants: &CudaSlice<i8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    k_tokens: usize,
    out: &mut CudaSlice<f32>,
    residual: i32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(
        residual == 0 || (lane == ProjQuant::Q1_0 && kern.fast_q1),
        "batched residual epilogue is implemented by the fast Q1 kernels"
    );
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
        ProjQuant::Q1_0 if bmma_ready && prism_bmma_shape_enabled(kern, cols, k_tokens) => {
            launch_prism_q1_q8_b128_bmma_gemm_batched(
                s,
                kern.prism_q1_q8_b128_bmma_gemm_batched
                    .as_ref()
                    .expect("BMMA policy requires the SM80 function"),
                q8_b128_bitplanes,
                q8_b128_scales,
                &weight.slice(0..weight.len()),
                rows,
                cols,
                k_tokens,
                kern.q1_tiled,
                out,
                residual,
            )
        }
        ProjQuant::Q1_0 if kern.fast_q1 && k_tokens >= 32 => {
            if let Some(f) = &kern.prism_q1_q8_wmma_gemm_batched {
                launch_prism_q1_q8_wmma_gemm_batched(
                    s,
                    f,
                    q8_0_quants,
                    q8_0_scales,
                    &weight.slice(0..weight.len()),
                    rows,
                    cols,
                    k_tokens,
                    kern.q1_tiled,
                    out,
                    residual,
                )
            } else {
                launch_prism_q1_q8_gemm_batched(
                    s,
                    &kern.prism_q1_q8_gemm_batched,
                    q8_0_quants,
                    q8_0_scales,
                    &weight.slice(0..weight.len()),
                    rows,
                    cols,
                    k_tokens,
                    kern.q1_tiled,
                    out,
                    residual,
                )
            }
        }
        ProjQuant::Q1_0 if kern.fast_q1 => launch_prism_q1_q8_gemm_batched(
            s,
            &kern.prism_q1_q8_gemm_batched,
            q8_0_quants,
            q8_0_scales,
            &weight.slice(0..weight.len()),
            rows,
            cols,
            k_tokens,
            kern.q1_tiled,
            out,
            residual,
        ),
        ProjQuant::Q1_0 => launch_prism_q1_f32_gemm_batched(
            s,
            &kern.prism_q1_f32_gemm_batched,
            input_f32,
            &weight.slice(0..weight.len()),
            rows,
            cols,
            k_tokens,
            kern.q1_tiled,
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
    q8_b128_bitplanes: &mut CudaSlice<u32>,
    q8_b128_scales: &mut CudaSlice<f32>,
    q8k_quants: &mut CudaSlice<i8>,
    q8k_scales: &mut CudaSlice<f32>,
    cols: usize,
    k_tokens: usize,
    lanes: &[ProjQuant],
) -> Result<bool, cudarc::driver::DriverError> {
    let bmma_ready =
        lanes.contains(&ProjQuant::Q1_0) && prism_bmma_shape_enabled(kern, cols, k_tokens);
    if bmma_ready {
        launch_prism_q8_b128_bitpack(
            s,
            kern.prism_q8_b128_bitpack
                .as_ref()
                .expect("BMMA policy requires the SM80 packer"),
            x,
            q8_b128_bitplanes,
            q8_b128_scales,
            cols,
            k_tokens,
        )?;
    }
    // A Q1-only BMMA group does not need the legacy Q8/32 copy. Mixed groups
    // still prepare it once for any Q8_0 projection sharing this activation.
    if lanes
        .iter()
        .any(|q| q.needs_q8_0(kern.fast_q1) && !(*q == ProjQuant::Q1_0 && bmma_ready))
    {
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
    Ok(bmma_ready)
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

/// Q8_K counterpart of [`launch_rms_inv_norm_quantize_q8_0`]. Normalization
/// stays device-side, while the CPU-provided RMS inverse and the existing
/// first-max/nearest-int Q8_K reducer preserve the reference bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rms_inv_norm_quantize_q8k(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n: usize,
    rms_inv: f32,
) -> Result<(), cudarc::driver::DriverError> {
    debug_assert!(n.is_multiple_of(256));
    let cfg = LaunchConfig {
        grid_dim: ((n / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x)
        .arg(w)
        .arg(quants)
        .arg(scales)
        .arg(&n_i)
        .arg(&rms_inv);
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

/// Batched shared-scratch Q4 GEMM launch geometry. Raw-wire Q4_0/Q4_1 and the
/// SoA Q4_0 twin differ only in their kernel-side weight addresses; all retain
/// the same [warp][token][block] scratch and ordered scalar fold.
#[allow(dead_code, clippy::too_many_arguments)]
fn launch_q4_shared_gemm_batched(
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
    const SHARED_BUDGET: u32 = 46 * 1024;
    assert!(k_tokens > 0, "Q4 batch must contain at least one token");
    let per_warp_bytes = (k_tokens as u32) * (blocks_per_row as u32) * 4;
    assert!(
        per_warp_bytes <= SHARED_BUDGET,
        "Q4 batch does not fit the shared-memory budget"
    );
    let warps_per_block = (SHARED_BUDGET / per_warp_bytes.max(1)).clamp(1, 8);
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (warps_per_block * 32, 1, 1),
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

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemm_batched(
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
    launch_q4_shared_gemm_batched(
        s,
        f,
        in_scales,
        in_quants,
        weight,
        rows,
        blocks_per_row,
        k_tokens,
        out,
    )
}

/// Shared-scratch batched Q4_0 GEMM for the quants-first SoA layout produced by
/// [`q4_0_wire_to_soa`]. This is the default dense projection lane; it exactly
/// preserves the prior raw-wire launch geometry and ordered accumulation.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemm_batched_soa_shared(
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
    launch_q4_shared_gemm_batched(
        s,
        f,
        in_scales,
        in_quants,
        weight,
        rows,
        blocks_per_row,
        k_tokens,
        out,
    )
}

/// Batched Q4_0 GEMM for the quants-first SoA layout produced by
/// [`q4_0_wire_to_soa`]. The exact lane-owner fold supports the production
/// verifier widths K=1..=14. It uses no dynamic shared memory: block-owner lanes
/// keep one weight block in registers while token-owner lanes accumulate terms
/// in exactly increasing block order.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemm_batched_soa(
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
    const MAX_TOKEN_OWNER_LANES: usize = 14;
    assert!(rows > 0, "Q4_0 SoA batch requires at least one output row");
    assert!(
        blocks_per_row > 0,
        "Q4_0 SoA batch requires at least one input block"
    );
    assert!(
        (1..=MAX_TOKEN_OWNER_LANES).contains(&k_tokens),
        "Q4_0 SoA batch width must be in 1..={MAX_TOKEN_OWNER_LANES}"
    );
    let input_blocks = k_tokens
        .checked_mul(blocks_per_row)
        .expect("Q4_0 SoA batch input size overflowed");
    assert!(
        in_scales.len() >= input_blocks,
        "Q4_0 SoA batch scale input is undersized"
    );
    assert!(
        in_quants.len() >= input_blocks.saturating_mul(32),
        "Q4_0 SoA batch quant input is undersized"
    );
    let weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(18))
        .expect("Q4_0 SoA batch weight size overflowed");
    assert!(
        weight.len() >= weight_bytes,
        "Q4_0 SoA batch weight input is undersized"
    );
    assert!(
        out.len() >= k_tokens.saturating_mul(rows),
        "Q4_0 SoA batch output is undersized"
    );

    const WARPS_PER_BLOCK: u32 = 8;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, bpr, kt) = (rows as i32, blocks_per_row as i32, k_tokens as i32);
    let mut builder = s.launch_builder(f);
    builder
        .arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(&kt)
        .arg(out);
    unsafe { builder.launch(cfg) }.map(|_| ())
}

/// Exact Ampere IMMA Q4_0 SoA batch. The CTA covers 128 output rows and all
/// verifier tokens; static shared memory holds one decoded 128x32 weight tile
/// plus the 16-token activation tile. Only the SM86 env-gated dispatch uses
/// this helper, while parity tests may launch the loaded SM80+ function directly.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemm_batched_soa_imma(
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
    const MAX_IMMA_TOKENS: usize = 14;
    assert!(rows > 0, "Q4_0 IMMA batch requires at least one output row");
    assert!(
        blocks_per_row > 0,
        "Q4_0 IMMA batch requires at least one input block"
    );
    assert!(
        (1..=MAX_IMMA_TOKENS).contains(&k_tokens),
        "Q4_0 IMMA batch width must be in 1..={MAX_IMMA_TOKENS}"
    );
    let input_blocks = k_tokens
        .checked_mul(blocks_per_row)
        .expect("Q4_0 IMMA batch input size overflowed");
    assert!(
        in_scales.len() >= input_blocks,
        "Q4_0 IMMA batch scale input is undersized"
    );
    assert!(
        in_quants.len() >= input_blocks.saturating_mul(32),
        "Q4_0 IMMA batch quant input is undersized"
    );
    let weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(18))
        .expect("Q4_0 IMMA batch weight size overflowed");
    assert!(
        weight.len() >= weight_bytes,
        "Q4_0 IMMA batch weight input is undersized"
    );
    assert!(
        out.len() >= k_tokens.saturating_mul(rows),
        "Q4_0 IMMA batch output is undersized"
    );

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(128), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, bpr, kt) = (rows as i32, blocks_per_row as i32, k_tokens as i32);
    let mut builder = s.launch_builder(f);
    builder
        .arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(&kt)
        .arg(out);
    unsafe { builder.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_1_gemm_batched(
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
    launch_q4_shared_gemm_batched(
        s,
        f,
        in_scales,
        in_quants,
        weight,
        rows,
        blocks_per_row,
        k_tokens,
        out,
    )
}

/// Launch one CSR-routed Q4 expert GEMM. Assignment rows are expert-major:
/// `token_offsets[e]..token_offsets[e + 1]` indexes `token_ids`, and `out` is
/// written in that same flat assignment order. The caller retains a separate
/// [token][router-rank] map for the strict weighted fold.
#[allow(dead_code, clippy::too_many_arguments)]
fn launch_q4_wire_gemm_routed(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight_arena: &CudaSlice<u8>,
    slot_ids: &CudaSlice<i32>,
    token_offsets: &CudaSlice<i32>,
    token_ids: &CudaSlice<i32>,
    weight_stride: usize,
    rows: usize,
    blocks_per_row: usize,
    expert_count: usize,
    max_assignments_per_expert: usize,
    chunked: bool,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    const SHARED_BUDGET: usize = 46 * 1024;
    const MAX_TILE: usize = 9;
    assert!(rows > 0, "routed Q4 GEMM requires at least one row");
    assert!(
        blocks_per_row > 0,
        "routed Q4 GEMM requires at least one input block"
    );
    assert!(weight_stride > 0, "routed Q4 GEMM weight stride is zero");
    assert!(
        slot_ids.len() >= expert_count,
        "routed Q4 GEMM slot map is shorter than its expert union"
    );
    assert!(
        token_offsets.len() > expert_count,
        "routed Q4 GEMM CSR offsets omit the terminal offset"
    );
    assert!(
        out.len() >= token_ids.len().saturating_mul(rows),
        "routed Q4 GEMM assignment output is undersized"
    );
    if expert_count == 0 || max_assignments_per_expert == 0 {
        return Ok(());
    }

    // Eight warps give the 26B-A4B expert matrices enough row parallelism. If a
    // future shape cannot fit even one assignment row at that width, halve the
    // CTA until one tile fits. The K-wide verifier currently needs at most 15
    // rows; capping a tile at nine bounds shared memory and lets K14 span grid.z.
    let scratch_blocks = if chunked {
        blocks_per_row.min(32)
    } else {
        blocks_per_row
    };
    let bytes_per_warp_tile = scratch_blocks
        .checked_mul(4)
        .expect("routed Q4 GEMM shared-memory size overflowed");
    let mut warps_per_block = 8usize;
    while warps_per_block > 1 && warps_per_block.saturating_mul(bytes_per_warp_tile) > SHARED_BUDGET
    {
        warps_per_block /= 2;
    }
    let tile_capacity = SHARED_BUDGET / (warps_per_block * bytes_per_warp_tile);
    assert!(
        tile_capacity > 0,
        "routed Q4 GEMM does not fit one assignment in shared memory"
    );
    let tile = max_assignments_per_expert
        .min(MAX_TILE)
        .min(tile_capacity)
        .max(1);
    let cfg = LaunchConfig {
        grid_dim: (
            (rows as u32).div_ceil(warps_per_block as u32),
            expert_count as u32,
            max_assignments_per_expert.div_ceil(tile) as u32,
        ),
        block_dim: ((warps_per_block * 32) as u32, 1, 1),
        shared_mem_bytes: (warps_per_block * tile * bytes_per_warp_tile) as u32,
    };
    let (stride, rows_i, bpr_i, experts_i, tile_i) = (
        weight_stride as u64,
        rows as i32,
        blocks_per_row as i32,
        expert_count as i32,
        tile as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight_arena)
        .arg(slot_ids)
        .arg(token_offsets)
        .arg(token_ids)
        .arg(&stride)
        .arg(&rows_i)
        .arg(&bpr_i)
        .arg(out)
        .arg(&experts_i)
        .arg(&tile_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_0_gemm_routed(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight_arena: &CudaSlice<u8>,
    slot_ids: &CudaSlice<i32>,
    token_offsets: &CudaSlice<i32>,
    token_ids: &CudaSlice<i32>,
    weight_stride: usize,
    rows: usize,
    blocks_per_row: usize,
    expert_count: usize,
    max_assignments_per_expert: usize,
    chunked: bool,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    launch_q4_wire_gemm_routed(
        s,
        f,
        in_scales,
        in_quants,
        weight_arena,
        slot_ids,
        token_offsets,
        token_ids,
        weight_stride,
        rows,
        blocks_per_row,
        expert_count,
        max_assignments_per_expert,
        chunked,
        out,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_q4_1_gemm_routed(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight_arena: &CudaSlice<u8>,
    slot_ids: &CudaSlice<i32>,
    token_offsets: &CudaSlice<i32>,
    token_ids: &CudaSlice<i32>,
    weight_stride: usize,
    rows: usize,
    blocks_per_row: usize,
    expert_count: usize,
    max_assignments_per_expert: usize,
    chunked: bool,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    launch_q4_wire_gemm_routed(
        s,
        f,
        in_scales,
        in_quants,
        weight_arena,
        slot_ids,
        token_offsets,
        token_ids,
        weight_stride,
        rows,
        blocks_per_row,
        expert_count,
        max_assignments_per_expert,
        chunked,
        out,
    )
}

/// Map expert-major CSR assignment rows back to token-major router order and
/// perform the strict rank-0..rank-N weighted fold for every hidden coordinate.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_moe_weighted_sum_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    expert_y: &CudaSlice<f32>,
    route_to_assignment: &CudaSlice<i32>,
    route_scales: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    hidden: usize,
    k_tokens: usize,
    route_count: usize,
) -> Result<(), cudarc::driver::DriverError> {
    assert!(hidden > 0, "batched MoE weighted sum has zero hidden width");
    assert!(
        (1..=15).contains(&k_tokens),
        "batched MoE weighted sum supports 1..=15 verifier rows"
    );
    assert!(
        (1..=8).contains(&route_count),
        "batched MoE weighted sum supports 1..=8 routes"
    );
    assert!(
        expert_y.len().is_multiple_of(hidden),
        "batched MoE assignment rows are not hidden-aligned"
    );
    let routes = k_tokens
        .checked_mul(route_count)
        .expect("batched MoE route count overflowed");
    assert!(
        expert_y.len() >= routes.saturating_mul(hidden),
        "batched MoE assignment rows are undersized"
    );
    // There is exactly one assignment row per logical token/rank pair. The
    // persistent verifier scratch can be physically wider than the current K;
    // do not let a malformed map address stale tail rows from that allocation.
    let assignment_count = routes;
    assert!(
        route_to_assignment.len() >= routes,
        "batched MoE route-to-assignment map is undersized"
    );
    assert!(
        route_scales.len() >= routes,
        "batched MoE route-scale matrix is undersized"
    );
    assert!(
        out.len() >= k_tokens.saturating_mul(hidden),
        "batched MoE output is undersized"
    );
    let cfg = LaunchConfig {
        grid_dim: ((hidden as u32).div_ceil(256), k_tokens as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (assignments_i, hidden_i, tokens_i, routes_i) = (
        assignment_count as i32,
        hidden as i32,
        k_tokens as i32,
        route_count as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(expert_y)
        .arg(route_to_assignment)
        .arg(route_scales)
        .arg(out)
        .arg(&assignments_i)
        .arg(&hidden_i)
        .arg(&tokens_i)
        .arg(&routes_i);
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

/// Bonsai fast RMSNorm -> Q8 activation for `k` token rows. This intentionally
/// uses a parallel reduction and is therefore part of the approximate fast Q1
/// contract, never the strict parity lane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prism_rms_norm_q8_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (k as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n, k) = (n as i32, k as i32);
    let mut b = s.launch_builder(f);
    b.arg(x)
        .arg(w)
        .arg(quants)
        .arg(scales)
        .arg(&n)
        .arg(&eps)
        .arg(&k);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Bonsai fast lane-parallel SwiGLU -> Q8 activation.
pub(crate) fn launch_prism_silu_mul_q8_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_blocks: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_blocks = n_blocks as i32;
    let mut b = s.launch_builder(f);
    b.arg(gate).arg(up).arg(quants).arg(scales).arg(&n_blocks);
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
    cache: &mut CudaSlice<u8>,
    base_position: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
    per_token_dim: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (k * n_kv_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
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
pub(crate) fn launch_kv_scatter_batched_q8_0(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u8>,
    base_position: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
    per_token_dim: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let blocks_per_head = head_dim / 32;
    let total = (k * n_kv_heads * blocks_per_head) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
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
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
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
    global_scores: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    // Scores live in the per-engine global scratch buffer. Shared memory only holds the query
    // and weighted-V partials (max_groups * head_dim, the decode-parity G-group reduction).
    let max_groups = (1024 / head_dim).max(1);
    let shared = ((head_dim + max_groups * head_dim) as u32) * 4;
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
        .arg(&splitk_active)
        .arg(global_scores);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Batched sliding-window attention (`attention_sw_batched`), the windowed counterpart of
/// [`launch_attention_batched`]. `window == 0` means no crop, i.e. plain causal.
///
/// `block_dim` is fixed at `head_dim` rather than exposed, because the scalar
/// `attention_decode_sw` this must stay bit-identical to derives its reduction group count
/// as `blockDim.x / head_dim`. The reference is the **gemma4** call site, which launches the
/// scalar kernel at exactly `head_dim` (G = 1); [`launch_attention_sw`] sizes G from the
/// position count instead and is a different reduction, so this is not a drop-in for it.
///
/// Lands ahead of its production caller, the same way `q4_0_gemm_routed` did: certified by
/// `attention_sw_batched_matches_gemma4_scalar_decode` first, wired into
/// `Gemma4CudaResident::verify_batch_moe` second. Proving the kernel bitwise against decode
/// is the part that is hard to get right and easy to get wrong quietly, so it is worth
/// having settled before any plumbing depends on it.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn launch_attention_sw_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_position: usize,
    max_pos: usize,
    scale: f32,
    window: usize,
    q_per_token: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    // Shared = query (head_dim) + scores over the widest window any token in the batch
    // sees. The last token has the longest prefix, and the window crops it.
    let widest = base_position + k;
    let span = if window > 0 {
        widest.min(window)
    } else {
        widest
    };
    let shared = ((head_dim + span) as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((k * n_heads) as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: shared,
    };
    let (nh, nkv, hd, bp, mp, win, qpt, ki) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        base_position as i32,
        max_pos as i32,
        window as i32,
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
        .arg(&win)
        .arg(&qpt)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kv_scatter_tree_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u8>,
    node_kvslot: &CudaSlice<i32>,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
    per_token_dim: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (k * n_kv_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
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
pub(crate) fn launch_kv_scatter_tree_batched_q8_0(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u8>,
    node_kvslot: &CudaSlice<i32>,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
    per_token_dim: usize,
    k: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let blocks_per_head = head_dim / 32;
    let total = (k * n_kv_heads * blocks_per_head) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
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
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
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
    global_scores: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    // Scores live in the per-engine global scratch buffer. Shared memory holds the query,
    // slot indices (<= base + k), and weighted-V partials (max_groups * head_dim).
    let max_groups = (1024 / head_dim).max(1);
    let shared = ((head_dim + base_position + k + max_groups * head_dim) as u32) * 4;
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
        .arg(&splitk_active)
        .arg(global_scores);
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

/// Weightless per-head RMS norm, used by Gemma 4's V projection. The CUDA
/// kernel guards the weight load with `use_weight != 0`, so a null pointer is a
/// valid placeholder and avoids forcing the caller to keep an unrelated norm
/// vector alive solely to satisfy the launch ABI.
#[allow(dead_code)]
pub(crate) fn launch_rms_norm_per_head_weightless(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    buf: &mut CudaSlice<f32>,
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    assert!(head_count > 0, "weightless RMS norm has zero heads");
    assert!(head_dim > 0, "weightless RMS norm has zero head width");
    assert!(
        buf.len() >= head_count.saturating_mul(head_dim),
        "weightless RMS norm buffer is undersized"
    );
    let cfg = LaunchConfig {
        grid_dim: (head_count as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (head_dim as u32) * 4,
    };
    let (head_dim_i, use_weight) = (head_dim as i32, 0i32);
    // Kernel arguments are untyped ABI payloads. CUDA device pointers are
    // 64-bit; this zero payload binds the unused `const float* weight` argument.
    let null_weight = 0u64;
    let mut b = s.launch_builder(f);
    b.arg(buf)
        .arg(&null_weight)
        .arg(&head_dim_i)
        .arg(&eps)
        .arg(&use_weight);
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
    decay_out: &mut CudaSlice<f32>,
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
        .arg(decay_out)
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
    decay: &CudaSlice<f32>,
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
        .arg(decay)
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
pub(crate) fn launch_ssm_gates_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    beta_raw: &CudaSlice<f32>,
    alpha_raw: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    decay_out: &mut CudaSlice<f32>,
    nv: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = nv * k_tokens;
    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nvi, ki) = (nv as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(beta_raw)
        .arg(alpha_raw)
        .arg(dt_bias)
        .arg(a)
        .arg(beta_out)
        .arg(decay_out)
        .arg(&nvi)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_conv1d_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    conv_w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    conv_state: &mut CudaSlice<f32>,
    conv_out: &mut CudaSlice<f32>,
    conv_dim: usize,
    d_conv: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (cdi, dci, ki) = (conv_dim as i32, d_conv as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(conv_w)
        .arg(x)
        .arg(conv_state)
        .arg(conv_out)
        .arg(&cdi)
        .arg(&dci)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_l2_norm_per_head_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    buf: &mut CudaSlice<f32>,
    token_stride: usize,
    offset: usize,
    head_count: usize,
    head_dim: usize,
    k_tokens: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((k_tokens * head_count) as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: (head_dim as u32) * 4,
    };
    let (stride, off, heads, dim, ki) = (
        token_stride as i32,
        offset as i32,
        head_count as i32,
        head_dim as i32,
        k_tokens as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(buf)
        .arg(&stride)
        .arg(&off)
        .arg(&heads)
        .arg(&dim)
        .arg(&ki)
        .arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ssm_delta_rule_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    state: &mut CudaSlice<f32>,
    conv_out: &CudaSlice<f32>,
    z: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    decay: &CudaSlice<f32>,
    ssm_norm: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    d_state: usize,
    nk: usize,
    nv: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    k_tokens: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 1),
        block_dim: (d_state as u32, 1, 1),
        shared_mem_bytes: (3 * d_state as u32) * 4,
    };
    let (ds, nk, nv, kd, vd, cd, ki) = (
        d_state as i32,
        nk as i32,
        nv as i32,
        key_dim as i32,
        value_dim as i32,
        conv_dim as i32,
        k_tokens as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(state)
        .arg(conv_out)
        .arg(z)
        .arg(beta)
        .arg(decay)
        .arg(ssm_norm)
        .arg(out)
        .arg(&ds)
        .arg(&nk)
        .arg(&nv)
        .arg(&kd)
        .arg(&vd)
        .arg(&cd)
        .arg(&ki)
        .arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Bonsai/Qwen3.5 four-tap convolution.  The history stays in registers for
/// every token in the prompt chunk and is committed once at the end.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_qwen35_ssm_conv1d_d4_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    conv_w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    conv_state: &mut CudaSlice<f32>,
    conv_out: &mut CudaSlice<f32>,
    conv_dim: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (cd, kt) = (conv_dim as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(conv_w)
        .arg(x)
        .arg(conv_state)
        .arg(conv_out)
        .arg(&cd)
        .arg(&kt);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Paired Q/K L2 normalization for the D=128 Bonsai SSM geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_qwen35_ssm_qk_l2_norm_d128_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    conv: &mut CudaSlice<f32>,
    conv_dim: usize,
    key_dim: usize,
    n_key_heads: usize,
    k_tokens: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((k_tokens * n_key_heads) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (cd, kd, nkh, kt) = (
        conv_dim as i32,
        key_dim as i32,
        n_key_heads as i32,
        k_tokens as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(conv).arg(&cd).arg(&kd).arg(&nkh).arg(&kt).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Register-sharded D=128 delta rule.  This launch is valid only for the
/// Qwen3.5/Bonsai state geometry checked by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_qwen35_ssm_delta_rule_d128_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    state: &mut CudaSlice<f32>,
    conv_out: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    decay: &CudaSlice<f32>,
    raw_out: &mut CudaSlice<f32>,
    nk: usize,
    nv: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 4),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    let (nk, nv, kd, vd, cd, kt) = (
        nk as i32,
        nv as i32,
        key_dim as i32,
        value_dim as i32,
        conv_dim as i32,
        k_tokens as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(state)
        .arg(conv_out)
        .arg(beta)
        .arg(decay)
        .arg(raw_out)
        .arg(&nk)
        .arg(&nv)
        .arg(&kd)
        .arg(&vd)
        .arg(&cd)
        .arg(&kt);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Fused D=128 gated RMSNorm and Q8 activation writer for the Q1 ssm_out
/// projection. The recurrence's raw f32 result is never materialized again.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_qwen35_ssm_rmsnorm_gate_q8_d128_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    raw: &CudaSlice<f32>,
    z: &CudaSlice<f32>,
    ssm_norm: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    nv: usize,
    value_dim: usize,
    k_tokens: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((k_tokens * nv) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nv, vd, kt) = (nv as i32, value_dim as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(raw)
        .arg(z)
        .arg(ssm_norm)
        .arg(quants)
        .arg(scales)
        .arg(&nv)
        .arg(&vd)
        .arg(&kt)
        .arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_deinterleave_qgate_batched(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    qg: &CudaSlice<f32>,
    q_out: &mut CudaSlice<f32>,
    gate_out: &mut CudaSlice<f32>,
    n_heads: usize,
    head_dim: usize,
    k_tokens: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = k_tokens * n_heads * head_dim;
    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, ki) = (n_heads as i32, head_dim as i32, k_tokens as i32);
    let mut b = s.launch_builder(f);
    b.arg(qg)
        .arg(q_out)
        .arg(gate_out)
        .arg(&nh)
        .arg(&hd)
        .arg(&ki);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kv_scatter(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u8>,
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
pub(crate) fn launch_kv_scatter_q8_0(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<u8>,
    position: &CudaSlice<i32>,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let blocks_per_head = head_dim / 32;
    let total = (n_kv_heads * blocks_per_head) as u32;
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
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
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
    is_q8_0: bool,
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
        let scores_fn = if is_q8_0 {
            &k.attn_sk_scores_q8_0
        } else if k.attn_coalesced {
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
        let partial_fn = if is_q8_0 {
            &k.attn_sk_partial_q8_0
        } else {
            &k.attn_sk_partial
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

/// Whether flash prefill attention is enabled.
/// Opt-in via `CAMELID_FLASH_PREFILL=1` (prefill-only, token-parity).
/// Default is off (retaining bit-identity with serial forward pass).
fn flash_prefill_enabled() -> bool {
    std::env::var("CAMELID_FLASH_PREFILL").is_ok_and(|v| {
        v != "0" && !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("off")
    })
}

/// Fused Tiled Flash Prefill Attention (online softmax).
/// Tiled across query blocks (B_Q = 16) and key blocks (B_K = 32) in shared memory and registers.
/// Computes causal attention in a single fused kernel pass with 0 bytes intermediate DRAM traffic.
///
/// One entry point per head_dim: 64/128/256 are compile-time specialized so the output accumulator
/// stays in registers (measured on sm_89: 0-byte stack frame), and every other head_dim takes the
/// runtime-bounded twin, whose accumulator is dynamically indexed and does use local memory.
///
/// Unlike the split-K kernels this replaces, there is no `SPLITK_THRESHOLD` engagement floor: with
/// `CAMELID_FLASH_PREFILL` set this runs at every prefix length, so the token-parity reassociation
/// now applies to short prompts too. Shared memory is `192 * head_dim` bytes and does NOT grow with
/// the prefix, which is why the floor is no longer needed to bound it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_flash_prefill(
    s: &Arc<CudaStream>,
    k: &CudaResidentKernels,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_position: usize,
    k_tokens: usize,
    q_per_token: usize,
    max_pos: usize,
    scale: f32,
) -> Result<(), cudarc::driver::DriverError> {
    const FLASH_PREFILL_BQ: usize = 16;
    const FLASH_PREFILL_BK: usize = 32;
    // 192 * head_dim bytes; 256 is exactly the 48 KiB default dynamic shared-memory limit, and the
    // block-per-query-tile layout assumes the 8 warps this launcher requests.
    assert!(head_dim <= 256, "flash prefill requires head_dim <= 256");
    let q_tiles = (k_tokens as u32).div_ceil(FLASH_PREFILL_BQ as u32);
    let block: u32 = 256;
    let shared_mem_bytes = (FLASH_PREFILL_BQ * head_dim * 4
        + FLASH_PREFILL_BK * head_dim * 2
        + FLASH_PREFILL_BK * head_dim * 2) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, q_tiles, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes,
    };
    let (nh, nkv, hd, bp, kt, qpt, mp) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        base_position as i32,
        k_tokens as i32,
        q_per_token as i32,
        max_pos as i32,
    );
    // Each head_dim has its own entry point so ptxas does not charge one kernel's register and
    // stack-frame footprint to the others; the runtime-bounded twin is the fallback.
    let f = match head_dim {
        64 => &k.flash_attention_prefill_tiled_d64,
        128 => &k.flash_attention_prefill_tiled_d128,
        256 => &k.flash_attention_prefill_tiled_d256,
        _ => &k.flash_attention_prefill_tiled_dyn,
    };
    let mut b = s.launch_builder(f);
    b.arg(q)
        .arg(cache_k)
        .arg(cache_v)
        .arg(out)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&bp)
        .arg(&kt)
        .arg(&qpt)
        .arg(&mp)
        .arg(&scale);
    unsafe { b.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: &CudaSlice<i32>,
    // Positions used to choose the weighted-V group count. The non-graph path passes the exact
    // current count; the graph-capture path passes `max_pos` so launch geometry is replay-stable.
    shared_positions: usize,
    max_pos: usize,
    scale: f32,
    global_scores: &mut CudaSlice<f32>,
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
        // qsh[head_dim] + vpart[groups*head_dim]; scores live in global scratch.
        shared_mem_bytes: (head_dim as u32 * (1 + groups)) * 4,
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
        .arg(&scale)
        .arg(global_scores);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// Sliding-window decode attention (`attention_decode_sw`). Identical launch
/// geometry to [`launch_attention`] — same block/G sizing, same shared-memory
/// budget — plus a trailing `window` scalar. The kernel computes
/// `start = position_count - window` on the device from `position_ptr`, so the
/// launch config does not vary with position. Scores use the same global scratch
/// buffer as the full-causal launcher.
///
/// `window` is the gemma3 convention: it INCLUDES the current position, so a
/// layer at `pos` attends `[pos + 1 - window ..= pos]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_sw(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<u8>,
    cache_v: &CudaSlice<u8>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: &CudaSlice<i32>,
    shared_positions: usize,
    max_pos: usize,
    scale: f32,
    window: usize,
    global_scores: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let max_groups = (1024 / head_dim as u32).max(1);
    let groups = (shared_positions.max(1) as u32)
        .div_ceil(head_dim as u32)
        .clamp(1, max_groups);
    let block = groups * head_dim as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (head_dim as u32 * (1 + groups)) * 4,
    };
    let (nh, nkv, hd, mp, win) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
        window as i32,
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
        .arg(&scale)
        .arg(&win)
        .arg(global_scores);
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
    Q1_0,
    Q2_0G64,
    Q2_0G128,
    Q4K,
    Q5K,
    Q6K,
    Q2K,
    Q3K,
    IQ4XS,
}

impl ProjQuant {
    /// Whether this projection consumes the compact Q8_0 activation buffers.
    fn needs_q8_0(self, fast_q1: bool) -> bool {
        self == ProjQuant::Q8_0 || (self == ProjQuant::Q1_0 && fast_q1)
    }

    /// Whether this is one of Prism's packed low-bit rows. These projections
    /// consume the original f32 activation to preserve the Metal oracle's
    /// reduction association.
    fn is_prism_low_bit(self, fast_q1: bool) -> bool {
        matches!(self, ProjQuant::Q2_0G64 | ProjQuant::Q2_0G128)
            || (self == ProjQuant::Q1_0 && !fast_q1)
    }

    fn prism_layout(self) -> Option<(usize, usize)> {
        match self {
            ProjQuant::Q1_0 => Some((1, 128)),
            ProjQuant::Q2_0G64 => Some((2, 64)),
            ProjQuant::Q2_0G128 => Some((2, 128)),
            _ => None,
        }
    }

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
        matches!(
            self,
            ProjQuant::Q8_0 | ProjQuant::Q1_0 | ProjQuant::Q4K | ProjQuant::Q6K
        )
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
            ProjQuant::Q8_0
                | ProjQuant::Q1_0
                | ProjQuant::Q2_0G64
                | ProjQuant::Q2_0G128
                | ProjQuant::Q4K
                | ProjQuant::Q6K
                | ProjQuant::Q3K
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
    /// gemma3 "sandwich" post-norms: `post_attention_norm` is applied to the O
    /// projection's output and `post_ffn_norm` to the down projection's output,
    /// each BEFORE its residual add. `None` for every other architecture, which
    /// keeps those layers on the fused gemv+residual path byte-identically.
    /// Uploaded by [`CudaResidentDecode::set_layer_gemma3_norms`], never by
    /// `set_layer_located` — so no existing call site changes shape.
    post_attn_norm: Option<CudaSlice<f32>>,
    post_ffn_norm: Option<CudaSlice<f32>>,
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
    d_decay: CudaSlice<f32>,     // num_v_heads (post-exp)
    d_ssm_mix: CudaSlice<f32>,   // value_dim (delta-rule output, before ssm_out)
    d_qgate: CudaSlice<f32>,     // 2*q_width (full-attn fused query+gate projection)
    d_gate_attn: CudaSlice<f32>, // q_width (deinterleaved attention gate)
}

/// gemma3 (windowed-attention arch) per-layer resident state. `None` on
/// [`CudaResidentDecode`] for every other architecture, so no extra VRAM and no
/// path change for anything else.
///
/// Mirrors `metal::ResidentLayerSchedule` deliberately: the Metal lane
/// (PR #560) and this one must express the same per-layer schedule, or the two
/// GPU lanes could disagree about which layers slide. Both vectors are indexed
/// by OWNED layer (a pipeline-sharded node holds a subrange), matching
/// `LlamaInferenceSession::gemma3_resident_schedule`.
#[cfg(feature = "cuda")]
struct Gemma3Gpu {
    /// Per owned layer: `true` selects the ALT (local-theta) rope tables.
    use_alt_rope: Vec<bool>,
    /// Per owned layer: `Some(w)` attends only the last `w` keys INCLUDING the
    /// current position (`[pos + 1 - w ..= pos]`), matching
    /// `Gemma3Metadata::layer_window` and the runnable reference. `None` is a
    /// full-causal (global) layer.
    window: Vec<Option<usize>>,
    /// ALT (local-theta) rope tables for THIS token, uploaded by
    /// [`CudaResidentDecode::upload_alt_rope`] before each forward. The primary
    /// (global-theta) tables stay in `d_cos`/`d_sin`.
    d_cos_alt: CudaSlice<f32>,
    d_sin_alt: CudaSlice<f32>,
    /// gemma3's FFN activation is GeGLU (`gelu_tanh(gate) * up`), not SiLU.
    ffn_geglu: bool,
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
    artifact: ResidentCudaArtifact,
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
    /// Exact Bonsai-27B projection fusion policy paired with `k.q1_tiled`.
    bonsai27b_fused_projections: bool,
    /// Exact Q8 bitplane+POPC decode policy for the SHA-pinned Windows SM86 lane.
    bonsai27b_popc: bool,
    layers: Vec<ResidentLayer>,
    final_norm: CudaSlice<f32>,
    output_weight: CudaSlice<u8>,
    /// Quant lane of the output (lm_head) projection. Q6_K for Q4_K_M models.
    output_quant: ProjQuant,
    /// KV cache stored as f16 bits (u16) or quantized Q8_0 blocks.
    cache_k: Vec<CudaSlice<u8>>,
    cache_v: Vec<CudaSlice<u8>>,
    pub kv_quant: crate::model::KvCacheQuantization,
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
    // Batched speculative-verification attention scores, k_tokens-major
    // (flat (token * n_heads + head)).
    d_verify_scores: CudaSlice<f32>, // MAX_VERIFY_K * n_heads * max_pos
    d_proj: CudaSlice<f32>,
    /// Holds a projection's output AFTER a gemma3 sandwich post-norm and before
    /// its residual add. A top-level field rather than one inside `Gemma3Gpu` so
    /// the layer loop can hold `&self.layers[li].post_*_norm` and `&mut` this
    /// buffer in one statement without a nested borrow through `self.gemma3`.
    /// Allocated unconditionally: `hidden` f32s is a few KiB, and an
    /// always-present buffer costs less than an `Option` on the hot path.
    d_post: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_ffn_act: CudaSlice<f32>,
    d_in_scales: CudaSlice<f32>,
    d_in_quants: CudaSlice<i8>,
    /// Persistent lossless bit-sliced view of `d_in_quants`, packed once per
    /// activation and reused by every Q1 projection consuming that activation.
    d_in_bitplanes: CudaSlice<u32>,
    d_in_qsums: CudaSlice<i32>,
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
    /// Qwen3.5 device-loop graph containing the resident layer stack, final norm,
    /// and lm_head. Token gather, RoPE row selection, and ring argmax stay outside
    /// because their source/destination views change per step. This still collapses
    /// the hundreds of stable transformer/SSM launches to one graph submission.
    device_forward_graph: Option<SendCudaGraph>,
    /// Phase 6 multi-stream overlap (side streams + join events). `None` unless
    /// `CAMELID_CUDA_STREAMS` was on at build — the flag-off path constructs
    /// nothing and stays in cudarc's single-stream mode.
    overlap: Option<StreamOverlap>,
    /// Lazily-allocated K-batched scratch for the speculative-verify forward.
    verify_scratch: Option<VerifyScratch>,
    /// Wider prompt-only scratch. Keeping it separate preserves the small,
    /// bounded speculative-verify allocation while Q1 prefill uses J=128.
    prefill_scratch: Option<VerifyScratch>,
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
    /// gemma3 windowed-arch schedule + ALT rope tables + post-norm scratch.
    /// `None` for every other architecture. Set by [`Self::set_gemma3`].
    gemma3: Option<Gemma3Gpu>,
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
const MAX_PRISM_PREFILL_K: usize = 128;
const DEFAULT_PRISM_BMMA_MIN_TOKENS: usize = 32;

/// Fast Q1 CUDA is the production default. It matches the established
/// Q1-by-Q8 contraction used by optimized CUDA runtimes and is deterministic,
/// but activation quantization means it is not bit-identical to Camelid's
/// original f32 reduction. Set `CAMELID_PRISM_CUDA_STRICT=1` for the exact lane.
fn prism_cuda_fast_policy(strict: Option<&str>) -> bool {
    !env_switch_enabled(strict)
}

fn prism_cuda_fast_from_env() -> bool {
    prism_cuda_fast_policy(std::env::var("CAMELID_PRISM_CUDA_STRICT").ok().as_deref())
}

/// Artifact identity for CUDA paths whose kernels/layouts are validated against
/// one exact model file rather than inferred from a potentially shared geometry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ResidentCudaArtifact {
    #[default]
    Generic,
    PrismBonsai27bQ1,
}

const PRISM_BONSAI27B_Q1_SHA256: &str =
    "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0";

pub(crate) fn resident_cuda_artifact_from_sha256(sha256: &str) -> ResidentCudaArtifact {
    if sha256.eq_ignore_ascii_case(PRISM_BONSAI27B_Q1_SHA256) {
        ResidentCudaArtifact::PrismBonsai27bQ1
    } else {
        ResidentCudaArtifact::Generic
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrismQ1ModelPolicy {
    /// Projection bytes are stored in the same-size Q1T128 tiled layout. This is
    /// immutable for the engine lifetime: every uploader and reader must agree.
    q1_tiled: bool,
    /// The three exact-shape Bonsai projection fusions may consume that layout.
    fused_projections: bool,
}

fn env_switch_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1") | Some("true") | Some("on") | Some("yes"))
}

fn is_bonsai27b_geometry(
    n_layers: usize,
    hidden: usize,
    ffn_dim: usize,
    q_width: usize,
    kv_width: usize,
) -> bool {
    n_layers == 64 && hidden == 5_120 && ffn_dim == 17_408 && q_width == 6_144 && kv_width == 1_024
}

/// Pure policy seam for the model-local Q1 upload/reader contract. The exact
/// Bonsai-27B Windows fast-Q1 lane defaults to Q1T128 and its matching projection
/// fusions together. Positive flags retain the diagnostic/bring-up paths; negative
/// flags always win so a single process can construct a fresh engine with a
/// different layout after a model reload.
#[allow(clippy::too_many_arguments)]
fn prism_q1_model_policy(
    fast_q1: bool,
    windows: bool,
    artifact: ResidentCudaArtifact,
    exact_geometry: bool,
    q1t128_opt_in: bool,
    q1t128_opt_out: bool,
    fused_opt_in: bool,
    fused_opt_out: bool,
) -> PrismQ1ModelPolicy {
    let exact_bonsai27b = artifact == ResidentCudaArtifact::PrismBonsai27bQ1 && exact_geometry;
    let exact_default = fast_q1 && windows && exact_bonsai27b;
    let q1_tiled = !q1t128_opt_out && (q1t128_opt_in || exact_default);
    let fused_projections =
        fast_q1 && exact_bonsai27b && q1_tiled && !fused_opt_out && (fused_opt_in || exact_default);
    PrismQ1ModelPolicy {
        q1_tiled,
        fused_projections,
    }
}

fn prism_q1_model_policy_for_geometry(
    fast_q1: bool,
    artifact: ResidentCudaArtifact,
    n_layers: usize,
    hidden: usize,
    ffn_dim: usize,
    q_width: usize,
    kv_width: usize,
) -> PrismQ1ModelPolicy {
    let flag = |name: &str| env_switch_enabled(std::env::var(name).ok().as_deref());
    prism_q1_model_policy(
        fast_q1,
        cfg!(target_os = "windows"),
        artifact,
        is_bonsai27b_geometry(n_layers, hidden, ffn_dim, q_width, kv_width),
        flag("CAMELID_PRISM_CUDA_Q1T128"),
        flag("CAMELID_PRISM_CUDA_NO_Q1T128"),
        flag("CAMELID_PRISM_CUDA_FUSED_PROJECTIONS"),
        flag("CAMELID_PRISM_CUDA_NO_FUSED_PROJECTIONS"),
    )
}

/// Promotion gate for the exact POPC decode lane. This is intentionally a pure,
/// per-engine decision: model reloads re-read the negative escape and an opt-in
/// Q1T layout on another artifact/device can never accidentally inherit it.
#[allow(clippy::too_many_arguments)]
fn prism_cuda_popc_policy(
    fast_q1: bool,
    windows: bool,
    artifact: ResidentCudaArtifact,
    exact_geometry: bool,
    q1_tiled: bool,
    sm86: bool,
    no_popc: bool,
) -> bool {
    fast_q1
        && windows
        && artifact == ResidentCudaArtifact::PrismBonsai27bQ1
        && exact_geometry
        && q1_tiled
        && sm86
        && !no_popc
}

fn parse_prism_bmma_min_tokens(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PRISM_BMMA_MIN_TOKENS)
        .clamp(1, MAX_PRISM_PREFILL_K)
}

/// Binary tensor-core prompt acceleration is default-on independently of the
/// broader fast-Q1 gate. This negative flag provides a stable A/B escape hatch
/// without opting the entire Q1 CUDA path back into strict f32 arithmetic.
fn prism_cuda_bmma_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("CAMELID_PRISM_CUDA_NO_BMMA").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

fn prism_cuda_bmma_min_tokens() -> usize {
    static MIN_TOKENS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        parse_prism_bmma_min_tokens(
            std::env::var("CAMELID_PRISM_CUDA_BMMA_MIN_TOKENS")
                .ok()
                .as_deref(),
        )
    })
}

fn prism_bmma_dispatch_policy(
    fast_q1: bool,
    bmma_enabled: bool,
    functions_available: bool,
    cols: usize,
    k_tokens: usize,
    min_tokens: usize,
) -> bool {
    fast_q1
        && bmma_enabled
        && functions_available
        && cols.is_multiple_of(128)
        && (min_tokens..=MAX_PRISM_PREFILL_K).contains(&k_tokens)
}

fn prism_bmma_shape_enabled(kern: &CudaResidentKernels, cols: usize, k_tokens: usize) -> bool {
    prism_bmma_dispatch_policy(
        kern.fast_q1,
        prism_cuda_bmma_enabled(),
        kern.prism_q8_b128_bitpack.is_some() && kern.prism_q1_q8_b128_bmma_gemm_batched.is_some(),
        cols,
        k_tokens,
        prism_cuda_bmma_min_tokens(),
    )
}

/// K-batched scratch buffers for `verify_batch`, sized `MAX_VERIFY_K * dim`.
struct VerifyScratch {
    vh: CudaSlice<f32>,
    vn: CudaSlice<f32>,
    viq: CudaSlice<i8>,
    vis: CudaSlice<f32>,
    viqk: CudaSlice<i8>,
    visk: CudaSlice<f32>,
    /// Q8/128 two's-complement bitplanes and one f16-rounded scale per
    /// token/block. Shared by every Q1 projection consuming the same activation.
    vibits: CudaSlice<u32>,
    vibscales: CudaSlice<f32>,
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

/// The fully-resident Bonsai-27B Q1 device loop is graph-safe and materially
/// reduces Windows/WDDM submission overhead. The shape gate lives at engagement;
/// keep the broader legacy graph switch opt-in for other architectures. This exact
/// lane defaults on only for Windows and has its own negative A/B escape hatch.
fn qwen35_device_graphs_enabled() -> bool {
    let disabled = matches!(
        std::env::var("CAMELID_PRISM_CUDA_NO_GRAPH").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    );
    !disabled && (cfg!(target_os = "windows") || cuda_graphs_enabled())
}

/// Use explicit resident-engine ordering instead of cudarc's per-slice automatic
/// events. This is the Windows default because the engine already owns one dedicated
/// stream and WDDM makes hundreds of redundant event API calls per token expensive.
/// CUDA graph capture also requires it: a wait on an event recorded before capture is
/// an external dependency and CUDA correctly rejects the graph as isolated.
fn cuda_manual_stream_order_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let safe_events = matches!(
            std::env::var("CAMELID_CUDA_SAFE_EVENTS").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        !safe_events && (cfg!(target_os = "windows") || cuda_graphs_enabled())
    })
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
        Self::new_with_kv_quant(
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
            split_half_pairing,
            crate::model::KvCacheQuantization::F16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_kv_quant(
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
        kv_quant: crate::model::KvCacheQuantization,
    ) -> Result<Self, String> {
        Self::new_for_artifact_with_kv_quant(
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
            split_half_pairing,
            ResidentCudaArtifact::Generic,
            kv_quant,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_for_artifact(
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
        artifact: ResidentCudaArtifact,
    ) -> Result<Self, String> {
        Self::new_for_artifact_with_kv_quant(
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
            split_half_pairing,
            artifact,
            crate::model::KvCacheQuantization::F16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_for_artifact_with_kv_quant(
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
        artifact: ResidentCudaArtifact,
        kv_quant: crate::model::KvCacheQuantization,
    ) -> Result<Self, String> {
        if kv_quant == crate::model::KvCacheQuantization::Q8_0 && !head_dim.is_multiple_of(32) {
            return Err(format!(
                "Q8_0 resident KV cache requires head_dim to be a multiple of 32, got {head_dim}"
            ));
        }
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let fast_q1 = prism_cuda_fast_from_env();
        let q1_policy = prism_q1_model_policy_for_geometry(
            fast_q1, artifact, n_layers, hidden, ffn_dim, q_width, kv_width,
        );
        let k = CudaResidentKernels::new_with_q1_policy(q1_policy.q1_tiled, fast_q1)?;
        let bonsai27b_popc = prism_cuda_popc_policy(
            fast_q1,
            cfg!(target_os = "windows"),
            artifact,
            is_bonsai27b_geometry(n_layers, hidden, ffn_dim, q_width, kv_width),
            q1_policy.q1_tiled,
            k.sm86,
            env_switch_enabled(std::env::var("CAMELID_PRISM_CUDA_NO_POPC").ok().as_deref()),
        );
        let s = &k.stream;
        let max_in = hidden.max(ffn_dim).max(q_width); // widest quantize input
        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("alloc: {e}"));
        let kv_bytes_per_elem = match kv_quant {
            crate::model::KvCacheQuantization::Q8_0 => (kv_width / 32) * 34,
            _ => kv_width * 2,
        };
        let cache_k = (0..n_layers)
            .map(|_| s.alloc_zeros::<u8>(kv_bytes_per_elem * max_pos))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("kv alloc: {e}"))?;
        let cache_v = (0..n_layers)
            .map(|_| s.alloc_zeros::<u8>(kv_bytes_per_elem * max_pos))
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
            artifact,
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
            bonsai27b_fused_projections: q1_policy.fused_projections,
            bonsai27b_popc,
            layers: Vec::with_capacity(n_layers),
            final_norm: alloc_f(hidden)?,
            output_weight: s.alloc_zeros::<u8>(1).map_err(|e| format!("alloc: {e}"))?,
            output_quant: ProjQuant::Q8_0,
            cache_k,
            cache_v,
            kv_quant,
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
            d_verify_scores: alloc_f(MAX_VERIFY_K * n_heads * max_pos)?,
            d_proj: alloc_f(hidden)?,
            d_post: alloc_f(hidden)?,
            d_gate: alloc_f(ffn_dim)?,
            d_up: alloc_f(ffn_dim)?,
            d_ffn_act: alloc_f(ffn_dim)?,
            d_in_scales: alloc_f(max_in / 32)?,
            d_in_quants: s
                .alloc_zeros::<i8>(max_in)
                .map_err(|e| format!("alloc: {e}"))?,
            d_in_bitplanes: s
                .alloc_zeros::<u32>(max_in / 4)
                .map_err(|e| format!("alloc: {e}"))?,
            d_in_qsums: s
                .alloc_zeros::<i32>(max_in / 32)
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
            device_forward_graph: None,
            overlap,
            verify_scratch: None,
            prefill_scratch: None,
            tree_scratch: None,
            offload: None,
            qwen35: None,
            gemma3: None,
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
        let q_rows = if self.qwen35.is_some() {
            2 * self.q_width
        } else {
            self.q_width
        };
        let projection_shapes = [
            (q_rows, self.hidden),
            (self.kv_width, self.hidden),
            (self.kv_width, self.hidden),
            (self.hidden, self.q_width),
            (self.ffn_dim, self.hidden),
            (self.ffn_dim, self.hidden),
            (self.hidden, self.ffn_dim),
        ];
        let (attn_norm, ffn_norm) = (up_f(attn_norm)?, up_f(ffn_norm)?);
        let q_norm_gpu = q_norm.map(up_f).transpose()?;
        let k_norm_gpu = k_norm.map(up_f).transpose()?;

        if resident {
            // Resident: each projection uploaded once to its own VRAM slice (repacked
            // into the layout its quant lane reads); no offload metadata.
            let vram = |i: usize| -> Result<CudaSlice<u8>, String> {
                let (rows, cols) = projection_shapes[i];
                let repacked =
                    repack_for_lane(projections[i], quants[i], rows, cols, self.k.q1_tiled)?;
                s.clone_htod(&repacked).map_err(|e| format!("htod: {e}"))
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
                post_attn_norm: None,
                post_ffn_norm: None,
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
            .map(|(i, b)| {
                let (rows, cols) = projection_shapes[i];
                repack_for_lane(b, quants[i], rows, cols, self.k.q1_tiled)
            })
            .collect::<Result<_, _>>()?;
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
            post_attn_norm: None,
            post_ffn_norm: None,
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
        let repacked = repack_for_lane(
            output_weight,
            output_quant,
            self.vocab,
            self.hidden,
            self.k.q1_tiled,
        )?;
        self.output_weight = s.clone_htod(&repacked).map_err(|e| format!("htod: {e}"))?;
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
            d_decay: af(num_v_heads)?,
            d_ssm_mix: af(value_dim)?,
            d_qgate: af(2 * self.q_width)?,
            d_gate_attn: af(self.q_width)?,
        });
        Ok(())
    }

    /// Install the gemma3 (windowed-attention arch) per-layer schedule. Call
    /// AFTER every `set_layer_located`, and pair it with
    /// [`Self::set_layer_gemma3_norms`] for each layer plus a per-token
    /// [`Self::upload_alt_rope`].
    ///
    /// `use_alt_rope` and `window` are indexed by OWNED layer and must both be
    /// exactly `self.layers.len()` long — a short or long schedule would
    /// silently mis-key which layers slide, which is the one failure mode that
    /// produces fluent, plausible, wrong text rather than an error, so it is a
    /// typed refusal rather than a clamp.
    pub fn set_gemma3(
        &mut self,
        use_alt_rope: Vec<bool>,
        window: Vec<Option<usize>>,
        ffn_geglu: bool,
    ) -> Result<(), String> {
        let n = self.layers.len();
        if use_alt_rope.len() != n || window.len() != n {
            return Err(format!(
                "gemma3 schedule length mismatch: {} layers resident but use_alt_rope={} window={}; \
                 refusing rather than running an unkeyed windowed forward",
                n,
                use_alt_rope.len(),
                window.len()
            ));
        }
        let s = self.k.stream.clone();
        let half = self.rope_dim / 2;
        let af = |n: usize| -> Result<CudaSlice<f32>, String> {
            s.alloc_zeros::<f32>(n).map_err(|e| format!("alloc: {e}"))
        };
        self.gemma3 = Some(Gemma3Gpu {
            use_alt_rope,
            window,
            d_cos_alt: af(half.max(1))?,
            d_sin_alt: af(half.max(1))?,
            ffn_geglu,
        });
        Ok(())
    }

    /// Upload one gemma3 layer's sandwich post-norms. `layer` indexes the OWNED
    /// range, matching the order `set_layer_located` pushed them in.
    pub fn set_layer_gemma3_norms(
        &mut self,
        layer: usize,
        post_attn_norm: &[f32],
        post_ffn_norm: &[f32],
    ) -> Result<(), String> {
        let s = self.k.stream.clone();
        let l = self.layers.get_mut(layer).ok_or_else(|| {
            format!("gemma3 post-norms for layer {layer}: no such resident layer")
        })?;
        l.post_attn_norm = Some(
            s.clone_htod(post_attn_norm)
                .map_err(|e| format!("htod: {e}"))?,
        );
        l.post_ffn_norm = Some(
            s.clone_htod(post_ffn_norm)
                .map_err(|e| format!("htod: {e}"))?,
        );
        Ok(())
    }

    /// Upload this token's ALT (local-theta) rope tables for a gemma3 session.
    /// The primary (global-theta) pair still rides `forward_token`'s `cos`/`sin`
    /// arguments; sliding layers read these instead. No-op when the engine
    /// carries no gemma3 schedule.
    pub fn upload_alt_rope(&mut self, cos_alt: &[f32], sin_alt: &[f32]) -> Result<(), String> {
        let s = self.k.stream.clone();
        let Some(g) = self.gemma3.as_mut() else {
            return Ok(());
        };
        s.memcpy_htod(cos_alt, &mut g.d_cos_alt)
            .map_err(|e| format!("htod alt cos: {e}"))?;
        s.memcpy_htod(sin_alt, &mut g.d_sin_alt)
            .map_err(|e| format!("htod alt sin: {e}"))?;
        Ok(())
    }

    /// True when this engine is running a windowed (gemma3) schedule. Consulted
    /// by the paths that have NO window support — batched prefill, batched and
    /// tree speculative verify, and CUDA graph capture — so they decline rather
    /// than silently evaluating a windowed model full-causal.
    pub fn is_windowed(&self) -> bool {
        self.gemma3.is_some()
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
                    .alloc_zeros::<u8>(1)
                    .map_err(|e| format!("kv placeholder: {e}"))?;
                self.cache_v[li] = s
                    .alloc_zeros::<u8>(1)
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
        resident_ffn: bool,
        conv_dim: usize,
        d_conv: usize,
        nv: usize,
        d_state: usize,
    ) -> Result<(), String> {
        let s = self.k.stream.clone();
        let value_dim = self
            .qwen35
            .as_ref()
            .ok_or_else(|| "set_layer_ssm_qwen35 called before set_qwen35".to_string())?
            .value_dim;
        let up_u8 =
            |b: &[u8], q: ProjQuant, rows: usize, cols: usize| -> Result<CudaSlice<u8>, String> {
                let repacked = repack_for_lane(b, q, rows, cols, self.k.q1_tiled)?;
                s.clone_htod(&repacked).map_err(|e| format!("htod: {e}"))
            };
        let up_f = |b: &[f32]| -> Result<CudaSlice<f32>, String> {
            s.clone_htod(b).map_err(|e| format!("htod: {e}"))
        };
        let ph = || s.clone_htod(&[0u8]).map_err(|e| format!("htod: {e}"));
        let ssm = SsmResident {
            wqkv: up_u8(wqkv, ssm_quants[0], conv_dim, self.hidden)?,
            wqkv_gate: up_u8(wqkv_gate, ssm_quants[1], value_dim, self.hidden)?,
            beta: up_u8(beta, ssm_quants[2], nv, self.hidden)?,
            alpha: up_u8(alpha, ssm_quants[3], nv, self.hidden)?,
            ssm_out: up_u8(ssm_out, ssm_quants[4], self.hidden, value_dim)?,
            quants: ssm_quants,
            conv1d: up_f(conv1d)?,
            dt_bias: up_f(dt_bias)?,
            a: up_f(a)?,
            ssm_norm: up_f(ssm_norm)?,
        };
        let (gate_g, up_g, down_g, offloaded) = if resident_ffn {
            (
                up_u8(gate, ffn_quants[0], self.ffn_dim, self.hidden)?,
                up_u8(up, ffn_quants[1], self.ffn_dim, self.hidden)?,
                up_u8(down, ffn_quants[2], self.hidden, self.ffn_dim)?,
                None,
            )
        } else {
            // Capacity path for large qwen35 Prism rows: the recurrent mixer and
            // state remain resident, while the three large FFN projections stream
            // through the existing per-layer offload scratch. Slots q/k/v/o are
            // empty because the SSM branch reads its dedicated resident weights;
            // the shared FFN tail consumes slots gate/up/down exactly as a full
            // attention layer does.
            let ctx = &self.k.ctx;
            let repacked = [
                repack_for_lane(
                    gate,
                    ffn_quants[0],
                    self.ffn_dim,
                    self.hidden,
                    self.k.q1_tiled,
                )?,
                repack_for_lane(
                    up,
                    ffn_quants[1],
                    self.ffn_dim,
                    self.hidden,
                    self.k.q1_tiled,
                )?,
                repack_for_lane(
                    down,
                    ffn_quants[2],
                    self.hidden,
                    self.ffn_dim,
                    self.k.q1_tiled,
                )?,
            ];
            let mut off = [0usize; 8];
            for (i, bytes) in repacked.iter().enumerate() {
                let slot = i + 4;
                off[slot + 1] = (off[slot] + bytes.len() + 15) & !15;
            }
            let mut packed = vec![0u8; off[7]];
            for (i, bytes) in repacked.iter().enumerate() {
                let slot = i + 4;
                packed[off[slot]..off[slot] + bytes.len()].copy_from_slice(bytes);
            }
            let pinned = CacheablePinned::from_bytes(ctx, &packed)?;
            (
                ph()?,
                ph()?,
                ph()?,
                Some(OffloadedLayer { host: pinned, off }),
            )
        };
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
            // qwen35 SSM layers are never windowed; gemma3 has no SSM layers.
            post_attn_norm: None,
            post_ffn_norm: None,
            offloaded,
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
        // A windowed (gemma3) schedule is declined here, which is the single
        // choke point for batched prefill AND batched speculative verify: the
        // batched/flash kernels (`attention_batched`, `attn_sk_*`,
        // `launch_attention_flash_prefill`) carry NO window parameter, so they
        // would evaluate a sliding layer full-causal — fluent, plausible, wrong.
        // The windowed lane instead prefills token-by-token through the
        // per-token forward, which is already what `session_prefill_chunk_tokens`
        // forces arch-wide for windowed archs.
        !self.is_windowed()
            && !self.is_offloaded()
            && self.layers.iter().all(|layer| {
                matches!(&layer.kind, LayerKind::Full)
                    && layer.quants.iter().all(|q| q.supports_batched())
            })
    }

    fn batched_layer_token_cap(&self) -> usize {
        if self.layers.iter().any(|layer| {
            layer.quants.contains(&ProjQuant::Q1_0)
                || matches!(&layer.kind, LayerKind::Ssm(ssm) if ssm.quants.contains(&ProjQuant::Q1_0))
        }) {
            let max = if self.k.fast_q1 {
                MAX_PRISM_PREFILL_K
            } else {
                2
            };
            return std::env::var("CAMELID_PRISM_BATCH_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(max)
                .clamp(1, max);
        }
        let has_kquant = self
            .layers
            .iter()
            .any(|layer| layer.quants.iter().any(|q| q.needs_q8k()));
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

    /// Largest prefill chunk this model's batched kernels can actually launch.
    ///
    /// Every batched GEMM stages `k` token rows of activations in shared memory, so `k` is bounded
    /// by the portable 46 KiB budget the launchers use. Exceeding it is not a slowdown, it is a
    /// failure: `launch_kquant_gemm_batched` asserts, and `launch_gemm_batched` builds a launch
    /// config the driver rejects. This is the ceiling any operator override is clamped to.
    fn prefill_token_cap_limit(&self) -> usize {
        const SHARED_BUDGET: usize = 46 * 1024;
        if self.layers.iter().any(|layer| {
            layer.quants.contains(&ProjQuant::Q1_0)
                || matches!(&layer.kind, LayerKind::Ssm(ssm) if ssm.quants.contains(&ProjQuant::Q1_0))
        }) {
            return MAX_PRISM_PREFILL_K;
        }
        let max_cols = self.hidden.max(self.q_width).max(self.ffn_dim);
        let has_kquant = self
            .layers
            .iter()
            .any(|layer| layer.quants.iter().any(|q| q.needs_q8k()));
        if has_kquant {
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
            return (SHARED_BUDGET / (n_sb * (260 + aux_lanes * 4)).max(1)).max(1);
        }
        let max_bpr = max_cols / 32;
        (SHARED_BUDGET / (max_bpr * 4).max(1)).max(1)
    }

    /// Maximum token batch size for prompt prefill.
    ///
    /// Defaults to the shared batched-stack cap, so the shipped prefill path is unchanged. Chunk
    /// size and the flash attention kernel are independent levers and are measured separately:
    /// `CAMELID_CUDA_PREFILL_BATCH_TOKENS` requests a chunk, clamped to what this lane's kernels
    /// can launch (see `prefill_token_cap_limit`).
    fn batched_prefill_token_cap(&self) -> usize {
        match std::env::var("CAMELID_CUDA_PREFILL_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(k) => k.clamp(1, self.prefill_token_cap_limit()),
            None => self.batched_layer_token_cap(),
        }
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
            || (!self.is_windowed()
                && !self.is_offloaded()
                && self.qwen35.is_some()
                && self.layers.iter().all(|layer| {
                    layer.quants.iter().all(|q| q.supports_batched())
                        && match &layer.kind {
                            LayerKind::Full => true,
                            LayerKind::Ssm(ssm) => ssm.quants.iter().all(|q| q.supports_batched()),
                        }
                }))
    }

    /// Default policy after same-host Windows/WDDM benchmarking. Q8_0 batching
    /// remains a clear win; the parity-correct Q4_K/Q6_K tiles stay opt-in until
    /// a device shows a sustained gain over the mature serial GEMVs.
    pub fn prefers_batched_prefill(&self) -> bool {
        self.supports_batched_prefill()
            && self.layers.iter().all(|layer| {
                layer
                    .quants
                    .iter()
                    .all(|q| matches!(q, ProjQuant::Q8_0 | ProjQuant::Q1_0))
            })
    }

    /// Whether linear speculative verification can use the batched stack,
    /// including its final vocabulary projection.
    pub fn supports_batched_verify(&self) -> bool {
        self.supports_batched_layer_stack() && self.output_quant.supports_batched()
    }

    /// Tree verification still uses the legacy Q8-only layer stack.
    pub fn supports_tree_verify(&self) -> bool {
        // Windowed (gemma3) declined for the same reason as the linear batched
        // stack: `attention_tree_batched` has no window parameter.
        !self.is_windowed()
            && !self.is_offloaded()
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
        let s = self.k.stream.clone();
        if self.kv_quant == crate::model::KvCacheQuantization::Q8_0 {
            let blocks_per_head = hd / 32;
            let bytes_per_head = blocks_per_head * 34;
            let mut k_bytes = vec![0u8; position * bytes_per_head];
            let mut v_bytes = vec![0u8; position * bytes_per_head];
            let mut q_blocks = vec![crate::tensor::kv_quant::BlockQ8_0::default(); blocks_per_head];
            for h in 0..n_kv {
                let hsrc = h * position * hd;
                for p in 0..position {
                    let p_src = hsrc + p * hd;
                    crate::tensor::kv_quant::quantize_row_q8_0(
                        &ck[p_src..p_src + hd],
                        &mut q_blocks,
                    );
                    let q_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(q_blocks.as_ptr() as *const u8, bytes_per_head)
                    };
                    k_bytes[p * bytes_per_head..(p + 1) * bytes_per_head].copy_from_slice(q_bytes);

                    crate::tensor::kv_quant::quantize_row_q8_0(
                        &cv[p_src..p_src + hd],
                        &mut q_blocks,
                    );
                    let q_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(q_blocks.as_ptr() as *const u8, bytes_per_head)
                    };
                    v_bytes[p * bytes_per_head..(p + 1) * bytes_per_head].copy_from_slice(q_bytes);
                }
                let gdst = h * max_pos * bytes_per_head;
                let span = position * bytes_per_head;
                let mut vk = self.cache_k[layer].slice_mut(gdst..gdst + span);
                s.memcpy_htod(&k_bytes, &mut vk)
                    .map_err(|e| format!("seed htod k: {e}"))?;
                let mut vv = self.cache_v[layer].slice_mut(gdst..gdst + span);
                s.memcpy_htod(&v_bytes, &mut vv)
                    .map_err(|e| format!("seed htod v: {e}"))?;
            }
        } else {
            let span = position * hd;
            for h in 0..n_kv {
                let hsrc = h * span; // host: head h's [0,position) block
                let gdst = h * max_pos * hd * 2; // gpu: head h's base byte offset
                let kbits: Vec<u16> = ck[hsrc..hsrc + span]
                    .iter()
                    .map(|&x| crate::inference::f32_to_f16_bits(x))
                    .collect();
                let k_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(kbits.as_ptr() as *const u8, kbits.len() * 2)
                };
                let mut vk = self.cache_k[layer].slice_mut(gdst..gdst + span * 2);
                s.memcpy_htod(k_bytes, &mut vk)
                    .map_err(|e| format!("seed htod k: {e}"))?;
                let vbits: Vec<u16> = cv[hsrc..hsrc + span]
                    .iter()
                    .map(|&x| crate::inference::f32_to_f16_bits(x))
                    .collect();
                let v_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(vbits.as_ptr() as *const u8, vbits.len() * 2)
                };
                let mut vv = self.cache_v[layer].slice_mut(gdst..gdst + span * 2);
                s.memcpy_htod(v_bytes, &mut vv)
                    .map_err(|e| format!("seed htod v: {e}"))?;
            }
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
        if self.kv_quant == crate::model::KvCacheQuantization::Q8_0 {
            let blocks_per_head = hd / 32;
            let bytes_per_head = blocks_per_head * 34;
            let span = n_positions * bytes_per_head;
            let mut k_bytes = vec![0u8; n_kv * span];
            let mut v_bytes = vec![0u8; n_kv * span];
            for h in 0..n_kv {
                let gsrc = h * max_pos * bytes_per_head;
                s.memcpy_dtoh(
                    &self.cache_k[layer].slice(gsrc..gsrc + span),
                    &mut k_bytes[h * span..(h + 1) * span],
                )
                .map_err(|e| format!("read_kv_layer K dtoh: {e}"))?;
                s.memcpy_dtoh(
                    &self.cache_v[layer].slice(gsrc..gsrc + span),
                    &mut v_bytes[h * span..(h + 1) * span],
                )
                .map_err(|e| format!("read_kv_layer V dtoh: {e}"))?;
            }
            self.k
                .ctx
                .synchronize()
                .map_err(|e| format!("read_kv_layer sync: {e}"))?;
            let mut k_out = vec![0.0f32; n_kv * n_positions * hd];
            let mut v_out = vec![0.0f32; n_kv * n_positions * hd];
            for h in 0..n_kv {
                for p in 0..n_positions {
                    let src_idx = h * span + p * bytes_per_head;
                    let dst_idx = (h * n_positions + p) * hd;
                    let qk_blocks: &[crate::tensor::kv_quant::BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            k_bytes[src_idx..].as_ptr()
                                as *const crate::tensor::kv_quant::BlockQ8_0,
                            blocks_per_head,
                        )
                    };
                    crate::tensor::kv_quant::dequantize_row_q8_0(
                        qk_blocks,
                        &mut k_out[dst_idx..dst_idx + hd],
                    );
                    let qv_blocks: &[crate::tensor::kv_quant::BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            v_bytes[src_idx..].as_ptr()
                                as *const crate::tensor::kv_quant::BlockQ8_0,
                            blocks_per_head,
                        )
                    };
                    crate::tensor::kv_quant::dequantize_row_q8_0(
                        qv_blocks,
                        &mut v_out[dst_idx..dst_idx + hd],
                    );
                }
            }
            Ok((k_out, v_out))
        } else {
            let span = n_positions * hd;
            let mut k_bits = vec![0u16; n_kv * span];
            let mut v_bits = vec![0u16; n_kv * span];
            for h in 0..n_kv {
                let gsrc = h * max_pos * hd * 2;
                let k_bytes_mut: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        k_bits[h * span..].as_mut_ptr() as *mut u8,
                        span * 2,
                    )
                };
                s.memcpy_dtoh(
                    &self.cache_k[layer].slice(gsrc..gsrc + span * 2),
                    k_bytes_mut,
                )
                .map_err(|e| format!("read_kv_layer K dtoh: {e}"))?;
                let v_bytes_mut: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        v_bits[h * span..].as_mut_ptr() as *mut u8,
                        span * 2,
                    )
                };
                s.memcpy_dtoh(
                    &self.cache_v[layer].slice(gsrc..gsrc + span * 2),
                    v_bytes_mut,
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
        let bonsai27b_fused_projections = self.bonsai27b_fused_projections
            && ov_streams.is_none()
            && self.hidden == 5_120
            && self.ffn_dim == 17_408
            && self.q_width == 6_144
            && self.kv_width == 1_024
            && self.qwen35.as_ref().is_some_and(|q| {
                q.conv_dim == 10_240 && q.value_dim == 6_144 && q.num_v_heads == 48
            });
        let bonsai27b_popc = self.bonsai27b_popc && bonsai27b_fused_projections;
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
            let attn_need_q8_0 = lq[..n_attn_consumers]
                .iter()
                .any(|q| q.needs_q8_0(self.k.fast_q1));
            let attn_need_q8k = lq[..n_attn_consumers].iter().any(|q| q.needs_q8k());
            let attn_need_f32 = lq[..n_attn_consumers]
                .iter()
                .any(|q| q.is_prism_low_bit(self.k.fast_q1));
            let attn_q1 = self.k.fast_q1
                && lq[..n_attn_consumers]
                    .iter()
                    .all(|lane| *lane == ProjQuant::Q1_0);
            if attn_q1 {
                launch_prism_rms_norm_q8_batched(
                    &s,
                    &self.k.prism_rms_norm_q8_batched,
                    &self.d_hidden,
                    &self.layers[li].attn_norm,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    self.hidden,
                    self.eps,
                    1,
                )
                .map_err(map)?;
            }
            if !attn_q1 && attn_need_f32 {
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
            }
            if !attn_q1 && attn_need_q8_0 {
                if fused && !attn_need_f32 {
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
                    if !attn_need_f32 {
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
                    }
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
            if !attn_q1 && attn_need_q8k {
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
            if bonsai27b_popc {
                launch_prism_q8_32_bitplanes_qsum(
                    &s,
                    &self.k.prism_q8_32_bitplanes_qsum,
                    &self.d_in_quants,
                    &mut self.d_in_bitplanes,
                    &mut self.d_in_qsums,
                    hb,
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
                    // Q,K,V (qwen35 full-attn: wq is fused [query|gate]). The
                    // opt-in Bonsai kernel only merges these read-only projections;
                    // deinterleave and every attention epilogue stay unchanged.
                    let fused_full = bonsai27b_fused_projections
                        && lq[..3].iter().all(|lane| *lane == ProjQuant::Q1_0);
                    if fused_full {
                        let q = self.qwen35.as_mut().expect("Bonsai fusion requires qwen35");
                        if bonsai27b_popc {
                            launch_prism_q1t128_q8_popc_fused_full_bonsai27b(
                                &s,
                                &self.k.prism_q1t128_q8_popc_fused_full_bonsai27b,
                                &self.d_in_bitplanes,
                                &self.d_in_qsums,
                                &self.d_in_scales,
                                &wq,
                                &wk,
                                &wv,
                                &mut q.d_qgate,
                                &mut self.d_k,
                                &mut self.d_v,
                            )
                            .map_err(map)?;
                        } else {
                            launch_prism_q1t128_fused_full_bonsai27b(
                                &s,
                                &self.k.prism_q1t128_fused_full_bonsai27b,
                                &self.d_in_quants,
                                &self.d_in_scales,
                                &wq,
                                &wk,
                                &wv,
                                &mut q.d_qgate,
                                &mut self.d_k,
                                &mut self.d_v,
                            )
                            .map_err(map)?;
                        }
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
                        if let Some(q) = self.qwen35.as_mut() {
                            dispatch_gemv(
                                &s,
                                &self.k,
                                lq[0],
                                &self.d_normed,
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
                                &self.d_normed,
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
                            &self.d_normed,
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
                            &self.d_normed,
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
                    }
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
                    // gemma3 per-layer schedule, read once as plain Copy values so
                    // no borrow of `self.gemma3` is held across the launches below.
                    // `None`/`false` for every other architecture.
                    let gemma3_window: Option<usize> =
                        self.gemma3.as_ref().and_then(|g| g.window[li]);
                    let gemma3_alt_rope: bool =
                        self.gemma3.as_ref().is_some_and(|g| g.use_alt_rope[li]);
                    // RoPE on Q and K.
                    //
                    // gemma3 dual-theta: a SLIDING layer ropes from the local-θ
                    // (ALT) tables, a GLOBAL layer from the primary. Every other
                    // arch has one table set and always takes the primary, so
                    // this selection is a no-op for them (byte-identical).
                    let pairing = if self.split_half_pairing { 1i32 } else { 0i32 };
                    let (rope_cos, rope_sin) = if gemma3_alt_rope {
                        let g = self.gemma3.as_ref().expect("alt rope implies gemma3 state");
                        (&g.d_cos_alt, &g.d_sin_alt)
                    } else {
                        (&self.d_cos, &self.d_sin)
                    };
                    launch_rope(
                        &s,
                        &self.k.rope,
                        &mut self.d_q,
                        rope_cos,
                        rope_sin,
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
                        rope_cos,
                        rope_sin,
                        self.n_kv_heads,
                        self.head_dim,
                        self.rope_dim,
                        pairing,
                    )
                    .map_err(map)?;
                    // KV write
                    let is_q8_kv = self.kv_quant == crate::model::KvCacheQuantization::Q8_0;
                    if is_q8_kv {
                        launch_kv_scatter_q8_0(
                            s_k,
                            &self.k.kv_scatter_q8_0,
                            &self.d_k,
                            &mut self.cache_k[li],
                            &self.d_position,
                            self.n_kv_heads,
                            self.head_dim,
                            self.max_pos,
                        )
                        .map_err(map)?;
                        launch_kv_scatter_q8_0(
                            s_v,
                            &self.k.kv_scatter_q8_0,
                            &self.d_v,
                            &mut self.cache_v[li],
                            &self.d_position,
                            self.n_kv_heads,
                            self.head_dim,
                            self.max_pos,
                        )
                        .map_err(map)?;
                    } else {
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
                    }
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
                    //
                    // gemma3 SLIDING layers take `attention_decode_sw` instead,
                    // which masks to the last `window` keys. They never take
                    // split-K, and that costs nothing: the split-K kernels carry
                    // no window parameter at all, and a sliding layer can never
                    // reach the threshold anyway — the 1B row's window is 512 and
                    // SPLITK_THRESHOLD is also 512, so a sliding layer's attended
                    // key count is bounded at exactly the point split-K would
                    // start to pay. GLOBAL gemma3 layers fall through to the
                    // unchanged logic below.
                    if let Some(window) = gemma3_window {
                        let attn_fn = if is_q8_kv {
                            &self.k.attention_sw_q8_0
                        } else {
                            &self.k.attention_sw
                        };
                        launch_attention_sw(
                            &s,
                            attn_fn,
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
                            window,
                            &mut self.d_sk_scores,
                        )
                        .map_err(map)?;
                    } else if !graph_capture && attn_shared > SPLITK_THRESHOLD {
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
                            is_q8_kv,
                        )
                        .map_err(map)?;
                    } else {
                        let attn_fn = if is_q8_kv {
                            &self.k.attention_q8_0
                        } else {
                            &self.k.attention
                        };
                        launch_attention(
                            &s,
                            attn_fn,
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
                            &mut self.d_sk_scores,
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
                    //
                    // gemma3 sandwich: the post-attention norm sits BETWEEN the O
                    // projection and the residual add —
                    //   h = h + post_attention_norm(o_proj(attn))
                    // — so the fused gemv+residual kernel (`output[row] += acc`)
                    // cannot be used on these layers. Take the unfused shape and
                    // insert the norm. Everything else keeps the fusion.
                    if self.gemma3.is_some() {
                        // Keyed on the SESSION being windowed, not on the norm
                        // being present: a gemma3 layer that reached here without
                        // its post-attention norm bound would otherwise silently
                        // fall through to the Llama path and drop the norm — the
                        // exact failure gemma2 is fail-closed for. Demand it.
                        let post_attn =
                            self.layers[li].post_attn_norm.as_ref().ok_or_else(|| {
                                format!(
                                    "gemma3 layer {li}: post_attention_norm is not bound on the \
                                 resident layer; refusing rather than running the Llama-shaped \
                                 forward, which would silently drop the sandwich norm"
                                )
                            })?;
                        // H5 keeps windowed-arch resident admission pinned to Q8_0,
                        // so a non-Q8_0 gemma3 layer must never reach here. Fail
                        // loudly: the alternative is a silently wrong forward.
                        if lq[3] != ProjQuant::Q8_0 {
                            return Err(format!(
                                "gemma3 layer {li}: O projection is {:?}, but the windowed resident \
                                 lane is pinned to Q8_0; refusing rather than running an \
                                 unvalidated quant through the sandwich-norm path",
                                lq[3]
                            ));
                        }
                        launch_quantize(
                            &s,
                            &self.k.quantize,
                            &self.d_attn,
                            &mut self.d_in_quants,
                            &mut self.d_in_scales,
                            qb,
                        )
                        .map_err(map)?;
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
                        launch_rmsnorm(
                            &s,
                            &self.k.rms_norm,
                            &self.d_proj,
                            post_attn,
                            &mut self.d_post,
                            self.hidden,
                            self.eps,
                        )
                        .map_err(map)?;
                        launch_residual(
                            &s,
                            &self.k.residual_add,
                            &mut self.d_hidden,
                            &self.d_post,
                            self.hidden,
                        )
                        .map_err(map)?;
                    } else if lq[3].needs_q8_0(self.k.fast_q1)
                        || lq[3].is_prism_low_bit(self.k.fast_q1)
                    {
                        if lq[3].needs_q8_0(self.k.fast_q1) {
                            launch_quantize(
                                &s,
                                &self.k.quantize,
                                &self.d_attn,
                                &mut self.d_in_quants,
                                &mut self.d_in_scales,
                                qb,
                            )
                            .map_err(map)?;
                        }
                        if fused {
                            dispatch_gemv(
                                &s,
                                &self.k,
                                lq[3],
                                &self.d_attn,
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
                        } else {
                            dispatch_gemv(
                                &s,
                                &self.k,
                                lq[3],
                                &self.d_attn,
                                &self.d_in_scales,
                                &self.d_in_quants,
                                &self.d_q8k_scales,
                                &self.d_q8k_quants,
                                &wo,
                                self.hidden,
                                self.q_width,
                                &mut self.d_proj,
                                0,
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
                            &self.d_attn,
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
                    let fused_ssm = bonsai27b_fused_projections
                        && sq[..4].iter().all(|lane| *lane == ProjQuant::Q1_0);
                    if fused_ssm {
                        if bonsai27b_popc {
                            launch_prism_q1t128_q8_popc_fused_ssm_bonsai27b(
                                &s,
                                &self.k.prism_q1t128_q8_popc_fused_ssm_bonsai27b,
                                &self.d_in_bitplanes,
                                &self.d_in_qsums,
                                &self.d_in_scales,
                                &ssm.wqkv.as_view(),
                                &ssm.wqkv_gate.as_view(),
                                &ssm.beta.as_view(),
                                &ssm.alpha.as_view(),
                                &mut q.d_qkv,
                                &mut q.d_z,
                                &mut self.d_k,
                                &mut self.d_v,
                            )
                            .map_err(map)?;
                        } else {
                            launch_prism_q1t128_fused_ssm_bonsai27b(
                                &s,
                                &self.k.prism_q1t128_fused_ssm_bonsai27b,
                                &self.d_in_quants,
                                &self.d_in_scales,
                                &ssm.wqkv.as_view(),
                                &ssm.wqkv_gate.as_view(),
                                &ssm.beta.as_view(),
                                &ssm.alpha.as_view(),
                                &mut q.d_qkv,
                                &mut q.d_z,
                                &mut self.d_k,
                                &mut self.d_v,
                            )
                            .map_err(map)?;
                        }
                    } else {
                        dispatch_gemv(
                            &s,
                            &self.k,
                            sq[0],
                            &self.d_normed,
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
                            &self.d_normed,
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
                            &self.d_normed,
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
                            &self.d_normed,
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
                    }
                    launch_ssm_gates(
                        &s,
                        &self.k.ssm_gates,
                        &self.d_k,
                        &self.d_v,
                        &ssm.dt_bias,
                        &ssm.a,
                        &mut q.d_beta,
                        &mut q.d_decay,
                        nv,
                    )
                    .map_err(map)?;
                    let bonsai_ssm_q8 =
                        self.k.fast_q1 && ds == 128 && d_conv == 4 && sq[4] == ProjQuant::Q1_0;
                    if bonsai_ssm_q8 {
                        launch_qwen35_ssm_conv1d_d4_batched(
                            &s,
                            &self.k.qwen35_ssm_conv1d_d4_batched,
                            &ssm.conv1d,
                            &q.d_qkv,
                            &mut self.ssm_conv_state[li],
                            &mut q.d_conv_out,
                            conv_dim,
                            1,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_qk_l2_norm_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_qk_l2_norm_d128_batched,
                            &mut q.d_conv_out,
                            conv_dim,
                            key_dim,
                            nk,
                            1,
                            self.eps,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_delta_rule_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_delta_rule_d128_batched,
                            &mut self.ssm_state[li],
                            &q.d_conv_out,
                            &q.d_beta,
                            &q.d_decay,
                            &mut q.d_ssm_mix,
                            nk,
                            nv,
                            key_dim,
                            value_dim,
                            conv_dim,
                            1,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_rmsnorm_gate_q8_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_rmsnorm_gate_q8_d128_batched,
                            &q.d_ssm_mix,
                            &q.d_z,
                            &ssm.ssm_norm,
                            &mut self.d_in_quants,
                            &mut self.d_in_scales,
                            nv,
                            value_dim,
                            1,
                            self.eps,
                        )
                        .map_err(map)?;
                    } else {
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
                            &q.d_decay,
                            &ssm.ssm_norm,
                            &mut q.d_ssm_mix,
                            ds,
                            nk,
                            nv,
                            self.eps,
                        )
                        .map_err(map)?;
                    }
                    // ssm_out projection + residual into d_hidden. Quantize the SSM mix to
                    // the activation format ssm_out's lane reads: Q8_K for a K-quant
                    // (Q4_K/Q6_K) ssm_out, Q8_0 otherwise.
                    if !bonsai_ssm_q8 && sq[4].needs_q8k() {
                        launch_quantize_q8k(
                            &s,
                            &self.k.quantize_q8k,
                            &q.d_ssm_mix,
                            &mut self.d_q8k_quants,
                            &mut self.d_q8k_scales,
                            value_dim / 256,
                        )
                        .map_err(map)?;
                    } else if !bonsai_ssm_q8 && sq[4].needs_q8_0(self.k.fast_q1) {
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
                        &q.d_ssm_mix,
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
            let ffn_need_q8_0 =
                lq[4].needs_q8_0(self.k.fast_q1) || lq[5].needs_q8_0(self.k.fast_q1);
            let ffn_need_q8k = lq[4].needs_q8k() || lq[5].needs_q8k();
            let ffn_need_f32 =
                lq[4].is_prism_low_bit(self.k.fast_q1) || lq[5].is_prism_low_bit(self.k.fast_q1);
            let ffn_q1 = self.k.fast_q1 && lq[4..6].iter().all(|lane| *lane == ProjQuant::Q1_0);
            if ffn_q1 {
                launch_prism_rms_norm_q8_batched(
                    &s,
                    &self.k.prism_rms_norm_q8_batched,
                    &self.d_hidden,
                    &self.layers[li].ffn_norm,
                    &mut self.d_in_quants,
                    &mut self.d_in_scales,
                    self.hidden,
                    self.eps,
                    1,
                )
                .map_err(map)?;
            }
            if !ffn_q1 && ffn_need_f32 {
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
            }
            if !ffn_q1 && ffn_need_q8_0 {
                if fused && !ffn_need_f32 {
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
                    if !ffn_need_f32 {
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
                    }
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
            if !ffn_q1 && ffn_need_q8k {
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
            if bonsai27b_popc {
                launch_prism_q8_32_bitplanes_qsum(
                    &s,
                    &self.k.prism_q8_32_bitplanes_qsum,
                    &self.d_in_quants,
                    &mut self.d_in_bitplanes,
                    &mut self.d_in_qsums,
                    hb,
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
            let fused_ffn = bonsai27b_fused_projections
                && !ffn_overlap
                && lq[4..6].iter().all(|lane| *lane == ProjQuant::Q1_0);
            if fused_ffn {
                if bonsai27b_popc {
                    launch_prism_q1t128_q8_popc_fused_ffn_bonsai27b(
                        &s,
                        &self.k.prism_q1t128_q8_popc_fused_ffn_bonsai27b,
                        &self.d_in_bitplanes,
                        &self.d_in_qsums,
                        &self.d_in_scales,
                        &wgate,
                        &wup,
                        &mut self.d_gate,
                        &mut self.d_up,
                    )
                    .map_err(map)?;
                } else {
                    launch_prism_q1t128_fused_ffn_bonsai27b(
                        &s,
                        &self.k.prism_q1t128_fused_ffn_bonsai27b,
                        &self.d_in_quants,
                        &self.d_in_scales,
                        &wgate,
                        &wup,
                        &mut self.d_gate,
                        &mut self.d_up,
                    )
                    .map_err(map)?;
                }
            } else {
                dispatch_gemv(
                    &s,
                    &self.k,
                    lq[4],
                    &self.d_normed,
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
                    &self.d_normed,
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
            }
            // Join: silu on main reads d_up, so it waits on the side up gemv. The
            // wait also transitively orders silu's write of the shared d_in_*
            // (/d_q8k_*) scratch after the up gemv's read of it (WAR), and the next
            // layer's attn norm write after that.
            if ffn_overlap {
                let o = self.overlap.as_ref().expect("overlap state");
                o.ev_up.record(s_up).map_err(map)?;
                s.wait(&o.ev_up).map_err(map)?;
            }
            // gemma3 FFN: GeGLU instead of SiLU, and the post-FFN sandwich norm
            // between the down projection and the residual —
            //   h = h + post_ffw_norm(down_proj(gelu_tanh(gate) * up))
            // — so this branch is the unfused shape for the same reason the O
            // projection above is. `geglu_mul` has the identical
            // `(gate, up, out, n)` signature and elementwise launch geometry as
            // `silu_mul`, so it reuses `launch_silu_mul`'s launcher; only the
            // CUfunction differs.
            if let Some(ffn_geglu) = self.gemma3.as_ref().map(|g| g.ffn_geglu) {
                // Two separate model properties, read from two separate places:
                // the ACTIVATION comes from parsed metadata (`Gemma3Metadata
                // .ffn_geglu`), the NORM from the bound weights. Keying one off
                // the other would mean a future windowed arch with SiLU silently
                // got GeGLU, so they stay independent and both are demanded.
                let post_ffn = self.layers[li].post_ffn_norm.as_ref().ok_or_else(|| {
                    format!(
                        "gemma3 layer {li}: post_ffn_norm is not bound on the resident layer; \
                         refusing rather than running the Llama-shaped forward, which would \
                         silently drop the sandwich norm"
                    )
                })?;
                let act_fn = if ffn_geglu {
                    &self.k.geglu_mul
                } else {
                    &self.k.silu_mul
                };
                if lq[6] != ProjQuant::Q8_0 {
                    return Err(format!(
                        "gemma3 layer {li}: down projection is {:?}, but the windowed resident \
                         lane is pinned to Q8_0; refusing rather than running an unvalidated \
                         quant through the sandwich-norm path",
                        lq[6]
                    ));
                }
                launch_silu_mul(
                    &s,
                    act_fn,
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
                launch_rmsnorm(
                    &s,
                    &self.k.rms_norm,
                    &self.d_proj,
                    post_ffn,
                    &mut self.d_post,
                    self.hidden,
                    self.eps,
                )
                .map_err(map)?;
                launch_residual(
                    &s,
                    &self.k.residual_add,
                    &mut self.d_hidden,
                    &self.d_post,
                    self.hidden,
                )
                .map_err(map)?;
            } else if lq[6].needs_q8_0(self.k.fast_q1) || lq[6].is_prism_low_bit(self.k.fast_q1) {
                if self.k.fast_q1 && lq[6] == ProjQuant::Q1_0 {
                    launch_prism_silu_mul_q8_batched(
                        &s,
                        &self.k.prism_silu_mul_q8_batched,
                        &self.d_gate,
                        &self.d_up,
                        &mut self.d_in_quants,
                        &mut self.d_in_scales,
                        fb,
                    )
                    .map_err(map)?;
                } else if lq[6].is_prism_low_bit(self.k.fast_q1) {
                    launch_silu_mul(
                        &s,
                        &self.k.silu_mul,
                        &self.d_gate,
                        &self.d_up,
                        &mut self.d_ffn_act,
                        self.ffn_dim,
                    )
                    .map_err(map)?;
                } else if fused {
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
                    dispatch_gemv(
                        &s,
                        &self.k,
                        lq[6],
                        &self.d_ffn_act,
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
                } else {
                    dispatch_gemv(
                        &s,
                        &self.k,
                        lq[6],
                        &self.d_ffn_act,
                        &self.d_in_scales,
                        &self.d_in_quants,
                        &self.d_q8k_scales,
                        &self.d_q8k_quants,
                        &wdown,
                        self.hidden,
                        self.ffn_dim,
                        &mut self.d_proj,
                        0,
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
                    &self.d_ffn_act,
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
            // gemma3→CUDA campaign, localization instrument. Mirrors the runnable
            // oracle's `CAMELID_LAYER_DUMP` (src/runnable/model.rs): append this
            // layer's hidden state so the two lanes can be diffed layer by layer.
            // Costs a full sync + D2H per layer, so it is strictly a debugging
            // lane — the var is never set in production or in any gate.
            if let Some(path) = std::env::var("CAMELID_LAYER_DUMP").ok().as_deref() {
                s.synchronize().map_err(map)?;
                let mut h = vec![0f32; self.hidden];
                s.memcpy_dtoh(&self.d_hidden, &mut h).map_err(map)?;
                let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                // Keyed on POSITION, not on call ordinal: the CUDA lane runs a
                // load-time warmup the CPU lane has no counterpart for, so ordinal
                // alignment silently compares different tokens.
                let line = format!(
                    "{position}\t{li}\t{l2:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                    h[0], h[1], h[2], h[3]
                );
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }

        if !compute_logits {
            return Ok(());
        }
        // final norm + output (lm_head) projection -> d_logits (no argmax / no sync).
        // Produce the activation in the lm_head lane's format (Q6_K for Q4_K_M).
        if self.k.fast_q1 && self.output_quant == ProjQuant::Q1_0 {
            launch_prism_rms_norm_q8_batched(
                &s,
                &self.k.prism_rms_norm_q8_batched,
                &self.d_hidden,
                &self.final_norm,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                self.hidden,
                self.eps,
                1,
            )
            .map_err(map)?;
        } else if self.output_quant.is_prism_low_bit(self.k.fast_q1) {
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
        } else if self.output_quant.needs_q8_0(self.k.fast_q1) {
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
        if bonsai27b_popc && self.output_quant == ProjQuant::Q1_0 {
            launch_prism_q8_32_bitplanes_qsum(
                &s,
                &self.k.prism_q8_32_bitplanes_qsum,
                &self.d_in_quants,
                &mut self.d_in_bitplanes,
                &mut self.d_in_qsums,
                hb,
            )
            .map_err(map)?;
            launch_prism_q1t128_q8_popc_gemv_m16(
                &s,
                &self.k.prism_q1t128_q8_popc_gemv_m16,
                &self.d_in_bitplanes,
                &self.d_in_qsums,
                &self.d_in_scales,
                &out_w,
                self.vocab,
                self.hidden,
                &mut self.d_logits,
                0,
            )
            .map_err(map)?;
        } else {
            dispatch_gemv(
                &s,
                &self.k,
                self.output_quant,
                &self.d_normed,
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
        }
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
        // gemma3 is excluded from graph capture alongside qwen35. The captured
        // graph freezes each layer's kernel identity and args, and a windowed
        // schedule alternates between `attention_decode_sw` and the full-causal
        // kernel per layer; nothing here has been proven under capture, and the
        // safe outcome of getting it wrong is not a crash but a wrong token.
        if cuda_graphs_enabled() && self.qwen35.is_none() && self.gemma3.is_none() {
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

    /// Run the layer stack for one token and return the resulting hidden state
    /// (PRE final-norm, `hidden` floats), without the lm_head projection.
    ///
    /// This is the hidden-state-threading form of [`Self::forward_token`], and it
    /// exists for windowed (gemma3) sessions: they prefill token-by-token through
    /// the per-token forward — the batched prefill has no window mask — and each
    /// prefill token needs its hidden state threaded back, not logits. The Metal
    /// lane has carried this shape since #560 (`ResidentTokenOut::Data` with
    /// `compute_logits == false`); this is its CUDA twin, so the two GPU lanes
    /// agree on what a windowed prefill step returns.
    ///
    /// Costs one D2H copy of `hidden` floats plus the usual single sync. Only the
    /// windowed lane calls it; every other architecture keeps `compute_logits ==
    /// false` on the CPU path exactly as before.
    pub fn forward_token_hidden(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward hidden: {e}");
        let s = self.k.stream.clone();
        self.forward_pass(embedding, cos, sin, position, scale, false, false, false)?;
        let mut out = vec![0f32; self.hidden];
        s.memcpy_dtoh(&self.d_hidden, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(out)
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
        let disabled = matches!(
            std::env::var("CAMELID_PRISM_CUDA_NO_DEVICE_DECODE")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        !disabled && self.embd_table.is_some()
    }

    /// One fully device-side forward: the input token id comes from
    /// `prev_step` of the on-device output ring (None => the host-fed
    /// d_token_in slot, uploaded here from `host_token`), the rope row is
    /// selected on-device, the embedding row is gathered/dequantized on-device
    /// (bit-identical elementwise mirror of the CPU dequant), then the standard
    /// per-launch forward runs; when `out_step` is Some the greedy argmax lands
    /// in d_out_tokens[out_step] — which the NEXT call can consume directly.
    /// `kv_position` is the physical cache slot/attention depth, while
    /// `rope_position` selects the logical RoPE row. They are equal for text and
    /// intentionally diverge after a multimodal image span.
    /// NO host synchronization: the caller syncs once per readback chunk.
    pub(crate) fn forward_token_device(
        &mut self,
        prev_step: Option<usize>,
        host_token: Option<u32>,
        kv_position: usize,
        rope_position: usize,
        scale: f32,
        out_step: Option<usize>,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda device-decode: {e}");
        let s = self.k.stream.clone();
        s.memcpy_htod(&[kv_position as i32], &mut self.d_position)
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
                rope_position,
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
                ProjQuant::Q1_0 => &this.k.embed_gather_q1_0,
                ProjQuant::Q2_0G64 => &this.k.embed_gather_q2_0_g64,
                ProjQuant::Q2_0G128 => &this.k.embed_gather_q2_0_g128,
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
        if self.device_forward_graph_eligible() && out_step.is_some() {
            self.forward_device_pass_graphed(kv_position, scale)?;
        } else {
            self.forward_pass(
                &[],
                &[],
                &[],
                kv_position,
                scale,
                out_step.is_some(),
                false,
                true,
            )?;
        }
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

    fn device_forward_graph_eligible(&self) -> bool {
        qwen35_device_graphs_enabled()
            && cuda_manual_stream_order_enabled()
            && self.artifact == ResidentCudaArtifact::PrismBonsai27bQ1
            && self.qwen35.is_some()
            && self.n_layers == 64
            && self.hidden == 5120
            && self.ffn_dim == 17_408
            && self.q_width == 6144
            && self.offload.is_none()
            && self.overlap.is_none()
            && self.gemma3.is_none()
            && std::env::var_os("CAMELID_LAYER_DUMP").is_none()
    }

    /// Capture/replay only the stable Qwen3.5 forward body. The device loop has
    /// already populated hidden/RoPE/position buffers. Keeping gather/select and
    /// the variable output-ring view outside lets one graph serve every token and
    /// both the text and multimodal physical/logical position clocks.
    fn forward_device_pass_graphed(&mut self, position: usize, scale: f32) -> Result<(), String> {
        use cudarc::driver::sys;
        let map = |step: &'static str| {
            move |e: cudarc::driver::DriverError| format!("cuda device graph {step}: {e}")
        };
        let s = self.k.stream.clone();

        if self.device_forward_graph.is_none() {
            // All input preparation precedes capture and must be complete. With
            // cudarc automatic event tracking disabled, the captured kernels have
            // no dependency on events recorded outside this capture graph.
            s.synchronize().map_err(map("pre-capture sync"))?;
            s.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(map("begin"))?;
            let recorded = self.forward_pass(&[], &[], &[], position, scale, true, true, true);
            let flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0) };
            let captured = s.end_capture(flags);
            recorded?;
            match captured.map_err(map("end"))? {
                Some(graph) => {
                    graph.upload().map_err(map("upload"))?;
                    self.device_forward_graph = Some(SendCudaGraph(graph));
                    if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
                        eprintln!(
                            "[cuda-graph] Qwen3.5 device forward captured: layer stack + norm + lm_head"
                        );
                    }
                }
                None => return Err("device forward graph capture produced no graph".into()),
            }
        }

        self.device_forward_graph
            .as_ref()
            .expect("device forward graph present")
            .0
            .launch()
            .map_err(map("replay"))
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
        let output_bmma = if self.k.fast_q1
            && self.output_quant == ProjQuant::Q1_0
            && !prism_bmma_shape_enabled(&self.k, hidden, k)
        {
            launch_prism_rms_norm_q8_batched(
                &s,
                &self.k.prism_rms_norm_q8_batched,
                &sc.vh,
                &self.final_norm,
                &mut sc.viq,
                &mut sc.vis,
                hidden,
                eps,
                k,
            )
            .map_err(map)?;
            false
        } else {
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
                &mut sc.vibits,
                &mut sc.vibscales,
                &mut sc.viqk,
                &mut sc.visk,
                hidden,
                k,
                std::slice::from_ref(&self.output_quant),
            )
            .map_err(map)?
        };
        dispatch_gemm_batched(
            &s,
            &self.k,
            self.output_quant,
            &sc.vn,
            &sc.vis,
            &sc.viq,
            &sc.vibits,
            &sc.vibscales,
            output_bmma,
            &sc.visk,
            &sc.viqk,
            &self.output_weight,
            vocab,
            hidden,
            k,
            &mut sc.vlogits,
            0,
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

    fn ensure_prefill_scratch(&mut self, cap: usize) -> Result<(), String> {
        if self.prefill_scratch.is_some() {
            return Ok(());
        }
        self.prefill_scratch = Some(self.alloc_scratch(cap, false)?);
        Ok(())
    }

    /// Allocate a `VerifyScratch` sized to `cap` rows (`cap * dim`). Used by the
    /// linear verify (`cap = MAX_VERIFY_K`), tree verify (`cap = TREE_MAX_NODES`),
    /// and prompt prefill (`cap = batch_cap`).
    fn alloc_verify_scratch(&self, cap: usize) -> Result<VerifyScratch, String> {
        self.alloc_scratch(cap, true)
    }

    fn alloc_scratch(&self, cap: usize, include_logits: bool) -> Result<VerifyScratch, String> {
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
            // One byte per activation stored as u32 bitplanes, plus one scale
            // per 128 values. Round up here; dispatch still requires an exact
            // 128-divisible projection width before it can consume the lane.
            vibits: st
                .alloc_zeros::<u32>(mk * max_in.div_ceil(128) * 32)
                .map_err(|e| format!("verify alloc: {e}"))?,
            vibscales: af(mk * max_in.div_ceil(128))?,
            vq: af(mk * q_width)?,
            vk: af(mk * kv_width)?,
            vv: af(mk * kv_width)?,
            vattn: af(mk * q_width)?,
            vproj: af(mk * hidden)?,
            vgate: af(mk * ffn_dim)?,
            vup: af(mk * ffn_dim)?,
            vact: af(mk * ffn_dim)?,
            vlogits: if include_logits {
                af(mk * vocab)?
            } else {
                af(1)?
            },
            vsamp: if include_logits {
                st.alloc_zeros::<u32>(mk)
                    .map_err(|e| format!("verify alloc: {e}"))?
            } else {
                st.alloc_zeros::<u32>(1)
                    .map_err(|e| format!("verify alloc: {e}"))?
            },
            vcos: af(mk * half)?,
            vsin: af(mk * half)?,
        })
    }

    /// Run the batched layer stack for `k` tokens (`1..=MAX_VERIFY_K` or prefill chunk cap) at consecutive
    /// positions `[base_position, base_position+k)`. Reads the staged per-token input
    /// from `sc.vh` / `sc.vcos` / `sc.vsin`, writes each token's K/V into the cache,
    /// and leaves the post-final-layer hidden state in `sc.vh`. The caller stages the
    /// inputs and (for `verify_batch`) projects logits afterward.
    ///
    /// This is the single source of truth for the batched forward, shared by
    /// `verify_batch` (speculative decode) and `prefill_batched`.
    /// On the default path (and for all `verify_batch` calls, where `flash_ok` is false),
    /// it is bit-identical to the serial `forward_pass` per token: each supported batched
    /// projection reproduces its decode GEMV's integer decomposition and ordered fp32 sum,
    /// and the batched norm/RoPE/scatter/attention kernels match their serial counterparts.
    /// When opt-in flash prefill is enabled (`CAMELID_FLASH_PREFILL=1`, prefill only), the fused
    /// online-softmax attention kernel preserves greedy token-parity while eliminating intermediate
    /// DRAM scratch.
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
            let mixer_lane_count = if matches!(&layer.kind, LayerKind::Ssm(_)) {
                4
            } else {
                3
            };
            let mixer_q1 = self.k.fast_q1
                && lq[..mixer_lane_count]
                    .iter()
                    .all(|lane| *lane == ProjQuant::Q1_0);
            let mixer_bmma = if mixer_q1 && !prism_bmma_shape_enabled(&self.k, hidden, k) {
                launch_prism_rms_norm_q8_batched(
                    &s,
                    &self.k.prism_rms_norm_q8_batched,
                    &sc.vh,
                    &layer.attn_norm,
                    &mut sc.viq,
                    &mut sc.vis,
                    hidden,
                    eps,
                    k,
                )
                .map_err(map)?;
                false
            } else {
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
                    &mut sc.vibits,
                    &mut sc.vibscales,
                    &mut sc.viqk,
                    &mut sc.visk,
                    hidden,
                    k,
                    &lq[..mixer_lane_count],
                )
                .map_err(map)?
            };
            match &layer.kind {
                LayerKind::Full => {
                    if self.qwen35.is_some() {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            lq[0],
                            &sc.vn,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            mixer_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &layer.q,
                            2 * q_width,
                            hidden,
                            k,
                            &mut sc.vgate,
                            0,
                        )
                        .map_err(map)?;
                        launch_deinterleave_qgate_batched(
                            &s,
                            &self.k.deinterleave_qgate_batched,
                            &sc.vgate,
                            &mut sc.vq,
                            &mut sc.vup,
                            n_heads,
                            head_dim,
                            k,
                        )
                        .map_err(map)?;
                    } else {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            lq[0],
                            &sc.vn,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            mixer_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &layer.q,
                            q_width,
                            hidden,
                            k,
                            &mut sc.vq,
                            0,
                        )
                        .map_err(map)?;
                    }
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        lq[1],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &layer.k,
                        kv_width,
                        hidden,
                        k,
                        &mut sc.vk,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        lq[2],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &layer.v,
                        kv_width,
                        hidden,
                        k,
                        &mut sc.vv,
                        0,
                    )
                    .map_err(map)?;
                    if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
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
                    let is_q8_kv = self.kv_quant == crate::model::KvCacheQuantization::Q8_0;
                    if is_q8_kv {
                        launch_kv_scatter_batched_q8_0(
                            &s,
                            &self.k.kv_scatter_batched_q8_0,
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
                        launch_kv_scatter_batched_q8_0(
                            &s,
                            &self.k.kv_scatter_batched_q8_0,
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
                    } else {
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
                    }
                    if !is_q8_kv
                        && flash_ok
                        && flash_prefill_enabled()
                        && q_width == n_heads * head_dim
                        && head_dim <= 256
                    {
                        launch_attention_flash_prefill(
                            &s,
                            &self.k,
                            &sc.vq,
                            &self.cache_k[li],
                            &self.cache_v[li],
                            &mut sc.vattn,
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
                        let attn_batched_fn = if is_q8_kv {
                            &self.k.attention_batched_q8_0
                        } else {
                            &self.k.attention_batched
                        };
                        launch_attention_batched(
                            &s,
                            attn_batched_fn,
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
                            &mut self.d_verify_scores,
                        )
                        .map_err(map)?;
                    }
                    if self.qwen35.is_some() {
                        launch_sigmoid_mul(
                            &s,
                            &self.k.sigmoid_mul,
                            &mut sc.vattn,
                            &sc.vup,
                            k * q_width,
                        )
                        .map_err(map)?;
                    }
                    let attention_bmma = quantize_batched_for_lanes(
                        &s,
                        &self.k,
                        &sc.vattn,
                        &mut sc.viq,
                        &mut sc.vis,
                        &mut sc.vibits,
                        &mut sc.vibscales,
                        &mut sc.viqk,
                        &mut sc.visk,
                        q_width,
                        k,
                        &lq[3..4],
                    )
                    .map_err(map)?;
                    if self.k.fast_q1 && lq[3] == ProjQuant::Q1_0 {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            lq[3],
                            &sc.vattn,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            attention_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &layer.o,
                            hidden,
                            q_width,
                            k,
                            &mut sc.vh,
                            1,
                        )
                        .map_err(map)?;
                    } else {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            lq[3],
                            &sc.vattn,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            attention_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &layer.o,
                            hidden,
                            q_width,
                            k,
                            &mut sc.vproj,
                            0,
                        )
                        .map_err(map)?;
                        launch_residual(
                            &s,
                            &self.k.residual_add,
                            &mut sc.vh,
                            &sc.vproj,
                            k * hidden,
                        )
                        .map_err(map)?;
                    }
                }
                LayerKind::Ssm(ssm) => {
                    let q = self
                        .qwen35
                        .as_ref()
                        .expect("SSM layer requires qwen35 state");
                    let (ds, nk, nv, key_dim, value_dim, conv_dim, d_conv) = (
                        q.d_state,
                        q.num_k_heads,
                        q.num_v_heads,
                        q.key_dim,
                        q.value_dim,
                        q.conv_dim,
                        q.d_conv,
                    );
                    let sq = ssm.quants;
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        sq[0],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &ssm.wqkv,
                        conv_dim,
                        hidden,
                        k,
                        &mut sc.vgate,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        sq[1],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &ssm.wqkv_gate,
                        value_dim,
                        hidden,
                        k,
                        &mut sc.vact,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        sq[2],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &ssm.beta,
                        nv,
                        hidden,
                        k,
                        &mut sc.vk,
                        0,
                    )
                    .map_err(map)?;
                    dispatch_gemm_batched(
                        &s,
                        &self.k,
                        sq[3],
                        &sc.vn,
                        &sc.vis,
                        &sc.viq,
                        &sc.vibits,
                        &sc.vibscales,
                        mixer_bmma,
                        &sc.visk,
                        &sc.viqk,
                        &ssm.alpha,
                        nv,
                        hidden,
                        k,
                        &mut sc.vv,
                        0,
                    )
                    .map_err(map)?;
                    launch_ssm_gates_batched(
                        &s,
                        &self.k.ssm_gates_batched,
                        &sc.vk,
                        &sc.vv,
                        &ssm.dt_bias,
                        &ssm.a,
                        &mut sc.vq,
                        &mut sc.vattn,
                        nv,
                        k,
                    )
                    .map_err(map)?;
                    let bonsai_ssm_q8 =
                        self.k.fast_q1 && ds == 128 && d_conv == 4 && sq[4] == ProjQuant::Q1_0;
                    if bonsai_ssm_q8 {
                        // Bonsai's hot SSM lane: fixed-geometry convolution,
                        // paired Q/K norm, and register-resident recurrence.
                        // No generic SSM kernel is allowed on this path.
                        launch_qwen35_ssm_conv1d_d4_batched(
                            &s,
                            &self.k.qwen35_ssm_conv1d_d4_batched,
                            &ssm.conv1d,
                            &sc.vgate,
                            &mut self.ssm_conv_state[li],
                            &mut sc.vup,
                            conv_dim,
                            k,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_qk_l2_norm_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_qk_l2_norm_d128_batched,
                            &mut sc.vup,
                            conv_dim,
                            key_dim,
                            nk,
                            k,
                            eps,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_delta_rule_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_delta_rule_d128_batched,
                            &mut self.ssm_state[li],
                            &sc.vup,
                            &sc.vq,
                            &sc.vattn,
                            &mut sc.vgate,
                            nk,
                            nv,
                            key_dim,
                            value_dim,
                            conv_dim,
                            k,
                        )
                        .map_err(map)?;
                        launch_qwen35_ssm_rmsnorm_gate_q8_d128_batched(
                            &s,
                            &self.k.qwen35_ssm_rmsnorm_gate_q8_d128_batched,
                            &sc.vgate,
                            &sc.vact,
                            &ssm.ssm_norm,
                            &mut sc.viq,
                            &mut sc.vis,
                            nv,
                            value_dim,
                            k,
                            eps,
                        )
                        .map_err(map)?;
                    } else {
                        launch_ssm_conv1d_batched(
                            &s,
                            &self.k.ssm_conv1d_batched,
                            &ssm.conv1d,
                            &sc.vgate,
                            &mut self.ssm_conv_state[li],
                            &mut sc.vup,
                            conv_dim,
                            d_conv,
                            k,
                        )
                        .map_err(map)?;
                        launch_ssm_l2_norm_per_head_batched(
                            &s,
                            &self.k.ssm_l2_norm_per_head_batched,
                            &mut sc.vup,
                            conv_dim,
                            0,
                            nk,
                            ds,
                            k,
                            eps,
                        )
                        .map_err(map)?;
                        launch_ssm_l2_norm_per_head_batched(
                            &s,
                            &self.k.ssm_l2_norm_per_head_batched,
                            &mut sc.vup,
                            conv_dim,
                            key_dim,
                            nk,
                            ds,
                            k,
                            eps,
                        )
                        .map_err(map)?;
                        launch_ssm_delta_rule_batched(
                            &s,
                            &self.k.ssm_delta_rule_batched,
                            &mut self.ssm_state[li],
                            &sc.vup,
                            &sc.vact,
                            &sc.vq,
                            &sc.vattn,
                            &ssm.ssm_norm,
                            &mut sc.vgate,
                            ds,
                            nk,
                            nv,
                            key_dim,
                            value_dim,
                            conv_dim,
                            k,
                            eps,
                        )
                        .map_err(map)?;
                    }
                    let ssm_out_bmma = if !bonsai_ssm_q8 {
                        quantize_batched_for_lanes(
                            &s,
                            &self.k,
                            &sc.vgate,
                            &mut sc.viq,
                            &mut sc.vis,
                            &mut sc.vibits,
                            &mut sc.vibscales,
                            &mut sc.viqk,
                            &mut sc.visk,
                            value_dim,
                            k,
                            &sq[4..5],
                        )
                        .map_err(map)?
                    } else {
                        // The fused Bonsai gated-RMSNorm kernel materializes
                        // Q8/32 directly, so no f32 row exists to bit-pack.
                        false
                    };
                    if bonsai_ssm_q8 {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            sq[4],
                            &sc.vgate,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            ssm_out_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &ssm.ssm_out,
                            hidden,
                            value_dim,
                            k,
                            &mut sc.vh,
                            1,
                        )
                        .map_err(map)?;
                    } else {
                        dispatch_gemm_batched(
                            &s,
                            &self.k,
                            sq[4],
                            &sc.vgate,
                            &sc.vis,
                            &sc.viq,
                            &sc.vibits,
                            &sc.vibscales,
                            ssm_out_bmma,
                            &sc.visk,
                            &sc.viqk,
                            &ssm.ssm_out,
                            hidden,
                            value_dim,
                            k,
                            &mut sc.vn,
                            0,
                        )
                        .map_err(map)?;
                        launch_residual(&s, &self.k.residual_add, &mut sc.vh, &sc.vn, k * hidden)
                            .map_err(map)?;
                    }
                }
            }
            let ffn_q1 = self.k.fast_q1 && lq[4..6].iter().all(|lane| *lane == ProjQuant::Q1_0);
            let ffn_bmma = if ffn_q1 && !prism_bmma_shape_enabled(&self.k, hidden, k) {
                launch_prism_rms_norm_q8_batched(
                    &s,
                    &self.k.prism_rms_norm_q8_batched,
                    &sc.vh,
                    &layer.ffn_norm,
                    &mut sc.viq,
                    &mut sc.vis,
                    hidden,
                    eps,
                    k,
                )
                .map_err(map)?;
                false
            } else {
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
                    &mut sc.vibits,
                    &mut sc.vibscales,
                    &mut sc.viqk,
                    &mut sc.visk,
                    hidden,
                    k,
                    &lq[4..6],
                )
                .map_err(map)?
            };
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[4],
                &sc.vn,
                &sc.vis,
                &sc.viq,
                &sc.vibits,
                &sc.vibscales,
                ffn_bmma,
                &sc.visk,
                &sc.viqk,
                &layer.gate,
                ffn_dim,
                hidden,
                k,
                &mut sc.vgate,
                0,
            )
            .map_err(map)?;
            dispatch_gemm_batched(
                &s,
                &self.k,
                lq[5],
                &sc.vn,
                &sc.vis,
                &sc.viq,
                &sc.vibits,
                &sc.vibscales,
                ffn_bmma,
                &sc.visk,
                &sc.viqk,
                &layer.up,
                ffn_dim,
                hidden,
                k,
                &mut sc.vup,
                0,
            )
            .map_err(map)?;
            let down_q1 = self.k.fast_q1 && lq[6] == ProjQuant::Q1_0;
            let down_bmma = if down_q1 && !prism_bmma_shape_enabled(&self.k, ffn_dim, k) {
                launch_prism_silu_mul_q8_batched(
                    &s,
                    &self.k.prism_silu_mul_q8_batched,
                    &sc.vgate,
                    &sc.vup,
                    &mut sc.viq,
                    &mut sc.vis,
                    k * (ffn_dim / 32),
                )
                .map_err(map)?;
                false
            } else {
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
                    &mut sc.vibits,
                    &mut sc.vibscales,
                    &mut sc.viqk,
                    &mut sc.visk,
                    ffn_dim,
                    k,
                    &lq[6..7],
                )
                .map_err(map)?
            };
            if down_q1 {
                dispatch_gemm_batched(
                    &s,
                    &self.k,
                    lq[6],
                    &sc.vact,
                    &sc.vis,
                    &sc.viq,
                    &sc.vibits,
                    &sc.vibscales,
                    down_bmma,
                    &sc.visk,
                    &sc.viqk,
                    &layer.down,
                    hidden,
                    ffn_dim,
                    k,
                    &mut sc.vh,
                    1,
                )
                .map_err(map)?;
            } else {
                dispatch_gemm_batched(
                    &s,
                    &self.k,
                    lq[6],
                    &sc.vact,
                    &sc.vis,
                    &sc.viq,
                    &sc.vibits,
                    &sc.vibscales,
                    down_bmma,
                    &sc.visk,
                    &sc.viqk,
                    &layer.down,
                    hidden,
                    ffn_dim,
                    k,
                    &mut sc.vproj,
                    0,
                )
                .map_err(map)?;
                launch_residual(&s, &self.k.residual_add, &mut sc.vh, &sc.vproj, k * hidden)
                    .map_err(map)?;
            }
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
            let is_q8_kv = self.kv_quant == crate::model::KvCacheQuantization::Q8_0;
            if is_q8_kv {
                launch_kv_scatter_tree_batched_q8_0(
                    &s,
                    &self.k.kv_scatter_tree_batched_q8_0,
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
                launch_kv_scatter_tree_batched_q8_0(
                    &s,
                    &self.k.kv_scatter_tree_batched_q8_0,
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
            } else {
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
            }
            // Tree attention: dense prefix [0, base) + ancestor slots only.
            let attn_tree_fn = if is_q8_kv {
                &self.k.attention_tree_batched_q8_0
            } else {
                &self.k.attention_tree_batched
            };
            launch_attention_tree_batched(
                &s,
                attn_tree_fn,
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
                &mut self.d_verify_scores,
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
        let bytes_per_pos = if self.kv_quant == crate::model::KvCacheQuantization::Q8_0 {
            (head_dim / 32) * 34
        } else {
            head_dim * 2
        };
        let mut row = vec![0u8; bytes_per_pos];
        for (r, &node) in path.iter().enumerate() {
            if node == r {
                continue; // identity (the whole linear case) — slot already correct
            }
            let src_pos = base + node;
            let dst_pos = base + r;
            for li in 0..self.n_layers {
                for kv_head in 0..n_kv {
                    let row_base = kv_head * max_pos * bytes_per_pos;
                    let src = row_base + src_pos * bytes_per_pos;
                    let dst = row_base + dst_pos * bytes_per_pos;
                    // K
                    s.memcpy_dtoh(&self.cache_k[li].slice(src..src + bytes_per_pos), &mut row)
                        .map_err(map)?;
                    s.memcpy_htod(
                        &row,
                        &mut self.cache_k[li].slice_mut(dst..dst + bytes_per_pos),
                    )
                    .map_err(map)?;
                    // V
                    s.memcpy_dtoh(&self.cache_v[li].slice(src..src + bytes_per_pos), &mut row)
                        .map_err(map)?;
                    s.memcpy_htod(
                        &row,
                        &mut self.cache_v[li].slice_mut(dst..dst + bytes_per_pos),
                    )
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
        let batch_cap = self.batched_prefill_token_cap();
        self.ensure_prefill_scratch(batch_cap)?;
        let s = self.k.stream.clone();
        let mut sc = self.prefill_scratch.take().expect("allocated above");
        let mut base = 0usize;
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
        self.prefill_scratch = Some(sc);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
