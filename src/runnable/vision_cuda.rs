//! Streaming CUDA execution for the Prism/Qwen3-VL image projector.
//!
//! The 27B language graph already occupies most of a 6 GiB card. Keeping the
//! 630 MiB projector resident beside it is needlessly fragile, so this lane
//! retains only activations and uploads one matrix at a time. CUDA stream order
//! keeps each weight alive through its contraction and frees it before later
//! layers need the same memory.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx};

use crate::gguf::GgufTensorType;

use super::vision::{PrismVisionInput, PrismVisionModel, VisionMat};

const KERNELS: &str = r#"
#ifdef CAMELID_HAS_WMMA
#include <mma.h>
#endif

__device__ __forceinline__ float vision_half_to_float(unsigned short bits) {
    unsigned int sign = ((unsigned int)bits & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)bits >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int)bits & 0x03ffu;
    unsigned int value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            value = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return __uint_as_float(value);
}

// IEEE-754 round-to-nearest-even f32 -> f16 conversion. Keeping this helper
// header-free lets the portable scalar module continue to compile on older
// CUDA installations while the Ampere path uses the same Q8_0 scale contract
// as Camelid's language kernels.
__device__ __forceinline__ unsigned short vision_float_to_half_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    int exponent = (int)((bits >> 23) & 0xffu);
    unsigned int mantissa = bits & 0x007fffffu;
    if (exponent == 0xff) {
        return (unsigned short)(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }
    int half_exponent = exponent - 127 + 15;
    if (half_exponent >= 0x1f) return (unsigned short)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        unsigned int normalized = mantissa | 0x00800000u;
        int shift = 14 - half_exponent;
        unsigned short half_mantissa = (unsigned short)(normalized >> shift);
        unsigned int round_bit = 1u << (shift - 1);
        if ((normalized & round_bit) != 0u &&
            ((normalized & (round_bit - 1u)) != 0u || (half_mantissa & 1u) != 0u)) {
            half_mantissa = (unsigned short)(half_mantissa + 1);
        }
        return (unsigned short)(sign | half_mantissa);
    }
    unsigned short half = (unsigned short)(sign
        | ((unsigned short)half_exponent << 10) | (unsigned short)(mantissa >> 13));
    if ((mantissa & 0x00001000u) != 0u &&
        ((mantissa & 0x00000fffu) != 0u || (half & 1u) != 0u)) {
        half = (unsigned short)(half + 1);
    }
    return half;
}

__device__ __forceinline__ float vision_f16_round(float value) {
    return vision_half_to_float(vision_float_to_half_bits(value));
}

// Token-major activation quantization for the Q8_0 projector matrices. Each
// thread owns one 32-value block. Quants use the unrounded inverse and the
// stored scale is rounded through f16, matching GGML Q8_0.
extern "C" __global__ void vision_quantize_q8_0(
    const float* __restrict__ input,
    signed char* __restrict__ quants,
    float* __restrict__ scales,
    int total_blocks
) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= total_blocks) return;
    const float* values = input + (long)block * 32;
    float maximum = 0.0f;
    #pragma unroll
    for (int index = 0; index < 32; ++index) maximum = fmaxf(maximum, fabsf(values[index]));
    float unrounded = maximum / 127.0f;
    scales[block] = vision_f16_round(unrounded);
    float inverse = unrounded == 0.0f ? 0.0f : 1.0f / unrounded;
    signed char* output = quants + (long)block * 32;
    #pragma unroll
    for (int index = 0; index < 32; ++index) {
        float value = rintf(values[index] * inverse);
        value = fminf(127.0f, fmaxf(-128.0f, value));
        output[index] = (signed char)value;
    }
}

extern "C" __global__ void vision_convert_f32_to_f16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int total
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) output[index] = vision_float_to_half_bits(input[index]);
}

extern "C" __global__ void vision_matmul_f32(
    const float* input, const unsigned char* wire, float* output,
    int tokens, int cols, int rows
) {
    long thread = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long index = thread >> 5;
    int lane = (int)thread & 31;
    long total = (long)tokens * rows;
    if (index >= total) return;
    int token = (int)(index / rows);
    int row = (int)(index - (long)token * rows);
    const float* x = input + (long)token * cols;
    const float* w = reinterpret_cast<const float*>(wire) + (long)row * cols;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) sum += x[col] * w[col];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) output[index] = sum;
}

extern "C" __global__ void vision_matmul_f16(
    const float* input, const unsigned char* wire, float* output,
    int tokens, int cols, int rows
) {
    long thread = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long index = thread >> 5;
    int lane = (int)thread & 31;
    long total = (long)tokens * rows;
    if (index >= total) return;
    int token = (int)(index / rows);
    int row = (int)(index - (long)token * rows);
    const float* x = input + (long)token * cols;
    const unsigned short* w = reinterpret_cast<const unsigned short*>(wire) + (long)row * cols;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) sum += x[col] * vision_half_to_float(w[col]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) output[index] = sum;
}

// Dense F16 projector GEMM for the 27 FFN-down matrices. Image-token rows and
// all certified Bonsai dimensions are multiples of 16. One warp owns one
// 16x16 (output-row x token) tile and accumulates through K in f32 while the
// F16 weights and converted activations contract on tensor cores.
extern "C" __global__ void vision_matmul_f16_wmma(
    const unsigned short* __restrict__ input,
    const unsigned char* __restrict__ wire,
    float* __restrict__ output,
    int tokens, int cols, int rows
) {
#if defined(CAMELID_HAS_WMMA) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int work = blockIdx.x * warps_per_block + warp;
    int row_tiles = rows >> 4;
    int token_tiles = tokens >> 4;
    if (work >= row_tiles * token_tiles) return;
    int token_tile = work / row_tiles;
    int row_tile = work - token_tile * row_tiles;
    int token_base = token_tile * 16;
    int row_base = row_tile * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> weights;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> activations;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators;
    wmma::fill_fragment(accumulators, 0.0f);
    const half* dense_weights = reinterpret_cast<const half*>(wire);
    const half* dense_input = reinterpret_cast<const half*>(input);
    for (int column = 0; column < cols; column += 16) {
        wmma::load_matrix_sync(
            weights, dense_weights + (long)row_base * cols + column, cols);
        wmma::load_matrix_sync(
            activations, dense_input + (long)token_base * cols + column, cols);
        wmma::mma_sync(accumulators, weights, activations, accumulators);
    }
    wmma::store_matrix_sync(
        output + (long)token_base * rows + row_base,
        accumulators,
        rows,
        wmma::mem_col_major);
#endif
}

extern "C" __global__ void vision_matmul_q8_0(
    const float* input, const unsigned char* wire, float* output,
    int tokens, int cols, int rows
) {
    long thread = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long index = thread >> 5;
    int lane = (int)thread & 31;
    long total = (long)tokens * rows;
    if (index >= total) return;
    int token = (int)(index / rows);
    int row = (int)(index - (long)token * rows);
    const float* x = input + (long)token * cols;
    int blocks = cols / 32;
    const unsigned char* wrow = wire + (long)row * blocks * 34;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const unsigned char* bytes = wrow + (long)block * 34;
        float scale = vision_half_to_float(*reinterpret_cast<const unsigned short*>(bytes));
        const signed char* quants = reinterpret_cast<const signed char*>(bytes + 2);
        const float* xb = x + block * 32;
        sum += xb[lane] * (float)quants[lane] * scale;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) output[index] = sum;
}

// Ampere/Turing Q8_0 x Q8_0 projector GEMM. A 256-thread CTA owns 128 output
// rows by 128 image tokens. Each warp owns a 16-token tile while all eight
// warps reuse the same 128x32 weight tile. The native m16n8k32 signed-int8 MMA
// contracts one quant block at a time; the per-block f16 scales are applied to
// its int32 result in f32. This replaces the scalar one-warp-per-output dot
// whose profiler cost dominates the Windows vision encoder.
extern "C" __global__ void vision_matmul_q8_0_imma(
    const signed char* __restrict__ input_quants,
    const float* __restrict__ input_scales,
    const unsigned char* __restrict__ wire,
    float* __restrict__ output,
    int tokens, int cols, int rows
) {
#if __CUDA_ARCH__ >= 750
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row_base = blockIdx.x * 128;
    int token_base = blockIdx.y * 128 + warp * 16;
    int blocks_per_row = cols >> 5;
    bool token_active = token_base < tokens;

    __shared__ __align__(32) signed char weight_tile[128 * 32];
    __shared__ __align__(32) signed char input_tile[8 * 16 * 32];
    __shared__ float weight_scales[128];
    __shared__ float activation_scales[128];
    float sums[64];
    #pragma unroll
    for (int index = 0; index < 64; ++index) sums[index] = 0.0f;

    for (int quant_block = 0; quant_block < blocks_per_row; ++quant_block) {
        if (threadIdx.x < 128) {
            int row = row_base + threadIdx.x;
            float scale = 0.0f;
            if (row < rows) {
                const unsigned char* block = wire
                    + ((long)row * blocks_per_row + quant_block) * 34;
                scale = vision_half_to_float((unsigned short)block[0]
                    | ((unsigned short)block[1] << 8));
            }
            weight_scales[threadIdx.x] = scale;
        }
        for (int index = threadIdx.x; index < 128 * 32; index += blockDim.x) {
            int row_in_tile = index >> 5;
            int column = index & 31;
            int row = row_base + row_in_tile;
            signed char value = 0;
            if (row < rows) {
                const unsigned char* block = wire
                    + ((long)row * blocks_per_row + quant_block) * 34;
                value = (signed char)block[2 + column];
            }
            weight_tile[index] = value;
        }
        #pragma unroll
        for (int token_column = 0; token_column < 16; ++token_column) {
            int token = token_base + token_column;
            input_tile[warp * 512 + token_column * 32 + lane] = token < tokens
                ? input_quants[(long)token * cols + quant_block * 32 + lane]
                : 0;
        }
        if (threadIdx.x < 128) {
            int token = blockIdx.y * 128 + threadIdx.x;
            activation_scales[threadIdx.x] = token < tokens
                ? input_scales[(long)token * blocks_per_row + quant_block]
                : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int row_group = 0; row_group < 8; ++row_group) {
            if (token_active && row_base + row_group * 16 < rows) {
                int a0, a1, a2, a3;
                const int* weight_rows = (const int*)(weight_tile + row_group * 16 * 32);
                const int* weight_source = weight_rows
                    + (lane % 16) * 8 + (lane / 16) * 4;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "l"(weight_source));

                const int* inputs = (const int*)(input_tile + warp * 512);
                #pragma unroll
                for (int token_half = 0; token_half < 2; ++token_half) {
                    const int* input_half = inputs + token_half * 8 * 8;
                    int b0 = input_half[(lane / 4) * 8 + (lane % 4)];
                    int b1 = input_half[(lane / 4) * 8 + (lane % 4) + 4];
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    int accumulators[4] = { c0, c1, c2, c3 };
                    #pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        int row_in_group = (item / 2) * 8 + lane / 4;
                        int token_column = token_half * 8
                            + (lane % 4) * 2 + (item % 2);
                        int row_in_tile = row_group * 16 + row_in_group;
                        int row = row_base + row_in_tile;
                        int token = token_base + token_column;
                        if (row < rows && token < tokens) {
                            sums[row_group * 8 + token_half * 4 + item] +=
                                (float)accumulators[item]
                                * weight_scales[row_in_tile]
                                * activation_scales[warp * 16 + token_column];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (token_active) {
        #pragma unroll
        for (int row_group = 0; row_group < 8; ++row_group) {
            #pragma unroll
            for (int token_half = 0; token_half < 2; ++token_half) {
                #pragma unroll
                for (int item = 0; item < 4; ++item) {
                    int row_in_group = (item / 2) * 8 + lane / 4;
                    int token_column = token_half * 8
                        + (lane % 4) * 2 + (item % 2);
                    int row = row_base + row_group * 16 + row_in_group;
                    int token = token_base + token_column;
                    if (row < rows && token < tokens) {
                        output[(long)token * rows + row] =
                            sums[row_group * 8 + token_half * 4 + item];
                    }
                }
            }
        }
    }
#endif
}

extern "C" __global__ void vision_patch_sum(
    const float* first, const float* second, const float* bias,
    const float* position, float* output, int width, int total
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) output[index] = first[index] + second[index] +
        bias[index % width] + position[index];
}

extern "C" __global__ void vision_layer_norm(
    const float* input, const float* weight, const float* bias, float* output,
    int width, float eps
) {
    extern __shared__ float scratch[];
    int token = blockIdx.x;
    int tid = threadIdx.x;
    long base = (long)token * width;
    float local = 0.0f;
    for (int index = tid; index < width; index += blockDim.x) local += input[base + index];
    scratch[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float mean = scratch[0] / (float)width;
    local = 0.0f;
    for (int index = tid; index < width; index += blockDim.x) {
        float centered = input[base + index] - mean;
        local += centered * centered;
    }
    scratch[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(scratch[0] / (float)width + eps);
    for (int index = tid; index < width; index += blockDim.x) {
        output[base + index] = (input[base + index] - mean) * inverse * weight[index] + bias[index];
    }
}

extern "C" __global__ void vision_add_bias(float* values, const float* bias, int width, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) values[index] += bias[index % width];
}

extern "C" __global__ void vision_bias_residual(
    const float* residual, const float* projected, const float* bias,
    float* output, int width, int total
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) output[index] = residual[index] + projected[index] + bias[index % width];
}

extern "C" __global__ void vision_bias_gelu(float* values, const float* bias, int width, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    float x = values[index] + bias[index % width];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    values[index] = 0.5f * x * (1.0f + tanhf(fminf(15.0f, fmaxf(-15.0f, inner))));
}

extern "C" __global__ void vision_rope(
    float* qkv, const float* cosine, const float* sine,
    int hidden, int heads, int head_dim, int tokens
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int per_token = heads * half;
    int total = tokens * per_token;
    if (index >= total) return;
    int token = index / per_token;
    int remainder = index - token * per_token;
    int head = remainder / half;
    int pair = remainder - head * half;
    float c = cosine[(long)token * half + pair];
    float s = sine[(long)token * half + pair];
    long token_base = (long)token * 3 * hidden;
    for (int qk = 0; qk < 2; ++qk) {
        long base = token_base + (long)qk * hidden + (long)head * head_dim;
        float x0 = qkv[base + pair];
        float x1 = qkv[base + pair + half];
        qkv[base + pair] = x0 * c - x1 * s;
        qkv[base + pair + half] = x0 * s + x1 * c;
    }
}

extern "C" __global__ void vision_attention(
    const float* qkv, float* output, int hidden, int heads,
    int head_dim, int tokens, float scale
) {
    extern __shared__ float scores[];
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= tokens || head >= heads) return;
    long qbase = (long)query * 3 * hidden + (long)head * head_dim;
    for (int key = tid; key < tokens; key += blockDim.x) {
        long kbase = (long)key * 3 * hidden + hidden + (long)head * head_dim;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) score += qkv[qbase + dim] * qkv[kbase + dim];
        scores[key] = score * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float maximum = -3.402823466e+38F;
        for (int key = 0; key < tokens; ++key) maximum = fmaxf(maximum, scores[key]);
        float sum = 0.0f;
        for (int key = 0; key < tokens; ++key) {
            float value = expf(scores[key] - maximum);
            scores[key] = value;
            sum += value;
        }
        scores[tokens] = 1.0f / sum;
    }
    __syncthreads();
    float inverse = scores[tokens];
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (int key = 0; key < tokens; ++key) {
            long vbase = (long)key * 3 * hidden + 2 * hidden + (long)head * head_dim;
            value += scores[key] * inverse * qkv[vbase + dim];
        }
        output[(long)query * hidden + (long)head * head_dim + dim] = value;
    }
}

// Prism vision flash attention. One warp owns a complete (query, head), keeps
// Q and the weighted-V accumulator in registers, and performs online softmax
// while walking K/V exactly once. This removes the scalar inner dot, the
// thread-0 max/sum bottleneck, and the token-sized shared score buffer from the
// portable oracle above. The certified projector has head_dim <= 256, so each
// lane owns at most eight coalesced dimensions.
extern "C" __global__ void vision_attention_online(
    const float* __restrict__ qkv, float* __restrict__ output,
    int hidden, int heads, int head_dim, int tokens, float scale
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int query = blockIdx.x * warps_per_block + warp;
    int head = blockIdx.y;
    if (query >= tokens || head >= heads) return;

    long qbase = (long)query * 3 * hidden + (long)head * head_dim;
    float qreg[8];
    float accum[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        int dim = lane + item * 32;
        qreg[item] = dim < head_dim ? qkv[qbase + dim] : 0.0f;
        accum[item] = 0.0f;
    }

    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    for (int key = 0; key < tokens; ++key) {
        long kbase = (long)key * 3 * hidden + hidden + (long)head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            int dim = lane + item * 32;
            if (dim < head_dim) dot += qreg[item] * qkv[kbase + dim];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);

        float alpha = 1.0f;
        float beta = 0.0f;
        if (lane == 0) {
            float score = dot * scale;
            if (score <= running_max) {
                beta = expf(score - running_max);
                running_sum += beta;
            } else {
                alpha = expf(running_max - score);
                beta = 1.0f;
                running_sum = running_sum * alpha + 1.0f;
                running_max = score;
            }
        }
        alpha = __shfl_sync(0xffffffffu, alpha, 0);
        beta = __shfl_sync(0xffffffffu, beta, 0);
        long vbase = (long)key * 3 * hidden + 2 * hidden + (long)head * head_dim;
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            int dim = lane + item * 32;
            if (dim < head_dim)
                accum[item] = accum[item] * alpha + beta * qkv[vbase + dim];
        }
    }

    float inverse = lane == 0 ? 1.0f / running_sum : 0.0f;
    inverse = __shfl_sync(0xffffffffu, inverse, 0);
    long outbase = (long)query * hidden + (long)head * head_dim;
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        int dim = lane + item * 32;
        if (dim < head_dim) output[outbase + dim] = accum[item] * inverse;
    }
}
"#;

pub(crate) struct CudaVisionEncoder {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    matmul_f32: CudaFunction,
    matmul_f16: CudaFunction,
    matmul_f16_wmma: Option<CudaFunction>,
    convert_f32_to_f16: CudaFunction,
    matmul_q8_0: CudaFunction,
    matmul_q8_0_imma: Option<CudaFunction>,
    quantize_q8_0: CudaFunction,
    patch_sum: CudaFunction,
    layer_norm: CudaFunction,
    add_bias: CudaFunction,
    bias_residual: CudaFunction,
    bias_gelu: CudaFunction,
    rope: CudaFunction,
    attention: CudaFunction,
    attention_online: Option<CudaFunction>,
    // Large projector matrices are immutable GGUF pages. On <=8 GiB cards we
    // retain them only after another substantial resident consumer (normally
    // the language engine) has already claimed memory; this prevents the first
    // image from stealing the budget needed to construct Bonsai-27B. Larger
    // cards cache immediately. Both cases preserve at least 1 GiB free.
    weight_cache: Vec<(usize, usize, CudaSlice<u8>)>,
}

impl CudaVisionEncoder {
    #[cfg(test)]
    pub(crate) fn new() -> Result<Self, String> {
        Self::new_with_strict_math(false)
    }

    pub(crate) fn new_with_strict_math(force_strict: bool) -> Result<Self, String> {
        let ordinal = crate::cuda::selected_device_ordinal();
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|_| "CUDA driver library not available".to_string())?
            .map_err(|error| format!("vision CudaContext::new({ordinal}): {error}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|error| format!("vision CUDA stream: {error}"))?;
        let cc_major = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        let tensor_core_q8 = cc_major > 7 || (cc_major == 7 && cc_minor >= 5);
        let strict = force_strict
            || std::env::var("CAMELID_PRISM_CUDA_STRICT").is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        let cuda_include = ["CUDA_PATH", "CUDA_HOME"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(std::path::PathBuf::from)
            .map(|root| root.join("include"))
            .chain([std::path::PathBuf::from("/usr/local/cuda/include")])
            .find(|include| include.join("mma.h").is_file());
        let tensor_core_f16 = cc_major >= 7 && cuda_include.is_some();
        let arch = if cc_major >= 8 {
            "compute_80"
        } else if tensor_core_q8 {
            "compute_75"
        } else if cc_major == 7 {
            "compute_70"
        } else {
            "compute_61"
        };
        let mut include_paths = Vec::new();
        let mut nvrtc_options = Vec::new();
        if let Some(include) = cuda_include {
            include_paths.push(include.to_string_lossy().into_owned());
            nvrtc_options.push("-DCAMELID_HAS_WMMA=1".to_string());
        }
        let options = CompileOptions {
            fmad: Some(false),
            arch: Some(arch),
            include_paths,
            options: nvrtc_options,
            ..Default::default()
        };
        let ptx: Ptx =
            std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx_with_opts(KERNELS, options))
                .map_err(|_| "CUDA NVRTC library not available".to_string())?
                .map_err(|error| format!("vision CUDA nvrtc: {error}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| format!("vision CUDA load module: {error}"))?;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|error| format!("vision CUDA load {name}: {error}"))
        };
        Ok(Self {
            matmul_f32: function("vision_matmul_f32")?,
            matmul_f16: function("vision_matmul_f16")?,
            matmul_f16_wmma: (tensor_core_f16 && !strict)
                .then(|| function("vision_matmul_f16_wmma"))
                .transpose()?,
            convert_f32_to_f16: function("vision_convert_f32_to_f16")?,
            matmul_q8_0: function("vision_matmul_q8_0")?,
            matmul_q8_0_imma: (tensor_core_q8 && !strict)
                .then(|| function("vision_matmul_q8_0_imma"))
                .transpose()?,
            quantize_q8_0: function("vision_quantize_q8_0")?,
            patch_sum: function("vision_patch_sum")?,
            layer_norm: function("vision_layer_norm")?,
            add_bias: function("vision_add_bias")?,
            bias_residual: function("vision_bias_residual")?,
            bias_gelu: function("vision_bias_gelu")?,
            rope: function("vision_rope")?,
            attention: function("vision_attention")?,
            attention_online: (!strict)
                .then(|| function("vision_attention_online"))
                .transpose()?,
            weight_cache: Vec::new(),
            ctx,
            stream,
        })
    }

    #[cfg(test)]
    pub(crate) fn disable_tensor_cores_for_test(&mut self) {
        self.matmul_q8_0_imma = None;
        self.matmul_f16_wmma = None;
        self.attention_online = None;
    }

    pub(crate) fn encode(
        &mut self,
        model: &PrismVisionModel,
        input: &PrismVisionInput,
    ) -> Result<Vec<Vec<f32>>, String> {
        let tokens = input
            .patch_width
            .checked_mul(input.patch_height)
            .ok_or_else(|| "vision CUDA patch geometry overflow".to_string())?;
        if tokens == 0
            || tokens > 4096
            || input.patches.len() != tokens * model.patch_0.input
            || input.position.len() != tokens * model.hidden
        {
            return Err("vision CUDA refused the preprocessed image geometry".into());
        }
        let head_dim = model.hidden / model.heads;
        let d_patches = self.upload_f32(&input.patches, "patches")?;
        let d_position = self.upload_f32(&input.position, "position")?;
        let patch_first = self.linear(&d_patches, &model.patch_0, tokens, "patch 0")?;
        let patch_second = self.linear(&d_patches, &model.patch_1, tokens, "patch 1")?;
        let mut hidden = self.patch_sum(
            &patch_first,
            &patch_second,
            &model.patch_bias,
            &d_position,
            tokens,
            model.hidden,
        )?;
        drop((d_patches, d_position, patch_first, patch_second));

        let (cosine, sine) =
            vision_rope_tables(input.patch_width, input.patch_height, model.merge, head_dim);
        let d_cosine = self.upload_f32(&cosine, "rope cosine")?;
        let d_sine = self.upload_f32(&sine, "rope sine")?;

        for (index, layer) in model.layers.iter().enumerate() {
            let normalized = self.layer_norm(
                &hidden,
                &layer.ln1_weight,
                &layer.ln1_bias,
                tokens,
                model.hidden,
                model.eps,
            )?;
            let mut qkv = self.linear(
                &normalized,
                &layer.qkv,
                tokens,
                &format!("layer {index} qkv"),
            )?;
            self.add_bias(&mut qkv, &layer.qkv_bias, 3 * model.hidden)?;
            self.rope(
                &mut qkv,
                &d_cosine,
                &d_sine,
                tokens,
                model.hidden,
                model.heads,
                head_dim,
            )?;
            let attention = self.attention(&qkv, tokens, model.hidden, model.heads, head_dim)?;
            let projected = self.linear(
                &attention,
                &layer.attn_output,
                tokens,
                &format!("layer {index} attention output"),
            )?;
            let after_attention =
                self.bias_residual(&hidden, &projected, &layer.attn_output_bias, model.hidden)?;

            let normalized = self.layer_norm(
                &after_attention,
                &layer.ln2_weight,
                &layer.ln2_bias,
                tokens,
                model.hidden,
                model.eps,
            )?;
            let mut ffn = self.linear(
                &normalized,
                &layer.ffn_up,
                tokens,
                &format!("layer {index} ffn up"),
            )?;
            self.bias_gelu(&mut ffn, &layer.ffn_up_bias, model.ffn)?;
            let projected = self.linear(
                &ffn,
                &layer.ffn_down,
                tokens,
                &format!("layer {index} ffn down"),
            )?;
            hidden = self.bias_residual(
                &after_attention,
                &projected,
                &layer.ffn_down_bias,
                model.hidden,
            )?;
        }

        let normalized = self.layer_norm(
            &hidden,
            &model.post_weight,
            &model.post_bias,
            tokens,
            model.hidden,
            model.eps,
        )?;
        let output_tokens = tokens / (model.merge * model.merge);
        let merged = model.hidden * model.merge * model.merge;
        let mut merger = self.linear(&normalized, &model.merger_0, output_tokens, "merger 0")?;
        self.bias_gelu(&mut merger, &model.merger_0_bias, merged)?;
        let mut projected = self.linear(&merger, &model.merger_2, output_tokens, "merger 2")?;
        self.add_bias(&mut projected, &model.merger_2_bias, model.projection)?;
        let mut host = vec![0.0f32; output_tokens * model.projection];
        self.stream
            .memcpy_dtoh(&projected, &mut host)
            .map_err(|error| format!("vision CUDA result copy: {error}"))?;
        self.ctx
            .synchronize()
            .map_err(|error| format!("vision CUDA synchronize: {error}"))?;
        if host.iter().any(|value| !value.is_finite()) {
            return Err("vision CUDA produced non-finite embeddings".into());
        }
        Ok(host
            .chunks_exact(model.projection)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn upload_f32(&self, values: &[f32], label: &str) -> Result<CudaSlice<f32>, String> {
        let mut device = self
            .stream
            .alloc_zeros::<f32>(values.len())
            .map_err(|error| format!("vision CUDA allocate {label}: {error}"))?;
        self.stream
            .memcpy_htod(values, &mut device)
            .map_err(|error| format!("vision CUDA upload {label}: {error}"))?;
        Ok(device)
    }

    fn linear(
        &mut self,
        input: &CudaSlice<f32>,
        matrix: &VisionMat,
        tokens: usize,
        label: &str,
    ) -> Result<CudaSlice<f32>, String> {
        let bytes = matrix.pages.byte_len();
        let key = matrix.pages.bytes().as_ptr() as usize;
        let mut cached = self
            .weight_cache
            .iter()
            .position(|(cached_key, cached_bytes, _)| *cached_key == key && *cached_bytes == bytes);
        let mut transient = None;
        if cached.is_none() {
            let mut uploaded = self
                .stream
                .alloc_zeros::<u8>(bytes)
                .map_err(|error| format!("vision CUDA allocate {label} weight: {error}"))?;
            self.stream
                .memcpy_htod(matrix.pages.bytes(), &mut uploaded)
                .map_err(|error| format!("vision CUDA upload {label} weight: {error}"))?;
            let retain = cudarc::driver::result::mem_get_info()
                .map(|(free, total)| {
                    const GIB: usize = 1usize << 30;
                    let low_vram = total <= 8 * GIB;
                    let resident_consumer_present = free <= total.saturating_mul(3) / 4;
                    (!low_vram || resident_consumer_present) && free >= GIB
                })
                .unwrap_or(false);
            if retain {
                self.weight_cache.push((key, bytes, uploaded));
                cached = Some(self.weight_cache.len() - 1);
            } else {
                transient = Some(uploaded);
            }
        }
        let weight = if let Some(index) = cached {
            &self.weight_cache[index].2
        } else {
            transient.as_ref().expect("uncached upload installed above")
        };
        let mut output = self
            .stream
            .alloc_zeros::<f32>(tokens * matrix.output)
            .map_err(|error| format!("vision CUDA allocate {label} output: {error}"))?;
        if matrix.tensor_type == GgufTensorType::Q8_0 {
            if let Some(function) = &self.matmul_q8_0_imma {
                let blocks_per_row = matrix.input / 32;
                if blocks_per_row * 32 == matrix.input {
                    let total_blocks = tokens * blocks_per_row;
                    let mut input_quants = self
                        .stream
                        .alloc_zeros::<i8>(tokens * matrix.input)
                        .map_err(|error| {
                            format!("vision CUDA allocate {label} activation quants: {error}")
                        })?;
                    let mut input_scales =
                        self.stream
                            .alloc_zeros::<f32>(total_blocks)
                            .map_err(|error| {
                                format!("vision CUDA allocate {label} activation scales: {error}")
                            })?;
                    let quant_config = linear_config(total_blocks);
                    let total_blocks_i32 = total_blocks as i32;
                    let mut quantize = self.stream.launch_builder(&self.quantize_q8_0);
                    quantize
                        .arg(input)
                        .arg(&mut input_quants)
                        .arg(&mut input_scales)
                        .arg(&total_blocks_i32);
                    unsafe { quantize.launch(quant_config) }.map_err(|error| {
                        format!("vision CUDA quantize {label} activations: {error}")
                    })?;

                    let config = LaunchConfig {
                        grid_dim: (
                            (matrix.output as u32).div_ceil(128),
                            (tokens as u32).div_ceil(128),
                            1,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let (token_count, cols, rows) =
                        (tokens as i32, matrix.input as i32, matrix.output as i32);
                    let mut launch = self.stream.launch_builder(function);
                    launch
                        .arg(&input_quants)
                        .arg(&input_scales)
                        .arg(weight)
                        .arg(&mut output)
                        .arg(&token_count)
                        .arg(&cols)
                        .arg(&rows);
                    unsafe { launch.launch(config) }.map_err(|error| {
                        format!("vision CUDA launch {label} tensor-core Q8: {error}")
                    })?;
                    return Ok(output);
                }
            }
        }
        if matrix.tensor_type == GgufTensorType::F16
            && tokens.is_multiple_of(16)
            && matrix.input.is_multiple_of(16)
            && matrix.output.is_multiple_of(16)
        {
            if let Some(function) = &self.matmul_f16_wmma {
                let input_values = tokens * matrix.input;
                let mut input_f16 =
                    self.stream
                        .alloc_zeros::<u16>(input_values)
                        .map_err(|error| {
                            format!("vision CUDA allocate {label} f16 activations: {error}")
                        })?;
                let input_values_i32 = input_values as i32;
                let mut convert = self.stream.launch_builder(&self.convert_f32_to_f16);
                convert
                    .arg(input)
                    .arg(&mut input_f16)
                    .arg(&input_values_i32);
                unsafe { convert.launch(linear_config(input_values)) }.map_err(|error| {
                    format!("vision CUDA convert {label} activations to f16: {error}")
                })?;

                let tile_count = (tokens / 16) * (matrix.output / 16);
                let block = 256u32;
                let warps_per_block = block / 32;
                let config = LaunchConfig {
                    grid_dim: ((tile_count as u32).div_ceil(warps_per_block), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (token_count, cols, rows) =
                    (tokens as i32, matrix.input as i32, matrix.output as i32);
                let mut launch = self.stream.launch_builder(function);
                launch
                    .arg(&input_f16)
                    .arg(weight)
                    .arg(&mut output)
                    .arg(&token_count)
                    .arg(&cols)
                    .arg(&rows);
                unsafe { launch.launch(config) }.map_err(|error| {
                    format!("vision CUDA launch {label} tensor-core F16: {error}")
                })?;
                return Ok(output);
            }
        }
        let function = match matrix.tensor_type {
            GgufTensorType::F32 => &self.matmul_f32,
            GgufTensorType::F16 => &self.matmul_f16,
            GgufTensorType::Q8_0 => &self.matmul_q8_0,
            other => {
                return Err(format!(
                    "vision CUDA {label}: unsupported matrix type {other:?}"
                ))
            }
        };
        let total = tokens * matrix.output;
        let block = 256u32;
        let warps_per_block = block / 32;
        let config = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(warps_per_block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (token_count, cols, rows) = (tokens as i32, matrix.input as i32, matrix.output as i32);
        let mut launch = self.stream.launch_builder(function);
        launch
            .arg(input)
            .arg(weight)
            .arg(&mut output)
            .arg(&token_count)
            .arg(&cols)
            .arg(&rows);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA launch {label}: {error}"))?;
        Ok(output)
    }

    fn patch_sum(
        &self,
        first: &CudaSlice<f32>,
        second: &CudaSlice<f32>,
        bias: &[f32],
        position: &CudaSlice<f32>,
        tokens: usize,
        width: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let d_bias = self.upload_f32(bias, "patch bias")?;
        let total = tokens * width;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|error| format!("vision CUDA allocate patch output: {error}"))?;
        let config = linear_config(total);
        let (width, total) = (width as i32, total as i32);
        let mut launch = self.stream.launch_builder(&self.patch_sum);
        launch
            .arg(first)
            .arg(second)
            .arg(&d_bias)
            .arg(position)
            .arg(&mut output)
            .arg(&width)
            .arg(&total);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA patch sum: {error}"))?;
        Ok(output)
    }

    fn layer_norm(
        &self,
        input: &CudaSlice<f32>,
        weight: &[f32],
        bias: &[f32],
        tokens: usize,
        width: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let d_weight = self.upload_f32(weight, "layer norm weight")?;
        let d_bias = self.upload_f32(bias, "layer norm bias")?;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(tokens * width)
            .map_err(|error| format!("vision CUDA allocate layer norm output: {error}"))?;
        let config = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let width = width as i32;
        let mut launch = self.stream.launch_builder(&self.layer_norm);
        launch
            .arg(input)
            .arg(&d_weight)
            .arg(&d_bias)
            .arg(&mut output)
            .arg(&width)
            .arg(&eps);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA layer norm: {error}"))?;
        Ok(output)
    }

    fn add_bias(
        &self,
        values: &mut CudaSlice<f32>,
        bias: &[f32],
        width: usize,
    ) -> Result<(), String> {
        let d_bias = self.upload_f32(bias, "bias")?;
        let total = values.len();
        let config = linear_config(total);
        let (width, total) = (width as i32, total as i32);
        let mut launch = self.stream.launch_builder(&self.add_bias);
        launch.arg(values).arg(&d_bias).arg(&width).arg(&total);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA add bias: {error}"))?;
        Ok(())
    }

    fn bias_residual(
        &self,
        residual: &CudaSlice<f32>,
        projected: &CudaSlice<f32>,
        bias: &[f32],
        width: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let d_bias = self.upload_f32(bias, "residual bias")?;
        let total = residual.len();
        let mut output = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|error| format!("vision CUDA allocate residual output: {error}"))?;
        let config = linear_config(total);
        let (width, total) = (width as i32, total as i32);
        let mut launch = self.stream.launch_builder(&self.bias_residual);
        launch
            .arg(residual)
            .arg(projected)
            .arg(&d_bias)
            .arg(&mut output)
            .arg(&width)
            .arg(&total);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA residual: {error}"))?;
        Ok(output)
    }

    fn bias_gelu(
        &self,
        values: &mut CudaSlice<f32>,
        bias: &[f32],
        width: usize,
    ) -> Result<(), String> {
        let d_bias = self.upload_f32(bias, "GELU bias")?;
        let total = values.len();
        let config = linear_config(total);
        let (width, total) = (width as i32, total as i32);
        let mut launch = self.stream.launch_builder(&self.bias_gelu);
        launch.arg(values).arg(&d_bias).arg(&width).arg(&total);
        unsafe { launch.launch(config) }.map_err(|error| format!("vision CUDA GELU: {error}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        qkv: &mut CudaSlice<f32>,
        cosine: &CudaSlice<f32>,
        sine: &CudaSlice<f32>,
        tokens: usize,
        hidden: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let total = tokens * heads * (head_dim / 2);
        let config = linear_config(total);
        let (hidden, heads, head_dim, tokens) =
            (hidden as i32, heads as i32, head_dim as i32, tokens as i32);
        let mut launch = self.stream.launch_builder(&self.rope);
        launch
            .arg(qkv)
            .arg(cosine)
            .arg(sine)
            .arg(&hidden)
            .arg(&heads)
            .arg(&head_dim)
            .arg(&tokens);
        unsafe { launch.launch(config) }.map_err(|error| format!("vision CUDA RoPE: {error}"))?;
        Ok(())
    }

    fn attention(
        &self,
        qkv: &CudaSlice<f32>,
        tokens: usize,
        hidden: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let mut output = self
            .stream
            .alloc_zeros::<f32>(tokens * hidden)
            .map_err(|error| format!("vision CUDA allocate attention output: {error}"))?;
        let fast = self.attention_online.as_ref().filter(|_| head_dim <= 256);
        let config = if fast.is_some() {
            LaunchConfig {
                grid_dim: ((tokens as u32).div_ceil(8), heads as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (tokens as u32, heads as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: ((tokens + 1) * 4) as u32,
            }
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (hidden, heads, head_dim, tokens) =
            (hidden as i32, heads as i32, head_dim as i32, tokens as i32);
        let function = fast.unwrap_or(&self.attention);
        let mut launch = self.stream.launch_builder(function);
        launch
            .arg(qkv)
            .arg(&mut output)
            .arg(&hidden)
            .arg(&heads)
            .arg(&head_dim)
            .arg(&tokens)
            .arg(&scale);
        unsafe { launch.launch(config) }
            .map_err(|error| format!("vision CUDA attention: {error}"))?;
        Ok(output)
    }
}

fn linear_config(total: usize) -> LaunchConfig {
    let block = 256u32;
    LaunchConfig {
        grid_dim: ((total as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn vision_rope_tables(
    patch_width: usize,
    patch_height: usize,
    merge: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let section = half / 2;
    let mut cosine = Vec::with_capacity(patch_width * patch_height * half);
    let mut sine = Vec::with_capacity(cosine.capacity());
    for tile_y in (0..patch_height).step_by(merge) {
        for tile_x in (0..patch_width).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    for pair in 0..half {
                        let (position, dimension) = if pair < section {
                            ((tile_y + dy) as f32, pair)
                        } else {
                            ((tile_x + dx) as f32, pair - section)
                        };
                        let angle =
                            position * 10_000.0f32.powf(-2.0 * dimension as f32 / half as f32);
                        cosine.push(angle.cos());
                        sine.push(angle.sin());
                    }
                }
            }
        }
    }
    (cosine, sine)
}
