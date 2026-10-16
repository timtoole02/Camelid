// Transposed Q8 activation planes for the original 28-byte PTQ weights.
// The plane layout and exact-sum technique follow the MIT-licensed PrismML
// llama.cpp PTQ1 CUDA reference; see THIRD_PARTY_NOTICES.md for attribution.
// Each 32-element activation group has a half scale and an exact signed sum.
// A plane contains the same packed word from 32 adjacent 128-element blocks,
// so lanes processing adjacent K blocks issue coalesced activation reads.
__device__ __forceinline__ void bonsai_store_q8(
    float value, unsigned int* output, int cols, int i, int token
) {
    const int lane = threadIdx.x & 31;
    float maximum = fabsf(value);
    for (int s = 16; s; s >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, s));
    const float scale = maximum / 127.0f;
    const int q = maximum == 0.0f ? 0 : (int)roundf(value / scale);
    unsigned int packed = (unsigned int)q & 255u;
    packed |= (__shfl_down_sync(0xffffffffu, packed, 1) & 255u) << 8;
    packed |= __shfl_down_sync(0xffffffffu, packed, 2) << 16;
    const int block = i / 128;
    const int sub = (i / 32) & 3;
    const int stride = ((cols / 128 + 31) / 32) * 36 * 32;
    unsigned int* base = output + (size_t)token * stride
        + (block / 32) * 36 * 32 + (block & 31);
    if ((lane & 3) == 0) base[(sub * 8 + lane / 4) * 32] = packed;
    int sum = q;
    for (int s = 16; s; s >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, s);
    if (lane == 0)
        base[(32 + sub) * 32] = (unsigned int)f32_to_f16_bits(scale)
            | (((unsigned int)sum & 65535u) << 16);
}

// Retained as an independent quantizer for numerical validation and profiling.
extern "C" __global__ void bonsai_quantize_q8(
    const float* input, unsigned int* output, int cols, int tokens
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) bonsai_store_q8(input[(size_t)blockIdx.y * cols + i], output, cols, i, blockIdx.y);
}

// Preserve bonsai_hadamard's butterflies and permutation, then quantize from
// shared memory. No intermediate global float write or extra quantizer launch.
extern "C" __global__ void bonsai_hadamard_q8(
    const float* input, const float* signs, unsigned int* output,
    int width, int block, int sign_count, int hd, int nk, int rep
) {
    extern __shared__ float values[];
    const int token = blockIdx.y;
    const int offset = blockIdx.x * block;
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        const int dst = offset + i;
        int src = dst;
        if (nk != 0) {
            const int channel = dst % hd;
            const int group = dst / hd;
            src = ((group % rep) * nk + group / rep) * hd + channel;
        }
        float value = input[(size_t)token * width + src];
        if (sign_count != 0) value *= signs[dst];
        values[i] = value;
    }
    __syncthreads();
    for (int stride = 1; stride < block; stride *= 2) {
        for (int pair = threadIdx.x; pair < block / 2; pair += blockDim.x) {
            const int first = (pair / stride) * (stride * 2) + pair % stride;
            const float a = values[first];
            const float b = values[first + stride];
            values[first] = a + b;
            values[first + stride] = a - b;
        }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf((float)block);
    for (int i = threadIdx.x; i < block; i += blockDim.x)
        bonsai_store_q8(values[i] * scale, output, width, offset + i, token);
}

template<int TOKENS, int LANES>
__device__ __forceinline__ void bonsai_q8_project(
    const unsigned int* input, const unsigned char* weights, float* output,
    int rows, int cols, int residual, int tokens
) {
    const int lane = threadIdx.x % LANES;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / LANES;
    const int token_base = blockIdx.y * TOKENS;
    const int count = min(TOKENS, tokens - token_base);
    if (row >= rows) return;
    const int blocks = cols / 128;
    const int stride = ((blocks + 31) / 32) * 36 * 32;
    float partial[TOKENS] = {};
    for (int b = lane; b < blocks; b += LANES) {
        const unsigned char* w = weights + ((size_t)row * blocks + b) * 28;
        const unsigned int* x = input + (size_t)token_base * stride
            + (b / 32) * 36 * 32 + (b & 31);
        int dots[TOKENS][4] = {};
        // Extract four base-three digits at once, retaining the fractional
        // byte for the next digit. Digits are 0,1,2; subtract the exact Q8 sum
        // after contraction to recover the signed ternary values -1,0,1.
        #pragma unroll
        for (int g = 0; g < 6; ++g) {
            const unsigned int word = ((const unsigned int*)w)[g];
            unsigned int lo = __byte_perm(word, 0, 0x4140);
            unsigned int hi = __byte_perm(word, 0, 0x4342);
            #pragma unroll
            for (int digit = 0; digit < 5; ++digit) {
                lo *= 3u;
                hi *= 3u;
                const int trits = __byte_perm(lo, hi, 0x7531);
                lo &= 0x00ff00ffu;
                hi &= 0x00ff00ffu;
                const int element = g < 4 ? digit * 16 + g * 4 : 80 + digit * 8 + (g - 4) * 4;
                #pragma unroll
                for (int t = 0; t < TOKENS; ++t) if (TOKENS == 1 || t < count)
                    dots[t][element / 32] = __dp4a(trits,
                        (int)x[t * stride + (element / 4) * 32], dots[t][element / 32]);
            }
        }
        unsigned int tail = (unsigned int)w[24] | ((unsigned int)w[25] << 16);
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const unsigned int first = tail * 3u;
            const unsigned int second = (first & 0x00ff00ffu) * 3u;
            tail = second & 0x00ff00ffu;
            const int trits = __byte_perm(first, second, 0x7531);
            #pragma unroll
            for (int t = 0; t < TOKENS; ++t) if (TOKENS == 1 || t < count)
                dots[t][3] = __dp4a(trits, (int)x[t * stride + (30 + pair) * 32], dots[t][3]);
        }
        const unsigned short weight_bits = ((const unsigned short*)w)[13];
        float weight_scale;
        asm("cvt.f32.f16 %0, %1;" : "=f"(weight_scale) : "h"(weight_bits));
        #pragma unroll
        for (int t = 0; t < TOKENS; ++t) if (TOKENS == 1 || t < count) {
            float sum = 0.0f;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const unsigned int ds = x[t * stride + (32 + sub) * 32];
                const unsigned short scale_bits = ds & 65535u;
                float scale;
                asm("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));
                const int bias = (int)ds >> 16;
                sum += scale * (float)(dots[t][sub] - bias);
            }
            partial[t] += weight_scale * sum;
        }
    }
    const unsigned int mask = __activemask();
    #pragma unroll
    for (int t = 0; t < TOKENS; ++t) if (TOKENS == 1 || t < count) {
        for (int step = LANES / 2; step; step >>= 1)
            partial[t] += __shfl_down_sync(mask, partial[t], step, LANES);
        if (lane == 0) {
            const size_t i = (size_t)(token_base + t) * rows + row;
            output[i] = residual ? output[i] + partial[t] : partial[t];
        }
    }
}

#define BONSAI_Q8(NAME, T, L) \
extern "C" __global__ void NAME( \
    const unsigned int* input, const unsigned char* weights, float* output, \
    int rows, int cols, int residual, int tokens) { \
    bonsai_q8_project<T, L>(input, weights, output, rows, cols, residual, tokens); \
}
BONSAI_Q8(bonsai_q8_t1_l16, 1, 16)
BONSAI_Q8(bonsai_q8_t1_l32, 1, 32)
BONSAI_Q8(bonsai_q8_t4_l16, 4, 16)
BONSAI_Q8(bonsai_q8_t4_l32, 4, 32)
BONSAI_Q8(bonsai_q8_t8_l16, 8, 16)
BONSAI_Q8(bonsai_q8_t8_l32, 8, 32)
#undef BONSAI_Q8
