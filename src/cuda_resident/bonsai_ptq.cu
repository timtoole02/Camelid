
// Bonsai 2 activation transform. One CTA owns one Sylvester block; the
// butterflies preserve the CPU operation order within every block. Grouped
// GDN output is permuted before signs and rotation, never after the projection.
extern "C" __global__ void bonsai_hadamard(
    const float* input, const float* signs, float* output,
    int width, int block, int sign_count, int hd, int nk, int rep, int inverse
) {
    extern __shared__ float values[];
    int row = blockIdx.y;
    int offset = blockIdx.x * block;
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        int dst = offset + i;
        int src = dst;
        if (!inverse && nk != 0) {
            int channel = dst % hd;
            int group = dst / hd;
            int key = group / rep;
            int repeat = group % rep;
            src = (repeat * nk + key) * hd + channel;
        }
        float value = input[row * width + src];
        if (!inverse && sign_count != 0) value *= signs[dst];
        values[i] = value;
    }
    __syncthreads();
    for (int stride = 1; stride < block; stride *= 2) {
        for (int pair = threadIdx.x; pair < block / 2; pair += blockDim.x) {
            int first = (pair / stride) * (stride * 2) + pair % stride;
            float a = values[first];
            float b = values[first + stride];
            values[first] = a + b;
            values[first + stride] = a - b;
        }
        __syncthreads();
    }
    float scale = 1.0f / sqrtf((float)block);
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        float value = values[i] * scale;
        if (inverse && sign_count != 0) value *= signs[offset + i];
        output[row * width + offset + i] = value;
    }
}

__device__ __forceinline__ int bonsai_ptq_trit(const unsigned char* block, int i) {
    int byte_index, power_index;
    if (i < 80) { byte_index = i % 16; power_index = i / 16; }
    else if (i < 120) { byte_index = 16 + (i - 80) % 8; power_index = (i - 80) / 8; }
    else { byte_index = 24 + (i - 120) % 2; power_index = (i - 120) / 2; }
    int power = power_index == 0 ? 1 : power_index == 1 ? 3 : power_index == 2 ? 9 : power_index == 3 ? 27 : 81;
    unsigned char code = (unsigned char)(block[byte_index] * power);
    return (((unsigned int)code * 3u) >> 8) - 1;
}

// One warp per output row and up to eight independent token accumulators.
// Each packed weight is decoded once and reused across the proposal chain.
// Every accumulator follows the scalar column/reduction order exactly, so
// batching changes neither target logits nor the seeded sampling stream.
extern "C" __global__ void bonsai_ptq_matmul(
    const float* input, const unsigned char* weights, float* output,
    int rows, int cols, int residual, int dense, int tokens
) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int token_base = blockIdx.y * 8;
    int remaining = tokens - token_base;
    int token_count = remaining < 8 ? remaining : 8;
    if (row >= rows) return;
    float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (dense != 0) {
        for (int i = lane; i < cols; i += 32) {
            size_t offset = (size_t)row * cols + i;
            float weight;
            if (dense == 2) {
                const unsigned short bits = ((const unsigned short*)weights)[offset];
                weight = __uint_as_float(((unsigned int)bits) << 16);
            } else {
                weight = ((const float*)weights)[offset];
            }
            #pragma unroll
            for (int token = 0; token < 8; ++token) {
                if (token < token_count)
                    partial[token] += weight * input[(size_t)(token_base + token) * cols + i];
            }
        }
    } else {
        // Each lane visits four trits of every 128-value block. Compute their
        // interleave geometry once, outside the contraction loop. These are
        // exactly the lane,lane+32,lane+64,lane+96 positions used previously.
        const int byte_indices[4] = {
            lane & 15,
            lane & 15,
            lane < 24 ? lane : lane - 8,
            lane < 8 ? lane + 16 : lane < 16 ? lane + 8 : lane < 24 ? lane : 24 + (lane & 1)
        };
        const int powers[4] = {
            lane < 16 ? 1 : 3,
            lane < 16 ? 9 : 27,
            lane < 16 ? 81 : lane < 24 ? 1 : 3,
            lane < 8 ? 9 : lane < 16 ? 27 : lane < 24 ? 81 : lane < 26 ? 1 : lane < 28 ? 3 : lane < 30 ? 9 : 27
        };
        const int blocks = cols / 128;
        const unsigned char* row_weights = weights + (size_t)row * blocks * 28;
        for (int block_index = 0; block_index < blocks; ++block_index) {
            const unsigned char* block = row_weights + block_index * 28;
            float scale = 0.0f;
            if (lane == 0) {
                unsigned short bits = (unsigned short)block[26] | ((unsigned short)block[27] << 8);
                scale = f16_bits_to_f32(bits);
            }
            scale = __shfl_sync(0xffffffffu, scale, 0);
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                unsigned char code = (unsigned char)(block[byte_indices[quarter]] * powers[quarter]);
                int trit = (((unsigned int)code * 3u) >> 8) - 1;
                float weight = scale * (float)trit;
                int i = block_index * 128 + lane + quarter * 32;
                #pragma unroll
                for (int token = 0; token < 8; ++token) {
                    if (token < token_count)
                        partial[token] += weight * input[(size_t)(token_base + token) * cols + i];
                }
            }
        }
    }
    #pragma unroll
    for (int token = 0; token < 8; ++token) {
        if (token < token_count) {
            for (int step = 16; step > 0; step /= 2)
                partial[token] += __shfl_down_sync(0xffffffffu, partial[token], step);
            if (lane == 0) {
                size_t index = (size_t)(token_base + token) * rows + row;
                output[index] = residual ? output[index] + partial[token] : partial[token];
            }
        }
    }
}

// The generic eight-token kernel above spends issue slots on inactive token
// lanes during scalar decode. Specialize the token tile, and let each warp
// reuse its activation loads across several output rows. Seven lanes read the
// complete 28-byte PTQ block as aligned words; shuffles distribute its bytes.
// The on-disk/device layout, per-lane column order, and final reduction stay
// unchanged. No activation quantization or additional weight allocation.
template<int TOKENS, int ROWS>
__device__ __forceinline__ void bonsai_ptq_project_tile(
    const float* input, const unsigned char* weights, float* output,
    int rows, int cols, int residual, int tokens
) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * ROWS;
    const int token_base = blockIdx.y * TOKENS;
    const int token_count = min(TOKENS, tokens - token_base);
    if (row >= rows) return;
    const int byte_index[4] = {
        lane & 15, lane & 15, lane < 24 ? lane : lane - 8,
        lane < 8 ? lane + 16 : lane < 16 ? lane + 8 : lane < 24 ? lane : 24 + (lane & 1)
    };
    const int power[4] = {
        lane < 16 ? 1 : 3, lane < 16 ? 9 : 27,
        lane < 16 ? 81 : lane < 24 ? 1 : 3,
        lane < 8 ? 9 : lane < 16 ? 27 : lane < 24 ? 81 : lane < 26 ? 1 : lane < 28 ? 3 : lane < 30 ? 9 : 27
    };
    float partial[ROWS][TOKENS] = {};
    const int blocks = cols / 128;
    for (int b = 0; b < blocks; ++b) {
        unsigned int codes[ROWS][4];
        float scales[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            unsigned int word = 0;
            if (row + r < rows && lane < 7) {
                const unsigned char* block = weights + ((size_t)(row + r) * blocks + b) * 28;
                word = ((const unsigned int*)block)[lane];
            }
            unsigned short bits = __shfl_sync(0xffffffffu, word, 6) >> 16;
            // Preserve NaN payload handling from the original converter too.
            if ((bits & 0x7c00u) == 0x7c00u) scales[r] = f16_bits_to_f32(bits);
            else asm("cvt.f32.f16 %0, %1;" : "=f"(scales[r]) : "h"(bits));
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                unsigned int byte = (__shfl_sync(0xffffffffu, word, byte_index[q] / 4)
                    >> ((byte_index[q] & 3) * 8)) & 255u;
                codes[r][q] = (((byte * power[q]) & 255u) * 3u) >> 8;
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float weight[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) weight[r] = scales[r] * (float)((int)codes[r][q] - 1);
            const int i = b * 128 + lane + q * 32;
            #pragma unroll
            for (int t = 0; t < TOKENS; ++t) {
                if (TOKENS == 1 || t < token_count) {
                    float x = input[(size_t)(token_base + t) * cols + i];
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) partial[r][t] += weight[r] * x;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (TOKENS == 1 || t < token_count) {
                for (int step = 16; step > 0; step /= 2)
                    partial[r][t] += __shfl_down_sync(0xffffffffu, partial[r][t], step);
                if (lane == 0 && row + r < rows) {
                    size_t index = (size_t)(token_base + t) * rows + row + r;
                    output[index] = residual ? output[index] + partial[r][t] : partial[r][t];
                }
            }
        }
    }
}

#define BONSAI_PTQ_TILE(NAME, T, R) \
extern "C" __global__ void NAME( \
    const float* input, const unsigned char* weights, float* output, \
    int rows, int cols, int residual, int dense, int tokens) { \
    bonsai_ptq_project_tile<T, R>(input, weights, output, rows, cols, residual, tokens); \
}
BONSAI_PTQ_TILE(bonsai_ptq_t1_r2, 1, 2)
BONSAI_PTQ_TILE(bonsai_ptq_t1_r4, 1, 4)
BONSAI_PTQ_TILE(bonsai_ptq_t2_r4, 2, 4)
BONSAI_PTQ_TILE(bonsai_ptq_t4_r4, 4, 4)
BONSAI_PTQ_TILE(bonsai_ptq_t8_r4, 8, 4)
#undef BONSAI_PTQ_TILE

template<int TOKENS, int DENSE>
__device__ __forceinline__ void bonsai_dense_project_tile(
    const float* input, const unsigned char* weights, float* output,
    int rows, int cols, int residual, int tokens
) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int token_base = blockIdx.y * TOKENS;
    int token_count = min(TOKENS, tokens - token_base);
    if (row >= rows) return;
    float partial[TOKENS] = {};
    for (int i = lane; i < cols; i += 32) {
        size_t index = (size_t)row * cols + i;
        float weight = DENSE == 2 ? __uint_as_float(((unsigned int)((const unsigned short*)weights)[index]) << 16)
                                 : ((const float*)weights)[index];
        #pragma unroll
        for (int t = 0; t < TOKENS; ++t)
            if (TOKENS == 1 || t < token_count)
                partial[t] += weight * input[(size_t)(token_base + t) * cols + i];
    }
    #pragma unroll
    for (int t = 0; t < TOKENS; ++t) {
        if (TOKENS == 1 || t < token_count) {
            for (int step = 16; step > 0; step /= 2)
                partial[t] += __shfl_down_sync(0xffffffffu, partial[t], step);
            if (lane == 0) {
                size_t index = (size_t)(token_base + t) * rows + row;
                output[index] = residual ? output[index] + partial[t] : partial[t];
            }
        }
    }
}
#define BONSAI_DENSE_TILE(NAME, T, D) \
extern "C" __global__ void NAME( \
    const float* input, const unsigned char* weights, float* output, \
    int rows, int cols, int residual, int dense, int tokens) { \
    bonsai_dense_project_tile<T, D>(input, weights, output, rows, cols, residual, tokens); \
}
BONSAI_DENSE_TILE(bonsai_f32_t1, 1, 1)
BONSAI_DENSE_TILE(bonsai_f32_t4, 4, 1)
BONSAI_DENSE_TILE(bonsai_f32_t8, 8, 1)
BONSAI_DENSE_TILE(bonsai_bf16_t1, 1, 2)
BONSAI_DENSE_TILE(bonsai_bf16_t4, 4, 2)
BONSAI_DENSE_TILE(bonsai_bf16_t8, 8, 2)
#undef BONSAI_DENSE_TILE

extern "C" __global__ void embed_gather_ptq1_0(
    const unsigned char* table, const unsigned int* token, int dim, float* output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const unsigned char* block = table + ((size_t)token[0] * (dim / 128) + i / 128) * 28;
    unsigned short bits = (unsigned short)block[26] | ((unsigned short)block[27] << 8);
    output[i] = f16_bits_to_f32(bits) * (float)bonsai_ptq_trit(block, i & 127);
}
