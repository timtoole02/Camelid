// Shared by NVRTC and the explicit native-host math tests. The host tests
// exercise these same scalar operations; they do not validate CUDA execution.
__device__ __forceinline__ float sampling_uniform_from_u24(unsigned int m) {
    // The legacy 24-bit midpoint rounds to 1.0f at m=0xffffff. Clamp that
    // endpoint only, keeping every other seeded uniform unchanged.
    return fminf(((float)m + 0.5f) / 16777216.0f, 0.999999940395355224609375f);
}

__device__ __forceinline__ float sampling_centered_logit(
    float logit, float maximum, float temperature
) {
    return (logit / temperature) - (maximum / temperature);
}

// -1 requests CPU sampling of the already-computed logits; 0 drops and 1 keeps.
// Compare untempered probability ratios, avoiding max_logit + log(min_p), whose
// addition loses the cutoff at large offsets. The CPU reference normalizes
// probabilities before comparing. Near a boundary, those f32 divisions and
// multiplication (and host/device expf differences) can change the retained set.
// Conservatively hand those rows back to that reference. This is an opt-in
// optimization with explicit device parity gates, not a universal libm guarantee.
__device__ __forceinline__ int min_p_candidate(
    float logit, float max_logit, float min_p, int n
) {
    // Keeping the normalized cutoff well inside the normal range also handles
    // CPU thresholds that would underflow to zero and admit zero probabilities.
    // At p=1, rounded probability ties can include unequal logits: use the CPU.
    if (n <= 0 || !isfinite(logit) || !isfinite(max_logit)
        || !isfinite(min_p) || min_p <= 0.0f || min_p >= 1.0f
        || min_p < (4.0f * 1.1754943508222875e-38f) * (float)n) {
        return -1;
    }
    float ratio = expf(logit - max_logit);
    float margin = (64.0f * 1.1920928955078125e-7f) * min_p;
    if (!isfinite(ratio) || fabsf(ratio - min_p) <= margin) return -1;
    return ratio > min_p ? 1 : 0;
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
    return sampling_uniform_from_u24(m);
}
#ifndef CAMELID_SAMPLING_HOST_TEST
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

// Shared maximum/rejection reduction for the filtered draw and its mask oracle.
__device__ __forceinline__ float sampling_row_max(
    const float* logits, int n, float* values, int* invalid
) {
    int tid = threadIdx.x;
    float local_max = -__int_as_float(0x7f800000);
    int bad = n <= 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float logit = logits[i];
        if (!isfinite(logit)) bad = 1;
        else local_max = fmaxf(local_max, logit);
    }
    values[tid] = local_max;
    invalid[tid] = bad;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] = fmaxf(values[tid], values[tid + stride]);
            invalid[tid] |= invalid[tid + stride];
        }
        __syncthreads();
    }
    return values[0];
}

// Min-p filtering precedes temperature. A UINT_MAX result requests a single
// readback of the existing logits and CPU sampling, never another forward pass.
extern "C" __global__ void sample_gumbel_filtered(
    const float* __restrict__ logits, int n, float temperature, float min_p,
    unsigned long long seed, unsigned int* __restrict__ out_idx
) {
    extern __shared__ float sh[];
    float* values = sh;
    int* indices = (int*)(sh + blockDim.x);
    int* invalid = (int*)(sh + 2 * blockDim.x);
    int tid = threadIdx.x;
    float maximum = sampling_row_max(logits, n, values, invalid);
    int bad = invalid[0] || !isfinite(temperature) || temperature <= 0.0f;
    // Every lane must read the maximum before values[0] becomes a sample score.
    __syncthreads();
    float best = -__int_as_float(0x7f800000);
    int best_index = 0x7fffffff;
    for (int i = tid; i < n; i += blockDim.x) {
        int keep = min_p_candidate(logits[i], maximum, min_p, n);
        if (keep < 0) bad = 1;
        if (keep == 1) {
            float u = splitmix_uniform(seed, (unsigned int)i);
            float gumbel = -logf(-logf(u));
            // Match the CPU's division-before-centering order. Center before
            // adding noise so a large common offset cannot erase the draw.
            float score = sampling_centered_logit(logits[i], maximum, temperature) + gumbel;
            if (!isfinite(score)) bad = 1;
            if (score > best || (score == best && i < best_index)) {
                best = score;
                best_index = i;
            }
        }
    }
    values[tid] = best;
    indices[tid] = best_index;
    invalid[tid] = bad;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = values[tid + stride];
            int index = indices[tid + stride];
            if (other > values[tid] || (other == values[tid] && index < indices[tid])) {
                values[tid] = other;
                indices[tid] = index;
            }
            invalid[tid] |= invalid[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_idx[0] = invalid[0] || indices[0] == 0x7fffffff
            ? 0xffffffffu : (unsigned int)indices[0];
    }
}

#ifdef CAMELID_CUDA_SAMPLING_TESTS
extern "C" __global__ void sample_min_p_mask(
    const float* logits, int n, float min_p, int* mask
) {
    extern __shared__ float sh[];
    int* invalid = (int*)(sh + blockDim.x);
    float maximum = sampling_row_max(logits, n, sh, invalid);
    int bad = invalid[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        mask[i] = bad ? -1 : min_p_candidate(logits[i], maximum, min_p, n);
    }
}

extern "C" __global__ void sampling_uniform_test(
    const unsigned int* mantissas, int n, float* output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = sampling_uniform_from_u24(mantissas[i]);
}
#endif
#endif // CAMELID_SAMPLING_HOST_TEST
