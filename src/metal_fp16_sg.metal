#include <metal_stdlib>
using namespace metal;

// packed_w[output_octet][k_octet][k_inner][output_inner] is the
// transposed operand W^T, one ordinary row-major half8x8 fragment.
// packed_x[token_octet][k_octet][token_inner][k_inner] is X.
// One 32-thread SIMD group computes NR*8 output columns and NT*8 tokens.
// FP32 accumulators persist across the complete K reduction.
template <uint NR, uint NT>
inline void fragment_mm(
    device const half* packed_w,
    device const half* packed_x,
    device float* output,
    uint kdim, uint output_columns, uint tokens, uint2 group, uint lane
) {
    const uint output_start = group.x * NR * 8;
    const uint token_start = group.y * NT * 8;
    if (output_start >= output_columns || token_start >= tokens) return;
    const uint k_octets = kdim / 8;
    simdgroup_float8x8 accum[NR][NT];
    #pragma unroll
    for (uint r = 0; r < NR; ++r) {
        #pragma unroll
        for (uint t = 0; t < NT; ++t) {
            accum[r][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k = 0; k < k_octets; ++k) {
        simdgroup_half8x8 weights[NR];
        #pragma unroll
        for (uint r = 0; r < NR; ++r) {
            simdgroup_load(weights[r], packed_w + ((group.x * NR + r) * k_octets + k) * 64, 8, 0, false);
        }
        #pragma unroll
        for (uint t = 0; t < NT; ++t) {
            simdgroup_half8x8 activation;
            simdgroup_load(activation,
                packed_x + ((token_start / 8 + t) * k_octets + k) * 64, 8, 0, false);
            #pragma unroll
            for (uint r = 0; r < NR; ++r) {
                simdgroup_multiply_accumulate(accum[r][t], activation, weights[r], accum[r][t]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < NR; ++r) {
      #pragma unroll
      for (uint t = 0; t < NT; ++t) {
        if (token_start + t * 8 < tokens) {
            if (token_start + t * 8 + 8 <= tokens) {
                simdgroup_store(accum[r][t],
                    output + (token_start + t * 8) * output_columns + output_start + r * 8,
                    output_columns, 0, false);
            } else {
                // Exact Apple-family ownership map used by Camelid's existing
                // kquant_v4_fragment_coord; avoid writing padded output rows.
                const uint fragment_row = 4u * (lane >> 4) + ((lane & 7u) >> 1);
                const uint column0 = 4u * ((lane >> 3) & 1u) + 2u * (lane & 1u);
                const uint token = token_start + t * 8 + fragment_row;
                if (token < tokens) {
                    output[token * output_columns + output_start + r * 8 + column0] = accum[r][t].thread_elements()[0];
                    output[token * output_columns + output_start + r * 8 + column0 + 1] = accum[r][t].thread_elements()[1];
                }
            }
        }
      }
    }
}

#define FRAGMENT_KERNEL(NAME, NR, NT) \
kernel void NAME( \
    device const half* packed_w [[buffer(0)]], \
    device const half* packed_x [[buffer(1)]], \
    device float* output [[buffer(2)]], \
    constant uint& kdim [[buffer(3)]], \
    constant uint& output_columns [[buffer(4)]], \
    constant uint& tokens [[buffer(5)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    fragment_mm<NR, NT>(packed_w, packed_x, output, kdim, output_columns, tokens, group, lane); \
}
FRAGMENT_KERNEL(sg_fragment_nt1, 1, 1)
FRAGMENT_KERNEL(sg_fragment_nt2, 1, 2)
FRAGMENT_KERNEL(sg_fragment_nt4, 1, 4)
FRAGMENT_KERNEL(sg_fragment_nt8, 1, 8)
FRAGMENT_KERNEL(sg_fragment_nr2_nt1, 2, 1)
FRAGMENT_KERNEL(sg_fragment_nr2_nt2, 2, 2)
FRAGMENT_KERNEL(sg_fragment_nr2_nt4, 2, 4)
FRAGMENT_KERNEL(sg_fragment_nr2_nt8, 2, 8)

// One-time activation-layout staging, timed separately from candidate GEMMs.
// Extra physical token rows are zero, allowing an NT8 experiment at M=32.
kernel void pack_activation_fragments(
    device const half* input [[buffer(0)]],
    device half* packed [[buffer(1)]],
    constant uint& kdim [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& physical_tokens [[buffer(4)]],
    constant uint& source_row_stride [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= physical_tokens * kdim) return;
    const uint fragment_index = index / 64;
    const uint element = index % 64;
    const uint k_octets = kdim / 8;
    const uint token = (fragment_index / k_octets) * 8 + element / 8;
    const uint k = (fragment_index % k_octets) * 8 + element % 8;
    packed[index] = token < tokens ? input[token * source_row_stride + k] : half(0);
}
