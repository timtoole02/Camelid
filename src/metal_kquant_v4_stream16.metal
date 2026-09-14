// Default-off width-16 production probe, copied from the bit-exact stream kbench.
// Removes per-tile uint4 B/prefetch arrays and explicitly unrolls the tile axis.
// Trades more scalar fragment loads for lower live register state.
// Each SIMDgroup owns eight output rows and NT independent eight-token tiles.
// A's packed bytes and decoded fragments are reused across token tiles in registers.
// B uses [token tile][existing fragment-major panel], with no new quantization.
// Every C tile executes the original 32 MMAs per superblock in the same order;
// each output keeps the original increasing-superblock f32 fold.
template <uint NT>
inline void q4k_token_tiles_stream_body(
    device const float* scales, device const uchar* weights, device float* output,
    uint n_sb, uint rows, uint n_tokens, device const half* y_reg,
    device const half* ysums_reg, uint tile, uint lane
) {
    const uint2 fc = kquant_v4_fragment_coord(lane);
    const uint fcol = fc.x;
    const uint row = tile * 8 + fc.y;
    if (tile * 8 >= rows) return;
    const uint safe_row = min(row, rows - 1u);
    device const uint* y2 = reinterpret_cast<device const uint*>(y_reg);
    device const uint2* ys2 = reinterpret_cast<device const uint2*>(ysums_reg);
    float accum[NT][2];
    #pragma unroll
    for (uint t = 0; t < NT; ++t) {
        accum[t][0] = 0.0f;
        accum[t][1] = 0.0f;
    }
    for (uint sb = 0; sb < n_sb; ++sb) {
        device const uchar* block = weights + (safe_row * n_sb + sb) * 144;
        uchar sc[8], mn[8];
        q4k_scale_min_v2(block, sc, mn);
        half slo[4], shi[4], mnl[2];
        ushort wq[16];
        for (uint g = 0; g < 4; ++g) {
            slo[g] = half(int(sc[2 * g]));
            shi[g] = half(int(sc[2 * g + 1]));
        }
        mnl[0] = half(int(mn[fcol >> 1]));
        mnl[1] = half(int(mn[4 + (fcol >> 1)]));
        device const ushort* w16 = reinterpret_cast<device const ushort*>(block + 16 + fcol);
        for (uint g = 0; g < 4; ++g)
            for (uint m = 0; m < 4; ++m)
                wq[g * 4 + m] = w16[g * 16 + m * 4];
        const float dw = float(*reinterpret_cast<device const half*>(block));
        const float dm = float(*reinterpret_cast<device const half*>(block + 2));
        simdgroup_float8x8 c_main[NT];
        #pragma unroll
        for (uint t = 0; t < NT; ++t)
            c_main[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint g = 0; g < 4; ++g) {
            for (uint m = 0; m < 4; ++m) {
                const uint w = uint(wq[g * 4 + m]);
                simdgroup_half8x8 a;
                a.thread_elements()[0] = slo[g] * half(int(w & 0x0fu));
                a.thread_elements()[1] = slo[g] * half(int((w >> 8) & 0x0fu));
                #pragma unroll
                for (uint t = 0; t < NT; ++t) {
                    const half2 bh = as_type<half2>(y2[(((t * n_sb + sb) * 8 + 2 * g) * 32 + lane) * 4 + m]);
                    simdgroup_half8x8 b;
                    b.thread_elements()[0] = bh.x;
                    b.thread_elements()[1] = bh.y;
                    simdgroup_multiply_accumulate(c_main[t], a, b, c_main[t]);
                }
            }
            for (uint m = 0; m < 4; ++m) {
                const uint w = uint(wq[g * 4 + m]);
                simdgroup_half8x8 a;
                a.thread_elements()[0] = shi[g] * half(int((w >> 4) & 0x0fu));
                a.thread_elements()[1] = shi[g] * half(int((w >> 12) & 0x0fu));
                #pragma unroll
                for (uint t = 0; t < NT; ++t) {
                    const half2 bh = as_type<half2>(y2[(((t * n_sb + sb) * 8 + 2 * g + 1) * 32 + lane) * 4 + m]);
                    simdgroup_half8x8 b;
                    b.thread_elements()[0] = bh.x;
                    b.thread_elements()[1] = bh.y;
                    simdgroup_multiply_accumulate(c_main[t], a, b, c_main[t]);
                }
            }
        }
        simdgroup_float8x8 c_min[NT];
        #pragma unroll
        for (uint t = 0; t < NT; ++t)
            c_min[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint m = 0; m < 2; ++m) {
            simdgroup_half8x8 a;
            a.thread_elements()[0] = mnl[m];
            a.thread_elements()[1] = mnl[m];
            #pragma unroll
            for (uint t = 0; t < NT; ++t) {
                const uint2 ysq = ys2[(t * n_sb + sb) * 32 + lane];
                const half2 bh = as_type<half2>(m == 0 ? ysq.x : ysq.y);
                simdgroup_half8x8 b;
                b.thread_elements()[0] = bh.x;
                b.thread_elements()[1] = bh.y;
                simdgroup_multiply_accumulate(c_min[t], a, b, c_min[t]);
            }
        }
        if (row < rows) {
            #pragma unroll
            for (uint t = 0; t < NT; ++t)
                for (uint cell = 0; cell < 2; ++cell) {
                    const uint token = t * 8 + fcol + cell;
                    if (token < n_tokens) {
                        const float da = scales[token * n_sb + sb];
                        accum[t][cell] += (dw * da) * c_main[t].thread_elements()[cell]
                                         - (dm * da) * c_min[t].thread_elements()[cell];
                    }
                }
        }
    }
    if (row < rows)
        #pragma unroll
        for (uint t = 0; t < NT; ++t)
            for (uint cell = 0; cell < 2; ++cell) {
                const uint token = t * 8 + fcol + cell;
                if (token < n_tokens) output[token * rows + row] = accum[t][cell];
            }
}

template <uint NT>
inline void q6k_token_tiles_stream_body(
    device const float* scales, device const uchar* weights, device float* output,
    uint n_sb, uint rows, uint n_tokens, device const half* y_reg,
    uint tile, uint lane
) {
    const uint2 fc = kquant_v4_fragment_coord(lane);
    const uint fcol = fc.x;
    const uint row = tile * 8 + fc.y;
    if (tile * 8 >= rows) return;
    const uint safe_row = min(row, rows - 1u);
    device const uint* y2 = reinterpret_cast<device const uint*>(y_reg);
    float accum[NT][2];
    #pragma unroll
    for (uint t = 0; t < NT; ++t) {
        accum[t][0] = 0.0f;
        accum[t][1] = 0.0f;
    }
    for (uint sb = 0; sb < n_sb; ++sb) {
        device const uchar* block = weights + (safe_row * n_sb + sb) * 210;
        ushort wl[8], wh[8], wq[8], ws[8];
        for (uint h = 0; h < 2; ++h)
            for (uint j = 0; j < 4; ++j) {
                const uint o = h * 64 + 8 * j + fcol;
                wl[h * 4 + j] = *reinterpret_cast<device const ushort*>(block + o);
                wh[h * 4 + j] = *reinterpret_cast<device const ushort*>(block + o + 32);
                wq[h * 4 + j] = *reinterpret_cast<device const ushort*>(block + 128 + h * 32 + 8 * j + fcol);
            }
        for (uint i = 0; i < 8; ++i)
            ws[i] = *reinterpret_cast<device const ushort*>(block + 192 + 2 * i);
        simdgroup_float8x8 c_main[NT];
        #pragma unroll
        for (uint t = 0; t < NT; ++t)
            c_main[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint g = 0; g < 4; ++g) {
            for (uint i = 0; i < 8; ++i) {
                const uint kk = g * 8 + i;
                const uint h = kk >> 4;
                const uint quarter = (kk >> 2) & 3u;
                const uint j = kk & 3u;
                const uint idx = h * 4 + j;
                const uint src = (quarter & 1u) ? uint(wh[idx]) : uint(wl[idx]);
                const uint nib = (quarter & 2u) ? ((src >> 4) & 0x0f0fu) : (src & 0x0f0fu);
                const uint hb = ((uint(wq[idx]) >> (2 * quarter)) & 0x0303u) << 4;
                const uint v = nib | hb;
                const uint sword = uint(ws[kk >> 2]);
                const int sc = int(char((sword >> (8 * ((kk >> 1) & 1u))) & 0xffu));
                simdgroup_half8x8 a;
                a.thread_elements()[0] = half(sc * (int(v & 0xffu) - 32));
                a.thread_elements()[1] = half(sc * (int(v >> 8) - 32));
                #pragma unroll
                for (uint t = 0; t < NT; ++t) {
                    const half2 bh = as_type<half2>(y2[(((t * n_sb + sb) * 8 + (kk >> 2)) * 32 + lane) * 4 + (kk & 3u)]);
                    simdgroup_half8x8 b;
                    b.thread_elements()[0] = bh.x;
                    b.thread_elements()[1] = bh.y;
                    simdgroup_multiply_accumulate(c_main[t], a, b, c_main[t]);
                }
            }
        }
        if (row < rows) {
            const float dw = float(*reinterpret_cast<device const half*>(block + 208));
            #pragma unroll
            for (uint t = 0; t < NT; ++t)
                for (uint cell = 0; cell < 2; ++cell) {
                    const uint token = t * 8 + fcol + cell;
                    if (token < n_tokens) {
                        const float da = scales[token * n_sb + sb];
                        accum[t][cell] += (dw * da) * c_main[t].thread_elements()[cell];
                    }
                }
        }
    }
    if (row < rows)
        #pragma unroll
        for (uint t = 0; t < NT; ++t)
            for (uint cell = 0; cell < 2; ++cell) {
                const uint token = t * 8 + fcol + cell;
                if (token < n_tokens) output[token * rows + row] = accum[t][cell];
            }
}

#define STREAM_Q4_ENTRY(NAME, NT) \
kernel void NAME(device const float* s [[buffer(0)]], device const uchar* w [[buffer(2)]], \
 device float* o [[buffer(3)]], constant uint& ns [[buffer(4)]], constant uint& r [[buffer(5)]], \
 constant uint& n [[buffer(6)]], device const half* y [[buffer(7)]], device const half* ym [[buffer(8)]], \
 uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
 q4k_token_tiles_stream_body<NT>(s,w,o,ns,r,n,y,ym,tile,lane); }
#define STREAM_Q6_ENTRY(NAME, NT) \
kernel void NAME(device const float* s [[buffer(0)]], device const uchar* w [[buffer(2)]], \
 device float* o [[buffer(3)]], constant uint& ns [[buffer(4)]], constant uint& r [[buffer(5)]], \
 constant uint& n [[buffer(6)]], device const half* y [[buffer(7)]], \
 uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
 q6k_token_tiles_stream_body<NT>(s,w,o,ns,r,n,y,tile,lane); }
STREAM_Q4_ENTRY(q4k_token_tiles_stream_16, 2)
STREAM_Q6_ENTRY(q6k_token_tiles_stream_16, 2)

#undef STREAM_Q4_ENTRY
#undef STREAM_Q6_ENTRY
