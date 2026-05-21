
            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and add offset 128
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = _mm512_add_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
            }
            const int offset = TILE_N * TILE_K;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const int offset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
            const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q4_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
