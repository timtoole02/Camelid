            &[packed_block],
            &[input_block],
            Q8_0PackedRows4Interleave::I8,
        );
        for lane in 0..4 {
            assert_eq!(
                actual[lane],
                expected[lane] as f32 * [0.25, 0.5, 0.75, 1.25][lane] * 0.125
            );
        }
        std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_q8_avx512vnni_dpwssd_packed_rows4_i8_matches_scalar_dot() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT", "on");
        let packed = std::array::from_fn(|idx| (idx as i8).wrapping_mul(11).wrapping_sub(37));
        let input = std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(19));
        let expected =
            q8_0_packed_rows4_block_dot_scalar(&packed, &input, Q8_0PackedRows4Interleave::I8);

        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            let actual =
                unsafe { q8_0_packed_4x8_block_avx512vnni_dpwssd(packed.as_ptr(), input.as_ptr()) };
            assert_eq!(actual, expected);

            let packed_block = Q8_0PackedRows4Block {
                scales: [0.25, 0.5, 0.75, 1.25],
                quants: packed,
            };
            let input_block = Q8_0Block {
                scale: 0.125,
                quants: input,
            };
            let actual = q8_0_packed_rows4_dot(
                &[packed_block],
                &[input_block],
                Q8_0PackedRows4Interleave::I8,
            );
            for lane in 0..4 {
                assert_eq!(
                    actual[lane],
                    expected[lane] as f32 * [0.25, 0.5, 0.75, 1.25][lane] * 0.125
                );
            }
        }
        std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT");
    }

    #[test]
    fn x86_q8_avx2_packed_rows4_hoisted_matmul_matches_scalar_dot() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST", "on");
        let packed_block = Q8_0PackedRows4Block {
            scales: [0.25, 0.5, 0.75, 1.25],
