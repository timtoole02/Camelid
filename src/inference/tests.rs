use super::*;
use crate::test_support::env_lock;
use std::io::Write;

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn projection_bias_broadcasts_across_every_token_row() {
    let projection =
        CpuTensor::from_f32("q", vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).unwrap();
    let bias = CpuTensor::from_f32("q.bias", vec![3], vec![0.5, -1.0, 2.0]).unwrap();

    let biased = add_projection_bias(projection, &bias, "test q projection").unwrap();

    assert_eq!(biased.shape.dims, vec![2, 3]);
    assert_eq!(biased.data, vec![1.5, 1.0, 5.0, -0.5, -3.0, -1.0]);
}

#[test]
fn projection_bias_rejects_a_mismatched_width() {
    let projection = CpuTensor::from_f32("q", vec![2, 3], vec![0.0; 6]).unwrap();
    let bias = CpuTensor::from_f32("q.bias", vec![2], vec![0.0; 2]).unwrap();

    let err = add_projection_bias(projection, &bias, "test q projection").unwrap_err();

    assert!(
        matches!(err, BackendError::RuntimeShapeMismatch(_)),
        "unexpected error: {err}"
    );
    assert!(format!("{err}").contains("does not match projection width 3"));
}

#[test]
fn resident_parity_forbids_reflects_the_cached_verdict() {
    // A key unlikely to collide with other parallel tests sharing the process-global map.
    let key = 0x5A5A_A5A5_1234_5678_u64;
    assert!(!resident_parity_forbids(key), "no verdict -> not forbidden");
    resident_parity_verdicts().lock().unwrap().insert(key, true);
    assert!(
        !resident_parity_forbids(key),
        "PASS verdict -> not forbidden"
    );
    resident_parity_verdicts()
        .lock()
        .unwrap()
        .insert(key, false);
    assert!(resident_parity_forbids(key), "FAIL verdict -> forbidden");
    // The fail-closed backstop (used when the probe panics) records a FAIL for a fresh key.
    let panic_key = 0x1357_9BDF_2468_ACE0_u64;
    assert!(
        !resident_parity_forbids(panic_key),
        "unprobed -> not forbidden"
    );
    record_resident_parity_fail(panic_key);
    assert!(
        resident_parity_forbids(panic_key),
        "recorded FAIL -> forbidden"
    );
    // Do not leak state to other tests.
    resident_parity_verdicts().lock().unwrap().remove(&key);
    resident_parity_verdicts()
        .lock()
        .unwrap()
        .remove(&panic_key);
}

#[test]
fn metal_resident_weight_gate_rejects_kquant_wire_storage() {
    let q8 = CpuTensor::from_q8_0_blocks(
        "q8.weight",
        TensorShape { dims: vec![1, 32] },
        vec![Q8_0Block {
            scale: 0.25,
            quants: [1; 32],
        }],
    )
    .unwrap();
    assert!(metal_resident_weight_eligible(&q8, false));

    let mut q4 = CpuTensor::from_f32("q4.weight", vec![1, 256], vec![0.0; 256]).unwrap();
    q4.source_type = Some(GgufTensorType::Q4K);
    q4.q4_k_wire_bytes = Some(Arc::new(vec![0; crate::tensor::Q4_K_BLOCK_BYTES]));
    q4.data.clear();

    assert!(
        !metal_resident_weight_eligible(&q4, false),
        "the Metal resident engine currently only implements Q8_0 weights"
    );
}

#[test]
fn resident_q8_moe_expert_view_slices_blocks_without_f32_materialization() {
    let blocks: Vec<_> = (0..4)
        .map(|index| Q8_0Block {
            scale: index as f32 + 1.0,
            quants: [index as i8; 32],
        })
        .collect();
    let blocks = Arc::new(blocks);
    let experts = CpuTensor::from_q8_0_shared_blocks(
        "experts",
        TensorShape {
            dims: vec![32, 2, 2],
        },
        Arc::clone(&blocks),
        0,
        4,
    )
    .unwrap();

    let second = expert_matrix_view(&experts, 1, 32, 2, "expert.1").unwrap();

    assert_eq!(second.shape.dims, vec![2, 32]);
    assert!(second.data.is_empty());
    assert!(
        second.q8_0_blocks.is_none(),
        "expert selection must not clone"
    );
    let shared = second
        .q8_0_shared_blocks
        .expect("shared resident Q8 blocks");
    assert!(Arc::ptr_eq(&shared.blocks, &blocks));
    assert_eq!(shared.as_slice().len(), 2);
    assert_eq!(shared.as_slice()[0].scale, 3.0);
    assert_eq!(shared.as_slice()[1].scale, 4.0);
}

#[test]
fn resident_moe_preflight_preserves_host_headroom() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let empty = MoeResidentMemoryEstimate::default();
    assert!(resident_moe_preflight_error(empty, Some((16 * GIB, 0))).is_none());
    let estimate = MoeResidentMemoryEstimate {
        resident_bytes: 2 * GIB,
        transient_bytes: GIB,
    };
    assert!(resident_moe_preflight_error(estimate, Some((16 * GIB, 7 * GIB))).is_none());
    let error = resident_moe_preflight_error(estimate, Some((16 * GIB, 6 * GIB)))
        .expect("resident blocks, loader scratch, and headroom should require seven GiB");
    assert!(error.contains("only 6.00 GiB is currently available"));
    assert!(error.contains("1.00 GiB loader scratch"));
    assert!(error.contains(crate::runtime_config::MOE_EXPERT_STORAGE_ENV));
}

#[test]
fn q3_k_wire_linear_routes_through_cpu_block_dot_without_dense_data() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q4K_DECODE", "1");

    let input_values = (0..Q6_K_VALUES_PER_BLOCK)
        .map(|index| (index as f32 - 127.0) / 64.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![1, Q6_K_VALUES_PER_BLOCK], input_values).unwrap();
    let output_width = 2;
    let mut wire = vec![0_u8; output_width * crate::tensor::Q3_K_BLOCK_BYTES];
    for (row, block) in wire
        .chunks_exact_mut(crate::tensor::Q3_K_BLOCK_BYTES)
        .enumerate()
    {
        for (index, byte) in block[..108].iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(17).wrapping_add(row as u8 * 29);
        }
        block[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    let mut weight = CpuTensor::from_f32(
        "blk.0.attn_q.weight",
        vec![Q6_K_VALUES_PER_BLOCK, output_width],
        vec![0.0; Q6_K_VALUES_PER_BLOCK * output_width],
    )
    .unwrap();
    weight.source_type = Some(GgufTensorType::Q3K);
    weight.q3_k_wire_bytes = Some(Arc::new(wire.clone()));
    weight.data.clear();

    let actual =
        linear_for_role_runtime(&input, &weight, "attention_q", "attention q", false).unwrap();
    let quantized_input = quantize_q8_k_blocks(&input.data);
    let expected = wire
        .chunks_exact(crate::tensor::Q3_K_BLOCK_BYTES)
        .map(|row| q3_k_wire_row_dot(row, &quantized_input))
        .collect::<Vec<_>>();

    assert_eq!(actual.shape.dims, vec![1, output_width]);
    assert_eq!(actual.data, expected);

    std::env::remove_var("CAMELID_X86_Q4K_DECODE");
}

#[test]
fn q3_k_wire_dot_matches_dequantized_reference() {
    let input = (0..Q6_K_VALUES_PER_BLOCK)
        .map(|index| ((index * 37 % 211) as f32 - 105.0) / 31.0)
        .collect::<Vec<_>>();
    let quantized = quantize_q8_k_blocks(&input);
    let mut wire = [0_u8; crate::tensor::Q3_K_BLOCK_BYTES];
    for (index, byte) in wire[..108].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(17).wrapping_add(23);
    }
    wire[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());

    let block = crate::tensor::Q3KBlock::from_bytes(&wire);
    let mut weights = [0.0_f32; Q6_K_VALUES_PER_BLOCK];
    block.dequantize(&mut weights);
    let dequantized_input = quantized[0]
        .qs
        .iter()
        .map(|quant| quantized[0].d * f32::from(*quant))
        .collect::<Vec<_>>();
    let expected = weights
        .iter()
        .zip(dequantized_input)
        .map(|(weight, input)| weight * input)
        .sum::<f32>();
    let actual = q3_k_wire_row_dot(&wire, &quantized);
    let tolerance = expected.abs().max(1.0) * 1e-5;

    assert!(
        (actual - expected).abs() <= tolerance,
        "Q3_K wire dot {actual} diverged from dequantized reference {expected}"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_row_dispatch_adversarial_parity() {
    let _env_guard = env_lock();

    let n_blocks = 4;

    let mut weight_blocks = Vec::with_capacity(n_blocks);
    let mut input_blocks = Vec::with_capacity(n_blocks);

    // Block 0: Mixed signs & normal values
    let mut w0 = [0_i8; 32];
    let mut in0 = [0_i8; 32];
    for idx in 0..32 {
        w0[idx] = if idx % 2 == 0 {
            (idx as i8) * 3
        } else {
            -(idx as i8) * 4
        };
        in0[idx] = if idx % 3 == 0 {
            29 - (idx as i8)
        } else {
            (idx as i8) - 45
        };
    }
    weight_blocks.push(Q8_0Block {
        scale: 0.125,
        quants: w0,
    });
    input_blocks.push(Q8_0Block {
        scale: 0.25,
        quants: in0,
    });

    // Block 1: Zero block (all zeros, scale 0)
    weight_blocks.push(Q8_0Block {
        scale: 0.0,
        quants: [0_i8; 32],
    });
    input_blocks.push(Q8_0Block {
        scale: 0.0,
        quants: [0_i8; 32],
    });

    // Block 2: Boundary case with i8::MIN (-128) and i8::MAX (127)
    let mut w2 = [0_i8; 32];
    let mut in2 = [0_i8; 32];
    for idx in 0..32 {
        w2[idx] = match idx % 4 {
            0 => i8::MIN,
            1 => i8::MAX,
            2 => 0,
            _ => -7,
        };
        in2[idx] = match idx % 5 {
            0 => i8::MIN,
            1 => i8::MAX,
            2 => 5,
            _ => 13,
        };
    }
    weight_blocks.push(Q8_0Block {
        scale: 1.5,
        quants: w2,
    });
    input_blocks.push(Q8_0Block {
        scale: 0.75,
        quants: in2,
    });

    // Block 3: Mixed small values/subnormal scales
    let mut w3 = [0_i8; 32];
    let mut in3 = [0_i8; 32];
    for idx in 0..32 {
        w3[idx] = (idx as i8) - 16;
        in3[idx] = 16 - (idx as i8);
    }
    weight_blocks.push(Q8_0Block {
        scale: 1e-37,
        quants: w3,
    });
    input_blocks.push(Q8_0Block {
        scale: 1e-38,
        quants: in3,
    });

    // Test single-row dot product
    std::env::set_var("CAMELID_Q8_ROW_DISPATCH", "off");
    let scalar_dot = q8_0_dot_rows(&weight_blocks, &input_blocks);

    std::env::set_var("CAMELID_Q8_ROW_DISPATCH", "on");
    let simd_dot = q8_0_dot_rows(&weight_blocks, &input_blocks);

    assert_eq!(
        scalar_dot, simd_dot,
        "Single-row dot product mismatch (scalar: {}, simd: {})",
        scalar_dot, simd_dot
    );

    // Test two-row dot product
    let mut second_weight_blocks = Vec::with_capacity(n_blocks);

    let mut w0_2 = [0_i8; 32];
    for idx in 0..32 {
        w0_2[idx] = if idx % 2 == 0 { -10 } else { 12 };
    }
    second_weight_blocks.push(Q8_0Block {
        scale: 0.5,
        quants: w0_2,
    });

    second_weight_blocks.push(Q8_0Block {
        scale: 0.0,
        quants: [0_i8; 32],
    });

    let mut w2_2 = [0_i8; 32];
    for idx in 0..32 {
        w2_2[idx] = if idx % 3 == 0 { i8::MIN } else { 45 };
    }
    second_weight_blocks.push(Q8_0Block {
        scale: 2.25,
        quants: w2_2,
    });

    let mut w3_2 = [0_i8; 32];
    for idx in 0..32 {
        w3_2[idx] = -(idx as i8);
    }
    second_weight_blocks.push(Q8_0Block {
        scale: 1e-35,
        quants: w3_2,
    });

    std::env::set_var("CAMELID_Q8_ROW_DISPATCH", "off");
    let scalar_two_dot = q8_0_two_dot_rows(&weight_blocks, &second_weight_blocks, &input_blocks);

    std::env::set_var("CAMELID_Q8_ROW_DISPATCH", "on");
    let simd_two_dot = q8_0_two_dot_rows(&weight_blocks, &second_weight_blocks, &input_blocks);

    assert_eq!(
        scalar_two_dot, simd_two_dot,
        "Two-row dot product mismatch (scalar: {:?}, simd: {:?})",
        scalar_two_dot, simd_two_dot
    );

    std::env::remove_var("CAMELID_Q8_ROW_DISPATCH");
}

fn assert_slice_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "slice length mismatch");
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() < 1e-5,
            "expected index {idx} to be {expected}, got {actual}"
        );
    }
}

fn assert_slice_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "slice length mismatch");
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "expected index {idx} to be within {tolerance} of {expected}, got {actual}"
        );
    }
}

fn no_rope_scaling() -> RopeScaling {
    RopeScaling {
        kind: RopeScalingKind::None,
        factor: 1.0,
        original_context_length: None,
        low_freq_factor: None,
        high_freq_factor: None,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_avx2_kernel_matches_scalar_dot() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
    let weight = std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(59));
    let input = std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(17));
    let encoded = weight.map(|value| value as u8);
    let expected = q8_0_block_int_dot_horizontal_sum_scalar(&weight, &input);

    assert_eq!(q8_0_block_int_dot_horizontal_sum(&weight, &input), expected);
    assert_eq!(
        q8_0_block_int_dot_horizontal_sum_encoded(&encoded, &input),
        expected
    );
    std::env::remove_var("CAMELID_X86_Q8_KERNEL");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_avx2_kernel_matches_scalar_dot_for_negative_128() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
    let weight = std::array::from_fn(|idx| match idx % 4 {
        0 => -128,
        1 => 127,
        2 => -7,
        _ => idx as i8,
    });
    let input = std::array::from_fn(|idx| match idx % 5 {
        0 => -128,
        1 => 127,
        2 => 5,
        _ => (idx as i8).wrapping_mul(-3),
    });
    let expected = q8_0_block_int_dot_horizontal_sum_scalar(&weight, &input);
    assert_eq!(q8_0_block_int_dot_horizontal_sum(&weight, &input), expected);
    std::env::remove_var("CAMELID_X86_Q8_KERNEL");
}

#[test]
fn x86_q8_packed_rows4_matmul_chunk_groups_env_override() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK");
    std::env::set_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL", "on");
    let mut plan = ResolvedRuntimePlan::from_env().unwrap();
    assert_eq!(
        plan.q8_packed_rows4_matmul_schedule.groups_per_chunk,
        q8_runtime::X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT
    );
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "32");
    plan = ResolvedRuntimePlan::from_env().unwrap();
    assert_eq!(plan.q8_packed_rows4_matmul_schedule.groups_per_chunk, 32);
    assert_eq!(
        q8_packed_rows4_matmul_parallel_chunk_floats(128, plan.q8_packed_rows4_matmul_schedule),
        128
    );
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "0");
    plan = ResolvedRuntimePlan::from_env().unwrap();
    assert_eq!(
        plan.q8_packed_rows4_matmul_schedule.groups_per_chunk,
        q8_runtime::X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT
    );
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL");
    plan = ResolvedRuntimePlan::from_env().unwrap();
    assert_eq!(
        plan.q8_packed_rows4_matmul_schedule.groups_per_chunk,
        q8_runtime::X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT
    );
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK");
}

#[test]
fn x86_q8_ffn_down_gemm4_row_group_schedule_respects_min_input_groups() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS");
    assert_eq!(
        x86_q8_ffn_down_gemm4_row_group_min_input_groups(),
        X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS
    );
    assert!(!should_use_x86_q8_ffn_down_gemm4_row_group_schedule(
        false,
        X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS
    ));
    assert!(!should_use_x86_q8_ffn_down_gemm4_row_group_schedule(
        true,
        X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS - 1
    ));
    assert_eq!(
        should_use_x86_q8_ffn_down_gemm4_row_group_schedule(
            true,
            X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS
        ),
        rayon::current_num_threads() > 1
    );

    std::env::set_var(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
        "2",
    );
    assert_eq!(x86_q8_ffn_down_gemm4_row_group_min_input_groups(), 2);
    assert_eq!(
        should_use_x86_q8_ffn_down_gemm4_row_group_schedule(true, 2),
        rayon::current_num_threads() > 1
    );
    std::env::set_var(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
        "0",
    );
    assert_eq!(
        x86_q8_ffn_down_gemm4_row_group_min_input_groups(),
        X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS
    );
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_avx2_packed_rows4_i8_matches_scalar_dot() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT", "on");
    let packed = std::array::from_fn(|idx| (idx as i8).wrapping_mul(11).wrapping_sub(37));
    let input = std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(19));
    let expected =
        q8_0_packed_rows4_block_dot_scalar(&packed, &input, Q8_0PackedRows4Interleave::I8);

    if std::arch::is_x86_feature_detected!("avx2") {
        let actual = unsafe { q8_0_packed_4x8_block_avx2(packed.as_ptr(), input.as_ptr()) };
        assert_eq!(actual, expected);
    }

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_avx512vnni_dpbusd_packed_rows4_i8_matches_scalar_dot() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT", "on");
    let packed = std::array::from_fn(|idx| ((idx * 11 % 127) as i8).wrapping_sub(63));
    let input = std::array::from_fn(|idx| ((idx * 5 % 127) as i8).wrapping_sub(63));
    let expected =
        q8_0_packed_rows4_block_dot_scalar(&packed, &input, Q8_0PackedRows4Interleave::I8);

    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni")
    {
        let actual =
            unsafe { q8_0_packed_4x8_block_avx512vnni_dpbusd(packed.as_ptr(), input.as_ptr()) };
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
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT");
}

#[test]
fn x86_q8_avx2_packed_rows4_hoisted_matmul_matches_scalar_dot() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST", "on");
    let packed_block = Q8_0PackedRows4Block {
        scales: [0.25, 0.5, 0.75, 1.25],
        quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(11).wrapping_sub(37)),
    };
    let input_block = Q8_0Block {
        scale: 0.125,
        quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(19)),
    };
    let expected = q8_0_packed_rows4_dot(
        std::slice::from_ref(&packed_block),
        std::slice::from_ref(&input_block),
        Q8_0PackedRows4Interleave::I8,
    );
    let actual = q8_0_packed_rows4_dot_i8_matmul(
        std::slice::from_ref(&packed_block),
        std::slice::from_ref(&input_block),
        x86_q8_packed_rows4_avx2_dot_hoist_enabled(),
    );
    assert_eq!(actual, expected);
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST");
}

#[test]
fn x86_q8_avx2_packed_rows4_decode_hoist_projection_matches_scalar_dot() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST", "on");
    let blocks_per_row = 2;
    let packed = Q8_0PackedRows4 {
        rows: 4,
        blocks_per_row,
        interleave: Q8_0PackedRows4Interleave::I8,
        amx_blocks: None,
        vnni_packed: None,
        blocks: (0..blocks_per_row)
            .map(|block_idx| Q8_0PackedRows4Block {
                scales: [0.25, 0.5, 0.75, 1.25],
                quants: std::array::from_fn(|idx| {
                    (idx as i8)
                        .wrapping_mul(3)
                        .wrapping_add((block_idx as i8).wrapping_mul(17))
                }),
            })
            .collect(),
    };
    let quantized_input: Vec<Q8_0Block> = (0..blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| {
                (idx as i8)
                    .wrapping_mul(5)
                    .wrapping_sub((block_idx as i8).wrapping_mul(13))
            }),
        })
        .collect();
    let expected = q8_0_packed_rows4_dot(
        &packed.blocks,
        &quantized_input,
        Q8_0PackedRows4Interleave::I8,
    );
    let mut actual = [0.0_f32; 4];
    q8_0_packed_rows4_single_input_projection_into(&packed, &quantized_input, &mut actual).unwrap();
    assert_eq!(actual, expected);
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_DECODE_HOIST");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn x86_q8_packed_rows4_decode_rawptr_avx2_matches_scalar_projection() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_DECODE_RAWPTR_AVX2");
    assert!(!x86_q8_packed_rows4_decode_rawptr_avx2_enabled());
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let blocks_per_row = 3;
    let rows = 8;
    let packed = Q8_0PackedRows4 {
        rows,
        blocks_per_row,
        interleave: Q8_0PackedRows4Interleave::I8,
        amx_blocks: None,
        vnni_packed: None,
        blocks: (0..rows / 4 * blocks_per_row)
            .map(|block_idx| Q8_0PackedRows4Block {
                scales: [
                    0.25 + block_idx as f32 * 0.01,
                    0.5,
                    0.75,
                    1.25 - block_idx as f32 * 0.005,
                ],
                quants: std::array::from_fn(|idx| {
                    (idx as i8)
                        .wrapping_mul(7)
                        .wrapping_add((block_idx as i8).wrapping_mul(11))
                        .wrapping_sub(61)
                }),
            })
            .collect(),
    };
    let quantized_input: Vec<Q8_0Block> = (0..blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.125 + block_idx as f32 * 0.03125,
            quants: std::array::from_fn(|idx| {
                (idx as i8)
                    .wrapping_mul(5)
                    .wrapping_sub((block_idx as i8).wrapping_mul(17))
                    .wrapping_add(29)
            }),
        })
        .collect();

    let mut expected = vec![0.0_f32; rows];
    q8_0_packed_rows4_single_input_projection_into(&packed, &quantized_input, &mut expected)
        .unwrap();

    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_DECODE_RAWPTR_AVX2", "on");
    assert!(x86_q8_packed_rows4_decode_rawptr_avx2_enabled());
    let mut actual = vec![0.0_f32; rows];
    q8_0_packed_rows4_single_input_projection_into_with_decode_chunking(
        &packed,
        &quantized_input,
        &mut actual,
        true,
    )
    .unwrap();

    assert_slice_close(&actual, &expected);
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_DECODE_RAWPTR_AVX2");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_ffn_down_decode_uses_avx2_reference_gate() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
    if !std::arch::is_x86_feature_detected!("avx2") {
        std::env::remove_var("CAMELID_X86_Q8_KERNEL");
        return;
    }

    let input = CpuTensor::from_f32(
        "hidden",
        vec![1, Q8_0_BLOCK_VALUES],
        (0..Q8_0_BLOCK_VALUES)
            .map(|idx| (idx as f32 - 15.0) / 8.0)
            .collect(),
    )
    .unwrap();
    let row_major_blocks: Vec<Q8_0Block> = (0..4)
        .map(|row| Q8_0Block {
            scale: 0.25 + row as f32 * 0.125,
            quants: std::array::from_fn(|idx| {
                (idx as i8).wrapping_mul(3).wrapping_add(row as i8 * 11)
            }),
        })
        .collect();
    let packed =
        Q8_0PackedRows4::from_rows(4, 1, Q8_0PackedRows4Interleave::I8, &row_major_blocks).unwrap();
    let weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        TensorShape {
            dims: vec![4, Q8_0_BLOCK_VALUES],
        },
        packed.clone(),
    );
    let runtime_plan = ResolvedRuntimePlan::from_env().unwrap();
    assert!(!runtime_plan.q8.ffn_down_decode_consumer);

    let actual = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &weight,
        "ffn_down",
        "ffn_down",
        &runtime_plan,
    )
    .unwrap()
    .unwrap();
    let quantized_input = quantize_q8_0_row(&input.data);
    let expected =
        q8_0_packed_rows4_single_input_projection(&packed, &quantized_input.blocks, 4, "expected")
            .unwrap();
    assert_eq!(actual.shape.dims, vec![1, 4]);
    assert_eq!(actual.data, expected.data);
    std::env::remove_var("CAMELID_X86_Q8_KERNEL");
}

#[test]
fn q8_0_block_reader_smoke() {
    let _q8_guard = crate::test_support::q8_file_state_lock();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let scale_bits = 0x3c00u16;
    let mut block_data = vec![0u8; Q8BlockReader::BLOCK_SIZE_BYTES];
    block_data[0..2].copy_from_slice(&scale_bits.to_le_bytes());
    block_data[2] = 10i8 as u8;
    block_data[3] = 20i8 as u8;

    temp_file.write_all(&block_data).unwrap();
    temp_file.flush().unwrap();

    let reader = Q8BlockReader::new(0, 1);
    let file = temp_file.reopen().unwrap();
    let mut dest = vec![0.0; Q8BlockReader::WEIGHTS_PER_BLOCK];
    reader
        .dequantize_block_to_slice(&file, 0, &mut dest)
        .unwrap();

    assert_eq!(dest[0], 10.0);
    assert_eq!(dest[1], 20.0);
    assert!(dest[2..].iter().all(|value| *value == 0.0));
}

fn write_q8_0_test_block(
    out: &mut impl Write,
    scale: f32,
    quants: [i8; Q8_0_BLOCK_VALUES],
) -> Q8_0Block {
    let scale_bits = f32_to_f16_bits(scale);
    out.write_all(&scale_bits.to_le_bytes()).unwrap();
    out.write_all(&quants.map(|value| value as u8)).unwrap();
    Q8_0Block {
        scale: f16_bits_to_f32(scale_bits),
        quants,
    }
}

#[test]
fn q8_file_backed_output_projection_reuses_weight_read_across_batch_rows() {
    let _q8_stats_guard = crate::tensor::q8_stats_test_lock();
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let mut first_row = [0_i8; Q8_0_BLOCK_VALUES];
    let mut second_row = [0_i8; Q8_0_BLOCK_VALUES];
    for idx in 0..Q8_0_BLOCK_VALUES {
        first_row[idx] = (idx as i8 % 7) - 3;
        second_row[idx] = 4 - (idx as i8 % 9);
    }
    let weight_blocks = [
        write_q8_0_test_block(&mut temp_file, 0.5, first_row),
        write_q8_0_test_block(&mut temp_file, 0.25, second_row),
    ];
    temp_file.flush().unwrap();

    let input = CpuTensor::from_f32(
        "prefill-output-norm",
        vec![3, Q8_0_BLOCK_VALUES],
        (0..(3 * Q8_0_BLOCK_VALUES))
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.05)
            .collect(),
    )
    .unwrap();
    let weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        TensorShape {
            dims: vec![2, Q8_0_BLOCK_VALUES],
        },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, weight_blocks.len()),
    );
    let start = q8_0_file_read_stats();

    let actual = output_projection_with_layout(
        &input,
        &weight,
        "logits",
        OutputProjectionLayout::TokenMajor,
    )
    .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    let mut expected = Vec::new();
    for input_row in input.data.chunks_exact(Q8_0_BLOCK_VALUES) {
        let quantized_input = quantize_q8_0_row(input_row);
        expected.push(q8_0_dot_rows(&weight_blocks[0..1], &quantized_input.blocks));
        expected.push(q8_0_dot_rows(&weight_blocks[1..2], &quantized_input.blocks));
    }
    assert_eq!(actual.shape.dims, vec![3, 2]);
    assert_slice_close(&actual.data, &expected);
    assert_eq!(reads.read_calls, 1);
    assert_eq!(
        reads.read_bytes,
        (weight_blocks.len() * Q8BlockReader::BLOCK_SIZE_BYTES) as u64
    );
}

#[test]
fn q8_file_backed_output_projection_empty_batch_skips_weight_reads() {
    let _q8_stats_guard = crate::tensor::q8_stats_test_lock();
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let weight_blocks = [
        write_q8_0_test_block(&mut temp_file, 1.0, [1_i8; Q8_0_BLOCK_VALUES]),
        write_q8_0_test_block(&mut temp_file, 1.0, [-1_i8; Q8_0_BLOCK_VALUES]),
    ];
    temp_file.flush().unwrap();

    let input = CpuTensor::from_f32(
        "empty-prefill-output-norm",
        vec![0, Q8_0_BLOCK_VALUES],
        Vec::new(),
    )
    .unwrap();
    let weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        TensorShape {
            dims: vec![weight_blocks.len(), Q8_0_BLOCK_VALUES],
        },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, weight_blocks.len()),
    );
    let start = q8_0_file_read_stats();

    let actual = output_projection_with_layout(
        &input,
        &weight,
        "logits",
        OutputProjectionLayout::TokenMajor,
    )
    .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_eq!(actual.shape.dims, vec![0, weight_blocks.len()]);
    assert!(actual.data.is_empty());
    assert_eq!(reads.read_calls, 0);
    assert_eq!(reads.read_bytes, 0);
    assert!(!weight
        .q8_0_file_backing
        .as_ref()
        .unwrap()
        .file_handle_cached());
}

fn memory_sample(
    rss_kib: u64,
    kv_position: usize,
    allocated_sequence_length: usize,
) -> LlamaMemorySample {
    let elements = allocated_sequence_length * 2;
    LlamaMemorySample {
        rss_kib: Some(rss_kib),
        kv_cache_position: kv_position,
        kv_cache_allocated_sequence_length: allocated_sequence_length,
        kv_cache_allocated_elements: elements,
        kv_cache_allocated_bytes: (elements * std::mem::size_of::<f32>()) as u64,
    }
}

fn test_forward_memory(start: LlamaMemorySample) -> LlamaForwardMemoryTimings {
    LlamaForwardMemoryTimings::new(
        start,
        LlamaWeightMaterializationStats::default(),
        q8_0_file_read_stats(),
    )
}

fn tiny_prefill_schedule_weights(attention_q: CpuTensor) -> LlamaLoadedWeights {
    LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32("token_embd.weight", vec![2, 2], vec![1.0; 4])
            .unwrap(),
        output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0; 2]).unwrap(),
        output: None,
        rope_freqs: None,
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
        layers: vec![LlamaLayerWeights {
            attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0; 2])
                .unwrap(),
            attention_q,
            attention_k: CpuTensor::from_f32("blk.0.attn_k.weight", vec![2, 2], vec![1.0; 4])
                .unwrap(),
            attention_v: CpuTensor::from_f32("blk.0.attn_v.weight", vec![2, 2], vec![1.0; 4])
                .unwrap(),
            attention_output: CpuTensor::from_f32(
                "blk.0.attn_output.weight",
                vec![2, 2],
                vec![1.0; 4],
            )
            .unwrap(),
            ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0; 2]).unwrap(),
            ffn_gate: CpuTensor::from_f32("blk.0.ffn_gate.weight", vec![2, 2], vec![1.0; 4])
                .unwrap(),
            ffn_up: CpuTensor::from_f32("blk.0.ffn_up.weight", vec![2, 2], vec![1.0; 4]).unwrap(),
            ffn_down: CpuTensor::from_f32("blk.0.ffn_down.weight", vec![2, 2], vec![1.0; 4])
                .unwrap(),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
    }
}

#[test]
fn prefill_chunk_token_count_accepts_full_prompt_probe() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
    assert_eq!(prefill_chunk_token_count(2047), 256);

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "256");
    assert_eq!(prefill_chunk_token_count(2047), 256);

    for value in ["all", "full", "prompt", "unbounded", " FULL "] {
        std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", value);
        assert_eq!(prefill_chunk_token_count(2047), 2047);
    }

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "0");
    assert_eq!(prefill_chunk_token_count(2047), 256);
    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
}

#[test]
fn resolve_prefill_thread_count_widens_only_on_measured_default() {
    // Explicit override wins, including disabling the wider pool.
    assert_eq!(
        resolve_prefill_thread_count_from(Some("6"), false, 16, true),
        Some(6)
    );
    assert_eq!(
        resolve_prefill_thread_count_from(Some(" 12 "), true, 16, true),
        Some(12)
    );
    assert_eq!(
        resolve_prefill_thread_count_from(Some("off"), false, 16, true),
        None
    );
    assert_eq!(
        resolve_prefill_thread_count_from(Some("GLOBAL"), false, 16, true),
        None
    );
    assert_eq!(
        resolve_prefill_thread_count_from(Some("0"), false, 16, true),
        None
    );
    // Unparseable / non-positive overrides fall back to the global pool.
    assert_eq!(
        resolve_prefill_thread_count_from(Some("abc"), false, 16, true),
        None
    );

    // No override: widen to logical cores only on a measured target...
    assert_eq!(
        resolve_prefill_thread_count_from(None, false, 16, true),
        Some(16)
    );
    // ...never on unmeasured targets...
    assert_eq!(
        resolve_prefill_thread_count_from(None, false, 16, false),
        None
    );
    // ...and never silently over an operator's hand-pinned global thread count.
    assert_eq!(
        resolve_prefill_thread_count_from(None, true, 16, true),
        None
    );
}

#[test]
fn resolve_decode_thread_count_promotes_physical_default_with_kill_switch() {
    // No flags, no policy input: no decode pool (pre-promotion inline path).
    assert_eq!(
        resolve_decode_thread_count_from(None, false, 16, None),
        None
    );
    // Promoted default policy: dedicated pool at detected physical cores...
    assert_eq!(
        resolve_decode_thread_count_from(None, false, 16, Some(8)),
        Some(8)
    );
    // ...including physical == global (the isolation configuration)...
    assert_eq!(
        resolve_decode_thread_count_from(None, false, 8, Some(8)),
        Some(8)
    );
    // ...but never wider than an operator-narrowed global.
    assert_eq!(
        resolve_decode_thread_count_from(None, false, 4, Some(8)),
        None
    );
    // Explicit width wins over the policy.
    assert_eq!(
        resolve_decode_thread_count_from(Some("4"), false, 16, Some(8)),
        Some(4)
    );
    assert_eq!(
        resolve_decode_thread_count_from(Some(" 6 "), true, 16, Some(8)),
        Some(6)
    );
    // Explicit 0/off is the KILL SWITCH: disables the pool entirely, over
    // both the dedicated flag and the default policy.
    assert_eq!(
        resolve_decode_thread_count_from(Some("0"), true, 16, Some(8)),
        None
    );
    assert_eq!(
        resolve_decode_thread_count_from(Some("off"), false, 16, Some(8)),
        None
    );
    // Empty/invalid specs fall through to dedicated, then the policy.
    assert_eq!(
        resolve_decode_thread_count_from(Some(""), false, 16, Some(8)),
        Some(8)
    );
    assert_eq!(
        resolve_decode_thread_count_from(Some("abc"), true, 16, None),
        Some(16)
    );
    assert_eq!(
        resolve_decode_thread_count_from(Some("abc"), false, 16, None),
        None
    );
    // Dedicated alone isolates at the global width.
    assert_eq!(
        resolve_decode_thread_count_from(None, true, 12, None),
        Some(12)
    );
}

#[test]
fn prefill_layer_major_chunk_token_count_has_separate_headroom_default() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
    assert_eq!(prefill_chunk_token_count(2047), 256);
    assert_eq!(prefill_layer_major_chunk_token_count(2047), 512);

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "128");
    assert_eq!(prefill_layer_major_chunk_token_count(2047), 128);

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "1024");
    assert_eq!(prefill_layer_major_chunk_token_count(2047), 1024);

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "all");
    assert_eq!(prefill_layer_major_chunk_token_count(2047), 2047);

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "0");
    assert_eq!(prefill_layer_major_chunk_token_count(2047), 512);
    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS");
}

#[test]
fn q8_file_reader_batch_chunk_rows_respect_output_scratch_budget() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "1024");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64");

    assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 32);
    assert_eq!(
        q8_0_file_reader_chunk_rows_for_batch(32, 100, 1, true).unwrap(),
        32
    );
    assert_eq!(
        q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, true).unwrap(),
        2
    );
    assert_eq!(
        q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, false).unwrap(),
        32
    );
    assert_eq!(q8_0_file_reader_chunk_rows(32, 63).unwrap(), 63);
    assert_eq!(q8_0_file_reader_chunk_rows(32, 64).unwrap(), 64);
    assert_eq!(q8_0_file_reader_chunk_rows(32, 65).unwrap(), 32);

    std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "1 KiB");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64_B");
    assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 32);
    assert_eq!(
        q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, true).unwrap(),
        2
    );
    assert_eq!(
        q8_0_file_reader_chunk_rows_for_batch(32, 100, 8, false).unwrap(),
        32
    );

    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");
}

#[test]
fn q8_file_reader_parallel_output_falls_back_when_default_scratch_fragments_full_tensor_read() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES", "4096");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(should_parallelize_q8_0_file_reader_output(100));
        assert_eq!(q8_0_file_reader_chunk_rows(32, 100).unwrap(), 100);
        assert_eq!(
            q8_0_file_reader_output_scratch_chunk_rows(1_000_000, 100).unwrap(),
            16
        );
        assert!(!should_use_q8_0_file_reader_parallel_output(32, 100, 1_000_000).unwrap());

        std::env::set_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES", "64");
        assert!(should_use_q8_0_file_reader_parallel_output(32, 100, 8).unwrap());
    });

    std::env::remove_var("CAMELID_PARALLEL_LINEAR");
    std::env::remove_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");
}

#[test]
fn q8_file_reader_default_coalesces_llama3_8b_ffn_q8_shapes() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES");

    let llama3_8b_hidden_row_bytes = 4096 / Q8_0_BLOCK_VALUES * Q8BlockReader::BLOCK_SIZE_BYTES;
    let llama3_8b_ffn_row_bytes = 14336 / Q8_0_BLOCK_VALUES * Q8BlockReader::BLOCK_SIZE_BYTES;

    assert_eq!(
        q8_0_file_reader_chunk_rows(llama3_8b_hidden_row_bytes, 14336).unwrap(),
        14336
    );
    assert_eq!(
        q8_0_file_reader_chunk_rows(llama3_8b_ffn_row_bytes, 4096).unwrap(),
        4096
    );
}

#[test]
fn prefill_layer_major_defaults_only_for_lazy_q8_backing() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let dense_weights = tiny_prefill_schedule_weights(
        CpuTensor::from_f32("blk.0.attn_q.weight", vec![2, 2], vec![1.0; 4]).unwrap(),
    );
    assert!(!prefill_layer_major_enabled(&dense_weights));

    let lazy_q8_attention_q = CpuTensor::from_f32_with_source_type(
        "blk.0.attn_q.weight",
        vec![2, 2],
        vec![1.0; 4],
        Some(GgufTensorType::Q8_0),
    )
    .unwrap()
    .with_q8_0_file_backing(Q8_0FileBacking::new("unused.gguf".into(), 0, 1));
    let lazy_q8_weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);
    assert!(prefill_layer_major_enabled(&lazy_q8_weights));

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", "1");
    assert!(prefill_layer_major_enabled(&dense_weights));

    for value in ["0", "false", "off", "disabled", " FALSE ", "Off"] {
        std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", value);
        assert!(!prefill_layer_major_enabled(&lazy_q8_weights));
    }

    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR");
}

#[test]
fn prefill_layer_major_q8_cache_uses_scoped_default_only_for_lazy_q8() {
    let _q8_stats_guard = crate::tensor::q8_stats_test_lock();
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let dense_weights = tiny_prefill_schedule_weights(
        CpuTensor::from_f32("blk.0.attn_q.weight", vec![2, 2], vec![1.0; 4]).unwrap(),
    );
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&dense_weights, 2),
        None
    );

    let lazy_q8_attention_q = CpuTensor::from_f32_with_source_type(
        "blk.0.attn_q.weight",
        vec![2, 2],
        vec![1.0; 4],
        Some(GgufTensorType::Q8_0),
    )
    .unwrap()
    .with_q8_0_file_backing(Q8_0FileBacking::new("unused.gguf".into(), 0, 1));
    let lazy_q8_weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
        None
    );
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 2),
        Some(DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES)
    );

    let large_layer_blocks =
        (DEFAULT_PREFILL_LAYER_MAJOR_Q8_FILE_CACHE_BYTES / Q8BlockReader::BLOCK_SIZE_BYTES) + 1;
    let large_layer_capacity = large_layer_blocks * Q8BlockReader::BLOCK_SIZE_BYTES;
    let large_lazy_q8_attention_q = CpuTensor::q8_0_file_backed_linear(
        "blk.0.attn_q.weight",
        TensorShape { dims: vec![1, 32] },
        Q8_0FileBacking::new("unused.gguf".into(), 0, large_layer_blocks),
    );
    let large_lazy_q8_weights = tiny_prefill_schedule_weights(large_lazy_q8_attention_q);
    assert_eq!(
        large_lazy_q8_weights.largest_q8_0_file_backed_layer_storage_bytes(),
        large_layer_capacity as u64
    );
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&large_lazy_q8_weights, 2),
        Some(large_layer_capacity)
    );

    std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "64 MiB");
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 2),
        None
    );

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES", "0");
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
        Some(0)
    );

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES", "1 MiB");
    assert_eq!(
        prefill_layer_major_q8_file_cache_capacity_override(&lazy_q8_weights, 1),
        Some(1024 * 1024)
    );
}

#[test]
fn prefill_layer_major_scoped_q8_cache_reuses_file_reads_across_chunks() {
    let _q8_stats_guard = crate::tensor::q8_stats_test_lock();
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    let _ = q8_0_file_read_stats();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for _ in 0..32 {
        temp_file
            .write_all(&f32_to_f16_bits(1.0).to_le_bytes())
            .unwrap();
        temp_file.write_all(&[0_u8; Q8_0_BLOCK_VALUES]).unwrap();
    }
    temp_file.flush().unwrap();

    let config = LlamaModelConfig {
        context_length: 2,
        embedding_length: 32,
        block_count: 1,
        feed_forward_length: 32,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(32),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(2),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let dense_vector = |name: &str| CpuTensor::from_f32(name, vec![32], vec![1.0; 32]).unwrap();
    let dense_matrix =
        |name: &str| CpuTensor::from_f32(name, vec![32, 32], vec![0.0; 32 * 32]).unwrap();
    let attention_q = CpuTensor::q8_0_file_backed_linear(
        "blk.0.attn_q.weight",
        TensorShape { dims: vec![32, 32] },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 32),
    );
    let weights = LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd.weight",
            vec![2, 32],
            (0..64).map(|idx| idx as f32 * 0.001).collect(),
        )
        .unwrap(),
        output_norm: dense_vector("output_norm.weight"),
        output: None,
        rope_freqs: None,
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
        layers: vec![LlamaLayerWeights {
            attention_norm: dense_vector("blk.0.attn_norm.weight"),
            attention_q,
            attention_k: dense_matrix("blk.0.attn_k.weight"),
            attention_v: dense_matrix("blk.0.attn_v.weight"),
            attention_output: dense_matrix("blk.0.attn_output.weight"),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            ffn_norm: dense_vector("blk.0.ffn_norm.weight"),
            ffn_gate: dense_matrix("blk.0.ffn_gate.weight"),
            ffn_up: dense_matrix("blk.0.ffn_up.weight"),
            ffn_down: dense_matrix("blk.0.ffn_down.weight"),
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
    };
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();
    let start = q8_0_file_read_stats();

    let timings = session
        .forward_prefill_layer_major_timed_fast(&[0, 1], 1)
        .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_eq!(timings.layers.len(), 1);
    assert_eq!(session.kv_cache.position, 2);
    assert_eq!(reads.read_calls, 1);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * 32) as u64
    );
    assert_eq!(reads.cache_hits, 1);
    assert_eq!(
        reads.cache_hit_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * 32) as u64
    );
}

/// Builds a minimal single-layer synthetic session (embedding 32, one KV head,
/// head_dim 32, `context_length` positions) for the KV-budget guard tests. The
/// weights are shape-valid but numerically trivial: these tests exercise only the
/// KV growth/refusal path, not output correctness. The backing temp file is
/// returned alongside the session because the q8 file-backed attention weight
/// reads it lazily during the forward pass, so it must outlive the session.
fn tiny_kv_budget_session(context_length: u32) -> (LlamaInferenceSession, tempfile::NamedTempFile) {
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for _ in 0..32 {
        temp_file
            .write_all(&f32_to_f16_bits(1.0).to_le_bytes())
            .unwrap();
        temp_file.write_all(&[0_u8; Q8_0_BLOCK_VALUES]).unwrap();
    }
    temp_file.flush().unwrap();

    let config = LlamaModelConfig {
        context_length,
        embedding_length: 32,
        block_count: 1,
        feed_forward_length: 32,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(32),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(2),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let dense_vector = |name: &str| CpuTensor::from_f32(name, vec![32], vec![1.0; 32]).unwrap();
    let dense_matrix =
        |name: &str| CpuTensor::from_f32(name, vec![32, 32], vec![0.0; 32 * 32]).unwrap();
    let attention_q = CpuTensor::q8_0_file_backed_linear(
        "blk.0.attn_q.weight",
        TensorShape { dims: vec![32, 32] },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 32),
    );
    let weights = LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd.weight",
            vec![2, 32],
            (0..64).map(|idx| idx as f32 * 0.001).collect(),
        )
        .unwrap(),
        output_norm: dense_vector("output_norm.weight"),
        output: None,
        rope_freqs: None,
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
        layers: vec![LlamaLayerWeights {
            attention_norm: dense_vector("blk.0.attn_norm.weight"),
            attention_q,
            attention_k: dense_matrix("blk.0.attn_k.weight"),
            attention_v: dense_matrix("blk.0.attn_v.weight"),
            attention_output: dense_matrix("blk.0.attn_output.weight"),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            ffn_norm: dense_vector("blk.0.ffn_norm.weight"),
            ffn_gate: dense_matrix("blk.0.ffn_gate.weight"),
            ffn_up: dense_matrix("blk.0.ffn_up.weight"),
            ffn_down: dense_matrix("blk.0.ffn_down.weight"),
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
    };
    let session = LlamaInferenceSession::new(config, weights).unwrap();
    (session, temp_file)
}

/// A NoPE model must never reach the resident GPU engines. They build ONE cos/sin
/// table for the whole forward and rope every layer unconditionally
/// (`cuda_resident` launch_rope / launch_rope_batched and the Metal equivalents),
/// so they cannot express the per-layer skip — a smollm3 file on a CUDA build
/// would silently produce different, wrong tokens from the fixed CPU path.
///
/// The gate is checked before the backend-enabled test, so this holds on a
/// CPU-only host too. Note that on a host where the resident path is not eligible
/// for some *other* reason this assertion is satisfied but not causal; the
/// ordering of the check in `resident_decode_eligible` is what makes it load-bearing.
#[test]
fn nope_config_disqualifies_the_resident_gpu_path() {
    let (mut session, _temp) = tiny_kv_budget_session(64);

    session.config.no_rope_layer_step = Some(4);
    assert!(
        !session
            .resident_decode_eligible(false)
            .expect("eligibility check should not error"),
        "a NoPE config must be refused by the resident GPU gate"
    );
    assert!(
        !session
            .resident_decode_eligible(true)
            .expect("eligibility check should not error"),
        "a NoPE config must be refused for the want-logits path too"
    );
}

/// End-to-end proof that the KV predict-and-abort budget guard fires through the
/// real forward path, not only at the `ensure_position_capacity` unit seam: a
/// prefill whose positions would exceed the session's KV byte budget returns a
/// typed error and never grows the cache past the budget, so the host ceiling is
/// refused up front instead of being discovered by OOMing mid-generation. This is
/// the session-level counterpart to the `kv_cache.rs` cache-unit tests and closes
/// the behavioral gap flagged in
/// `qa/validation-notes/2026-06-25-capability-context-host-limits.md`.
#[test]
fn prefill_fails_closed_when_kv_budget_exceeded() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    let _ = q8_0_file_read_stats();

    // context_length is generous (16) so the *budget*, not the context-length cap,
    // is the reason the prefill is refused.
    let (mut session, _temp_file) = tiny_kv_budget_session(16);

    // Budget for at most two f32 tokens (2 * 32 * 2 * 4 = 512 bytes). Overriding
    // the resolved budget on the constructed cache keeps the test independent of
    // the host's RAM and of any CAMELID_MAX_KV_CACHE_BYTES in the environment.
    let kv_budget_bytes = 512;
    session.kv_cache.kv_budget_bytes = kv_budget_bytes;

    // A five-token prompt needs more KV than the budget allows, so the prefill must
    // refuse instead of allocating it.
    let err = session
        .forward_prefill_layer_major_timed_fast(&[0, 1, 0, 1, 0], 1)
        .unwrap_err();
    assert!(
        matches!(err, crate::BackendError::KvCacheBudgetExceeded { .. }),
        "prefill over budget must return the typed KV budget error, got: {err:?}"
    );
    // The message still names the override env so the operator sees the knob.
    assert!(
        err.to_string().contains("CAMELID_MAX_KV_CACHE_BYTES"),
        "the forward error must name the override env: {err}"
    );
    // Fail-closed invariants that hold whether the prefill sizes the cache in one
    // batch or grows it token-by-token: the full over-budget length is never
    // allocated, and whatever was allocated stays within the byte budget.
    assert!(
        session.kv_cache.allocated_sequence_length < 5,
        "the over-budget prompt length must never be fully allocated"
    );
    assert!(
        session.kv_cache.allocated_bytes() <= kv_budget_bytes,
        "KV allocation must stay within the byte budget after a refused prefill"
    );
    assert!(
        session.kv_cache.position < 5,
        "the decode position must not advance to the refused length"
    );
}

/// The decode counterpart to `prefill_fails_closed_when_kv_budget_exceeded`: the
/// guard's other forward-path call site is the single-token growth in
/// `write_kv_cache` (`position + 1`), which is the exact "OOM mid-generation" case
/// the host-limit note calls out. Decode two tokens under an unbounded budget, then
/// tighten the budget to precisely what is already allocated and decode once more:
/// the growth to the next position is refused with a typed error, the position does
/// not advance, and the allocation never exceeds the budget. Driving the assertions
/// off the live `allocated_bytes()` keeps this independent of the active KV dtype
/// and layout.
#[test]
fn decode_fails_closed_when_kv_budget_exceeded_mid_generation() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    let _ = q8_0_file_read_stats();

    // context_length 16 (< 512 => grow chunk 1, so allocated length tracks the
    // position exactly).
    let (mut session, _temp_file) = tiny_kv_budget_session(16);

    // Two decode steps with no effective budget cap.
    session.kv_cache.kv_budget_bytes = u64::MAX;
    session.forward_single_token(0).unwrap();
    session.forward_single_token(1).unwrap();
    assert_eq!(session.kv_cache.position, 2);
    let allocated_bytes = session.kv_cache.allocated_bytes();
    assert!(allocated_bytes > 0, "two decodes must have allocated KV");

    // Tighten the budget to exactly what is already resident: the next token cannot
    // grow the cache, so the decode must fail closed mid-generation.
    session.kv_cache.kv_budget_bytes = allocated_bytes;
    let err = session.forward_single_token(0).unwrap_err();
    assert!(
        matches!(err, crate::BackendError::KvCacheBudgetExceeded { .. }),
        "the mid-generation refusal must be the typed KV budget error, got: {err:?}"
    );
    assert!(
        err.to_string().contains("CAMELID_MAX_KV_CACHE_BYTES"),
        "the mid-generation error must name the override env: {err}"
    );
    assert_eq!(
        session.kv_cache.position, 2,
        "the decode position must not advance past the refused growth"
    );
    assert_eq!(
        session.kv_cache.allocated_bytes(),
        allocated_bytes,
        "no KV may be allocated beyond the tightened budget"
    );
}

#[test]
fn materialization_stats_quantify_lazy_q8_file_backing_tradeoff() {
    let lazy_q8_attention_q = CpuTensor::q8_0_file_backed_linear(
        "blk.0.attn_q.weight",
        crate::tensor::TensorShape { dims: vec![2, 64] },
        Q8_0FileBacking::new("unused.gguf".into(), 0, 4),
    );
    let weights = tiny_prefill_schedule_weights(lazy_q8_attention_q);

    let stats = collect_weight_materialization_stats(&weights);

    assert_eq!(stats.q8_0_file_backed_tensor_count, 1);
    assert_eq!(stats.q8_0_file_backed_storage_bytes, 4 * 34);
    assert_eq!(stats.q8_0_file_backed_f32_bytes_avoided, 4 * 32 * 4);
    assert_eq!(
        stats.q8_0_file_backed_retained_block_bytes_if_enabled,
        4 * std::mem::size_of::<Q8_0Block>() as u64
    );
    assert_eq!(stats.q8_0_retained_block_bytes, 0);
    assert!(stats.has_lazy_q8_0_file_backing);
    assert!(!stats.has_retained_q8_0_blocks);
    assert!(!stats.has_q8_0_f32_materialization);
}

#[test]
fn memory_timing_merge_tracks_forward_passes_and_peak_rss() {
    let mut first = LlamaForwardTimings {
        memory: Some(test_forward_memory(memory_sample(100, 0, 0))),
        ..LlamaForwardTimings::default()
    };
    first
        .memory
        .as_mut()
        .unwrap()
        .record_after_logits(memory_sample(110, 0, 1));
    first
        .memory
        .as_mut()
        .unwrap()
        .q8_file_read_phases
        .push(LlamaQ8FileReadPhaseTrace {
            phase: "logits_done".to_string(),
            q8_file_reads: Q8_0FileReadStats {
                read_calls: 3,
                read_bytes: 256,
                cache_hits: 1,
                cache_hit_bytes: 64,
                cache_entries: 2,
                cache_bytes: 512,
                cache_capacity_bytes: 1024,
                ..Q8_0FileReadStats::default()
            },
        });

    let mut second = LlamaForwardTimings {
        memory: Some(test_forward_memory(memory_sample(105, 1, 1))),
        ..LlamaForwardTimings::default()
    };
    second
        .memory
        .as_mut()
        .unwrap()
        .record_after_layers(memory_sample(140, 1, 2));
    second
        .memory
        .as_mut()
        .unwrap()
        .q8_file_read_phases
        .push(LlamaQ8FileReadPhaseTrace {
            phase: "layers_done".to_string(),
            q8_file_reads: Q8_0FileReadStats {
                read_calls: 4,
                read_bytes: 1024,
                cache_hits: 2,
                cache_hit_bytes: 128,
                cache_entries: 3,
                cache_bytes: 768,
                cache_capacity_bytes: 1024,
                ..Q8_0FileReadStats::default()
            },
        });
    first.memory.as_mut().unwrap().q8_file_reads = Q8_0FileReadStats {
        read_calls: 3,
        read_bytes: 256,
        cache_hits: 1,
        cache_hit_bytes: 64,
        cache_entries: 2,
        cache_bytes: 512,
        cache_capacity_bytes: 1024,
        ..Q8_0FileReadStats::default()
    };
    second.memory.as_mut().unwrap().q8_file_reads = Q8_0FileReadStats {
        read_calls: 4,
        read_bytes: 1024,
        cache_hits: 2,
        cache_hit_bytes: 128,
        cache_entries: 3,
        cache_bytes: 768,
        cache_capacity_bytes: 1024,
        ..Q8_0FileReadStats::default()
    };

    first.add_assign(&second);

    let memory = first.memory.expect("merged memory timings");
    assert_eq!(memory.forward_passes, 2);
    assert_eq!(
        memory.q8_file_reads,
        Q8_0FileReadStats {
            read_calls: 7,
            read_bytes: 1280,
            cache_hits: 3,
            cache_hit_bytes: 192,
            cache_entries: 3,
            cache_bytes: 768,
            cache_capacity_bytes: 1024,
            ..Q8_0FileReadStats::default()
        }
    );
    assert_eq!(memory.peak_rss_kib, Some(140));
    assert_eq!(memory.peak_rss_delta_kib, Some(40));
    assert_eq!(memory.peak_phase.as_deref(), Some("layers_done"));
    assert_eq!(memory.q8_file_read_phases.len(), 2);
    assert_eq!(memory.q8_file_read_phases[0].phase, "logits_done");
    assert_eq!(
        memory.q8_file_read_phases[0].q8_file_reads.cache_hit_bytes,
        64
    );
    assert_eq!(memory.q8_file_read_phases[1].phase, "layers_done");
    assert_eq!(memory.q8_file_read_phases[1].q8_file_reads.read_bytes, 1024);
    assert_eq!(memory.end, None);
    assert_eq!(
        memory
            .after_layers
            .unwrap()
            .kv_cache_allocated_sequence_length,
        2
    );
}

#[test]
fn q8_file_read_stats_merge_keeps_peak_cache_state() {
    let mut target = Q8_0FileReadStats {
        read_calls: 2,
        read_bytes: 128,
        cache_entries: 4,
        cache_bytes: 1024,
        cache_capacity_bytes: 2048,
        ..Q8_0FileReadStats::default()
    };
    let delta = Q8_0FileReadStats {
        read_calls: 3,
        read_bytes: 256,
        cache_hits: 1,
        cache_hit_bytes: 64,
        cache_entries: 0,
        cache_bytes: 0,
        cache_capacity_bytes: 0,
        ..Q8_0FileReadStats::default()
    };

    add_q8_file_read_stats_delta(&mut target, delta);

    assert_eq!(target.read_calls, 5);
    assert_eq!(target.read_bytes, 384);
    assert_eq!(target.cache_hits, 1);
    assert_eq!(target.cache_hit_bytes, 64);
    assert_eq!(target.cache_entries, 4);
    assert_eq!(target.cache_bytes, 1024);
    assert_eq!(target.cache_capacity_bytes, 2048);
}

#[test]
fn layer_memory_record_end_captures_tail_q8_file_read_phase() {
    let _q8_guard = crate::test_support::q8_file_state_lock();
    let mut memory = LlamaLayerMemoryTimings::new(7, memory_sample(100, 0, 0));

    record_q8_0_file_read(32);
    memory.record_after_attention_q(memory_sample(110, 0, 0));
    record_q8_0_file_read(64);
    memory.record_end();

    assert_eq!(memory.q8_file_reads.read_calls, 2);
    assert_eq!(memory.q8_file_reads.read_bytes, 96);
    assert_eq!(memory.q8_file_read_phases.len(), 2);
    assert_eq!(memory.q8_file_read_phases[0].phase, "attention_q_done");
    assert_eq!(memory.q8_file_read_phases[0].q8_file_reads.read_calls, 1);
    assert_eq!(memory.q8_file_read_phases[0].q8_file_reads.read_bytes, 32);
    assert_eq!(memory.q8_file_read_phases[1].phase, "layer_end");
    assert_eq!(memory.q8_file_read_phases[1].q8_file_reads.read_calls, 1);
    assert_eq!(memory.q8_file_read_phases[1].q8_file_reads.read_bytes, 64);
}

#[test]
fn layer_memory_merge_accumulates_q8_file_reads() {
    let mut first = LlamaLayerMemoryTimings::new(3, memory_sample(100, 0, 0));
    first.q8_file_reads = Q8_0FileReadStats {
        read_calls: 2,
        read_bytes: 128,
        cache_hits: 1,
        cache_hit_bytes: 32,
        cache_entries: 1,
        cache_bytes: 256,
        cache_capacity_bytes: 512,
        ..Q8_0FileReadStats::default()
    };
    let mut second = LlamaLayerMemoryTimings::new(3, memory_sample(105, 1, 1));
    second.q8_file_reads = Q8_0FileReadStats {
        read_calls: 5,
        read_bytes: 512,
        cache_hits: 3,
        cache_hit_bytes: 96,
        cache_entries: 2,
        cache_bytes: 384,
        cache_capacity_bytes: 512,
        ..Q8_0FileReadStats::default()
    };
    first.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
        phase: "attention_q_done".to_string(),
        q8_file_reads: Q8_0FileReadStats {
            read_calls: 2,
            read_bytes: 128,
            cache_hits: 1,
            cache_hit_bytes: 32,
            cache_entries: 1,
            cache_bytes: 256,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        },
    });
    second.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
        phase: "attention_q_done".to_string(),
        q8_file_reads: Q8_0FileReadStats {
            read_calls: 3,
            read_bytes: 384,
            cache_hits: 2,
            cache_hit_bytes: 64,
            cache_entries: 2,
            cache_bytes: 384,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        },
    });
    second.q8_file_read_phases.push(LlamaQ8FileReadPhaseTrace {
        phase: "ffn_down_done".to_string(),
        q8_file_reads: Q8_0FileReadStats {
            read_calls: 2,
            read_bytes: 128,
            cache_hits: 1,
            cache_hit_bytes: 32,
            cache_entries: 2,
            cache_bytes: 384,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        },
    });

    first.merge_assign(&second);

    assert_eq!(first.forward_passes, 2);
    assert_eq!(
        first.q8_file_reads,
        Q8_0FileReadStats {
            read_calls: 7,
            read_bytes: 640,
            cache_hits: 4,
            cache_hit_bytes: 128,
            cache_entries: 2,
            cache_bytes: 384,
            cache_capacity_bytes: 512,
            ..Q8_0FileReadStats::default()
        }
    );
    assert_eq!(first.q8_file_read_phases.len(), 2);
    assert_eq!(first.q8_file_read_phases[0].phase, "attention_q_done");
    assert_eq!(first.q8_file_read_phases[0].q8_file_reads.read_calls, 5);
    assert_eq!(first.q8_file_read_phases[0].q8_file_reads.read_bytes, 512);
    assert_eq!(first.q8_file_read_phases[0].q8_file_reads.cache_hits, 3);
    assert_eq!(
        first.q8_file_read_phases[0].q8_file_reads.cache_hit_bytes,
        96
    );
    assert_eq!(first.q8_file_read_phases[1].phase, "ffn_down_done");
    assert_eq!(first.q8_file_read_phases[1].q8_file_reads.read_calls, 2);
    assert_eq!(
        first.q8_file_read_phases[1].q8_file_reads.cache_hit_bytes,
        32
    );

    first.record_after_attention_output(memory_sample(160, 1, 1));
    assert_eq!(first.peak_rss_kib, Some(160));
    assert_eq!(first.peak_rss_delta_kib, Some(60));
    assert_eq!(first.peak_phase.as_deref(), Some("attention_output_done"));
}

#[cfg(target_os = "linux")]
#[test]
fn parses_linux_proc_status_rss_kib() {
    assert_eq!(
        parse_proc_status_rss_kib("Name:\tcamelid\nVmRSS:\t  12345 kB\n"),
        Some(12_345)
    );
}

fn clear_dense_diagnostic_env() {
    for key in [
        "CAMELID_ATTENTION_SCORE_SCALE",
        "CAMELID_FFN_GATE_UP_ORDER",
        "CAMELID_FORWARD_MEMORY_TRACE",
        "CAMELID_FORWARD_RSS_TIMINGS",
        "CAMELID_GQA_HEAD_MAPPING",
        "CAMELID_LINEAR_ACCUMULATION",
        "CAMELID_METAL_Q8",
        "CAMELID_METAL_Q8_RETAINED",
        "CAMELID_HYBRID_Q8_GPU_PERCENT",
        "CAMELID_HYBRID_Q8_GPU_ROWS",
        "CAMELID_HYBRID_Q8_RETAINED",
        "CAMELID_OUTPUT_PROJECTION_LAYOUT",
        "CAMELID_PREFILL_LAYER_MAJOR",
        "CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES",
        "CAMELID_PARALLEL_LINEAR",
        "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS",
        "CAMELID_Q8_0_BLOCK_DOT",
        "CAMELID_MAC_Q8_REPACK",
        "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
        "CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING",
        "CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK",
        "CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS",
        "CAMELID_Q8_0_PACKED_4X4_DOT",
        "CAMELID_Q8_0_PACKED_4X8_DOT",
        "CAMELID_X86_KQUANT_MATMUL_OWNER",
        "CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI",
        "CAMELID_X86_Q6K_AVX2",
        "CAMELID_Q8_0_FILE_READER_BLOCK_DOT",
        "CAMELID_Q8_0_FILE_CACHE_BYTES",
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        "CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES",
        "CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES",
        "CAMELID_PARALLEL_LINEAR",
        "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_V",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_DOWN",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_GATE",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_UP",
        "CAMELID_RMS_NORM_EPSILON",
        "CAMELID_ROPE_DIRECTION",
        "CAMELID_ROPE_PAIRING",
        "CAMELID_ROPE_POSITION_MODE",
        "CAMELID_RUNTIME_PROFILE",
        "CAMELID_SQUARE_LINEAR_LAYOUT",
        "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
        "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
        "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
        "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
        "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
        "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
        "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
        "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
    ] {
        std::env::remove_var(key);
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[test]
fn final_norm_diagnostics_reconstruct_output_norm_values() {
    let hidden = CpuTensor::from_f32("hidden", vec![1, 4], vec![3.0, 4.0, 0.0, -5.0]).unwrap();
    let weight =
        CpuTensor::from_f32("output_norm.weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output_norm = hidden.rms_norm(&weight, 1e-5, "output_norm").unwrap();

    let diagnostic = final_norm_diagnostics(&hidden, &weight, &output_norm, 1e-5).unwrap();

    assert_close(diagnostic.hidden_mean_square, 12.5);
    assert_close(diagnostic.hidden_rms, 12.5_f32.sqrt());
    assert_eq!(diagnostic.hidden_first_values, vec![3.0, 4.0, 0.0, -5.0]);
    assert_eq!(diagnostic.weight_first_values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(diagnostic.reconstructed_first_values.len(), 4);
    assert_eq!(diagnostic.reported_first_values, output_norm.data);
    assert_eq!(diagnostic.reported_max_abs_index, 3);
    assert_close(diagnostic.reported_max_abs, output_norm.data[3].abs());
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, output_norm.data);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        output_norm.data
    );
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn rms_norm_diagnostics_report_peak_window() {
    let input = CpuTensor::from_f32("input", vec![1, 4], vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    let weight = CpuTensor::from_f32("norm.weight", vec![4], vec![0.5, 1.0, 1.5, 2.0]).unwrap();
    let reported = input.rms_norm(&weight, 1e-5, "reported").unwrap();

    let diagnostic = rms_norm_diagnostics(&input, &weight, &reported, 1e-5).unwrap();

    assert_eq!(diagnostic.reported_max_abs_index, 3);
    assert_close(diagnostic.reported_max_abs, reported.data[3].abs());
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, reported.data);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        reported.data
    );
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn residual_diagnostics_report_delta_scale_and_alignment() {
    let input = CpuTensor::from_f32("input", vec![1, 4], vec![3.0, 4.0, 0.0, -5.0]).unwrap();
    let delta = CpuTensor::from_f32("delta", vec![1, 4], vec![1.0, -2.0, 0.0, 2.0]).unwrap();
    let reported = input.add(&delta, "reported").unwrap();

    let diagnostic = residual_reconstruction_diagnostic(&input, &delta, &reported).unwrap();

    assert_close(diagnostic.input_rms, 12.5_f32.sqrt());
    assert_close(diagnostic.delta_rms, 2.25_f32.sqrt());
    assert_close(diagnostic.reported_rms, 7.25_f32.sqrt());
    assert_close(
        diagnostic.delta_to_input_rms_ratio,
        2.25_f32.sqrt() / 12.5_f32.sqrt(),
    );
    assert_close(
        diagnostic.delta_input_cosine_similarity,
        -15.0 / (50.0_f32.sqrt() * 9.0_f32.sqrt()),
    );
    assert_eq!(diagnostic.reconstructed_first_values, reported.data);
    assert_eq!(diagnostic.reported_max_abs_index, 0);
    assert_close(diagnostic.reported_max_abs, 4.0);
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, reported.data);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        reported.data
    );
    assert_eq!(diagnostic.delta_reported_max_abs_window, delta.data);
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn linear_projection_diagnostics_reconstruct_descriptor_layout() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K");
    std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
    let weight =
        CpuTensor::from_f32("weight", vec![3, 2], vec![1.0, 2.0, -3.0, 4.0, 0.5, -2.0]).unwrap();
    let reported = linear_with_diagnostic_layouts(
        &input,
        &weight,
        "reported",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Descriptor,
    )
    .unwrap();

    let diagnostic =
        linear_projection_diagnostics(&input, &weight, &reported, "attention_k").unwrap();

    assert_eq!(diagnostic.layout, "descriptor");
    assert_eq!(diagnostic.input_width, 3);
    assert_eq!(diagnostic.output_width, 2);
    assert_eq!(diagnostic.weight_shape, vec![3, 2]);
    assert_eq!(diagnostic.reconstructed_first_values, reported.data);
    assert_eq!(diagnostic.reported_max_abs_index, 0);
    assert_close(diagnostic.reported_max_abs, 5.25);
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, reported.data);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        reported.data
    );
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn linear_projection_diagnostics_reconstruct_transposed_layout() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_V");
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
    let weight =
        CpuTensor::from_f32("weight", vec![2, 3], vec![1.0, -3.0, 0.5, 2.0, 4.0, -2.0]).unwrap();
    let reported = linear_with_diagnostic_layouts(
        &input,
        &weight,
        "reported",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Auto,
    )
    .unwrap();

    let diagnostic =
        linear_projection_diagnostics(&input, &weight, &reported, "attention_v").unwrap();

    assert_eq!(diagnostic.layout, "transposed_auto");
    assert_eq!(diagnostic.input_width, 3);
    assert_eq!(diagnostic.output_width, 2);
    assert_eq!(diagnostic.weight_shape, vec![2, 3]);
    assert_eq!(diagnostic.reconstructed_first_values, reported.data);
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn linear_projection_diagnostics_report_nonzero_reconstruction_delta() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_Q");
    std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
    let weight =
        CpuTensor::from_f32("weight", vec![3, 2], vec![1.0, 2.0, -3.0, 4.0, 0.5, -2.0]).unwrap();
    let reported = CpuTensor::from_f32("reported", vec![1, 2], vec![5.25, -2.75]).unwrap();

    let diagnostic =
        linear_projection_diagnostics(&input, &weight, &reported, "attention_q").unwrap();

    assert_eq!(diagnostic.layout, "descriptor");
    assert_eq!(diagnostic.reconstructed_first_values, vec![5.25, -1.0]);
    assert_eq!(diagnostic.reported_first_values, vec![5.25, -2.75]);
    assert_eq!(diagnostic.reported_max_abs_index, 0);
    assert_close(diagnostic.reported_max_abs, 5.25);
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, vec![5.25, -2.75]);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        vec![5.25, -1.0]
    );
    assert_eq!(diagnostic.max_abs_delta_index, 1);
    assert_close(diagnostic.max_abs_delta, 1.75);
}

#[test]
fn parallel_linear_matches_serial_descriptor_transposed_and_q8_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");

    let input = CpuTensor::from_f32("input", vec![1, 4], vec![2.0, -1.0, 0.5, 3.0]).unwrap();
    let descriptor_weight = CpuTensor::from_f32(
        "descriptor.weight",
        vec![4, 3],
        vec![
            1.0, -2.0, 0.25, -3.0, 4.0, 0.5, 0.5, -1.0, 2.0, 2.0, 0.25, -0.75,
        ],
    )
    .unwrap();
    let transposed_weight = CpuTensor::from_f32(
        "transposed.weight",
        vec![3, 4],
        vec![
            1.0, -3.0, 0.5, 2.0, -2.0, 4.0, -1.0, 0.25, 0.25, 0.5, 2.0, -0.75,
        ],
    )
    .unwrap();

    let serial_descriptor = linear_with_diagnostic_layouts(
        &input,
        &descriptor_weight,
        "serial_descriptor",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Descriptor,
    )
    .unwrap();
    let serial_transposed = linear_with_diagnostic_layouts(
        &input,
        &transposed_weight,
        "serial_transposed",
        SquareLinearLayout::Transposed,
        RectangularLinearLayout::Transposed,
    )
    .unwrap();

    std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
    let parallel_descriptor = linear_with_diagnostic_layouts(
        &input,
        &descriptor_weight,
        "parallel_descriptor",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Descriptor,
    )
    .unwrap();
    let parallel_transposed = linear_with_diagnostic_layouts(
        &input,
        &transposed_weight,
        "parallel_transposed",
        SquareLinearLayout::Transposed,
        RectangularLinearLayout::Transposed,
    )
    .unwrap();

    assert_eq!(parallel_descriptor.data, serial_descriptor.data);
    assert_eq!(parallel_transposed.data, serial_transposed.data);

    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");
    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let q8_input = CpuTensor::from_f32("q8_input", vec![1, 32], input_values).unwrap();
    let row0 = Q8_0Block {
        scale: 0.5,
        quants: std::array::from_fn(|idx| idx as i8 - 16),
    };
    let row1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
    };
    let mut dequantized_weight = Vec::with_capacity(64);
    for block in [&row0, &row1] {
        dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
    }
    let q8_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "q8_weight",
        vec![2, 32],
        dequantized_weight,
        vec![row0, row1],
    )
    .unwrap();
    let serial_q8 =
        matmul_rhs_transposed_with_precision(&q8_input, &q8_weight, "serial_q8").unwrap();
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
    let parallel_q8 =
        matmul_rhs_transposed_with_precision(&q8_input, &q8_weight, "parallel_q8").unwrap();

    assert_eq!(parallel_q8.data, serial_q8.data);
}

#[test]
fn q8_0_hot_path_uses_resolved_plan_not_current_env() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");

    let input = CpuTensor::from_f32("input", vec![1, 32], vec![0.25; 32]).unwrap();
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![1, 32],
        vec![1.25; 32],
        vec![Q8_0Block {
            scale: 1.0,
            quants: [1; 32],
        }],
    )
    .unwrap();
    let plan = ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags {
            block_dot: true,
            file_reader_block_dot: false,
            attention_projection_decode_consumer: false,
            attention_output_decode_consumer: false,
            attention_output_packed_rows4_matmul: false,
            attention_qkv_decode_consumer: false,
            attention_qkv_decode_group_chunking: false,
            attention_qkv_packed_rows4_matmul: false,
            output_packed_rows4_matmul: false,
            output_amx_prefill: false,
            output_decode_owner: false,
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_decode_group_chunking: false,
            ffn_gate_up_decode_fused_activation: false,
            ffn_gate_up_decode_paired_dot: false,
            ffn_decode_chain: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_decode_group_chunking: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            ffn_down_vnni_decode_rawptr: false,
            q8_matmul_owner: Q8MatmulOwnerScope::Off,
            q8_matmul_owner_avx2: false,
            q8_matmul_owner_vnni: false,
            q8_matmul_owner_4x8: false,
            metal: false,
            cuda: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
        q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::default(),
    };

    let actual =
        matmul_rhs_transposed_with_precision_with_plan(&input, &weight, "resolved_plan_out", &plan)
            .unwrap();

    assert!(
        (actual.data[0] - 8.0).abs() < 1.0e-3,
        "got {}",
        actual.data[0]
    );
}

#[test]
fn q8_0_block_dot_uses_quantized_fast_path_when_explicitly_enabled() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let input_values = vec![0.25; 32];
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values).unwrap();
    let block = Q8_0Block {
        scale: 1.0,
        quants: [1; 32],
    };
    let weight =
        CpuTensor::from_f32_with_q8_0_blocks("weight", vec![1, 32], vec![1.0; 32], vec![block])
            .unwrap();

    assert!(should_use_q8_0_block_dot(&weight, 32));
    let actual = matmul_rhs_transposed_with_precision(&input, &weight, "out").unwrap();

    assert_eq!(actual.shape.dims, vec![1, 1]);
    assert!(
        (actual.data[0] - 8.0).abs() < 1.0e-3,
        "expected quantized fast path to stay close to dequantized output, got {}",
        actual.data[0]
    );
}

#[test]
fn q8_0_compute_gates_preserve_default_on_and_explicit_escape_hatches() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    assert!(q8_0_block_dot_enabled());
    assert!(q8_0_file_reader_block_dot_enabled());

    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
    assert!(!q8_0_block_dot_enabled());
    assert!(q8_0_file_reader_block_dot_enabled());

    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "0");
    assert!(!q8_0_block_dot_enabled());
    assert!(!q8_0_file_reader_block_dot_enabled());
}

#[test]
fn experimental_q8_acceleration_gates_default_off_and_require_explicit_opt_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    assert!(!q8_0_metal_enabled());
    assert!(!q8_0_metal_retained_enabled());
    assert!(!q8_0_hybrid_retained_enabled());

    std::env::set_var("CAMELID_METAL_Q8", "true");
    std::env::set_var("CAMELID_METAL_Q8_RETAINED", "enabled");
    std::env::set_var("CAMELID_HYBRID_Q8_RETAINED", "yes");

    assert!(q8_0_metal_enabled());
    assert!(q8_0_metal_retained_enabled());
    assert!(q8_0_hybrid_retained_enabled());
}

#[test]
fn resolved_runtime_plan_captures_q8_env_once() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "1");
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER", "on");
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER", "on");
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER", "yes");
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL", "on");
    std::env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER", "true");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
    std::env::set_var("CAMELID_HYBRID_Q8_GPU_ROWS", "7");
    std::env::set_var("CAMELID_HYBRID_Q8_GPU_PERCENT", "25");

    let plan = ResolvedRuntimePlan::from_env().unwrap();

    assert_eq!(
        plan.linear_accumulation_precision,
        LinearAccumulationPrecision::F32
    );
    assert!(plan.q8.block_dot);
    assert!(plan.q8.file_reader_block_dot);
    assert!(plan.q8.attention_projection_decode_consumer);
    assert!(plan.q8.attention_output_decode_consumer);
    assert!(plan.q8.attention_output_packed_rows4_matmul);
    assert!(plan.q8.attention_qkv_decode_consumer);
    assert!(plan.q8.attention_qkv_packed_rows4_matmul);
    assert!(plan.q8.output_packed_rows4_matmul);
    assert!(plan.q8.output_amx_prefill);
    assert!(plan.q8.output_decode_owner);
    assert!(plan.q8.ffn_gate_up_decode_consumer);
    assert!(plan.q8.ffn_gate_up_decode_group_chunking);
    assert!(plan.q8.ffn_gate_up_decode_fused_activation);
    assert!(plan.q8.ffn_gate_up_decode_paired_dot);
    assert!(plan.q8.ffn_decode_chain);
    assert!(plan.q8.ffn_gate_up_packed_rows4_matmul);
    assert!(plan.q8.ffn_gate_up_single_owner);
    assert!(plan.q8.ffn_down_decode_consumer);
    assert!(plan.q8.ffn_down_decode_group_chunking);
    assert!(plan.q8.ffn_down_packed_rows4_matmul);
    assert!(plan.q8.ffn_down_vnni_decode);
    assert!(plan.q8.ffn_down_vnni_decode_rawptr);
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL");
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
    assert!(
        plan.q8.attention_projection_decode_consumer,
        "resolved plan should cache the attention projection consumer gate"
    );
    assert!(
        plan.q8.attention_output_decode_consumer,
        "resolved plan should cache the attention output consumer gate"
    );
    assert!(
        plan.q8.attention_output_packed_rows4_matmul,
        "resolved plan should cache the attention output packed-rows4 matmul gate"
    );
    assert!(
        plan.q8.attention_qkv_decode_consumer,
        "resolved plan should cache the attention QKV consumer gate"
    );
    assert!(
        plan.q8.attention_qkv_packed_rows4_matmul,
        "resolved plan should cache the attention QKV packed-rows4 matmul gate"
    );
    assert!(
        plan.q8.output_packed_rows4_matmul,
        "resolved plan should cache the output packed-rows4 matmul gate"
    );
    assert!(
        plan.q8.output_amx_prefill,
        "resolved plan should cache the output AMX prefill gate"
    );
    assert!(
        plan.q8.output_decode_owner,
        "resolved plan should cache the output decode-owner gate"
    );
    assert!(
        plan.q8.ffn_gate_up_decode_consumer,
        "resolved plan should cache the FFN gate/up consumer gate"
    );
    assert!(
        plan.q8.ffn_gate_up_decode_group_chunking,
        "resolved plan should cache the FFN gate/up decode group-chunking gate"
    );
    assert!(
        plan.q8.ffn_gate_up_decode_fused_activation,
        "resolved plan should cache the FFN gate/up fused activation gate"
    );
    assert!(
        plan.q8.ffn_gate_up_decode_paired_dot,
        "resolved plan should cache the FFN gate/up paired dot gate"
    );
    assert!(
        plan.q8.ffn_decode_chain,
        "resolved plan should cache the FFN decode-chain gate"
    );
    assert!(
        plan.q8.ffn_gate_up_packed_rows4_matmul,
        "resolved plan should cache the FFN gate/up packed-rows4 matmul gate"
    );
    assert!(
        plan.q8.ffn_gate_up_single_owner,
        "resolved plan should cache the FFN gate/up single-owner gate"
    );
    assert!(
        plan.q8.ffn_down_decode_consumer,
        "resolved plan should cache the FFN-down consumer gate"
    );
    assert!(
        plan.q8.ffn_down_decode_group_chunking,
        "resolved plan should cache the FFN-down decode group-chunking gate"
    );
    assert!(
        plan.q8.ffn_down_packed_rows4_matmul,
        "resolved plan should cache the packed-rows4 matmul gate"
    );
    assert!(
        plan.q8.ffn_down_vnni_decode,
        "resolved plan should cache the FFN-down VNNI decode gate"
    );
    assert!(
        plan.q8.ffn_down_vnni_decode_rawptr,
        "resolved plan should cache the FFN-down VNNI rawptr decode gate"
    );
    assert_eq!(plan.q8.hybrid_gpu_rows, Some(7));
    assert_eq!(plan.q8.hybrid_gpu_percent, 25);
    assert_eq!(plan.q8.hybrid_gpu_rows_for_output(100), 7);
}

#[test]
fn x86_q8_ffn_down_packed_rows4_matmul_accepts_role_gate_and_legacy_alias() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    assert!(!Q8RuntimeFlags::from_env().ffn_down_packed_rows4_matmul);

    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_down_packed_rows4_matmul);

    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL");
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_down_packed_rows4_matmul);

    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL");
}

#[test]
fn runtime_profile_defaults_keep_experimental_q8_gates_closed() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    for profile in ["safe", "auto", "experimental", "debug"] {
        std::env::set_var("CAMELID_RUNTIME_PROFILE", profile);
        let plan = ResolvedRuntimePlan::from_env().unwrap();
        assert!(
            plan.q8.block_dot,
            "{profile} should preserve Q8 block-dot default-on behavior"
        );
        assert!(
            plan.q8.file_reader_block_dot,
            "{profile} should preserve Q8 file-reader block-dot default-on behavior"
        );
        assert!(
            !plan.q8.attention_projection_decode_consumer,
            "{profile} should not enable attention projection consumer by default"
        );
        assert!(
            !plan.q8.attention_output_decode_consumer,
            "{profile} should not enable attention output consumer by default"
        );
        assert!(
            !plan.q8.attention_output_packed_rows4_matmul,
            "{profile} should not enable attention output packed-rows4 matmul by default"
        );
        assert!(
            !plan.q8.attention_qkv_decode_consumer,
            "{profile} should not enable attention QKV consumer by default"
        );
        assert!(
            !plan.q8.attention_qkv_packed_rows4_matmul,
            "{profile} should not enable attention QKV packed-rows4 matmul by default"
        );
        assert!(
            !plan.q8.output_packed_rows4_matmul,
            "{profile} should not enable output packed-rows4 matmul by default"
        );
        assert!(
            !plan.q8.output_amx_prefill,
            "{profile} should not enable output AMX prefill by default"
        );
        assert!(
            !plan.q8.output_decode_owner,
            "{profile} should not enable output decode owner by default"
        );
        assert!(
            !plan.q8.ffn_gate_up_decode_consumer,
            "{profile} should not enable FFN gate/up consumer by default"
        );
        assert!(
            !plan.q8.ffn_gate_up_decode_group_chunking,
            "{profile} should not enable FFN gate/up decode group chunking by default"
        );
        assert!(
            !plan.q8.ffn_gate_up_decode_fused_activation,
            "{profile} should not enable FFN gate/up fused activation by default"
        );
        assert!(
            !plan.q8.ffn_decode_chain,
            "{profile} should not enable FFN decode chain by default"
        );
        assert!(
            !plan.q8.ffn_gate_up_packed_rows4_matmul,
            "{profile} should not enable FFN gate/up packed-rows4 matmul by default"
        );
        assert!(
            !plan.q8.ffn_gate_up_single_owner,
            "{profile} should not enable FFN gate/up single owner by default"
        );
        assert!(
            !plan.q8.ffn_down_decode_consumer,
            "{profile} should not enable FFN-down consumer by default"
        );
        assert!(
            !plan.q8.ffn_down_packed_rows4_matmul,
            "{profile} should not enable packed-rows4 matmul by default"
        );
        assert!(
            !plan.q8.ffn_down_vnni_decode,
            "{profile} should not enable FFN-down VNNI decode by default"
        );
        assert!(
            !plan.q8.ffn_down_vnni_decode_rawptr,
            "{profile} should not enable FFN-down VNNI rawptr decode by default"
        );
        assert!(
            !plan.q8.metal,
            "{profile} should not enable Metal Q8 by default"
        );
        assert!(
            !plan.q8.metal_retained,
            "{profile} should not enable retained Metal Q8 by default"
        );
        assert!(
            !plan.q8.hybrid_retained,
            "{profile} should not enable hybrid Q8 by default"
        );
    }
    std::env::remove_var("CAMELID_RUNTIME_PROFILE");
}

#[test]
fn q8_0_block_dot_env_flags_ignore_outer_whitespace() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", " on ");
    assert!(q8_0_block_dot_enabled());

    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", " f32 ");
    assert!(!q8_0_file_reader_block_dot_enabled());

    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", " dequantized ");
    assert!(!q8_0_file_reader_block_dot_enabled());

    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", " maybe ");
    assert!(!q8_0_file_reader_block_dot_enabled());

    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn q8_0_input_quantization_uses_unrounded_scale_for_quants() {
    let unrounded_scale = 1.0_f32 / 127.0;
    let mut input_values = vec![0.0; Q8_0_BLOCK_VALUES];
    input_values[0] = 1.0;
    input_values[1] = 2.49995 * unrounded_scale;

    let quantized = quantize_q8_0_row(&input_values);
    let block = &quantized.blocks[0];

    assert_eq!(
        block.scale,
        f16_bits_to_f32(f32_to_f16_bits(unrounded_scale))
    );
    assert_eq!(block.quants[0], 127);
    assert_eq!(block.quants[1], 2);
    assert_eq!((input_values[1] / block.scale).round() as i8, 3);
}

#[test]
fn q8_0_two_dot_rows_matches_individual_dot_rows() {
    let input = vec![
        Q8_0Block {
            scale: 0.25,
            quants: std::array::from_fn(|idx| idx as i8 - 16),
        },
        Q8_0Block {
            scale: 0.5,
            quants: std::array::from_fn(|idx| 15 - idx as i8),
        },
    ];
    let first_weight = vec![
        Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| (idx as i8 % 9) - 4),
        },
        Q8_0Block {
            scale: 0.375,
            quants: std::array::from_fn(|idx| (idx as i8 % 7) - 3),
        },
    ];
    let second_weight = vec![
        Q8_0Block {
            scale: 0.625,
            quants: std::array::from_fn(|idx| (idx as i8 % 11) - 5),
        },
        Q8_0Block {
            scale: 0.875,
            quants: std::array::from_fn(|idx| (idx as i8 % 13) - 6),
        },
    ];

    let (first, second) = q8_0_two_dot_rows(&first_weight, &second_weight, &input);

    assert_eq!(first, q8_0_dot_rows(&first_weight, &input));
    assert_eq!(second, q8_0_dot_rows(&second_weight, &input));
}

fn assert_packed_rows4_matches_retained(interleave: Q8_0PackedRows4Interleave) {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    match interleave {
        Q8_0PackedRows4Interleave::I4 => {
            std::env::set_var("CAMELID_Q8_0_PACKED_4X4_DOT", "on");
            std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");
        }
        Q8_0PackedRows4Interleave::I8 => {
            std::env::set_var("CAMELID_Q8_0_PACKED_4X8_DOT", "on");
            std::env::remove_var("CAMELID_Q8_0_PACKED_4X4_DOT");
        }
    }

    let rows = 4;
    let blocks_per_row = 3;
    let mut weight_blocks = Vec::new();
    let mut dequantized = Vec::new();
    for row in 0..rows {
        for block_idx in 0..blocks_per_row {
            let block = Q8_0Block {
                scale: 0.125 + row as f32 * 0.03125 + block_idx as f32 * 0.015625,
                quants: std::array::from_fn(|idx| {
                    ((row as i32 * 11 + block_idx as i32 * 7 + idx as i32) % 41 - 20) as i8
                }),
            };
            dequantized.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
            weight_blocks.push(block);
        }
    }
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows, blocks_per_row * Q8_0_BLOCK_VALUES],
        dequantized,
        weight_blocks.clone(),
    )
    .unwrap();
    let packed = match interleave {
        Q8_0PackedRows4Interleave::I4 => weight.q8_0_packed_rows4_4x4.as_ref(),
        Q8_0PackedRows4Interleave::I8 => weight.q8_0_packed_rows4_4x8.as_ref(),
    }
    .expect("packed rows4 sidecar should be built when opted in");
    assert_eq!(packed.rows, rows);
    assert_eq!(packed.blocks_per_row, blocks_per_row);
    assert_eq!(packed.interleave, interleave);

    let input = quantize_q8_0_blocks(
        &(0..blocks_per_row * Q8_0_BLOCK_VALUES)
            .map(|idx| (idx as f32 - 31.0) * 0.02125)
            .collect::<Vec<_>>(),
    );
    let expected = (0..rows)
        .map(|row| {
            let start = row * blocks_per_row;
            q8_0_dot_rows(&weight_blocks[start..start + blocks_per_row], &input)
        })
        .collect::<Vec<_>>();
    let actual = q8_0_packed_rows4_dot(&packed.blocks, &input, interleave);

    assert_eq!(actual.as_slice(), expected.as_slice());
}

#[test]
fn q8_0_packed_4x4_rows4_matches_retained_block_dot() {
    assert_packed_rows4_matches_retained(Q8_0PackedRows4Interleave::I4);
}

#[test]
fn q8_0_packed_4x8_rows4_matches_retained_block_dot() {
    assert_packed_rows4_matches_retained(Q8_0PackedRows4Interleave::I8);
}

#[test]
fn q8_0_file_backed_packed_rows4_dot_matches_retained_without_q8_blocks() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");
    std::env::remove_var("CAMELID_Q8_0_PACKED_4X4_DOT");
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let blocks_per_row = 1;
    let rows = 4;
    let row_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.25,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(3).wrapping_add(row as i8)),
        })
        .collect();
    let input_values: Vec<f32> = (0..Q8_0_BLOCK_VALUES)
        .map(|idx| (idx as f32 - 16.0) * 0.5)
        .collect();
    let input = CpuTensor::from_f32("input", vec![1, Q8_0_BLOCK_VALUES], input_values).unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_weight",
        vec![rows, Q8_0_BLOCK_VALUES],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();

    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
    let packed_file_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.attn_q.weight",
        TensorShape {
            dims: vec![rows, Q8_0_BLOCK_VALUES],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    assert!(packed_file_weight.q8_0_blocks.is_none());
    assert!(packed_file_weight.q8_0_file_backing.is_none());
    assert!(packed_file_weight.q8_0_packed_rows4_4x8.is_none());
    assert!(packed_file_weight.q8_0_runtime_storage.is_some());

    let actual =
        matmul_rhs_transposed_with_precision(&input, &packed_file_weight, "actual").unwrap();
    assert_slice_close(&actual.data, &expected.data);

    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[test]
fn q8_0_runtime_packed_rows4_f32_fallback_handles_empty_runtime_data() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");

    let blocks_per_row = 1;
    let rows = 4;
    let row_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.0625,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_sub(row as i8)),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, Q8_0_BLOCK_VALUES],
        (0..Q8_0_BLOCK_VALUES)
            .map(|idx| (idx as f32 - 12.0) * 0.25)
            .collect(),
    )
    .unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_weight",
        vec![rows, Q8_0_BLOCK_VALUES],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();

    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.attn_q.weight",
        TensorShape {
            dims: vec![rows, Q8_0_BLOCK_VALUES],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    assert!(packed_weight.data.is_empty());
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(packed_weight.q8_0_file_backing.is_none());

    let actual = matmul_rhs_transposed_with_precision(&input, &packed_weight, "actual")
        .expect("runtime-owned packed Q8 fallback must not crash when block-dot is off");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);

    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[test]
fn q8_0_runtime_packed_ffn_transposed_f32_fallback_handles_empty_runtime_data() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");

    let rows = 64;
    let input_width = Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.2 + row as f32 * 0.004,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_add(row as i8)),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 8.0) * 0.125)
            .collect(),
    )
    .unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_ffn_gate_transposed",
        vec![rows, input_width],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();

    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_gate.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(rows, 1, Q8_0PackedRows4Interleave::I8, &row_blocks).unwrap(),
    );
    assert!(packed_weight.data.is_empty());
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(packed_weight.q8_0_file_backing.is_none());

    let actual = linear_for_role_runtime(&input, &packed_weight, "actual", "ffn gate", false)
        .expect("transposed runtime-owned packed Q8 fallback must not crash when block-dot is off");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);

    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[test]
fn transposed_runtime_packed_attention_k_without_row_major_data_returns_error_instead_of_panicking()
{
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let input_width = Q8_0_BLOCK_VALUES;
    let kv_width = 16;
    let rows = input_width;
    let blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.00390625,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(3).wrapping_add(row as i8)),
        })
        .collect();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.attn_k.weight",
        TensorShape {
            dims: vec![input_width, kv_width],
        },
        Q8_0PackedRows4::from_rows(rows, 1, Q8_0PackedRows4Interleave::I8, &blocks).unwrap(),
    );
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 16.0) * 0.125)
            .collect(),
    )
    .unwrap();

    let outcome = std::panic::catch_unwind(|| {
        linear_for_role_runtime(&input, &packed_weight, "actual", "attention k", false)
    });
    assert!(
        outcome.is_ok(),
        "runtime-packed K tensor must not panic when row-major data is empty"
    );
    let err = outcome.unwrap().expect_err(
            "transposed runtime-packed attention K should be rejected unless a matching packed consumer path is available",
        );
    let err_text = err.to_string();
    assert!(
        err_text.contains(
            "matmul rhs-transposed rhs cannot read tensor blk.0.attn_k.weight as row-major f32"
        ),
        "{err_text}"
    );
    assert!(err_text.contains("storage=no-row-major-data"), "{err_text}");

    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
}

fn assert_q8_0_runtime_packed_ffn_transposed_view_matches_retained_blocks(
    tensor_name: &str,
    role_name: &str,
    descriptor_dims: Vec<usize>,
    rows: usize,
    input_width: usize,
    row_blocks: Vec<Q8_0Block>,
    input_values: Vec<f32>,
) {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let input = CpuTensor::from_f32("input", vec![1, input_width], input_values).unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        format!("retained_{role_name}_transposed"),
        vec![rows, input_width],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();

    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        tensor_name,
        TensorShape {
            dims: descriptor_dims,
        },
        Q8_0PackedRows4::from_rows(
            rows,
            input_width / Q8_0_BLOCK_VALUES,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    let actual =
        linear_for_role_runtime(&input, &packed_weight, "actual", role_name, false).unwrap();

    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(packed_weight.q8_0_file_backing.is_none());

    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[test]
fn q8_0_runtime_packed_ffn_gate_transposed_view_matches_retained_blocks() {
    let rows = 64;
    let input_width = Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.25 + row as f32 * 0.01,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(row as i8)),
        })
        .collect();
    assert_q8_0_runtime_packed_ffn_transposed_view_matches_retained_blocks(
        "blk.0.ffn_gate.weight",
        "ffn gate",
        vec![input_width, rows],
        rows,
        input_width,
        row_blocks,
        (0..input_width)
            .map(|idx| (idx as f32 - 12.0) * 0.25)
            .collect(),
    );
}

#[test]
fn q8_0_runtime_packed_ffn_down_transposed_view_matches_retained_blocks() {
    let rows = 32;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let row_blocks: Vec<Q8_0Block> = (0..rows * 2)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.006,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(9).wrapping_sub(row as i8)),
        })
        .collect();
    assert_q8_0_runtime_packed_ffn_transposed_view_matches_retained_blocks(
        "blk.0.ffn_down.weight",
        "ffn_down",
        vec![input_width, rows],
        rows,
        input_width,
        row_blocks,
        (0..input_width)
            .map(|idx| (idx as f32 - 16.0) * 0.1875)
            .collect(),
    );
}

fn attention_projection_consumer_plan(enabled: bool) -> ResolvedRuntimePlan {
    q8_attention_consumer_plan(enabled, false)
}

fn attention_qkv_consumer_plan(enabled: bool) -> ResolvedRuntimePlan {
    q8_attention_consumer_plan(false, enabled)
}

fn attention_qkv_packed_rows4_matmul_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = q8_attention_consumer_plan(false, false);
    plan.q8.attention_qkv_packed_rows4_matmul = enabled;
    plan
}

fn attention_output_consumer_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = q8_attention_consumer_plan(false, false);
    plan.q8.attention_output_decode_consumer = enabled;
    plan
}

fn attention_output_packed_rows4_matmul_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = q8_attention_consumer_plan(false, false);
    plan.q8.attention_output_packed_rows4_matmul = enabled;
    plan
}

fn output_packed_rows4_matmul_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = q8_attention_consumer_plan(false, false);
    plan.q8.output_packed_rows4_matmul = enabled;
    plan
}

fn q8_attention_consumer_plan(
    attention_projection_decode_consumer: bool,
    attention_qkv_decode_consumer: bool,
) -> ResolvedRuntimePlan {
    ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags {
            block_dot: false,
            file_reader_block_dot: false,
            attention_projection_decode_consumer,
            attention_output_decode_consumer: false,
            attention_output_packed_rows4_matmul: false,
            attention_qkv_decode_consumer,
            attention_qkv_decode_group_chunking: false,
            attention_qkv_packed_rows4_matmul: false,
            output_packed_rows4_matmul: false,
            output_amx_prefill: false,
            output_decode_owner: false,
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_decode_group_chunking: false,
            ffn_gate_up_decode_fused_activation: false,
            ffn_gate_up_decode_paired_dot: false,
            ffn_decode_chain: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_decode_group_chunking: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            ffn_down_vnni_decode_rawptr: false,
            q8_matmul_owner: Q8MatmulOwnerScope::Off,
            q8_matmul_owner_avx2: false,
            q8_matmul_owner_vnni: false,
            q8_matmul_owner_4x8: false,
            metal: false,
            cuda: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
        q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::default(),
    }
}

fn runtime_packed_attention_projection_case(
    role_name: &str,
    tensor_name: &str,
) -> (CpuTensor, CpuTensor, CpuTensor) {
    let rows = 12;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.004,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_sub(row as i8)),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 20.0) * 0.15625)
            .collect(),
    )
    .unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        format!("retained_{role_name}"),
        vec![rows, input_width],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        tensor_name,
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    assert!(packed_weight.data.is_empty());
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(packed_weight.q8_0_file_backing.is_none());
    assert!(matches!(
        packed_weight.q8_0_runtime_storage.as_ref(),
        Some(Q8_0RuntimeStorage::PackedRows4(_))
    ));
    (input, packed_weight, expected)
}

#[test]
fn q8_attention_projection_consumer_matches_runtime_packed_baseline_for_qkv() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let plan = attention_projection_consumer_plan(true);

    for (role, tensor_name) in [
        ("attention_q", "blk.0.attn_q.weight"),
        ("attention_k", "blk.0.attn_k.weight"),
        ("attention_v", "blk.0.attn_v.weight"),
    ] {
        let (input, packed_weight, expected) =
            runtime_packed_attention_projection_case(role, tensor_name);
        let actual = linear_for_role_runtime_with_plan(
            &input,
            &packed_weight,
            format!("actual_{role}"),
            role,
            &plan,
            false,
        )
        .unwrap();
        assert_eq!(actual.shape.dims, expected.shape.dims, "{role}");
        assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    }
}

#[test]
fn q8_attention_qkv_consumer_quantizes_once_for_runtime_packed_qkv() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, q_weight, q_expected) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, k_expected) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, v_expected) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");
    let plan = attention_qkv_consumer_plan(true);

    let (q, k, v) = try_x86_q8_attention_qkv_decode_consumer_path(
        &input, &q_weight, &k_weight, &v_weight, &plan,
    )
    .unwrap()
    .expect("QKV consumer should accept runtime-packed attention Q/K/V weights");

    assert_eq!(q.name, "attention_q_x86_q8_qkv_consumer");
    assert_eq!(k.name, "attention_k_x86_q8_qkv_consumer");
    assert_eq!(v.name, "attention_v_x86_q8_qkv_consumer");
    assert_slice_close_with_tolerance(&q.data, &q_expected.data, 5e-4);
    assert_slice_close_with_tolerance(&k.data, &k_expected.data, 5e-4);
    assert_slice_close_with_tolerance(&v.data, &v_expected.data, 5e-4);
}

#[test]
fn q8_attention_qkv_decode_group_chunking_matches_unchunked_triplet_projection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "7");

    let output_width = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.05 + (idx % 11) as f32 * 0.003,
            quants: std::array::from_fn(|lane| ((idx * 13 + lane * 7) as i16 % 127 - 63) as i8),
        })
        .collect();
    let q_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let k_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let v_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let input: Vec<f32> = (0..input_width)
        .map(|idx| (idx as f32 - 17.0) * 0.0625)
        .collect();
    let quantized_input = quantize_q8_0_row(&input);

    let (q_expected, k_expected, v_expected) =
        q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
            &q_packed,
            &k_packed,
            &v_packed,
            output_width,
            output_width,
            output_width,
            &quantized_input.blocks,
            false,
        )
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let (q_actual, k_actual, v_actual) = pool
        .install(|| {
            q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
                &q_packed,
                &k_packed,
                &v_packed,
                output_width,
                output_width,
                output_width,
                &quantized_input.blocks,
                true,
            )
        })
        .unwrap();

    assert_eq!(x86_q8_attention_qkv_decode_groups_per_chunk(), 7);
    assert_slice_close_with_tolerance(&q_actual.data, &q_expected.data, 1e-6);
    assert_slice_close_with_tolerance(&k_actual.data, &k_expected.data, 1e-6);
    assert_slice_close_with_tolerance(&v_actual.data, &v_expected.data, 1e-6);
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK");
}

#[test]
fn q8_attention_qkv_gqa_parallel_decode_bitwise_matches_serial_triplet_projection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    // An ambient serial-decode toggle would silently turn the parallel leg
    // into serial-vs-serial and prove nothing — clear it explicitly.
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE");

    // GQA shape: q wider than k/v (the unequal-width branch under test).
    let kv_width = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let q_width = kv_width * 3;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let make_packed = |rows: usize, salt: usize| {
        let row_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
            .map(|idx| Q8_0Block {
                scale: 0.05 + ((idx + salt) % 11) as f32 * 0.003,
                quants: std::array::from_fn(|lane| {
                    (((idx + salt) * 13 + lane * 7) as i16 % 127 - 63) as i8
                }),
            })
            .collect();
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap()
    };
    let q_packed = make_packed(q_width, 0);
    let k_packed = make_packed(kv_width, 5);
    let v_packed = make_packed(kv_width, 9);
    let input: Vec<f32> = (0..input_width)
        .map(|idx| (idx as f32 - 17.0) * 0.0625)
        .collect();
    let quantized_input = quantize_q8_0_row(&input);

    for decode_group_chunking in [false, true] {
        std::env::set_var("CAMELID_X86_Q8_QKV_GQA_PARALLEL_DECODE", "0");
        let (q_expected, k_expected, v_expected) =
            q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
                &q_packed,
                &k_packed,
                &v_packed,
                q_width,
                kv_width,
                kv_width,
                &quantized_input.blocks,
                decode_group_chunking,
            )
            .unwrap();
        std::env::set_var("CAMELID_X86_Q8_QKV_GQA_PARALLEL_DECODE", "1");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            // Guard against silent serial-vs-serial degradation: the parallel
            // leg must actually satisfy the helper's parallel predicate for
            // every width in play (kv_width is the narrowest).
            assert!(
                should_parallelize_x86_q8_packed_rows4_decode_output(kv_width),
                "parallel precondition not met — test would degrade to serial-vs-serial"
            );
        });
        let (q_actual, k_actual, v_actual) = pool
            .install(|| {
                q8_0_packed_rows4_single_input_projection_triplet_from_quantized(
                    &q_packed,
                    &k_packed,
                    &v_packed,
                    q_width,
                    kv_width,
                    kv_width,
                    &quantized_input.blocks,
                    decode_group_chunking,
                )
            })
            .unwrap();

        // The lane's contract is BYTE-identical output, not close output.
        assert_eq!(
            q_actual.data, q_expected.data,
            "q diverged (chunking={decode_group_chunking})"
        );
        assert_eq!(
            k_actual.data, k_expected.data,
            "k diverged (chunking={decode_group_chunking})"
        );
        assert_eq!(
            v_actual.data, v_expected.data,
            "v diverged (chunking={decode_group_chunking})"
        );
    }
    std::env::remove_var("CAMELID_X86_Q8_QKV_GQA_PARALLEL_DECODE");
}

/// STAMPEDE Phase 3 Lane B: the batched Q4_K prefill owner must be bitwise
/// identical to the per-cell block-dot path — tile-aligned and ragged row
/// counts, multiple in/out shapes, deterministic wire bytes.
#[test]
fn q4_k_owner_prefill_bitwise_matches_block_dot_core() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");

    for (in_dim, out_dim) in [(512usize, 96usize), (768, 40)] {
        let row_bytes = (in_dim / 256) * 144;
        // Deterministic wire: small finite f16 d/dmin, adversarial scale/min
        // and nibble bytes with genuine full-byte coverage via a
        // multiplicative hash (a linear-in-index generator collapses to one
        // parity class because the 144-byte block stride is even; every bit
        // pattern is structurally valid Q4_K — the kmask unpack caps live
        // scales at 63).
        let wire: Vec<u8> = (0..out_dim * row_bytes)
            .map(|i| match i % 144 {
                0 => 0x66,
                1 => 0x2e, // d ~= 0.1
                2 => 0x99,
                3 => 0x24, // dmin ~= 0.018
                _ => ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8,
            })
            .collect();
        for n_rows in [4usize, 5, 13, 64, 67] {
            let input_data: Vec<f32> = (0..n_rows * in_dim)
                .map(|i| ((i as f32) * 0.37).sin() * 3.0 - 0.8)
                .collect();
            let input =
                CpuTensor::from_f32("owner-test-in", vec![n_rows, in_dim], input_data).unwrap();
            std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
            let base = q4_k_block_dot_core(&input, &wire, out_dim, in_dim, "base").unwrap();
            // Cover BOTH owner inners: VNNI (default when the CPU has it) and
            // the AVX2 fallback (VNNI sub-flag forced off).
            for vnni in ["1", "0"] {
                std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER", "1");
                std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI", vnni);
                reset_q8_schedule_telemetry();
                let owner = q4_k_block_dot_core(&input, &wire, out_dim, in_dim, "owner").unwrap();
                // Engaged-check: if the dispatch silently stopped firing, the
                // "owner" leg would be the base path and this test would pass
                // vacuously.
                let telemetry = snapshot_q8_schedule_telemetry();
                assert!(
                    telemetry.kquant_owner_prefill_taken > 0,
                    "owner leg did not dispatch (n_rows={n_rows}, vnni={vnni}) — vacuous comparison"
                );
                // Per-inner engaged check: the vnni=1 leg must actually run
                // the VNNI inner when the host has the features; on hosts
                // without them the leg legitimately falls back to AVX2 (the
                // AVX2 inner is still covered by the vnni=0 leg).
                let vnni_expected = vnni == "1" && q8_owner_avx512vnni_available();
                assert_eq!(
                    vnni_expected,
                    telemetry.kquant_owner_vnni_taken > 0,
                    "VNNI inner engagement mismatch (n_rows={n_rows}, vnni={vnni}, \
                     host_vnni={})",
                    q8_owner_avx512vnni_available()
                );
                std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
                std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI");
                assert_eq!(owner.data.len(), base.data.len());
                for (i, (a, b)) in owner.data.iter().zip(base.data.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "cell {i} diverged (n_rows={n_rows}, in_dim={in_dim}, \
                         out_dim={out_dim}, vnni={vnni})"
                    );
                }
            }
        }
    }
}

/// STAMPEDE Lane B v5: the 8-row repack owner must be bitwise identical to
/// the per-cell block-dot path — including the ragged tail (out_dim % 8 != 0)
/// and multi-row-block shapes. On hosts without avx512vnni the repack path
/// legitimately never engages (asserted) and the comparison covers the
/// fallback instead.
#[test]
fn q4_k_repack8_owner_bitwise_matches_block_dot_core() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    for (in_dim, out_dim) in [(512usize, 96usize), (768, 40), (512, 19)] {
        let row_bytes = (in_dim / 256) * 144;
        // d/dmin VARY PER ROW (finite f16s, sign included): uniform values
        // would mask a per-row lane permutation in the pack's d/dmin arrays
        // (review finding). Exponent nibbles stay in a finite range.
        let wire: Vec<u8> = (0..out_dim * row_bytes)
            .map(|i| {
                let row = i / row_bytes;
                match i % 144 {
                    0 => ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8,
                    1 => 0x28 | ((row % 8) as u8) | ((((row / 8) % 2) as u8) << 7),
                    2 => ((i as u32).wrapping_mul(0x9E37_79B9) >> 24) as u8,
                    3 => 0x20 | (((row + 3) % 8) as u8),
                    _ => ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8,
                }
            })
            .collect();
        let superblocks = in_dim / 256;
        let pack_rows = out_dim / 8 * 8;
        let pack = crate::tensor::Q4KPackedRows8::from_q4_k_wire(
            pack_rows,
            superblocks,
            &wire[..pack_rows * row_bytes],
        )
        .unwrap();
        for n_rows in [4usize, 13, 65] {
            let input_data: Vec<f32> = (0..n_rows * in_dim)
                .map(|i| ((i as f32) * 0.41).sin() * 2.0 - 0.6)
                .collect();
            let input =
                CpuTensor::from_f32("repack8-in", vec![n_rows, in_dim], input_data).unwrap();
            std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
            let base = q4_k_block_dot_core(&input, &wire, out_dim, in_dim, "base").unwrap();
            std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER", "1");
            reset_q8_schedule_telemetry();
            let owner = q4_k_block_dot_core_with_repack(
                &input,
                &wire,
                out_dim,
                in_dim,
                "owner",
                Some(&pack),
            )
            .unwrap();
            let telemetry = snapshot_q8_schedule_telemetry();
            std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
            assert!(telemetry.kquant_owner_prefill_taken > 0);
            // Engaged-check: the repack path must run exactly when the host
            // has the features (the pack is passed in explicitly here).
            assert_eq!(
                q8_owner_avx512vnni_available(),
                telemetry.kquant_owner_repack8_taken > 0,
                "repack8 engagement mismatch (n_rows={n_rows}, out_dim={out_dim})"
            );
            assert_eq!(owner.data.len(), base.data.len());
            for (i, (a, b)) in owner.data.iter().zip(base.data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "cell {i} diverged (n_rows={n_rows}, in_dim={in_dim}, out_dim={out_dim})"
                );
            }
        }
    }
}

/// Lane B v5 repack round-trip: the packed block must reproduce the wire's
/// d/dmin/scales/mins and nibble bytes exactly (straight byte permutation).
#[test]
fn q4_k_repack8_round_trip_matches_wire() {
    let rows = 16usize;
    let superblocks = 2usize;
    let row_bytes = superblocks * 144;
    let wire: Vec<u8> = (0..rows * row_bytes)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let pack = crate::tensor::Q4KPackedRows8::from_q4_k_wire(rows, superblocks, &wire).unwrap();
    for row_group in 0..rows / 8 {
        for sb in 0..superblocks {
            let gb = &pack.blocks[row_group * superblocks + sb];
            for r in 0..8 {
                let src = ((row_group * 8 + r) * superblocks + sb) * 144;
                let d =
                    crate::tensor::f16_bits_to_f32(u16::from_le_bytes([wire[src], wire[src + 1]]));
                let dmin = crate::tensor::f16_bits_to_f32(u16::from_le_bytes([
                    wire[src + 2],
                    wire[src + 3],
                ]));
                assert_eq!(gb.d[r].to_bits(), d.to_bits());
                assert_eq!(gb.dmin[r].to_bits(), dmin.to_bits());
                let (scales, mins) =
                    crate::tensor::q4_k_unpack_kmask_scales(&wire[src + 4..src + 16]);
                for g in 0..8 {
                    assert_eq!(gb.scales[g][r], scales[g]);
                    assert_eq!(gb.mins[g][r], mins[g]);
                }
                for c in 0..4 {
                    for qc in 0..8 {
                        for b in 0..4 {
                            assert_eq!(
                                gb.qs[c * 256 + qc * 32 + r * 4 + b],
                                wire[src + 16 + c * 32 + qc * 4 + b],
                                "nibble byte mismatch (r={r}, c={c}, qc={qc}, b={b})"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Lane B v5 budget policy: degrade, never refuse — budget 0 denies the build,
/// output stays bit-identical via the single-row inner.
#[test]
fn q4_k_repack8_budget_zero_degrades_bit_identically() {
    #[cfg(target_arch = "x86_64")]
    {
        let _env_guard = env_lock();
        clear_dense_diagnostic_env();
        let in_dim = 512usize;
        let out_dim = 32usize;
        let row_bytes = (in_dim / 256) * 144;
        let wire: Vec<u8> = (0..out_dim * row_bytes)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let input_data: Vec<f32> = (0..8 * in_dim).map(|i| ((i as f32) * 0.3).cos()).collect();
        let input = CpuTensor::from_f32("budget-in", vec![8usize, in_dim], input_data).unwrap();
        let mut weight = CpuTensor::from_f32(
            "budget-w",
            vec![out_dim, in_dim],
            vec![0f32; out_dim * in_dim],
        )
        .unwrap();
        weight.source_type = Some(GgufTensorType::Q4K);
        weight.q4_k_wire_bytes = Some(std::sync::Arc::new(wire.clone()));

        std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
        let base = q4_k_block_dot_core(&input, &wire, out_dim, in_dim, "base").unwrap();

        std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER", "1");
        std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8", "1");
        std::env::set_var("CAMELID_X86_KQUANT_REPACK8_BUDGET_MIB", "0");
        reset_q8_schedule_telemetry();
        let denied = matmul_rhs_transposed_q4_k_block_dot(&input, &weight, "denied").unwrap();
        let telemetry = snapshot_q8_schedule_telemetry();
        std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
        std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8");
        std::env::remove_var("CAMELID_X86_KQUANT_REPACK8_BUDGET_MIB");
        if q8_owner_avx512vnni_available() {
            assert_eq!(telemetry.kquant_owner_repack8_built, 0);
            assert!(telemetry.kquant_owner_repack8_budget_denied > 0);
        }
        assert_eq!(telemetry.kquant_owner_repack8_taken, 0);
        for (a, b) in denied.data.iter().zip(base.data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

/// STAMPEDE Phase 3 Lane B sibling: the batched Q6_K prefill owner must be
/// bitwise identical to the per-cell block-dot path — its f32 shape (8 lane
/// accumulators, final left-fold) is the bits-sensitive part KQUANT_RECON
/// flagged, so this is the load-bearing check.
#[test]
fn q6_k_owner_prefill_bitwise_matches_block_dot_core() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");

    for (in_dim, out_dim) in [(512usize, 96usize), (768, 40)] {
        let row_bytes = (in_dim / 256) * 210;
        // Deterministic wire with genuine full-byte coverage (multiplicative
        // hash — a linear-in-index generator collapses to one parity class
        // because the 210-byte block stride is even); d f16 fixed small, and
        // one scale byte per superblock forced to 0x80 (-128, the i8
        // extreme) so the scale sign path is pinned end-to-end.
        let wire: Vec<u8> = (0..out_dim * row_bytes)
            .map(|i| match i % 210 {
                192 => 0x80, // first per-16 scale = -128
                208 => 0x66,
                209 => 0x2e, // d ~= 0.1
                _ => ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8,
            })
            .collect();
        for n_rows in [4usize, 5, 13, 64, 67] {
            let input_data: Vec<f32> = (0..n_rows * in_dim)
                .map(|i| ((i as f32) * 0.29).cos() * 2.5 - 0.4)
                .collect();
            let input =
                CpuTensor::from_f32("q6k-owner-in", vec![n_rows, in_dim], input_data).unwrap();
            std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
            let base = q6_k_block_dot_core(&input, &wire, out_dim, in_dim, "base").unwrap();
            std::env::set_var("CAMELID_X86_KQUANT_MATMUL_OWNER", "1");
            reset_q8_schedule_telemetry();
            let owner = q6_k_block_dot_core(&input, &wire, out_dim, in_dim, "owner").unwrap();
            assert!(
                snapshot_q8_schedule_telemetry().kquant_owner_prefill_taken > 0,
                "owner leg did not dispatch (n_rows={n_rows}) — vacuous comparison"
            );
            std::env::remove_var("CAMELID_X86_KQUANT_MATMUL_OWNER");
            assert_eq!(owner.data.len(), base.data.len());
            for (i, (a, b)) in owner.data.iter().zip(base.data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "cell {i} diverged (n_rows={n_rows}, in_dim={in_dim}, out_dim={out_dim})"
                );
            }
        }
    }
}

/// Scalar-vs-AVX2 twin for the Q6_K owner's lifted integer lane dot.
#[test]
fn q6_k_owner_aux32_scalar_matches_avx2() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for salt in 0..4usize {
            let a: [i8; 256] =
                std::array::from_fn(|i| ((((i * 17 + salt * 5) % 64) as i16) - 32) as i8);
            // Includes -128 (0x80) among the i8 scales when salt selects it.
            let scales: [u8; 16] = std::array::from_fn(|j| {
                if j == 0 {
                    0x80
                } else {
                    ((j * 37 + salt * 13) % 256) as u8
                }
            });
            let q8: [i8; 256] =
                std::array::from_fn(|i| (((i * 23 + salt * 11) % 255) as i16 - 127) as i8);
            let scalar = q6_k_owner_aux32_scalar(&a, &scales, &q8);
            // SAFETY: avx2 confirmed present above.
            let simd = unsafe { q6_k_owner_aux32_avx2(&a, &scales, &q8) };
            assert_eq!(scalar, simd, "aux32 twin diverged (salt={salt})");
        }
    }
}

/// D15: the Q8 matmul owner default is scoped to exactly win-x86_64 — pin it
/// so a default regression (either direction, on any target) fails loudly.
#[test]
fn q8_matmul_owner_default_scope_is_win_x86_64_only() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER");
    let default_scope_on = super::q8_runtime::Q8RuntimeFlags::from_env()
        .q8_matmul_owner
        .is_on();
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    assert!(
        default_scope_on,
        "D15: owner default must be All on win-x86_64"
    );
    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    assert!(
        !default_scope_on,
        "owner default must stay Off on non-win-x86_64 targets"
    );
    std::env::set_var("CAMELID_X86_Q8_MATMUL_OWNER", "off");
    assert!(
        !super::q8_runtime::Q8RuntimeFlags::from_env()
            .q8_matmul_owner
            .is_on(),
        "explicit =off must always win"
    );
    std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER");
}

/// Scalar-vs-AVX2 twin for the owner's lifted main-side superblock dot.
#[test]
fn q4_k_owner_main_side_scalar_matches_avx2() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for salt in 0..4usize {
            let qs: Vec<u8> = (0..128)
                .map(|i| ((i * 37 + salt * 13 + 5) % 256) as u8)
                .collect();
            let q8: Vec<i8> = (0..256)
                .map(|i| (((i * 29 + salt * 7) % 255) as i16 - 127) as i8)
                .collect();
            let scales: [u8; 8] = std::array::from_fn(|g| ((g * 23 + salt * 3 + 1) % 64) as u8);
            let scalar = q4_k_owner_main_side_scalar(&qs, &q8, &scales);
            // SAFETY: avx2 confirmed present above.
            let simd = unsafe { q4_k_owner_main_side_avx2(&qs, &q8, &scales) };
            assert_eq!(scalar, simd, "main-side twin diverged (salt={salt})");
        }
    }
}

#[test]
fn x86_q8_ffn_gate_up_small_batch_owner_predicate_boundaries() {
    // STAMPEDE P5 follow-up: 2..=16-row batches route past the bespoke
    // gate/up arms to the tiled owner; decode (rows==1) and large chunks are
    // untouched. Both routes are bit-exact vs the same packed-rows4 baseline
    // (their own twin tests), so the swap cannot change bits — this test pins
    // the boundary so the routing doesn't silently widen.
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(0));
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(1));
    // 2..=3 stay on the bespoke arms: the owner's floor is rows >= 4, so
    // declining them would strand short verify chains on the generic
    // per-row fallback (review finding).
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(2));
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(3));
    assert!(x86_q8_ffn_gate_up_small_batch_prefers_owner(4));
    assert!(x86_q8_ffn_gate_up_small_batch_prefers_owner(8));
    assert!(x86_q8_ffn_gate_up_small_batch_prefers_owner(16));
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(17));
    assert!(!x86_q8_ffn_gate_up_small_batch_prefers_owner(512));
}

#[test]
fn q8_ffn_gate_up_decode_group_chunking_matches_unchunked_pair_projection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "5");

    let output_width = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.04 + (idx % 13) as f32 * 0.002,
            quants: std::array::from_fn(|lane| ((idx * 11 + lane * 5) as i16 % 127 - 63) as i8),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.03 + (idx % 17) as f32 * 0.0025,
            quants: std::array::from_fn(|lane| ((idx * 7 + lane * 9) as i16 % 127 - 63) as i8),
        })
        .collect();
    let gate_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &gate_blocks,
    )
    .unwrap();
    let up_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &up_blocks,
    )
    .unwrap();
    let input: Vec<f32> = (0..input_width)
        .map(|idx| (idx as f32 - 29.0) * 0.03125)
        .collect();
    let quantized_input = quantize_q8_0_row(&input);
    let mut gate_expected = vec![0.0_f32; output_width];
    let mut up_expected = vec![0.0_f32; output_width];
    q8_0_packed_rows4_single_input_projection_pair_into_with_decode_chunking(
        &gate_packed,
        &up_packed,
        &quantized_input.blocks,
        &mut gate_expected,
        &mut up_expected,
        false,
    )
    .unwrap();

    let mut gate_actual = vec![0.0_f32; output_width];
    let mut up_actual = vec![0.0_f32; output_width];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        q8_0_packed_rows4_single_input_projection_pair_into_with_decode_chunking(
            &gate_packed,
            &up_packed,
            &quantized_input.blocks,
            &mut gate_actual,
            &mut up_actual,
            true,
        )
    })
    .unwrap();

    assert_eq!(x86_q8_ffn_gate_up_decode_groups_per_chunk(), 5);
    assert_slice_close_with_tolerance(&gate_actual, &gate_expected, 1e-6);
    assert_slice_close_with_tolerance(&up_actual, &up_expected, 1e-6);
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK");
}

#[test]
fn q8_ffn_gate_up_decode_fused_activation_matches_pair_projection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let output_width = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.04 + (idx % 13) as f32 * 0.002,
            quants: std::array::from_fn(|lane| ((idx * 11 + lane * 5) as i16 % 127 - 63) as i8),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.03 + (idx % 17) as f32 * 0.0025,
            quants: std::array::from_fn(|lane| ((idx * 7 + lane * 9) as i16 % 127 - 63) as i8),
        })
        .collect();
    let gate_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &gate_blocks,
    )
    .unwrap();
    let up_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &up_blocks,
    )
    .unwrap();
    let input: Vec<f32> = (0..input_width)
        .map(|idx| (idx as f32 - 29.0) * 0.03125)
        .collect();
    let quantized_input = quantize_q8_0_row(&input);
    let mut gate = vec![0.0_f32; output_width];
    let mut up = vec![0.0_f32; output_width];
    q8_0_packed_rows4_single_input_projection_pair_into_with_decode_chunking(
        &gate_packed,
        &up_packed,
        &quantized_input.blocks,
        &mut gate,
        &mut up,
        false,
    )
    .unwrap();
    let expected: Vec<f32> = gate
        .into_iter()
        .zip(up)
        .map(|(gate_value, up_value)| {
            apply_ffn_gate_up_order(gate_value, up_value, FfnGateUpOrder::GateUp)
        })
        .collect();

    let actual = q8_0_packed_rows4_single_input_projection_pair_activated_from_quantized(
        &gate_packed,
        &up_packed,
        output_width,
        "actual",
        FfnGateUpOrder::GateUp,
        &quantized_input.blocks,
        false,
    )
    .unwrap();

    assert_slice_close_with_tolerance(&actual.data, &expected, 1e-6);
}

#[test]
fn q8_ffn_gate_up_decode_paired_dot_matches_separate_fused_activation() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let output_width = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let blocks_per_row = 3;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.0275 + (idx % 11) as f32 * 0.003,
            quants: std::array::from_fn(|lane| ((idx * 13 + lane * 3) as i16 % 127 - 63) as i8),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|idx| Q8_0Block {
            scale: 0.033 + (idx % 19) as f32 * 0.0015,
            quants: std::array::from_fn(|lane| ((idx * 5 + lane * 11) as i16 % 127 - 63) as i8),
        })
        .collect();
    let gate_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &gate_blocks,
    )
    .unwrap();
    let up_packed = Q8_0PackedRows4::from_rows(
        output_width,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &up_blocks,
    )
    .unwrap();
    let input: Vec<f32> = (0..input_width)
        .map(|idx| ((idx as i32 % 23) as f32 - 11.0) * 0.01953125)
        .collect();
    let quantized_input = quantize_q8_0_row(&input);

    let expected = q8_0_packed_rows4_single_input_projection_pair_activated_from_quantized(
        &gate_packed,
        &up_packed,
        output_width,
        "expected",
        FfnGateUpOrder::GateUp,
        &quantized_input.blocks,
        false,
    )
    .unwrap();
    let actual = q8_0_packed_rows4_single_input_projection_pair_activated_from_quantized(
        &gate_packed,
        &up_packed,
        output_width,
        "actual",
        FfnGateUpOrder::GateUp,
        &quantized_input.blocks,
        true,
    )
    .unwrap();

    assert_slice_close_with_tolerance(&actual.data, &expected.data, 1e-6);
}

#[test]
fn q8_attention_qkv_consumer_is_default_off_and_requires_all_runtime_packed_inputs() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, q_weight, _) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, _) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, _) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");

    assert!(
        try_x86_q8_attention_qkv_decode_consumer_path(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &attention_qkv_consumer_plan(false),
        )
        .unwrap()
        .is_none(),
        "default-off plan should not enter the fused QKV consumer"
    );

    let dense_v = CpuTensor::from_f32(
        "dense_v",
        vec![12, Q8_0_BLOCK_VALUES * 2],
        vec![0.0; 12 * Q8_0_BLOCK_VALUES * 2],
    )
    .unwrap();
    assert!(
        try_x86_q8_attention_qkv_decode_consumer_path(
            &input,
            &q_weight,
            &k_weight,
            &dense_v,
            &attention_qkv_consumer_plan(true),
        )
        .unwrap()
        .is_none(),
        "fused QKV consumer must fail closed unless every Q/K/V projection is runtime-packed Q8_0"
    );
}

#[test]
fn q8_attention_qkv_route_resolver_preserves_decode_and_prefill_guards() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (decode_input, q_weight, _) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, _) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, _) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");
    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, decode_input.dim(1).unwrap()],
        vec![0.0; 2 * decode_input.dim(1).unwrap()],
    )
    .unwrap();

    let decode_route = resolve_x86_q8_attention_qkv_route(
        &decode_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_consumer_plan(true),
        X86Q8AttentionQkvRouteKind::Decode,
    )
    .unwrap()
    .expect("decode route should accept one-row runtime-packed Q/K/V weights");
    assert_eq!(decode_route.input_width, decode_input.dim(1).unwrap());
    assert_eq!(decode_route.q_width, 12);
    assert_eq!(decode_route.k_width, 12);
    assert_eq!(decode_route.v_width, 12);

    assert!(resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_consumer_plan(true),
        X86Q8AttentionQkvRouteKind::Decode,
    )
    .unwrap()
    .is_none());

    let prefill_route = resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(true),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .expect("prefill route should accept multi-row runtime-packed Q/K/V weights");
    assert_eq!(prefill_route.input_width, prefill_input.dim(1).unwrap());
    assert_eq!(prefill_route.q_width, 12);
    assert_eq!(prefill_route.k_width, 12);
    assert_eq!(prefill_route.v_width, 12);

    assert!(resolve_x86_q8_attention_qkv_route(
        &decode_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(true),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());

    assert!(resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(false),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());
}

#[test]
fn q8_attention_qkv_route_policy_records_denials() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();

    let (decode_input, q_weight, _) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, _) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, _) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");
    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, decode_input.dim(1).unwrap()],
        vec![0.0; 2 * decode_input.dim(1).unwrap()],
    )
    .unwrap();

    assert_eq!(
        X86Q8AttentionQkvRouteKind::Decode.telemetry_name(),
        "decode_consumer"
    );
    assert_eq!(
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul.telemetry_name(),
        "packed_rows4_matmul_prefill"
    );

    assert!(resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_consumer_plan(true),
        X86Q8AttentionQkvRouteKind::Decode,
    )
    .unwrap()
    .is_none());
    assert!(resolve_x86_q8_attention_qkv_route(
        &decode_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(true),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());
    assert!(resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(false),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());

    let dense_v = CpuTensor::from_f32(
        "dense_v",
        vec![12, Q8_0_BLOCK_VALUES * 2],
        vec![0.0; 12 * Q8_0_BLOCK_VALUES * 2],
    )
    .unwrap();
    assert!(resolve_x86_q8_attention_qkv_route(
        &prefill_input,
        &q_weight,
        &k_weight,
        &dense_v,
        &attention_qkv_packed_rows4_matmul_plan(true),
        X86Q8AttentionQkvRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());

    let telemetry = snapshot_q8_schedule_telemetry();
    assert!(telemetry
        .projection_route_denials
        .contains_key("attention_qkv.decode_consumer.prefill_or_empty_input"));
    assert!(telemetry
        .projection_route_denials
        .contains_key("attention_qkv.packed_rows4_matmul_prefill.decode_or_empty_input"));
    assert!(telemetry
        .projection_route_denials
        .contains_key("attention_qkv.packed_rows4_matmul_prefill.plan_off"));
    assert!(telemetry
        .projection_route_denials
        .contains_key("attention_qkv.packed_rows4_matmul_prefill.missing_v_runtime_packed_rows4"));

    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
}

#[test]
fn q8_attention_qkv_prefill_consumer_gate_is_default_off() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER");
    assert!(!x86_q8_attention_qkv_prefill_consumer_enabled());
    std::env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER", "on");
    assert!(x86_q8_attention_qkv_prefill_consumer_enabled());
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER");
}

#[test]
fn q8_attention_qkv_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, q_weight, _q_expected) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, _k_expected) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, _v_expected) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");
    let input_width = q_weight.dim(0).unwrap();
    let output_width = q_weight.dim(1).unwrap();
    let rows = 3;
    let input = CpuTensor::from_f32(
        "prefill_qkv_context",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 13.0) * 0.078125
                    + (idx / input_width) as f32 * 0.046875
            })
            .collect(),
    )
    .unwrap();
    let plan = attention_qkv_packed_rows4_matmul_plan(true);

    let (q, k, v) = try_x86_q8_attention_qkv_packed_rows4_matmul_path(
        &input, &q_weight, &k_weight, &v_weight, &plan,
    )
    .unwrap()
    .expect("QKV packed-rows4 matmul should accept multi-row runtime-packed Q/K/V weights");

    assert_eq!(q.name, "attention_q_x86_q8_qkv_packed_rows4_matmul");
    assert_eq!(k.name, "attention_k_x86_q8_qkv_packed_rows4_matmul");
    assert_eq!(v.name, "attention_v_x86_q8_qkv_packed_rows4_matmul");
    assert_eq!(q.shape.dims, vec![rows, output_width]);
    assert_eq!(k.shape.dims, q.shape.dims);
    assert_eq!(v.shape.dims, q.shape.dims);

    let expected_q = q8_0_packed_rows4_matmul_projection(
        &input,
        match q_weight.q8_0_runtime_storage.as_ref() {
            Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
            other => panic!("expected runtime-packed Q weight, got {other:?}"),
        },
        output_width,
        "expected_q",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    let expected_k = q8_0_packed_rows4_matmul_projection(
        &input,
        match k_weight.q8_0_runtime_storage.as_ref() {
            Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
            other => panic!("expected runtime-packed K weight, got {other:?}"),
        },
        output_width,
        "expected_k",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    let expected_v = q8_0_packed_rows4_matmul_projection(
        &input,
        match v_weight.q8_0_runtime_storage.as_ref() {
            Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
            other => panic!("expected runtime-packed V weight, got {other:?}"),
        },
        output_width,
        "expected_v",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    assert_slice_close_with_tolerance(&q.data, &expected_q.data, 5e-4);
    assert_slice_close_with_tolerance(&k.data, &expected_k.data, 5e-4);
    assert_slice_close_with_tolerance(&v.data, &expected_v.data, 5e-4);
}

#[test]
fn q8_attention_qkv_packed_rows4_matmul_is_plan_gated_and_prefill_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (decode_input, q_weight, _) =
        runtime_packed_attention_projection_case("attention_q", "blk.0.attn_q.weight");
    let (_, k_weight, _) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");
    let (_, v_weight, _) =
        runtime_packed_attention_projection_case("attention_v", "blk.0.attn_v.weight");

    assert!(
        try_x86_q8_attention_qkv_packed_rows4_matmul_path(
            &decode_input,
            &q_weight,
            &k_weight,
            &v_weight,
            &attention_qkv_packed_rows4_matmul_plan(true),
        )
        .unwrap()
        .is_none(),
        "the matrix path intentionally leaves one-row decode to the decode consumer"
    );

    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, decode_input.dim(1).unwrap()],
        vec![0.0; 2 * decode_input.dim(1).unwrap()],
    )
    .unwrap();
    assert!(try_x86_q8_attention_qkv_packed_rows4_matmul_path(
        &prefill_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &attention_qkv_packed_rows4_matmul_plan(false),
    )
    .unwrap()
    .is_none());

    let dense_v = CpuTensor::from_f32(
        "dense_v",
        vec![12, Q8_0_BLOCK_VALUES * 2],
        vec![0.0; 12 * Q8_0_BLOCK_VALUES * 2],
    )
    .unwrap();
    assert!(try_x86_q8_attention_qkv_packed_rows4_matmul_path(
        &prefill_input,
        &q_weight,
        &k_weight,
        &dense_v,
        &attention_qkv_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
}

#[test]
fn q8_attention_projection_consumer_is_plan_gated_and_role_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) =
        runtime_packed_attention_projection_case("attention_k", "blk.0.attn_k.weight");

    let disabled = attention_projection_consumer_plan(false);
    assert!(
        try_x86_q8_attention_projection_decode_consumer_path(
            &input,
            &packed_weight,
            "disabled",
            "attention_k",
            &disabled,
        )
        .unwrap()
        .is_none(),
        "default-off plan should not enter the Q/K/V consumer"
    );

    let enabled = attention_projection_consumer_plan(true);
    assert!(
        try_x86_q8_attention_projection_decode_consumer_path(
            &input,
            &packed_weight,
            "wrong_role",
            "attention_output",
            &enabled,
        )
        .unwrap()
        .is_none(),
        "attention_output must not use the Q/K/V consumer slice"
    );
}

#[test]
fn q8_attention_output_consumer_matches_runtime_packed_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, expected) =
        runtime_packed_attention_projection_case("attention_output", "blk.0.attn_output.weight");

    let actual = linear_runtime_with_plan(
        &input,
        &packed_weight,
        "actual_attention_output",
        &attention_output_consumer_plan(true),
        false,
    )
    .unwrap();

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(matches!(
        packed_weight.q8_0_runtime_storage.as_ref(),
        Some(Q8_0RuntimeStorage::PackedRows4(_))
    ));
}

#[test]
fn q8_attention_output_consumer_is_separate_default_off_x86_gate() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) =
        runtime_packed_attention_projection_case("attention_output", "blk.0.attn_output.weight");

    assert!(
        try_x86_q8_attention_output_decode_consumer_path(
            &input,
            &packed_weight,
            "disabled",
            "linear",
            &attention_output_consumer_plan(false),
        )
        .unwrap()
        .is_none(),
        "attention output consumer must stay default-off"
    );
    assert!(
        try_x86_q8_attention_output_decode_consumer_path(
            &input,
            &packed_weight,
            "wrong_role",
            "attention_q",
            &attention_output_consumer_plan(true),
        )
        .unwrap()
        .is_none(),
        "Q/K/V roles must not enter the attention-output consumer"
    );
    let projection_only = attention_projection_consumer_plan(true);
    assert!(
        try_x86_q8_attention_output_decode_consumer_path(
            &input,
            &packed_weight,
            "projection_only",
            "linear",
            &projection_only,
        )
        .unwrap()
        .is_none(),
        "old Q/K/V projection gate must not enable attention output"
    );
}

#[test]
fn q8_attention_output_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) =
        runtime_packed_attention_projection_case("attention_output", "blk.0.attn_output.weight");
    let input_width = packed_weight.dim(0).unwrap();
    let output_width = packed_weight.dim(1).unwrap();
    let rows = 3;
    let input = CpuTensor::from_f32(
        "prefill_attention_context",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 11.0) * 0.09375 + (idx / input_width) as f32 * 0.03125
            })
            .collect(),
    )
    .unwrap();
    let packed = match packed_weight.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed rows4 weight, got {other:?}"),
    };
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let mut expected_values = vec![0.0_f32; rows * output_width];
    for row_idx in 0..rows {
        let input_start = row_idx * input_width;
        let quantized = quantize_q8_0_row(&input.data[input_start..input_start + input_width]);
        for (group_idx, output_chunk) in expected_values
            [row_idx * output_width..(row_idx + 1) * output_width]
            .chunks_exact_mut(4)
            .enumerate()
        {
            let group_start = group_idx * blocks_per_row;
            let sums = q8_0_packed_rows4_dot(
                &packed.blocks[group_start..group_start + blocks_per_row],
                &quantized.blocks,
                Q8_0PackedRows4Interleave::I8,
            );
            output_chunk.copy_from_slice(&sums);
        }
    }
    let expected =
        CpuTensor::from_f32("expected", vec![rows, output_width], expected_values).unwrap();
    let plan = attention_output_packed_rows4_matmul_plan(true);

    let actual = linear_for_role_runtime_with_plan(
        &input,
        &packed_weight,
        "actual_attention_output_prefill",
        "linear",
        &plan,
        false,
    )
    .unwrap();

    assert_eq!(actual.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

#[test]
fn q8_attention_output_packed_rows4_matmul_is_plan_gated_and_shape_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) =
        runtime_packed_attention_projection_case("attention_output", "blk.0.attn_output.weight");

    assert!(
        try_x86_q8_attention_output_packed_rows4_matmul_path(
            &input,
            &packed_weight,
            "decode_row",
            "linear",
            &attention_output_packed_rows4_matmul_plan(true),
        )
        .unwrap()
        .is_none(),
        "the matrix path intentionally leaves one-row decode to the decode consumer"
    );

    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, input.dim(1).unwrap()],
        vec![0.0; 2 * input.dim(1).unwrap()],
    )
    .unwrap();
    assert!(try_x86_q8_attention_output_packed_rows4_matmul_path(
        &prefill_input,
        &packed_weight,
        "disabled",
        "linear",
        &attention_output_packed_rows4_matmul_plan(false),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_attention_output_packed_rows4_matmul_path(
        &prefill_input,
        &packed_weight,
        "wrong_role",
        "attention_q",
        &attention_output_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
}

fn ffn_down_consumer_plan(enabled: bool) -> ResolvedRuntimePlan {
    ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags {
            block_dot: false,
            file_reader_block_dot: false,
            attention_projection_decode_consumer: false,
            attention_output_decode_consumer: false,
            attention_output_packed_rows4_matmul: false,
            attention_qkv_decode_consumer: false,
            attention_qkv_decode_group_chunking: false,
            attention_qkv_packed_rows4_matmul: false,
            output_packed_rows4_matmul: false,
            output_amx_prefill: false,
            output_decode_owner: false,
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_decode_group_chunking: false,
            ffn_gate_up_decode_fused_activation: false,
            ffn_gate_up_decode_paired_dot: false,
            ffn_decode_chain: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: enabled,
            ffn_down_decode_group_chunking: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            ffn_down_vnni_decode_rawptr: false,
            q8_matmul_owner: Q8MatmulOwnerScope::Off,
            q8_matmul_owner_avx2: false,
            q8_matmul_owner_vnni: false,
            q8_matmul_owner_4x8: false,
            metal: false,
            cuda: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
        q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::default(),
    }
}

fn ffn_down_vnni_decode_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_down_consumer_plan(false);
    plan.q8.ffn_down_vnni_decode = enabled;
    plan.q8.ffn_down_vnni_decode_rawptr =
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
    plan
}

fn ffn_down_packed_rows4_matmul_plan(enabled: bool) -> ResolvedRuntimePlan {
    ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags {
            block_dot: false,
            file_reader_block_dot: false,
            attention_projection_decode_consumer: false,
            attention_output_decode_consumer: false,
            attention_output_packed_rows4_matmul: false,
            attention_qkv_decode_consumer: false,
            attention_qkv_decode_group_chunking: false,
            attention_qkv_packed_rows4_matmul: false,
            output_packed_rows4_matmul: false,
            output_amx_prefill: false,
            output_decode_owner: false,
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_decode_group_chunking: false,
            ffn_gate_up_decode_fused_activation: false,
            ffn_gate_up_decode_paired_dot: false,
            ffn_decode_chain: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_decode_group_chunking: false,
            ffn_down_packed_rows4_matmul: enabled,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            ffn_down_vnni_decode_rawptr: false,
            q8_matmul_owner: Q8MatmulOwnerScope::Off,
            q8_matmul_owner_avx2: false,
            q8_matmul_owner_vnni: false,
            q8_matmul_owner_4x8: false,
            metal: false,
            cuda: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
        q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::default(),
    }
}

fn ffn_down_gemm4_prefill_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_down_packed_rows4_matmul_plan(false);
    plan.q8.ffn_down_gemm4_prefill = enabled;
    plan
}

fn ffn_down_single_owner_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_down_packed_rows4_matmul_plan(false);
    plan.q8.ffn_down_single_owner = enabled;
    plan
}

fn ffn_gate_up_consumer_plan(enabled: bool) -> ResolvedRuntimePlan {
    ResolvedRuntimePlan {
        linear_accumulation_precision: LinearAccumulationPrecision::F32,
        q8: Q8RuntimeFlags {
            block_dot: false,
            file_reader_block_dot: false,
            attention_projection_decode_consumer: false,
            attention_output_decode_consumer: false,
            attention_output_packed_rows4_matmul: false,
            attention_qkv_decode_consumer: false,
            attention_qkv_decode_group_chunking: false,
            attention_qkv_packed_rows4_matmul: false,
            output_packed_rows4_matmul: false,
            output_amx_prefill: false,
            output_decode_owner: false,
            ffn_gate_up_decode_consumer: enabled,
            ffn_gate_up_decode_group_chunking: false,
            ffn_gate_up_decode_fused_activation: false,
            ffn_gate_up_decode_paired_dot: false,
            ffn_decode_chain: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_decode_group_chunking: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            ffn_down_vnni_decode_rawptr: false,
            q8_matmul_owner: Q8MatmulOwnerScope::Off,
            q8_matmul_owner_avx2: false,
            q8_matmul_owner_vnni: false,
            q8_matmul_owner_4x8: false,
            metal: false,
            cuda: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
        q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::default(),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn ffn_gate_up_packed_rows4_matmul_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_gate_up_consumer_plan(false);
    plan.q8.ffn_gate_up_packed_rows4_matmul = enabled;
    plan
}

fn ffn_gate_up_single_owner_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_gate_up_consumer_plan(false);
    plan.q8.ffn_gate_up_single_owner = enabled;
    plan
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn ffn_decode_chain_plan() -> ResolvedRuntimePlan {
    let mut plan = ffn_gate_up_consumer_plan(true);
    plan.q8.ffn_decode_chain = true;
    plan.q8.ffn_down_decode_consumer = true;
    plan
}

fn runtime_packed_ffn_gate_up_case() -> (CpuTensor, CpuTensor, CpuTensor, GatedFfnActivation) {
    let rows = 64;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.125 + block_idx as f32 * 0.005,
            quants: std::array::from_fn(|idx| {
                (idx as i8).wrapping_mul(3).wrapping_add(block_idx as i8)
            }),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.2 + block_idx as f32 * 0.003,
            quants: std::array::from_fn(|idx| {
                (idx as i8).wrapping_mul(7).wrapping_sub(block_idx as i8)
            }),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 8.0) * 0.125)
            .collect(),
    )
    .unwrap();
    let retained_gate = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_gate",
        vec![rows, input_width],
        dequantized_q8_0_rows(&gate_blocks),
        gate_blocks.clone(),
    )
    .unwrap();
    let retained_up = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_up",
        vec![rows, input_width],
        dequantized_q8_0_rows(&up_blocks),
        up_blocks.clone(),
    )
    .unwrap();
    let expected = gated_ffn_activation_with_plan(
        &input,
        &retained_gate,
        &retained_up,
        "expected",
        &ffn_gate_up_consumer_plan(false),
        false,
    )
    .unwrap();
    let packed_gate = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_gate.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &gate_blocks,
        )
        .unwrap(),
    );
    let packed_up = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_up.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &up_blocks,
        )
        .unwrap(),
    );
    (input, packed_gate, packed_up, expected)
}

fn runtime_packed_ffn_down_case() -> (CpuTensor, CpuTensor, CpuTensor) {
    let rows = 32;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.006,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(9).wrapping_sub(row as i8)),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 16.0) * 0.1875)
            .collect(),
    )
    .unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_ffn_down_transposed",
        vec![rows, input_width],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    assert!(packed_weight.data.is_empty());
    assert!(packed_weight.q8_0_blocks.is_none());
    assert!(packed_weight.q8_0_file_backing.is_none());
    assert!(matches!(
        packed_weight.q8_0_runtime_storage.as_ref(),
        Some(Q8_0RuntimeStorage::PackedRows4(packed))
            if packed.rows == rows && packed.blocks_per_row == blocks_per_row
    ));
    (input, packed_weight, expected)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn runtime_vnni_packed_ffn_down_case() -> (CpuTensor, CpuTensor, CpuTensor) {
    const Q8_0_BLOCK_BYTES: usize = 34;
    let rows = 64;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let mut raw = Vec::with_capacity(rows * blocks_per_row * Q8_0_BLOCK_BYTES);
    let mut row_blocks = Vec::with_capacity(rows * blocks_per_row);
    for row in 0..rows {
        for block_idx in 0..blocks_per_row {
            let scale = 0.125 + row as f32 * 0.003 + block_idx as f32 * 0.017;
            let scale_bits = f32_to_f16_bits(scale);
            let quants = std::array::from_fn(|idx| {
                (idx as i8)
                    .wrapping_mul(5)
                    .wrapping_add((row as i8).wrapping_mul(3))
                    .wrapping_sub((block_idx as i8).wrapping_mul(11))
            });
            raw.extend_from_slice(&scale_bits.to_le_bytes());
            raw.extend(quants.iter().map(|value| *value as u8));
            row_blocks.push(Q8_0Block {
                scale: f16_bits_to_f32(scale_bits),
                quants,
            });
        }
    }
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 21.0) * 0.15625)
            .collect(),
    )
    .unwrap();
    let retained_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_ffn_down_transposed",
        vec![rows, input_width],
        dequantized_q8_0_rows(&row_blocks),
        row_blocks,
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_with_precision(&input, &retained_weight, "expected").unwrap();
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
    let packed =
        Q8_0PackedRows4::from_q8_0_bytes(rows, blocks_per_row, Q8_0PackedRows4Interleave::I8, &raw)
            .unwrap();
    assert!(packed.vnni_packed.is_some());
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        packed,
    );
    (input, packed_weight, expected)
}

#[test]
fn mac_q8_ffn_down_decode_consumer_alias_is_default_off_and_opt_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER");
    assert!(!Q8RuntimeFlags::from_env().ffn_down_decode_consumer);

    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_down_decode_consumer);
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mac_q8_ffn_down_decode_consumer_uses_mac_route_telemetry_name() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    assert_eq!(
        q8_ffn_down_decode_consumer_route_name(false),
        "x86_decode_consumer"
    );

    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
    assert_eq!(
        q8_ffn_down_decode_consumer_route_name(false),
        "mac_decode_consumer"
    );
    assert_eq!(
        q8_ffn_down_decode_consumer_route_name(true),
        "mac_decode_consumer_group_chunking"
    );
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mac_q8_ffn_down_single_projection_scheduler_counters_are_default_off_and_opt_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS");
    assert!(!mac_q8_ffn_down_single_projection_scheduler_counters_enabled());

    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS", "on");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    assert!(mac_q8_ffn_down_single_projection_scheduler_counters_enabled());
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    assert!(!mac_q8_ffn_down_single_projection_scheduler_counters_enabled());

    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS");
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn mac_q8_ffn_down_single_projection_scheduler_counters_fail_closed_off_mac() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS", "on");
    assert!(!mac_q8_ffn_down_single_projection_scheduler_counters_enabled());
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS");
}

#[test]
fn q8_projection_route_telemetry_records_layer_route_bucket() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();

    assert_eq!(
        q8_schedule_layer_index_for_projection_name("layer_21_ffn_down"),
        Some(21)
    );
    assert_eq!(q8_schedule_layer_index_for_projection_name("logits"), None);

    record_q8_schedule_output_projection_route_call(
        "ffn_down",
        "mac_decode_consumer",
        Some("layer_21_ffn_down"),
        1,
        8192,
        3072,
        12_345,
    );
    record_q8_schedule_output_projection_route_call(
        "logits",
        "q8_0_retained_blocks",
        Some("logits"),
        1,
        3072,
        128_256,
        6_789,
    );

    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.output_projection_calls, 2);
    assert_eq!(telemetry.ffn_gate_up_decode_consumer_taken, 0);
    assert_eq!(telemetry.ffn_gate_up_decode_fused_activation_taken, 0);
    assert_eq!(telemetry.ffn_gate_up_decode_consumer_activation_us, 0);
    assert_eq!(telemetry.ffn_gate_up_decode_consumer_tensor_us, 0);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.mac_decode_consumer"));
    let layer_route = telemetry
        .output_projection_by_layer_route
        .get("layer_21.ffn_down.mac_decode_consumer")
        .expect("layer route telemetry");
    assert_eq!(layer_route.layer_index, 21);
    assert_eq!(layer_route.calls, 1);
    assert_eq!(layer_route.elapsed_us, 12_345);
    assert!(!telemetry
        .output_projection_by_layer_route
        .contains_key("layer_0.logits.q8_0_retained_blocks"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
}

#[test]
fn mac_q8_ffn_gate_up_decode_consumer_alias_is_default_off_and_opt_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT");
    assert!(!Q8RuntimeFlags::from_env().ffn_gate_up_decode_consumer);
    assert!(!Q8RuntimeFlags::from_env().ffn_gate_up_decode_group_chunking);
    assert!(!Q8RuntimeFlags::from_env().ffn_gate_up_decode_fused_activation);
    assert!(!Q8RuntimeFlags::from_env().ffn_gate_up_decode_paired_dot);

    std::env::set_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_consumer);
    std::env::remove_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER");

    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_group_chunking);
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING");

    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_fused_activation);
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION");

    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_fused_activation);
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED");

    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_paired_dot);
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_ffn_decode_chain_is_default_off_and_matches_split_consumers() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN");
    assert!(!q8_0_env_flag_enabled_default_off(
        "CAMELID_X86_Q8_FFN_DECODE_CHAIN"
    ));
    assert!(!Q8RuntimeFlags::from_env().ffn_decode_chain);

    let (input, packed_gate, packed_up, _expected_gate_up) = runtime_packed_ffn_gate_up_case();
    let (_down_input, packed_down, _expected_down) = runtime_packed_ffn_down_case();
    let plan = ffn_decode_chain_plan();

    assert!(try_x86_q8_ffn_decode_chain_path(
        &input,
        &packed_gate,
        &packed_up,
        &packed_down,
        "layer_0_ffn_activated",
        "layer_0_ffn_down",
        &ResolvedRuntimePlan {
            q8: Q8RuntimeFlags {
                ffn_decode_chain: false,
                ..plan.q8
            },
            ..plan
        },
    )
    .unwrap()
    .is_none());

    std::env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
    assert!(q8_0_env_flag_enabled_default_off(
        "CAMELID_X86_Q8_FFN_DECODE_CHAIN"
    ));
    assert!(Q8RuntimeFlags::from_env().ffn_decode_chain);
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();

    let activated = gated_ffn_activation_with_plan(
        &input,
        &packed_gate,
        &packed_up,
        "expected_activated",
        &ffn_gate_up_consumer_plan(true),
        false,
    )
    .unwrap();
    let expected = linear_for_role_runtime_with_plan(
        &activated.tensor,
        &packed_down,
        "expected_down",
        "ffn_down",
        &ffn_down_consumer_plan(true),
        false,
    )
    .unwrap();
    reset_q8_schedule_telemetry();

    // Run the chain enough times for the stage timings to be deterministic:
    // they accumulate nanoseconds, but on Windows `Instant` ticks at QPC
    // granularity (commonly 100ns), so one sub-tick stage (the 64-element row
    // quantize is tens of ns) can legitimately measure 0 in a single call.
    // Across 128 calls the accumulated readings cannot all round to zero.
    const CHAIN_CALLS: u64 = 128;
    let mut actual = None;
    for _ in 0..CHAIN_CALLS {
        actual = Some(
            try_x86_q8_ffn_decode_chain_path(
                &input,
                &packed_gate,
                &packed_up,
                &packed_down,
                "layer_0_ffn_activated",
                "layer_0_ffn_down",
                &plan,
            )
            .unwrap()
            .expect("x86 FFN decode chain should cover runtime-packed gate/up/down"),
        );
    }
    let actual = actual.expect("at least one chain call");

    assert_eq!(actual.tensor.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.tensor.data, &expected.data, 5e-4);
    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_gate_up_decode_consumer_taken, 0);
    assert_eq!(
        telemetry.ffn_gate_up_decode_fused_activation_taken,
        CHAIN_CALLS
    );
    assert_eq!(telemetry.ffn_decode_chain_taken, CHAIN_CALLS);
    assert_eq!(telemetry.ffn_down_decode_consumer_taken, CHAIN_CALLS);
    assert!(telemetry.ffn_decode_chain_total_ns > 0);
    assert!(telemetry.ffn_decode_chain_input_quantize_ns > 0);
    assert!(telemetry.ffn_decode_chain_activation_quantize_ns > 0);
    assert!(telemetry.ffn_decode_chain_down_ns > 0);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_gate_up.decode_fused_activation"));
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_decode_consumer"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_decode_chain_uses_vnni_down_when_gated() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        return;
    }

    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();

    let (input, packed_gate, packed_up, _expected_gate_up) = runtime_packed_ffn_gate_up_case();
    let (_down_input, packed_down, _expected_down) = runtime_vnni_packed_ffn_down_case();
    let mut plan = ffn_decode_chain_plan();
    plan.q8.ffn_down_vnni_decode = true;

    let activated = gated_ffn_activation_with_plan(
        &input,
        &packed_gate,
        &packed_up,
        "expected_activated",
        &ffn_gate_up_consumer_plan(true),
        false,
    )
    .unwrap();
    let expected = try_x86_q8_ffn_down_decode_consumer_path(
        &activated.tensor,
        &packed_down,
        "expected_down",
        "ffn_down",
        &ffn_down_vnni_decode_plan(true),
    )
    .unwrap()
    .expect("standalone VNNI FFN-down decode output");
    reset_q8_schedule_telemetry();

    let actual = try_x86_q8_ffn_decode_chain_path(
        &input,
        &packed_gate,
        &packed_up,
        &packed_down,
        "layer_0_ffn_activated",
        "layer_0_ffn_down",
        &plan,
    )
    .unwrap()
    .expect("x86 FFN decode chain should cover VNNI-packed down projection");

    assert_eq!(actual.tensor.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.tensor.data, &expected.data, 5e-4);
    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_decode_chain_taken, 1);
    assert_eq!(telemetry.ffn_down_vnni_decode_taken, 1);
    assert_eq!(telemetry.ffn_down_vnni_decode_reject_no_vnni_pack, 0);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_vnni_decode_consumer"));

    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[test]
fn q8_ffn_down_consumer_matches_runtime_packed_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, expected) = runtime_packed_ffn_down_case();
    let plan = ffn_down_consumer_plan(true);

    let actual = linear_for_role_runtime_with_plan(
        &input,
        &packed_weight,
        "actual",
        "ffn_down",
        &plan,
        false,
    )
    .unwrap();

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_consumer_matches_rows4_decode_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        return;
    }
    let (input, packed_weight, expected) = runtime_vnni_packed_ffn_down_case();
    let plan = ffn_down_vnni_decode_plan(true);

    let actual = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_0_ffn_down",
        "ffn_down",
        &plan,
    )
    .unwrap()
    .expect("VNNI FFN-down decode output");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_records_selected_route() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        return;
    }
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (input, packed_weight, _expected) = runtime_vnni_packed_ffn_down_case();

    let _ = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_7_ffn_down",
        "ffn_down",
        &ffn_down_vnni_decode_plan(true),
    )
    .unwrap()
    .expect("VNNI FFN-down decode output");

    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_down_vnni_decode_taken, 1);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_vnni_decode_consumer"));
    assert!(telemetry
        .output_projection_by_layer_route
        .contains_key("layer_7.ffn_down.x86_vnni_decode_consumer"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_rawptr_matches_rows4_decode_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
        return;
    }
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
    let (input, packed_weight, expected) = runtime_vnni_packed_ffn_down_case();
    let plan = ffn_down_vnni_decode_plan(true);

    let actual = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_0_ffn_down",
        "ffn_down",
        &plan,
    )
    .unwrap()
    .expect("rawptr VNNI FFN-down decode output");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_rawptr_records_selected_route() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
        return;
    }
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (input, packed_weight, _expected) = runtime_vnni_packed_ffn_down_case();

    let _ = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_11_ffn_down",
        "ffn_down",
        &ffn_down_vnni_decode_plan(true),
    )
    .unwrap()
    .expect("rawptr VNNI FFN-down decode output");

    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_down_vnni_decode_taken, 1);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_vnni_decode_rawptr_consumer"));
    assert!(telemetry
        .output_projection_by_layer_route
        .contains_key("layer_11.ffn_down.x86_vnni_decode_rawptr_consumer"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_rawptr_env_does_not_bypass_runtime_plan() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !x86_q8_vnni_decode_cpu_supported() {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
        return;
    }
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (input, packed_weight, _expected) = runtime_vnni_packed_ffn_down_case();
    let mut plan = ffn_down_vnni_decode_plan(true);
    plan.q8.ffn_down_vnni_decode_rawptr = false;

    let _ = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_13_ffn_down",
        "ffn_down",
        &plan,
    )
    .unwrap()
    .expect("planned VNNI FFN-down decode output");

    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_down_vnni_decode_taken, 1);
    assert!(telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_vnni_decode_consumer"));
    assert!(!telemetry
        .output_projection_by_route
        .contains_key("ffn_down.x86_vnni_decode_rawptr_consumer"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_vnni_decode_rawptr_avx2_matches_rows4_decode_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if !std::arch::is_x86_feature_detected!("avx2") {
        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
        return;
    }
    let (input, packed_weight, expected) = runtime_vnni_packed_ffn_down_case();
    let quantized_input = quantize_q8_0_row(&input.data);
    let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = packed_weight.q8_0_runtime_storage.as_ref()
    else {
        panic!("expected packed rows4 runtime storage");
    };
    let vnni_packed = packed.vnni_packed.as_ref().expect("VNNI sidecar");
    let mut output = vec![0.0_f32; expected.data.len()];

    unsafe {
        q8_0_vnni_decode_1x64_projection_rawptr_avx2(
            vnni_packed,
            &quantized_input.blocks,
            &mut output,
        );
    }

    assert_slice_close_with_tolerance(&output, &expected.data, 5e-4);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_vnni_tile16_avx2_matches_scalar_for_extreme_i8_values() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let input_block = Q8_0Block {
        scale: 1.0,
        quants: std::array::from_fn(|idx| match idx % 4 {
            0 => -128,
            1 => -3,
            2 => 0,
            _ => 127,
        }),
    };
    let mut tile = Q8_0VnniTile16 {
        quants: [0; 512],
        scale_f16: [0x3c00; 16],
        scale_f32: [1.0; 16],
        comp: [0; 16],
    };
    for lane in 0..16 {
        let mut comp = 0_i32;
        for g in 0..8 {
            for r in 0..4 {
                let value = match (lane + g + r) % 5 {
                    0 => -128,
                    1 => -17,
                    2 => 0,
                    3 => 63,
                    _ => 127,
                };
                tile.quants[g * 64 + lane * 4 + r] = value;
                comp += i32::from(value);
            }
        }
        tile.comp[lane] = 128 * comp;
    }

    let expected = q8_0_vnni_tile16_dot_scalar(&tile, &input_block);
    let actual = unsafe { q8_0_vnni_tile16_dot_avx2(&tile, &input_block) };
    assert_eq!(actual, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "manual x86 Q8 VNNI AVX2 pair-reducer benchmark"]
fn q8_vnni_avx2_pair_reducer_benchmark() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    // SAFETY: runtime feature detection above confirms AVX2 support.
    unsafe { q8_vnni_avx2_pair_reducer_benchmark_impl() };
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_vnni_avx2_pair_reducer_benchmark_impl() {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_set1_epi32, _mm256_setr_epi32, _mm_storeu_si128,
    };
    use std::hint::black_box;

    let seed = _mm256_setr_epi32(3, -7, 11, -13, 17, -19, 23, -29);
    let iterations = 10_000_000_i32;

    let started = Instant::now();
    let mut legacy_checksum = 0_i32;
    for idx in 0..iterations {
        let acc = _mm256_add_epi32(seed, _mm256_set1_epi32(black_box(idx)));
        let lanes = q8_vnni_avx2_pair_sums_legacy_store(acc);
        legacy_checksum = legacy_checksum.wrapping_add(black_box(lanes[0]));
    }
    let legacy_us = started.elapsed().as_micros();

    let started = Instant::now();
    let mut register_checksum = 0_i32;
    for idx in 0..iterations {
        let acc = _mm256_add_epi32(seed, _mm256_set1_epi32(black_box(idx)));
        let mut lanes = [0_i32; 4];
        _mm_storeu_si128(
            lanes.as_mut_ptr().cast(),
            q8_0_vnni_avx2_pair_sums_i128(acc),
        );
        register_checksum = register_checksum.wrapping_add(black_box(lanes[0]));
    }
    let register_us = started.elapsed().as_micros();

    assert_eq!(legacy_checksum, register_checksum);
    eprintln!(
        "q8_vnni_avx2_pair_reducer_benchmark iterations={iterations} legacy_store_us={legacy_us} register_hadd_us={register_us} checksum={register_checksum}"
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx2")]
unsafe fn q8_vnni_avx2_pair_sums_legacy_store(acc: std::arch::x86_64::__m256i) -> [i32; 4] {
    use std::arch::x86_64::_mm256_storeu_si256;

    let mut pair_sums = [0_i32; 8];
    _mm256_storeu_si256(pair_sums.as_mut_ptr().cast(), acc);
    [
        pair_sums[0] + pair_sums[1],
        pair_sums[2] + pair_sums[3],
        pair_sums[4] + pair_sums[5],
        pair_sums[6] + pair_sums[7],
    ]
}

#[test]
fn q8_ffn_down_vnni_decode_falls_back_when_gate_off_or_pack_missing() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, expected) = runtime_packed_ffn_down_case();

    let gate_off = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "gate_off",
        "ffn_down",
        &ffn_down_consumer_plan(true),
    )
    .unwrap()
    .expect("rows4 fallback with VNNI gate off");
    assert_slice_close_with_tolerance(&gate_off.data, &expected.data, 5e-4);

    let vnni_on = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "pack_missing",
        "ffn_down",
        &ffn_down_vnni_decode_plan(true),
    )
    .unwrap()
    .expect("rows4 fallback when VNNI pack is unavailable or CPU-gated");
    assert_slice_close_with_tolerance(&vnni_on.data, &expected.data, 5e-4);
}

#[test]
fn q8_ffn_down_vnni_decode_records_route_denials() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();

    let _ = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "layer_3_ffn_down",
        "ffn_down",
        &ffn_down_consumer_plan(true),
    )
    .unwrap();
    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_down_vnni_decode_candidates, 1);
    assert_eq!(telemetry.ffn_down_vnni_decode_reject_gate_off, 1);
    assert!(telemetry
        .projection_route_denials
        .contains_key("ffn_down.x86_vnni_decode_consumer.gate_off"));
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mac_q8_ffn_down_decode_group_chunking_is_default_off_and_matches_consumer() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    assert!(!mac_q8_ffn_down_decode_group_chunking_enabled());

    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();
    let plan = ffn_down_consumer_plan(true);
    let unchunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "unchunked",
        "ffn_down",
        &plan,
    )
    .unwrap()
    .expect("unchunked ffn_down consumer");

    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING", "on");
    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK", "2");
    assert!(mac_q8_ffn_down_decode_group_chunking_enabled());
    assert_eq!(mac_q8_ffn_down_decode_groups_per_chunk(), 2);
    let mut chunked_plan = plan;
    chunked_plan.q8.ffn_down_decode_group_chunking =
        Q8RuntimeFlags::from_env().ffn_down_decode_group_chunking;
    assert!(chunked_plan.q8.ffn_down_decode_group_chunking);

    let chunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "chunked",
        "ffn_down",
        &chunked_plan,
    )
    .unwrap()
    .expect("chunked ffn_down consumer");

    assert_eq!(chunked.shape.dims, unchunked.shape.dims);
    assert_slice_close_with_tolerance(&chunked.data, &unchunked.data, 1e-6);
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_ffn_down_decode_group_chunking_is_default_off_and_matches_consumer() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK");
    assert!(!x86_q8_ffn_down_decode_group_chunking_enabled());

    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();
    let plan = ffn_down_consumer_plan(true);
    let unchunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "unchunked",
        "ffn_down",
        &plan,
    )
    .unwrap()
    .expect("unchunked ffn_down consumer");

    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK", "2");
    assert!(x86_q8_ffn_down_decode_group_chunking_enabled());
    assert!(Q8RuntimeFlags::from_env().ffn_down_decode_group_chunking);
    assert_eq!(q8_ffn_down_decode_groups_per_chunk(), 2);
    assert_eq!(
        q8_ffn_down_decode_consumer_route_name(true),
        "x86_decode_consumer_group_chunking"
    );

    let mut chunked_plan = plan;
    chunked_plan.q8.ffn_down_decode_group_chunking =
        Q8RuntimeFlags::from_env().ffn_down_decode_group_chunking;
    let chunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "chunked",
        "ffn_down",
        &chunked_plan,
    )
    .unwrap()
    .expect("chunked ffn_down consumer");

    assert_eq!(chunked.shape.dims, unchunked.shape.dims);
    assert_slice_close_with_tolerance(&chunked.data, &unchunked.data, 1e-6);
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK");
}

#[test]
fn q8_ffn_down_consumer_is_plan_gated_and_distinct_from_old_owner_gate() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER", "on");
    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();

    let disabled = ffn_down_consumer_plan(false);
    assert!(
        try_x86_q8_ffn_down_decode_consumer_path(
            &input,
            &packed_weight,
            "disabled",
            "ffn_down",
            &disabled,
        )
        .unwrap()
        .is_none(),
        "old owner gate must not enable the new FFN-down consumer"
    );

    let enabled = ffn_down_consumer_plan(true);
    assert!(
        try_x86_q8_ffn_down_decode_consumer_path(
            &input,
            &packed_weight,
            "wrong_role",
            "attention_output",
            &enabled,
        )
        .unwrap()
        .is_none(),
        "attention-output must not use the FFN-down consumer"
    );

    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER");
}

#[test]
fn q8_ffn_down_consumer_fails_closed_for_non_runtime_or_mismatched_storage() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();
    let plan = ffn_down_consumer_plan(true);

    let element_count = packed_weight.shape.element_count().unwrap();
    let retained_like = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_ffn_down_transposed",
        packed_weight.shape.dims.clone(),
        vec![0.0; element_count],
        vec![
            Q8_0Block {
                scale: 1.0,
                quants: [0; Q8_0_BLOCK_VALUES],
            };
            element_count / Q8_0_BLOCK_VALUES
        ],
    )
    .unwrap();
    assert!(
        try_x86_q8_ffn_down_decode_consumer_path(
            &input,
            &retained_like,
            "retained_like",
            "ffn_down",
            &plan,
        )
        .unwrap()
        .is_none(),
        "consumer must require backend-owned runtime-packed storage"
    );

    let mut mismatched = packed_weight.clone();
    if let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = mismatched.q8_0_runtime_storage.as_mut()
    {
        packed.rows += 4;
    }
    assert!(
        try_x86_q8_ffn_down_decode_consumer_path(
            &input,
            &mismatched,
            "mismatched",
            "ffn_down",
            &plan,
        )
        .unwrap()
        .is_none(),
        "consumer must fail closed when packed rows do not match output width"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mac_q8_ffn_down_single_projection_counter_probe_records_scheduler_shape() {
    // i8mm is ARMv8.6; Apple M1 (and virtualized CI runners) lack it. This test
    // executes the i8mm kernel directly, so skip when the feature is absent
    // rather than SIGILL on an illegal instruction.
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    std::env::set_var("CAMELID_MAC_Q8_FFN_DOWN_SINGLE_PROJECTION_COUNTERS", "on");
    reset_q8_schedule_telemetry();

    let (_decode_input, packed_weight_tensor, _expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight_tensor.dim(0).unwrap();
    let output_width = packed_weight_tensor.dim(1).unwrap();
    let packed_weight = match packed_weight_tensor.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected packed rows4 runtime storage, got {other:?}"),
    };
    let rows = 4;
    let input = CpuTensor::from_f32(
        "ffn_down_input",
        vec![rows, input_width],
        vec![0.25; rows * input_width],
    )
    .unwrap();

    let output = matmul_rhs_transposed_q8_0_packed_rows4_prefill_i8mm(
        &input,
        packed_weight,
        output_width,
        "ffn_down",
        "ffn_down_probe",
    )
    .unwrap();
    assert_eq!(output.shape.dims, vec![rows, output_width]);

    let telemetry = snapshot_q8_schedule_telemetry();
    let role = telemetry
        .i8mm_single_projection_by_role
        .get("ffn_down")
        .expect("ffn_down role telemetry");
    assert_eq!(role.calls, 1);
    assert_eq!(role.rows, rows as u64);
    assert_eq!(role.scheduler_chunk_calls, 1);
    assert_eq!(role.scheduler_output_groups, (output_width / 4) as u64);
    assert_eq!(role.scheduler_row_groups, 1);
    assert_eq!(role.scheduler_groups_per_chunk, 1);
}

#[test]
fn q8_ffn_down_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let output_width = packed_weight.dim(1).unwrap();
    let rows = 3;
    let input = CpuTensor::from_f32(
        "prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 9.0) * 0.125 + (idx / input_width) as f32 * 0.0625
            })
            .collect(),
    )
    .unwrap();
    let packed = match packed_weight.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed rows4 weight, got {other:?}"),
    };
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let mut expected_values = vec![0.0_f32; rows * output_width];
    for row_idx in 0..rows {
        let input_start = row_idx * input_width;
        let quantized = quantize_q8_0_row(&input.data[input_start..input_start + input_width]);
        for (group_idx, output_chunk) in expected_values
            [row_idx * output_width..(row_idx + 1) * output_width]
            .chunks_exact_mut(4)
            .enumerate()
        {
            let group_start = group_idx * blocks_per_row;
            let sums = q8_0_packed_rows4_dot(
                &packed.blocks[group_start..group_start + blocks_per_row],
                &quantized.blocks,
                Q8_0PackedRows4Interleave::I8,
            );
            output_chunk.copy_from_slice(&sums);
        }
    }
    let expected =
        CpuTensor::from_f32("expected", vec![rows, output_width], expected_values).unwrap();
    let plan = ffn_down_packed_rows4_matmul_plan(true);

    let actual = linear_for_role_runtime_with_plan(
        &input,
        &packed_weight,
        "actual",
        "ffn_down",
        &plan,
        false,
    )
    .unwrap();

    assert_eq!(actual.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

#[test]
fn q8_ffn_down_packed_rows4_matmul_is_plan_gated() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();

    assert!(try_x86_q8_ffn_down_packed_rows4_matmul_path(
        &input,
        &packed_weight,
        "disabled",
        "ffn_down",
        &ffn_down_packed_rows4_matmul_plan(false),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_ffn_down_packed_rows4_matmul_path(
        &input,
        &packed_weight,
        "wrong_role",
        "attention_output",
        &ffn_down_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
}

#[test]
#[ignore = "manual x86 Q8 scheduler tracer bullet benchmark"]
fn q8_ffn_down_gemm4_row_group_threshold_benchmark() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let input_width = Q8_0_BLOCK_VALUES * 8;
    let output_width = 1024;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|row| Q8_0Block {
            scale: 0.03125 + row as f32 * 0.000001,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(9).wrapping_sub(row as i8)),
        })
        .collect();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        TensorShape {
            dims: vec![input_width, output_width],
        },
        Q8_0PackedRows4::from_rows(
            output_width,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    let rows = 16;
    let input = CpuTensor::from_f32(
        "ffn_down_gemm4_threshold_bench_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 9.0) * 0.125 + (idx / input_width) as f32 * 0.0625
            })
            .collect(),
    )
    .unwrap();
    let Some((packed, packed_output_width)) =
        q8_0_runtime_packed_projection(&packed_weight, input_width).unwrap()
    else {
        panic!("expected runtime packed FFN-down weight")
    };
    assert_eq!(packed_output_width, output_width);

    let iterations = std::env::var("CAMELID_X86_Q8_SCHED_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(50);
    let run = |label: &str, min_groups: &str, row_group_schedule: bool| {
        std::env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            min_groups,
        );
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iterations {
            last = Some(
                q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
                    &input,
                    packed,
                    output_width,
                    label,
                    row_group_schedule,
                    false,
                    Q8PackedRows4MatmulSchedule::default(),
                )
                .unwrap(),
            );
        }
        let elapsed = started.elapsed().as_micros();
        (elapsed, last.unwrap())
    };

    let (baseline_us, baseline) = run("baseline_output_group", "8", false);
    let (old_row_group_us, old_row_group) = run("forced_row_group", "1", true);
    let (thresholded_us, thresholded) = run("thresholded_row_group", "8", true);
    assert_slice_close_with_tolerance(&old_row_group.data, &baseline.data, 5e-4);
    assert_slice_close_with_tolerance(&thresholded.data, &baseline.data, 5e-4);
    println!(
            "rows={rows} input_groups={} input_width={input_width} output_width={output_width} iterations={iterations} baseline_us={baseline_us} forced_row_group_us={old_row_group_us} thresholded_us={thresholded_us}",
            rows / 4
        );
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS");
}

#[test]
fn q8_ffn_down_gemm4_prefill_matches_runtime_packed_matmul_with_tail() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let output_width = packed_weight.dim(1).unwrap();
    let rows = 5;
    let input = CpuTensor::from_f32(
        "ffn_down_gemm4_prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 9.0) * 0.125 + (idx / input_width) as f32 * 0.0625
            })
            .collect(),
    )
    .unwrap();

    let actual = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "actual_ffn_down_gemm4_prefill",
        "ffn_down",
        &ffn_down_gemm4_prefill_plan(true),
    )
    .unwrap()
    .expect("gemm4 prefill should cover rows4 plus tail FFN-down input");
    let expected = try_x86_q8_ffn_down_packed_rows4_matmul_path(
        &input,
        &packed_weight,
        "expected_ffn_down_matmul",
        "ffn_down",
        &ffn_down_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("packed rows4 matmul should cover FFN-down prefill baseline");

    assert_eq!(actual.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

/// Lane 1: the unified tiled prefill owner must be BIT-IDENTICAL (to_bits, zero ULP) to the
/// trusted packed-rows4 matmul baseline across tile-aligned AND ragged-tail row counts, and for
/// any role under scope=All (the kernel is role-agnostic). Tighter than the 5e-4 the GEMM4 tests
/// use, because the owner is a token-identical (bit_exact) prefill drop-in, not argmax_stable.
#[cfg(target_arch = "x86_64")]
#[test]
fn q8_unified_owner_prefill_is_bit_identical_to_packed_matmul() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let output_width = packed_weight.dim(1).unwrap();

    // Test ALL THREE owner microkernels: AVX2 (v1), 4x4 VNNI (v2), 4x8 VNNI (v3). Each falls back
    // gracefully if the CPU lacks avx512vnni, and each must be byte-identical to the baseline.
    for (vnni, x8) in [("0", "0"), ("1", "0"), ("1", "1")] {
        std::env::set_var("CAMELID_X86_Q8_MATMUL_OWNER", "all");
        std::env::set_var("CAMELID_X86_Q8_MATMUL_OWNER_AVX2", "on");
        std::env::set_var("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", vnni);
        std::env::set_var("CAMELID_X86_Q8_MATMUL_OWNER_4X8", x8);
        let owner_plan = ResolvedRuntimePlan::from_env().unwrap();
        std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER");
        std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER_AVX2");
        std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER_VNNI");
        std::env::remove_var("CAMELID_X86_Q8_MATMUL_OWNER_4X8");

        // Sweep row counts: 4/8 = exact tile groups, 5/13 = ragged tail, and a couple of roles to
        // prove the dispatch is role-agnostic under scope=All.
        for &rows in &[4usize, 5, 8, 13, 16] {
            for role in ["ffn_down", "linear", "attention_k"] {
                let input = CpuTensor::from_f32(
                    "owner_prefill_input",
                    vec![rows, input_width],
                    (0..rows * input_width)
                        .map(|idx| {
                            ((idx % input_width) as f32 - 9.0) * 0.125
                                + (idx / input_width) as f32 * 0.0625
                        })
                        .collect(),
                )
                .unwrap();

                let actual = try_q8_matmul_owner_prefill(
                    &input,
                    &packed_weight,
                    "owner_actual",
                    role,
                    &owner_plan,
                )
                .unwrap()
                .unwrap_or_else(|| panic!("owner should cover role={role} rows={rows}"));
                let expected = try_x86_q8_ffn_down_packed_rows4_matmul_path(
                    &input,
                    &packed_weight,
                    "owner_expected",
                    "ffn_down",
                    &ffn_down_packed_rows4_matmul_plan(true),
                )
                .unwrap()
                .expect("packed rows4 matmul baseline");

                assert_eq!(actual.shape.dims, vec![rows, output_width]);
                for (idx, (a, b)) in actual.data.iter().zip(&expected.data).enumerate() {
                    assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "owner vs packed-matmul bit mismatch at vnni={vnni} x8={x8} role={role} rows={rows} idx={idx}: {a} vs {b}"
                );
                }
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn q8_ffn_down_amx_prefill_matches_rows4_matmul_when_supported() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if unsafe { camelid_x86_q8_amx_supported() } == 0 {
        return;
    }

    std::env::set_var("CAMELID_X86_Q8_AMX_REPACK", "on");
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let output_width = packed_weight.dim(1).unwrap();
    let packed = match packed_weight.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed rows4 weight, got {other:?}"),
    };
    assert!(
        packed.amx_blocks.is_some(),
        "explicit AMX repack gate should create AMX tile sidecar"
    );

    let rows = 20;
    let input = CpuTensor::from_f32(
        "ffn_down_amx_prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 7.0) * 0.09375 + (idx / input_width) as f32 * 0.03125
            })
            .collect(),
    )
    .unwrap();

    let mut amx_plan = ffn_down_gemm4_prefill_plan(false);
    amx_plan.q8.ffn_down_amx_prefill = true;
    let actual = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "actual_amx_prefill",
        "ffn_down",
        &amx_plan,
    )
    .unwrap()
    .expect("AMX prefill should cover rows4 FFN-down input plus scalar tail");
    let expected = try_x86_q8_ffn_down_packed_rows4_matmul_path(
        &input,
        &packed_weight,
        "expected_rows4_matmul",
        "ffn_down",
        &ffn_down_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("packed rows4 matmul should cover AMX baseline");

    assert_eq!(actual.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
    std::env::remove_var("CAMELID_X86_Q8_AMX_REPACK");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
#[ignore = "manual x86 Q8 AMX prefill tracer bullet benchmark"]
fn q8_ffn_down_amx_prefill_benchmark() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    if unsafe { camelid_x86_q8_amx_supported() } == 0 {
        println!("amx_supported=0");
        return;
    }

    std::env::set_var("CAMELID_X86_Q8_AMX_REPACK", "on");
    let input_width = Q8_0_BLOCK_VALUES * 8;
    let output_width = 1024;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|row| Q8_0Block {
            scale: 0.03125 + row as f32 * 0.000001,
            quants: std::array::from_fn(|idx| {
                (idx as i8)
                    .wrapping_mul(7)
                    .wrapping_add((row as i8).wrapping_mul(3))
            }),
        })
        .collect();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        TensorShape {
            dims: vec![input_width, output_width],
        },
        Q8_0PackedRows4::from_rows(
            output_width,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &row_blocks,
        )
        .unwrap(),
    );
    let packed = match packed_weight.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed rows4 weight, got {other:?}"),
    };
    assert!(packed.amx_blocks.is_some());

    let rows = 16;
    let input = CpuTensor::from_f32(
        "ffn_down_amx_bench_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 11.0) * 0.0625 + (idx / input_width) as f32 * 0.015625
            })
            .collect(),
    )
    .unwrap();
    let iterations = std::env::var("CAMELID_X86_Q8_SCHED_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(50);

    let bench = |mut run: Box<dyn FnMut() -> CpuTensor + '_>| {
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iterations {
            last = Some(run());
        }
        let elapsed = started.elapsed().as_micros();
        (elapsed, last.unwrap())
    };

    let (rows4_us, rows4) = bench(Box::new(|| {
        q8_0_packed_rows4_matmul_projection(
            &input,
            packed,
            output_width,
            "rows4",
            Q8PackedRows4MatmulSchedule::default(),
        )
        .unwrap()
    }));
    let (gemm4_us, gemm4) = bench(Box::new(|| {
        q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
            &input,
            packed,
            output_width,
            "gemm4",
            false,
            true,
            Q8PackedRows4MatmulSchedule::default(),
        )
        .unwrap()
    }));
    let (amx_us, amx) = bench(Box::new(|| {
        try_q8_0_packed_rows4_amx_prefill_projection(&input, packed, output_width, "amx")
            .unwrap()
            .expect("AMX path should be available")
    }));

    assert_slice_close_with_tolerance(&gemm4.data, &rows4.data, 5e-4);
    assert_slice_close_with_tolerance(&amx.data, &rows4.data, 5e-4);
    println!(
            "rows={rows} input_width={input_width} output_width={output_width} iterations={iterations} rows4_matmul_us={rows4_us} gemm4_avx2_us={gemm4_us} amx_prefill_us={amx_us}"
        );
    std::env::remove_var("CAMELID_X86_Q8_AMX_REPACK");
}

#[test]
fn q8_ffn_down_gemm4_prefill_is_plan_gated_and_rows4_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let input = CpuTensor::from_f32(
        "too_short",
        vec![3, input_width],
        vec![0.0; 3 * input_width],
    )
    .unwrap();
    let rows4_input =
        CpuTensor::from_f32("rows4", vec![4, input_width], vec![0.0; 4 * input_width]).unwrap();

    assert!(try_x86_q8_ffn_down_gemm4_prefill_path(
        &rows4_input,
        &packed_weight,
        "disabled",
        "ffn_down",
        &ffn_down_gemm4_prefill_plan(false),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "too_short",
        "ffn_down",
        &ffn_down_gemm4_prefill_plan(true),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_ffn_down_gemm4_prefill_path(
        &rows4_input,
        &packed_weight,
        "wrong_role",
        "attention_output",
        &ffn_down_gemm4_prefill_plan(true),
    )
    .unwrap()
    .is_none());
}

#[test]
fn q8_ffn_down_gemm4_avx2_matches_default_gemm4() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let rows = 8;
    let input = CpuTensor::from_f32(
        "ffn_down_gemm4_avx2_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 5.0) * 0.125 + (idx / input_width) as f32 * 0.046875
            })
            .collect(),
    )
    .unwrap();

    let default_plan = ffn_down_gemm4_prefill_plan(true);
    let expected = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "expected_default_gemm4_avx2",
        "ffn_down",
        &default_plan,
    )
    .unwrap()
    .expect("default gemm4 should cover rows4 FFN-down input");
    let mut avx2_plan = ffn_down_gemm4_prefill_plan(true);
    avx2_plan.q8.ffn_down_gemm4_avx2 = true;
    let actual = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "actual_avx2_gemm4",
        "ffn_down",
        &avx2_plan,
    )
    .unwrap()
    .expect("AVX2 gemm4 should cover rows4 FFN-down input");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

#[test]
fn q8_ffn_down_gemm4_row_group_schedule_matches_default_gemm4() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_weight, _decode_expected) = runtime_packed_ffn_down_case();
    let input_width = packed_weight.dim(0).unwrap();
    let rows = 8;
    let input = CpuTensor::from_f32(
        "ffn_down_gemm4_row_group_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 7.0) * 0.15625 + (idx / input_width) as f32 * 0.03125
            })
            .collect(),
    )
    .unwrap();

    let default_plan = ffn_down_gemm4_prefill_plan(true);
    let expected = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "expected_default_gemm4_schedule",
        "ffn_down",
        &default_plan,
    )
    .unwrap()
    .expect("default gemm4 should cover rows4 FFN-down input");
    let mut row_group_plan = ffn_down_gemm4_prefill_plan(true);
    row_group_plan.q8.ffn_down_gemm4_row_group_schedule = true;
    let actual = try_x86_q8_ffn_down_gemm4_prefill_path(
        &input,
        &packed_weight,
        "actual_row_group_gemm4_schedule",
        "ffn_down",
        &row_group_plan,
    )
    .unwrap()
    .expect("row-group gemm4 should cover rows4 FFN-down input");

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 5e-4);
}

#[test]
fn q8_ffn_down_single_owner_matches_decode_and_prefill_owners() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (decode_input, packed_weight, _expected) = runtime_packed_ffn_down_case();

    let actual_decode = try_x86_q8_ffn_down_single_owner_path(
        &decode_input,
        &packed_weight,
        "actual_decode",
        "ffn_down",
        &ffn_down_single_owner_plan(true),
    )
    .unwrap()
    .expect("single owner should cover FFN-down decode");
    let expected_decode = try_x86_q8_ffn_down_decode_consumer_path(
        &decode_input,
        &packed_weight,
        "expected_decode",
        "ffn_down",
        &ffn_down_consumer_plan(true),
    )
    .unwrap()
    .expect("decode consumer should cover FFN-down decode");
    assert_eq!(actual_decode.shape.dims, expected_decode.shape.dims);
    assert_slice_close_with_tolerance(&actual_decode.data, &expected_decode.data, 5e-4);

    let input_width = packed_weight.dim(0).unwrap();
    let rows = 3;
    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 9.0) * 0.125 + (idx / input_width) as f32 * 0.0625
            })
            .collect(),
    )
    .unwrap();
    let actual_prefill = try_x86_q8_ffn_down_single_owner_path(
        &prefill_input,
        &packed_weight,
        "actual_prefill",
        "ffn_down",
        &ffn_down_single_owner_plan(true),
    )
    .unwrap()
    .expect("single owner should cover FFN-down prefill");
    let expected_prefill = try_x86_q8_ffn_down_packed_rows4_matmul_path(
        &prefill_input,
        &packed_weight,
        "expected_prefill",
        "ffn_down",
        &ffn_down_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("packed rows4 matmul should cover FFN-down prefill");
    assert_eq!(actual_prefill.shape.dims, expected_prefill.shape.dims);
    assert_slice_close_with_tolerance(&actual_prefill.data, &expected_prefill.data, 5e-4);
}

#[test]
fn q8_ffn_down_single_owner_is_plan_gated_and_role_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_weight, _expected) = runtime_packed_ffn_down_case();

    assert!(try_x86_q8_ffn_down_single_owner_path(
        &input,
        &packed_weight,
        "disabled",
        "ffn_down",
        &ffn_down_single_owner_plan(false),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_ffn_down_single_owner_path(
        &input,
        &packed_weight,
        "wrong_role",
        "attention_output",
        &ffn_down_single_owner_plan(true),
    )
    .unwrap()
    .is_none());
}

#[test]
fn q8_ffn_gate_up_consumer_matches_runtime_packed_baseline() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (input, packed_gate, packed_up, expected) = runtime_packed_ffn_gate_up_case();

    let actual = gated_ffn_activation_with_plan(
        &input,
        &packed_gate,
        &packed_up,
        "actual",
        &ffn_gate_up_consumer_plan(true),
        false,
    )
    .unwrap();

    assert_slice_close_with_tolerance(&actual.tensor.data, &expected.tensor.data, 5e-4);
    assert!(packed_gate.q8_0_blocks.is_none());
    assert!(packed_up.q8_0_blocks.is_none());
    assert!(matches!(
        packed_gate.q8_0_runtime_storage.as_ref(),
        Some(Q8_0RuntimeStorage::PackedRows4(_))
    ));
    assert!(matches!(
        packed_up.q8_0_runtime_storage.as_ref(),
        Some(Q8_0RuntimeStorage::PackedRows4(_))
    ));
    let telemetry = snapshot_q8_schedule_telemetry();
    assert_eq!(telemetry.ffn_gate_up_decode_consumer_taken, 1);
    assert_eq!(telemetry.ffn_gate_up_decode_fused_activation_taken, 0);
    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
}

#[test]
fn q8_ffn_gate_up_consumer_is_plan_gated_and_requires_runtime_storage() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER", "on");
    let (input, packed_gate, packed_up, _expected) = runtime_packed_ffn_gate_up_case();
    let mut gate = vec![0.0; 64];
    let mut up = vec![0.0; 64];

    assert!(
        try_x86_q8_ffn_gate_up_decode_consumer_path(
            &input,
            &packed_gate,
            &packed_up,
            "layer_7_ffn_activated",
            &mut gate,
            &mut up,
            &ffn_gate_up_consumer_plan(false),
        )
        .unwrap()
        .is_none(),
        "default-off plan and old owner gate must not enter the FFN gate/up consumer"
    );

    let retained_like = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_gate",
        packed_gate.shape.dims.clone(),
        vec![0.0; packed_gate.shape.element_count().unwrap()],
        vec![
            Q8_0Block {
                scale: 1.0,
                quants: [0; Q8_0_BLOCK_VALUES],
            };
            packed_gate.shape.element_count().unwrap() / Q8_0_BLOCK_VALUES
        ],
    )
    .unwrap();
    assert!(
        try_x86_q8_ffn_gate_up_decode_consumer_path(
            &input,
            &retained_like,
            &packed_up,
            "layer_7_ffn_activated",
            &mut gate,
            &mut up,
            &ffn_gate_up_consumer_plan(true),
        )
        .unwrap()
        .is_none(),
        "consumer must require backend-owned runtime-packed storage for both gate and up"
    );

    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_ffn_gate_up_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (_decode_input, packed_gate, packed_up, _decode_expected) =
        runtime_packed_ffn_gate_up_case();
    let input_width = packed_gate.dim(0).unwrap();
    let output_width = packed_gate.dim(1).unwrap();
    let rows = 3;
    let input = CpuTensor::from_f32(
        "prefill_gate_up_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 7.0) * 0.109375
                    + (idx / input_width) as f32 * 0.046875
            })
            .collect(),
    )
    .unwrap();

    let actual = try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &input,
        &packed_gate,
        &packed_up,
        "actual",
        &ffn_gate_up_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("FFN gate/up packed-rows4 matmul should accept multi-row runtime-packed weights");

    let gate_packed = match packed_gate.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed gate weight, got {other:?}"),
    };
    let up_packed = match packed_up.q8_0_runtime_storage.as_ref() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => packed,
        other => panic!("expected runtime-packed up weight, got {other:?}"),
    };
    let mut gate = q8_0_packed_rows4_matmul_projection(
        &input,
        gate_packed,
        output_width,
        "expected_gate",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    let up = q8_0_packed_rows4_matmul_projection(
        &input,
        up_packed,
        output_width,
        "expected_up",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    for (gate_value, up_value) in gate.data.iter_mut().zip(up.data) {
        *gate_value = (*gate_value / (1.0 + (-*gate_value).exp())) * up_value;
    }

    assert_eq!(actual.tensor.name, "actual");
    assert_eq!(actual.tensor.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(&actual.tensor.data, &gate.data, 5e-4);
    assert!(packed_gate.q8_0_blocks.is_none());
    assert!(packed_up.q8_0_blocks.is_none());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_ffn_gate_up_packed_rows4_matmul_is_plan_gated_and_prefill_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (decode_input, packed_gate, packed_up, _expected) = runtime_packed_ffn_gate_up_case();
    assert!(try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &decode_input,
        &packed_gate,
        &packed_up,
        "decode_row",
        &ffn_gate_up_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());

    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, decode_input.dim(1).unwrap()],
        vec![0.0; 2 * decode_input.dim(1).unwrap()],
    )
    .unwrap();
    assert!(try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &prefill_input,
        &packed_gate,
        &packed_up,
        "disabled",
        &ffn_gate_up_packed_rows4_matmul_plan(false),
    )
    .unwrap()
    .is_none());

    let dense_up = CpuTensor::from_f32(
        "dense_up",
        packed_up.shape.dims.clone(),
        vec![0.0; packed_up.shape.element_count().unwrap()],
    )
    .unwrap();
    assert!(try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &prefill_input,
        &packed_gate,
        &dense_up,
        "dense_up",
        &ffn_gate_up_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_ffn_gate_up_prefill_route_resolver_records_route_and_denials() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();
    let (decode_input, packed_gate, packed_up, _expected) = runtime_packed_ffn_gate_up_case();
    let input_width = packed_gate.dim(0).unwrap();
    let output_width = packed_gate.dim(1).unwrap();
    let rows = 3;
    let prefill_input = CpuTensor::from_f32(
        "prefill_gate_up_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 7.0) * 0.109375
                    + (idx / input_width) as f32 * 0.046875
            })
            .collect(),
    )
    .unwrap();
    let route_name = X86Q8FfnGateUpRouteKind::PackedRows4Matmul.telemetry_name();

    let route = resolve_x86_q8_ffn_gate_up_route(
        &prefill_input,
        &packed_gate,
        &packed_up,
        &ffn_gate_up_packed_rows4_matmul_plan(true),
        X86Q8FfnGateUpRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .expect("prefill route should accept multi-row runtime-packed FFN gate/up weights");
    assert_eq!(route.rows, rows);
    assert_eq!(route.input_width, input_width);
    assert_eq!(route.output_width, output_width);

    assert!(resolve_x86_q8_ffn_gate_up_route(
        &decode_input,
        &packed_gate,
        &packed_up,
        &ffn_gate_up_packed_rows4_matmul_plan(true),
        X86Q8FfnGateUpRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());
    assert!(resolve_x86_q8_ffn_gate_up_route(
        &prefill_input,
        &packed_gate,
        &packed_up,
        &ffn_gate_up_packed_rows4_matmul_plan(false),
        X86Q8FfnGateUpRouteKind::PackedRows4Matmul,
    )
    .unwrap()
    .is_none());

    let actual = try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &prefill_input,
        &packed_gate,
        &packed_up,
        "layer_3_ffn_activated",
        &ffn_gate_up_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("prefill route should produce the fused gate/up activation");
    assert_eq!(actual.tensor.shape.dims, vec![rows, output_width]);

    let telemetry = snapshot_q8_schedule_telemetry();
    let by_route = telemetry
        .output_projection_by_route
        .get(&format!("ffn_gate_up.{route_name}"))
        .expect("FFN gate/up prefill route telemetry");
    assert_eq!(by_route.calls, 1);
    assert_eq!(by_route.rows, rows as u64);
    assert_eq!(by_route.input_width, input_width as u64);
    assert_eq!(by_route.output_width, output_width as u64);
    let layer_route = telemetry
        .output_projection_by_layer_route
        .get(&format!("layer_3.ffn_gate_up.{route_name}"))
        .expect("layer-scoped FFN gate/up prefill route telemetry");
    assert_eq!(layer_route.layer_index, 3);
    assert_eq!(layer_route.calls, 1);
    assert!(telemetry
        .projection_route_denials
        .contains_key(&format!("ffn_gate_up.{route_name}.decode_or_empty_input")));
    assert!(telemetry
        .projection_route_denials
        .contains_key(&format!("ffn_gate_up.{route_name}.plan_off")));

    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_ffn_gate_up_single_owner_matches_decode_and_prefill_owners() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (decode_input, packed_gate, packed_up, expected_decode) = runtime_packed_ffn_gate_up_case();

    let actual_decode = try_x86_q8_ffn_gate_up_single_owner_path(
        &decode_input,
        &packed_gate,
        &packed_up,
        "actual_decode",
        &ffn_gate_up_single_owner_plan(true),
    )
    .unwrap()
    .expect("single owner should cover FFN gate/up decode");
    assert_eq!(
        actual_decode.tensor.shape.dims,
        expected_decode.tensor.shape.dims
    );
    assert_slice_close_with_tolerance(
        &actual_decode.tensor.data,
        &expected_decode.tensor.data,
        5e-4,
    );

    let input_width = packed_gate.dim(0).unwrap();
    let output_width = packed_gate.dim(1).unwrap();
    let rows = 3;
    let prefill_input = CpuTensor::from_f32(
        "prefill_gate_up_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| {
                ((idx % input_width) as f32 - 7.0) * 0.109375
                    + (idx / input_width) as f32 * 0.046875
            })
            .collect(),
    )
    .unwrap();
    let actual_prefill = try_x86_q8_ffn_gate_up_single_owner_path(
        &prefill_input,
        &packed_gate,
        &packed_up,
        "actual_prefill",
        &ffn_gate_up_single_owner_plan(true),
    )
    .unwrap()
    .expect("single owner should cover FFN gate/up prefill");
    let expected_prefill = try_x86_q8_ffn_gate_up_packed_rows4_matmul_path(
        &prefill_input,
        &packed_gate,
        &packed_up,
        "expected_prefill",
        &ffn_gate_up_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .expect("packed rows4 matmul should cover FFN gate/up prefill");
    assert_eq!(actual_prefill.tensor.name, "actual_prefill");
    assert_eq!(actual_prefill.tensor.shape.dims, vec![rows, output_width]);
    assert_slice_close_with_tolerance(
        &actual_prefill.tensor.data,
        &expected_prefill.tensor.data,
        5e-4,
    );
}

#[test]
fn q8_ffn_gate_up_single_owner_is_default_off_and_requires_runtime_storage() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let (input, packed_gate, packed_up, _expected) = runtime_packed_ffn_gate_up_case();

    assert!(try_x86_q8_ffn_gate_up_single_owner_path(
        &input,
        &packed_gate,
        &packed_up,
        "disabled",
        &ffn_gate_up_single_owner_plan(false),
    )
    .unwrap()
    .is_none());

    let dense_up = CpuTensor::from_f32(
        "dense_up",
        packed_up.shape.dims.clone(),
        vec![0.0; packed_up.shape.element_count().unwrap()],
    )
    .unwrap();
    assert!(try_x86_q8_ffn_gate_up_single_owner_path(
        &input,
        &packed_gate,
        &dense_up,
        "dense_up",
        &ffn_gate_up_single_owner_plan(true),
    )
    .unwrap()
    .is_none());
}

#[test]
fn q8_0_runtime_packed_ffn_gate_up_activation_matches_retained_blocks() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let rows = 64;
    let input_width = Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.005,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(3).wrapping_add(row as i8)),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..rows)
        .map(|row| Q8_0Block {
            scale: 0.2 + row as f32 * 0.003,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(row as i8)),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| (idx as f32 - 8.0) * 0.125)
            .collect(),
    )
    .unwrap();
    let retained_gate = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_gate",
        vec![rows, input_width],
        dequantized_q8_0_rows(&gate_blocks),
        gate_blocks.clone(),
    )
    .unwrap();
    let retained_up = CpuTensor::from_f32_with_q8_0_blocks(
        "retained_up",
        vec![rows, input_width],
        dequantized_q8_0_rows(&up_blocks),
        up_blocks.clone(),
    )
    .unwrap();
    let expected =
        gated_ffn_activation(&input, &retained_gate, &retained_up, "expected", false).unwrap();

    let packed_gate = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_gate.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(rows, 1, Q8_0PackedRows4Interleave::I8, &gate_blocks).unwrap(),
    );
    let packed_up = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_up.weight",
        TensorShape {
            dims: vec![input_width, rows],
        },
        Q8_0PackedRows4::from_rows(rows, 1, Q8_0PackedRows4Interleave::I8, &up_blocks).unwrap(),
    );
    let actual = gated_ffn_activation(&input, &packed_gate, &packed_up, "actual", false).unwrap();

    assert_slice_close(&actual.tensor.data, &expected.tensor.data);
    assert!(packed_gate.q8_0_blocks.is_none());
    assert!(packed_up.q8_0_blocks.is_none());

    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn q8_0_runtime_packed_prefill_i8mm_matches_current_gemv_path() {
    // i8mm is ARMv8.6; Apple M1 (and virtualized CI runners) lack it. This test
    // executes the i8mm kernel directly, so skip when the feature is absent
    // rather than SIGILL on an illegal instruction.
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let rows = 8;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let weight_blocks: Vec<Q8_0Block> = (0..rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0625 + block_idx as f32 * 0.00390625,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 17 + idx as i32 * 5) % 59 - 29) as i8
            }),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![5, input_width],
        (0..5 * input_width)
            .map(|idx| (idx as f32 - 151.0) * 0.0078125)
            .collect(),
    )
    .unwrap();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.attn_q.weight",
        TensorShape {
            dims: vec![rows, input_width],
        },
        Q8_0PackedRows4::from_rows(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &weight_blocks,
        )
        .unwrap(),
    );

    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    let expected =
        matmul_rhs_transposed_q8_0_block_dot(&input, &packed_weight, "expected").unwrap();
    std::env::set_var("CAMELID_MAC_Q8_PREFILL_I8MM", "on");
    let actual = matmul_rhs_transposed_q8_0_block_dot(&input, &packed_weight, "actual").unwrap();

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 1.0e-3);
    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn q8_0_small_m_i8mm_kernel_matches_prefill_i8mm_kernel() {
    // i8mm is ARMv8.6; Apple M1 (and virtualized CI runners) lack it. This test
    // executes the i8mm kernels directly, so skip when the feature is absent
    // rather than SIGILL on an illegal instruction.
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let output_width = 16;
    let blocks_per_row = 3;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    // 13 rows: three packed 4-row groups through the small-M kernel plus a
    // conservative GEMV tail row, matching a speculative verify chunk shape.
    let input_rows = 13;
    let weight_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0546875 + block_idx as f32 * 0.0009765625,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 13 + idx as i32 * 7) % 63 - 31) as i8
            }),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![input_rows, input_width],
        (0..input_rows * input_width)
            .map(|idx| (idx as f32 - 311.0) * 0.0048828125)
            .collect(),
    )
    .unwrap();
    let packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.attn_q.weight",
        TensorShape {
            dims: vec![output_width, input_width],
        },
        Q8_0PackedRows4::from_rows(
            output_width,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &weight_blocks,
        )
        .unwrap(),
    );

    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    let expected =
        matmul_rhs_transposed_q8_0_block_dot(&input, &packed_weight, "expected").unwrap();

    std::env::set_var("CAMELID_MAC_Q8_PREFILL_I8MM", "on");
    // Force the input-outer prefill kernel by setting the small-M ceiling to zero rows.
    std::env::set_var("CAMELID_MAC_Q8_I8MM_SMALL_M_MAX_ROWS", "0");
    let prefill = matmul_rhs_transposed_q8_0_block_dot(&input, &packed_weight, "prefill").unwrap();
    // Default ceiling (64 rows) routes 13 rows through the weight-resident small-M kernel.
    std::env::remove_var("CAMELID_MAC_Q8_I8MM_SMALL_M_MAX_ROWS");
    let small_m = matmul_rhs_transposed_q8_0_block_dot(&input, &packed_weight, "small_m").unwrap();

    assert_eq!(small_m.shape.dims, expected.shape.dims);
    assert_eq!(prefill.shape.dims, expected.shape.dims);
    // The two i8mm kernels perform identical block arithmetic in a different
    // order-of-loops, so the full 4-row groups must agree bit-for-bit. The final
    // partial row rides the zero-padded i8mm group on the small-M path but the
    // per-row GEMV tail on the prefill path, so it is compared by tolerance only.
    let full_group_values = 12 * output_width;
    assert_eq!(
        small_m.data[..full_group_values],
        prefill.data[..full_group_values]
    );
    assert_slice_close_with_tolerance(&small_m.data, &prefill.data, 1.0e-3);
    assert_slice_close_with_tolerance(&small_m.data, &expected.data, 1.0e-3);
    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn q8_0_runtime_packed_prefill_i8mm_respects_min_row_threshold() {
    assert!(!mac_q8_prefill_i8mm_row_threshold_met(
        MAC_Q8_PREFILL_I8MM_MIN_ROWS - 1
    ));
    assert!(mac_q8_prefill_i8mm_row_threshold_met(
        MAC_Q8_PREFILL_I8MM_MIN_ROWS
    ));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn q8_0_runtime_packed_prefill_gate_up_sched_matches_unfused_path() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");

    let output_width = 8;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let input_rows = 5;
    let gate_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.046875 + block_idx as f32 * 0.001953125,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 11 + idx as i32 * 3) % 61 - 30) as i8
            }),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0390625 + block_idx as f32 * 0.0029296875,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 7 + idx as i32 * 5) % 67 - 33) as i8
            }),
        })
        .collect();
    let input = CpuTensor::from_f32(
        "input",
        vec![input_rows, input_width],
        (0..input_rows * input_width)
            .map(|idx| (idx as f32 - 123.0) * 0.0068359375)
            .collect(),
    )
    .unwrap();
    let packed_gate = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_gate.weight",
        TensorShape {
            dims: vec![input_width, output_width],
        },
        Q8_0PackedRows4::from_rows(
            output_width,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &gate_blocks,
        )
        .unwrap(),
    );
    let packed_up = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_up.weight",
        TensorShape {
            dims: vec![input_width, output_width],
        },
        Q8_0PackedRows4::from_rows(
            output_width,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &up_blocks,
        )
        .unwrap(),
    );

    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    std::env::remove_var("CAMELID_MAC_Q8_SCHED");
    let expected = gated_ffn_activation_batch(&input, &packed_gate, &packed_up, "expected")
        .unwrap()
        .tensor;
    std::env::set_var("CAMELID_MAC_Q8_PREFILL_I8MM", "on");
    std::env::set_var("CAMELID_MAC_Q8_SCHED", "packed_prefill");
    let actual = gated_ffn_activation_batch(&input, &packed_gate, &packed_up, "actual")
        .unwrap()
        .tensor;

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close_with_tolerance(&actual.data, &expected.data, 1.0e-3);
    std::env::remove_var("CAMELID_MAC_Q8_SCHED");
    std::env::remove_var("CAMELID_MAC_Q8_PREFILL_I8MM");
    std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
}

#[test]
fn q8_0_file_reader_quantized_input_buffer_reuses_capacity() {
    // The retained-scratch limit is process-global environment state. Serialize
    // against the bounded-retention test so its 128-byte cap cannot clear this
    // thread-local allocation between the two capacity observations.
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let first = CpuTensor::from_f32(
        "first",
        vec![2, Q8_0_BLOCK_VALUES],
        (0..2 * Q8_0_BLOCK_VALUES)
            .map(|idx| idx as f32 - 17.0)
            .collect(),
    )
    .unwrap();
    let second = CpuTensor::from_f32(
        "second",
        vec![1, Q8_0_BLOCK_VALUES],
        (0..Q8_0_BLOCK_VALUES).map(|idx| idx as f32).collect(),
    )
    .unwrap();

    let retained_capacity = with_q8_0_file_reader_quantized_inputs(|blocks| {
        *blocks = Vec::new();

        {
            let quantized = quantize_q8_0_rows_into(&first, Q8_0_BLOCK_VALUES, blocks)?;
            assert_eq!(quantized.rows().len(), 2);
            assert_eq!(quantized.row(0)[0].quants[0], -127);
        }
        let retained_capacity = blocks.capacity();

        {
            let quantized = quantize_q8_0_rows_into(&second, Q8_0_BLOCK_VALUES, blocks)?;
            assert_eq!(quantized.rows().len(), 1);
            assert_eq!(quantized.row(0)[0].quants[0], 0);
        }

        assert_eq!(blocks.capacity(), retained_capacity);
        Ok(blocks.capacity())
    })
    .unwrap();

    with_q8_0_file_reader_quantized_inputs(|blocks| {
        assert!(blocks.is_empty());
        assert_eq!(blocks.capacity(), retained_capacity);
        Ok(())
    })
    .unwrap();
}

#[test]
fn q8_0_file_reader_scratch_retention_is_bounded() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES", "128");

    with_q8_0_file_reader_row_chunk(512, |row_chunk| {
        row_chunk.fill(7);
        Ok(())
    })
    .unwrap();
    let (row_capacity, _, _, _) = q8_0_file_reader_scratch_capacities();
    assert!(
        row_capacity <= 128,
        "row scratch capacity should be capped after an oversized use, got {row_capacity}"
    );

    with_q8_0_file_reader_chunk_scales(256, |scales| {
        scales.fill(3.0);
        Ok(())
    })
    .unwrap();
    let (_, scale_capacity, _, _) = q8_0_file_reader_scratch_capacities();
    assert!(
            scale_capacity * mem::size_of::<f32>() <= 128,
            "scale scratch capacity should be capped after an oversized use, got {scale_capacity} entries"
        );

    with_q8_0_file_reader_output_chunk(256, |output_chunk| {
        output_chunk.fill(5.0);
        Ok(())
    })
    .unwrap();
    let (_, _, _, output_capacity) = q8_0_file_reader_scratch_capacities();
    assert!(
            output_capacity * mem::size_of::<f32>() <= 128,
            "output scratch capacity should be capped after an oversized use, got {output_capacity} entries"
        );

    with_q8_0_file_reader_quantized_inputs(|blocks| {
        blocks.resize(
            32,
            Q8_0Block {
                scale: 1.0,
                quants: [0; Q8_0_BLOCK_VALUES],
            },
        );
        Ok(())
    })
    .unwrap();
    let (_, _, quantized_capacity, _) = q8_0_file_reader_scratch_capacities();
    assert!(
            quantized_capacity * mem::size_of::<Q8_0Block>() <= 128,
            "quantized-input scratch capacity should be capped after an oversized use, got {quantized_capacity} entries"
        );

    std::env::remove_var("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES");
}

#[test]
fn q8_0_block_reader_linear_matches_existing_q8_path() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "off");
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let row0 = Q8_0Block {
        scale: 0.5,
        quants: std::array::from_fn(|idx| idx as i8 - 16),
    };
    let row1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
    };
    for block in [&row0, &row1] {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
        temp_file.write_all(&bytes).unwrap();
    }
    temp_file.flush().unwrap();

    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

    let mut dequantized_weight = Vec::with_capacity(64);
    for block in [&row0, &row1] {
        dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
    }
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![2, 32],
        dequantized_weight,
        vec![row0.clone(), row1.clone()],
    )
    .unwrap();

    let expected = matmul_rhs_transposed_with_precision(&input, &weight, "expected").unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 2);
    let reader = Q8BlockReader::new(0, 2);
    let actual =
        matmul_rhs_transposed_q8_0_block_reader(&input, &backing, reader, 2, "actual").unwrap();

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close(&actual.data, &expected.data);
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "on");
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "off");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "off");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let rows: Vec<Q8_0Block> = (0..5)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
            quants: std::array::from_fn(|idx| idx as i8 - 12 + row as i8),
        })
        .collect();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
        temp_file.write_all(&bytes).unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..64)
        .map(|idx| idx as f32 * 0.25 - 4.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![2, 32], input_values).unwrap();
    let mut dequantized_weight = Vec::with_capacity(rows.len() * 32);
    for block in &rows {
        dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
    }
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows.len(), 32],
        dequantized_weight,
        rows,
    )
    .unwrap();
    let expected = matmul_rhs_transposed_with_precision(&input, &weight, "expected").unwrap();

    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 5);
    let reader = Q8BlockReader::new(0, 5);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let actual = pool
        .install(|| {
            assert!(should_parallelize_linear_output(5));
            matmul_rhs_transposed_q8_0_block_reader(&input, &backing, reader, 5, "actual")
        })
        .unwrap();

    assert_eq!(actual.shape.dims, expected.shape.dims);
    assert_slice_close(&actual.data, &expected.data);
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn q8_0_file_reader_parallelizes_wide_outputs_by_default() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    pool.install(|| {
        assert!(!should_parallelize_q8_0_file_reader_output(1023));
        assert!(should_parallelize_q8_0_file_reader_output(1024));
    });
}

#[test]
fn q8_0_file_reader_parallel_respects_explicit_linear_off() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_PARALLEL_LINEAR", "off");
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "1");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    pool.install(|| assert!(!should_parallelize_q8_0_file_reader_output(14336)));
}

#[test]
fn q8_0_file_reader_parallel_uses_existing_linear_threshold_env() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", "2048");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    pool.install(|| {
        assert!(!should_parallelize_q8_0_file_reader_output(2047));
        assert!(should_parallelize_q8_0_file_reader_output(2048));
    });
}

#[test]
fn q8_0_encoded_row_matches_decoded_scale_helper() {
    let row = Q8_0Block {
        scale: f16_bits_to_f32(f32_to_f16_bits(0.375)),
        quants: std::array::from_fn(|idx| idx as i8 - 12),
    };
    let input = QuantizedQ8_0Row {
        blocks: PooledQ8Blocks(vec![Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.25)),
            quants: std::array::from_fn(|idx| 15 - idx as i8),
        }]),
    };
    let mut row_bytes = Vec::with_capacity(Q8BlockReader::BLOCK_SIZE_BYTES);
    row_bytes.extend_from_slice(&f32_to_f16_bits(row.scale).to_le_bytes());
    row_bytes.extend(row.quants.iter().map(|q| *q as u8));
    let mut scales = vec![0.0; 1];
    decode_q8_0_encoded_row_scales(&row_bytes, &mut scales);

    let direct = dot_q8_0_encoded_row(&input.blocks, &row_bytes);
    let decoded = dot_q8_0_encoded_row_with_scales(&input.blocks, &row_bytes, &scales);

    assert!((direct - decoded).abs() < 1e-6);
}

#[test]
fn q8_0_file_backed_accumulate_matches_q8_block_dot_across_chunks() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let rows: Vec<Q8_0Block> = (0..5)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
            quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
        })
        .collect();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
        temp_file.write_all(&bytes).unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..32)
        .map(|idx| idx as f32 * 0.5 - 3.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();
    let mut dequantized_weight = Vec::with_capacity(rows.len() * 32);
    for block in &rows {
        dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
    }
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows.len(), 32],
        dequantized_weight,
        rows.clone(),
    )
    .unwrap();
    let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();

    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
    let mut actual = vec![0.0; rows.len()];
    let start = q8_0_file_read_stats();
    accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
        .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_slice_close(&actual, &expected.data);
    assert_eq!(reads.read_calls, 3);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
}

#[test]
fn q8_0_file_backed_accumulate_coalesces_exact_two_chunk_tensor_read() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let rows: Vec<Q8_0Block> = (0..4)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.25 + row as f32 * 0.125)),
            quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..32)
        .map(|idx| idx as f32 * 0.5 - 3.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows.len(), 32],
        dequantized_q8_0_rows(&rows),
        rows.clone(),
    )
    .unwrap();
    let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
    let mut actual = vec![0.0; rows.len()];
    let start = q8_0_file_read_stats();

    accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
        .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_slice_close(&actual, &expected.data);
    assert_eq!(reads.read_calls, 1);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
}

#[test]
fn q8_0_file_backed_accumulate_can_use_quantized_input_block_dot() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");

    let rows: Vec<Q8_0Block> = (0..3)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.125 + row as f32 * 0.0625)),
            quants: std::array::from_fn(|idx| idx as i8 - 9 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..32)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.37)
        .collect::<Vec<_>>();
    let quantized_input = quantize_q8_0_row(&input_values);
    let expected = rows
        .iter()
        .map(|row| q8_0_dot_rows(std::slice::from_ref(row), &quantized_input.blocks))
        .collect::<Vec<_>>();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
    let mut actual = vec![0.0; rows.len()];

    accumulate_transposed_linear_row_q8_0_file_reader(&input_values, &backing, &mut actual)
        .unwrap();

    assert_slice_close(&actual, &expected);
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn q8_0_file_backed_accumulate_rejects_unaligned_input_width() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 1);
    let input = vec![0.0_f32; Q8_0_BLOCK_VALUES + 1];
    let mut output = vec![0.0_f32; 1];

    let err = accumulate_transposed_linear_row_q8_0_file_reader(&input, &backing, &mut output)
        .unwrap_err()
        .to_string();

    assert!(err.contains("not a multiple of 32"));
}

#[test]
fn q8_0_file_backed_batch_matmul_reuses_chunk_reads_across_input_rows() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let rows: Vec<Q8_0Block> = (0..5)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.03125,
            quants: std::array::from_fn(|idx| idx as i8 - 8 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for row in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..96)
        .map(|idx| idx as f32 * 0.1 - 3.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![3, 32], input_values).unwrap();
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows.len(), 32],
        dequantized_q8_0_rows(&rows),
        rows.clone(),
    )
    .unwrap();
    let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());
    let start = q8_0_file_read_stats();

    let actual = matmul_rhs_transposed_q8_0_block_reader(
        &input,
        &backing,
        Q8BlockReader::new(0, rows.len()),
        rows.len(),
        "actual",
    )
    .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_slice_close(&actual.data, &expected.data);
    assert_eq!(reads.read_calls, 3);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
}

#[test]
fn q8_0_file_backed_batch_matmul_can_use_quantized_input_block_dot() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");

    let rows: Vec<Q8_0Block> = (0..4)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.1875 + row as f32 * 0.03125)),
            quants: std::array::from_fn(|idx| idx as i8 - 11 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..64)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.21)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![2, 32], input_values.clone()).unwrap();
    let mut expected = Vec::new();
    for input_row in input_values.chunks_exact(32) {
        let quantized_input = quantize_q8_0_row(input_row);
        expected.extend(
            rows.iter()
                .map(|row| q8_0_dot_rows(std::slice::from_ref(row), &quantized_input.blocks)),
        );
    }
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());

    let actual = matmul_rhs_transposed_q8_0_block_reader(
        &input,
        &backing,
        Q8BlockReader::new(0, rows.len()),
        rows.len(),
        "actual",
    )
    .unwrap();

    assert_slice_close(&actual.data, &expected);
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn q8_0_file_backed_batch_matmul_reuses_cached_chunks_across_calls() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "1024");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let rows: Vec<Q8_0Block> = (0..5)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.03125,
            quants: std::array::from_fn(|idx| idx as i8 - 7 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for row in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..96)
        .map(|idx| idx as f32 * 0.075 - 2.5)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("input", vec![3, 32], input_values).unwrap();
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![rows.len(), 32],
        dequantized_q8_0_rows(&rows),
        rows.clone(),
    )
    .unwrap();
    let expected = matmul_rhs_transposed_q8_0_block_dot(&input, &weight, "expected").unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len());

    let start = q8_0_file_read_stats();
    let first = matmul_rhs_transposed_q8_0_block_reader(
        &input,
        &backing,
        Q8BlockReader::new(0, rows.len()),
        rows.len(),
        "first",
    )
    .unwrap();
    let after_first = q8_0_file_read_stats();
    let first_reads = after_first.saturating_delta_since(start);

    let second = matmul_rhs_transposed_q8_0_block_reader(
        &input,
        &backing,
        Q8BlockReader::new(0, rows.len()),
        rows.len(),
        "second",
    )
    .unwrap();
    let second_reads = q8_0_file_read_stats().saturating_delta_since(after_first);

    assert_slice_close(&first.data, &expected.data);
    assert_slice_close(&second.data, &expected.data);
    assert_eq!(first_reads.read_calls, 3);
    assert_eq!(
        first_reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    assert_eq!(second_reads.read_calls, 0);
    assert_eq!(second_reads.read_bytes, 0);
    assert_eq!(second_reads.cache_hits, 3);
    assert_eq!(
        second_reads.cache_hit_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
}

#[test]
fn q8_0_file_backed_borrowed_batch_matmul_reuses_chunk_reads_across_input_rows() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    std::env::set_var(
        "CAMELID_Q8_0_FILE_READER_CHUNK_BYTES",
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2).to_string(),
    );

    let rows: Vec<Q8_0Block> = (0..5)
        .map(|row| Q8_0Block {
            scale: 0.125 + row as f32 * 0.03125,
            quants: std::array::from_fn(|idx| idx as i8 - 9 + row as i8),
        })
        .collect();
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for row in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(row.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&row.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let input_values = (0..96)
        .map(|idx| idx as f32 * 0.05 - 2.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("output_norm_batch", vec![3, 32], input_values).unwrap();
    let expected_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "expected.weight",
        vec![rows.len(), 32],
        dequantized_q8_0_rows(&rows),
        rows.clone(),
    )
    .unwrap();
    let expected =
        matmul_rhs_transposed_q8_0_block_dot(&input, &expected_weight, "expected").unwrap();
    let output_weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        crate::tensor::TensorShape {
            dims: vec![32, rows.len()],
        },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len()),
    );
    let start = q8_0_file_read_stats();

    let actual = output_projection_runtime(&input, &output_weight, "actual", false).unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert_eq!(actual.shape.dims, vec![3, 5]);
    assert_slice_close(&actual.data, &expected.data);
    assert_eq!(reads.read_calls, 3);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * rows.len()) as u64
    );
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_CHUNK_BYTES");
}

#[test]
fn q8_0_file_backing_cache_reuses_exact_chunk_reads() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
    let _ = q8_0_file_read_stats();
    std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "1024");

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&[1_u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    temp_file.flush().unwrap();
    let backing = Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 1);
    let start = q8_0_file_read_stats();
    let mut first = [0_u8; 4];
    let mut second = [0_u8; 4];

    backing.read_exact_at_cached(&mut first, 2).unwrap();
    let after_first = q8_0_file_read_stats().saturating_delta_since(start);
    backing.read_exact_at_cached(&mut second, 2).unwrap();
    let after_second = q8_0_file_read_stats().saturating_delta_since(start);

    assert_eq!(first, [3, 4, 5, 6]);
    assert_eq!(second, first);
    assert_eq!(after_first.read_calls, 1);
    assert_eq!(after_first.read_bytes, 4);
    assert_eq!(after_first.cache_hits, 0);
    assert_eq!(after_first.cache_entries, 1);
    assert_eq!(after_first.cache_bytes, 4);
    assert_eq!(after_first.cache_capacity_bytes, 1024);
    assert_eq!(after_second.read_calls, after_first.read_calls);
    assert_eq!(after_second.read_bytes, after_first.read_bytes);
    assert_eq!(after_second.cache_hits, 1);
    assert_eq!(after_second.cache_entries, 1);
    assert_eq!(after_second.cache_bytes, 4);
    assert_eq!(after_second.cache_capacity_bytes, 1024);
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
}

#[test]
fn q8_0_block_dot_uses_raw_weight_blocks_and_quantized_input_when_opted_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

    let row0 = Q8_0Block {
        scale: 0.5,
        quants: std::array::from_fn(|idx| idx as i8 - 16),
    };
    let row1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 7 } else { -9 }),
    };
    let mut dequantized_weight = Vec::with_capacity(64);
    for block in [&row0, &row1] {
        dequantized_weight.extend(block.quants.iter().map(|q| block.scale * f32::from(*q)));
    }
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![2, 32],
        dequantized_weight,
        vec![row0.clone(), row1.clone()],
    )
    .unwrap();

    let actual = matmul_rhs_transposed_with_precision(&input, &weight, "out").unwrap();

    assert_eq!(actual.shape.dims, vec![1, 2]);
    assert_slice_close(
        &actual.data,
        &[
            expected_q8_0_block_dot(&input_values, &row0),
            expected_q8_0_block_dot(&input_values, &row1),
        ],
    );
}

#[test]
fn rectangular_shape_reinterpretation_preserves_q8_0_blocks_for_transposed_dot() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let block = Q8_0Block {
        scale: 1.0,
        quants: [0; 32],
    };
    let weight = CpuTensor::from_f32_with_q8_0_blocks(
        "weight",
        vec![32, 64],
        vec![0.0; 2048],
        vec![block; 64],
    )
    .unwrap();

    let reinterpreted = weight_with_swapped_matrix_shape(&weight);

    assert_eq!(reinterpreted.shape.dims, vec![64, 32]);
    assert_eq!(reinterpreted.source_type, Some(GgufTensorType::Q8_0));
    assert!(reinterpreted.q8_0_blocks.is_some());
    assert!(should_use_q8_0_block_dot(&reinterpreted, 32));
}

#[test]
fn q8_0_block_dot_reads_descriptor_shaped_blocks_as_transposed_rows_when_opted_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let input = CpuTensor::from_f32("input", vec![1, 32], input_values.clone()).unwrap();

    let row0 = Q8_0Block {
        scale: 0.125,
        quants: std::array::from_fn(|idx| (idx % 5) as i8 - 2),
    };
    let row1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 3 } else { -1 }),
    };
    let descriptor_shaped_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "descriptor.weight",
        vec![32, 2],
        dequantized_q8_0_rows(&[row0.clone(), row1.clone()]),
        vec![row0.clone(), row1.clone()],
    )
    .unwrap();

    let actual = linear_with_diagnostic_layouts(
        &input,
        &descriptor_shaped_weight,
        "out",
        SquareLinearLayout::Transposed,
        RectangularLinearLayout::Auto,
    )
    .unwrap();

    assert_eq!(actual.shape.dims, vec![1, 2]);
    assert_slice_close(
        &actual.data,
        &[
            expected_q8_0_block_dot(&input_values, &row0),
            expected_q8_0_block_dot(&input_values, &row1),
        ],
    );
}

#[test]
fn output_projection_q8_0_descriptor_shape_uses_storage_token_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values.clone()).unwrap();

    let token_0 = Q8_0Block {
        scale: 0.125,
        quants: std::array::from_fn(|idx| (idx % 7) as i8 - 3),
    };
    let token_1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 5 } else { -4 }),
    };
    let output_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "output.weight",
        vec![32, 2],
        dequantized_q8_0_rows(&[token_0.clone(), token_1.clone()]),
        vec![token_0.clone(), token_1.clone()],
    )
    .unwrap();

    let runtime =
        output_projection_runtime(&input, &output_weight, "runtime_logits", false).unwrap();
    let token_major = output_projection_with_layout(
        &input,
        &output_weight,
        "token_major_logits",
        OutputProjectionLayout::TokenMajor,
    )
    .unwrap();
    let descriptor = output_projection_with_layout(
        &input,
        &output_weight,
        "descriptor_logits",
        OutputProjectionLayout::Descriptor,
    )
    .unwrap();
    let expected = [
        expected_q8_0_block_dot(&input_values, &token_0),
        expected_q8_0_block_dot(&input_values, &token_1),
    ];

    assert_eq!(runtime.shape.dims, vec![1, 2]);
    assert_eq!(token_major.shape.dims, vec![1, 2]);
    assert_slice_close(&runtime.data, &expected);
    assert_slice_close(&token_major.data, &expected);
    assert!(
        descriptor
            .data
            .iter()
            .zip(expected.iter())
            .any(|(actual, expected)| (actual - expected).abs() > 1e-3),
        "descriptor-column interpretation should not alias token-major Q8_0 storage rows"
    );
}

#[test]
fn gated_ffn_activation_uses_q8_0_descriptor_blocks_for_gate_and_up_when_opted_in() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");

    let mut input_values = Vec::with_capacity(32);
    input_values.push(127.0);
    input_values.extend((1..32).map(|idx| idx as f32 - 17.0));
    let input = CpuTensor::from_f32("ffn_norm", vec![1, 32], input_values.clone()).unwrap();

    let gate0 = Q8_0Block {
        scale: 0.0625,
        quants: std::array::from_fn(|idx| (idx % 7) as i8 - 3),
    };
    let gate1 = Q8_0Block {
        scale: 0.03125,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 4 } else { -2 }),
    };
    let up0 = Q8_0Block {
        scale: 0.125,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 1 } else { -3 }),
    };
    let up1 = Q8_0Block {
        scale: 0.25,
        quants: std::array::from_fn(|idx| (idx % 5) as i8 - 2),
    };
    let gate_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "blk.0.ffn_gate.weight",
        vec![32, 2],
        dequantized_q8_0_rows(&[gate0.clone(), gate1.clone()]),
        vec![gate0.clone(), gate1.clone()],
    )
    .unwrap();
    let up_weight = CpuTensor::from_f32_with_q8_0_blocks(
        "blk.0.ffn_up.weight",
        vec![32, 2],
        dequantized_q8_0_rows(&[up0.clone(), up1.clone()]),
        vec![up0.clone(), up1.clone()],
    )
    .unwrap();

    let actual = gated_ffn_activation(&input, &gate_weight, &up_weight, "ffn", false)
        .unwrap()
        .tensor;

    let expected_gate = [
        expected_q8_0_block_dot(&input_values, &gate0),
        expected_q8_0_block_dot(&input_values, &gate1),
    ];
    let expected_up = [
        expected_q8_0_block_dot(&input_values, &up0),
        expected_q8_0_block_dot(&input_values, &up1),
    ];
    let expected = [
        silu(expected_gate[0]) * expected_up[0],
        silu(expected_gate[1]) * expected_up[1],
    ];

    assert_eq!(actual.shape.dims, vec![1, 2]);
    assert_slice_close(&actual.data, &expected);
}

#[test]
fn q8_0_horizontal_sum_matches_linear_int_sum() {
    let weight = std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(111));
    let input = std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(97));
    let linear_sum: i32 = weight
        .iter()
        .zip(input.iter())
        .map(|(w, x)| i32::from(*w) * i32::from(*x))
        .sum();

    assert_eq!(
        q8_0_block_int_dot_horizontal_sum(&weight, &input),
        linear_sum
    );
}

#[test]
fn q8_0_encoded_horizontal_sum_matches_linear_int_sum() {
    let weight: [i8; Q8_0_BLOCK_VALUES] =
        std::array::from_fn(|idx| (idx as i8).wrapping_mul(7).wrapping_sub(111));
    let input: [i8; Q8_0_BLOCK_VALUES] =
        std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_add(97));
    let encoded_weight = weight.map(|quant| quant as u8);
    let linear_sum: i32 = weight
        .iter()
        .zip(input.iter())
        .map(|(w, x)| i32::from(*w) * i32::from(*x))
        .sum();

    assert_eq!(
        q8_0_block_int_dot_horizontal_sum_encoded(&encoded_weight, &input),
        linear_sum
    );
}

fn expected_q8_0_block_dot(input_values: &[f32], weight: &Q8_0Block) -> f32 {
    // The input vector deliberately contains a 127.0 max-absolute value, so Camelid's
    // Q8_0 activation quantizer uses an exactly representable scale of 1.0 and preserves
    // these integer samples as their Q8 quants. That keeps the expected dot independent
    // from the production quantization helper while still exercising the block-dot path.
    input_values
        .iter()
        .zip(weight.quants.iter())
        .map(|(input, weight_quant)| input * f32::from(*weight_quant) * weight.scale)
        .sum()
}

fn dequantized_q8_0_rows(rows: &[Q8_0Block]) -> Vec<f32> {
    rows.iter()
        .flat_map(|block| block.quants.iter().map(|q| block.scale * f32::from(*q)))
        .collect()
}

#[test]
fn applies_rope_to_each_attention_head() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ROPE_PAIRING");
    std::env::remove_var("CAMELID_ROPE_DIRECTION");
    std::env::remove_var("CAMELID_ROPE_POSITION_MODE");

    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let rotated = apply_rope(&tensor, 1, 2, &config, None, 0, "query_rope").unwrap();

    let (sin, cos) = 1.0_f32.sin_cos();
    assert_eq!(rotated.shape.dims, vec![1, 4]);
    assert_close(rotated.data[0], cos);
    assert_close(rotated.data[1], sin);
    assert_close(rotated.data[2], -sin);
    assert_close(rotated.data[3], cos);
}

#[test]
fn apply_rope_uses_configured_frequency_base() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 8192,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(4),
        rope_freq_base: Some(500_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-5,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 1, 1, &config, None, 0, "query_rope").unwrap();
    let diagnostic =
        rope_diagnostics(&tensor, &rotated, 1, 1, &config, None, "attention_q").unwrap();

    let theta_500k = 500_000.0_f32.powf(-0.5);
    let (sin_500k, cos_500k) = theta_500k.sin_cos();
    let theta_10k = 10_000.0_f32.powf(-0.5);
    let (sin_10k, _) = theta_10k.sin_cos();

    assert_eq!(rotated.shape.dims, vec![1, 4]);
    assert_close(rotated.data[2], cos_500k);
    assert_close(rotated.data[3], sin_500k);
    assert!(
            (rotated.data[3] - sin_10k).abs() > 1e-3,
            "RoPE rotation unexpectedly matched the TinyLlama 10000 fallback instead of GGUF freq_base=500000"
        );
    assert_eq!(diagnostic.freq_base, 500_000.0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn apply_rope_uses_llama3_frequency_scaling_metadata() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 32,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(4),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: Some("llama3".to_string()),
        rope_scaling_factor: Some(8.0),
        rope_scaling_original_context_length: Some(16),
        rope_scaling_low_freq_factor: Some(1.0),
        rope_scaling_high_freq_factor: Some(4.0),
        rms_norm_epsilon: 1e-5,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 8, 1, &config, None, 0, "query_rope").unwrap();
    let diagnostic =
        rope_diagnostics(&tensor, &rotated, 8, 1, &config, None, "attention_q").unwrap();

    let base_theta = 10_000.0_f32.powf(-0.5);
    let scaled_theta = base_theta / 8.0;
    let (scaled_sin, scaled_cos) = (8.0 * scaled_theta).sin_cos();
    let (unscaled_sin, _) = (8.0 * base_theta).sin_cos();

    assert_eq!(rotated.shape.dims, vec![1, 4]);
    assert_close(rotated.data[2], scaled_cos);
    assert_close(rotated.data[3], scaled_sin);
    assert!(
        (rotated.data[3] - unscaled_sin).abs() > 1e-2,
        "RoPE rotation unexpectedly ignored llama3 scaling metadata"
    );
    assert_eq!(diagnostic.scaling_type, "llama3");
    assert_eq!(diagnostic.scaling_factor, 8.0);
    assert_eq!(diagnostic.scaling_original_context_length, Some(16));
    assert_eq!(diagnostic.scaling_low_freq_factor, Some(1.0));
    assert_eq!(diagnostic.scaling_high_freq_factor, Some(4.0));
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn apply_rope_uses_gguf_rope_frequency_factors() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 32,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(4),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-5,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let rope_freqs = CpuTensor::from_f32("rope_freqs.weight", vec![2], vec![1.0, 4.0]).unwrap();

    let rotated = apply_rope(&tensor, 8, 1, &config, Some(&rope_freqs), 0, "query_rope").unwrap();
    let diagnostic = rope_diagnostics(
        &tensor,
        &rotated,
        8,
        1,
        &config,
        Some(&rope_freqs),
        "attention_q",
    )
    .unwrap();

    let derived_theta = 10_000.0_f32.powf(-0.5);
    let factor_theta = derived_theta / 4.0;
    let (factor_sin, factor_cos) = (8.0_f32 * factor_theta).sin_cos();
    let (derived_sin, _) = (8.0_f32 * derived_theta).sin_cos();

    assert_close(rotated.data[2], factor_cos);
    assert_close(rotated.data[3], factor_sin);
    assert!(
        (rotated.data[3] - derived_sin).abs() > 0.05,
        "RoPE rotation unexpectedly ignored rope_freqs.weight factors"
    );
    assert_eq!(diagnostic.frequency_source, "rope_freqs.weight");
    assert_eq!(diagnostic.rope_freqs_count, Some(2));
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn rope_diagnostics_reconstruct_reported_rotation() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ROPE_PAIRING");
    std::env::remove_var("CAMELID_ROPE_DIRECTION");
    std::env::remove_var("CAMELID_ROPE_POSITION_MODE");

    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let reported = apply_rope(&tensor, 1, 2, &config, None, 0, "query_rope").unwrap();

    let diagnostic =
        rope_diagnostics(&tensor, &reported, 1, 2, &config, None, "attention_q").unwrap();

    assert_eq!(diagnostic.role, "attention_q");
    assert_eq!(diagnostic.pairing, "adjacent_even_odd");
    assert_eq!(diagnostic.direction, "forward");
    assert_eq!(diagnostic.position_mode, "zero_based");
    assert_eq!(diagnostic.position, 1);
    assert_eq!(diagnostic.effective_position, 1);
    assert_eq!(diagnostic.head_count, 2);
    assert_eq!(diagnostic.head_dim, 2);
    assert_eq!(diagnostic.rope_dim, 2);
    assert_eq!(diagnostic.input_first_values, tensor.data);
    assert_eq!(diagnostic.reported_first_values, reported.data);
    assert_eq!(diagnostic.reconstructed_first_values, reported.data);
    assert_eq!(diagnostic.reported_max_abs_index, 1);
    assert_close(diagnostic.reported_max_abs, reported.data[1]);
    assert_eq!(diagnostic.reported_max_abs_window_start, 0);
    assert_eq!(diagnostic.reported_max_abs_window, reported.data);
    assert_eq!(
        diagnostic.reconstructed_reported_max_abs_window,
        reported.data
    );
    assert_eq!(diagnostic.max_abs_delta_index, 0);
    assert!(diagnostic.max_abs_delta < 1e-7);
}

#[test]
fn zero_delta_selector_accepts_all_none_and_layer_lists() {
    assert!(diagnostic_zero_delta_value("TEST_ZERO", "all", 7).unwrap());
    assert!(diagnostic_zero_delta_value("TEST_ZERO", "true", 7).unwrap());
    assert!(!diagnostic_zero_delta_value("TEST_ZERO", "none", 7).unwrap());
    assert!(!diagnostic_zero_delta_value("TEST_ZERO", "", 7).unwrap());
    assert!(diagnostic_zero_delta_value("TEST_ZERO", "1, 7, 9", 7).unwrap());
    assert!(!diagnostic_zero_delta_value("TEST_ZERO", "1, 2, 9", 7).unwrap());
    assert!(diagnostic_zero_delta_value("TEST_ZERO", "oops", 7).is_err());
}

#[test]
fn split_half_rope_pairing_is_available_for_diagnostics() {
    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(4),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let head_dim = 4;
    let rope_dim = 4;
    let freq_base = config.rope_freq_base.unwrap();

    let adjacent = apply_rope_with_pairing(
        &tensor,
        RopeParams {
            position: 1,
            head_count: 1,
            head_dim,
            rope_dim,
            freq_base,
            pairing: RopePairing::AdjacentEvenOdd,
            direction: RopeDirection::Forward,
            position_mode: RopePositionMode::ZeroBased,
            scaling: no_rope_scaling(),
            rope_freqs: None,
            rope_offset: 0,
        },
        "adjacent",
    )
    .unwrap();
    let split = apply_rope_with_pairing(
        &tensor,
        RopeParams {
            position: 1,
            head_count: 1,
            head_dim,
            rope_dim,
            freq_base,
            pairing: RopePairing::SplitHalf,
            direction: RopeDirection::Forward,
            position_mode: RopePositionMode::ZeroBased,
            scaling: no_rope_scaling(),
            rope_freqs: None,
            rope_offset: 0,
        },
        "split",
    )
    .unwrap();

    let (sin, cos) = 1.0_f32.sin_cos();
    assert_eq!(adjacent.shape.dims, vec![1, 4]);
    assert_eq!(split.shape.dims, vec![1, 4]);
    assert_close(adjacent.data[0], cos);
    assert_close(adjacent.data[1], sin);
    assert_close(adjacent.data[2], 0.0);
    assert_close(adjacent.data[3], 0.0);
    assert_close(split.data[0], cos);
    assert_close(split.data[1], 0.0);
    assert_close(split.data[2], sin);
    assert_close(split.data[3], 0.0);
}

#[test]
fn inverse_rope_direction_is_available_for_diagnostics() {
    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
    let head_dim = 2;
    let rope_dim = 2;
    let freq_base = config.rope_freq_base.unwrap();

    let forward = apply_rope_with_pairing(
        &tensor,
        RopeParams {
            position: 1,
            head_count: 1,
            head_dim,
            rope_dim,
            freq_base,
            pairing: RopePairing::AdjacentEvenOdd,
            direction: RopeDirection::Forward,
            position_mode: RopePositionMode::ZeroBased,
            scaling: no_rope_scaling(),
            rope_freqs: None,
            rope_offset: 0,
        },
        "forward",
    )
    .unwrap();
    let inverse = apply_rope_with_pairing(
        &tensor,
        RopeParams {
            position: 1,
            head_count: 1,
            head_dim,
            rope_dim,
            freq_base,
            pairing: RopePairing::AdjacentEvenOdd,
            direction: RopeDirection::Inverse,
            position_mode: RopePositionMode::ZeroBased,
            scaling: no_rope_scaling(),
            rope_freqs: None,
            rope_offset: 0,
        },
        "inverse",
    )
    .unwrap();

    let (sin, cos) = 1.0_f32.sin_cos();
    assert_eq!(forward.shape.dims, vec![1, 2]);
    assert_eq!(inverse.shape.dims, vec![1, 2]);
    assert_close(forward.data[0], cos);
    assert_close(forward.data[1], sin);
    assert_close(inverse.data[0], cos);
    assert_close(inverse.data[1], -sin);
}

#[test]
fn one_based_rope_position_mode_is_available_for_diagnostics() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_ROPE_POSITION_MODE", "one_based");

    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 8,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 0, 1, &config, None, 0, "query_rope").unwrap();
    let diagnostic =
        rope_diagnostics(&tensor, &rotated, 0, 1, &config, None, "attention_q").unwrap();

    let (sin, cos) = 1.0_f32.sin_cos();
    assert_close(rotated.data[0], cos);
    assert_close(rotated.data[1], sin);
    assert_eq!(diagnostic.position_mode, "one_based");
    assert_eq!(diagnostic.position, 0);
    assert_eq!(diagnostic.effective_position, 1);
    assert!(diagnostic.max_abs_delta < 1e-7);

    std::env::set_var("CAMELID_ROPE_POSITION_MODE", "diagonal");
    assert!(diagnostic_rope_position_mode().is_err());
    std::env::remove_var("CAMELID_ROPE_POSITION_MODE");
}

#[test]
fn tied_output_projection_uses_token_major_embedding_layout() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let embedding = CpuTensor::from_f32(
        "token_embd.weight",
        vec![3, 2],
        vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            2.0, 3.0, // token 2
        ],
    )
    .unwrap();
    let hidden = CpuTensor::from_f32("hidden", vec![1, 2], vec![2.0, 3.0]).unwrap();

    let logits = linear(&hidden, &embedding, "logits").unwrap();

    assert_eq!(logits.shape.dims, vec![1, 3]);
    assert_close(logits.data[0], 2.0);
    assert_close(logits.data[1], 3.0);
    assert_close(logits.data[2], 13.0);
}

#[test]
fn output_projection_diagnostics_reconstruct_tied_output_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

    let output_norm = CpuTensor::from_f32("output_norm", vec![1, 3], vec![2.0, -1.0, 0.5]).unwrap();
    let tied_output = CpuTensor::from_f32(
        "token_embd.weight",
        vec![4, 3],
        vec![
            0.5, 1.0, -2.0, // token 0
            -1.0, 0.25, 0.75, // token 1
            2.0, -0.5, 1.5, // token 2
            0.0, 3.0, -1.0, // token 3
        ],
    )
    .unwrap();
    let logits = output_projection_with_layout(
        &output_norm,
        &tied_output,
        "logits",
        OutputProjectionLayout::Descriptor,
    )
    .unwrap();

    let diagnostics =
        output_projection_diagnostics(&output_norm, &tied_output, &logits, &[2], None, None, None)
            .unwrap();

    assert_eq!(logits.shape.dims, vec![1, 4]);
    assert_close(logits.data[2], 5.25);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].token_id, 2);
    assert_eq!(diagnostics[0].layout, "output_input");
    assert_close(diagnostics[0].reported_logit, 5.25);
    assert_close(diagnostics[0].reconstructed_logit, 5.25);
    assert_close(diagnostics[0].absolute_delta, 0.0);
    assert_eq!(diagnostics[0].output_row_first_values, vec![2.0, -0.5, 1.5]);
    assert_eq!(
        diagnostics[0].component_products_first_values,
        vec![4.0, 0.5, 0.75]
    );
}

#[test]
fn token_major_output_projection_diagnostic_reinterprets_descriptor_shape() {
    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let descriptor_weight = CpuTensor::from_f32(
        "output.weight",
        vec![2, 3],
        vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            2.0, 3.0, // token 2
        ],
    )
    .unwrap();

    let logits = output_projection_with_layout(
        &input,
        &descriptor_weight,
        "logits",
        OutputProjectionLayout::TokenMajor,
    )
    .unwrap();
    assert_eq!(logits.shape.dims, vec![1, 3]);
    assert_close(logits.data[0], 2.0);
    assert_close(logits.data[1], 3.0);
    assert_close(logits.data[2], 13.0);
}

#[test]
fn output_projection_defaults_to_token_major_runtime_layout() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    assert_eq!(
        diagnostic_output_projection_layout().unwrap(),
        OutputProjectionLayout::TokenMajor
    );

    let input = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let token_major_weight = CpuTensor::from_f32(
        "output.weight",
        vec![2, 3],
        vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            2.0, 3.0, // token 2
        ],
    )
    .unwrap();

    let logits = output_projection_runtime(&input, &token_major_weight, "logits", false).unwrap();
    assert_eq!(logits.shape.dims, vec![1, 3]);
    assert_close(logits.data[0], 2.0);
    assert_close(logits.data[1], 3.0);
    assert_close(logits.data[2], 13.0);
}

#[test]
fn output_projection_diagnostics_support_q8_0_file_backed_token_major_rows() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

    let input_values = (0..32)
        .map(|idx| idx as f32 * 0.25 - 2.0)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values.clone()).unwrap();
    let row0 = Q8_0Block {
        scale: 0.125,
        quants: std::array::from_fn(|idx| idx as i8 - 8),
    };
    let row1 = Q8_0Block {
        scale: 0.0625,
        quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 6 } else { -5 }),
    };
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for block in [&row0, &row1] {
        use std::io::Write;
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        let bytes = block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
        temp_file.write_all(&bytes).unwrap();
    }
    temp_file.flush().unwrap();

    let output_weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        crate::tensor::TensorShape { dims: vec![32, 2] },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, 2),
    );

    let logits = output_projection_runtime(&input, &output_weight, "logits", false).unwrap();
    let read_start = q8_0_file_read_stats();
    let diagnostics =
        output_projection_diagnostics(&input, &output_weight, &logits, &[0, 1], None, None, None)
            .unwrap();
    let reads = q8_0_file_read_stats().saturating_delta_since(read_start);

    assert_eq!(diagnostics.len(), 2);
    assert_close(diagnostics[0].reconstructed_logit, logits.data[0]);
    assert_close(diagnostics[1].reconstructed_logit, logits.data[1]);
    assert_close(
        diagnostics[0].q8_direct_reconstructed_logit.unwrap(),
        logits.data[0],
    );
    assert_close(
        diagnostics[1].q8_direct_reconstructed_logit.unwrap(),
        logits.data[1],
    );
    assert_eq!(diagnostics[0].q8_direct_absolute_delta, Some(0.0));
    assert_eq!(diagnostics[1].q8_direct_absolute_delta, Some(0.0));
    assert!(diagnostics[0]
        .q8_direct_decoded_component_delta
        .is_some_and(|delta| delta.is_finite()));
    assert_eq!(reads.read_calls, 2);
    assert_eq!(
        reads.read_bytes,
        (Q8BlockReader::BLOCK_SIZE_BYTES * 2) as u64
    );
}

#[test]
fn output_projection_diagnostics_support_runtime_packed_tied_output_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let output_norm_values = (0..32)
        .map(|idx| idx as f32 * 0.125 - 1.5)
        .collect::<Vec<_>>();
    let output_norm =
        CpuTensor::from_f32("output_norm", vec![1, 32], output_norm_values.clone()).unwrap();
    let row_blocks = vec![
        Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| idx as i8 - 8),
        },
        Q8_0Block {
            scale: 0.0625,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 6 } else { -5 }),
        },
        Q8_0Block {
            scale: 0.09375,
            quants: std::array::from_fn(|idx| (idx as i8 % 9) - 4),
        },
        Q8_0Block {
            scale: 0.15625,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 7 } else { -3 }),
        },
    ];
    let packed =
        Q8_0PackedRows4::from_rows(4, 1, Q8_0PackedRows4Interleave::I8, &row_blocks).unwrap();
    let output_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape { dims: vec![32, 4] },
        packed,
    );

    let logits = row_blocks
        .iter()
        .map(|block| {
            output_norm_values
                .iter()
                .zip(block.quants.iter())
                .map(|(input, quant)| *input * block.scale * f32::from(*quant))
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let logits = CpuTensor::from_f32("logits", vec![1, 4], logits).unwrap();

    let diagnostics = output_projection_diagnostics(
        &output_norm,
        &output_weight,
        &logits,
        &[0, 1],
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(diagnostics.len(), 2);
    for (idx, diagnostic) in diagnostics.iter().enumerate() {
        assert_eq!(diagnostic.token_id as usize, idx);
        assert_eq!(diagnostic.layout, "token_major");
        assert_close(diagnostic.reconstructed_logit, diagnostic.reported_logit);
        assert_close(
            diagnostic.decoded_component_reconstructed_logit,
            diagnostic.reported_logit,
        );
        assert_eq!(diagnostic.q8_direct_reconstructed_logit, None);
        assert_eq!(diagnostic.q8_direct_absolute_delta, None);
        assert_eq!(diagnostic.q8_direct_decoded_component_delta, None);
    }
}

/// The CPU repack execution plan retains the output projection as loader-packed rows
/// (`q8_0_packed_rows4_4x8`) or plain retained blocks (`q8_0_blocks`) with no dense
/// values, no file backing, and no runtime storage. Dense diagnostics used to 503 on
/// these ("... with 0 values") â€” both storages must now decode token rows.
#[test]
fn output_projection_diagnostics_support_loader_packed_and_retained_block_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let output_norm_values = (0..32)
        .map(|idx| idx as f32 * 0.125 - 1.5)
        .collect::<Vec<_>>();
    let output_norm =
        CpuTensor::from_f32("output_norm", vec![1, 32], output_norm_values.clone()).unwrap();
    let row_blocks = vec![
        Q8_0Block {
            scale: 0.125,
            quants: std::array::from_fn(|idx| idx as i8 - 8),
        },
        Q8_0Block {
            scale: 0.0625,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(2) { 6 } else { -5 }),
        },
        Q8_0Block {
            scale: 0.09375,
            quants: std::array::from_fn(|idx| (idx as i8 % 9) - 4),
        },
        Q8_0Block {
            scale: 0.15625,
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 7 } else { -3 }),
        },
    ];
    let logits = row_blocks
        .iter()
        .map(|block| {
            output_norm_values
                .iter()
                .zip(block.quants.iter())
                .map(|(input, quant)| *input * block.scale * f32::from(*quant))
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let logits = CpuTensor::from_f32("logits", vec![1, 4], logits).unwrap();

    // Loader-packed direct field (no runtime storage).
    let packed =
        Q8_0PackedRows4::from_rows(4, 1, Q8_0PackedRows4Interleave::I8, &row_blocks).unwrap();
    let mut packed_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape { dims: vec![32, 4] },
        packed,
    );
    packed_weight.q8_0_packed_rows4_4x8 = match packed_weight.q8_0_runtime_storage.take() {
        Some(Q8_0RuntimeStorage::PackedRows4(packed)) => Some(packed),
        other => panic!("expected packed rows4 runtime storage, got {other:?}"),
    };

    // Plain retained blocks only.
    let mut blocks_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape { dims: vec![32, 4] },
        Q8_0PackedRows4::from_rows(4, 1, Q8_0PackedRows4Interleave::I8, &row_blocks).unwrap(),
    );
    blocks_weight.q8_0_runtime_storage = None;
    blocks_weight.q8_0_blocks = Some(row_blocks.clone());

    for (label, output_weight) in [
        ("loader_packed", &packed_weight),
        ("blocks", &blocks_weight),
    ] {
        assert!(
            output_weight.data.is_empty(),
            "{label} must have no dense values"
        );
        let diagnostics = output_projection_diagnostics(
            &output_norm,
            output_weight,
            &logits,
            &[0, 1],
            None,
            None,
            None,
        )
        .unwrap_or_else(|err| panic!("{label} diagnostics failed: {err}"));
        assert_eq!(diagnostics.len(), 2, "{label}");
        for (idx, diagnostic) in diagnostics.iter().enumerate() {
            assert_eq!(diagnostic.token_id as usize, idx, "{label}");
            assert_eq!(diagnostic.layout, "token_major", "{label}");
            assert_close(diagnostic.reconstructed_logit, diagnostic.reported_logit);
            assert_close(
                diagnostic.decoded_component_reconstructed_logit,
                diagnostic.reported_logit,
            );
        }
    }
}

/// The zero-clone borrowed block-dot path must be bitwise-identical to the
/// cloning path it replaces (`linear_weight_reinterpreted_as_transposed` +
/// tensor block-dot), for both weight forms the swap can present: retained
/// blocks only, and runtime packed rows4 that survives the orientation swap.
#[test]
fn borrowed_q8_0_block_dot_matches_cloned_swapped_tensor_path() {
    let _env_guard = env_lock();
    let input_width = 64usize; // 2 Q8_0 blocks per weight row
    let output_width = 8usize;
    let blocks_per_row = input_width / Q8_0_BLOCK_VALUES;
    let weight_blocks: Vec<Q8_0Block> = (0..output_width * blocks_per_row)
        .map(|row| Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.03 + row as f32 * 0.011)),
            quants: std::array::from_fn(|idx| ((row * 31 + idx * 7) % 251) as i8),
        })
        .collect();
    let input_data: Vec<f32> = (0..input_width)
        .map(|i| (((i % 89) as f32) - 44.0) * 0.017)
        .collect();
    let input =
        CpuTensor::from_f32("borrowed_dot_probe", vec![1, input_width], input_data).unwrap();

    // GGUF orientation [input_width, output_width]: rows == input width, so the
    // linear chain reinterprets by swapping the matrix shape.
    let make_weight = || {
        CpuTensor::q8_0_runtime_packed_rows4_linear(
            "blk.0.ffn_down.weight",
            TensorShape {
                dims: vec![input_width, output_width],
            },
            Q8_0PackedRows4::from_rows(
                output_width,
                blocks_per_row,
                Q8_0PackedRows4Interleave::I8,
                &weight_blocks,
            )
            .unwrap(),
        )
    };
    // Form 1: retained blocks only. Form 2: runtime packed rows4 that matches
    // the swapped orientation (kept across the swap by both paths).
    let mut blocks_weight = make_weight();
    blocks_weight.q8_0_runtime_storage = None;
    blocks_weight.q8_0_blocks = Some(weight_blocks.clone());
    let packed_weight = make_weight();

    let runtime_plan = ResolvedRuntimePlan::from_env().expect("plan");
    for (label, weight) in [
        ("blocks", &blocks_weight),
        ("runtime_packed", &packed_weight),
    ] {
        let cloned =
            linear_weight_reinterpreted_as_transposed(weight, input_width).expect("reinterpret");
        assert!(
            should_use_q8_0_block_dot_with_plan(&cloned, input_width, &runtime_plan),
            "{label}: cloned path must route to block-dot for this probe"
        );
        let expected = matmul_rhs_transposed_q8_0_block_dot_with_plan(
            &input,
            &cloned,
            "borrowed_dot_out",
            &runtime_plan,
        )
        .expect("cloned kernel");

        let borrowed =
            borrowed_linear_weight_as_transposed(weight, input_width).expect("borrowed view");
        assert!(
            should_use_borrowed_q8_0_block_dot_with_plan(borrowed, input_width, &runtime_plan),
            "{label}: borrowed predicate must match the cloned predicate"
        );
        let actual = matmul_rhs_transposed_q8_0_block_dot_borrowed_with_plan(
            &input,
            borrowed,
            "borrowed_dot_out",
            &runtime_plan,
        )
        .expect("borrowed kernel");

        assert_eq!(expected.shape.dims, actual.shape.dims, "{label}");
        for (idx, (a, b)) in expected.data.iter().zip(actual.data.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: divergence at output {idx}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn output_projection_diagnostics_reject_q8_0_file_backed_unaligned_rows_before_read() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

    let output_norm = CpuTensor::from_f32("output_norm", vec![1, 33], vec![0.0; 33]).unwrap();
    let logits = CpuTensor::from_f32("logits", vec![1, 1], vec![0.0]).unwrap();
    let output_weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        crate::tensor::TensorShape { dims: vec![33, 1] },
        Q8_0FileBacking::new("unused-q8-output.gguf".into(), 0, 1),
    );

    let start = q8_0_file_read_stats();
    let err = output_projection_diagnostics(
        &output_norm,
        &output_weight,
        &logits,
        &[0],
        None,
        None,
        None,
    )
    .unwrap_err()
    .to_string();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert!(err.contains("hidden width 33 is not block aligned"));
    assert_eq!(reads.read_calls, 0);
    assert_eq!(reads.read_bytes, 0);
}

#[test]
fn output_projection_diagnostics_reject_q8_0_file_backing_block_mismatch_before_read() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

    let output_norm = CpuTensor::from_f32("output_norm", vec![1, 32], vec![0.0; 32]).unwrap();
    let logits = CpuTensor::from_f32("logits", vec![1, 2], vec![0.0, 0.0]).unwrap();
    let output_weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        crate::tensor::TensorShape { dims: vec![32, 2] },
        Q8_0FileBacking::new("unused-q8-output.gguf".into(), 0, 1),
    );

    let start = q8_0_file_read_stats();
    let err = output_projection_diagnostics(
        &output_norm,
        &output_weight,
        &logits,
        &[1],
        None,
        None,
        None,
    )
    .unwrap_err()
    .to_string();
    let reads = q8_0_file_read_stats().saturating_delta_since(start);

    assert!(err.contains("expected 2 blocks"));
    assert!(err.contains("got 1"));
    assert_eq!(reads.read_calls, 0);
    assert_eq!(reads.read_bytes, 0);
}

#[test]
fn output_projection_diagnostics_match_q8_0_file_backed_block_dot_probe() {
    let _env_guard = env_lock();
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_Q8_0_BLOCK_DOT", "on");
    std::env::set_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT", "on");
    std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");

    let input_values = (0..32)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.17)
        .collect::<Vec<_>>();
    let input = CpuTensor::from_f32("output_norm", vec![1, 32], input_values).unwrap();
    let rows = [
        Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.15625)),
            quants: std::array::from_fn(|idx| idx as i8 - 10),
        },
        Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.09375)),
            quants: std::array::from_fn(|idx| if idx.is_multiple_of(3) { 7 } else { -4 }),
        },
    ];
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    for block in &rows {
        temp_file
            .write_all(&f32_to_f16_bits(block.scale).to_le_bytes())
            .unwrap();
        temp_file
            .write_all(&block.quants.iter().map(|q| *q as u8).collect::<Vec<_>>())
            .unwrap();
    }
    temp_file.flush().unwrap();

    let output_weight = CpuTensor::q8_0_file_backed_linear(
        "output.weight",
        crate::tensor::TensorShape { dims: vec![32, 2] },
        Q8_0FileBacking::new(temp_file.path().to_path_buf(), 0, rows.len()),
    );

    let logits = output_projection_runtime(&input, &output_weight, "logits", false).unwrap();
    let diagnostics =
        output_projection_diagnostics(&input, &output_weight, &logits, &[0, 1], None, None, None)
            .unwrap();

    assert_eq!(diagnostics.len(), 2);
    assert_close(
        diagnostics[0].q8_direct_reconstructed_logit.unwrap(),
        logits.data[0],
    );
    assert_close(
        diagnostics[1].q8_direct_reconstructed_logit.unwrap(),
        logits.data[1],
    );
    assert_eq!(diagnostics[0].q8_direct_absolute_delta, Some(0.0));
    assert_eq!(diagnostics[1].q8_direct_absolute_delta, Some(0.0));
    assert!(diagnostics[0]
        .q8_direct_decoded_component_delta
        .is_some_and(|delta| delta.is_finite()));
    std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
    std::env::remove_var("CAMELID_Q8_0_FILE_READER_BLOCK_DOT");
}

#[test]
fn output_projection_runtime_ignores_diagnostic_layout_env_without_dense_collection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

    let input = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let token_major_weight = CpuTensor::from_f32(
        "output.weight",
        vec![2, 3],
        vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            2.0, 3.0, // token 2
        ],
    )
    .unwrap();

    let runtime_logits =
        output_projection_runtime(&input, &token_major_weight, "runtime_logits", false).unwrap();
    let diagnostic_logits =
        output_projection_runtime(&input, &token_major_weight, "diagnostic_logits", true).unwrap();

    assert_eq!(runtime_logits.shape.dims, vec![1, 3]);
    assert_close(runtime_logits.data[0], 2.0);
    assert_close(runtime_logits.data[1], 3.0);
    assert_close(runtime_logits.data[2], 13.0);
    assert_eq!(diagnostic_logits.shape.dims, vec![1, 3]);
    assert_close(diagnostic_logits.data[0], 5.0);
    assert_close(diagnostic_logits.data[1], 6.0);
    assert_close(diagnostic_logits.data[2], 9.0);
}

#[test]
fn q8_packed_rows4_matmul_projection_chunked_prefill_matches_manual_output() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let rows = 5;
    let output_rows = 128;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..output_rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.03125 + (block_idx % 17) as f32 * 0.001953125,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 7 + idx as i32 * 11) % 71 - 35) as i8
            }),
        })
        .collect();
    let packed = Q8_0PackedRows4::from_rows(
        output_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let input = CpuTensor::from_f32(
        "chunked_prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.078125)
            .collect(),
    )
    .unwrap();
    let quantized_inputs = q8_0_quantized_matmul_input_rows(&input, blocks_per_row).unwrap();

    let actual = q8_0_packed_rows4_matmul_projection_from_quantized(
        rows,
        &packed,
        output_rows,
        "actual_chunked_prefill",
        &quantized_inputs,
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();

    let mut expected = Vec::with_capacity(rows * output_rows);
    for row_idx in 0..rows {
        let input_start = row_idx * blocks_per_row;
        let quantized_row = &quantized_inputs[input_start..input_start + blocks_per_row];
        for group_blocks in packed.blocks.chunks_exact(blocks_per_row) {
            expected.extend_from_slice(&q8_0_packed_rows4_dot(
                group_blocks,
                quantized_row,
                Q8_0PackedRows4Interleave::I8,
            ));
        }
    }

    assert_eq!(actual.shape.dims, vec![rows, output_rows]);
    assert_eq!(actual.data, expected);
}

#[test]
fn q8_packed_rows4_gate_up_fused_prefill_matches_separate_pair_activation() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let rows = 6;
    let output_rows = 128;
    let blocks_per_row = 3;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let gate_blocks: Vec<Q8_0Block> = (0..output_rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0234375 + (block_idx % 19) as f32 * 0.001953125,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 5 + idx as i32 * 7) % 83 - 41) as i8
            }),
        })
        .collect();
    let up_blocks: Vec<Q8_0Block> = (0..output_rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.03125 + (block_idx % 17) as f32 * 0.001953125,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 11 + idx as i32 * 3) % 79 - 39) as i8
            }),
        })
        .collect();
    let gate_packed = Q8_0PackedRows4::from_rows(
        output_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &gate_blocks,
    )
    .unwrap();
    let up_packed = Q8_0PackedRows4::from_rows(
        output_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &up_blocks,
    )
    .unwrap();
    let input = CpuTensor::from_f32(
        "gate_up_fused_prefill_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.0390625)
            .collect(),
    )
    .unwrap();
    let quantized_inputs = q8_0_quantized_matmul_input_rows(&input, blocks_per_row).unwrap();
    let (mut gate, up) = q8_0_packed_rows4_matmul_projection_pair_from_quantized(
        rows,
        &gate_packed,
        &up_packed,
        output_rows,
        output_rows,
        "gate",
        "up",
        &quantized_inputs,
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();
    for (gate_value, up_value) in gate.data.iter_mut().zip(up.data) {
        *gate_value = apply_ffn_gate_up_order(*gate_value, up_value, FfnGateUpOrder::GateUp);
    }

    let fused = q8_0_packed_rows4_matmul_projection_pair_activated_from_quantized(
        rows,
        &gate_packed,
        &up_packed,
        output_rows,
        "fused",
        FfnGateUpOrder::GateUp,
        &quantized_inputs,
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();

    assert_eq!(fused.shape.dims, vec![rows, output_rows]);
    assert_eq!(fused.data, gate.data);
}

#[test]
fn q8_packed_rows4_parallel_input_quantize_matches_serial() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let rows = 11;
    let blocks_per_row = 3;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let input = CpuTensor::from_f32(
        "parallel_quantize_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0546875)
            .collect(),
    )
    .unwrap();

    std::env::remove_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE");
    let serial = q8_0_quantized_matmul_input_rows(&input, blocks_per_row).unwrap();
    std::env::set_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "on");
    let parallel = q8_0_quantized_matmul_input_rows(&input, blocks_per_row).unwrap();
    std::env::remove_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE");

    assert_eq!(parallel, serial);
}

#[test]
fn q8_packed_rows4_matmul_quantized_input_scratch_matches_owned_rows() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let rows = 4;
    let blocks_per_row = 3;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let input = CpuTensor::from_f32(
        "scratch_quantized_input",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.0625)
            .collect(),
    )
    .unwrap();
    let owned = q8_0_quantized_matmul_input_rows(&input, blocks_per_row).unwrap();

    let scratch = with_q8_0_quantized_matmul_input_rows(
        &input,
        blocks_per_row,
        |scratch_rows, quantized_inputs| {
            assert_eq!(scratch_rows, rows);
            Ok(quantized_inputs.to_vec())
        },
    )
    .unwrap();

    assert_eq!(scratch, owned);
    let (_, _, quantized_capacity, _) = q8_0_file_reader_scratch_capacities();
    assert!(quantized_capacity >= rows * blocks_per_row);
}

#[test]
fn x86_q8_output_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let rows = 3;
    let vocab_rows = 8;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..vocab_rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0625 + (block_idx % 13) as f32 * 0.00390625,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 11 + idx as i32 * 5) % 67 - 33) as i8
            }),
        })
        .collect();
    let packed = Q8_0PackedRows4::from_rows(
        vocab_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let output_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape {
            dims: vec![input_width, vocab_rows],
        },
        packed.clone(),
    );
    let input = CpuTensor::from_f32(
        "output_prefill_hidden",
        vec![rows, input_width],
        (0..rows * input_width)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.109375)
            .collect(),
    )
    .unwrap();
    let plan = output_packed_rows4_matmul_plan(true);

    let actual = output_projection_runtime_with_plan(
        &input,
        &output_weight,
        "output_prefill_logits",
        &plan,
        false,
    )
    .unwrap();
    let expected = q8_0_packed_rows4_matmul_projection(
        &input,
        &packed,
        vocab_rows,
        "expected_output_prefill_logits",
        Q8PackedRows4MatmulSchedule::default(),
    )
    .unwrap();

    assert_eq!(actual.shape.dims, vec![rows, vocab_rows]);
    assert_eq!(actual.data, expected.data);
}

#[test]
fn x86_q8_output_packed_rows4_matmul_is_plan_gated_and_prefill_limited() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let vocab_rows = 8;
    let blocks_per_row = 1;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let packed = Q8_0PackedRows4::from_rows(
        vocab_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &vec![
            Q8_0Block {
                scale: 0.125,
                quants: [3; Q8_0_BLOCK_VALUES],
            };
            vocab_rows * blocks_per_row
        ],
    )
    .unwrap();
    let output_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape {
            dims: vec![input_width, vocab_rows],
        },
        packed,
    );
    let prefill_input = CpuTensor::from_f32(
        "prefill_input",
        vec![2, input_width],
        vec![0.25; 2 * input_width],
    )
    .unwrap();
    let decode_input = CpuTensor::from_f32(
        "decode_input",
        vec![1, input_width],
        vec![0.25; input_width],
    )
    .unwrap();

    assert!(try_x86_q8_output_packed_rows4_matmul_path(
        &prefill_input,
        &output_weight,
        "disabled",
        &output_packed_rows4_matmul_plan(false),
    )
    .unwrap()
    .is_none());
    assert!(try_x86_q8_output_packed_rows4_matmul_path(
        &decode_input,
        &output_weight,
        "decode_limited",
        &output_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
    let non_output_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "blk.0.ffn_down.weight",
        output_weight.shape.clone(),
        match output_weight.q8_0_runtime_storage.as_ref().unwrap() {
            Q8_0RuntimeStorage::PackedRows4(packed) => packed.clone(),
        },
    );
    assert!(try_x86_q8_output_packed_rows4_matmul_path(
        &prefill_input,
        &non_output_weight,
        "non_output",
        &output_packed_rows4_matmul_plan(true),
    )
    .unwrap()
    .is_none());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_output_decode_owner_path_uses_runtime_packed_storage() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_X86_Q8_REPACK", "on");
    std::env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
    std::env::set_var(Q8_SCHEDULE_TELEMETRY_ENV, "on");
    reset_q8_schedule_telemetry();

    let vocab_rows = 8;
    let input_width = Q8_0_BLOCK_VALUES * 2;
    let row_blocks: Vec<Q8_0Block> = (0..vocab_rows * 2)
        .map(|row| Q8_0Block {
            scale: 0.1 + row as f32 * 0.004,
            quants: std::array::from_fn(|idx| (idx as i8).wrapping_mul(5).wrapping_sub(row as i8)),
        })
        .collect();
    let packed = Q8_0PackedRows4::from_rows(
        vocab_rows,
        input_width / Q8_0_BLOCK_VALUES,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let output_weight = CpuTensor::q8_0_runtime_packed_rows4_linear(
        "output.weight",
        TensorShape {
            dims: vec![input_width, vocab_rows],
        },
        packed.clone(),
    );
    let input = CpuTensor::from_f32(
        "output_norm",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.25)
            .collect(),
    )
    .unwrap();

    let logits = output_projection_runtime(&input, &output_weight, "logits", false).unwrap();

    assert_eq!(logits.shape.dims, vec![1, vocab_rows]);
    let quantized_input = quantize_q8_0_row(&input.data);
    let mut expected = Vec::new();
    for group_blocks in packed.blocks.chunks_exact(packed.blocks_per_row) {
        expected.extend_from_slice(&q8_0_packed_rows4_dot(
            group_blocks,
            &quantized_input.blocks,
            Q8_0PackedRows4Interleave::I8,
        ));
    }
    assert_eq!(logits.data, expected);
    let telemetry = snapshot_q8_schedule_telemetry();
    let route = telemetry
        .output_projection_by_route
        .get("logits.x86_output_decode_owner")
        .expect("output decode-owner route telemetry");
    assert_eq!(route.calls, 1);
    assert_eq!(route.rows, 1);
    assert_eq!(route.input_width, input_width as u64);
    assert_eq!(route.output_width, vocab_rows as u64);

    reset_q8_schedule_telemetry();
    std::env::remove_var(Q8_SCHEDULE_TELEMETRY_ENV);
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER");
    std::env::remove_var("CAMELID_X86_Q8_REPACK");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_packed_rows4_decode_projection_matches_manual_wide_output() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let output_rows = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    let blocks_per_row = 2;
    let input_width = blocks_per_row * Q8_0_BLOCK_VALUES;
    let row_blocks: Vec<Q8_0Block> = (0..output_rows * blocks_per_row)
        .map(|block_idx| Q8_0Block {
            scale: 0.0625 + (block_idx % 11) as f32 * 0.00390625,
            quants: std::array::from_fn(|idx| {
                ((block_idx as i32 * 13 + idx as i32 * 7) % 61 - 30) as i8
            }),
        })
        .collect();
    let packed = Q8_0PackedRows4::from_rows(
        output_rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        &row_blocks,
    )
    .unwrap();
    let input = CpuTensor::from_f32(
        "wide_decode_input",
        vec![1, input_width],
        (0..input_width)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.125)
            .collect(),
    )
    .unwrap();
    let quantized_input = quantize_q8_0_row(&input.data);

    let actual = q8_0_packed_rows4_single_input_projection(
        &packed,
        &quantized_input.blocks,
        output_rows,
        "actual_wide_decode",
    )
    .unwrap();

    let mut expected = Vec::with_capacity(output_rows);
    for group_blocks in packed.blocks.chunks_exact(blocks_per_row) {
        expected.extend_from_slice(&q8_0_packed_rows4_dot(
            group_blocks,
            &quantized_input.blocks,
            Q8_0PackedRows4Interleave::I8,
        ));
    }
    assert_eq!(actual.shape.dims, vec![1, output_rows]);
    assert_eq!(actual.data, expected);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_q8_packed_rows4_serial_decode_gate_disables_decode_parallelism() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE");

    let output_rows = X86_Q8_PACKED_ROWS4_DECODE_PARALLEL_MIN_OUTPUTS;
    if rayon::current_num_threads() > 1 {
        assert!(should_parallelize_x86_q8_packed_rows4_decode_output(
            output_rows
        ));
    }

    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE", "on");
    assert!(!should_parallelize_x86_q8_packed_rows4_decode_output(
        output_rows
    ));
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE");
}

#[test]
fn output_projection_diagnostics_reconstruct_selected_logits() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");

    let output_norm = CpuTensor::from_f32("output_norm", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let output_weight = CpuTensor::from_f32(
        "output.weight",
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0, // hidden dim 0 to tokens 0..2
            4.0, 5.0, 6.0, // hidden dim 1 to tokens 0..2
        ],
    )
    .unwrap();
    let logits = output_projection_with_layout(
        &output_norm,
        &output_weight,
        "logits",
        OutputProjectionLayout::Descriptor,
    )
    .unwrap();

    let final_hidden = CpuTensor::from_f32("final_hidden", vec![1, 2], vec![4.0, 6.0]).unwrap();
    let output_norm_weight =
        CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0, 1.0]).unwrap();
    let diagnostics = output_projection_diagnostics(
        &output_norm,
        &output_weight,
        &logits,
        &[2],
        Some(&final_hidden),
        Some(&output_norm_weight),
        Some(0.5),
    )
    .unwrap();

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].token_id, 2);
    assert_eq!(diagnostics[0].layout, "descriptor");
    assert_close(diagnostics[0].reported_logit, 24.0);
    assert_close(diagnostics[0].reconstructed_logit, 24.0);
    assert_close(diagnostics[0].absolute_delta, 0.0);
    assert_eq!(diagnostics[0].output_row_first_values, vec![3.0, 6.0]);
    assert_eq!(
        diagnostics[0].component_products_first_values,
        vec![6.0, 18.0]
    );
    assert_eq!(diagnostics[0].component_products_max_abs_window_start, 0);
    assert_eq!(
        diagnostics[0].component_products_max_abs_window,
        vec![6.0, 18.0]
    );
    assert_eq!(diagnostics[0].max_abs_component_index, 1);
    assert_close(diagnostics[0].max_abs_component, 18.0);
    assert_close(diagnostics[0].positive_component_sum, 24.0);
    assert_close(diagnostics[0].negative_component_sum, 0.0);
    assert_eq!(diagnostics[0].top_positive_components.len(), 2);
    assert_eq!(diagnostics[0].top_positive_components[0].index, 1);
    assert_close(
        diagnostics[0].top_positive_components[0].output_norm_value,
        3.0,
    );
    assert_close(
        diagnostics[0].top_positive_components[0].output_row_value,
        6.0,
    );
    assert_close(diagnostics[0].top_positive_components[0].component, 18.0);
    assert_eq!(
        diagnostics[0].top_positive_components[0].final_hidden_value,
        Some(6.0)
    );
    assert_eq!(
        diagnostics[0].top_positive_components[0].output_norm_weight_value,
        Some(1.0)
    );
    assert_eq!(
        diagnostics[0].top_positive_components[0].output_norm_scale,
        Some(0.5)
    );
    assert_eq!(
        diagnostics[0].top_positive_components[0].reconstructed_output_norm_value,
        Some(3.0)
    );
    assert_eq!(
        diagnostics[0].top_positive_components[0].output_norm_reconstruction_delta,
        Some(0.0)
    );
    assert_eq!(diagnostics[0].top_positive_components[1].index, 0);
    assert_close(diagnostics[0].top_positive_components[1].component, 6.0);
    assert!(diagnostics[0].top_negative_components.is_empty());
}

#[test]
fn output_projection_diagnostics_report_signed_component_extremes() {
    let output_norm =
        CpuTensor::from_f32("output_norm", vec![1, 4], vec![2.0, -3.0, 4.0, -5.0]).unwrap();
    let output_weight =
        CpuTensor::from_f32("output.weight", vec![4, 1], vec![1.5, 2.0, -0.5, -4.0]).unwrap();
    let logits = output_projection_with_layout(
        &output_norm,
        &output_weight,
        "logits",
        OutputProjectionLayout::Descriptor,
    )
    .unwrap();

    let diagnostics = output_projection_diagnostics(
        &output_norm,
        &output_weight,
        &logits,
        &[0],
        None,
        None,
        None,
    )
    .unwrap();

    assert_close(diagnostics[0].reported_logit, 15.0);
    assert_close(diagnostics[0].positive_component_sum, 23.0);
    assert_close(diagnostics[0].negative_component_sum, -8.0);
    assert_eq!(diagnostics[0].top_positive_components.len(), 2);
    assert_eq!(diagnostics[0].top_positive_components[0].index, 3);
    assert_close(diagnostics[0].top_positive_components[0].component, 20.0);
    assert_eq!(diagnostics[0].top_positive_components[1].index, 0);
    assert_close(diagnostics[0].top_positive_components[1].component, 3.0);
    assert_eq!(diagnostics[0].top_negative_components.len(), 2);
    assert_eq!(diagnostics[0].top_negative_components[0].index, 1);
    assert_close(diagnostics[0].top_negative_components[0].component, -6.0);
    assert_eq!(diagnostics[0].top_negative_components[1].index, 2);
    assert_close(diagnostics[0].top_negative_components[1].component, -2.0);
}

#[test]
fn square_linear_transposed_diagnostic_reinterprets_ambiguous_square_weight() {
    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let square_weight = CpuTensor::from_f32(
        "attention_q.weight",
        vec![2, 2],
        vec![
            1.0, 2.0, // descriptor row for input dim 0
            3.0, 4.0, // descriptor row for input dim 1
        ],
    )
    .unwrap();

    let descriptor = linear_with_diagnostic_layouts(
        &input,
        &square_weight,
        "descriptor",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Auto,
    )
    .unwrap();
    let transposed = linear_with_diagnostic_layouts(
        &input,
        &square_weight,
        "transposed",
        SquareLinearLayout::Transposed,
        RectangularLinearLayout::Auto,
    )
    .unwrap();

    assert_eq!(descriptor.shape.dims, vec![1, 2]);
    assert_eq!(transposed.shape.dims, vec![1, 2]);
    assert_close(descriptor.data[0], 11.0);
    assert_close(descriptor.data[1], 16.0);
    assert_close(transposed.data[0], 8.0);
    assert_close(transposed.data[1], 18.0);
}

#[test]
fn rectangular_linear_role_override_reinterprets_only_named_projection() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let descriptor_weight = CpuTensor::from_f32(
        "attention_k.weight",
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0, // descriptor row for input dim 0
            4.0, 5.0, 6.0, // descriptor row for input dim 1
        ],
    )
    .unwrap();

    std::env::set_var(
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K",
        "transposed",
    );
    let overridden =
        linear_for_role(&input, &descriptor_weight, "overridden", "attention_k").unwrap();
    let unaffected =
        linear_for_role(&input, &descriptor_weight, "unaffected", "attention_v").unwrap();
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_K");
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");

    assert_eq!(overridden.shape.dims, vec![1, 3]);
    assert_eq!(unaffected.shape.dims, vec![1, 3]);
    assert_close(overridden.data[0], 8.0);
    assert_close(overridden.data[1], 18.0);
    assert_close(overridden.data[2], 28.0);
    assert_close(unaffected.data[0], 14.0);
    assert_close(unaffected.data[1], 19.0);
    assert_close(unaffected.data[2], 24.0);
}

#[test]
fn linear_accumulation_precision_f64_reconstructs_descriptor_layout() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_LINEAR_ACCUMULATION", "f64");
    std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, 1.0e-3, -2.0]).unwrap();
    let weight = CpuTensor::from_f32(
        "weight",
        vec![3, 2],
        vec![1.0e8, -1.0e8, -1.0e8, 1.0e8, 0.25, -0.5],
    )
    .unwrap();

    let actual = linear(&input, &weight, "out").unwrap();

    std::env::remove_var("CAMELID_LINEAR_ACCUMULATION");
    std::env::remove_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT");
    let expected = vec![
        (1.0_f64 * 1.0e8 + 1.0e-3 * -1.0e8 + -2.0 * 0.25) as f32,
        (1.0_f64 * -1.0e8 + 1.0e-3 * 1.0e8 + -2.0 * -0.5) as f32,
    ];
    assert_eq!(actual.shape.dims, vec![1, 2]);
    assert_eq!(actual.data, expected);
}

#[test]
fn linear_accumulation_precision_f64_reconstructs_transposed_layout() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_LINEAR_ACCUMULATION", "f64");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, 1.0e-3, -2.0]).unwrap();
    let weight = CpuTensor::from_f32(
        "weight",
        vec![2, 3],
        vec![1.0e8, -1.0e8, 0.25, -1.0e8, 1.0e8, -0.5],
    )
    .unwrap();

    let actual = linear(&input, &weight, "out").unwrap();

    std::env::remove_var("CAMELID_LINEAR_ACCUMULATION");
    let expected = vec![
        (1.0_f64 * 1.0e8 + 1.0e-3 * -1.0e8 + -2.0 * 0.25) as f32,
        (1.0_f64 * -1.0e8 + 1.0e-3 * 1.0e8 + -2.0 * -0.5) as f32,
    ];
    assert_eq!(actual.shape.dims, vec![1, 2]);
    assert_eq!(actual.data, expected);
}

#[test]
fn rectangular_linear_transposed_diagnostic_reinterprets_descriptor_weight() {
    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, 3.0]).unwrap();
    let descriptor_weight = CpuTensor::from_f32(
        "attention_k.weight",
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0, // descriptor row for input dim 0
            4.0, 5.0, 6.0, // descriptor row for input dim 1
        ],
    )
    .unwrap();

    let auto = linear_with_diagnostic_layouts(
        &input,
        &descriptor_weight,
        "auto",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Auto,
    )
    .unwrap();
    let forced_transposed = linear_with_diagnostic_layouts(
        &input,
        &descriptor_weight,
        "forced_transposed",
        SquareLinearLayout::Descriptor,
        RectangularLinearLayout::Transposed,
    )
    .unwrap();

    assert_eq!(auto.shape.dims, vec![1, 3]);
    assert_eq!(forced_transposed.shape.dims, vec![1, 3]);
    assert_close(auto.data[0], 8.0);
    assert_close(auto.data[1], 18.0);
    assert_close(auto.data[2], 28.0);
    assert_close(forced_transposed.data[0], 8.0);
    assert_close(forced_transposed.data[1], 18.0);
    assert_close(forced_transposed.data[2], 28.0);
}

#[test]
fn gated_ffn_activation_matches_separate_linear_silu_mul_for_transposed_weights() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");

    let input = CpuTensor::from_f32("input", vec![1, 3], vec![1.0, -2.0, 0.5]).unwrap();
    let gate = CpuTensor::from_f32(
        "gate",
        vec![4, 3],
        vec![
            1.0, 0.0, 0.0, // x
            0.0, 1.0, 0.0, // y
            0.0, 0.0, 1.0, // z
            0.5, -0.5, 1.0,
        ],
    )
    .unwrap();
    let up = CpuTensor::from_f32(
        "up",
        vec![4, 3],
        vec![
            -1.0, 0.0, 0.0, // -x
            0.0, 2.0, 0.0, // 2y
            0.0, 0.0, 3.0, // 3z
            1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let separate = linear(&input, &gate, "gate_out")
        .unwrap()
        .silu_mul(&linear(&input, &up, "up_out").unwrap(), "separate")
        .unwrap();
    let fused = gated_ffn_activation(&input, &gate, &up, "fused", true)
        .unwrap()
        .tensor;

    assert_eq!(fused.shape.dims, vec![1, 4]);
    for (actual, expected) in fused.data.iter().zip(separate.data) {
        assert_close(*actual, expected);
    }
}

#[test]
fn gated_ffn_activation_matches_reference_for_wide_transposed_weights() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let input_values = vec![1.0, -2.0, 0.5];
    let output_width = 1031;
    let input =
        CpuTensor::from_f32("input", vec![1, input_values.len()], input_values.clone()).unwrap();
    let mut gate_values = Vec::with_capacity(output_width * input_values.len());
    let mut up_values = Vec::with_capacity(output_width * input_values.len());
    for idx in 0..output_width {
        gate_values.extend_from_slice(&[
            0.01 * ((idx % 7) as f32 - 3.0),
            -0.02 * ((idx % 5) as f32 - 2.0),
            0.03 * ((idx % 11) as f32 - 5.0),
        ]);
        up_values.extend_from_slice(&[
            -0.015 * ((idx % 13) as f32 - 6.0),
            0.012 * ((idx % 17) as f32 - 8.0),
            0.02 * ((idx % 19) as f32 - 9.0),
        ]);
    }
    let gate = CpuTensor::from_f32(
        "gate",
        vec![output_width, input_values.len()],
        gate_values.clone(),
    )
    .unwrap();
    let up = CpuTensor::from_f32(
        "up",
        vec![output_width, input_values.len()],
        up_values.clone(),
    )
    .unwrap();

    let fused = gated_ffn_activation(&input, &gate, &up, "fused", false)
        .unwrap()
        .tensor;

    assert_eq!(fused.shape.dims, vec![1, output_width]);
    for idx in 0..output_width {
        let row_start = idx * input_values.len();
        let gate_value = input_values
            .iter()
            .zip(&gate_values[row_start..row_start + input_values.len()])
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let up_value = input_values
            .iter()
            .zip(&up_values[row_start..row_start + input_values.len()])
            .map(|(left, right)| left * right)
            .sum::<f32>();
        assert_close(fused.data[idx], silu(gate_value) * up_value);
    }
}

#[test]
fn gated_ffn_activation_matches_separate_linear_silu_mul_for_direct_weights() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");

    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, -1.0]).unwrap();
    let gate = CpuTensor::from_f32(
        "gate",
        vec![2, 3],
        vec![
            1.0, 0.0, -1.0, // input col 0 contributions
            0.5, 2.0, 1.0, // input col 1 contributions
        ],
    )
    .unwrap();
    let up = CpuTensor::from_f32("up", vec![2, 3], vec![-1.0, 0.25, 0.5, 1.5, -0.5, 2.0]).unwrap();

    let separate = linear(&input, &gate, "gate_out")
        .unwrap()
        .silu_mul(&linear(&input, &up, "up_out").unwrap(), "separate")
        .unwrap();
    let fused = gated_ffn_activation(&input, &gate, &up, "fused", true)
        .unwrap()
        .tensor;

    assert_eq!(fused.shape.dims, vec![1, 3]);
    for (actual, expected) in fused.data.iter().zip(separate.data) {
        assert_close(*actual, expected);
    }
}

#[test]
fn ffn_gate_up_order_diagnostic_can_apply_silu_to_up_projection() {
    let _env_guard = env_lock();
    std::env::set_var("CAMELID_FFN_GATE_UP_ORDER", "up_gate");

    let input = CpuTensor::from_f32("input", vec![1, 2], vec![2.0, -1.0]).unwrap();
    let gate = CpuTensor::from_f32(
        "gate",
        vec![2, 3],
        vec![
            1.0, 0.0, -1.0, // input col 0 contributions
            0.5, 2.0, 1.0, // input col 1 contributions
        ],
    )
    .unwrap();
    let up = CpuTensor::from_f32("up", vec![2, 3], vec![-1.0, 0.25, 0.5, 1.5, -0.5, 2.0]).unwrap();

    let separate = linear(&input, &up, "up_out")
        .unwrap()
        .silu_mul(&linear(&input, &gate, "gate_out").unwrap(), "separate")
        .unwrap();
    let fused = gated_ffn_activation(&input, &gate, &up, "fused", true).unwrap();

    assert_eq!(fused.tensor.shape.dims, vec![1, 3]);
    for (actual, expected) in fused.tensor.data.iter().zip(separate.data) {
        assert_close(*actual, expected);
    }
    let diagnostic = fused.activation_diagnostic.expect("activation diagnostic");
    assert_eq!(diagnostic.activation_order, "up_gate");
    assert_close(diagnostic.max_abs_delta, 0.0);

    std::env::remove_var("CAMELID_FFN_GATE_UP_ORDER");
}

#[test]
fn single_token_forward_diagnostics_follow_llama_stage_order() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    std::env::set_var("CAMELID_SQUARE_LINEAR_LAYOUT", "descriptor");
    std::env::set_var("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "descriptor");
    std::env::set_var("CAMELID_OUTPUT_PROJECTION_LAYOUT", "descriptor");
    std::env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "1");

    let config = LlamaModelConfig {
        context_length: 4,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 0.0,
        vocab_size: Some(3),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let weights = Arc::new(LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd.weight",
            vec![3, 2],
            vec![
                1.0, 1.0, // token 0, selected by the prompt
                -1.0, 0.5, // token 1
                0.25, -0.75, // token 2
            ],
        )
        .unwrap(),
        output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![1.0, 1.0]).unwrap(),
        output: Some(
            CpuTensor::from_f32(
                "output.weight",
                vec![2, 3],
                vec![
                    1.0, 0.0, -1.0, // hidden dim 0 -> vocab logits
                    0.0, 1.0, -1.0, // hidden dim 1 -> vocab logits
                ],
            )
            .unwrap(),
        ),
        rope_freqs: None,
        layers: vec![LlamaLayerWeights {
            attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0, 1.0])
                .unwrap(),
            attention_q: CpuTensor::from_f32(
                "blk.0.attn_q.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_k: CpuTensor::from_f32(
                "blk.0.attn_k.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_v: CpuTensor::from_f32(
                "blk.0.attn_v.weight",
                vec![2, 2],
                vec![0.25, 0.25, 0.25, 0.25],
            )
            .unwrap(),
            attention_output: CpuTensor::from_f32(
                "blk.0.attn_output.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0, 1.0])
                .unwrap(),
            ffn_gate: CpuTensor::from_f32(
                "blk.0.ffn_gate.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 2.0],
            )
            .unwrap(),
            ffn_up: CpuTensor::from_f32(
                "blk.0.ffn_up.weight",
                vec![2, 2],
                vec![3.0, 0.0, 0.0, 4.0],
            )
            .unwrap(),
            ffn_down: CpuTensor::from_f32(
                "blk.0.ffn_down.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
    });
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let step = session
        .generate_next_token_with_history_diagnostics(&[0], LlamaSampler::Greedy, &[0], true, None)
        .unwrap();

    assert_eq!(step.prompt_token_count, 1);
    assert_eq!(step.prefill_token_count, 0);
    assert_eq!(step.prefill_timings.total, 0);
    assert_eq!(
        step.first_token_timings
            .memory
            .as_ref()
            .unwrap()
            .forward_passes,
        1
    );
    assert_eq!(step.next_token_id, 1);
    let memory = step
        .timings
        .memory
        .as_ref()
        .expect("memory timings requested");
    assert_eq!(memory.forward_passes, 1);
    assert!(memory.after_embedding.is_some());
    assert!(memory.after_layers.is_some());
    assert!(memory.after_logits.is_some());
    assert_eq!(memory.layers.len(), 1);
    assert_eq!(memory.layers[0].layer_index, 0);
    assert!(memory.layers[0].after_kv_cache_write.is_some());
    assert_eq!(memory.end.as_ref().unwrap().kv_cache_position, 1);
    let diagnostics = step.diagnostics.expect("dense diagnostics requested");
    assert_slice_close(&diagnostics.embedding.checkpoint.first_values, &[1.0, 1.0]);
    assert_eq!(diagnostics.layers.len(), 1);
    let layer = &diagnostics.layers[0];
    assert_eq!(layer.layer_index, 0);

    assert_slice_close(
        &layer.residual_flow.attention_input.checkpoint.first_values,
        &[1.0, 1.0],
    );
    assert_slice_close(&layer.attention_norm.checkpoint.first_values, &[1.0, 1.0]);
    assert_close(layer.attention_norm_reconstruction.input_rms, 1.0);
    assert_close(layer.attention_norm_reconstruction.max_abs_delta, 0.0);
    assert_slice_close(&layer.attention_q.checkpoint.first_values, &[1.0, 1.0]);
    assert_slice_close(&layer.attention_k.checkpoint.first_values, &[1.0, 1.0]);
    assert_slice_close(&layer.attention_q_rope.checkpoint.first_values, &[1.0, 1.0]);
    assert_slice_close(&layer.attention_k_rope.checkpoint.first_values, &[1.0, 1.0]);
    assert_slice_close(&layer.attention_v.checkpoint.first_values, &[0.5, 0.5]);
    assert_eq!(layer.kv_cache_trace.layer_index, 0);
    assert_eq!(layer.kv_cache_trace.position_count, 1);
    assert_eq!(layer.kv_cache_trace.key_value_width, 2);
    assert_close(layer.kv_cache_trace.key_checksum as f32, 3.0);
    assert_close(layer.kv_cache_trace.value_checksum as f32, 1.5);
    assert_close(layer.kv_cache_trace.key_rms, 1.0);
    assert_close(layer.kv_cache_trace.value_rms, 0.5);
    assert_eq!(layer.kv_cache_trace.sampled_positions.len(), 1);
    assert_slice_close(
        &layer.kv_cache_trace.sampled_positions[0].key_first_values,
        &[1.0, 1.0],
    );
    assert_slice_close(
        &layer.kv_cache_trace.sampled_positions[0].value_first_values,
        &[0.5, 0.5],
    );
    assert_slice_close(
        &layer.attention_context.checkpoint.first_values,
        &[0.5, 0.5],
    );
    assert_eq!(layer.attention_trace.position_count, 1);
    assert_close(layer.attention_trace.heads[0].positions[0].probability, 1.0);
    assert_slice_close(&layer.attention_output.checkpoint.first_values, &[0.5, 0.5]);
    assert_slice_close(
        &layer.attention_residual.checkpoint.first_values,
        &[1.5, 1.5],
    );
    assert_slice_close(
        &layer.residual_flow.attention_delta.delta_first_values,
        &[0.5, 0.5],
    );
    assert_close(layer.residual_flow.attention_delta.max_abs_delta, 0.0);

    assert_slice_close(&layer.ffn_norm.checkpoint.first_values, &[1.0, 1.0]);
    assert_slice_close(
        &layer.ffn_gate.as_ref().unwrap().checkpoint.first_values,
        &[1.0, 2.0],
    );
    assert_slice_close(
        &layer.ffn_up.as_ref().unwrap().checkpoint.first_values,
        &[3.0, 4.0],
    );
    let expected_activation = vec![silu(1.0) * 3.0, silu(2.0) * 4.0];
    assert_slice_close(
        &layer
            .ffn_activation
            .as_ref()
            .unwrap()
            .checkpoint
            .first_values,
        &expected_activation,
    );
    assert_eq!(
        layer
            .ffn_activation_reconstruction
            .as_ref()
            .unwrap()
            .activation_order,
        "gate_up"
    );
    assert_close(
        layer
            .ffn_activation_reconstruction
            .as_ref()
            .unwrap()
            .max_abs_delta,
        0.0,
    );
    assert_slice_close(
        &layer.ffn_output.as_ref().unwrap().checkpoint.first_values,
        &expected_activation,
    );

    let expected_hidden = vec![1.5 + expected_activation[0], 1.5 + expected_activation[1]];
    assert_slice_close(
        &layer.ffn_residual.checkpoint.first_values,
        &expected_hidden,
    );
    assert_slice_close(
        &diagnostics.final_hidden.checkpoint.first_values,
        &expected_hidden,
    );
    assert_close(layer.residual_flow.ffn_delta.max_abs_delta, 0.0);

    let final_mean_square = expected_hidden
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / expected_hidden.len() as f32;
    let final_scale = 1.0 / final_mean_square.sqrt();
    let expected_output_norm = expected_hidden
        .iter()
        .map(|value| value * final_scale)
        .collect::<Vec<_>>();
    assert_slice_close(
        &diagnostics.output_norm.checkpoint.first_values,
        &expected_output_norm,
    );
    assert_close(diagnostics.final_norm.max_abs_delta, 0.0);

    let expected_logits = vec![
        expected_output_norm[0],
        expected_output_norm[1],
        -expected_output_norm[0] - expected_output_norm[1],
    ];
    assert_slice_close(&step.logits.data, &expected_logits);
    assert_slice_close(
        &diagnostics.logits.checkpoint.first_values,
        &expected_logits,
    );
}

#[test]
fn chunked_prefill_matches_sequential_prefill_outputs_and_cache() {
    let _env_guard = env_lock();
    // This test asserts the chunked prefill's q8_file_reads delta is all-zero.
    // The delta is measured from the PROCESS-GLOBAL Q8 file-read counters
    // (tensor::q8_0_file_read_stats), so any concurrently running test that
    // records reads — e.g. layer_memory_record_end_captures_tail_q8_file_read_
    // phase's record_q8_0_file_read(32)+(64) = the exact "2 calls / 96 bytes"
    // seen in flaky runs — bleeds into the snapshot. Hold the same
    // q8_file_state_lock every other stats-touching test holds; lock order
    // (env -> q8) matches the dual-guard tests in this file.
    let _q8_guard = crate::test_support::q8_file_state_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 12,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(4),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let weights = Arc::new(LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd.weight",
            vec![4, 2],
            vec![1.0, 0.25, -0.5, 0.75, 0.3, -0.8, 0.2, 0.4],
        )
        .unwrap(),
        output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![0.9, 1.1]).unwrap(),
        output: Some(
            CpuTensor::from_f32(
                "output.weight",
                vec![4, 2],
                vec![0.7, -0.2, -0.4, 0.6, 0.1, 0.3, -0.5, -0.1],
            )
            .unwrap(),
        ),
        rope_freqs: None,
        layers: vec![LlamaLayerWeights {
            attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0, 0.8])
                .unwrap(),
            attention_q: CpuTensor::from_f32(
                "blk.0.attn_q.weight",
                vec![2, 2],
                vec![0.5, -0.1, 0.25, 0.7],
            )
            .unwrap(),
            attention_k: CpuTensor::from_f32(
                "blk.0.attn_k.weight",
                vec![2, 2],
                vec![0.3, 0.2, -0.4, 0.6],
            )
            .unwrap(),
            attention_v: CpuTensor::from_f32(
                "blk.0.attn_v.weight",
                vec![2, 2],
                vec![0.2, -0.3, 0.5, 0.4],
            )
            .unwrap(),
            attention_output: CpuTensor::from_f32(
                "blk.0.attn_output.weight",
                vec![2, 2],
                vec![0.6, 0.1, -0.2, 0.9],
            )
            .unwrap(),
            ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.2, 0.7])
                .unwrap(),
            ffn_gate: CpuTensor::from_f32(
                "blk.0.ffn_gate.weight",
                vec![2, 2],
                vec![0.4, -0.6, 0.8, 0.2],
            )
            .unwrap(),
            ffn_up: CpuTensor::from_f32(
                "blk.0.ffn_up.weight",
                vec![2, 2],
                vec![-0.3, 0.9, 0.5, 0.1],
            )
            .unwrap(),
            ffn_down: CpuTensor::from_f32(
                "blk.0.ffn_down.weight",
                vec![2, 2],
                vec![0.7, -0.2, 0.4, 0.3],
            )
            .unwrap(),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
    });

    let prompt = [0, 1, 2, 3, 0, 1, 2];

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "1");
    let mut sequential = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
    let sequential_step = sequential
        .generate_next_token_with_history_diagnostics(
            &prompt,
            LlamaSampler::Greedy,
            &prompt,
            false,
            None,
        )
        .unwrap();

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "2");
    std::env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "1");
    let mut chunked = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
    let chunked_step = chunked
        .generate_next_token_with_history_diagnostics(
            &prompt,
            LlamaSampler::Greedy,
            &prompt,
            false,
            None,
        )
        .unwrap();

    let prefill_memory = chunked_step
        .prefill_timings
        .memory
        .as_ref()
        .expect("chunked prefill records structured memory timings");
    assert_eq!(prefill_memory.forward_passes, 3);
    assert_eq!(prefill_memory.layers.len(), 1);
    assert_eq!(prefill_memory.end.as_ref().unwrap().kv_cache_position, 6);
    for layer_memory in &prefill_memory.layers {
        assert_eq!(layer_memory.forward_passes, 3);
        assert!(layer_memory.after_attention_norm.is_some());
        assert!(layer_memory.after_attention_q.is_some());
        assert!(layer_memory.after_attention_k.is_some());
        assert!(layer_memory.after_attention_rope.is_some());
        assert!(layer_memory.after_attention_v.is_some());
        assert!(layer_memory.after_kv_cache_write.is_some());
        assert!(layer_memory.after_attention_context.is_some());
        assert!(layer_memory.after_attention_output.is_some());
        assert!(layer_memory.after_attention_residual.is_some());
        assert!(layer_memory.after_ffn_norm.is_some());
        assert!(layer_memory.after_ffn_activation.is_some());
        assert!(layer_memory.after_ffn_down.is_some());
        assert!(layer_memory.after_ffn_residual.is_some());
        assert_eq!(layer_memory.q8_file_reads, Q8_0FileReadStats::default());
    }
    assert_eq!(prefill_memory.q8_file_reads, Q8_0FileReadStats::default());

    assert_eq!(chunked_step.next_token_id, sequential_step.next_token_id);
    assert_slice_close(&chunked_step.logits.data, &sequential_step.logits.data);
    assert_slice_close(
        &chunked_step.hidden_state.data,
        &sequential_step.hidden_state.data,
    );
    assert_eq!(chunked.kv_cache.position, sequential.kv_cache.position);
    assert_slice_close(&chunked.kv_cache.keys, &sequential.kv_cache.keys);
    assert_slice_close(&chunked.kv_cache.values, &sequential.kv_cache.values);

    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR", "1");
    std::env::set_var("CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION", "1");
    let mut layer_major = LlamaInferenceSession::new(config, weights).unwrap();
    let layer_major_step = layer_major
        .generate_next_token_with_history_diagnostics(
            &prompt,
            LlamaSampler::Greedy,
            &prompt,
            false,
            None,
        )
        .unwrap();
    let layer_major_memory = layer_major_step
        .prefill_timings
        .memory
        .as_ref()
        .expect("layer-major attribution enables structured prefill memory");
    assert!(!layer_major_memory
        .prefill_layer_major_attribution
        .is_empty());
    let first_attribution = &layer_major_memory.prefill_layer_major_attribution[0];
    assert_eq!(first_attribution.layer_index, 0);
    assert_eq!(first_attribution.chunk_start, 0);
    assert!(first_attribution.chunk_rows > 0);
    assert!(first_attribution.hidden_bytes > 0);
    assert!(first_attribution.next_hidden_bytes > 0);
    assert!(first_attribution.chunk_input_bytes > 0);
    assert!(first_attribution.kv_cache_bytes_after >= first_attribution.kv_cache_bytes_before);
    let serialized_memory = serde_json::to_value(layer_major_memory).unwrap();
    assert!(serialized_memory
        .get("prefill_layer_major_attribution")
        .and_then(|value| value.as_array())
        .is_some_and(|value| !value.is_empty()));
    assert_eq!(
        layer_major_step.next_token_id,
        sequential_step.next_token_id
    );
    assert_slice_close(&layer_major_step.logits.data, &sequential_step.logits.data);
    assert_slice_close(
        &layer_major_step.hidden_state.data,
        &sequential_step.hidden_state.data,
    );
    assert_eq!(layer_major.kv_cache.position, sequential.kv_cache.position);
    assert_slice_close(&layer_major.kv_cache.keys, &sequential.kv_cache.keys);
    assert_slice_close(&layer_major.kv_cache.values, &sequential.kv_cache.values);

    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR");
    std::env::remove_var("CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION");
    std::env::remove_var("CAMELID_FORWARD_RSS_TIMINGS");
}

#[test]
fn prefill_layer_rejects_misaligned_kv_cache_cursor() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 8,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(4),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let layer = LlamaLayerWeights {
        attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0, 1.0])
            .unwrap(),
        attention_q: CpuTensor::from_f32(
            "blk.0.attn_q.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        attention_k: CpuTensor::from_f32(
            "blk.0.attn_k.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        attention_v: CpuTensor::from_f32(
            "blk.0.attn_v.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        attention_output: CpuTensor::from_f32(
            "blk.0.attn_output.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.0, 1.0]).unwrap(),
        ffn_gate: CpuTensor::from_f32(
            "blk.0.ffn_gate.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        ffn_up: CpuTensor::from_f32("blk.0.ffn_up.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap(),
        ffn_down: CpuTensor::from_f32(
            "blk.0.ffn_down.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
        attention_biases: None,
        attention_q_norm: None,
        attention_k_norm: None,
        moe_router: None,
        mla_q_a_proj: None,
        mla_q_a_layernorm: None,
        mla_q_b_proj: None,
        mla_kv_a_proj_with_mqa: None,
        mla_kv_a_layernorm: None,
        mla_kv_b_proj: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        decode_bindings: DecodeLinearBindings::default(),
    };
    let hidden = CpuTensor::from_f32("hidden", vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    let plan = LlamaKvCachePlan::from_config(&config).unwrap();
    let mut kv_cache = LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).unwrap();
    kv_cache.position = 1;

    let err = forward_prefill_layer_chunk_timed(
        &hidden,
        &layer,
        PrefillLayerChunkParams {
            config: &config,
            rope_freqs: None,
            rms_norm_epsilon: config.rms_norm_epsilon,
            layer_idx: 0,
            base_position: 0,
            chunk_start: 0,
            chunk_rows: 2,
        },
        &mut kv_cache,
    )
    .unwrap_err();

    assert!(err
        .to_string()
        .contains("prefill chunk base position 0 does not match KV cache cursor 1"));
}

#[test]
fn batch_attention_rejects_reads_beyond_allocated_kv_cache() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 2,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(4),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let kv_cache = LlamaKvCache::new(
        LlamaKvCachePlan::from_config(&config).unwrap(),
        crate::model::KvCacheQuantization::F16,
    )
    .unwrap();
    let query = CpuTensor::from_f32("query", vec![1, 2], vec![0.1, 0.2]).unwrap();

    let err = causal_attention_context_batch(&kv_cache, 0, 0, &query, 1, 1, "context").unwrap_err();

    assert!(err
        .to_string()
        .contains("attention batch needs 1 cached position(s), but KV cache has 0 allocated"));
}

#[test]
fn batch_attention_parallel_context_matches_serial() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let rows = 8;
    let attention_heads = 32;
    let kv_heads = 8;
    let head_dim = 2;
    let expected_width = attention_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let plan = LlamaKvCachePlan {
        max_sequence_length: rows,
        layer_count: 1,
        kv_head_count: kv_heads,
        head_dim,
        k_head_dim: head_dim,
        v_head_dim: head_dim,
        key_shape: vec![1, rows, kv_heads, head_dim],
        value_shape: vec![1, rows, kv_heads, head_dim],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");
    let key_data: Vec<f32> = (0..rows * kv_width)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
        .collect();
    let value_data: Vec<f32> = (0..rows * kv_width)
        .map(|idx| 10.0 + ((idx % 17) as f32) * 0.25)
        .collect();
    let query_data: Vec<f32> = (0..rows * expected_width)
        .map(|idx| ((idx % 19) as f32 - 9.0) * 0.0625)
        .collect();

    let key = CpuTensor::from_f32("key", vec![rows, kv_width], key_data).unwrap();
    let value = CpuTensor::from_f32("value", vec![rows, kv_width], value_data).unwrap();
    write_kv_cache_batch(&mut kv_cache, 0, 0, &key, &value).unwrap();
    let query = CpuTensor::from_f32("query", vec![rows, expected_width], query_data).unwrap();

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let serial = serial_pool
        .install(|| {
            assert!(!should_parallelize_attention_context_batch(
                rows,
                attention_heads
            ));
            causal_attention_context_batch(
                &kv_cache,
                0,
                0,
                &query,
                attention_heads,
                kv_heads,
                "serial",
            )
        })
        .unwrap();

    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let parallel = parallel_pool
        .install(|| {
            assert!(should_parallelize_attention_context_batch(
                rows,
                attention_heads
            ));
            causal_attention_context_batch(
                &kv_cache,
                0,
                0,
                &query,
                attention_heads,
                kv_heads,
                "parallel",
            )
        })
        .unwrap();

    assert_eq!(parallel.shape.dims, serial.shape.dims);
    assert_slice_close(&parallel.data, &serial.data);
}

#[test]
fn batch_attention_parallel_context_respects_threshold_and_thread_count() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();
    let single_thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    single_thread_pool.install(|| {
        assert!(!should_parallelize_attention_context_batch(16, 32));
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(!should_parallelize_attention_context_batch(7, 32));
        assert!(should_parallelize_attention_context_batch(8, 32));
    });
}

#[test]
fn zero_prefill_chunk_env_falls_back_without_panicking() {
    let _env_guard = env_lock();
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 8,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(2),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1.0e-5,
        vocab_size: Some(4),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let weights = Arc::new(LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd.weight",
            vec![4, 2],
            vec![1.0, 0.25, -0.5, 0.75, 0.3, -0.8, 0.2, 0.4],
        )
        .unwrap(),
        output_norm: CpuTensor::from_f32("output_norm.weight", vec![2], vec![0.9, 1.1]).unwrap(),
        output: Some(
            CpuTensor::from_f32(
                "output.weight",
                vec![4, 2],
                vec![0.7, -0.2, -0.4, 0.6, 0.1, 0.3, -0.5, -0.1],
            )
            .unwrap(),
        ),
        rope_freqs: None,
        layers: vec![LlamaLayerWeights {
            attention_norm: CpuTensor::from_f32("blk.0.attn_norm.weight", vec![2], vec![1.0, 0.8])
                .unwrap(),
            attention_q: CpuTensor::from_f32(
                "blk.0.attn_q.weight",
                vec![2, 2],
                vec![0.5, -0.1, 0.25, 0.7],
            )
            .unwrap(),
            attention_k: CpuTensor::from_f32(
                "blk.0.attn_k.weight",
                vec![2, 2],
                vec![0.3, 0.2, -0.4, 0.6],
            )
            .unwrap(),
            attention_v: CpuTensor::from_f32(
                "blk.0.attn_v.weight",
                vec![2, 2],
                vec![0.2, -0.3, 0.5, 0.4],
            )
            .unwrap(),
            attention_output: CpuTensor::from_f32(
                "blk.0.attn_output.weight",
                vec![2, 2],
                vec![0.6, 0.1, -0.2, 0.9],
            )
            .unwrap(),
            ffn_norm: CpuTensor::from_f32("blk.0.ffn_norm.weight", vec![2], vec![1.2, 0.7])
                .unwrap(),
            ffn_gate: CpuTensor::from_f32(
                "blk.0.ffn_gate.weight",
                vec![2, 2],
                vec![0.4, -0.6, 0.8, 0.2],
            )
            .unwrap(),
            ffn_up: CpuTensor::from_f32(
                "blk.0.ffn_up.weight",
                vec![2, 2],
                vec![-0.3, 0.9, 0.5, 0.1],
            )
            .unwrap(),
            ffn_down: CpuTensor::from_f32(
                "blk.0.ffn_down.weight",
                vec![2, 2],
                vec![0.7, -0.2, 0.4, 0.3],
            )
            .unwrap(),
            attention_biases: None,
            attention_q_norm: None,
            attention_k_norm: None,
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
    });

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "0");
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();
    let step = session
        .generate_next_token_with_history_diagnostics(
            &[0, 1, 2],
            LlamaSampler::Greedy,
            &[0, 1, 2],
            false,
            None,
        )
        .unwrap();

    assert_eq!(step.prefill_token_count, 2);
    assert!(step.prefill_timings.total > 0);

    std::env::remove_var("CAMELID_PREFILL_CHUNK_TOKENS");
}

#[test]
fn kv_cache_allocates_positions_lazily_without_losing_prior_layers() {
    let plan = LlamaKvCachePlan {
        max_sequence_length: 10,
        layer_count: 2,
        kv_head_count: 1,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![2, 10, 1, 2],
        value_shape: vec![2, 10, 1, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");
    assert_eq!(kv_cache.allocated_sequence_length, 0);
    assert!(kv_cache.keys.is_empty());
    assert!(kv_cache.values.is_empty());

    let layer0_key = CpuTensor::from_f32("layer0_key", vec![1, 2], vec![1.0, 2.0]).unwrap();
    let layer0_value = CpuTensor::from_f32("layer0_value", vec![1, 2], vec![3.0, 4.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &layer0_key, &layer0_value).unwrap();
    assert_eq!(kv_cache.allocated_sequence_length, 1);
    assert_eq!(kv_cache.keys.len(), 4);
    assert_eq!(kv_cache.values.len(), 4);

    let layer1_key = CpuTensor::from_f32("layer1_key", vec![1, 2], vec![5.0, 6.0]).unwrap();
    let layer1_value = CpuTensor::from_f32("layer1_value", vec![1, 2], vec![7.0, 8.0]).unwrap();
    write_kv_cache(&mut kv_cache, 1, &layer1_key, &layer1_value).unwrap();

    kv_cache.position = 1;
    let layer0_next_key =
        CpuTensor::from_f32("layer0_next_key", vec![1, 2], vec![9.0, 10.0]).unwrap();
    let layer0_next_value =
        CpuTensor::from_f32("layer0_next_value", vec![1, 2], vec![11.0, 12.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &layer0_next_key, &layer0_next_value).unwrap();
    assert_eq!(kv_cache.allocated_sequence_length, 2);
    assert_eq!(kv_cache.keys.len(), 8);
    assert_eq!(kv_cache.values.len(), 8);

    let prior_layer1_start = kv_cache.offset(1, 0, 0);
    assert_eq!(
        &kv_cache.keys[prior_layer1_start..prior_layer1_start + 2],
        &[5.0, 6.0]
    );
    assert_eq!(
        &kv_cache.values[prior_layer1_start..prior_layer1_start + 2],
        &[7.0, 8.0]
    );
}

#[test]
fn kv_cache_uses_paged_growth_for_model_sized_contexts() {
    let plan = LlamaKvCachePlan {
        max_sequence_length: 1024,
        layer_count: 2,
        kv_head_count: 1,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![2, 1024, 1, 2],
        value_shape: vec![2, 1024, 1, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");
    let key = CpuTensor::from_f32("key", vec![1, 2], vec![1.0, 2.0]).unwrap();
    let value = CpuTensor::from_f32("value", vec![1, 2], vec![3.0, 4.0]).unwrap();

    write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();

    assert_eq!(kv_cache.allocated_sequence_length, 256);
    assert_eq!(kv_cache.keys.len(), 1024);
    assert_eq!(kv_cache.values.len(), 1024);
}

#[test]
fn kv_cache_storage_matches_llama_cpp_f16_rounding() {
    let plan = LlamaKvCachePlan {
        max_sequence_length: 1,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![1, 1, 1, 2],
        value_shape: vec![1, 1, 1, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");
    let key = CpuTensor::from_f32("key", vec![1, 2], vec![1.0001, -2.0003]).unwrap();
    let value = CpuTensor::from_f32("value", vec![1, 2], vec![3.0007, -4.0009]).unwrap();

    write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();

    assert_eq!(
        kv_cache.keys,
        key.data
            .iter()
            .copied()
            .map(|value| f16_bits_to_f32(f32_to_f16_bits(value)))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        kv_cache.values,
        value
            .data
            .iter()
            .copied()
            .map(|value| f16_bits_to_f32(f32_to_f16_bits(value)))
            .collect::<Vec<_>>()
    );
    assert_ne!(kv_cache.keys, key.data);
    assert_ne!(kv_cache.values, value.data);
}

#[test]
fn quantized_attention_matches_its_dequantized_cache_for_tail_rows() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let head_dim = 40;
    let plan = LlamaKvCachePlan {
        max_sequence_length: 3,
        layer_count: 1,
        kv_head_count: 1,
        head_dim,
        k_head_dim: head_dim,
        v_head_dim: head_dim,
        key_shape: vec![1, 3, 1, head_dim],
        value_shape: vec![1, 3, 1, head_dim],
    };
    let query_data: Vec<f32> = (0..head_dim)
        .map(|index| (index as f32 - 13.0) * 0.03125)
        .collect();
    let query = CpuTensor::from_f32("query", vec![1, head_dim], query_data.clone()).unwrap();

    for dtype in [KvDtype::Q8_0, KvDtype::Q4_0] {
        for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
            let mut cache =
                LlamaKvCache::new_with_layout_and_dtype(plan.clone(), layout, dtype).unwrap();
            cache.ensure_position_capacity(3).unwrap();
            for position in 0..3 {
                let key: Vec<f32> = (0..head_dim)
                    .map(|index| (index as f32 - 17.0) * 0.025 + position as f32 * 0.0625)
                    .collect();
                let value: Vec<f32> = (0..head_dim)
                    .map(|index| (9.0 - index as f32) * 0.04 + position as f32 * 0.125)
                    .collect();
                cache.store_kv_head_row(0, position, 0, &key, &value);
            }
            cache.position = 2;

            let actual = causal_attention_context(&cache, 0, &query, 1, 1, "context", false)
                .unwrap()
                .tensor;

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut scores = Vec::new();
            let mut values = Vec::new();
            for position in 0..3 {
                let mut key = vec![0.0; head_dim];
                let mut value = vec![0.0; head_dim];
                cache.copy_key_row_into(0, position, 0, &mut key);
                cache.copy_value_row_into(0, position, 0, &mut value);
                scores.push(dot_product(&query_data, &key) * scale);
                values.push(value);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (score - max_score).exp()).sum();
            let mut expected = vec![0.0; head_dim];
            for (score, value) in scores.iter().zip(values) {
                let probability = (*score - max_score).exp() / denominator;
                for (out, value) in expected.iter_mut().zip(value) {
                    *out += probability * value;
                }
            }

            for (index, (actual, expected)) in actual.data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{dtype:?}/{layout:?} attention mismatch at {index}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn causal_attention_context_attends_over_prior_and_current_positions() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let plan = LlamaKvCachePlan {
        max_sequence_length: 3,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![1, 3, 1, 2],
        value_shape: vec![1, 3, 1, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");

    let prior_key = CpuTensor::from_f32("prior_key", vec![1, 2], vec![1.0, 0.0]).unwrap();
    let prior_value = CpuTensor::from_f32("prior_value", vec![1, 2], vec![10.0, 0.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &prior_key, &prior_value).unwrap();
    kv_cache.position = 1;
    let current_key = CpuTensor::from_f32("current_key", vec![1, 2], vec![0.0, 1.0]).unwrap();
    let current_value = CpuTensor::from_f32("current_value", vec![1, 2], vec![0.0, 20.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &current_key, &current_value).unwrap();

    let query = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
    let context = causal_attention_context(&kv_cache, 0, &query, 1, 1, "context", true).unwrap();

    let first_score = (1.0_f32 / 2.0_f32.sqrt()).exp();
    let first_probability = first_score / (first_score + 1.0);
    assert_eq!(context.tensor.shape.dims, vec![1, 2]);
    assert_close(context.tensor.data[0], first_probability * 10.0);
    assert_close(context.tensor.data[1], (1.0 - first_probability) * 20.0);
    let trace = context.trace.expect("trace diagnostics requested");
    assert_eq!(trace.position_count, 2);
    assert_eq!(trace.head_dim, 2);
    assert_eq!(trace.heads.len(), 1);
    let head = &trace.heads[0];
    assert_eq!(head.attention_head, 0);
    assert_eq!(head.kv_head, 0);
    assert_eq!(head.query_first_values, vec![1.0, 0.0]);
    assert_close(head.probability_sum, 1.0);
    assert_close(
        head.probability_entropy,
        -(first_probability * first_probability.ln()
            + (1.0 - first_probability) * (1.0 - first_probability).ln()),
    );
    assert_close(
        head.probability_rms,
        ((first_probability * first_probability
            + (1.0 - first_probability) * (1.0 - first_probability))
            / 2.0)
            .sqrt(),
    );
    assert_eq!(head.max_probability_position, 0);
    assert_close(head.max_probability, first_probability);
    assert_eq!(head.top_probability_positions.len(), 2);
    assert_eq!(head.top_probability_positions[0].position, 0);
    assert_close(
        head.top_probability_positions[0].score,
        1.0 / 2.0_f32.sqrt(),
    );
    assert_close(
        head.top_probability_positions[0].probability,
        first_probability,
    );
    assert_eq!(
        head.top_probability_positions[0].key_first_values,
        vec![1.0, 0.0]
    );
    assert_eq!(
        head.top_probability_positions[0].value_first_values,
        vec![10.0, 0.0]
    );
    assert_eq!(head.context_reconstruction_max_abs_delta_index, 0);
    assert_close(head.context_reconstruction_max_abs_delta, 0.0);
    assert_eq!(head.positions.len(), 2);
    assert_close(head.positions[0].score, 1.0 / 2.0_f32.sqrt());
    assert_close(head.positions[0].probability, first_probability);
    assert_eq!(head.positions[0].key_first_values, vec![1.0, 0.0]);
    assert_eq!(head.positions[0].value_first_values, vec![10.0, 0.0]);
    assert_close(head.context_first_values[0], first_probability * 10.0);
    assert_close(
        head.context_first_values[1],
        (1.0 - first_probability) * 20.0,
    );
    assert_eq!(head.reconstructed_context_first_values.len(), 2);
    assert_close(
        head.reconstructed_context_first_values[0],
        first_probability * 10.0,
    );
    assert_close(
        head.reconstructed_context_first_values[1],
        (1.0 - first_probability) * 20.0,
    );
}

#[test]
fn causal_attention_context_repeats_grouped_kv_heads_for_single_position() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let plan = LlamaKvCachePlan {
        max_sequence_length: 1,
        layer_count: 1,
        kv_head_count: 2,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![1, 1, 2, 2],
        value_shape: vec![1, 1, 2, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");

    let key = CpuTensor::from_f32("key", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let value = CpuTensor::from_f32("value", vec![1, 4], vec![10.0, 11.0, 20.0, 21.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();

    let query = CpuTensor::from_f32(
        "query",
        vec![1, 8],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
    )
    .unwrap();
    let context = causal_attention_context(&kv_cache, 0, &query, 4, 2, "context", true).unwrap();

    assert_eq!(
        context.tensor.data,
        vec![10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]
    );

    let trace = context.trace.expect("trace diagnostics requested");
    assert_eq!(trace.position_count, 1);
    assert_eq!(trace.heads.len(), 4);
    assert_eq!(trace.heads[0].attention_head, 0);
    assert_eq!(trace.heads[0].kv_head, 0);
    assert_eq!(trace.heads[1].attention_head, 1);
    assert_eq!(trace.heads[1].kv_head, 0);
    assert_eq!(trace.heads[2].attention_head, 2);
    assert_eq!(trace.heads[2].kv_head, 1);
    assert_eq!(trace.heads[3].attention_head, 3);
    assert_eq!(trace.heads[3].kv_head, 1);
    assert_eq!(trace.heads[1].context_first_values, vec![10.0, 11.0]);
    assert_eq!(trace.heads[1].positions.len(), 1);
    assert_close(trace.heads[1].probability_entropy, 0.0);
    assert_close(trace.heads[1].probability_rms, 1.0);
    assert_close(trace.heads[1].positions[0].score, 0.0);
    assert_close(trace.heads[1].positions[0].reconstructed_score, 0.0);
    assert_close(trace.heads[1].positions[0].score_reconstruction_delta, 0.0);
    assert_eq!(
        trace.heads[1].positions[0].qk_products_first_values,
        vec![0.0, 0.0]
    );
    assert_close(trace.heads[1].context_reconstruction_max_abs_delta, 0.0);
}

#[test]
fn causal_attention_context_repeats_grouped_kv_heads_across_positions() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let plan = LlamaKvCachePlan {
        max_sequence_length: 2,
        layer_count: 1,
        kv_head_count: 2,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![1, 2, 2, 2],
        value_shape: vec![1, 2, 2, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");

    let prior_key = CpuTensor::from_f32("prior_key", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let prior_value =
        CpuTensor::from_f32("prior_value", vec![1, 4], vec![10.0, 0.0, 20.0, 0.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &prior_key, &prior_value).unwrap();
    kv_cache.position = 1;
    let current_key =
        CpuTensor::from_f32("current_key", vec![1, 4], vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    let current_value =
        CpuTensor::from_f32("current_value", vec![1, 4], vec![0.0, 11.0, 0.0, 21.0]).unwrap();
    write_kv_cache(&mut kv_cache, 0, &current_key, &current_value).unwrap();

    let query = CpuTensor::from_f32(
        "query",
        vec![1, 8],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let context = causal_attention_context(&kv_cache, 0, &query, 4, 2, "context", true).unwrap();

    let high_score = (1.0_f32 / 2.0_f32.sqrt()).exp();
    let high_probability = high_score / (high_score + 1.0);
    let low_probability = 1.0 - high_probability;
    assert_eq!(context.tensor.shape.dims, vec![1, 8]);
    assert_close(context.tensor.data[0], high_probability * 10.0);
    assert_close(context.tensor.data[1], low_probability * 11.0);
    assert_close(context.tensor.data[2], low_probability * 10.0);
    assert_close(context.tensor.data[3], high_probability * 11.0);
    assert_close(context.tensor.data[4], high_probability * 20.0);
    assert_close(context.tensor.data[5], low_probability * 21.0);
    assert_close(context.tensor.data[6], low_probability * 20.0);
    assert_close(context.tensor.data[7], high_probability * 21.0);

    let trace = context.trace.expect("trace diagnostics requested");
    assert_eq!(trace.position_count, 2);
    assert_eq!(trace.heads.len(), 4);
    assert_eq!(trace.heads[0].attention_head, 0);
    assert_eq!(trace.heads[0].kv_head, 0);
    assert_eq!(trace.heads[1].attention_head, 1);
    assert_eq!(trace.heads[1].kv_head, 0);
    assert_eq!(trace.heads[2].attention_head, 2);
    assert_eq!(trace.heads[2].kv_head, 1);
    assert_eq!(trace.heads[3].attention_head, 3);
    assert_eq!(trace.heads[3].kv_head, 1);
    assert_close(trace.heads[0].positions[0].probability, high_probability);
    assert_close(
        trace.heads[0].positions[0].reconstructed_score,
        1.0 / 2.0_f32.sqrt(),
    );
    assert_close(trace.heads[0].positions[0].score_reconstruction_delta, 0.0);
    assert_eq!(
        trace.heads[0].positions[0].qk_products_first_values,
        vec![1.0, 0.0]
    );
    assert_eq!(
        trace.heads[0].positions[0].qk_products_max_abs_window_start,
        0
    );
    assert_eq!(
        trace.heads[0].positions[0].qk_products_max_abs_window,
        vec![1.0, 0.0]
    );
    assert_close(trace.heads[1].positions[1].probability, high_probability);
    assert_close(trace.heads[0].context_reconstruction_max_abs_delta, 0.0);
    assert_close(trace.heads[1].context_reconstruction_max_abs_delta, 0.0);
}

#[test]
fn attention_trace_reports_top_probability_positions_outside_edge_samples() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let plan = LlamaKvCachePlan {
        max_sequence_length: 10,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 2,
        k_head_dim: 2,
        v_head_dim: 2,
        key_shape: vec![1, 10, 1, 2],
        value_shape: vec![1, 10, 1, 2],
    };
    let mut kv_cache =
        LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16).expect("KV cache");

    for position in 0..10 {
        kv_cache.position = position;
        let key_values = if position == 5 {
            vec![10.0, 0.0]
        } else {
            vec![0.0, 0.0]
        };
        let key = CpuTensor::from_f32("key", vec![1, 2], key_values).unwrap();
        let value = CpuTensor::from_f32(
            "value",
            vec![1, 2],
            vec![position as f32, -(position as f32)],
        )
        .unwrap();
        write_kv_cache(&mut kv_cache, 0, &key, &value).unwrap();
    }
    kv_cache.position = 9;

    let query = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();
    let context = causal_attention_context(&kv_cache, 0, &query, 1, 1, "context", true).unwrap();
    let trace = context.trace.expect("trace diagnostics requested");
    let head = &trace.heads[0];

    assert_eq!(
        head.positions
            .iter()
            .map(|position| position.position)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 6, 7, 8, 9]
    );
    assert_eq!(head.top_probability_positions.len(), 4);
    assert_eq!(head.top_probability_positions[0].position, 5);
    assert!(
        head.top_probability_positions[0].probability
            > head.top_probability_positions[1].probability
    );
    assert_eq!(
        head.top_probability_positions[0].key_first_values,
        vec![10.0, 0.0]
    );
    assert_eq!(
        head.top_probability_positions[0].value_first_values,
        vec![5.0, -5.0]
    );
    assert_close(
        head.top_probability_positions[0].score,
        10.0 / 2.0_f32.sqrt(),
    );
}

#[test]
fn attention_score_scale_diagnostic_supports_default_and_unscaled_modes() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    assert_eq!(
        diagnostic_attention_score_scale().unwrap(),
        AttentionScoreScale::HeadDim
    );
    assert_close(
        attention_score_scale_value(4, diagnostic_attention_score_scale().unwrap()),
        0.5,
    );

    std::env::set_var("CAMELID_ATTENTION_SCORE_SCALE", "none");
    assert_eq!(
        diagnostic_attention_score_scale().unwrap(),
        AttentionScoreScale::None
    );
    assert_close(
        attention_score_scale_value(4, diagnostic_attention_score_scale().unwrap()),
        1.0,
    );

    std::env::set_var("CAMELID_ATTENTION_SCORE_SCALE", "bogus");
    assert!(diagnostic_attention_score_scale().is_err());
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
}

#[test]
fn gqa_head_mapping_supports_grouped_and_modulo_indexing() {
    assert_eq!(
        map_attention_head_to_kv_head(0, 2, 2, GqaHeadMapping::Grouped),
        0
    );
    assert_eq!(
        map_attention_head_to_kv_head(1, 2, 2, GqaHeadMapping::Grouped),
        0
    );
    assert_eq!(
        map_attention_head_to_kv_head(2, 2, 2, GqaHeadMapping::Grouped),
        1
    );
    assert_eq!(
        map_attention_head_to_kv_head(3, 2, 2, GqaHeadMapping::Grouped),
        1
    );

    assert_eq!(
        map_attention_head_to_kv_head(0, 2, 2, GqaHeadMapping::Modulo),
        0
    );
    assert_eq!(
        map_attention_head_to_kv_head(1, 2, 2, GqaHeadMapping::Modulo),
        1
    );
    assert_eq!(
        map_attention_head_to_kv_head(2, 2, 2, GqaHeadMapping::Modulo),
        0
    );
    assert_eq!(
        map_attention_head_to_kv_head(3, 2, 2, GqaHeadMapping::Modulo),
        1
    );
}

#[test]
fn attention_trace_samples_prompt_prefix_and_current_tail_positions() {
    assert_eq!(sampled_attention_trace_positions(0), Vec::<usize>::new());
    assert_eq!(sampled_attention_trace_positions(3), vec![0, 1, 2]);
    assert_eq!(
        sampled_attention_trace_positions(8),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        sampled_attention_trace_positions(18),
        vec![0, 1, 2, 3, 14, 15, 16, 17]
    );
}

#[test]
fn attention_trace_samples_gqa_kv_group_anchors_and_tail_heads() {
    assert_eq!(
        sampled_attention_trace_heads(4, 2, 2, GqaHeadMapping::Grouped),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        sampled_attention_trace_heads(32, 8, 4, GqaHeadMapping::Grouped),
        vec![0, 8, 16, 24, 28, 29, 30, 31]
    );
    assert_eq!(
        sampled_attention_trace_heads(32, 8, 4, GqaHeadMapping::Modulo),
        vec![0, 1, 2, 3, 28, 29, 30, 31]
    );
}

#[test]
fn softmax_top_k_renormalizes_selected_router_weights() {
    let top = softmax_top_k(&[0.0, 1.0, 2.0], 2);
    assert_eq!(top[0].0, 2);
    assert_eq!(top[1].0, 1);
    let selected_sum = top.iter().map(|(_, weight)| *weight).sum::<f32>();
    assert!((selected_sum - 1.0).abs() < 1.0e-6, "{top:?}");
    let full_sum = 0.0_f32.exp() + 1.0_f32.exp() + 2.0_f32.exp();
    let expected_first =
        (2.0_f32.exp() / full_sum) / ((2.0_f32.exp() / full_sum) + (1.0_f32.exp() / full_sum));
    assert!((top[0].1 - expected_first).abs() < 1.0e-6, "{top:?}");
}

#[test]
fn softmax_top_k_breaks_router_ties_by_expert_id() {
    let top = softmax_top_k(&[1.0, 1.0, 1.0, 0.0], 2);
    assert_eq!(top[0].0, 0);
    assert_eq!(top[1].0, 1);
    assert!((top[0].1 - 0.5).abs() < 1.0e-6, "{top:?}");
    assert!((top[1].1 - 0.5).abs() < 1.0e-6, "{top:?}");
}

#[test]
fn mixtral_moe_ffn_routes_top_k_experts() {
    let input = CpuTensor::from_f32("input", vec![1, 2], vec![1.0, 1.0]).unwrap();
    let router = CpuTensor::from_f32("router", vec![2, 2], vec![10.0, 0.0, 0.0, 0.0]).unwrap();
    let gate_experts = CpuTensor::from_f32(
        "gate_experts",
        vec![2, 2, 2],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let up_experts = CpuTensor::from_f32(
        "up_experts",
        vec![2, 2, 2],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let down_experts = CpuTensor::from_f32(
        "down_experts",
        vec![2, 2, 2],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let (out, ..) = mixtral_moe_ffn(
        &input,
        &router,
        &gate_experts,
        &up_experts,
        &down_experts,
        2,
        MixtralMoeFfnOptions::new("out", false),
    )
    .unwrap();

    let expected = 1.0 / (1.0 + (-1.0_f32).exp());
    assert!((out.data[0] - expected).abs() < 1.0e-3, "{:?}", out.data);
    assert!((out.data[1] - expected).abs() < 1.0e-3, "{:?}", out.data);
}

#[test]
fn mixtral_moe_ffn_captures_router_logits_and_selected_experts() {
    let input = CpuTensor::from_f32("input", vec![1, 2], vec![1.0, 1.0]).unwrap();
    let router =
        CpuTensor::from_f32("router", vec![2, 3], vec![3.0, 0.0, 2.0, 0.0, 0.0, 0.0]).unwrap();
    let gate_experts = CpuTensor::from_f32(
        "gate_experts",
        vec![2, 2, 3],
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.25],
    )
    .unwrap();
    let up_experts = gate_experts.clone();
    let down_experts = CpuTensor::from_f32(
        "down_experts",
        vec![2, 2, 3],
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.25],
    )
    .unwrap();

    let (out, _, _, _, _, trace) = mixtral_moe_ffn(
        &input,
        &router,
        &gate_experts,
        &up_experts,
        &down_experts,
        2,
        MixtralMoeFfnOptions::new("out", true),
    )
    .unwrap();

    let trace = trace.expect("trace should be captured");
    assert_eq!(trace.expert_used_count, 2);
    assert_eq!(trace.rows.len(), 1);
    assert_eq!(trace.rows[0].row_index, 0);
    assert_eq!(trace.rows[0].router_logits, vec![3.0, 2.0, 0.0]);
    assert_eq!(trace.rows[0].selected_experts, vec![0, 1]);
    let selected_sum = trace.rows[0].selected_weights.iter().sum::<f32>();
    assert!((selected_sum - 1.0).abs() < 1.0e-6, "{trace:?}");
    assert_eq!(trace.rows[0].experts.len(), 2);
    assert_eq!(trace.rows[0].experts[0].expert_index, 0);
    assert_eq!(trace.rows[0].experts[1].expert_index, 1);
    for column in 0..out.data.len() {
        let reconstructed = trace.rows[0]
            .experts
            .iter()
            .map(|expert| expert.weighted_down.checkpoint.first_values[column])
            .sum::<f32>();
        assert!(
            (reconstructed - out.data[column]).abs() < 1.0e-6,
            "weighted expert checkpoints must reconstruct MoE output column {column}"
        );
    }
}

#[test]
fn q8_0_residency_report_counts_resident_blocks_and_flags_file_backed() {
    let shape = TensorShape { dims: vec![2, 32] };
    let blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [0; 32],
        };
        2
    ];
    let resident = CpuTensor::from_q8_0_blocks("resident.weight", shape.clone(), blocks).unwrap();
    let file_backed = CpuTensor::q8_0_file_backed_linear(
        "lazy.weight",
        shape,
        Q8_0FileBacking::new(std::path::PathBuf::from("/nonexistent.gguf"), 0, 2),
    );
    let placeholder = CpuTensor::from_f32("placeholder", vec![0], vec![]).unwrap();

    let weights = LlamaLoadedWeights {
        token_embedding: resident,
        output_norm: placeholder,
        output: Some(file_backed),
        rope_freqs: None,
        layers: Vec::new(),
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
    };

    let report = weights.q8_0_residency_report();
    assert_eq!(report.resident_tensors, 1);
    assert_eq!(
        report.resident_block_bytes,
        (2 * std::mem::size_of::<Q8_0Block>()) as u64
    );
    assert_eq!(report.violations.len(), 1);
    assert!(
        report.violations[0].contains("lazy.weight")
            && report.violations[0].contains("file-backed"),
        "{:?}",
        report.violations
    );
}

#[test]
fn resident_prefill_rope_tables_match_per_position_builder() {
    // Llama3-scaled config so the batched builder exercises the smooth-factor path the
    // 3B row actually uses; the claim is bit-identical tables, not approximate ones.
    let config = LlamaModelConfig {
        context_length: 64,
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(8),
        rope_freq_base: Some(500_000.0),
        rope_scaling_type: Some("llama3".to_string()),
        rope_scaling_factor: Some(32.0),
        rope_scaling_original_context_length: Some(8192),
        rope_scaling_low_freq_factor: Some(1.0),
        rope_scaling_high_freq_factor: Some(4.0),
        rms_norm_epsilon: 1e-6,
        vocab_size: None,
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    };
    let n = 7;
    let head_dim = 8;
    let tables = rope::resident_prefill_rope_tables(n, head_dim, &config, None)
        .unwrap()
        .expect("batched tables");
    let (cos_all, sin_all, split_half) = (tables.cos, tables.sin, tables.split_half_pairing);
    let half = head_dim / 2;
    assert_eq!(cos_all.len(), n * half);
    assert_eq!(sin_all.len(), n * half);
    for pos in 0..n {
        let t = rope::resident_decode_rope_tables(pos, head_dim, &config, None)
            .unwrap()
            .expect("per-position tables");
        assert_eq!(
            &cos_all[pos * half..(pos + 1) * half],
            &t.cos[..],
            "cos pos {pos}"
        );
        assert_eq!(
            &sin_all[pos * half..(pos + 1) * half],
            &t.sin[..],
            "sin pos {pos}"
        );
        assert_eq!(split_half, t.split_half_pairing);
    }
}

/// Q4_0 wire dot: the scalar and NEON paths must agree bit-exactly with each
/// other and track a plain f32 dequantÂ·f32 reference closely (the integer dot
/// is exact per block; only the per-block f32 scale accumulate rounds).
#[test]
fn q4_0_wire_row_dot_scalar_matches_dequant_reference() {
    // Deterministic synthetic row: 4 blocks = 128 weights.
    let blocks = 4usize;
    let mut wire = Vec::with_capacity(blocks * super::Q4_0_WIRE_BYTES_PER_BLOCK);
    let mut expected_weights = Vec::with_capacity(blocks * 32);
    for b in 0..blocks {
        let scale = 0.0125f32 * (b as f32 + 1.0);
        let scale_f16 = super::f32_to_f16_bits(scale);
        wire.extend_from_slice(&scale_f16.to_le_bytes());
        let scale_back = super::f16_bits_to_f32(scale_f16);
        let mut nibbles = [0u8; 16];
        for (j, nib) in nibbles.iter_mut().enumerate() {
            let lo = ((b * 7 + j * 3) % 16) as u8;
            let hi = ((b * 11 + j * 5) % 16) as u8;
            *nib = (hi << 4) | lo;
        }
        wire.extend_from_slice(&nibbles);
        for nib in &nibbles {
            expected_weights.push(((nib & 0x0F) as i32 - 8) as f32 * scale_back);
        }
        for nib in &nibbles {
            expected_weights.push(((nib >> 4) as i32 - 8) as f32 * scale_back);
        }
    }

    let activation: Vec<f32> = (0..blocks * 32)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let xq = super::quantize_q8_0_blocks(&activation);

    // Reference: dequantized weights Ã— dequantized activation, block-sequential.
    let mut reference = 0f32;
    for (b, xb) in xq.iter().enumerate() {
        let mut isum = 0i32;
        for j in 0..32 {
            let w = expected_weights[b * 32 + j];
            let wq =
                (w / super::f16_bits_to_f32(super::f32_to_f16_bits(0.0125f32 * (b as f32 + 1.0))))
                    .round() as i32;
            isum += wq * (xb.quants[j] as i32);
        }
        reference += isum as f32
            * super::f16_bits_to_f32(super::f32_to_f16_bits(0.0125f32 * (b as f32 + 1.0)))
            * xb.scale;
    }

    let scalar = super::q4_0_wire_row_dot_scalar(&wire, &xq);
    assert!(
        (scalar - reference).abs() <= reference.abs() * 1e-6 + 1e-6,
        "scalar {scalar} vs reference {reference}"
    );

    let dispatched = super::q4_0_wire_row_dot(&wire, &xq);
    assert_eq!(
        dispatched.to_bits(),
        scalar.to_bits(),
        "NEON and scalar q4_0 dots must agree bit-exactly: {dispatched} vs {scalar}"
    );
}

/// The interleaved 8-row Q4_0 GEMV (both the packed-scalar reference and the
/// AVX2/dispatch path) must be BIT-EXACT to `q4_0_wire_row_dot_scalar` run over
/// each of the eight rows. Repacking is a layout change, not a math change.
#[test]
fn q4_0_packed_gemv8_matches_scalar_bit_exact() {
    use crate::tensor::Q4_0PackedRows8;

    let blocks_per_row = 6usize; // 192 weights per row
    let rows = 8usize;

    // Deterministic pseudo-random wire bytes for 8 rows.
    let mut wire = vec![0u8; rows * blocks_per_row * super::Q4_0_WIRE_BYTES_PER_BLOCK];
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for b in wire.chunks_exact_mut(super::Q4_0_WIRE_BYTES_PER_BLOCK) {
        let scale = 0.005f32 + (next() % 97) as f32 * 0.001f32;
        b[0..2].copy_from_slice(&super::f32_to_f16_bits(scale).to_le_bytes());
        for byte in &mut b[2..18] {
            *byte = (next() & 0xFF) as u8;
        }
    }

    // Random activation row.
    let activation: Vec<f32> = (0..blocks_per_row * 32)
        .map(|i| ((i as f32) * 0.19 + 0.3).sin() * 4.0 - 1.0)
        .collect();
    let xq = super::quantize_q8_0_blocks(&activation);

    // Per-row scalar oracle.
    let row_bytes = blocks_per_row * super::Q4_0_WIRE_BYTES_PER_BLOCK;
    let mut oracle = [0f32; 8];
    for (r, o) in oracle.iter_mut().enumerate() {
        *o = super::q4_0_wire_row_dot_scalar(&wire[r * row_bytes..(r + 1) * row_bytes], &xq);
    }

    let packed = Q4_0PackedRows8::from_q4_0_bytes(rows, blocks_per_row, &wire).unwrap();

    let mut packed_scalar = [0f32; 8];
    super::q4_0_packed_gemv8_scalar(&packed, 0, &xq, &mut packed_scalar);

    let mut dispatched = [0f32; 8];
    super::q4_0_packed_gemv8(&packed, 0, &xq, &mut dispatched);

    for r in 0..8 {
        assert_eq!(
            packed_scalar[r].to_bits(),
            oracle[r].to_bits(),
            "packed-scalar row {r}: {} vs oracle {}",
            packed_scalar[r],
            oracle[r]
        );
        assert_eq!(
            dispatched[r].to_bits(),
            oracle[r].to_bits(),
            "dispatched (AVX2) row {r}: {} vs oracle {}",
            dispatched[r],
            oracle[r]
        );
    }
}

#[test]
fn q4_0_wire_block_dequant_matches_nibble_layout() {
    let scale = 0.5f32;
    let mut block = Vec::new();
    block.extend_from_slice(&super::f32_to_f16_bits(scale).to_le_bytes());
    let mut nibbles = [0u8; 16];
    for (j, nib) in nibbles.iter_mut().enumerate() {
        *nib = (((j % 16) as u8) << 4) | ((15 - j % 16) as u8);
    }
    block.extend_from_slice(&nibbles);
    let out = super::q4_0_wire_block_dequant(&block);
    for j in 0..16 {
        assert_eq!(out[j], ((15 - j as i32 % 16) - 8) as f32 * scale, "lo {j}");
        assert_eq!(out[j + 16], ((j as i32 % 16) - 8) as f32 * scale, "hi {j}");
    }
}

/// Q6_K: the wire-row dot must equal the dequant-array reference computed
/// through the same integer path (weights rebuild exactly; the dot's lane
/// structure is the reference generic kernel's).
#[test]
fn q6_k_wire_dot_consistent_with_dequant() {
    // One synthetic 210-byte superblock with full nibble/2-bit/scale coverage.
    let mut block = vec![0u8; super::Q6_K_WIRE_BYTES_PER_BLOCK];
    for (i, b) in block.iter_mut().enumerate().take(128) {
        *b = ((i * 37 + 11) % 256) as u8; // ql
    }
    for i in 0..64 {
        block[128 + i] = ((i * 73 + 5) % 256) as u8; // qh
    }
    for i in 0..16 {
        block[192 + i] = ((i as i32 * 9 - 60) & 0xFF) as u8; // signed scales
    }
    let d = 0.0375f32;
    block[208..210].copy_from_slice(&super::f32_to_f16_bits(d).to_le_bytes());

    let activation: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.21).cos() * 4.0).collect();
    let xq = super::quantize_q8_k_blocks(&activation);
    assert_eq!(xq.len(), 1);

    let dot = super::q6_k_wire_row_dot(&block, &xq);

    // Reference: same integer math via the dequant array (w = d*sc*q exactly,
    // so dividing back out per group is exact in i32 range).
    let deq = super::q6_k_wire_block_dequant(&block);
    let d_back = super::f16_bits_to_f32(super::f32_to_f16_bits(d));
    let mut sums = [0f32; 8];
    let mut aux32 = [0i32; 8];
    for j in 0..16 {
        let scale = block[192 + j] as i8 as i32;
        let off = j * 16;
        for l in 0..16 {
            let w = deq[off + l];
            let q = if scale == 0 || d_back == 0.0 {
                0
            } else {
                (w / (d_back * scale as f32)).round() as i32
            };
            aux32[l % 8] += scale * (xq[0].qs[off + l] as i32) * q;
        }
    }
    for l in 0..8 {
        sums[l] += d_back * xq[0].d * aux32[l] as f32;
    }
    let reference: f32 = sums.iter().sum();
    assert!(
        (dot - reference).abs() <= reference.abs() * 1e-5 + 1e-4,
        "q6_k dot {dot} vs dequant reference {reference}"
    );
}

/// Phase-2 follow-up parity gate: the opt-in AVX2 Q6_K row dot must be BIT-IDENTICAL
/// to the 8-lane scalar oracle `q6_k_wire_row_dot` (not merely close) â€” it vectorizes
/// only the associative integer dot and reproduces the same 8-lane f32 reduction. This
/// is the proof obligation that lets `CAMELID_X86_Q6K_AVX2` ship without a parity risk.
#[cfg(target_arch = "x86_64")]
#[test]
fn q6_k_wire_row_dot_avx2_bit_identical() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("skipping: avx2 not available");
        return;
    }
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const WIRE: usize = super::Q6_K_WIRE_BYTES_PER_BLOCK;
    for nblk in [1usize, 2, 5, 11] {
        let mut wire = vec![0u8; nblk * WIRE];
        for b in wire.iter_mut() {
            *b = (next() & 0xff) as u8;
        }
        let mut blocks = Vec::with_capacity(nblk);
        for _ in 0..nblk {
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = (next() & 0xff) as u8 as i8;
            }
            let d = (next() % 1000) as f32 / 333.0 + 0.001;
            blocks.push(super::Q8KBlock { d, qs });
        }
        let s = super::q6_k_wire_row_dot(&wire, &blocks);
        let v = unsafe { super::q6_k_wire_row_dot_avx2(&wire, &blocks) };
        assert_eq!(
            s.to_bits(),
            v.to_bits(),
            "q6_k 8-lane avx2 != scalar at nblk={nblk}: scalar={s} avx2={v}"
        );
    }
}

/// Q4_K: the wire-row dot must agree with an f64 dot of the tensor-layer
/// dequant (`Q4KBlock::dequantize`, an independent implementation of the same
/// format) against the dequantized Q8_K activations. The integer kernel and
/// the dequant dot compute the same sum in different float orders, so the
/// tolerance covers f32-vs-f64 accumulation only. (DiffusionGemma lane.)
#[test]
fn q4_k_wire_dot_consistent_with_tensor_dequant() {
    let blocks = 3usize;
    let mut wire = vec![0u8; blocks * super::Q4_K_WIRE_BYTES_PER_BLOCK];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 131 + 17) % 256) as u8;
    }
    // sane f16 scales: d at +0, dmin at +2 of each superblock
    for blk in wire.chunks_exact_mut(super::Q4_K_WIRE_BYTES_PER_BLOCK) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.0173).to_le_bytes());
        blk[2..4].copy_from_slice(&super::f32_to_f16_bits(0.0049).to_le_bytes());
    }

    let activation: Vec<f32> = (0..blocks * 256)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let xq = super::quantize_q8_k_blocks(&activation);

    let dot = super::q4_k_wire_row_dot(&wire, &xq);

    let decoded = crate::tensor::decode_q4_k_blocks(&wire).expect("decode q4_k blocks");
    let mut reference = 0f64;
    let mut vals = [0f32; 256];
    for (bi, block) in decoded.iter().enumerate() {
        block.dequantize(&mut vals);
        let y = &xq[bi];
        for (l, &w) in vals.iter().enumerate() {
            reference += w as f64 * (y.d as f64 * y.qs[l] as f64);
        }
    }
    assert!(
        (dot as f64 - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
        "q4_k dot {dot} vs tensor dequant reference {reference}"
    );
}

/// Q5_K: the wire-row dot must agree with an f64 dot of the tensor-layer
/// dequant (`Q5KBlock::dequantize`, an independent implementation of the same
/// format via `q4_k_scale_min` rather than the kmask scheme) against the
/// dequantized Q8_K activations — the same cross-check as
/// `q4_k_wire_dot_consistent_with_tensor_dequant`, exercising the added qh fifth bit.
#[test]
fn q5_k_wire_dot_consistent_with_tensor_dequant() {
    let blocks = 3usize;
    let mut wire = vec![0u8; blocks * super::Q5_K_WIRE_BYTES_PER_BLOCK];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 131 + 17) % 256) as u8;
    }
    // sane f16 scales: d at +0, dmin at +2 of each superblock
    for blk in wire.chunks_exact_mut(super::Q5_K_WIRE_BYTES_PER_BLOCK) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.0173).to_le_bytes());
        blk[2..4].copy_from_slice(&super::f32_to_f16_bits(0.0049).to_le_bytes());
    }

    let activation: Vec<f32> = (0..blocks * 256)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let xq = super::quantize_q8_k_blocks(&activation);

    let dot = super::q5_k_wire_row_dot(&wire, &xq);

    let decoded = crate::tensor::decode_q5_k_blocks(&wire).expect("decode q5_k blocks");
    let mut reference = 0f64;
    let mut vals = [0f32; 256];
    for (bi, block) in decoded.iter().enumerate() {
        block.dequantize(&mut vals);
        let y = &xq[bi];
        for (l, &w) in vals.iter().enumerate() {
            reference += w as f64 * (y.d as f64 * y.qs[l] as f64);
        }
    }
    assert!(
        (dot as f64 - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
        "q5_k dot {dot} vs tensor dequant reference {reference}"
    );
}

/// IQ4_XS: the streaming wire-row dot must equal the tensor-layer decode dotted against
/// the SAME Q8_K-dequantised activation. `iq4_xs_wire_row_dot` accumulates integers per
/// sub-block; the reference decodes to f32 and dots in f64 — they agree within f32 rounding.
#[test]
fn iq4_xs_wire_dot_consistent_with_tensor_dequant() {
    let blocks = 3usize;
    let mut wire = vec![0u8; blocks * crate::tensor::IQ4_XS_BLOCK_BYTES];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 131 + 17) % 256) as u8;
    }
    // Sane f16 super-block scale `d` at byte offset 0 of each super-block (leave scales_h,
    // scales_l, qs as the pseudo-random fill to exercise all 8 sub-block scales + codebook).
    for blk in wire.chunks_exact_mut(crate::tensor::IQ4_XS_BLOCK_BYTES) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.0173).to_le_bytes());
    }

    let activation: Vec<f32> = (0..blocks * 256)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let xq = super::quantize_q8_k_blocks(&activation);

    let dot = super::iq4_xs_wire_row_dot(&wire, &xq);

    let decoded = crate::tensor::decode_iq4_xs_blocks(&wire).expect("decode iq4_xs blocks");
    let mut reference = 0f64;
    let mut vals = [0f32; 256];
    for (bi, block) in decoded.iter().enumerate() {
        block.dequantize(&mut vals);
        let y = &xq[bi];
        for (l, &w) in vals.iter().enumerate() {
            reference += w as f64 * (y.d as f64 * y.qs[l] as f64);
        }
    }
    assert!(
        (dot as f64 - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
        "iq4_xs dot {dot} vs tensor dequant reference {reference}"
    );
}

/// IQ4_XS prefill linear: `iq4_xs_block_dot_core` (multi-row, rayon over output) must equal
/// the per-(row, output) `iq4_xs_wire_row_dot` oracle exactly — same kernel, so the tiling
/// is bit-identical, not merely close.
#[test]
fn iq4_xs_block_dot_core_matches_row_dot_oracle() {
    let (in_dim, out_dim, n_rows) = (512usize, 5usize, 4usize);
    let sb = in_dim / 256;
    let mut wire = vec![0u8; out_dim * sb * crate::tensor::IQ4_XS_BLOCK_BYTES];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 191 + 7) % 256) as u8;
    }
    for blk in wire.chunks_exact_mut(crate::tensor::IQ4_XS_BLOCK_BYTES) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.021).to_le_bytes());
    }
    let input_data: Vec<f32> = (0..n_rows * in_dim)
        .map(|i| ((i as f32) * 0.017).cos() * 2.5)
        .collect();
    let input = super::CpuTensor::from_f32("in", vec![n_rows, in_dim], input_data.clone()).unwrap();

    let out = super::iq4_xs_block_dot_core(&input, &wire, out_dim, in_dim, "iq4xs").unwrap();
    assert_eq!(out.shape.dims, vec![n_rows, out_dim]);

    let row_bytes = sb * crate::tensor::IQ4_XS_BLOCK_BYTES;
    for r in 0..n_rows {
        let xq = super::quantize_q8_k_blocks(&input_data[r * in_dim..(r + 1) * in_dim]);
        for o in 0..out_dim {
            let w_row = &wire[o * row_bytes..(o + 1) * row_bytes];
            let want = super::iq4_xs_wire_row_dot(w_row, &xq);
            assert_eq!(
                out.data[r * out_dim + o].to_bits(),
                want.to_bits(),
                "row {r} col {o}"
            );
        }
    }
}

/// IQ4_XS decode (single-row) path must equal the first row of the prefill core.
#[test]
fn iq4_xs_decode_row_matches_block_dot_core() {
    let (in_dim, out_dim) = (256usize, 6usize);
    let sb = in_dim / 256;
    let mut wire = vec![0u8; out_dim * sb * crate::tensor::IQ4_XS_BLOCK_BYTES];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 53 + 29) % 256) as u8;
    }
    for blk in wire.chunks_exact_mut(crate::tensor::IQ4_XS_BLOCK_BYTES) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.031).to_le_bytes());
    }
    let input_row: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.05).sin()).collect();
    let input = super::CpuTensor::from_f32("in", vec![1, in_dim], input_row.clone()).unwrap();

    let prefill = super::iq4_xs_block_dot_core(&input, &wire, out_dim, in_dim, "iq4xs").unwrap();
    let mut decode = vec![0f32; out_dim];
    super::accumulate_transposed_linear_row_iq4_xs(&input_row, &wire, &mut decode);
    for (o, &d) in decode.iter().enumerate() {
        assert_eq!(prefill.data[o].to_bits(), d.to_bits(), "col {o}");
    }
}

/// Real-weight Q5_K parity: for each 2-D Q5_K linear in a downloaded Q5_K_M GGUF,
/// the CPU block-dot (`q5_k_block_dot_core`, which quantises the activation to Q8_K)
/// must match an independent f32 reference — the tensor-layer decoder
/// (`decode_q5_k_tensor`, a different scale-unpack path) dotted against the SAME
/// Q8_K-dequantised activation. Same methodology as the synthetic Q4_K/Q5_K unit
/// tests, but on real model weights, exercising the load + block-dot wiring.
/// Skips unless `CAMELID_Q5KM_GGUF` points to a `*-Q5_K_M.gguf`.
#[test]
fn q5_k_block_dot_matches_decode_on_real_model() {
    let Some(path) = std::env::var_os("CAMELID_Q5KM_GGUF") else {
        eprintln!(
            "skipping q5_k block-dot real-model parity: set CAMELID_Q5KM_GGUF to a *-Q5_K_M.gguf"
        );
        return;
    };
    let path = std::path::PathBuf::from(path);
    let gguf = crate::gguf::read_metadata(&path).expect("read gguf metadata");
    let store = crate::tensor::TensorStore::open(&path, &gguf);

    // Deterministic activation in [-1, 1) (xorshift; no RNG dependency).
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };

    let mut tested = 0usize;
    for desc in &gguf.tensors {
        if desc.tensor_type != crate::gguf::GgufTensorType::Q5K || desc.dimensions.len() != 2 {
            continue;
        }
        // GGUF linear: dimensions[0] = contraction (in), dimensions[1] = output rows.
        let in_dim = desc.dimensions[0] as usize;
        let out_dim = desc.dimensions[1] as usize;
        if !in_dim.is_multiple_of(256) {
            continue;
        }
        let wire = store.tensor_bytes(&desc.name).expect("wire bytes");
        let f32w = crate::tensor::decode_q5_k_tensor(&desc.name, &wire, in_dim * out_dim)
            .expect("decode q5_k tensor");

        let n_rows = 2usize;
        let input_data: Vec<f32> = (0..n_rows * in_dim).map(|_| next_f32()).collect();
        let input =
            crate::tensor::CpuTensor::from_f32("q5k_in", vec![n_rows, in_dim], input_data.clone())
                .expect("input tensor");

        let out_bd = super::q5_k_block_dot_core(&input, &wire, out_dim, in_dim, "q5k_bd")
            .expect("q5_k block dot");

        for r in 0..n_rows {
            let xq = super::quantize_q8_k_blocks(&input_data[r * in_dim..(r + 1) * in_dim]);
            for o in 0..out_dim {
                let mut reference = 0f64;
                for (blk, y) in xq.iter().enumerate() {
                    for l in 0..256 {
                        let k = blk * 256 + l;
                        reference += f32w[o * in_dim + k] as f64 * (y.d as f64 * y.qs[l] as f64);
                    }
                }
                let got = out_bd.data[r * out_dim + o] as f64;
                assert!(
                    (got - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
                    "q5_k block-dot mismatch in {} row {r} out {o}: got {got} ref {reference}",
                    desc.name
                );
            }
        }
        tested += 1;
        if tested >= 3 {
            break;
        }
    }
    assert!(tested > 0, "no 2-D Q5_K linears found in {path:?}");
}

/// Real-weight IQ4_XS parity: for each 2-D IQ4_XS linear in a downloaded GGUF, the CPU
/// streaming block-dot (`iq4_xs_block_dot_core`, which quantises the activation to Q8_K)
/// must match an independent f32 reference — the tensor-layer decoder (`decode_iq4_xs_tensor`)
/// dotted against the SAME Q8_K-dequantised activation. Exercises the real load + block-dot
/// wiring on real model weights. Skips unless `CAMELID_IQ4XS_GGUF` points to an IQ4_XS GGUF.
#[test]
fn iq4_xs_block_dot_matches_decode_on_real_model() {
    let Some(path) = std::env::var_os("CAMELID_IQ4XS_GGUF") else {
        eprintln!(
            "skipping iq4_xs block-dot real-model parity: set CAMELID_IQ4XS_GGUF to an IQ4_XS gguf"
        );
        return;
    };
    let path = std::path::PathBuf::from(path);
    let gguf = crate::gguf::read_metadata(&path).expect("read gguf metadata");
    let store = crate::tensor::TensorStore::open(&path, &gguf);

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };

    let mut tested = 0usize;
    for desc in &gguf.tensors {
        if desc.tensor_type != crate::gguf::GgufTensorType::IQ4XS || desc.dimensions.len() != 2 {
            continue;
        }
        let in_dim = desc.dimensions[0] as usize;
        let out_dim = desc.dimensions[1] as usize;
        if !in_dim.is_multiple_of(256) {
            continue;
        }
        let wire = store.tensor_bytes(&desc.name).expect("wire bytes");
        let f32w = crate::tensor::decode_iq4_xs_tensor(&desc.name, &wire, in_dim * out_dim)
            .expect("decode iq4_xs tensor");

        let n_rows = 2usize;
        let input_data: Vec<f32> = (0..n_rows * in_dim).map(|_| next_f32()).collect();
        let input =
            crate::tensor::CpuTensor::from_f32("iq4_in", vec![n_rows, in_dim], input_data.clone())
                .expect("input tensor");

        let out_bd = super::iq4_xs_block_dot_core(&input, &wire, out_dim, in_dim, "iq4_bd")
            .expect("iq4_xs block dot");

        for r in 0..n_rows {
            let xq = super::quantize_q8_k_blocks(&input_data[r * in_dim..(r + 1) * in_dim]);
            for o in 0..out_dim {
                let mut reference = 0f64;
                for (blk, y) in xq.iter().enumerate() {
                    for l in 0..256 {
                        let k = blk * 256 + l;
                        reference += f32w[o * in_dim + k] as f64 * (y.d as f64 * y.qs[l] as f64);
                    }
                }
                let got = out_bd.data[r * out_dim + o] as f64;
                assert!(
                    (got - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
                    "iq4_xs block-dot mismatch in {} row {r} out {o}: got {got} ref {reference}",
                    desc.name
                );
            }
        }
        tested += 1;
        if tested >= 3 {
            break;
        }
    }
    assert!(tested > 0, "no 2-D IQ4_XS linears found in {path:?}");
}

/// Q5_0: the wire-row dot must agree with an f64 dot of the tensor-layer
/// dequant (`Q5_0Block`) against the dequantized Q8_0 activations.
/// (DiffusionGemma lane.)
#[test]
fn q5_0_wire_dot_consistent_with_tensor_dequant() {
    let blocks = 5usize;
    let mut wire = vec![0u8; blocks * super::Q5_0_WIRE_BYTES_PER_BLOCK];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = ((i * 89 + 41) % 256) as u8;
    }
    for blk in wire.chunks_exact_mut(super::Q5_0_WIRE_BYTES_PER_BLOCK) {
        blk[0..2].copy_from_slice(&super::f32_to_f16_bits(0.031).to_le_bytes());
    }

    let activation: Vec<f32> = (0..blocks * 32)
        .map(|i| ((i as f32) * 0.83).cos() * 2.0)
        .collect();
    let xq = super::quantize_q8_0_blocks(&activation);

    let dot = super::q5_0_wire_row_dot(&wire, &xq);

    let decoded = crate::tensor::decode_q5_0_blocks(&wire).expect("decode q5_0 blocks");
    let mut reference = 0f64;
    for (bi, block) in decoded.iter().enumerate() {
        let scale = block.scale_f32();
        let y = &xq[bi];
        for (l, &q) in block.unpack_values().iter().enumerate() {
            reference += (scale * q as f32) as f64 * (y.scale as f64 * y.quants[l] as f64);
        }
    }
    assert!(
        (dot as f64 - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
        "q5_0 dot {dot} vs tensor dequant reference {reference}"
    );
}

/// The Q8_K quantizer must mirror the reference exactly: iscale uses the
/// SIGNED max (not amax), magic-number nearest-even rounding, clamp to 127.
#[test]
fn q8_k_quantizer_mirrors_reference_semantics() {
    // Signed-max behavior: a negative max flips iscale's sign.
    let mut row = vec![0f32; 256];
    row[0] = -10.0; // amax element is negative
    row[1] = 4.0;
    let q = super::quantize_q8_k_blocks(&row);
    // iscale = -127 / -10 = 12.7; d = 1/iscale
    assert!((q[0].d - (1.0 / 12.7)).abs() < 1e-6);
    // qs[0] = nearest_int(12.7 * -10.0) = -127 (the signed-max element).
    assert_eq!(q[0].qs[0], -127);
    assert_eq!(q[0].qs[1], 51); // nearest_int(12.7*4.0) = nearest_int(50.8) = 51

    // All-zero block short-circuits.
    let z = super::quantize_q8_k_blocks(&vec![0f32; 256]);
    assert_eq!(z[0].d, 0.0);
    assert!(z[0].qs.iter().all(|&v| v == 0));
}

#[test]
#[ignore] // perf micro-bench, run explicitly: cargo test --release perf_q4_q8_q6_dot -- --ignored --nocapture
fn perf_q4_q8_q6_dot() {
    use std::time::Instant;
    let in_dim = 2560usize;
    let nblk = in_dim / 32; // 80
    let act: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.013).sin()).collect();
    let xq = super::quantize_q8_0_blocks(&act);
    // Q8_0 weight row: 34 bytes/block
    let mut q8row: Vec<u8> = (0..nblk * 34).map(|i| (i % 251) as u8).collect();
    for b in 0..nblk {
        q8row[b * 34] = 0x66;
        q8row[b * 34 + 1] = 0x2e;
    } // valid f16 ~0.1 scale
      // Q4_0 weight row: 18 bytes/block
    let mut q4row: Vec<u8> = (0..nblk * 18).map(|i| (i % 251) as u8).collect();
    for b in 0..nblk {
        q4row[b * 18] = 0x66;
        q4row[b * 18 + 1] = 0x2e;
    }
    // Q6_K: 210 bytes / 256-block; in_dim 2560 = 10 superblocks
    let q6blk = in_dim / 256;
    let mut q6row: Vec<u8> = (0..q6blk * 210).map(|i| (i % 251) as u8).collect();
    for b in 0..q6blk {
        q6row[b * 210 + 208] = 0x66;
        q6row[b * 210 + 209] = 0x2e;
    } // f16 super-scale
    let xqk = super::quantize_q8_k_blocks(&act);
    let iters = 200_000usize;
    let t = Instant::now();
    let mut s = 0f32;
    for _ in 0..iters {
        s += super::q8_0_wire_row_dot(&q8row, &xq);
    }
    let q8 = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..iters {
        s += super::q4_0_wire_row_dot(&q4row, &xq);
    }
    let q4 = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..iters {
        s += super::q6_k_wire_row_dot(&q6row, &xqk);
    }
    let q6 = t.elapsed().as_secs_f64();
    eprintln!("[dotbench] {iters} iters @ in_dim {in_dim}: q8={:.3}s ({:.1} Mrow/s)  q4={:.3}s ({:.1} Mrow/s, {:.1}x q8)  q6={:.3}s ({:.1} Mrow/s, {:.1}x q8)  sink={s}",
        q8, iters as f64/q8/1e6, q4, iters as f64/q4/1e6, q4/q8, q6, iters as f64/q6/1e6, q6/q8);
}

/// Build a 1-layer `LlamaLoadedWeights` for the CUDA arch-guard test, with per-head
/// QK-norm tensors present (`qk_norm = true`, i.e. a Qwen3-shaped row) or absent
/// (`false`, a plain Llama row). All other tensors are trivial 2Ã—2 placeholders; the
/// guard only inspects `attention_q_norm` / `attention_k_norm`.
#[cfg(feature = "cuda")]
fn minimal_weights_with_qk_norm(qk_norm: bool) -> LlamaLoadedWeights {
    let t = |name: &str, shape: Vec<usize>, n: usize| {
        CpuTensor::from_f32(name, shape, vec![1.0; n]).unwrap()
    };
    let (q_norm, k_norm) = if qk_norm {
        (
            Some(t("blk.0.attn_q_norm.weight", vec![2], 2)),
            Some(t("blk.0.attn_k_norm.weight", vec![2], 2)),
        )
    } else {
        (None, None)
    };
    LlamaLoadedWeights {
        token_embedding: t("token_embd.weight", vec![2, 2], 4),
        output_norm: t("output_norm.weight", vec![2], 2),
        output: None,
        rope_freqs: None,
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
        layers: vec![LlamaLayerWeights {
            attention_norm: t("blk.0.attn_norm.weight", vec![2], 2),
            attention_q: t("blk.0.attn_q.weight", vec![2, 2], 4),
            attention_k: t("blk.0.attn_k.weight", vec![2, 2], 4),
            attention_v: t("blk.0.attn_v.weight", vec![2, 2], 4),
            attention_output: t("blk.0.attn_output.weight", vec![2, 2], 4),
            attention_biases: None,
            attention_q_norm: q_norm,
            attention_k_norm: k_norm,
            ffn_norm: t("blk.0.ffn_norm.weight", vec![2], 2),
            ffn_gate: t("blk.0.ffn_gate.weight", vec![2, 2], 4),
            ffn_up: t("blk.0.ffn_up.weight", vec![2, 2], 4),
            ffn_down: t("blk.0.ffn_down.weight", vec![2, 2], 4),
            moe_router: None,
            mla_q_a_proj: None,
            mla_q_a_layernorm: None,
            mla_q_b_proj: None,
            mla_kv_a_proj_with_mqa: None,
            mla_kv_a_layernorm: None,
            mla_kv_b_proj: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            decode_bindings: DecodeLinearBindings::default(),
        }],
    }
}

/// Loading a Qwen3 GGUF (per-head QK-norm present) on the CUDA resident path must fail
/// closed with the typed `UnsupportedModelArchitecture` error: the engine has no
/// QK-norm kernel, so running it would silently feed un-normalized Q/K into RoPE. A
/// Qwen3 per-head QK-norm is now supported by the CUDA resident decode engine.
/// This test verifies the engine *accepts* models with QK-norm (the guard was removed).
#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_accepts_qwen3_qk_norm() {
    // With the QK-norm kernel ported, models with attention_q_norm/attention_k_norm
    // tensors should no longer be rejected. The guard function has been removed,
    // so this test just documents that the architecture is now supported.
    let qwen3 = minimal_weights_with_qk_norm(true);
    assert!(
        qwen3.layers.iter().any(|l| l.attention_q_norm.is_some()),
        "test fixture should have QK-norm weights"
    );
}

/// Item 2 bitwise lock: the decode attention head-parallel lane is a
/// scheduling-only change, so its full attention output must be bit-identical
/// to the serial loop for every GQA shape, head_dim, position count, dot
/// lane, and random content â€” and identical across repeated parallel runs.
#[test]
fn decode_attention_parallel_lane_is_bitwise_identical_to_serial() {
    struct XorShift64Star(u64);
    impl XorShift64Star {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Finite f32 spanning subnormals through +-1e12, both signs.
        fn next_f32(&mut self) -> f32 {
            let unit = ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match self.next_u64() % 8 {
                0 => unit * f32::MIN_POSITIVE * 0.5,
                1 => unit * 1e-12,
                2..=4 => unit * 2.0,
                5 => unit * 1e4,
                _ => unit * 1e12,
            }
        }
        fn fill(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.next_f32()).collect()
        }
    }

    let gqa_shapes = [(32usize, 4usize), (24, 8), (32, 8)];
    let head_dims = [64usize, 128];
    let position_counts = [2usize, 63, 64, 65, 511, 2048];
    let cases_per_point = 100u64;
    #[cfg(target_arch = "x86_64")]
    let dot_lanes: &[bool] = if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        &[false, true]
    } else {
        &[false]
    };
    #[cfg(not(target_arch = "x86_64"))]
    let dot_lanes: &[bool] = &[false];

    let mut points = Vec::new();
    for &(attention_heads, kv_heads) in &gqa_shapes {
        for &head_dim in &head_dims {
            for &position_count in &position_counts {
                for case in 0..cases_per_point {
                    points.push((attention_heads, kv_heads, head_dim, position_count, case));
                }
            }
        }
    }

    // The case grid is large (36 shape points x 100 cases x up to 2048
    // positions); spread cases across the pool so the suite stays fast. Any
    // scheduling sensitivity this introduces is exactly what the assertion
    // must be immune to.
    let mismatches: usize = points
        .par_iter()
        .map(
            |&(attention_heads, kv_heads, head_dim, position_count, case)| {
                let mut rng = XorShift64Star(
                    0x1770_0001
                        ^ ((attention_heads as u64) << 48)
                        ^ ((kv_heads as u64) << 40)
                        ^ ((head_dim as u64) << 24)
                        ^ ((position_count as u64) << 8)
                        ^ case,
                );
                let plan = LlamaKvCachePlan {
                    max_sequence_length: position_count,
                    layer_count: 1,
                    kv_head_count: kv_heads,
                    head_dim,
                    k_head_dim: head_dim,
                    v_head_dim: head_dim,
                    key_shape: vec![1, position_count, kv_heads, head_dim],
                    value_shape: vec![1, position_count, kv_heads, head_dim],
                };
                let mut kv_cache = LlamaKvCache::new(plan, crate::model::KvCacheQuantization::F16)
                    .expect("kv cache");
                let kv_len = position_count * kv_heads * head_dim;
                kv_cache.keys = rng.fill(kv_len);
                kv_cache.values = rng.fill(kv_len);
                kv_cache.allocated_sequence_length = position_count;
                kv_cache.position = position_count - 1;
                let query = rng.fill(attention_heads * head_dim);
                let params = DecodeAttentionHeadsParams {
                    kv_cache: &kv_cache,
                    layer_idx: 0,
                    query_data: &query,
                    attention_heads,
                    repeats: attention_heads / kv_heads,
                    kv_heads,
                    head_mapping: GqaHeadMapping::Grouped,
                    position_count,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                };

                let width = attention_heads * head_dim;
                let repeats = attention_heads / kv_heads;
                let mut mismatch = 0usize;

                // Production driver, serial vs parallel, parallel run twice to
                // expose scheduling nondeterminism. The per-head body resolves the
                // dot lane from env (untouched in tests => the legacy lane).
                let mut serial = vec![0.0f32; width];
                decode_attention_all_heads_into_with_mode(&params, &mut serial, false)
                    .expect("serial");
                let mut parallel_a = vec![0.0f32; width];
                decode_attention_all_heads_into_with_mode(&params, &mut parallel_a, true)
                    .expect("parallel run 1");
                let mut parallel_b = vec![0.0f32; width];
                decode_attention_all_heads_into_with_mode(&params, &mut parallel_b, true)
                    .expect("parallel run 2");
                for ((s, a), b) in serial.iter().zip(&parallel_a).zip(&parallel_b) {
                    if s.to_bits() != a.to_bits() || a.to_bits() != b.to_bits() {
                        mismatch += 1;
                    }
                }

                // Dot-lane lock: run every head through the explicit-kernel body
                // with the lane pinned (no process-env writes), serial vs a rayon
                // scope mirroring the production parallel arm, compared bitwise.
                // Covers composition with the Item-1 blocked lane on AVX2 hosts.
                for &use_blocked in dot_lanes {
                    let mut serial_pinned = vec![0.0f32; width];
                    let mut scores = Vec::new();
                    for head in 0..attention_heads {
                        let kv_head = map_attention_head_to_kv_head(
                            head,
                            repeats,
                            kv_heads,
                            params.head_mapping,
                        );
                        attention_context_for_head_into_with_kernels(
                            AttentionContextHeadParams {
                                kv_cache: &kv_cache,
                                layer_idx: 0,
                                kv_head,
                                query_slice: &query[head * head_dim..(head + 1) * head_dim],
                                position_count,
                                scale: params.scale,
                            },
                            &mut serial_pinned[head * head_dim..(head + 1) * head_dim],
                            &mut scores,
                            use_blocked,
                        )
                        .expect("serial pinned");
                    }
                    let mut parallel_pinned = vec![0.0f32; width];
                    parallel_pinned
                        .par_chunks_exact_mut(head_dim)
                        .enumerate()
                        .try_for_each_init(Vec::new, |scratch, (head, out_slice)| {
                            let kv_head = map_attention_head_to_kv_head(
                                head,
                                repeats,
                                kv_heads,
                                params.head_mapping,
                            );
                            attention_context_for_head_into_with_kernels(
                                AttentionContextHeadParams {
                                    kv_cache: &kv_cache,
                                    layer_idx: 0,
                                    kv_head,
                                    query_slice: &query[head * head_dim..(head + 1) * head_dim],
                                    position_count,
                                    scale: params.scale,
                                },
                                out_slice,
                                scratch,
                                use_blocked,
                            )
                        })
                        .expect("parallel pinned");
                    for (s, p) in serial_pinned.iter().zip(&parallel_pinned) {
                        if s.to_bits() != p.to_bits() {
                            mismatch += 1;
                        }
                    }
                }
                mismatch
            },
        )
        .sum();

    assert_eq!(
        mismatches, 0,
        "decode attention parallel lane diverged from serial in {mismatches} element(s)"
    );
}

/// Item 3 Lane A bitwise lock: the head-major KV layout is address math only.
/// (KV content magnitudes cap below the f16 finite max â€” the real write path
/// f16-rounds every stored element, on main too, and inf KV rejects softmax.)
/// For every GQA shape, head_dim, and position count: build one cache per
/// layout, drive them through the REAL write paths (write_kv_cache /
/// write_kv_cache_batch) including growth re-layout (positions > the 256
/// grow chunk), a rollback + divergent overwrite sequence, and the CUDA
/// mirror-back accessor pattern (offset-addressed per (position, head)
/// writes, covering copy_resident_cuda_kv_to_host); then assert (a) logical
/// buffer equality element-by-element via offset(), and (b) bitwise-equal
/// attention outputs in both scheduling modes and both dot lanes.
#[test]
fn kv_head_major_layout_is_bitwise_identical_to_position_major() {
    struct XorShift64Star(u64);
    impl XorShift64Star {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_f32(&mut self) -> f32 {
            let unit = ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match self.next_u64() % 8 {
                0 => unit * f32::MIN_POSITIVE * 0.5,
                1 => unit * 1e-12,
                2..=4 => unit * 2.0,
                5 => unit * 1e4,
                // Cap below the f16 finite max (65504): the REAL write path
                // rounds every stored element through f16, so bigger inputs
                // become Â±inf in the cache (true on main too) and inf KV
                // makes softmax reject the row.
                _ => unit * 6e4,
            }
        }
        fn fill(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.next_f32()).collect()
        }
    }

    let gqa_shapes = [(32usize, 4usize), (24, 8), (32, 8)];
    let head_dims = [64usize, 128];
    let position_counts = [2usize, 63, 64, 65, 511, 2048];
    let cases_per_point = 8u64;

    let mut points = Vec::new();
    for &(attention_heads, kv_heads) in &gqa_shapes {
        for &head_dim in &head_dims {
            for &position_count in &position_counts {
                for case in 0..cases_per_point {
                    points.push((attention_heads, kv_heads, head_dim, position_count, case));
                }
            }
        }
    }

    let mismatches: usize = points
        .par_iter()
        .map(
            |&(attention_heads, kv_heads, head_dim, position_count, case)| {
                let mut rng = XorShift64Star(
                    0x1a1e_0001
                        ^ ((attention_heads as u64) << 48)
                        ^ ((kv_heads as u64) << 40)
                        ^ ((head_dim as u64) << 24)
                        ^ ((position_count as u64) << 8)
                        ^ case,
                );
                let width = kv_heads * head_dim;
                let plan = || LlamaKvCachePlan {
                    // max_sequence_length >= 512 engages the 256-position grow
                    // chunking, so deep points exercise the head-major
                    // re-layout-on-growth path.
                    max_sequence_length: position_count.max(512),
                    layer_count: 2,
                    kv_head_count: kv_heads,
                    head_dim,
                    k_head_dim: head_dim,
                    v_head_dim: head_dim,
                    key_shape: vec![2, position_count.max(512), kv_heads, head_dim],
                    value_shape: vec![2, position_count.max(512), kv_heads, head_dim],
                };
                let mut pm = LlamaKvCache::new_with_layout(plan(), KvLayout::PositionMajor)
                    .expect("pm cache");
                let mut hm =
                    LlamaKvCache::new_with_layout(plan(), KvLayout::HeadMajor).expect("hm cache");
                let mut pm16 = LlamaKvCache::new_with_layout_and_dtype(
                    plan(),
                    KvLayout::PositionMajor,
                    KvDtype::F16,
                )
                .expect("pm16 cache");
                let mut hm16 = LlamaKvCache::new_with_layout_and_dtype(
                    plan(),
                    KvLayout::HeadMajor,
                    KvDtype::F16,
                )
                .expect("hm16 cache");

                // Phase A: append `position_count` tokens through the real
                // single-token write path on layer 0 and the batch path on
                // layer 1 (same data), on BOTH caches.
                let mut token_rows = Vec::with_capacity(position_count);
                for _ in 0..position_count {
                    token_rows.push((rng.fill(width), rng.fill(width)));
                }
                for (p, (k_row, v_row)) in token_rows.iter().enumerate() {
                    let k = CpuTensor::from_f32("k", vec![1, width], k_row.clone()).unwrap();
                    let v = CpuTensor::from_f32("v", vec![1, width], v_row.clone()).unwrap();
                    for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                        cache.position = p;
                        write_kv_cache(cache, 0, &k, &v).expect("write layer0");
                    }
                }
                let flat_k: Vec<f32> = token_rows.iter().flat_map(|(k, _)| k.clone()).collect();
                let flat_v: Vec<f32> = token_rows.iter().flat_map(|(_, v)| v.clone()).collect();
                let bk = CpuTensor::from_f32("bk", vec![position_count, width], flat_k).unwrap();
                let bv = CpuTensor::from_f32("bv", vec![position_count, width], flat_v).unwrap();
                for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                    write_kv_cache_batch(cache, 1, 0, &bk, &bv).expect("batch write");
                    cache.position = position_count - 1;
                }

                // Phase B: rollback to half depth and overwrite the tail with
                // DIFFERENT data (spec-decode shape), through the real path.
                if position_count >= 4 {
                    let half = position_count / 2;
                    for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                        cache.rollback_to_position(half).unwrap();
                    }
                    for p in half..position_count {
                        let k = CpuTensor::from_f32("k2", vec![1, width], rng.fill(width)).unwrap();
                        let v = CpuTensor::from_f32("v2", vec![1, width], rng.fill(width)).unwrap();
                        for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                            cache.position = p;
                            write_kv_cache(cache, 0, &k, &v).expect("overwrite layer0");
                        }
                    }
                    for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                        cache.position = position_count - 1;
                    }
                }

                // Phase C: CUDA mirror-back pattern (copy_resident_cuda_kv_to_host):
                // per-(position, head) stores of f16-exact rows into layer 1
                // through the canonical store, exactly like the fixed site.
                for p in 0..position_count.min(8) {
                    for h in 0..kv_heads {
                        let row: Vec<f32> = (0..head_dim)
                            .map(|_| f16_bits_to_f32(f32_to_f16_bits(rng.next_f32())))
                            .collect();
                        for cache in [&mut pm, &mut hm, &mut pm16, &mut hm16] {
                            cache.store_kv_head_row(1, p, h, &row, &row);
                        }
                    }
                }

                let mut mismatch = 0usize;

                // (a) Logical buffer equality across layouts, every element.
                for layer in 0..2 {
                    for p in 0..position_count {
                        for h in 0..kv_heads {
                            let po = pm.offset(layer, p, h);
                            let ho = hm.offset(layer, p, h);
                            for d in 0..head_dim {
                                if pm.keys[po + d].to_bits() != hm.keys[ho + d].to_bits()
                                    || pm.values[po + d].to_bits() != hm.values[ho + d].to_bits()
                                {
                                    mismatch += 1;
                                }
                            }
                        }
                    }
                }

                // (b) Attention outputs bitwise-equal across layouts, both
                // scheduling modes, both dot lanes (pinned per head, env-free).
                let query = rng.fill(attention_heads * head_dim);
                let scale = 1.0 / (head_dim as f32).sqrt();
                let repeats = attention_heads / kv_heads;
                // is_x86_feature_detected! cannot COMPILE on non-x86 targets
                // (aarch64 macOS CI), so this needs a cfg split, not cfg!().
                #[cfg(target_arch = "x86_64")]
                let dot_lanes: &[bool] = if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    &[false, true]
                } else {
                    &[false]
                };
                #[cfg(not(target_arch = "x86_64"))]
                let dot_lanes: &[bool] = &[false];
                for layer in 0..2 {
                    for &parallel in &[false, true] {
                        let run = |cache: &LlamaKvCache| {
                            let params = DecodeAttentionHeadsParams {
                                kv_cache: cache,
                                layer_idx: layer,
                                query_data: &query,
                                attention_heads,
                                repeats,
                                kv_heads,
                                head_mapping: GqaHeadMapping::Grouped,
                                position_count,
                                scale,
                            };
                            let mut out = vec![0.0f32; attention_heads * head_dim];
                            decode_attention_all_heads_into_with_mode(&params, &mut out, parallel)
                                .expect("attention");
                            out
                        };
                        let a = run(&pm);
                        let b = run(&hm);
                        for (x, y) in a.iter().zip(&b) {
                            if x.to_bits() != y.to_bits() {
                                mismatch += 1;
                            }
                        }
                    }
                    for &use_blocked in dot_lanes {
                        let mut scores = Vec::new();
                        for head in 0..attention_heads {
                            let kv_head = map_attention_head_to_kv_head(
                                head,
                                repeats,
                                kv_heads,
                                GqaHeadMapping::Grouped,
                            );
                            let mut out_pm = vec![0.0f32; head_dim];
                            let mut out_hm = vec![0.0f32; head_dim];
                            for (cache, out) in [(&pm, &mut out_pm), (&hm, &mut out_hm)] {
                                attention_context_for_head_into_with_kernels(
                                    AttentionContextHeadParams {
                                        kv_cache: cache,
                                        layer_idx: layer,
                                        kv_head,
                                        query_slice: &query[head * head_dim..(head + 1) * head_dim],
                                        position_count,
                                        scale,
                                    },
                                    out,
                                    &mut scores,
                                    use_blocked,
                                )
                                .expect("pinned attention");
                            }
                            for (x, y) in out_pm.iter().zip(&out_hm) {
                                if x.to_bits() != y.to_bits() {
                                    mismatch += 1;
                                }
                            }

                            // Lane B: f16 storage under the blocked lane must
                            // be bitwise-equal to f32 storage under the
                            // blocked lane, in both layouts (the values are
                            // identical â€” the write path always rounded â€” and
                            // the fused kernel is expand-then-blocked).
                            if use_blocked {
                                let mut out_pm16 = vec![0.0f32; head_dim];
                                let mut out_hm16 = vec![0.0f32; head_dim];
                                for (cache, out) in [(&pm16, &mut out_pm16), (&hm16, &mut out_hm16)]
                                {
                                    attention_context_for_head_into_with_kernels(
                                        AttentionContextHeadParams {
                                            kv_cache: cache,
                                            layer_idx: layer,
                                            kv_head,
                                            query_slice: &query
                                                [head * head_dim..(head + 1) * head_dim],
                                            position_count,
                                            scale,
                                        },
                                        out,
                                        &mut scores,
                                        true,
                                    )
                                    .expect("pinned f16 attention");
                                }
                                for ((x, y), z) in out_pm.iter().zip(&out_pm16).zip(&out_hm16) {
                                    if x.to_bits() != y.to_bits() || y.to_bits() != z.to_bits() {
                                        mismatch += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                // (c) Logical content equality across dtypes: the f16 caches
                // expand to exactly the values the f32 caches hold.
                let mut row_ref = vec![0.0f32; head_dim];
                let mut row_alt = vec![0.0f32; head_dim];
                for layer in 0..2 {
                    for p in 0..position_count {
                        for h in 0..kv_heads {
                            pm.copy_key_row_into(layer, p, h, &mut row_ref);
                            for cache in [&pm16, &hm16] {
                                cache.copy_key_row_into(layer, p, h, &mut row_alt);
                                for (x, y) in row_ref.iter().zip(&row_alt) {
                                    if x.to_bits() != y.to_bits() {
                                        mismatch += 1;
                                    }
                                }
                            }
                            pm.copy_value_row_into(layer, p, h, &mut row_ref);
                            for cache in [&pm16, &hm16] {
                                cache.copy_value_row_into(layer, p, h, &mut row_alt);
                                for (x, y) in row_ref.iter().zip(&row_alt) {
                                    if x.to_bits() != y.to_bits() {
                                        mismatch += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                mismatch
            },
        )
        .sum();

    assert_eq!(
        mismatches, 0,
        "KV layout/dtype lanes diverged from the position-major f32 reference in \
         {mismatches} element(s)"
    );
}

/// Items 4+5 P1.1 binder equivalence: for EVERY projection weight of a real
/// loaded 3B model, across runtime-plan flag combinations, (a) the recording
/// cascade selects a stable arm across repeated fresh runs, (b) the bound
/// fast path replays that arm without rebinding, and (c) bound and cascade
/// outputs are bitwise-identical. Skips when the 3B GGUF is absent (CI).
#[test]
fn decode_linear_binder_matches_cascade_on_real_3b_weights() {
    // Opt-in: set CAMELID_3B_GGUF to a 3B GGUF path to run this. No hardcoded
    // default (a hardcoded operator path leaks the home dir and isn't portable).
    let Some(model) = std::env::var_os("CAMELID_3B_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("skipping: set CAMELID_3B_GGUF to the 3B GGUF path to run this test");
        return;
    };
    if !model.exists() {
        eprintln!("skipping: 3B GGUF not present at {}", model.display());
        return;
    }
    let gguf = crate::gguf::read_metadata(&model).expect("metadata");
    let config = crate::model::LlamaModelConfig::from_gguf(&gguf).expect("config");
    let binding = crate::model::LlamaTensorBinding::bind(&gguf, &config).expect("binding");
    let store = crate::tensor::TensorStore::open(&model, &gguf);
    let weights = LlamaLoadedWeights::load(&store, &binding, None).expect("weights");

    let hidden = config.embedding_length as usize;
    let ffn = config.feed_forward_length as usize;
    let fill = |width: usize, scale: f32| -> CpuTensor {
        let data: Vec<f32> = (0..width)
            .map(|i| (((i % 97) as f32) - 48.0) * scale)
            .collect();
        CpuTensor::from_f32("binder_probe", vec![1, width], data).unwrap()
    };

    let base_plan = ResolvedRuntimePlan::from_env().expect("plan");
    let mut packed_on = base_plan;
    packed_on.q8.attention_output_packed_rows4_matmul = true;
    packed_on.q8.ffn_down_packed_rows4_matmul = true;
    packed_on.q8_packed_rows4_matmul_schedule =
        Q8PackedRows4MatmulSchedule::from_q8_flags(packed_on.q8);
    let mut consumers_on = packed_on;
    consumers_on.q8.attention_output_decode_consumer = true;
    consumers_on.q8.attention_projection_decode_consumer = true;
    consumers_on.q8.ffn_down_decode_consumer = true;
    consumers_on.q8_packed_rows4_matmul_schedule =
        Q8PackedRows4MatmulSchedule::from_q8_flags(consumers_on.q8);
    let plans = [
        ("default", base_plan),
        ("packed", packed_on),
        ("consumers", consumers_on),
    ];

    let mut checked = 0usize;
    for (plan_name, plan) in &plans {
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let slots: [(&CpuTensor, &str, usize); 5] = [
                (&layer.attention_q, "linear", hidden),
                (&layer.attention_k, "attention_k", hidden),
                (&layer.attention_v, "attention_v", hidden),
                (&layer.attention_output, "linear", hidden),
                (&layer.ffn_down, "ffn_down", ffn),
            ];
            for (slot_idx, (weight, role, width)) in slots.iter().enumerate() {
                let input = fill(*width, 0.001 + slot_idx as f32 * 1e-4);
                let cell_a = DecodeBindingCell::default();
                let out_a = decode_linear_cascade(
                    &input,
                    weight,
                    format!("binder_a_{layer_idx}_{slot_idx}"),
                    role,
                    plan,
                    Some(&cell_a),
                )
                .expect("cascade a");
                let arm_a = cell_a.load();
                let cell_b = DecodeBindingCell::default();
                let out_b = decode_linear_cascade(
                    &input,
                    weight,
                    format!("binder_b_{layer_idx}_{slot_idx}"),
                    role,
                    plan,
                    Some(&cell_b),
                )
                .expect("cascade b");
                assert_eq!(
                    arm_a,
                    cell_b.load(),
                    "unstable arm: plan {plan_name} layer {layer_idx} slot {slot_idx}"
                );
                let bound = linear_for_role_bound(
                    &input,
                    weight,
                    format!("binder_c_{layer_idx}_{slot_idx}"),
                    role,
                    plan,
                    false,
                    &cell_a,
                )
                .expect("bound replay");
                assert_eq!(
                    arm_a,
                    cell_a.load(),
                    "bound path rebound: plan {plan_name} layer {layer_idx} slot {slot_idx}"
                );
                assert_eq!(out_a.data.len(), out_b.data.len());
                assert_eq!(out_a.data.len(), bound.data.len());
                for (i, ((a, b), c)) in out_a
                    .data
                    .iter()
                    .zip(&out_b.data)
                    .zip(&bound.data)
                    .enumerate()
                {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "cascade nondeterminism at {plan_name}/{layer_idx}/{slot_idx}/{i}"
                    );
                    assert_eq!(
                        a.to_bits(),
                        c.to_bits(),
                        "bound output diverged at {plan_name}/{layer_idx}/{slot_idx}/{i}"
                    );
                }
                checked += 1;
            }
        }
    }
    eprintln!("binder equivalence: {checked} (weight x plan) combinations checked, 0 mismatches");
}

// ---------------------------------------------------------------------------
// BASALT Phase 3: nvfp4_wire_row_dot vs an in-test straightforward reference
// over the Phase 2 produced-pilot-row fixture.
// ---------------------------------------------------------------------------

/// Minimal RFC 4648 base64 decoder (mirrors tests/nvfp4_e4b_spotcheck.rs; the
/// fixture stores wire bytes base64-encoded and this module takes no new deps).
#[cfg(test)]
fn nvfp4_b64_decode(s: &str) -> Vec<u8> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let rev = {
        let mut r = [255u8; 256];
        for (i, &c) in TABLE.iter().enumerate() {
            r[c as usize] = i as u8;
        }
        r
    };
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            let v = rev[c as usize];
            assert_ne!(v, 255, "invalid base64 byte {c}");
            acc = (acc << 6) | u32::from(v);
        }
        let bits = chunk.len() * 6;
        acc <<= 24 - bits.min(24);
        let full = [(acc >> 16) as u8, (acc >> 8) as u8, acc as u8];
        out.extend_from_slice(&full[..(bits / 8).min(3)]);
    }
    out
}

/// In-test straightforward reference for `nvfp4_wire_row_dot`, written against
/// the pin contract directly (`ggml_vec_dot_nvfp4_q8_0_generic`, llama.cpp
/// acd79d603 ggml-cpu/quants.c): first decode each superblock's 64 E2M1 codes
/// to their integer kvalues, then per sub-block take the exact i32 dot against
/// the owning Q8_0 activation block and accumulate
/// `x_scale * UE4M3_TO_F32[d[s]] * sumi` in f32, sub-block-major. Integer adds
/// are order-free; the f32 accumulation order is the load-bearing part and
/// matches the pin (and the production kernel) exactly.
#[cfg(test)]
fn nvfp4_reference_row_dot(wire: &[u8], input: &[Q8_0Block]) -> f32 {
    use crate::tensor::{KVALUES_MXFP4_I8, UE4M3_TO_F32};
    assert_eq!(wire.len() % 36, 0);
    assert_eq!(input.len(), wire.len() / 36 * 2);
    let mut sumf = 0.0_f32;
    for (ib, blk) in wire.chunks_exact(36).enumerate() {
        // Decode the 64 integer code values (pre-scale): sub-block s owns qs
        // bytes s*8..s*8+8; low nibble of byte j -> element s*16+j, high
        // nibble -> element s*16+8+j.
        let mut codes = [0i32; 64];
        for s in 0..4 {
            for j in 0..8 {
                let byte = blk[4 + s * 8 + j];
                codes[s * 16 + j] = i32::from(KVALUES_MXFP4_I8[(byte & 0x0F) as usize]);
                codes[s * 16 + 8 + j] = i32::from(KVALUES_MXFP4_I8[(byte >> 4) as usize]);
            }
        }
        for s in 0..4 {
            let y = &input[2 * ib + s / 2];
            let off = (s % 2) * 16;
            let mut sumi = 0i32;
            for j in 0..16 {
                sumi += codes[s * 16 + j] * i32::from(y.quants[off + j]);
            }
            sumf += y.scale * UE4M3_TO_F32[blk[s] as usize] * sumi as f32;
        }
    }
    sumf
}

/// The three §-pre-registered deterministic activation patterns for the row-dot
/// test: constant, alternating-sign, and seeded pseudo-random i8 quants.
#[cfg(test)]
fn nvfp4_test_activation(pattern: usize, blocks: usize) -> Vec<Q8_0Block> {
    match pattern {
        0 => (0..blocks)
            .map(|_| Q8_0Block {
                scale: 0.0625,
                quants: [3i8; 32],
            })
            .collect(),
        1 => (0..blocks)
            .map(|b| {
                let mut quants = [0i8; 32];
                for (j, q) in quants.iter_mut().enumerate() {
                    *q = if j % 2 == 0 { 5 } else { -7 };
                }
                Q8_0Block {
                    scale: if b % 2 == 0 { 1.5 } else { 0.75 },
                    quants,
                }
            })
            .collect(),
        _ => {
            // xorshift64: deterministic pseudo-random i8 quants + positive scales.
            let mut state = 0x243F_6A88_85A3_08D3_u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            (0..blocks)
                .map(|_| {
                    let mut quants = [0i8; 32];
                    for q in quants.iter_mut() {
                        *q = (next() & 0xFF) as u8 as i8;
                    }
                    Q8_0Block {
                        scale: (next() % 1000) as f32 / 333.0 + 0.001,
                        quants,
                    }
                })
                .collect()
        }
    }
}

/// BASALT Phase 3 (test a): `nvfp4_wire_row_dot` must match the in-test pin-order
/// reference BITWISE over ALL 5,120 blocks of the produced-pilot-row fixture
/// (`tests/fixtures/dequant/nvfp4_e4b_spotcheck.json`, real wire bytes from
/// gemma-4-E4B-it-NVFP4-mm.gguf) x 3 deterministic activation patterns — both as
/// 5,120 x 3 single-superblock rows and as 64-superblock rows (8 per tensor) that
/// exercise the cross-superblock f32 accumulation order.
#[test]
fn nvfp4_wire_row_dot_matches_pin_order_reference_on_pilot_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dequant")
        .join("nvfp4_e4b_spotcheck.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let fx: serde_json::Value = serde_json::from_str(&raw).expect("spotcheck fixture parses");
    assert_eq!(
        fx["provenance"]["pin_sha"].as_str(),
        Some("acd79d603"),
        "fixture provenance pin mismatch"
    );
    let blocks = fx["blocks"].as_array().expect("blocks");
    assert_eq!(blocks.len(), 5120, "5120 sampled blocks (512 x 10 tensors)");

    let wires: Vec<Vec<u8>> = blocks
        .iter()
        .map(|b| {
            let wire = nvfp4_b64_decode(b["w"].as_str().expect("wire b64"));
            assert_eq!(wire.len(), 36);
            wire
        })
        .collect();

    let mut single_rows_checked = 0usize;
    let mut multi_rows_checked = 0usize;
    for pattern in 0..3 {
        // Every fixture block as its own single-superblock row.
        let xq2 = nvfp4_test_activation(pattern, 2);
        for (bi, wire) in wires.iter().enumerate() {
            let got = super::nvfp4_wire_row_dot(wire, &xq2);
            let want = nvfp4_reference_row_dot(wire, &xq2);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "pattern {pattern} fixture block {bi}: got {got} want {want}"
            );
            single_rows_checked += 1;
        }
        // 64-superblock rows (80 per pattern): cross-superblock accumulation order.
        let xq128 = nvfp4_test_activation(pattern, 128);
        for (ri, row_blocks) in wires.chunks_exact(64).enumerate() {
            let row: Vec<u8> = row_blocks.iter().flatten().copied().collect();
            let got = super::nvfp4_wire_row_dot(&row, &xq128);
            let want = nvfp4_reference_row_dot(&row, &xq128);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "pattern {pattern} 64-superblock row {ri}: got {got} want {want}"
            );
            multi_rows_checked += 1;
        }
    }
    assert_eq!(single_rows_checked, 5120 * 3);
    assert_eq!(multi_rows_checked, 80 * 3);
    eprintln!(
        "nvfp4 row dot: {single_rows_checked} single-superblock + {multi_rows_checked} \
         64-superblock rows bitwise-identical to the pin-order reference"
    );
}

/// A session whose positions were all produced by a GPU-resident engine carries a
/// non-zero `kv_cache.position` over CPU K/V buffers that were never grown — the
/// state the resident reseed path used to index blindly.
#[test]
fn f32_history_addressable_declines_kv_buffers_a_resident_session_never_grew() {
    let (mut session, _temp_file) = tiny_kv_budget_session(64);

    // Position 0 needs no history, so seeding from it is always safe.
    assert!(session.kv_cache.f32_history_addressable(0, 0));

    // The resident lane advances `position` without touching the CPU buffers.
    session.kv_cache.position = 5;
    assert!(session.kv_cache.keys.is_empty());
    assert!(
        !session.kv_cache.f32_history_addressable(0, 5),
        "an unallocated CPU KV history must never be reported as readable"
    );

    // Once the CPU path has grown the buffers the same range is readable...
    session.kv_cache.ensure_position_capacity(5).unwrap();
    assert!(session.kv_cache.f32_history_addressable(0, 5));
    // ...but only up to what was actually allocated.
    let past_end = session.kv_cache.allocated_sequence_length + 1;
    assert!(!session.kv_cache.f32_history_addressable(0, past_end));
    // A layer index past the plan is out of range for the offset math.
    assert!(!session
        .kv_cache
        .f32_history_addressable(session.kv_cache.plan.layer_count, 5));
}

/// Rejected-draft rollback must move the Metal resident engine's `filled` with the KV
/// position. Leaving it stale is what made the next resident decode treat the session as
/// a new sequence and reseed from a CPU KV history the GPU-resident drafter never wrote
/// (`range end index <head_dim> out of range for slice of length 0`).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_resident_rollback_moves_filled_with_the_kv_position() {
    let (mut session, _temp_file) = tiny_kv_budget_session(64);
    // tiny_kv_budget_session: 1 layer, 1 head, 1 kv head, head_dim 32, hidden 32, ffn 32.
    let Some(mut state) =
        crate::metal::ResidentDecodeState::new(1, 1, 1, 32, 32, 32, 16, 64, 1.0e-5, false)
    else {
        // No usable Metal lane. This is the ONLY guard for the draft-rollback regression that
        // needs no model file, so a silent skip here means the regression is unguarded — and
        // indistinguishable from a pass. Under CAMELID_REQUIRE_METAL_TESTS=1 (set by CI on the
        // macOS leg once the runner is known to be capable) that is a failure, not a skip.
        // `metal_lane_capability_probe` in src/metal.rs reports which capability is missing.
        assert!(
            std::env::var("CAMELID_REQUIRE_METAL_TESTS").as_deref() != Ok("1"),
            "CAMELID_REQUIRE_METAL_TESTS=1 but ResidentDecodeState::new returned None — the \
             draft-rollback regression is UNGUARDED on this runner"
        );
        return; // no Metal device on this host — nothing to exercise
    };

    // Six positions decoded on the GPU: `filled` and `position` advance together, and the
    // CPU K/V buffers stay empty (only `ensure_position_capacity` grows them).
    state.set_filled(6);
    session.resident_decode = Some(state);
    session.kv_cache.position = 6;
    assert!(session.kv_cache.keys.is_empty());

    // The target accepted four of those positions and rejected the tail.
    session.rollback_resident_to_position(4).unwrap();

    let filled = session
        .resident_decode
        .as_ref()
        .expect("rollback must not drop the resident session")
        .filled();
    assert_eq!(session.kv_cache.position, 4);
    assert_eq!(
        filled, session.kv_cache.position,
        "a stale `filled` sends the next decode into the reseed path"
    );

    // Rolling back to the current position is a no-op, and rolling forward is not a
    // rollback: neither may raise `filled` past what the GPU cache holds.
    session.rollback_resident_to_position(4).unwrap();
    assert_eq!(session.resident_decode.as_ref().unwrap().filled(), 4);
    session.rollback_resident_to_position(0).unwrap();
    assert_eq!(session.resident_decode.as_ref().unwrap().filled(), 0);
}
