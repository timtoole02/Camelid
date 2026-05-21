use super::*;
use crate::test_support::env_lock;
use std::io::Write;

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
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
fn x86_q8_packed_rows4_matmul_chunk_groups_env_override() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK");
    assert_eq!(
        x86_q8_packed_rows4_matmul_groups_per_chunk(),
        X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK
    );
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "32");
    assert_eq!(x86_q8_packed_rows4_matmul_groups_per_chunk(), 32);
    assert_eq!(q8_packed_rows4_matmul_parallel_chunk_floats(128), 128);
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "0");
    assert_eq!(
        x86_q8_packed_rows4_matmul_groups_per_chunk(),
        X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK
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
            moe_router: None,
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
        moe: None,
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
        layers: vec![LlamaLayerWeights {
            attention_norm: dense_vector("blk.0.attn_norm.weight"),
            attention_q,
            attention_k: dense_matrix("blk.0.attn_k.weight"),
            attention_v: dense_matrix("blk.0.attn_v.weight"),
            attention_output: dense_matrix("blk.0.attn_output.weight"),
            ffn_norm: dense_vector("blk.0.ffn_norm.weight"),
            ffn_gate: dense_matrix("blk.0.ffn_gate.weight"),
            ffn_up: dense_matrix("blk.0.ffn_up.weight"),
            ffn_down: dense_matrix("blk.0.ffn_down.weight"),
            moe_router: None,
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
        "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
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
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            metal: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
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
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER", "true");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER", "on");
    std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
    std::env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", "on");
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
    assert!(plan.q8.ffn_gate_up_decode_consumer);
    assert!(plan.q8.ffn_gate_up_packed_rows4_matmul);
    assert!(plan.q8.ffn_gate_up_single_owner);
    assert!(plan.q8.ffn_down_decode_consumer);
    assert!(plan.q8.ffn_down_packed_rows4_matmul);
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL");
    std::env::remove_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER");
    std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER");
    std::env::remove_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL");
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
        plan.q8.ffn_gate_up_decode_consumer,
        "resolved plan should cache the FFN gate/up consumer gate"
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
        plan.q8.ffn_down_packed_rows4_matmul,
        "resolved plan should cache the packed-rows4 matmul gate"
    );
    assert_eq!(plan.q8.hybrid_gpu_rows, Some(7));
    assert_eq!(plan.q8.hybrid_gpu_percent, 25);
    assert_eq!(plan.q8.hybrid_gpu_rows_for_output(100), 7);
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
            !plan.q8.ffn_gate_up_decode_consumer,
            "{profile} should not enable FFN gate/up consumer by default"
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
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            metal: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
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
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: enabled,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            metal: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
    }
}

fn ffn_down_vnni_decode_plan(enabled: bool) -> ResolvedRuntimePlan {
    let mut plan = ffn_down_consumer_plan(false);
    plan.q8.ffn_down_vnni_decode = enabled;
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
            ffn_gate_up_decode_consumer: false,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_packed_rows4_matmul: enabled,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            metal: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
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
            ffn_gate_up_decode_consumer: enabled,
            ffn_gate_up_packed_rows4_matmul: false,
            ffn_gate_up_single_owner: false,
            ffn_down_decode_consumer: false,
            ffn_down_packed_rows4_matmul: false,
            ffn_down_gemm4_prefill: false,
            ffn_down_gemm4_row_group_schedule: false,
            ffn_down_gemm4_avx2: false,
            ffn_down_amx_prefill: false,
            ffn_down_single_owner: false,
            ffn_down_vnni_decode: false,
            metal: false,
            metal_retained: false,
            hybrid_retained: false,
            hybrid_gpu_rows: None,
            hybrid_gpu_percent: 10,
        },
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
    assert!(!Q8RuntimeFlags::from_env().ffn_gate_up_decode_consumer);

    std::env::set_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
    assert!(Q8RuntimeFlags::from_env().ffn_gate_up_decode_consumer);
    std::env::remove_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER");
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

    let chunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "chunked",
        "ffn_down",
        &plan,
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
    assert_eq!(q8_ffn_down_decode_groups_per_chunk(), 2);
    assert_eq!(
        q8_ffn_down_decode_consumer_route_name(true),
        "x86_decode_consumer_group_chunking"
    );

    let chunked = try_x86_q8_ffn_down_decode_consumer_path(
        &input,
        &packed_weight,
        "chunked",
        "ffn_down",
        &plan,
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
        q8_0_packed_rows4_matmul_projection(&input, packed, output_width, "rows4").unwrap()
    }));
    let (gemm4_us, gemm4) = bench(Box::new(|| {
        q8_0_packed_rows4_gemm4_projection_with_row_group_schedule(
            &input,
            packed,
            output_width,
            "gemm4",
            false,
            true,
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
    let mut gate =
        q8_0_packed_rows4_matmul_projection(&input, gate_packed, output_width, "expected_gate")
            .unwrap();
    let up = q8_0_packed_rows4_matmul_projection(&input, up_packed, output_width, "expected_up")
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
        blocks: vec![Q8_0Block {
            scale: f16_bits_to_f32(f32_to_f16_bits(0.25)),
            quants: std::array::from_fn(|idx| 15 - idx as i8),
        }],
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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let rotated = apply_rope(&tensor, 1, 2, &config, None, "query_rope").unwrap();

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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 1, 1, &config, None, "query_rope").unwrap();
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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 8, 1, &config, None, "query_rope").unwrap();
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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let rope_freqs = CpuTensor::from_f32("rope_freqs.weight", vec![2], vec![1.0, 4.0]).unwrap();

    let rotated = apply_rope(&tensor, 8, 1, &config, Some(&rope_freqs), "query_rope").unwrap();
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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let reported = apply_rope(&tensor, 1, 2, &config, None, "query_rope").unwrap();

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
        moe: None,
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
        moe: None,
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
        moe: None,
    };
    let tensor = CpuTensor::from_f32("query", vec![1, 2], vec![1.0, 0.0]).unwrap();

    let rotated = apply_rope(&tensor, 0, 1, &config, None, "query_rope").unwrap();
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
        moe: None,
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
            moe_router: None,
        }],
    });
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let step = session
        .generate_next_token_with_history_diagnostics(&[0], LlamaSampler::Greedy, &[0], true)
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
    assert_slice_close(&layer.ffn_gate.checkpoint.first_values, &[1.0, 2.0]);
    assert_slice_close(&layer.ffn_up.checkpoint.first_values, &[3.0, 4.0]);
    let expected_activation = vec![silu(1.0) * 3.0, silu(2.0) * 4.0];
    assert_slice_close(
        &layer.ffn_activation.checkpoint.first_values,
        &expected_activation,
    );
    assert_eq!(
        layer.ffn_activation_reconstruction.activation_order,
        "gate_up"
    );
    assert_close(layer.ffn_activation_reconstruction.max_abs_delta, 0.0);
    assert_slice_close(
        &layer.ffn_output.checkpoint.first_values,
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
    clear_dense_diagnostic_env();

    let config = LlamaModelConfig {
        context_length: 12,
        embedding_length: 2,
        block_count: 1,
        feed_forward_length: 2,
        attention_head_count: 1,
        attention_head_count_kv: 1,
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
        moe: None,
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
            moe_router: None,
        }],
    });

    let prompt = [0, 1, 2, 3, 0, 1, 2];

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "1");
    let mut sequential = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
    let sequential_step = sequential
        .generate_next_token_with_history_diagnostics(&prompt, LlamaSampler::Greedy, &prompt, false)
        .unwrap();

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "2");
    std::env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "1");
    let mut chunked = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
    let chunked_step = chunked
        .generate_next_token_with_history_diagnostics(&prompt, LlamaSampler::Greedy, &prompt, false)
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
        .generate_next_token_with_history_diagnostics(&prompt, LlamaSampler::Greedy, &prompt, false)
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
        moe: None,
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
        moe_router: None,
    };
    let hidden = CpuTensor::from_f32("hidden", vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    let plan = LlamaKvCachePlan::from_config(&config).unwrap();
    let mut kv_cache = LlamaKvCache::new(plan).unwrap();
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
        moe: None,
    };
    let kv_cache = LlamaKvCache::new(LlamaKvCachePlan::from_config(&config).unwrap()).unwrap();
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
        key_shape: vec![1, rows, kv_heads, head_dim],
        value_shape: vec![1, rows, kv_heads, head_dim],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
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
        moe: None,
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
            moe_router: None,
        }],
    });

    std::env::set_var("CAMELID_PREFILL_CHUNK_TOKENS", "0");
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();
    let step = session
        .generate_next_token_with_history_diagnostics(
            &[0, 1, 2],
            LlamaSampler::Greedy,
            &[0, 1, 2],
            false,
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
        key_shape: vec![2, 10, 1, 2],
        value_shape: vec![2, 10, 1, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
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

    let prior_layer1_start = kv_cache_offset(&kv_cache, 1, 0, 0);
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
        key_shape: vec![2, 1024, 1, 2],
        value_shape: vec![2, 1024, 1, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
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
        key_shape: vec![1, 1, 1, 2],
        value_shape: vec![1, 1, 1, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");
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
fn causal_attention_context_attends_over_prior_and_current_positions() {
    let _env_guard = env_lock();
    std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
    std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

    let plan = LlamaKvCachePlan {
        max_sequence_length: 3,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 2,
        key_shape: vec![1, 3, 1, 2],
        value_shape: vec![1, 3, 1, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

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
        key_shape: vec![1, 1, 2, 2],
        value_shape: vec![1, 1, 2, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

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
        key_shape: vec![1, 2, 2, 2],
        value_shape: vec![1, 2, 2, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

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
        key_shape: vec![1, 10, 1, 2],
        value_shape: vec![1, 10, 1, 2],
    };
    let mut kv_cache = LlamaKvCache::new(plan).expect("KV cache");

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
fn softmax_top_k_preserves_full_router_softmax_weights() {
    let top = softmax_top_k(&[0.0, 1.0, 2.0], 2);
    assert_eq!(top[0].0, 2);
    assert_eq!(top[1].0, 1);
    let selected_sum = top.iter().map(|(_, weight)| *weight).sum::<f32>();
    assert!(selected_sum < 1.0, "{top:?}");
    let full_sum = 0.0_f32.exp() + 1.0_f32.exp() + 2.0_f32.exp();
    let expected_first = 2.0_f32.exp() / full_sum;
    assert!((top[0].1 - expected_first).abs() < 1.0e-6, "{top:?}");
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
        "out",
    )
    .unwrap();

    let expected = 1.0 / (1.0 + (-1.0_f32).exp());
    assert!((out.data[0] - expected).abs() < 1.0e-3, "{:?}", out.data);
    assert!((out.data[1] - expected).abs() < 1.0e-3, "{:?}", out.data);
}
