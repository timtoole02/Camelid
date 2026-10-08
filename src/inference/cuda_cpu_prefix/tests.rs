use super::*;

const HIDDEN: usize = 64;
const HEAD_DIM: usize = 32;
const KV_HEADS: usize = 2;
const FFN: usize = 128;
const VOCAB: usize = 32;
const PREFIX: usize = 2;

fn fixture() -> LlamaInferenceSession {
    let config = LlamaModelConfig {
        architecture: "llama".into(),
        // The production builder requires at least 256 resident positions.
        context_length: 256,
        embedding_length: HIDDEN as u32,
        block_count: 3,
        feed_forward_length: FFN as u32,
        attention_head_count: KV_HEADS as u32,
        attention_head_count_kv: KV_HEADS as u32,
        kv_quant: crate::model::KvCacheQuantization::F16,
        rope_dimension_count: Some(HEAD_DIM as u32),
        rope_freq_base: Some(10_000.0),
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-5,
        vocab_size: Some(VOCAB as u32),
        file_type: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma3: None,
        gemma4: None,
        qwen35: None,
        lfm2: None,
        mla: None,
    };
    let norm = |name: &str| CpuTensor::from_f32(name, vec![HIDDEN], vec![1.0; HIDDEN]).unwrap();
    // Exact binary scales and varied signed coefficients produce nonzero,
    // ordinary-magnitude activations without any platform random source.
    let projection = |name: &str, input: usize, output: usize, salt: u32| {
        let blocks = (0..input * output / 32)
            .map(|block| Q8_0Block {
                scale: (1 + block % 3) as f32 / 1024.0,
                quants: std::array::from_fn(|i| {
                    let value = ((block * 32 + i) as u32)
                        .wrapping_add(salt)
                        .wrapping_mul(2_654_435_761);
                    ((value >> 24) as i32 % 63 - 31) as i8
                }),
            })
            .collect();
        CpuTensor::from_q8_0_blocks(
            name,
            TensorShape {
                dims: vec![input, output],
            },
            blocks,
        )
        .unwrap()
    };
    let layers = (0..3)
        .map(|index| {
            let salt = index * 101;
            LlamaLayerWeights {
                attention_norm: norm("attn_norm"),
                attention_q: projection("attn_q", HIDDEN, HIDDEN, salt + 1),
                attention_k: projection("attn_k", HIDDEN, HIDDEN, salt + 2),
                attention_v: projection("attn_v", HIDDEN, HIDDEN, salt + 3),
                attention_output: projection("attn_output", HIDDEN, HIDDEN, salt + 4),
                attention_q_norm: None,
                attention_k_norm: None,
                post_attention_norm: None,
                post_ffw_norm: None,
                attention_biases: None,
                ffn_norm: norm("ffn_norm"),
                ffn_gate: projection("ffn_gate", HIDDEN, FFN, salt + 5),
                ffn_up: projection("ffn_up", HIDDEN, FFN, salt + 6),
                ffn_down: projection("ffn_down", FFN, HIDDEN, salt + 7),
                moe_router: None,
                mla_q_a_proj: None,
                mla_q_a_layernorm: None,
                mla_q_b_proj: None,
                mla_kv_a_proj_with_mqa: None,
                mla_kv_a_layernorm: None,
                mla_kv_b_proj: None,
                moe_expert_bias: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                decode_bindings: DecodeLinearBindings::default(),
            }
        })
        .collect();
    let weights = LlamaLoadedWeights {
        token_embedding: CpuTensor::from_f32(
            "token_embd",
            vec![VOCAB, HIDDEN],
            (0..VOCAB * HIDDEN)
                .map(|i| (((i * 73 + i / HIDDEN * 11) % 257) as f32 - 128.0) / 128.0)
                .collect(),
        )
        .unwrap(),
        output_norm: norm("output_norm"),
        output: Some(projection("output", HIDDEN, VOCAB, 997)),
        rope_freqs: None,
        layers,
        layer_range: None,
        output_projection_binding: DecodeBindingCell::default(),
    };
    LlamaInferenceSession::new(config, weights).unwrap()
}

fn embedding(session: &LlamaInferenceSession, token: u32) -> CpuTensor {
    session
        .weights
        .token_embedding
        .embedding_lookup(&[token], "test_embedding")
        .unwrap()
}

#[test]
fn cpu_prefix_rejects_per_projection_cuda_before_mutation() {
    let _lock = crate::test_support::env_lock();
    struct Restore {
        prefix: Option<std::ffi::OsString>,
        resident_decode: Option<std::ffi::OsString>,
        projection_cuda: bool,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            for (key, value) in [
                (PREFIX_ENV, self.prefix.take()),
                ("CAMELID_CUDA_RESIDENT_DECODE", self.resident_decode.take()),
            ] {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
            crate::cuda::set_runtime_enabled(self.projection_cuda);
        }
    }
    let _restore = Restore {
        prefix: std::env::var_os(PREFIX_ENV),
        resident_decode: std::env::var_os("CAMELID_CUDA_RESIDENT_DECODE"),
        projection_cuda: crate::cuda::runtime_enabled(),
    };
    // Avoid probing a CUDA device after the new gate passes: this regression
    // checks admission ordering and state preservation entirely on the CPU.
    std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
    let mut session = fixture();
    let before = session.kv_cache.clone();
    crate::cuda::set_runtime_enabled(true);
    for prefix in [None, Some("0")] {
        match prefix {
            Some(value) => std::env::set_var(PREFIX_ENV, value),
            None => std::env::remove_var(PREFIX_ENV),
        }
        assert_eq!(session.cuda_cpu_prefix_layers().unwrap(), None);
    }

    std::env::set_var(PREFIX_ENV, "2");
    let input = embedding(&session, 3);
    let error = session
        .try_resident_decode_forward_cuda(&input, true, None, None, None)
        .err()
        .expect("an active projection override must reject the split");
    assert!(error
        .to_string()
        .contains("incompatible with per-projection CUDA Q8"));
    assert!(
        session.kv_cache == before,
        "admission must not change host KV"
    );
    assert!(session.cuda_cpu_prefix.is_none());
    assert!(session.cuda_resident_pin.is_none());

    crate::cuda::set_runtime_enabled(false);
    let error = session.cuda_cpu_prefix_layers().unwrap_err().to_string();
    assert!(error.contains("requires one unsharded CUDA sequence"));
    assert!(!error.contains("per-projection CUDA Q8"));
}

fn split_step(session: &mut LlamaInferenceSession, token: u32) -> Vec<u32> {
    split_step_with_prefix(session, token, PREFIX)
}

fn split_step_with_prefix(
    session: &mut LlamaInferenceSession,
    token: u32,
    prefix: usize,
) -> Vec<u32> {
    let embedding = embedding(session, token);
    let result = session
        .forward_cuda_cpu_prefix(&embedding, prefix, None, None)
        .unwrap();
    let ResidentForward::Logits(logits) = result else {
        panic!("expected logits")
    };
    session.kv_cache.position += 1;
    assert!(session.cpu_kv_authoritative());
    assert!(logits.data.iter().all(|v| v.is_finite()));
    assert!(logits.data.iter().any(|v| v.abs() > 0.01));
    logits.data.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_batched_kv_gather_matches_spans_and_rebinds_slots() {
    use crate::cuda_resident::CudaResidentDecode;
    const LAYERS: usize = 3;
    const CAPACITY: usize = 7;
    let make_engine = |quant| {
        CudaResidentDecode::new_with_kv_quant(
            LAYERS, KV_HEADS, KV_HEADS, HEAD_DIM, HIDDEN, FFN, HEAD_DIM, CAPACITY, VOCAB, 1e-5,
            false, quant,
        )
        .unwrap()
    };
    let seed = |engine: &mut CudaResidentDecode, salt: usize| {
        // Include signed zero, subnormals, the normal boundary, and both signs
        // while varying layer/head/position so any stride mixup is observable.
        let special = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff,
        ];
        for layer in 0..LAYERS {
            let values = |side: usize| {
                (0..KV_HEADS * CAPACITY * HEAD_DIM)
                    .map(|index| {
                        if index % 11 == 0 {
                            f16_bits_to_f32(special[(index / 11 + layer + side) % special.len()])
                        } else {
                            let value = (index * 19 + layer * 73 + side * 131 + salt * 47) % 1021;
                            (value as f32 - 510.0) / 256.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            engine
                .seed_layer(layer, &values(0), &values(1), CAPACITY)
                .unwrap();
        }
    };
    let compare = |engine: &mut CudaResidentDecode, position| {
        let (keys, values) = engine.read_kv_row_all_layers(position).unwrap();
        let mut expected_keys = Vec::new();
        let mut expected_values = Vec::new();
        for layer in 0..LAYERS {
            let (key, value) = engine.read_kv_layer_range(layer, position, 1).unwrap();
            expected_keys.extend(key.into_iter().map(f32::to_bits));
            expected_values.extend(value.into_iter().map(f32::to_bits));
        }
        let keys: Vec<_> = keys.into_iter().map(f32::to_bits).collect();
        let values: Vec<_> = values.into_iter().map(f32::to_bits).collect();
        assert_eq!(keys, expected_keys, "K position {position}");
        assert_eq!(values, expected_values, "V position {position}");
        (keys, values)
    };
    let mut engine = make_engine(crate::model::KvCacheQuantization::F16);
    engine.enable_second_kv_slot().unwrap();
    seed(&mut engine, 0);
    let original = compare(&mut engine, 0);
    for position in [3, CAPACITY - 1, 1] {
        compare(&mut engine, position);
    }
    assert!(engine.read_kv_row_all_layers(CAPACITY).is_err());
    assert!(engine.read_kv_row_all_layers(usize::MAX).is_err());
    engine.select_kv_slot(1).unwrap();
    let zero = compare(&mut engine, 0);
    assert!(zero.0.iter().chain(&zero.1).all(|&bits| bits == 0));
    seed(&mut engine, 1);
    assert_ne!(compare(&mut engine, 0), original);
    engine.select_kv_slot(0).unwrap();
    assert_eq!(compare(&mut engine, 0), original);
    engine.sparsify_kv(&[true, false, true]).unwrap();
    assert!(engine.read_kv_row_all_layers(0).is_err());

    // Q8 continues to use its checked span reader. The gather must decline it
    // explicitly instead of interpreting quantized bytes as F16.
    let mut q8 = make_engine(crate::model::KvCacheQuantization::Q8_0);
    assert!(q8.read_kv_row_all_layers(0).is_err());
    let (keys, values) = q8.read_kv_layer_range(1, 2, 1).unwrap();
    assert!(keys.iter().chain(&values).all(|&value| value == 0.0));

    let mut windowed = make_engine(crate::model::KvCacheQuantization::F16);
    // No weights are needed for readback; set the schedule for the empty layer
    // stack to exercise the layout gate independently of the quantization gate.
    windowed.set_gemma3(Vec::new(), Vec::new(), false).unwrap();
    assert!(windowed.read_kv_row_all_layers(0).is_err());
    let mut paged = CudaResidentDecode::new_paged_with_kv_quant(
        LAYERS,
        KV_HEADS,
        KV_HEADS,
        HEAD_DIM,
        HIDDEN,
        FFN,
        HEAD_DIM,
        CAPACITY,
        VOCAB,
        1e-5,
        false,
        crate::model::KvCacheQuantization::F16,
    )
    .unwrap();
    assert!(paged.read_kv_row_all_layers(0).is_err());
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_multilayer_gather_preserves_clone_and_rollback() {
    let mut session = fixture();
    split_step_with_prefix(&mut session, 3, 1);
    let mut expected = session.clone();
    for token in [9, 7] {
        assert_eq!(
            split_step_with_prefix(&mut session, token, 1),
            split_step_with_prefix(&mut expected, token, 1)
        );
        assert_eq!(history(&session), history(&expected));
    }
    split_step_with_prefix(&mut session, 29, 1);
    split_step_with_prefix(&mut session, 31, 1);
    session.rollback_to_position(3).unwrap();
    assert_eq!(
        split_step_with_prefix(&mut session, 13, 1),
        split_step_with_prefix(&mut expected, 13, 1)
    );
    assert_eq!(history(&session), history(&expected));
}

fn history(session: &LlamaInferenceSession) -> Vec<u32> {
    let mut result = Vec::new();
    for layer in 0..3 {
        for position in 0..session.kv_position() {
            for head in 0..KV_HEADS {
                let mut row = [0.0; HEAD_DIM];
                session
                    .kv_cache
                    .copy_key_row_into(layer, position, head, &mut row);
                result.extend(row.map(f32::to_bits));
                session
                    .kv_cache
                    .copy_value_row_into(layer, position, head, &mut row);
                result.extend(row.map(f32::to_bits));
            }
        }
    }
    result
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_clone_rollback_and_row_range_preserve_history() {
    let mut uninterrupted = fixture();
    // Exercise seeding from actual CPU prefill before any GPU suffix exists.
    uninterrupted.set_resident_paths_disabled(true);
    for token in [3, 9] {
        uninterrupted.forward_single_token_timed(token).unwrap();
    }
    uninterrupted.set_resident_paths_disabled(false);
    split_step(&mut uninterrupted, 7);
    assert_eq!(uninterrupted.kv_position(), 3);
    let mut fork = uninterrupted.clone();
    assert!(
        fork.cuda_cpu_prefix.is_none(),
        "clone must construct and seed its own GPU suffix"
    );
    for token in [11, 2] {
        assert_eq!(
            split_step(&mut uninterrupted, token),
            split_step(&mut fork, token)
        );
        assert_eq!(history(&uninterrupted), history(&fork));
    }

    // Inspect a nonzero range spanning two positions and both KV heads. This
    // catches a readback that accidentally starts at zero or mixes head strides.
    let mut expected_k = Vec::new();
    let mut expected_v = Vec::new();
    for head in 0..KV_HEADS {
        for position in 2..4 {
            let mut row = [0.0; HEAD_DIM];
            uninterrupted
                .kv_cache
                .copy_key_row_into(PREFIX, position, head, &mut row);
            expected_k.extend(row.map(f32::to_bits));
            uninterrupted
                .kv_cache
                .copy_value_row_into(PREFIX, position, head, &mut row);
            expected_v.extend(row.map(f32::to_bits));
        }
    }
    let state = uninterrupted
        .cuda_cpu_prefix
        .as_mut()
        .unwrap()
        .get_mut()
        .unwrap();
    let (keys, values) = state.engine.read_kv_layer_range(0, 2, 2).unwrap();
    assert_eq!(
        keys.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected_k
    );
    assert_eq!(
        values.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected_v
    );

    // Advance one fork down a rejected branch, then use the public rollback.
    // Its GPU filled watermark is now ahead and must reseed from retained KV.
    let expected = split_step(&mut uninterrupted, 13);
    split_step(&mut fork, 29);
    split_step(&mut fork, 31);
    fork.rollback_to_position(5).unwrap();
    assert_eq!(split_step(&mut fork, 13), expected);
    assert_eq!(history(&uninterrupted), history(&fork));
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_rollback_then_cpu_rows_reseed_at_the_old_gpu_position() {
    for resident_rollback in [false, true] {
        let mut session = fixture();
        for token in [3, 9, 7] {
            split_step(&mut session, token);
        }
        let mut expected = session.clone();
        // The rejected GPU branch ends at 5. Replacing its two rows on the
        // CPU also ends at 5: a cursor-only freshness check would miss this.
        for token in [29, 31] {
            split_step(&mut session, token);
        }
        let old_gpu_position = session.kv_position();
        if resident_rollback {
            session.rollback_resident_to_position(3).unwrap();
        } else {
            session.rollback_to_position(3).unwrap();
        }
        assert!(session.cuda_cpu_prefix.is_some(), "keep uploaded weights");
        for token in [11, 2] {
            // The diagnostic entry deliberately runs the whole token on CPU.
            session.forward_single_token_timed(token).unwrap();
            expected.forward_single_token_timed(token).unwrap();
        }
        assert_eq!(session.kv_position(), old_gpu_position);
        assert_eq!(history(&session), history(&expected));
        assert_eq!(
            split_step(&mut session, 13),
            split_step(&mut expected, 13),
            "GPU continuation must use the rewritten host history"
        );
        assert_eq!(history(&session), history(&expected));
    }
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_gpu_verifiers_decline_without_pinning_and_cpu_verify_can_resume() {
    assert!(
        resident_decode_cuda_enabled(),
        "exercise the GPU verifier gate"
    );
    let mut session = fixture();
    for token in [3, 9] {
        split_step(&mut session, token);
    }
    let mut expected = session.clone();
    let before = history(&session);
    let before_position = session.kv_position();
    assert!(session.cuda_resident_pin.is_none());
    assert!(session.verify_drafts_gpu(7, &[11, 2]).unwrap().is_none());
    assert!(session.cuda_resident_pin.is_none());
    let tree = crate::inference::spec_tree::TokenTree::linear(7, &[11, 2]);
    assert!(session.verify_tree_gpu(&tree).unwrap().is_none());
    assert!(session.cuda_resident_pin.is_none());
    assert_eq!(session.kv_position(), before_position);
    assert_eq!(history(&session), before);

    // Match the serving fallback: CPU chunk verify, retain the accepted
    // anchor, roll rejected drafts back, then continue the hybrid session.
    let (predictions, _) = session.forward_greedy_verify_chunk(&[7, 11, 2]).unwrap();
    let (expected_predictions, _) = expected.forward_greedy_verify_chunk(&[7, 11, 2]).unwrap();
    assert_eq!(predictions, expected_predictions);
    assert_eq!(history(&session), history(&expected));
    session.rollback_to_position(before_position + 1).unwrap();
    expected.rollback_to_position(before_position + 1).unwrap();
    assert_eq!(split_step(&mut session, 13), split_step(&mut expected, 13));
    assert_eq!(history(&session), history(&expected));
    assert!(session.cuda_resident_pin.is_none());
}

#[test]
#[ignore = "requires a CUDA device; run alone with no CPU-prefix environment override"]
fn cpu_prefix_partial_failure_poisons_state_and_restores_authority() {
    let mut session = fixture();
    split_step(&mut session, 3);
    let before_history = history(&session);
    let before_position = session.kv_position();
    let before_watermark = session.kv_cache.materialized_through;
    // Inject a recoverable shape error in the SECOND CPU layer, after the
    // first has written the next KV row. Keep the same Arc identity so this
    // reaches the partial-forward guard rather than model-change preflight.
    let weights = Arc::get_mut(&mut session.weights).unwrap();
    weights.layers[1].attention_norm.shape.dims[0] -= 1;
    weights.layers[1].attention_norm.data.pop();
    let input = embedding(&session, 9);
    let error = session
        .forward_cuda_cpu_prefix(&input, PREFIX, None, None)
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("rms_norm weight shape"),
        "{error}"
    );
    assert_eq!(session.kv_position(), before_position);
    assert_eq!(session.kv_cache.materialized_through, before_watermark);
    assert_eq!(history(&session), before_history);
    let mut changed_row = [0.0; HEAD_DIM];
    session
        .kv_cache
        .copy_key_row_into(0, before_position, 0, &mut changed_row);
    assert!(
        changed_row.iter().any(|v| *v != 0.0),
        "fixture must reach a partial KV write"
    );
    assert!(
        session
            .cuda_cpu_prefix
            .as_mut()
            .unwrap()
            .get_mut()
            .unwrap()
            .poisoned
    );
    let retry = session
        .forward_cuda_cpu_prefix(&input, PREFIX, None, None)
        .err()
        .unwrap();
    assert!(
        retry
            .to_string()
            .contains("previous partial forward failed"),
        "{retry}"
    );
    assert_eq!(session.kv_position(), before_position);
    assert_eq!(session.kv_cache.materialized_through, before_watermark);
}
