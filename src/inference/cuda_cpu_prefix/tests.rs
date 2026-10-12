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
    const CHILD_ENV: &str = "CAMELID_TEST_CPU_PREFIX_PROJECTION_GUARD_CHILD";
    const CHILD_VALUE: &str = "run-admission-assertions";
    const COMPLETED: &str = "CPU_PREFIX_PROJECTION_GUARD_ASSERTIONS_PASSED";
    if std::env::var_os(CHILD_ENV).is_none() {
        // The shared mutex only covers participating tests. Isolate all flag
        // changes from parallel inference tests that read them without it.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "inference::cuda_cpu_prefix::integration_tests::cpu_prefix_rejects_per_projection_cuda_before_mutation",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, CHILD_VALUE)
            .env(PREFIX_ENV, "0")
            .env("CAMELID_CUDA_Q8", "0")
            .env("CAMELID_CUDA_RESIDENT_DECODE", "0")
            .output()
            .expect("launch the isolated CPU-prefix admission regression");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "isolated regression failed: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status,
        );
        // An outdated exact-test name can exit successfully after zero tests.
        // Only the end of the child assertion body emits this receipt.
        assert!(
            stdout.lines().any(|line| line == COMPLETED),
            "isolated regression did not complete its assertions\nstdout:\n{stdout}\nstderr:\n{stderr}",
        );
        return;
    }
    assert_eq!(
        std::env::var(CHILD_ENV).as_deref(),
        Ok(CHILD_VALUE),
        "invalid subprocess sentinel",
    );
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
    println!("\n{COMPLETED}");
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

fn min_p_model_forward(
    session: &mut LlamaInferenceSession,
    token: u32,
    config: Option<&SamplingConfig>,
) -> ResidentForward {
    let input = embedding(session, token);
    let sample = config.map(|config| ResidentTemperatureSampling {
        inv_temp: 1.0 / config.temperature,
        temperature: config.temperature,
        seed: config.seed.unwrap_or(0)
            ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(session.kv_position() as u64 + 1),
        min_p: config.min_p,
    });
    session
        .try_resident_decode_forward_cuda(&input, true, None, sample, Some(token))
        .unwrap()
        .expect("the selected CUDA model lane must execute")
}

fn min_p_model_assert_position(session: &mut LlamaInferenceSession, route: &str, position: usize) {
    assert_eq!(session.kv_position(), position);
    if route == "cpu-prefix" {
        let state = session.cuda_cpu_prefix.as_mut().unwrap().get_mut().unwrap();
        assert_eq!(state.engine.filled(), position);
        assert!(!state.poisoned);
        assert!(session.cpu_kv_authoritative());
    } else {
        let cache = session.existing_resident_cache().unwrap();
        let guard = cache.lock().unwrap();
        let slot = guard.as_ref().unwrap();
        assert_eq!(slot.engine.filled(), position);
        assert_eq!(slot.paged_kv.is_some(), route == "paged");
        if route == "paged" {
            assert_eq!(
                slot.paged_sequence_len(session.cuda_sequence_id).unwrap(),
                position
            );
            let metadata = slot.paged_kv.as_ref().unwrap().pool.snapshot();
            let device = slot.engine.paged_kv_page_snapshot().unwrap();
            assert_eq!(metadata.allocated_pages, position.div_ceil(16));
            assert_eq!(device.resident_pages, metadata.allocated_pages);
        }
    }
}

fn min_p_model_history(session: &mut LlamaInferenceSession, route: &str) -> Vec<u32> {
    if route != "cpu-prefix" {
        assert!(session
            .recover_cpu_kv_from_cuda_resident(session.kv_position())
            .unwrap());
    }
    assert!(session.cpu_kv_authoritative());
    history(session)
}

#[test]
#[ignore = "requires a CUDA device; runs the three model lanes in isolated subprocesses"]
fn cuda_min_p_model_tails_preserve_fallback_kv_and_seeded_replay() {
    const CHILD_ENV: &str = "CAMELID_TEST_MIN_P_MODEL_ROUTE";
    const COMPLETED: &str = "CUDA_MIN_P_MODEL_ASSERTIONS_PASSED";
    let Ok(route) = std::env::var(CHILD_ENV) else {
        // Isolate OnceLock/runtime flags and the resident arena. An explicitly
        // requested device gate must fail if no CUDA device is available.
        for route in ["contiguous", "paged", "cpu-prefix"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "inference::cuda_cpu_prefix::integration_tests::cuda_min_p_model_tails_preserve_fallback_kv_and_seeded_replay",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, route)
                .env(PREFIX_ENV, if route == "cpu-prefix" { "2" } else { "0" })
                .env("CAMELID_CUDA_PAGED_KV", if route == "paged" { "1" } else { "0" })
                .env("CAMELID_CUDA_RESIDENT_DECODE", "1")
                .env("CAMELID_GPU_TEMP_SAMPLING", "1")
                .env("CAMELID_CUDA_MIN_P_SAMPLING", "1")
                .env("CAMELID_CUDA_Q8", "0")
                .env("CAMELID_DETERMINISTIC", "0")
                .env("CAMELID_CUDA_TWO_SEQUENCE_KV", "0")
                .env("CAMELID_CUDA_SEQUENCE_SLOTS", "1")
                .env("CAMELID_CUDA_TRUE_BATCH2", "0")
                .env("CAMELID_CUDA_TRUE_BATCH", "0")
                .output()
                .expect("launch the isolated CUDA min-p model gate");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "{route} min-p model gate failed: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            assert!(
                stdout.lines().any(|line| line == COMPLETED),
                "{route} did not execute its assertions\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            eprintln!("CUDA min-p {route}: fallback, KV continuation, seeded replay, and filter admission passed");
        }
        return;
    };
    assert!(["contiguous", "paged", "cpu-prefix"].contains(&route.as_str()));
    assert!(
        crate::cuda::is_available(),
        "CUDA qualification requires a usable device"
    );
    crate::cuda::set_gpu_accel_enabled(true);
    crate::cuda::set_runtime_enabled(false);
    assert!(resident_decode_cuda_enabled());

    // Cross the 16-token paged boundary while retaining ordinary, nonzero
    // three-layer Q8 model activations. All runs consume the same input tokens.
    let inputs: Vec<u32> = (0..18).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let mut reference = fixture();
    let mut logits = Vec::new();
    for (position, &input) in inputs.iter().enumerate() {
        let ResidentForward::Logits(row) = min_p_model_forward(&mut reference, input, None) else {
            panic!("normal forward must return logits");
        };
        assert!(row.data.iter().all(|value| value.is_finite()));
        reference.kv_cache.position += 1;
        min_p_model_assert_position(&mut reference, &route, position + 1);
        logits.push(row);
    }
    let expected_history = min_p_model_history(&mut reference, &route);
    drop(reference);

    let boundary = SamplingConfig {
        temperature: 0.8,
        min_p: Some(1.0),
        seed: Some(27),
        ..SamplingConfig::default()
    };
    let mut fallback = fixture();
    for (position, (&input, expected)) in inputs.iter().zip(&logits).enumerate() {
        let output = min_p_model_forward(&mut fallback, input, Some(&boundary));
        let ResidentForward::Logits(actual) = &output else {
            panic!("p=1 must take the exact CPU boundary fallback");
        };
        assert_eq!(actual.data, expected.data, "fallback logits at {position}");
        let token = fallback
            .finish_resident_temperature_sample(output, &boundary, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            token,
            sample_with_config(expected, &boundary, &inputs[..=position]).unwrap()
        );
        min_p_model_assert_position(&mut fallback, &route, position + 1);
    }
    assert_eq!(min_p_model_history(&mut fallback, &route), expected_history);
    drop(fallback);

    let config = SamplingConfig {
        min_p: Some(0.25),
        ..boundary
    };
    let mut direct = fixture();
    let mut direct_tokens = Vec::new();
    let mut device_samples = 0;
    for (position, (&input, expected)) in inputs.iter().zip(&logits).enumerate() {
        let output = min_p_model_forward(&mut direct, input, Some(&config));
        device_samples += usize::from(matches!(&output, ResidentForward::Sampled(_)));
        let token = direct
            .finish_resident_temperature_sample(output, &config, true)
            .unwrap()
            .unwrap();
        assert!(
            sampling_weights_with_config(expected, &config, &inputs[..=position])
                .unwrap()
                .iter()
                .any(|(candidate, weight)| *candidate == token as usize && *weight > 0.0)
        );
        direct_tokens.push(token);
        min_p_model_assert_position(&mut direct, &route, position + 1);
    }
    assert!(
        device_samples > 0,
        "exercise the on-device min-p sampling tail"
    );
    assert_eq!(min_p_model_history(&mut direct, &route), expected_history);
    drop(direct);

    let mut replayed = Vec::new();
    for seed in [27, 27, 83] {
        let mut replay = fixture();
        let config = SamplingConfig {
            seed: Some(seed),
            ..config.clone()
        };
        let mut tokens = Vec::new();
        for (position, (&input, expected)) in inputs.iter().zip(&logits).enumerate() {
            let (token, _) = replay
                .generate_next_token_sampled_resident(input, &config)
                .unwrap()
                .expect("the public sampled resident entry must execute");
            assert!(
                sampling_weights_with_config(expected, &config, &inputs[..=position])
                    .unwrap()
                    .iter()
                    .any(|(candidate, weight)| *candidate == token as usize && *weight > 0.0)
            );
            tokens.push(token);
            min_p_model_assert_position(&mut replay, &route, position + 1);
        }
        assert_eq!(min_p_model_history(&mut replay, &route), expected_history);
        // A filtered call must decline before even looking up this invalid
        // input token, preserving the already-active lane's KV and pages.
        for unsupported in [
            SamplingConfig {
                min_keep: Some(2),
                ..config.clone()
            },
            SamplingConfig {
                top_k: Some(2),
                ..config.clone()
            },
            SamplingConfig {
                presence_penalty: 0.1,
                ..config.clone()
            },
        ] {
            assert!(replay
                .generate_next_token_sampled_resident(u32::MAX, &unsupported)
                .unwrap()
                .is_none());
            min_p_model_assert_position(&mut replay, &route, inputs.len());
            assert_eq!(history(&replay), expected_history);
        }
        replayed.push(tokens);
    }
    assert_eq!(
        replayed[0], direct_tokens,
        "public entry must thread the same per-position seed"
    );
    assert_eq!(
        replayed[0], replayed[1],
        "same seed must replay token-for-token"
    );
    assert_ne!(
        replayed[0], replayed[2],
        "different seeds must change the sampled stream"
    );
    println!("\n{COMPLETED}");
}

fn min_p_gguf_history(session: &mut LlamaInferenceSession, route: &str) -> Vec<u32> {
    if route != "cpu-prefix" {
        assert!(session
            .recover_cpu_kv_from_cuda_resident(session.kv_position())
            .unwrap());
    }
    assert!(session.cpu_kv_authoritative());
    let dims = DenseLlamaDims::from_config(&session.config).unwrap();
    let mut result = Vec::new();
    let mut row = vec![0.0; dims.head_dim];
    for layer in 0..dims.block_count {
        for position in 0..session.kv_position() {
            for head in 0..dims.attention_head_count_kv {
                session
                    .kv_cache
                    .copy_key_row_into(layer, position, head, &mut row);
                result.extend(row.iter().map(|value| value.to_bits()));
                session
                    .kv_cache
                    .copy_value_row_into(layer, position, head, &mut row);
                result.extend(row.iter().map(|value| value.to_bits()));
            }
        }
    }
    result
}

fn min_p_gguf_assert_support(logits: &CpuTensor, config: &SamplingConfig, token: u32) {
    assert!(
        sampling_weights_with_config(logits, config, &[])
            .unwrap()
            .iter()
            .any(|(candidate, weight)| *candidate == token as usize && *weight > 0.0),
        "sampled token {token} must belong to the CPU sampler's retained support"
    );
}

fn min_p_gguf_benchmark(
    model_config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    route: &str,
    prompt: &[u32],
    config: &SamplingConfig,
) -> serde_json::Value {
    const WARM: usize = 8;
    const MEASURED: usize = 64;
    const TRIALS: usize = 5;
    let inputs: Vec<_> = prompt
        .iter()
        .copied()
        .cycle()
        .take(WARM + MEASURED)
        .collect();
    let mut trials = Vec::new();
    let mut mean_token_us = [Vec::new(), Vec::new()];
    let mut device_samples = [0_usize; 2];
    let mut cpu_samples = [0_usize; 2];
    for trial in 0..TRIALS {
        // Alternate which arm runs first to reduce ordering/thermal bias.
        for device_tail in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let arm = usize::from(device_tail);
            let mut session =
                LlamaInferenceSession::new(model_config.clone(), weights.clone()).unwrap();
            for &input in &inputs[..WARM] {
                let output =
                    min_p_model_forward(&mut session, input, device_tail.then_some(config));
                session
                    .finish_resident_temperature_sample(output, config, true)
                    .unwrap()
                    .unwrap();
            }
            min_p_model_assert_position(&mut session, route, WARM);
            let mut token_us = Vec::with_capacity(MEASURED);
            let mut trial_device_samples = 0;
            let mut trial_cpu_samples = 0;
            for &input in &inputs[WARM..] {
                // Both arms consume the same token at the same context position.
                // Loading, JIT, warmup, bookkeeping and KV verification are all
                // outside this timer. Each forward synchronizes its result.
                let started = std::time::Instant::now();
                let output =
                    min_p_model_forward(&mut session, input, device_tail.then_some(config));
                let sampled_on_device = matches!(&output, ResidentForward::Sampled(_));
                session
                    .finish_resident_temperature_sample(output, config, true)
                    .unwrap()
                    .unwrap();
                let elapsed = started.elapsed();
                token_us.push(elapsed.as_secs_f64() * 1_000_000.0);
                trial_device_samples += usize::from(sampled_on_device);
                trial_cpu_samples += usize::from(!sampled_on_device);
            }
            min_p_model_assert_position(&mut session, route, inputs.len());
            let total_us = token_us.iter().sum::<f64>();
            mean_token_us[arm].push(total_us / MEASURED as f64);
            device_samples[arm] += trial_device_samples;
            cpu_samples[arm] += trial_cpu_samples;
            trials.push(serde_json::json!({
                "trial": trial + 1,
                "arm": if device_tail { "cuda_min_p_tail" } else { "cuda_logits_cpu_sampler" },
                "measured_tokens": MEASURED,
                "total_forward_and_sample_ms": total_us / 1_000.0,
                "token_latency_us": token_us,
                "device_samples": trial_device_samples,
                "cpu_samples": trial_cpu_samples,
            }));
        }
    }
    let summaries: Vec<_> = mean_token_us
        .iter()
        .enumerate()
        .map(|(arm, means)| {
            let mut sorted = means.clone();
            sorted.sort_by(f64::total_cmp);
            serde_json::json!({
                "arm": if arm == 1 { "cuda_min_p_tail" } else { "cuda_logits_cpu_sampler" },
                "mean_token_us_by_trial": means,
                "median_mean_token_us": sorted[TRIALS / 2],
                "minimum_mean_token_us": sorted[0],
                "maximum_mean_token_us": sorted[TRIALS - 1],
                "measured_tokens": TRIALS * MEASURED,
                "device_samples": device_samples[arm],
                "cpu_samples": cpu_samples[arm],
            })
        })
        .collect();
    serde_json::json!({
        "mode": "warmed_matched_input_forward_and_sample",
        "timing_scope": "synchronized forward and sampling only; excludes load, JIT, warmup, session creation, assertions and KV recovery",
        "warm_positions": WARM,
        "measured_positions": MEASURED,
        "paired_trials": TRIALS,
        "alternating_arm_order": true,
        "input_ids": inputs,
        "seed": config.seed,
        "temperature": config.temperature,
        "min_p": config.min_p,
        "summaries": summaries,
        "trials": trials,
    })
}

#[test]
#[ignore = "requires CAMELID_MIN_P_GGUF and CUDA; runs full GGUF lanes in isolated subprocesses"]
fn cuda_min_p_full_gguf_qualification() {
    const CHILD_ENV: &str = "CAMELID_TEST_MIN_P_GGUF_ROUTE";
    const COMPLETED: &str = "CUDA_MIN_P_GGUF_ASSERTIONS_PASSED";
    let model = std::env::var_os("CAMELID_MIN_P_GGUF")
        .map(std::path::PathBuf::from)
        .expect("set CAMELID_MIN_P_GGUF to a complete dense Llama Q8_0 GGUF");
    assert!(model.is_file(), "GGUF does not exist: {}", model.display());
    let Ok(route) = std::env::var(CHILD_ENV) else {
        for route in ["contiguous", "paged", "cpu-prefix"] {
            eprintln!("Starting full GGUF min-p qualification: {route}");
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "inference::cuda_cpu_prefix::integration_tests::cuda_min_p_full_gguf_qualification",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, route)
                .env(PREFIX_ENV, if route == "cpu-prefix" { "2" } else { "0" })
                .env("CAMELID_CUDA_PAGED_KV", if route == "paged" { "1" } else { "0" })
                .env("CAMELID_CUDA_RESIDENT_DECODE", "1")
                .env("CAMELID_GPU_TEMP_SAMPLING", "1")
                .env("CAMELID_CUDA_MIN_P_SAMPLING", "1")
                .env("CAMELID_CUDA_Q8", "0")
                .env("CAMELID_DETERMINISTIC", "0")
                .env("CAMELID_CUDA_TWO_SEQUENCE_KV", "0")
                .env("CAMELID_CUDA_SEQUENCE_SLOTS", "1")
                .env("CAMELID_CUDA_TRUE_BATCH2", "0")
                .env("CAMELID_CUDA_TRUE_BATCH", "0")
                .output()
                .expect("launch full GGUF qualification subprocess");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "{route} GGUF qualification failed: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            assert!(
                stdout.lines().any(|line| line == COMPLETED),
                "{route} did not execute its assertions\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            for line in stdout.lines().filter(|line| line.starts_with('{')) {
                println!("{line}");
            }
        }
        return;
    };
    assert!(["contiguous", "paged", "cpu-prefix"].contains(&route.as_str()));
    assert!(
        crate::cuda::is_available(),
        "a usable CUDA device is required"
    );
    crate::cuda::set_gpu_accel_enabled(true);
    crate::cuda::set_runtime_enabled(false);
    assert!(resident_decode_cuda_enabled());

    let gguf = crate::gguf::read_metadata(&model).expect("read full GGUF metadata");
    let mut model_config = LlamaModelConfig::from_gguf(&gguf).expect("dense model config");
    assert_eq!(
        model_config.architecture, "llama",
        "the three-route gate requires Llama because paged KV currently admits only Llama"
    );
    // Bound only KV allocation, retaining every original model layer and weight.
    model_config.context_length = 256;
    model_config.kv_quant = crate::model::KvCacheQuantization::F16;
    let binding = crate::model::LlamaTensorBinding::bind(&gguf, &model_config).unwrap();
    let store = crate::tensor::TensorStore::open(&model, &gguf);
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).unwrap());
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let prompt = tokenizer
        .encode("Once upon a time, there was", true, false)
        .unwrap();
    assert!(!prompt.is_empty() && prompt.len() < 32);
    let session = || LlamaInferenceSession::new(model_config.clone(), weights.clone()).unwrap();
    let config = SamplingConfig {
        temperature: 1.1,
        min_p: Some(0.05),
        seed: Some(27),
        ..SamplingConfig::default()
    };

    // Compare every position against an ordinary logits-returning CUDA forward.
    // Eighteen real tokenizer IDs cross a page boundary in the paged lane.
    let inputs: Vec<u32> = prompt.iter().copied().cycle().take(18).collect();
    let mut reference = session();
    let mut reference_logits = Vec::new();
    for (position, &input) in inputs.iter().enumerate() {
        let ResidentForward::Logits(logits) = min_p_model_forward(&mut reference, input, None)
        else {
            panic!("reference forward must return logits");
        };
        assert!(logits.data.iter().all(|value| value.is_finite()));
        reference.kv_cache.position += 1;
        min_p_model_assert_position(&mut reference, &route, position + 1);
        reference_logits.push(logits);
    }
    let reference_history = min_p_gguf_history(&mut reference, &route);
    drop(reference);

    // Alternate forced boundary fallbacks and ordinary device draws. Tiny
    // thresholds retain multiple candidates, so fallback seed progression is
    // observable instead of testing only a p=1 argmax. Position 15 falls back
    // and position 16 starts a new KV page.
    let mut fallback = session();
    let mut boundary_fallbacks = 0;
    let mut device_samples = 0;
    for (position, (&input, expected)) in inputs.iter().zip(&reference_logits).enumerate() {
        let boundary = SamplingConfig {
            min_p: Some(match position % 4 {
                1 => 1.0,
                3 => f32::MIN_POSITIVE,
                _ => 0.05,
            }),
            ..config.clone()
        };
        let output = min_p_model_forward(&mut fallback, input, Some(&boundary));
        if position % 2 == 1 {
            let ResidentForward::Logits(actual) = &output else {
                panic!("boundary fallback must return the existing logits for CPU sampling");
            };
            assert_eq!(actual.data, expected.data, "boundary logits at {position}");
            boundary_fallbacks += 1;
            if position % 4 == 3 {
                assert!(
                    sampling_weights_with_config(expected, &boundary, &[])
                        .unwrap()
                        .len()
                        > 1
                );
            }
        } else {
            device_samples += usize::from(matches!(&output, ResidentForward::Sampled(_)));
        }
        let token = fallback
            .finish_resident_temperature_sample(output, &boundary, true)
            .unwrap()
            .unwrap();
        min_p_gguf_assert_support(expected, &boundary, token);
        if position % 2 == 1 {
            assert_eq!(
                token,
                sample_with_config(expected, &boundary, &inputs[..=position]).unwrap()
            );
        }
        min_p_model_assert_position(&mut fallback, &route, position + 1);
    }
    assert_eq!(boundary_fallbacks, 9);
    assert!(
        device_samples > 0,
        "full model must execute device min-p draws"
    );
    assert_eq!(min_p_gguf_history(&mut fallback, &route), reference_history);
    drop(fallback);

    // Exercise the real general-generation fallback, not only the admission
    // predicate: it must consume one row and agree with the CPU sampler on the
    // independently captured reference logits, including its advancing seed.
    let mut general_samples = 0;
    for unsupported in [
        SamplingConfig {
            min_p: Some(1.0),
            min_keep: Some(4),
            ..config.clone()
        },
        SamplingConfig {
            top_k: Some(32),
            top_p: Some(0.8),
            min_p: Some(0.2),
            min_keep: Some(2),
            ..config.clone()
        },
    ] {
        let mut general = session();
        for (position, (&input, expected)) in inputs.iter().zip(&reference_logits).enumerate() {
            assert!(general
                .generate_next_token_sampled_resident(u32::MAX, &unsupported)
                .unwrap()
                .is_none());
            assert_eq!(general.kv_position(), position);
            let step = general
                .generate_next_token_with_history_diagnostics(
                    &[input],
                    LlamaSampler::Sampling(unsupported.clone()),
                    &inputs[..=position],
                    false,
                    None,
                )
                .unwrap();
            assert_eq!(
                step.logits.data, expected.data,
                "general logits at {position}"
            );
            assert_eq!(
                step.next_token_id,
                sample_with_config(expected, &unsupported, &inputs[..=position]).unwrap()
            );
            if unsupported.min_p == Some(1.0) {
                assert!(
                    sampling_weights_with_config(expected, &unsupported, &[])
                        .unwrap()
                        .len()
                        >= 4
                );
            }
            min_p_model_assert_position(&mut general, &route, position + 1);
            general_samples += 1;
        }
        assert_eq!(min_p_gguf_history(&mut general, &route), reference_history);
    }

    // Real autoregressive requests: generated tokens feed subsequent forwards.
    // The first request exposes the internal tail for execution counting; its
    // same-seed replay and the changed-seed request use the public fast entry.
    let mut streams = Vec::new();
    let mut autoregressive_device_samples = 0;
    let mut supported_samples = 0;
    for (request, seed) in [27, 27, 83].into_iter().enumerate() {
        let request_config = SamplingConfig {
            seed: Some(seed),
            ..config.clone()
        };
        let mut generated = session();
        for &input in &prompt[..prompt.len() - 1] {
            min_p_model_forward(&mut generated, input, None);
            generated.kv_cache.position += 1;
        }
        let mut input = *prompt.last().unwrap();
        let mut tokens = Vec::new();
        for _ in 0..24 {
            let token = if request == 0 {
                let output = min_p_model_forward(&mut generated, input, Some(&request_config));
                autoregressive_device_samples +=
                    usize::from(matches!(&output, ResidentForward::Sampled(_)));
                generated
                    .finish_resident_temperature_sample(output, &request_config, true)
                    .unwrap()
                    .unwrap()
            } else {
                generated
                    .generate_next_token_sampled_resident(input, &request_config)
                    .unwrap()
                    .expect("public resident min-p request must execute")
                    .0
            };
            tokens.push(token);
            input = token;
            min_p_model_assert_position(&mut generated, &route, prompt.len() + tokens.len() - 1);
        }
        let generated_history = min_p_gguf_history(&mut generated, &route);
        drop(generated);

        // Replay the exact consumed inputs through the ordinary logits path,
        // checking the CPU's retained support for every emitted device token.
        let mut oracle = session();
        let consumed: Vec<_> = prompt
            .iter()
            .chain(&tokens[..tokens.len() - 1])
            .copied()
            .collect();
        for (position, &input) in consumed.iter().enumerate() {
            let ResidentForward::Logits(logits) = min_p_model_forward(&mut oracle, input, None)
            else {
                panic!("autoregressive oracle must return logits");
            };
            oracle.kv_cache.position += 1;
            if position + 1 >= prompt.len() {
                min_p_gguf_assert_support(
                    &logits,
                    &request_config,
                    tokens[position + 1 - prompt.len()],
                );
                supported_samples += 1;
            }
        }
        assert_eq!(min_p_gguf_history(&mut oracle, &route), generated_history);
        streams.push(tokens);
    }
    assert!(autoregressive_device_samples > 0);
    assert_eq!(
        streams[0], streams[1],
        "same seed must replay the entire request"
    );
    assert_ne!(
        streams[0], streams[2],
        "changed seed must change the request"
    );
    let mut receipt = serde_json::json!({
        "schema": "camelid.cuda-min-p-gguf/v1",
        "model": model.file_name().unwrap().to_string_lossy(),
        "architecture": model_config.architecture,
        "layers": model_config.block_count,
        "vocab": model_config.vocab_size,
        "route": route,
        "context_capacity": 256,
        "prompt_ids": prompt,
        "fixed_input_positions": inputs.len(),
        "boundary_fallbacks": boundary_fallbacks,
        "fixed_input_device_samples": device_samples,
        "general_sampler_reference_matches": general_samples,
        "autoregressive_device_samples": autoregressive_device_samples,
        "autoregressive_support_matches": supported_samples,
        "seeds": [27, 27, 83],
        "generated_tokens": streams,
        "same_seed_replay": true,
        "changed_seed_changes_output": true,
        "complete_kv_history_matches": true,
        "one_kv_row_per_forward": true,
        "passed": true,
    });
    if std::env::var("CAMELID_MIN_P_BENCH").as_deref() == Ok("1") {
        receipt["performance"] =
            min_p_gguf_benchmark(&model_config, &weights, &route, &prompt, &config);
    }
    println!("\n{receipt}");
    println!("{COMPLETED}");
}
