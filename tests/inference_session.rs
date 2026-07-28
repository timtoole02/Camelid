use camelid::{
    inference::{
        LlamaInferenceSession, LlamaKvCachePlan, LlamaLayerWeights, LlamaLoadedWeights,
        LlamaSampler, SamplingConfig,
    },
    model::LlamaModelConfig,
    tensor::CpuTensor,
};

#[test]
fn plans_llama_kv_cache_shape() {
    let config = tiny_config();

    let plan = LlamaKvCachePlan::from_config(&config).unwrap();

    assert_eq!(plan.max_sequence_length, 4);
    assert_eq!(plan.layer_count, 1);
    assert_eq!(plan.kv_head_count, 1);
    assert_eq!(plan.head_dim, 2);
    assert_eq!(plan.key_shape, vec![1, 4, 1, 2]);
    assert_eq!(plan.value_shape, vec![1, 4, 1, 2]);
}

#[test]
fn runs_single_token_dense_llama_forward_skeleton() {
    let config = tiny_config();
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let output = session.forward_single_token(1).unwrap();

    assert_eq!(output.logits.shape.dims, vec![1, 3]);
    assert_eq!(output.hidden_state.shape.dims, vec![1, 4]);
    assert_eq!(session.kv_cache.position, 1);
    assert_approx_eq(session.kv_cache.keys[0], 1.999984);
    assert_approx_eq(session.kv_cache.keys[1], 0.0);
    assert_approx_eq(session.kv_cache.values[0], 0.999992);
    assert_approx_eq(session.kv_cache.values[1], 0.0);
    assert!(output.logits.data.iter().all(|value| value.is_finite()));
}

#[test]
fn applies_rope_before_writing_current_key_to_cache() {
    let config = tiny_config();
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    session.forward_single_token(1).unwrap();
    session.forward_single_token(2).unwrap();

    let unrotated_key_y = 1.0 / (0.25_f32 + 1e-6).sqrt();
    let (sin, cos) = 1.0_f32.sin_cos();
    assert_eq!(session.kv_cache.position, 2);
    assert_approx_eq(session.kv_cache.keys[2], -unrotated_key_y * sin);
    assert_approx_eq(session.kv_cache.keys[3], unrotated_key_y * cos);
}

#[test]
fn writes_all_layers_to_same_token_position_before_advancing_cache() {
    let mut config = tiny_config();
    config.block_count = 2;
    let mut weights = tiny_weights();
    weights.layers.push(weights.layers[0].clone());
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    session.forward_single_token(1).unwrap();

    let plan = &session.kv_cache.plan;
    let layer_0_position_0 = 0;
    let layer_1_position_0 = plan.head_dim;

    assert_eq!(session.kv_cache.position, 1);
    assert_eq!(session.kv_cache.allocated_sequence_length, 1);
    assert_eq!(
        session.kv_cache.keys.len(),
        plan.layer_count * plan.kv_head_count * plan.head_dim
    );
    assert!(
        session.kv_cache.keys[layer_0_position_0..layer_0_position_0 + plan.head_dim]
            .iter()
            .any(|value| *value != 0.0)
    );
    assert!(
        session.kv_cache.keys[layer_1_position_0..layer_1_position_0 + plan.head_dim]
            .iter()
            .any(|value| *value != 0.0)
    );
}

#[test]
fn generates_next_token_after_prompt_prefill_with_greedy_sampling() {
    let config = tiny_config();
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let step = session
        .generate_next_token(&[1, 2], LlamaSampler::Greedy)
        .unwrap();

    assert_eq!(step.prompt_token_count, 2);
    assert_eq!(step.logits.shape.dims, vec![1, 3]);
    assert_eq!(session.kv_cache.position, 2);
    assert!(step.logits.data.iter().all(|value| value.is_finite()));
    assert_eq!(
        step.next_token_id,
        LlamaSampler::Greedy.sample(&step.logits).unwrap()
    );
    assert_eq!(step.timings.layers.len(), 1);
    assert_eq!(step.timings.layers[0].layer_index, 0);
}

#[test]
fn prompt_prefill_writes_every_layer_at_each_token_position() {
    let mut config = tiny_config();
    config.block_count = 2;
    let mut weights = tiny_weights();
    weights.layers.push(weights.layers[0].clone());
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    session
        .generate_next_token(&[1, 2], LlamaSampler::Greedy)
        .unwrap();

    let plan = &session.kv_cache.plan;
    assert_eq!(session.kv_cache.allocated_sequence_length, 2);
    assert_eq!(
        session.kv_cache.keys.len(),
        2 * plan.layer_count * plan.kv_head_count * plan.head_dim
    );

    for layer_idx in 0..plan.layer_count {
        for position in 0..2 {
            let start =
                ((position * plan.layer_count + layer_idx) * plan.kv_head_count) * plan.head_dim;
            assert!(
                session.kv_cache.keys[start..start + plan.head_dim]
                    .iter()
                    .any(|value| *value != 0.0),
                "expected layer {layer_idx} position {position} to be populated"
            );
        }
    }

    assert_eq!(session.kv_cache.position, 2);
}

#[test]
fn greedy_sampler_selects_highest_logit_and_lowest_tie() {
    let logits = tensor("logits", vec![1, 4], vec![0.5, 2.0, 2.0, -1.0]);

    let token_id = LlamaSampler::Greedy.sample(&logits).unwrap();

    assert_eq!(token_id, 1);
}

#[test]
fn temperature_zero_sampling_preserves_greedy_tie_breaking() {
    let logits = tensor("logits", vec![1, 4], vec![0.5, 2.0, 2.0, -1.0]);
    let sampler = LlamaSampler::Sampling(SamplingConfig {
        temperature: 0.0,
        top_k: Some(2),
        top_p: Some(0.5),
        seed: Some(42),
        ..SamplingConfig::default()
    });

    let token_id = sampler.sample(&logits).unwrap();

    assert_eq!(token_id, 1);
}

#[test]
fn seeded_temperature_sampling_honors_top_k_and_top_p_filters() {
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);

    let top_k_token = LlamaSampler::Sampling(SamplingConfig {
        temperature: 1.0,
        top_k: Some(1),
        top_p: None,
        seed: Some(0),
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap();
    let top_p_token = LlamaSampler::Sampling(SamplingConfig {
        temperature: 1.0,
        top_k: None,
        top_p: Some(0.8),
        seed: Some(0),
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap();

    assert_eq!(top_k_token, 0);
    assert_eq!(top_p_token, 1);
}

#[test]
fn rejects_invalid_sampling_config() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, 1.0]);
    let err = LlamaSampler::Sampling(SamplingConfig {
        temperature: 1.0,
        top_k: Some(0),
        top_p: None,
        seed: None,
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap_err()
    .to_string();

    assert!(err.contains("top_k"));
}

#[test]
fn logit_bias_adjusts_greedy_selection_deterministically() {
    let logits = tensor("logits", vec![1, 3], vec![0.0, 0.5, 0.4]);
    let token_id = LlamaSampler::Sampling(SamplingConfig {
        logit_bias: vec![(2, 0.2)],
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap();

    assert_eq!(token_id, 2);
}

#[test]
fn penalties_apply_to_seen_tokens_before_sampling() {
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.0]);
    let token_id = LlamaSampler::Sampling(SamplingConfig {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &[0, 0])
    .unwrap();

    assert_eq!(token_id, 1);
}

#[test]
fn rejects_logit_bias_outside_vocabulary() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, 1.0]);
    let err = LlamaSampler::Sampling(SamplingConfig {
        logit_bias: vec![(2, 1.0)],
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap_err()
    .to_string();

    assert!(err.contains("outside vocabulary"));
}

#[test]
fn top_k_one_is_argmax_regardless_of_seed() {
    // class-I invariant: top_k=1 collapses to the greedy argmax for any seed.
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);
    for seed in [0u64, 1, 7, 42, 123_456_789] {
        let token = LlamaSampler::Sampling(SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            seed: Some(seed),
            ..SamplingConfig::default()
        })
        .sample(&logits)
        .unwrap();
        assert_eq!(token, 0, "top_k=1 must pick the argmax for seed {seed}");
    }
}

#[test]
fn min_p_one_keeps_only_argmax() {
    // class-I invariant: min_p=1.0 keeps only the max-probability token.
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);
    for seed in [0u64, 5, 99, 2_024] {
        let token = LlamaSampler::Sampling(SamplingConfig {
            temperature: 1.0,
            min_p: Some(1.0),
            seed: Some(seed),
            ..SamplingConfig::default()
        })
        .sample(&logits)
        .unwrap();
        assert_eq!(token, 0, "min_p=1.0 must pick the argmax for seed {seed}");
    }
}

#[test]
fn min_p_zero_is_a_noop() {
    // class-I invariant: min_p=0.0 must not change the sampled token vs no min_p.
    let logits = tensor("logits", vec![1, 4], vec![1.0, 0.5, 0.25, 0.0]);
    for seed in [0u64, 3, 17, 555] {
        let baseline = LlamaSampler::Sampling(SamplingConfig {
            temperature: 1.0,
            seed: Some(seed),
            ..SamplingConfig::default()
        })
        .sample(&logits)
        .unwrap();
        let with_zero = LlamaSampler::Sampling(SamplingConfig {
            temperature: 1.0,
            min_p: Some(0.0),
            seed: Some(seed),
            ..SamplingConfig::default()
        })
        .sample(&logits)
        .unwrap();
        assert_eq!(
            baseline, with_zero,
            "min_p=0 changed the draw for seed {seed}"
        );
    }
}

#[test]
fn rejects_min_p_out_of_range() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, 1.0]);
    let err = LlamaSampler::Sampling(SamplingConfig {
        temperature: 1.0,
        min_p: Some(1.5),
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap_err()
    .to_string();
    assert!(err.contains("min_p"), "got {err}");
}

#[test]
fn repeat_penalty_one_is_a_noop() {
    // class-I invariant: repeat_penalty=1.0 leaves the seen-token logits untouched.
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.0]);
    let with_one = LlamaSampler::Sampling(SamplingConfig {
        repeat_penalty: 1.0,
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &[0, 0])
    .unwrap();
    assert_eq!(with_one, 0);
}

#[test]
fn repeat_penalty_demotes_a_repeated_token() {
    // class-I direction invariant: a penalty > 1 pushes a seen token below an
    // unseen rival. Greedy would pick token 0 (1.0); dividing its positive logit
    // by 2.0 (=0.5) lets the unseen token 1 (0.9) win.
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.0]);
    let token = LlamaSampler::Sampling(SamplingConfig {
        repeat_penalty: 2.0,
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &[0])
    .unwrap();
    assert_eq!(token, 1);
}

#[test]
fn rejects_non_positive_repeat_penalty() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, 1.0]);
    let err = LlamaSampler::Sampling(SamplingConfig {
        repeat_penalty: 0.0,
        ..SamplingConfig::default()
    })
    .sample(&logits)
    .unwrap_err()
    .to_string();
    assert!(err.contains("repeat_penalty"), "got {err}");
}

#[test]
fn seeded_sampling_advances_per_decode_step() {
    // Regression for the degenerate-RNG bug: with a fixed seed the per-step draw
    // used to be constant, so every decode step returned the same token. With the
    // per-position advance a uniform distribution yields more than one distinct
    // token across steps — and the whole sequence stays reproducible.
    let logits = tensor("logits", vec![1, 4], vec![0.0, 0.0, 0.0, 0.0]);
    let run = || {
        (0..48u32)
            .map(|step| {
                let history = vec![0u32; step as usize];
                LlamaSampler::Sampling(SamplingConfig {
                    temperature: 1.0,
                    seed: Some(0x00C0_FFEE),
                    ..SamplingConfig::default()
                })
                .sample_with_history(&logits, &history)
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let first = run();
    let second = run();
    let distinct: std::collections::BTreeSet<u32> = first.iter().copied().collect();
    assert!(
        distinct.len() >= 2,
        "per-step draw did not advance: all {} steps returned the same token",
        first.len()
    );
    assert_eq!(
        first, second,
        "a fixed seed must reproduce the sequence token-for-token"
    );
}

#[test]
fn rejects_empty_prompt_for_next_token_generation() {
    let config = tiny_config();
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let err = session
        .generate_next_token(&[], LlamaSampler::Greedy)
        .unwrap_err()
        .to_string();

    assert!(err.contains("at least one prompt token"));
    assert_eq!(session.kv_cache.position, 0);
}

#[test]
fn rejects_non_finite_sampler_logits() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, f32::NAN]);

    let err = LlamaSampler::Greedy
        .sample(&logits)
        .unwrap_err()
        .to_string();

    assert!(err.contains("non-finite"));
}

#[test]
fn rejects_loaded_weight_shape_before_forward() {
    let config = tiny_config();
    let mut weights = tiny_weights();
    weights.layers[0].attention_k = tensor(
        "blk.0.attn_k.weight",
        vec![3, config.embedding_length as usize],
        vec![0.0; 3 * config.embedding_length as usize],
    );

    let err = LlamaInferenceSession::new(config, weights)
        .unwrap_err()
        .to_string();

    assert!(err.contains("attention k"));
    assert!(err.contains("blk.0.attn_k.weight"));
}

#[test]
fn rejects_token_past_context_length() {
    let mut config = tiny_config();
    config.context_length = 1;
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    session.forward_single_token(0).unwrap();
    let err = session.forward_single_token(0).unwrap_err().to_string();

    assert!(err.contains("KV cache is full"));
}

#[test]
fn rejects_generation_prompt_that_exceeds_remaining_context_before_cache_advance() {
    let mut config = tiny_config();
    config.context_length = 2;
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    session.forward_single_token(0).unwrap();
    let err = session
        .generate_next_token(&[1, 2], LlamaSampler::Greedy)
        .unwrap_err()
        .to_string();

    assert!(err.contains("exceeds remaining context capacity 1"));
    assert_eq!(session.kv_cache.position, 1);
}

#[test]
fn rejects_invalid_sampling_config_before_cache_advance() {
    let config = tiny_config();
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let err = session
        .generate_next_token(
            &[1],
            LlamaSampler::Sampling(SamplingConfig {
                temperature: f32::NAN,
                ..SamplingConfig::default()
            }),
        )
        .unwrap_err()
        .to_string();

    assert!(err.contains("temperature"));
    assert_eq!(session.kv_cache.position, 0);
}

#[test]
fn rejects_invalid_rope_dimension_before_cache_advance() {
    let mut config = tiny_config();
    config.rope_dimension_count = Some(3);
    let weights = tiny_weights();
    let mut session = LlamaInferenceSession::new(config, weights).unwrap();

    let err = session.forward_single_token(1).unwrap_err().to_string();

    assert!(err.contains("RoPE dimension count 3"));
    assert_eq!(session.kv_cache.position, 0);
}

#[test]
fn rejects_loaded_layer_count_mismatch_before_forward() {
    let config = tiny_config();
    let mut weights = tiny_weights();
    weights.layers.clear();

    let err = LlamaInferenceSession::new(config, weights)
        .unwrap_err()
        .to_string();

    assert!(err.contains("config block count 1"));
    assert!(err.contains("loaded layer count 0"));
}

#[test]
fn rejects_attention_head_configuration_that_cannot_share_kv_heads() {
    let mut config = tiny_config();
    config.attention_head_count_kv = 3;
    let weights = tiny_weights();

    let err = LlamaInferenceSession::new(config, weights)
        .unwrap_err()
        .to_string();

    assert!(err.contains("attention head count 2"));
    assert!(err.contains("kv head count 3"));
}

fn tiny_config() -> LlamaModelConfig {
    LlamaModelConfig {
        context_length: 4,
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 6,
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
        kv_quant: camelid::model::KvCacheQuantization::F16,
        vocab_size: Some(3),
        file_type: Some(0),
        rope_neox_pairing: false,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma4: None,
        qwen35: None,
        mla: None,
    }
}

fn tiny_weights() -> LlamaLoadedWeights {
    let hidden = 4;
    let ffn = 6;
    LlamaLoadedWeights {
        token_embedding: tensor(
            "token_embd.weight",
            vec![3, hidden],
            vec![
                1.0, 0.0, 0.0, 0.0, // token 0
                0.5, 0.0, 0.0, 0.0, // token 1
                0.0, 1.0, 0.0, 0.0, // token 2
            ],
        ),
        output_norm: ones("output_norm.weight", hidden),
        output: Some(tensor(
            "output.weight",
            vec![3, hidden],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )),
        rope_freqs: None,
        layers: vec![LlamaLayerWeights {
            attention_norm: ones("blk.0.attn_norm.weight", hidden),
            attention_q: select_rows("blk.0.attn_q.weight", hidden, hidden, &[0, 1, 2, 3]),
            attention_k: select_rows("blk.0.attn_k.weight", 2, hidden, &[0, 1]),
            attention_v: scaled_select_rows("blk.0.attn_v.weight", 2, hidden, &[0, 1], 0.5),
            attention_output: select_rows(
                "blk.0.attn_output.weight",
                hidden,
                hidden,
                &[0, 1, 2, 3],
            ),
            ffn_norm: ones("blk.0.ffn_norm.weight", hidden),
            ffn_gate: select_rows("blk.0.ffn_gate.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
            ffn_up: select_rows("blk.0.ffn_up.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
            ffn_down: select_rows("blk.0.ffn_down.weight", hidden, ffn, &[0, 1, 2, 3]),
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
            decode_bindings: camelid::inference::DecodeLinearBindings::default(),
        }],
        layer_range: None,
        output_projection_binding: camelid::inference::DecodeBindingCell::default(),
    }
}

fn assert_approx_eq(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 5e-4,
        "expected {actual} to be within tolerance of {expected}"
    );
}

fn ones(name: &str, width: usize) -> CpuTensor {
    tensor(name, vec![width], vec![1.0; width])
}

fn select_rows(name: &str, rows: usize, cols: usize, source_cols: &[usize]) -> CpuTensor {
    scaled_select_rows(name, rows, cols, source_cols, 1.0)
}

fn scaled_select_rows(
    name: &str,
    rows: usize,
    cols: usize,
    source_cols: &[usize],
    scale: f32,
) -> CpuTensor {
    let mut data = vec![0.0; rows * cols];
    for (row, source_col) in source_cols.iter().copied().enumerate() {
        data[row * cols + source_col] = scale;
    }
    tensor(name, vec![rows, cols], data)
}

fn tensor(name: &str, dims: Vec<usize>, data: Vec<f32>) -> CpuTensor {
    CpuTensor::from_f32(name, dims, data).unwrap()
}
