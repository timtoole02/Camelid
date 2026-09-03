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
fn penalty_last_n_windows_the_history() {
    // logits [1.0, 0.9, 0.8], history [0, 0, 0, 1], presence 0.5 / frequency 0.25.
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.8]);
    let history = [0u32, 0, 0, 1];

    // Default (`penalty_last_n: None`) measures the WHOLE history, so token 0
    // takes 0.5 + 3*0.25 = -0.25, token 1 takes 0.15, and token 2 (never seen,
    // 0.8) wins. This is the pre-existing behaviour and must not change.
    let whole = LlamaSampler::Sampling(SamplingConfig {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &history)
    .unwrap();
    assert_eq!(
        whole, 2,
        "unwindowed penalties should demote token 0 hardest"
    );

    // A one-token window sees only the trailing `1`, so token 0 keeps its 1.0
    // logit and wins. Long-context/RAG callers want this: without it, every
    // token in a large prompt is penalized forever.
    let windowed = LlamaSampler::Sampling(SamplingConfig {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        penalty_last_n: Some(1),
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &history)
    .unwrap();
    assert_eq!(
        windowed, 0,
        "penalty_last_n=1 must only penalize the final history token"
    );
}

#[test]
fn penalty_last_n_zero_disables_penalties() {
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.8]);
    let token = LlamaSampler::Sampling(SamplingConfig {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        penalty_last_n: Some(0),
        ..SamplingConfig::default()
    })
    .sample_with_history(&logits, &[0u32, 0, 0, 1])
    .unwrap();
    assert_eq!(token, 0, "penalty_last_n=0 must leave the logits untouched");
}

#[test]
fn penalty_last_n_beyond_history_matches_the_whole_history() {
    // Guards the `saturating_sub` bound: a window longer than the history is the
    // whole history, not a panic and not an empty window.
    let logits = tensor("logits", vec![1, 3], vec![1.0, 0.9, 0.8]);
    let history = [0u32, 0, 0, 1];
    let config = |penalty_last_n| SamplingConfig {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        penalty_last_n,
        ..SamplingConfig::default()
    };

    let unbounded = LlamaSampler::Sampling(config(None))
        .sample_with_history(&logits, &history)
        .unwrap();
    let oversized = LlamaSampler::Sampling(config(Some(4096)))
        .sample_with_history(&logits, &history)
        .unwrap();

    assert_eq!(unbounded, oversized);
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
fn min_p_measures_the_untempered_distribution() {
    // Parity regression against llama.cpp's default sampler chain
    // (`common/common.h`: ... top_k -> typical -> top_p -> min_p -> TEMPERATURE
    // -> dist). Temperature is applied LAST, so min_p measures the raw softmax,
    // not a temperature-sharpened one.
    //
    // logits [3, 2, 1] -> untempered softmax [0.6652, 0.2447, 0.0900].
    // min_p 0.15 -> threshold 0.15 * 0.6652 = 0.0998, so {0, 1} survive and
    // token 2 (0.0900) is cut.
    //
    // The previous chain divided by temperature *before* min_p: at T=0.5 the
    // distribution sharpens to [0.8668, 0.1173, 0.0159], the threshold becomes
    // 0.1300, and only token 0 survived — token 1 was unreachable for every
    // seed. Reaching token 1 is exactly what the reorder restores.
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);
    let mut drawn = std::collections::BTreeSet::new();
    for seed in 0u64..64 {
        drawn.insert(
            LlamaSampler::Sampling(SamplingConfig {
                temperature: 0.5,
                min_p: Some(0.15),
                seed: Some(seed),
                ..SamplingConfig::default()
            })
            .sample(&logits)
            .unwrap(),
        );
    }
    assert!(
        drawn.contains(&1),
        "token 1 must be reachable once min_p measures the untempered distribution, drew {drawn:?}"
    );
    assert!(
        !drawn.contains(&2),
        "min_p must still cut token 2, drew {drawn:?}"
    );
}

#[test]
fn top_p_measures_the_untempered_distribution() {
    // Same parity regression, on the top_p stage.
    //
    // logits [3, 2, 1] -> untempered softmax [0.6652, 0.2447, 0.0900].
    // top_p 0.8 accumulates 0.6652 (< 0.8), then 0.9099 (>= 0.8), keeping {0, 1}.
    //
    // Dividing by temperature first (T=0.5) gave [0.8668, ...], whose very first
    // element already clears 0.8, so the old chain kept only token 0.
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);
    let mut drawn = std::collections::BTreeSet::new();
    for seed in 0u64..64 {
        drawn.insert(
            LlamaSampler::Sampling(SamplingConfig {
                temperature: 0.5,
                top_p: Some(0.8),
                seed: Some(seed),
                ..SamplingConfig::default()
            })
            .sample(&logits)
            .unwrap(),
        );
    }
    assert!(
        drawn.contains(&1),
        "token 1 must be reachable once top_p measures the untempered distribution, drew {drawn:?}"
    );
    assert!(
        !drawn.contains(&2),
        "top_p must still cut token 2, drew {drawn:?}"
    );
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

#[test]
fn smollm3_nope_layers_are_every_fourth_zero_based() {
    // llama.cpp models/smollm3.cpp:69 — use_rope = (il + 1) % 4 != 0, il 0-based.
    // For the 36-layer SmolLM3-3B (smollm3.cpp:7-10 maps case 36 => LLM_TYPE_3B)
    // that skips 9 of 36 layers, INCLUDING the final one.
    let config = LlamaModelConfig {
        architecture: "llama".to_string(),
        block_count: 36,
        no_rope_layer_step: Some(4),
        ..tiny_config()
    };

    let skipped: Vec<usize> = (0..36).filter(|i| !config.layer_uses_rope(*i)).collect();
    assert_eq!(skipped, vec![3, 7, 11, 15, 19, 23, 27, 31, 35]);
    assert_eq!((0..36).filter(|i| config.layer_uses_rope(*i)).count(), 27);

    // Layer 35 is the case an off-by-one silently gets wrong.
    assert!(!config.layer_uses_rope(35));
    assert!(config.layer_uses_rope(0));
}

#[test]
fn layer_uses_rope_matches_the_reference_formula() {
    // Proves the predicate is the reference formula across steps, not something
    // tuned to reproduce step 4.
    for step in [1u32, 2, 3, 4, 5, 8] {
        let config = LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: Some(step),
            ..tiny_config()
        };
        for il in 0..24usize {
            assert_eq!(
                config.layer_uses_rope(il),
                (il + 1) % (step as usize) != 0,
                "step {step} layer {il}"
            );
        }
    }

    // None and Some(0) both mean "rope every layer"; Some(0) defensively, so the
    // modulo stays total (mirrors the `step > 0 &&` guard in llama.cpp
    // models/afmoe.cpp:137).
    for step in [None, Some(0)] {
        let config = LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: step,
            ..tiny_config()
        };
        assert!(
            (0..24).all(|il| config.layer_uses_rope(il)),
            "step {step:?} must rope every layer"
        );
    }
}

#[test]
fn nope_layer_is_identity_at_position_zero() {
    // At position 0 the rotation angle is 0 (cos=1, sin=0), so RoPE is the
    // identity and skipping it must give bit-identical logits. This proves the
    // skip branch does not corrupt the tensor, its shape, or the decode scratch
    // pool recycle that reuses q_before_rope/k_before_rope.
    let mut roped = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
    let mut nope = LlamaInferenceSession::new(
        LlamaModelConfig {
            architecture: "llama".to_string(),
            // (0 + 1) % 1 == 0, so the single layer 0 is NoPE.
            no_rope_layer_step: Some(1),
            ..tiny_config()
        },
        tiny_weights(),
    )
    .unwrap();

    let roped_out = roped.forward_single_token(1).unwrap();
    let nope_out = nope.forward_single_token(1).unwrap();

    assert_eq!(roped_out.logits.data, nope_out.logits.data);
}

#[test]
fn nope_layer_changes_output_at_nonzero_position() {
    // Companion to the identity test: without this, that test would still pass if
    // the gate were dead code. At position 1 the rotation is not the identity, so
    // skipping it must change the logits.
    let mut roped = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
    let mut nope = LlamaInferenceSession::new(
        LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: Some(1),
            ..tiny_config()
        },
        tiny_weights(),
    )
    .unwrap();

    roped.forward_single_token(1).unwrap();
    nope.forward_single_token(1).unwrap();
    let roped_out = roped.forward_single_token(2).unwrap();
    let nope_out = nope.forward_single_token(2).unwrap();

    assert_ne!(
        roped_out.logits.data, nope_out.logits.data,
        "the NoPE gate must be live on the decode path"
    );
}

#[test]
fn nope_step_two_ropes_layer_zero() {
    // Pins the 0-based `(il + 1)` convention numerically. With step 2, layer 0 has
    // (0 + 1) % 2 == 1 != 0, so it IS roped and must match the baseline exactly.
    // An `il % step` implementation — the convention llama.cpp uses in
    // models/smallthinker.cpp:109 — would skip layer 0 here and fail.
    let mut baseline = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
    let mut stepped = LlamaInferenceSession::new(
        LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: Some(2),
            ..tiny_config()
        },
        tiny_weights(),
    )
    .unwrap();

    for token in [1u32, 2] {
        let baseline_out = baseline.forward_single_token(token).unwrap();
        let stepped_out = stepped.forward_single_token(token).unwrap();
        assert_eq!(
            baseline_out.logits.data, stepped_out.logits.data,
            "step 2 must rope layer 0"
        );
    }
}

#[test]
fn nope_gate_is_live_on_the_prompt_path() {
    // The prompt path has its own rope call site (forward_prefill_layer_chunk_timed);
    // this proves the skip is wired there too and not only in single-token decode.
    let prompt = [1u32, 2, 1];
    let mut roped = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
    let mut nope = LlamaInferenceSession::new(
        LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: Some(1),
            ..tiny_config()
        },
        tiny_weights(),
    )
    .unwrap();

    let roped_step = roped
        .generate_next_token(&prompt, LlamaSampler::Greedy)
        .unwrap();
    let nope_step = nope
        .generate_next_token(&prompt, LlamaSampler::Greedy)
        .unwrap();
    assert_ne!(
        roped_step.logits.data, nope_step.logits.data,
        "the NoPE gate must be live on the prompt path"
    );

    // A step that ropes layer 0 must be indistinguishable from the baseline.
    let mut stepped = LlamaInferenceSession::new(
        LlamaModelConfig {
            architecture: "llama".to_string(),
            no_rope_layer_step: Some(2),
            ..tiny_config()
        },
        tiny_weights(),
    )
    .unwrap();
    let stepped_step = stepped
        .generate_next_token(&prompt, LlamaSampler::Greedy)
        .unwrap();
    assert_eq!(roped_step.logits.data, stepped_step.logits.data);
}

#[test]
fn top_n_sigma_keeps_only_logits_within_n_standard_deviations() {
    // logits [10, 9, 1, 0]: mean 5.0, population variance
    // (25 + 16 + 16 + 25)/4 = 20.5, std 4.5277. With n = 1 the threshold is
    // 10 - 4.5277 = 5.4723, so only tokens 0 and 1 survive.
    let logits = tensor("logits", vec![1, 4], vec![10.0, 9.0, 1.0, 0.0]);
    let mut drawn = std::collections::BTreeSet::new();
    for seed in 0u64..64 {
        drawn.insert(
            LlamaSampler::Sampling(SamplingConfig {
                temperature: 1.0,
                top_n_sigma: Some(1.0),
                seed: Some(seed),
                ..SamplingConfig::default()
            })
            .sample(&logits)
            .unwrap(),
        );
    }
    assert!(
        drawn.is_subset(&std::collections::BTreeSet::from([0, 1])),
        "tokens beyond one sigma must be cut, drew {drawn:?}"
    );
    assert!(drawn.contains(&0) && drawn.contains(&1), "drew {drawn:?}");
}

#[test]
fn typical_p_can_drop_the_argmax() {
    // The property that distinguishes locally-typical sampling from every other
    // filter here: its survivors are not a prefix of the logit ordering, so the
    // most probable token can be cut.
    //
    // Written as logs of the target probabilities rather than decimal literals:
    // softmax is shift-invariant, so softmax(ln p) == p exactly for a normalized p.
    // That gives [0.6, 0.3, 0.1], entropy 0.89795 nats, surprisals 0.51083 /
    // 1.20397 / 2.30259, so |surprisal - entropy| is 0.38712 / 0.30602 / 1.40464 --
    // token 1, NOT the argmax, is the most typical. With typical_p = 0.25 the
    // first candidate alone (0.3) already exceeds it, so token 1 is the only
    // survivor.
    let logits = tensor(
        "logits",
        vec![1, 3],
        vec![0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()],
    );

    // Baseline: the argmax is token 0.
    assert_eq!(LlamaSampler::Greedy.sample(&logits).unwrap(), 0);

    for seed in [0u64, 1, 7, 42, 12345] {
        let token = LlamaSampler::Sampling(SamplingConfig {
            temperature: 1.0,
            typical_p: Some(0.25),
            seed: Some(seed),
            ..SamplingConfig::default()
        })
        .sample(&logits)
        .unwrap();
        assert_eq!(
            token, 1,
            "typical_p must keep the most typical token and drop the argmax (seed {seed})"
        );
    }
}

#[test]
fn min_keep_floors_the_truncating_filters() {
    // min_p = 1.0 keeps only the argmax; min_keep = 2 raises the floor to two
    // candidates, so token 1 becomes reachable again.
    let logits = tensor("logits", vec![1, 3], vec![3.0, 2.0, 1.0]);
    let draw = |min_keep| {
        let mut drawn = std::collections::BTreeSet::new();
        for seed in 0u64..64 {
            drawn.insert(
                LlamaSampler::Sampling(SamplingConfig {
                    temperature: 1.0,
                    min_p: Some(1.0),
                    min_keep,
                    seed: Some(seed),
                    ..SamplingConfig::default()
                })
                .sample(&logits)
                .unwrap(),
            );
        }
        drawn
    };

    assert_eq!(
        draw(None),
        std::collections::BTreeSet::from([0]),
        "min_p=1 keeps only the argmax"
    );
    let floored = draw(Some(2));
    assert!(
        floored.contains(&1),
        "min_keep=2 must keep a second candidate, drew {floored:?}"
    );
    assert!(
        !floored.contains(&2),
        "min_keep=2 must not keep a third, drew {floored:?}"
    );
}

#[test]
fn rejects_out_of_range_typical_p_top_n_sigma_and_min_keep() {
    let logits = tensor("logits", vec![1, 2], vec![0.0, 1.0]);
    let err = |config: SamplingConfig| {
        LlamaSampler::Sampling(config)
            .sample(&logits)
            .unwrap_err()
            .to_string()
    };

    assert!(err(SamplingConfig {
        typical_p: Some(1.5),
        ..SamplingConfig::default()
    })
    .contains("typical_p"));
    assert!(err(SamplingConfig {
        top_n_sigma: Some(-1.0),
        ..SamplingConfig::default()
    })
    .contains("top_n_sigma"));
    assert!(err(SamplingConfig {
        min_keep: Some(0),
        ..SamplingConfig::default()
    })
    .contains("min_keep"));
}

fn tiny_config() -> LlamaModelConfig {
    LlamaModelConfig {
        architecture: "llama".to_string(),
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
        no_rope_layer_step: None,
        attention_key_length: None,
        logit_scale: None,
        moe: None,
        gemma3: None,
        gemma4: None,
        qwen35: None,
        lfm2: None,
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
            moe_expert_bias: None,
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
            post_attention_norm: None,
            post_ffw_norm: None,
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
