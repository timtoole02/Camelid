//! Full learned EAGLE generation against resident CUDA, including pooled requests.
#![cfg(all(feature = "cuda", not(target_os = "macos")))]
use camelid::{
    eagle3::Eagle3DraftModel,
    eagle3_serving::{Eagle3ServeHeadKey, Eagle3ServingConfig, Eagle3ServingState},
    gguf::read_metadata,
    inference::{LlamaInferenceSession, LlamaLoadedWeights},
    model::{LlamaModelConfig, LlamaTensorBinding},
    tensor::TensorStore,
    tokenizer::Tokenizer,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

fn step(session: &mut LlamaInferenceSession, token: u32) -> u32 {
    session
        .generate_next_token_greedy_resident(token)
        .unwrap()
        .expect("CUDA reference must stay resident")
        .0
}

#[test]
#[ignore = "requires the pinned Qwen3 target, pinned EAGLE head, and CUDA"]
fn learned_cuda_generation_matches_resident_and_reuses_head() {
    let path = PathBuf::from(
        std::env::var_os("CAMELID_QWEN3_4B_GGUF").expect("set CAMELID_QWEN3_4B_GGUF"),
    );
    let head_path =
        PathBuf::from(std::env::var_os("CAMELID_EAGLE3_MODEL").expect("set CAMELID_EAGLE3_MODEL"));
    let gguf = read_metadata(&path).unwrap();
    let mut config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    config.context_length = 512;
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let store = TensorStore::open(&path, &gguf);
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).unwrap());
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let checkpoint = Arc::new(Eagle3DraftModel::load(&head_path).unwrap());
    let key = Eagle3ServeHeadKey {
        checkpoint_path: head_path,
        checkpoint_sha256: camelid::eagle3::QWEN_WEIGHTS_SHA256.into(),
        target_sha256: camelid::eagle3::QWEN_TARGET_SHA256.into(),
        draft_wire: "cuda-q8_128-f32-linear1-v2".into(),
        max_positions: 512,
    };
    let make_session = || {
        let mut s = LlamaInferenceSession::new(config.clone(), weights.clone()).unwrap();
        s.set_resident_encode_ahead_enabled(false);
        s
    };
    let cases = [
        ("The capital of France is", 40),
        ("Write a Python function that adds two numbers:\n", 48),
        ("Count from one to ten: one, two, three,", 32),
        ("Explain why water freezes when", 32),
    ];
    let mut accepted_total = 0;
    let mut rejected_total = 0;
    let mut declined_total = 0;
    let repeats = std::env::var("CAMELID_EAGLE3_BENCH_REPEATS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("integer benchmark repeat count")
        })
        .unwrap_or(1)
        .clamp(1, 10);
    let mut plain_totals = [0u128; 4];
    let mut eagle_totals = [0u128; 4];
    let prefix = std::env::var("CAMELID_EAGLE3_BENCH_PREFIX").unwrap_or_default();
    for (index, (text, count)) in cases.iter().cycle().take(cases.len() * repeats).enumerate() {
        let prompt = tokenizer
            .encode(&format!("{prefix}{text}"), true, false)
            .unwrap();
        assert!(
            prompt.len() + count <= 512,
            "benchmark prompt exceeds its cache"
        );
        let mut reference = make_session();
        let mut first = 0;
        for &token in &prompt {
            first = step(&mut reference, token);
        }
        let plain_started = Instant::now();
        let mut expected = vec![first];
        while expected.len() < *count {
            expected.push(step(&mut reference, *expected.last().unwrap()));
        }
        let plain_ms = plain_started.elapsed().as_millis();
        drop(reference);
        let mut target = make_session();
        let mut state =
            Eagle3ServingState::new(checkpoint.clone(), Eagle3ServingConfig::new(7).unwrap())
                .with_pooled_head(key.clone());
        assert_eq!(state.dynamic_tree().verify_nodes, 2);
        let bootstrap = state
            .bootstrap(&mut target, &weights, &prompt, *count)
            .unwrap();
        assert_eq!(state.head_reused(), index > 0);
        let mut actual = vec![bootstrap.first_token];
        let mut history = prompt.clone();
        history.push(bootstrap.first_token);
        let started = Instant::now();
        let mut accepted = 0;
        let mut rejected = 0;
        let mut declined = 0;
        while actual.len() < *count {
            let round = state
                .run_round(&mut target, &weights, &history, *count - actual.len())
                .unwrap()
                .expect("context available");
            assert!(!round.suffix, "must execute learned speculation");
            if round.offered == 1 {
                if round.emitted.len() == 2 {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            } else if *count - actual.len() > 1 {
                declined += 1;
            }
            history.extend_from_slice(&round.emitted);
            actual.extend(round.emitted);
        }
        assert_eq!(
            actual, expected,
            "full learned generation diverged for {text:?}"
        );
        assert!(state.phase_timings().head_update_us > 0);
        accepted_total += accepted;
        rejected_total += rejected;
        declined_total += declined;
        let eagle_ms = started.elapsed().as_millis();
        let case = index % cases.len();
        plain_totals[case] += plain_ms;
        eagle_totals[case] += eagle_ms;
        eprintln!("CUDA EAGLE trial={} case={case} prompt_tokens={} tokens={count} accepted={accepted} rejected={rejected} declined={declined} head_reused={} plain_decode_ms={plain_ms} eagle_decode_ms={eagle_ms} verify_ms={} head_ms={} parity=true",index / cases.len(), prompt.len(), state.head_reused(), state.phase_timings().verify_us / 1000, state.phase_timings().head_update_us / 1000);
        drop(state);
    }
    assert!(
        accepted_total > 0,
        "learned head never supplied an accepted draft"
    );
    assert!(
        rejected_total > 0,
        "fixture did not exercise rejected-head updates"
    );
    assert!(
        declined_total > 0,
        "fixture must exercise confidence-declined drafts"
    );
    let plain_total: u128 = plain_totals.iter().sum();
    let eagle_total: u128 = eagle_totals.iter().sum();
    eprintln!("CUDA EAGLE aggregate repeats={repeats} plain_ms={plain_total} eagle_ms={eagle_total} speedup={:.3}x", plain_total as f64 / eagle_total as f64);
    // Opt-in hardware performance gate: ordinary CI must not depend on GPU
    // clock state or an unqualified device, while a speed receipt must pass it.
    if std::env::var("CAMELID_EAGLE3_REQUIRE_SPEEDUP").as_deref() == Ok("1") {
        assert!(
            repeats >= 3,
            "performance qualification requires at least three trials"
        );
        for case in 0..cases.len() {
            assert!(
                eagle_totals[case] < plain_totals[case],
                "case {case} regressed: plain={}ms EAGLE={}ms",
                plain_totals[case],
                eagle_totals[case]
            );
        }
        assert!(
            plain_total * 100 > eagle_total * 105,
            "aggregate speedup must exceed 5%"
        );
    }
}
