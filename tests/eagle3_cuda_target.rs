//! Bounded live Qwen3-4B CUDA target test; this is not an EAGLE head parity test.
//! Set CAMELID_QWEN3_4B_GGUF and explicitly run with --ignored --nocapture.
#![cfg(feature = "cuda")]

use camelid::{
    gguf::read_metadata,
    inference::{LlamaInferenceSession, LlamaLoadedWeights},
    model::{LlamaModelConfig, LlamaTensorBinding},
    tensor::TensorStore,
    tokenizer::Tokenizer,
};
use std::{path::PathBuf, sync::Arc};

fn resident_step(session: &mut LlamaInferenceSession, token: u32) -> u32 {
    session
        .generate_next_token_greedy_resident(token)
        .unwrap()
        .expect("every reference and continuation step must use the resident GPU")
        .0
}

fn seed(session: &mut LlamaInferenceSession, prompt: &[u32]) -> u32 {
    let (&first, rest) = prompt.split_first().expect("nonempty prompt");
    let mut prediction = resident_step(session, first);
    for &token in rest {
        prediction = resident_step(session, token);
    }
    prediction
}

#[test]
#[ignore = "requires a Qwen3-4B GGUF and a CUDA device; fails if either is unavailable"]
fn qwen3_cuda_target_capture_acceptance_and_rollback() {
    let path = PathBuf::from(
        std::env::var_os("CAMELID_QWEN3_4B_GGUF").expect("set CAMELID_QWEN3_4B_GGUF"),
    );
    let gguf = read_metadata(&path).unwrap();
    let mut config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    assert_eq!(config.architecture, "qwen3");
    assert_eq!(config.embedding_length, 2560);
    assert_eq!(config.block_count, 36);
    // The resident allocator deliberately declines capacities below 256.
    config.context_length = 512;
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let store = TensorStore::open(&path, &gguf);
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).unwrap());
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let prompt = tokenizer
        .encode("The capital of France is", true, false)
        .unwrap();
    let make_session = || {
        let mut session = LlamaInferenceSession::new(config.clone(), Arc::clone(&weights)).unwrap();
        session.set_resident_encode_ahead_enabled(false);
        session
    };
    let mut plain = make_session();
    let mut expected = vec![seed(&mut plain, &prompt)];
    for _ in 1..10 {
        let next = resident_step(&mut plain, *expected.last().unwrap());
        expected.push(next);
    }
    drop(plain);
    let mut session = make_session();
    assert_eq!(seed(&mut session, &prompt), expected[0]);
    // Ensure the resident decode engine is materialized before the verify seam.
    assert_eq!(resident_step(&mut session, expected[0]), expected[1]);
    let position = session.kv_position();
    // The pinned Q4_K_M target's FFN projection limits this CUDA stack to
    // two rows. Preserve the real four-row refusal as a regression, then
    // exercise full acceptance at the supported width (one draft + bonus).
    let too_wide = session
        .verify_drafts_cuda_with_layer_inputs(expected[1], &expected[2..5], &[2, 18, 33])
        .unwrap_err();
    assert!(too_wide.to_string().contains("shared-memory cap of 2"));
    assert_eq!(session.kv_position(), position);
    let capture = session
        .verify_drafts_cuda_with_layer_inputs(expected[1], &expected[2..3], &[33, 2, 18])
        .unwrap()
        .expect("CUDA capture must execute; CPU fallback is not evidence");
    assert_eq!(capture.predictions, expected[2..4]);
    assert_eq!(session.kv_position(), position + 2);
    for tap in &capture.layer_inputs {
        assert_eq!(tap.shape.dims, [2, 2560]);
        assert!(tap.data.iter().all(|x| x.is_finite()));
        assert!(tap.data.iter().any(|x| *x != 0.0));
    }
    let position = session.kv_position();
    let wrong = (expected[4] + 1) % 151936;
    let rejected = session
        .verify_drafts_cuda_with_layer_inputs(expected[3], &[wrong], &[2, 18, 33])
        .unwrap()
        .expect("CUDA mismatch verification must execute");
    assert_eq!(rejected.predictions[0], expected[4]);
    assert_eq!(
        session.kv_position(),
        position + 1,
        "rejected rows must not be committed"
    );
    let mut last = expected[4];
    for &token in &expected[5..] {
        last = resident_step(&mut session, last);
        assert_eq!(
            last, token,
            "continuation after rejected speculative KV differs"
        );
    }
    eprintln!("CUDA target: greedy tokens {expected:?}; taps [33,2,18], 2x2560 each; four-row batch refused without advancing KV; accepted 1 draft plus bonus; mismatch committed only bonus; continuation matched. No learned head executed.");
}
