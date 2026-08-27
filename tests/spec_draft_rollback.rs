//! Regression: rejected-draft rollback on the GPU-resident draft-model path.
//!
//! `ModelDrafter::draft` rolls its session back to the accepted prefix before each
//! round (`rollback_resident_to_position`). That lowers `kv_cache.position`, so the
//! resident engine's own `filled` must move with it — otherwise the next resident
//! decode reads `filled() != position` as a new sequence and reseeds the GPU cache
//! from a CPU KV history a GPU-resident drafter never wrote (`keys` is grown only by
//! `ensure_position_capacity`), indexing an empty buffer:
//!
//!   thread 'main' panicked at src/inference/metal_resident.rs:
//!   range end index 64 out of range for slice of length 0
//!
//! On macOS that reset existed only under `#[cfg(feature = "cuda")]`, so every
//! `--drafter draft` run without `--cpu-draft` panicked at the first partially
//! rejected round. This test drives exactly that sequence: draft a round, then draft
//! again from a history that diverges inside the drafted tail.
//!
//! Skips cleanly when the env var is unset (CI carries no model files); run locally:
//!   CAMELID_DRAFT_ROLLBACK_GGUF=/path/Llama-3.2-1B-Instruct-Q8_0.gguf \
//!     cargo test --release --test spec_draft_rollback -- --nocapture

use camelid::gguf::read_metadata;
use camelid::inference::speculative::ModelDrafter;
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;
use camelid::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn draft_model_rollback_after_rejection_keeps_the_resident_engine_in_sync() {
    let Some(model) = std::env::var_os("CAMELID_DRAFT_ROLLBACK_GGUF").map(PathBuf::from) else {
        // Skipping is correct for contributors and for CI legs that carry no model files, but a
        // skip that reports "ok" is indistinguishable from coverage. A job that is CONFIGURED to
        // prove this fix opts in with CAMELID_REQUIRE_MODEL_TESTS=1 and then cannot silently
        // degrade to a no-op.
        assert!(
            std::env::var("CAMELID_REQUIRE_MODEL_TESTS").as_deref() != Ok("1"),
            "CAMELID_REQUIRE_MODEL_TESTS=1 but CAMELID_DRAFT_ROLLBACK_GGUF is unset — this job \
             is configured to prove the draft-rollback fix and cannot"
        );
        eprintln!("SKIP draft rollback regression: set CAMELID_DRAFT_ROLLBACK_GGUF");
        return;
    };
    // The defect lives on the GPU-resident lane; the CLI turns this on via the execution
    // planner, so a bare test process must ask for it explicitly.
    std::env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");

    let gguf = read_metadata(&model).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("model config");
    let binding = LlamaTensorBinding::bind(&gguf, &config).expect("tensor binding");
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).expect("load weights"));
    let session = LlamaInferenceSession::new(config, weights).expect("session");
    let mut drafter = ModelDrafter::new(session);

    let prompt: Vec<u32> = tokenizer
        .encode(
            "The capital of France is Paris, and the capital of Italy is",
            true,
            false,
        )
        .expect("encode prompt");
    assert!(
        prompt.len() > 4,
        "prompt must give the drafter real history"
    );

    let gamma = 5;
    let first = drafter.draft(&prompt, gamma).expect("first draft round");
    assert_eq!(first.len(), gamma, "drafter should fill the window");
    let (_, resident_steps, cpu_steps, _) = drafter.take_forward_stats();
    let resident_lane = resident_steps > 0;
    eprintln!("round 1: drafts={first:?} resident_steps={resident_steps} cpu_steps={cpu_steps}");
    // A model WAS supplied, so the caller asked for the resident lane. Falling back to CPU is
    // not a softer pass — the panic this file guards lives in the resident reseed, so a CPU-only
    // run proves nothing and every assertion below also holds with the fix reverted. Opt out
    // explicitly (CAMELID_ALLOW_CPU_LANE=1) if you are deliberately smoke-testing the CPU path.
    if !resident_lane && std::env::var("CAMELID_ALLOW_CPU_LANE").as_deref() != Ok("1") {
        panic!(
            "a model was supplied but the resident lane never engaged (resident_steps=0, \
             cpu_steps={cpu_steps}) — this run cannot reproduce the pre-fix panic, which lives \
             in the resident reseed. Set CAMELID_ALLOW_CPU_LANE=1 to accept a CPU-only run."
        );
    }
    // LOAD-BEARING PRECONDITION, not a nicety. The pre-fix failure mode is the panic in the
    // round-2 reseed, and that panic needs `kv_cache.keys` to still be EMPTY — i.e. round 1
    // must have stayed entirely on the resident lane. A single CPU-fallback step here calls
    // `ensure_position_capacity`, which grows the buffers; the round-2 reseed then finds them
    // addressable, does NOT panic, and silently seeds zeros instead. Since this test asserts
    // nothing about draft CONTENT, that outcome would pass on a build with the fix reverted —
    // the regression test would quietly stop being one. Assert the precondition so the test
    // fails loudly rather than passing for the wrong reason.
    if resident_lane {
        assert_eq!(
            cpu_steps, 0,
            "round 1 fell back to the CPU ({cpu_steps} steps), which grows kv_cache.keys and \
             masks the pre-fix panic — this run cannot prove the rollback fix"
        );
    }

    // The target accepts the first draft and then emits something else — the classic
    // partial rejection. `history` therefore keeps drafts[0] and diverges after it, so the
    // next round must roll the draft KV back past the speculative tail it fed last round.
    let rejected = (first[1] + 1) % tokenizer.tokens.len().max(2) as u32;
    assert_ne!(rejected, first[1], "the second history token must diverge");
    let mut history = prompt.clone();
    history.push(first[0]);
    history.push(rejected);

    // Pre-fix this call panicked inside the resident reseed.
    let second = drafter.draft(&history, gamma).expect("second draft round");
    assert_eq!(second.len(), gamma, "drafter should fill the window again");
    let (_, resident_steps_2, cpu_steps_2, _) = drafter.take_forward_stats();
    eprintln!(
        "round 2: drafts={second:?} resident_steps={resident_steps_2} cpu_steps={cpu_steps_2}"
    );

    // A round that quietly fell back to the CPU forward would also "not panic", so assert
    // the rolled-back session stayed on the resident lane it was on before the rollback.
    if resident_lane {
        assert_eq!(
            cpu_steps_2, 0,
            "rollback pushed the draft session off the resident lane \
             ({resident_steps_2} resident vs {cpu_steps_2} CPU steps)"
        );
    }

    // A third round where the target accepted the whole fed tail must keep working too:
    // the rollback target equals the current position, so it is a no-op that must leave the
    // engine (and its encode-ahead graph) alone. History grows by the accepted drafts plus
    // the target's bonus token, which the draft session has not consumed yet.
    let mut history = history.clone();
    history.extend_from_slice(&second[..gamma - 1]);
    history.push(rejected);
    let third = drafter.draft(&history, gamma).expect("third draft round");
    assert_eq!(third.len(), gamma);
    let (_, resident_steps_3, cpu_steps_3, _) = drafter.take_forward_stats();
    eprintln!(
        "round 3: drafts={third:?} resident_steps={resident_steps_3} cpu_steps={cpu_steps_3}"
    );
    if resident_lane {
        assert_eq!(
            cpu_steps_3, 0,
            "a no-op rollback must not disturb the engine"
        );
    }
}
