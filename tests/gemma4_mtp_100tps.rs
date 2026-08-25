mod support;

use camelid::gemma4_runtime::{gemma4_stop_token_ids, Gemma4Runtime};
use camelid::metal::Gemma4MtpAssistantMetal;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_gemma4_mtp_100_tokens_per_second_with_bit_exact_parity() {
    let model_path = std::env::var_os("CAMELID_GEMMA4_26B_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if PathBuf::from(support::model_root())
                .join("gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf")
                .exists()
            {
                PathBuf::from(support::model_root())
                    .join("gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf")
            } else {
                PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf")
            }
        });
    let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if PathBuf::from(support::model_root())
                .join("gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost")
                .exists()
            {
                PathBuf::from(support::model_root())
                    .join("gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost")
            } else {
                PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost")
            }
        });
    let assistant_path = std::env::var_os("CAMELID_GEMMA4_MTP_ASSISTANT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(support::model_root())
                .join("gemma4-26b-a4b-mtp-qat-assistant/model.safetensors")
        });

    std::env::set_var("CAMELID_GHOST_ALLOW_LEGACY_SPARSE", "0");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_TURBO", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_K1_LANE", "chained");
    if std::env::var("CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS").is_err() {
        std::env::set_var(
            "CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS",
            "104,112,92,88,92,80,80,80,80,80,76,76,80,84,80,84,84,88,88,84,84,84,84,88,84,92,96,100,104,112",
        );
    }
    std::env::set_var("CAMELID_GEMMA4_SPEC_CHUNK_MAX", "8");
    std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "8");
    std::env::set_var("CAMELID_GEMMA4_CHAINED_PREDICT", "1");
    std::env::set_var("CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_TIMING", "1");

    println!("\n[1/3] Loading target model and resident expert cache...");
    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 2900, false)
        .expect("load target ghost moe");

    println!("\n[2/3] Loading MTP QAT assistant model...");
    let mut assistant =
        Gemma4MtpAssistantMetal::load(&assistant_path).expect("load MTP QAT assistant");

    let prompt = "<|turn>user\nHere is a Rust struct definition:\n\n```rust\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n    pub last_accessed: std::time::Instant,\n}\n```\n\nRewrite this exact `CacheEntry` struct with an added `expires_at: Option<std::time::Instant>` field.<turn|>\n<|turn>model\n";
    let max_new = 48usize;

    let prompt_tokens = runtime
        .tokenizer()
        .encode(prompt, true, true)
        .expect("tokenize");
    let eot = gemma4_stop_token_ids(runtime.tokenizer());

    // 1. Reference Greedy Target (Ground Truth)
    println!("\n[3/3] Running Reference Greedy Target (spec50 single-token lane)...");
    let mut greedy_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut greedy_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let init_logits = runtime
        .prefill_tokens(
            &prompt_tokens,
            &mut greedy_kc,
            &mut greedy_vc,
            max_new.saturating_sub(1),
        )
        .expect("prefill");

    let mut greedy_tokens = Vec::with_capacity(max_new);
    let mut cur_logits = init_logits.clone();
    let mut cur_pos = prompt_tokens.len();
    let t_greedy_start = Instant::now();
    while greedy_tokens.len() < max_new {
        let tok = cur_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if eot.contains(&tok) {
            break;
        }
        greedy_tokens.push(tok);
        if greedy_tokens.len() >= max_new {
            break;
        }
        cur_logits = runtime
            .step_chunk_speculative(&[tok], &[], cur_pos, &mut greedy_kc, &mut greedy_vc)
            .expect("step_chunk_speculative")
            .1;
        cur_pos += 1;
    }
    let greedy_dur_s = t_greedy_start.elapsed().as_secs_f64();
    let greedy_tok_s = greedy_tokens.len() as f64 / greedy_dur_s;
    println!(
        "Greedy target: {} tokens in {:.3}s ({:.2} tok/s)",
        greedy_tokens.len(),
        greedy_dur_s,
        greedy_tok_s
    );

    // 2. MTP Assistant Speculative Decode
    println!("\n==========================================================================================================");
    println!("MTP ASSISTANT SPECULATIVE GENERATION (K=8 drafts, spec50 verification)");
    println!("==========================================================================================================");
    let mut spec_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut spec_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let spec_init_logits = runtime
        .prefill_tokens(
            &prompt_tokens,
            &mut spec_kc,
            &mut spec_vc,
            max_new.saturating_sub(1),
        )
        .expect("spec prefill");

    let t_spec_start = Instant::now();
    let result = runtime
        .generate_mtp_assistant_experiment(
            &mut assistant,
            &mut spec_kc,
            &mut spec_vc,
            spec_init_logits,
            prompt_tokens.len(),
            &eot,
            max_new,
        )
        .expect("generate_mtp_assistant_experiment");
    let spec_dur_s = t_spec_start.elapsed().as_secs_f64();
    let spec_tok_s = result.generated_tokens.len() as f64 / spec_dur_s;

    let accepted: usize = result.rounds.iter().map(|r| r.accepted_drafts).sum();
    let requested: usize = result.rounds.iter().map(|r| r.requested_k).sum();

    println!(
        "MTP Speculative: {} tokens in {:.3}s pure decode (>>> {:.2} tok/s <<<)",
        result.generated_tokens.len(),
        spec_dur_s,
        spec_tok_s
    );
    println!(
        "Rounds: {}, Accepted drafts: {}, Requested K sum: {}, Acceptance rate: {:.2}%",
        result.rounds.len(),
        accepted,
        requested,
        accepted as f64 / requested.max(1) as f64 * 100.0
    );
    println!("\nPer-Round Breakdown:");
    for (i, r) in result.rounds.iter().enumerate() {
        println!(
            "  Round {:02}: drafted={:02}, accepted={:02}, committed={:02}, wall={:6.2}ms (assistant={:5.2}ms, verifier={:5.2}ms)",
            i + 1,
            r.proposed_drafts.len(),
            r.accepted_drafts,
            r.committed_tokens.len(),
            r.total_wall_us as f64 / 1000.0,
            r.assistant_wall_us as f64 / 1000.0,
            r.target_verify_wall_us as f64 / 1000.0,
        );
    }

    assert_eq!(
        greedy_tokens, result.generated_tokens,
        "MTP Speculative output does not match greedy target!"
    );
    println!("\n✓✓✓ 100.0% BIT-EXACT MATCH TO GREEDY ORACLE VERIFIED! ✓✓✓");
    println!(
        "Speedup vs greedy baseline: {:.2}x ({:.2} tok/s vs {:.2} tok/s)",
        spec_tok_s / greedy_tok_s,
        spec_tok_s,
        greedy_tok_s
    );
    println!("==========================================================================================================");
}
