//! Gemma 4 26B Advancing Speculative Generation Benchmark (Milestone 2 & 3)

use camelid::gemma4_runtime::{gemma4_stop_token_ids, Gemma4Runtime};
use std::{path::PathBuf, time::Instant};

#[test]
fn test_gemma4_26b_speculative_advancing() {
    let model_path = std::env::var_os("CAMELID_GEMMA4_26B_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if PathBuf::from("/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf").exists() {
                PathBuf::from("/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf")
            } else {
                PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf")
            }
        });
    let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if PathBuf::from("/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost").exists() {
                PathBuf::from("/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost")
            } else {
                PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost")
            }
        });

    std::env::set_var("CAMELID_GHOST_ALLOW_LEGACY_SPARSE", "0");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_K1_LANE", "chained");
    if std::env::var("CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS").is_err() {
        std::env::set_var(
            "CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS",
            "104,112,92,88,92,80,80,80,80,80,76,76,80,84,80,84,84,88,88,84,84,84,84,88,84,92,96,100,104,112",
        );
    }
    std::env::set_var("CAMELID_GEMMA4_SPEC_TIMING", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_MULTIHOP", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_MIN_MATCH", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_MAX_MATCH", "6");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 2900, false)
        .expect("load ghost moe");

    let test_prompts: [(&str, &str, usize); 2] = [
        (
            "Code Infilling & Refactoring (High Prompt Overlap)",
            "<|turn>user\nRewrite this struct with an added expires_at field:\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n}\n<turn|>\n<|turn>model\n",
            48usize,
        ),
        (
            "JSON Extraction & Schema Transformation (High Prompt Overlap)",
            "<|turn>user\nConvert this configuration payload to YAML:\n{\"cluster_id\": \"prod-1\", \"min_replicas\": 4, \"max_replicas\": 32, \"enabled\": true}\n<turn|>\n<|turn>model\n",
            48usize,
        ),
    ];

    for (workload_name, prompt, max_new) in test_prompts {
        println!("\n==========================================================================================================");
        println!("WORKLOAD: {}", workload_name);
        println!("==========================================================================================================");

        let prompt_tokens = runtime
            .tokenizer()
            .encode(prompt, true, true)
            .expect("tokenize");
        let eot = gemma4_stop_token_ids(runtime.tokenizer());

        // 1. Reference Greedy Baseline (Pure Decode Measurement)
        let mut greedy_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
        let mut greedy_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
        let t_prefill_start = Instant::now();
        let init_logits = runtime
            .prefill_tokens(
                &prompt_tokens,
                &mut greedy_kc,
                &mut greedy_vc,
                max_new.saturating_sub(1),
            )
            .expect("prefill");
        let prefill_dur = t_prefill_start.elapsed();
        println!(
            "Prefill ({} tokens): {:.2} ms ({:.2} tok/s)",
            prompt_tokens.len(),
            prefill_dur.as_secs_f64() * 1000.0,
            prompt_tokens.len() as f64 / prefill_dur.as_secs_f64()
        );

        println!("\nGenerating reference greedy tokens (K=1 decode)...");
        let mut greedy_tokens = Vec::with_capacity(max_new);
        let mut cur_logits = init_logits;
        let mut cur_pos = prompt_tokens.len();
        let t_decode_greedy_start = Instant::now();
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
        let greedy_decode_dur_s = t_decode_greedy_start.elapsed().as_secs_f64();
        let greedy_tok_s = greedy_tokens.len() as f64 / greedy_decode_dur_s;
        println!(
            "Greedy reference: {} tokens in {:.2}s pure decode ({:.2} tok/s)",
            greedy_tokens.len(),
            greedy_decode_dur_s,
            greedy_tok_s
        );

        std::env::set_var("CAMELID_GEMMA4_SPEC_CHUNK_MAX", "16");
        // 2. Speculative Decode Across Draft Widths (up to K=16 widened batch)
        for draft_k in [2, 4, 8, 12, 16] {
            std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", draft_k.to_string());
            println!("\n----------------------------------------------------------------------------------------------------------");
            println!("ADVANCING SPECULATIVE DECODE (Draft K = {})", draft_k);
            println!("----------------------------------------------------------------------------------------------------------");

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

            let t_decode_spec_start = Instant::now();
            let spec_tokens = runtime
                .spec_decode_generate(
                    &mut spec_kc,
                    &mut spec_vc,
                    spec_init_logits,
                    &prompt_tokens,
                    &eot,
                    max_new,
                )
                .expect("spec_decode_generate");
            let spec_decode_dur_s = t_decode_spec_start.elapsed().as_secs_f64();
            let spec_tok_s = spec_tokens.len() as f64 / spec_decode_dur_s;

            println!(
                "Speculative: {} tokens in {:.2}s pure decode ({:.2} tok/s)",
                spec_tokens.len(),
                spec_decode_dur_s,
                spec_tok_s
            );
            assert_eq!(
                greedy_tokens, spec_tokens,
                "Speculative decode token mismatch for draft K={draft_k}!"
            );
            println!("✓ 100% BIT-EXACT MATCH to greedy target!");
            println!(
                "Speedup vs greedy pure decode: {:.2}x ({:.2} tok/s vs {:.2} tok/s)",
                spec_tok_s / greedy_tok_s,
                spec_tok_s,
                greedy_tok_s
            );
            println!("----------------------------------------------------------------------------------------------------------");
        }
    }
}
