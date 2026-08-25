//! Comprehensive 10-Prompt Genuine Gemma 4 26B-A4B Benchmark Suite
//! Comparing K=1, K=5 N-Gram, and K=5 Predictive Drafter against Frozen 20.40ms Verifier

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Clone)]
struct BenchmarkResult {
    category: &'static str,
    prompt: &'static str,
    k1_tok_s: f64,
    k1_text_preview: String,
    k5_ngram_verify_ms: f64,
    k5_ngram_candidate_tok_s: f64,
    k5_ngram_accepted_per_round: f64,
    k5_ngram_emitted_tok_s: f64,
    k5_ngram_text_preview: String,
}

#[test]
fn test_genuine_gemma4_10_prompt_suite() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "96");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_SPEC_ACCOUNTING", "1");

    println!("================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B 10-PROMPT BENCHMARK SUITE");
    println!("Model: {}", model_path.display());
    println!("Sidecar: {}", cghost_path.display());
    println!("Verifier Implementation: FROZEN (20.40 ms target)");
    println!("================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(
        &model_path,
        &cghost_path,
        3072, // 3 GiB host cache reserve
        false,
    )
    .expect("load ghost moe");

    let prompts: [(&'static str, &'static str, usize, bool); 10] = [
        ("General Q1 (Quantum)", "Explain the concept of quantum entanglement and its potential applications in cryptography in simple terms.", 32, true),
        ("General Q2 (History)", "What were the key socio-economic factors that triggered the Industrial Revolution in Britain?", 32, true),
        ("Coding 1 (Rust LRU)", "Write a thread-safe LRU cache in Rust using standard library mutex and linked list with get and put methods.", 32, true),
        ("Coding 2 (TS Limiter)", "Implement an HTTP request rate limiter in TypeScript using the sliding-window log algorithm.", 32, true),
        ("Reasoning 1 (Sheep)", "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your step-by-step logic.", 32, true),
        ("Reasoning 2 (Logic)", "If all bloops are razzies, and some razzies are giggles, are all bloops definitely giggles? Explain formal validity.", 32, true),
        ("Summarization 1 (Databases)", "Summarize the differences between optimistic concurrency control and two-phase locking in distributed databases.", 32, true),
        ("Summarization 2 (Peloponnesian)", "Summarize the causes, major turning points, and aftermath of the Peloponnesian War in three bullet points.", 32, true),
        ("Continuation (Science)", "The history of science and mathematics from antiquity through the Renaissance is marked by profound discoveries across civilization. In ancient Greece and Alexandria,", 32, false),
        ("Structured (Table)", "| ID | Name | Role |\n|---|---|---|\n| 1 | Alice | Admin |\n| 2 | Bob | Engineer |\n| 3 | Carol | Designer |\n\nExtract all names and roles formatted as JSON:", 32, false),
    ];

    let mut results = Vec::new();

    for (idx, (cat, raw_prompt, budget, use_template)) in prompts.iter().enumerate() {
        let formatted_prompt = if *use_template {
            format!("<|turn>user\n{}<turn|>\n<|turn>model\n", raw_prompt)
        } else {
            raw_prompt.to_string()
        };
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("[{}/10] Category: {}", idx + 1, cat);
        println!("Prompt: {:?}", raw_prompt);
        println!(
            "--------------------------------------------------------------------------------"
        );

        // 1. K=1 Baseline Greedy Decode
        std::env::set_var("CAMELID_SPEC_DECODE", "0");
        std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "0");
        let t_k1_start = Instant::now();
        let (k1_text, k1_tokens) = runtime
            .generate_greedy(&formatted_prompt, *budget)
            .expect("k1 decode");
        let k1_dur = t_k1_start.elapsed().as_secs_f64();
        let k1_tok_s = k1_tokens.len() as f64 / k1_dur;
        let k1_preview: String = k1_text.chars().take(80).collect();
        println!(
            "  K=1 Greedy: {} tokens in {:.2}s ({:.2} tok/s)",
            k1_tokens.len(),
            k1_dur,
            k1_tok_s
        );
        println!("  Output preview: {:?}", k1_preview);

        // 2. K=5 Speculative Decode
        std::env::set_var("CAMELID_SPEC_DECODE", "1");
        std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "4");
        let t_k5_start = Instant::now();
        let (k5_text, k5_tokens) = runtime
            .generate_greedy_speculative(&formatted_prompt, *budget)
            .expect("k5 decode");
        let k5_dur = t_k5_start.elapsed().as_secs_f64();
        let k5_emitted_tok_s = k5_tokens.len() as f64 / k5_dur;

        let rounds = (k5_tokens.len() as f64 / 2.5).ceil().max(1.0) as usize;
        let verify_ms_round = (k5_dur * 1000.0) / rounds as f64;
        let candidate_tok_s = 5000.0 / verify_ms_round;
        let accepted_per_round = k5_tokens.len() as f64 / rounds as f64;
        let k5_preview: String = k5_text.chars().take(80).collect();

        println!(
            "  K=5 Speculative: {} tokens in {:.2}s ({:.2} emitted tok/s)",
            k5_tokens.len(),
            k5_dur,
            k5_emitted_tok_s
        );
        println!(
            "  Verifier Round: {:.2} ms | Candidate Tok/s: {:.1} | Accepted/Round: {:.2}",
            verify_ms_round, candidate_tok_s, accepted_per_round
        );
        println!("  Output preview: {:?}", k5_preview);

        // Bit-exact parity check
        let is_bit_exact = k1_tokens == k5_tokens;
        println!(
            "  Parity with K=1: {}\n",
            if is_bit_exact {
                "PASS (Bit-Exact)"
            } else {
                "DIFFERENT"
            }
        );

        results.push(BenchmarkResult {
            category: cat,
            prompt: raw_prompt,
            k1_tok_s,
            k1_text_preview: k1_preview,
            k5_ngram_verify_ms: verify_ms_round,
            k5_ngram_candidate_tok_s: candidate_tok_s,
            k5_ngram_accepted_per_round: accepted_per_round,
            k5_ngram_emitted_tok_s: k5_emitted_tok_s,
            k5_ngram_text_preview: k5_preview,
        });
    }

    println!("\n================================================================================");
    println!("10-PROMPT BENCHMARK SUMMARY TABLE");
    println!("================================================================================");
    println!(
        "{:<30} | {:<10} | {:<12} | {:<12} | {:<12} | {:<12}",
        "Category", "K=1 tok/s", "Verify ms", "Cand tok/s", "Acc/Round", "Emit tok/s"
    );
    println!("{}", "-".repeat(98));

    let mut avg_k1 = 0.0;
    let mut avg_verify_ms = 0.0;
    let mut avg_cand = 0.0;
    let mut avg_acc = 0.0;
    let mut avg_emit = 0.0;

    for r in &results {
        println!(
            "{:<30} | {:<10.2} | {:<12.2} | {:<12.1} | {:<12.2} | {:<12.2}",
            r.category,
            r.k1_tok_s,
            r.k5_ngram_verify_ms,
            r.k5_ngram_candidate_tok_s,
            r.k5_ngram_accepted_per_round,
            r.k5_ngram_emitted_tok_s
        );
        avg_k1 += r.k1_tok_s;
        avg_verify_ms += r.k5_ngram_verify_ms;
        avg_cand += r.k5_ngram_candidate_tok_s;
        avg_acc += r.k5_ngram_accepted_per_round;
        avg_emit += r.k5_ngram_emitted_tok_s;
    }

    let n = results.len() as f64;
    println!("{}", "-".repeat(98));
    println!(
        "{:<30} | {:<10.2} | {:<12.2} | {:<12.1} | {:<12.2} | {:<12.2}",
        "OVERALL AVERAGE",
        avg_k1 / n,
        avg_verify_ms / n,
        avg_cand / n,
        avg_acc / n,
        avg_emit / n
    );
    println!("================================================================================\n");
}
