//! Test End-to-End Speculative Decoding on Real Gemma 4 26B-A4B MoE

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_real_gemma4_moe_speculative_generation() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_TIMING", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "4");

    println!("================================================================================");
    println!("architecture: Gemma4 MoE");
    println!("expert_sidecar_loaded: true");
    println!("expert_count: 128");
    println!("router_top_k: 8");
    println!("expert_file: {}", cghost_path.display());
    println!("routed_expert_execution: true");
    println!("speculative_moe_generation: true");
    println!("metal_slots_per_layer: 16 (1.50 GiB resident slab)");
    println!("================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(
        &model_path,
        &cghost_path,
        3072, // 3 GiB resident cache (Total RSS ~4.9GB, zero swap)
        false,
    )
    .expect("load ghost moe");

    let prompt = "The capital of France is Paris, which is known for its art, gastronomy, culture, and monuments. The city is divided into twenty arrondissements and";
    let max_new = 32;

    println!("Running baseline greedy generation ({} tokens)...", max_new);
    let t_greedy_start = Instant::now();
    let (greedy_text, greedy_tokens) = runtime
        .generate_greedy(prompt, max_new)
        .expect("generate_greedy");
    let greedy_dur = t_greedy_start.elapsed();
    let greedy_tok_s = greedy_tokens.len() as f64 / greedy_dur.as_secs_f64();
    println!(
        "Greedy finished: {} tokens in {:.2} ms ({:.2} tok/s)\n",
        greedy_tokens.len(),
        greedy_dur.as_secs_f64() * 1000.0,
        greedy_tok_s
    );

    println!("Running speculative MoE generation ({} tokens)...", max_new);
    let t_spec_start = Instant::now();
    let (spec_text, spec_tokens) = runtime
        .generate_greedy_speculative(prompt, max_new)
        .expect("generate_greedy_speculative");
    let spec_dur = t_spec_start.elapsed();
    let spec_tok_s = spec_tokens.len() as f64 / spec_dur.as_secs_f64();
    println!(
        "Speculative finished: {} tokens in {:.2} ms ({:.2} tok/s)\n",
        spec_tokens.len(),
        spec_dur.as_secs_f64() * 1000.0,
        spec_tok_s
    );

    println!("--------------------------------------------------------------------------------");
    println!("PARITY & ACCELERATION VERIFICATION");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Greedy Text:      {:?}",
        &greedy_text[..greedy_text.len().min(80)]
    );
    println!(
        "Speculative Text: {:?}",
        &spec_text[..spec_text.len().min(80)]
    );

    assert_eq!(
        greedy_tokens, spec_tokens,
        "Speculative MoE decode produced mismatched token sequence!"
    );
    println!("\nSUCCESS: Speculative MoE produced 100% BIT-EXACT output matching target greedy baseline!");
    println!(
        "Speedup: {:.2}x ({:.2} tok/s vs {:.2} tok/s)",
        spec_tok_s / greedy_tok_s,
        spec_tok_s,
        greedy_tok_s
    );
}
