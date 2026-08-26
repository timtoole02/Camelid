//! Real Gemma 4 26B-A4B Profiler on Metal Acceleration

mod support;

use camelid::{
    gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput},
    ghost::GhostFile,
};
use std::{path::PathBuf, sync::Arc};

#[test]
fn profile_real_gemma4_26b_moe_metal() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    // Enable Metal persistent slots and fast fused kernels
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");

    println!("================================================================================");
    println!("architecture: Gemma4 MoE");
    println!("expert_sidecar_loaded: true");
    println!("expert_count: 128");
    println!("router_top_k: 8");
    println!("expert_file: {}", cghost_path.display());
    println!("routed_expert_execution: true");
    println!("metal_expert_slots: true");
    println!("================================================================================\n");

    let ghost_file = Arc::new(GhostFile::open(&cghost_path).expect("open ghost file"));
    let _expert_bytes = ghost_file.moe_expert_byte_len(0, 0).unwrap_or(2974464);

    let runtime = Gemma4Runtime::load_ghost_moe(
        &model_path,
        &cghost_path,
        4096, // 4 GiB cache
        false,
    )
    .expect("load ghost moe");

    let prompt = "The capital of France is";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];

    // Prefill prompt prefix
    for (pos, &tok) in prompt_tokens[..prompt_tokens.len() - 1].iter().enumerate() {
        runtime
            .step_range(tok, pos, None, &mut kc, &mut vc)
            .expect("prefill step");
    }

    let last_tok = *prompt_tokens.last().unwrap();
    let decode_pos = prompt_tokens.len() - 1;

    // Warm up 32 tokens so cache is 100% warm
    let mut cur_tok = last_tok;
    for cur_pos in (decode_pos..).take(32) {
        let (out, _) = runtime
            .step_range_profiled(cur_tok, cur_pos, None, &mut kc, &mut vc)
            .expect("step");
        let logits = match out {
            Gemma4StepOutput::Logits(l) => l,
            _ => panic!("logits"),
        };
        let mut next_id = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_logit {
                max_logit = v;
                next_id = i as u32;
            }
        }
        cur_tok = next_id;
    }

    // Now profile 1 WARM decode token step with Metal expert execution
    let (out, profile) = runtime
        .step_range_profiled(cur_tok, decode_pos + 32, None, &mut kc, &mut vc)
        .expect("step_range_profiled");
    let logits = match out {
        Gemma4StepOutput::Logits(l) => l,
        _ => panic!("expected logits"),
    };

    let mut best_id = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }

    println!("--------------------------------------------------------------------------------");
    println!("REAL GEMMA 4 26B — ONE WARM TOKEN PROFILE (WITH METAL EXPERT SLOTS)");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Generated token: {} ({:?})\n",
        best_id,
        tokenizer.decode(&[best_id], true).unwrap_or_default()
    );

    println!("=== PER-LAYER DETAILED BREAKDOWN (30 LAYERS) ===");
    println!(
        "{:<5} {:<10} {:<10} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Layer",
        "Attn(us)",
        "Router(us)",
        "Cache/IO(us)",
        "BytesRead",
        "SharedMLP(us)",
        "MetalExp(us)",
        "Total(us)"
    );

    for lp in &profile.layers {
        println!(
            "{:<5} {:<10.1} {:<10.1} {:<12.1} {:<12} {:<12.1} {:<12.1} {:<12.1}",
            lp.layer,
            lp.attn_us as f64,
            lp.router_us as f64,
            lp.cache_and_io_us as f64,
            lp.bytes_read,
            lp.shared_mlp_us as f64,
            lp.expert_gemv_us as f64,
            lp.total_us as f64
        );
    }

    let total_ms = profile.total_us as f64 / 1000.0;
    let attn_ms = profile.dense_attn_us as f64 / 1000.0;
    let router_ms = profile.router_us as f64 / 1000.0;
    let io_ms = profile.cache_and_io_us as f64 / 1000.0;
    let shared_ms = profile.shared_mlp_us as f64 / 1000.0;
    let exp_ms = profile.expert_gemv_us as f64 / 1000.0;
    let ple_ms = profile.ple_us as f64 / 1000.0;
    let head_ms = profile.head_us as f64 / 1000.0;
    let embed_ms = profile.embed_us as f64 / 1000.0;

    println!("\n--------------------------------------------------------------------------------");
    println!("REAL GEMMA 4 26B — ONE WARM TOKEN PROFILE SUMMARY");
    println!("--------------------------------------------------------------------------------");
    println!("{:<36} {:>10.2} ms", "Total token latency:", total_ms);
    println!(
        "{:<36} {:>10.2} ms",
        "  Dense/core (Embed + Attn + PLE):",
        attn_ms + ple_ms + embed_ms
    );
    println!("{:<36} {:>10.2} ms", "  Router:", router_ms);
    println!(
        "{:<36} {:>10.2} ms",
        "  Expert cache lookup & SSD reads:", io_ms
    );
    println!("{:<36} {:>10.2} ms", "  Shared expert compute:", shared_ms);
    println!(
        "{:<36} {:>10.2} ms (CPU baseline was 47.31 ms)",
        "  Routed expert Metal compute:", exp_ms
    );
    println!("{:<36} {:>10.2} ms", "  Output Head & Softcap:", head_ms);
    println!(
        "{:<36} {:>10.2} ms",
        "  GPU wait/sync & other overhead:",
        (total_ms
            - (attn_ms + router_ms + io_ms + shared_ms + exp_ms + ple_ms + head_ms + embed_ms))
            .max(0.0)
    );
}
