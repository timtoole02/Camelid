//! Test Metal Routed Expert Acceleration for Real Gemma 4 26B-A4B

use camelid::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};
use std::{path::PathBuf, time::Instant};

#[test]
fn test_metal_expert_moe() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    // Force Metal slots on + fast fused kernel + common core
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_COMMON", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");

    println!("================================================================================");
    println!("Testing Gemma 4 26B-A4B with Metal Fused Fast & Common Metal Core...");
    println!("================================================================================");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load ghost moe");

    let prompt = "The capital of France is";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];

    // Prefill
    for (pos, &tok) in prompt_tokens[..prompt_tokens.len() - 1].iter().enumerate() {
        runtime
            .step_range(tok, pos, None, &mut kc, &mut vc)
            .expect("prefill step");
    }

    let last_tok = *prompt_tokens.last().unwrap();
    let decode_pos = prompt_tokens.len() - 1;

    // Measure decode steps
    let num_tokens = 32;
    let mut cur_tok = last_tok;
    let mut cur_pos = decode_pos;
    let mut generated = Vec::new();

    let t_start = Instant::now();
    for _ in 0..num_tokens {
        let (out, _prof) = runtime
            .step_range_profiled(cur_tok, cur_pos, None, &mut kc, &mut vc)
            .expect("step");
        let logits = match out {
            Gemma4StepOutput::Logits(l) => l,
            _ => panic!("expected logits"),
        };

        let mut next_id = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_logit {
                max_logit = v;
                next_id = i as u32;
            }
        }

        generated.push(next_id);
        cur_tok = next_id;
        cur_pos += 1;
    }
    let dur = t_start.elapsed();

    let tok_s = num_tokens as f64 / dur.as_secs_f64();
    println!(
        "Generated {} tokens in {:.2} ms ({:.2} tok/s)",
        num_tokens,
        dur.as_secs_f64() * 1000.0,
        tok_s
    );
    let decoded = tokenizer.decode(&generated, true).unwrap_or_default();
    println!("Generated text: {:?}", decoded);
}
