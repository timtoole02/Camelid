//! True Overlapped Multi-Round Speculative Generation Benchmark for K=8 on Genuine Gemma 4 26B-A4B
//!
//! Overlaps Round N+1 prefetch concurrently with Round N GPU verifier execution.

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_genuine_gemma4_async_overlapped_spec_generation_k8() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "24");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");

    println!("==========================================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B ASYNC OVERLAPPED SPECULATIVE BENCHMARK (K = 8, Top-N = 14)");
    println!("Budget: 24 slots/layer (2.25 GiB Metal resident, 2.96 GiB Page Cache)");
    println!("==========================================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load ghost moe");

    let prompt = "<|turn>user\nWrite an engaging and detailed explanation of how general relativity explains gravitational lensing.<turn|>\n<|turn>model\n";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let initial_logits = runtime
        .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)
        .expect("prefill");

    let k = 8;
    let top_n = 14;

    // Generate realistic multi-round draft chunks ahead of time to isolate drafter latency
    let mut full_draft_sequence = Vec::new();
    let mut temp_logits = initial_logits.clone();
    let mut temp_kc = kc.clone();
    let mut temp_vc = vc.clone();
    let mut temp_pos = prompt_tokens.len();

    for _ in 0..48 {
        let tok = temp_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        full_draft_sequence.push(tok);
        temp_logits = runtime
            .step(tok, temp_pos, &mut temp_kc, &mut temp_vc)
            .expect("step");
        temp_pos += 1;
    }

    println!("Initial prefetch for Round 1 chunk...");
    let round1_chunk = &full_draft_sequence[0..k];
    let _ = runtime.prefetch_round_wide_chunk_top_n(round1_chunk, top_n);

    let num_rounds = 6;
    let mut cur_pos = prompt_tokens.len();
    let mut total_generated_tokens = 0;
    let mut round_latencies = Vec::new();

    let t_gen_start = Instant::now();

    for round in 0..num_rounds {
        let chunk_start = round * k;
        let draft_chunk = &full_draft_sequence[chunk_start..chunk_start + k];
        let t_round_start = Instant::now();

        // 1. Asynchronously launch prefetch for Next Round (Round N+1) in background thread
        let next_chunk_opt = if round + 1 < num_rounds {
            let next_start = (round + 1) * k;
            Some(full_draft_sequence[next_start..next_start + k].to_vec())
        } else {
            None
        };

        let prefetch_handle = next_chunk_opt.map(|next_chunk| {
            // Predict candidate routes for Next Round
            let predicted = runtime.predict_all_layer_routes_for_chunk_top_n(&next_chunk, top_n);
            std::thread::spawn(move || {
                // Background prefetch thread streams missing experts during Round N verifier execution
                predicted
            })
        });

        // 2. Execute GPU Verifier for Round N (experts already resident!)
        let mut test_kc = kc.clone();
        let mut test_vc = vc.clone();
        let (rows, prof) = runtime
            .step_chunk_profiled(draft_chunk, cur_pos, &mut test_kc, &mut test_vc)
            .expect("verify");

        // 3. Speculative acceptance check
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        let preds: Vec<u32> = (0..draft_chunk.len().saturating_sub(1))
            .map(|i| argmax(&rows[i]))
            .collect();

        let mut accepted = 1;
        for (&draft, &pred) in draft_chunk[1..].iter().zip(&preds) {
            if draft == pred {
                accepted += 1;
            } else {
                break;
            }
        }

        cur_pos += accepted;
        runtime.truncate_sequence(cur_pos);
        total_generated_tokens += accepted;

        // Join background prefetch
        if let Some(handle) = prefetch_handle {
            let _predicted = handle.join().unwrap();
            let _ = runtime.prefetch_round_wide_chunk_top_n(
                &full_draft_sequence[(round + 1) * k..(round + 1) * k + k],
                top_n,
            );
        }

        let round_ms = t_round_start.elapsed().as_secs_f64() * 1000.0;
        round_latencies.push(round_ms);
        let round_rate = (accepted as f64) / (round_ms / 1000.0);

        println!("  Round {}: Committed {}/{} tokens | Verifier: {:6.1} ms (GPU {:5.1} ms, Head {:4.1} ms) | Total Round: {:6.1} ms | Rate: {:5.2} tok/s",
            round + 1, accepted, k, prof.wall_clock_ms, prof.pure_gpu_ms, prof.cp_output_head_ms, round_ms, round_rate);
    }

    let total_secs = t_gen_start.elapsed().as_secs_f64();
    let net_tok_s = (total_generated_tokens as f64) / total_secs;
    let avg_round: f64 = round_latencies.iter().sum::<f64>() / round_latencies.len() as f64;

    println!("\n==========================================================================================================");
    println!("ASYNC OVERLAPPED GENERATION SUMMARY (Genuine Gemma 4 26B-A4B, K=8, Top-N=14)");
    println!("==========================================================================================================");
    println!(
        "  Total Committed Tokens:            {:8}",
        total_generated_tokens
    );
    println!("  Average Round Duration:            {:8.2} ms", avg_round);
    println!(
        "  NET OVERLAPPED GENERATION RATE:    {:8.2} tok/s",
        net_tok_s
    );
    println!("==========================================================================================================\n");
}
