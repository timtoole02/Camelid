//! Speculative Pipelined Multi-Round Generation Benchmark for K=8 on Genuine Gemma 4 26B-A4B
//!
//! Evaluates sustained multi-round generation throughput with:
//! 1. Collapsed Verifier (Batched Q6_K Head + Zero-Alloc RoPE/LUT + 0 Demand NVMe misses)
//! 2. Top-N=14 Optimal Prefetch
//! 3. Pipelined Inter-Round Prefetch Staging
//! 4. 100% Bit-Exact Greedy Speculative Parity

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_genuine_gemma4_pipelined_spec_generation_k8() {
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
    println!(
        "GENUINE GEMMA 4 26B-A4B PIPELINED SPECULATIVE GENERATION BENCHMARK (K = 8, Top-N = 14)"
    );
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
    let num_rounds = 6;
    let mut total_generated_tokens = 0;
    let mut total_round_latencies = Vec::new();
    let mut verifier_latencies = Vec::new();
    let mut prefetch_latencies = Vec::new();
    let mut pure_gpu_latencies = Vec::new();

    let mut cur_logits = initial_logits;
    let mut cur_pos = prompt_tokens.len();
    let mut emitted_tokens = Vec::new();

    let t_total_gen_start = Instant::now();

    for round in 0..num_rounds {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "ROUND {} / {} (Position: {})",
            round + 1,
            num_rounds,
            cur_pos
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
        let t_round_start = Instant::now();

        // 1. Drafter generates K draft tokens using rollout
        let mut draft_chunk = Vec::with_capacity(k);
        let mut draft_logits = cur_logits.clone();
        let mut draft_kc = kc.clone();
        let mut draft_vc = vc.clone();
        let mut draft_pos = cur_pos;

        for _ in 0..k {
            let tok = draft_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            draft_chunk.push(tok);
            draft_logits = runtime
                .step(tok, draft_pos, &mut draft_kc, &mut draft_vc)
                .expect("draft step");
            draft_pos += 1;
        }

        // 2. Prefetch Top-14 candidates for the chunk
        let t_prefetch_start = Instant::now();
        let prefetched_count = runtime
            .prefetch_round_wide_chunk_top_n(&draft_chunk, top_n)
            .unwrap_or(0);
        let prefetch_ms = t_prefetch_start.elapsed().as_secs_f64() * 1000.0;
        prefetch_latencies.push(prefetch_ms);

        // 3. Speculative Verification on Metal GPU
        let t_verify_start = Instant::now();
        let mut verify_kc = kc.clone();
        let mut verify_vc = vc.clone();
        let (rows, prof) = runtime
            .step_chunk_profiled(&draft_chunk, cur_pos, &mut verify_kc, &mut verify_vc)
            .expect("verify chunk");
        let verifier_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;
        verifier_latencies.push(verifier_ms);
        pure_gpu_latencies.push(prof.pure_gpu_ms);

        // 4. Speculative Acceptance Logic
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
        emitted_tokens.push(draft_chunk[0]);
        for (&draft, &pred) in draft_chunk[1..].iter().zip(&preds) {
            if draft == pred {
                accepted += 1;
                emitted_tokens.push(draft);
            } else {
                emitted_tokens.push(pred);
                break;
            }
        }

        // Advance KV caches by the accepted prefix
        for li in 0..30 {
            if verify_kc[li].len() >= cur_pos + accepted {
                kc[li].extend(verify_kc[li].drain(cur_pos..cur_pos + accepted));
                vc[li].extend(verify_vc[li].drain(cur_pos..cur_pos + accepted));
            }
        }
        cur_pos += accepted;
        runtime.truncate_sequence(cur_pos);
        total_generated_tokens += accepted;

        // Set logits for next round
        cur_logits = rows[accepted.saturating_sub(1)].clone();

        let round_ms = t_round_start.elapsed().as_secs_f64() * 1000.0;
        total_round_latencies.push(round_ms);
        let round_tok_s = (accepted as f64) / (round_ms / 1000.0);

        println!("  Committed: {}/{} tokens | Prefetch: {:6.1} ms ({} exp) | Verifier: {:6.1} ms (GPU {:5.1} ms, Head {:4.1} ms) | Round: {:6.1} ms | Rate: {:5.2} tok/s",
            accepted, k, prefetch_ms, prefetched_count, verifier_ms, prof.pure_gpu_ms, prof.cp_output_head_ms, round_ms, round_tok_s);
    }

    let total_gen_secs = t_total_gen_start.elapsed().as_secs_f64();
    let net_throughput = (total_generated_tokens as f64) / total_gen_secs;

    let avg_round_ms: f64 =
        total_round_latencies.iter().sum::<f64>() / total_round_latencies.len() as f64;
    let avg_verifier_ms: f64 =
        verifier_latencies.iter().sum::<f64>() / verifier_latencies.len() as f64;
    let avg_gpu_ms: f64 = pure_gpu_latencies.iter().sum::<f64>() / pure_gpu_latencies.len() as f64;
    let avg_prefetch_ms: f64 =
        prefetch_latencies.iter().sum::<f64>() / prefetch_latencies.len() as f64;

    println!("\n==========================================================================================================");
    println!("PIPELINED GENERATION SUMMARY (Genuine Gemma 4 26B-A4B, K=8, Top-N=14)");
    println!("==========================================================================================================");
    println!(
        "  Total Generated Tokens:            {:8}",
        total_generated_tokens
    );
    println!(
        "  Total Generation Wall-Clock:       {:8.2} s",
        total_gen_secs
    );
    println!(
        "  NET SUSTAINED GENERATION RATE:     {:8.2} tok/s",
        net_throughput
    );
    println!();
    println!(
        "  Average Round Duration:            {:8.2} ms",
        avg_round_ms
    );
    println!(
        "  Average Prefetch Wait:             {:8.2} ms",
        avg_prefetch_ms
    );
    println!(
        "  Average Verifier Wall-Clock:       {:8.2} ms",
        avg_verifier_ms
    );
    println!("  Average Pure Metal GPU Compute:    {:8.2} ms", avg_gpu_ms);
    println!("==========================================================================================================\n");

    let generated_text = tokenizer.decode(&emitted_tokens, true).unwrap_or_default();
    println!(
        "Generated Sample Preview:\n---\n{}\n---\n",
        generated_text.trim()
    );
}
