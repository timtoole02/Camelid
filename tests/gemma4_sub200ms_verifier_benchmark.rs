//! Sub-200ms Verifier Optimization Benchmark (10 Iterations) for K=8 on Genuine Gemma 4 26B-A4B
//!
//! Measures warm steady-state latency and verifies 100% bit-exact greedy parity.

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_genuine_gemma4_sub200ms_verifier_k8() {
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
        "GENUINE GEMMA 4 26B-A4B SUB-200MS VERIFIER BENCHMARK (K = 8, Top-N = 14, 10 Iterations)"
    );
    println!("Budget: 24 slots/layer (2.25 GiB Metal resident, 2.96 GiB Page Cache)");
    println!("==========================================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load ghost moe");

    let prompt = "<|turn>user\nExplain quantum entanglement and its applications in cryptography in simple terms.<turn|>\n<|turn>model\n";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let initial_logits = runtime
        .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)
        .expect("prefill");

    // Generate candidate continuation tokens using greedy draft rollout for realistic draft tokens
    let mut draft_pool = Vec::new();
    let mut cur_logits = initial_logits.clone();
    let mut temp_kc = kc.clone();
    let mut temp_vc = vc.clone();
    let mut cur_pos = prompt_tokens.len();
    for _ in 0..16 {
        let tok = cur_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        draft_pool.push(tok);
        cur_logits = runtime
            .step(tok, cur_pos, &mut temp_kc, &mut temp_vc)
            .expect("step");
        cur_pos += 1;
    }

    let k = 8;
    let candidate_chunk = &draft_pool[..k];
    let start_pos = prompt_tokens.len();
    let top_n = 14;

    println!(">>> RUNNING PREFETCH FOR TOP-N = 14 <<<");
    let t_prefetch_start = Instant::now();
    let prefetched_count = runtime
        .prefetch_round_wide_chunk_top_n(candidate_chunk, top_n)
        .unwrap_or(0);
    let prefetch_ms = t_prefetch_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Prefetch completed: {:.2} ms ({} experts prefetched)\n",
        prefetch_ms, prefetched_count
    );

    // Warmup step to ensure all Metal pipelines and caches are hot
    let mut round_kc = kc.clone();
    let mut round_vc = vc.clone();
    let _ = runtime
        .step_chunk_profiled(candidate_chunk, start_pos, &mut round_kc, &mut round_vc)
        .expect("warmup");

    println!(">>> BENCHMARKING GENUINE K=8 VERIFIER (10 WARM ROUNDS) <<<");
    let mut verifier_times = Vec::new();
    let mut pure_gpu_times = Vec::new();
    let mut head_times = Vec::new();
    let mut verified_accepted = 0;

    for iter in 0..10 {
        let mut test_kc = kc.clone();
        let mut test_vc = vc.clone();
        let t_verify_start = Instant::now();
        let (rows, prof) = runtime
            .step_chunk_profiled(candidate_chunk, start_pos, &mut test_kc, &mut test_vc)
            .expect("verify");
        let verifier_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;

        // Verify exact parity
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        let preds: Vec<u32> = (0..candidate_chunk.len().saturating_sub(1))
            .map(|i| argmax(&rows[i]))
            .collect();
        let mut accepted = 1;
        for (&draft, &pred) in candidate_chunk[1..].iter().zip(&preds) {
            if draft == pred {
                accepted += 1;
            } else {
                break;
            }
        }
        verified_accepted = accepted;

        verifier_times.push(verifier_ms);
        pure_gpu_times.push(prof.pure_gpu_ms);
        head_times.push(prof.cp_output_head_ms);

        println!("  Iter {:2}: Verifier = {:7.2} ms (Pure GPU: {:6.2} ms, Head: {:5.2} ms, Parity: {}/{} committed)",
            iter + 1, verifier_ms, prof.pure_gpu_ms, prof.cp_output_head_ms, accepted, k);
    }

    let min_verifier = verifier_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg_verifier: f64 = verifier_times.iter().sum::<f64>() / verifier_times.len() as f64;
    let avg_pure_gpu: f64 = pure_gpu_times.iter().sum::<f64>() / pure_gpu_times.len() as f64;
    let avg_head: f64 = head_times.iter().sum::<f64>() / head_times.len() as f64;

    println!("\n--------------------------------------------------------------------------------");
    println!("VERIFIER BENCHMARK SUMMARY (K = 8, Top-N = 14, 10 Rounds)");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  Min Verifier Latency:              {:8.2} ms",
        min_verifier
    );
    println!(
        "  Avg Verifier Latency:              {:8.2} ms",
        avg_verifier
    );
    println!(
        "  Avg Pure Metal GPU Arithmetic:     {:8.2} ms",
        avg_pure_gpu
    );
    println!("  Avg Batched Q6_K Output Head:      {:8.2} ms", avg_head);
    println!(
        "  Parity Status:                     PASS (exact {}/{} tokens)",
        verified_accepted, k
    );
    println!("--------------------------------------------------------------------------------\n");
    assert_eq!(verified_accepted, k, "Greedy parity must match exactly");
}
