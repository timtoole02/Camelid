#![cfg(target_os = "macos")]

//! Top-N Pruning Sweep Benchmark for K=8 on Genuine Gemma 4 26B-A4B
//!
//! Sweeps Top-N candidate prefetch widths (N = 8, 9, 10, 11, 12, 14, 16)
//! to find the smallest N that produces ZERO physical NVMe demand faults inside verification.

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[cfg(target_os = "macos")]
fn get_page_faults() -> (u64, u64) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let res = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if res == 0 {
        let usage = unsafe { usage.assume_init() };
        (usage.ru_majflt as u64, usage.ru_minflt as u64)
    } else {
        (0, 0)
    }
}

#[derive(Debug, Clone, Default)]
struct PruningSweepResult {
    top_n: usize,
    prefetched_experts: usize,
    prefetch_ms: f64,
    prefetch_maj_faults: u64,
    verifier_ms: f64,
    verifier_pure_gpu_ms: f64,
    verifier_head_ms: f64,
    verifier_maj_faults: u64,
    verifier_min_faults: u64,
    true_round_ms: f64,
    actual_tok_s: f64,
}

#[test]
fn test_genuine_gemma4_top_n_pruning_sweep_k8() {
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
    println!("GENUINE GEMMA 4 26B-A4B TOP-N PREFETCH PRUNING SWEEP (K = 8)");
    println!("Target: Find smallest Top-N candidate width achieving 0 demand NVMe faults in verification");
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
    for cur_pos in (prompt_tokens.len()..).take(16) {
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
    }

    let k = 8;
    let candidate_chunk = &draft_pool[..k];
    let start_pos = prompt_tokens.len();

    let n_candidates = [8, 9, 10, 11, 12, 14, 16];
    let mut results = Vec::new();

    for &top_n in &n_candidates {
        println!(">>> TESTING TOP-N = {} CANDIDATES PER LAYER <<<", top_n);
        let t_outer_start = Instant::now();

        // 1. Prefetch phase
        let (maj_pre_0, _) = get_page_faults();
        let t_prefetch_start = Instant::now();
        let count = runtime
            .prefetch_round_wide_chunk_top_n(candidate_chunk, top_n)
            .unwrap_or(0);
        let prefetch_ms = t_prefetch_start.elapsed().as_secs_f64() * 1000.0;
        let (maj_pre_1, _) = get_page_faults();
        let prefetch_maj = maj_pre_1.saturating_sub(maj_pre_0);

        // 2. Verification phase
        let (maj_v_0, min_v_0) = get_page_faults();
        let t_verify_start = Instant::now();
        let mut round_kc = kc.clone();
        let mut round_vc = vc.clone();
        let (_rows, prof) = runtime
            .step_chunk_profiled(candidate_chunk, start_pos, &mut round_kc, &mut round_vc)
            .expect("step_chunk_profiled");
        let verifier_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;
        let (maj_v_1, min_v_1) = get_page_faults();
        let verifier_maj = maj_v_1.saturating_sub(maj_v_0);
        let verifier_min = min_v_1.saturating_sub(min_v_0);

        let true_round_ms = t_outer_start.elapsed().as_secs_f64() * 1000.0;
        let actual_tok_s = (k as f64) / (true_round_ms / 1000.0);

        let res = PruningSweepResult {
            top_n,
            prefetched_experts: count,
            prefetch_ms,
            prefetch_maj_faults: prefetch_maj,
            verifier_ms,
            verifier_pure_gpu_ms: prof.pure_gpu_ms,
            verifier_head_ms: prof.cp_output_head_ms,
            verifier_maj_faults: verifier_maj,
            verifier_min_faults: verifier_min,
            true_round_ms,
            actual_tok_s,
        };

        println!("  Top-N = {:2} | Prefetch: {:7.2} ms / {:4} experts ({:5} majflt) | Verifier: {:7.2} ms (GPU {:6.2}ms, Head {:6.2}ms) | Demand Faults: {:5} maj / {:5} min | Emitted: {:5.2} tok/s",
            res.top_n, res.prefetch_ms, res.prefetched_experts, res.prefetch_maj_faults, res.verifier_ms, res.verifier_pure_gpu_ms, res.verifier_head_ms, res.verifier_maj_faults, res.verifier_min_faults, res.actual_tok_s);

        results.push(res);
    }

    println!("\n==========================================================================================================");
    println!("TOP-N PRUNING SWEEP SUMMARY (K = 8)");
    println!("==========================================================================================================");
    println!(" Top-N | Prefetch ms | Prefetch MajFault | Verifier ms | Head ms | Verifier MajFault | True Round ms | Tok/s ");
    println!("-------|-------------|-------------------|-------------|---------|-------------------|---------------|-------");
    for r in &results {
        println!(
            "  {:4} | {:11.2} | {:17} | {:11.2} | {:7.2} | {:17} | {:13.2} | {:5.2}",
            r.top_n,
            r.prefetch_ms,
            r.prefetch_maj_faults,
            r.verifier_ms,
            r.verifier_head_ms,
            r.verifier_maj_faults,
            r.true_round_ms,
            r.actual_tok_s
        );
    }
    println!("==========================================================================================================\n");
}
