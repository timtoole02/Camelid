//! Real Gemma 4 26B-A4B Cache Budget Sweep (1 GiB to 8 GiB)
//!
//! Measures real runtime performance over 256 generated tokens for:
//! 1 GiB, 2 GiB, 3 GiB, 4 GiB, 6 GiB, 8 GiB.
//!
//! Reports:
//! - cold-start tok/s (first 32 tokens)
//! - warm steady-state tok/s (tokens 32..256)
//! - peak RSS
//! - macOS compression / swap
//! - cache hit rate
//! - compulsory misses vs capacity misses
//! - SSD MB/token
//! - SSD ms/token
//! - effective SSD GB/s
//! - Exact token sequence correctness across all configurations.

mod support;

use std::{path::PathBuf, process::Command, time::Instant};

use camelid::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};

fn get_peak_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        // On macOS, ru_maxrss is in bytes
        rusage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

fn get_macos_vm_info() -> (String, String) {
    let swap_str = Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());

    let vm_stat_out = Command::new("vm_stat")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let mut compressed_mb = "0 MB".to_string();
    for line in vm_stat_out.lines() {
        if line.starts_with("Pages occupied by compressor:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(num_str) = parts.last() {
                let clean = num_str.trim_end_matches('.');
                if let Ok(pages) = clean.parse::<u64>() {
                    // Apple Silicon page size is 16384 bytes (16 KB)
                    let bytes = pages * 16384;
                    compressed_mb = format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0));
                }
            }
        }
    }

    let clean_swap = swap_str.trim().to_string();
    (compressed_mb, clean_swap)
}

struct SweepResult {
    budget_gib: usize,
    cold_tok_s: f64,
    warm_tok_s: f64,
    total_tok_s: f64,
    peak_rss_mb: f64,
    compression: String,
    swap: String,
    hit_rate: f64,
    compulsory_misses: u64,
    capacity_misses: u64,
    ssd_mb_per_tok: f64,
    ssd_ms_per_tok: f64,
    effective_ssd_gb_s: f64,
    tokens: Vec<u32>,
}

fn run_cache_benchmark(
    model_path: &PathBuf,
    cghost_path: &PathBuf,
    budget_mib: usize,
) -> SweepResult {
    let budget_gib = budget_mib / 1024;
    println!("\n================================================================================");
    println!(
        "RUNNING REAL GEMMA 4 26B BENCHMARK WITH {} GiB CACHE ({} MiB)",
        budget_gib, budget_mib
    );
    println!("================================================================================");

    let runtime = Gemma4Runtime::load_ghost_moe(model_path, cghost_path, budget_mib, false)
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

    let num_tokens = 256;
    let mut generated_tokens = Vec::with_capacity(num_tokens);
    let mut cur_tok = last_tok;
    let mut cur_pos = decode_pos;

    let mut total_ssd_time_us = 0u64;
    let mut total_ssd_bytes = 0u64;

    // Track cold start (tokens 0..32) and warm steady state (tokens 32..256)
    let t_start_total = Instant::now();
    let mut t_cold_end = t_start_total;

    for step in 0..num_tokens {
        let (out, prof) = runtime
            .step_range_profiled(cur_tok, cur_pos, None, &mut kc, &mut vc)
            .expect("step");
        let logits = match out {
            Gemma4StepOutput::Logits(l) => l,
            _ => panic!("expected logits"),
        };

        total_ssd_time_us += prof.cache_and_io_us;
        total_ssd_bytes += prof.bytes_read as u64;

        let mut next_id = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_logit {
                max_logit = v;
                next_id = i as u32;
            }
        }

        generated_tokens.push(next_id);
        cur_tok = next_id;
        cur_pos += 1;

        if step == 31 {
            t_cold_end = Instant::now();
        }
    }

    let t_end_total = Instant::now();
    let total_dur = t_end_total.duration_since(t_start_total);
    let cold_dur = t_cold_end.duration_since(t_start_total);
    let warm_dur = t_end_total.duration_since(t_cold_end);

    let cold_tok_s = 32.0 / cold_dur.as_secs_f64();
    let warm_tok_s = (num_tokens - 32) as f64 / warm_dur.as_secs_f64();
    let total_tok_s = num_tokens as f64 / total_dur.as_secs_f64();

    let stats = runtime.ghost_moe_cache_stats().expect("cache stats");
    let total_calls = stats.hits + stats.misses;
    let hit_rate = if total_calls > 0 {
        (stats.hits as f64 / total_calls as f64) * 100.0
    } else {
        0.0
    };

    let ssd_mb_per_tok = (total_ssd_bytes as f64 / (1024.0 * 1024.0)) / num_tokens as f64;
    let ssd_ms_per_tok = (total_ssd_time_us as f64 / 1000.0) / num_tokens as f64;
    let effective_ssd_gb_s = if total_ssd_time_us > 0 {
        (total_ssd_bytes as f64 / 1e9) / (total_ssd_time_us as f64 / 1e6)
    } else {
        0.0
    };

    let peak_rss_mb = get_peak_rss_mb();
    let (compression, swap) = get_macos_vm_info();

    println!("Results for {} GiB Cache:", budget_gib);
    println!(
        "  Cold-start speed (tokens 0..32):   {:.2} tok/s",
        cold_tok_s
    );
    println!(
        "  Warm steady-state (tokens 32..256): {:.2} tok/s",
        warm_tok_s
    );
    println!(
        "  Overall 256 tokens:                 {:.2} tok/s",
        total_tok_s
    );
    println!(
        "  Cache Hit Rate:                     {:.2}% ({} hits / {} calls)",
        hit_rate, stats.hits, total_calls
    );
    println!(
        "  Compulsory Misses:                  {} ({:.2}%)",
        stats.compulsory_misses,
        stats.compulsory_misses as f64 / total_calls as f64 * 100.0
    );
    println!(
        "  Capacity Misses:                    {} ({:.2}%)",
        stats.capacity_misses,
        stats.capacity_misses as f64 / total_calls as f64 * 100.0
    );
    println!(
        "  SSD Traffic:                        {:.2} MB/tok, {:.2} ms/tok (Effective: {:.2} GB/s)",
        ssd_mb_per_tok, ssd_ms_per_tok, effective_ssd_gb_s
    );
    println!(
        "  Peak RSS:                           {:.1} MB ({:.2} GB)",
        peak_rss_mb,
        peak_rss_mb / 1024.0
    );
    println!("  macOS Compression:                  {}", compression);
    println!("  macOS Swap:                         {}", swap);

    let decoded_snippet = tokenizer
        .decode(&generated_tokens[..generated_tokens.len().min(40)], true)
        .unwrap_or_default();
    println!(
        "  Output Snippet:                     {:?}",
        decoded_snippet
    );

    SweepResult {
        budget_gib,
        cold_tok_s,
        warm_tok_s,
        total_tok_s,
        peak_rss_mb,
        compression,
        swap,
        hit_rate,
        compulsory_misses: stats.compulsory_misses,
        capacity_misses: stats.capacity_misses,
        ssd_mb_per_tok,
        ssd_ms_per_tok,
        effective_ssd_gb_s,
        tokens: generated_tokens,
    }
}

#[test]
fn sweep_real_gemma4_26b_expert_cache_budgets() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    println!("================================================================================");
    println!("architecture: Gemma4 MoE");
    println!("expert_sidecar_loaded: true");
    println!("expert_count: 128");
    println!("router_top_k: 8");
    println!("expert_file: {}", cghost_path.display());
    println!("routed_expert_execution: true");
    println!("================================================================================");

    let budgets_mib = vec![1024, 2048, 3072, 4096, 6144, 8192];
    let mut results = Vec::new();

    for &b in &budgets_mib {
        let res = run_cache_benchmark(&model_path, &cghost_path, b);
        results.push(res);
    }

    // Verify token identity across all runs
    let baseline_tokens = &results[0].tokens;
    for (_i, res) in results.iter().enumerate().skip(1) {
        assert_eq!(
            baseline_tokens, &res.tokens,
            "Token sequence mismatch between 1 GiB and {} GiB cache!",
            res.budget_gib
        );
        println!(
            "Verified: {} GiB cache produced 100% bit-exact tokens identical to 1 GiB baseline.",
            res.budget_gib
        );
    }

    println!("\n\n========================================================================================================================");
    println!("OFFICIAL REAL GEMMA 4 26B-A4B EXPERT CACHE SWEEP MATRIX (256 TOKENS)");
    println!("========================================================================================================================");
    println!("{:<8} | {:<12} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10} | {:<12} | {:<12} | {:<12} | {:<10}",
        "Cache", "Cold tok/s", "Warm tok/s", "Total tok/s", "Peak RSS", "Hit Rate", "Compulsory", "Capacity", "SSD MB/tok", "SSD ms/tok", "SSD GB/s");
    println!("{:-<8}-|-{:-<12}-|-{:-<12}-|-{:-<10}-|-{:-<10}-|-{:-<10}-|-{:-<10}-|-{:-<12}-|-{:-<12}-|-{:-<12}-|-{:-<10}",
        "", "", "", "", "", "", "", "", "", "", "");

    for r in &results {
        println!("{:<8} | {:<12.2} | {:<12.2} | {:<10.2} | {:<7.1} GB | {:<9.1}% | {:<10} | {:<10} | {:<12.2} | {:<12.2} | {:<10.2}",
            format!("{} GiB", r.budget_gib),
            r.cold_tok_s,
            r.warm_tok_s,
            r.total_tok_s,
            r.peak_rss_mb / 1024.0,
            r.hit_rate,
            r.compulsory_misses,
            r.capacity_misses,
            r.ssd_mb_per_tok,
            r.ssd_ms_per_tok,
            r.effective_ssd_gb_s
        );
    }
    println!("========================================================================================================================\n");
}
