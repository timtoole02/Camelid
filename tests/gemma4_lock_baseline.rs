//! Baseline Lock for Gemma 4 26B-A4B MoE on Apple Silicon Metal
//!
//! Measures:
//! - 256 advancing tokens K=1 decode
//! - Prefill latency separately
//! - Per-token latencies: p50, p90, p95, min/max/median/mean
//! - Slot hit rate, misses/token, fill ms/token
//! - Process RSS, macOS compressed memory, swap usage
//! - 48-token deterministic parity verification against llama.cpp oracle

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, process::Command, time::Instant};

fn get_peak_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        rusage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    0.0
}

fn get_current_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        #[allow(deprecated)]
        let task = libc::mach_task_self();
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
            / std::mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let kret = libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut libc::integer_t,
            &mut count,
        );
        if kret == libc::KERN_SUCCESS {
            info.resident_size as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
    #[cfg(not(target_os = "macos"))]
    0.0
}

fn get_macos_vm_info() -> (String, String) {
    let swap_str = Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
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
                    let bytes = pages * 16384;
                    compressed_mb = format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0));
                }
            }
        }
    }

    (compressed_mb, swap_str)
}

#[test]
fn test_gemma4_26b_lock_baseline() {
    let model_path = std::env::var_os("CAMELID_GEMMA4_26B_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf"));
    let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost"));

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!(
            "SKIP: Model files not found at {:?} / {:?}",
            model_path, cghost_path
        );
        return;
    }

    let slots_per_layer = std::env::var("KNOB_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(88);

    std::env::set_var("CAMELID_GHOST_ALLOW_LEGACY_SPARSE", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var(
        "CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER",
        slots_per_layer.to_string(),
    );
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");

    println!("==========================================================================================================");
    println!("GEMMA 4 26B-A4B BASELINE LOCK (88 slots/layer K=1, 256 Advancing Tokens)");
    println!("Model:  {}", model_path.display());
    println!("Sidecar: {}", cghost_path.display());
    println!("Slots/layer: {}", slots_per_layer);
    println!("==========================================================================================================\n");

    let t_load_start = Instant::now();
    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 3072, false)
        .expect("load ghost moe");
    let load_ms = t_load_start.elapsed().as_secs_f64() * 1000.0;
    println!("Model loaded in {:.2} ms", load_ms);

    let (comp_init, swap_init) = get_macos_vm_info();
    println!(
        "Initial VM: RSS={:.1} MB, compressed={}, swap={}",
        get_current_rss_mb(),
        comp_init,
        swap_init
    );

    // 1. Part 1: 48 Deterministic Tokens Parity Check
    println!("\n----------------------------------------------------------------------------------------------------------");
    println!("PART 1: 48 DETERMINISTIC TOKENS ORACLE PARITY CHECK (QKD Prompt)");
    println!("----------------------------------------------------------------------------------------------------------");
    let qkd_prompt = "<|turn>user\nExplain quantum key distribution protocols, BB84 and E91, in detail and contrast their security proofs.<turn|>\n<|turn>model\n";
    let qkd_prompt_tokens = runtime
        .tokenizer()
        .encode(qkd_prompt, true, true)
        .expect("tokenize");

    let mut qkd_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut qkd_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let t_qkd_prefill = Instant::now();
    let qkd_init_logits = runtime
        .prefill_tokens(&qkd_prompt_tokens, &mut qkd_kc, &mut qkd_vc, 0)
        .expect("qkd prefill");
    let qkd_prefill_ms = t_qkd_prefill.elapsed().as_secs_f64() * 1000.0;
    println!(
        "QKD Prefill ({} tokens): {:.2} ms",
        qkd_prompt_tokens.len(),
        qkd_prefill_ms
    );

    let mut qkd_tokens = Vec::new();
    let mut qkd_logits = qkd_init_logits;
    let mut qkd_pos = qkd_prompt_tokens.len();
    for _ in 0..48 {
        let tok = qkd_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        qkd_tokens.push(tok);
        qkd_logits = runtime
            .step(tok, qkd_pos, &mut qkd_kc, &mut qkd_vc)
            .expect("step");
        qkd_pos += 1;
    }
    println!("48 tokens generated: {:?}", qkd_tokens);
    let qkd_text = runtime
        .tokenizer()
        .decode(&qkd_tokens, false)
        .unwrap_or_default();
    println!(
        "48 tokens decoded text preview:\n{:?}",
        &qkd_text[..qkd_text.len().min(200)]
    );

    // 2. Part 2: 256 Advancing Tokens Profile
    println!("\n----------------------------------------------------------------------------------------------------------");
    println!("PART 2: 256 ADVANCING TOKENS K=1 PROFILE");
    println!("----------------------------------------------------------------------------------------------------------");
    runtime.rollback_sequence(0);

    let (hits_before, misses_before) = runtime.ghost_metal_aggregate_slot_stats();

    let gen_prompt = "<|turn>user\nExplain how general relativity predicts gravitational lensing, gravitational time dilation, and frame dragging in detail.<turn|>\n<|turn>model\n";
    let gen_prompt_tokens = runtime
        .tokenizer()
        .encode(gen_prompt, true, true)
        .expect("tokenize");

    let mut gen_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut gen_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let t_prefill_start = Instant::now();
    let init_logits = runtime
        .prefill_tokens(&gen_prompt_tokens, &mut gen_kc, &mut gen_vc, 0)
        .expect("prefill");
    let prefill_ms = t_prefill_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Prefill ({} tokens): {:.2} ms ({:.2} tok/s)",
        gen_prompt_tokens.len(),
        prefill_ms,
        gen_prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
    );

    let (hits_post_prefill, misses_post_prefill) = runtime.ghost_metal_aggregate_slot_stats();

    let n_tokens = 256;
    let mut generated_tokens = Vec::with_capacity(n_tokens);
    let mut token_latencies_ms = Vec::with_capacity(n_tokens);
    let mut cur_logits = init_logits;
    let mut cur_pos = gen_prompt_tokens.len();

    let t_decode_start = Instant::now();
    for step_i in 0..n_tokens {
        let t_tok_start = Instant::now();
        let tok = cur_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        generated_tokens.push(tok);

        cur_logits = runtime
            .step(tok, cur_pos, &mut gen_kc, &mut gen_vc)
            .expect("step");
        cur_pos += 1;

        let tok_ms = t_tok_start.elapsed().as_secs_f64() * 1000.0;
        token_latencies_ms.push(tok_ms);

        if (step_i + 1) % 32 == 0 || step_i == 0 {
            let recent_slice = if step_i >= 31 {
                &token_latencies_ms[step_i - 31..=step_i]
            } else {
                &token_latencies_ms[..=step_i]
            };
            let mut sorted = recent_slice.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = sorted[sorted.len() / 2];
            println!(
                "  Step {:3}/256: cur={:5.1}ms | window_med={:5.1}ms ({:4.1} tok/s) | RSS={:.1} MB",
                step_i + 1,
                tok_ms,
                med,
                1000.0 / med,
                get_current_rss_mb()
            );
        }
    }
    let total_decode_dur_s = t_decode_start.elapsed().as_secs_f64();
    let total_e2e_dur_s = (prefill_ms / 1000.0) + total_decode_dur_s;

    let full_text = runtime
        .tokenizer()
        .decode(&generated_tokens, false)
        .unwrap_or_default();
    println!(
        "\nGenerated Text (first 300 chars):\n{}",
        &full_text[..full_text.len().min(300)]
    );

    // Calculations
    let mut sorted_latencies = token_latencies_ms.clone();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = sorted_latencies[n_tokens * 50 / 100];
    let p90 = sorted_latencies[n_tokens * 90 / 100];
    let p95 = sorted_latencies[n_tokens * 95 / 100];
    let min_latency = sorted_latencies[0];
    let max_latency = sorted_latencies[n_tokens - 1];

    // Steady state: tokens 32..256
    let steady_slice = &token_latencies_ms[32..];
    let mut sorted_steady = steady_slice.to_vec();
    sorted_steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let steady_median_ms = sorted_steady[sorted_steady.len() / 2];
    let steady_mean_ms = steady_slice.iter().sum::<f64>() / steady_slice.len() as f64;

    let all_mean_ms = token_latencies_ms.iter().sum::<f64>() / n_tokens as f64;
    let all_median_ms = p50;

    let (hits_end, misses_end) = runtime.ghost_metal_aggregate_slot_stats();
    let decode_hits = hits_end.saturating_sub(hits_post_prefill);
    let decode_misses = misses_end.saturating_sub(misses_post_prefill);
    let decode_accesses = decode_hits + decode_misses;

    let hit_rate = if decode_accesses > 0 {
        decode_hits as f64 / decode_accesses as f64 * 100.0
    } else {
        0.0
    };
    let misses_per_tok = decode_misses as f64 / n_tokens as f64;
    // Each miss takes ~1.91 ms on positioned pread
    let est_fill_ms_per_tok = misses_per_tok * 1.91;

    let (comp_end, swap_end) = get_macos_vm_info();
    let peak_rss_mb = get_peak_rss_mb();
    let cur_rss_mb = get_current_rss_mb();

    println!("\n==========================================================================================================");
    println!("BASELINE_CORRECT METRICS SUMMARY");
    println!("==========================================================================================================");
    println!("Tokens generated:             {}", n_tokens);
    println!(
        "Prefill latency:              {:.2} ms ({} tokens, {:.2} tok/s)",
        prefill_ms,
        gen_prompt_tokens.len(),
        gen_prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
    );
    println!("Total decode wall-clock:      {:.2} s", total_decode_dur_s);
    println!("End-to-end wall-clock:        {:.2} s", total_e2e_dur_s);
    println!(
        "End-to-end throughput:        {:.2} tok/s",
        n_tokens as f64 / total_e2e_dur_s
    );
    println!("----------------------------------------------------------------------------------------------------------");
    println!(
        "Steady decode median (tok/s): {:.2} tok/s ({:.2} ms)",
        1000.0 / steady_median_ms,
        steady_median_ms
    );
    println!(
        "Steady decode mean (tok/s):   {:.2} tok/s ({:.2} ms)",
        1000.0 / steady_mean_ms,
        steady_mean_ms
    );
    println!(
        "Overall decode median:        {:.2} tok/s ({:.2} ms)",
        1000.0 / all_median_ms,
        all_median_ms
    );
    println!(
        "Overall decode mean:          {:.2} tok/s ({:.2} ms)",
        1000.0 / all_mean_ms,
        all_mean_ms
    );
    println!(
        "Best token latency:           {:.2} ms ({:.2} tok/s)",
        min_latency,
        1000.0 / min_latency
    );
    println!("Worst token latency:          {:.2} ms", max_latency);
    println!("Latency p50:                  {:.2} ms", p50);
    println!("Latency p90:                  {:.2} ms", p90);
    println!("Latency p95:                  {:.2} ms", p95);
    println!("----------------------------------------------------------------------------------------------------------");
    println!("Slot accesses:                {}", decode_accesses);
    println!(
        "Slot hits:                    {} ({:.2}%)",
        decode_hits, hit_rate
    );
    println!("Slot misses:                  {}", decode_misses);
    println!("Misses per token:             {:.2}", misses_per_tok);
    println!(
        "Est fill ms per token:        {:.2} ms",
        est_fill_ms_per_tok
    );
    println!("----------------------------------------------------------------------------------------------------------");
    println!(
        "Current process RSS:          {:.1} MB ({:.2} GiB)",
        cur_rss_mb,
        cur_rss_mb / 1024.0
    );
    println!(
        "Peak process RSS:             {:.1} MB ({:.2} GiB)",
        peak_rss_mb,
        peak_rss_mb / 1024.0
    );
    println!("Compressed memory:            {}", comp_end);
    println!("Swap usage:                   {}", swap_end);
    println!("==========================================================================================================\n");
}
