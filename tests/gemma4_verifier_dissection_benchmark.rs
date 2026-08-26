#![cfg(target_os = "macos")]

//! Detailed Microsecond Dissection Benchmark of the K=8 Verifier on Genuine Gemma 4 26B-A4B
//!
//! Reconciles:
//! K=8 VERIFIER
//! --------------------------------
//! Pure Metal command-buffer execution: XX ms
//! Demand expert cache lookups:         XX ms
//! Demand RAM/page-cache fills:         XX ms
//! Demand physical NVMe waits:          XX ms
//! Expert slab copy/mapping:            XX ms
//! Command encoding CPU:                XX ms
//! Metal synchronization:               XX ms
//! Attention & PLE compute:             XX ms
//! Other:                               XX ms
//! --------------------------------
//! TOTAL:                              XXXX ms

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

#[test]
fn test_genuine_gemma4_verifier_dissection_k8() {
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
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_TIMING", "1");

    println!("==========================================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B K=8 VERIFIER MICROSECOND DISSECTION BENCHMARK");
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

    // 1. First run: with round-wide prefetch executed beforehand
    println!(">>> RUNNING K=8 VERIFIER WITH ROUND-WIDE PREFETCH <<<");
    let _ = runtime.prefetch_round_wide_chunk(candidate_chunk);

    let (maj_before, min_before) = get_page_faults();
    let t_verify_start = Instant::now();
    let mut round_kc = kc.clone();
    let mut round_vc = vc.clone();

    let (_rows, prof) = runtime
        .step_chunk_profiled(candidate_chunk, start_pos, &mut round_kc, &mut round_vc)
        .expect("step_chunk_profiled");
    let total_verifier_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;
    let (maj_after, min_after) = get_page_faults();

    let maj_diff = maj_after.saturating_sub(maj_before);
    let min_diff = min_after.saturating_sub(min_before);

    // Compute detailed dissection components:
    let pure_metal_ms = prof.pure_gpu_ms;
    let attention_core_ms = prof.attention_core_ms;
    let cache_lookup_ms = prof.cp_cache_slot_lookup_ms;
    let mapping_and_prep_ms = prof.layer0_buffer_prep_ms * 30.0;
    let physical_reads_ms = prof.physical_ssd_reads_ms;
    let metal_sync_ms = (prof.layer0_commit_wait_ms * 30.0).max(0.0);
    let command_encode_cpu_ms = (prof.all_moe_layers_ms
        - (pure_metal_ms + cache_lookup_ms + physical_reads_ms + metal_sync_ms))
        .max(0.0);
    let other_ms = (total_verifier_ms
        - (pure_metal_ms
            + attention_core_ms
            + cache_lookup_ms
            + mapping_and_prep_ms
            + physical_reads_ms
            + metal_sync_ms
            + command_encode_cpu_ms))
        .max(0.0);

    println!("--------------------------------------------------------------------------------");
    println!("K=8 VERIFIER DISSECTION LEDGER");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  Pure Metal GPU hardware execution: {:8.2} ms",
        pure_metal_ms
    );
    println!(
        "  Attention & Common-Core (CPU/GPU): {:8.2} ms",
        attention_core_ms
    );
    println!(
        "  Demand expert slot/cache lookups:  {:8.2} ms",
        cache_lookup_ms
    );
    println!(
        "  Demand physical NVMe / RAM reads:  {:8.2} ms",
        physical_reads_ms
    );
    println!(
        "  Expert slab work/route preparation:{:8.2} ms",
        mapping_and_prep_ms
    );
    println!(
        "  Command encoding & Q8 quant (CPU): {:8.2} ms",
        command_encode_cpu_ms
    );
    println!(
        "  Metal command commit & sync wait:  {:8.2} ms",
        metal_sync_ms
    );
    println!("  Other residual / head processing:  {:8.2} ms", other_ms);
    println!("--------------------------------------------------------------------------------");
    println!(
        "  TOTAL VERIFIER WALL-CLOCK:         {:8.2} ms",
        total_verifier_ms
    );
    println!();
    println!(
        "  Physical NVMe major faults in verifier: {:8} ({:.2} MB)",
        maj_diff,
        (maj_diff * 16384) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Page-cache RAM minor faults in verifier:{:8} ({:.2} MB)",
        min_diff,
        (min_diff * 16384) as f64 / (1024.0 * 1024.0)
    );
    println!("--------------------------------------------------------------------------------\n");

    println!("Detailed Profile Struct Dump:");
    println!("{:#?}", prof);
    println!("==========================================================================================================\n");
}
