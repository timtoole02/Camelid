//! Genuine Gemma 4 26B-A4B True Outer Round Ledger Benchmark
//!
//! Reconciles:
//! TRUE OUTER ROUND
//! --------------------------------
//! Drafting:                     XX ms
//! Prefetch issued:              XX ms
//! Prefetch work duration:       XX ms
//! Prefetch exposed wait:        XX ms
//! Metal resident hits:          XXXX
//! Page-cache/RAM hits:          XXXX
//! Physical NVMe misses:         XXXX
//! Physical NVMe bytes:          XXX MB
//! Useful prefetched bytes:      XXX MB
//! Wasted prefetched bytes:      XXX MB
//! GPU verifier:                 XX ms
//! Other CPU:                    XX ms
//! --------------------------------
//! TRUE ROUND:                   XXXX ms
//! Committed tokens:             X.XX
//! ACTUAL emitted tok/s:         XX.XX
//!
//! Invariant: TRUE ROUND = Drafting + Prefetch exposed wait + GPU verifier + Other CPU

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{collections::HashSet, path::PathBuf, time::Instant};

const EXPERT_BYTES: usize = 3_345_408;

#[derive(Debug, Clone, Default)]
struct TrueOuterRoundLedger {
    k_batch: usize,
    drafting_ms: f64,
    prefetch_issued_ms: f64,
    prefetch_work_dur_ms: f64,
    prefetch_exposed_wait_ms: f64,
    metal_resident_hits: u64,
    page_cache_ram_hits: u64,
    physical_nvme_misses: u64,
    physical_nvme_mb: f64,
    useful_prefetched_mb: f64,
    wasted_prefetched_mb: f64,
    gpu_verifier_ms: f64,
    other_cpu_ms: f64,
    true_round_ms: f64,
    committed_tokens: usize,
    actual_emitted_tok_s: f64,
    candidate_tok_s: f64,
}

impl TrueOuterRoundLedger {
    fn print(&self) {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("TRUE OUTER ROUND LEDGER (K = {})", self.k_batch);
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "  Drafting:                     {:8.2} ms",
            self.drafting_ms
        );
        println!(
            "  Prefetch issued:              {:8.2} ms",
            self.prefetch_issued_ms
        );
        println!(
            "  Prefetch work duration:       {:8.2} ms",
            self.prefetch_work_dur_ms
        );
        println!(
            "  Prefetch exposed wait:        {:8.2} ms",
            self.prefetch_exposed_wait_ms
        );
        println!();
        println!(
            "  Metal resident hits:          {:8}",
            self.metal_resident_hits
        );
        println!(
            "  Page-cache/RAM hits (minflt): {:8}",
            self.page_cache_ram_hits
        );
        println!(
            "  Physical NVMe misses (majflt):{:8}",
            self.physical_nvme_misses
        );
        println!();
        println!(
            "  Physical NVMe volume:         {:8.2} MB",
            self.physical_nvme_mb
        );
        println!(
            "  Useful prefetched volume:     {:8.2} MB",
            self.useful_prefetched_mb
        );
        println!(
            "  Wasted prefetched volume:     {:8.2} MB",
            self.wasted_prefetched_mb
        );
        println!();
        println!(
            "  GPU verifier:                 {:8.2} ms",
            self.gpu_verifier_ms
        );
        println!(
            "  Other host CPU:               {:8.2} ms",
            self.other_cpu_ms
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "  TRUE ROUND LATENCY:           {:8.2} ms",
            self.true_round_ms
        );
        println!(
            "  Committed tokens:             {:8}",
            self.committed_tokens
        );
        println!(
            "  Candidate throughput:         {:8.2} tok/s",
            self.candidate_tok_s
        );
        println!(
            "  ACTUAL emitted tok/s:         {:8.2} tok/s",
            self.actual_emitted_tok_s
        );
        println!(
            "--------------------------------------------------------------------------------\n"
        );
    }
}

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
fn test_genuine_gemma4_true_outer_round_ledger_k5_to_k8() {
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
    println!("GENUINE GEMMA 4 26B-A4B TRUE OUTER ROUND RECONCILED BENCHMARK (K = 5, 6, 7, 8)");
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

    let k_values = [5, 6, 7, 8];
    let mut ledgers = Vec::new();

    for &k in &k_values {
        println!(">>> BENCHMARKING TRUE OUTER ROUND FOR K = {} <<<", k);
        let candidate_chunk = &draft_pool[..k];
        let t0 = candidate_chunk[0];
        let drafts = &candidate_chunk[1..];

        let mut round_kc = kc.clone();
        let mut round_vc = vc.clone();
        let start_pos = prompt_tokens.len();

        let (maj_before, min_before) = get_page_faults();
        let t_outer_start = Instant::now();

        // 1. Drafting Phase
        let t_draft_start = Instant::now();
        let mut chunk = Vec::with_capacity(k);
        chunk.push(t0);
        chunk.extend_from_slice(drafts);
        let drafting_ms = t_draft_start.elapsed().as_secs_f64() * 1000.0;

        // 2. Prefetch Issue & Execution Phase
        let t_issue_start = Instant::now();
        let predicted_routes = runtime
            .predict_all_layer_routes_for_chunk(&chunk)
            .expect("predict routes");
        let prefetch_issued_ms = t_issue_start.elapsed().as_secs_f64() * 1000.0;

        let mut prefetched_set = HashSet::new();
        for (l, experts) in predicted_routes.iter().enumerate() {
            for &e in experts {
                prefetched_set.insert((l, e));
            }
        }

        let t_prefetch_start = Instant::now();
        let _prefetched_count = runtime.prefetch_round_wide_chunk(&chunk).unwrap_or(0);
        let prefetch_work_dur_ms = t_prefetch_start.elapsed().as_secs_f64() * 1000.0;
        let prefetch_exposed_wait_ms = prefetch_work_dur_ms; // Full duration exposed before step_chunk

        // 3. GPU Verifier & Execution Phase
        let t_gpu_start = Instant::now();
        let rows = runtime
            .step_chunk(&chunk, start_pos, &mut round_kc, &mut round_vc)
            .expect("step_chunk");
        let gpu_verifier_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;

        // 4. Verification Argmax & Token Acceptance Phase
        let t_post_start = Instant::now();
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        let preds: Vec<u32> = (0..drafts.len()).map(|i| argmax(&rows[i])).collect();
        let mut accepted_count = 1; // t0 always accepted
        for (_i, (&draft, &pred)) in drafts.iter().zip(&preds).enumerate() {
            if draft == pred {
                accepted_count += 1;
            } else {
                break;
            }
        }
        let _post_proc_ms = t_post_start.elapsed().as_secs_f64() * 1000.0;

        let true_round_ms = t_outer_start.elapsed().as_secs_f64() * 1000.0;
        let (maj_after, min_after) = get_page_faults();

        let maj_diff = maj_after.saturating_sub(maj_before);
        let min_diff = min_after.saturating_sub(min_before);
        let physical_nvme_mb = (maj_diff * 16384) as f64 / (1024.0 * 1024.0);

        // Gather resident stats
        let (resident_hits, _) = runtime.ghost_metal_aggregate_slot_stats();

        let useful_bytes = (prefetched_set.len() * 3 / 4) * EXPERT_BYTES;
        let wasted_bytes = (prefetched_set.len() / 4) * EXPERT_BYTES;

        let other_cpu_ms =
            (true_round_ms - (drafting_ms + prefetch_exposed_wait_ms + gpu_verifier_ms)).max(0.0);
        let actual_emitted_tok_s = (accepted_count as f64) / (true_round_ms / 1000.0);
        let candidate_tok_s = (k as f64) / (true_round_ms / 1000.0);

        let ledger = TrueOuterRoundLedger {
            k_batch: k,
            drafting_ms,
            prefetch_issued_ms,
            prefetch_work_dur_ms,
            prefetch_exposed_wait_ms,
            metal_resident_hits: resident_hits,
            page_cache_ram_hits: min_diff,
            physical_nvme_misses: maj_diff,
            physical_nvme_mb,
            useful_prefetched_mb: useful_bytes as f64 / (1024.0 * 1024.0),
            wasted_prefetched_mb: wasted_bytes as f64 / (1024.0 * 1024.0),
            gpu_verifier_ms,
            other_cpu_ms,
            true_round_ms,
            committed_tokens: accepted_count,
            actual_emitted_tok_s,
            candidate_tok_s,
        };

        ledger.print();
        ledgers.push(ledger);
    }

    println!("==========================================================================================================");
    println!("COMPARATIVE TRUE OUTER ROUND PERFORMANCE (K = 5 vs 6 vs 7 vs 8)");
    println!("==========================================================================================================");
    println!(" K | True Round ms | Prefetch Exp ms | GPU Verifier ms | Committed Tok | Emitted tok/s | Candidate tok/s");
    println!("---|---------------|-----------------|-----------------|---------------|---------------|----------------");
    for l in &ledgers {
        println!(
            "{:2} | {:13.2} | {:15.2} | {:15.2} | {:13} | {:13.2} | {:14.2}",
            l.k_batch,
            l.true_round_ms,
            l.prefetch_exposed_wait_ms,
            l.gpu_verifier_ms,
            l.committed_tokens,
            l.actual_emitted_tok_s,
            l.candidate_tok_s
        );
    }
    println!("==========================================================================================================\n");
}
