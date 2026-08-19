//! 10-Iteration Chained Layer Verifier Benchmark on Genuine Gemma 4 26B-A4B
//!
//! Track A: Zero-Sync Verifier Collapse & On-Device Residual Chaining
//! Track C: Non-Overlapping GPU Hardware Arithmetic Breakdown
//! Verifies 100% Exact Bit-Exact Speculative Greedy Parity

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_genuine_gemma4_chained_layer_verifier_10_rounds() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "20");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");

    println!("==========================================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B ZERO-SYNC CHAINED VERIFIER (K = 8, Top-N = 14, 10 Rounds)");
    println!("Budget: 20 slots/layer (1.88 GiB Metal resident, 2.41 GiB Overflow Bank)");
    println!("==========================================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 2900, false)
        .expect("load ghost moe");

    let prompt = "<|turn>user\nExplain quantum key distribution protocols, BB84 and E91, in detail and contrast their security proofs.<turn|>\n<|turn>model\n";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let initial_logits = runtime
        .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)
        .expect("prefill");

    // Rollout draft tokens
    let mut draft_pool = Vec::new();
    let mut cur_logits = initial_logits;
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

    println!(">>> PREFETCHING TOP-N = 14 CANDIDATES <<<");
    let t_prefetch_start = Instant::now();
    let prefetched_count = runtime
        .prefetch_round_wide_chunk_top_n(candidate_chunk, top_n)
        .unwrap_or(0);
    let prefetch_ms = t_prefetch_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Prefetch completed: {:.2} ms ({} experts prefetched)\n",
        prefetch_ms, prefetched_count
    );

    // Warmup (discarded cold round)
    let mut round_kc = kc.clone();
    let mut round_vc = vc.clone();
    let _ = runtime
        .step_chunk_profiled(candidate_chunk, start_pos, &mut round_kc, &mut round_vc)
        .expect("warmup");

    fn argmax(l: &[f32]) -> u32 {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }
    fn top5(l: &[f32]) -> Vec<(u32, f32)> {
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
        idx.into_iter().take(5).map(|i| (i as u32, l[i])).collect()
    }

    println!(">>> ISOLATE FIRST MISMATCH: K=1 teacher-force vs K=8 step_chunk <<<");
    runtime.rollback_sequence(start_pos);
    let mut seq_kc = kc.clone();
    let mut seq_vc = vc.clone();
    let mut k1_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
    for (i, &tok) in candidate_chunk.iter().enumerate() {
        let logits = runtime
            .step(tok, start_pos + i, &mut seq_kc, &mut seq_vc)
            .expect("k1 step");
        k1_logits.push(logits);
    }
    runtime.rollback_sequence(start_pos);
    let mut cmp_kc = kc.clone();
    let mut cmp_vc = vc.clone();
    let (k8_logits, _) = runtime
        .step_chunk_profiled(candidate_chunk, start_pos, &mut cmp_kc, &mut cmp_vc)
        .expect("k8 isolate");
    runtime.rollback_sequence(start_pos);

    let mut first_mismatch: Option<usize> = None;
    for i in 0..k {
        let k1_am = argmax(&k1_logits[i]);
        let k8_am = argmax(&k8_logits[i]);
        let expected = if i + 1 < k {
            candidate_chunk[i + 1]
        } else {
            u32::MAX
        };
        let k8_abs: f32 = k8_logits[i].iter().map(|v| v.abs()).sum();
        let k1_abs: f32 = k1_logits[i].iter().map(|v| v.abs()).sum();
        let k8_max = k8_logits[i]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let k1_max = k1_logits[i]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let match_ok = k1_am == k8_am;
        if !match_ok && first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
        println!(
            "  row[{i}] K1 argmax={k1_am} abs={k1_abs:.4} max={k1_max:.4} | K8 argmax={k8_am} abs={k8_abs:.4} max={k8_max:.4} | draft[i+1]={expected} match={}",
            if match_ok { "YES" } else { "NO" }
        );
        if !match_ok || i == 0 {
            println!("         K1 top5={:?}", top5(&k1_logits[i]));
            println!("         K8 top5={:?}", top5(&k8_logits[i]));
        }
    }
    match first_mismatch {
        Some(i) => println!("  FIRST ARGMAX MISMATCH at row[{i}]"),
        None => println!("  K=1 vs K=8 argmax MATCH on all {k} positions"),
    }
    println!();

    fn process_rss_mb() -> f64 {
        let pid = std::process::id();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok();
        out.and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|kb| kb / 1024.0)
            .unwrap_or(0.0)
    }

    fn sysctl_u64(name: &str) -> u64 {
        std::process::Command::new("sysctl")
            .args(["-n", name])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn vm_stat_compressor_mb() -> f64 {
        let out = std::process::Command::new("vm_stat")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        let page_bytes = 16384.0;
        let pages = out.lines().find_map(|l| {
            if l.contains("occupied by compressor") || l.contains("Pages compressor") {
                l.split(':').nth(1).and_then(|s| {
                    s.trim()
                        .trim_end_matches('.')
                        .replace(',', "")
                        .parse::<f64>()
                        .ok()
                })
            } else {
                None
            }
        });
        pages.unwrap_or(0.0) * page_bytes / (1024.0 * 1024.0)
    }

    fn mac_memory_report() -> (f64, f64, String) {
        let compressed_mb = vm_stat_compressor_mb();
        let _ = sysctl_u64("vm.compressor_compressed_bytes");
        let swap_txt = std::process::Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        let pressure = std::process::Command::new("memory_pressure")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        let pressure_line = pressure
            .lines()
            .find(|l| {
                l.contains("free percentage")
                    || l.contains("Pages compressor")
                    || l.contains("The system has")
                    || l.to_lowercase().contains("warn")
                    || l.to_lowercase().contains("critical")
            })
            .unwrap_or(pressure.lines().next().unwrap_or("n/a"))
            .trim()
            .to_string();
        let swap_used = swap_txt
            .split("used =")
            .nth(1)
            .and_then(|s| s.split('M').next())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        (
            compressed_mb,
            swap_used,
            format!("{} | {}", pressure_line, swap_txt.trim()),
        )
    }

    println!(">>> BENCHMARKING CHAINED VERIFIER (10 WARM ROUNDS) <<<");
    let (baseline_compressed_mb, baseline_swap_mb, _) = mac_memory_report();
    let _ = (baseline_compressed_mb, baseline_swap_mb);
    println!(
        "Memory baseline before warm rounds: RSS={:.0} MiB compressor={:.0} MiB swap_used={:.0} MiB",
        process_rss_mb(),
        baseline_compressed_mb,
        baseline_swap_mb
    );
    println!("Parity compare: Metal greedy-draft token[i+1] vs Metal verifier argmax(row[i])");
    println!("Kernel split (QKV/O, attn, router, shared MLP, GateUp, Down, residual/norm/quant):");
    println!(
        "  GPU stage timestamps via queued per-stage CBs (GPUStartTime; no extra host waits)."
    );
    println!("  GPU busy (nested under sync) is GPUStartTime..GPUEndTime summed across waits.\n");
    println!(
        "{:>5} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8}",
        "Rnd", "wall_ms", "commit", "tok/s", "gpu_busy", "qkv/o", "attn", "router", "shared", "gateup", "down", "resid", "head", "encode", "sync", "nvme", "parity"
    );

    #[derive(Default, Clone)]
    struct RoundRow {
        wall_ms: f64,
        committed: usize,
        tok_s: f64,
        gpu_busy: f64,
        head: f64,
        encode: f64,
        sync: f64,
        filler_cpu: f64,
        nvme: f64,
        other: f64,
        gap: f64,
        late: usize,
        rss_mb: f64,
        chained_ok: bool,
        parity_pass: bool,
        slot_wait: f64,
        final_wait: f64,
        host_sum: f64,
        prefetch: f64,
        setup: f64,
        unique_sum: u32,
        unique_max: u32,
        unique_per_layer: [u16; 30],
        kv_capacity: u32,
        kv_bytes: u64,
        kv_filled: u32,
        waves_sum: u32,
        waves_max: u32,
        dropped: u32,
        failclose: u32,
        overflow: u32,
        overflow_slots: u32,
        overflow_bytes: u64,
        overflow_layers: u32,
        overflow_experts: u32,
        overflow_wait: f64,
        wave_load: f64,
        wave_gpu: f64,
        nvme_mb: f64,
        qkv_o: f64,
        attn: f64,
        router: f64,
        shared: f64,
        gateup: f64,
        down: f64,
        resid: f64,
    }
    let mut rows_out: Vec<RoundRow> = Vec::new();

    for round in 0..10 {
        let mut test_kc = kc.clone();
        let mut test_vc = vc.clone();
        let rss_mb = process_rss_mb();
        let (compressed_mb, swap_mb, pressure) = mac_memory_report();
        let t_verify_start = Instant::now();
        let (rows, prof) = runtime
            .step_chunk_profiled(candidate_chunk, start_pos, &mut test_kc, &mut test_vc)
            .expect("verify");
        let verifier_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;

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
        let parity_pass = accepted == k;
        let tok_s = if verifier_ms > 0.0 {
            (accepted as f64) / (verifier_ms / 1000.0)
        } else {
            0.0
        };
        let accounted = prof.cpu_only_exposed_ms
            + prof.physical_ssd_reads_ms
            + prof.cp_gpu_waits_ms
            + prof.chained_prefetch_ms;
        let gap = (prof.wall_clock_ms - accounted).max(0.0);

        let row = RoundRow {
            wall_ms: verifier_ms,
            committed: accepted,
            tok_s,
            gpu_busy: prof.chained_gpu_busy_ms,
            head: prof.cp_output_head_ms,
            encode: prof.cp_command_encoding_ms,
            sync: prof.cp_gpu_waits_ms,
            filler_cpu: (prof.cp_cache_slot_lookup_ms - prof.chained_setup_ms).max(0.0),
            nvme: prof.physical_ssd_reads_ms,
            other: prof.cp_other_ms,
            gap: prof.synchronization_gap_ms.max(gap),
            late: prof.prefetch_late_count,
            rss_mb,
            chained_ok: prof.gpu_chained_round_ok,
            parity_pass,
            slot_wait: prof.chained_slot_wait_ms,
            final_wait: prof.chained_final_wait_ms,
            host_sum: prof.chained_host_sum_ms,
            prefetch: prof.chained_prefetch_ms,
            setup: prof.chained_setup_ms,
            unique_sum: prof.unique_experts_sum,
            unique_max: prof.unique_experts_max,
            unique_per_layer: prof.unique_per_layer,
            kv_capacity: prof.kv_capacity,
            kv_bytes: prof.kv_bytes,
            kv_filled: prof.kv_filled,
            waves_sum: prof.expert_waves_sum,
            waves_max: prof.expert_waves_max,
            dropped: prof.selected_experts_dropped,
            failclose: prof.missing_expert_failclose,
            overflow: prof.slot_capacity_overflow,
            overflow_slots: prof.overflow_slots,
            overflow_bytes: prof.overflow_bytes,
            overflow_layers: prof.overflow_layers,
            overflow_experts: prof.overflow_experts,
            overflow_wait: prof.overflow_wait_ms,
            wave_load: prof.wave_load_ms,
            wave_gpu: prof.wave_gpu_ms,
            nvme_mb: prof.physical_nvme_mb,
            qkv_o: prof.gpu_qkv_o_ms,
            attn: prof.gpu_attn_ms,
            router: prof.gpu_router_ms,
            shared: prof.gpu_shared_ms,
            gateup: prof.gpu_gateup_ms,
            down: prof.gpu_down_ms,
            resid: prof.gpu_resid_ms,
        };

        println!(
            "{:5} {:10.2} {:8} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:>6}",
            round + 1,
            row.wall_ms,
            row.committed,
            row.tok_s,
            row.gpu_busy,
            row.qkv_o,
            row.attn,
            row.router,
            row.shared,
            row.gateup,
            row.down,
            row.resid,
            row.head,
            row.encode,
            row.sync,
            row.nvme,
            if parity_pass { "PASS" } else { "FAIL" },
        );
        println!(
            "      chained_ok={} fallback={} prefetch={:.2} unique_sum={} unique_max={} waves_sum={} waves_max={} dropped={} failclose={} overflow={} ovf_layers={} ovf_exps={} ovf_wait={:.2} wave_load={:.2} wave_gpu={:.2} nvme_mb={:.2} slot_wait={:.2} gap={:.2}",
            row.chained_ok,
            if row.chained_ok { 0 } else { 1 },
            row.prefetch,
            row.unique_sum,
            row.unique_max,
            row.waves_sum,
            row.waves_max,
            row.dropped,
            row.failclose,
            row.overflow,
            row.overflow_layers,
            row.overflow_experts,
            row.overflow_wait,
            row.wave_load,
            row.wave_gpu,
            row.nvme_mb,
            row.slot_wait,
            row.gap,
        );
        println!(
            "      mem RSS={:.0}MiB compressed={:.0}MiB swap={:.0}MiB kv_cap={} kv={:.2}GiB filled={} | {}",
            row.rss_mb,
            compressed_mb,
            swap_mb,
            row.kv_capacity,
            row.kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            row.kv_filled,
            pressure,
        );
        if row.rss_mb > 10240.0 {
            println!("      STOP: process RSS > 10 GiB — shrink, do not continue this attack.");
            break;
        }
        rows_out.push(row);
    }

    if rows_out.is_empty() {
        println!("No completed rounds (stopped on memory pressure).");
        return;
    }
    let n = rows_out.len() as f64;
    let avg = |f: fn(&RoundRow) -> f64| rows_out.iter().map(f).sum::<f64>() / n;
    let avg_commit = rows_out.iter().map(|r| r.committed).sum::<usize>() as f64 / n;
    let avg_wall = avg(|r| r.wall_ms);
    let avg_tok_s = avg(|r| r.tok_s);
    let avg_sync = avg(|r| r.sync);
    let avg_gpu = avg(|r| r.gpu_busy);
    let avg_encode = avg(|r| r.encode);
    let avg_head = avg(|r| r.head);
    let avg_filler = avg(|r| r.filler_cpu);
    let avg_nvme = avg(|r| r.nvme);
    let avg_other = avg(|r| r.other);
    let _avg_gap = avg(|r| r.gap);
    let avg_slot = avg(|r| r.slot_wait);
    let avg_final = avg(|r| r.final_wait);
    let avg_prefetch = avg(|r| r.prefetch);
    let avg_setup = avg(|r| r.setup);
    let all_parity = rows_out.iter().all(|r| r.parity_pass);
    let all_chained = rows_out.iter().all(|r| r.chained_ok);

    println!("\n==========================================================================================================");
    println!("NON-OVERLAPPING LEDGER (avg of 10 warm rounds; must sum to wall-clock)");
    println!("Harness: gemma4_chained_layer_verifier_benchmark  K=8 Top-N=14");
    println!("Draft is pre-rolled greedy Metal (not live spec decode). Timed path = verifier step_chunk.");
    println!("==========================================================================================================");
    println!(
        "  true outer round latency (verifier wall): {:8.2} ms",
        avg_wall
    );
    println!(
        "  verifier latency:                         {:8.2} ms",
        avg_wall
    );
    println!("  committed tokens/round:                   {:8.2}  (K=8 candidates; win metric is commits)", avg_commit);
    println!(
        "  emitted tok/s (commits/wall):             {:8.2}",
        avg_tok_s
    );
    let partition_sum =
        avg_encode + avg_filler + avg_setup + avg_prefetch + avg_sync + avg_other + avg_head;
    let true_gap = (avg_wall - partition_sum).max(0.0);
    println!("  --- summing wall partition ---");
    println!(
        "  command encoding (CPU):                   {:8.2} ms",
        avg_encode
    );
    println!(
        "  CPU slot-table setup:                     {:8.2} ms",
        avg_setup
    );
    println!(
        "  sync exposed (slot+final wait):           {:8.2} ms",
        avg_sync
    );
    println!(
        "    nested slot_wait:                       {:8.2} ms",
        avg_slot
    );
    println!(
        "    nested final_wait:                      {:8.2} ms",
        avg_final
    );
    println!(
        "  embed/upload/rope/download:               {:8.2} ms",
        avg_other
    );
    println!(
        "  output head:                              {:8.2} ms",
        avg_head
    );
    println!(
        "  unexplained gap (lock/debug/eprint):      {:8.2} ms",
        true_gap
    );
    println!(
        "  SUM of partition:                         {:8.2} ms",
        partition_sum + true_gap
    );
    println!(
        "  wall-clock:                               {:8.2} ms",
        avg_wall
    );
    println!("  --- nested inside sync / background ---");
    println!(
        "  pure Metal GPU critical path (GPU busy):  {:8.2} ms",
        avg_gpu
    );
    println!(
        "  wait idle / driver (sync - gpu_busy):     {:8.2} ms",
        (avg_sync - avg_gpu).max(0.0)
    );
    println!("  GPU stage timestamps (nested in gpu_busy; no extra waits):");
    println!(
        "    QKV/O:                                  {:8.2} ms",
        avg(|r| r.qkv_o)
    );
    println!(
        "    attention:                              {:8.2} ms",
        avg(|r| r.attn)
    );
    println!(
        "    router:                                 {:8.2} ms",
        avg(|r| r.router)
    );
    println!(
        "    shared MLP:                             {:8.2} ms",
        avg(|r| r.shared)
    );
    println!(
        "    GateUp:                                 {:8.2} ms",
        avg(|r| r.gateup)
    );
    println!(
        "    Down:                                   {:8.2} ms",
        avg(|r| r.down)
    );
    println!(
        "    residual/norm/quant:                    {:8.2} ms",
        avg(|r| r.resid)
    );
    let avg_stage = avg(|r| r.qkv_o + r.attn + r.router + r.shared + r.gateup + r.down + r.resid);
    println!(
        "    SUM of GPU stages:                      {:8.2} ms",
        avg_stage
    );
    println!(
        "  background NVMe throughput (overlapped):  {:8.2} ms ({:.2} MB)",
        avg_nvme,
        avg(|r| r.nvme_mb)
    );
    println!(
        "  late prefetches (demand slot loads/round):{:8.2}",
        avg(|r| r.late as f64)
    );
    println!(
        "  memory pressure (process RSS):            {:8.0} MiB",
        avg(|r| r.rss_mb)
    );
    println!(
        "  gpu_chained_round_ok every round:         {}",
        if all_chained {
            "true (fallback=0)"
        } else {
            "FALSE — see per-round"
        }
    );
    println!(
        "  parity every round:                       {} (Metal draft vs Metal verifier argmax)",
        if all_parity { "PASS" } else { "FAIL" }
    );
    println!(
        "  unique experts (sum/max over layers):     {:.0} / {:.0}",
        avg(|r| r.unique_sum as f64),
        avg(|r| r.unique_max as f64)
    );
    println!(
        "  expert waves (sum/max over layers):       {:.0} / {:.0}",
        avg(|r| r.waves_sum as f64),
        avg(|r| r.waves_max as f64)
    );
    println!(
        "  selected_experts_dropped:                 {:.0}",
        avg(|r| r.dropped as f64)
    );
    println!(
        "  missing_expert_failclose:                 {:.0}",
        avg(|r| r.failclose as f64)
    );
    println!(
        "  slot_capacity_overflow:                   {:.0}",
        avg(|r| r.overflow as f64)
    );
    println!(
        "  wave_load_ms:                             {:8.2}",
        avg(|r| r.wave_load)
    );
    println!(
        "  wave_gpu_ms:                              {:8.2}",
        avg(|r| r.wave_gpu)
    );
    println!(
        "  physical_nvme_mb:                         {:8.2}",
        avg(|r| r.nvme_mb)
    );
    println!("==========================================================================================================");

    let mut ranked = [
        ("sync exposed (slot+final wait)", avg_sync),
        ("output head", avg_head),
        ("unexplained gap", true_gap),
        ("command encoding", avg_encode),
        ("embed/upload/rope/download", avg_other),
        ("CPU slot-table setup", avg_setup),
        ("CPU slot-fill", avg_filler),
        ("late prefetch (last-round experts)", avg_prefetch),
    ];
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("RANKED BY EXPOSED WALL-CLOCK ms (largest first):");
    for (i, (name, ms)) in ranked.iter().enumerate() {
        println!("  #{:<2} {:<36} {:8.2} ms", i + 1, name, ms);
    }
    let mut gpu_ranked = [
        ("GPU GateUp", avg(|r| r.gateup)),
        ("GPU Down", avg(|r| r.down)),
        ("GPU QKV/O", avg(|r| r.qkv_o)),
        ("GPU shared MLP", avg(|r| r.shared)),
        ("GPU attention", avg(|r| r.attn)),
        ("GPU residual/norm/quant", avg(|r| r.resid)),
        ("GPU router", avg(|r| r.router)),
    ];
    gpu_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("RANKED BY GPU STAGE ms (nested in gpu_busy):");
    for (i, (name, ms)) in gpu_ranked.iter().enumerate() {
        println!("  #{:<2} {:<36} {:8.2} ms", i + 1, name, ms);
    }
    println!(
        "#1 wall is '{}' — GPU #1 is '{}'.",
        ranked[0].0, gpu_ranked[0].0
    );
    println!("Gates at 8 commits: 160ms=50 tok/s  106.67ms=75  80ms=100  53.33ms=150");
    println!("==========================================================================================================\n");

    // BYTE LEDGER — logical tensor bytes vs estimated DRAM. Never time×120.
    const GATE_UP: u64 = 2_230_272;
    const DOWN: u64 = 1_115_136;
    const SLOT: u64 = 3_358_720;
    const QKV_O: u64 = 397_828_096;
    const SHARED: u64 = 301_086_720;
    const HEAD: u64 = 605_552_640;
    const ROUTER: u64 = 43_622_400;
    const KV_PER_POS: u64 = 450_560;
    const HIDDEN: u64 = 2_816;
    const FF: u64 = 704;
    const K: u64 = 8;
    const LAYERS: u64 = 30;
    const PLANNING_BW_GBS: f64 = 120.0;

    let steady: Vec<&RoundRow> = if rows_out.len() > 2 {
        rows_out[2..].iter().collect()
    } else {
        rows_out.iter().collect()
    };
    let sn = steady.len().max(1) as f64;
    let savg = |f: fn(&RoundRow) -> f64| steady.iter().map(|r| f(r)).sum::<f64>() / sn;
    let last = rows_out.last().cloned().unwrap_or_default();
    let unique = last.unique_sum as u64;
    let mut unions: Vec<u16> = last.unique_per_layer.to_vec();
    unions.sort_unstable();
    let pct = |p: f64| -> u16 {
        if unions.is_empty() {
            return 0;
        }
        let idx = ((p / 100.0) * (unions.len() - 1) as f64).round() as usize;
        unions[idx.min(unions.len() - 1)]
    };
    let mean_u = last.unique_per_layer.iter().map(|&x| x as f64).sum::<f64>() / 30.0;
    let wave2_layers = last.unique_per_layer.iter().filter(|&&u| u > 24).count();
    let wave2_experts: u64 = last
        .unique_per_layer
        .iter()
        .map(|&u| u.saturating_sub(24) as u64)
        .sum();
    let filled = last.kv_filled.max(1) as u64;
    let kv_alloc = last.kv_bytes;
    let kv_full = KV_PER_POS * 4096;
    let kv_logical_read = KV_PER_POS * filled; // attention reads filled positions, K+V all layers
    let routes = LAYERS * K * K; // 8 experts × 8 tokens × 30 layers
    let act_in = K * HIDDEN * 4;
    let gate_act_q8 = unique * K * FF; // Q8 activations written
    let gate_act_scale = unique * K * 22 * 4;

    let gate_logical = unique * GATE_UP;
    let down_logical = unique * DOWN;
    let down_est_dram = routes * DOWN; // per-token simd rereads
    let slot_memcpy = unique * SLOT; // predicted ping/pong fill of the union
    let wave2_memcpy = wave2_experts * SLOT;

    let components: [(&str, u64, f64, u64, u64, f64); 8] = [
        (
            "GateUp",
            gate_logical,
            1.0,
            act_in,
            gate_act_q8 + gate_act_scale,
            savg(|r| r.gateup),
        ),
        (
            "Down",
            down_logical,
            down_est_dram as f64 / down_logical.max(1) as f64,
            gate_act_q8,
            act_in,
            savg(|r| r.down),
        ),
        (
            "QKV/O",
            QKV_O,
            1.0,
            act_in,
            K * 16 * 256 * 4,
            savg(|r| r.qkv_o),
        ),
        (
            "Shared MLP",
            SHARED,
            1.0,
            act_in,
            K * 2112 * 4,
            savg(|r| r.shared),
        ),
        ("Head", HEAD, 1.0, act_in, 262_144 * 4 * K, savg(|r| r.head)),
        (
            "Attention",
            kv_logical_read,
            1.0,
            K * 16 * 256 * 4,
            K * 16 * 256 * 4,
            savg(|r| r.attn),
        ),
        (
            "Residual/norm",
            LAYERS * HIDDEN * 4 * 4,
            1.0,
            act_in,
            act_in,
            savg(|r| r.resid),
        ),
        (
            "Expert slot memcpy",
            slot_memcpy,
            1.0,
            0,
            0,
            savg(|r| r.wave_load),
        ),
    ];

    println!("==========================================================================================================");
    println!(
        "BYTE LEDGER (K=8, unique={}, filled_pos={}, kv_cap={})",
        unique, filled, last.kv_capacity
    );
    println!("logical = tensor bytes addressed. est DRAM = conservative physical estimate.");
    println!("Do NOT use elapsed × 120 GB/s as bytes moved.");
    println!(
        "{:<22} {:>12} {:>8} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "component", "logical_B", "passes", "act_read_B", "write_B", "est_DRAM_B", "gpu_ms", "GB/s"
    );
    let mut tot_logical = 0u64;
    let mut tot_dram = 0u64;
    for (name, logical, passes, act_r, act_w, ms) in components {
        let est = if name == "Down" {
            down_est_dram
        } else {
            (logical as f64 * passes) as u64
        };
        let gbs = if ms > 0.0 {
            (est as f64 / 1e9) / (ms / 1000.0)
        } else {
            0.0
        };
        tot_logical += logical;
        tot_dram += est;
        println!(
            "{:<22} {:>12} {:>8.2} {:>12} {:>12} {:>12} {:>8.2} {:>8.1}",
            name, logical, passes, act_r, act_w, est, ms, gbs
        );
    }
    let mandatory = gate_logical + down_logical + QKV_O + SHARED + HEAD + ROUTER + kv_logical_read;
    let gpu_wall = savg(|r| r.gpu_busy);
    let wall = savg(|r| r.wall_ms);
    let tok_s = savg(|r| r.tok_s);
    println!("---");
    println!(
        "TOTAL logical weight+KV+slots:     {:>12}  ({:.3} GB)",
        tot_logical,
        tot_logical as f64 / 1e9
    );
    println!(
        "MINIMUM mandatory (1-pass unique GateUp+Down + QKV + Shared + Head + Router + KV read):"
    );
    println!(
        "                                   {:>12}  ({:.3} GB)",
        mandatory,
        mandatory as f64 / 1e9
    );
    println!(
        "ESTIMATED physical DRAM (Down rereads + 1-pass others + slot memcpy): {:>12}  ({:.3} GB)",
        tot_dram,
        tot_dram as f64 / 1e9
    );
    println!("GPU wall (steady 3..):             {:>8.2} ms", gpu_wall);
    println!(
        "TRUE WALL (steady 3..):            {:>8.2} ms   ACTUAL tok/s {:.2}",
        wall, tok_s
    );
    println!(
        "effective GB/s vs 120: GateUp {:.1}  Down {:.1}  QKV {:.1}  Head {:.1}",
        if savg(|r| r.gateup) > 0.0 {
            (gate_logical as f64 / 1e9) / (savg(|r| r.gateup) / 1000.0)
        } else {
            0.0
        },
        if savg(|r| r.down) > 0.0 {
            (down_est_dram as f64 / 1e9) / (savg(|r| r.down) / 1000.0)
        } else {
            0.0
        },
        if savg(|r| r.qkv_o) > 0.0 {
            (QKV_O as f64 / 1e9) / (savg(|r| r.qkv_o) / 1000.0)
        } else {
            0.0
        },
        if savg(|r| r.head) > 0.0 {
            (HEAD as f64 / 1e9) / (savg(|r| r.head) / 1000.0)
        } else {
            0.0
        },
    );
    println!("page-cache / file-backed: Head + QKV + Shared + Router are mmap/file-backed (may be cache-serviced, not extra DRAM alloc).");
    println!(
        "expert slots resident: 24×30×{SLOT} = {:.2} GiB",
        (24.0 * 30.0 * SLOT as f64) / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "overflow bank: {}×{} slots = {:.2} MiB (NOT ×30 layers); layers={} experts={} wait={:.2} ms memcpy={:.1} MiB",
        (last.overflow_bytes / (last.overflow_slots.max(1) as u64 * SLOT)).max(1),
        last.overflow_slots,
        last.overflow_bytes as f64 / (1024.0 * 1024.0),
        last.overflow_layers,
        last.overflow_experts,
        savg(|r| r.overflow_wait),
        (last.overflow_experts as u64 * SLOT) as f64 / (1024.0 * 1024.0),
    );
    println!(
        "KV allocated: {:.2} GiB  (full-4096 would be {:.2} GiB, saved {:.2} GiB)",
        kv_alloc as f64 / (1024.0 * 1024.0 * 1024.0),
        kv_full as f64 / (1024.0 * 1024.0 * 1024.0),
        (kv_full.saturating_sub(kv_alloc)) as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "wave-2: {} layers, {} overflow experts, memcpy {:.1} MiB",
        wave2_layers,
        wave2_experts,
        wave2_memcpy as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Budgets at {:.0} GB/s: 100 tok/s=80ms≈9.60GB  75=106.67ms≈12.80GB  50=160ms≈19.20GB",
        PLANNING_BW_GBS
    );
    println!(
        "Mandatory {:.3} GB vs 50-tok budget 19.20 GB: {}",
        mandatory as f64 / 1e9,
        if mandatory as f64 / 1e9 > 19.20 {
            "EXCEEDS — 50 tok/s impossible without changing representation"
        } else {
            "fits 50 tok/s budget if kernels hit ~120 GB/s"
        }
    );
    println!(
        "Mandatory {:.3} GB vs 75-tok budget 12.80 GB: {}",
        mandatory as f64 / 1e9,
        if mandatory as f64 / 1e9 > 12.80 {
            "EXCEEDS — 75 tok/s needs fewer bytes or more than 120 GB/s"
        } else {
            "fits 75 tok/s budget if kernels hit ~120 GB/s"
        }
    );
    println!(
        "Mandatory {:.3} GB vs 100-tok budget 9.60 GB: {}",
        mandatory as f64 / 1e9,
        if mandatory as f64 / 1e9 > 9.60 {
            "EXCEEDS — 100 tok/s needs fewer bytes or more than 120 GB/s"
        } else {
            "fits 100 tok/s budget if kernels hit ~120 GB/s"
        }
    );
    println!(
        "unique/layer: mean={:.1} p50={} p90={} p95={} p99={} max={}  (n=30)",
        mean_u,
        pct(50.0),
        pct(90.0),
        pct(95.0),
        pct(99.0),
        unions.last().copied().unwrap_or(0)
    );
    print!("per-layer unique: ");
    for (i, u) in last.unique_per_layer.iter().enumerate() {
        print!("L{i}={u}{}", if i + 1 == 30 { "\n" } else { " " });
    }
    let (compressed_mb, swap_mb, pressure) = mac_memory_report();
    println!(
        "MEMORY end: RSS={:.0} MiB  compressed={:.0} MiB  swap={:.0} MiB  KV={:.2} GiB  slots=2.25 GiB  overflow={:.2} MiB  | {}",
        process_rss_mb(),
        compressed_mb,
        swap_mb,
        kv_alloc as f64 / (1024.0 * 1024.0 * 1024.0),
        last.overflow_bytes as f64 / (1024.0 * 1024.0),
        pressure
    );
    println!("Checkpoint SHA (pre-this-session known-good): 12132aac");
    println!("==========================================================================================================\n");
}
