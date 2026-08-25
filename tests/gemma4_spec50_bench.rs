//! Gemma 4 26B emitted-throughput bench: greedy K=1 vs lossless speculative decode.
//!
//! Measures what a user sees — emitted tokens per wall second after prefill — on
//! several workloads, and asserts the speculative stream is token-identical to the
//! greedy stream. Prints per-token latency percentiles for the K=1 lane and
//! per-round acceptance for the speculative lane.
//!
//! Env:
//!   CAMELID_GEMMA4_26B_GGUF / CAMELID_GEMMA4_26B_CGHOST   model pair
//!   SPEC50_MAX_NEW        tokens per workload (default 64)
//!   SPEC50_K              comma list of draft widths (default "4")
//!   SPEC50_WORKLOADS      comma list of workload keys (default all)
//!   SPEC50_SKIP_GREEDY    skip the greedy reference (no parity check)
//!   SPEC50_CACHE_MIB      host expert cache MiB (default 2900)
//!   SPEC50_GREEDY_LANE    chained (default) | head — lane for the greedy reference
//!   SPEC50_MIN_MATCH      n-gram min match length for the drafter (default 3)
//!   SPEC50_ADAPTIVE       set to enable adaptive draft width

mod support;

use camelid::gemma4_runtime::{gemma4_stop_token_ids, Gemma4Runtime};
use std::{path::PathBuf, time::Instant};

/// Exposed-idle decomposition counters read by the [k-idle] report line.
/// Times are in microseconds; the tail entries are event counts. The report
/// indexes this array by position, so order is load-bearing.
const IDLE_STATS: [(&str, &std::sync::atomic::AtomicU64); 21] = [
    ("route_us", &camelid::metal::SPEC_FILLER_ROUTE_US),
    ("fill_us", &camelid::metal::SPEC_FILLER_FILL_US),
    ("copy_us", &camelid::metal::SPEC_FILL_COPY_US),
    ("encode_us", &camelid::metal::SPEC_ENCODE_US),
    ("pre_encode_us", &camelid::metal::SPEC_PRE_ENCODE_US),
    ("slot_wait_us", &camelid::metal::SPEC_SLOT_WAIT_US),
    ("wave_load_us", &camelid::metal::SPEC_WAVE_LOAD_US),
    ("final_wait_us", &camelid::metal::SPEC_FINAL_WAIT_US),
    ("other_host_us", &camelid::metal::SPEC_HOST_OTHER_US),
    ("boundary_us", &camelid::metal::SPEC_BOUNDARY_US),
    ("draft_us", &camelid::metal::SPEC_DRAFT_US),
    ("truncate_us", &camelid::metal::SPEC_TRUNCATE_US),
    ("embed_us", &camelid::metal::SPEC_EMBED_US),
    ("slot_hits", &camelid::metal::SPEC_SLOT_HITS),
    ("slot_misses", &camelid::metal::SPEC_SLOT_MISSES),
    ("slot_evictions", &camelid::metal::SPEC_SLOT_EVICTIONS),
    ("prev_union_hits", &camelid::metal::SPEC_PREV_UNION_HITS),
    ("prev_union_total", &camelid::metal::SPEC_PREV_UNION_TOTAL),
    (
        "resident_start_hits",
        &camelid::metal::SPEC_RESIDENT_START_HITS,
    ),
    ("overlap_layers", &camelid::metal::SPEC_OVERLAP_LAYERS),
    ("overlap_fallbacks", &camelid::metal::SPEC_OVERLAP_FALLBACKS),
];

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

const PARA: &str = "The ghost lane keeps a directory of resident experts per layer and pages the rest in from the \
sparse file on demand. Each decode step routes eight experts per layer, so the union over a short chunk of \
tokens grows roughly linearly with the chunk length until the hot set saturates. Because decode is bandwidth \
bound, reading every weight once per round and amortising it over several accepted tokens is the only way \
past the single-token wall on a sixteen gigabyte machine.";

fn workloads() -> Vec<(&'static str, String)> {
    vec![
        (
            "code-refactor",
            "<|turn>user\nRewrite this struct with an added expires_at field:\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n}\n<turn|>\n<|turn>model\n".to_string(),
        ),
        (
            "json-yaml",
            "<|turn>user\nConvert this configuration payload to YAML:\n{\"cluster_id\": \"prod-1\", \"min_replicas\": 4, \"max_replicas\": 32, \"enabled\": true}\n<turn|>\n<|turn>model\n".to_string(),
        ),
        (
            "prose",
            "<|turn>user\nExplain in three short paragraphs how a hash map works and when it degrades.\n<turn|>\n<|turn>model\n".to_string(),
        ),
        (
            "copy",
            format!("<|turn>user\nRepeat the following paragraph exactly, word for word:\n\n{PARA}\n<turn|>\n<|turn>model\n"),
        ),
        // The realistic agent-edit shape: nearly all of the output is a literal
        // span of the input, which is exactly what a prompt-lookup drafter can
        // predict. This is the workload that decides whether wide waves pay.
        (
            "code-edit",
            "<|turn>user\nAdd a `pub expires_at: u64,` field at the end of this struct and output the COMPLETE struct definition again, unchanged otherwise, with no explanation:\n\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n    pub created_at: u64,\n    pub last_hit: u64,\n}\n<turn|>\n<|turn>model\n".to_string(),
        ),
    ]
}

#[test]
fn gemma4_spec50_bench() {
    let model_path = std::env::var_os("CAMELID_GEMMA4_26B_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf"));
    let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost"));
    if !model_path.exists() || !cghost_path.exists() {
        eprintln!("model pair missing; skipping");
        return;
    }

    let set_default = |k: &str, v: &str| {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    };
    set_default("CAMELID_GHOST_ALLOW_LEGACY_SPARSE", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_TURBO", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_COMMON", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_CONTEXT", "1024");
    set_default("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT", "1");
    set_default("CAMELID_GEMMA4_GHOST_READ_THREADS", "4");
    set_default("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "88");
    set_default("CAMELID_GEMMA4_SPEC_TIMING", "1");
    if let Ok(v) = std::env::var("SPEC50_MIN_MATCH") {
        std::env::set_var("CAMELID_GEMMA4_SPEC_MIN_MATCH", v);
    }
    if std::env::var_os("SPEC50_ADAPTIVE").is_some() {
        std::env::set_var("CAMELID_GEMMA4_SPEC_ADAPTIVE", "1");
    }

    let max_new: usize = env_or("SPEC50_MAX_NEW", "64").parse().unwrap();
    let ks: Vec<usize> = env_or("SPEC50_K", "4")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let cache_mib: usize = env_or("SPEC50_CACHE_MIB", "2900").parse().unwrap();
    let skip_greedy = std::env::var_os("SPEC50_SKIP_GREEDY").is_some();
    // Greedy reference lane: "chained" (default; the same lane the speculative
    // verifier uses, so the parity assert is lane-internal) or "head".
    let greedy_head_lane = env_or("SPEC50_GREEDY_LANE", "chained").eq_ignore_ascii_case("head");
    let selected: Option<Vec<String>> = std::env::var("SPEC50_WORKLOADS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());

    let t_load = Instant::now();
    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, cache_mib, false)
        .expect("load ghost moe");
    eprintln!("[spec50] loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    let eot = gemma4_stop_token_ids(runtime.tokenizer());
    let n_layers = 30usize;

    // Warm-up: one short generation so slots hold a realistic hot set.
    {
        let prompt = "<|turn>user\nSay hello and name three colours.\n<turn|>\n<|turn>model\n";
        let toks = runtime.tokenizer().encode(prompt, true, true).unwrap();
        let mut kc = vec![Vec::new(); n_layers];
        let mut vc = vec![Vec::new(); n_layers];
        let t = Instant::now();
        let mut logits = runtime.prefill_tokens(&toks, &mut kc, &mut vc, 31).unwrap();
        let mut pos = toks.len();
        for _ in 0..24 {
            let tok = argmax(&logits);
            if eot.contains(&tok) {
                break;
            }
            logits = runtime.step(tok, pos, &mut kc, &mut vc).unwrap();
            pos += 1;
        }
        eprintln!("[spec50] warm-up done in {:.1}s", t.elapsed().as_secs_f64());
    }

    let mut summary: Vec<String> = Vec::new();
    for (name, prompt) in workloads() {
        if let Some(sel) = &selected {
            if !sel.iter().any(|s| s == name) {
                continue;
            }
        }
        println!("\n================ WORKLOAD {name} ================");
        let prompt_tokens = runtime.tokenizer().encode(&prompt, true, true).unwrap();

        // Greedy K=1 reference.
        let mut greedy_tokens: Vec<u32> = Vec::new();
        let mut greedy_tok_s = 0.0;
        let mut greedy_med_ms = 0.0;
        if !skip_greedy {
            let mut kc = vec![Vec::new(); n_layers];
            let mut vc = vec![Vec::new(); n_layers];
            let t0 = Instant::now();
            let mut logits = runtime
                .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))
                .unwrap();
            let prefill_s = t0.elapsed().as_secs_f64();
            let mut pos = prompt_tokens.len();
            let mut per_tok_ms: Vec<f64> = Vec::new();
            let t1 = Instant::now();
            loop {
                let tok = argmax(&logits);
                if eot.contains(&tok) {
                    break;
                }
                greedy_tokens.push(tok);
                if greedy_tokens.len() >= max_new {
                    break;
                }
                let ts = Instant::now();
                logits = if greedy_head_lane {
                    runtime.step(tok, pos, &mut kc, &mut vc).unwrap()
                } else {
                    runtime
                        .step_chunk(&[tok], pos, &mut kc, &mut vc)
                        .unwrap()
                        .pop()
                        .unwrap()
                };
                per_tok_ms.push(ts.elapsed().as_secs_f64() * 1000.0);
                pos += 1;
            }
            let decode_s = t1.elapsed().as_secs_f64();
            greedy_tok_s = greedy_tokens.len() as f64 / decode_s;
            per_tok_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            greedy_med_ms = pct(&per_tok_ms, 0.5);
            println!(
                "[greedy:{}] prefill {} tok {:.2}s | {} tokens in {:.2}s = {:.2} tok/s | step ms min {:.1} p50 {:.1} p90 {:.1} max {:.1}",
                if greedy_head_lane { "head" } else { "chained" },
                prompt_tokens.len(),
                prefill_s,
                greedy_tokens.len(),
                decode_s,
                greedy_tok_s,
                pct(&per_tok_ms, 0.0),
                greedy_med_ms,
                pct(&per_tok_ms, 0.9),
                pct(&per_tok_ms, 1.0)
            );
            println!(
                "[greedy] text: {:?}",
                runtime
                    .tokenizer()
                    .decode(&greedy_tokens, true)
                    .unwrap_or_default()
            );
        }

        for &k in &ks {
            std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", k.to_string());
            let mut kc = vec![Vec::new(); n_layers];
            let mut vc = vec![Vec::new(); n_layers];
            let t0 = Instant::now();
            let logits = runtime
                .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))
                .unwrap();
            let prefill_s = t0.elapsed().as_secs_f64();
            use std::sync::atomic::Ordering::Relaxed;
            let hw0 = camelid::metal::HEAD_WALL_US.load(Relaxed);
            let ue0 = camelid::metal::SPEC_EXPERT_UNIQUE_SUM.load(Relaxed);
            let cg0 = camelid::metal::SPEC_CHAINED_GPU_US.load(Relaxed);
            let cr0 = camelid::metal::SPEC_CHAINED_ROUNDS.load(Relaxed);
            let hist0: Vec<u32> = camelid::metal::SPEC_UNION_HIST
                .iter()
                .map(|a| a.load(Relaxed))
                .collect();
            let hg0 = camelid::metal::HEAD_GPU_US.load(Relaxed);
            let hc0 = camelid::metal::HEAD_CALLS.load(Relaxed);
            let hr0 = camelid::metal::HEAD_ROWS.load(Relaxed);
            let rounds0 =
                camelid::metal::SPEC_VERIFY_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
            let acc0 =
                camelid::metal::SPEC_ACCEPTED_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
            // Exposed-idle decomposition baselines.
            let idle0: Vec<u64> = IDLE_STATS.iter().map(|(_, a)| a.load(Relaxed)).collect();
            // Eviction-cause baselines: cold misses + re-miss distance histogram.
            let cold0 = camelid::metal::SPEC_MISS_COLD.load(Relaxed);
            let remiss0: Vec<u64> = camelid::metal::SPEC_REMISS_DIST_HIST
                .iter()
                .map(|a| a.load(Relaxed))
                .collect();
            // Victim-ring baselines.
            let victim0 = [
                camelid::metal::SPEC_VICTIM_HITS.load(Relaxed),
                camelid::metal::SPEC_VICTIM_FILL_US.load(Relaxed),
                camelid::metal::SPEC_VICTIM_SALVAGE_COPIES.load(Relaxed),
                camelid::metal::SPEC_VICTIM_SALVAGE_US.load(Relaxed),
                camelid::metal::SPEC_VICTIM_VERIFY_FAILS.load(Relaxed),
            ];
            let t1 = Instant::now();
            let spec_tokens = runtime
                .spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)
                .unwrap();
            let decode_s = t1.elapsed().as_secs_f64();
            let rounds = camelid::metal::SPEC_VERIFY_ROUNDS
                .load(std::sync::atomic::Ordering::Relaxed)
                - rounds0;
            let acc = camelid::metal::SPEC_ACCEPTED_TOKENS
                .load(std::sync::atomic::Ordering::Relaxed)
                - acc0;
            let head_wall_ms = (camelid::metal::HEAD_WALL_US.load(Relaxed) - hw0) as f64 / 1000.0;
            let head_gpu_ms = (camelid::metal::HEAD_GPU_US.load(Relaxed) - hg0) as f64 / 1000.0;
            let head_calls = camelid::metal::HEAD_CALLS.load(Relaxed) - hc0;
            let head_rows = camelid::metal::HEAD_ROWS.load(Relaxed) - hr0;
            let spec_tok_s = spec_tokens.len() as f64 / decode_s;
            let alpha = acc as f64 / rounds.max(1) as f64;
            let exact = skip_greedy || spec_tokens == greedy_tokens;
            println!(
                "[spec K={k}] prefill {:.2}s | {} tokens in {:.2}s = {:.2} tok/s | rounds {} alpha {:.2} | {:.1} ms/round | exact={} | vs greedy {:.2}x",
                prefill_s,
                spec_tokens.len(),
                decode_s,
                spec_tok_s,
                rounds,
                alpha,
                decode_s * 1000.0 / rounds.max(1) as f64,
                exact,
                if greedy_tok_s > 0.0 { spec_tok_s / greedy_tok_s } else { 0.0 }
            );
            // Per-K sweep report: everything needed to price a wider wave.
            {
                let uniq = camelid::metal::SPEC_EXPERT_UNIQUE_SUM.load(Relaxed) - ue0;
                let gpu_ms =
                    (camelid::metal::SPEC_CHAINED_GPU_US.load(Relaxed) - cg0) as f64 / 1000.0;
                let cr = camelid::metal::SPEC_CHAINED_ROUNDS.load(Relaxed) - cr0;
                // Physical traffic. Experts: unique loads x record bytes. Dense
                // core (attention + shared MLP + router) and the tied head are
                // read once per chained round / head call respectively.
                const EXPERT_RECORD_MB: f64 = 3.345408;
                const DENSE_CORE_MB: f64 = 742.4;
                const HEAD_MB: f64 = 605.5;
                let expert_mb = uniq as f64 * EXPERT_RECORD_MB;
                let dense_mb = cr as f64 * DENSE_CORE_MB + head_calls as f64 * HEAD_MB;
                let total_mb = expert_mb + dense_mb;
                let decode_ms = decode_s * 1000.0;
                let idle_ms = (decode_ms - gpu_ms - head_wall_ms).max(0.0);
                // Union distribution across layer-rounds, from the histogram delta.
                let du: Vec<(usize, u32)> = camelid::metal::SPEC_UNION_HIST
                    .iter()
                    .enumerate()
                    .map(|(u, a)| (u, a.load(Relaxed) - hist0[u]))
                    .filter(|&(_, n)| n > 0)
                    .collect();
                let n_lr: u64 = du.iter().map(|&(_, n)| n as u64).sum();
                // `du` is built from an ascending enumerate, so it is already
                // sorted by union size; the closure below relies on that order.
                let pct = |q: f64| -> usize {
                    let target = (q * n_lr as f64) as u64;
                    let mut seen = 0u64;
                    for &(u, n) in &du {
                        seen += n as u64;
                        if seen > target {
                            return u;
                        }
                    }
                    du.last().map(|&(u, _)| u).unwrap_or(0)
                };
                {
                    let d: Vec<u64> = IDLE_STATS
                        .iter()
                        .zip(&idle0)
                        .map(|((_, a), b)| a.load(Relaxed) - b)
                        .collect();
                    let ms = |i: usize| d[i] as f64 / 1000.0;
                    // Order must match IDLE_STATS below.
                    println!(
                        "[k-idle K={k}] route={:.0}ms fill={:.0}ms (copy={:.0}ms) encode={:.0}ms \
                         pre_encode={:.0}ms slot_wait={:.0}ms wave_load={:.0}ms final_wait={:.0}ms \
                         other_host={:.0}ms boundary={:.0}ms draft={:.0}ms truncate={:.0}ms \
                         embed={:.0}ms | slots hit={} miss={} evict={} | union_vs_prev={}/{} \
                         ({:.0}%) resident_at_start={}/{} ({:.0}%) | overlap_layers={} fallbacks={}",
                        ms(0),
                        ms(1),
                        ms(2),
                        ms(3),
                        ms(4),
                        ms(5),
                        ms(6),
                        ms(7),
                        ms(8),
                        ms(9),
                        ms(10),
                        ms(11),
                        ms(12),
                        d[13],
                        d[14],
                        d[15],
                        d[16],
                        d[17],
                        100.0 * d[16] as f64 / (d[17].max(1)) as f64,
                        d[18],
                        d[17],
                        100.0 * d[18] as f64 / (d[17].max(1)) as f64,
                        d[19],
                        d[20],
                    );
                    // Eviction cause: are misses first touches (cold) or
                    // re-misses of recently evicted experts (churn)? Low
                    // distance buckets dominating = the policy evicts experts
                    // the router immediately re-routes.
                    let rd: Vec<u64> = camelid::metal::SPEC_REMISS_DIST_HIST
                        .iter()
                        .zip(&remiss0)
                        .map(|(a, b)| a.load(Relaxed) - b)
                        .collect();
                    println!(
                        "[k-remiss K={k}] cold={} | re-miss distance (rounds): \
                         d0={} d1={} d2={} d3={} d4={} d5_8={} d9_16={} d17+={}",
                        camelid::metal::SPEC_MISS_COLD.load(Relaxed) - cold0,
                        rd[0],
                        rd[1],
                        rd[2],
                        rd[3],
                        rd[4],
                        rd[5],
                        rd[6],
                        rd[7],
                    );
                    // Victim ring: misses served by host memcpy instead of
                    // pread, plus what salvaging cost. verify_fails must be 0.
                    println!(
                        "[k-victim K={k}] ring_hits={} fill={:.0}ms salvage_copies={} \
                         salvage={:.0}ms verify_fails={}",
                        camelid::metal::SPEC_VICTIM_HITS.load(Relaxed) - victim0[0],
                        (camelid::metal::SPEC_VICTIM_FILL_US.load(Relaxed) - victim0[1]) as f64
                            / 1000.0,
                        camelid::metal::SPEC_VICTIM_SALVAGE_COPIES.load(Relaxed) - victim0[2],
                        (camelid::metal::SPEC_VICTIM_SALVAGE_US.load(Relaxed) - victim0[3]) as f64
                            / 1000.0,
                        camelid::metal::SPEC_VICTIM_VERIFY_FAILS.load(Relaxed) - victim0[4],
                    );
                }
                println!(
                    "[k-report K={k}] rounds={cr} accepted_tokens={} expert_bytes={:.0}MB \
                     dense+head_bytes={:.0}MB total={:.0}MB gpu={:.0}ms head={:.0}ms \
                     idle(host+sync)={:.0}ms realized={:.1}GB/s union/layer \
                     min={} p25={} p50={} p75={} p95={} max={}",
                    spec_tokens.len(),
                    expert_mb,
                    dense_mb,
                    total_mb,
                    gpu_ms,
                    head_wall_ms,
                    idle_ms,
                    total_mb / decode_ms,
                    du.first().map(|&(u, _)| u).unwrap_or(0),
                    pct(0.25),
                    pct(0.50),
                    pct(0.75),
                    pct(0.95),
                    du.last().map(|&(u, _)| u).unwrap_or(0),
                );
            }
            // Head measured DIRECTLY (wall around the head call, plus the command
            // buffer's own GPU time), never inferred by subtracting stages.
            println!(
                "[head K={k}] calls={head_calls} rows={head_rows} wall={head_wall_ms:.1}ms \
                 gpu={head_gpu_ms:.1}ms | per-round wall={:.1}ms gpu={:.1}ms | \
                 share of decode={:.0}%",
                head_wall_ms / rounds.max(1) as f64,
                head_gpu_ms / rounds.max(1) as f64,
                100.0 * head_wall_ms / (decode_s * 1000.0)
            );
            if !exact {
                println!(
                    "[spec K={k}] MISMATCH\n greedy={:?}\n spec  ={:?}",
                    greedy_tokens, spec_tokens
                );
            }
            summary.push(format!(
                "{{\"workload\":\"{name}\",\"k\":{k},\"greedy_tok_s\":{greedy_tok_s:.2},\"greedy_p50_ms\":{greedy_med_ms:.1},\"spec_tok_s\":{spec_tok_s:.2},\"rounds\":{rounds},\"alpha\":{alpha:.2},\"tokens\":{},\"exact\":{exact}}}",
                spec_tokens.len()
            ));
            assert!(
                exact,
                "speculative stream diverged from greedy on {name} K={k}"
            );
        }
    }
    println!("\n[spec50-summary]");
    for line in summary {
        println!("{line}");
    }
}
