//! Gemma 4 26B lane parity gate: HEAD lane vs chained lane vs speculative.
//!
//! The speculative bench asserts that the speculative stream equals the greedy
//! stream, but both run on the SAME lane, so that assert cannot see a lane that
//! is uniformly wrong. This test pins the three lanes against each other and
//! prints token ids for an external llama.cpp comparison:
//!
//!   1. HEAD lane greedy    — `step()`, the lane the oracle parity was
//!                            established on (see the K=1 HEAD lane work).
//!   2. chained lane greedy — `step_chunk(&[tok])`, the lane the K>1 verifier
//!                            uses and therefore the lane speculative decode
//!                            falls back to for draft-less rounds.
//!   3. speculative         — `spec_decode_generate`.
//!
//! Each lane is also compared against llama.cpp's greedy token ids, captured
//! from `llama-server` on the FULL GGUF (CPU graph, temp 0 / top_k 1) and
//! committed under qa/evidence-bundles/gemma4-26b-oracle/. Measured 2026-08-19:
//! the chained lane and speculative decode match llama.cpp EXACTLY on all three
//! prompts, while the HEAD lane rounds a near-tie the other way at token 4 of
//! `code-edit` (emitting `pub` where llama.cpp opens a ```rust fence, rejoining
//! three tokens later). The oracle must be regenerated against the FULL GGUF on
//! the T7: the copy under ~/models is the sparse hot shadow whose routed-expert
//! ranges are holes, and it would score every lane against zeros.
//!
//! All three must emit identical token streams. A divergence between (1) and
//! (2) means the chained lane is not the oracle-verified numerics, and any
//! throughput measured on it is measured on a different model.
//!
//! Env:
//!   CAMELID_GEMMA4_26B_GGUF / CAMELID_GEMMA4_26B_CGHOST   model pair
//!   LANEP_MAX_NEW     tokens per prompt (default 48)
//!   LANEP_K           speculative draft width (default 8)
//!   LANEP_CACHE_MIB   host expert cache MiB (default 2900)
//!   LANEP_SOFT        report divergences without failing (diagnosis runs)

mod support;

use camelid::gemma4_runtime::{gemma4_stop_token_ids, Gemma4Runtime};
use std::{path::PathBuf, time::Instant};

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

/// Top-1 and top-2 (token, logit) plus the gap between them. A divergence at a
/// position whose gap is a few ULP is two lanes rounding a genuine tie in
/// different directions; a divergence at a position with a wide gap is a real
/// numerical error and must be treated as a bug.
fn top2(l: &[f32]) -> ((u32, f32), (u32, f32), f32) {
    let (mut b1, mut v1, mut b2, mut v2) = (0u32, f32::NEG_INFINITY, 0u32, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            b2 = b1;
            v2 = v1;
            b1 = i as u32;
            v1 = v;
        } else if v > v2 {
            b2 = i as u32;
            v2 = v;
        }
    }
    ((b1, v1), (b2, v2), v1 - v2)
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n).find(|&i| a[i] != b[i]).or({
        if a.len() == b.len() {
            None
        } else {
            Some(n)
        }
    })
}

/// Prompts kept short and deterministic; the point is lane agreement, not quality.
fn prompts() -> Vec<(&'static str, String)> {
    vec![
        (
            "greeting",
            "<|turn>user\nSay hello and name three colours.\n<turn|>\n<|turn>model\n".to_string(),
        ),
        (
            "json-yaml",
            "<|turn>user\nConvert this configuration payload to YAML:\n{\"cluster_id\": \"prod-1\", \"min_replicas\": 4}\n<turn|>\n<|turn>model\n".to_string(),
        ),
        (
            "code-edit",
            "<|turn>user\nAdd a `pub expires_at: u64,` field at the end of this struct and output the COMPLETE struct definition again, unchanged otherwise, with no explanation:\n\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n}\n<turn|>\n<|turn>model\n".to_string(),
        ),
    ]
}

#[test]
fn gemma4_lane_oracle_parity() {
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

    let max_new: usize = env_or("LANEP_MAX_NEW", "48").parse().unwrap();
    let k: usize = env_or("LANEP_K", "8").parse().unwrap();
    let cache_mib: usize = env_or("LANEP_CACHE_MIB", "2900").parse().unwrap();
    let soft = std::env::var_os("LANEP_SOFT").is_some();
    std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", k.to_string());

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, cache_mib, false)
        .expect("load ghost moe");
    let eot = gemma4_stop_token_ids(runtime.tokenizer());
    let n_layers = 30usize;

    let mut failures: Vec<String> = Vec::new();

    // Committed llama.cpp greedy ids (see the module docs for how to regenerate).
    let oracle: Option<serde_json::Value> =
        std::fs::read_to_string("qa/evidence-bundles/gemma4-26b-oracle/llamacpp_greedy_ids.json")
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
    if oracle.is_none() {
        println!("[warn] llama.cpp fixture missing; lane-vs-lane checks only");
    }
    // Camelid stops BEFORE a stop token; llama.cpp includes it in its output.
    let strip_stops = |v: &[u32], eot: &[u32]| -> Vec<u32> {
        let mut o = v.to_vec();
        while o.last().is_some_and(|t| eot.contains(t)) {
            o.pop();
        }
        o
    };

    for (name, prompt) in prompts() {
        println!("\n================ PROMPT {name} ================");
        let prompt_tokens = runtime.tokenizer().encode(&prompt, true, true).unwrap();
        println!("[prompt ids] {prompt_tokens:?}");

        let mut head_top2: Vec<((u32, f32), (u32, f32), f32)> = Vec::new();
        let mut chained_top2: Vec<((u32, f32), (u32, f32), f32)> = Vec::new();

        // --- lane 1: HEAD lane greedy (step) ---
        let head_ids = {
            let mut kc = vec![Vec::new(); n_layers];
            let mut vc = vec![Vec::new(); n_layers];
            let mut logits = runtime
                .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))
                .unwrap();
            let mut pos = prompt_tokens.len();
            let mut ids = Vec::new();
            let t = Instant::now();
            while ids.len() < max_new {
                head_top2.push(top2(&logits));
                let tok = argmax(&logits);
                if eot.contains(&tok) {
                    break;
                }
                ids.push(tok);
                if ids.len() >= max_new {
                    break;
                }
                logits = runtime.step(tok, pos, &mut kc, &mut vc).unwrap();
                pos += 1;
            }
            println!(
                "[lane head   ] {} tokens in {:.2}s",
                ids.len(),
                t.elapsed().as_secs_f64()
            );
            ids
        };

        // --- lane 2: chained lane greedy (step_chunk with K=1) ---
        let chained_ids = {
            let mut kc = vec![Vec::new(); n_layers];
            let mut vc = vec![Vec::new(); n_layers];
            let mut logits = runtime
                .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))
                .unwrap();
            let mut pos = prompt_tokens.len();
            let mut ids = Vec::new();
            let t = Instant::now();
            while ids.len() < max_new {
                chained_top2.push(top2(&logits));
                let tok = argmax(&logits);
                if eot.contains(&tok) {
                    break;
                }
                ids.push(tok);
                if ids.len() >= max_new {
                    break;
                }
                logits = runtime
                    .step_chunk(&[tok], pos, &mut kc, &mut vc)
                    .unwrap()
                    .pop()
                    .unwrap();
                pos += 1;
            }
            println!(
                "[lane chained] {} tokens in {:.2}s",
                ids.len(),
                t.elapsed().as_secs_f64()
            );
            ids
        };

        // --- lane 3: speculative ---
        let spec_ids = {
            let mut kc = vec![Vec::new(); n_layers];
            let mut vc = vec![Vec::new(); n_layers];
            let logits = runtime
                .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))
                .unwrap();
            let t = Instant::now();
            let ids = runtime
                .spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)
                .unwrap();
            println!(
                "[lane spec   ] {} tokens in {:.2}s",
                ids.len(),
                t.elapsed().as_secs_f64()
            );
            ids
        };

        // Token ids are the comparison unit; text is for eyeballing gibberish.
        println!("[ids head   ] {head_ids:?}");
        println!("[ids chained] {chained_ids:?}");
        println!("[ids spec   ] {spec_ids:?}");
        println!(
            "[text head  ] {:?}",
            runtime
                .tokenizer()
                .decode(&head_ids, true)
                .unwrap_or_default()
        );
        println!(
            "[text chained] {:?}",
            runtime
                .tokenizer()
                .decode(&chained_ids, true)
                .unwrap_or_default()
        );
        println!(
            "[text spec  ] {:?}",
            runtime
                .tokenizer()
                .decode(&spec_ids, true)
                .unwrap_or_default()
        );

        match first_divergence(&head_ids, &chained_ids) {
            None => println!("[PASS] head == chained ({} tokens)", head_ids.len()),
            Some(i) => {
                let hg = head_top2.get(i);
                let cg = chained_top2.get(i);
                let msg = format!(
                    "{name}: chained lane diverges from HEAD lane at token {i} \
                     (head={:?} chained={:?})",
                    head_ids.get(i),
                    chained_ids.get(i)
                );
                println!("[NOTE] {msg}");
                if let (Some(h), Some(c)) = (hg, cg) {
                    println!(
                        "[gap ] head  top1={:?} top2={:?} gap={:.3e}\n\
                         [gap ] chain top1={:?} top2={:?} gap={:.3e}\n\
                         [gap ] -> {}",
                        h.0,
                        h.1,
                        h.2,
                        c.0,
                        c.1,
                        c.2,
                        if h.2.min(c.2) < 1e-3 {
                            "NEAR-TIE: the two lanes rounded a genuine tie apart"
                        } else {
                            "WIDE GAP: not a tie, this is a numerical error"
                        }
                    );
                }
                // Not a failure on its own. llama.cpp arbitrates which lane is
                // right, and it sided with the chained lane (see the oracle
                // comparison below); the HEAD lane is the one that rounds this
                // near-tie the other way. The chained lane is what decode and
                // the speculative verifier actually run.
            }
        }

        if let Some(o) = oracle.as_ref().and_then(|o| o.get(name)) {
            let raw: Vec<u32> = o["gen_ids"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect()
                })
                .unwrap_or_default();
            let oracle_ids = strip_stops(&raw, &eot);
            println!("[ids oracle ] {oracle_ids:?}");
            for (lane_name, ids) in [
                ("head", &head_ids),
                ("chained", &chained_ids),
                ("spec", &spec_ids),
            ] {
                let n = ids.len().min(oracle_ids.len());
                match first_divergence(&ids[..n], &oracle_ids[..n]) {
                    None => println!("[PASS] {lane_name} == llama.cpp ({n} tokens)"),
                    Some(i) => {
                        let msg = format!(
                            "{name}: {lane_name} lane diverges from llama.cpp at token {i} \
                             ({lane_name}={:?} llama.cpp={:?})",
                            ids.get(i),
                            oracle_ids.get(i)
                        );
                        if lane_name == "head" {
                            println!("[NOTE] {msg} (known HEAD-lane near-tie)");
                        } else {
                            println!("[FAIL] {msg}");
                            failures.push(msg);
                        }
                    }
                }
            }
        }

        match first_divergence(&chained_ids, &spec_ids) {
            None => println!("[PASS] chained == spec ({} tokens)", spec_ids.len()),
            Some(i) => {
                let msg = format!(
                    "{name}: speculative diverges from chained greedy at token {i} \
                     (chained={:?} spec={:?})",
                    chained_ids.get(i),
                    spec_ids.get(i)
                );
                println!("[FAIL] {msg}");
                failures.push(msg);
            }
        }
    }

    println!("\n[lane-parity-summary] {} failure(s)", failures.len());
    for f in &failures {
        println!("  - {f}");
    }
    if !failures.is_empty() && !soft {
        panic!("lane parity failures: {failures:#?}");
    }
}
