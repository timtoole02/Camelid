//! Teacher-forced chunk-width bisect for the 15/16 oracle divergence.
//!
//! Same tokens, same start position, different chunk widths: per-token logits
//! must be bitwise identical regardless of how wide the covering chunk is
//! (width only changes how many tokens share one weight read). Measured
//! violation: widths 15/16 flip the code-edit near-tie at token 4.
//!
//! This test feeds the ORACLE token stream (teacher forcing, so every width
//! sees identical inputs) through `step_chunk` at widths {8, 12, 14, 15, 16}
//! and compares each row's logits bitwise against the width-8 baseline.
//! Reports the first differing (width, row) and the max ULP, plus whether the
//! row-3 argmax (the token-4 near-tie) flips.
//!
//! Env: CAMELID_GEMMA4_26B_GGUF / CAMELID_GEMMA4_26B_CGHOST, WIDTHS override.

use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

fn ulp_diff(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

fn argmax(l: &[f32]) -> usize {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn gemma4_width_bisect() {
    let model_path = std::env::var_os("CAMELID_GEMMA4_26B_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf"));
    let cghost_path = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost"));
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
    set_default("CAMELID_GEMMA4_GHOST_METAL_CONTEXT", "1024");
    set_default("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT", "1");
    set_default("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "88");
    set_default("CAMELID_GEMMA4_SPEC_CHUNK_MAX", "16");

    // The oracle fixture's code-edit prompt and its verified greedy ids.
    let prompt = "<|turn>user\nAdd a `pub expires_at: u64,` field at the end of this struct and output the COMPLETE struct definition again, unchanged otherwise, with no explanation:\n\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n}\n<turn|>\n<|turn>model\n";
    let oracle_ids: Vec<u32> = vec![
        100, 45518, 107, 101, 2717, 22413, 107, 9430, 2456, 47910, 10356, 236820, 236855, 236764,
        782, 236813, 642, 107, 140, 9430, 2307, 236787, 751, 236764,
    ];

    let widths: Vec<usize> = std::env::var("WIDTHS")
        .unwrap_or_else(|_| "8,12,14,15,16".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 2900, false)
        .expect("load ghost moe");
    let prompt_tokens = runtime.tokenizer().encode(prompt, true, true).unwrap();
    let n_layers = 30usize;

    // Per width: fresh prefill, then ONE teacher-forced chunk of that width
    // starting at the same position with the same oracle tokens.
    let mut rows_by_width: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
    for &w in &widths {
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let _ = runtime
            .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 31)
            .expect("prefill");
        let chunk: Vec<u32> = oracle_ids[..w].to_vec();
        let rows = runtime
            .step_chunk(&chunk, prompt_tokens.len(), &mut kc, &mut vc)
            .expect("step_chunk");
        println!(
            "[width {w}] rows={} argmax(row3)={} argmax(row2)={}",
            rows.len(),
            rows.get(3).map(|r| argmax(r) as i64).unwrap_or(-1),
            rows.get(2).map(|r| argmax(r) as i64).unwrap_or(-1),
        );
        rows_by_width.push((w, rows));
        // Reset the resident Metal sequence so the next width starts clean.
        runtime.rollback_sequence(0);
    }

    let (base_w, base_rows) = &rows_by_width[0];
    let mut any_diff = false;
    for (w, rows) in &rows_by_width[1..] {
        let shared = base_rows.len().min(rows.len());
        let mut first_bad_row = None;
        for r in 0..shared {
            let a = &base_rows[r];
            let b = &rows[r];
            let n_diff = a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            if n_diff > 0 {
                let max_ulp = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| ulp_diff(*x, *y))
                    .max()
                    .unwrap_or(0);
                let flip = argmax(a) != argmax(b);
                println!(
                    "[width {w} vs {base_w}] row {r}: {n_diff} values differ, max ULP {max_ulp}, argmax {} ({} -> {})",
                    if flip { "FLIPS" } else { "same" },
                    argmax(a),
                    argmax(b)
                );
                if first_bad_row.is_none() {
                    first_bad_row = Some(r);
                }
                any_diff = true;
            }
        }
        match first_bad_row {
            None => println!("[width {w} vs {base_w}] all {shared} shared rows bitwise identical"),
            Some(r) => println!("[width {w} vs {base_w}] FIRST differing row: {r}"),
        }
    }
    // Diagnosis test: report, do not gate CI. The lane oracle test is the gate.
    if any_diff {
        println!("[width-bisect] WIDTH-DEPENDENT NUMERICS CONFIRMED (see rows above)");
    } else {
        println!("[width-bisect] all widths bitwise identical on shared rows");
    }
}
