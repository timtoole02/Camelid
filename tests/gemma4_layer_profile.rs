//! Gemma 4 26B Layer-by-Layer Expert Routing and Working Set Profiler

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_gemma4_26b_layer_profile() {
    let Some(model_path) = std::env::var_os("CAMELID_GEMMA4_26B_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP: CAMELID_GEMMA4_26B_GGUF not set");
        return;
    };
    let Some(cghost_path) = std::env::var_os("CAMELID_GEMMA4_26B_CGHOST").map(PathBuf::from) else {
        eprintln!("SKIP: CAMELID_GEMMA4_26B_CGHOST not set");
        return;
    };

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    if std::env::var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER").is_err() {
        std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "88");
    }
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 2900, false)
        .expect("load ghost moe");

    let prompt = "<|turn>user\nExplain in technical depth how general relativity predicts gravitational time dilation and frame dragging, with mathematical reasoning.<turn|>\n<|turn>model\n";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer
        .encode(prompt, true, true)
        .expect("tokenize prompt");

    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let initial_logits = runtime
        .prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)
        .expect("prefill");

    let total_steps = 256;
    let mut cur_logits = initial_logits;
    let mut step_latencies = Vec::with_capacity(total_steps);
    let t_decode_start = Instant::now();

    for cur_pos in prompt_tokens.len()..prompt_tokens.len() + total_steps {
        let tok = cur_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap();

        let t_step = Instant::now();
        cur_logits = runtime
            .step(tok, cur_pos, &mut kc, &mut vc)
            .expect("step decode");
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
        step_latencies.push(step_ms);
    }

    let decode_wall_s = t_decode_start.elapsed().as_secs_f64();
    let steady_latencies = &step_latencies[16..];
    let mut sorted_steady = steady_latencies.to_vec();
    sorted_steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = sorted_steady[sorted_steady.len() / 2];
    let median_toks = 1000.0 / median_ms;

    println!("\n==========================================================================================================");
    println!("256 ADVANCING TOKENS PROFILE RESULTS");
    println!(
        "Total decode wall: {:.2}s | Steady median: {:.2} tok/s ({:.2} ms)",
        decode_wall_s, median_toks, median_ms
    );
    println!("----------------------------------------------------------------------------------------------------------");
    println!("Layer | Slots | Hits   | Misses | Hit Rate %");
    println!("------+-------+--------+--------+-----------");
    let layer_stats = runtime.ghost_metal_layer_slot_stats();
    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    for (layer, hits, misses, slots) in &layer_stats {
        let accesses = hits + misses;
        let rate = if accesses > 0 {
            (*hits as f64 / accesses as f64) * 100.0
        } else {
            100.0
        };
        total_hits += hits;
        total_misses += misses;
        println!(
            "{:5} | {:5} | {:6} | {:6} | {:8.2}%",
            layer, slots, hits, misses, rate
        );
    }
    let total_acc = total_hits + total_misses;
    let overall_rate = if total_acc > 0 {
        (total_hits as f64 / total_acc as f64) * 100.0
    } else {
        100.0
    };
    println!("------+-------+--------+--------+-----------");
    println!(
        "TOTAL | {:5} | {:6} | {:6} | {:8.2}%",
        layer_stats.iter().map(|s| s.3).sum::<usize>(),
        total_hits,
        total_misses,
        overall_rate
    );
    println!("==========================================================================================================");
}
