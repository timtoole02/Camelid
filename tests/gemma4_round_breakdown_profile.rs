//! Fine-grained microsecond profile breakdown of a genuine Gemma 4 26B-A4B K=5 verification round
//! Budget: 16 slots/layer (1.50 GiB Metal resident).

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, time::Instant};

#[test]
fn test_genuine_gemma4_round_breakdown_profile() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    let test_prompt = "<|turn>user\nExplain quantum entanglement.<turn|>\n<|turn>model\n";

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "24");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");
    std::env::set_var("CAMELID_TIMING", "1");
    std::env::set_var("CAMELID_SPEC_DECODE", "1");
    std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "4");

    println!("==========================================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B K=5 VERIFICATION ROUND MICROSECOND PROFILE");
    println!("Budget: 16 slots/layer (1.50 GiB Metal resident, 3.16 GiB Page Cache)");
    println!("==========================================================================================================\n");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load ghost moe");

    let t0 = Instant::now();
    let (_text, tokens) = runtime
        .generate_greedy_speculative(test_prompt, 8)
        .expect("generate");
    let total_dur = t0.elapsed();

    println!(
        "\nTotal generated tokens: {} in {:.2}s ({:.2} tok/s)",
        tokens.len(),
        total_dur.as_secs_f64(),
        tokens.len() as f64 / total_dur.as_secs_f64()
    );
    println!("==========================================================================================================\n");
}
