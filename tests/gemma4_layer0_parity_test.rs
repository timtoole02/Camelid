use camelid::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};
use std::path::PathBuf;

#[test]
fn test_layer0_substep_parity() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        return;
    }

    let token = 2u32; // <bos>
    let pos = 0usize;

    // 1. Pure CPU execution
    println!("=== [1/2] RUNNING PURE CPU REFERENCE ===");
    std::env::set_var("CAMELID_GEMMA4_DUMP_LAYERS", "1");
    std::env::remove_var("CAMELID_GEMMA4_GHOST_METAL");
    std::env::remove_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS");
    std::env::remove_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST");

    let runtime_cpu =
        Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false).expect("load CPU");

    let mut kc_cpu = vec![Vec::new(); 30];
    let mut vc_cpu = vec![Vec::new(); 30];
    let out_cpu = runtime_cpu
        .step_range(token, pos, None, &mut kc_cpu, &mut vc_cpu)
        .expect("step cpu");
    let logits_cpu = match out_cpu {
        Gemma4StepOutput::Logits(v) => v,
        Gemma4StepOutput::Hidden(v) => v,
    };
    drop(runtime_cpu);
    drop(kc_cpu);
    drop(vc_cpu);

    // 2. Metal execution
    println!("\n=== [2/2] RUNNING METAL RUNTIME ===");
    std::env::remove_var("CAMELID_DETERMINISTIC");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");

    let runtime_metal =
        Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false).expect("load Metal");

    let mut kc_metal = vec![Vec::new(); 30];
    let mut vc_metal = vec![Vec::new(); 30];
    let logits_metal = runtime_metal
        .step(token, pos, &mut kc_metal, &mut vc_metal)
        .expect("step metal");

    println!("\n=== COMPARISON SUMMARY ===");
    println!(
        "CPU logits len={}, sum={:.4}",
        logits_cpu.len(),
        logits_cpu.iter().sum::<f32>()
    );
    println!(
        "Metal logits len={}, sum={:.4}",
        logits_metal.len(),
        logits_metal.iter().sum::<f32>()
    );

    let mut cpu_ranked: Vec<(usize, f32)> = logits_cpu.iter().copied().enumerate().collect();
    cpu_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut metal_ranked: Vec<(usize, f32)> = logits_metal.iter().copied().enumerate().collect();
    metal_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 tokens on CPU:");
    for (rank, (tok, logit)) in cpu_ranked.iter().take(10).enumerate() {
        println!("  #{}: token_id={} logit={:.4}", rank + 1, tok, logit);
    }

    println!("\nTop 10 tokens on Metal:");
    for (rank, (tok, logit)) in metal_ranked.iter().take(10).enumerate() {
        println!("  #{}: token_id={} logit={:.4}", rank + 1, tok, logit);
    }

    assert_eq!(
        cpu_ranked[0].0, metal_ranked[0].0,
        "Top-1 token must match between CPU and Metal reference"
    );
    println!(
        "\n>>> TEST PASSED: CPU and Metal top-1 token matches (token_id={}) <<<",
        metal_ranked[0].0
    );
}
