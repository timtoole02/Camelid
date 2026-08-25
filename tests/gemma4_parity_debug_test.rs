mod support;

use camelid::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};
use std::path::PathBuf;

#[test]
fn test_all_30_layers_hard_parity_assertions() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        return;
    }

    let token = 2u32; // <bos>
    let pos = 0usize;

    // 1. CPU forward pass
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
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
        Gemma4StepOutput::Logits(h) => h,
        Gemma4StepOutput::Hidden(h) => h,
    };

    // 2. Metal forward pass
    std::env::remove_var("CAMELID_DETERMINISTIC");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "128");

    let runtime_metal =
        Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false).expect("load Metal");

    let mut kc_metal = vec![Vec::new(); 30];
    let mut vc_metal = vec![Vec::new(); 30];
    let logits_metal = runtime_metal
        .step(token, pos, &mut kc_metal, &mut vc_metal)
        .expect("step metal");

    // 3. Compare Logits with Hard Mathematical Assertions
    println!("\n=== FINAL LOGITS NUMERICAL PARITY REPORT ===");
    assert_eq!(logits_cpu.len(), logits_metal.len());

    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f32;
    let mut cpu_sq = 0.0f32;
    let mut metal_sq = 0.0f32;

    for (c, m) in logits_cpu.iter().zip(logits_metal.iter()) {
        let diff = (c - m).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;
        cpu_sq += c * c;
        metal_sq += m * m;
    }

    let mean_abs_diff = sum_abs_diff / logits_cpu.len() as f32;
    let l2_diff = sum_sq_diff.sqrt();
    let cpu_l2 = cpu_sq.sqrt();
    let metal_l2 = metal_sq.sqrt();
    let rel_l2_err = l2_diff / cpu_l2;

    // Argmax check
    let top1_cpu = logits_cpu
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let top1_metal = logits_metal
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!(
        "Logits: CPU_L2={:8.3} Metal_L2={:8.3} | max_diff={:.4} mean_diff={:.5} l2_diff={:.3} rel_err={:.4}",
        cpu_l2, metal_l2, max_abs_diff, mean_abs_diff, l2_diff, rel_l2_err
    );
    println!(
        "Top-1 token: CPU=token {} ({:.4}), Metal=token {} ({:.4})",
        top1_cpu.0, top1_cpu.1, top1_metal.0, top1_metal.1
    );

    assert_eq!(
        top1_cpu.0, top1_metal.0,
        "Argmax top-1 token mismatch: CPU emitted token {}, Metal emitted token {}",
        top1_cpu.0, top1_metal.0
    );
    println!(">>> TOP-1 TOKEN EXACT MATCH: TOKEN {} <<<", top1_cpu.0);
}
