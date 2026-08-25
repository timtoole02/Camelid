//! Target Audit: Prove whether step_chunk performs genuine multi-vector MoE execution or sequential execution at K=5

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[test]
fn test_real_gemma4_moe_layer_audit_k5() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_MOE_AUDIT", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 3072, false)
        .expect("load ghost moe");

    // 5 candidate tokens (K=5)
    let candidate_tokens = vec![236778u32, 236770, 236764, 236743, 236800];
    let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];

    // Execute step_chunk on the 5 candidate tokens simultaneously starting at position 0
    println!("\nExecuting step_chunk on 5 candidate tokens simultaneously (K=5)...");
    let rows = runtime
        .step_chunk(&candidate_tokens, 0, &mut kc, &mut vc)
        .expect("step_chunk");
    assert_eq!(rows.len(), 5, "step_chunk must return 5 prediction rows");
    println!("Step chunk executed successfully: 5 prediction rows generated.\n");

    println!("Executing sequential step_range for bit-exact parity check...");
    let mut kc_seq: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc_seq: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    for (i, &tok) in candidate_tokens.iter().enumerate() {
        let out = runtime
            .step_range(tok, i, None, &mut kc_seq, &mut vc_seq)
            .expect("step_range");
        let seq_logits = match out {
            camelid::gemma4_runtime::Gemma4StepOutput::Logits(l) => l,
            _ => panic!("expected logits"),
        };
        let chunk_l = &rows[i];
        let dot: f32 = chunk_l.iter().zip(&seq_logits).map(|(a, b)| a * b).sum();
        let norm_chunk: f32 = chunk_l.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_seq: f32 = seq_logits.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_chunk * norm_seq);
        let max_diff = chunk_l
            .iter()
            .zip(&seq_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let mut top_chunk: Vec<(usize, f32)> = chunk_l.iter().copied().enumerate().collect();
        top_chunk.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut top_seq: Vec<(usize, f32)> = seq_logits.iter().copied().enumerate().collect();
        top_seq.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Token {i}: cos_sim = {cos_sim:.7}, max_diff = {max_diff:.4}");
        println!("  chunk top-3: {:?}", &top_chunk[..3]);
        println!("  seq   top-3: {:?}", &top_seq[..3]);
    }
    println!("AUDIT AND COMPARISON COMPLETE!\n");
}
