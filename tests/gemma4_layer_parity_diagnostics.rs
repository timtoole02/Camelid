//! Layer-by-Layer and Checkpoint-by-Checkpoint Parity Diagnostic for Gemma 4 26B-A4B MoE

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_d = 0.0f32;
    let mut max_idx = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_d {
            max_d = d;
            max_idx = i;
        }
    }
    (max_d, max_idx)
}

#[test]
fn test_layer_by_layer_parity_diagnosis_k2() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 3072, false)
        .expect("load ghost moe");

    let prompt = "The history of science and mathematics from antiquity through the Renaissance is marked by profound discoveries across civilization. In ancient Greece and Alexandria,";
    let tokenizer = runtime.tokenizer();
    let prompt_tokens = tokenizer.encode(prompt, true, true).expect("tokenize");

    println!("Prompt token count: {}", prompt_tokens.len());

    // 1. Run prompt prefill
    let mut kc_ref: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];
    let mut vc_ref: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 30];

    let prefill_logits = runtime
        .prefill_tokens(&prompt_tokens, &mut kc_ref, &mut vc_ref, 0)
        .expect("prefill");
    let mut top_tok = 0;
    let mut max_l = f32::NEG_INFINITY;
    for (i, &v) in prefill_logits.iter().enumerate() {
        if v > max_l {
            max_l = v;
            top_tok = i as u32;
        }
    }
    println!(
        "Prefill top token for pos {}: tok={} (logit={:.3})",
        prompt_tokens.len(),
        top_tok,
        max_l
    );

    // Now define candidate tokens for K = 2
    let candidate_tokens = vec![236778u32, 236770];
    let start_pos = prompt_tokens.len();

    println!("\n================================================================================");
    println!("TEST 1: CAUSAL INVARIANCE OF CANDIDATE 0 (SINGLE VS CHUNK)");
    println!("================================================================================");

    // Run Single token forward for candidate 0
    let mut kc_single = kc_ref.clone();
    let mut vc_single = vc_ref.clone();
    let single_0 = runtime
        .step_chunk(
            &[candidate_tokens[0]],
            start_pos,
            &mut kc_single,
            &mut vc_single,
        )
        .expect("single 0");
    let logits_single_0 = &single_0[0];

    // Rollback sequence to start_pos for chunk test
    runtime.rollback_sequence(start_pos);

    // Run Single token forward for candidate 1 after candidate 0 (sequential)
    let single_1 = runtime
        .step_chunk(
            &[candidate_tokens[1]],
            start_pos + 1,
            &mut kc_single,
            &mut vc_single,
        )
        .expect("single 1");
    let logits_single_1 = &single_1[0];

    // Rollback sequence to start_pos for chunk test
    runtime.rollback_sequence(start_pos);

    println!(
        "Sequential Cand 0 top-1 token: {}",
        logits_single_0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    println!(
        "Sequential Cand 1 top-1 token: {}",
        logits_single_1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );

    // Now run batched step_chunk for both candidates together
    println!("\n================================================================================");
    println!("TEST 2: BATCHED CHUNK FORWARD FOR [CAND 0, CAND 1]");
    println!("================================================================================");

    let mut kc_chunk = kc_ref.clone();
    let mut vc_chunk = vc_ref.clone();
    let chunk_logits = runtime
        .step_chunk(&candidate_tokens, start_pos, &mut kc_chunk, &mut vc_chunk)
        .expect("step_chunk");

    println!(
        "Chunk Cand 0 top-1 token: {}",
        chunk_logits[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    println!(
        "Chunk Cand 1 top-1 token: {}",
        chunk_logits[1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );

    let (d0, idx0) = max_abs_diff(logits_single_0, &chunk_logits[0]);
    let (d1, idx1) = max_abs_diff(logits_single_1, &chunk_logits[1]);

    println!("\nLogits Max Difference (Sequential vs Chunk):");
    println!(
        "  Cand 0: max diff = {:.6} at index {} (Seq={:.4}, Chunk={:.4})",
        d0, idx0, logits_single_0[idx0], chunk_logits[0][idx0]
    );
    println!(
        "  Cand 1: max diff = {:.6} at index {} (Seq={:.4}, Chunk={:.4})",
        d1, idx1, logits_single_1[idx1], chunk_logits[1][idx1]
    );

    if d0 > 0.05 || d1 > 0.05 {
        println!(
            "\n[PARITY FAILURE DETECTED] Batched chunk forward does not match sequential forward!"
        );
    } else {
        println!(
            "\n[PARITY SUCCESS] 100% Bit-Exact Match between Batched Chunk and Sequential forward!"
        );
    }

    assert!(
        d0 < 0.05,
        "Cand 0 diverges between single and chunk: max diff = {}",
        d0
    );
    assert!(
        d1 < 0.05,
        "Cand 1 diverges between single and chunk: max diff = {}",
        d1
    );
}
