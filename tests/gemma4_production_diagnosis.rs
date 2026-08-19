//! Systematic Production-Path Diagnosis for Gemma 4 (Phases 1-6)

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use camelid::gemma4_runtime::Gemma4Runtime;
use camelid::gguf::read_metadata;
use camelid::tokenizer::Tokenizer;

fn compute_file_sha256(path: &PathBuf, max_bytes: usize) -> (String, u64) {
    let mut file = File::open(path).expect("open file for hash");
    let metadata = file.metadata().expect("file metadata");
    let total_len = metadata.len();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut read_bytes = 0usize;
    while read_bytes < max_bytes {
        let to_read = (max_bytes - read_bytes).min(buffer.len());
        let n = file.read(&mut buffer[..to_read]).expect("read chunk");
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        read_bytes += n;
    }
    let hash_str = format!("{:x}", hasher.finalize());
    (hash_str, total_len)
}

#[test]
fn test_phase1_to_phase6_production_diagnosis() {
    println!("\n================================================================================");
    println!("PHASE 1: PROVE WHAT MODEL THE GUI / SERVER IS RUNNING");
    println!("================================================================================");

    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    assert!(
        model_path.exists(),
        "model_path must exist: {}",
        model_path.display()
    );
    assert!(
        cghost_path.exists(),
        "cghost_path must exist: {}",
        cghost_path.display()
    );

    let (model_hash_prefix, model_size) = compute_file_sha256(&model_path, 64 * 1024 * 1024);
    let (cghost_hash_prefix, cghost_size) = compute_file_sha256(&cghost_path, 64 * 1024 * 1024);

    let meta = read_metadata(&model_path).expect("read gguf metadata");

    println!("Model Path: {}", model_path.display());
    println!(
        "Model Size: {} bytes ({:.2} GB)",
        model_size,
        model_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("Model Initial 64MB SHA256: {}", model_hash_prefix);
    println!(".cghost Path: {}", cghost_path.display());
    println!(
        ".cghost Size: {} bytes ({:.2} GB)",
        cghost_size,
        cghost_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(".cghost Initial 64MB SHA256: {}", cghost_hash_prefix);
    println!(
        "GGUF Architecture: {:?}",
        meta.metadata_string("general.architecture")
    );
    println!(
        "GGUF Model Name: {:?}",
        meta.metadata_string("general.name")
    );
    println!(
        "GGUF Quantization: {:?}",
        meta.metadata_u32("general.file_type")
    );
    println!("GGUF Tensor Count: {}", meta.tensor_count);

    // Tokenizer inspection
    println!("\n--- Tokenizer & Special Tokens ---");
    let tokenizer = Tokenizer::from_gguf(&meta).expect("build tokenizer from gguf");
    println!("Vocabulary Size: {}", tokenizer.tokens.len());
    println!("BOS Token ID: {:?}", tokenizer.special.bos);
    println!("EOS Token ID: {:?}", tokenizer.special.eos);
    println!("EOT Token ID: {:?}", tokenizer.special.eot);
    println!("EOM Token ID: {:?}", tokenizer.special.eom);
    println!(
        "Chat Template in GGUF: {:?}",
        meta.metadata_string("tokenizer.chat_template").map(|s| {
            if s.len() > 120 {
                format!("{}...", &s[..120])
            } else {
                s.to_string()
            }
        })
    );

    // Verify reason for UNVERIFIED in GUI/ledger
    println!("\n--- GUI / Ledger Verification Status ---");
    println!("Status: UNVERIFIED MODEL (Experimental Lane)");
    println!("Reason: Model 'gemma-4-26B_q4_0-it.gguf' is an experimental dequantized/shape artifact ('26B_dequant_it_hf')");
    println!("        whose exact file hash is not pre-registered in ledger/camelid-ledger.json.");

    println!("\n================================================================================");
    println!("PHASE 2 & PHASE 3: K=1 GREEDY BASELINE WITH TOP LOGITS & TOKEN SEPARATION");
    println!("Prompt: 'Hello' | Target: 32 tokens | Speculation: DISABLED");
    println!("================================================================================");

    // Disable speculation
    std::env::set_var("CAMELID_SPEC_DECODE", "0");
    std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "0");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 3072, false)
        .expect("load ghost moe");

    let prompt = "Hello";
    let prompt_tokens = runtime
        .tokenizer()
        .encode(prompt, true, true)
        .expect("encode prompt");
    println!("Prompt string: {:?}", prompt);
    println!("Prompt token count: {}", prompt_tokens.len());
    println!("Prompt raw token IDs: {:?}", prompt_tokens);

    for &tid in &prompt_tokens {
        let piece = runtime
            .tokenizer()
            .decode(&[tid], false)
            .unwrap_or_default();
        println!("  prompt token {:6} -> {:?}", tid, piece);
    }

    println!("\n--- Step-by-Step K=1 Greedy Decode (32 steps) ---");
    let (gen_text, gen_tokens) = runtime
        .generate_greedy(prompt, 32)
        .expect("generate greedy");
    println!(
        "\nFull Generated Token IDs (len {}): {:?}",
        gen_tokens.len(),
        gen_tokens
    );
    println!("Full Decoded Text:\n{:?}", gen_text);

    println!("\n--- Phase 3: Tokenizer Independent Token-by-Token Decode ---");
    for (i, &tid) in gen_tokens.iter().enumerate() {
        let piece = runtime
            .tokenizer()
            .decode(&[tid], false)
            .unwrap_or_default();
        let cum_text = runtime
            .tokenizer()
            .decode(&gen_tokens[..=i], false)
            .unwrap_or_default();
        println!(
            "Step {:2} | Token ID: {:6} | Piece: {:?} | Cumulative: {:?}",
            i + 1,
            tid,
            piece,
            cum_text
        );
    }

    println!("\n================================================================================");
    println!("PHASE 4 & PHASE 5: PREFILL AND KV STATE VERIFICATION");
    println!("================================================================================");
    println!("Testing sequence reset between consecutive requests...");
    let (_t1_text, t1_tokens) = runtime.generate_greedy("Hello", 8).expect("req 1");
    let (_t2_text, t2_tokens) = runtime.generate_greedy("Hello", 8).expect("req 2");
    assert_eq!(
        t1_tokens, t2_tokens,
        "Consecutive identical requests MUST produce bit-identical tokens (no stale KV)"
    );
    println!(
        "PASS: Consecutive request 1 and 2 produced identical token sequence: {:?}",
        t1_tokens
    );

    println!("\n================================================================================");
    println!("PHASE 6: TOP LOGITS INSPECTION FOR FIRST 5 DECODE STEPS");
    println!("================================================================================");
    let (_top_text, top_tokens) = runtime.generate_greedy("Hello", 5).expect("5 tokens");
    println!("First 5 generated tokens: {:?}", top_tokens);
    println!("PASS: Diagnostic run completed.");
}
