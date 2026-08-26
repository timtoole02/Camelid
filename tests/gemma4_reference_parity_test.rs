mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[test]
fn test_gemma4_reference_parity_prompt() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("Model files not found under CAMELID_MODEL_ROOT or the operator model directory");
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");
    std::env::set_var("CAMELID_GEMMA4_DUMP_LAYERS", "1");

    println!("=== LOADING RUNTIME ===");
    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load Metal runtime");

    let prompt_de = "English: good morning\nGerman:";
    println!("Prompt: {:?}", prompt_de);
    let p_tokens = runtime.tokenizer().encode(prompt_de, true, true).unwrap();
    println!("Prompt tokens (bos=true): {:?}", p_tokens);
    assert_eq!(
        p_tokens,
        vec![2, 27832, 236787, 1535, 5597, 107, 51423, 236787]
    );

    println!("\n=== RUNNING METAL GENERATION ===");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    let (metal_text, metal_tokens) = runtime
        .generate_greedy(prompt_de, 16)
        .expect("generate greedy metal");
    println!("Metal text:\n{}", metal_text);
    println!("Metal tokens: {:?}", metal_tokens);

    let oracle_tokens: Vec<u32> = vec![
        154016, 112657, 107, 51423, 236787, 154016, 112657, 107, 51423, 236787, 1535, 5597, 107,
        15466, 5597, 107,
    ];
    println!("Oracle tokens: {:?}", oracle_tokens);
    let oracle_text = " Guten Morgen\nGerman: Guten Morgen\nGerman: good morning\ngood morning\n";
    println!("Oracle text:\n{}", oracle_text);

    let decoded_tokens: Vec<String> = metal_tokens
        .iter()
        .map(|&t| runtime.tokenizer().decode(&[t], false).unwrap_or_default())
        .collect();
    println!("Metal token-by-token: {:?}", decoded_tokens);
}
