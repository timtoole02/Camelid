use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[test]
fn test_gemma4_generate_hello() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "24");

    let runtime =
        Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false).expect("load Metal");

    println!("Model loaded successfully.");
    let prompt = "<|turn>user\nExplain how BB84 quantum key distribution works in two short paragraphs.<turn|>\n<|turn>model\n";
    println!("Prompt: {:?}", prompt);

    let t0 = std::time::Instant::now();
    let (text, tokens) = runtime
        .generate_greedy(prompt, 64)
        .expect("generate greedy");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tok_s = if ms > 0.0 {
        (tokens.len() as f64) / (ms / 1000.0)
    } else {
        0.0
    };
    println!("Generated text: {:?}", text);
    println!(
        "Generated tokens: {} in {:.1} ms ({:.2} tok/s)",
        tokens.len(),
        ms,
        tok_s
    );
    println!("token ids: {:?}", tokens);
    assert!(!tokens.is_empty(), "Tokens should not be empty");
    assert!(tokens.len() >= 16, "expected a real generation run");
}
