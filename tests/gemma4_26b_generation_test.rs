use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[test]
fn test_gemma4_26b_metal_greedy_generation() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        return;
    }

    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER", "16");

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
        .expect("load Metal runtime");

    let mut kc = vec![Vec::new(); 30];
    let mut vc = vec![Vec::new(); 30];

    // Start with <bos> = 2
    let mut cur_tok = 2u32;
    let mut generated_tokens = Vec::new();

    println!("\n=== GENERATING 20 TOKENS (METAL GPU ACCELERATED) ===");
    for pos in 0..20 {
        let logits = runtime
            .step(cur_tok, pos, &mut kc, &mut vc)
            .expect("step metal");
        let (next_tok, max_logit) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let next_tok = next_tok as u32;
        let token_str = runtime
            .tokenizer()
            .decode(&[next_tok], false)
            .unwrap_or_default();
        println!(
            "Pos {:2}: tok={:6} logit={:8.4} | {:?}",
            pos, next_tok, max_logit, token_str
        );

        generated_tokens.push(next_tok);
        cur_tok = next_tok;
    }

    let full_text = runtime
        .tokenizer()
        .decode(&generated_tokens, false)
        .unwrap_or_default();
    println!("\nFull decoded text:\n{}", full_text);

    assert_eq!(generated_tokens.len(), 20);
    assert_eq!(generated_tokens[0], 236772, "First token is <|channel>");
    assert_eq!(generated_tokens[1], 236764, "Second token is thought");
    println!("\n>>> GENERATION TEST PASSED SUCCESSFULLY <<<");
}
