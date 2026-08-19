//! Parity test for genuine Gemma 4 E4B Q4_0 against reference

use camelid::gemma4_runtime::{Gemma4GpuRuntime, Gemma4Runtime};
use std::path::PathBuf;

#[test]
fn test_genuine_gemma4_e4b_q4_0_parity() {
    let model_path = PathBuf::from("/Volumes/Untitled/models/gemma-4-E4B_q4_0-it.gguf");
    if !model_path.is_file() {
        eprintln!("SKIP: gemma-4-E4B_q4_0-it.gguf not found");
        return;
    }

    println!("Loading Gemma 4 E4B Q4_0 CPU runtime...");
    let cpu_runtime = Gemma4Runtime::load(&model_path).expect("load cpu runtime");

    let prompt = "<|turn>user\nHello<turn|>\n<|turn>model\n";
    let prompt_tokens = cpu_runtime
        .tokenizer()
        .encode(prompt, true, true)
        .expect("encode prompt");
    println!("Prompt tokens: {:?}", prompt_tokens);

    let (cpu_text, cpu_tokens) = cpu_runtime.generate_greedy(prompt, 64).expect("cpu greedy");
    println!(
        "CPU Generated {} tokens: {:?}",
        cpu_tokens.len(),
        cpu_tokens
    );
    println!("CPU Decoded Text:\n{}", cpu_text);

    #[cfg(target_os = "macos")]
    {
        println!("\nLoading Gemma 4 E4B Q4_0 Metal GPU runtime...");
        let gpu_runtime = Gemma4GpuRuntime::load(&model_path, 2048).expect("load gpu runtime");
        let (gpu_text, gpu_tokens) = gpu_runtime.generate_greedy(prompt, 64).expect("gpu greedy");
        println!(
            "GPU Generated {} tokens: {:?}",
            gpu_tokens.len(),
            gpu_tokens
        );
        println!("GPU Decoded Text:\n{}", gpu_text);

        assert_eq!(
            cpu_tokens, gpu_tokens,
            "CPU and GPU token IDs must match exactly"
        );
        println!("PASS: CPU and GPU match 100% bit-exactly on genuine Gemma 4 E4B Q4_0!");
    }
}
