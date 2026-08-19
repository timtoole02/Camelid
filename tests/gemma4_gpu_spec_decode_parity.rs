//! Speculative-decode losslessness gate for the gemma4 GPU runtime.

#[cfg(target_os = "macos")]
#[test]
fn gemma4_gpu_speculative_decode_matches_greedy_token_for_token() {
    use camelid::gemma4_runtime::Gemma4GpuRuntime;
    use std::path::PathBuf;

    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4 GPU spec-decode parity: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let rt = Gemma4GpuRuntime::load(&model, 512).expect("load gemma4 GPU runtime");

    let prompt = "Explain the theory of relativity in simple terms.";
    let max_new = 10;
    let (_, greedy_ids) = rt
        .generate_greedy(prompt, max_new)
        .expect("greedy generation");
    eprintln!("GREEDY IDS: {:?}", greedy_ids);
    let (_, spec_ids) = rt
        .generate_greedy_speculative_gpu(prompt, max_new, 5)
        .expect("speculative generation");
    eprintln!("SPEC IDS:   {:?}", spec_ids);
    assert_eq!(greedy_ids, spec_ids);
}
