//! Permanent QA Gate: Gemma 4 MoE Fail-Closed & Routed Expert Execution.
//!
//! Asserts that:
//! 1. An MoE model refuses fail-closed in `Gemma4GpuRuntime::load` (never enters dense mode).
//! 2. An MoE model with `.cghost` correctly records and executes `routed_expert_calls > 0`.

mod support;

#[cfg(target_os = "macos")]
use camelid::gemma4_runtime::Gemma4GpuRuntime;
use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
#[test]
fn gemma4_moe_fails_closed_in_dense_gpu_runtime() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    if !model_path.is_file() {
        eprintln!(
            "SKIP: 26B MoE test model not found at {}",
            model_path.display()
        );
        return;
    }

    // Dense GPU runtime MUST refuse MoE models fail-closed
    let result = Gemma4GpuRuntime::load(&model_path, 512);
    assert!(
        result.is_err(),
        "Gemma4GpuRuntime::load must fail closed when given an MoE model without experts"
    );
    let err_msg = result.err().unwrap().to_string();
    assert!(
        err_msg.contains("MoE") || err_msg.contains("Ghost-MoE"),
        "Expected MoE refusal error message, got: {err_msg}"
    );
}

#[test]
fn gemma4_moe_executes_routed_experts() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");
    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 1024, false)
        .expect("load ghost moe");

    let prompt = "The capital of France is";
    let (text, tokens) = runtime.generate_greedy(prompt, 5).expect("generate");

    eprintln!("[QA-MoE] generated text: {:?}, tokens: {:?}", text, tokens);
    assert!(!tokens.is_empty(), "Must generate tokens");
    assert_eq!(tokens[0], 9079, "First token must be 9079 (' Paris')");

    let stats = runtime
        .ghost_moe_cache_stats()
        .expect("stats must be available for MoE");
    let total_calls = stats.hits + stats.misses;
    assert!(
        total_calls > 0,
        "MoE model generation must execute routed expert calls; found total_calls = 0"
    );
    eprintln!(
        "[QA-MoE] routed expert stats: hits={}, misses={}, total_calls={}, bytes_read={}",
        stats.hits, stats.misses, total_calls, stats.bytes_read
    );
}
