# Qwen3-4B EAGLE-3 on Windows/CUDA

This port implements learned EAGLE generation on Windows/CUDA for the pinned
Qwen3-4B Q4_K_M target and AngelSlim Qwen3-4B EAGLE-3 head. It is stacked on
#767 (`codex/qwen4b-eagle-chat`), which depends on #766.

## Execution

The BF16 checkpoint is quantized once to symmetric Q8 weights with one FP32
scale per 128 weights. These approximately 215 MiB of draft matrices remain
resident; activations, norms, and private KV use FP32. Quantization affects the
learned proposal only. The target verifies every emitted token with its existing
arithmetic. The head executes feature fusion, RMS normalization, GQA, split-half
RoPE, residual attention, SwiGLU, output normalization, the 32,000-row output
head, and draft-to-target vocabulary mapping.

Prompt seeding pairs the next token's embedding with target inputs to layers
2, 18, and 33. Only target-authoritative rows enter the head cache. CUDA offers
one learned draft when its softmax probability is at least 0.6. Otherwise it runs
one target row and updates the head from that row. Admitted proposals use two
verification rows (draft plus bonus/correction), matching the configured small
K-quant batch. This avoids paying for a second row on low-confidence proposals.
There is no suffix/ngram substitution. Metal retains its existing tree scheduler.
Metal tree width, suffix, and early-exit tuning do not widen CUDA rounds.

Q4_K and Q6_K one/two-row projections use DP4A kernels that retain the existing
ordered float accumulations without per-token integer scratch in shared memory.
One-row captures use the ordinary fused target decoder; tap buffers are reused.
A process-wide head pool reuses uploaded weights across requests and resets its
logical cache. Target KV is bounded to the selected 2048/4096 logical serving
rung before prompt capture. Unneeded final head predictions are skipped.
Capture and verification fail explicitly if the resident target becomes
unavailable; they do not switch to CPU KV mid-generation. Pipeline-sharded
capture is unsupported.

## Run

Use #767's pinned target (SHA-256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`)
and `AngelSlim/Qwen3-4B_eagle3` revision
`fd331e59626c8e95c392381a16ee59d518727fbb`.
The head directory must contain `config.json` and `model.safetensors`.

```powershell
cargo build --bin camelid
$env:CAMELID_SPEC_DECODE = 'eagle3'
$env:CAMELID_EAGLE3_MODEL = '<head-directory>'
$env:CAMELID_EAGLE3_LOGICAL_TOKENS = '2048' # default; 4096 also admitted
./target/debug/camelid.exe serve --model '<model-directory>/Qwen3-4B-Q4_K_M.gguf' --gpu on --no-open
```

Generation is greedy under the existing EAGLE API contract. CUDA/NVRTC and enough
VRAM for the target, draft matrices, and caches are required. CUDA EAGLE currently
supports this Qwen head only; it does not implement the Llama head.

## Validation and performance

On the RTX 3060 Laptop GPU, three trials over four prompts produced **1.078x
aggregate decode throughput** versus ordinary resident CUDA: 12,701 ms versus
11,777 ms. Each prompt improved when its three trials were combined (1.012x to
1.123x). All 456 output token IDs matched. These measurements exclude prompt/head
setup and use the same dev/test profiles on both sides; they are not release,
cold-start, or universal workload speed claims.

The Q8 head matched an independent NumPy Q8/128 implementation across 64,000
logits, maximum absolute error 0.00000859, with matching mapped argmax and
confidence. Q4_K and Q6_K kernel tests check bitwise agreement with the established
target paths. See the [performance receipt](../qa/validation-notes/2026-09-14-qwen3-eagle3-cuda-performance.md)
for full evidence and the earlier [baseline receipt](../qa/validation-notes/2026-09-14-qwen3-eagle3-windows.md).

## Reproduce tests

```powershell
$env:CARGO_PROFILE_DEV_DEBUG = '0'
$env:CARGO_PROFILE_TEST_DEBUG = '0'
$env:CARGO_INCREMENTAL = '0'
$env:RUST_MIN_STACK = '8388608'
cargo test --test eagle3_platform
cargo test --lib eagle3 -- --test-threads=2
cargo test --lib speculative -- --test-threads=2
cargo test --lib cuda_resident::tests::verify_batch_matches_sequential -- --ignored --exact --nocapture
cargo test --lib cuda_resident::tests::q4k_gemm_batched_matches_oracle -- --ignored --exact --nocapture
cargo test --lib cuda_resident::tests::q6k_gemm_batched_anchor_dp4a_matches_production_bitwise -- --ignored --exact --nocapture
$env:CAMELID_QWEN3_4B_GGUF = '<model-directory>/Qwen3-4B-Q4_K_M.gguf'
$env:CAMELID_EAGLE3_MODEL = '<head-directory>'
python scripts/eagle3-cuda-reference.py --q8 '<head-directory>' target/eagle3-reference.json
$env:CAMELID_EAGLE3_REFERENCE = "$PWD/target/eagle3-reference.json"
cargo test --test eagle3_cuda_head -- --ignored --nocapture
cargo test --test eagle3_cuda_target -- --ignored --nocapture
$env:CAMELID_EAGLE3_BENCH_REPEATS = '3'
$env:CAMELID_EAGLE3_REQUIRE_SPEEDUP = '1'
cargo test --test eagle3_cuda_serving -- --ignored --nocapture
```

The opt-in speed gate requires at least three trials, an improvement for each
prompt, and an aggregate gain above 5%. Hardware tests fail when their required
device/artifacts are absent; CPU fallback is not evidence. The reference generator
requires NumPy. Omitting `--q8` produces the original BF16 math reference for
comparison, not the current CUDA weight format. For a longer-context fixture,
set `CAMELID_EAGLE3_BENCH_PREFIX` to the raw contents of
`tests/fixtures/eagle3_cuda_benchmark_context.txt`.

```powershell
$env:CAMELID_EAGLE3_BENCH_PREFIX = (Get-Content tests/fixtures/eagle3_cuda_benchmark_context.txt -Raw) + "`n"
```
