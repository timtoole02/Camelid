# Qwen3-4B EAGLE-3 on Windows/CUDA

This port implements learned EAGLE generation on Windows/CUDA for the pinned
Qwen3-4B Q4_K_M target and AngelSlim Qwen3-4B EAGLE-3 head. It is stacked on
#767 (`codex/qwen4b-eagle-chat`), which depends on #766.

## Execution

The CUDA head keeps the checkpoint's BF16 matrices resident and uses FP32
activations and private KV. It executes feature fusion, embedding/feature RMS
normalization, GQA with split-half RoPE, residual attention, SwiGLU, output
normalization, the 32,000-row output head, and draft-to-target vocabulary mapping.
Prompt seeding pairs the next token's embedding with target inputs to layers
2, 18, and 33. Only target-authoritative rows enter the head cache.

The existing Q4_K_M CUDA verifier admits two rows under its shared-memory budget.
Consequently CUDA uses one learned draft plus target bonus/correction per round
(N2/K1/X1). Rejected target rows are rolled back; accepted rows update the head.
There is no suffix/ngram substitution. Metal retains its existing tree scheduler.
Metal tree width, suffix, and draft early-exit tuning do not widen CUDA rounds.

A process-wide head pool reuses uploaded weights across requests and resets its
logical cache. Target KV is bounded to the selected 2048/4096 logical serving
rung before prompt capture; an oversized startup-warmup allocation is rebuilt.
The final single-token step uses ordinary resident CUDA. Capture and verification
fail explicitly if the resident target becomes unavailable; they do not switch
to CPU KV mid-generation. Pipeline-sharded capture is unsupported.

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
VRAM for the target plus the approximately 417 MiB head are required. CUDA EAGLE
currently supports this Qwen head only; it does not implement the Llama head.

## Validation and performance

The actual pinned head matched an independent NumPy implementation across 64,000
logits from two recurrent rows: maximum absolute error 0.00000787, identical mapped
argmax tokens. Full generation matched resident target token IDs for all 152
output tokens across four prompts, exercising 46 accepted and 54 rejected drafts
and pooled head reuse. These are bounded fixtures, not an exhaustive parity proof.

On an RTX 3060 Laptop GPU, measured debug-build decode times were 1.4-1.6 times
ordinary resident CUDA (prompt setup excluded). **This implementation does not
yet provide a speedup on that host.** Wider target verification and head/kernel
optimization remain performance work; the complete learned serving path runs.
See the [validation receipt](../qa/validation-notes/2026-09-14-qwen3-eagle3-windows.md).

## Reproduce tests

```powershell
cargo test --test eagle3_platform
cargo test --lib eagle3 -- --test-threads=2
cargo test --lib speculative -- --test-threads=2
cargo test --lib cuda_resident::tests::verify_batch_matches_sequential -- --ignored --exact --nocapture
$env:CAMELID_QWEN3_4B_GGUF = '<model-directory>/Qwen3-4B-Q4_K_M.gguf'
$env:CAMELID_EAGLE3_MODEL = '<head-directory>'
python scripts/eagle3-cuda-reference.py '<head-directory>' target/eagle3-reference.json
$env:CAMELID_EAGLE3_REFERENCE = "$PWD/target/eagle3-reference.json"
cargo test --test eagle3_cuda_head -- --ignored --nocapture
cargo test --test eagle3_cuda_target -- --ignored --nocapture
cargo test --test eagle3_cuda_serving -- --ignored --nocapture
```

The reference generator requires NumPy. Explicit GPU tests fail when their
required device/artifacts are unavailable. They never count CPU fallback as CUDA
evidence. The ordinary executable test checks startup validation outside
`cfg(test)`, guarding the original Windows production-build defect.
