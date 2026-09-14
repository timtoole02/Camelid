# Windows Qwen3-4B learned CUDA EAGLE baseline validation

Historical receipt for the BF16 implementation in 69a7ea9e. The current Q8/128
head and faster serving path are covered by the [performance receipt](2026-09-14-qwen3-eagle3-cuda-performance.md).

Tested September 14, 2026, as a follow-up to #767 head
`7642db46e81a1465e960a5f65a3af31b0a137822`. The PR base remains
`codex/qwen4b-eagle-chat` (stacked on #766); the source branch is preserved.
This receipt supersedes the initial target-capture-only result in commit
`5806978a`.

**Result: the learned CUDA head and complete one-draft serving path execute and
pass bounded numerical, token-parity, cache, and HTTP checks. No speedup was
observed on this host.**

## Host and artifacts

- Windows x86_64 MSVC, Rust 1.95.0, default Windows CUDA features.
- NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB VRAM; driver 576.83, CUDA 12.9.
- Target: Qwen3-4B Q4_K_M, SHA-256
  `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Head: `AngelSlim/Qwen3-4B_eagle3`, revision
  `fd331e59626c8e95c392381a16ee59d518727fbb`; weights SHA-256
  `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`;
  config SHA-256
  `1fc560b1fe78e79cd31255da282651f7802bb41a8dc42c7c79e7333f036c5195`.
- Dev/test profiles, `CARGO_PROFILE_DEV_DEBUG=0`,
  `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`,
  `RUST_MIN_STACK=8388608`. NumPy 2.3.5 for independent head validation.

## Learned-head and generation evidence

The independent Python implementation reads BF16 safetensors directly and
implements all head operations in NumPy. Two recurrent rows produced 64,000
logits with maximum absolute error 0.00000787 against CUDA; both mapped target
argmax tokens matched (`320`, `11`). Reset followed by KV-only catch-up reproduced
the stable logits. Cache exhaustion is checked before another row is appended.

Full generation compared exact output token IDs with ordinary resident CUDA,
requiring GPU execution on both paths. All 152 output tokens matched. The head
was uploaded on the first request and reused on the next three; accepted and
rejected drafts both exercised target-authoritative head updates.

| Prompt | Tokens | Accepted drafts | Rejected drafts | Plain decode ms | EAGLE decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| The capital of France is | 40 | 15 | 9 | 1132 | 1628 |
| Write a Python function that adds two numbers | 48 | 14 | 19 | 1370 | 2031 |
| Count from one to ten: one, two, three, | 32 | 10 | 10 | 904 | 1260 |
| Explain why water freezes when | 32 | 7 | 16 | 892 | 1409 |

These are single-run debug-build measurements with prompt setup excluded, not a
release benchmark. EAGLE took approximately 1.4-1.6 times the plain decode time.
The existing Q4_K_M shared-memory limit permits only two verification rows, so
CUDA uses one learned draft per round. No throughput improvement is claimed.

The target-capture fixture independently checks layers `[33, 2, 18]`, finite
nonzero `[2, 2560]` residuals, acceptance plus bonus, rejection rollback, and five
subsequent resident tokens. Its greedy reference is
`[12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576]`.
Four-row Qwen verification is rejected without advancing logical KV. The
synthetic Q8 fixture separately checks capture-on/off predictions, layer-zero
embeddings, reversed layer order, multi-row versus sequential captures, and
invalid IDs/capacity. It does not widen the Qwen kernel support claim.

## Production HTTP checks

Ran the ordinary `camelid serve` executable with the pinned target/head,
`CAMELID_SPEC_DECODE=eagle3`, GPU on, and the default 2048-token EAGLE logical
limit. No special CUDA context environment override was used. Health reported
`generation_ready=true` and `cuda_resident_kquant_runtime`; all 36 target layers
were resident.

- Eight-token `/v1/completions` for `The capital of France is` returned
  ` Paris. The capital of Germany is Berlin`, exactly matching the previously
  measured ordinary CUDA response and token IDs. Streaming returned identical
  text and `[DONE]`.
- `/v1/chat/completions`, temperature zero and `camelid_enable_thinking=false`,
  answered `Paris.` identically in streaming and nonstreaming modes.
- Closed a longer chat stream after its first content chunk, then verified that
  a fresh completion reproduced the original token IDs.
- A one-token output budget returned the original first token.
- A 2050-token prompt returned HTTP 413 with
  `eagle3_context_limit_exceeded`; the next normal request again matched.
- Stopped the owned test server after validation.

## Regression and tooling checks

- Normal production server build; executable startup validation integration test.
- EAGLE library filter: 86 passed, 1 ignored.
- Speculative filter: 23 passed, 1 ignored; original suffix confidence regression
  remains unchanged.
- Tree filter: 58 passed, 4 ignored; streaming-delta filter: 3 passed.
  Substring filters overlap and are not additive suite totals.
- Explicit learned-head, target-capture, full learned-generation, and synthetic
  CUDA tests require their device/artifacts and never count fallback as success.
- Strict Clippy covers library, server, and all four EAGLE integration targets.
  The inherited EAGLE literal casts and ranking-loop warnings were cleaned up.
- Formatting, diff whitespace, and public-scrub checks.

This is not a full all-target CI, release/package, Metal, or 4096-position runtime
qualification. Metal's tree scheduler is retained; wider CUDA verification and
head optimization remain performance work. See [scope and reproduction](../../docs/qwen3-eagle3-windows.md).
