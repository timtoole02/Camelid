# Qwen3-4B CUDA EAGLE performance validation

Validated September 14, 2026, after the initial BF16 implementation in
`69a7ea9e`. The original #767 branch remains the PR base and is unchanged.
The target, checkpoint revision, and SHA-256 hashes are the same as the
[baseline receipt](2026-09-14-qwen3-eagle3-windows.md).

## Result

**1.078x aggregate decode throughput over three trials**, with exact parity for
all 456 output tokens. Each of the four prompts improved when its three trials
were combined. The explicit performance gate passed: every prompt must improve,
and aggregate throughput must improve by more than 5%.

| Prompt | Output tokens per trial | Plain decode ms, three trials | EAGLE decode ms, three trials | Speedup |
| --- | ---: | ---: | ---: | ---: |
| The capital of France is | 40 | 3341 | 2985 | 1.119x |
| Write a Python function that adds two numbers | 48 | 4056 | 3812 | 1.064x |
| Count from one to ten: one, two, three, | 32 | 2666 | 2374 | 1.123x |
| Explain why water freezes when | 32 | 2638 | 2606 | 1.012x |
| Total | 152 | 12701 | 11777 | 1.078x |

Three trials ran in one process. Only the first request uploaded the head;
subsequent requests reused it with reset logical KV. Every reference forward
required ordinary resident CUDA. Timing covers decode after the first target
prediction, excluding prompt processing, checkpoint loading, quantization, and
head seeding. Timing runs were kept separate from this task's builds and other GPU tests.
Individual legs varied; the result is the combined measurement, not a claim
that every individual request is faster.

A separate longer-context run prepended
[`eagle3_cuda_benchmark_context.txt`](../../tests/fixtures/eagle3_cuda_benchmark_context.txt).
An additional newline separated the fixture from each prompt. The four prompts contained 217, 221, 224, and 218 tokens. All 152 output tokens
matched. Plain decode totaled 4533 ms; EAGLE totaled 4218 ms (1.075x), with an
improvement on each prompt. This was one trial, not the three-trial speed gate.
Together the two runs checked **608 output token IDs**.

## Why the implementation changed

The initial profile assigned about 96% of EAGLE decode time to target
verification. Unconditionally verifying two rows cost more than two ordinary
steps, and low-acceptance prompts wasted many second rows.

- Small Q4_K/Q6_K DP4A kernels remove the per-token integer scratch in shared
  memory. They keep the established ordered float accumulations. Q6_K is
  specialized for one/two rows; wider batches retain their existing dispatch.
- A learned proposal is admitted when its softmax probability is at least 0.6.
  Otherwise one target row updates the authoritative head cache. Across the
  three short trials there were 108 accepted drafts, 18 rejected drafts, and
  201 confidence-declined proposals. No suffix/ngram drafter substitutes for
  the learned head.
- Single-row captures use the ordinary fused decoder, and capture allocations
  are reused across rounds. Unneeded final head predictions are skipped.
- Draft matrices use symmetric Q8/128 weights with FP32 block scales, about
  215 MiB instead of 417 MiB. Activations and private KV remain FP32. A warp
  computes each output row, with eight rows per CTA. Only the draft weights
  are quantized; the target remains the pinned Q4_K_M model and decides every
  emitted token.

## Numerical and runtime checks

The independent NumPy reference now has an explicit `--q8` mode. It reads the
pinned BF16 checkpoint, computes the weight quantization independently, and
executes the full learned head. Across two recurrent rows, all 64,000 CUDA
logits agreed within the declared tolerance; maximum absolute error was
0.0000085831. Mapped argmax tokens (`320`, `11`) and confidence agreed. Reset plus
KV-only catch-up reproduced the stable logits, and cache overflow was rejected.
This compares CUDA to the quantized reference, not to unchanged BF16 logits.

Q4_K tests compare against resident GEMV bit-for-bit, including one/two rows at
Qwen's 2560- and 9728-column contractions and a partial final output tile. Q6_K
tests compare the specialized kernel bit-for-bit with the existing production
kernel at those dimensions, while retaining checks for the general 7/14-row
kernel. The synthetic capture fixture checks capture-on/off predictions,
caller order, embeddings, scalar/batched taps, and invalid-ID/capacity refusal.
The pinned-target test checks accepted-prefix commit, rejected-row rollback,
continuation parity, and refusal of an unconfigured wider batch.

## Environment and scope

Windows x86_64 MSVC, Rust 1.95.0; RTX 3060 Laptop GPU with 6 GiB VRAM; NVIDIA
576.83 / CUDA 12.9. Default Windows CUDA features. Dev/test symbols and
incremental compilation were disabled; `RUST_MIN_STACK=8388608`. NumPy 2.3.5.
All 36 target layers stayed on the GPU; the benchmark cache was 512 positions.

These are bounded measurements on this host, not release/package, cold-start,
4096-position, other-GPU, or universal workload claims. The weakest prompt's
measured gain is small. Commands and the explicit speed gate are documented in
[the port guide](../../docs/qwen3-eagle3-windows.md).

## Final checks

The production server ran at the default 2048-token EAGLE limit with all 36
layers resident and `cuda_resident_kquant_runtime` selected. Streaming and
nonstreaming completion/chat text matched, including the ordinary CUDA baseline
` Paris. The capital of Germany is Berlin`. Closing a stream early left the
next request correct. A one-token budget returned the expected token; a
2050-token prompt returned HTTP 413, and the following normal request recovered.
The owned test server was stopped afterward.

The numerical head, pinned-target capture/rollback, synthetic capture, Q4_K,
Q6_K, and production-executable startup checks passed. Library filters passed:
86 EAGLE tests (1 ignored), 23 speculative tests (1 ignored), 58 tree tests
(4 ignored), and 3 streaming-delta tests. These filters overlap. Strict Clippy
passed for the library, server, and all four EAGLE integration targets.
Formatting, whitespace, and public-scrub checks passed. This is not a claim that
the entire unrelated all-target CI suite was run.
