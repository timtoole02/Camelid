# Qwen3 4B learned speculative chat on mini2

Camelid's opt-in EAGLE-3 chat lane reaches median **52.26 tok/s for prose, 64.75 for code, and 68.11 for JSON** on mini2 (Apple M4, 16 GiB). These are actual `/v1/chat/completions` SSE measurements, including drafting, target verification, and orchestration after the first content token. **100 tok/s was not achieved.**

Three fresh server pairs used the same binary, target model, kernel flags, requests, and 256-token cap. Prose ended naturally at 162 tokens; code and JSON reached the cap. All nine speculative streams matched their paired ordinary stream's text and usage. A separate non-streaming pair matched every generated token ID on all three prompts.

| Prompt | Matched plain tok/s | EAGLE tok/s | EAGLE request tok/s, including startup work | EAGLE first content |
| --- | ---: | ---: | ---: | ---: |
| Prose | 25.24 | 52.26 | 46.39 | 409 ms |
| Python merge function | 25.09 | 64.75 | 58.58 | 430 ms |
| JSON books | 25.11 | 68.11 | 61.29 | 432 ms |

Headline rates are `(completion_tokens - 1) / (generate_seconds - first_content_seconds)`, using tokenizer usage, not SSE chunk count. End-to-end rates use completion tokens divided by client request wall time. The server warms the draft head before declaring generation readiness: its initial head upload/bootstrap took about 1.15 seconds. The table's first-content measurements use that warm head, with a new request's prompt processing and head seeding included. Recorded startup-to-ready times include model loading and server warmup.

The faster ordinary V3 lane measured approximately 31.6–32.6 tok/s in the initial 128-token experiment. The 25 tok/s matched control above uses the matrix-oriented kernel profile needed by the faster verifier. Therefore the result is roughly twice the earlier best ordinary coding decode rate, rather than a claim that all ordinary decode used to run at 25 tok/s. The 128-token and 256-token measurements are separate experiments.

## Implementation

- Validate the pinned Qwen3-4B EAGLE checkpoint and target pairing. The draft residual width is 2560, while its 32 query heads have width 4096; projections, buffers, RoPE, normalization, and feature capture now honor that asymmetric geometry. Llama geometry remains supported.
- Capture target layer inputs `[2, 18, 33]` according to the pinned upstream Qwen inference implementation. Maintain target-authoritative verification and rebuild/catch up draft state from accepted target activations.
- Add a bounded learned tree with suffix-confidence admission, a reusable uploaded draft head, per-request cache reset, and target verification through Metal batching and fused operations.
- Stream committed text incrementally while withholding partial stop strings. Preserve whole-step byte-exact fallback when per-token decoding cannot safely reproduce the step.
- Record head bootstrap, drafting, target verification, and head update timings. The harness records model/binary hashes, every flag, requests, responses, and parity failures.

The selected profile uses eight verifier nodes, top four draft candidates, at most five expansions, early-exit threshold 0.24, and Q4 draft body/output head. It is exposed through the benchmark's `qwenmma-eagle3-15` arm and remains opt-in. The `15` is the maximum draft allowance for suffix rounds, not the learned tree width.

## What limits speed

For the first selected-profile code run, 256 tokens used 98 verification rounds: 3.406 seconds in target verification, 0.351 seconds drafting, and 0.158 seconds updating the draft head. Target verification dominates the measured phase time. Increasing expansion depth or tree width did not pay for itself.

The following are single tuning runs, with the same 256-token requests and matched plain reference. Every run retained text and usage parity:

| Variant | Prose tok/s | Code tok/s | JSON tok/s |
| --- | ---: | ---: | ---: |
| Selected Q4, N8/K4/X5, early exit 0.24 | 52.26 | 64.75 | 68.11 |
| Disable early exit, N8/K4/X5 | 47.35 | 64.18 | 64.75 |
| N16/K4/X8, no early exit | 20.92 | 30.67 | 33.82 |
| Q8 body and output head | 49.86 | 63.40 | 63.97 |
| Q4 body, Q8 output head | 50.69 | 62.79 | 66.54 |
| BF16 body and output head | 43.70 | 52.54 | 54.17 |

The full receipt also includes N8/K4/X8 without early exit and the slower initial target-kernel profiles. Further work toward 100 tok/s should reduce verifier cost per accepted token or increase accepted tokens per round without proportionally increasing target work. These measurements do not establish general coding correctness, tool-use quality, or agent task completion speed.

## Artifacts and reproduction

[Machine-readable receipt](receipts/qwen4b-eagle-chat-20260913.json) includes all experiment summaries, configurations, phase logs, exact-token witnesses, and hashes of raw responses/logs. Full raw runs remain under `target/qwen-eagle/` in the mini2 worktree; no user chat history or project data is included.

- Target SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- [AngelSlim Qwen3-4B EAGLE checkpoint](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3/tree/fd331e59626c8e95c392381a16ee59d518727fbb), revision `fd331e59626c8e95c392381a16ee59d518727fbb`.
- Head weights SHA-256: `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
- Head config SHA-256: `1fc560b1fe78e79cd31255da282651f7802bb41a8dc42c7c79e7333f036c5195`.
- [Pinned target capture implementation](https://github.com/Tencent/AngelSlim/blob/7fa71bc2fd4d7f9e28aa2e5c04dfe907bf2ac8e0/angelslim/compressor/speculative/inference/models/eagle3/target/modeling_qwen3_kv.py).
- Measured binary SHA-256: `65bcd751083c59163002b7de7df625b59bd5f141c29bf3ba500059f1f3197d6a`; release build with Rust 1.95, LTO off and 16 codegen units. Final source differs from that binary only in comments and harness error wording.

Run only on mini2, sequentially, with other model servers stopped. Build with `CARGO_BUILD_JOBS=1`. Replace the binary/model locations as necessary:

```sh
/usr/bin/python3 scripts/bench-qwen4b-metal-spec.py \
  --binary target/qwen-eagle/camelid-candidate \
  --model /workspace/camelid/models/Qwen3-4B-Q4_K_M.gguf \
  --eagle-model /workspace/camelid/models/Qwen3-4B-Eagle3-AngelSlim-fd331e59 \
  --arms qwenmma,qwenmma-eagle3-15 \
  --cases prose,code,json --max-tokens 256 --stream \
  --env CAMELID_BENCH_EAGLE3_AUTHORITATIVE_CB_FUSION=1 \
  --env CAMELID_BENCH_EAGLE3_REPLAY_SCATTER=1 \
  --out target/qwen-eagle/new-stream-pair
```

Omit `--stream` and choose a new output directory for exact generated-token-ID comparison. The harness refuses occupied preview/comparison ports, runs one server at a time, and terminates only its owned server.

## Validation and scope

All builds, tests, and inference ran on mini2. Passed: loader/schema (12), real pinned checkpoint load (1), draft runtime (34), serving lifecycle/budgets (9), suffix confidence (14), stop-prefix helpers (4), stream splitting/stop boundaries (3), and the batch-admission guard (1). Final `cargo fmt --check` passed. The initial stream test expected one delta; its assertion was corrected to accept the two safe stop-prefix chunks and its final rerun passed. Test results and that superseded failure are recorded with the receipt. The independent CPU/GPU Qwen cell proof uses the original BF16 head over three positions and checks feature fusion, GQA attention, recurrent hidden state, exact draft argmax, and target mapping; worst hidden relative L2 was approximately `4.05e-5` against a `0.002` threshold.

The verifier proof also passed with prefetch/w32/128-thread attention, batched RoPE/scatter, attention reuse, and batch argmax enabled: 36 Q/K glue encodes and 40 batched RoPE/scatter encodes, with exact logits and captures. Synthetic Metal regression checks cover Qwen-style per-head Q/K normalization with group-four attention, batch glue across attention boundaries, prefix-cache pending graph cleanup, and matrix-prefill continuation. Those tests complement the real-model short-chat parity; they do not qualify all long-context behavior.

The default logical prompt-plus-output envelope is 2048 tokens. A 4096-token operator option exists, but the real-model speed/parity results here use short prompts; native model context is not qualified by this report. Non-greedy sampling, broad agent tasks, and experimental runtime/kernel configurations beyond the selected profile are outside these measurements. The patch contains substantial shared Metal/runtime changes with retained gated diagnostic paths; those require broader regression review before promotion. Normal serving defaults and the original hosted preview are preserved.
