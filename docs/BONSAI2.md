# Bonsai 2

Camelid implements Bonsai 2's `qwen35` text and single-image graph with packed
CPU and CUDA language execution. Get models lists all four official compact
packs and downloads each matching Q8_0 vision projector alongside the language
model. Windows CPU and CUDA support is qualified for all four exact publisher
packs. CUDA API/GUI checks, the complete native/reference benchmark, and all
fourteen final state/kernel and host memory regressions passed. The
[CUDA manifest](../qa/bonsai2/cuda/manifest.json) records the qualified artifacts;
the earlier CPU qualification remains available separately.

| Model | Language file | Vision projector | Total download |
| --- | ---: | ---: | ---: |
| Bonsai 2 2B PTQ1_0 | 429,183,456 B | 364,664,352 B | 793,847,808 B |
| Bonsai 2 4B PTQ1_0 | 949,549,920 B | 366,894,624 B | 1,316,444,544 B |
| Bonsai 2 9B PTQ1_0 | 1,997,486,976 B | 624,229,664 B | 2,621,716,640 B |
| Bonsai 2 27B PTQ1_0 | 5,946,648,928 B | 629,246,976 B | 6,575,895,904 B |

These are file sizes, not total runtime RAM requirements. The catalog ids are
`ternary_bonsai_2_2b_ptq1_0`, `ternary_bonsai_2_4b_ptq1_0`,
`ternary_bonsai_2_9b_ptq1_0`, and `ternary_bonsai_2_27b_ptq1_0`.
All four use Apache 2.0 licensed publisher artifacts. Exact model/projector
SHA256 pins live in `src/bonsai2.rs`.

## GUI images

Download a model from Get models and start it. Once the matching projector is
loaded, `/v1/health` reports `vision_ready=true` and Chat enables image attachment.
The existing interface accepts one local PNG/JPEG per request. An older Bonsai
27B projector does not substitute for a Bonsai 2 projector. Missing, truncated,
or replaced projectors leave image attachment unavailable; text still works.

For API clients, use one `image_url` content part containing a PNG/JPEG data URL
with `/v1/chat/completions`. Text and images support streaming and request
sampling. The portable image decoder has an 8192-slot capacity; the advertised
262K model-native context is not a validated Camelid context claim.

## Execution and qualification

PTQ1_0 (GGUF tensor type 143) stores 128 ternary values in a 28-byte block.
Weights remain packed; matrix operations dequantize rows transiently. Hadamard
metadata controls activation rotation, inverse embedding lookup, the shared
embedding/output heads in 2B and 4B, and grouped recurrent features. Unknown or
inconsistent transform contracts fail at load rather than silently producing
incorrect text. PQ2_0 and resolved Q2_0 layouts also use the transform-aware path,
but the download catalog and artifact identities above cover PTQ1_0 only.

The Windows CUDA language path now implements these transforms for all four
PTQ1_0 packs. Packed projections, activation rotation, attention, recurrent
layers, and feed-forward layers execute on CUDA with f32 activations and f16
attention KV storage. The graph supports ordinary greedy and sampled decoding,
multimodal prefill, and batched speculative verification. GPU selection follows
the existing GUI/server acceleration setting; a selected Bonsai 2 CUDA path
reports device/setup failures instead of silently changing to CPU execution.
The portable CPU path remains available when GPU acceleration is disabled.

Chat labels a client-measured rate as `tok/s request`: generated tokens divided
by request duration, including prefill. Verified-token bursts during drafting
are not used as a claim of device decode speed. The benchmarks below separately
define their measured interval.

The language loader shares a read-only mapping of the PTQ file across tensors
and tied heads. CUDA preparation borrows unchanged packed bytes, avoiding a
second full model allocation in private host RAM. The capacity planner keeps
packed weights on the device where VRAM permits. When the target and its caches
exceed the available budget, it streams selected trailing-layer weights from
pinned host RAM while those layers still compute on CUDA. This provides the
27B execution strategy on the 6 GiB qualification GPU; it is not a claim that
the entire 27B model fits in that VRAM. With drafting,
the drafter is initialized first so target placement accounts for both models.
Transfer overhead and available VRAM affect the resulting throughput.

Image encoding uses the matching projector backend, including accurate f32 CUDA
arithmetic for these models, with a CPU path available. Windows API/GUI checks
cover all four exact rows with CUDA language decoding: text, single-image chat,
request sampling, and function-tool round trips. The 27B GUI passes both fresh
loading and repeated model switches after the host memory fix. Each row passes
context recall at 1,044 prompt tokens; this does not certify larger windows.
Apple Silicon remains a separate qualification target.

The [historical CPU manifest](../qa/bonsai2/manifest.json) retains the earlier
CPU-language results. The [CUDA manifest](../qa/bonsai2/cuda/manifest.json) and
[reproduction workflow](../qa/bonsai2/cuda/README.md) record the completed CUDA
qualification. The reproducible native probe is `examples/bonsai2_probe.rs`.

In the completed CPU comparison, the pinned Prism reference matches text,
counting and code on all four models, and prose on 27B. The 2B/4B/9B prose runs first differ at measured near-ties
(the native choice is the reference runner-up, with log-probability margins
0.00476, 0.00241 and 0.00042). These differences are retained in the evidence;
they are not claims of whole-output reference identity.

Windows x86_64 is the first qualification target. Apple Silicon has a separate
qualification gate; a Windows result does not certify the Metal backend.

## Draft-model decoding

The Bonsai 2 CPU and CUDA text runtimes accept an independent Qwen3.5 draft
model through the existing server options:

```sh
camelid serve --model TARGET.gguf --spec-decode draft --spec-draft-model Ternary-Bonsai-2-2B-PTQ1_0.gguf --spec-draft-tokens 3
```

The draft and target must have the same token mapping. The target verifies every
proposal and remains responsible for request sampling. On CUDA, both models use
the packed PTQ1_0 graph with f32 activations. Batched verification reuses packed
weights across proposed tokens. Rejection restores device convolution and
recurrent snapshots plus the attention cache's valid-position watermark before
replaying the accepted prefix. The drafter is cached across requests, while
request state is reset; changing its file or context configuration rebuilds it.
Image requests use the ordinary multimodal decoder.

The Windows CUDA API checks retain identical 32-token seeded continuations with
and without drafting for all four targets. The
[active-drafter receipt](../qa/bonsai2/cuda/sampled-draft-runtime.json) preserves
the server log, proposal/acceptance counters, and the sequential associations
with those API requests.

The CPU drafter uses integer PTQ1_0 x Q8_0 projections; its target verification
keeps the qualified f32 arithmetic. This CPU optimization and its historical
measurements below are distinct from the CUDA implementation.

Prism's published DSpark assistants target the earlier Bonsai generation.
[The publisher's speculative-decoding guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md)
does not provide a Bonsai 2 DSpark assistant. The independent 2B draft route above
is a separate implementation. Use `qa/bonsai2/spec-benchmark.mjs` to compare its
output and elapsed time against the same target without drafting; acceptance
rate and net speed depend on the target, prompt, and hardware.

### Completed CPU measurements

Measured on Windows with an Intel i7-11800H, 16 GB RAM and eight CPU threads,
using a three-token draft window and 32 generated tokens per case. All twelve
draft runs produced exactly the same token IDs as their ordinary target runs.
The table gives **ordinary -> draft generation tokens/second**, including draft
loading and both prefills. Target loading is excluded from both measurements.

| Target | Counting | Code | Prose |
| --- | ---: | ---: | ---: |
| 2B | 1.08 -> 1.35 | 1.14 -> 1.34 | 1.1 -> 1.38 |
| 4B | 0.507 -> 0.878 | 0.541 -> 0.998 | 0.547 -> 0.392 |
| 9B | 0.297 -> 0.633 | 0.293 -> 0.641 | 0.3 -> 0.32 |
| 27B | 0.0869 -> 0.221 | 0.0808 -> 0.218 | 0.0844 -> 0.145 |

Counting and code improved 1.18-2.70x end to end. Prose ranged from 0.72x for
4B to 1.72x for 27B: low draft acceptance can make a request slower. Drafting is
therefore opt-in. These are individual measurements, not a universal speedup
or a GPU language-throughput claim. The [full timings and acceptance counts](../qa/bonsai2/performance.json)
include every case, including the slowdown. Streaming rates in the raw receipts
exclude the first visible token and are affected by verified-token bursts.

## CUDA measurements and qualification

The CUDA projection optimization is enabled by default. It specializes scalar
and batched token tiles, shares activation loads across output rows, uses grouped
packed-weight reads, and specializes dense F32/BF16 projections. It retains the
28-byte PTQ blocks and the original float contraction/reduction order. The
same-binary control is available with `CAMELID_BONSAI2_CUDA_PROJECTION=baseline`.

Nsight Systems profiles of 2B, 4B, and 9B identify projections as the main GPU
cost. In the final 2B trace, they account for 88.6% of kernel time before the
change; the vocabulary projection falls from 6.45 to 2.01 ms. The profiles include
an eight-token warmup plus the 32-token request and both prefills. They are
diagnostic traces, separate from the unprofiled request measurements below.
The [profile receipts](../qa/bonsai2/cuda-optimized/profile.json) include before/after
traces for all three resident sizes and the isolated experiments.

The pinned [Prism CUDA kernel](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10777-e3de715/ggml/src/ggml-cuda/vecdotq.cuh)
uses Q8 activations and integer dot products with a transposed activation layout.
Camelid retains its qualified float target. Independent integer-drafter
prototypes did not beat the selected float kernel on the 2B proposal-model
shapes, so they are excluded. A faster isolated weight-repacking experiment
increased packed storage by 29% and was also excluded. The port remains faster
on all measured 2B/4B/9B cases; this patch reduces the gap without reaching parity.

Measurements use Windows x86_64, an i7-11800H, 16 GB RAM, RTX 3060 Laptop 6 GiB,
CUDA 12.9 and driver 576.83. The reference is the official
[prism-b10777-e3de715 release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715),
Windows CUDA 12.4 package with installed CUDA 12.9 runtime DLLs. Both engines use
the exact checked-in counting, code and prose prompts, greedy sampling, 32 output
tokens, eight CPU threads, 2048 context slots, f16 KV, an eight-token warmup and
three measured repetitions. Drafting uses the official Bonsai 2 2B pack with a
three-token window.

All rates are **request tokens/second**, based on the median complete uncached
request duration. Loading and warmup are excluded; prefill and generation are
included. Native timing starts after tokenization, while the reference also
includes localhost HTTP, tokenization and JSON parsing. Callback-delivery
`decode_tokens_per_second` fields are diagnostics, not GPU decode-clock rates.
Ranges below cover the three prompts, not variation across repetitions.

| Model | Before ordinary | Optimized ordinary | Optimized + draft | Port ordinary | Port + draft |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 29.17–29.89 | 65.35–67.50 | 41.22–42.89 | 198.08–208.80 | 58.34–81.18 |
| 4B | 13.33–13.67 | 29.90–31.47 | 12.32–27.52 | 115.25–121.05 | 45.99–71.09 |
| 9B | 7.21–7.39 | 16.10–16.72 | 10.03–16.75 | 77.60–78.84 | 39.26–62.35 |
| 27B | 2.01–2.14 | 3.45–3.82 | 3.93–5.23 | 1.60–1.66 | 0.81–1.09 |

Ordinary 2B/4B/9B throughput improves 2.23–2.30x against the original kernel in
the same executable. 27B improves 1.61–1.90x. Drafting is optional: it slows all
2B and 4B cases, adds only 2–4% on 9B counting/code and slows 9B prose by 40%.
It improves 27B counting/code by 1.37–1.52x; the 3% prose gain is a near tie,
not evidence of a broad advantage. These short prompts do not establish a
universal speedup. The [twelve-case table](../qa/bonsai2/cuda-optimized/RESULTS.md)
and [raw comparison](../qa/bonsai2/cuda-optimized/comparison.json) retain all cases.

The first three targets keep their packed weights on CUDA in both engines.
Native 27B streams 19/64 trailing layers ordinarily and 29/64 with drafting;
all language-layer computation remains on CUDA. The port places 45/65 target
layers on GPU ordinarily and 31/65 with drafting, plus all 25/25 drafter layers.
Its remaining target layers compute on CPU. This compares complete operating
configurations on a constrained GPU, not entirely resident 27B implementations.
Native 27B is 2.15–2.39x faster ordinarily and 4.80–5.15x faster with drafting
than those corresponding port configurations.

All twelve native ordinary/draft outputs match, match the original kernel,
and match the port's ordinary output at all 32 generated tokens. Repetitions
are stable within each mode. The port's own ordinary and drafted prose output
differs for 2B and 9B; those differences are retained in the receipts.

The optimized build passed thirteen CUDA checks and two host memory regressions,
including the new bitwise legacy/tiled fixture, independent CPU references,
all four real-model cache/rollback tests, attention boundaries, and all four
models' greedy/seeded/stop/cancel draft checks. The test-only build used Rust
opt-level=1 and one job after an opt-level=3 build exhausted host memory. Probe
and server benchmarks retain opt-level=3; both builds use identical CUDA source
and NVRTC settings. The [manifest](../qa/bonsai2/cuda-optimized/manifest.json)
binds binaries, source identity and receipts.

Fresh production-GUI text/image checks passed for all four models through the
native library-server harness, using the default image settings and an
8192-slot allocation. They verify supported Windows CUDA routing, image input,
answers, and request-rate labels. The original broader API qualification
(including tools, context recall and seeded API continuations) remains in the
[initial CUDA bundle](../qa/bonsai2/cuda/README.md). These runs exercise the
production frontend/API, not an installed Tauri shell. Desktop CI is separate;
Apple Silicon qualification remains separate. Reproduce this optimization pass
with the [new workflow](../qa/bonsai2/cuda-optimized/README.md).

Publisher references: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf),
[4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), and
[27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
