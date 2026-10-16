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

All four Windows CUDA production GUI/API checks through the native
library-server harness, seeded draft comparisons, and native benchmarks have
passed. The final rebuilt suite also passed all twelve CUDA tests and both
host memory regressions, recorded in the [cache and memory receipt](../qa/bonsai2/cuda/cache-parity.json).
The [CUDA manifest](../qa/bonsai2/cuda/manifest.json) binds the qualified
artifacts, and the [reproduction workflow](../qa/bonsai2/cuda/README.md) lists
the commands and required receipts. These model runs did not execute the
installed Tauri desktop shell; desktop shell CI is separate. Apple Silicon
qualification remains separate.

Measurements use an Intel i7-11800H, 16 GB RAM, an RTX 3060 Laptop GPU with
6 GiB VRAM, and driver 576.83. The reference is the official PrismML-Eng/llama.cpp
[`prism-b10777-e3de715`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715)
Windows CUDA 12.4 package, using installed CUDA 12.9 runtime
DLLs. Both harnesses use the same ChatML counting, code, and prose prompts,
32 generated tokens, eight CPU threads, 2048 context slots, f16 KV, greedy
sampling, an eight-token warmup, and three measured repetitions. Rates use the
median complete request duration, including uncached prefill and excluding
model loading and warmup. Native timing begins after tokenization at the Rust
generation call; reference timing also includes localhost HTTP, tokenization,
and JSON response overhead. These timings use a different protocol from the
historical CPU table above.

The raw native `decode_tokens_per_second` and `decode_speedup` fields describe
callback delivery timing, which can be bursty on CUDA. They are diagnostics,
not GPU decode-clock measurements, and are excluded from this comparison.
All rates below are **request tokens/second**. Drafting uses the same Bonsai 2
2B pack and a three-token window. A speedup below 1.00x means drafting is slower.

| Target | Prompt | Camelid ordinary | Camelid draft | Camelid draft speedup | Port ordinary | Port draft |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2B | Counting | 29.50 | 22.27 | 0.75x | 205.24 | 86.05 |
| 2B | Code | 29.31 | 22.29 | 0.76x | 220.95 | 80.08 |
| 2B | Prose | 29.87 | 22.90 | 0.77x | 207.75 | 57.03 |
| 4B | Counting | 13.35 | 16.05 | 1.20x | 117.87 | 70.35 |
| 4B | Code | 13.27 | 16.02 | 1.21x | 116.71 | 66.51 |
| 4B | Prose | 13.62 | 6.17 | 0.45x | 112.59 | 46.55 |
| 9B | Counting | 7.14 | 11.36 | 1.59x | 76.97 | 61.50 |
| 9B | Code | 7.17 | 11.41 | 1.59x | 77.58 | 55.38 |
| 9B | Prose | 7.38 | 5.71 | 0.77x | 79.53 | 37.85 |
| 27B | Counting | 2.06 | 4.50 | 2.19x | 1.74 | 1.19 |
| 27B | Code | 2.10 | 4.36 | 2.08x | 1.68 | 1.17 |
| 27B | Prose | 2.12 | 2.92 | 1.38x | 1.76 | 0.89 |

Drafting improves counting/code by about 20% on 4B and 59% on 9B. It improves
all three 27B requests by 1.38–2.19x. It slows every 2B request by 23–25%, even
with full proposal acceptance, and slows 4B prose by 55% and 9B prose by 23%.
Drafting remains opt-in; these three prompts do not establish a universal gain.
The port is faster for every measured 2B/4B/9B case. On the tested 27B
configuration, Camelid is 1.18–1.25x faster in ordinary mode and 3.29–3.78x faster
in draft mode. The entry-point difference is retained in the protocol; these
measurements do not isolate the causes of the performance gap.

The first three sizes keep their packed target weights on CUDA in both engines.
For 27B, Camelid streams weights for 19/64 trailing layers ordinarily and 29/64
with drafting, while all layer computation stays on CUDA. The port places
45/65 target layers on CUDA ordinarily and 31/65 with drafting; its remaining
layers compute on CPU. Its entire 25-layer 2B drafter stays on CUDA. These counts
use each engine's own layer convention. Automatic placement accounts for the
drafter and speculative state. The comparison measures those complete
configurations on this constrained GPU; it does not predict all-resident 27B
performance on a larger GPU.

All twelve native ordinary/draft outputs match exactly, and each ordinary
output matches the pinned port baseline at all 32 generated tokens. Repeated
outputs are stable within each mode. The port's own ordinary and draft prose
outputs differ for 2B and 9B; those discrepancies remain in the receipts.
[Complete rates, placement excerpts, and parity results](../qa/bonsai2/cuda/comparison.json)
retain every case, including slowdowns. These are bounded test results, not a
claim of whole-output identity for arbitrary prompts.

Publisher references: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf),
[4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), and
[27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
