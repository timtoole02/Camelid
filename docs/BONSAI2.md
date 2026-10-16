# Bonsai 2

Camelid implements Bonsai 2's `qwen35` text and single-image graph with packed
CPU and CUDA language execution. Get models lists all four official compact
packs and downloads each matching Q8_0 vision projector alongside the language
model. The existing Windows qualification covers CPU language execution; the
new CUDA path is undergoing its own benchmark, API, GUI, and state-parity checks.

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

The capacity planner keeps packed weights on the device where VRAM permits.
When the target and its caches exceed the available budget, it streams selected
trailing-layer weights from pinned host RAM while those layers still compute on
CUDA. This provides the 27B execution strategy on the 6 GiB qualification GPU;
it is not a claim that the entire 27B model fits in that VRAM. With drafting,
the drafter is initialized first so target placement accounts for both models.
Transfer overhead and available VRAM affect the resulting throughput.

Image encoding uses the matching projector backend, including accurate f32 CUDA
arithmetic for these models, with a CPU path available. The existing evidence
qualifies these four exact rows on Windows x86_64 for CPU language decoding,
single-image GUI/API chat, request sampling, and function-tool round trips.
Each row passes context recall at 1,044 prompt tokens; this does not certify
larger windows. Apple Silicon remains a separate qualification target.

Those completed results and exact source references are recorded in
[`qa/bonsai2/manifest.json`](../qa/bonsai2/manifest.json). The reproducible native
probe is `examples/bonsai2_probe.rs`. CUDA qualification is tracked separately
below; implementation alone does not extend the completed evidence boundary.

In the completed CPU comparison, the pinned Prism reference matches text,
counting and code on all four models,
and prose on 27B. The 2B/4B/9B prose runs first differ at measured near-ties
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

## CUDA qualification in progress

The CUDA implementation is undergoing native benchmark, API, GUI text/image,
and cache/rollback qualification on Windows. Completed native CUDA support or
speedup is not yet claimed here. The reproduction commands and receipt checks
are in [`qa/bonsai2/cuda/README.md`](../qa/bonsai2/cuda/README.md). Apple Silicon
will be qualified separately after Windows.

The reference side is complete using the official PrismML-Eng/llama.cpp
`prism-b10777-e3de715` Windows CUDA 12.4 package on an RTX 3060 Laptop GPU with
6 GiB VRAM and driver 576.83. Both benchmark harnesses use the same ChatML
counting, code, and prose prompts, 32 generated tokens, eight CPU threads,
2048 context slots, f16 KV, greedy sampling, an eight-token warmup, and three
measured repetitions. Rates use the median complete request duration, including
uncached prefill and excluding model loading and warmup. These timings use a
different protocol from the historical CPU table above.

The table contains **reference-port results only**, as the range across the
three prompts. Drafting uses the same Bonsai 2 2B pack and a three-token window.

| Target | Reference ordinary tok/s | Reference draft tok/s | Reference CUDA target layers, ordinary / draft |
| --- | ---: | ---: | --- |
| 2B | 205–221 | 57–86 | 25/25 / 25/25 |
| 4B | 113–118 | 47–70 | 33/33 / 33/33 |
| 9B | 77–80 | 38–62 | 33/33 / 33/33 |
| 27B | 1.68–1.76 | 0.89–1.19 | 45/65 / 31/65 |

The port's automatic fit reduces 27B target placement when it also loads the
2B drafter and speculative state; all 25 drafter layers stay on CUDA. Its
remaining target layers compute on CPU. Camelid's implemented 27B strategy
instead streams packed weights and computes those layers on CUDA. A comparison
therefore measures the complete configurations on this constrained GPU, not
just their kernel arithmetic. The same 2B drafter slowed every measured
reference case on this host.

All reference repetitions within each mode are token-identical. The port's
ordinary and draft prose outputs differ for 2B and 9B; those discrepancies are
retained in the receipts. They do not waive Camelid's requirement for matching
ordinary and draft token IDs. The final native comparison will also record
reference divergence explicitly rather than assume whole-output identity.

Publisher references: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf),
[4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), and
[27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
