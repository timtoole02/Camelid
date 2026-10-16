# Bonsai 2

Camelid implements Bonsai 2's `qwen35` text and single-image graph with packed
CPU and CUDA language execution. Get models lists all four official compact
packs and downloads each matching Q8_0 vision projector alongside the language
model. Windows CPU and CUDA support is qualified for all four exact publisher
packs. CUDA API/GUI checks, the complete native/reference benchmark, and all
sixteen final state/kernel and host memory regressions passed. The
[Q8 CUDA manifest](../qa/bonsai2/cuda-q8/manifest.json) records the qualified artifacts;
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
Weights remain packed. CPU operations dequantize rows transiently; CUDA integer
projections decode packed digits during contraction. Hadamard
metadata controls activation rotation, inverse embedding lookup, the shared
embedding/output heads in 2B and 4B, and grouped recurrent features. Unknown or
inconsistent transform contracts fail at load rather than silently producing
incorrect text. PQ2_0 and resolved Q2_0 layouts also use the transform-aware path,
but the download catalog and artifact identities above cover PTQ1_0 only.

The Windows CUDA language path now implements these transforms for all four
PTQ1_0 packs. Packed projections, activation rotation, attention, recurrent
layers, and feed-forward layers execute on CUDA. Hidden states use f32, packed
PTQ projections quantize rotated inputs to Q8, and attention KV uses f16.
The graph supports ordinary greedy and sampled decoding,
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
request sampling, and function-tool round trips. The initial CUDA qualification
checked 27B fresh loading and repeated model switches after the host memory fix;
the Q8 qualification repeats model switching across all four sizes. Each row passes
context recall at 1,044 prompt tokens; this does not certify larger windows.
Apple Silicon remains a separate qualification target.

The [historical CPU manifest](../qa/bonsai2/manifest.json) retains the earlier
CPU-language results. The [Q8 CUDA manifest](../qa/bonsai2/cuda-q8/manifest.json) and
[reproduction workflow](../qa/bonsai2/cuda-q8/README.md) record the completed CUDA
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
the packed PTQ1_0 graph with Q8 projection inputs. Batched verification reuses packed
weights across proposed tokens. Rejection restores device convolution and
recurrent snapshots plus the attention cache's valid-position watermark before
replaying the accepted prefix. The drafter is cached across requests, while
request state is reset; changing its file or context configuration rebuilds it.
Image requests use the ordinary multimodal decoder.

The initial Windows CUDA API checks retain identical 32-token seeded continuations with
and without drafting for all four targets. The
[active-drafter receipt](../qa/bonsai2/cuda/sampled-draft-runtime.json) preserves
the server log, proposal/acceptance counters, and the sequential associations
with those API requests.
The Q8 hardware tests below repeat greedy and seeded draft parity for all four
real models.

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

The default CUDA PTQ projection now uses transposed Q8 activations and DP4A
integer dot products. Each group of 32 rotated inputs stores a half scale and
an exact signed sum; the sum corrects the unsigned ternary digit bias. Coalesced
activation planes, a fused Hadamard/quantization launch, and four-token tiles
reduce projection cost without repacking the original 28-byte weights. Hidden
states, attention, recurrent updates, and dense F32/BF16 gates retain their
previous arithmetic. Scratch is preallocated before CUDA graph capture.

The pinned [Prism CUDA reference](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10777-e3de715/ggml/src/ggml-cuda/vecdotq.cuh)
informed the layout and exact-sum method; attribution is in THIRD_PARTY_NOTICES.
Q8 changes activation arithmetic. The previous float policy remains available
with `CAMELID_BONSAI2_CUDA_PROJECTION=tiled`; `baseline` selects the original
generic float kernel. `q8` explicitly selects the default. Setting
`CAMELID_BONSAI2_CUDA_Q8_UNFUSED=1` separates rotation and quantization for
diagnostics while preserving the quantized bytes.

Six final Nsight Systems traces cover 2B, 4B, and 9B under both float and Q8
policies. Projections account for 75.26–83.00% of float kernel
time and 49.51–60.90% with Q8. Each trace includes an eight-token
warmup plus the 32-token request and both prefills. These diagnostic profiles
are separate from the unprofiled request measurements below; the
[kernel summaries](../qa/bonsai2/cuda-q8/profile.json) retain all six traces.

Measurements use Windows x86_64, an i7-11800H, 16 GB RAM, RTX 3060 Laptop 6 GiB,
CUDA 12.9 and driver 576.83. The reference is the official
[prism-b10777-e3de715 release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715),
Windows CUDA 12.4 package with installed CUDA 12.9 runtime DLLs. Both engines use
the exact checked-in counting, code and prose prompts, greedy sampling, 32 output
tokens, eight CPU threads, 2048 context slots, f16 KV, an eight-token warmup and
three measured repetitions. Drafting uses the official Bonsai 2 2B pack with a
three-token window.

All rates are **request tokens/second**, based on the median complete uncached
request duration. Loading, CUDA compilation, and warmup are excluded; prefill
and generation are included. Native timing starts after tokenization, while
the reference also includes localhost HTTP, tokenization and JSON parsing.
Each native prompt/mode now retains one loaded process across its three uncached
requests, resetting request KV/recurrent state. The prior bundle used one
process per repetition, so speedup ratios here use the fresh float control in
the same final executable. The port also reuses its loaded server.
Callback-delivery rates remain diagnostic fields, not GPU decode-clock rates.
Ranges below span the three prompts, not variation across repetitions.

| Model | Float ordinary | Q8 ordinary | Q8 + draft | Port ordinary | Port + draft |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 64.73–66.57 | 141.28–145.74 | 78.90–82.32 | 177.83–194.66 | 52.92–79.22 |
| 4B | 29.85–31.10 | 61.87–66.49 | 22.41–47.13 | 109.42–118.53 | 45.65–69.82 |
| 9B | 16.05–16.77 | 39.68–39.72 | 23.62–39.64 | 64.72–68.57 | 28.37–42.46 |
| 27B | 3.22–3.98 | 5.10–6.03 | 4.55–8.07 | 1.46–1.66 | 0.80–1.10 |

Ordinary 2B/4B/9B throughput improves 2.07–2.47x against the fresh optimized float
control. The port's remaining lead is 1.26–1.92x, down from the earlier
3–5x gap. Ordinary 27B improves 1.28–1.66x. Drafting slows
all measured 2B and 4B requests. On 9B, counting/code are near ties while prose
is slower. On 27B, drafting changes throughput by 0.89–1.51x.
These short prompts do not establish a universal draft advantage, so drafting
remains opt-in. The [twelve-case table](../qa/bonsai2/cuda-q8/RESULTS.md) includes
both float draft controls and all Q8/reference measurements.

The first three targets keep packed weights on CUDA in both engines. Native
27B streams 19/64 trailing layers ordinarily and 29/64 with drafting;
all language-layer computation remains on CUDA. The port places
45/65 target layers on GPU ordinarily and 31/65 with drafting, plus the full
25/25-layer drafter. Its remaining target layers compute on CPU. This compares
complete configurations on a constrained GPU, not entirely resident 27B engines.
The [raw comparison](../qa/bonsai2/cuda-q8/comparison.json) retains actual placement.

All twelve native ordinary/draft outputs match within each policy, match the
float control, and match the port's ordinary output at all 32 generated tokens.
Every repetition is stable within its mode. The port's own ordinary/drafted
prose output differs for 2B and 9B; those differences remain in the receipts.

The final build passes fourteen CUDA checks and two host memory regressions.
Coverage includes independently calculated Q8 bytes/integer contractions,
fused/separate quantization, scalar/batch identity, partial tiles, independent
float references, all four real-model cache/rollback tests, attention boundaries,
and greedy/seeded/stop/cancel draft behavior. Fixed-continuation scoring checks
96 tokens per model under both policies: all 384 top choices agree, with a
maximum mean NLL increase of 0.000734 nats/token. This is a bounded numerical
regression check, not broad task-quality or corpus-perplexity qualification.

An initial 27B state test overlapped release compilation and failed a 240 MiB
host allocation. The complete serial rerun after compilation passed all sixteen
checks. The test-only Rust build uses opt-level=1 and one job; probe/server use
opt-level=3, with identical CUDA source and NVRTC settings. The
[manifest](../qa/bonsai2/cuda-q8/manifest.json) binds source identity, retained
binaries, build settings, and receipts.
The measured runtime is commit `d86878671f35301a3f84584405f0fa7431219612`;
the subsequent evidence commit updates QA, documentation, and capability
evidence links without changing inference code.

Fresh production-GUI text/image and broader API checks pass for all four models
with the default Q8 policy, supported Windows CUDA routing, default image
settings, and an 8192-slot allocation. API checks cover text, tool-call/result
round trips, 1,044-token context recall, seeded sampling, and spatial vision.
GUI checks verify attachment, answers, supported labels, and request-rate labels.
These runs exercise the production frontend/API through the native library-server
harness; installed Tauri execution and Apple Silicon qualification are separate.
The [reproduction workflow](../qa/bonsai2/cuda-q8/README.md) describes the checks.
The [earlier float optimization](../qa/bonsai2/cuda-optimized/README.md) and
[initial CUDA qualification](../qa/bonsai2/cuda/README.md) remain historical evidence.

Publisher references: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf),
[4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), and
[27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
