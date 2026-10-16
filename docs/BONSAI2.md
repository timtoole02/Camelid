# Bonsai 2

Camelid implements Bonsai 2's `qwen35` text and single-image graph on the
portable runnable lane. Get models lists these official compact packs and
downloads each matching Q8_0 vision projector alongside the language model.

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

Language decoding uses the portable CPU path. The existing Metal/CUDA resident
language graphs do not yet implement these rotations. Image encoding can use
the existing projector backend, with CPU fallback on non-Metal hosts. These
four exact rows are supported on Windows x86_64 for text, single-image GUI/API
chat, request sampling, and function-tool round trips. The CUDA image encoder
automatically uses accurate f32 arithmetic for these models. Each row passes
context recall at 1,044 prompt tokens; this does not certify larger windows.
Apple Silicon and resident GPU language decoding require separate validation.
See the evidence manifest for measured performance and reference comparisons.

Validation results and exact source references are recorded in
[`qa/bonsai2/manifest.json`](../qa/bonsai2/manifest.json). The reproducible native
probe is `examples/bonsai2_probe.rs`.

Windows x86_64 is the first qualification target. Apple Silicon has a separate
qualification gate; a Windows result does not certify the Metal backend.

## Draft-model decoding

The portable Bonsai 2 text runtime accepts an independent Qwen3.5 draft model
through the existing server options:

```sh
camelid serve --model TARGET.gguf --spec-decode draft --spec-draft-model Ternary-Bonsai-2-2B-PTQ1_0.gguf --spec-draft-tokens 3
```

The draft and target must have the same token mapping. The target verifies every
proposal, and rejection restores attention, convolution, and recurrent state.
The target remains responsible for request sampling. The drafter uses integer
PTQ1_0 x Q8_0 projections; target verification keeps the qualified f32 arithmetic. Image requests currently
use the ordinary multimodal decoder.

Prism's published DSpark assistants target the earlier Bonsai generation.
[The publisher's speculative-decoding guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md)
does not provide a Bonsai 2 DSpark assistant. The independent 2B draft route above
is a separate implementation. Use `qa/bonsai2/spec-benchmark.mjs` to compare its
output and elapsed time against the same target without drafting; acceptance
rate and net speed depend on the target, prompt, and hardware.

Publisher references: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf),
[4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), and
[27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
