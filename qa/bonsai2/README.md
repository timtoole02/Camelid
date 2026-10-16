# Bonsai 2 smoke probes

The artifact identities are pinned in `src/bonsai2.rs`. Download the language
file and matching projector from the corresponding publisher repository. These
fixtures exercise short, greedy text and image requests with thinking disabled.
`red.png` is a generated 64x64 solid-red RGB image, not an external photograph.

```sh
cargo run --release --example bonsai2_probe -- MODEL.gguf qa/bonsai2/text-prompt.txt 16
cargo run --release --example bonsai2_probe -- MODEL.gguf qa/bonsai2/image-prompt.txt 16 MMPROJ.gguf qa/bonsai2/red.png
```

The probe uses four merged image tokens. Its output includes generated token IDs,
decoded text, and elapsed generation time. EOS is consumed rather than returned.
For encoder diagnostics, `CAMELID_PROBE_EMBEDDINGS` may name a JSON output file.

`gui-smoke.mjs` exercises an actual image attachment and the rendered streamed
answer through a dedicated test server. The `bonsai2_server` example runs the
production API without CLI startup work; set `CAMELID_DATA_DIR`,
`CAMELID_RESPONSES_DB` and `CAMELID_WORKSPACE_MEMORY_DB` to isolated test paths.

The draft benchmark runs plain and speculative decoding in separate processes:

```sh
node qa/bonsai2/spec-benchmark.mjs PROBE TARGET.gguf DRAFT.gguf RESULT.json 48
```

It records token IDs, first-token and total latency, streaming decode rate,
accepted/proposed draft counts, and before/after ratios. Token disagreement fails
the run. Streaming rate excludes time before the first visible token; total
latency includes draft loading and both prefills. Neither metric is an SLA.
Prism's official DSpark drafters target the earlier Bonsai generation; this
benchmark uses an independent Qwen3.5 model with an identical token mapping.

Reference: PrismML-Eng/llama.cpp release `prism-b10777-e3de715`, CPU execution,
greedy sampling, thinking disabled, image minimum/maximum tokens both four.
Text comparison uses the exact raw prompt in `text-prompt.txt`; image comparison
uses the same image and question through the reference chat endpoint. Image
answers are semantic smoke checks, not a token-parity claim across templates.

These short probes do not qualify tools, long context, throughput, all images,
or other platforms. See `manifest.json` for the recorded per-model results.
