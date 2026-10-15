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

Reference: PrismML-Eng/llama.cpp release `prism-b10777-e3de715`, CPU execution,
greedy sampling, thinking disabled, image minimum/maximum tokens both four.
Text comparison uses the exact raw prompt in `text-prompt.txt`; image comparison
uses the same image and question through the reference chat endpoint. Image
answers are semantic smoke checks, not a token-parity claim across templates.

These short probes do not qualify tools, long context, throughput, all images,
or other platforms. See `manifest.json` for the recorded per-model results.
