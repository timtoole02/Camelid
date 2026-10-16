# Bonsai 2 Windows qualification

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
`CAMELID_PROBE_IMAGE_MIN_TOKENS` and `CAMELID_PROBE_IMAGE_MAX_TOKENS` override
the probe's image sizing without changing its requested output-token limit.

`reference-text.mjs` captures four 32-token cases from the pinned reference
server. `compare-reference.mjs` compares native token IDs with those captures
and retains mismatches in its report. For a divergent case, `reference-score.mjs`
captures the reference's top 20 token log probabilities along the same greedy
continuation; compare the first divergent position, where prefixes still agree.
This distinguishes a measured close decision from an unexamined disagreement.

`gui-smoke.mjs` exercises an actual image attachment and the rendered streamed
answer through a dedicated test server. The `bonsai2_server` example runs the
production API without CLI startup work; set `CAMELID_DATA_DIR`,
`CAMELID_RESPONSES_DB` and `CAMELID_WORKSPACE_MEMORY_DB` to isolated test paths.

The draft benchmark runs plain and speculative decoding in separate processes:

```sh
node qa/bonsai2/spec-benchmark.mjs PROBE TARGET.gguf DRAFT.gguf RESULT.json 48
```

Set `CAMELID_BENCH_CASES=count,code,explain` to select cases and
`CAMELID_SPEC_DRAFT_TOKENS` to select the draft window. The default window is 3.
Build the probe from the same source tree whose hashes the report records.
Run timing comparisons with other model servers and compiler jobs stopped.
`CAMELID_BENCH_REFERENCE` can name a reference-text capture for the same target;
the report separately records agreement with its available token prefix.

The API qualification harness checks text, a function call and its result,
context recall, and sampled generation through the production endpoint:

```sh
node qa/bonsai2/api-qualification.mjs MODEL.gguf RESULT.json http://127.0.0.1:28182 text,tools,context,sampling,vision
```

For a GUI run that retains normal image sizing, add `--send --default-image`
to `gui-smoke.mjs`. Output length is bounded to 16 tokens; the actual attachment,
request construction, streaming renderer, and image limits remain in use.
`blue-yellow.png` tests spatial color recognition at default sizing; regenerate
it with `node qa/bonsai2/make-vision-fixture.mjs`.

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

The API pack also qualifies function-tool round trips, sampled generation and
context recall at 1,044 prompt tokens. Add `--require-supported` to the GUI
harness to assert the Windows support badge. See `manifest.json` for per-model
results; larger context and other platforms remain separate qualification work.
