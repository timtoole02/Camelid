# Bonsai 2 Windows qualification

The artifact identities are pinned in `src/bonsai2.rs`. Download the language
file and matching projector from the corresponding publisher repository. These
fixtures exercise text, single-image requests, tools and sampling with thinking disabled.
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

The benchmark records token IDs, first-token and total latency, streaming decode
rate, accepted/proposed drafts, and before/after ratios. Token disagreement fails
the run. Generation latency includes draft loading and both prefills; target
model loading happens before the timer in both runs. Streaming rate excludes
time before the first visible token and can be inflated by verified token bursts.
Use total generation throughput for the net comparison. These are individual
32-token measurements on one host, not a throughput guarantee.

The API qualification harness checks text, a function call and its result,
context recall, sampled generation and spatial vision through the production endpoint:

```sh
node qa/bonsai2/api-qualification.mjs MODEL.gguf RESULT.json http://127.0.0.1:28182 text,tools,context,sampling,vision
```

For a GUI run that retains normal image sizing, add `--send --default-image`
to `gui-smoke.mjs`. Output length is bounded to 16 tokens; the actual attachment,
request construction, streaming renderer, and image limits remain in use.
`blue-yellow.png` tests spatial color recognition at default sizing; regenerate
it with `node qa/bonsai2/make-vision-fixture.mjs`.

Reference: PrismML-Eng/llama.cpp release `prism-b10777-e3de715`, CPU execution,
greedy sampling, thinking disabled, image minimum/maximum tokens both four.
Text comparison uses the exact raw prompt in `text-prompt.txt`; image comparison
uses the same image and question through the reference chat endpoint. Image
answers are semantic smoke checks, not a token-parity claim across templates.

The API pack also qualifies function-tool round trips, sampled generation and
context recall at 1,044 prompt tokens. Add `--require-supported` to the GUI
harness to assert the Windows support badge. See `manifest.json` for per-model
results; larger context and other platforms remain separate qualification work.

The final GUI pass uses the built backend's frontend on port 28182:

```sh
node qa/bonsai2/gui-smoke.mjs MODEL.gguf RESULT_PREFIX http://127.0.0.1:28182 --send --default-image --require-supported
```

`image-lanes.json` retains the encoder-selection logs. CPU fallback was forced
with `CAMELID_CUDA_DEVICE=99` on the one-GPU qualification host. Normal CUDA runs
used device 0 with no strict-math override; Bonsai 2 selects accurate arithmetic
automatically. `cache-parity.json` records the real-file batched/serial logits,
attention cache, convolution state and recurrent state comparison.

Validate the committed receipts without downloading models:

```sh
node qa/bonsai2/check-evidence.mjs
```

This checks all API cases, supported GUI receipts, three draft comparisons per
size, model identities, and the ordinary-versus-draft seeded API continuation.
It validates the evidence bundle; use the commands above to rerun inference.
