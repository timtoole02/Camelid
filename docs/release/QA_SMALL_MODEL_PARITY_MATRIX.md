# QA Small-Model Parity Matrix

Last updated: 2026-05-24

> [!NOTE]
> This matrix is a QA evidence summary, not the public support ledger. For current support truth,
> use [`COMPATIBILITY.md`](../../COMPATIBILITY.md), [`STATUS.md`](../../STATUS.md), and the owner matrix in
> [`FULL_SUPPORT_BLOCKER_MATRIX.md`](../../FULL_SUPPORT_BLOCKER_MATRIX.md).

## Scope

This matrix summarizes the four currently relevant Q8_0 rows without turning partial evidence into
full-support language:

- TinyLlama 1.1B Chat Q8_0
- Llama 3.2 1B Instruct Q8_0
- Llama 3.2 3B Instruct Q8_0
- Llama 3 8B Instruct Q8_0

## Matrix

| Target | Quant | Current QA position | Prompt-token parity | First-token parity | Short generation parity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat | Q8_0 | Supported gate evidence is green | PASS | PASS | PASS | Matches known-good llama-server on the active TinyLlama gate. Keep this as the release anchor and refresh artifacts when packaging the four-row evidence set. |
| Llama 3.2 1B Instruct | Q8_0 | Verified bounded support | PASS for compact-header prompt, broader prompt pack, and checked bounded context packs | PASS | PASS for compact and broader short-generation packs plus checked 512/1024/2048/4096/8192 context packs | Exact 1B Instruct Q8_0 support is bounded to the documented envelopes; the checked 512/1024/2048/4096/8192 context packs are green, with 2048 exact-row only after the RoPE frequency-factor fix and 4096/8192 tied to their cited source/runtime-head bundles, while model-native/larger context beyond checked packs, stronger memory/perf, portability, and broader template coverage remain expansion gates. |
| Llama 3.2 3B Instruct | Q8_0 | Supported exact-row smoke | PASS for compact-header prompt, post-Q8-dot broader prompt pack, and checked bounded context packs | PASS | PASS for compact 1/5/50-token, broader 3-prompt/50-token, and checked 512/1024/2048 context packs | The previous JSON-shaped broader prompt blocker is fixed for the current pack, and checked 512/1024/2048 context packs are green. The row remains limited to exact-row smoke until model-native/larger context beyond checked packs, memory/perf, portability, and broader template evidence land. |
| Llama 3 8B Instruct | Q8_0 | Verified bounded support through checked 512/1024/2048 bounded context packs where row-specific PASS artifacts are cited | PASS for compact `hello`, broader 50-token pack, bounded 512/1024/2048 context packs, and compact chat-template-shapes pack | PASS for compact `hello` and covered packs | PASS for compact `hello` 5-token, bounded 50-token, broader 50-token, bounded 512/1024/2048 context packs, and compact chat-template-shapes pack | Exact 8B Instruct Q8_0 support is bounded to the documented envelopes; the 1024/2048 bounded buckets are tied to the published source/runtime-head PASS bundle for source/runtime head `8e26be0a73c0`, while older 1024/2048 PASS artifacts remain historical source-head evidence. Model-native/larger context, arbitrary template execution, stronger memory/perf, portability, and broader template coverage remain expansion gates. |

## Current evidence summary

### TinyLlama 1.1B Chat Q8_0

- Prompt IDs match known-good reference.
- First generated token matches `29907` / `"C"`.
- Short deterministic generation matches.
- This is the live supported generation gate.

Representative artifacts cited by the public docs:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/tinyllama-q8-chat-parity-5tok.json`
- `target/chat-parity-postfix-50-token-audit.json`

### Llama 3.2 1B Instruct Q8_0

- Compact-header prompt IDs match known-good reference.
- First generated token matches `9906` / `"Hello"`.
- Compact deterministic generation matches `[9906,0,2650,649,358]` / `"Hello! How can I"`.
- The broader downloaded prompt pack also passed for prompt tokens, generated token IDs, and generated text.
- The second bounded 1024-context pack passed with 881 reference prompt tokens, generated tokens `[34,2735,35,12,4278]`, and generated text `CMLD-102`.
- The third bounded 2048-context pack passed after the RoPE frequency-factor fix with 1910 reference prompt tokens, generated tokens `[34,2735,35,12,7854]`, and generated text `CMLD-204`.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are documented for the exact row.
- This is verified bounded support for the exact row, not broad Llama-family support.

Representative artifacts cited by the public docs:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/llama32-1b-q8-chat-parity-5tok.json`
- `target/qa-small-model-parity-20260429T1338Z-head-35bfd58/`
- `target/parity-50tok-20260502T031820Z/llama32-1b-50tok/report.json`
- `target/downloaded-llama-matrix-20260502T231000Z/summary.json`
- `qa/evidence-bundles/llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json`

### Llama 3.2 3B Instruct Q8_0

- The exact GGUF exists in the tracked model-dir lane used by the validation runs.
- Metadata and `/api/models/load` work for the exact row.
- Compact prompt-token, deterministic 1-token, deterministic 5-token, and bounded 50-token parity passed.
- The post-Q8-dot broader three-prompt 50-token pack passes for prompt tokens, generated token IDs, and generated text.
- The first bounded 512-context pack, second bounded 1024-context pack, and third bounded 2048-context pack pass for this exact row only.
- `/v1/completions`, `/v1/chat/completions`, frontend smoke, and a five-prompt API smoke pack are documented for the exact row.
- This is a supported exact-row smoke lane, not broad Llama-family support.

Representative artifacts cited by the public docs:

- `target/parity-20260502T030911Z/llama32-3b-1tok/report.json`
- `target/parity-20260502T030911Z/llama32-3b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama32-3b-50tok/report.json`
- `target/camelid-regression-q8dot-20260502T232633Z/llama32-3b-compact/summary.json`
- `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`

### Llama 3 8B Instruct Q8_0

- Tokenizer, metadata, config/template, retained-Q8, and lazy/file-backed Q8 groundwork exist.
- Compact-header `hello` now has prompt-token parity plus deterministic 1-token, 5-token, and bounded 50-token generation parity.
- Basic API smoke, frontend smoke, and bounded memory evidence are documented for the exact tracked Q8_0 GGUF.
- The later broader three-prompt 50-token pack passed for `hello`, alpacas, and JSON and is summarized at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`.
- The first bounded 512-context pack passed on the reopened Ubuntu validation lane and is summarized at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`.
- The published source/runtime-head bounded 1024/2048-context bundle at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` passed for source/runtime head `8e26be0a73c0`: 1024 used 881 reference prompt tokens and generated `CMLD-102` (`[34,2735,35,12,4278]`), while 2048 used 1910 reference prompt tokens and generated `CMLD-204` (`[34,2735,35,12,7854]`). Prompt-token, generated-token, and generated-text parity all matched llama.cpp for both checked buckets.
- Older bounded 1024/2048-context PASS artifacts remain historical for their source heads only; do not cite them as current-head evidence after later runtime/source commits.
- The bounded compact chat-template-shapes pack passed on the reopened Ubuntu validation lane and is summarized at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`.
- Do not treat this as broad/full 8B support: the checked 512/1024/2048 bounded packs are exact-row only, and larger/model-native context, arbitrary template execution, broader chat-template coverage, support-grade memory/perf, production throughput, and portability remain required before widening the claim.

Representative artifacts cited by the public docs:

- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/`
- `target/parity-20260502T030911Z/llama3-8b-1tok/report.json`
- `target/parity-20260502T030911Z/llama3-8b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama3-8b-50tok/report.json`
- `target/downloaded-llama-matrix-20260502T231000Z/summary.json`
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` (published source/runtime-head bounded 1024/2048 PASS for source/runtime head `8e26be0a73c0`)
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T024342Z-head-b49034007f2e/manifest.json` (historical source-head PASS after later commits)
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260507T194559Z-head-ab8e465b50c3/manifest.json` (historical source-head PASS only)
- `qa/evidence-bundles/llama3-8b-context-1024-20260506T182100Z-head-e146d3b335d8/manifest.json` (historical/superseded for current-head promotion)
- `qa/evidence-bundles/llama3-8b-context-2048-20260506T182534Z-head-e146d3b335d8/manifest.json` (historical/superseded for current-head promotion)
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`

## Artifact caveat

Most representative artifacts live under gitignored `target/` paths and are not present in a fresh
public checkout. That is fine for local validation, but a four-row full-support release should also
publish a durable artifact manifest with exact commands, model SHA256 values, current commit, and
checksums for every cited report.

## Usage rule

Treat this file as QA context only. Support changes must be reflected in `COMPATIBILITY.md`,
`STATUS.md`, `/api/capabilities`, and frontend readiness copy together.
