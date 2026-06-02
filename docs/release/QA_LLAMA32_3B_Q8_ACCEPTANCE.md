# Llama 3.2 3B Instruct Q8_0 Parity Acceptance

Last updated: 2026-05-05

> [!NOTE]
> This QA checklist is an acceptance document for one exact model row. It does not change the
> public support contract by itself. For current support truth, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md)
> and [`STATUS.md`](../../STATUS.md).

QA checklist for the exact Llama 3.2 3B WebUI real-chat acceptance gate. The short local-chat
smoke gate is now accepted for this exact row; this file remains the QA boundary for preserving
that claim and for naming the still-missing expansion evidence.

## Exact target artifact

- **Source repo:** `bartowski/Llama-3.2-3B-Instruct-GGUF`
- **Required filename:** `Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Expected local path:** `$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Resolve URL:** `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Expected size from earlier HEAD check:** `3,421,899,296 bytes` (`3.187 GiB`)
- **Earlier HEAD ETag/Xet hash:** `291ce1d4ca0fcef86407b7c6531bf85a1c348c65d5d3c69c57c98fec6483bb1f`

Current state: the exact GGUF is now present at the expected model-dir path, Camelid metadata/API
load evidence exists, the Ubuntu compact-header `hello` harness has prompt-token parity plus
deterministic 1-token, 5-token, and bounded 50-token generation parity, the broader three-prompt
50-token pack now matches the known-good baseline, reopened-lane API/WebUI smoke is captured, the second bounded 1024-context pack and third bounded 2048-context pack are captured, bounded
compact chat-template-shapes plus unique-chat perf/RSS envelopes are captured, and the opt-in parallel Q8 first-token runtime direction sub-box is captured for this exact row.
The support claim is therefore **supported exact-row smoke** only. The blocker has moved from
short-chat parity/API/WebUI/template/perf acceptance to model-native/larger context beyond the checked 512/1024/2048 packs,
production performance/portability beyond the first-token direction probe, and arbitrary/Jinja template acceptance before any
broader/full-support language.

Durable public anchors:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_3b_instruct_q8_0.bundle.json` preserves the carry-forward exact-row smoke boundary.
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json` is the current-head citation target for the compact parity, post-Q8-dot broader-pack handoff, and next context/template/perf tracks.
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json` records the reopened-lane API + frontend smoke refresh for this exact row.
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json` records the first bounded 512-context pass for this exact row.
- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json` records the second bounded 1024-context pass for this exact row.
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json` records the third bounded 2048-context pass for this exact row only.
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json` records bounded compact chat-template-shape coverage for this exact row.
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json` records the bounded unique-chat perf/RSS envelope for this exact row.
- `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json` records the opt-in parallel Q8 first-token runtime direction sub-box for this exact row.

## Current blocker summary

- `/api/models/load` succeeds for the exact 3B target.
- The latest file-backed lazy-Q8 recovery materially reduced the earlier eager dense-load spike.
- The Ubuntu compact-header `hello` harness now matches the known-good baseline for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
- The former broader JSON-shaped prompt blocker is resolved by the post-Q8-dot clean rerun. The public citation target is `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json`; the raw drill-down source path was `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json`. In that checked pack, `hello`, alpacas, and `answer with valid JSON for {"ok":true,"value":2}` all match the known-good baseline for prompt tokens, generated token IDs, and generated text.
- Therefore the row is no longer parity-blocked for the current three-prompt 50-token pack, the first bounded 512-context pack, the second bounded 1024-context pack, the third bounded 2048-context pack, the bounded compact chat-template-shapes pack, the bounded unique-chat perf/RSS envelope, or the opt-in parallel Q8 first-token runtime direction sub-box; remaining expansion gates are model-native/larger context beyond the checked packs, production performance/portability beyond the first-token direction probe, and arbitrary/Jinja chat-template evidence.

## Disk and memory expectations

- Keep the artifact in the configured `$CAMELID_MODEL_DIR` location.
- Use bounded runs with process-memory sampling before any WebUI promotion.
- Do not infer safety from the 1B or 8B rows.

## Acceptance checklist

The exact-row short-chat smoke items below are accepted for the current row. Do not mark any
broader/full-support expansion green until all applicable items for that larger claim have durable
artifact paths.

1. **Model presence** — exact filename exists at the expected model-dir path; record size and hash.
2. **Readiness/inspect** — `scripts/small-model-readiness.mjs` or equivalent reports the row and
   records the exact blocker or safe candidate state.
3. **Rendered prompt** — capture the compact Llama 3 prompt Camelid currently renders.
4. **Reference token IDs** — use llama.cpp `llama-tokenize --ids` against the exact 3B GGUF.
5. **Camelid prompt-token parity** — run `scripts/chat-parity-llama3.mjs --require-prompt-match`.
6. **First generated token parity** — run deterministic greedy `--max-tokens 1 --require-generated-match`.
7. **Short greedy output parity** — run deterministic greedy `--max-tokens 5 --require-generated-match`.
8. **API load/chat smoke** — capture `/v1/health`, `/api/models/current`, `/api/models/tokenizer`,
   `/v1/chat/completions`, and process-memory samples.
9. **WebUI smoke** — only after API parity is green, capture real chat evidence plus memory samples.
10. **Regression preservation** — keep TinyLlama Q8_0 and Llama 3.2 1B evidence green.

## Current status

Status: **accepted exact-row parity/API/WebUI smoke with broader three-prompt parity, compact template-shape, bounded 512-context, bounded 1024-context, bounded 2048-context, unique-chat perf/RSS, and opt-in parallel Q8 first-token runtime direction evidence**

The exact 3B artifact now exists, and the Ubuntu compact-header `hello` harness matches the known-good baseline
for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation. The former
JSON-shaped broader prompt blocker is now fixed. The raw drill-down paths were `target/camelid-regression-q8dot-20260502T232633Z/llama32-3b-compact/summary.json` for the compact pack and `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` for the broader three-prompt 50-token pack; the durable public citation target is `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json`, with API/WebUI freshness in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`, first bounded 512-context evidence in `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`, second bounded 1024-context evidence in `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json`, third bounded 2048-context evidence in `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`, bounded compact template-shape evidence in `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`, bounded unique-chat perf/RSS evidence in `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`, and opt-in parallel Q8 first-token runtime direction evidence in `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`. The current work is to preserve that evidence and expand only after model-native/larger-context, production performance/portability beyond the first-token direction probe, and arbitrary/Jinja chat-template gates land.
