# Validation Matrix

Last updated: 2026-05-31

Run the smallest meaningful validation lane for your change. If a change affects support language, readiness behavior, or exact-row claims, update docs and evidence references together.

## Current support-surface alignment rule

Every public surface should tell the same exact-row support story:

- TinyLlama 1.1B Chat Q8_0 is the supported current gate, with checked 512-context/template/RSS evidence.
- Llama 3.2 1B Instruct Q8_0 has verified bounded support through checked 512/1024/2048/4096/8192 context packs where row-specific PASS artifacts are cited, plus a supported metadata-Jinja row-template renderer path for the recognized Llama 3 template shape; the latest 8192 pack is `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`. Llama 3.2 3B Instruct Q8_0 remains supported as exact-row smoke through the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF (`qa/evidence-bundles/llama32-3b-context-512-8192-anchored-20260710T2119-head-6527a770/manifest.json`; the May template-context packs remain historical evidence for the prior upload).
- Llama 3 8B Instruct Q8_0 has verified bounded support through checked bounded 512/1024/2048 context packs where row-specific PASS artifacts are cited; the 1024/2048 buckets are backed by `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` for source/runtime head `8e26be0a73c0`, and older 1024/2048 bundles remain historical source-head evidence.
- Mistral 7B Instruct v0.3 Q8_0 is supported exact-row smoke only: `Mistral-7B-Instruct-v0.3.Q8_0.gguf` has tokenizer/template, 1-token, broader 50-token, checked 512/1024/2048/4096/8192 context, GPU-vs-CPU greedy parity, and support-promotion API/WebUI smoke evidence at `qa/evidence-bundles/mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json`.
- Current-head language requires a fresh canonical PASS after later runtime/source commits; broad/full support for every non-TinyLlama row still requires model-native/larger context beyond checked packs, arbitrary-template evidence, throughput, portability, and durable normalized current-head bundles.

Next-family public language is locked to row-by-row evidence, not family-wide support:

- Mixtral 8x7B Instruct: “Bounded one-token backend MoE runtime evidence exists for `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf`; later-generation/API/WebUI/frontend readiness is not yet established.”
- Qwen 2.5 7B Instruct: “Planned exact-row candidate for `Qwen2.5-7B-Instruct-Q8_0.gguf`; not supported yet.”
- Gemma 2 9B Instruct: “Verified runnable exact row for `gemma-2-9b-it-q8_0.gguf`: tokenizer/template identity, real-weight load, six short deterministic greedy probes, and guarded API/WebUI checks pass; bounded 512-context, broader performance, and portability evidence remain before Supported.”
- Nine hash-pinned Phase 2 rows are “Runnable with numerical variance”: they pass exact-artifact load, generation, and template gates, but at least one strict deterministic token-ID probe differs from pinned llama.cpp. The Models UI must download, start, and chat through the normal path with an amber warning; it must not call these rows Experimental, Verified, or Supported. Load-blocked and template-blocked rows remain fail-closed.

First promotion for any unsupported row requires row-specific source/SHA/license, tokenizer/template references, bounded load/readiness, parity, API/WebUI, RSS/timing, scrubbed manifest, and checksum evidence. Mixtral has not cleared support promotion under the current blocker record: later-generation divergence and the continuation backend HTTP hang must be fixed before rerunning API/WebUI/RSS/frontend readiness. The current long-generation continuation lane is blocker work and does not widen support wording.

| Change type | Minimum expected checks | Extra checks when relevant | Notes |
| --- | --- | --- | --- |
| Docs-only | `git diff --check`<br>`node scripts/check-public-evidence-claims.mjs`<br>`bash scripts/check-public-scrub.sh` | n/a | Keep support language synchronized with `README.md`, `COMPATIBILITY.md`, `docs/reference/STATUS.md`, and UI copy when claims change. |
| Frontend-only copy/layout | `cd frontend && npm ci && npm run build` | `npm run smoke` or `npm run smoke:tiny` when chat/model-load/readiness surfaces change | Do not loosen readiness gates or support wording without matching evidence/docs updates. |
| Backend-only non-inference changes | `cargo fmt --all -- --check`<br>`cargo clippy --all-targets --all-features -- -D warnings`<br>`cargo test --all-targets --all-features`<br>`cargo doc --no-deps --all-features`<br>`bash scripts/check-public-scrub.sh` | frontend build if API shape or delivery may be affected | Good default lane for parser, API, CLI, and non-runtime refactors. |
| Inference/tokenizer/runtime changes | Standard backend gate above | targeted parity, readiness, or smoke artifacts for the affected exact row(s) | Do not broaden support from seam evidence alone. |
| Frontend + backend readiness/chat-path changes | Standard backend gate + `cd frontend && npm ci && npm run build` | frontend smoke against the affected exact row(s) | Required when `/v1/health`, `/api/capabilities`, model loading, or WebUI chat gating changes. |
| Support-contract / compatibility-row changes | Validation appropriate to the underlying code/docs change | fresh evidence bundles and synchronized updates to public sources of truth | A support claim is a release decision, not a wording tweak. |
| QA / evidence-publication changes | Validate the producing scripts or manifests you changed | scrub/publication checks and updated artifact references | Keep public bundle paths, manifests, and summaries internally consistent. |

## Public vs maintainer-only validation

Public contributor expectations stop at local reproducible checks plus public artifact references.

The following may still be maintainer-only workflows rather than baseline contributor requirements:

- promotion-grade reruns on the approved Ubuntu validation lane
- SSH-backed remote execution
- private operator recovery/debug procedures

Public docs may reference those workflows at a high level, but should not depend on unpublished infrastructure details.

## When in doubt

- choose the smallest lane that could realistically catch your change
- if a claim gets stronger, the evidence must get stronger too
- if code, docs, frontend copy, and compatibility rows disagree, the task is not finished
