# Frontend 3B Catalog Exact-Row Badge

## Summary

Retained frontend slice for Llama 3.2 3B Instruct Q8_0 WebUI closure.

- Source head before slice: `93a6502c1726e75a1d86def9edd753acc1f59db1`
- Branch: `frontend-3b-closure-6e9017a8-20260520`
- Slice: catalog model cards now show the exact `/api/capabilities` row badge when a catalog item resolves to a supported exact row, using the same `findCompatibilityHint` path as chat/API/System readiness.
- Regression: `frontend/scripts/frontend-3b-closure-smoke.mjs` now covers 3B catalog filename plus Q8_0 resolution and asserts that ModelsView renders row-scoped catalog exact-row badges.

## Gate Results

- `npm run build`: pass
- `npm run smoke:model-state`: pass
- `npm run smoke:streaming`: pass
- `npm run smoke:ui`: pass
- `npm run smoke:integration`: pass
- `npm run smoke:3b-closure`: pass
- local backend plus frontend `npm run smoke:tiny`: pass; WebUI chat gate stayed blocked for the unsupported tiny fixture despite `generation_ready=true`
- `scripts/check-public-scrub.sh`: pass
- `node scripts/test-audit-evidence-bundle-privacy.mjs`: pass
- `node scripts/test-check-public-evidence-claims.mjs`: pass

## Remote 3B API Evidence

Canonical Ubuntu host was probed in this run with the project-private SSH command. The public evidence keeps only scrubbed results.

- Remote repo head: `93a6502c1726e75a1d86def9edd753acc1f59db1`
- Exact 3B GGUF load: `id=llama-3.2-3b-instruct-q8`, tokenizer `available`, tensors bound, GGUF `file_type=7`
- `/v1/health`: `loaded_now=true`, `generation_ready=true`, `active_model_id=llama-3.2-3b-instruct-q8`
- `/api/capabilities`: `llama32_3b_instruct_q8_0`, `supported_exact_row_smoke`, `Q8_0`, `metadata_jinja_supported_for_exact_row`, `bounded_unique_chat_perf_rss_validated`
- `/v1/chat/completions` one-token probe: `finish_reason=length`, `completion_tokens=1`, non-empty content

## Retain/Reject

Retain. This is a frontend support-surface slice only: it improves catalog visibility for an already supported exact row and does not widen 3B support beyond exact Q8_0 row, runtime `loaded_now=true`, runtime `generation_ready=true`, and matching `/api/capabilities` evidence.
