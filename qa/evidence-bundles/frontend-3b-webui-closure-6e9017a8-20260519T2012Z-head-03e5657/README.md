# Frontend 3B WebUI closure refresh

Retained slice: current-main frontend/WebUI 3B closure for `Llama 3.2 3B Instruct Q8_0`.

What changed:
- TopBar support-contract detail now prioritizes the active/selected exact-row hint before falling back to the first current-gate row, so an active 3B row does not display TinyLlama/other-row detail.
- Chat keeps exact-row runtime readiness, support-contract readiness, and row-scoped capability lanes visible in both empty and live/non-empty chat states.
- Models tracked-row cards use the shared chat gate, so stale browser rows cannot claim chat unlock unless `loaded_now`, `generation_ready`, `active_model_id`, and exact supported compatibility row all pass.
- Frontend regression smokes now lock the 3B quant-mismatch, 512/1024/2048 context-boundary, active-alias, live-chat capability lane, and shared-gate behavior.

Support boundary: exact-row smoke only. This does not widen model-native/larger context beyond checked packs, production throughput, portability, neighboring-row, arbitrary GGUF, or broad Llama-family support.

Head before commit: 03e5657aae2b101397483e1bf8db6e04681fc9e0 (`origin/main`).
Live backend evidence: canonical Ubuntu host, current-main checkout `03e5657aae2b101397483e1bf8db6e04681fc9e0`, exact local `Llama-3.2-3B-Instruct-Q8_0.gguf`; public artifact intentionally uses scrubbed host/path details.

Gates captured:
- `npm run smoke:model-state`
- `npm run smoke:streaming`
- `npm run smoke:3b-closure`
- `npm run smoke:ui`
- `npm run smoke:integration`
- `npm run build`
- Live 3B WebUI smoke through local frontend preview + SSH-tunneled canonical backend: `npm run smoke -- --require-generation --expect-compatibility-row llama32_3b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled`
- `bash scripts/check-public-scrub.sh`
- `node scripts/audit-evidence-bundle-privacy.mjs --strict`
