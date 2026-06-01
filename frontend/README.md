# Camelid frontend

This frontend targets Camelid. During the naming transition, the backend crate, binary, and some diagnostics still use `camelid`; the commands below keep those current implementation names.

## Source of truth

This UI is the Camelid frontend. It was adapted from an existing local-model frontend implementation rather than rebuilt as a throwaway replacement, so future frontend work should preserve the mature app shell, views, components, styling, and UX structure while wiring behavior to Camelid.

The backend data hook is adapted for Camelid's current API surface:

- checks `GET /v1/health`
- lists `GET /v1/models`
- treats `/v1/models` public `meta` fields as descriptive model-shape and quant metadata only; those fields must not expose local paths and must not unlock support or chat readiness by themselves
- does not use native llama-server `/models` discovery, router-mode load/unload route presence, or model-management metadata to unlock chat; WebUI readiness still comes only from the exact `/api/capabilities` row plus `/v1/health` runtime state
- loads local GGUF paths through `POST /api/models/load`
- reads the support contract from `GET /api/capabilities`
- shows the support gate, current compatibility row, model-family/quantization evidence, and guarded API feature rows directly in chat, model setup, per-model/catalog cards, API, analytics, and system surfaces
- keeps the current runtime chat gate and `/api/capabilities` support gate visible in the page top bar outside the Chat/Models views, with a direct jump to the API contract before users interpret model-family or quant support
- keeps the API tab first-class in desktop/sidebar/mobile navigation, browser tab restore, and chat readiness prompts so the support contract is easy to find during readiness checks
- keeps API examples readiness-gated: `/api/capabilities` explains evidence boundaries, while `/v1/health` `loaded_now`/`generation_ready` plus `active_model_id` decide whether chat calls should run for the selected local GGUF
- normalizes loaded-model `general.file_type` values into GGUF quant labels (for example file type `7` → `Q8_0`) before comparing them to `/api/capabilities`, so loaded model cards get useful quant evidence without treating filenames as support claims
- keeps the shipped exact Llama 3.2 1B/3B Instruct Q8_0 and Llama 3 8B Instruct Q8_0 smoke rows visible as row-specific wins, while still requiring the loaded local GGUF to match its exact supported row before chat unlocks
- sends readiness-gated streaming chat requests to `POST /v1/chat/completions` with `stream: true`, preserves backend usage when provided, treats structured SSE `event: error` payloads as failures instead of empty assistant replies, and keeps the non-streaming JSON parser only as a response-shape fallback
- blocks chat until `/v1/health` reports the selected `active_model_id` with `loaded_now: true` and `generation_ready: true` and `/api/capabilities` has an exact supported model/quant compatibility row; streaming transport behavior is UI reliability evidence only, not support evidence for any additional model row; the exact Llama 3.2 1B/3B Instruct Q8_0 plus Llama 3 8B Instruct Q8_0 rows are supported only for their bounded local-chat smoke/parity envelopes

For claim-source order and public evidence policy, see [`../docs/WAR_ROOM_EVIDENCE_INDEX.md`](../docs/WAR_ROOM_EVIDENCE_INDEX.md). That file keeps WebUI readiness copy aligned with `COMPATIBILITY.md`, `/api/capabilities`, and scrubbed evidence; it does not add any supported row or frontend unlock condition.

Server features Camelid does not expose yet are kept honest: catalog downloads, external-provider setup, planned or blocked quantization lanes, and unsupported or partial API parameters show disabled or typed-guardrail copy instead of pretending to work. The analytics view treats conversation telemetry as usage only, not compatibility evidence. The UI mirrors the compatibility contract documented in `../COMPATIBILITY.md`; filenames, catalog metadata, saved browser paths, and prior usage are not treated as support evidence by themselves.

## Exact-row smoke wins shown in the UI

The frontend should make these shipped wins easy to see without turning them into broad Llama-family support:

- **Llama 3.2 1B Instruct Q8_0:** exact-row API/WebUI smoke plus compact/broader parity, the supported metadata-Jinja row-template path for the recognized Llama 3 template shape, and bounded 512/1024/2048/4096/8192 context-pack evidence are represented as supported exact-row evidence; the 2048 pack is exact-row only after the RoPE frequency-factor fix, and the 4096/8192 packs stay tied to their cited source/runtime heads.
- **Llama 3.2 3B Instruct Q8_0:** canonical Ubuntu main-lane exact-row API/WebUI support-gate refresh at source head `e9f926ed1a65`, compact parity, broader three-prompt 50-token parity, five-prompt API smoke, row-scoped metadata-Jinja/template-shape evidence, bounded unique-chat perf/RSS, opt-in parallel Q8 first-token direction evidence, and bounded 512/1024/2048 context packs are represented as a supported exact-row smoke lane. Bounded perf/RSS and first-token direction evidence must not be labeled production-throughput support.
- **Llama 3 8B Instruct Q8_0:** exact-row API/WebUI smoke, clean-main timing/RSS smoke, broader 50-token parity, checked bounded 512/1024/2048 context packs, compact chat-template-shapes pack, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path cost probe are represented as supported exact-row wins. The published source/runtime-head 1024/2048 bounded-pack PASS at `../qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` closes those exact bounded buckets; older 1024/2048 bundles remain historical source-head evidence.

All three rows still fail closed in the WebUI unless the active local GGUF matches the exact row and `/v1/health` reports `loaded_now=true` plus `generation_ready=true`. When `/api/capabilities` reports bounded template-shape, exact-row renderer, or perf/RSS fields, the frontend may show those as row-scoped evidence only; those fields do **not** establish broad arbitrary/Jinja-template readiness or production-throughput readiness. For the 3B row specifically, the canonical Ubuntu main-lane refresh at `../qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json` keeps chat green only for the exact `supported_exact_row_smoke` contract and does not widen context, production-throughput, portability, neighboring-row, or broad-family claims. The cards still do not infer neighboring sizes, base variants, other quantizations, larger contexts beyond checked packs, or portability.

## Run locally

Start Camelid first, usually on `127.0.0.1:8181`:

```bash
cargo run -- serve --addr 127.0.0.1:8181
```

Then run the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open:

```text
http://127.0.0.1:4175
```

## Configuration

For repo-wide local toolchain and path guidance, see [`../docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).

The default API base is:

```text
http://127.0.0.1:8181
```

Override it at build/dev time with:

```bash
VITE_CAMELID_API_BASE=http://127.0.0.1:8181 npm run dev
```

You can also edit the API base in the UI sidebar while testing.

## Validation

Build the frontend:

```bash
cd frontend
npm run build
```

For parser-only chat-path changes, run the streaming parser smoke before involving a real backend:

```bash
cd frontend
npm run smoke:streaming
```

This smoke covers partial SSE chunks, backend usage chunks, non-streaming JSON fallback parsing, and structured SSE `event: error` failures after headers. Passing it does not prove model parity, API/WebUI readiness, or support for any new row; it only validates frontend stream parsing behavior.

Smoke-test a running backend + frontend:

```bash
# terminal 1
cargo run -- serve --addr 127.0.0.1:8181

# terminal 2
cd frontend
npm run dev

# terminal 3
cd frontend
npm run smoke
```

For a self-contained local generation smoke test, use the tiny generated GGUF fixture:

```bash
cd frontend
npm run smoke:tiny
```

`smoke:tiny` creates a temporary tiny Camelid-compatible GGUF fixture, loads it through `POST /api/models/load`, verifies `generation_ready=true`, checks `/v1/models`, and confirms the WebUI chat guard stays blocked when that fixture does not have an exact supported `/api/capabilities` compatibility row. Real chat smoke only runs for models that are both `generation_ready=true` and support-contract matched.

To smoke-test a downloaded local GGUF without committing model files, pass its path explicitly:

```bash
cd frontend
npm run smoke -- --model ../models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --model-id tinyllama-q8
```

This verifies the frontend is reachable, loads the GGUF through the backend API, checks `/v1/health`, `/v1/models`, and the UI guardrails around `/api/capabilities`, and only sends a chat request when `generation_ready=true` **and** the active model has an exact supported compatibility row. The smoke output includes coarse timings for frontend reachability, model load, health/model listing, support-contract matching, and chat completion so real-model runs produce repeatable latency evidence. Add `--require-generation` when the model is expected to run end-to-end; otherwise the smoke exits successfully after confirming the UI/API guardrail state for metadata-only or unsupported-runtime models.

For the exact smoke-supported Llama rows, use the exact local path when a backend and frontend are running:

```bash
cd frontend
npm run smoke -- --model '$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf' --model-id llama-3.2-1b-instruct-q8 --require-generation --expect-compatibility-row llama32_1b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled

npm run smoke -- --model '$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf' --model-id llama-3.2-3b-instruct-q8 --require-generation --expect-compatibility-row llama32_3b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled

npm run smoke -- --model '$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf' --model-id llama-3-8b-instruct-q8 --require-generation --expect-compatibility-row llama3_8b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled
```

These commands must still fail closed if the loaded model is the wrong row, lacks Q8_0 metadata, is not `loaded_now=true` + `generation_ready=true`, or is outside the exact supported `/api/capabilities` row. That is intentional: the UI supports only the exact 1B/3B/8B smoke rows without making a broad Llama-family claim. The older public reopened-lane API + frontend smoke summary for all four exact rows remains `../qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; for the exact 3B row, use `../qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json` as the canonical Ubuntu API/WebUI support-gate refresh. The exact 1B third bounded 2048-context pack now passed after the RoPE frequency-factor fix at `../qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json`; the older 1B blocker notes remain historical. The exact 3B third bounded 2048-context pack passed at `../qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`; the exact 8B broader three-prompt 50-token pack passed at `../qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`; the first 8B bounded 512-context pack passed at `../qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`; and the published source/runtime-head 8B 1024/2048 bounded-pack pass at `../qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` matched prompt tokens, generated token IDs, and generated text for `CMLD-102`/`CMLD-204`, closing those exact bounded buckets. Older 8B 1024/2048 bounded-pack artifacts remain historical for their source heads only. The compact chat-template-shapes pack passed at `../qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`, and the lazy-Q8 hot-path measurement is summarized at `../qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`; the frontend surfaces those as bounded exact-row evidence only. They do not establish arbitrary/Jinja-template readiness, production-throughput support, model-native/larger context, portability, neighboring rows, or broad Llama-family support.
