# Camelid ForgeLocal and Fathom Integration Plan

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

## What was inspected

ForgeLocal:

- `frontend/src/hooks/useDashboardData.js`
- `frontend/src/App.jsx`
- `frontend/src/views/ModelsView.jsx`
- `frontend/src/views/SystemView.jsx`
- `backend/src/main.rs`
- `README.md`

Fathom:

- `docs/api/v1-contract.md`
- `docs/api/backend-only-quickstart.md`
- `docs/architecture/rust-runtime-architecture.md`
- `crates/fathom-server/src/main.rs` route surface by search

## Source-of-truth rule

Camelid is the project/product name going forward. Current implementation names may still say `camelid` in crate, binary, route diagnostic, and metadata contexts until a separate code/package rename is planned.

Use the full ForgeLocal repository as the product/frontend source of truth:

```text
https://github.com/timtoole02/forgelocal/
```

Frontend work should preserve and adapt the existing ForgeLocal codebase wherever possible instead of rebuilding replacement screens from scratch.

## Current integration status

A frontend integration now exists in `frontend/`. It is based directly on the existing ForgeLocal frontend source — app shell, views, components, and styling are preserved — while the dashboard/data hook is adapted to Camelid's current route surface:

- `GET /v1/health` for server/generation readiness
- `GET /v1/models` for loaded model selection
- `GET /api/models/current` for the active local GGUF state when available
- `POST /api/models/load` for loading a local GGUF path
- `POST /v1/chat/completions` for non-streaming chat

ForgeLocal-only server features such as model catalog install, hosted-provider chat routing, analytics persistence, and runtime process management are not faked because Camelid does not expose equivalent app-control routes. Browser-local conversations and memory notes are kept for UI continuity, while backend truth comes from Camelid. The Models page now separates current local GGUF loading from planned hosted/catalog features: it reads Camelid health/current-model/list state, saves imported local file paths in browser state, calls `/api/models/load` for those paths, hides catalog browsing unless an endpoint responds, and labels hosted API links as planned rather than runnable.

## Current ForgeLocal shape

ForgeLocal is already split into:

- React/Vite frontend
- Rust/Axum control plane
- SQLite app state
- local runtime bridge
- OpenAI-compatible `/v1` surface

The frontend does not call `llama-server` directly. It calls the ForgeLocal Rust backend. That backend either starts/calls local `llama-server` or forwards to an external OpenAI-compatible API.

## Key ForgeLocal routes

Representative backend routes:

```text
GET    /api/dashboard
GET    /api/runtime
GET    /api/models/catalog
POST   /api/models/catalog/install
POST   /api/conversations
POST   /api/conversations/:id/chat
POST   /api/models/install
POST   /api/models/register
POST   /api/models/external
POST   /api/models/:id/activate
GET    /v1/models
GET    /v1/health
POST   /v1/chat/completions
```

## Current runtime paths

### Local runtime path

ForgeLocal stores local model records with `model_path`, `runtime_model_name`, and `provider_kind = "local"`. For chat, it starts `llama-server` with a model path and calls:

```text
GET  http://127.0.0.1:<runtime_port>/v1/models
POST http://127.0.0.1:<runtime_port>/v1/chat/completions
```

It expects an OpenAI-style chat response with `choices[].message.content` and `usage`.

### External OpenAI-compatible path

ForgeLocal stores external providers with `provider_kind = "external"`, `api_base`, `api_key`, and `runtime_model_name`. It verifies models through:

```text
GET <api_base>/models
Authorization: Bearer <api_key>
```

Then sends chat to:

```text
POST <api_base>/chat/completions
Authorization: Bearer <api_key>
```

This is the best initial adapter path for Camelid.

## Phase 0 integration hypothesis

Camelid should first expose a local OpenAI-compatible `/v1` service. ForgeLocal can use it through the existing external-provider path with minimal or no frontend changes.

Recommended default base URL:

```text
http://127.0.0.1:8181/v1
```

Rationale:

- Fathom commonly uses `8180`.
- ForgeLocal uses `8080`.
- llama runtime commonly starts at `8081`.
- `8181` avoids likely conflicts.

## Minimum Camelid HTTP contract

### `GET /v1/health`

```json
{
  "ok": true,
  "engine": "camelid",
  "generation_ready": false,
  "active_model_id": null
}
```

The frontend treats `generation_ready` as the chat gate. A model can be loaded but still not generation-ready if tokenizer/runtime pieces are unsupported or incomplete.

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "loaded-model-id",
      "object": "model",
      "created": 0,
      "owned_by": "camelid"
    }
  ]
}
```

ForgeLocal's external verifier expects the chosen model name to appear in `data[].id`.

### `GET /v1/models/{model}`

Returns the same OpenAI-style model object for the currently loaded runnable model, or a typed `404 model_not_found` error if the requested id is not active. This supports OpenAI-compatible consumers that retrieve a model detail after listing models without advertising any unsupported model registry behavior.

### `POST /v1/chat/completions`

Current behavior:

- Accept OpenAI-shaped non-streaming and SSE streaming requests, including `stop` as a string or string array.
- Support a single generated choice (`n` omitted or `1`; `best_of` omitted or `1`) from a loaded Camelid-supported dense GGUF model/tokenizer.
- Return typed `400 unsupported_parameter` errors for unsupported multi-choice and logprob requests such as `n > 1`, `best_of > 1`, completion `logprobs`, chat `logprobs: true`, or `top_logprobs`.
- Never fake inference output or logprob payloads.

### Error envelope

Use OpenAI/Fathom-like errors:

```json
{
  "error": {
    "message": "Human-readable explanation.",
    "type": "invalid_request",
    "code": "not_implemented",
    "param": null
  }
}
```

Recommended status/code pairs:

- `400 invalid_request`
- `400 model_not_found`
- `400 invalid_model`
- `400 unsupported_parameter`
- `501 not_implemented`
- `501 stream_not_supported`
- `503 runtime_unavailable`

### `GET /api/models/current`

Returns Camelid's active loaded-model state and readiness diagnostics for local GGUF paths. The frontend should use this as local truth for the current model instead of inferring readiness from catalog records.

### `POST /api/models/load`

Loads a local GGUF path into Camelid. Catalog download/install remains out of scope until Camelid exposes an explicit model-management surface.

## Adapter strategy

### Phase 0A — no ForgeLocal code changes

Run Camelid locally and register it in ForgeLocal as an external OpenAI-compatible API:

- API base: `http://127.0.0.1:8181/v1`
- API key: non-empty placeholder if ForgeLocal requires one
- Model name: one of `GET /v1/models` IDs

Concern: ForgeLocal may require an API key for external providers. A local Camelid service should ideally not need one.

### Phase 0B — thin ForgeLocal adapter

Later, add `provider_kind = "camelid"` while reusing most external-provider code.

Potential adapter trait:

```rust
trait ChatRuntimeAdapter {
    async fn list_models(&self) -> Result<Vec<RuntimeModel>>;
    async fn health(&self) -> Result<RuntimeHealth>;
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;
}
```

Implementations:

- Existing local process adapter
- `OpenAiCompatibleHttpAdapter`
- `CamelidHttpAdapter`

`CamelidHttpAdapter` can initially wrap the OpenAI-compatible HTTP adapter.

## Fathom alignment

Fathom already exposes a narrow truthful OpenAI-style API:

```text
GET  /v1/health
GET  /v1/models
POST /v1/chat/completions
POST /v1/embeddings
```

It also has runtime/control concepts:

```text
GET  /api/runtime
GET  /api/capabilities
POST /api/models/catalog/install
POST /api/models/register
POST /api/models/:id/activate
```

Principles to preserve for future Fathom integration:

- `/v1/models` should list only actually runnable generation models.
- Unsupported models should be excluded or refused truthfully.
- `stream: true` should return real OpenAI-compatible SSE chunks for supported chat/completion requests, and typed errors for unsupported runtime states or unsupported request options.
- Errors should use a consistent envelope.
- Capability reporting should distinguish detected/imported/runnable states.

## Future Fathom plan

Do not integrate yet. Prepare Camelid so it can later become a runtime target by:

1. Maintaining `/v1` compatibility.
2. Providing `/api/capabilities`.
3. Returning structured model metadata.
4. Keeping runtime-specific metadata namespaced under `camelid`.
5. Avoiding ForgeLocal-only assumptions in core APIs.
