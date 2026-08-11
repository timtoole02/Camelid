# RECON_CHAT.md — Phase 0 working note for `camelid chat`

Answers come from the actual repo (`/Volumes/Untitled/Camelid-push`, branch `main`), not
from the spec. Where the spec and the repo disagree, the repo wins and the discrepancy is
recorded here (§8).

## 1. Subcommand registration

- **File:** `src/main.rs`.
- **Pattern:** clap **derive**. A single `#[derive(Subcommand)] enum Command` (line 56) whose
  variants are matched in `#[tokio::main] async fn main()` (line 440, `match Cli::parse().command`).
  `Serve`, `Pull`, `Inspect`, `TensorDump` are all variants of this one enum; there is **one**
  bin target (`[[bin]] name = "camelid" path = "src/main.rs"` in `Cargo.toml`).
- **What `chat` must do:** add a `Chat { … }` variant to `Command` and a `Command::Chat { … } => …`
  arm to the `match`. Version string is the `const VERSION` at `src/main.rs:40`
  (git-describe via `CAMELID_GIT_DESCRIBE`, else `CARGO_PKG_VERSION`).

## 2. Model load path (`serve --model`)

- `Command::Serve { model, .. }` → `api::serve(addr, threads, model, open_ui)`
  (`src/main.rs:473`).
- **`pub async fn serve(addr: SocketAddr, configured_threads: Option<usize>, initial_model:
  Option<PathBuf>, open_ui: bool) -> std::io::Result<()>`** — `src/api/mod.rs:1293`.
- At startup, if `initial_model` is `Some`, serve calls `load_model_from_path(&state, path, None)`
  → `load_model_from_path_with_activation(state, path, id, true)` (`src/api/mod.rs:3418`), which
  reads GGUF metadata, builds the execution plan, parses `LlamaModelConfig::from_gguf`, inserts a
  `LoadedModel` into `state.loaded_models` and sets `state.active_model_id`.
- **HTTP equivalent (what `chat` will drive):** `POST /api/models/load` → handler
  `load_model(Json<LoadModelRequest>)` (`src/api/mod.rs:2674`).
  - `LoadModelRequest { path: PathBuf, id: Option<String> }` (`:236`).
  - Returns `200 OK` + `Json<LoadedModel>` on success, or `400` `api_error("invalid_model", …)`
    on a hard failure.
  - `LoadedModel` (`:166`) carries `id, path, gguf, llama_config: Option<…>,
    llama_tensors: Option<…>, unsupported_runtime: Option<UnsupportedRuntimeSummary>, tokenizer,
    lane`. **An unsupported architecture does not fail the load** — it returns `200` with
    `unsupported_runtime: Some(...)` and `llama_config: None` (see §5/§8).

## 3. Generation path (`/v1/chat/completions`)

- Route `.route("/v1/chat/completions", post(chat_completions))` (`src/api/mod.rs:1278`).
- Handler `async fn chat_completions(State<AppState>, Json<ChatCompletionRequest>) -> Response`
  (`:4522`): rejects multimodal, optionally routes to the gemma4 runtime (behind
  `CAMELID_GEMMA4_SERVE`), converts to a `GenerationSessionRequest`, calls
  `prepare_generation(&state, req)` (`:5174`), then:
  - `stream=true`  → `stream_completion(prepared, /*chat=*/true)` (`:7144`) — SSE.
  - `stream=false` → `generate_decoded_tokens_blocking(prepared)` (`:6012`) →
    `ChatCompletionResponse`.
- **Unsupported gate at generation:** `prepare_generation` requires `model.llama_config`; if
  absent it returns `422 UNPROCESSABLE_ENTITY` with code `"unsupported_model_architecture"` and
  message = `unsupported_runtime.message` (`src/api/mod.rs:5316`).
- **SSE wire format** (OpenAI-compatible, emitted via `sse_json_event` `:7510`):
  - role chunk first (chat): `delta: { role: "assistant" }`.
  - per token: `data: {"id":"chatcmpl-…","object":"chat.completion.chunk","model":…,
    "choices":[{"index":0,"delta":{"content":"<text>"},"finish_reason":null}]}\n\n`.
  - final: `delta: {}`, `finish_reason: "stop"|"length"`.
  - terminator: `data: [DONE]\n\n`.
  - Chunk structs: `ChatCompletionStreamChunk` / `ChatCompletionStreamChoice` /
    `ChatCompletionDelta` (`src/api/mod.rs:1063`).
- **Health:** `GET /v1/health` (and `/health`) → `health` (`:1364`) → `HealthResponse { ok,
  engine, loaded_now, generation_ready, active_model_id, backend, model_family,
  gemma4_available, … }` (`:241`).

## 4. `pull` path

- `Command::Pull { model, models_dir }` → `catalog::run_pull(model.as_deref(), &dir)`
  (`src/main.rs:575`; default dir `./models`).
- **`pub fn run_pull(query: Option<&str>, models_dir: &Path) -> anyhow::Result<()>`**
  (`src/catalog.rs:16`). No query → prints catalog. With a query → `resolve()` (id or name
  fragment, separator-insensitive) → `download()` via `curl -L -C - --fail -o models/<filename>
  <hf-url>`; skips if a complete copy exists; prints the `camelid serve --model …` next step.
- Catalog data: **`pub fn curated_catalog() -> Vec<CatalogItem>`** and `pub struct CatalogItem
  { catalog_id, name, repo_id, filename, size_bytes, downloads, likes, quant, license }`, both in
  `src/api/mod.rs` (re-exported; `catalog.rs` does `use crate::api::{curated_catalog, CatalogItem}`).
  Also exposed over HTTP at `GET /api/models/catalog` (`get_catalog`).
- **Catalog ids (8, all Q8_0 except the last):** `llama32_1b_instruct_q8_0`,
  `llama32_3b_instruct_q8_0`, `tinyllama_1_1b_chat_q8_0`, `llama3_8b_instruct_q8_0`,
  `gemma4_e4b_it_q8_0`, `gemma4_e2b_it_q8_0`, `gemma4_12b_it_q8_0`, `gemma4_26b_a4b_it_q4_0`.

## 5. BLOCKING DECISION A — compatibility source: **PROGRAMMATIC**

The supported set is available as structured data, two ways:

1. **HTTP:** `GET /api/capabilities` → `capabilities` (`:1914`) →
   `CapabilitiesResponse { …, model_compatibility: Vec<ModelCompatibilityTarget>, … }`.
2. **Rust:** `capabilities_response_with_plan(Option<ExecutionPlan>) -> CapabilitiesResponse`
   (`src/api/mod.rs:~1939`, currently private).

`ModelCompatibilityTarget` (`src/api/mod.rs:~300`) per row includes:
`id: &str` (e.g. `"llama32_3b_instruct_q8_0"`), `family: &str`, `quantization: &str`,
`status: &str`, `support_scope`, `full_support_status`, `frontend_readiness_gate`, plus a large
evidence/blocker surface.

**Supported-status values (verbatim):** rows carry `status` like `"supported_exact_row_smoke"`,
`"supported_exact_row_smoke_lane"`, `"supported"`; **non-supported** rows carry `"planned"`,
`"active_validation_partial"`, `"unsupported"`. The picker's supported predicate is therefore
**`status.starts_with("supported")`**, read from this ledger at runtime — no hardcoded list.

→ **Decision A is NOT prose-only.** The "expose structured support data" scope boundary in the
spec's working method does **not** trigger. The picker consumes `/api/capabilities` directly.

**Catalog vs ledger are two lists that must be JOINED** (spec §0.7): `model_compatibility` lists
~14 rows (supported + planned + unsupported); `curated_catalog()` lists the 8 *pullable* rows.
Join key is `model_compatibility[i].id == CatalogItem.catalog_id`. The picker source is the
ledger (supported rows); availability/pull is the catalog join. Some supported rows
(e.g. `mistral_*`, `qwen3_*`) have **no** catalog entry → render `[supported · no pull alias]`.

## 6. BLOCKING DECISION B — architecture: **Option B (HTTP client, child-process serve)**

Recorded + justified in `DECISIONS.md`. Key enabling find: **the repo has no HTTP-client
dependency, but already hand-rolls a blocking HTTP/1.1 client over `std::net::TcpStream`** in
`src/receipt/verify.rs` (`http_json(host, port, method, path, body, timeout)` `:681`,
`parse_http_response` `:725`). Option B therefore needs **zero new network deps** — reuse that
pattern for `load`/`capabilities`/`health`, and extend it with a line-buffered `data:` reader for
the SSE stream. This inherits the already-audited SSE wire path (constraint 2/4/5) for free and
avoids re-plumbing streaming + a token-for-token parity test that Option A would require.

- **Spawn-vs-attach:** GET `/v1/health` on `--addr`; if it answers, **attach**. Else **spawn**
  `camelid serve --addr <addr> --no-open` as a child (`std::process::Child`), poll `/v1/health`
  until ready (bounded), and **tear the child down on exit** (only if we spawned it).
- **Control-plane reuse without HTTP:** `pull` and on-disk availability run **in-process** via
  the pub `catalog::run_pull` / `curated_catalog()` + a `models/<filename>` fs check; only the
  audited lanes (load, capabilities, health, generation) go over HTTP.
- The `chat` command body is **fully synchronous** (blocking TcpStream client + blocking line
  editor + `std::process::Child`); no tokio is needed in the chat path.

## 7. Typed unsupported-state error

- `src/error.rs`: `BackendError::UnsupportedModelArchitecture(String)` →
  `#[error("unsupported model architecture: {0}")]`. (Related: `UnsupportedGguf`,
  `UnsupportedTokenizer`, `UnsupportedTensorType`, `ModelNotLoaded`.)
- Surfaced as `UnsupportedRuntimeSummary { code: "unsupported_model_architecture", message:
  String }` on `LoadedModel.unsupported_runtime` at load, and as `422` `unsupported_model_architecture`
  + that message at generation (`prepare_generation`, `:5316`).

## 8. Spec ↔ repo discrepancy (recorded per the working method)

**Constraint #4** ("`--model <arbitrary path>` at a non-supported row must refuse with the typed
error") is enforceable **only at the architecture level**, because that is the only gate the
backend actually has:

- Unrecognized/unsupported **architecture** → backend already fails closed (`unsupported_runtime`
  populated; `generation_ready=false`; `422` at generation). `chat` surfaces this verbatim and
  exits non-zero. ✅ fully covered.
- A **recognized-arch GGUF that is not a ledger exact row** (e.g. some other llama finetune Q8)
  → the backend (and today's React frontend) **would still generate**; the frontend's
  `frontend_readiness_gate` is an *advertising* signal, not a generation block. Inventing a new
  per-file/hash gate in the TUI would **reimplement and widen** behavior the engine doesn't have,
  violating constraint #2 ("reuse, don't reimplement").

**Resolution:** `chat` reuses the existing typed gate (refuse when `unsupported_runtime.is_some()`
or `generation_ready == false`) and does **not** invent new matching. The **picker** guarantees
supported-only selection (it only ever offers `status.starts_with("supported")` rows), so the
normal path cannot reach a non-supported row; the `--model` backstop catches the
arch-unsupported bypass exactly as the engine/frontend already do.
