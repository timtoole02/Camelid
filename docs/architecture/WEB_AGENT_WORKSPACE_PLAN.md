# Web Workspace Feature Plan

**Status:** Implemented preview feature on `feat/web-agent-workspace`; not production-promoted.
**Last verified:** 2026-07-22.

## Purpose

Web Workspace is an additive Camelid feature, not a product rebrand or a replacement for Chat, the CLI/TUI agent, model management, or the OpenAI-compatible API.

It lets a user select one local directory and hold a resumable, multi-turn conversation about the files in that directory. The browser owns presentation only. The server owns authorization, canonical path confinement, model eligibility, tool selection, context budgeting, persistence, and cancellation.

## Current Product Boundary

Workspace provides:

- one canonical local directory per conversation;
- one active Workspace session per Camelid process;
- follow-up messages after a completed turn;
- explicit resume and deletion of saved conversations;
- local SQLite/FTS5 episodic memory;
- exact prompt budgeting and a context inspector;
- automatic and manual reversible compaction;
- deterministic, evidence-backed file inventories;
- bounded SSE activity and turn-scoped cancellation;
- exact-model eligibility through the existing `tool_capable` capability contract.

Workspace is strictly read-only. Its advertised tool profile contains exactly:

- `read_file`;
- `list_dir`;
- literal-content `search`.

Workspace does not expose writes, edits, shell commands, network access, GUI control, subagents, or unattended execution. A legacy request with `allow_writes: true` fails with typed `400 workspace_read_only` before path or model access.

The CLI/TUI agent remains a separate preview surface with its existing broader, approval-gated capabilities.

## Governing Invariants

1. **Additive surface.** Removing the Workspace routes, view, memory module, and capability entry restores the prior product without changing ordinary Chat or CLI/TUI agent behavior.
2. **Exact model gate.** The loaded artifact must match an existing supported row with `tool_capable: true`. Saved threads also bind the model ID and exact GGUF SHA-256.
3. **Loopback management boundary.** Workspace routes require a loopback-bound server, a loopback `Host`, and exact same-authority `Origin` or same-origin fetch metadata. Vite development reaches the backend through a same-origin proxy.
4. **Server-owned sandbox.** The server canonicalizes the selected root. Every tool path is resolved under that root and escapes fail closed.
5. **Read-only enforcement at both layers.** The API rejects legacy write mode, and `ToolProfile::WorkspaceReadOnly` advertises and validates only the three read tools.
6. **Untrusted observations.** Persisted conversation memory and tool evidence enter prompts as user-role, explicitly untrusted data, never as system policy.
7. **Bounded execution.** Steps, generation, events, tool output, retained evidence, memory context, and total prompt-plus-generation tokens are capped.
8. **No partial durable turn.** Cancellation or failure during a model step cannot persist a partial assistant answer.
9. **No support widening.** Workspace availability does not promote a model, quantization, backend, platform, context window, or latency claim.

## HTTP Surface

All routes are under `/api/agent/workspace` and use Camelid's normal typed error envelope.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/models` | List exact eligible model artifacts and fit state |
| `GET` | `/browse` | Browse local directories for the folder picker |
| `GET` | `/threads?workspace=...` | List saved threads for the canonical root and active exact model |
| `GET` | `/threads/:id?workspace=...` | Load a saved transcript and metadata |
| `DELETE` | `/threads/:id?workspace=...` | Delete saved thread memory when it is not active |
| `POST` | `/threads/:id/compact?workspace=...` | Compact completed turns |
| `DELETE` | `/threads/:id/compact?workspace=...` | Undo the most recent compaction |
| `POST` | `/sessions` | Start a new thread or explicitly resume one |
| `GET` | `/sessions/:id` | Read non-sensitive session state and context capacity |
| `GET` | `/sessions/:id/events` | Claim the bounded SSE stream for the pending turn |
| `POST` | `/sessions/:id/messages` | Send an idempotent follow-up message |
| `DELETE` | `/sessions/:id` | Request cooperative cancellation |
| `POST` | `/sessions/:id/decisions` | Shared bridge compatibility route; current read-only tools do not request approval |

A create request includes the root, first goal, optional saved `thread_id`, and bounded step/generation settings. A follow-up includes both text and a caller-generated `client_message_id`; duplicate IDs return the existing turn rather than inserting it twice.

## Turn Lifecycle

The active session state is one of:

- `waiting_for_events`;
- `running`;
- `idle`;
- `cancelling`;
- `cancelled`;
- `failed`.

A new or follow-up turn is installed as `waiting_for_events`. Claiming its event stream within 30 seconds moves it to `running`; an unclaimed turn is persisted as aborted and becomes `cancelled`. A successful persisted answer returns the session to `idle`. Cancellation preserves `cancelled` even when it races event-stream claim or worker completion, and a failed turn becomes `failed`. `idle`, `cancelled`, and `failed` can all accept a follow-up, so terminal outcomes are turn-scoped rather than thread-fatal.

Only `waiting_for_events`, `running`, and `cancelling` block model load or unload transitions.

The SSE event vocabulary is:

- `session.started`;
- `turn.started`;
- `memory.updated`;
- `memory.compacted`;
- `model.delta`;
- `model.timing`;
- `model.answer`;
- `tool.call`;
- `approval.required` (shared bridge compatibility; not expected for the read-only profile);
- `tool.result`;
- `session.notice`;
- `session.finished`;
- `session.error`.

Every envelope carries a session ID and monotonically increasing sequence number.

## Local Conversation Memory

`src/chat/workspace_memory.rs` owns an app-managed SQLite schema at:

- `%LOCALAPPDATA%\camelid\workspace-memory.sqlite3` on Windows;
- `$XDG_DATA_HOME/camelid/workspace-memory.sqlite3` or the standard home fallback elsewhere;
- `CAMELID_WORKSPACE_MEMORY_DB` when an isolated path is explicitly configured.

Schema v6 stores:

- threads keyed by ID, canonical root, model ID, and model SHA-256, with a deterministic title from the first non-empty initial-prompt line (normalized and capped at 80 characters);
- idempotent ordered turns keyed by `client_message_id`, including terminal outcome;
- bounded tool evidence with SHA-256 observation integrity checks;
- reversible compaction boundaries and history;
- an FTS5 index over user and assistant turn text.

Writes use immediate transactions. Unknown newer schema versions and malformed current schemas fail closed. Foreign keys use cascade deletion and connections enable WAL mode with a bounded busy timeout.

For a turn, memory retrieval keeps up to three recent uncompacted answered turns, lexically retrieves up to six older relevant answered turns, then includes bounded evidence associated with selected turns. Aborted, capped, repeated, and failed attempts remain visible and idempotent but do not enter future model context. Raw unbounded tool output, hidden reasoning, and approval grants are not persisted.

Compaction changes which completed turns are always recent; it does not delete their transcript, FTS entry, or evidence. Undo restores the previous boundary. Automatic compaction runs only after a successful durable turn when there are at least four turns and exact prompt plus reserved generation reaches 75% of the Workspace envelope.

## Exact Context Budget

Workspace resolves one adaptive total envelope when a session starts. Camelid
reads the active GGUF's native context and KV dimensions, samples currently
available host RAM after the model is loaded, divides 70% of that memory across
active generation slots and retained prompt-prefix caches for conservative f32 KV,
and rounds down to a 1,024-token quantum. The result is clamped by Camelid's
validated agent-context envelope, the model and server limits, a 65,536-token
operational ceiling, and the optional `CAMELID_AGENT_CONTEXT_MAX_TOKENS`
operator cap. Native GGUF metadata does not itself widen a supported window;
the bounded-paging exception below is keyed to exact earned Qwen3 4B rows.
Unknown telemetry falls back to 8,192 tokens. The selected value stays fixed
for the session and is inherited by delegated Code workers. The API reports the
raw memory-derived capacity separately from the 8,192-token minimum operational
recommendation; under severe pressure the allocator remains authoritative.

Generation allowance and action steps remain separate controls. Code has no
arbitrary action-step limit, accepts a 64 KiB written task specification, and
uses context paging by default so durable task/source state does not depend on
replaying an ever-growing transcript. Read-only Workspace retains its bounded
step policy.

The logical session envelope and the paging working set are intentionally
different. With paging enabled, the exact Qwen3 4B Q8_0 Code row exposes a 16K
logical task envelope while the paging capsule remains bounded to its 8K
input/output/safety working set. This avoids turning a larger session ceiling
into an unbounded cold prefill on every step. It does not authorize a 16K model
prompt; the active request remains inside the exact row's validated envelope.
Q4_K_M does not inherit this exception. It keeps the legacy 8K operational
agent envelope needed to fit the paging capsule and reserves, but that value is
not a context-support promotion: its formal parity ladder remains 512/1,024
until the intervening 2K near-tie and the longer buckets are qualified.

Before every model step, Camelid renders the real chat template with the actual tool schemas and tokenizer. The required system policy, current user message, and latest native tool call/result pair stay intact. Earlier tool exchanges are reduced to bounded observations. If the request is too large, optional memory is removed first, followed by complete older user/assistant turn pairs. If required content still cannot fit, the turn fails instead of overflowing.

The Workspace request carries the private `camelid_context_budget_tokens` field. The generation server independently verifies prompt tokens plus requested generation against that ceiling, so the UI/agent compiler is not the sole guard.

`memory.updated` reports exact prompt tokens, generation reserve, and total budget. Its category split for system, tools, messages, recent memory, retrieved memory, evidence, and current tool results is an estimate that reconciles to the exact prompt total; it is not separate tokenizer accounting for each category.

## Grounded File Answers

File-inspection requests must obtain a successful observation before the final answer. The model cannot turn a positive observation into an unsupported absence claim.

When the user explicitly names an existing file path, Workspace resolves it through the sandbox and requires a successful `read_file` observation for that exact file before accepting a final answer. Broader requests remain bounded by the step limit and must disclose inspected and uninspected scope rather than claim complete coverage.

For immediate non-recursive extension inventories, Camelid derives the answer from successful `list_dir` output:

- replacement is limited to unqualified list requests with exactly one directory observation;
- case-distinct matching filenames remain distinct;
- results are lexically sorted;
- directories and nonmatching entries are excluded;
- the result is rendered as stable Markdown bullets;
- nested-folder scope and truncation are disclosed;
- empty results are grounded in the observation;
- control characters, percent signs, and backticks are represented safely.

Directory listing retains at most 4,096 observed entries and explicitly says when additional entries cannot be paged. Reads, searches, pages, hits, and observations are separately bounded; Workspace observations are clipped before history insertion.

## Cancellation and Deadlines

Cancellation is turn-scoped rather than process-global. The model client uses one absolute 90-second deadline starting with the first prompt preflight and covering every fitting retry, HTTP header wait, and SSE body stream, with periodic wakeups to observe cancellation. This is the read-only Workspace lane described by this document; the separate Code surface runs the same stream cancellable but without the wall-clock deadline (see `WEB_CODE_MODE.md`).

The agent checks cancellation before work and again after each model step. If cancellation arrives during streaming, partial text is discarded before reporting a final answer or committing a turn. The API also protects cancellation state from stale worker completion and event-claim races.

Dropping the event stream requests cancellation. A failed cancellation request is shown as an error in the UI and is never mislabeled as stopped.

## Frontend Behavior

Workspace is a normal operational view in the existing Web UI. It adds:

- a canonical folder picker and exact-model eligibility state;
- a transcript with user and assistant turns;
- a follow-up composer while the thread is idle;
- saved-thread resume and delete actions;
- a collapsible activity history for tool/model events;
- a context inspector with exact budget, estimated categories, model timing, resident CUDA status when available, and Compact/Undo controls;
- responsive desktop and mobile layouts.

The Web UI contains no write toggle or write-approval workflow. Clearing the active view does not silently delete saved memory; deletion is explicit.

## Validation

The branch is covered by:

- focused agent, tool, memory, cancellation, budget, inventory, and rendering unit tests;
- eight Workspace HTTP integration tests, including authorization, inaccessible roots, model gating, follow-up routing, and fail-closed legacy write mode;
- frontend build, reducer, Markdown, integration, and deterministic browser UI smokes;
- desktop and 390x844 mobile checks for overflow, overlap, list spacing, compaction, Undo, and cancellation failure display;
- the full macOS, Ubuntu all-features, Windows, desktop, frontend, public-scrub, validation-script, and final CI gates.

Exact-model receipts use the pinned 2,497,280,256-byte `Qwen3-4B-Q4_K_M.gguf` with SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` on an RTX 4060 Laptop GPU with all 36 layers resident. They cover multi-turn recall, idempotency, explicit resume, reversible compaction, FTS recall, sustained follow-ups, and cancellation races with no partial persisted turn.

The historical bundle `qa/evidence-bundles/workspace-qwen3-4b-q4km-20260717T165404Z-head-8c2a2b74/` records the earlier write-capable prototype. It remains an auditable historical receipt but does not define the current read-only product boundary.

## Known Limits and Non-Claims

This preview does not claim:

- production-ready interactive latency;
- a population-level latency or retrieval-recall SLA;
- per-request proof of resident GPU execution versus CPU fallback;
- population-level validation of adaptive context sizing under latency and
  memory pressure;
- prefix reuse or appendable GPU KV sessions;
- semantic or embedding-based retrieval;
- recursive inventory without explicit bounded observation;
- support for neighboring models, quantizations, or hardware paths.

On the measured sustained exact-model run, follow-up elapsed p50 was 15.110 seconds and p95/max was 18.299 seconds; TTFT p95 was 17.593 seconds. Repeated prompt prefill remains the main limitation. These are single-host measurements, not a production SLA.

The final exact-model rerun after narrowing hostile-filename display encoding was blocked locally by enterprise Code Integrity rejecting newly linked executable hashes. Compiler, renderer, and CI coverage pass, but that final model-backed rerun must not be claimed.

## Release and Rollback

Workspace remains labeled preview until its latency, retrieval, and execution-path evidence meet the project's promotion standard. Public claims must stay narrower than the receipts above.

Rollback is additive: remove the Workspace routes and manager from `src/api`, the Workspace bridge/memory modules, the Web UI view and styles, and the `web_workspace` capability entry. Ordinary Chat, model loading, the OpenAI-compatible API, and CLI/TUI agent behavior remain unchanged.
