# Workspace Conversational Memory — Design Spec

**Codename:** Budgeted Working‑Set Memory (BWSM)
**Status:** Implementation candidate on `feat/web-agent-workspace`. **Not production-promoted.**
**Target branch context:** `feat/web-agent-workspace` (fork of Camelid), the branch that adds the Web Agent Workspace.
**Date drafted:** 2026‑07‑20.
**Implementation update:** 2026‑07‑21.

> **Important:** Sections 1–17 preserve the original proposal for review history. Where they conflict with the implementation update below, the update governs. The original rolling-memo/folder-map design was rejected as the primary memory architecture after code and performance review.

## Implementation update — 2026‑07‑21

The implemented candidate is a **thread-scoped episodic context compiler**, not a rolling model-written memo:

- `src/chat/workspace_memory.rs` owns an app-managed bundled SQLite/FTS5 schema-v6 store under `%LOCALAPPDATA%\camelid\workspace-memory.sqlite3` (or the platform data-directory fallback). `CAMELID_WORKSPACE_MEMORY_DB` overrides the path for isolated tests and controlled deployments.
- Durable records are explicit threads, user/final-answer turns, and bounded tool evidence with SHA-256 observation digests. Approval grants, hidden reasoning, and raw unbounded tool dumps are not persisted.
- Thread creation, turn+evidence commit, schema migration, idempotency, and deletion use immediate transactions. Newer unknown schemas fail closed without overwrite.
- Saved conversations are listed only for a caller-supplied canonical folder. Each title is deterministically derived from the first non-empty line of the initial prompt, normalized and capped at 80 characters; existing threads migrate from their earliest stored user turn. Resume is explicit in the UI; a new conversation remains the default. Resume verifies canonical root, original model ID, and the exact loaded GGUF SHA-256.
- Successful turns return the durable thread to `idle`; only waiting/running/cancelling turns block model transitions. Workspace is server-enforced read-only; write/edit requests fail closed even if a caller submits the legacy `allow_writes` field.
- Model streaming uses the turn-scoped cancellation flag rather than the process-global CLI flag. One 90-second model-step deadline starts with the first prompt preflight and is shared by every fitting retry, the wait for generation HTTP headers, and the SSE body. Cancellation is observed during preflight and streaming, then rechecked after every model step before any partial answer can be reported or persisted. Cancellation state is preserved if DELETE races event-stream claim or worker completion.
- `POST /api/agent/workspace/sessions/:id/messages` accepts idempotent follow-ups. The existing SSE route streams one turn at a time.
- Workspace file observations are clipped to 2 KiB before reporter/history insertion. `read_file` supports line ranges, `list_dir` supports pages, and `search` supports a validated hit limit.
- Before every model step, old native tool exchanges are compacted into bounded untrusted evidence while the latest native call/result pair remains intact.
- `POST /api/generation/preflight` uses the real model template, tool schemas, and tokenizer without decode. Workspace evicts older untrusted memory, then complete prior turn groups, until exact prompt tokens plus generation allowance fit.
- `/v1/chat/completions` independently enforces `camelid_context_budget_tokens`. Workspace currently uses a conservative static total envelope of 4,096 tokens and a maximum 1,024-token generation allowance (512 default).
- Persisted user/assistant episodes and evidence are injected only as `<workspace_memory untrusted="true">` user-role data, never as system policy or trusted facts.
- Conversation compaction is reversible and lossless: it moves completed turns out of the always-recent set while retaining raw transcript, FTS retrieval, evidence, and undo history. After a successful durable turn, Workspace automatically compacts at 75% exact prompt-plus-reserved-generation use once at least four turns exist; manual compact and undo remain available.
- The context inspector shows exact prompt+generation allocation, an explicitly estimated category breakdown that reconciles to the exact prompt total, measured model-call/TTFT timing, authoritative resident CUDA capacity when available, and compact/undo controls.
- Workspace exposes only the three scoped read tools (`read_file`, `list_dir`, and literal-content `search`). Explicit file-inspection requests must obtain a successful read observation before answering, and observed filenames prevent contradictory absence claims.
- Existing file paths explicitly named by the user are resolved through the sandbox and must receive a successful `read_file` observation before Workspace accepts a final answer. Broad requests that exceed bounded inspection must disclose remaining scope.
- Unqualified, immediate non-recursive extension inventories are deterministic after exactly one `list_dir`: case-distinct matching filenames remain distinct, results are sorted and rendered as stable bullets, and directory/nesting scope is explicit. Qualified or semantic requests remain model-owned. Model prose cannot add directory entries, reverse a positive observation, or fabricate the empty result.

### Validation completed

- 150 `chat::` library tests pass, including deterministic inventory, hostile-filename encoding, evidence-first and list-only contradiction guards, file-target content search, honest capped-directory pagination, cancellation/deadline, bounded-tool, persistence, FTS, and compaction cases.
- 8 Workspace HTTP integration tests pass, including fail-closed rejection of legacy write mode before path/model access.
- The complete Rust library suite passes with the packaged NVRTC runtime on PATH: 975 passed, 56 external-fixture/manual-perf tests ignored, 0 failed.
- SQLite/FTS/evidence/compaction tests cover concurrent first-open, v1→current migration, malformed-current-schema refusal, evidence tamper detection, deletion, lossless compaction, FTS recall after compaction, and undo. Debug test execution may be blocked locally by Windows Application Control; all targets compile and release-path HTTP compaction passed.
- Exact budget fitter, bounded file continuation, strict all-target Clippy, scoped rustfmt, frontend production build, Markdown/reducer/integration/UI smokes, and a deterministic read-only browser E2E pass. The browser E2E verifies 14 exact files, no directory leakage, two-digit list-marker spacing, no horizontal or vertical overlap at 1280×800 and 390×844, automatic compaction plus Undo, and zero write controls.
- A current-source exact-Qwen inventory receipt reached the deterministic backend and proved the 14-file count, directory/non-matching exclusion, write-mode rejection, and scope disclosure. It then exposed over-encoding of underscores in displayed filenames. The narrowed display encoding compiles under strict Clippy and passes hostile-filename renderer coverage, but the final exact-model rerun is blocked on this host because enterprise Code Integrity rejects each newly linked executable hash until it is signed/allowlisted. Do not claim a final model-backed receipt for the narrowed display encoding yet.
- Final-artifact exact Qwen3‑4B‑Q4_K_M normal-flow receipt: `target/workspace-memory-final-artifact-soak/summary.json`. The SHA-pinned 2,497,280,256-byte model ran on RTX 4060 Laptop CUDA resident K-quant with all 36 layers resident. Authoritative resident capacity was 29,946 positions with no layer offload. Four read-only turns passed bounded ranged read, in-session recall without re-read, duplicate-message idempotency, explicit new-session resume, cross-session recall, reversible compaction+undo, compacted FTS recall without re-read, exact category reconciliation, and four persisted turns. Informational wall times were 19.801 s, 13.174 s, 14.067 s, and 15.209 s.
- Final-artifact sustained receipt: `target/workspace-memory-final-artifact-sustained/summary.json`. One initial exact-model file read plus 10 follow-ups produced 11 persisted turns. Every follow-up recalled the code without tools, stayed within budget, and returned the session to `idle`. Follow-up elapsed p50 was 15.110 s and p95/max was 18.299 s; model-call p95 was 17.894 s and TTFT p95 was 17.593 s. Prompt size ranged from 576 to 846 tokens.
- Historical final-artifact cancellation receipt: `target/workspace-memory-final-cancel-races/summary.json`. At the time of that run, DELETE after a real `model.delta` produced `aborted` in 21 ms and DELETE before event-stream claim produced `aborted` in 26 ms; neither path persisted a turn in that pre-schema-v5 implementation. Current schema-v5 behavior supersedes only that persistence detail: each terminal attempt is durable with outcome `aborted`, while partial assistant text remains discarded and unsuccessful attempts remain outside future model context. Focused tests cover the claim deadline, preflight/stream deadline, and cancellation during a model step.

### Current implementation update (2026-07-28)

The original proposal below predates Camelid's embedding lane. Workspace now has an optional semantic assist: if the exact supported `nomic-embed-text-v1.5.Q8_0.gguf` row is registered, a session lazily indexes a bounded set of UTF-8 source/document chunks in memory, embeds them with `search_document:` prefixes, and injects the top cosine-ranked excerpts as untrusted memory. It never writes an index into the workspace, skips symlinks/build/vendor/model data, and fails soft to the existing lexical/tool path. The historical design statements below remain useful proposal context but no longer establish the current absence of embeddings.

### Production promotion blockers

- The 4,096-token application envelope is below the observed authoritative 29,946-position resident capacity, but is not yet dynamically selected from capacity plus a population-level latency profile.
- Runtime responses do not yet expose per-request resident-prefill/decode versus CPU-fallback counters, so no-fallback must not be claimed.
- There is no appendable GPU KV session; every agent step still re-prefills its compiled prompt.
- Lexical retrieval has deterministic unit coverage and an optional exact-row semantic assist, but neither has a measured production recall benchmark.
- Prompt/tool reduction and bounded memory keep the final 10-follow-up run correct, but measured elapsed p95 is still 18.299 seconds and TTFT p95 is 17.593 seconds. This is one host/run, not a production SLA, and is too slow to claim an interactive production experience. Repeated prefill dominates; prefix/KV reuse remains the likely next engine lever.

Do not describe this candidate as production-ready until those gates have receipts.

---

**Original proposal purpose:** provide the design context that led to the implementation. Tags below reflect the 2026‑07‑20 proposal, not the current implementation unless explicitly updated.

---

## 0. How to read this document (anti‑hallucination contract)

Every claim below is tagged:

- **[VERIFIED]** — read directly from the codebase during design and believed accurate as of this branch. Still re‑confirm before coding; the tree is dirty/uncommitted in places.
- **[DOCUMENTED]** — taken from repo docs/notes (`COMPATIBILITY.md`, `STATUS.md`, memory notes), not independently re‑measured here.
- **[ASSUMED]** — a reasonable inference **not** verified. Must be confirmed by reading code before relying on it.
- **[PROPOSED]** — new design in this spec. Does not exist yet.
- **[TO‑VERIFY]** — an explicit open question the implementer must resolve first.

If a statement has no tag it is either a definition or a design decision internal to this spec (i.e., **[PROPOSED]**).

Do not treat any number in the "budget math" as ground truth — they are illustrative until calibrated on a real device (see §5, §12, §16).

---

## 1. Problem statement

### 1.1 What Workspace was when this proposal was drafted (2026‑07‑20) **[SUPERSEDED]**

> **Correction (2026‑07‑22):** the shipped Workspace is server-enforced read-only. Its tool surface is exactly `read_file`, `list_dir`, and literal-content `search` (§2.3); no write or edit tool exists, so nothing is approval-gated, and follow-up turns with durable memory shipped with this design. The bullets below describe the five-tool write-capable prototype this proposal was drafted against, which the final Web Workspace PR replaced before merge. `COMPATIBILITY.md` states the governing rule.

The Web Agent Workspace was a **single‑shot, bounded agent task** over one local folder:

- The user picks **one canonical directory** and **one goal**, then Starts.
- A plan‑act‑observe agent loop ran with five file tools, read‑only tools auto‑run, every write approval‑gated.
- When the loop ended, the session was over. There was **no follow‑up turn and no memory** — each Start was a fresh session. The original 2026‑07‑17 Workspace design treated **"background persistence"** as *out of scope* for the first release, so the absence of memory was intentional for v1, not an accident.

### 1.2 The product gap
Users' natural mental model is **"chat with my folder"** — ask, get an answer, then follow up ("tell me more", "now look at the architecture folder"), with the assistant remembering the conversation. The current one‑shot model does not support this. The missing primitive is **conversational memory (continuity)**, both **within a session** (multi‑turn) and **across sessions** (resumable).

### 1.3 The hard constraint that shapes the whole design
On a constrained device (reference box: **RTX 4060 Laptop, 8 GB VRAM, ~16 GB RAM**), the binding limit is **not** the model's logical context ceiling — it's the **resident KV window**: how many tokens stay in VRAM before decode falls back to CPU/host and slows to seconds/token (feels like a hang). Relevant verified constraints:

- **No cross‑request prefix/KV cache** in the serving path. **[VERIFIED]** (grep of `src/api`, `src/inference/kv_cache.rs` for prefix/prompt/kv reuse returned nothing.) → **every model call re‑processes the full prompt**, so per‑turn cost grows with conversation length.
- **No KV‑cache quantization** (no 8/4‑bit KV). **[DOCUMENTED]** (memory note "No 8/4‑bit KV"; `CAMELID_KV_F16` is an opt‑in CPU f16 KV path only.)
- **Semantic embeddings service now exists.** **[UPDATED 2026-07-28]** The exact Nomic v1.5 Q8_0 row serves `/v1/embeddings` and an optional bounded Workspace semantic index; generic embedding models and a persistent vector database remain unsupported.
- Reference model **Qwen3‑4B‑Q4_K_M**: native context **40,960** (KV) / single‑shot prefill ceiling **16,384**. **[DOCUMENTED]** (`COMPATIBILITY.md`). Fully GPU‑resident model peak ≈ **4.9 GB** VRAM. **[DOCUMENTED]**
- Exact resident **token** capacity on this device is **[TO‑VERIFY]** (estimated low‑five‑figures but not measured — do not rely on a number until §16‑1 is done).

### 1.4 Design thesis (one line)
> **The folder is the long‑term store; the model window is a *budgeted working set*; "memory" is a rolling memo + a folder map that persist to disk; the agent retrieves on demand; and a hard *budget guard* keeps the live prompt inside the device's fast resident window.**

This makes a *small, fast* window behave like a *large, persistent* memory, and it requires **no engine changes** for v1.

---

## 2. Current architecture reference (for the implementer)

### 2.1 Backend files **[VERIFIED]**
- `src/api/workspace.rs` — Workspace HTTP handlers + `WorkspaceSessionManager` (one active session), request/response structs, loopback `authorize` guard, the browse endpoint, `simplify_path` (Windows `\\?\` stripping).
- `src/chat/workspace_bridge.rs` — `run_live(...)`, `WorkspaceRunConfig`, the bounded event bridge (`bridge`, `WorkspaceEvent`, `WorkspaceBridgeControl`), and construction of the `Sandbox` + system prompt + `AgentConfig`.
- `src/chat/agent.rs` — the agent loop (`run_loop`), traits `ModelDriver`/`Reporter`/`Approver`, `LiveDriver` (calls the chat API), `system_prompt(sandbox, tools)`, `history_to_messages(...)`, `AgentMsg` enum.
- `src/chat/tools.rs` — `Sandbox` (canonical-root confinement, `resolve`, `rel`, `root`, `root_display`), `ToolProfile` (`Full`, `WorkspaceReadOnly`), tool specs, tool execution, size caps.
- `src/chat/tool_parse.rs` — parses model output into `ToolCall`s (Qwen/Hermes `<tool_call>{json}</tool_call>`, Llama JSON, Mistral, Ornith XML), plus the lenient Windows‑path JSON repair added on this branch.

### 2.2 HTTP routes (current) **[VERIFIED]**
```
POST   /api/agent/workspace/sessions            create_session (one active session; 409 if busy)
GET    /api/agent/workspace/models              compatible_models (tool_capable exact rows)
GET    /api/agent/workspace/browse              browse (read-only dir listing; loopback-guarded)
GET    /api/agent/workspace/threads             list saved threads for one canonical root
GET    /api/agent/workspace/threads/:id         load a saved transcript
DELETE /api/agent/workspace/threads/:id         delete an inactive saved thread
POST   /api/agent/workspace/threads/:id/compact compact completed turns
DELETE /api/agent/workspace/threads/:id/compact undo the latest compaction
GET    /api/agent/workspace/sessions/:id        session_status
DELETE /api/agent/workspace/sessions/:id        cancel_session
GET    /api/agent/workspace/sessions/:id/events SSE event stream (event name: "workspace")
POST   /api/agent/workspace/sessions/:id/messages idempotent follow-up turn
POST   /api/agent/workspace/sessions/:id/decisions  vestigial approval decision (the read-only lane never creates an approval, so it can only answer 409)
```
Auth: loopback-bound server + loopback `Host` + exact same-authority `Origin` or same-origin fetch metadata (`authorize` / `local_management_request_allowed`). **[VERIFIED]**

### 2.3 The three tools (WorkspaceReadOnly profile) **[VERIFIED]**
`read_file`, `list_dir`, and literal-content `search`. All are confined to the canonical workspace root by `Sandbox::resolve(raw, must_exist)`, which joins relative paths to the root, canonicalizes after symlink resolution, and rejects anything not under the root. No write/edit/exec/network tool is advertised or accepted.

Relevant caps in `src/chat/tools.rs` **[VERIFIED]**:
```
MAX_READ_BYTES   = 64 * 1024   // read_file output cap
MAX_OUTPUT_BYTES = 16 * 1024
MAX_RANGED_FILE_BYTES = 8 * 1024 * 1024
MAX_LIST_ENTRIES = 4096
Workspace search = 20 hits, 5000 files, 2 seconds
```

### 2.4 Session limits (`src/api/workspace.rs`) **[VERIFIED]**
```
DEFAULT_MAX_STEPS = 12    MAX_STEPS  = 32
DEFAULT_MAX_TOKENS= 512   MAX_TOKENS = 1024
MAX_GOAL_BYTES    = 4 * 1024
EVENT_BACKLOG     = 128   EVENT_STREAM_BUFFER = 128
EVENT_CLAIM_TIMEOUT = 30 seconds
```

### 2.5 Agent loop shape **[VERIFIED, with specifics TO‑VERIFY]**
- `WorkspaceRunConfig { addr, workspace: PathBuf, goal, model_id, family, max_steps, max_tokens, temperature }`.
- History is a `Vec<AgentMsg>` where `AgentMsg ∈ { System(String), User(String), Assistant(...), ToolResult { name, outcome } }`. **[VERIFIED]** (from `history_to_messages`).
- `LiveDriver::step(history, tools)` builds a chat request with **the full history every call** (`history_to_messages`), sends to the local chat API, and returns `ModelStep::{Calls(..)|Text(..)}` by reading structured `tool_calls` or falling back to `tool_parse::parse(content, family)`. **[VERIFIED]**
- For Qwen family, prior tool calls/results are replayed as native `<tool_call>` / `<tool_response>` markers. **[VERIFIED]**
- The exact signature of `run_loop(...)` and how `Reporter`/`Approver` are wired is **[TO‑VERIFY]** by reading `src/chat/agent.rs` in full before integrating.

### 2.6 Frontend **[VERIFIED]**
- `frontend/src/views/WorkspaceView.jsx` — the view. Recently redesigned to a **Result‑first** layout: a prominent markdown Answer panel (`AssistantMarkdown` + `copyText` from `frontend/src/lib/markdown.jsx`) with a collapsible **"What Camelid did"** timeline (`<details>`).
- `frontend/src/lib/workspaceAgent.js` — API client + `reduceWorkspaceEvent(state, envelope)` reducer + endpoint helpers (`workspaceEndpoint`, `workspaceModelsEndpoint`, `workspaceBrowseEndpoint`, `createWorkspaceSession`, `browseWorkspaceFolders`, etc.).
- Event vocabulary handled by the reducer **[VERIFIED 2026‑07‑22]**: `session.starting`, `session.started`, `turn.starting`, `turn.user`, `turn.stopping`, `turn.stop_failed`, `thread.restored`, `model.delta`→`model.live`, `tool.call`, `tool.result`, `model.answer`, `session.finished`, `session.error`, `session.reset`; `session.notice`, `memory.updated`, and `memory.compacted` flow through the shared event log to the view. `approval.resolved` is no longer handled, and `approval.required` fails closed: the read-only reducer surfaces it as a session error, not an approval prompt.
- One session at a time; SSE via `EventSource`.

### 2.7 Signals that already exist and this design depends on **[VERIFIED / TO‑VERIFY]**
- `crate::capability::HardwareProfile::cached()` is used in `src/api/workspace.rs` (workspace_model_options). **[VERIFIED]** — exposes hardware info; **exact fields (free VRAM, etc.) are [TO‑VERIFY].**
- A model‑fit advisor exists (`crate::fit::FitVerdict`, `docs/architecture/MODEL_FIT_ADVISOR_PLAN.md`). **[VERIFIED]** — computes whether a model fits a host, so VRAM/size signals are available somewhere.
- `src/inference/kv_cache.rs` holds KV cache logic and (per memory notes) a **VRAM‑sized resident KV cap** with CPU fallback beyond it. **[VERIFIED file exists; exact API TO‑VERIFY]**.
- Tokenizer access for **token counting** from a manager layer is **[TO‑VERIFY]** (there is `/api/models/tokenizer/encode`; the manager needs a way to count tokens — either that endpoint, or a direct tokenizer handle in‑process).

---

## 3. Goals and non‑goals

### 3.1 Goals
1. **Multi‑turn conversation** over one folder: ask → answer → follow‑up, with continuity.
2. **Cross‑session memory**: resume a folder conversation later (persisted memo).
3. **Bounded, fast context**: the live prompt never exceeds the device's resident window → no CPU‑fallback hang.
4. **No engine work for v1**: implement as a manager layer over the three read-only tools, exact token counting, and existing hardware/fit signals.
5. **Graceful degradation**: when memory pressure hits, trim/summarize with a visible signal — never hang, never silently corrupt.
6. **Preserve the safety contract**: still one folder, three read-only tools, exact-origin loopback management, no writes/shell/network/GUI/subagents/unattended.

### 3.2 Non‑goals (explicitly out of scope for this design)
- Becoming an IDE (no editor, diff view, run/debug).
- **RoPE context extension (YaRN/NTK/linear scaling)** — rejected: it *increases* KV memory (wrong direction on 8 GB) and going past native context is not parity‑validated (Camelid is parity‑strict).
- **KV‑cache quantization** and **prefix/KV reuse** — high value but **engine work**; deferred to a later phase (§15 Phase 4). v1 must not depend on them.
- **Semantic embeddings / vector DB** — historical v1 non-goal. The current exact-row in-memory semantic assist is implemented; a persistent vector database remains deferred.
- Multi‑folder, multiple concurrent sessions, unattended/background execution.

---

## 4. The three durable artifacts **[PROPOSED]**

All memory state lives in **three artifacts**, not in an ever‑growing prompt. Two persist to disk so memory survives across sessions.

### 4.1 Folder Map (`map`)
A cheap, once‑per‑attach index of the folder — a **menu** the agent retrieves from, never the contents.

```jsonc
{
  "schema": "camelid.workspace.map/v1",
  "root": "C:\\path\\to\\folder",           // display form (\\?\ stripped)
  "generated_at": "2026-07-20T..Z",
  "entry_count": 128,
  "truncated": false,                        // true if entry/byte caps hit
  "entries": [
    {
      "path": "src/auth.rs",                 // relative to root
      "kind": "file",                        // "file" | "dir"
      "size_bytes": 8123,
      "signature": "fn login(...), struct Session, // handles OAuth"  // headings/symbols/first-lines, capped ~200 chars
    }
  ]
}
```
- Built by walking the folder (`list_dir` recursively) with **bounded depth, entry count, and total bytes**; dot‑dirs skipped (mirror the browse endpoint rules).
- `signature` is cheap and **extractive** (no model): for code, top‑level symbol lines; for docs/markdown, headings; else first N non‑blank lines. **Never** the full file.
- Persisted (see §9). Rebuilt/refreshed on demand or when the folder changes materially (staleness policy in §9.3).

### 4.2 Rolling Memo (`memo`)
The compressed, durable memory of the conversation.

```jsonc
{
  "schema": "camelid.workspace.memo/v1",
  "root": "C:\\path\\to\\folder",
  "updated_at": "2026-07-20T..Z",
  "summary": "Running abstractive summary of the conversation so far (bounded).",
  "facts": [                                  // pinned, high-value, rarely evicted
    "Auth logic lives in src/auth.rs (login, Session).",
    "User wants a security review focused on token handling."
  ],
  "turn_count": 14                            // total turns folded so far
}
```
- `summary` + `facts` are **token‑bounded** (see slot budget §5).
- Persisted → this is what makes memory **cross‑session**.

### 4.3 Recent‑Turns Buffer (`recent`)
The last *K* user/assistant turns kept **verbatim** for immediate coherence. In memory (and optionally persisted for resume). Each entry: `{ role: "user"|"assistant", content, tokens }`.

---

## 5. The budget model (device‑aware) **[PROPOSED]**

### 5.1 Definitions
- `W` = **resident window** (tokens that stay fast in VRAM for the loaded model on this device). **[TO‑VERIFY how to obtain]** — see §16‑1. Until measured, use a conservative static default per model class.
- `B` = usable budget = `floor(SAFETY * W)`, `SAFETY = 0.7` (leave headroom for generation + fragmentation). **[PROPOSED]**
- Everything assembled into the prompt each turn must satisfy `assembled_tokens ≤ B ≤ W`.

### 5.2 Slot partition of `B`
| Slot | Share of B | Contents | Overflow handling |
|---|---|---|---|
| `S` System + tool defs | fixed (measure once) | system prompt (`system_prompt`), tool JSON | constant; subtract first |
| `M` Memo | ~15% | `summary` + `facts` | hierarchical re‑summarize |
| `R` Recent turns | ~25% | last K verbatim turns | evict oldest → fold into `M` |
| `Q` Retrieved working set | ~45% | folder chunks/excerpts used this turn | drop lowest‑ranked chunks |
| `G` Generation headroom | ~15% | reserved for the answer | maps to `max_tokens` |

Shares are **[PROPOSED]** starting points; make them config constants and tune after calibration. `S` is measured at session start (it's constant for a given model/tool set) and deducted before splitting the rest.

### 5.3 Estimating `W`
Two acceptable paths (implementer picks after §16‑1):
1. **Derived**: `W ≈ (free_VRAM_bytes * VRAM_SAFETY) / kv_bytes_per_token`, where `kv_bytes_per_token = 2 (K+V) * n_layers * n_kv_heads * head_dim * bytes_per_elem`. Requires reading these from the loaded model + `HardwareProfile`. **[TO‑VERIFY these are exposed]**.
2. **Calibrated static table**: a per‑(model, quant, gpu‑class) constant measured once and stored, chosen because it is robust and avoids trusting live VRAM math. Recommended for v1.

Whichever path, **clamp `W` to the model's native ceiling** (`min(W, native_ctx)`; for Qwen3‑4B that is 40,960, but on 8 GB the resident value will bind first).

---

## 6. The per‑turn algorithm **[PROPOSED]**

### 6.1 Session start / folder attach
```
on attach(folder, model):
    ensure Sandbox(folder) is valid (reuse existing create_session validation)
    W  = resolve_resident_window(model, device)     // §5.3
    B  = floor(0.7 * W)
    S  = measure_tokens(system_prompt + tool_defs)
    load or build:
        map   = load_map(folder) or build_map(folder)      // §4.1
        memo  = load_memo(folder) or empty_memo()          // §4.2 (cross-session)
        recent = []                                         // §4.3
    slots = partition(B - S)                                // §5.2
```

### 6.2 Handling one user message
```
on user_message(text):
    # 1) ASSEMBLE (never exceed budget)
    ctx = []
    ctx += System(system_prompt)                            # S
    ctx += render_memo(memo, cap=slots.M)                   # M (summary + facts)
    ctx += render_recent(recent, cap=slots.R)               # R (verbatim, newest first until cap)
    working = select_working_set(text, memo, map, cap=slots.Q)   # Q  (see §7)
    ctx += render_working(working)
    ctx += User(text)
    assert measure_tokens(ctx) <= B                         # budget guard (§6.3)

    # 2) GENERATE (agent loop, max_tokens ~ slots.G)
    #    The model may ALSO call list_dir/search/read_file mid-loop.
    #    Tool results are TRANSIENT: used, then reduced (retrieve-don't-retain, §7.3).
    answer, transcript = run_agent_loop(ctx, tools, budget_guard, max_tokens=slots.G)

    # 3) UPDATE MEMORY
    recent.append(Turn(user=text))
    recent.append(Turn(assistant=answer))
    while measure_tokens(recent) > slots.R:
        old = recent.pop_oldest_pair()
        memo = fold_into_memo(memo, old)                    # lazy, incremental summarize (§8)
    if measure_tokens(memo) > slots.M:
        memo = compact_memo(memo)                           # hierarchical re-summarize (§8)

    # 4) PERSIST (cross-session durability)
    save_memo(folder, memo)
    # map already persisted; refresh only on staleness (§9.3)

    emit(answer)                                            # to the Result panel
```

### 6.3 The budget guard (the core safety mechanism)
A function invoked (a) before generation and (b) whenever the agent tries to *retain* a tool result. Trim order when over `B`:
1. Drop **lowest‑ranked** retrieved chunks from `Q`.
2. Compress the **oldest** `recent` turns into `M`.
3. If `M` over its slot, compact `M`.
4. If still over (pathological), refuse to retain more and emit a `session.notice` ("context is full — older detail summarized"). **Never** exceed `W`; **never** hang.

`measure_tokens` must use the **actual model tokenizer** for correctness (see §16‑4), with a cheap char‑based estimate as a pre‑filter.

---

## 7. Retrieval design **[PROPOSED]**

### 7.1 Primary: agent‑driven retrieval + governor
The original v1 path assumed no embeddings. Its agent-driven retrieval remains the mandatory fallback and is now optionally preceded by the bounded in-memory semantic assist described at the top of this document:
- Put the **folder map** (or a relevant slice of it) in front of the model as the retrieval menu.
- The model uses the existing `search` / `read_file` / `list_dir` tools to pull what it needs (this already works in the current loop).
- The **budget guard governs retention**: retrieved/tool content that would exceed `Q` is reduced (§7.3) or dropped.

### 7.2 Optional assist: lexical pre‑rank
To spend fewer agent steps hunting, optionally pre‑rank map entries by a **lexical score** (BM25 / tf‑idf over `path + signature` vs the user query + memo). Surface the top‑N as "likely relevant files" in the assembled context. Deterministic, cheap, no model. **[PROPOSED]**

### 7.3 Retrieve‑don't‑retain rule (critical)
When a `read_file` returns up to 64 KB, **do not keep the raw dump** in `recent`/history. Retain only the **excerpt actually used** — e.g., the lines around search hits, or a one‑paragraph extractive gist — with a **citation** (`path` + line range). This is the single biggest reduction in context growth. The raw file is always re‑readable on demand.

### 7.4 Chunking
A "chunk" is a bounded excerpt (~target 300–800 tokens) aligned to natural boundaries where possible (a function, a markdown section). Each chunk carries `{ path, line_start, line_end, text }` for citation and dedup. Dedup against chunks already present in `recent`/`memo`.

---

## 8. Summarization design **[PROPOSED]**

Summarization is the "hidden tax." Keep it cheap:
- **Lazy**: only when `recent` overflows its slot (not every turn).
- **Incremental**: fold **one** evicted (user, assistant) pair into `memo.summary` per compaction (small prompt).
- **Extractive for tool output** (no model call): keep matched/used lines, drop the rest.
- **Abstractive for conversation turns** (model call): only when folding into `memo`.
- **Pinned facts**: let the model emit explicit durable facts (e.g., a `remember:` convention or a lightweight tool) that go into `memo.facts` and are evicted only on `compact_memo`.

**Drift mitigation**: repeated re‑summarization loses fidelity. The `facts` list (verbatim, pinned) anchors the most important state; the abstractive `summary` carries the rest. Add a test that a fact once pinned survives N compactions (§14).

---

## 9. Persistence & resumability **[PROPOSED]**

### 9.1 What persists
- `memo` (cross‑session memory) — **required**.
- `map` (folder index) — **cached**, rebuildable.
- `recent` — optional (persist to resume mid‑conversation; otherwise rebuilt empty).

### 9.2 Where **[TO‑VERIFY / DECISION NEEDED]**
Two options, pick with the maintainer:
- **(A) Inside the workspace folder** (e.g., `.camelid/memory.json`). Pros: travels with the folder, obviously local. Cons: writes into the user's folder → must be approval‑gated or explicitly excluded from the "no writes without approval" contract; risk of polluting repos.
- **(B) App‑managed store** (e.g., under the Camelid config dir keyed by canonical folder path hash). Pros: never writes into the user's folder; cleaner. Cons: not portable with the folder.

**Recommendation: (B)** to preserve the "writes are approval‑gated" guarantee (memory writes are system writes, not agent writes, and should not appear as approvals nor mutate the user's tree). Document this clearly.

### 9.3 Staleness
Store folder `mtime`/size digest in `map`. On attach, if the folder changed materially, offer to refresh the map (cheap) — do not silently trust a stale index.

### 9.4 Privacy
Memory files contain conversation content and file excerpts. They are **local only**. Never included in evidence bundles, telemetry, or any network path. Add to the public‑scrub allowlist/ignore as appropriate.

---

## 10. Integration points (where code changes) **[PROPOSED, exact hooks TO‑VERIFY]**

1. **Session becomes multi‑turn** (biggest change). Today a session runs once and ends. Needed:
   - The `WorkspaceSessionManager` must keep a session **alive and idle** after a turn completes, holding `memo` + `recent` + `map` + the loaded model lock semantics, ready for the next user message.
   - **[TO‑VERIFY]** how the current session lifecycle (`WorkspaceSessionState`) ends and whether it can be extended to a `waiting_for_next_message` state without breaking the model‑transition exclusion logic.
2. **New endpoint** to send a follow‑up message to an existing session:
   - **[PROPOSED]** `POST /api/agent/workspace/sessions/:id/messages { text }` → runs one turn (§6.2), streams events on the existing SSE channel.
3. **Memory manager module** (new, `src/chat/workspace_memory.rs` or similar): owns `map`/`memo`/`recent`, budget math, budget guard, summarization calls, persistence. Pure‑ish and unit‑testable.
4. **Agent loop injection**: the assembled context (§6.2) must be turned into the `Vec<AgentMsg>` history that `LiveDriver::step` consumes. Memo + working set are injected as synthetic `System`/`User`/`ToolResult` messages. **[TO‑VERIFY]** the cleanest injection shape given `history_to_messages`.
5. **Token counting**: the manager needs `measure_tokens`. **[TO‑VERIFY]** whether to call `/api/models/tokenizer/encode` or hold an in‑process tokenizer handle (preferred for latency).
6. **Frontend**: `WorkspaceView.jsx` gains a **follow‑up composer** (a message box after the first answer) and renders a running transcript; `reduceWorkspaceEvent` gains handling for multi‑turn (multiple answers, per‑turn activity). The Result panel becomes a **conversation** of answers, with "What Camelid did" per turn.

---

## 11. API & data contract changes **[PROPOSED]**

- **Create session** (`POST /sessions`) — unchanged shape, but the session no longer auto‑terminates; it enters an idle/awaiting‑message state after the first turn. The first `goal` is treated as the first user message.
- **New**: `POST /sessions/:id/messages { text }` → 202/200; turn runs; events stream on the SSE channel.
- **New events** (extend the SSE vocabulary; keep names consistent with the reducer):
  - `turn.started { turn_index }`
  - `memory.updated { summary_tokens, facts_count, recent_tokens, budget_used, budget_total }` (drives a "context budget" UI meter)
  - reuse existing `model.delta`/`model.answer`/`tool.call`/`tool.result`/`approval.required`.
- **Status** (`GET /sessions/:id`) — add memory/budget fields for resume.
- **Frontend reducer** — support an array of turns; keep the Result‑first design per turn.

---

## 12. Budget math — worked example (ILLUSTRATIVE, calibrate before trusting) **[TO‑VERIFY]**

> These numbers are placeholders to show the *shape* of the math. Do **not** ship them without §16‑1 calibration.

Assume (illustrative) `W = 12,000` tokens resident for Qwen3‑4B‑Q4_K_M on the 8 GB box, `SAFETY = 0.7`:
```
B = 0.7 * 12000 = 8400
S (system+tools) measured = 600
remaining = 7800
  M (15%) = 1170
  R (25%) = 1950
  Q (45%) = 3510
  G (15%) = 1170   -> max_tokens ~ 1170
```
Implication: even with a modest resident window, ~3,500 tokens of *relevant* retrieved content + ~1,950 tokens of recent verbatim turns + a 1,170‑token memo gives a conversation that *feels* large, while the live prompt stays ≈ 8,400 tokens — inside the fast window. A 500‑file repo is fine because only retrieved chunks enter `Q`.

---

## 13. Failure modes & graceful degradation **[PROPOSED]**
| Failure | Handling |
|---|---|
| Assembled context > B | Budget guard trims Q → folds R → compacts M; emit `session.notice`. |
| Retrieval misses relevant file | Agent can still `search`/`read` on demand; lexical pre‑rank reduces misses; measure recall (§14). |
| Summarization latency spike | Lazy + incremental; show a subtle "updating memory…" state; never block input beyond the turn. |
| Model native ceiling reached | Clamp W to native ceiling; guard prevents exceeding it. |
| Long single tool result (64 KB) | Retrieve‑don't‑retain reduces before retention. |
| User cancels mid‑turn | Reuse existing cancellation/fail‑closed path. |
| Memory file corrupt/missing | Treat as empty memo; log; never crash the session. |

---

## 14. Testing & validation strategy **[PROPOSED]**
- **Unit (no model)**: budget partition math; budget‑guard trim order; `measure_tokens` accuracy vs tokenizer; map builder caps; memo compaction preserves pinned facts across N compactions; retrieve‑don't‑retain never stores > cap.
- **Integration (real model, exact Qwen3‑4B‑Q4_K_M)**: multi‑turn conversation keeps live prompt ≤ B (assert measured token counts); follow‑up correctly recalls an earlier fact; a write mid‑conversation still triggers approval with exact content.
- **Retrieval recall harness**: on a fixture folder with known answers, measure top‑k hit‑rate for lexical + agent‑driven retrieval.
- **Performance**: assert the live prompt stays under the resident window on the reference box (no CPU‑fallback) across a scripted N‑turn conversation; record tokens/sec per turn to prove no progressive slowdown beyond expectations.
- **Parity discipline**: memory is a prompt‑assembly layer; it must not alter decode determinism for a fixed assembled prompt. Any supported‑row parity claims are unaffected (document this).
- **Frontend**: reducer multi‑turn unit smoke; visual smoke for the composer + budget meter (fresh browser profile — see the first‑load lesson below).
- **Lesson to honor**: always test the frontend in a **fresh browser profile** (no localStorage). A prior crash (`DEFAULT_SETUP_PERCENT`/`MIN_SETUP_PX` undefined) only reproduced with empty localStorage.

---

## 15. Phased delivery plan **[PROPOSED]**
- **Phase 1 — Multi‑turn + budget guard (no summarizer, no persistence).** Keep session alive; add `POST /messages`; `recent` buffer + budget guard + retrieve‑don't‑retain; frontend composer + budget meter. Ships the core "chat with your folder" with in‑session memory bounded to the window.
- **Phase 2 — Rolling memo + cross‑session persistence.** Add `memo` (summary + pinned facts), lazy incremental summarization, persistence store (§9), resume.
- **Phase 3 — Folder map + lexical pre‑rank.** Build the map on attach; lexical pre‑rank to cut retrieval steps; staleness refresh.
- **Phase 4 — Engine ceiling-raisers (separate, higher-cost).** KV-cache quantization, prefix/KV reuse, and exact-row local embeddings are now implemented; semantic retrieval remains optional and separately evidence-gated. Broader retrieval quality, portability, and encoder support still require their own validation.

---

## 16. Open questions to resolve BEFORE coding (the must‑verify list)
1. **Resident window `W`**: how to obtain per (model, device)? Read `kv_cache.rs` for the resident cap logic; decide derived‑math vs. calibrated‑static table. Measure the real value on the reference box. *(Blocks §5, §12.)*
2. **KV bytes/token for Qwen3‑4B**: confirm `n_layers`, `n_kv_heads`, `head_dim`, KV dtype from the loaded model metadata.
3. **HardwareProfile fields**: confirm free‑VRAM and model‑size fields are actually exposed to a manager layer.
4. **Token counting**: confirm the cleanest `measure_tokens` path (in‑process tokenizer handle vs `/api/models/tokenizer/encode`) and its latency.
5. **Session lifecycle extension**: read `WorkspaceSessionState` + `run_live` + `WorkspaceSessionManager` fully; confirm a session can idle between turns without breaking the model‑transition exclusion and the one‑active‑session invariant.
6. **Agent history injection**: confirm how to inject memo/working‑set as `AgentMsg`s so `history_to_messages` renders them correctly for the Qwen native tool format.
7. **`run_loop` signature**: read it in full; confirm where the assembled context enters and where per‑turn tool transcripts come out.
8. **Persistence location decision** (§9.2 A vs B) with the maintainer; ensure memory writes do **not** appear as agent approvals and do **not** mutate the user's tree without consent.
9. **Reset/clear semantics**: how "Clear activity" / new‑thread interacts with persisted memo (clear memory vs. keep).

---

## 17. Summary
BWSM makes a **small, fast resident window behave like a large, persistent memory** by treating the **folder as the store**, the **model window as a budgeted working set**, and **memory as a rolling memo + folder map persisted to disk**, with a **budget guard** that keeps the live prompt inside the device's fast window. It needs **no engine changes** for Phases 1–3; KV‑quant and prefix‑cache are optional later ceiling‑raisers. The design deliberately rejects RoPE context extension on constrained GPUs. Every device‑specific number here is illustrative until calibrated (§16).
