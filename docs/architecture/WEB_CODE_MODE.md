# Web Code Mode

**Status:** Implemented preview on `feat/web-code-mode`; not merged or production-promoted.

Code is an additive WebUI/Desktop surface that places the existing terminal agent loop behind a
Chat/Code switch. It does not replace Chat, the read-only Workspace, or the terminal agent.

## User experience

- **Chat** retains the ordinary prompt conversation.
- **Code** selects one local workspace, runs an approval-gated coding turn, and streams model and
  tool activity.
- The left rail shows durable coding threads independently of chat history.
- A completed session exposes its checkpoint summary and diff, plus guarded single-step undo.
- Saved coding turns can be reopened and resumed when the same exact model artifact is active.

The Windows Desktop app receives this surface automatically because its Tauri WebView hosts the
same loopback WebUI served by the engine.

## Security boundary

The server, not the browser, owns the boundary:

1. Requests require the existing loopback and exact same-origin authorization.
2. The workspace is canonicalized and every file or shell target stays confined to that root.
3. Admission requires a loaded, generation-ready, supported exact model row with
   `tool_capable: true`.
4. `ToolProfile::WebCode` advertises and accepts only:
   `read_file`, `list_dir`, literal-content `search`, `update_plan`, `write_file`, `edit_file`, and
   sandboxed `run_shell`.
5. File mutations and shell commands use the existing approval bridge. Network, GUI, MCP,
   subagent, Windows computer-control, and unattended tools are absent.
6. Starting another workspace session is refused while a turn is active.
7. The existing read-only mode remains the default. A legacy `allow_writes: true` request without
   `mode: "code"` still fails with `400 workspace_read_only`.

## API additions

All endpoints remain under `/api/agent/workspace`:

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/sessions` with `mode: "code"` | Start or resume a Code session |
| `POST` | `/sessions/:id/decisions` | Resolve an exact pending approval |
| `GET` | `/sessions/:id/changes` | Read checkpoint summary, files, and diff |
| `POST` | `/sessions/:id/undo` | Undo the latest checkpoint after the turn stops |
| `GET` | `/threads?workspace=…&mode=code` | List Code threads for one root |
| `GET` | `/threads/recent?mode=code` | Populate the global Code-history rail |

Code thread IDs use the `code-` prefix. Read-only Workspace IDs continue to use `workspace-`;
resuming a thread through the wrong mode fails closed.

## State and persistence

Transcripts use the existing SQLite/FTS5 Workspace memory store. A terminal outcome is written
durably only when the turn completes, aborts, or fails under the existing lifecycle rules.
Checkpoint snapshots remain workspace-local under `.camelid/checkpoints`; Code never invokes git.

The checkpoint log is process-local and corresponds to the one active workspace session. Starting
a new Code session clears that log. Historical transcripts remain available, while historical file
diffs are intentionally not reconstructed after a new session or process restart.

## Validation boundary

Local Windows validation covers the allowlist and authorization tests, approval/cancellation bridge
tests, a real Qwen3-4B-Q4_K_M read/write turn, exact-action approval, created-file verification,
diff, durable history restore, guarded undo, desktop and 390 px responsive WebUI checks, the full
core Rust suite, Desktop tests/build/Clippy, frontend build/smokes, and dependency audit.

This preview does not promote any model, quantization, backend, context window, operating system,
latency, throughput, or broader product-support claim.
