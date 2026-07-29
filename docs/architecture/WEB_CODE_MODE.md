# Web Code Mode

**Status:** Implemented preview on `feat/web-code-mode`; not merged or production-promoted.

Code is an additive WebUI/Desktop surface that places the existing terminal agent loop behind a
Chat/Code switch. It does not replace Chat, the read-only Workspace, or the terminal agent.

## User experience

- **Chat** retains the ordinary prompt conversation.
- **Code** selects one local workspace and streams readable model, plan, tool, approval, and change
  activity for the coding turn.
- Approval-gated is the default. A separately confirmed full-auto choice applies to one session;
  network/web search is controlled by an independent switch.
- Switching back to Chat hides rather than unmounts Code, so the run continues until it finishes or
  the user presses Stop.
- The left rail shows durable coding threads independently of chat history.
- A completed session exposes its checkpoint summary and diff, plus guarded single-step undo.
- Saved coding turns can be reopened and resumed when the same exact model artifact is active.

The Windows Desktop app receives this surface automatically because its Tauri WebView hosts the
same loopback WebUI served by the engine.

## Security boundary

The server, not the browser, owns the boundary:

1. Requests require the existing loopback and exact same-origin authorization.
2. The workspace is canonicalized and every file-tool target stays confined to that root.
3. Admission requires a loaded, generation-ready, supported exact model row with
   `tool_capable: true`.
4. `ToolProfile::WebCode` advertises and accepts:
   `read_file`, `list_dir`, literal-content `search`, `update_plan`, `write_file`, `edit_file`, and
   sandboxed `run_shell`; bounded `spawn_subagent`/`check_subagent_status` are added only while the
   Code subagent runtime is active.
5. `allow_network: true` adds only the built-in `web_search` and `http_fetch` tools. The sandbox
   rejects both when the switch is off. This is not an OS egress firewall for `run_shell`.
6. `approval_mode: "approval_gated"` sends exact write/Exec/network decisions through the approval
   bridge. The separately confirmed `"full_auto"` policy promotes them for that Code session only
   and is refused when `CAMELID_PRODUCTION` is set. File tools remain root-confined.
7. Child agents inherit the WebCode allowlist, network switch, shell sandbox, and approval posture.
   They cannot spawn grandchildren and are killed when the parent turn ends or is stopped.
8. GUI, Windows computer-control, and MCP tools are never advertised or accepted by WebCode.
   General shell execution is nevertheless general process execution: on Windows the existing
   `ShellSandbox::Sandboxed` contract is cwd-pin + hard timeout, not filesystem or network
   isolation. The full-auto confirmation states this explicitly.
9. Starting another workspace session is refused while a turn is active.
10. The existing read-only mode remains the default. A legacy `allow_writes: true` request without
   `mode: "code"` still fails with `400 workspace_read_only`.

Code uses cancellation and a result-aware repetition guard instead of an arbitrary model/tool step
count. Its local-model stream has no wall-clock model-step deadline; the visible Stop control remains
authoritative.

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
