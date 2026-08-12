# Web Code Mode

**Status:** Revived on `codex/web-code-revival`; ready for review, not production-promoted.

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
   Code subagent runtime is active. `run_shell` is offered only where the sandbox can actually be
   enforced — Linux x86_64/aarch64, macOS, and Windows. On any other host the session resolves the
   shell to `disabled`, so the tool is never advertised, and says so in the transcript; a Code
   session there is a read/write coding surface without command execution.
5. `allow_network: true` adds only the built-in `web_search` and `http_fetch` tools. The sandbox
   rejects both when the switch is off. This is not an OS egress firewall for `run_shell`.
6. `approval_mode: "approval_gated"` sends exact write/Exec/network decisions through the approval
   bridge. The separately confirmed `"full_auto"` policy promotes them for that Code session only
   and is refused when `CAMELID_PRODUCTION` is set. File tools remain root-confined.
7. Child agents inherit the WebCode allowlist, network switch, shell sandbox, and approval posture.
   They cannot spawn grandchildren and are killed when the parent turn ends or is stopped.
8. GUI, Windows computer-control, and MCP tools are never advertised or accepted by WebCode.
   General shell execution is nevertheless general process execution, and what confinement means
   depends on the host:
   - **Linux**: seccomp (blocking the `socket` family, so no egress) + uid-drop + rlimits + cwd-pin.
   - **macOS**: kernel Sandbox (Seatbelt) via `sandbox-exec` — writes confined to the workspace plus
     the process temp directory, network denied, credential stores denied for read and write. Reads
     elsewhere are still permitted; the workspace read jail belongs to the file tools, not the
     profile. The temp-directory allowance is required for compilers and is the jail's one widening.
   - **Windows**: the existing `ShellSandbox::Sandboxed` contract is cwd-pin + hard timeout, not
     filesystem or network isolation.
   - Elsewhere the shell is not offered at all (item 4).

   The full-auto confirmation states this explicitly. Timeout teardown reaches the whole process tree
   on Windows (job object) and for delegated work on Unix (the worker's process group); a command the
   server itself runs on Unix is killed as a single process, so a descendant build tree can outlive
   its deadline.
9. Starting another workspace session is refused while a turn is active.
10. The existing read-only mode remains the default. A legacy `allow_writes: true` request without
   `mode: "code"` still fails with `400 workspace_read_only`.

Code uses cancellation and a result-aware repetition guard instead of an arbitrary model/tool step
count. Its local-model stream has no wall-clock model-step deadline; the visible Stop control remains
authoritative. This applies to Code only — the read-only Workspace lane keeps its documented
90-second model-step deadline. The repetition guard never counts waiting on a running subagent:
polling one is how a turn waits, and treating it as a stall would end the turn and kill the child
with it.

The Code driver consumes both streaming `delta.content` and indexed OpenAI-compatible
`delta.tool_calls`. Camelid's dense chat stream intentionally buffers a possible tool envelope until
it can classify it, then emits a structured tool-call delta with empty content. The client therefore
accumulates name/argument fragments by call index and executes those calls before falling back to
family-specific text parsing. This keeps the browser's live stream behavior identical to the
non-streaming agent-eval path that earned the model's `tool_capable` receipt.

Code also enforces an artifact-completion contract for actionable coding requests: it will not emit
an `answered`/Complete outcome until `write_file` or `edit_file` has successfully created a
checkpointed workspace change. Natural-language surrender after a failed tool call is re-prompted;
repeated surrender ends as `repeated`/No progress rather than a false success. `run_shell` rejects
multi-line source pasted as a command and directs the model to write the source first. Runtime
recovery guidance requires a host probe before proposing installation (including the Windows `py`
launcher before the Microsoft Store `python.exe` alias), and any real package-manager install still
crosses the existing Exec approval boundary.

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

The checkpoint log corresponds to the one active workspace session. Starting a new Code session
clears it. Historical transcripts remain available, while historical file diffs are intentionally
not reconstructed after a new session or process restart.

Committed checkpoints are journaled next to the backups (`.camelid/checkpoints/journal.jsonl`),
because a subagent is a separate process writing into the same workspace: the change set and the
undo stack are read back from that journal, so a delegated write is as visible and as revertible as
one the server made itself, and it unwinds in the order it was actually committed.

## Validation boundary

Local Windows validation covers the allowlist and authorization tests, approval/cancellation bridge
tests, a real Qwen3-4B-Q4_K_M read/write turn, exact-action approval, created-file verification,
diff, durable history restore, guarded undo, raw-source shell rejection, failed-tool coding recovery,
false-completion prevention, desktop and 390 px responsive WebUI checks, the full core Rust suite,
Desktop tests/build/Clippy, frontend build/smokes, and dependency audit.

This preview does not promote any model, quantization, backend, context window, operating system,
latency, throughput, or broader product-support claim.
