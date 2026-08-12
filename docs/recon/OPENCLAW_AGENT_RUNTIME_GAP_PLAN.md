# Agent Runtime Recovery Plan

Status: Runtime recovery implemented on `codex/web-code-revival`; transport/tool gate passes, behavioral real-model release gate remains on hold
Reference: OpenClaw commit [`b82ad646`](https://github.com/openclaw/openclaw/tree/b82ad646a5151e5fb6378e72dbbe257fd5012813)

## Failure that triggered this work

For a one-file Python tic-tac-toe request, the model called `spawn_subagent` with
`generate_tic_tac_toe_code`. Camelid required the model to invent an internal
filename-safe id matching `^[a-z0-9-]{1,64}$`, rejected the underscore before
execution, and allowed the identical invalid call to run through inference twice.
The user then had to stop the turn.

The model made a recoverable formatting choice. The runtime turned it into a
terminal product failure.

## What OpenClaw gets right

OpenClaw separates the model-facing task name from the runtime-owned child
identity. Its spawn contract requires the task, makes the readable task name
optional, and returns a runtime session key and run id. See
[`subagent-spawn-contract.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/subagents/spawn/subagent-spawn-contract.ts)
and
[`sessions-spawn-tool.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/tools/sessions-spawn-tool.ts).

Its loop protection also operates before tool execution. Calls are canonicalized,
hashed by tool plus stable arguments, admitted atomically, and checked for repeat,
argument churn, unknown-tool repetition, polling without progress, ping-pong, and
a global circuit breaker. See
[`tool-loop-argument-churn.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/tool-loop-argument-churn.ts)
and
[`tool-loop-call-reconciliation.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/tool-loop-call-reconciliation.ts).

Finally, child completion is captured authoritatively and delivered back to the
parent instead of becoming a model-managed polling ritual. The registry owns
execution state, capture state, delivery state, generations, deadlines,
idempotency, recovery, and cancellation. The wait bridge parks until the
registry wakes it and then re-checks to close the subscribe race. See
[`subagent-registry.types.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/subagents/registry/subagent-registry.types.ts),
[`subagent-registry-completion.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/subagents/registry/subagent-registry-completion.ts),
[`subagent-completion-delivery.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/subagents/completion/subagent-completion-delivery.ts),
[`subagent-announce.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/subagents/announce/subagent-announce.ts),
and
[`agents-wait-tool.ts`](https://github.com/openclaw/openclaw/blob/b82ad646a5151e5fb6378e72dbbe257fd5012813/src/agents/tools/agents-wait-tool.ts).

## Camelid target architecture

The model proposes intent. The runtime owns identity, admission, lifecycle,
retry state, and completion delivery.

1. **Canonical tool boundary — implemented.** `subtask_id` is optional.
   Readable aliases accept case, spaces, underscores, and hyphens, then normalize
   into Camelid's strict storage id. Unsafe punctuation and traversal still fail
   closed. Omitted ids are deterministic per goal, and a repeated admitted spawn
   returns status instead of creating duplicate work or another tool error.
2. **Fast validation recovery — implemented.** The first invalid call gets an
   explicit correction message. Repeating the identical invalid call stops after
   two attempts, not three minute-long inference rounds. Small single-file coding
   work is directed to `write_file`/`edit_file` instead of unnecessary delegation.
3. **Runtime-owned child completion — implemented.** `await_subagent` parks one
   cancellable tool execution until the authoritative result becomes terminal or
   its bounded wait expires. It performs no additional model inference.
   `check_subagent_status` remains a non-blocking inspection/recovery tool and its
   schema explicitly tells the model not to poll it.
4. **Run-level circuit breakers — implemented.** Every Workspace turn has an
   absolute 64-call admission ceiling in addition to the eight-call per-step
   ceiling. A narrow tail-churn detector stops the same tool after four varied
   argument signatures produce the same error; exact invalid repeats still stop
   after two. Unknown-tool variants share one churn lane.
5. **Structured outcomes and run-level retry state.** Replace control decisions
   based on display text with typed statuses such as `accepted`, `running`,
   `completed`, `failed`, `inconclusive`, `validation_error`, and `timed_out`.
   Retry counters live for the whole run so compaction or a driver retry cannot
   reset them.
6. **Per-turn lifecycle ownership — implemented.** Camelid's former process-wide
   registry let one Web Code turn reconfigure or cancel another turn's children.
   Registry state is now isolated to the turn worker thread. A readable alias is
   only the turn-local idempotency key; every admitted child receives a unique
   runtime/storage id, so stale or concurrent result files cannot be mistaken for
   a new request. Stop remains authoritative for that turn's whole child tree.
7. **Real-model release gate — required before merge.** Run a fixed Qwen
   Web Code matrix: one-file Python GUI creation, missing-runtime recovery,
   underscore task alias, omitted task alias, duplicate spawn, invalid argument
   correction, child timeout, Stop cascade, and successful direct edit. Assert a
   bounded number of model turns and tool calls for every case.

## Additional recovery work implemented

- Obvious standalone creation stays on the parent model. It does not advertise
  subagents, planning, or patch editing; every draft/correction is a complete
  `write_file` replacement. Repository work retains orchestration.
- Direct creation owns a deterministic fallback artifact name (`tic_tac_toe.py`
  for the triggering request, otherwise `app.py` for a small Python GUI). It is
  supplied only when a recognizable write contains real content but omits only
  `path`; explicit model paths are preserved and every call still passes normal
  schema, sandbox, approval, checkpoint, and audit validation.
- Llama/Qwen malformed-call recovery handles invalid JSON escapes, semicolon-
  separated native calls, unescaped source quotes, a malformed write followed by
  valid calls, and the observed one-brace-short write envelope. Malformed shell
  and network envelopes remain inert.
- Mutation results report workspace-relative paths. Windows extended paths no
  longer leak into model context and come back double-escaped in repair calls.
- Windows Python recovery probes `py --version`, treats success as authoritative,
  blocks `pip install tkinter`, and turns both `python game.py` and `py game.py`
  verification attempts into bounded `py -m py_compile game.py` checks. A real
  traceback or syntax error locks a direct task to a complete source rewrite.
- The host captures exact changed paths itself before accepting completion,
  retains the observations, runs Python syntax validation, and applies a narrow
  source contract for the explicit tic-tac-toe request: automatic legal O move,
  return to X inside the computer-move block, terminal states/draw, GUI result,
  and correctly bound Tkinter loop callbacks.
- No-progress guards cover exact invalid repeats, same-result repeats, varied-
  argument error churn, malformed tool syntax, repeated completion without a
  mutation, and completion without exact post-change verification.

## 2026-08-12 live-model evidence

All runs used the exact user prompt, an empty temporary workspace, Code mode,
full-auto approval, network disabled, and the real authenticated Workspace API.
The CUDA-resident server was stopped between builds and at handoff.

- `Llama-3.2-3B-Instruct-Q8_0.gguf` proved the Llama structured boundary:
  malformed writes were recovered, a deterministic filename was supplied, files
  were checkpointed, the Windows Store alias recovered to `py`, relative-path
  rewrites succeeded, and syntax failures were caught. It did not converge on a
  behaviorally correct game; successive rewrites regressed computer turns and
  emitted invalid Python. This row is useful for fast boundary smoke, not the
  recommended coding row.
- `Qwen3-4B-Q4_K_M.gguf` proved the Qwen route through real writes, exact host
  capture, syntax validation, and semantic rejection. Its first drafts still had
  material defects (late-bound Tkinter callbacks, incomplete O terminal handling,
  bad initialization/minimax behavior), then ignored the required full rewrite
  and was stopped by the no-progress guard. It is stronger than the 3B row but
  has not passed this behavioral release gate.
- `Qwen3-4B-Q8_0.gguf` is tool-certified but not operational for this workflow on
  the 6 GiB RTX 3060 laptop: context residency leaves too little headroom and the
  correction turn becomes impractically slow.
- `Qwen3-8B-Q4_K_M.gguf` was not used for Code because its exact compatibility
  row is not tool-capable. An attempted promotion evaluation was inconclusive;
  support was not widened and no receipt was fabricated.

Release verdict: the runtime bugs that caused the original `spawn_subagent`
failure and later silent write loss are fixed and regression-tested. The exact
one-file acceptance prompt still lacks a complete behavioral pass from an
installed model, so this branch must not claim that model-quality gate as passed.
The next product step is to certify a stronger coding/tool row that fits the host,
then rerun the fixed matrix above; do not weaken the host audit to make a smaller
model appear successful.

## Non-negotiable invariants

- A harmless model formatting choice cannot make a tool unusable.
- Internal ids and filesystem paths are runtime concerns, never prompt trivia.
- The same admitted request cannot launch duplicate children.
- A tool failure must say whether execution happened and what can be corrected.
- Waiting is runtime state, not repeated model inference.
- Stop ends the entire run tree.
- No Code turn may claim completion for an actionable coding request without a
  successful workspace mutation.
- Validation and builds must be disk-bounded: one Cargo job, incremental off,
  shared dependency cache, and free-space checks before and after.
