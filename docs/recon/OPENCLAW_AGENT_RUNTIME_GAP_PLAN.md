# Agent Runtime Recovery Plan

Status: Phase 1 implemented on `codex/web-code-revival`
Reference: OpenClaw commit [`630aac9`](https://github.com/openclaw/openclaw/tree/630aac9b25ae6f42c760226662d4a7b3d1545f82)

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
optional, normalizes it before admission, and returns a runtime session key and
run id. See
[`subagent-task-name.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/subagents/spawn/subagent-task-name.ts),
[`subagent-spawn-contract.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/subagents/spawn/subagent-spawn-contract.ts),
and
[`sessions-spawn-tool.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/tools/sessions-spawn-tool.ts).

Its loop protection also operates before tool execution. Calls are canonicalized,
hashed by tool plus stable arguments, admitted atomically, and checked for repeat,
argument churn, unknown-tool repetition, polling without progress, ping-pong, and
a global circuit breaker. See
[`tool-loop-recovery.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/embedded-agent-runner/run/tool-loop-recovery.ts),
[`tool-loop-admission.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/tool-loop-admission.ts),
and
[`tool-loop-detection.ts`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/src/agents/tool-loop-detection.ts).

Finally, child completion is an event delivered back to the parent instead of a
model-managed polling ritual. Child liveness, cascade stop, stale-run pruning,
depth policy, concurrency, and delivery backpressure belong to the runtime. See
[`docs/tools/subagents.md`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/docs/tools/subagents.md)
and
[`docs/agent-runtime-architecture.md`](https://github.com/openclaw/openclaw/blob/630aac9b25ae6f42c760226662d4a7b3d1545f82/docs/agent-runtime-architecture.md).

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
3. **Event-driven child completion.** Add a run-scoped completion queue. A parent
   that has live children suspends without another model inference and resumes
   when a completion, timeout, cancellation, or failure event arrives. Keep
   `check_subagent_status` for user inspection and recovery, not normal control
   flow.
4. **Pre-execution batch admission.** Canonicalize every proposed call before
   signatures are computed. Evaluate a whole tool-call batch before launching any
   action. Record only admitted actions as launched; record denied actions as
   denial evidence. Add detectors for argument churn, unknown-tool repetition,
   ping-pong, and a global call ceiling.
5. **Structured outcomes and run-level retry state.** Replace control decisions
   based on display text with typed statuses such as `accepted`, `running`,
   `completed`, `failed`, `inconclusive`, `validation_error`, and `timed_out`.
   Retry counters live for the whole run so compaction or a driver retry cannot
   reset them.
6. **Lifecycle ownership and backpressure.** Give every run a runtime id, parent
   id, depth, captured policy, deadline, and cancellation token. Cascade Stop to
   descendants. Reap stale task files. Refuse new delegation when the completion
   delivery queue is saturated.
7. **Real-model release gate.** Before shipping agent changes, run a fixed Qwen
   Web Code matrix: one-file Python GUI creation, missing-runtime recovery,
   underscore task alias, omitted task alias, duplicate spawn, invalid argument
   correction, child timeout, Stop cascade, and successful direct edit. Assert a
   bounded number of model turns and tool calls for every case.

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
