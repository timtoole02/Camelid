# Context Paging Runtime implementation state

## Objective

Build an opt-in, bounded-context Web Code runtime that constructs one fresh
context capsule per action while keeping canonical task/project/tool state
outside the model.

## Decisions

- Reuse Camelid's exact generation preflight as the final tokenizer budget gate.
- Persist runtime state under host-protected `.camelid/context-paging`.
- Start with deterministic Rust/Python structural extraction; no embeddings.
- Treat exact file hashes and source pages as authority.
- Preserve all existing sandbox, approval, checkpoint, and audit boundaries.
- Use JSON typed actions; first executable actions are `NEED_CONTEXT` and
  hash-checked full-page `PATCH`.
- Keep rollout behind `CAMELID_CONTEXT_PAGING=1`.

## Completed work

- Mapped inference, prompt, history, tool, source, output, and persistence flow.
- Added persistent canonical ledgers, runtime metrics, raw artifacts, project
  maps, symbol cards, exact pages, content-hash invalidation, and restart load.
- Added deterministic budgeted capsules whose accounting includes native tool
  schemas, plus exact tokenizer enforcement at the live inference boundary.
- Added typed actions, repeated page-fault pinning, hash-checked PATCH-to-edit
  translation, phase tool enforcement, and compact diagnostic inspection.
- Integrated a fresh-capsule-per-action Web Code loop behind
  `CAMELID_CONTEXT_PAGING=1`; the old loop is unchanged when disabled.
- Added the deterministic benchmark report and an end-to-end three-request
  test proving an oversized transcript is not replayed.
- Preserved narrow Qwen malformed-`write_file` parser regressions developed
  before this larger brief arrived.
- Restored completed work and persisted verified-source paths when a task is
  reopened, so generic completion gates cannot send a complete ledger back to
  Modify or demand redundant reads.
- Passed a real Qwen3 4B restart gate in one tool-free typed `COMPLETE` action:
  438 exact prompt tokens, `answered` terminal outcome, completed frontend
  activity, revision-25 complete ledger, and a fresh Python compile check.
- Passed the final capped full library regression: 1,874 passed, 0 failed, and
  86 ignored. The run used two test threads, two-core affinity, below-normal
  priority, and Camelid's shared Cargo target.

## Current focus

Implementation and validation are complete; keep rollout opt-in while the live
evidence is reviewed.

## Remaining work

- Review the opt-in rollout evidence and decide when
  `CAMELID_CONTEXT_PAGING=1` should become the default Web Code path.

## Failed approaches

- Retrying malformed Qwen tool envelopes without structural recovery repeated
  expensive inference and ended as no progress.
- Transcript-only pruning bounds size but does not provide canonical restartable
  task state, hash invalidation, or exact-source authority.
