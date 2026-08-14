# Context Paging Runtime implementation state

## Objective

Build a bounded-context Web Code runtime that constructs one fresh
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
- Keep a fail-safe rollback behind `CAMELID_CONTEXT_PAGING=0`.

## Completed work

- Mapped inference, prompt, history, tool, source, output, and persistence flow.
- Added persistent canonical ledgers, runtime metrics, raw artifacts, project
  maps, symbol cards, exact pages, content-hash invalidation, and restart load.
- Added deterministic budgeted capsules whose accounting includes native tool
  schemas, plus exact tokenizer enforcement at the live inference boundary.
- Added typed actions, repeated page-fault pinning, hash-checked PATCH-to-edit
  translation, phase tool enforcement, and compact diagnostic inspection.
- Integrated a fresh-capsule-per-action Web Code loop as the default; the old
  loop is unchanged when explicitly disabled with `CAMELID_CONTEXT_PAGING=0`.
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
- Completed a 14-finding adversarial audit and fixed every confirmed defect:
  startup on repositories with >1 MB files, symbol-ID collisions, host-owned
  completion gating, typed-action loop bounds, tail-inclusive raw capture,
  search-result visibility, the eviction ladder, ledger bounding, tokenizer
  calibration rebuilds, deletion invalidation, parent symbols, and atomic
  persistence. Added regression tests for each fix (16 new unit tests).
- Live Qwen3-4B smoke runs on a real playground exposed and fixed three more
  greedy-small-model defects: structural recovery for typed-action JSON whose
  string values carry unescaped quotes (a Python docstring's `"""` ended the
  strict string mid-value and the model repeated the same bytes forever), a
  body-fragment guard plus full-file write_file escalation for PATCHes that
  would replace a whole page with a headless method, trailing-newline
  normalization on page replacement, and duplicate-NEED_CONTEXT steering (a
  re-request of a page already in the capsule now changes the canonical focus
  so the next capsule breaks the greedy fixed point).

## Current focus

Final gate run (fmt, clippy, full tests, scrub), then merge origin/main and
push.

## Remaining work

- Merge codex/web-code-revival with origin/main and push.
- Continue expanding live-model paging evidence while retaining the explicit
  rollback switch and the Qwen 4B-safe default 8K active working envelope.

## Failed approaches

- Retrying malformed Qwen tool envelopes without structural recovery repeated
  expensive inference and ended as no progress.
- Transcript-only pruning bounds size but does not provide canonical restartable
  task state, hash invalidation, or exact-source authority.
- Pinning a repeatedly faulted page without changing the rendered focus does
  not break a greedy model's fault loop: the capsule bytes stay identical, so
  the model re-emits the identical fault. Steering must alter canonical state.
- A process-global extended-capture flag leaked across concurrent sessions and
  the test process; the capture mode is thread-scoped and set per run instead.
