# Context Paging Runtime implementation state

## Objective

Build a default-on, bounded-context Web Code runtime that constructs one fresh
context capsule per action while keeping canonical task/project/tool state
outside the model.

## Decisions

- Reuse Camelid's exact generation preflight as the final tokenizer budget gate.
- Persist runtime state under host-protected `.camelid/context-paging`.
- Start with deterministic Rust/Python structural extraction; no embeddings.
- Treat exact file hashes and source pages as authority.
- Preserve all existing sandbox, approval, checkpoint, and audit boundaries.
- Advertise the same native file/shell tools the local model already uses;
  successful `read_file` calls load exact hash-authorized source pages.
- Retain the earlier JSON typed actions as backward-compatible input, not as a
  second protocol the stable kernel asks a small model to learn.
- Make Context Paging the Web Code default, with
  `CAMELID_CONTEXT_PAGING=0` retained as the explicit rollback switch.

## Completed work

- Mapped inference, prompt, history, tool, source, output, and persistence flow.
- Added persistent canonical ledgers, runtime metrics, raw artifacts, project
  maps, symbol cards, exact pages, content-hash invalidation, and restart load.
- Added deterministic budgeted capsules whose accounting includes native tool
  schemas, plus exact tokenizer enforcement at the live inference boundary.
- Added typed actions, repeated page-fault pinning, hash-checked PATCH-to-edit
  translation, phase tool enforcement, and compact diagnostic inspection.
- Integrated a fresh-capsule-per-action Web Code loop. It is now the default;
  the old loop remains available only through `CAMELID_CONTEXT_PAGING=0`.
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
- A default-on 8K greenfield run then exposed a native-tool recovery defect:
  structured calls correctly carried empty assistant text, but an invalid call's
  error/reminder was stored only in legacy transcript history. Fresh paging
  capsules therefore repeated the same 21-token call and hit the two-strike
  guard. Trailing retry feedback now enters the next mandatory capsule with the
  exact validation error, is UTF-8 safely bounded, and is consumed by the next
  tool result.
- Simplified the stable kernel to one native tool protocol, repaired
  `list_dir {}` to the deterministic workspace root, made empty creation start
  with write tools, kept the active Modify/Verify tool schema stable, and made
  native reads load exact source pages. A reread alone no longer marks Code
  verified when a shell verification tool is available.
- Added a conservative host-owned manifest for explicit workspace artifacts in
  the exact objective. Missing artifacts keep write tools available after an
  earlier file passes; successful environment probes do not count as
  verification, and the shell-disabled host verification path remains usable.
- Preserved the user's complete immutable objective verbatim. If the exact task
  contract cannot fit the mandatory input budget, capsule construction fails
  closed instead of silently hiding requirements after byte 600.
- Fixed the legacy rollback lane's reminder boundary and added a fixed 5.5K
  compiled-prompt high-water mark (4K low-water), so widening a nominal window
  to 16K cannot defer compaction past the measured cold-prefill cliff.

## Current focus

Run the full regression/scrub gates and capture a fresh live default-on receipt
with the native-tool compatibility path.

## Remaining work

- Capture a live long-turn comparison on the default path, including peak
  capsule size, time to first token, tool progress, and completion outcome.

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
