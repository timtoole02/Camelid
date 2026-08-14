# Context Paging Runtime

## Implementation map

Camelid's current Web Code request flow is:

1. `api::workspace::create_session` creates a durable Workspace thread and a
   `WorkspaceRunConfig`.
2. `chat::workspace_bridge::run_live` builds the system prompt, restores bounded
   thread memory, and creates `LiveDriver`.
3. `chat::agent::run_loop` calls `compile_history_for_step`, then
   `fit_history_to_budget`, then `LiveDriver::step`.
4. `LiveDriver::request` converts `AgentMsg` history and all currently offered
   `ToolSpec`s to the OpenAI-style chat request. The loaded model tokenizer is
   used by generation preflight to report the exact prompt token count.
5. Model calls pass through `tools::validate_for`, the approval policy, audited
   execution, checkpoints, and bounded `ToolOutcome` observations.
6. `WorkspaceMemoryStore` persists thread turns and selected evidence in
   SQLite. `agent_session` separately persists CLI transcript resumes.

Relevant integration points:

- `src/chat/context_paging.rs`: canonical ledger, structural index, hash-backed
  pages, compact artifact store, typed actions, and deterministic capsule
  builder.
- `src/chat/workspace_bridge.rs`: enables the feature-gated runtime for Web
  Code.
- `src/chat/agent.rs`: constructs a new capsule and phase-filtered tool set for
  every action. The existing tokenizer preflight remains the final hard request
  gate; tool validation and approval boundaries remain authoritative.
- `src/chat/tools.rs`: bounded file/search primitives and audited writes remain
  the execution layer. Raw paging artifacts are never executable authority.
- `src/chat/workspace_memory.rs`: remains the user-visible thread store. Context
  Paging state is independent so a fresh inference session can resume from the
  ledger without replaying chat.

Compatibility constraints:

- Context paging is the default Web Code memory path. Set
  `CAMELID_CONTEXT_PAGING=0` only as a deliberate rollback; the legacy loop is
  unchanged while disabled.
- `.camelid` is already protected from model-authored writes. Runtime state is
  stored below `.camelid/context-paging` through host code only.
- Exact source is authoritative. A card or page whose file hash no longer
  matches is excluded and must be rebuilt.
- Typed patches are translated to ordinary `edit_file` calls and pass the
  workspace sandbox, approval, checkpoint, and audit layers. Native edits are
  also rejected unless their old source occurs in an exact page in the current
  capsule; overwrites of existing files require a full exact file page.
- No embeddings, vector database, multi-agent dependency, or growing KV cache
  is required by the first slice.

## Vertical-slice architecture

The first slice supports Rust and Python symbol extraction with deterministic
source-derived signatures and balanced source ranges. It persists:

- a canonical `TaskLedger`;
- a `ProjectMap`, `SymbolCard`s, and exact `SourcePage`s;
- compact tool-result records plus content-addressed raw artifacts;
- page-fault counts and temporary pins;
- runtime metrics (including the composition of the most recent capsule as
  `lastCapsuleComposition`) and capsule selection/exclusion explanations.

Symbol extraction is deterministic and collision-free: same-named declarations
in one file receive `#N` ordinal suffixes on their IDs (the first occurrence
keeps the unsuffixed ID), and `parent_symbol` is populated from the innermost
containing declaration. `impl Trait for Type` blocks are named after the
implemented type, and generic `impl<T>` headers are recognized. Brace counting
ignores braces inside string/char literals and `//` comments; multi-line raw
strings remain a documented heuristic limit.

`ContextCapsuleBuilder` emits a fresh capsule for one action. The stable kernel
is a byte-stable prefix; task-specific state follows it. Items are ordered by
category and stable identifiers. Eviction strictly follows the spec priority
ladder as a prefix take — first removed to last removed: failed-attempt
history, low-relevance cards, dependency pages, completed-work detail,
repository map, task detail — never a greedy fill that would keep small
low-priority history while dropping higher-priority evidence. Never-evict
content is the stable kernel, the bounded task contract, the current
diagnostic, the phase tool list, the modification target page (the most
recently faulted symbol), and every pinned page. Pinned pages are capped at 4;
the least-faulted pin is released first. The task contract is split:
`objective`, `currentAction`, `currentFocus`, `acceptanceCriteria`,
`criticalInvariants`, and `verificationStatus` stay mandatory and bounded (at
most 6 items of 240 chars per list, 600 chars per field), while `decisions`
and `openQuestions` render as evictable task detail. Ledger lists themselves
are bounded (32 items, 480 chars per item), so ledger growth can never brick
capsule construction. The builder uses a `TokenEstimator` interface that is
continuously calibrated from the live tokenizer's measured tokens-per-byte
rate; the integrated request is still checked with Camelid's exact
loaded-model tokenizer preflight before inference.

The typed action protocol is JSON with one of: `NEED_CONTEXT`, `SEARCH`,
`PATCH`, `RUN_TEST`, `INSPECT_DIAGNOSTIC`, `UPDATE_PLAN`, `COMPLETE`, or
`BLOCKED`. The stable kernel includes the one-line JSON shape of all eight
actions and remains byte-stable. The live slice executes `NEED_CONTEXT` and
translates `PATCH`, `SEARCH`, and `RUN_TEST` through the existing tool
boundary. Diagnostic inspection and plan updates remain host-owned state
actions. `INSPECT_DIAGNOSTIC` accepts an optional `startLine` to page bounded
slices of a stored raw artifact by reference; `UPDATE_PLAN` and `BLOCKED`
validate non-empty fields. The parser also accepts an action inside a plain
(unlabeled) code fence or as a standalone JSON line inside prose; anything
looser stays rejected. `PATCH.patch` is the complete replacement text for the
exact target page. The action must include the page's expected file hash; a
mismatch is rejected.

Every tool result is stored content-addressed under
`.camelid/context-paging/artifacts`. Only its bounded structured diagnostic is
eligible for the next capsule. When paging is active, shell output capture
becomes tail-inclusive (64 KiB head plus 192 KiB tail) before external
storage, because test failures print their assertions near the end of logs;
the model still sees only the compact bounded summary. The capture mode is
scoped to the paging session's own agent-loop thread and set explicitly each
run, so a concurrent session without paging keeps the legacy head-only 16 KiB
clip. Successful `search`, `list_dir`, and `read_file`
results reach the next capsule as compact "ok"-status diagnostics with
reference IDs — fresh capsules never replay history, so the compact diagnostic
is the only channel through which any tool result reaches the model. Runtime
metrics and repeated page-fault pins are persisted separately from the ledger,
so restarting the inference session does not reset canonical progress or
observability.

## Robustness and loop bounds

Indexing failure is contained per file: a file that cannot be indexed (over
the 1 MiB per-file limit, non-UTF-8, or changed mid-walk) is skipped and its
stale records are purged instead of failing the whole runtime. Files deleted
mid-session have their map entries, cards, and pages purged on the next
refresh. A corrupt project index or runtime-state file is rebuilt/reset from
source instead of refusing to start; the canonical task ledger stays strict.

Verification is host-owned. `COMPLETE` is accepted only when host-run
verification has passed. A prose answer after a workspace change but before
verification is reprompted (bounded) and can never overwrite a failed
verification status with "complete"; `BLOCKED` never marks the task complete.

The paging loop is bounded: 16 consecutive typed-action steps that execute no
workspace action end the run. An exact-tokenizer overflow recalibrates the
estimator from the measured count and rebuilds a smaller capsule (up to 3
times per run) instead of failing the run.

Persistence is crash-safe and Windows-concurrency-safe: the ledger, index,
runtime-state, and raw artifacts are written through unique same-directory
temp files and atomic replacement, so parent and delegated workers cannot
truncate or rename one another's staging files. Delegated workers also receive
stable task scopes so identical objectives do not share canonical ledger or
runtime-state files. Retrieval misses are persisted even when the fault fails.

## Configuration

- `CAMELID_CONTEXT_PAGING=0`: disable the default Web Code paging integration.
- `CAMELID_CONTEXT_MAX_INPUT_TOKENS`: optional explicit input ceiling. Unset or
  invalid values keep the default `5500`-token active input working set; with
  the default reserves this stays inside the checked Qwen 4B 8K envelope.
  An explicit value can opt into a larger working set, but is still clamped to
  the session context window minus the output and safety reserves.
- `CAMELID_CONTEXT_OUTPUT_RESERVE`: output reserve, default `1300`.
- `CAMELID_CONTEXT_SAFETY_RESERVE`: safety reserve, default `1200`.
- `CAMELID_CONTEXT_TOOL_RESULT_BYTES`: compact tool-result preview bytes,
  default `2048`, minimum `256`.
- `CAMELID_CONTEXT_TOOL_RESULT_LINES`: compact tool-result preview lines,
  default `32`, minimum `4`.
- `CAMELID_CONTEXT_DEBUG=1`: record item inclusion/exclusion explanations.

Invalid numeric values fail closed to their documented defaults. A capsule that
cannot retain its mandatory task contract, current focus, critical invariants,
diagnostic, and exact target source returns a budget error rather than silently
dropping them.

## Known first-slice limits

- Structural extraction covers ordinary Rust/Python declarations; macros and
  generated sources fall back to bounded file pages.
- Caller/callee edges are lexical heuristics, not a compiler call graph, and
  `imports` and `dependencies` currently duplicate each other.
- Multi-line raw strings can still fool Rust block-end detection; the error is
  absorbed safely by source-hash checks and unique-match patching.
- Output-token metrics depend on the driver reporting completion tokens;
  streaming responses may not.
- Typed `PATCH` uses exact page replacement rather than arbitrary unified diff.
- File-level exact pages are limited to 16 KiB; larger files must be changed by
  symbol page or a later bounded-range paging adapter.
- The benchmark is a deterministic fixture, not a live-model comparison; the
  legacy loop remains available through the rollback switch while live-model
  coverage expands.

The next iteration should add compiler/LSP diagnostics adapters, more language
indexers, and direct UI controls for capsule-debug and page-fault metrics.
