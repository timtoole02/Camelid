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

- Rollout is opt-in with `CAMELID_CONTEXT_PAGING=1`; existing agent behavior is
  unchanged when disabled.
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
- runtime metrics and capsule selection/exclusion explanations.

`ContextCapsuleBuilder` emits a fresh capsule for one action. The stable kernel
is a byte-stable prefix; task-specific state follows it. Items are ordered by
category and stable identifiers. The builder uses a `TokenEstimator` interface;
the integrated request is still checked with Camelid's exact loaded-model
tokenizer preflight before inference.

The typed action protocol is JSON with one of: `NEED_CONTEXT`, `SEARCH`,
`PATCH`, `RUN_TEST`, `INSPECT_DIAGNOSTIC`, `UPDATE_PLAN`, `COMPLETE`, or
`BLOCKED`. The live slice executes `NEED_CONTEXT` and translates `PATCH`,
`SEARCH`, and `RUN_TEST` through the existing tool boundary. Diagnostic
inspection and plan updates remain host-owned state actions.
`PATCH.patch` is the complete replacement text for the exact target page. The
action must include the page's expected file hash; a mismatch is rejected.

Every tool result is stored content-addressed under
`.camelid/context-paging/artifacts`. Only its bounded structured diagnostic is
eligible for the next capsule. Runtime metrics and repeated page-fault pins are
persisted separately from the ledger, so restarting the inference session does
not reset canonical progress or observability.

## Configuration

- `CAMELID_CONTEXT_PAGING=1`: enable the experimental Web Code integration.
- `CAMELID_CONTEXT_MAX_INPUT_TOKENS`: input ceiling, default `5500`.
- `CAMELID_CONTEXT_OUTPUT_RESERVE`: output reserve, default `1300`.
- `CAMELID_CONTEXT_SAFETY_RESERVE`: safety reserve, default `1200`.
- `CAMELID_CONTEXT_DEBUG=1`: record item inclusion/exclusion explanations.

Invalid numeric values fail closed to the documented defaults. A capsule that
cannot retain its mandatory task contract, current focus, critical invariants,
diagnostic, and exact target source returns a budget error rather than silently
dropping them.

## Known first-slice limits

- Structural extraction covers ordinary Rust/Python declarations; macros and
  generated sources fall back to bounded file pages.
- Call/dependency edges are lexical evidence, not a compiler call graph.
- Typed `PATCH` uses exact page replacement rather than arbitrary unified diff.
- File-level exact pages are limited to 16 KiB; larger files must be changed by
  symbol page or a later bounded-range paging adapter.
- The rollout is opt-in while live-model benchmark coverage is expanded beyond
  the deterministic acceptance fixture.

The next iteration should add compiler/LSP diagnostics adapters, more language
indexers, and direct UI controls for capsule-debug and page-fault metrics.
