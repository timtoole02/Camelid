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
  pages, compact artifact store, backward-compatible typed-action parsing, and
  the deterministic capsule builder.
- `src/chat/workspace_bridge.rs`: enables the default-on runtime for Web Code
  and retains an explicit rollback switch.
- `src/chat/agent.rs`: constructs a new capsule for every action and keeps one
  stable native-tool vocabulary through active modification and verification.
  The existing tokenizer preflight remains the final hard request gate; tool
  validation and approval boundaries remain authoritative.
- `src/chat/tools.rs`: bounded file/search primitives and audited writes remain
  the execution layer. Raw paging artifacts are never executable authority.
- `src/chat/workspace_memory.rs`: remains the user-visible thread store. Context
  Paging state is independent so a fresh inference session can resume from the
  ledger without replaying chat.

Compatibility constraints:

- Context Paging is the default Web Code runtime. Set `CAMELID_CONTEXT_PAGING=0`
  only as a rollback/diagnostic switch. Terminal agent mode and read-only
  Workspace retain their existing history behavior. Web Code subagents inherit
  paging and the parent's fixed context envelope, but keep isolated task state.
- `.camelid` is already protected from model-authored writes. Runtime state is
  stored below `.camelid/context-paging` through host code only.
- Exact source is authoritative. A card or page whose file hash no longer
  matches is excluded and must be rebuilt.
- Typed patches are translated to ordinary `edit_file` calls and pass the
  workspace sandbox, approval, checkpoint, and audit layers. When a valid
  native edit or overwrite targets indexed source that was evicted from the
  current capsule, the host raises a page fault, makes that exact page mandatory,
  and asks the model to retry. Wrong edits remain rejected. An edit whose exact
  old and new text are identical is acknowledged as already satisfied only when
  the current indexed source and hash prove that text is on disk; it advances
  to verification instead of consuming the invalid-action retry budget.
  Overwrites of existing files require a bounded full exact file page.
- No embeddings, vector database, multi-agent dependency, or growing KV cache
  is required by the first slice.

## Vertical-slice architecture

The first slice gives common UTF-8 source, build, configuration, test, and
documentation artifacts hash-backed path authority and exact bounded pages.
Rust and Python additionally receive deterministic symbol extraction with
source-derived signatures and balanced source ranges. The runtime retains up
to 8,192 authority paths and 256 hydrated files at once; an explicitly named
supported text path can be admitted and hydrated on demand when it lies beyond
that initial inventory. Binary, oversized, sensitive, and ambiguous targets
fail closed instead of being treated as new files. It persists:

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

`ContextCapsuleBuilder` emits a fresh capsule for one action. Prefix order is
chosen for the actual Qwen cache key: the byte-stable kernel, immutable task
contract, stable active-work tool list, project map, cards, and exact pages all
precede the late mutable task state. The persisted ledger revision is not model
input; source hashes and project-index revisions enforce freshness. Items within
a category use stable identifiers. Eviction strictly follows the spec priority
ladder as a prefix take — first removed to last removed: failed-attempt
history, low-relevance cards, dependency pages, completed-work detail,
repository map, task detail — never a greedy fill that would keep small
low-priority history while dropping higher-priority evidence. Never-evict
content is the stable kernel, the bounded task contract, the current
diagnostic, the phase tool list, the modification target page (the most
recently faulted symbol), and every pinned page. Pinned pages are capped at 4;
the least-faulted pin is released first. The immutable contract contains the
exact `objective`, `acceptanceCriteria`, and `criticalInvariants`; late mandatory
task state contains `action`, `focus`, and `verification`. The immutable user
objective is preserved verbatim and fails closed if it cannot fit; mutable focus
is bounded to 600 bytes, and criterion/invariant lists render at most 6 items of
240 bytes. `decisions` and `openQuestions` render as evictable task detail.
Ledger lists themselves are bounded (128 items, 480 bytes per item), so
model-authored state cannot grow without bound. The builder uses a
`TokenEstimator` interface that is continuously calibrated from the live
tokenizer's measured tokens-per-byte rate; the integrated request is still
checked with Camelid's exact loaded-model tokenizer preflight before inference.

Dense Qwen3 with F32 resident Metal KV admits a partial prefix only when the
common-prefix/divergent-suffix token ratio is at least 48:1; that threshold is a
measured M4 break-even, not a paging policy knob. The late task state is kept
compact so ordinary Modify/Verify transitions can clear it. A real write still
changes source hashes, map rows, and pages and may correctly force one cold
prefill; the layout does not claim that every action is cacheable. On the exact
recorded multi-file Python/Qwen3-4B-Q8_0 benchmark fixture with six active-work
schemas and unchanged preceding evidence, the measured integer reuse ratios
are 53:1 (Modify to pending Verify), 66:1 (pending to plain Verify), and 67:1
(source-fault retry). The fixture is only a benchmark workload; the runtime
does not match its project name, file layout, language, or application domain.

The advertised action protocol is the model's existing native function-call
format: one advertised tool call per step, followed by a concise plain-text
answer only after host verification. `read_file` doubles as the exact-source
page-fault operation, so a small model does not need to learn a second JSON
protocol before it can edit. `list_dir` with an omitted path is deterministically
repaired to the workspace root. Modify and Verify expose the same scoped native
tool set, allowing multi-file work to continue after the first write and keeping
tool schemas stable across active steps. Once the model declares modification
settled (including an already-satisfied edit or a premature completion claim),
the next step exposes only `run_shell` when available; this deliberate one-step
cache break prevents another cosmetic edit loop and makes execution verification
mandatory. A failing command restores the complete active-work tool set and its
diagnostic. The earlier typed actions
(`NEED_CONTEXT`, `PATCH`, `SEARCH`, `RUN_TEST`, `INSPECT_DIAGNOSTIC`,
`UPDATE_PLAN`, `COMPLETE`, and `BLOCKED`) remain accepted for persisted/older
clients but are no longer advertised in the stable kernel. Typed `PATCH` still
requires a complete exact-page replacement and the expected file hash, but any
recovery after a legacy typed action is expressed back to the model only with
advertised native `read_file`, `edit_file`, or `write_file` calls.
Explicit relative artifact names in the immutable objective form a conservative
host-owned completion manifest. Missing entries keep the runtime in Modify even
after an earlier file passed verification, preventing multi-file creation from
ending after its first artifact.

Every tool result is stored content-addressed under
`.camelid/context-paging/artifacts`. Only its bounded structured diagnostic is
eligible for the next capsule. When paging is active, shell output capture
becomes tail-inclusive (64 KiB head plus 192 KiB tail) before external
storage, because test failures print their assertions near the end of logs;
the model still sees only the compact bounded summary. The capture mode is
scoped to the paging session's own agent-loop thread and set explicitly each
run, so a concurrent session without paging keeps the legacy head-only 16 KiB
clip. Successful `search` and `list_dir` results, plus reads that cannot be
represented by one bounded full-file page, reach the next capsule as compact
"ok"-status diagnostics with reference IDs. A successful small-file read instead
faults in its canonical hash-backed full page and drops the duplicate numbered
preview. Fresh capsules never replay history, so one of those bounded channels
must carry each observation. Runtime
metrics and repeated page-fault pins are persisted separately from the ledger,
so restarting the inference session does not reset canonical progress or
observability.

## Robustness and loop bounds

Indexing failure is contained per file: a supported path that cannot be
hydrated (over the 1 MiB per-file limit, non-UTF-8, or changed mid-read) retains
at most path authority while stale cards/pages are purged; a later modification
fails closed rather than guessing bytes. Files deleted mid-session have their
map entries, cards, and pages purged on the next refresh. A corrupt project
index or runtime-state file is rebuilt/reset from source instead of refusing to
start; the canonical task ledger stays strict.

Verification is host-owned. `COMPLETE` is accepted only when host-run
verification has passed. A prose answer after a workspace change but before
verification is reprompted (bounded) and can never overwrite a failed
verification status with "complete"; `BLOCKED` never marks the task complete.
A user-declared test, launch, or manual-validation command is retained as an
exact host obligation independent of language or framework. Ecosystem adapters
can recognize additional conventional runners, but an allowlist is not the
completion authority; successful probes, help, collection-only, or syntax-only
commands cannot satisfy application-execution evidence.
A successful reread proves the saved bytes but does not by itself mark Code
verification passed when `run_shell` is available; a post-write test, build, or
syntax command must also succeed. Successful environment probes such as
`python --version`, `ls`, and `pwd` are not verification. When `run_shell` is not
advertised, the bounded host read-verification path remains valid and the stable
kernel does not instruct the model to call a missing tool.

The paging loop is bounded: 16 consecutive model steps that execute no workspace
action end the run. A trailing host retry reminder is copied, bounded, into the
next capsule's mandatory `currentAction`; this prevents invalid native calls,
malformed envelopes, or capped replies from receiving a byte-identical retry
prompt. When a deterministic model repeats the same successful observation
twice, that observation tool is also omitted from the next native schema until
a different action succeeds; the recovery changes what the model can select
instead of relying on prose alone. Any later tool result consumes the one-shot
feedback. An exact-tokenizer
overflow recalibrates the estimator from the measured count and rebuilds a
smaller capsule (up to 3 times per run) instead of failing the run.

Persistence publication is atomic against concurrent writers and process
interruption: the ledger, index, runtime-state, and raw artifacts are written
through unique same-directory temporary files and atomically replaced.
Concurrent workers cannot collide on a shared temp name or publish half-written
JSON. This is a consistency guarantee, not power-loss durability; files and
parent directories are not explicitly synced to stable storage. Child ledgers
and runtime state are task-scoped. The project index remains shared, derived
workspace state and is published only after a fresh index or structural edit;
task-local ledger/metrics saves never rewrite a possibly stale project copy.
Retrieval misses are persisted even when the fault fails.

## Configuration

- `CAMELID_AGENT_CONTEXT_MAX_TOKENS`: optional process-wide cap for adaptive
  session context selection. Automatic selection divides its memory allowance
  across each active engine generation slot's resident and CPU-mirrored KV plus
  retained prompt-prefix cache entries, then clamps the result to Camelid's
  validated agent-context ceiling (and any lower model/server limit). When that
  aggregate allowance cannot hold the selected active working set, Camelid
  reselects for exactly one owner, permanently serializes KV-owning engine work
  for the process, and disables the retained prompt cache before the agent
  starts. If even that one-owner raw estimate is below the required active
  envelope, Workspace fails closed before the first model request rather than
  relying on memory compression or swap; the operational floor is never
  relabeled as memory-safe.
  A GGUF's larger native-context declaration alone does not widen supported
  agent context. The status API separates raw memory-derived capacity from the
  8K minimum operational recommendation. Invalid and zero values leave
  automatic selection in control.
  The exact Qwen3 4B Q8_0 Code row is the narrow exception: when paging is
  enabled it receives a 16K logical task envelope, while the enforced active
  input + output + safety working set remains 8K and inside the validated
  request envelope. This is not a 16K single-prompt support claim. The exact
  Qwen3-4B-Q4_K_M row keeps the legacy 8K operational agent envelope needed to
  fit the paging capsule and output/safety reserves, but it receives no 16K
  logical exception. That operational value is not a Q4 context-support claim:
  the row's promoted parity-context ladder remains 512/1,024, and its
  non-contiguous 4K/8K sweep matches remain unclaimed while the 2K bucket is a
  disclosed near-tie.
- `CAMELID_CONTEXT_PAGING=0`: disable Context Paging for Web Code and use the
  legacy growing-transcript loop. Context Paging is enabled when the variable is
  absent; `1` explicitly keeps it enabled.
- `CAMELID_CONTEXT_MAX_INPUT_TOKENS`: active input ceiling. Unset or invalid
  values use `5500`; an explicit value may raise it only within the session
  context remaining after output and safety reserves.
- `CAMELID_CONTEXT_OUTPUT_RESERVE`: output reserve, default `1300`.
- `CAMELID_CONTEXT_SAFETY_RESERVE`: safety reserve, default `1200`.
- `CAMELID_CONTEXT_TOOL_RESULT_BYTES`: compact tool-result preview bytes,
  default `2048`, minimum `256`.
- `CAMELID_CONTEXT_TOOL_RESULT_LINES`: compact tool-result preview lines,
  default `32`, minimum `4`.
- `CAMELID_CONTEXT_DEBUG=1`: record item inclusion/exclusion explanations.

Invalid numeric values fail closed to the documented defaults. A capsule that
cannot retain its immutable task contract, late mandatory task state, current
diagnostic, and exact target source returns a budget error rather than silently
dropping them.

## Known first-slice limits

- Structural extraction covers ordinary Rust/Python declarations. Other
  supported text ecosystems use exact full-file or overlapping line-safe chunk
  pages; macros and generated sources use the same bounded fallback.
- Caller/callee edges are lexical heuristics, not a compiler call graph, and
  `imports` and `dependencies` currently duplicate each other.
- Multi-line raw strings can still fool Rust block-end detection; the error is
  absorbed safely by source-hash checks and unique-match patching.
- Output-token metrics depend on the driver reporting completion tokens;
  streaming responses may not.
- Typed `PATCH` uses exact page replacement rather than arbitrary unified diff.
- Complete-file exact pages are limited to 8 KiB and the text-authority reader
  is limited to 1 MiB. Larger supported text files use overlapping exact chunk
  pages for narrow edits; whole-file overwrite remains fail-closed unless the
  complete current file fits the bounded page.
- The benchmark is a deterministic fixture, not a live-model throughput claim;
  default-on live-model receipts remain part of release validation.

The next iteration should add compiler/LSP diagnostics adapters, more language
indexers, and direct UI controls for capsule-debug and page-fault metrics.
