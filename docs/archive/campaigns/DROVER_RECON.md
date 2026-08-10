# DROVER_RECON.md — Phase 0 reconnaissance

**Campaign:** DROVER — finish the `camelid` CLI agent.
**Base:** `origin/main` @ `b9f9a40`, worktree `/Volumes/Untitled/Camelid-drover`, branch `drover/phase0-recon`.
**Date:** 2026-07-22. **Host:** Mac mini M4, 16 GiB (`hw.model = Mac16,10`).
**Status:** awaiting Tim's signature on §7 (Gate G0). No code has been written.

Every claim below is `file:line` against the base commit and was produced by a read-only sweep
(20 agents, cross-checked by an adversarial completeness pass). Where the campaign brief and the
repo disagree, **the repo wins** and the disagreement is recorded in §8 (Amendment log).

---

## 1. Baseline — the tree is green before DROVER touches it

Measured on this worktree, `CARGO_TARGET_DIR=/Volumes/Untitled/drover-target`:

| Gate | Result |
|---|---|
| `cargo check --bin camelid` | clean (19.2 s) |
| `cargo fmt --check` | clean |
| `cargo clippy --bin camelid -- -D warnings` | clean (15.4 s) |
| `cargo test --bin camelid chat::` | **91 passed, 0 failed**, 19 filtered |
| `node scripts/check-ledger-drift.mjs` | passed — ledger == code contract |

That is the reference every DROVER gate returns to.

> **Correction to §2 of the brief.** `CONTRIBUTING.md:56-60` prescribes `--all-targets --all-features`.
> On macOS that is **wrong and does not reproduce CI**: `.github/workflows/ci.yml:100-102` sets
> macOS to `features: ""` (`macos -> default (Metal lane; no cudarc)`), and `--all-features` turns
> on `#[cfg(feature = "cuda")]` code (`src/main.rs:645`) with cudarc excluded from the graph by
> `Cargo.toml:92-99`. **On this box the standing suite must run with `features=""`.**

---

## 2. §0.1 inventory — verified against the live tree

All anchors exist. Line numbers corrected where the brief was approximate.

### 2.1 The loop — `src/chat/agent.rs`

| Item | Anchor | Note |
|---|---|---|
| `run_loop` | `agent.rs:189` | 8 params, `#[allow(clippy::too_many_arguments)]` at `:188`. **No doc comment** — the three doc lines at `:156-158` that read like its contract actually attach to `const REPEAT_LIMIT` at `:159`. |
| `ModelDriver` | `agent.rs:69` | `fn step(&mut self, history: &[AgentMsg], tools: &[ToolSpec]) -> Result<ModelStep, String>` |
| `Approver` | `agent.rs:87` | `fn approve(&mut self, action: &Action, sandbox: &Sandbox) -> Decision` |
| `Reporter` | `agent.rs:92` | 4 methods: `model_text`, `tool_call`, `tool_result`, `notice` |
| `note_no_progress` | `agent.rs:167` | **Result-aware, as claimed**: a changed result resets the counter (`:172-178`). `REPEAT_LIMIT = 3` (`:159`). Fires on both the validation-error path (`:240`) and the executed/denied path (`:296`) — 3 identical *denials* also end a run. |
| Step cap | `agent.rs:205` | `for _ in 0..cfg.max_steps` — counts **model steps, not tool calls**; one `ModelStep::Calls` executes every call in the vec inside one step (`:225`). |
| Cancellation | `agent.rs:206`, `:226` | `AtomicBool`, checked before each model step and each tool call. Backed by `session::CANCEL` (`session.rs:18`), set by `on_sigint` (`mod.rs:280`). |
| `AgentMsg` | `agent.rs:60` | The transcript element. |
| `system_prompt` | `agent.rs:351` | Brief said "~line 351" — **exact**. |
| `LiveDriver` | `agent.rs:395` | `impl ModelDriver` at `:448`. |
| `run_agent` (line renderer) | `agent.rs:713` | |
| `MockDriver` (test) | `agent.rs:933` | The HARNESS-first substrate the brief relies on. Out-of-script → `ModelStep::Text("(out of script)")` (`:940`). |

### 2.2 Tools, sandbox, approvals

| Item | Anchor | Note |
|---|---|---|
| `specs()` | `tools.rs:324` | **`specs(allow_net: bool, shell_mode: ShellSandbox) -> Vec<ToolSpec>`** — built at runtime, not a const table. Good news for G7. |
| `Sandbox` | `tools.rs:202` | Jail check `canon.starts_with(&self.root)` at `tools.rs:294`. This is `Path::starts_with`, **component-wise**, so `/rootevil` does *not* match `/root` — there is no sibling-prefix bug. Do not "fix" it. |
| `ToolSpec` | `tools.rs:169` | `name: &'static str`, `description: &'static str`. **This typing is the G7 blocker — see §5.1.** |
| `validate()` | `tools.rs:872-1156` | |
| `ShellSandbox` | `shell_sandbox.rs` | disabled / sandboxed / unrestricted |
| `audit.rs` | `audit.rs` | webhook events |
| `NonInteractiveApprover` | `subagent.rs` | reused verbatim by G4 |

### 2.3 The shipped gates and the identity gate

- `agent-eval`, `agent-syscap-eval`, `agent-orchestration-eval`, `agent-orchestration-bench` all exist
  (`mod.rs:210/229/252/271`).
- `tool_capable` is declared at **`src/api/mod.rs:399`**. Enforced at `agent.rs:714-727` (inline) and
  `agent_tui.rs:151` (TUI); refusal exits 2. The predicate is
  `Session::active_tool_capable()` (`session.rs:197-203`) — exact string match on ledger row id.
- **Exactly 5 rows are `tool_capable: true`** (verified first-hand):
  `src/api/mod.rs:2936, 3026, 3201, 3926, 3977`.

---

## 3. The gaps — all eight confirmed ABSENT

Every G1–G8 absence in the brief's table is real. Proven by word-boundary grep, not inference.
Notable specifics:

- **G1** — `system_prompt` (`agent.rs:351-385`) reads no file, consults no config, takes only
  `(&Sandbox, &[ToolSpec])`. No `CAMELID.md` / `AGENTS.md` / `CLAUDE.md` exists in the tree and no
  code references such a name.
- **G2** — nothing ever removes from `history`. Per-tool output truncation exists; transcript
  compaction does not.
- **G3** — the loop is **validate/approve/execute/observe**. There is no plan phase; "plan-act-observe"
  appears only in a doc comment (`agent.rs:1`, `agent_tui.rs:10`). No plan struct, no plan tool.
- **G7** — `rg -ni '\bmcp\b'` over `*.rs`/`*.md`/`*.toml`/`*.ts*`/`*.json` (excluding `qa/`, `target/`,
  `node_modules/`) returns **zero** hits. The only repo-wide match is `▁MCP` inside a tokenizer vocab
  dump in a QA log.

**One correction to the brief's framing of G2**: the brief calls compaction a robustness fix against
"the model's window." The real ceiling is tighter — see §4.3.

---

## 4. Findings that change the plan

These are the recon results a Phase-1 implementer would otherwise discover the hard way.

### 4.1 There is no untrusted-data wrapper to reuse — DROVER must create it

The brief's §0.3(1) instructs every new surface to "route its output through the same 'untrusted'
framing the file/shell tools already use." **That framing does not exist in code.** Verified
first-hand at `src/chat/agent.rs:629-631`:

```rust
AgentMsg::ToolResult { name, outcome } => {
    out.push(json!({"role":"tool","name":name,"content":outcome.text()}));
}
```

No prefix, no delimiter, no fence. The entire untrusted-data treatment today is **one sentence in
the system prompt** (`agent.rs:380-383`) plus prose inside two tool descriptions (`tools.rs:368, 394`).

This does not weaken the *enforcement* invariant — enforcement is in `validate()` + `Sandbox` and is
real. But it means the brief's repeated "same framing as the existing tools" is a reference to
something that isn't there. **Recommendation: a pre-Phase-1 commit adds the wrapper at `agent.rs:630`
so G7/G8 have a real thing to inherit**, rather than each phase inventing its own.

### 4.2 G1 invalidates the `tool_capable` promotion basis — the campaign's biggest hidden coupling

`system_prompt` has **five** callers, and one of them is the promotion harness:

```
agent.rs:890 · agent_tui.rs:438 · subagent.rs:613 · agent_orchestration.rs:520 · agent_eval.rs:290
```

The shipped ledger prose states `tool_capable` is *"earned ONLY by those receipts"*
(`src/api/mod.rs:3019-3023`, `:3062-3063`). All five `true` rows were promoted under the **current**
prompt. G1 changes what `agent-eval` feeds the model, so the receipts no longer attest the shipped
prompt.

Re-minting is not locally possible for 3 of the 5 rows (§4.4). **G1 must therefore either (a) exclude
`agent_eval.rs:290` from project-instruction injection and record that in DECISIONS, or (b) accept
stale promotion evidence.** I recommend (a): the eval harness should attest the *baseline* prompt.
Two tests pin the flag values and will break if a row flips: `tests/api_vertical_slice.rs:1140, :1207`.

### 4.3 G2's budget is a ledger ceiling (8192), not the model's `n_ctx`

For the recommended CERT row `qwen3_4b_instruct_q8_0`, `src/api/mod.rs:3941` records
`bounded_512_1024_2048_4096_8192_context_raw_decode_parity`, topping out at
`bounded_context_8192_window: 8192` (`:3945-3960`), and `:3932` lists model-native/larger context as a
`full_support_blocker`.

So an agent transcript beyond 8192 tokens is **outside the row's support claim even if the server
accepts it**. The brief's pre-registered "80% of the active model's training `ctx`" should be
re-pinned to **80% of the row's validated envelope**. On the recommended row that is
**6554 tokens, compaction target 4096** — which also makes G2 *urgent*, not theoretical: a tool-heavy
coding session crosses 6.5k well inside an hour.

### 4.4 CERT model availability on this box

| `tool_capable` row | Local weights | Verdict |
|---|---|---|
| `qwen3_4b_instruct_q8_0` | `/Volumes/Untitled/models/Qwen3-4B-Q8_0.gguf` (4,280,404,704 B) | **PRIMARY PIN** |
| `llama32_3b_instruct_q8_0` | `/Volumes/Untitled/models/Llama-3.2-3B-Instruct-Q8_0.gguf` (3,421,899,296 B) | secondary |
| `Ornith 1.0 9B` (Q8_0) | present, 8.87 GiB | unusable — CPU-only lane, ~1 s/token |
| `ornith_1_0_9b_q4_k_m` | present but **+288 B vs its receipt** | not the certified artifact |
| `qwen3_4b_q4_k_m` | **absent** (whole-volume `find`, zero matches) | unavailable |

**Primary pin: `qwen3_4b_instruct_q8_0`.** It is the only row that is simultaneously present,
byte-size-identical to its committed agent-eval PASS receipt, byte-size-identical to the curated
catalog constant (so `active_tool_capable()` passes via *both* the picker and `--model`), and small
enough for 16 GiB with the default-on Metal resident stack (`src/main.rs:5044-5058`).

Two caveats to record: the Llama-3.2-3B catalog constant is **+480 B off** from the on-disk file
(`src/api/mod.rs:17785`), so the picker renders it `NotDownloaded` despite it being the receipted
artifact; and both locally-viable rows carry the **thinnest** existing evidence (1-case batteries,
vs 3-case for the Ornith/Q4_K_M rows).

### 4.5 A second, ungated tool executor exists

Only two non-test `action.execute()` sites exist:

- `agent.rs:338` — inside `execute_audited` (approver + policy + audit + production check)
- **`agent_syscap.rs:198` — no Approver, no policy, no audit, no `is_production()`**

`agent_syscap.rs:192-200` goes `ToolCall → tools::validate → action.execute(&sandbox)` directly.
**Any tool G3 or G8 adds to `validate()` becomes reachable there with no approval and no audit event.**
G3's `update_plan` is harmless there; **G8's `web_search` is not.** This must be closed in the same
commit that lands a network tool.

### 4.6 `LoopEnd` has one silent catch-all

`agent.rs:903-909` and `agent_tui.rs:1033-1041` are exhaustive and will break loudly on a new variant.
But `subagent.rs:660-664` is:

```rust
let status = match end {
    agent::LoopEnd::Answered => "completed",
    agent::LoopEnd::Aborted  => "inconclusive",
    _ => "failed",
};
```

A new variant (G2 `ContextExhausted`, anything G4 adds) **compiles silently and becomes `"failed"`**
in every subagent result file and `__subagent` exit code (`subagent.rs:548-552`). `LoopEnd` is
already mapped to an outcome three times with three vocabularies (`agent.rs:903` labels,
`subagent.rs:660` strings, `agent_eval.rs` `EvalOutcome`); G4 adding exit codes as a fourth without
unifying guarantees drift.

### 4.7 Nothing pins the surfaces G1/G3/G8 change

Hunted and confirmed absent: no `insta`, no `expect_test`, no `.snap`, no `assert_cmd`/`trycmd`,
no `--help` capture anywhere in `scripts/`, `tests/`, `.github/`. Nothing asserts the system prompt
text, the tool list, the tool count, the help output, or the slash list.

That is **bad** news: those phases would ship completely unguarded. Related: `agent_tui.rs` has
**zero** tests, and nothing in `scripts/`/`tests/`/`.github/` ever launches `chat --agent`
(`rg -n -F -- "--agent" scripts/ tests/ .github/` → exit 1).

### 4.8 Smaller landmines

- **Symlink write-escape.** `tools.rs:281-292` (`must_exist=false`) canonicalizes only the *parent*,
  then joins the filename — the final component is never resolved. Used by `write_file`
  (`tools.rs:1251`) and Windows `screenshot` (`tools.rs:1152`). **G5 hooks exactly those two write
  sites** and must not widen it.
- **`.camelid/` is not in `.gitignore`.** `subagent.rs:31` already writes `.camelid/subagents/` into
  the user's workspace and never deletes `result_*.json` (`subagent.rs:546` removes only the task
  file). G5/G6 will follow that convention into every repo the agent runs in. Worse: `search` skips
  only `.git`/`target`/`node_modules` (`tools.rs:1220`), so the agent will **index its own
  checkpoints and transcripts and feed them back as untrusted context** — precisely the loop
  `agent.rs:380` warns about.
- **`AgentConfig` has no `Default`** (`agent.rs:28-48`) and 5 construction sites — `mod.rs:153`,
  `subagent.rs:592`, `agent_eval.rs:293`, `agent_orchestration.rs:504`, `agent.rs:974`. `audit: Box<dyn
  AuditSink>` (`:45`) blocks `#[derive(Default)]` forever. **G1, G4, G6, G7 each add a field here**,
  so each phase edits all five — two of which mint promotion evidence. Highest-traffic edit surface
  in the campaign.
- **Every phase ships twice.** `LiveDriver::step` (blocking, structured `tool_calls`, `agent.rs:481-490`)
  vs `step_streamed` (TUI, `tool_parse` only, `agent.rs:548`). `ChatCompletionDelta`
  (`src/api/mod.rs:1333`) has no `tool_calls` field, so this split is structural. Neither path has a
  test.
- **Agent-family receipts are unverifiable.** `camelid verify-receipt` hard-rejects any schema
  != `camelid.parity-receipt/v1` (`src/receipt/verify.rs:151-154`); `verify_self_digest` is private
  and called only from `debug_assert!` — **compiled out of release builds**. A DROVER receipt
  mirroring the syscap template inherits an artifact no shipped tool can check.
- **`agent_eval` receipts are not hashed**, contradicting `DECISIONS.md:454-455`. Only the
  syscap/orchestration/bench family is tamper-evident.
- **A real bug in a gate harness:** `agent_orchestration.rs:341-350` `family_for` goes
  `qwen → mistral → llama` with **no ornith arm**, while `agent_eval.rs:424-442` has the ornith arm
  *with a comment explaining it must precede `qwen`*. Two filename→family guessers; one silently wrong.
- **Doc drift to fix in passing:** `agent.rs:7-9` and `DECISIONS.md:395` still call the TUI agent "a
  documented follow-up" — it shipped and is the default (`mod.rs:169-170`). `RECON_AGENT.md:41`
  describes an in-session `/agent` toggle that does not exist (`tui.rs:406-409` only prints a
  relaunch hint).

---

## 5. Sequencing — the recommended order changes

### 5.1 G7 (MCP) must go first, or the tools phases get rewritten

`ToolSpec.name`/`.description` are `&'static str` (`tools.rs:169-174`, verified first-hand) and
`Action::tool_name(&self) -> &'static str` (`tools.rs:684`). Runtime-registered MCP tools force both
to `String`/`Cow`. G3 and G8 each add tools as `&'static str` literals across the same seven sites
(`specs` 324, `Action` 561, `risk` 664, `tool_name` 684, `call_line` 708, `execute` 813, `validate`
872-1156). **Landing G3 and G8 before G7 means rewriting both.** G7 also adds a parameter to
`specs()`, moving all 8 call sites (`agent.rs:199`, `agent.rs:826`, `agent_tui.rs:260/437/514`,
`agent_orchestration.rs:461`, `subagent.rs:583`, `agent_eval.rs:271`).

### 5.2 Other hazards

- **G3, G5, G7, G8 are a four-way collision on `tools.rs`** — same seven `match self` arms, same
  `validate()` block. They cannot be parallel branches; per the worktree rule they must be serialized.
- **G1 and G6 collide** on the five `AgentMsg::System(system_prompt(...))` seed sites; both rewrite
  `agent.rs:889-892` and `agent_tui.rs:439`.
- **G4's tests are invalidated by G1 and G2** — a headless gate asserting step counts measures a
  prompt (G1) and a transcript policy (G2) that later phases change. Either order G1 → G2 → G4, or
  make G4's gate outcome-only.
- **G4 would create a fourth copy of the bounded-model-load block**, already verbatim three times
  (`agent_eval.rs:186-241`, `agent_orchestration.rs:391-424`, `agent_bench.rs:220-243`). Extract
  before copying or the INCONCLUSIVE semantics drift a fourth way.

### 5.3 Recommended order

**P0.5 (new, pre-Phase-1) → G1 → G2 → G7 → G3 → G8 → G4 → G5 → G6**

`P0.5` is a small standalone commit that costs ~half a day and de-risks everything after it:
1. the untrusted-data wrapper at `agent.rs:630` (§4.1),
2. the three missing pins — a `system_prompt` shape test in the existing `mod tests` (`agent.rs:924`),
   an exhaustive advertised-tool-name-set assertion beside `tools.rs:1854`, and a slash-command
   parity test across `agent.rs:844` and `agent_tui.rs:472` (§4.7),
3. `.camelid/` into `.gitignore` **and** into `search`'s skip list (`tools.rs:1220`) (§4.8),
4. kill the `LoopEnd` catch-all at `subagent.rs:663` (§4.6).

The slash-parity test will fail on write — today `/theme` and `/sidebar` are TUI-only, `/stop`
(`agent.rs:882`) is a stub that always prints `"nothing running"`, and the `/help` string
(`agent.rs:880`) omits `/help` and `/sidebar`. That is a finding, not a blocker; fix or document.

G7 moves ahead of G3/G8 per §5.1. G4/G5/G6 stay last — they are the most self-contained and the
most droppable.

---

## 6. Pins for G4 / G6 / G7 — collision-checked

The CLI root (`struct Cli`, `src/main.rs:57`) has **no global flags**; every flag is subcommand-scoped.
`enum Command` (`:265-1178`) has **39 variants** (25 visible, 14 hidden). Prefix inference is **off**,
so `agent` can never be absorbed into `agent-eval`.

| Pin | Free? | Evidence |
|---|---|---|
| `camelid agent` (top-level verb) | **YES** | `rg -n '^\s{4}Agent\s*[{,(]' src/main.rs` → rc=1. Existing names are `agent-eval` (428), `agent-syscap-eval` (451), `agent-orchestration-eval` (467), `agent-orchestration-bench` (484). |
| `camelid agent exec "<goal>"` | **YES** | `exec` unclaimed as a subcommand; the only `Exec` is `Risk::Exec` (`tools.rs:32`). Copy the `gait` nested shape (`main.rs:1015-1020` + `GaitAction` `:256-262`) — the only existing nested subcommand. |
| `--resume` | **YES** | absent everywhere; the only `resume` strings are download-resume comments. |
| `-p` / `--print` | `--print` free; `-p` taken **only** on `tokenize` (`main.rs:555`) | The binary has exactly one short flag total. With no global args, `-p` is reusable on a new subcommand. |
| `--allow-mcp` | **YES** | zero `mcp` hits repo-wide. Fits beside `--allow-net` (`:397`) / `--allow-fs` (`:402`). |
| Hidden-until-stable | `#[command(hide = true)]` | 14 uses. House style for engine-internal verbs is a `__` prefix: `#[command(name = "__subagent", hide = true)]` (`:458`). |

**The sandbox-root flag is `--workdir` (`main.rs:381-382`), not `--sandbox-root`.** If DROVER proposes
`--sandbox-root` that is a *new* name sitting beside `--workdir`, not a reuse.

**`--max-steps` already means three different things** — `chat` default 25 (`:384`), `agent-eval`
default 6 (`:439`), and `diffusion-gemma-chat` where it is denoise steps per block (`:683-684`). Any
"consistent `--max-steps` semantics" claim is already false.

**Adding a flag to `chat` requires three edits in lockstep**: the `#[arg]` field (`main.rs:341-424`),
the destructuring pattern (`:1260-1283`), and `ChatOptions` (`src/chat/mod.rs:63-98`) plus its
construction (`:1284-1307`). The destructure is exhaustive, so a miss is a compile error — a good net.

### 6.1 Help and doc surfaces that must move in the same commit

`--help` is 100% clap-generated from doc comments (no `after_help`, no `clap_complete`). But **three
hand-written surfaces drift**:

1. `src/chat/palette.rs:13-124` — the declared single source of truth for the `/` palette (its
   `agent` entry at `:102-107` says "relaunch with `camelid chat --agent`", which a new `camelid agent`
   verb makes wrong).
2. `src/chat/inline.rs:502-533` — a second hard-coded help list that **already drifts** from
   palette.rs (omits `/agent`, `/theme`, `/sidebar`, `/stop`).
3. `README.md:108-111` (interface table) and `:113-118` (agent-mode paragraph).

### 6.2 Docs hygiene — three CI gates guard the files DROVER edits

- `scripts/check-public-evidence-claims.mjs:63-72` requires **exact substrings** in `README.md`,
  `STATUS.md`, `ROADMAP.md`, `COMPATIBILITY.md` and four more. Rewording the wrong line fails CI.
- `scripts/check-ledger-drift.mjs` parses the README supported-models table and the COMPATIBILITY
  at-a-glance table **by header regex** — reshaping either breaks parsing outright.
- `.github/workflows/ci.yml:159-161` is a **negative** grep: a new README section must not be titled
  `## Current Status` nor use five specific bullet labels.
- Any `src/api/mod.rs` edit fails CI unless `ledger/camelid-ledger.json` is regenerated in the same
  commit (`scripts/extract-capabilities-to-ledger.mjs`, wired at `ci.yml:149-150`).
- `scripts/check-public-scrub.sh:50-65` scans `docs/` recursively but **not** repo-root markdown —
  which is why this file is at the root, matching `RECON_AGENT.md` / `BASALT_CONDUCTOR.md`.

### 6.3 DECISIONS numbering

**Two `DECISIONS.md` files exist.** The live one is the repo root (`DECISIONS.md`, 1042 lines) — all
agent-mode decisions live there (D10, `:347`, plus five continuations at `:380/:412/:436/:473/:527`).
`docs/architecture/DECISIONS.md` is stale (D0001–D0007, last touched 2026-04-28), yet `DOCS.md:36`
links *that* one as the decision log — a hygiene defect worth fixing in passing.

**The root sequence is not monotonic**: D5 appears *after* D6/D7, and D6, D7, D11 each appear twice.
**Last plain number used is D17** (`:730`); there is no D18–D20.

**Precedent for the namespaced form exists**: BASALT used `D-B1..D-B5` as bullets inside the D17
entry, then promoted `D-B6` to its own `##` header (`:1021`). So `D-DROVER-1..5` may either nest under
one `## D18 — DROVER` entry or take their own headers, and **the namespaced form does not consume a
plain integer**. Recommendation: nest under `## D18 — DROVER`, matching the D17/D-B1..5 pattern.

### 6.4 Test/smoke surface

- **There is no agent-mode smoke script.** No `.sh` in the repo drives `--agent`.
- **`scripts/chat-terminal-smoke.sh` is not invoked by CI** — the only reference outside `qa/` is
  `DECISIONS.md:228`. It is a 2-case, non-agent smoke (`CAMELID_CHAT_SUPPORTED_GGUF` /
  `CAMELID_CHAT_UNSUPPORTED_GGUF`, both skip when unset). There is no case table; a new case is a
  hand-written `if [[ -n "${CAMELID_CHAT_<X>:-}" ]]` block plus a header declaration.
- **No `tests/` integration test covers the agent loop.** All 91 passing `chat::` tests are inline
  `#[cfg(test)]` modules.

---

## 7. Gate G0 — SIGNED (Tim, 2026-07-22)

All eight items ruled. The four load-bearing ones were put to Tim directly; the remaining four had
unambiguous evidence and were accepted as recommended.

1. **Phase order — RULED: `P0.5 → G1 → G2 → G7 → G3 → G8 → G4 → G5 → G6`.** A new pre-phase P0.5 is
   authorized, and G7 is promoted ahead of G3/G8 per the `&'static str` constraint (§5.1).
2. **Project file name — `CAMELID.md` primary, `AGENTS.md` fallback.** As written; no collision.
3. **Headless verb — `camelid agent exec "<goal>"`.** As written; free, and the `gait` nested shape
   exists to copy (§6).
4. **Checkpoint substrate — RULED: snapshot-only.** Content snapshots always; the git-stash variant
   is dropped. Rationale: one code path to prove against the jail check, and the agent never mutates
   git state the sandbox does not own.
5. **MCP config — RULED: `camelid.mcp.json` at the sandbox root, stdio-only, v1 reads nothing else.**
   Importing a Claude/Codex config would pull in third-party server declarations the user never
   placed in this workspace, against the opt-in posture.
6. **CERT pin — `qwen3_4b_instruct_q8_0`** at `/Volumes/Untitled/models/Qwen3-4B-Q8_0.gguf`;
   secondary `llama32_3b_instruct_q8_0` (§4.4). Host is the **Mac mini M4**, not the RTX 3060 box.
7. **G1 vs. the promotion basis — RULED: `agent-eval` is EXCLUDED from project-instruction
   injection.** `agent_eval.rs:290` keeps feeding the baseline prompt so the five existing receipts
   remain valid attestations. Recorded as **D-DROVER-6** (§9.1).
8. **G2's budget — re-pinned to 80% of the row's *validated envelope*:** 6554 tokens, compaction
   target 4096, on the recommended row (§4.3). Not the model's `n_ctx`.

### 7.1 P0.5 — authorized scope

Four independent concerns, one commit each, all HARNESS-provable with no model:

1. Frame tool results as untrusted data in the transcript (`agent.rs:630`) — §4.1.
2. Add the three missing regression pins: system-prompt shape, advertised-tool-name set, and
   slash-command parity across the two front ends — §4.7.
3. Keep agent scratch state out of the workspace and out of its own search index: `.camelid/` into
   `.gitignore` and into the `search` skip list (`tools.rs:1220`) — §4.8.
4. Make a new `LoopEnd` variant a compile error in the subagent status map (`subagent.rs:663`) — §4.6.

**Carry-over risk on P0.5(1):** the wrapper changes tool-result framing on *every* lane, including
`agent-eval`. This is the same coupling as §4.2 but weaker — it changes the *observation* text, not
the instruction that elicits tool use. Mitigation: after landing, re-run `agent-eval` on the primary
CERT row and refresh that receipt, converting a staleness risk into current evidence.

### 7.2 P0.5 — delivered

| Commit | Concern | Tests |
|---|---|---|
| `7d718ff` | Fence tool results as untrusted data; system prompt names the markers | +5 |
| `9d62750` | Pin the system prompt shape, the advertised tool set, and the slash-command table | +7 |
| `6085645` | `.camelid/` out of `.gitignore` and out of the `search` index | +1 |
| `59d2996` | `LoopEnd` catch-all → exhaustive match (behavior unchanged) | 0 |

`chat::` tests **91 → 103**. Baseline gates green throughout (`fmt`, `clippy --all-targets
-D warnings`, `test`).

### 7.3 Phase delivery

| Gate | Commit | `chat::` |
|---|---|---|
| P0.5 | `7d718ff` `9d62750` `6085645` `59d2996` | 91 → 103 |
| G1 project context + prompt | `28501d5` | 103 → 112 |
| G2 compaction | `974f0c8` | 112 → 120 |
| G7 MCP (+ `a2908ac` retype prep) | `c88a82b` | 120 → 130 |
| G3 plan surface | `521cf68` | 130 → 135 |
| G8 web search + slash commands | `4e2c349` | 135 → 142 |
| G4 headless `agent exec` | `71776ad` | 142 → 145 |
| G5 checkpoint / diff / undo | `2cdf0c5` | 145 → 152 |
| G6 session save / resume | `44cf547` | 152 → 159 |

**All nine phases delivered.** `chat::` tests 91 → 159.

### 7.4 CERT status

- **G2 (`camelid.agent-context-receipt/v1`)** — not minted. Compaction is proven by HARNESS
  (forced-overflow through a real `run_loop`), and the live `agent exec` run exercised the loop
  end-to-end, but no long-task run crossed a compaction boundary on a real model. **INCONCLUSIVE**
  under the campaign rule, never FAIL.
- **G7 (`camelid.agent-mcp-receipt/v1`)** — not minted. HARNESS covers the full path against a stub
  stdio server (namespacing, tier, approval routing, untrusted output, timeouts). A real-model
  round-trip through an MCP tool was not run. **INCONCLUSIVE.**
- **Promotion receipt refreshed** — `qa/agent-eval/Qwen3-4B-Q8_0-1784747759-PASS.json`. P0.5 changed
  tool-result framing on every lane including `agent-eval`, so the committed receipt attested a
  prompt that no longer ships. Re-run: **PASS on the full 3-case battery**, where the previous
  receipt covered 1 case. Strictly stronger evidence than it replaces. An earlier attempt returned
  INCONCLUSIVE on a contended box and correctly changed no flag.
- **`agent exec` verified live** on the pinned row: read a workspace file, answered from it, exit 0.
  `CAMELID_PRODUCTION=1 … --yolo` exits 1 with the refusal.

### 7.5 Deliberately not done

- **`/compact` and `/model`** (G8). The transcript is created per goal *inside* the loop, so a
  REPL-level `/compact` has nothing to compact between goals; `/model` needs mid-session model
  switching plus a fresh `tool_capable` re-check. Both are real work, not wiring.
- **A2's exec-tier hole** (`agent_syscap.rs:198`) is still open — a second `action.execute()` with no
  approver, no audit and no production check. `update_plan` and `web_search` are now reachable from
  it. Neither is dangerous there (`update_plan` has no side effects; `web_search` needs
  `sandbox.allow_net`, which that harness does not set), but the hole itself should be closed.
- **A18** — `LoopEnd::StepCapped` maps to subagent exit **1**, not **3**. G4 defined the tri-state for
  `exec` but did not retrofit `subagent.rs`, so the two disagree. Worth reconciling.

The slash-command work went further than "add a pin" because the divergence the critic predicted was
real: `/sidebar` was undocumented in **both** front ends and the inline `/help` omitted itself. Both
help surfaces now derive from one shared `SLASH_COMMANDS` table, so the pin has something true to
pin. `/theme` and `/sidebar` remain deliberately TUI-only and the test asserts exactly that pair.

---

## 8. Amendment log

Discoveries that correct the campaign brief. Recorded at G0; each later phase appends here in the
same commit that acts on the finding.

| # | Brief said | Repo says |
|---|---|---|
| A1 | New surfaces route output "through the same untrusted framing the file/shell tools already use" | **No such framing exists.** Tool output reaches the model verbatim (`agent.rs:629-631`). DROVER must build it. §4.1 |
| A2 | Compaction budget = 80% of the model's training `ctx` | The binding ceiling is the row's **validated envelope (8192)**, not `n_ctx`. §4.3 |
| A3 | The loop is "plan → validate → approve → execute → observe" | There is **no plan phase**; "plan-act-observe" is a doc comment only. §3 |
| A4 | Standing suite per `CONTRIBUTING.md` (`--all-features`) | On macOS CI uses `features: ""`; `--all-features` breaks the cuda cfg. §1 |
| A5 | CERT hosts "RTX 3060 Laptop and Mac mini M4" | Only 2 of 5 `tool_capable` rows have usable local weights **here**; `qwen3_4b_q4_k_m` has none anywhere on this box. §4.4 |
| A6 | `agent-eval` emits a hashed receipt (`DECISIONS.md:454-455`) | **Not hashed.** Only syscap/orchestration/bench are tamper-evident. §4.8 |
| A7 | Receipts are the evidence bar | **No shipped tool can verify an agent-family receipt** — `verify-receipt` rejects non-parity schemas; `verify_self_digest` is `debug_assert!`-only. §4.8 |
| A8 | The `tool_capable` gate protects agent mode | It is **entry-only**. `agent-eval`, `__subagent`, `agent-orchestration` all build real loops with no capability check. §2.3 |
| A9 | (unstated) | A **second, ungated tool executor** exists at `agent_syscap.rs:198` — no approver, no audit, no production check. §4.5 |
| A10 | (unstated) | `agent_orchestration.rs:341-350` `family_for` is **missing the ornith arm** its twin at `agent_eval.rs:424-442` has. Real bug in a gate harness. §4.8 |
| A11 | Sandbox-root flag | It is `--workdir` (`main.rs:381-382`). §6 |
| A12 | DECISIONS numbering | Root sequence is **non-monotonic** (D5 after D6/D7; D6/D7/D11 duplicated); last plain number is **D17**. `DOCS.md:36` links the *stale* decision log. §6.3 |
| A13 | Extend `scripts/chat-terminal-smoke.sh` and "the agent smoke lane" | **There is no agent smoke lane**, and the chat smoke **is not run by CI**. §6.4 |
| A14 | The TUI agent is "a documented follow-up" (`agent.rs:7-9`, `DECISIONS.md:395`) | It shipped and is the **default** (`mod.rs:169-170`). `RECON_AGENT.md:41`'s `/agent` toggle does not exist. §4.8 |
| A15 | (found during P0.5) | The **orchestration tools are gated on `subagent::is_enabled()`**, which is false until a subagent config is installed. An ordinary `chat --agent` session is never offered `spawn_subagent` / `check_subagent_status`, and `inspect_system` is Windows-only — so the cross-platform baseline tool set is **five** tools, not the nine a reading of `specs()` suggests. Now pinned by `advertised_tool_set_is_pinned`. |
| A16 | (found during P0.5) | `Sandbox::new(root, allow_net, timeout)`'s second parameter is **`allow_net`, not `fs_unrestricted`** — the latter is a builder (`with_fs_unrestricted`). Easy to misread when adding a sandboxed surface. |
| A17 | (found during P0.5) | The `search` tool's parameter is **`pattern`**, not `query` (`tools.rs:350`). |
| A19 | (found during G2) | Retaining N recent messages verbatim **does not bound the transcript**: one `read_file` may return 64 KiB (`tools.rs:216`), more than the whole 8192-token budget, so a tail of recent results can exceed it with nothing old left to elide. Compaction needs a second pass that clips oversized retained results in place. Found by the test, not by inspection. |
| A20 | (found during G7) | A process that is not an MCP server **can still pass the handshake**: `cat` echoes the request back with a matching id, so `initialize` returns `Ok(null)`. Safe outcome (zero tools adopted, MCP stays disabled) but it means a successful handshake is not evidence of a real server. Pinned by `an_echo_server_adopts_no_tools`. |
| A21 | (found during G7) | The MCP registry is process-wide, so its tests race the `tools.rs` tool-set pins under cargo's parallel runner — a pin can observe MCP tools appearing mid-assertion. Serialized by a shared test mutex. Any future process-global agent state inherits this hazard. |
| A18 | (open question, deferred to G4) | `LoopEnd::StepCapped` maps to subagent status `"failed"` → exit **1**, not `"inconclusive"` → exit **3**. A step-capped run arguably did not fail. Left unchanged in P0.5 because it decides an exit code on a shipped gate lane; **G4 owns this call** when it defines the tri-state contract. |

---

## 9. Decision records

### 9.1 Draft entries for `DECISIONS.md`

To be numbered into the live root `DECISIONS.md` on acceptance. Per §6.3 the namespaced form does
not consume a plain integer, so these nest as bullets under a single `## D18 — DROVER` entry,
matching the D17/D-B1..5 precedent.

- **D-DROVER-1 — Compaction retains the safety spine.** The system prompt and the data-not-commands
  rule are never compacted; summaries of untrusted output remain untrusted. The budget is 80% of the
  row's *validated context envelope*, not the model's `n_ctx` (§4.3).
- **D-DROVER-2 — `agent exec` is a subcommand, not a binary.** Tri-state exit 0/1/3; non-interactive
  approver; `--yolo` refused under `CAMELID_PRODUCTION`.
- **D-DROVER-3 — MCP is opt-in, Confirm-tier, production-off, namespaced, output-untrusted**, stdio
  only, declared solely by a workspace-local `camelid.mcp.json`.
- **D-DROVER-4 — Resume re-validates model identity** and never re-executes historical tool calls.
- **D-DROVER-5 — Checkpoint and session stores live under the sandbox root**, pass the jail check,
  and are content snapshots only — never a git mutation (G0 item 4).
- **D-DROVER-6 — The promotion harness attests the baseline prompt.** `agent-eval` is excluded from
  project-instruction injection so `tool_capable` receipts keep attesting a fixed, reproducible
  prompt rather than whatever `CAMELID.md` a workspace happens to carry (G0 item 7, §4.2).

---

## 10. What is NOT true — myths this recon retires

- *"The loop needs rebuilding."* It does not. `run_loop` is complete, cancellable, step-capped, and
  result-aware. Every phase wires into it.
- *"`specs()` is a static table so MCP is hard."* It is a runtime `Vec` builder (`tools.rs:324`). The
  hard part is `&'static str` in `ToolSpec` (§5.1), which is a typing change, not an architecture one.
- *"The sandbox has a prefix-matching hole."* It does not — `Path::starts_with` is component-wise
  (`tools.rs:294`). The real hole is the unresolved final component on the `must_exist=false` path
  (§4.8).
- *"G1/G3/G8 will break snapshot tests."* There are no snapshot tests at all (§4.7). The risk is the
  opposite: those phases ship unguarded unless P0.5 adds the pins.
