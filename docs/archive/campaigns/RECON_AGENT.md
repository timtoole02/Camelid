# RECON_AGENT.md — Phase 0 working note for `camelid chat` agent mode

From the actual repo (`/Volumes/Untitled/Camelid-push`, `main` @ `bf16a5d`). Where spec and
repo disagree, the repo wins and the discrepancy is recorded here.

## 0.1 Tool-call support in the inference path — **NOT supported today (fail-closed)**

- `/v1/chat/completions` accepts `ChatCompletionRequest` (`src/api/mod.rs:378`): model, messages,
  stream, max_tokens, temperature, top_k/top_p, seed, penalties, logit_bias, stop, n, logprobs,
  camelid_* extras. **No `tools` / `tool_choice` field.**
- The request-field guard **explicitly rejects** `tools`, `tool_choice`, `parallel_tool_calls`,
  `parse_tool_calls` (`:5839`): *"tool/function calling is not supported by Camelid generation
  routes yet."* So tool-calling must be **added** — this is the substantial Phase 1.
- **However**, the prompt is rendered through the model's own **Jinja chat template via
  minijinja**: `render_jinja_chat_template` (`:7829`) → `compiled.render(context!{…})` (`:7849`),
  with a per-model template env cache (`JINJA_CHAT_TEMPLATE_ENV_CACHE`, `:77`). The render
  `context!` passes `messages` + `add_generation_prompt` + `bos/eos` today, **not** `tools`.
  The repo even carries the real **Llama 3.x template with `{%- set tools = custom_tools %}` /
  `{%- for t in tools %}`** logic (`LLAMA3_METADATA_FULL_TEMPLATE`, `:10449`, currently a test
  fixture). So a tool-capable model's *own template already knows how to render tools* — enabling
  rendering is "thread `tools`/`custom_tools` into the `context!`," not authoring a template.
- **Output parsing is the model-family-specific part:** Llama 3.2 emits tool calls as JSON
  (`{"name":…,"parameters":…}`) / `<|python_tag|>`; Qwen3/Hermes emit `<tool_call>…</tool_call>`.
  Parsing the generated text into `(name, args)` is per-family and is NOT in the repo yet.
- `ChatMessage` is `{ role, content }` (`:444`); roles handled are system/user/assistant. A
  tool-result message (`role:"tool"`/`"ipython"`) renders fine through the Llama/Qwen templates,
  but assistant messages have **no `tool_calls` field** — a client-parsed-from-text loop sidesteps
  that; a fully OpenAI-shaped server response would add it.

→ **Phase 1 is substantial** (touches the carefully-gated chat handler + the fail-closed
unsupported-field guard + family output parsing). Per the working method, this is a scope boundary
to confirm before building. See "Phase 1 options" below.

## 0.2 Where chat mode lives

`src/chat/` (bin-local): `mod.rs` (entry/dispatch), `session.rs` (UI-agnostic core),
`inline.rs` (line REPL), `tui.rs` (full-screen), `client.rs` (HTTP/SSE), `server.rs`, `models.rs`,
`palette.rs`, `markdown.rs`, `theme.rs`, `clipboard.rs`, `banner.rs`. The agent loop is a new
`agent.rs` module driven from the session core, reusing `client.chat_stream`/`chat_blocking`, the
splash/turn-marker (`banner.rs`), and the color/TTY helpers — beside the plain turn loop, not a
fork of it. Entry via `--agent` flag + in-session `/agent` toggle (Phase 6).

## 0.3 BLOCKING DECISION A — ledger `tool_capable` — **PROGRAMMATIC, tractable**

`ModelCompatibilityTarget` (`src/api/mod.rs:~300`) is a struct of `&'static str` fields surfaced by
`/api/capabilities` (`capabilities_response_with_plan`). Add **`tool_capable: bool`** (default
false) per row, populated once, and read by the agent gate + a future frontend from the same
source — no second parser. NOT prose-only → the "stop if prose-only" boundary does not trigger.
Initial population: `false` for every current row **unless** we also promote a verified row in
this work (e.g. Llama 3.2 3B Instruct Q8_0) — promotion requires a real tool-call round-trip as
evidence, exactly like the support gate. Until then agent mode is built+tested but refuses every
supported row (honest, per the capability tension).

## 0.4 BLOCKING DECISION B — sandbox root

**Root = cwd at launch, overridable with `--workdir <path>`.** Resolved once via
`std::fs::canonicalize` to an absolute, symlink-free root. Every file-tool path is: joined to root
(or taken as-is if absolute), `canonicalize`d, then required to be the root or a descendant
(canonical-prefix check). This rejects `..` traversal, absolute paths outside root, and escaping
symlinks (canonicalize resolves them before the check). For a not-yet-existing write target, the
**parent** dir is canonicalized + checked, then the final component re-appended. Enforced in code
before any I/O — not a prompt instruction (constraint 5).

## 0.5 BLOCKING DECISION C — shell execution

`run_shell(command)` runs **`/bin/sh -c <command>`** with `cwd` pinned to the sandbox root, a
**timeout** (default 30s, `--shell-timeout`), captured stdout/stderr/exit code (truncated for
display), and **no auto-interpolation** — the model supplies the whole command string and the
**verbatim string is shown at approval** (rendered from the parsed tool call, not model prose), so
"what you approve is what runs." Direct-argv (no shell) was considered but rejected: an agent
shell tool needs pipes/globs to be useful; the protections are **mandatory approval + verbatim
display + sandbox cwd + timeout + exec never auto-approved by default**, not argv-splitting.

**Honest limitation (recorded):** a path-sandbox truly confines the *file* tools, but `/bin/sh`
runs with the user's full permissions and can `cd` out of the root — `run_shell` is confined to
the root as *cwd*, not as a filesystem jail. Genuine jailing needs OS sandboxing (`sandbox-exec`
on macOS / namespaces on Linux) and is a documented hardening follow-up, not claimed here. This is
why exec is the highest risk class, always prompts, and is never in an auto-approve default.

## Phase 1 options (the confirmation point)

- **(A) Server-side, fully aligned:** add `tools` to `ChatCompletionRequest`, stop rejecting it,
  thread into the Jinja `context!`, render `role:"tool"` messages, parse family output into
  OpenAI `tool_calls` in the response. Most aligned; biggest change to the gated handler.
- **(B) Client-side agent loop:** the loop renders the tool system-prompt in the model family's
  documented format and parses the model's *text* output for calls; the existing chat path is
  unchanged. Smallest, isolated to `src/chat`; the API itself stays tool-unaware (mild divergence
  the spec cautions against, though it uses the real model format).
- **(C) Hybrid (recommended):** a *small, aligned* server change — accept `tools` and thread it
  into the existing template render (reuses the model's own tool template) + allow `role:"tool"`;
  the **agent loop does the family-specific output parsing**. Keeps rendering template-driven and
  aligned, keeps parsing isolated and iterable, avoids rewriting the response path.

Independent of the option: the **whole deterministic core** (agent loop + tool set + sandbox +
approval gate + transcript rendering + a test-only mock model + every test + the `tool_capable`
gate) can be built and proven **without** real-model tool-calling — exactly as the working method
recommends ("build the loop against the mock first"). Phase 1 only gates the *live-model demo*.
