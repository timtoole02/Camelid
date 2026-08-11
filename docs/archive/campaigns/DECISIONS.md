# DECISIONS.md — distributed parity lane

Authoritative record of binding decisions for the distributed parity lane. Each entry is
dated and justified. Companion to `DISTRIBUTED_RECON.md`.

## D1 — Topology: pipeline parallelism by contiguous layer block (2026-06-13)

Cut the decoder layer stack into contiguous blocks; each node loads only its block's
weights and owns only its block's KV. One hidden-state vector walks the stack node→node,
one network hop per block per token. Coordinator holds embeddings-in + final-norm/output-
projection/sampling-out and lives on the fastest node; shards hold a contiguous layer block
and its KV only.

**Wire:** raw little-endian f32 activations, row-major, length-prefixed framed TCP,
synchronous request/response per hop per token. The scalar absolute position travels with
the activation (needed for RoPE and the KV write offset; positions are not reconstructable
on a shard that skipped earlier layers).

**Why this and not tensor parallelism:** the goal is to make memory add up while keeping the
math exactly sequential — sequential math is what makes token-identity (the gate) provable.
Tensor parallelism splits within each layer, forcing a network sync per layer (latency
multiplier) and a far larger numeric-divergence surface. **Tensor parallelism is rejected
for this lane.**

**Split ratio by RAM-fit, not speed.** A ~10× Mac↔Pi compute gap cannot be balanced; do not
try. Mac gets the large block (+coordinator); each Pi gets a contiguous tail block sized
under its 8 GB ceiling.

This decision matches what the repo already implements (`distribute-master`/
`distribute-worker`, `src/cluster.rs` framed activation protocol, `forward_layer_range_from_hidden`).
It is recorded here as binding, not as new construction.

## D2 — Naming: use `camelid` / `CAMELID_*`, NOT `backendinference` (2026-06-13)

The spec's operating rule #2 says "names stay on `backendinference`" and references
`BACKENDINFERENCE_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`. **This is stale.** The actual,
current package identifiers are:

- crate/binary: `camelid` (`Cargo.toml:2`), subcommands under `camelid …`.
- env var: `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` (`src/api/mod.rs:57`).
- `backendinference` / `BACKENDINFERENCE_*` is **legacy branding that the repo's own public
  scrub CI actively forbids** (`scripts/check-public-scrub.sh:46-48`, branding_pattern
  `backendinference|BackendInference|backend inference`).

Following rule #2 literally would reintroduce branding that CI rejects and that an earlier
rename deliberately removed. The **intent** of rule #2 — "keep current package identifiers,
no rename in this lane" — is satisfied by using `camelid` / `CAMELID_*`, and is *violated*
by introducing `backendinference`. **Decision: use `camelid` / `CAMELID_*` everywhere.** The
memory-cap discipline the spec invokes is enforced via
`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`.

**Addendum (2026-07-10):** the last eight live `BACKENDINFERENCE_*` env vars — perf-lane
knobs introduced by the attention/KV/decode-scheduler campaigns (`DECODE_TIMINGS`,
`DECODE_THREADS`, `DECODE_POOL_DEDICATED`, `ATTENTION_F32_BLOCKED_DOT`,
`ATTENTION_DECODE_PARALLEL`, `ATTENTION_DECODE_PARALLEL_MIN_POSITIONS`, `KV_F16`,
`KV_LAYOUT_HEAD_MAJOR`) — are renamed to `CAMELID_*` as a **clean break** (operator
decision: no fallback reads; the flags were undocumented and never appeared in release
notes). Defaults and behavior are unchanged under the new names. Historical receipts,
archived docs, and this file's older entries keep the old names verbatim. The public-scrub
branding guard now also scans `src/` so legacy-prefixed identifiers cannot reappear in live
code.

## D3 — Reuse existing infrastructure; the lane's deliverable is the receipt, not plumbing (2026-06-13)

Recon (see `DISTRIBUTED_RECON.md`) found the transport, shard servers, coordinator,
per-token pipeline, and layer-range forward already exist and are tested. Gemma 4 already
has a passing distributed parity gate vs a llama.cpp oracle. The genuinely missing work is:

1. A bitwise in-process chained-partition parity test for **Llama** (Phase 1), with the
   execution lane pinned (see D4).
2. A distributed parity **receipt** for the Llama path in the spec's artifact schema,
   built on the existing `camelid.parity-receipt/v1` framework (`src/receipt/`), not a new
   one (Phases 2–4).
3. The cluster frontend tab with a standing experimental banner (Phase 5).

**Decision: do not re-implement working code.** Retrofit parity + receipts onto the
existing Llama distributed path and lift the generic `cluster.rs` protocol (add the
versioning + FNV checksum hygiene Gemma 4 already proved) rather than inventing a parallel
stack. Pending user confirmation of this re-scope (the spec was written as if greenfield).

## D4 — Pin the parity reference lane (2026-06-13, RESOLVED at Phase 1 gate)

`forward_layer_range_from_hidden` routes `seq_len==1` decode to the GPU-resident path
(`src/inference.rs:2110-2117`), which is a different implementation from the CPU chunk path.
Token-identity must be asserted against a **single, named** reference lane, not "whatever
the engine happens to pick." **Decision (provisional): the parity reference is single-node
`camelid` greedy (temp 0, seed 0) on the same exact GGUF, with the distributed run forced
onto the same execution lane (CPU vs resident) as the reference.** Finalize at the Phase 1
gate once the in-process split test exists. Never paper over a divergence with a tolerance
window (operating rule #6); a single differing f32 bit is a finding to document, not smooth.

**RESOLVED:** the reference lane is the **CPU chunk path** (`forward_layer_range_from_hidden`
with resident paths disabled via `set_resident_paths_disabled(true)`), driven identically by
both the single-node reference and the split so the test isolates loop-cut state leaks from
kernel differences. Phase 1 gate **PASSED** 2026-06-13: `tests/distributed_llama_parity.rs`
proves 8 greedy steps (prefill + 7 decode) bitwise-identical between full `[0,22)` and chained
`[0,11)+[11,22)` on TinyLlama 1.1B Chat Q8_0 (`fmt`/`clippy -D warnings` clean; test-only, so
the single-node 50-token gate is unchanged). Open follow-up: a separate gate may later validate
the GPU-resident lane across a split, but the distributed pipeline's parity reference stays the
CPU lane.

## D6 — Phase 3 driver: a same-binary `parity_node` example (2026-06-13, PASS)

Phase 3 (two Macs over LAN) runs through `examples/parity_node.rs`, a single binary with
`worker` and `coordinator` modes. Rationale: the existing `distribute-master`/`-worker`
CLI does not pin the CPU lane or emit a `DistributedParityReceipt`, both of which the gate
requires. The example reuses the library's public session API, `src/cluster.rs` wire, and
`src/receipt`. **The same binary must run on every node** — a different build would defeat
the parity claim — so it is built once and copied to each node.

Transport hardening (not parity relaxation): the worker resets its KV cache per connection
(a persistent worker serves many runs); the coordinator resets its KV per run, uses bounded
connect-retry + whole-run retry, and the worker survives a single bad run. mini2's worker
runs under `caffeinate` (App Nap / Wi-Fi power-save otherwise stalled `accept()`).

**Result PASS 2026-06-13:** two consecutive runs token-identical, mac-m4 `[0,11)` → mini2
`[11,22)`, TinyLlama 1.1B Q8_0 (byte-identical GGUF, sha `a4c9bb1d…`); deterministic
`receipt_id 33b79d8d…`; artifacts `qa/distributed/{two-mac-tinyllama-q8.json,
cluster-topology.json}` + `CLUSTER_BENCH.md` (honest latency, ~5.5 tok/s, capability not
speed).

## D7 — Phase 4 (heterogeneous Mac+Pi): PORTABILITY SOLVED, Pi inference BLOCKED (2026-06-13)

Reported, not faked (operating rule #5). Two distinct results:

**RESOLVED — camelid is portable to aarch64-Linux (the spec's assumed hard blocker).**
- All 3 Pis found: `camelid1`/`camelid2`/`camelid3` on the local LAN (IPs redacted) — Pi 5, 16 GB,
  aarch64 Linux (kernel 6.12 rpi-2712), running NanoCamelid. Operator SSH key path redacted,
  user `tooleman`. (An older ssh-config LAN entry now points at a different device.)
- `build.rs` portable (x86 AMX gated to linux+x86_64; Accelerate macOS; Metal cfg(macos)).
  No rust cross-target locally (Homebrew, no rustup), so built **natively on camelid1**:
  installed rustup (1.96), rsynced source, `cargo build --release --example parity_node`
  succeeded in **2m10s** with zero errors. The Linux binary runs (prints usage; clean `ldd`,
  no Metal/Accelerate).

**RESOLVED — the Pi runs camelid inference; the earlier "crash" was an SSH self-kill.** The
launch failures were `pkill -f parity_node` matching the *SSH command's own cmdline* (which
contains "parity_node") and killing its own session before the launch ran — not a code crash.
Launching without that pkill, the Pi worker loads and listens fine (TinyLlama [11,22) ~2.2 GB
f32 est; llama-2-13b [20,40) 26 GB f32 est / ~6.5 GB Q8 resident, within a 40 GB cap).

**RESULT — cross-ARM parity is EXPERIMENTAL-DIVERGENT (a real finding, recorded).** Run
`hetero-mac-pi-tinyllama-q8`: Mac (Apple) coordinator [0,11) + reference, Pi (Cortex) worker
[11,22) + head, TinyLlama Q8_0 (byte-identical GGUF). The heterogeneous output is
**token-identical to the all-Apple single-node reference for the first 24 generated tokens,
then diverges at generated token 25** (`first_divergent_generated_token_index=25`,
`generated_tokens_match=false`). Cross-platform IEEE-754 differences (FMA contraction /
accumulation order, Apple vs ARM Cortex) compound until a greedy argmax flips. Sealed receipt
`qa/distributed/hetero-mac-pi-tinyllama-q8.json` (`receipt_id 11bbe0e1…`, `validated:false`) —
documented, NOT hidden behind a tolerance (operating rule #6). The lane stays **experimental**
for heterogeneous Apple+Cortex configs. parity_node now emits the receipt on divergence
instead of aborting.

**Too-big (llama-2-13b) capability evidence + reference gap.** Each shard loads only its slice
(Pi [20,40) ~6.5 GB Q8 within a 40 GB f32-est cap; full model's 52 GB f32 est would exceed it
→ "fits no single capped node"). But the **single-node reference OOM'd the 16 GB Mac** loading
the full 13 GB model — which is itself evidence the full model exceeds one 16 GB node. So a
same-engine reference can't be computed on-device for 13b; completing the 13b parity verdict
needs the **llama.cpp oracle** reference (spec-sanctioned for models that fit nowhere whole) —
a documented next step (add an oracle/external-reference mode to parity_node).

Phase 4 prep landed: per-node materialization cap (`--max-weight-bytes` /
`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`) with a typed refuse-to-start (verified:
TinyLlama [11,22) ~2.2 GB refuses at cap=1000).

## D5 — Branch / naming for the lane (2026-06-13)

Work proceeds on `feat/distributed-parity-lane` off `origin/main`. Single-node TinyLlama
1.1B Chat Q8_0 gate and the full validation suite (`fmt`/`clippy -D warnings`/`test`/`doc`)
must stay green before every commit (operating rule #3).

## D6 — `camelid chat` terminal mode (2026-06-13)

Work proceeds on `feat/terminal-chat` off `origin/main`. Full validation suite
(`fmt`/`clippy -D warnings`/`test`/`doc`) stays green before every commit (operating rule #3).
Recon detail backing these decisions: `RECON_CHAT.md`.

**Decision A — compatibility source: PROGRAMMATIC.** The supported set is structured data at
`GET /api/capabilities` → `CapabilitiesResponse.model_compatibility: Vec<ModelCompatibilityTarget>`
(and the Rust `capabilities_response_with_plan`). The picker filters rows by
`status.starts_with("supported")` at runtime — no hardcoded list, no second prose parser of
`COMPATIBILITY.md`. The "prose-only → stop for confirmation" boundary did not trigger. The
capabilities ledger (supported rows) and `curated_catalog()` (8 pullable rows) are two lists
joined on `model_compatibility.id == CatalogItem.catalog_id`; supported rows with no catalog
entry render `[supported · no pull alias]`.

**Decision B — architecture: Option B (HTTP client over a child-process `serve`).** Justification:
the engine already ships an audited OpenAI-compatible SSE lane at `/v1/chat/completions`; driving
it over HTTP makes terminal output provably identical to the validated lane (constraints 2/4/5)
and avoids Option A's re-plumbed streaming + token-for-token parity-test burden. The usual cost
of Option B (an HTTP client) is **zero here**: the repo has no HTTP-client dep but already
hand-rolls a blocking HTTP/1.1 client over `std::net::TcpStream` in `src/receipt/verify.rs`
(`http_json`/`parse_http_response`); `chat` reuses that pattern and extends it with a
line-buffered `data:` SSE reader. Control-plane actions reuse pub Rust in-process —
`catalog::run_pull`/`curated_catalog()` for pull, a `models/<filename>` fs check for availability;
only load/capabilities/health/generation go over HTTP. The `chat` command body is fully
synchronous (blocking client + blocking line editor + `std::process::Child`); no tokio in the
chat path.

- **Spawn-vs-attach:** probe `GET /v1/health` on `--addr` (default `127.0.0.1:8181`); attach if
  it answers, else spawn `camelid serve --addr <addr> --no-open` as a child and poll health until
  ready. Tear the child down on exit **only if we spawned it** (an attached server is left alone).

**History reset on model switch.** Switching the active model mid-session clears conversation
history (a different model = a different context window; carrying history is a footgun) and prints
a one-line notice. The `--system`/`/system` system prompt is **re-applied** across a switch
(it is a session-level instruction, not model-specific context).

**Constraint #4 gate (repo wins).** The backend's only generation gate is the typed
architecture error (`unsupported_runtime`/`generation_ready=false`). A recognized-arch
non-ledger GGUF would generate, same as the engine + React frontend today. `chat` reuses that
typed gate verbatim and does not invent new per-file matching (which would violate "reuse, don't
reimplement"); the picker's supported-only selection keeps the normal path from reaching a
non-supported row. Full rationale in `RECON_CHAT.md` §8.

**Line editor dependency: `rustyline`.** Chosen over `reedline` for a smaller, mature, sync API
(line editing + in-session history) that fits the synchronous chat loop with no async glue. It is
the single new dependency this feature adds.

**Ctrl-C semantics.** A SIGINT handler flips a cancel flag. During `rustyline` readline the
terminal is raw, so Ctrl-C arrives as a byte → `Interrupted` → an idle hint, not a quit. During
a stream the terminal is cooked, so Ctrl-C raises SIGINT → the read loop aborts cleanly and the
**entire in-flight turn (the user message and the partial reply) is discarded**, so history stays
coherent. Ctrl-D / `/exit` quit cleanly; a spawned server is torn down by `ServerHandle`'s Drop.

**Phase 8 evidence.** SSE/de-chunk parser and ledger-derived picker are covered by bin unit
tests (`cargo test --bin camelid chat::`). The supported-path and unsupported-gate end-to-end
checks live in `scripts/chat-terminal-smoke.sh`, gated on `CAMELID_CHAT_SUPPORTED_GGUF` /
`CAMELID_CHAT_UNSUPPORTED_GGUF` (no-op when unset, like the parity scripts) so they never block
`cargo test`.

## D7 — `camelid chat` full-screen TUI (2026-06-13)

`chat` grew a full-screen ratatui front end (the default on an interactive terminal). Both UIs
share one UI-agnostic `session::Session` core (state, sampling settings, request shape, save/load
— no I/O); `inline.rs` is the line REPL (now also the `--plain`/non-TTY fallback that the smoke
scripts drive) and `tui.rs` is the ratatui app.

- **Concurrency:** the redraw loop runs on the main thread; each generation streams on a
  background thread that forwards deltas over an `mpsc` channel, polled non-blocking each frame.
  Ctrl-C (a key event in raw mode) flips the shared `session::CANCEL` flag → the worker aborts.
- **Reuse intact:** the worker calls the same `Client::chat_stream` SSE lane; `Client` is now
  `Clone` (just a `SocketAddr`). No second generation path.
- **Deps:** `ratatui = "0.29"` (+ its `crossterm`). Only network/HTTP stays hand-rolled.
- **Downloads in the TUI:** selecting a not-downloaded picker row (or `/pull`) suspends the
  alt-screen, runs the existing `catalog::run_pull` with visible `curl` progress, then re-enters
  — so the audited pull path is reused and progress isn't swallowed by the alt-screen.
- **Front-end choice:** TUI when stdin+stdout are both TTYs and `--plain` is unset; else inline.
  The `--model` unsupported gate runs before either UI (typed error + exit 1, no screen takeover).
- **New options:** `/set` (temperature, top_p, top_k, max_tokens, seed, stream), `/system`,
  `/reset`, `/retry`, `/save`/`/load` (session JSON), `/info`, `/tokens`, `/pull`, plus CLI
  `--top-p/--top-k/--seed/--plain`. Mascot redesigned to the dromedary line-art.

## D8 — `camelid chat` production feature pass (2026-06-14)

Added on top of the TUI (D7), all sharing the `session::Session` core and the same audited
HTTP/SSE lane:

- **Slash command palette** (`palette.rs`): typing `/` opens a filtering popup over the input
  (prefix-ranked, ↑↓ select, Tab complete, Enter run). One static `COMMANDS` registry is the
  single source of truth for the palette, `/help`, and dispatch (alias-aware) in both front ends.
- **Instant loaded-model switching** — **zero backend change**: `get_or_load_model` already
  activates an already-loaded model by the request's `model` id with no reload, and `GET
  /v1/models` lists every loaded model (id + ctx/params/size). `/switch` and the model browser's
  "● loaded (instant)" section just send the chosen id; `/model <id>` prefers a loaded model.
- **Markdown rendering** (`markdown.rs`, dependency-free): fenced code blocks, headings, bullets,
  block quotes, inline `code`/**bold**/*italic*, with style-aware width wrapping. Assistant turns
  render as Markdown; user turns stay plain.
- **Themes** (`theme.rs`): sandstorm/mono/ocean/nord, `/theme [name]` cycles/picks; every widget
  + the markdown renderer pull styles from the active theme.
- **Live streaming** spinner + tok/s in the status bar (frame-driven), a **context gauge** in the
  sidebar (last-turn tokens vs the model's `n_ctx_train`), and **`/copy`** to the clipboard via an
  OSC 52 escape with a hand-rolled base64 (`clipboard.rs`) — works inside the alt-screen and over
  SSH, no dep.
- New deps this pass: **none** beyond `ratatui` (added in D7). New unit tests: palette
  resolve/rank, markdown render/inline-code, base64 vectors (bin chat tests 18 → 27).

Note: ratatui's diff renderer interleaves cursor escapes between characters, so PTY screen-scrape
string matching is unreliable; verify features by reconstructing the screen, not substring search.

## D9 — Deterministic CPU forward pass, Pillar One (2026-06-14)

Opt-in deterministic inference for the supported **TinyLlama 1.1B Chat Q8_0** lane, behind
`--deterministic` (`serve` and `bench-generate`; env `CAMELID_DETERMINISTIC=1`). The default
(GPU resident-decode) fast path is untouched — verified byte-for-byte identical token stream
before/after the change at ~88 tok/s on M4. Credit: the pinned reduction order mirrors the
llama.cpp reference block-wise Q8_0 dot layout the parity contract is gated against.

**Flag, not a Cargo feature.** Determinism is a *runtime path choice* (CPU vs the Metal GPU
fast stack), not a compile-time one: the same shipped binary serves both the default fast path
and the deterministic path, and nothing in the default build changes. A Cargo feature would
fork the binary and risk a different default artifact; a runtime flag keeps one byte-identical
default binary and lets a caller opt in per invocation (or per process via the env var, which
the engine reads directly so library embedders get the same guarantee).

**Mechanism (fail-closed in the engine).** `apply_deterministic_mode()` (CLI) sets
`CAMELID_DETERMINISTIC=1`, forces the whole Metal stack off, and disables GPU sampling. The
engine reads `inference::deterministic_mode_enabled()` and **ANDs every Metal/GPU dispatch gate
with `!deterministic`** (`q8_0_metal_enabled`, `resident_decode_metal_enabled`,
`q8_0_metal_retained_enabled`, `q8_0_hybrid_retained_enabled`, and the two
`try_*_linear_row_metal` paths). So even if a `CAMELID_METAL_*` override is present, deterministic
mode still resolves to the order-stable CPU kernels. The default path (`deterministic == false`)
takes the existing branch unchanged.

**Why this is bit-exact for free on the CPU path.** Phase 0 (see
`qa/determinism/determinism-baseline-*.md`) established that the CPU forward pass is *already*
order-stable: every reduction site parallelizes over the **output** dimension (one thread owns a
disjoint set of outputs and runs that output's entire reduction serially), with no cross-thread
float combine, no atomics, no parallel sum/reduce/fold, and a compile-time-fixed SIMD lane
layout. Empirically the CPU stream is byte-identical across runs, across processes, and between
`--threads 1` and `--threads 10`. So `--deterministic` adds **no determinism penalty inside the
CPU computation** — its only cost is forgoing the GPU fast path (~12.5 vs ~88 tok/s on M4).

### The pinned reduction order (the contract a future portable trace depends on)

This order is **order-stable** on a given binary + host (identical across runs/processes/thread
counts). The *values* remain ISA-dependent (i8mm vs dotprod vs scalar round differently), so a
cross-machine trace must pin the host/ISA, not assume cross-ISA bit-equality.

- **Site 1 — Linear / matmul** (Q, K, V, attention-output, FFN gate/up/down) **and Site 3 — lm_head**
  (same kernels, `q8_0_packed_rows4_dot_i8_matmul` / `dot_product_row`):
  1. Output-partitioned across rayon — each output element (or packed group of 4) is computed
     entirely by one thread; parallelism **never** splits a single output's reduction.
  2. The activation row is quantized once (`quantize_q8_0_row`, an order-independent transform),
     then the dot over the K dimension accumulates **block by block in ascending Q8_0 block
     index** (32 values/block), each block scaled by its f32 scale, summed left-to-right into a
     fixed scalar/lane accumulator.
  3. The per-block 32-wide product reduces in a **compile-time-fixed lane layout** (i8mm/dotprod
     `q8_0_packed_4x8` → fixed `sums[0..4]`; scalar fallback unrolled-by-4, left-to-right).

- **Site 2 — Attention** (`attention_context_for_head_into`):
  1. **QKᵀ scores:** each score is a serial `dot_product_row(query, key@pos)` over `head_dim`,
     fixed left-to-right order (vDSP_dotpr on macOS / unrolled-by-4 scalar elsewhere).
  2. **Softmax:** max via `fold(NEG_INFINITY, f32::max)` over positions in **ascending position
     order**; exp + sum accumulated in ascending position order; normalize by `1.0 / sum`.
  3. **Value accumulation:** `out += prob * value` summed over cached positions in **ascending
     position order**. Prefill batches partition over output rows (one thread per row); single-
     token decode is serial per head. Neither splits a reduction across threads.

### Measured overhead (M4, TinyLlama 1.1B Q8_0, hello → 50 tok, greedy, median of 5)

| Path | tok/s | Determinism |
|---|---|---|
| Default (GPU resident decode) | ~88 | self-identical per process; **diverges from CPU at tok 25** (not a bit-reproducible reference) |
| `--deterministic` (CPU) | ~12.5 | **bit-exact** across runs, processes, and thread counts |

## D10 — `camelid chat` agent mode (2026-06-14) — Phase 0

Recon: `RECON_AGENT.md`. Agent mode = a tool-calling plan-act-observe loop built as a mode of the
existing `chat` subcommand (new `src/chat/agent.rs`, driven from the session core; `--agent` flag
+ `/agent` toggle). Reuses the inference client, splash/turn-marker, and TTY/color helpers.

**Decision A — ledger `tool_capable`: PROGRAMMATIC.** `ModelCompatibilityTarget` is structured
Rust surfaced by `/api/capabilities`; add `tool_capable: bool` per row (default false), read by the
agent gate + a future frontend from the same source — no second parser. Rows stay `false` until a
real tool-call round-trip promotes one (same evidence bar as the support gate), so agent mode is
built+tested but honestly refuses every supported row until then. The "stop if prose-only" boundary
does not trigger.

**Decision B — sandbox root: cwd (or `--workdir`), enforced.** Canonicalized absolute root; every
file-tool path is joined, canonicalized, and required to be the root or a descendant (rejects
`..`, outside-absolute, escaping symlinks). For new write targets the parent is checked. Enforced
in code before I/O.

**Decision C — shell: `/bin/sh -c`, cwd-pinned, timed, verbatim-approved.** Model supplies the
whole command; the verbatim string (from the parsed call, not model prose) is shown at approval;
cwd = sandbox root; default 30s timeout; captured stdout/stderr/exit. Honest limitation: `sh` runs
with the user's permissions and can leave the root cwd — `run_shell` is cwd-confined + approval-
gated, NOT a filesystem jail (true jailing = `sandbox-exec`/namespaces, a documented follow-up).
Exec is therefore the highest risk class: always prompts, never in an auto-approve default.

**Auto-approve stance.** `--auto-approve` may exist for power users but prints a prominent warning,
is never a README default, and only relaxes *prompting* — never the sandbox (file tools) or the
prompt-injection rule (tool-result content is data, never permission to escalate or act).

**Phase 1 (tool-calling in inference) is a SUBSTANTIAL scope boundary** — confirming the approach
(server-side / client-side / hybrid; see RECON_AGENT) before building it. The deterministic core
(loop/tools/sandbox/approval/mock/tests) is buildable first and independent of that choice.

### D10 (cont.) — agent mode: deterministic core DONE; Phase 1 + promotion pending

**Built + tested (Phases 0, 2–6, 8-core):** `src/chat/{tools,tool_parse,agent}.rs`. The
plan-act-observe loop is UI/model-agnostic (`ModelDriver`/`Approver`/`Reporter` traits), bounded
(`--max-steps`, default 25) and cancellable; the full tool set (read_file/list_dir/search/
write_file/edit_file/run_shell/http_fetch) is sandbox-confined; the approval gate is
y/a/n/q with session policy; the Llama/Hermes tool-call parser is in `tool_parse.rs`. **44 bin
unit tests** cover sandbox-escape rejection, each tool, parse (valid Llama/Qwen + malformed →
clean), loop-with-mock (threads results, step cap), denial handling, **prompt-injection in a tool
result does not execute**, and **--auto-approve still enforces the sandbox**. fmt/clippy/test/doc
green. Capability gate verified live: `--agent` on a non-tool-capable row refuses (typed error,
exit 2).

**Front-end decision:** agent mode runs in the **line renderer** (synchronous, readline approvals,
clean redirected transcripts) — entered via `--agent`. `/agent` in the chat front ends is a
discoverable pointer to it. The full-screen TUI agent (modal approvals inside the redraw loop) is
a documented follow-up.

**Phase 1 (Hybrid tool-calling) — PENDING, scoped:** the server currently rejects `tools` (it
falls into `#[serde(flatten)] unsupported_fields`). To enable: add `tools: Option<Vec<Value>>` to
`ChatCompletionRequest` + `GenerationSessionRequest` (+ the handler conversion), and thread it into
the Jinja render context (`render_jinja_chat_template` → add `custom_tools`/`tools` to `context!`;
`render_metadata_jinja_chat_template_prompt` + `render_chat_prompt_for_tokenization_for_model_result`
gain an `Option<&[Value]>` param — the ~12 existing callers, mostly tests, pass `None`). Backward-
compatible: `tools=none` → the template's no-tools path → byte-identical render. The model's own
chat template then renders tools; the client parses output (already built).

**Promotion — PENDING, evidence-gated:** set `tool_capable=true` on `llama32_3b_instruct_q8_0`
**only after** a real round-trip shows it emits a parseable tool call. Llama 3.2 3B is a small
model and tool-calling reliability is genuinely uncertain — if it doesn't round-trip cleanly, the
row stays `false` and that is reported honestly (no demo claimed without evidence).

### D10 (cont.) — Phase 1 SHIPPED; promotion NOT earned (evidence)

**Phase 1 (Hybrid tool-calling) — DONE.** `tools` is now an accepted field on
`ChatCompletionRequest`/`GenerationSessionRequest` and is threaded into the model's own Jinja
chat template (`render_jinja_chat_template` gained `custom_tools`/`tools` in its `context!`; a
dedicated `render_chat_prompt_for_tokenization_with_tools` is used only when a request carries
tools, so the shared render chain and its ~12 callers are untouched). Backward-compatible:
`tools=None` → the template's no-tools path → byte-identical render (lib's 438 tests still pass).
Verified live: a request with `tools` is no longer rejected (it reached model resolution), and the
model renders + emits tool calls.

**Promotion — NOT done (honest, evidence-gated).** A live round-trip on Llama 3.2 **1B** Q8_0
(the 3B would not load under severe box contention — 140–168s even for the 1B) showed the model
**emits the correct Llama tool-call format** (`{"name":"read_file","parameters":{…}}`) — so the
threading + parser work end-to-end — **but with malformed arguments** (it echoed the JSON schema
instead of `{"path":"notes.txt"}`) and looped to the length cap. That is not a usable tool call,
so **no row is promoted**: `tool_capable` stays false for every row and agent mode keeps refusing
(verified: exit 2). This matches the spec's capability tension exactly — small Llama 3.2 models
don't tool-call reliably, and the more-capable 3B/8B (or Qwen3/Mistral) round-trip awaits a calm
box. The server-side `tool_capable` ledger column is added the moment a row earns it (add the
field to `ModelCompatibilityTarget` + set the promoted row true); until then the client's
`CompatRow.tool_capable` defaults false, so the gate is correct without it. No capability is
published that no model has demonstrated.

### D10 (cont.) — agent-mode hardening + promotion harness (2026-06-14)

**Phase 0 — render bug, fixed (`TOOLCALL_DIAG.md`).** The malformed args were a render bug, not
just model size: tools were threaded as **OpenAI-nested** `{type, function:{…}}`, so the model
saw the envelope leak into the prompt and its `parameters` field was the JSON *schema*
(`properties`/`required`/`type`) — which a weak model echoes. Fix: the server now **normalizes**
each tool to its flat `function` object before rendering (matching llama.cpp/vLLM), localized to
the tools-present path; `tools=None` stays byte-identical (438 lib tests pass). Proven offline by
`tool_render_nested_vs_flat_diagnostic`. Conclusion: render is now canonical-correct, so a future
big-model failure cannot be a leftover template bug. The parser maps `parameters`/`arguments`
correctly (it never keyed off the schema's `properties`).

**Phase 1 — parser robustness.** `tool_parse` tests now cover plain/`python_tag` JSON, Hermes
`<tool_call>` tags, the `function` envelope, **double-encoded args** (a JSON-string normalized to
an object), multiple calls per turn, leading/trailing prose, the schema-echo failure mode (name
parsed, no real args → the gate rejects), and malformed/truncated/empty (clean, no panic).

**Phase 2 — `agent-eval` promotion harness.** A subcommand that loads a model with a **bounded
timeout** and runs a fixed tool-use battery, reporting **PASS / FAIL / INCONCLUSIVE** + a hashed
receipt (`camelid.agent_eval/v1`: model id, GGUF+quant+size, raw output, parsed calls, per-case
pass, host loadavg, timestamp, `promotion_eligible`). **INCONCLUSIVE** (load timed out) never
changes a flag and never counts as FAIL — this is the noisy-box fix. Promotion = flip
`tool_capable` true only after a PASS receipt. Verified live: 1B → **FAIL** (loaded 48s, malformed
args, exit 1) even with the corrected render; 1s timeout → **INCONCLUSIVE** (exit 3, flag
untouched). No row promoted (no PASS earned).

**Phase 3 — polish.** Loop: **repeat-call detection** (3 identical calls → break with an
explanation, not the whole budget) + a step-cap summary of what ran. Approval grants are
**session-scoped** (the `a` choice persists across goals; `/tools` shows which are auto-allowed).
Tool output truncates with an explicit "(N more lines)". Injection resistance proven
source-agnostically: a fooled model that *follows* injected content into a destructive call is
still **denied by the gate** (the gate, not the model, is the backstop) — covers file and
http_fetch result content alike.

**Honesty.** No `tool_capable` flag is set; nothing in the docs claims tool-calling works on a row
without a PASS receipt. The capability only ever moves on harness evidence.

### D10 (cont.) — first promotion: Llama 3.2 3B Instruct Q8_0 is tool_capable (2026-06-14)

`llama32_3b_instruct_q8_0` earned a **PASS** from `agent-eval` and is promoted: `tool_capable: true`
on that `ModelCompatibilityTarget` row (the field is added to the struct, all other rows `false`).
Receipt committed at `qa/agent-eval/Llama-3.2-3B-Instruct-Q8_0-…-PASS.json`: with the corrected
flat-tools render the 3B emitted a well-formed `read_file(notes.txt)` call (args `{"path":…}`, not
the schema echo), read the fixture (`alpha\nbeta\ngamma\n`), and answered `3` — `promotion_eligible:
true`, host loadavg 2.5. This confirms the Phase 0 render fix was the actual blocker: the 3B was
capable; the OpenAI-nested render had been breaking it (it `FAIL`ed before the fix, `PASS`es after).

`--agent --model <the 3B GGUF>` now runs the live loop (the catalog-label match sets the active
label to the ledger id, so `active_tool_capable()` matches). The 1B remains `FAIL`/gated (too weak);
Qwen3-4B `FAIL`ed by reasoning in `<think>` instead of emitting the call (and isn't a ledger row);
both are honest non-promotions. The gate, harness, and ledger all read the one `tool_capable` flag.
## D11 — Execution-trace rollup, Pillar Two (2026-06-14)

Built on D9: now that the deterministic CPU forward pass is bit-exact, a receipt can carry a
cryptographic **execution-trace rollup** — proof of *how* the computation ran, not just which
tokens came out. This is the "later portable execution-trace feature" D9 was the foundation for.

**Shape — single rollup (chosen over per-layer-localized or final-logits-only).** One streaming
SHA-256 (`ExecutionTraceHasher`, `camelid.execution-trace/v1`, `sha256-rollup-v1`) folds, in
forward order across every generated token, each transformer layer's output hidden state and the
final logits (domain-separated by a kind byte + index + length prefix; little-endian f32 bytes).
A mismatch on re-derivation proves the run differs but does not localize the token/layer — that
is the single-rollup tradeoff, chosen for the simplest end-to-end slice. Runtime cost is
negligible (the bytes are already materialized; SHA-256 is multi-GB/s — sub-1% of decode), so
scope, not speed, drove the choice.

**Only meaningful on the deterministic lane; fail-closed.** `LlamaInferenceSession::
enable_execution_trace()` refuses to arm unless `deterministic_mode_enabled()` (RECEIPTS.md
rule 2 — a digest over a non-reproducible run is meaningless). The default (non-deterministic)
path never allocates the hasher and is byte-for-byte unchanged; the receipt field is
`Option<ExecutionTraceBlock>` with `skip_serializing_if = None`, so non-traced receipts serialize
and digest exactly as before (proven by `execution_trace_absent_keeps_receipt_byte_identical`).

**Emission and verification cannot desync — they share one path.** Both the served generation and
`verify-receipt`'s re-run funnel through `replay_receipt_request → generate_decoded_tokens`, where
the rollup is armed (when deterministic) and captured once. When tracing, the prompt-prefix cache
is bypassed (a cache hit would skip the prompt forwards on one side only). Verification re-derives
the digest from an independent re-run and checks it matches; the rollup is included in the
`receipt_id` digest, so it is itself tamper-evident.

**ISA-pinned, not cross-ISA-portable.** The digest is reduction-order-stable on the deterministic
CPU lane but ISA-specific (the Q8_0 dot rounds differently across ISAs), so the block records
`lane` (`deterministic-cpu`) and `host_isa` (e.g. `aarch64-i8mm`). `verify-receipt` re-derives only
when the verifier's ISA matches; otherwise it prints `SKIP execution-trace` rather than a false
`FAIL`. The committed test digest is the M4/i8mm reference (i8mm-guarded), matching the D9 pattern.

**Proof:** `tests/execution_trace.rs` (engine rollup: run-to-run + thread-count invariant +
prompt-sensitive + pinned + fail-closed) and `tests/execution_trace_receipt.rs` (the full
emit→re-derive round trip through the real API replay path). Not built here: per-layer/per-token
localization, distributed (per-shard) trace chains, signing.

### D10 (cont.) — evaluating 8B / Mistral; system-prompt fold (2026-06-14)

Ran `agent-eval` against the other tool-capable-family ledger rows; **neither earns a PASS, so
neither is promoted** (the gate holds):
- `llama3_8b_instruct_q8_0` — **FAIL** (genuine): Meta-Llama-3 8B is *original* Llama 3, which lacks
  the Llama 3.1+ tool-calling training/template. It hallucinated a result in prose instead of
  emitting a structured call. Bigger ≠ tool-capable; it's training+template.
- `mistral_7b_instruct_v0_3_q8_0` — **FAIL** (not a clean capability verdict): its v0.3 template
  first rejected a standalone `system` role ("roles must alternate"). Fix below cleared that, after
  which it *rendered* but still didn't emit a structured call — the GGUF template doesn't present
  tools in Mistral's `[AVAILABLE_TOOLS]` form, so Mistral improvised a shell command in prose.
  Promoting it needs Mistral-specific tool-template rendering — a real follow-up, not done here.

**System-prompt fold (kept).** `LiveDriver::step` now retries with the system prompt folded into the
first user message **only when the template rejects a standalone system role** (Mistral v0.3, Gemma).
The system-role path is tried first and is unchanged, so the promoted 3B is unaffected (re-verified
PASS). This is a correct robustness fix (system-less templates no longer error) and a stepping stone
toward Mistral support. The only promoted row remains `llama32_3b_instruct_q8_0`.

## D11 — Native Windows desktop app (`camelid-desktop`): sidecar + embedded-UI WebView (2026-06-19)

Add a second, **additive** Windows executable (`camelid-desktop`) that gives users a native
desktop chat experience with no browser. Two binding choices, both confirmed before coding:

**1. Engine integration = sidecar, not in-process.** The desktop process spawns the shipped
`camelid serve --addr 127.0.0.1:<ephemeral> --no-open` bound to **loopback only**, health-gates
on `/v1/health`, and kills it on window close. `api::router_with_state()` is `pub` so in-process
*was* feasible, but sidecar is chosen for v1 because it guarantees byte-identical generation
behavior to the shipped server and isolates engine crashes from the UI. Any divergence in
generation between desktop and `camelid serve` would be a regression, not a feature.

**2. UI delivery = point the WebView at the sidecar's already-embedded UI, NOT re-bundle the
frontend.** The `camelid` router's fallback route (`*` → `crate::web_ui::handler`, `src/api/mod.rs`)
already serves the same React app the web path uses, embedded in the binary via `rust-embed`.
The desktop WebView therefore navigates to `http://127.0.0.1:<port>/`, making UI **and** API
same-origin: the frontend's `defaultApiBase()` resolves to `window.location.origin`
(`frontend/src/hooks/useDashboardData.js`), so no base-URL injection or ephemeral-port plumbing
is needed, and the runtime-ready + exact-supported-row capability gate is inherited **verbatim**
(it is literally the same UI hitting the same `/api/capabilities`). The brief's literal Phase 2
("bundle the frontend as Tauri assets") is therefore intentionally collapsed: the frontend is
already bundled *inside `camelid.exe`*; the desktop `.exe` needs no npm/Vite at runtime and keeps
no second copy of the UI. Tauri ships only a tiny static "starting engine…" splash page.

**Stack:** Tauri v2 (Rust backend + WebView2). WebView2 ships with supported Windows 10/11.

**Workspace:** the root `Cargo.toml` becomes a workspace (`resolver = "2"`,
`default-members = ["."]`) with `camelid-desktop` added as a member. CI builds the server with
`cargo build --release --locked --bin camelid`, which does not pull the desktop member into its
build graph, so the server binary stays byte-for-byte unaffected. `default-members = ["."]` keeps
bare `cargo build`/`cargo test` scoped to `camelid` as before; the desktop app is built explicitly
with `-p camelid-desktop`.

**No new inference / no metric fabrication / no support-contract drift / additive CI:** the desktop
app reuses the shipped engine and gate wholesale; any tokens/sec readout (Phase 3) is sourced from
the existing SSE `decode_tps` real generation event, rendered unavailable when absent. The new
release job is additive and independently skippable; existing server artifacts are untouched.
See `camelid-desktop/README.md`.

### D11 (cont.) — NSIS in-place upgrades must remove sidecar files the version no longer ships (2026-07-31)

An overwrite-only installer has a blind spot: it rewrites the files it ships and leaves
everything else alone, so a file installed by an **older** version that the current one dropped
survives every future upgrade. Found in the field on a v0.4.6 box — `sidecar\nvrtc64_120_0.alt.dll`,
85.7 MB, from an install predating the `.alt` filter in `scripts/package-windows-cuda.ps1`. The
published v0.4.6 artifact correctly contains no `.alt` entry and
`scripts/check-release-artifact.mjs` has a forbidden rule for it, so **the packaging and artifact
gates were both already correct** — the defect was only ever in the upgrade path, which is why
neither gate could see it and why re-running the one-command installer never cleared it.

**Fix: `NSIS_HOOK_PREINSTALL`** (`camelid-desktop/windows/installer-hooks.nsh`, wired via
`bundle.windows.nsis.installerHooks`), deleting `nvrtc64_*.dll` and `nvrtc-builtins64_*.dll` from
`$INSTDIR\sidecar` before the file copy re-lays the current set.

Deleting the whole NVRTC family rather than blacklisting `.alt` is the deliberate choice: these
filenames encode the CUDA version (`nvrtc-builtins64_129.dll` is 12.9; `nvrtc64_120_0.dll` the
12.x ABI), so a `CUDA_VERSION` bump renames them and strands the previous ones identically. A
rename is exactly what an overwrite cannot fix, so the blacklist would have to be extended by
hand every bump; re-laying the family is self-healing. Cost is that the good DLLs are deleted and
immediately re-extracted — verified below.

**Rejected: clearing `sidecar\` wholesale before extracting.** It looks safe because `sidecar\`
is shipped content, but it is not: `sidecar_models_dir` (`camelid-desktop/src/engine.rs`) makes
the desktop's model store `models\` *beside the engine binary* — i.e. `sidecar\models\`. The box
that reported the orphan had **4.42 GB of user GGUF weights there**. A wholesale clear is
therefore a data-loss bug, not a cleanup.

**Verified against a real in-place upgrade,** not by reading the template: installer built from
this config, planted `sidecar\nvrtc64_120_0.alt.dll`, upgraded over the existing install. The
`.alt` orphan is gone, the shipped NVRTC pair is present and re-extracted at full size, the
`models\` subtree survives byte-for-byte, and the app relaunches and health-gates.

Failure is non-fatal by construction: `Delete` on a missing or locked file sets the error flag
without aborting, so the worst case is the orphan surviving to the next upgrade — never a broken
install.

## D11 cont. — the binary inside the Windows installer is signed by the bundler, not by a folder pass (2026-08-01)

`camelid-desktop.exe` exists as **two distinct copies**, and a folder-based signing pass can
only ever reach one of them. Tauri stamps the bundle type into the copy it stages for the
installer, rewriting `__TAURI_BUNDLE_TYPE_VAR_UNK` to `..._NSS` so the installed app knows how
it was installed — and it rewrites a *staged copy*, leaving `target/release` untouched.

That gives a window with exactly one entrance. A signature applied **before** `tauri build` is
invalidated by the rewrite; a signature applied **after** cannot reach a binary already sealed
inside the installer. Two releases shipped from the two sides of it:

| release | when the desktop exe was signed | what users installed |
| --- | --- | --- |
| v0.4.6 | after bundling only | `NotSigned` |
| v0.4.7 | before bundling only | `HashMismatch` — a *broken* signature |

v0.4.7 was the worse outcome: an invalid signature earns no SmartScreen reputation and reads
as tampering, where an unsigned binary merely reads as unsigned. It was diagnosed by byte-diffing
the two published copies, which differ in exactly two places — the 3-byte marker at `0x8043c8`
and the PE CheckSum, which Authenticode excludes from its hash.

**Fix: `bundle.windows.signCommand`** (`camelid-desktop/windows/sign-artifact-signing.ps1`),
the only hook Tauri invokes between the rewrite and packaging. It drives `signtool` through
`Azure.CodeSigning.Dlib`.

**Rejected: `artifact-signing-cli`,** which Tauri's own documentation suggests for Azure. It
requires `AZURE_CLIENT_SECRET`, and this repo authenticates to Artifact Signing with OIDC
federated credentials and stores no secret. Adopting it would have traded a short-lived
federated token for a long-lived secret purely for convenience. The dlib authenticates through
`DefaultAzureCredential`, which resolves the `azure/login` session the workflow already
establishes, so OIDC is preserved.

**The gate matters more than the fix.** Every prior signing check inspected a file handed to
the signer; none could see the installer's payload, which is why two consecutive releases
shipped defects there. `Prove the INSTALLED binary is signed` silently installs the built
installer on the runner and asserts the unpacked binary verifies. Runners are ephemeral, so
this costs nothing, and it would have caught both defects before publishing.

## D11 cont. — signCommand paths must be absolute, and an incomplete release must never be `latest` (2026-08-01)

v0.4.8 published with **no Windows desktop installer**. `desktop-windows` died with
`failed to bundle project: failed to run powershell`, the server artifacts published anyway, the
release became `latest`, and the documented one-command install broke for every user with an
error they could do nothing about.

**Cause: `signCommand` is invoked seven times per bundle with a working directory that is not
constant.** Measured on a real bundle — the app binary, five NSIS plugin DLLs, and the
uninstaller staged as `%TEMP%\nst*.tmp`. Six run from the Tauri project directory; the
uninstaller call runs from `target\release\nsis\x64`. The project-relative script path resolved
for six and failed the build on the seventh. A bare `powershell` also failed to spawn on the
runner, so the interpreter is resolved to a full path too.

**Fix:** the release workflow generates `camelid-desktop/tauri.signing.conf.json` at release
time with absolute paths, passed as an extra `--config`. It is gitignored and deliberately
absent from the committed `tauri.conf.json`, so a developer's `tauri build` needs no signing
tooling. The overlay re-states `installerHooks` — read out of `tauri.conf.json`, derived rather
than duplicated — so it cannot silently drop the NVRTC-orphan cleanup if Tauri's config merge is
ever shallower than a deep merge at `bundle.windows`.

**Verified by a real bundle, not an assertion.** Text checks are what let v0.4.8 through: the
config was present, the script parsed, and its path resolved *from the project directory*. The
fix was proven by running `tauri build` with the generated config and observing all seven
invocations spawn.

**Two guards, because the fix alone only covers this bug:**

1. `verify-release-assets` (`if: always()`, needs every publishing job) checks the published
   asset set and, if anything is missing, flips the release to a **prerelease** so
   `releases/latest` skips it and the previous complete release keeps serving installs. It then
   fails loudly. The desktop jobs stay independent — they still cannot block server artifacts —
   but their failure can no longer reach users. **Rejected:** making the desktop jobs blocking,
   which would let a desktop-only failure withhold the server release.
2. `scripts/get-desktop-windows.ps1` no longer binds to `releases/latest`. It walks back to the
   most recent release that actually publishes an installer and warns which it chose. An
   explicit `CAMELID_DESKTOP_TAG` still fails hard rather than silently substituting a version
   the user did not ask for.

## D12 — CPU KV cache: f16-rounded values were stored in f32 buffers; f16 storage + head-major layout lanes (2026-07-01)

Measured finding (Item-3 recon): the CPU KV write path has rounded every stored
key/value element through f16 unconditionally since the f16-KV work landed
(`copy_to_f16_kv_cache_storage`, now `store_kv_head_row` in
`src/inference/kv_cache.rs`), but kept the results in f32 buffers — 2x the
bytes for zero additional precision. For Llama-3.2-3B at 8192 ctx that is
~1.79 GiB of KV instead of ~0.9 GiB, which is exactly what host-limited
deep-context runs on the 15.7 GiB Windows dev box.

Consequences, recorded as binding context for future parity claims:

- All Item-1 (blocked-dot, PR #355) and Item-2 (head-parallel, PR #358)
  parity and A/B evidence was earned on f16-rounded KV values.
- The Item-3 f16 storage lane (`BACKENDINFERENCE_KV_F16`, PR #359) therefore
  introduces ZERO new rounding: it stores the same bits as u16 and expands
  exactly on read. Both Item-3 lanes (with `BACKENDINFERENCE_KV_LAYOUT_HEAD_MAJOR`)
  carry the bitwise-identity contract — end-to-end token identity was proven
  in 17/17 flag combinations plus the serve stack, and the 8192-ctx run
  completes on the dev box (4.037 tok/s, 5.72 GB peak working set).
- Canonical conversions are pinned in `src/inference/kv_f16.rs`: store = the
  historical helper semantics (RNE, NaN -> sign|0x7E00, F16C fast path with a
  NaN fixup); read = exact `vcvtph2ps` semantics on the full 2^16 domain
  (quiets signalling NaNs — unreachable from the canonical store; the delta
  vs the older helper is documented there and locked by exhaustive tests).

Receipts: `target/kv-lanes-baseline-20260701T180113/`,
`target/kv-lanes-ab-20260701T185422/`, `target/kv-lanes-ab2-20260701T195346/`
(SHA256-sealed, dev box) and the PR #359 description.


## D13 - Decode thread-width policy: detected physical cores, never SMT logical count (2026-07-02)

**Decision:** on Windows x86_64, single-token decode runs by default on a dedicated
rayon pool sized to the DETECTED physical core count (GetLogicalProcessorInformation,
one RelationProcessorCore record per core). The SMT logical count is never used for
decode. Detection failure, an operator-pinned global (CAMELID_THREADS), or a global
already narrower than physical all fail closed to the previous inline-on-global
behavior. `BACKENDINFERENCE_DECODE_THREADS=N` remains the explicit width override and
`=0`/`=off` is the kill switch; `BACKENDINFERENCE_DECODE_POOL_DEDICATED=1` remains the
isolate-at-global-width override.

**Basis (receipts):** the Items-4/5 P2 sweep (`target/decode-sched-p2-20260702T011105/`,
SHA256-sealed): decode tok/s is flat across widths 4-8 (within ~3%) and falls
monotonically past the physical count (width 16 = -20% vs the plateau at depth 512,
-19% at 2048); the serve-path A/B in the same receipt shows the dedicated-pool
configuration at -4.5% wall and an explicit narrower width at -7.1%. This host's
measured optimum (6) is deliberately NOT hardcoded: the portable, defensible policy is
"physical, not logical"; per-host refinement of the exact width within the flat region
is deferred to GAIT calibration.


## D14 — Models page: five-zone IA, derived membership only, no localStorage truth (2026-07-02)

**Decision:** the Models page is exactly five zones in one scroll — (1) active model
bar with the only Unload action, (2) Supported, (3) Experimental
(compatible → eligible → not-anchored), (4) one global Downloads panel, (5) Get
models (curated + live Hugging Face search with confirmed downloads). Section
membership is computed at render time from `/api/models/local` +
`/api/capabilities` via `frontend/src/lib/modelLanes.js` (extracted verbatim from
the old LocalLaneSections); no hand-authored array places a model. The
`SUPPORTED_MODELS` list survives only as curated-download decoration
(blurb/Recommended) in Zone 5.

**localStorage download tracking is removed as a source of truth.** Download
progress renders only from the `/api/models/catalog/downloads` poll
(bytes/total), owned by one spine hook (`useModelsPageData`); "downloaded" means
the file is present in the live `/api/models/local` scan, nothing else.

**Diagnostics relocation:** Tokenizer Playground and the Model Inspector moved
into a collapsed "Diagnostics" disclosure at the page bottom (only ModelsView
consumed them). Import-a-GGUF-by-path also lives there — it is the only way to
load a GGUF stored outside `models/`, so the capability is preserved but off the
primary surface. The permanently-disabled Hosted API fieldset, the legacy
FILTERS grid, the hero ledger, the runtime status grid, the acceptance/tracked-
row panels, and the localStorage-driven SupportedModels section were deleted
outright. (The import/hosted-API panel was not listed in the rebuild conductor's
inventory; this disposition is the flagged resolution.)

**"Not-yet-runnable, visible but disabled" interpretation:** the backend exposes
no pre-download refusal signal per catalog row, so the disabled-with-reason state
is driven by the real derivable gates — runtime offline or the backend not
advertising `hf_catalog_install`/`model_downloads`. Per-row implementability is
still decided at load time by the inspect-first typed-blocker flow (fail-closed,
rendered verbatim).

## D15 - Q8 prefill GEMM owner: default ON, win-x86_64 only (2026-07-08)

**Decision:** `CAMELID_X86_Q8_MATMUL_OWNER` defaults to `All` on Windows x86_64 and
stays `Off` everywhere else. Explicit rollback: `CAMELID_X86_Q8_MATMUL_OWNER=off`.
The variant inside the owner is unchanged (4x8 AVX-512 VNNI when the CPU has it,
AVX2 4x4 otherwise; both bit-exact with twin tests).

**Basis (receipts):** re-validated at the llama.cpp b9918 re-pin on main 582781da with
the hardened paired in-process sweep, now carrying an ENGAGED-CHECK (the sweep aborts
if an owner-on config never dispatches the owner arm, and the per-record
`owner_prefill_taken` counts are in the receipt — off=0, owner=280): 3B Q8 prefill
26.97 -> 30.30 tok/s (+12.3%, CI [1.115,1.133], 8/8 rounds), Qwen3-4B 20.00 -> 22.40
(+11.9%, CI [1.113,1.126], 8/8). Bit-exactness unchanged from the #345 twin tests.
`PERF_RECEIPTS/same-host/q8-prefill-owner-b9918-revalidation-20260708/`.

**Treaty note (Tim-visible, not buried):** BENCHMARK_TREATY.md asks for a both-host
(Windows + Ubuntu) win before promoting a default; the Ubuntu host remains
PENDING/paused per Tim's earlier call. This flip follows the D13 precedent instead:
single-host receipts promote a default that is cfg-SCOPED to exactly the host class
that was measured (win-x86_64), leaving every other target Off. If the treaty should
bind even scoped flips, revert by deleting the cfg branch in
`Q8MatmulOwnerScope::from_env`.

## D16 — API engine inversion: one worker thread owns all decode compute (2026-07-09)

**Decision:** `AppState::generation_lock` is deleted. Every decode (streaming,
non-streaming, multi-choice, receipt replay) and every mutation of
engine-owned state (the GPU-runnable parity probe, `reset_resident_caches` on
unload) executes as a job on ONE dedicated engine worker thread behind a
bounded queue (`CAMELID_QUEUE_DEPTH`, default 8). HTTP handlers validate and
prepare OUTSIDE any serialization, post a job, and consume its events.

**Ownership invariant (enforced by tests, binding on future work):** only the
engine thread touches `LlamaInferenceSession` decode state, resident GPU
decode/KV state, and the prompt-prefix cache; all mutations of that state are
engine jobs. Anything else is a regression of the orphan-decode SEV.

**Why:** the lock's guard lifetime was decoupled from the `spawn_blocking`
decode it guarded. Client disconnect, the server's own generation timeout, or
an SSE hangup dropped the handler future, freed the lock, and left the decode
running — the next request then decoded concurrently with the orphan against
shared CUDA-resident KV state (garbled output, non-deterministic greedy,
OOB-slice panics). Demonstrated on all three triggers at the pin
(`qa/evidence-bundles/engine-inversion-gate0-*`); fixed structurally
(`engine-inversion-gate1-*`, `engine-inversion-gate2-*`), mirroring
llama.cpp's `server_queue` single-consumer ownership model.

**Contract consequences (deliberate):** cancellation is cooperative — a
dropped request stops its decode within one token step (`GenerationCancel` /
`CancelOnDrop`); the wall-clock timeout is enforced inside the decode loop and
never detaches compute; burst beyond queue depth is a typed 503
(`engine_queue_full` + Retry-After) with depth observable in `/v1/health` and
`/v1/slots`; a decode wedged inside one forward waits rather than
503-and-orphan. Parity: supported-row outputs byte-identical pre/post across
greedy + seeded sampling, stream + non-stream, cache hit + miss, CPU + resident
CUDA lanes (receipts in the gate bundles). Multi-slot/continuous batching was
evaluated and KILLED for this hardware class
(docs/recon/ENGINE_INVERSION_PHASE5_MULTISLOT_RECON.md).

## D17 — BASALT v1 NVFP4 decision set (2026-07-16)

**Signed by Tim at Gate G0 (PR #466): T1 no Blackwell available; T2–T7 accepted as
recommended.** Full context: `BASALT_RECON.md` §9/§11; eval design:
`basalt_eval_protocol.md`. The five campaign decisions, as accepted:

- **D-B1 — Wire compatibility:** adopt the pin's (`llama.cpp acd79d603`) `GGML_TYPE_NVFP4`
  layout byte-for-byte — type id 40, 64-element/36-byte superblock `{d[4] UE4M3, qs[32]}`,
  MXFP4-style nibble split, doubled-LUT × half-scale decode pair kept together as one
  convention. No Camelid-private layout.
- **D-B2 — Per-tensor scale seam (reframed by pin truth):** there is no in-block tensor
  scale. v1 implements the four in-block UE4M3 sub-scales only and **fails closed at
  admission on NVFP4 GGUFs carrying `.scale`/`.input_scale` sidecar tensors** (silently
  ignoring them would compute wrong logits for ModelOpt-converted files). Sidecar
  application is a follow-on once a sidecar-bearing fixture exists. Folding scales at load
  remains rejected.
- **D-B3 — Runnable-lane scope:** pilot-model-only until Gate G3 passes, then lane-wide
  admission; smoke stays gated on oracle-qualified combos.
- **D-B4 — Blackwell kernel route:** moot — no sm_100/sm_120 silicon (T1). Phase 5 is
  BLOCKED-HW; recorded with zero performance claims. Decide NVRTC-vs-precompiled-PTX only
  if hardware materializes.
- **D-B5 — Quantizer ownership:** pin-tool-only for v1, via the per-tensor override path
  (`llama-quantize --allow-requantize --tensor-type '<regex>=nvfp4'
  --override-kv general.file_type=int:39 <src> <dst> <base-ftype>`), empirically
  deterministic. Native Camelid quantizer stays optional (Phase 2b), deprioritized.

Also accepted under the same sign-off: **T5** NaN-scale posture (decode semantics match the
pin bit-for-bit — NaN sentinels flush to 0.0 — while Camelid admission scans NVFP4 tensors
and refuses files containing `0x7F`/`0xFF` scale bytes; zero scales admit); **T6** the
campaign proceeds on decode-bandwidth + partial-residency grounds after the 6 GB
full-residency refutation; **T7** the gated quality comparison is the format-isolated
`NVFP4-mm` vs `Q4K-mm` pair, with standard Q4_K_M rows and `NVFP4-all` report-only; and the
two `basalt_eval_protocol.md` §6 amendments (gemma4 lane-native packs; 80% sanity guard).

**Why:** interop with real files over format invention; refusal over silent corruption at
every ambiguity (sidecars, NaN scales); a controlled experiment over a confounded one for
the quality gate; and no claims — performance or otherwise — without hardware and receipts.

**D17 addendum (2026-07-16, G1):** the Phase 1 golden vectors (pin-generated, fixture
arbiter) corrected the T5 premise: the pin's CPU decode flushes only raw `0x7F` to 0.0,
while `0xFF` decodes to 240.0 — and the pin's CUDA mirror flushes both, so the pin's own
backends disagree on `0xFF`-scaled blocks. The accepted posture is unchanged and
strengthened: Camelid decode is pin-CPU-bitwise (`0xFF` → 240.0), and admission refuses
files containing either sentinel byte — such files cannot even produce a well-defined
cross-backend oracle. Also fixture-corrected: `decode(0x7E)` = 224.0 (a Phase 0 aside said
112.0). See BASALT_RECON.md §1 [G1 errata] and the Phase 1 evidence bundle.

**D17 addendum 2 (2026-07-16, G2 signed):** Tim signed Gate G2 (PR #470 merged), which
included the flagged D-B3 implementation shape: since gemma4 is deliberately outside the
runnable lane's covered architectures, the pilot scoping is an architecture-axis
carve-out — a gemma4 GGUF passes that axis iff it carries ≥1 NVFP4 tensor; gemma4 files
without NVFP4 and all other architectures keep their pre-BASALT refusals byte-for-byte.
Additionally (Phase 3): the D-B2 sidecar refusal and the T5 NaN-sentinel refusal are
enforced in BOTH lanes — runnable admission/decode (Phase 2) and the gemma4 wire lane
(`nvfp4_sidecar_check` at load + `WireQuant::new` sentinel scan) — because the wire lane
never runs the runnable decoder and would otherwise silently bypass both signed postures.
Closure accepted against Amendment 3 §1 bar on signature (2026-07-16): the synthetic
sidecar-tripping wire-lane fixture per §1.2 is committed (`tests/fixtures/gguf/`,
`tests/nvfp4_wire_lane_refusals.rs`), and the four sanctioned rows re-hashed
byte-unchanged per §1.3 (`qa/evidence-bundles/basalt/phase3/row_rehash_post_closure.txt`).

**D17 addendum 3 (2026-07-16, SHA_E3):** latent pre-BASALT K-quant projection routing gap
fixed under the Amendment 3 §3 freeze-move mechanism (crash-fix, no design change),
discovered by the S3 legs on the Q4K-mm row: the per-layer/batched projection call sites
bypassed the top-level matvec's Q4_K/Q6_K → Q8_K-activation routing and panicked
`unreachable!` at forward time; projections now dispatch per activation family
(byte-identical for NVFP4/Q8_0/Q4_0/Q4_1), Q5_K matvec roles refuse typed at load, and the
L2 I-unknown-type cell note is corrected (`qa/invariant_lanes.json`).

**D17 addendum 4 — Gate G3 outcome: NO-GO (2026-07-17, final freeze SHA `8038abba`).** The
pre-registered §5.2 rule (GO iff `agreement(NVFP4-mm)` ≥ `agreement(Q4K-mm)` − 2.0 pp) was
applied verbatim to the eval legs: `agreement(NVFP4-mm)` = 88.5, `agreement(Q4K-mm)` = 92.6,
threshold 90.6 → **NO-GO** (gap 4.1 pp, > 2× tolerance; sanity guard held at 92.6 ≥ 80.0).
NVFP4-mm is the worst of the four produced rows on both top-1 agreement and mean KL vs the
Q8_0 parent (all figures **vs Q8_0 parent, matched 4.5 bpw**, Amendment 3 §4). The
format-isolated comparison isolates this to the weight format alone (proven identical
elsewhere at G2). Cross-engine token parity passed independently (Leg B 8/9, one attributed
0.084-logit near-tie). **Conclusion recorded as measured; no threshold adjusted.** The
decode-bandwidth motivation (Phase 4) is a separate axis untouched by this quality verdict.
Scope decision — postmortem-and-stop (A) vs continue-to-Phase-4-on-bandwidth-grounds (B) —
is Tim's; recommendation (A) as the honest default, in the G3 PR. Receipts:
`qa/evidence-bundles/basalt/phase3/BASALT_G3_SUMMARY.md` + `legs/`.

**RESOLVED — Tim chose Option B (2026-07-17):** continue to Phase 4 on decode-bandwidth
grounds, treating NVFP4 as a space/speed format, NOT a quality-competitive one. Binding
consequence (claim-lint, enforced at Phase 6): every user-visible NVFP4 surface — README,
capability/support matrices, CAIRN ledger rows, Evidence Chip copy — carries the G3 quality
delta ("behind Q4_K at matched 4.5 bpw on the pilot: 88.5 % vs 92.6 % top-1 agreement,
0.111 vs 0.065 mean-KL nats, vs the Q8_0 parent"); no NVFP4 surface may imply
quality-competitiveness. The G3 NO-GO stands as a recorded, receipted result; Option B is a
forward-scope choice on a different axis, not a reversal of it.

**D17 addendum 5 (2026-07-17, Phase 4 — NVFP4 CUDA decode kernel landed).** The NVFP4
dequant-in-kernel CUDA-resident decode GEMV (`nvfp4_gemv`, `src/cuda_resident.rs`) shipped
behind a unit-level bit-parity gate: warp-per-row, raw 36-byte wire, exact in-kernel UE4M3
sub-scale decode (bit-for-bit `tensor::ue4m3_to_f32_const` — flush raw 0x00/0x7F, 0xFF->240.0
pin-CPU-bitwise), scalar E2M1 LUT integer dot, and the ordered lane-0 sum reproducing
`nvfp4_wire_row_dot`'s superblock-major/sub-block-minor order — so the kernel is 100%
BIT-identical to the CPU oracle on the same bytes (`nvfp4_gemv_matches_oracle`, the same
ordered-sum family as q8/q4_0/q4_1; the 1e-4 close() is a compiler backstop only). The
gemma4 CUDA lane now covers Q8_0/Q4_0/Q4_1/NVFP4 (`nvfp4_cuda_lane_check` admit arm;
`GemmaLayerQuant::Nvfp4` raw-wire passthrough); the K-quants stay CPU-only (still typed-refused
on the CUDA lane). The scalar-LUT kernel is v1 per the orchestrator §5 Q1 decision; the pin's
`__byte_perm`+`__dp4a` inner loop is recorded in a kernel comment as a measured
PARITY-NEUTRAL follow-up (identical i32 sumi). The six L3 `open:P4` invariant cells closed in
this commit (I-nan-scale/I-sidecar/I-scale-once/I-k-div/I-plat -> enforced with lane-native
tests; I-cache-quant -> na structural per §5 Q3), satisfying ratchet R3; the pre-Phase-4
CUDA-lane refusal text is gone and its pinning test inverted. Scope of THIS commit is
implementation + the unit bit-parity gate ONLY — the end-to-end CPU-vs-CUDA self-parity CERT
and the Gate-G4 perf table (medians, achieved GB/s) are a separate later step (no model loaded,
no benchmark here).

**D17 addendum 6 (2026-07-17, Phase 4 — Gate G4 CERT + perf table, measured this box:
RTX 3060 Laptop sm_86, driver 576.83, CUDA 12.9).** End-to-end self-parity CERT (NVFP4-mm
greedy, Camelid CPU wire lane vs CUDA-resident lane, gemma4 lane-native 9-prompt pack) =
**6/9 token-identical**; the 3 divergences are all near-tie argmax flips in which the CUDA
token is exactly the CPU's #2 candidate at a 0.047–0.111 raw-logit gap — attributed to the
accepted CUDA-f16-KV vs CPU-f32-KV difference (the same greedy-token contract the Q8_0 row
ships under), NOT a wiring/kernel bug. **CERT PASS.** G4 perf (median of 5 warm runs, 128
greedy tokens, decode tok/s): Q8_0 CUDA **25.80 tok/s** (39.8% of the 336 GB/s DRAM roofline,
peak 5559 MiB), NVFP4-mm CUDA **14.64 tok/s** (13.3% of roofline, peak 3479 MiB), NVFP4-mm
CPU 1.57 tok/s. Per-token weight read shrinks a **measured 1.70×** (matmul-only exactly
1.889×; format-isolated 1.647×, matching the recon's pre-registered ~1.6×). **Surprise/headline:
the byte reduction did NOT translate to speed — NVFP4-mm CUDA decodes at 0.57× the Q8_0 lane
because the v1 scalar-LUT dequant kernel is COMPUTE-bound (13.3% vs 39.8% of roofline), leaving
its bandwidth advantage on the table.** This is exactly the receipt recon §5 Q1 pre-registered:
the parity-neutral `__byte_perm`+`__dp4a` inner-loop upgrade (and the gpu_head lever, §5 Q4) are
now WARRANTED as the Phase-4-follow-up perf bite. NVFP4's realized win on this box today is VRAM
headroom (2.08 GB more free), not decode speed. Pin-GPU cross-engine row skipped (memory-safety:
a 6.06 GB model full-offload on a 6144 MiB card is not comfortable headroom). Receipts:
`qa/evidence-bundles/basalt/phase4/cert/` (parity_cert.json, perf_table.json, byte_accounting.json,
G4_PERF.md, sanitized command/resource logs).

**D17 addendum 7 (2026-07-17, Phase 4b — dp4a kernel upgrade landed, Option B executed).** Tim
chose **Option B** at G4. The pre-registered parity-neutral upgrade is done: `nvfp4_gemv`'s
per-sub-block integer dot (`src/cuda_resident.rs`) was rewritten from the scalar nibble-unpack +
E2M1 KV-LUT × q8 multiply to the pin's `get_int_from_table_16` `__byte_perm` codebook expansion
(nibble → signed int8) + `__dp4a` 4-way int8 dot — ported exactly from ggml-cuda/vecdotq.cuh
(`sm_86` already has `__dp4a`; no arch change; the v1 scalar loop is kept as a code comment for
the before/after receipt). Because the accumulated i32 `sumi` is identical by construction, it
cannot move parity, and the gate confirms it: **`nvfp4_gemv_matches_oracle` stays 46/46
bit-identical, worst rel diff 0.000e0**, with the sentinel-decode, residual-fusion, and
even-bpr guards all green (plus fmt/clippy-all-features-deny/plain-test clean). **Perf (median
of 5 warm runs, 128 greedy tokens, this box): NVFP4-mm CUDA decode 14.64 → 26.51 tok/s
(+81.1 %, 1.81×)**, moving from 13.3 % to **24.0 %** of the 336 GB/s DRAM roofline (peak VRAM
unchanged at 3479 MiB; byte read set unchanged at 3.048 GB/token). It **did NOT reach the
roofline** (Q8_0 still sits higher at 39.8 %, so the kernel is not yet fully memory-bound), but
the ~1.8× lift is enough to **overtake Q8_0: 26.51 vs 25.80 tok/s (1.03×)** — a narrow but
measured decode-speed win, because NVFP4 reads 1.70× fewer bytes/token. **Option-B outcome:
NVFP4-mm on this box is now BOTH faster than Q8_0 (1.03×) AND 2.08 GB lighter in VRAM.** The
Phase-6 claim-lint statement updates accordingly (faster + lighter on this card, decode-only,
narrow; the G3 quality delta still travels unchanged). The gpu_head lever (§5 Q4) and closing
the remaining roofline gap remain open follow-ons, not scheduled. Receipts:
`qa/evidence-bundles/basalt/phase4/cert/` (perf_table.json `NVFP4mm_cuda_dp4a` row kept beside
the v1 row; BASALT_G4_SUMMARY.md §2/§3/§4; p4b_dp4a_perf_log.md).

**D17 addendum 8 (2026-07-17, Phase 6 — surface alignment).** Every user-visible surface now
carries the single honest NVFP4 story, per the Option-B claim-lint binding (addendum 4). NVFP4
is a **`planned_quantization` axis item only** (`status = planned_beyond_named_certified_rows`
in the `/api/capabilities` contract + regenerated ledger) — deliberately **no
`model_compatibility` row and no frontend catalog entry**, which keeps the CAIRN drift gate
(checks A–E) and the `capabilities_support_statuses_stay_exact_row_allowlisted` tripwire green
(a supported-row add would have tripped both). The same story lands on README (quant tier +
Experimental-lanes row), COMPATIBILITY (quantization-formats row), STATUS (durable evidence
anchors + what-changed), SUPPORT_MATRIX (unsupported/downgraded row, the target of the TK2
Windows-only error text), CAPABILITY_MATRIX (`load.quant_breadth` note), and the new
`docs/architecture/NVFP4_FORMAT.md` spec: gemma-4-E4B pilot only, Windows-only; behind Q4_K on
quality (Gate G3 NO-GO, 88.5% vs 92.6% top-1 agreement, 0.111 vs 0.065 mean-KL nats vs the
Q8_0 parent at matched 4.5 bpw); narrowly faster than Q8_0 CUDA decode (26.51 vs 25.80 tok/s,
1.03×) and 2.08 GB lighter VRAM on an RTX 3060 Laptop (Gate G4, decode-only, this box); not a
supported/certified row, not quality-competitive. **Phase 5 (Blackwell sm_120 tensor cores) is
recorded BLOCKED-HW** (no Blackwell silicon on the target; no forward-looking perf claim). Still
pending Tim's signature (carried, non-blocking): **D-B6/TK3** per-tensor sidecar admission
(draft banked, Option A recommended) and the **§2.4** invariant-matrix-mechanism deviation
(disclosed above). Surface-alignment only — no support claim moved.

**D17 addendum 9 (2026-07-18, GABBRO M2 — macOS admit + §9 gate narrowed).** GABBRO Gate
G-M1 proved the NVFP4 CPU wire-lane decode bit-exact on Apple Silicon/ARM (13/13 committed
`nvfp4_*` tests on an Apple M4; receipt `qa/evidence-bundles/gabbro/phase1/`), so the
Amendment 3 §9 platform gate is narrowed to admit NVFP4 on **Windows AND macOS**, refusing
only on other targets. Both lanes (`nvfp4_windows_only_check` in the gemma4 wire path and the
mirrored `runnable::admit` gate) now test `!windows && !macos`; the TK2 error text is
truthed-up to **"NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"** and every
support surface (SUPPORT_MATRIX, CAPABILITY_MATRIX, STATUS, COMPATIBILITY, README,
`docs/architecture/NVFP4_FORMAT.md`, `qa/invariant_lanes.json`) is updated in the same ratchet
PR (Tim's ruling folded the surface truth-up into M2). Scope on macOS is the **CPU wire lane
only** — the Metal GPU kernel is GABBRO Phase M3 and is not yet wired (`Gemma4GpuRuntime`
still typed-refuses NVFP4 via `nvfp4_metal_lane_check`), so no macOS GPU/perf claim is made;
the CUDA dp4a kernel and its RTX-3060 numbers stay Windows-only. Still NOT a supported/certified
row and NOT quality-competitive (G3 NO-GO stands). The L4-metal invariant cells were already
`enforced` by BASALT (S1 upgrade over the prescribed na), so GABBRO's "flip L4 na→enforced"
step is a no-op — the pin was ahead of the GABBRO conductor. The internal fn name
`nvfp4_windows_only_check` is retained (optional rename follow-up; pub(crate), not a user
surface).

**D17 addendum 10 (2026-07-18, GABBRO M3-followup — NVFP4 wired through the Metal resident
lane).** M3 landed the `nvfp4_block_linear_row_ksplit_f32y_wire` Metal kernel and its
CPU-parity gate (PR #478); this follow-up wires it into the runtime so `Gemma4GpuRuntime`
actually runs NVFP4 layer projections on the GPU. **L1 (forward):** the resident forward's
`blocks_per_row` is made format-aware (`fmt.block_elements()`, 64 for NVFP4 vs 32 for
Q8_0/Q4_0) in `encode_gemma4_ffn`/`encode_gemma4_attention` — the old hardcoded `dim/32`
would have mis-strided NVFP4 rows; proven by `metal_gemma4_resident_nvfp4_forward_matches_cpu`.
**L2 (load admission):** the blanket `nvfp4_metal_lane_check` typed-refusal is lifted and
`Gemma4GpuRuntime::load` gains an `NVFP4` layer-fmt arm via the testable covered-set helper
`gemma4_metal_layer_fmt` (Q8_0/Q4_0/NVFP4; others refuse typed). **Safety (D17/T5 carried
into the GPU lane):** the resident lane reads NVFP4 wire bytes RAW via `WirePages`, bypassing
`WireQuant::new`'s NaN-sentinel scan — so the fail-closed guard is re-established as
`nvfp4_metal_sentinel_check`, run once the mmap is available: it scans every NVFP4 tensor's
UE4M3 scale bytes and refuses the `0x7F`/`0xFF` sentinels, matching the CPU wire lane; the
D-B2 sidecar check already runs up top. **Invariant matrix (L4-metal):** `I-nan-scale` flips
`na → enforced` (the new sentinel scan is the enforcing test), and `I-unknown-type`/`I-plat`/
`I-carveout` rebind to the new helper tests. **Scope unchanged:** the Metal GPU lane is
reached only via the macOS-only `gemma4-generate-gpu` subcommand (opt-in; `serve` still uses
the CPU wire lane), it is self-parity-proven against the CPU oracle but **not yet exercised
end-to-end with a real NVFP4 artifact and carries no perf claim**, and it remains NOT a
supported/certified row and NOT quality-competitive (G3 NO-GO stands). The live
`/api/capabilities` note + its ledger mirror are truthed-up here; the human support matrices
(SUPPORT_MATRIX/CAPABILITY/STATUS/COMPATIBILITY/README/`NVFP4_FORMAT`/DOCS) currently
**understate** the lane ("Metal GPU not yet wired") and are corrected in the M4 surface-
alignment pass — a conservative staleness, not an overclaim.

**D17 addendum 11 (2026-07-18, GABBRO — NVFP4 gemma4-E4B PROMOTED to a supported row; G3
gate REVISITED, not reversed).** Tim directed "make these supported models on the mac, do
whatever it takes"; then, shown that pure-NVFP4 quality is a knife-edge, "grind it" and chose
**pure NVFP4-mm**. Promotion basis (evidence bundle `qa/evidence-bundles/gabbro/support/`):
**(1) Decode-parity anchor** — NVFP4 CPU decode bit-exact on ARM (G-M1), Metal resident forward
== CPU oracle (`metal_gemma4_resident_nvfp4_forward_matches_cpu`), BASALT Leg B cross-engine
8/9+near-tie, and a fresh macOS llama.cpp `acd79d603` spot-check (both → "Paris"). **(2) End-to-end**
— first real-artifact run on the Metal resident lane produces coherent output; isolated 128-tok
decode **12.12 tok/s** (fastest of NVFP4/Q8_0/Q4_0, 1.45× the Q8_0 parent, 26% smaller).
**(3) Quality** — a NEW, disclosed current-engine teacher-forced re-eval (harness
`camelid gemma4-eval-pack`, validated: Q8_0 self-agreement 296/296, baseline token-identical to
BASALT's committed baseline for 8/9 prompts): NVFP4-mm **90.5%** (268/296) vs format-isolated
Q4K-mm **91.9%** (272/296) → gap **1.4pp ≤ the pre-registered 2.0pp GO tolerance = GO**, vs the
frozen G3's 4.1pp NO-GO. **The frozen Gate G3 NO-GO (88.5% vs 92.6%, engine `8038abba`) STANDS
as a recorded historical result — this is not a reversal but a disclosed re-measurement on the
current engine.** HONESTY (binding, Option-B claim-lint preserved): the pure-NVFP4 pass is a
**comparator-sensitive NEAR-TIE** (against a 92.6%-level Q4_K comparator the gap is 2.1pp,
marginal NO-GO); the +6-match NVFP4 gain (88.5→90.5) is a real current-engine decode-fidelity
improvement (byte-identical file, near-identical baseline), the −2 on Q4K reflects an ARM-quantized
comparator (Q4_K's float-search quantizer is ISA-sensitive; NVFP4 RTN is not). imatrix is proven
dead for NVFP4 (`ggml-quants.c GGML_UNUSED(quant_weights)`). No surface may claim NVFP4 is
quality-competitive-beyond-tolerance or better than Q4_K; it stays a **space/speed** quant. New
row `gemma4_e4b_it_nvfp4`: `supported_exact_row_smoke`, scope
`exact_row_gpu_resident_raw_decode_parity_smoke_only`, `full_support_status` blocked (no bounded
packs, perf/RSS, portability, arbitrary templates). Ledger + capabilities builder + the committed-set
/ scope guard tests in `tests/gemma4_capabilities.rs` updated in sync; a robust alternative
(ffn_down→Q8_0 hybrid, 91.9% = Q4_K) is recorded but NOT shipped (needs Metal per-tensor-fmt).

**Micro-decisions (Amendment 3):**

- **§9.1 — runtime platform gate over a `#[cfg]` wall:** NVFP4 admission refuses on
  non-Windows targets via `cfg!(target_os = "windows")` INSIDE ordinary code, in both
  lanes (`runnable::admit` after the D-B3/D-B2 checks; the gemma4 wire-lane load via
  `nvfp4_windows_only_check`, after the sidecar check), with the named TK2 error
  "NVFP4 is Windows-only in this release; see SUPPORT_MATRIX". Rationale: the crate
  compiles identically on every target (no cfg-walled decode code rotting unseen on
  platforms that never build it), the refusal is a typed, testable error rather than a
  missing symbol, and cfg-gated twin tests pin BOTH sides on the CI legs that actually
  run there (ubuntu/macos assert the refusal; Windows asserts admission). **[NARROWED by
  addendum 9 (GABBRO M2, 2026-07-18): the gate now admits on `windows` OR `macos`; only
  non-(Windows/macOS) legs assert the refusal, and the TK2 text is now "NVFP4 is
  Windows/macOS-only in this release; see SUPPORT_MATRIX".]**
- **§2.4 — invariant-matrix enforcement mechanism (S2):** `include_str!` file
  binding + test-fn name assertion + fixture-tripping references, all in
  `tests/invariant_matrix_binding.rs` against `qa/invariant_lanes.json`
  (schema `qa/invariant_lanes.schema.json`, `camelid.invariant-lanes/v1`).
  Every file an enforced cell names is `include_str!`-bound (a rename/move
  breaks the BUILD); the meta-test validates the matrix against the schema
  (hand validator, serde_json, no new deps), asserts full population (no empty
  cells), asserts every enforced cell's test-fn NAME appears in its bound
  file's text, and trips the committed `tests/fixtures/gguf/` fixtures (or
  references the S1 per-lane test that already trips them — no duplicate
  execution). HONESTLY NOTED: this is file-level compile-time + name-level
  test-time — a fn RENAME fails the meta-test, not the build. **Disclosed
  deviation, not a self-granted §2.4 pass** (SHA_E review correction): §2.4
  permits substitutions only if strictly STRONGER than the prescribed
  compile-time fn reference, and on the fn-rename axis test-time detection is
  weaker. The prescribed mechanism is infeasible for private `#[cfg(test)]`
  unit-test fns and cfg-twinned tests without restructuring every suite; the
  practical gap is one CI stage (the meta-test runs in the same `ci-gate`
  suite as the build — either way the PR goes red before merge). This
  deviation is flagged for Tim's explicit nod at the G3 gate PR. Open-cell teeth: the
  meta-test fails if the Phase-4 CUDA refusal text (or the P2b test-anchoring
  marker) vanishes while a cell still cites that phase as open — ratchet
  R3/R4 are enforced, not aspirational.

  **SIGNED — Tim, 2026-07-17.** The disclosed test-time-mechanism deviation
  (fn-rename caught by the meta-test, not the build) is accepted as-is; no
  restructuring required. The invariant-matrix enforcement stands as shipped.

## D-B6 — Admission of the pilot's BF16 tensor: Option A (2026-07-17, SIGNED)

**SIGNED by Tim, 2026-07-17: Option A** (draft `scratchpad/basalt-db6/D-B6_DRAFT.md`).
Add BF16 to the runnable lane's covered quant set as an exact-decode type (the dispatch arm
reuses the existing lossless `crate::tensor::decode_bf16_tensor` — no new numeric code), so
legitimate mixed-type NVFP4 files (the gemma-4-E4B pilot's single `per_layer_model_proj`
BF16 tensor) admit under the existing **whole-file** coverage model rather than requiring a
per-tensor admission predicate (B/C, rejected — they mirror load-path code, the CAIRN
anti-pattern, and carry the admit-then-fail hazard D-B3 excluded). TK3's conditional ("IF
mixed-type files REQUIRE per-tensor coverage") is not met; the pilot's mixed type is
coverable losslessly at ~5 lines. Signed riders (all land in the implementation PR):
(1) BF16 dequant-parity fixture vs the pin (M-B5 exit condition (a)); (2) L1 `I-carveout`
matrix re-sign — the `gemma4_nvfp4_with_bf16_refuses_on_bf16` refusal-pin test inverts to an
admission-pin test, off-Windows §9 twin retained; (3) surface rows (README/COMPAT/
SUPPORT_MATRIX/ledger) updated in the same PR — **this REVERSES the #475 execution-truth
clause** ("refuses generic runnable admission on its BF16 tensor") to full admission;
(4) disclosures: the SHA_E `ends_with("IQ4_XS")` generic-message pin retires (sanctioned
covered-set change, IQ4_XS precedent); BF16 runnable loads f32-materialize (memory doubling
— ornith-9B BF16 ~17.9→~35 GB, guarded by the fit advisor as a host limit per the
runnable-lane memory policy, not an engine refusal; M-B5 HOLD stands on host grounds);
(5) NO support-status / smoke / `oracle_qualified` / eligibility change — the pilot's bucket
stays `not_anchored`. Implementation is a bounded pass; scheduled separately.

## D18 — DROVER: finishing the `camelid chat --agent` coding agent (2026-07-22)

**Decision:** take agent mode from a correct sandboxed tool-calling loop to a daily-drivable
terminal coding agent, in gated phases over the existing `run_loop` / `tools.rs` / `Sandbox` /
`LiveDriver` machinery. No token-level generation path is touched and no parity receipt is
affected; DROVER is client-side loop, UX, and tests. Reconnaissance and the signed phase roster
are in `DROVER_RECON.md`.

**Order (signed 2026-07-22):** P0.5 → G1 project context → G2 compaction → G7 MCP → G3 plan
surface → G8 web search + slash commands → G4 headless exec → G5 checkpoint/diff/undo →
G6 session resume. G7 precedes G3/G8 because `ToolSpec::name`/`description` are `&'static str`;
runtime-registered MCP tools retype them, so adding static tools first means writing them twice.

- **D-DROVER-1 — Compaction retains the safety spine.** The system prompt and the
  data-not-commands rule are never compacted, and a summary of untrusted output is itself
  untrusted. The budget is 80% of the row's *validated context envelope* (8192 on the pinned
  row), not the model's `n_ctx`: a transcript past the validated window is outside the row's
  support claim even where the server accepts it.
- **D-DROVER-2 — `agent exec` is a subcommand, not a binary.** Tri-state exit (0 answered /
  1 failed-blocked / 3 inconclusive), non-interactive approver, `--yolo` refused under
  `CAMELID_PRODUCTION`.
- **D-DROVER-3 — MCP is opt-in, Confirm-tier, production-off, namespaced, output-untrusted.**
  stdio transport only in v1, servers declared solely by a workspace-local `camelid.mcp.json`.
  An external MCP config is not read: importing server declarations the user never placed in
  this workspace contradicts the opt-in posture.
- **D-DROVER-4 — Resume re-validates model identity** and never re-executes historical tool
  calls. A resumed transcript is replayed as context; its old tool results stay untrusted.
- **D-DROVER-5 — Checkpoint and session stores are content snapshots under the sandbox root**
  and pass the same canonical-prefix jail check as any other path. The git-stash variant is
  rejected: one code path is easier to prove, and the agent must not mutate git state the
  sandbox does not own.
- **D-DROVER-6 — The promotion harness attests the baseline prompt.** `agent-eval` and
  `agent-orchestration-eval` call `agent::system_prompt`, which carries no workspace content;
  only user-facing lanes call `system_prompt_with_project`. The ledger states `tool_capable` is
  earned only by those receipts, so the prompt they attest must be fixed and reproducible rather
  than whatever `CAMELID.md` a workspace happens to carry. Enforced by construction — the
  project block arrives through a separate function, so a gate lane cannot pick it up by
  forgetting a flag — and pinned by `baseline_prompt_never_carries_project_context`.

**Invariant across every phase:** each new road into the model — a project file, a plan, a
compaction summary, a resumed transcript, MCP output, a search result — is untrusted data. The
day one of them can change an approval tier, the sandbox, or the `tool_capable` gate, DROVER has
regressed the thing it exists to protect.

**D18 addendum 1 (2026-07-22, promotion to Supported (experimental)):** agent mode's
"(preview)" label drops in favor of **Supported (experimental)**, scoped by the new
"Agent mode" section of `COMPATIBILITY.md` and evidenced by
`qa/evidence-bundles/agent-mode-supported-experimental-20260722/` (a live
compaction-boundary run and a live end-to-end MCP round-trip on the pinned row, plus the
re-minted 3-case agent-eval PASS). The full (non-experimental) claim is deliberately NOT
made; its remaining preconditions are listed in the ROADMAP lane entry. Two recorded
divergences ride along unresolved by choice: (1) `LoopEnd::StepCapped` maps to subagent
exit 1 ("failed") but to `agent exec` exit 3 ("inconclusive") — the subagent mapping
predates the tri-state contract and changing a shipped gate lane's exit codes does not
belong in a promotion window; (2) the `api_features` ledger row for the feature axis is
deferred with the FULLY milestone. The syscap harness's direct executor is now
allowlisted to its two battery tools, closing the one execute path outside the
approval loop.

**D18 addendum 2 (2026-07-22, plan auto-completion on a final answer):** the interactive
front ends (`agent::run_agent` line renderer, `agent_tui`) now call `plan::complete_all()` when
`run_loop` returns `LoopEnd::Answered`, so a successful run does not leave the plan panel showing
e.g. "0/1 done" because a small model set a step `[~]` in-progress, did the work, and answered
without the closing `update_plan` call. This lives in the front ends, NOT in `run_loop`, so the
promotion/gate harnesses (`agent-eval`, `agent-orchestration-eval`, the subagent worker) stay
byte-identical and untouched by plan state. It does not weaken the G3 inertness invariant: the
loop still never *reads* a plan step to decide anything, and the plan still cannot change a tier,
the sandbox, or a tool choice — this only *writes* the model's own to-do to done at the natural
completion point. It fires only on `Answered`; a step-capped, aborted, repeated, or errored run
leaves the plan as the model left it, which is the honest signal that the run did not finish.

**D18 addendum 3 (2026-07-24, reconciling the subagent exit code with the tri-state
contract):** the divergence D18 addendum 1 parked -- `LoopEnd::StepCapped` (and `Repeated`)
mapping to subagent exit 1 ("failed") but to `agent exec` exit 3 ("inconclusive") -- is now
resolved in favour of the tri-state contract `agent exec` defines (`D-DROVER-2`). A step-capped
or repeating run means the run ran out of budget or got stuck, not that the task failed; a
re-run with more steps or a different model may still answer, so it is *inconclusive*, not a
failure. Only a driver error remains a hard failure. The fix is structural: both lanes previously
classified `LoopEnd` independently (and a third copy lived in a test), so they could drift; they
now share one `agent::RunOutcome` classifier that owns the whole `0` completed / `1` failed / `3`
inconclusive space (`2` stays reserved for the `tool_capable` refusal). The subagent worker carries
that typed outcome beside its serialized report until it chooses the process exit code; status text
is output, not control-flow input. Adding a `LoopEnd` variant forces a deliberate tier in that one
exhaustive match. No token-level path, parity receipt, or approval tier is touched; the subagent
worker's exit code moves from 1 to 3 for a step-capped or repeating child, matching what a headless
`agent exec` already reported. Pinned by `exec_exit_codes_are_tri_state`,
`run_outcome_status_and_exit_contract_is_pinned`, and the real canned-worker regression
`worker_step_cap_is_inconclusive_in_status_and_exit_code`. The same typed outcome
also drives the serialized result-file `status` token, so a step-capped or repeating
child now reports `inconclusive` rather than `failed` to `check_subagent_status` and
`/subagents` — the parent model sees the reclassification, not just the exit code.

## D19 — HARDPAN: Windows agent-terminal parity fixes (2026-07-23)

The full-screen agent terminal shipped assuming a Unix terminal; HARDPAN
(`qa/hardpan/REPRO.md`) reproduced the Windows-specific failures live before
fixing them, and every ruling below is receipt-backed rather than reasoned.

**Exec surface (Class A).** (1) `run_shell` drains stdout/stderr on per-pipe
reader threads — the poll-then-`wait_with_output` shape wedged any child
emitting more than the 64 KiB per-pipe quota and reported a *successful*
command to the model as a timeout with its output discarded. The fix lifts
`run_windows_command`'s existing pattern verbatim; it is cross-platform (Linux
has the same quota) and the regression tests run on both CI legs. (2) A
kill-on-close job object tears down `run_shell`'s whole process tree on
timeout. Ruled at GATE 0: the `CREATE_NO_WINDOW` half of the original finding
was DROPPED as unreachable (no console-less surface registers `run_shell`), so
no creation flags land. The negative control exposed a real coupling worth
recording: without the job object, an orphaned grandchild holds the inherited
pipe write-end, so the W1 drain-thread join wedges for the orphan's entire
lifetime (274.79 s vs 4.76 s on the same test). Tree teardown is what makes a
drained timeout path prompt — the two fixes are load-bearing together.
(3) `run_windows_command` feeds PowerShell 5.1 a base64-inside-pure-ASCII
stdin preamble and re-raises `$LASTEXITCODE`. Mechanism correction that
matters for future work: a `CREATE_NO_WINDOW` child gets a fresh windowless
console on the OEM code page (437 here), NOT "no console" and NOT the ANSI
page — which is why `win_console::init()` can't help and why ASCII (identical
under every code page) is the only airtight envelope. Deliberately NOT
`-EncodedCommand` (would reintroduce the ~32 KiB command-line ceiling) and
deliberately NOT "prefer pwsh.exe" (absent on the reference host — an
untestable branch does not ship; 5.1 is pinned by absolute path and made
UTF-8-correct by the preamble). Residual, asserted in tests: `$LASTEXITCODE`
tracks only the last native command.

**Terminal front ends (Class B).** (4) One shared RAII teardown guard +
chaining panic hook (`term_guard.rs`) covers both TUIs; it restores only for
panics on the arming thread, because agent-loop worker panics are survivable
today and yanking a live TUI out of its alternate screen would be a
regression. Known limit: non-unwinding deaths bypass any hook. (5) The W5
paste fix deviates from the campaign's prescription, recorded as amendment
A12: a Ctrl+V arm alone cannot fix paste on default Windows hosts, because the
terminal intercepts the chord and injects the clipboard as keystrokes — the
Phase 0 event log contains no Ctrl+V event at all. The shipped fix is
three-part: non-fatal `EnableBracketedPaste`, a Ctrl+V clipboard-read arm (new
read-only `win_clipboard.rs`; the OSC 52 write path is untouched), and a
pasted-Enter guard — an Enter processed while more input is already queued is
a pasted newline, not a submit. (6) AltGr: a `Char` carrying CONTROL|ALT is a
composition and inserts; CONTROL alone stays a chord — one shared predicate,
no per-character cases.

**Struck / null, on evidence.** W7 (`/copy` lies) did not reproduce — OSC 52
is honoured by this host's terminals in 2/2 real-console runs, so the tool
told the truth and `clipboard.rs` is untouched. W9 (LineWriter buffering) is a
measured null (8× the writes cost 1.07× the time; ratatui writes diffs and
already flushes once per frame) — receipt under
`qa/evidence-bundles/hardpan-windows-terminal-20260723/`, no product change,
per the SIROCCO precedent for unmeasured perf claims.

Boundaries held: no agent-loop, tool-schema, approval-policy, or parity-lane
change; no new crate (two `windows-sys` features only, justified by the
clipboard *read* path); Unix behavior preserved (the one deliberate
cross-platform fix, W1, is a strict improvement there). Follow-ups on the
amendment log: `win_uia.rs` shares W3(a)'s output leg (A11/W10); the real-TUI
CERT rows of `qa/hardpan/MANUAL_CHECKLIST.md` are owed post-merge
(merge-ahead authorized 2026-07-23).

## D20 — Windowed-attention architectures: routing invariants (2026-07-30)

Binding for every architecture whose attention is windowed (`gemma3` today;
`model::arch_has_windowed_attention` is the predicate). Adopted during the
gemma3→Metal campaign (`GEMMA3_METAL_CONDUCTOR.md` §§10-13), where each of the
three rules below was written *after* an adversarial review found the code
without it. All three survived two review passes.

### D20.1 — Routing is capability- and quant-aware, never a bare arch list

A windowed architecture is not "runnable-only" or "resident-capable" as a
property of its name. It is one or the other **per host and per file**:

- `model::arch_requires_runnable_bridge(arch)` is the live predicate; its pure
  half `..._given(arch, capable)` makes the split unit-testable with no env and
  no device.
- `inference::windowed_arch_resident_host_available()` is the capability probe:
  macOS build **and** resident Metal decode enabled (live env; deterministic
  mode forces it off) **and not** CUDA-resident (the CUDA engine has no windowed
  forward) **and** a real Metal device.
- Admission is additionally pinned to the quant the receipt covers. For windowed
  archs `is_resident_quant` returns true only for Q8_0, with an explicit typed
  decline when any layer linear is not Q8_0. A gemma3 Q4_K_M would otherwise
  ride the Metal K-quant admission onto an F16-primary lane whose gather drops
  `embed_scale`, with no windowed receipt behind it.

Serve and the CLI direct lanes consult the *same* predicate, so they cannot
disagree about which lane a file gets. The reason this matters is not tidiness:
the window mask is the difference between correct and wrong above 512 tokens,
and "which lane" is therefore a correctness decision, not a performance one.

### D20.2 — The fail-closed is a CHOKE POINT, never a list of entry-point guards

The CPU dense forward has no window mask. It must therefore refuse windowed
architectures, and *where* the refusal lives is the whole decision. The first
implementation guarded the three CPU dense entry points that were reachable at
the time. Phase 3c's review then found that the routing flip had made four MORE
reachable — `forward_greedy_verify_chunk`, `forward_layer_range_from_hidden`,
`forward_worker_layers`, `ghost_forward_one_layer` — and that guarding those
four would simply leave the fifth. **Entry-point enumeration is not an
invariant; it is a list that goes stale on someone else's commit.**

So the guard moved to the narrowest point every CPU dense layer walk must pass
through. `ensure_windowed_arch_off_cpu_dense_layer` (a free function keyed on
`model::arch_has_windowed_attention`, i.e. parsed metadata rather than the arch
string, so a future windowed arch inherits it with no edit) returns a typed
`BackendError::UnsupportedModelArchitecture` naming both correct lanes, and it
is called from the only two per-layer dense forwards that exist:

| Choke point | Covers |
|---|---|
| `forward_layer_timed` | every decode-side CPU dense layer walk |
| `forward_prefill_layer_chunk_timed` | every prefill-side CPU dense layer walk |

The three session-level guards that remain (`batch prefill`, `layer-major
prefill`, `decode forward`) are **courtesy legs only**: they delegate to the
same function so the operator sees the lane that failed instead of the generic
per-layer message. Deleting one degrades an error string; deleting a choke-point
call is the actual regression.

The invariant is therefore "two per-layer forwards, both guarded", and it is
pinned as a completeness proof AND a revert detector in `src/inference/tests.rs`:
two choke-point tests (fail if either call is deleted), one non-windowed control
(proves the guard is what causes the refusal rather than the fixture), three
lane-name tests (fail if a courtesy leg is deleted — the message's lane changes),
and four entry-point tests asserting the inherited callers above are covered
*through* the choke point rather than by their own guards.

Why this matters more than most fail-closeds: a full-causal forward on a windowed
model does not crash. It returns fluent, plausible, wrong text.

### D20.3 — A plan OUTPUT can never be a routing INPUT

The execution plan describes what the engine decided. Feeding any part of that
description back into the decision makes the two circular and lets a single env
var mean two different things at once. Concretely: the resident-host probe once
consulted `CAMELID_MAC_Q8_REPACK`, a variable that is simultaneously an operator
opt-out and a plan-output label — so an operator disabling repack silently
changed which *lane* served the model, while the plan kept reporting a lane it
was no longer on. The probe now consults only inputs (platform, env gates,
device presence).

The corollary is that where the resident plan selection cannot fire, the plan
**fails closed** to `safe_q8_plan` with a windowed-arch reason rather than
advertising a CPU dense lane D20.2 forbids. A plan that names a lane the engine
would refuse to run is worse than no plan.

### D20.4 — A plan OUTPUT can never be a PLANNING input either (2026-07-31)

D20.3 removed the plan's own output from the *routing* probe. The same variable
was still an input to the *next plan*. `plan_for_model` read
`CAMELID_MAC_Q8_REPACK` live; a successful Metal-resident selection writes it to
`off`; `PlannerEnv::apply` sets it. So the second and every later plan in a
process read the first plan's output as an operator opt-out and failed closed to
`safe_q8_plan`. Serve plans once per model load, so from the second load onward
`/v1/health`, `/api/capabilities` and `/execution-plan` reported `cpu_reference`
/ `safe_cpu_decode` for a row the engine was decoding on the Metal resident lane
— reproduced 2026-07-31 at `56ff2eb3`: `gemma-3-1b-it-Q8_0.gguf` loaded through
`POST /api/models/load` disclosed `cpu_reference` at 51 tok/s measured (its CPU
bridge does ~0.2), and the *same* Llama 3.2 1B file disclosed
`metal_resident_q8_runtime` when loaded at startup and `cpu_reference` when
re-loaded. It reached every desktop user, because the desktop always loads its
model at runtime.

The decision: the planner reads operator opt-outs from a `PlannerEnv` snapshot
captured before any plan applied its `env_updates`, never from live environment.
`AppState` captures one at construction and every load plans against it. Keys the
planner does not write are deliberately NOT snapshotted — they are genuine
runtime inputs, and a stale copy of those would be its own disclosure bug.

This is a disclosure fix only. The plan cannot disarm the runtime gates, so no
lane, no routing decision and no output token changed; what changed is that the
disclosure now matches the lane that was already running.

### What this does not decide

Nothing here is a performance decision, and none of it constitutes a throughput
claim for any lane. `support_level` in the execution plan stays deliberately
platform-blind (`supported_exact_row_smoke_sub512` for the gemma3 row) because
that table is keyed on the row name alone and is reported on fallback hosts too;
the lane-aware context claim lives in `/api/capabilities`, which is the support
source of truth and states the lane it applies to.
