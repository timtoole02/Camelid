<div align="center">

# 🐪 Camelid

**Run supported GGUF language models locally with a Rust-native engine.**

Desktop app, browser chat, terminal UI, and an OpenAI-style API — all backed by the same local runtime.

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-64748b.svg)](#platform-support)

[Download][latest-release] · [Quick start](#quick-start) · [Model compatibility](COMPATIBILITY.md) · [Documentation](DOCS.md) · [Contributing](CONTRIBUTING.md)

</div>

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

<div align="center"><sub>Camelid's local web UI — a dark, collapsed-rail chat surface, served straight from the engine binary.</sub></div>

Camelid loads GGUF models directly and runs inference on your own hardware. The tokenizer, model
loader, CPU kernels, and the Metal and CUDA execution paths are implemented in this repository and
distributed as a single Rust binary — no Python, Node.js, or Docker at runtime.

Camelid deliberately supports a curated set of exact model-and-quantization combinations. Each
supported row is validated token-for-token against a pinned reference before it is presented as
ready to use.

## Why Camelid

- **Local by default.** Models and inference stay on your machine unless you choose to expose the server.
- **One engine, several interfaces.** Desktop app, browser chat, terminal chat, or HTTP API — all the same runtime.
- **Nothing else to install.** The engine and web UI ship together as one binary.
- **Hardware acceleration.** Native Metal on Apple Silicon and CUDA on validated NVIDIA paths, with a CPU fallback everywhere.
- **Evidence-backed compatibility.** Support is tied to an exact GGUF row and published validation artifacts, never a broad claim.

## Quick start

> **Before you begin.** The engine itself is a single download, but model files are large —
> roughly 1–8 GB each. Give yourself some free disk space and a few minutes for the first model
> to download.

### Option A — Windows desktop app (easiest)

1. Download the signed installer from the [latest release][latest-release]:
   - `Camelid.Desktop_<version>_x64-setup.exe` — signed installer; installs per-user, no admin rights.
   - `camelid-desktop-windows-x64.zip` — portable desktop app, no installation required.
2. Run it. The app installs per-user under `%LOCALAPPDATA%\Camelid Desktop`.
3. It bundles the CUDA runtime, so GPU acceleration works with just the normal NVIDIA driver (CPU
   otherwise), and it embeds the same engine as everything below.

### Option B — Prebuilt engine (Windows, macOS, or Linux)

Prefer the command line? Download the engine archive for your platform from the
[latest release][latest-release] and unpack it.

| Platform | Archive |
|---|---|
| Windows x86_64 | `camelid-windows-x64.zip` |
| macOS Apple Silicon | `camelid-macos-arm64.tar.gz` |
| Linux x86_64 | `camelid-linux-x86_64.tar.gz` |

Every archive ships a matching `.sha256` for verification. On macOS, if Gatekeeper blocks the
binary, clear the quarantine attribute once: `xattr -d com.apple.quarantine ./camelid`.

### First chat in two commands

```bash
camelid pull llama32_3b
camelid serve --model models/Llama-3.2-3B-Instruct-Q8_0.gguf
```

That's it — your browser opens to a local chat at `http://127.0.0.1:8181`; start typing to talk to
the model.

`camelid pull` downloads the model into `./models`; run it with no argument to list the curated
catalog. `camelid serve` starts the engine, the OpenAI-style API, and the web UI on one port
(`127.0.0.1:8181` by default) and opens the browser automatically — pass `--no-open` to skip that.
Prefer the terminal? Run `camelid chat` instead for a full-screen chat UI over the same engine.

> [!WARNING]
> `camelid serve --addr 0.0.0.0:8181` makes the API and UI reachable by every device that can
> reach the host. Only bind `0.0.0.0` on a trusted network, behind your own access controls.

## Choose a model

Not sure where to start? Pick **Llama 3.2 3B** — it's a good balance of quality and size. Catalog
ids resolve by unique substring, so the short id below is all `camelid pull` needs.

| Goal | Model | Pull id |
|---|---|---|
| Smallest end-to-end test (~1.2 GB) | TinyLlama 1.1B Chat Q8_0 | `tinyllama` |
| Recommended first model | Llama 3.2 3B Instruct Q8_0 | `llama32_3b` |
| Larger, fits a 16 GB Apple Silicon Mac | Mistral 7B Instruct v0.3 Q8_0 | `mistral` |

All three are `Q8_0` quantizations. See [COMPATIBILITY.md](COMPATIBILITY.md) for the full set of
supported rows.

## Ways to use Camelid

Every interface talks to the same local engine — pick whichever fits your workflow.

| Interface | How to start it | Best for |
|---|---|---|
| **Desktop app** | Install `Camelid.Desktop_<version>_x64-setup.exe` (Windows) | A one-click, no-terminal setup |
| **Browser chat** | `camelid serve --model <gguf>` opens the web UI automatically | Everyday chatting in a familiar UI |
| **Terminal UI** | `camelid chat` — full-screen; `--plain` for a line REPL over SSH | Working entirely in the shell |
| **HTTP API** | OpenAI-style `/v1/*`, served alongside the UI on the same port | Wiring Camelid into your own apps |
| **Agent mode** | `camelid chat --agent --model <gguf>` — approval-gated tool calls | Coding-agent work in your own repo |
| **Code** (preview) | Open **Code** in the Web UI or Desktop app | Approval-gated coding with activity, diffs, undo, and history |
| **Workspace** (preview) | Open **Workspace** in the Web UI | Read-only, resumable analysis of a local folder |

**Agent mode — Supported (experimental).** `camelid chat --agent` is an approval-gated
tool-calling loop that can
read, write, and search files and run shell commands, with opt-in URL fetch. File tools are confined
to a workspace root (`--workdir`, default the current directory; path escapes are refused), and the
network stays off unless you pass `--allow-net`. Tool results are treated as untrusted data, and
only models the compatibility ledger marks `tool_capable` are eligible (promoted only after a
`camelid agent-eval` PASS). The supported scope — what is claimed, its boundary, and what is
explicitly not claimed — is pinned in [COMPATIBILITY.md](COMPATIBILITY.md), backed by the live-lane
bundle `qa/evidence-bundles/agent-mode-supported-experimental-20260722/`. Review every requested
action: approval is the contract.

**Code (preview).** Switch from Chat to Code in the Web UI or Desktop app, choose one local
workspace, and give Camelid a coding task. Code uses the terminal agent loop with a deliberately
scoped tool profile: read, list, literal search, plan, write, edit, and sandboxed shell. Every file
mutation and shell command pauses for an exact-action approval; network, GUI control, MCP tools,
subagents, and unattended execution remain unavailable. Automatic checkpoints power the Changes
view and guarded single-step undo without touching git. Completed turns are stored in the local
Workspace SQLite store and appear as coding history in the left rail. Code fails closed unless the
loaded exact model row is supported and has earned `tool_capable: true`.

**Workspace (preview).** Choose one local directory and ask follow-up questions across a durable
conversation. Workspace can only list, read, and search within that canonical root; writes, shell,
network, GUI control, and subagents are unavailable. File inventories are grounded in observed
directory entries, and reversible compaction keeps long threads within an exact context budget.
Workspace requires a loaded exact model row that has earned `tool_capable: true`.

The same read-only Workspace is available from a terminal while `camelid serve` is running:

```bash
camelid workspace ask . "Which files configure authentication?"
camelid workspace threads .
camelid workspace show workspace-123 --workspace .
camelid workspace ask . "What changed our conclusion?" --thread workspace-123
camelid workspace compact workspace-123 --workspace .
camelid workspace compact workspace-123 --workspace . --undo
camelid workspace delete workspace-123 --workspace .
```

Use `camelid workspace --json ...` for compact JSON; `ask` emits one JSON event per line. The CLI
is a client of the existing Workspace API, not a second agent: it uses the same three-tool profile,
canonical root confinement, SQLite/FTS5 threads, grounding checks, cancellation behavior, and exact
context budget as the web UI.

Browser authorization remains same-origin. At each loopback server start Camelid also rotates a
256-bit Workspace CLI bearer credential and stores it in the current user's runtime directory
(`%LOCALAPPDATA%\camelid\runtime` on Windows, `$XDG_RUNTIME_DIR/camelid/runtime` or
`~/.cache/camelid/runtime` on Unix). Unix files are created mode `0600`; Windows files inherit the
current user's LocalAppData ACL. `CAMELID_WORKSPACE_TOKEN_FILE` can override the location. The token
is never accepted on a non-loopback listener or with a non-loopback `Host`, and a clean shutdown
removes it when destructors run. After a crash or forced termination a stale file may remain, but
no server is present to honor it; the next server replaces it only after binding the same loopback
address and completing startup model loading. This capability protects against browser cross-site
requests; like the model files themselves, it does not defend against another process already
running as the same OS user.

With `--allow-net` the agent also gets `web_search` (ranked title/url/snippet results) alongside
`http_fetch`. Results are untrusted data — reading one is a separate, separately-approved
`http_fetch`. Point it at a different engine with `CAMELID_SEARCH_URL` (a template containing
`{query}`).

Every file the agent writes or edits is snapshotted first, so `/diff` shows what it changed,
`/undo` reverts the last change, and `/checkpoints` lists them. Snapshots are file copies under
`.camelid/checkpoints/` in the workspace — the agent never touches your git state.

`/save <id>` and `/resume <id>` carry an agent session across restarts, storing the transcript and
plan under `.camelid/sessions/`. A resumed transcript is replayed as context and never re-executed;
"always allow" grants are listed but never restored from a file; and resume is refused if the
active model is not the one that recorded it, or is no longer marked `tool_capable`.

In-session: `/init` scaffolds a `CAMELID.md`, `/plan` shows the agent's current checklist, `/copy`
puts the last answer on the clipboard, and `/help` lists the rest.

**Headless.** `camelid agent exec "<goal>" --model <gguf>` runs one goal to completion with no
prompts, prints the answer to stdout (progress goes to stderr), and exits 0 answered / 1 failed /
3 inconclusive. With no operator to approve anything, every gated tool is denied unless you pass
`--today-is-a-good-day-to-die` (alias: `--yolo`).

**MCP servers (opt-in).** `--allow-mcp` loads the servers declared in a `camelid.mcp.json` at the
workspace root (stdio transport) and offers their tools alongside the native ones, namespaced
`mcp__<server>__<tool>` so none can shadow a built-in:

```json
{ "servers": { "git": { "command": "uvx", "args": ["mcp-server-git"] } } }
```

An MCP server is third-party code, so every MCP tool is classified exec-tier — always approval-gated,
and *not* promoted by `--auto-approve` — its output is treated as untrusted data like any other tool
result, and the whole feature is refused under `CAMELID_PRODUCTION`. A server that fails to start or
never answers is dropped with a message; it does not stop your session.

Drop a `CAMELID.md` (or `AGENTS.md`) at the workspace root to tell the agent about your project —
build commands, layout, conventions. It is loaded into the agent's context as reference material,
fenced and labelled as untrusted: it can inform the agent, but it cannot grant permissions, change
an approval tier, or widen file access, and text inside it asking for any of those is ignored.

## Call the API

The served model id comes from the GGUF's `general.name`. Run `GET /v1/models` to read the exact
id, then send a standard chat-completions request:

```bash
curl http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama 3.2 3B Instruct",
    "messages": [{"role": "user", "content": "Explain why local inference is useful."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

## How support is validated

Camelid's core commitment is that every supported claim is backed by reproducible evidence.

Support is granted per **exact GGUF row** — a specific model file, at a specific quantization, on a
specific execution path. Each row is validated token-for-token against a pinned llama.cpp reference
before it is presented as supported. Models outside that set fail closed with a typed error rather
than silently producing unverified output, and experimental lanes are labeled separately and do not
inherit supported status.

The authoritative records live in the repository:

- [COMPATIBILITY.md](COMPATIBILITY.md) — the supported-row ledger.
- [RECEIPTS.md](RECEIPTS.md) — reproducible validation receipts.
- [docs/benchmarks/BENCHMARKS.md](docs/benchmarks/BENCHMARKS.md) — performance measurements.
- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — how the engine is built.

A selection of currently supported exact rows is below; [COMPATIBILITY.md](COMPATIBILITY.md) is the complete, authoritative ledger.

| Model row | Quant | Serve lane | Evidence |
|---|---|---|---|
| TinyLlama 1.1B Chat | Q8_0 | single-node | Current verified gate |
| Llama 3.2 3B Instruct | Q8_0 | single-node | Exact-row smoke + bounded context 512→8192 |
| Mistral 7B Instruct v0.3 | Q8_0 | single-node | Exact-row smoke + bounded context 512→8192 + GPU/CPU parity |
| Llama 3 8B Instruct | Q8_0 | single-node | Exact-row + bounded context 512→2048 |
| Qwen3 4B | Q8_0 | single-node | Exact-row ChatML parity (thinking-disabled) |
| Gemma 4 E2B-It | Q8_0 | single-node | 5/5 greedy parity (CPU + Metal) |

## Build from source

Camelid builds with a pinned toolchain (see [rust-toolchain.toml](rust-toolchain.toml)). The web UI
lives in `frontend/` (React/Vite) and is embedded into the binary at build time.

```bash
(cd frontend && npm ci && npm run build)
cargo build --release --locked --bin camelid
```

rustup reads the pinned toolchain automatically, so a standard Rust install is enough. See
[docs/CONTRIBUTOR_QUICKSTART.md](docs/CONTRIBUTOR_QUICKSTART.md) to get set up.

## Platform support

Camelid ships for three platforms today.

| Platform | Distribution | Acceleration |
|---|---|---|
| Windows x86_64 | Desktop installer, portable desktop ZIP, engine ZIP | NVIDIA CUDA on validated paths; CPU fallback |
| macOS Apple Silicon | Engine archive (`.tar.gz`) | Metal and CPU |
| Linux x86_64 | Engine archive (`.tar.gz`) | NVIDIA CUDA compiled in by default; CPU fallback |

CUDA is compiled into the default build on Windows and x86_64 Linux, so a machine with the normal
NVIDIA driver gets the GPU path with no build flags and no CUDA SDK (the driver and NVRTC load
dynamically at runtime; without a GPU the build still runs CPU-only). `camelid serve --gpu
auto|on|off` (or `CAMELID_GPU`) overrides the automatic choice. Other Linux targets — aarch64, the
Raspberry Pi — stay CPU-only, with CUDA opt-in via `--features cuda`. Note the validated GPU parity
rows in [COMPATIBILITY.md](COMPATIBILITY.md) were recorded on Windows; compiling the path in on
Linux does not by itself extend those row-level parity claims.

## Documentation

Deeper references live alongside the code:

- [DOCS.md](DOCS.md) — documentation index.
- [COMPATIBILITY.md](COMPATIBILITY.md) — supported models and quantizations.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — configuration reference.
- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — engine internals.
- [docs/benchmarks/BENCHMARKS.md](docs/benchmarks/BENCHMARKS.md) — performance measurements.
- [docs/VALIDATION_MATRIX.md](docs/VALIDATION_MATRIX.md) — validation coverage.
- [RECEIPTS.md](RECEIPTS.md) — reproducible validation receipts.
- [ROADMAP.md](ROADMAP.md) — what's planned next.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md) first, and start with
[docs/CONTRIBUTOR_QUICKSTART.md](docs/CONTRIBUTOR_QUICKSTART.md).

## License

Camelid is released under the [MIT License](LICENSE).

Camelid's tokenizer, compatibility layouts, and validation are checked against llama.cpp
(MIT, © the ggml authors), which serves as the reference oracle for supported rows. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
[release-badge]: https://img.shields.io/github/v/release/timtoole02/Camelid?display_name=tag
[latest-release]: https://github.com/timtoole02/Camelid/releases/latest
