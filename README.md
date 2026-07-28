<div align="center">

# 🐪 Camelid

**Run supported GGUF language models locally with a Rust-native engine.**

Desktop app, browser chat, terminal UI, and an OpenAI-style API — all backed by the same local runtime.

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-64748b.svg)](#platform-support)

[Download][latest-release] · [Quick start](#quick-start) · [Supported models](#supported-models) · [Model compatibility](COMPATIBILITY.md) · [Documentation](DOCS.md) · [Contributing](CONTRIBUTING.md)

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
- **Hardware acceleration.** Native Metal on Apple Silicon and experimental Windows CUDA for exact, recorded NVIDIA paths, with a CPU fallback everywhere.
- **Evidence-backed compatibility.** Support is tied to an exact GGUF row and published validation artifacts, never a broad claim.

## Quick start

> **Before you begin.** The engine itself is a single download, but model files are large —
> roughly 1–8 GB each. Give yourself some free disk space and a few minutes for the first model
> to download.

### Option A — Desktop app (easiest)

#### Windows

1. Download the signed installer from the [latest release][latest-release]:
   - `Camelid.Desktop_<version>_x64-setup.exe` — signed installer; installs per-user, no admin rights.
   - `camelid-desktop-windows-x64.zip` — portable desktop app, no installation required.
2. Run it. The app installs per-user under `%LOCALAPPDATA%\Camelid Desktop`.
3. It bundles the CUDA runtime, so no separate CUDA Toolkit is required. Its experimental Windows
  CUDA path still requires a compatible NVIDIA driver (CPU otherwise), and it embeds the same
  engine as everything below. Windows CUDA evidence is limited to the exact rows and recorded GPU,
  driver, and CUDA versions in [COMPATIBILITY.md](COMPATIBILITY.md); it makes no general
  token-parity or throughput claim.

#### macOS Apple Silicon — command-line install

The macOS desktop app currently uses an ad-hoc signature and is **not notarized**. Until a
notarized release is available, the simplest honest installation path is to build the current
source and install it from Terminal. It requires macOS 12 or newer on Apple Silicon, the Xcode
Command Line Tools, [Rust](https://rustup.rs/), and Node.js 22 with npm.

```bash
git clone https://github.com/timtoole02/Camelid.git
cd Camelid
./scripts/install-macos-desktop.sh
```

The install script builds the web UI and Metal-enabled engine, creates an ad-hoc-signed app,
installs it as `/Applications/Camelid Desktop.app`, and launches it. If `/Applications` requires
administrator access, macOS asks for your password through `sudo`. To update later, run
`git pull` followed by `./scripts/install-macos-desktop.sh` again.

Downloaded models live in
`~/Library/Application Support/app.camelid.desktop/models` and are preserved when the app is
rebuilt or reinstalled. See [Camelid Desktop](camelid-desktop/README.md) for the sidecar design
and manual packaging details.

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

## Supported models

> [!IMPORTANT]
> **Camelid's model policy: exact rows, not families.** Support is granted to a specific *model file
> at a specific quantization*, validated token-for-token against a pinned llama.cpp reference and
> backed by a committed parity receipt. A neighboring size, a different quant, another upload of
> the "same" model, or a wider template does **not** inherit that support — it fails closed with a
> typed error rather than quietly producing unverified output. The boundary for each row, and what
> is explicitly *not* claimed, is pinned in [SUPPORT_MATRIX_v0.1.md](SUPPORT_MATRIX_v0.1.md) and
> [COMPATIBILITY.md](COMPATIBILITY.md).

### Start here

Not sure where to begin? Pick **Llama 3.2 3B** — the best balance of quality and size.

| Goal | Model | Pull id |
|---|---|---|
| Smallest end-to-end test (~1.2 GB) | TinyLlama 1.1B Chat Q8_0 | `tinyllama` |
| **Recommended first model** | Llama 3.2 3B Instruct Q8_0 | `llama32_3b` |
| Fits a 16 GB Apple Silicon Mac | Mistral 7B Instruct v0.3 Q8_0 | `mistral` |
| Reasoning + coding on a small budget | Qwen3 4B Q4_K_M | `qwen3_4b_q4` |

### Catalog models — `camelid pull`

Twenty-one curated rows ship in the `camelid pull` catalog. Run `camelid pull` with no argument to
print the list, or `camelid pull <id>` to download into `./models`. Ids resolve by **unique
substring**, so the short ids below are all you need — `camelid pull llama32_3b` works exactly like
the full `llama32_3b_instruct_q8_0`.

| Model | Quant | Arch | Size | Pull id | GGUF file |
|---|---|---|---:|---|---|
| **TinyLlama 1.1B Chat** | `Q8_0` | `llama` | 1.2 GB | `tinyllama` | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` |
| **Llama 3.2 1B Instruct** | `Q8_0` | `llama` | 1.3 GB | `llama32_1b` | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| **Llama 3.2 3B Instruct** | `Q8_0` | `llama` | 3.4 GB | `llama32_3b` | `Llama-3.2-3B-Instruct-Q8_0.gguf` |
| **Llama 3 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama3_8b` | `Meta-Llama-3-8B-Instruct.Q8_0.gguf` |
| **Llama 3.1 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama31_8b` | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| **Gemma 3 1B-It** | `Q8_0` | `gemma3` | 1.1 GB | `gemma3_1b` | `gemma-3-1b-it-Q8_0.gguf` |
| **Gemma 4 E2B-It** | `Q8_0` | `gemma4` | 5.0 GB | `gemma4_e2b` | `gemma-4-E2B-it-Q8_0.gguf` |
| **Gemma 4 E4B-It** | `Q8_0` | `gemma4` | 8.2 GB | `gemma4_e4b` | `gemma-4-E4B-it-Q8_0.gguf` |
| **Gemma 4 12B-It** — two-Mac distributed | `Q8_0` | `gemma4` | 12.7 GB | `gemma4_12b` | `gemma-4-12b-it-Q8_0.gguf` |
| **Gemma 4 26B-A4B-It QAT** — two-Mac distributed MoE | `Q4_0` | `gemma4` | 14.4 GB | `gemma4_26b` | `gemma-4-26B_q4_0-it.gguf` |
| **Qwen3 0.6B** | `Q8_0` | `qwen3` | 0.6 GB | `qwen3_0_6b` | `Qwen3-0.6B-Q8_0.gguf` |
| **Qwen3 1.7B** | `Q8_0` | `qwen3` | 1.8 GB | `qwen3_1_7b` | `Qwen3-1.7B-Q8_0.gguf` |
| **Qwen3 4B** | `Q8_0` | `qwen3` | 4.3 GB | `qwen3_4b_q8` | `Qwen3-4B-Q8_0.gguf` |
| **Qwen3 4B** | `Q4_K_M` | `qwen3` | 2.5 GB | `qwen3_4b_q4` | `Qwen3-4B-Q4_K_M.gguf` |
| **Qwen3 8B** | `Q8_0` | `qwen3` | 8.7 GB | `qwen3_8b` | `Qwen3-8B-Q8_0.gguf` |
| **Mistral 7B Instruct v0.3** | `Q8_0` | `llama` | 7.7 GB | `mistral` | `Mistral-7B-Instruct-v0.3-Q8_0.gguf` |
| **Phi-3-mini-4k-instruct** | `Q8_0` | `phi3` | 4.1 GB | `phi3` | `Phi-3-mini-4k-instruct-Q8_0.gguf` |
| **DeepSeek R1 Distill Qwen 7B** | `Q8_0` | `qwen25` | 8.1 GB | `distill_qwen` | `DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf` |
| **DeepSeek R1 Distill Llama 8B** | `Q8_0` | `llama` | 8.5 GB | `distill_llama` | `DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf` |
| **Qwen2.5 Coder 7B** | `Q8_0` | `qwen25` | 8.1 GB | `qwen25_coder` | `qwen2.5-coder-7b-instruct-q8_0.gguf` |
| **Cohere Command R v01** | `Q8_0` | `command-r` | 37.2 GB | `command_r` | `c4ai-command-r-v01-Q8_0.gguf` |

The two Gemma 4 rows marked *two-Mac distributed* are validated on the layer-sharded two-host lane —
they are memory-infeasible on a single 16 GB machine. Command R is listed for completeness; at
37 GB it needs a workstation-class host.

### Also parity-certified

These exact rows carry committed parity receipts but are **not** in the `camelid pull` catalog —
point `--model` at the file yourself. Several are local requantizations rather than a single
canonical upstream upload, which is precisely why they aren't offered as a one-command download.

| Model | Quant | Arch | GGUF file | Lane |
|---|---|---|---|---|
| **Llama 3.2 1B Instruct** | `IQ4_XS` | `llama` | `Llama-3.2-1B-Instruct-IQ4_XS.gguf` | First i-quant row — GPU-resident + CPU wire-streamed raw-decode parity smoke |
| **Llama 3.2 1B Instruct** | `Q4_K_M` | `llama` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | GPU-resident K-quant raw greedy decode (16/16 layers VRAM-resident) |
| **Llama 3.2 3B Instruct** | `Q4_K_M` | `llama` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | GPU-resident K-quant raw greedy decode (28/28 layers VRAM-resident) |
| **Llama 3.2 3B Instruct** | `Q5_K_M` | `llama` | `Llama-3.2-3B-Instruct-Q5_K_M.gguf` | GPU-resident Q5 certification, token-and-text identical at 1/5/50 |
| **Ornith 1.0 9B** | `Q8_0` | `qwen35` | `ornith-1.0-9b-Q8_0.gguf` | Hybrid DeltaNet + sparse attention on the runnable serve lane; `tool_capable` |
| **Ornith 1.0 9B** | `Q4_K_M` | `qwen35` | `ornith-1.0-9b-Q4_K_M.gguf` | Fully GPU-resident CUDA lane (in-house requant); `tool_capable` |
| **Ornith 1.0 9B** | `Q3_K_M` | `qwen35` | `ornith-1.0-9b-Q3_K_M.gguf` | Fully GPU-resident at 16K context on a 6 GiB card (imatrix requant) |
| **Ternary Bonsai 4B** | `TQ2_0` | `qwen3` | `Ternary-Bonsai-4B-TQ2_0.gguf` | Ternary 2.06 bpw, single-node CPU completion smoke (~3.1 GB RSS) |
| **Gemma 4 E4B-It** | `NVFP4` | `gemma4` | `gemma-4-E4B-it-NVFP4-mm.gguf` | BASALT / GABBRO NVFP4 pilot — Windows CUDA + macOS Metal, fails closed elsewhere |

Each row's exact envelope — which surfaces are certified, which contexts were checked, and what is
explicitly not claimed — lives in [SUPPORT_MATRIX_v0.1.md](SUPPORT_MATRIX_v0.1.md).
[COMPATIBILITY.md](COMPATIBILITY.md) is the complete, authoritative supported-row ledger.

## Ways to use Camelid

Every interface talks to the same local engine — pick whichever fits your workflow.

| Interface | How to start it | Best for |
|---|---|---|
| **Desktop app** | Windows installer, or `./scripts/install-macos-desktop.sh` on Apple Silicon | A native app with the engine bundled as a local sidecar |
| **Browser chat** | `camelid serve --model <gguf>` opens the web UI automatically | Everyday chatting in a familiar UI |
| **Terminal UI** | `camelid chat` — full-screen; `--plain` for a line REPL over SSH | Working entirely in the shell |
| **HTTP API** | OpenAI-style `/v1/*`, served alongside the UI on the same port | Wiring Camelid into your own apps |
| **Agent mode** | `camelid chat --agent --model <gguf>` — approval-gated tool calls | Coding-agent work in your own repo |
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
- [SUPPORT_MATRIX_v0.1.md](SUPPORT_MATRIX_v0.1.md) — the per-row support boundary and claim limits.
- [RECEIPTS.md](RECEIPTS.md) — reproducible validation receipts.
- [docs/benchmarks/BENCHMARKS.md](docs/benchmarks/BENCHMARKS.md) — performance measurements.
- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — how the engine is built.

Every row in [Supported models](#supported-models) is backed by that evidence chain. The serve lane
and evidence envelope for a selection of those rows:

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
| Windows x86_64 | Desktop installer, portable desktop ZIP, engine ZIP | Experimental CUDA on named exact rows and recorded NVIDIA configurations; CPU fallback |
| macOS Apple Silicon | Source-installed desktop app (ad-hoc signed), engine archive (`.tar.gz`) | Metal and CPU |
| Linux x86_64 | Engine archive (`.tar.gz`) | NVIDIA CUDA compiled in by default; CPU fallback |

CUDA is compiled into the default build on Windows and x86_64 Linux. On Windows, the GPU path is
experimental: it needs a compatible NVIDIA driver, but no separate CUDA Toolkit or build flag. Its
evidence is limited to the named exact rows and recorded GPU, driver, and CUDA configuration in
[COMPATIBILITY.md](COMPATIBILITY.md); other configurations are not covered by those parity or
throughput claims. The driver and NVRTC load dynamically at runtime; without a usable GPU the build
still runs CPU-only. `camelid serve --gpu auto|on|off` (or `CAMELID_GPU`) overrides the automatic
choice. Other Linux targets — aarch64, the Raspberry Pi — stay CPU-only, with CUDA opt-in via
`--features cuda`. Compiling the path in on Linux does not by itself extend the Windows row-level
parity claims.

## Documentation

Deeper references live alongside the code:

- [DOCS.md](DOCS.md) — documentation index.
- [COMPATIBILITY.md](COMPATIBILITY.md) — supported models and quantizations.
- [SUPPORT_MATRIX_v0.1.md](SUPPORT_MATRIX_v0.1.md) — exact-row support boundary and claim limits.
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
