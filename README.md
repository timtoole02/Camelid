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
| **Agent mode** (preview) | `camelid chat --agent --model <gguf>` — approval-gated tool calls | Experimenting with tool-calling |

**Agent mode (preview).** `camelid chat --agent` is an approval-gated tool-calling loop that can
read, write, and search files and run shell commands, with opt-in URL fetch. File tools are confined
to a workspace root (`--workdir`, default the current directory; path escapes are refused), and the
network stays off unless you pass `--allow-net`. Tool results are treated as untrusted data, and
only models the compatibility ledger marks `tool_capable` are eligible (promoted only after a
`camelid agent-eval` PASS). Treat it as a preview and review every requested action.

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
| Linux x86_64 | Engine archive (`.tar.gz`) | CPU; CUDA buildable for supported NVIDIA setups |

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
