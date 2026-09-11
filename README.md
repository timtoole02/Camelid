<div align="center">

# 🐪 Camelid

**Local AI, powered by Rust.**

Run supported GGUF models on your own hardware through a desktop app, browser chat, terminal, or OpenAI-compatible API.

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-64748b.svg)](#platform-support)

[Download][latest-release] · [Quick start](#quick-start) · [Models](#supported-models) · [Documentation](DOCS.md) · [Contributing](CONTRIBUTING.md)

</div>

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

<div align="center"><sub>Camelid's local web UI—a dark, collapsed-rail chat surface served directly from the engine binary.</sub></div>

## Why Camelid?

- **Private local inference.** Run language, vision, and embedding models on your own hardware, offline after downloading the model files.
- **One Rust engine.** The engine and web UI ship as a single binary, with no Python, Node.js, or Docker required at runtime.
- **Hardware acceleration.** Use Metal on Apple Silicon, CUDA on supported NVIDIA paths, or CPU fallback on Windows, macOS, and Linux.
- **Tested compatibility.** Supported model files and quantizations are validated against a pinned llama.cpp reference, with explicit limits for each tested configuration.

## Quick start

Model downloads are typically 1–8 GB. For the simplest setup, install the desktop app and download a model from its **Models** page.

### Desktop app

**Windows 10 or 11 (x86_64):**

```powershell
irm https://raw.githubusercontent.com/timtoole02/Camelid/main/scripts/get-desktop-windows.ps1 | iex
```

This installs the signed app per user and bundles the CUDA runtime. A compatible NVIDIA driver is still required for CUDA; otherwise Camelid uses the CPU.

**macOS 12 or newer (Apple Silicon):**

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/Camelid/main/scripts/get-desktop-macos.sh | bash
```

This installs Camelid Desktop in `/Applications`. Run the same command again to update without removing models or settings.

Portable downloads and engine archives for Windows, macOS, and Linux are available from the [latest release][latest-release]. See the [desktop documentation](camelid-desktop/README.md) for manual installation and packaging details.

### Command line

After downloading and unpacking an engine archive, start a browser chat with:

```bash
camelid pull 3b_instruct_q8
camelid serve --model models/Llama-3.2-3B-Instruct-Q8_0.gguf
```

Camelid opens `http://127.0.0.1:8181`. Use `camelid chat` for the terminal UI, or add `--no-open` to run the server without opening a browser.

Run `camelid pull` without an argument to list the curated model catalog.

> [!NOTE]
> To chat from another device, see the [remote browser chat guide](docs/REMOTE_CHAT.md). Non-loopback listeners require authentication and either TLS or an explicit cleartext acknowledgement.

## Supported models

Start with one of these five options. Support applies to exact model files, quantizations, and tested execution paths; see the [compatibility ledger](COMPATIBILITY.md) for each model's limits.

| Best for | Model | Pull ID |
|---|---|---|
| **First chat** | Llama 3.2 3B Instruct Q8_0 | `3b_instruct_q8` |
| Small end-to-end test (~1.2 GB) | TinyLlama 1.1B Chat Q8_0 | `tinyllama` |
| Compact Windows CPU or tested M4 Metal chat (~2.9 GB) | LFM2.5 2.6B Q8_0 | `lfm2_5_2_6b` |
| Local embeddings (CPU) | Nomic Embed Text v1.5 Q8_0 | `nomic` |
| Chat on a 16 GB Apple Silicon Mac | Mistral 7B Instruct v0.3 Q8_0 | `mistral` |

Llama 3.2 3B and LFM2.5 have support limited to the documented exact-file smoke tests; broader context, hardware, and behavior do not inherit that support.

[Browse the full catalog and validation details](docs/MODELS.md), including coding, vision, and experimental models. Gemma 4 12B-It QAT Q4_0 is available for a tested 16 GB M4 Mac, with **validation pending**. Download availability does not imply verified support.

<a id="full-camelid-pull-catalog"></a>
<a id="multimodal-image-chat"></a>
<a id="embeddings-and-reranking"></a>

For [image chat](docs/MODELS.md#multimodal-image-chat) and [embeddings and reranking](docs/MODELS.md#embeddings-and-reranking), see the model guide for supported files and setup.

## Ways to use Camelid

| Interface | Start it with | Best for |
|---|---|---|
| **Desktop app** | Install from [Quick start](#quick-start) | Native app with bundled engine |
| **Browser chat** | `camelid serve --model <gguf>` | Everyday local chat |
| **Terminal UI** | `camelid chat` | Shell and SSH workflows |
| **HTTP API** | Start `camelid serve` | Chat, image input, embeddings, and reranking |
| **Agent mode** | `camelid chat --agent --model <gguf>` | Approval-gated tools in a repository |
| **Workspace** (preview) | Open **Workspace** in the web UI | Read-only analysis of a local folder |

Agent mode confines file tools to a workspace root and keeps network access off unless enabled. Workspace is read-only and resumable. Both require a model marked `tool_capable` in the compatibility ledger. Review the [agent documentation](DOCS.md) and every requested action before enabling additional tools or network access.

## OpenAI-compatible API

`camelid serve` exposes the browser UI and API on the same port. Read the loaded model ID from `GET /v1/models`, then call the chat-completions endpoint:

```bash
curl http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama 3.2 3B Instruct",
    "messages": [{"role": "user", "content": "Why is local inference useful?"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Camelid also supports `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, streaming, local image input on supported VLM rows, function tools, structured text formats, conversations, and optional local SQLite storage. The machine-readable route and feature inventory is available from `/api/capabilities`.

## Platform support

| Platform | Distribution | Acceleration |
|---|---|---|
| Windows x86_64 | Desktop installer, portable app, engine archive | Supported CUDA exact-row paths; CPU fallback |
| macOS Apple Silicon | Desktop DMG or engine archive | Metal and CPU |
| Linux x86_64 | Engine archive | CUDA compiled in; CPU fallback |

Hardware support is row- and configuration-specific. Consult [COMPATIBILITY.md](COMPATIBILITY.md) before relying on a particular GPU, model, or quantization combination.

## Build from source

Camelid uses the toolchain pinned in [rust-toolchain.toml](rust-toolchain.toml). The React/Vite web UI in `frontend/` is embedded in the engine binary.

```bash
(cd frontend && npm ci && npm run build)
cargo build --release --locked --bin camelid
```

See the [contributor quick start](docs/CONTRIBUTOR_QUICKSTART.md) for prerequisites and development setup.

## Documentation

- [Documentation index](DOCS.md)
- [Model catalog and validation](docs/MODELS.md)
- [Remote browser chat](docs/REMOTE_CHAT.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [CUDA continuous batching and model residency](docs/CUDA_CONTINUOUS_BATCHING.md)
- [Architecture](docs/architecture/ARCHITECTURE.md)
- [Validation matrix](docs/VALIDATION_MATRIX.md)
- [Roadmap](ROADMAP.md)

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [contributor quick start](docs/CONTRIBUTOR_QUICKSTART.md).

## License

Camelid is released under the [MIT License](LICENSE). llama.cpp (MIT, © the ggml authors) serves as the reference oracle for supported rows; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
[release-badge]: https://img.shields.io/github/v/release/timtoole02/Camelid?display_name=tag
[latest-release]: https://github.com/timtoole02/Camelid/releases/latest
