<div align="center">

# 🐪 Camelid

**Run supported GGUF language, vision, and embedding models locally with a Rust-native engine.**

Desktop app, browser chat, terminal UI, and an OpenAI-compatible API—all backed by the same local runtime.

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-64748b.svg)](#platform-support)

[Download][latest-release] · [Quick start](#quick-start) · [Models](#supported-models) · [Documentation](DOCS.md) · [Contributing](CONTRIBUTING.md)

</div>

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

<div align="center"><sub>Camelid's local web UI—a dark, collapsed-rail chat surface served directly from the engine binary.</sub></div>

Camelid loads GGUF models and runs inference on your hardware. Its tokenizer, model loader, CPU kernels, and Metal and CUDA execution paths are implemented in this repository and distributed as a single Rust binary—no Python, Node.js, or Docker is required at runtime.

## What is Camelid?

Camelid is an open-source, Rust-native local AI inference engine for running supported GGUF large language models (LLMs), vision-language models (VLMs), and embedding models on Windows, macOS, and Linux. Use it as a desktop app, a browser-based local AI chat interface, a terminal application, or a self-hosted OpenAI-compatible API for your own tools and applications.

Model inference runs on your hardware, with CPU execution, Apple Silicon Metal acceleration, and NVIDIA CUDA acceleration available within the documented support boundaries. Camelid is designed for private local AI chat, offline inference after model download, GGUF model testing, and local LLM application development without a Python, Node.js, or Docker runtime.

## Why Camelid?

- **Local by default.** Models and inference stay on your machine unless you expose the server.
- **One engine, several interfaces.** Use the desktop app, browser chat, terminal UI, or HTTP API.
- **Simple distribution.** The engine and web UI ship together as one binary.
- **Hardware acceleration.** Use Metal on Apple Silicon, CUDA on supported NVIDIA paths, or CPU fallback.
- **Developer APIs.** Build against chat, Responses, embeddings, reranking, and structured-output endpoints.
- **Evidence-backed compatibility.** Support applies to exact model files and quantizations validated against a pinned llama.cpp reference.

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

> [!WARNING]
> A non-loopback listener requires authentication. Prefer an API key file:
>
> ```bash
> camelid serve --addr 0.0.0.0:8181 --api-key-file ./camelid-api.key
> ```
>
> See [configuration](docs/CONFIGURATION.md) for CORS, TLS, and remote-deployment options.

## Supported models

Camelid deliberately supports exact model-and-quantization combinations rather than entire model families. Each supported file is validated token-for-token against a pinned llama.cpp reference. Files outside the supported set fail closed instead of silently using an unverified path.

Good starting points:

| Goal | Model | Pull ID |
|---|---|---|
| Smallest end-to-end test (~1.2 GB) | TinyLlama 1.1B Chat Q8_0 | `tinyllama` |
| **Recommended first model** | Llama 3.2 3B Instruct Q8_0 | `3b_instruct_q8` |
| Compact Windows CPU or tested M4 Metal chat (~2.9 GB) | LFM2.5 2.6B Q8_0 | `lfm2_5_2_6b` |
| Local embeddings and semantic retrieval | Nomic Embed Text v1.5 Q8_0 | `nomic` |
| Fits a 16 GB Apple Silicon Mac | Mistral 7B Instruct v0.3 Q8_0 | `mistral` |
| Reasoning and coding on a small budget | Qwen3 4B Q4_K_M | `qwen3_4b_q4` |
| Compact PrismML GPU model | Bonsai 4B Q1_0 | `bonsai_4b_q1` |
| PrismML browser/API vision | Bonsai 27B Q1_0 | `bonsai_27b_q1` |

### Full `camelid pull` catalog

Run `camelid pull <id>` to download a model into `./models`. Pull IDs resolve by unique substring; if a fragment matches several rows, Camelid lists the matches instead of guessing.

| Model | Quant | Arch | Size | Pull ID | GGUF file |
|---|---|---|---:|---|---|
| **Microsoft BitNet b1.58 2B 4T** *(experimental)* | `I2_S` | `bitnet-b1.58` | 1.2 GB | `bitnet_b1_58_2b_4t_i2_s` | `ggml-model-i2_s.gguf` |
| **Microsoft BitNet Embedding 0.6B** *(experimental)* | `I2_S` | `qwen3` | 0.4 GB | `bitnet_embedding_0_6b_i2_s` | `bitnet-embeddings-0.6b-bf16-i2_s.gguf` |
| **Microsoft BitNet Embedding 270M** *(experimental)* | `I2_S` | `gemma3` | 0.4 GB | `bitnet_embedding_270m_i2_s` | `bitnet-embeddings-270m-bf16-i2_s.gguf` |
| **Nomic Embed Text v1.5** | `Q8_0` | `nomic-bert` | 0.15 GB | `nomic` | `nomic-embed-text-v1.5.Q8_0.gguf` |
| **TinyLlama 1.1B Chat** | `Q8_0` | `llama` | 1.2 GB | `tinyllama` | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` |
| **Llama 3.2 1B Instruct** | `Q8_0` | `llama` | 1.3 GB | `1b_instruct_q8` | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| **Llama 3.2 1B Instruct** | `IQ4_XS` | `llama` | 0.7 GB | `iq4_xs` | `Llama-3.2-1B-Instruct-IQ4_XS.gguf` |
| **Llama 3.2 3B Instruct** | `Q8_0` | `llama` | 3.4 GB | `3b_instruct_q8` | `Llama-3.2-3B-Instruct-Q8_0.gguf` |
| **Llama 3.2 3B Instruct** | `Q4_K_M` | `llama` | 2.0 GB | `3b_instruct_q4` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| **Llama 3.2 3B Instruct** | `Q5_K_M` | `llama` | 2.3 GB | `3b_instruct_q5` | `Llama-3.2-3B-Instruct-Q5_K_M.gguf` |
| **Llama 3 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama3_8b` | `Meta-Llama-3-8B-Instruct.Q8_0.gguf` |
| **Llama 3.1 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama31_8b` | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| **Gemma 3 1B-It** | `Q8_0` | `gemma3` | 1.1 GB | `gemma_3_1b` | `gemma-3-1b-it-Q8_0.gguf` |
| **Gemma 4 E2B-It** | `Q8_0` | `gemma4` | 5.0 GB | `gemma4_e2b` | `gemma-4-E2B-it-Q8_0.gguf` |
| **Gemma 4 E4B-It** | `Q8_0` | `gemma4` | 8.2 GB | `gemma4_e4b` | `gemma-4-E4B-it-Q8_0.gguf` |
| **Gemma 4 12B-It** — two-Mac distributed | `Q8_0` | `gemma4` | 12.7 GB | `gemma4_12b` | `gemma-4-12b-it-Q8_0.gguf` |
| **Gemma 4 26B-A4B-It QAT** — two-Mac distributed MoE | `Q4_0` | `gemma4` | 14.4 GB | `gemma4_26b` | `gemma-4-26B_q4_0-it.gguf` |
| **Qwen3 0.6B** | `Q8_0` | `qwen3` | 0.6 GB | `qwen3_0_6b` | `Qwen3-0.6B-Q8_0.gguf` |
| **Qwen3 1.7B** | `Q8_0` | `qwen3` | 1.8 GB | `qwen3_1_7b` | `Qwen3-1.7B-Q8_0.gguf` |
| **Qwen3 4B** | `Q8_0` | `qwen3` | 4.3 GB | `qwen3_4b_q8` | `Qwen3-4B-Q8_0.gguf` |
| **Qwen3 4B** | `Q4_K_M` | `qwen3` | 2.5 GB | `qwen3_4b_q4` | `Qwen3-4B-Q4_K_M.gguf` |
| **Qwen3 8B** | `Q8_0` | `qwen3` | 8.7 GB | `qwen3_8b` | `Qwen3-8B-Q8_0.gguf` |
| **Qwen3 14B** *(active validation; not supported)* | `Q4_K_M` | `qwen3` | 9.0 GB | `qwen3_14b` | `Qwen3-14B-Q4_K_M.gguf` |
| **Mistral 7B Instruct v0.3** | `Q8_0` | `llama` | 7.7 GB | `mistral` | `Mistral-7B-Instruct-v0.3-Q8_0.gguf` |
| **Mistral Nemo Instruct 2407** *(validation hold; not supported)* | `Q4_K_M` | `llama` | 7.5 GB | `mistral_nemo` | `Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` |
| **LFM2.5 2.6B** *(supported exact-row smoke)* | `Q8_0` | `lfm2` | 2.9 GB | `lfm2_5_2_6b` | `LFM2.5-2.6B-Q8_0.gguf` |
| **Phi-3-mini-4k-instruct** | `Q8_0` | `phi3` | 4.1 GB | `phi3` | `Phi-3-mini-4k-instruct-Q8_0.gguf` |
| **DeepSeek R1 Distill Qwen 7B** | `Q8_0` | `qwen25` | 8.1 GB | `distill_qwen` | `DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf` |
| **DeepSeek R1 Distill Llama 8B** | `Q8_0` | `llama` | 8.5 GB | `distill_llama` | `DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf` |
| **DeepSeek R1 0528 Qwen3 8B** *(validation hold; not supported)* | `Q4_K_M` | `qwen3` | 5.0 GB | `deepseek_r1_0528` | `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` |
| **Qwen2.5 Coder 7B** | `Q8_0` | `qwen25` | 8.1 GB | `qwen25_coder` | `qwen2.5-coder-7b-instruct-q8_0.gguf` |
| **Ornith 1.0 9B** — hybrid DeltaNet, `tool_capable` | `Q8_0` | `qwen35` | 9.5 GB | `ornith` | `ornith-1.0-9b-Q8_0.gguf` |
| **Bonsai 4B** | `Q1_0` | `qwen3` | 0.6 GB | `bonsai_4b_q1` | `Bonsai-4B-Q1_0.gguf` |
| **Ternary Bonsai 4B** | `Q2_0` | `qwen3` | 1.1 GB | `bonsai_4b_q2` | `Ternary-Bonsai-4B-Q2_0.gguf` |
| **Ternary Bonsai 4B** | `PQ2_0` | `qwen3` | 1.1 GB | `bonsai_4b_pq2` | `Ternary-Bonsai-4B-PQ2_0.gguf` |
| **Bonsai 8B** | `Q1_0` | `qwen3` | 1.2 GB | `bonsai_8b_q1` | `Bonsai-8B-Q1_0.gguf` |
| **Ternary Bonsai 8B** | `Q2_0` | `qwen3` | 2.2 GB | `bonsai_8b_q2` | `Ternary-Bonsai-8B-Q2_0.gguf` |
| **Bonsai 27B** | `Q1_0` | `qwen35` | 3.8 GB | `bonsai_27b_q1` | `Bonsai-27B-Q1_0.gguf` |
| **Ternary Bonsai 27B** | `Q2_0` | `qwen35` | 7.2 GB | `bonsai_27b_q2` | `Ternary-Bonsai-27B-Q2_0.gguf` |

The two distributed Gemma 4 rows are validated on a layer-sharded two-host lane and do not fit on a single 16 GB machine. Aya Expanse 8B is tracked in the compatibility ledger as a header-only Command R planning candidate and is intentionally absent from the supported `camelid pull` catalog: chat fails closed until the exact artifact, Aya template/tokenizer parity, real-weight generation parity, and frontend/context gates are complete.

Mistral Nemo Instruct 2407 Q4_K_M, Qwen3 14B Q4_K_M, and DeepSeek R1 0528 Qwen3 8B Q4_K_M are downloadable catalog rows, not supported rows. The pinned bring-up evidence records Mistral Nemo cross-backend divergence and a blocked external comparator, Qwen3 14B without an external oracle or chat/API proof, and DeepSeek cross-backend divergence plus a missing native R1 marker/tool renderer. None inherits support from its architecture or a smaller sibling.

The three BitNet rows are bring-up targets, not promoted support rows yet. Camelid
can parse and execute their official canonical `I2_S` GGUF graphs through cleanroom
CPU, Metal, and CUDA projection kernels. Runtime-selectable `i2_s`, `tl1`, and `tl2`
strategies operate on the same published bytes; they do not claim compatibility with
BitNet.cpp's separately permuted TL files. Reference parity and bounded-context /
embedding-vector receipts remain outstanding. See
[the BitNet runtime notes](docs/architecture/BITNET.md).

Experimental exception: the Gemma 4 26B MoE row can also run on a single 16 GB
Apple-silicon Mac through the opt-in Ghost-MoE Metal lane, which repacks the routed experts into a
paged `.cghost` artifact and keeps a bounded, persistent expert working set in unified memory
(measured 17–20 tok/s steady-state decode on an M4). Replies on that lane are marked
experimental with no parity guarantee. Setup, the recommended serve profile, and measured receipts
live in [docs/runtime/ghost-mode.md](docs/runtime/ghost-mode.md#ghost-moe-v2-gemma-4-26b-a4b).

The full catalog, exact hashes, supported execution paths, and claim boundaries live in:

- [COMPATIBILITY.md](COMPATIBILITY.md)—authoritative supported-row ledger
- [SUPPORT_MATRIX_v0.1.md](./docs/reference/SUPPORT_MATRIX_v0.1.md)—per-row support boundaries
- [RECEIPTS.md](./docs/reference/RECEIPTS.md)—reproducible validation receipts
- [benchmarks](docs/benchmarks/BENCHMARKS.md)—recorded performance measurements

Selected validation highlight:

| Model row | Quant | Evidence |
|---|---|---|
| Mistral 7B Instruct v0.3 | Q8_0 | Exact-row smoke + bounded context 512→8192 + GPU/CPU parity |
| LFM2.5 2.6B | Q8_0 | Hash-pinned exact-row smoke on Windows CPU/runnable and Apple M4 macOS 26.5 arm64 resident Metal: 96/96 short greedy tokens, exact 512-token chat prompt + 8/8 reference-oracle tokens/text, and API/Models-page/WebUI/SSE smoke |

The LFM2.5 promotion is limited to `LiquidAI/LFM2.5-2.6B-GGUF@b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`, file `LFM2.5-2.6B-Q8_0.gguf` (2,874,779,456 bytes, SHA-256 `36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`). The Windows x86_64 CPU/runnable proof remains recorded at `qa/evidence-bundles/lfm2-2.6b-q8-phase1-promotion-20260810/`. The Apple M4 macOS 26.5 arm64 receipt at `qa/evidence-bundles/lfm2-2.6b-q8-macos-metal-20260810-head-d31e5cb0/` independently asserts the resident-Metal execution plan, 96/96 short greedy token IDs, the exact 512-token prompt plus 8/8 generated IDs/text against pinned llama.cpp b9632 (`acd79d603`), and API/Models-page/WebUI non-streaming plus 128-ceiling SSE smoke. The 512-token/8-token oracle checks are reference-only; raw `/v1/completions` and tools remain typed fail-closed. Sampling beyond deterministic greedy, context above 512, neighboring rows, production throughput, CUDA, other Apple hardware, broad platform portability, and broader LFM2 support remain unclaimed.

### Multimodal image chat

Seven hash-pinned PrismML Bonsai GGUFs are supported on Apple Silicon Metal and Windows x86_64 CUDA: 4B Q1/Q2/PQ2, 8B Q1/Q2, and 27B Q1/Q2. Both 27B rows support multimodal PNG/JPEG input in browser chat and OpenAI-compatible Chat Completions when paired with the Qwen3-VL projector.

The **Arch** column reports what each GGUF declares in `general.architecture`, so it is not uniform across this family: the 4B and 8B files declare `qwen3` and only the 27B files declare `qwen35`. The two labels bind different engines, so the split is real rather than a typo.

The desktop **Models** page downloads the projector automatically with either 27B model. For a CLI installation, place `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` beside the model GGUF or set `CAMELID_MMPROJ`, then run `camelid serve` normally. See [COMPATIBILITY.md](COMPATIBILITY.md) for the exact artifacts and validated scope.

### Embeddings and reranking

The exact Nomic Embed Text v1.5 Q8_0 row supports OpenAI-compatible `/v1/embeddings`, Matryoshka dimensions, cosine-similarity reranking through `/v1/rerank`, and optional in-memory semantic retrieval for Workspace. The encoder currently runs on CPU; other embedding families and quantizations fail closed. See the [embedding API guide](docs/architecture/EMBEDDINGS.md) for loading and request examples.

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
- [Configuration reference](docs/CONFIGURATION.md)
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
