# 🐫 Camelid

[![CI][ci-badge]][ci-workflow]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-orange.svg)](rust-toolchain.toml)

**Camelid** is a state-of-the-art, Rust-native local inference engine for GGUF language models. It is designed for maximum speed, strict trust guarantees, and hardware saturation on modern platforms, especially Apple Silicon macOS. 

Instead of optimistic compatibility claims, Camelid matches trusted inference baselines with 1:1 mathematical token parity and enforces an exact-row evidence-backed supported matrix.

---

## ⚡ Hardware-Saturated Performance

Camelid bypasses high-level compilation abstractions to write bare-metal math kernels directly in AArch64 assembly and vectorized NEON SIMD.

```text
               ┌────────────────────────────────────────────────────────┐
               │              Camelid AArch64 SIMD Engine               │
               └────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
 ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
 │   i8mm GEMM   │          │  DotProd GEMV │          │   NEON Core   │
 │ (smmla, 4x8)  │          │ (sdot, 4x4)   │          │ (vadd, vmul)  │
 └───────────────┘          └───────────────┘          └───────────────┘
   Prefill Phase             Decode Phase               Activation/Norm
   64 MACs/cycle             4 MACs/cycle               128-bit Vectors
```

*   **Bare-Metal AArch64 Matrix Multiplication (`smmla`)**: Emits matrix multiply-accumulate instructions in hardware, performing 64 8-bit MAC operations per clock cycle to saturate prefill throughput.
*   **Vectorized Decode Kernels (`sdot`)**: Performs high-speed row-major matrix-vector dot-products directly on repacked Q8_0 weights, eliminating dequantization overhead on Apple Silicon.
*   **128-bit NEON Element-Wise & Reduction Pipelines**: Vectorizes input block quantization, RMS Normalization (`rms_norm`), SiLU activations, and element-wise additions/multiplications into parallel SIMD register operations.
*   **Performance-Core Thread Scheduling**: Binds Rayon multi-threaded loops strictly to physical Performance (P) Cores, avoiding Efficiency core synchronization delays that derail inference latency.

---

## 🌐 High-Speed Distributed Clustering

Scale model inference across multiple machines (such as Mac minis) using high-speed interfaces like direct **Thunderbolt 4 Bridge Networking** (IP-over-Thunderbolt) for microsecond-scale bus latency.

*   **Split-Model Parallelization**: Partition layers dynamically across nodes (e.g., Coordinator evaluates layers $0..K$ locally, Worker evaluates layers $K..N$).
*   **Zero dequantization/repack overhead**: Coordinator and Worker communicate raw activations over a custom high-performance socket transport.
*   **TCP Bus Telemetry**: Includes a built-in `bench-network` tool to physically measure link round-trip time (RTT) and throughput.

---

## 💎 Google Gemini-Style Frontend

Camelid includes a built-in React/Vite web interface inside [frontend/](frontend/) that replicates a premium Google Gemini experience:
*   **Curated Aesthetics**: Harmonious dark mode palettes, subtle glowing gradients, glassmorphic layout rails, and micro-animations.
*   **Honest Readiness Signals**: Interaction panels dynamically reflect loaded GGUF capabilities, locking inputs until a model is fully validated.
*   **Intuitive Chat Surface**: Product-forward conversation area that emphasizes responsiveness.

---

## 📋 Exact-Row Support Matrix

Camelid enforces a strict evidence boundary. Support is exact-row only; neighboring sizes, formats, or tokenizers must have their own verification benchmarks before public readiness.

| Exact Model Row | Public Status | Current Evidence Boundary |
| :--- | :--- | :--- |
| **TinyLlama 1.1B Chat Q8_0** | **Verified Support** | End-to-end generation, 50-token reference parity, and checked 512-context. |
| **Llama 3.2 1B Instruct Q8_0** | **Verified Bounded Support** | Load, OpenAI API, WebUI chat, 1:1 token parity, and checked 512/1024 context. |
| **Llama 3.2 3B Instruct Q8_0** | **Smoke Supported** | Compact/broader 50-token reference parity, API/WebUI smoke, and 2048 context. |
| **Llama 3 8B Instruct Q8_0** | **Verified Bounded Support** | completions, chat completions, WebUI validation, and lazy-Q8 read hot-paths. |
| **Mistral-7B-Instruct-v0.3 Q8_0**| **Smoke Supported** | Exact-row load, tokenizer, 50-token parity, and checked 4096 context. |
| **Mixtral-8x7B-Instruct-v0.1** | **Backend Support Only** | One-token MoE backend execution. Later-generation divergence remains blocked. |

---

## 🚀 Quickstart

Verify that Camelid builds cleanly, starts the backend, and serves a live API endpoint locally.

### 1) Build and Run the Server

Ensure you are using Rust 1.87+ and target native CPU instruction sets during compilation:

```bash
# Build optimized for your CPU
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Serve a local GGUF model
./target/release/camelid serve --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf --threads 4
```
*Note: Set `--threads` exactly to your machine's physical Performance Core count (e.g. `sysctl -a | grep hw.perflevel0.physicalcpu` on macOS) to bypass scheduling traps.*

### 2) Verify Capabilities API

Verify that the local capabilities discovery is reachable:

```bash
curl -s http://127.0.0.1:8181/api/capabilities
```

### 3) Start the Gemini WebUI

Run the React/Vite development server locally to access the premium front-end chat surface:

```bash
cd frontend
npm ci
npm run dev
```

---

## High-Speed Cluster Commands

To scale across two local machines using a private link or trusted LAN:

#### 1. On the worker:
```bash
./target/release/camelid serve-distributed \
    --role worker \
    --addr <worker-host>:8089 \
    --layer-range 16..32 \
    --model /path/to/model.gguf
```

#### 2. On the coordinator:
```bash
./target/release/camelid serve-distributed \
    --role coordinator \
    --addr <coordinator-host>:8181 \
    --worker-addr <worker-host>:8089 \
    --layer-range 0..16 \
    --model /path/to/model.gguf
```

---

## 🧪 Validation & Formatting

Keep the repository clean, fully validated, and formatted before committing changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

For front-end development, check compiling outputs:

```bash
cd frontend
npm run build
```

---

## 🗺️ Documentation Map

*   [`COMPATIBILITY.md`](COMPATIBILITY.md) — Exact support criteria and ledgers
*   [`STATUS.md`](STATUS.md) — Active blockers, evidence checkpoints, and benchmarks
*   [`ROADMAP.md`](ROADMAP.md) — Engineering sequencing and delivery goals
*   [`ARCHITECTURE.md`](ARCHITECTURE.md) — Deep architectural module layouts

---

## 📜 License and Reference Credits

Camelid is open-source and licensed under the [MIT License](LICENSE).

Camelid's tokenizer, reference compatibility layouts, and validation benchmarks are inspired by and checked against [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (Copyright (c) 2023-2026 The ggml authors, MIT License). Camelid maintains its original Rust-native codebase while proudly crediting the extraordinary reference work of the broader `ggml` ecosystem.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
