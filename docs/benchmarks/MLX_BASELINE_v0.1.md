# MLX Baseline for Camelid v0.1

Status: pending for `v0.1.0-rc1`; historical memory-only context exists.

MLX is an Apple Silicon market-context comparator. It should be reported separately from CPU-only, Metal llama.cpp, Ollama user experience, and distributed Mac comparisons. MLX-LM uses MLX-format weights and Apple Silicon acceleration, so it is not a strict quant-equivalent comparison against Camelid GGUF Q8_0.

## Current Evidence

Historical retained artifact:

- Bundle: `qa/evidence-bundles/apple-silicon-camelid-vs-mlx-memory-20260514T001835Z-head-775db673af32/`
- Source head: `775db673af32`
- Scope: same-host Apple Silicon resident-memory comparison
- Camelid profile: memory-first lazy GGUF Q8_0
- MLX profile: public `mlx-community` 4-bit MLX-LM models
- Claim boundary: memory comparison only; MLX-LM was much faster in the short probe

Historical rows:

| Model family | Camelid row/profile | Camelid RSS MiB | MLX row/profile | MLX RSS MiB | MLX avg generate ms |
| --- | --- | ---: | --- | ---: | ---: |
| Llama 3.2 1B Instruct | GGUF Q8_0, memory-first lazy | 257.72 | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 1062.06 | 136.16 |
| Llama 3.2 3B Instruct | GGUF Q8_0, memory-first lazy | 328.92 | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 2139.70 | 211.14 |

Release boundary: this historical artifact is not the v0.1 MLX baseline because it was captured at source head `775db673af32`, not the release branch SHA. It also omits exact local model paths by design.

## Environment Probe

Observed in this environment on 2026-05-31:

- Host: macOS 26.5, Apple M4, arm64, 10 logical CPUs, 16 GiB RAM
- Python module probe: `mlx_lm` is not installed in the default `python3` environment

Because `mlx_lm` is missing, no fresh v0.1 MLX run was performed.

## Reproduction Commands

Use an isolated Python environment that has `mlx-lm` installed. Do not commit virtualenvs or downloaded model caches.

```sh
cd <repo>

PYTHON=/path/to/python-with-mlx-lm \
node scripts/bench-mlx-memory.mjs \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 16 \
  --warmup 1 \
  --repeats 3 \
  --sample-ms 100 \
  --message-prefix "v0.1 mlx baseline" \
  --out "qa/evidence-bundles/v0.1/$(date -u +%Y%m%dT%H%M%SZ)/mlx-llama32-3b-4bit.json"
```

For a 1B comparison row:

```sh
PYTHON=/path/to/python-with-mlx-lm \
node scripts/bench-mlx-memory.mjs \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --max-tokens 16 \
  --warmup 1 \
  --repeats 3 \
  --sample-ms 100 \
  --message-prefix "v0.1 mlx baseline" \
  --out "qa/evidence-bundles/v0.1/$(date -u +%Y%m%dT%H%M%SZ)/mlx-llama32-1b-4bit.json"
```

## Required Evidence Field Ledger

- Camelid commit SHA: pending; expected release branch HEAD
- Comparator commit or version: pending; record `mlx-lm` package version and Python version
- Model name: pending; recommended `mlx-community/Llama-3.2-3B-Instruct-4bit` and optionally 1B row
- Model path: pending; record sanitized local cache path or Hugging Face model ID if path is not public-safe
- Model SHA256 hash: pending; MLX model consists of multiple files, so record file hashes or a manifest hash for the cached revision
- Quantization: pending; expected `MLX 4-bit` for the recommended rows
- Prompt: pending; record generated prompt/messages from the harness
- Context size: pending; record tokenizer prompt token count and any explicit context setting if configured
- Max generated tokens: pending; recommended 16
- Thread count: pending; MLX does not expose a direct thread count equivalent in this harness; record `not exposed` if unchanged
- Batch settings: pending; record MLX-LM defaults and any generation options
- Runtime flags: pending; record `PYTHON`, script options, and MLX environment variables
- Environment variables: pending; record relevant `MLX_*`, `HF_*`, and Python environment values or state `none set`
- Hardware details: pending; record Apple Silicon model, CPU count, RAM, and Metal/GPU availability if observable
- OS version: pending; record `sw_vers`
- Raw command: pending; preserve exact command
- Raw output: pending; preserve stdout/stderr and JSON artifact
- Timing data: pending; use load, TTFT, and generate timings from `scripts/bench-mlx-memory.mjs`
- Memory data: pending; use sampled peak RSS/VSZ from the harness
- Pass/fail status: pending; pass requires successful model load and measured repeats

## Blockers

- `mlx_lm` is not installed in the current default Python environment.
- No fresh v0.1 MLX artifact exists under `qa/evidence-bundles/v0.1/`.
- Historical MLX evidence is memory-first context only and must not be presented as a speed win for Camelid.
- MLX rows use MLX-format 4-bit weights, not Camelid's exact GGUF Q8_0 rows.
