# Camelid vs MLX benchmark harness

A reproducible, evidence-first harness for comparing Camelid against Apple's
MLX-LM (and llama.cpp as a throughput reference) on the **same Apple Silicon
machine, same prompts, same token counts, temperature 0**.

This is an engineering evidence exercise, not a marketing one. It is designed to
answer: *where does Camelid beat MLX today, by how much, under what exact
conditions, and with what tradeoffs?* — and to report losses as plainly as wins.

## What it measures

Per runtime × prompt size, over 1 warmup + N measured iterations:

- **TTFT** (time to first token, model already loaded)
- **Decode throughput** (tok/s over tokens 2..N — pure decode)
- **Prefill** (prompt evaluation time)
- **Peak memory** (RSS via `getrusage`, cross-checked with `/usr/bin/time -l`)
- **Load time** (reported separately, once)

All runtimes emit the **same JSON schema** (`bench-generate` for Camelid,
`lib/mlx_generate.py` for MLX). llama.cpp uses `llama-bench` and is a
throughput-only reference (no per-request TTFT / peak RSS).

## Lanes (kept separate — never mixed)

1. **TTFT** — startup / first-token latency.
2. **Memory efficiency** — peak RSS, footprint.
3. **GGUF direct-load** — Camelid loads GGUF directly; MLX uses converted weights.
4. **Distributed** (future) — multi-Mac Camelid vs single-node MLX.

Do not compare across different models, quantizations, prompts, or token counts.

## Usage

```bash
# 1. Build the Camelid release binary (with the bench-generate subcommand)
cargo build --release --bin camelid

# 2. Run the full matrix
MODEL=/path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
MLX_VENV=/path/to/mlx-venv \
MLX_MODEL=mlx-community/Llama-3.2-3B-Instruct-8bit \
HF_HOME=/path/to/hf-cache \
CAMELID_BIN=/path/to/release/camelid \
ITERS=10 MAX_TOKENS=128 PROMPTS="128 512 2k 8k" \
bash tools/bench/mlx-compare/bench.sh
```

Output lands in `qa/evidence-bundles/mlx-compare-<UTC timestamp>/`:

```
hardware.txt        machine + system_profiler
versions.txt        camelid commit, cargo/rustc, python, mlx, mlx-lm, llama.cpp
commands.txt        the exact run configuration
prompts/            the prompt files used
raw/camelid/*.jsonl one JSON record per measured iteration
raw/mlx-lm/*.jsonl  same schema
raw/llamacpp/*.json llama-bench -o json (reference)
summaries/results.json + results.md   median/min/max/p95 per lane
```

## Single-runtime quick runs

```bash
# Camelid only
CAMELID_COMMIT=$(git rev-parse --short HEAD) \
  ./target/release/camelid bench-generate <model.gguf> \
  --prompt-file tools/bench/mlx-compare/prompts/prompt-128.txt \
  --max-tokens 128 --temperature 0 --warmup --iterations 10 --json

# MLX only
source <mlx-venv>/bin/activate
python3 tools/bench/mlx-compare/lib/mlx_generate.py \
  --model mlx-community/Llama-3.2-3B-Instruct-8bit \
  --prompt-file tools/bench/mlx-compare/prompts/prompt-128.txt \
  --max-tokens 128 --temperature 0 --warmup --iterations 10
```

## JSON schema (one object per measured iteration)

```json
{
  "runtime": "camelid",
  "commit": "...",
  "model": "...",
  "quantization": "Q8_0",
  "iteration": 0,
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "load_ms": 0.0,
  "prefill_ms": 0.0,
  "ttft_ms": 0.0,
  "decode_ms": 0.0,
  "tokens_per_second": 0.0,
  "peak_memory_bytes": 0
}
```

(Camelid additionally includes `output_text` and `output_token_ids` for
correctness inspection.)

## Honesty rules

- Same machine, thermal state, prompts, token counts, temperature 0.
- 1 warmup + N measured; report median/min/max/p95.
- Token counts differ per tokenizer; each runtime reports its actual
  `prompt_tokens`. Exact cross-runtime token parity is **not** claimed.
- Report both wins and losses. A win must state model, quant, prompt, output
  length, hardware, run count, and margin.
