# Ollama Baseline for Camelid v0.1

Status: deferred pending release-captain acceptance of the exact Ollama row.

Ollama is a practical user-experience comparator, not a CPU-only, Metal, MLX, or quant-equivalent runtime comparator. It should answer: how does a local user-facing runtime behave on the same host for a comparable prompt budget?

## Environment Probe

Observed in this environment on 2026-05-31:

- Ollama client: `0.24.0`
- Installed model visible to `ollama list`: `llama3.1:8b`
- Installed model ID shown by Ollama: `46e0c10c039e`
- Installed model size shown by Ollama: `4.9 GB`
- Host: macOS 26.5, Apple M4, arm64, 10 logical CPUs, 16 GiB RAM

This is not enough for a v0.1 release baseline because the installed Ollama row is not one of Camelid's exact GGUF Q8_0 release rows, and the release evidence contract requires model path/hash, raw command/output, timing, memory, and pass/fail status.

## Required Baseline Shape

Use this baseline only as a user-experience comparison:

- Comparator class: Ollama local app/service
- Backend mode: Ollama-selected local backend, not normalized CPU-only or Metal
- Model row: release captain must choose either an installed Ollama tag or a pullable tag that maps clearly to the release comparison story
- Prompt: use the same short marker prompt as the llama.cpp baseline when possible
- Output budget: match the release baseline token budget as closely as Ollama permits
- Metrics: wall time, Ollama prompt/eval durations, eval token count, approximate tok/s, and observed process RSS

## Reproduction Commands

Start or verify the local Ollama service:

```sh
ollama serve
```

Record version and model inventory:

```sh
ollama --version
ollama list
ollama show llama3.1:8b --modelfile > ollama-llama31-8b.modelfile.txt
ollama show llama3.1:8b --parameters > ollama-llama31-8b.parameters.txt
```

Run a streaming-disabled JSON request so raw timing fields can be captured:

```sh
RUN_ROOT="qa/evidence-bundles/v0.1/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

/usr/bin/time -lp curl -sS http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.1:8b",
    "prompt": "Reply with exactly this single line and nothing else: CMLD-BENCH",
    "stream": false,
    "options": {
      "temperature": 0,
      "num_predict": 16,
      "num_ctx": 512
    }
  }' \
  > "$RUN_ROOT/ollama-generate.json"
```

Capture memory while the request is running:

```sh
ps -axo pid,rss,vsz,command | rg '[o]llama'
```

If the selected model comes from an Ollama blob, record the blob SHA256 without committing local user paths:

```sh
shasum -a 256 "$OLLAMA_MODEL_BLOB"
```

## Required Evidence Field Ledger

- Camelid commit SHA: pending; expected release branch HEAD
- Comparator commit or version: pending; record `ollama --version`
- Model name: pending release-captain row decision; installed candidate is `llama3.1:8b`
- Model path: pending; Ollama blob path must be sanitized if committed
- Model SHA256 hash: pending; hash the selected Ollama blob
- Quantization: pending; record Ollama tag details and any quantization visible in model metadata
- Prompt: pending; recommended marker prompt above
- Context size: pending; recommended `num_ctx: 512`
- Max generated tokens: pending; recommended `num_predict: 16`
- Thread count: pending; Ollama may not expose a comparable thread count for the selected backend; record if configured
- Batch settings: pending; record Ollama options that affect batching/context
- Runtime flags: pending; record all Ollama options and environment variables
- Environment variables: pending; record relevant `OLLAMA_*` values or state `none set`
- Hardware details: pending; record host facts and whether Apple GPU/Metal is used by Ollama if observable
- OS version: pending; record `sw_vers`
- Raw command: pending; preserve the exact curl and timing command
- Raw output: pending; preserve Ollama JSON and timing stderr
- Timing data: pending; use Ollama `total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`, `prompt_eval_count`, and `eval_count`
- Memory data: pending; record observed Ollama RSS/VSZ
- Pass/fail status: pending; pass requires successful response and expected marker behavior if marker guard is used

## Blockers

- No release-captain-approved Ollama row is named.
- The installed `llama3.1:8b` row is not quant-equivalent to Camelid's GGUF Q8_0 rows.
- No v0.1 Ollama artifact exists under `qa/evidence-bundles/v0.1/`.
- This comparison should stay in the user-experience lane and must not be used as CPU-only, Metal, MLX, or parity evidence.
