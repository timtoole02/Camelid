# llama.cpp Baseline for Camelid v0.1

Status: CPU-only v0.1 baseline captured for one exact row; Metal mode deferred.

This file defines the reproducible llama.cpp comparator baseline required by the v0.1 evidence release. It separates CPU-only and Metal modes because the release directive requires backend-mode separation. Historical same-host llama.cpp evidence exists, but it was not captured at the current release branch SHA, so it is prior context only.

## Current Evidence

v0.1 CPU-only release artifact:

- Bundle: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`
- Camelid source SHA in bundle: `8026339531463ade269d7be7078da331ba3e4085`
- Release worktree status at run time: clean `release/v0.1-evidence`
- llama.cpp source commit: `399739d5c5978351f39e3454bfbfbab4f369088f`
- llama.cpp version output: `version: 1 (399739d)`, built with AppleClang `17.0.0.17000404`
- Mode: CPU-only llama.cpp server (`-ngl 0`); the binary had Metal support available, but this row is not Metal evidence
- Host class: macOS Darwin `25.5.0`, Apple M4, arm64, 10 logical CPUs, 16 GiB RAM
- Row: `llama32_3b_instruct_q8_0`
- Model: `Llama-3.2-3B-Instruct-Q8_0.gguf`
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`
- Prompt contract: marker prompt requiring `CMLD-BENCH`
- Context: 512
- Max generated tokens: 16
- Threads: 8
- Warmup/repeats: 1 warmup, 3 measured repeats
- Guardrails: passed for both Camelid and llama.cpp measured runs
- Result boundary: Camelid lost this bounded row on TTFT and total elapsed; do not use streamed chunk estimates as tokenizer-ground-truth throughput

Historical retained artifact:

- Bundle: `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260522T1620Z-main-samehost-bench/`
- Camelid source head in artifact: `84a4a83bf881550f29dcea8349c2284439dfd900`
- llama.cpp source head in artifact: `4f0e43da6f8f6e9390d88409610098ec2d2dc5c7`
- Mode: CPU-only llama.cpp server (`-ngl 0`)
- Host class: Linux x86_64, Intel Xeon Platinum 8488C, 16 logical CPUs, 123.79 GiB RAM
- Row: `llama32_3b_instruct_q8_0`
- Model: `Llama-3.2-3B-Instruct-Q8_0.gguf`
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`
- Prompt contract: marker prompt requiring `CMLD-BENCH`
- Context: 512
- Max generated tokens: 8
- Threads: 8
- Warmup/repeats: 0 warmup, 2 measured repeats
- Guardrails: passed
- Result: Camelid avg TTFT `8669.83 ms`; llama.cpp avg TTFT `309.52 ms`

Release boundary: this historical artifact is not the v0.1 baseline because the release SHA is different and the run predates the v0.1 evidence worktree.

## v0.1 Reproduction Commands

### CPU-only

Run from the release worktree:

```sh
cd <repo>
cargo build --release

CAMELID_BIN=target/release/camelid \
LLAMA3_LLAMA_SERVER="$LLAMA_CPP_BUILD/bin/llama-server" \
node scripts/bench-llama3-same-host.mjs \
  --model "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf" \
  --model-id llama32-3b-q8-throughput \
  --row-id llama32_3b_instruct_q8_0 \
  --max-tokens 16 \
  --warmup 1 \
  --repeats 3 \
  --threads 8 \
  --require-marker \
  --expected-marker CMLD-BENCH \
  --out "qa/evidence-bundles/v0.1/$(date -u +%Y%m%dT%H%M%SZ)/llamacpp-cpu-only.json"
```

The harness starts llama.cpp with `-ngl 0`, so this is the CPU-only comparator. The committed v0.1 bundle above was captured with an external Cargo target directory and an external llama.cpp build directory because the local release worktree filesystem had only about 246 MiB free after an attempted checkout; the source worktree itself remained clean.

### Metal

The existing same-host harness can compare against a manually-started Metal llama.cpp server by disabling llama-server startup:

```sh
cd <repo>

"$LLAMA_CPP_BUILD/bin/llama-server" \
  --host 127.0.0.1 \
  --port 8183 \
  -m "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf" \
  -ngl 999 \
  -c 512 \
  -t 8 \
  --no-warmup

CAMELID_BIN=target/release/camelid \
node scripts/bench-llama3-same-host.mjs \
  --model "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf" \
  --model-id llama32-3b-q8-throughput \
  --row-id llama32_3b_instruct_q8_0 \
  --llama-url http://127.0.0.1:8183 \
  --start-llama-server=false \
  --max-tokens 16 \
  --warmup 1 \
  --repeats 3 \
  --threads 8 \
  --require-marker \
  --expected-marker CMLD-BENCH \
  --out "qa/evidence-bundles/v0.1/$(date -u +%Y%m%dT%H%M%SZ)/llamacpp-metal.json"
```

The Metal run must record the llama.cpp build flags proving Metal support is enabled. If `llama-server` reports no Metal backend, mark the Metal baseline deferred rather than treating CPU fallback as Metal evidence.

## Evidence Field Ledger

The v0.1 CPU-only bundle records:

- Camelid commit SHA: `8026339531463ade269d7be7078da331ba3e4085`
- Comparator commit or version: `399739d5c5978351f39e3454bfbfbab4f369088f` and `llama-server --version`
- Model name/path/hash: exact 3B Q8_0 row with sanitized `$CAMELID_MODEL_DIR` path and SHA256 above
- Quantization: GGUF Q8_0
- Prompt: exact marker prompt and compact chat rendering in `same-host-llama32-3b-q8.json`
- Context size: 512
- Max generated tokens: 16
- Thread count: 8
- Batch settings/runtime flags: raw command and `-ngl 0` are in `commands.md` and `raw_logs/`
- Environment variables: `CAMELID_STREAM_TIMING_DIAGNOSTICS=on` is recorded in the command metadata
- Hardware/OS details: `machine.json`
- Raw command/output: `commands.md`, `raw_logs/`, and `results.json`
- Timing/memory data: `results.json` and `same-host-llama32-3b-q8.json`
- Pass/fail status: `entries_ok: 1`, marker guardrails passed

## Remaining Gaps

- Metal comparison remains deferred; the captured v0.1 row is CPU-only even though the binary reported Metal support.
- Only the Llama 3.2 3B Instruct Q8_0 row was captured for v0.1. This is not a full comparator table for every public support row.
- GGUF model files and comparator build artifacts are not vendored in the release worktree; the committed bundle uses sanitized placeholders for local paths.
