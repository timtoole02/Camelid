# llama.cpp Baseline for Camelid v0.1

Status: pending for `v0.1.0-rc1`.

This file defines the reproducible llama.cpp comparator baseline required by the v0.1 evidence release. It separates CPU-only and Metal modes because the release directive requires backend-mode separation. Historical same-host llama.cpp evidence exists, but it was not captured at the the current release branch SHA, so it is prior context only.

## Current Evidence

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

## v0.1 Required Runs

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

The harness starts llama.cpp with `-ngl 0`, so this is the CPU-only comparator.

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

## Required Evidence Field Ledger

Each v0.1 llama.cpp baseline must record:

- Camelid commit SHA: pending for current release run; expected release branch HEAD
- Comparator commit or version: pending; record llama.cpp commit and `llama-server --version` output
- Model name: pending; expected exact row `Llama 3.2 3B Instruct Q8_0`
- Model path: pending; use a sanitized placeholder such as `$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf`
- Model SHA256 hash: pending; expected `shasum -a 256 "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf"`
- Quantization: pending; expected `GGUF Q8_0`
- Prompt: pending; record the exact benchmark messages and rendered prompt mode
- Context size: pending; expected 512 unless release captain changes the row
- Max generated tokens: pending; expected 16 for the release baseline
- Thread count: pending; expected 8 unless release captain changes it
- Batch settings: pending; record llama.cpp context/batch flags and Camelid defaults
- Runtime flags: pending; CPU-only must include `-ngl 0`; Metal must include the exact `-ngl` and Metal build flag evidence
- Environment variables: pending; record `CAMELID_BIN`, `LLAMA3_LLAMA_SERVER`, `CAMELID_MODEL_DIR`, and any Camelid runtime flags
- Hardware details: pending; record `uname -a`, CPU model, logical CPU count, memory, and GPU/Metal device for Metal mode
- OS version: pending; record `sw_vers` on macOS or `/etc/os-release` on Linux
- Raw command: pending; preserve the exact shell command
- Raw output: pending; preserve stdout/stderr or JSON artifact paths
- Timing data: pending; use harness TTFT, total elapsed, and streamed decode estimates
- Memory data: pending; record process RSS snapshots if available
- Pass/fail status: pending; require marker guard pass and no unexpected server fallback

## Blockers

- No `target/reference/llama.cpp/build/bin/llama-server` exists in this release worktree.
- No GGUF model files are present inside this release worktree.
- Existing retained same-host artifact is from source head `84a4a83bf881550f29dcea8349c2284439dfd900`, not the v0.1 release SHA.
- Metal comparison needs a Metal-enabled llama.cpp build and an explicit manually-started server because the default harness path starts llama.cpp in CPU-only mode.
