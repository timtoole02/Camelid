# cron-0719640b main FFN decode-chain same-host benchmark

UTC: 2026-05-22T12:45Z

Host gate: `Linux x86_64` on the canonical Ubuntu validation host.

Source: `origin/main`, head `68fa3a65edf3c70d31bd8f15108dc851d54782ec` (`perf(q8): gate x86 FFN decode chain (#82)`).

Model: `/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf`

llama.cpp: `/home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server`

## Result

Retain this as a current-head same-host rejection for any FFN decode-chain throughput promotion. Marker parity passed for both engines, but Camelid remained substantially slower than llama.cpp on the bounded short streaming shape.

Default-off Camelid gates:

```bash
CAMELID_PROFILE=experimental
CAMELID_X86_Q8_REPACK=on
CAMELID_X86_Q8_KERNEL=avx2
CAMELID_X86_Q8_FFN_DECODE_CHAIN=on
CAMELID_STREAM_TIMING_DIAGNOSTICS=on
CAMELID_Q8_SCHED_TELEMETRY=on
```

Same-host `max_tokens=8`, `warmup=0`, `repeats=1`, `threads=4`, unique prompt, marker required:

- Camelid text guard: `CMLD-BENCH`, passed.
- llama.cpp text guard: `CMLD-BENCH`, passed.
- Camelid TTFT / total elapsed: 6799.74 ms / 6800.25 ms.
- Camelid backend first content / generate: 2661 ms / 3015 ms.
- Camelid backend Q8 calls: 174.
- llama.cpp TTFT / total elapsed: 575.91 ms / 934.20 ms.
- Delta vs llama.cpp: TTFT +1080.69%, total elapsed +627.92%.

The reported Camelid post-first-token throughput is not a promotion signal here; this prompt emits the marker in a small number of streamed chunks and the retained decision is based on the same-host TTFT/total elapsed guard.

## Ubuntu validation

Cargo target discipline:

```bash
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/cron-0719640b-main-20260522T1245Z
```

Validation/build commands:

```bash
cargo fmt --check
cargo test --lib x86_ffn_decode_chain_fuses_gate_up_activation_and_down_projection -- --nocapture
cargo test --lib ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins -- --nocapture
cargo check --lib
cargo build --release --bin camelid
```

Note: the first filtered unit-test command matched zero tests on this source head; the execution-plan opt-in test did run and passed. Build, check, and benchmark rc were all zero.

Benchmark command:

```bash
timeout 600s node scripts/bench-llama3-same-host.mjs --backend http://127.0.0.1:8221 --llama-url http://127.0.0.1:8223 --model "$LLAMA3_GGUF" --model-id llama32-3b-q8-ffn-chain --row-id llama32_3b_instruct_q8_0 --llama-server "$LLAMA3_LLAMA_SERVER" --backend-bin "$CAMELID_BIN" --max-tokens 8 --warmup 0 --repeats 1 --threads 4 --unique-prompt --require-marker --out "$OUTDIR/same-host-main-ffn-decode-chain.json"
```

## Files

- `same-host-main-ffn-decode-chain.json`: full same-host benchmark artifact.
- `summary.json`: compact metrics summary copied from the benchmark artifact.
- `SHA256SUMS`: artifact checksums using repository-relative filenames.

Boundary: exact Ubuntu x86_64 host class, exact 3B Q8 row, exact short streaming marker prompt, and the explicit default-off FFN decode-chain route only. This does not promote support, portability, default-on behavior, RSS, production throughput, or broader model-family performance.
