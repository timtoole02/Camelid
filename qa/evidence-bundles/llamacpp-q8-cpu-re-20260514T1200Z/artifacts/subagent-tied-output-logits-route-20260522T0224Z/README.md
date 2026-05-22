# Ubuntu x86 Q8 tied output logits route

Date: 2026-05-22T02:22Z

Base commit: `b95dd3051397b160f3ad1e0a464e33c14ab3c168`

Ubuntu host: `canonical-private-ubuntu-validation-host`

Shared target dir: `$CAMELID_SHARED_TARGET`

Disk guard evidence:

- Setup: `2026-05-22T02:22:28Z CHECK avail_kb=158613992 use_pct=22 target_count=4`
- Tests/build: `2026-05-22T02:22:44Z CHECK avail_kb=158496660 use_pct=22 target_count=4`
- One-token route run: `2026-05-22T02:25:40Z CHECK avail_kb=157561960 use_pct=23 target_count=4`
- Benchmark: `2026-05-22T02:26:14Z CHECK avail_kb=157553716 use_pct=23 target_count=4`

Commands:

```sh
export CARGO_TARGET_DIR=$CAMELID_SHARED_TARGET
cargo fmt --all -- --check
cargo test q8_x86_repack_materializes_tied_embedding_as_output_runtime_storage --lib
cargo test x86_q8_output_decode_owner_path_uses_runtime_packed_storage --lib
cargo build --release
```

One-token route/parity command:

```sh
env RAYON_NUM_THREADS=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores \
  CAMELID_PROFILE=experimental \
  CAMELID_X86_Q8_REPACK=on \
  CAMELID_X86_Q8_KERNEL=avx2 \
  CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on \
  CAMELID_Q8_SCHED_TELEMETRY=on \
  CAMELID_STREAM_TIMING_DIAGNOSTICS=on \
  taskset -c 0-15 node scripts/bench-llama3-same-host.mjs \
    --model $CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf \
    --backend-bin $CAMELID_SHARED_TARGET/release/camelid \
    --llama-server $CAMELID_LLAMA_CPP_BIN/llama-server \
    --threads 16 --warmup 0 --repeats 1 --max-tokens 1 \
    --out parity-one-token-telemetry.json
```

One-token result:

- Camelid text: `C`
- llama.cpp text: `C`
- Required route: `logits.x86_output_decode_owner`
- Rejected fallback route: no `logits.q8_0_retained_blocks`
- Route detail: `calls=1`, `input_width=3072`, `output_width=128256`, `elapsed_us=4958`

Benchmark command:

```sh
env RAYON_NUM_THREADS=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores \
  CAMELID_PROFILE=experimental \
  CAMELID_X86_Q8_REPACK=on \
  CAMELID_X86_Q8_KERNEL=avx2 \
  CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on \
  CAMELID_Q8_SCHED_TELEMETRY=on \
  CAMELID_STREAM_TIMING_DIAGNOSTICS=on \
  taskset -c 0-15 node scripts/bench-llama3-same-host.mjs \
    --model $CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf \
    --backend-bin $CAMELID_SHARED_TARGET/release/camelid \
    --llama-server $CAMELID_LLAMA_CPP_BIN/llama-server \
    --threads 16 --warmup 1 --repeats 3 --max-tokens 16 \
    --require-marker --expected-marker CMLD-BENCH \
    --out benchmark-r3.json
```

Benchmark result:

- Marker guard: pass for all Camelid and llama.cpp measured runs.
- Camelid TTFT mean: `445.737 ms`
- llama.cpp TTFT mean: `176.527 ms`
- Camelid total elapsed mean: `446.04 ms`
- llama.cpp total elapsed mean: `410.41 ms`
- Camelid backend first content mean: `117.333 ms`
- Camelid backend generate mean: `383.333 ms`
- Top logits route: `logits.x86_output_decode_owner`, `elapsed_mean=24.665 ms`, `calls_mean=5`, `width=3072x128256`
- No benchmark measured run recorded `logits.q8_0_retained_blocks`.

Artifacts:

- `parity-one-token-telemetry.json`
- `logs/route-check-one-token-telemetry.json`
- `benchmark-r3.json`
- `benchmark-r3.stream-summary.json`
- `logs/route-check-benchmark-r3.json`
