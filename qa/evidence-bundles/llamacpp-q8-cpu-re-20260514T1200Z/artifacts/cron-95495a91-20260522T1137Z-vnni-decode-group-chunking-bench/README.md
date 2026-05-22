# cron-95495a91 VNNI decode group chunking benchmark

UTC: 2026-05-22T11:37Z

Host gate: `Linux x86_64` on the canonical Ubuntu validation host.

Source branch: `archaeology-1eeef0a5-20260522T0809Z`, head `ec584a847458`.

Model: `/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf`

llama.cpp: `/home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server`

## Result

The default-off VNNI raw-pointer decode route was reached, but the group-chunking throughput promotion is rejected for this run: the chunked candidate did not improve the bounded same-host measurement against the unchunked VNNI raw-pointer control.

Control, raw-pointer VNNI without group chunking, `max_tokens=4`:

- Camelid TTFT: 9646.58 ms
- Camelid total elapsed: 9647.15 ms
- Camelid backend generate: 4052 ms
- FFN-down VNNI route: 420 candidates, 112 taken
- FFN-down VNNI kernel time: 72964 us
- llama.cpp TTFT / total elapsed: 564.14 ms / 779.58 ms

Candidate, raw-pointer VNNI with `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUP_CHUNKING=on` and groups-per-chunk 16, `max_tokens=4`:

- Camelid TTFT: 9647.69 ms
- Camelid total elapsed: 9648.19 ms
- Camelid backend generate: 4082 ms
- FFN-down VNNI route: 420 candidates, 112 taken
- FFN-down VNNI kernel time: 79568 us
- llama.cpp TTFT / total elapsed: 559.40 ms / 773.47 ms

Measured delta, candidate vs control:

- TTFT: +1.11 ms, +0.01%
- Total elapsed: +1.04 ms, +0.01%
- Backend generate: +30 ms, +0.74%
- FFN-down VNNI kernel counter: +6604 us, +9.05%

Marker/parity guard, candidate, `max_tokens=8`:

- Camelid text: `CMLD-BENCH`
- llama.cpp text: `CMLD-BENCH`
- Guardrails passed: true
- Camelid TTFT / total elapsed: 9868.83 ms / 9869.91 ms
- Camelid backend generate: 4285 ms
- FFN-down VNNI route: 532 candidates, 168 taken
- FFN-down VNNI kernel time: 127766 us
- llama.cpp TTFT / total elapsed: 563.54 ms / 922.59 ms

## Ubuntu commands

Disk guard was run before cargo and before each benchmark segment:

```text
2026-05-22T11:41:50Z CHECK avail_kb=140056952 use_pct=31 target_count=8
2026-05-22T11:41:53Z AFTER avail_kb=140056952 use_pct=31 target_count=8
2026-05-22T11:43:12Z CHECK avail_kb=139981652 use_pct=31 target_count=8
2026-05-22T11:43:16Z AFTER avail_kb=139981644 use_pct=31 target_count=8
2026-05-22T11:43:49Z CHECK avail_kb=139981600 use_pct=31 target_count=8
2026-05-22T11:43:52Z AFTER avail_kb=139981600 use_pct=31 target_count=8
2026-05-22T11:45:24Z CHECK avail_kb=139981552 use_pct=31 target_count=8
2026-05-22T11:45:27Z AFTER avail_kb=139981552 use_pct=31 target_count=8
2026-05-22T11:46:38Z CHECK avail_kb=139981452 use_pct=31 target_count=8
2026-05-22T11:46:41Z AFTER avail_kb=139981448 use_pct=31 target_count=8
```

Cargo target discipline:

```bash
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
```

Validation/build commands:

```bash
cargo fmt --check
cargo test --lib q8_ffn_down_vnni_decode_rawptr_group_chunking_matches_unchunked_rawptr -- --nocapture
cargo test ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins -- --nocapture
cargo check --lib
cargo build --release --bin camelid
```

Benchmark environment for the corrected route-taking runs:

```bash
export CAMELID_BIN=/home/ubuntu/work/camelid-targets/backend-95495a91/release/camelid
export LLAMA3_LLAMA_SERVER=/home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server
export LLAMA3_GGUF=/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf
export CAMELID_PROFILE=experimental
export CAMELID_X86_Q8_REPACK=on
export CAMELID_X86_Q8_KERNEL=avx2
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on
export CAMELID_STREAM_TIMING_DIAGNOSTICS=on
export CAMELID_Q8_SCHED_TELEMETRY=on
```

Control command:

```bash
unset CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUP_CHUNKING
unset CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUPS_PER_CHUNK
timeout 600s node scripts/bench-llama3-same-host.mjs --backend http://127.0.0.1:8201 --llama-url http://127.0.0.1:8203 --model "$LLAMA3_GGUF" --model-id llama32-3b-q8-vnni-rawptr --row-id llama32_3b_instruct_q8_0 --llama-server "$LLAMA3_LLAMA_SERVER" --backend-bin "$CAMELID_BIN" --max-tokens 4 --warmup 0 --repeats 1 --threads 4 --unique-prompt --out "$OUTDIR/same-host-vnni-rawptr-control-kernel.json"
```

Candidate command:

```bash
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUP_CHUNKING=on
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUPS_PER_CHUNK=16
timeout 600s node scripts/bench-llama3-same-host.mjs --backend http://127.0.0.1:8205 --llama-url http://127.0.0.1:8207 --model "$LLAMA3_GGUF" --model-id llama32-3b-q8-vnni-chunk --row-id llama32_3b_instruct_q8_0 --llama-server "$LLAMA3_LLAMA_SERVER" --backend-bin "$CAMELID_BIN" --max-tokens 4 --warmup 0 --repeats 1 --threads 4 --unique-prompt --out "$OUTDIR/same-host-vnni-chunk-kernel.json"
```

Marker/parity command:

```bash
timeout 600s node scripts/bench-llama3-same-host.mjs --backend http://127.0.0.1:8211 --llama-url http://127.0.0.1:8213 --model "$LLAMA3_GGUF" --model-id llama32-3b-q8-vnni-chunk-marker --row-id llama32_3b_instruct_q8_0 --llama-server "$LLAMA3_LLAMA_SERVER" --backend-bin "$CAMELID_BIN" --max-tokens 8 --warmup 0 --repeats 1 --threads 4 --unique-prompt --require-marker --out "$OUTDIR/same-host-vnni-chunk-marker-kernel.json"
```

## Files

- `same-host-vnni-rawptr-control-kernel.json`: corrected route-taking control measurement.
- `same-host-vnni-chunk-kernel.json`: corrected route-taking chunked candidate measurement.
- `same-host-vnni-chunk-marker-kernel.json`: candidate marker/parity guard measurement.
- Earlier `same-host-vnni-rawptr-control.json` and `same-host-vnni-chunk.json` are retained as failed-gate evidence; they omitted `CAMELID_X86_Q8_KERNEL=avx2`, so the runtime plan rejected the VNNI route with `gate_off` and they are not used for the candidate/control timing decision.

Boundary: exact Ubuntu host, exact 3B Q8 row, exact short streaming prompt, and default-off route only. This does not promote support, portability, default-on behavior, RSS, production throughput, or broader model-family performance.
