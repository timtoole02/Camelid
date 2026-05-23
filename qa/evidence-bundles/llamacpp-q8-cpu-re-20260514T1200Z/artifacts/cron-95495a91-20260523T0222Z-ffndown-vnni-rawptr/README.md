# Cron 95495a91 FFN-Down VNNI Rawptr Same-Host Check

## Scope

Default-off Ubuntu/Linux x86_64 check for `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on` on exact row `llama32_3b_instruct_q8_0` using `/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf` and the local llama.cpp CPU server under `/home/ubuntu/work`.

This is bounded parity and timing evidence only. It does not promote the raw-pointer path to default-on and does not widen support, portability, or throughput claims.

## Host And Disk Gate

- `uname -sm`: `Linux x86_64`
- Disk guard before release build: `avail_kb=115575248 use_pct=43 target_count=4`
- Disk guard before rawptr-on benchmark: `avail_kb=115575224 use_pct=43 target_count=4`
- Disk guard before rawptr-off benchmark: `avail_kb=115575140 use_pct=43 target_count=4`
- Disk guard before targeted tests: `avail_kb=115575052 use_pct=43 target_count=4`
- Shared build target: `CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91`

## Commands

```bash
/home/ubuntu/bin/camelid-disk-guard.sh
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
cargo build --release
```

```bash
/home/ubuntu/bin/camelid-disk-guard.sh
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
export CAMELID_PROFILE=experimental
export CAMELID_X86_Q8_REPACK=on
export CAMELID_X86_Q8_KERNEL=avx2
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on
export CAMELID_STREAM_TIMING_DIAGNOSTICS=on
export CAMELID_Q8_SCHED_TELEMETRY=on
node scripts/bench-llama3-same-host.mjs \
  --backend http://127.0.0.1:8191 \
  --llama-url http://127.0.0.1:8193 \
  --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-id llama32-3b-q8 \
  --row-id llama32_3b_instruct_q8_0 \
  --backend-bin /home/ubuntu/work/camelid-targets/backend-95495a91/release/camelid \
  --llama-server /home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server \
  --max-tokens 8 --warmup 1 --repeats 2 --threads 8 --require-marker \
  --out /home/ubuntu/work/artifacts/cron-95495a91-20260523T0222Z-ffndown-vnni-rawptr/rawptr-on.json
```

```bash
/home/ubuntu/bin/camelid-disk-guard.sh
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
export CAMELID_PROFILE=experimental
export CAMELID_X86_Q8_REPACK=on
export CAMELID_X86_Q8_KERNEL=avx2
export CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on
unset CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR
export CAMELID_STREAM_TIMING_DIAGNOSTICS=on
export CAMELID_Q8_SCHED_TELEMETRY=on
node scripts/bench-llama3-same-host.mjs \
  --backend http://127.0.0.1:8195 \
  --llama-url http://127.0.0.1:8197 \
  --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-id llama32-3b-q8 \
  --row-id llama32_3b_instruct_q8_0 \
  --backend-bin /home/ubuntu/work/camelid-targets/backend-95495a91/release/camelid \
  --llama-server /home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server \
  --max-tokens 8 --warmup 1 --repeats 2 --threads 8 --require-marker \
  --out /home/ubuntu/work/artifacts/cron-95495a91-20260523T0222Z-ffndown-vnni-rawptr/rawptr-off.json
```

```bash
/home/ubuntu/bin/camelid-disk-guard.sh
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
cargo test q8_ffn_down_vnni_decode_rawptr -- --nocapture
```

## Results

Both benchmark runs passed the `CMLD-BENCH` marker guard for Camelid and llama.cpp.

| Slice | Camelid TTFT ms | Camelid total ms | Camelid backend generate ms | llama.cpp TTFT ms | llama.cpp total ms | FFN-down VNNI taken | FFN-down VNNI kernel us | Route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rawptr on | 495.67 | 495.97 | 438 | 189.72 | 379.58 | 140 | 95727 | `ffn_down.x86_vnni_decode_rawptr_consumer` |
| rawptr off | 505.40 | 505.58 | 444 | 184.67 | 375.22 | 140 | 92086 | `ffn_down.x86_vnni_decode_consumer` |

Targeted Rust tests passed: `3 passed; 0 failed` for `q8_ffn_down_vnni_decode_rawptr*`.

## Decision

Keep the raw-pointer VNNI FFN-down decode path default-off. It passed parity/marker and route-use checks, but this short same-host run does not justify a throughput promotion: Camelid remained slower than llama.cpp on TTFT and total elapsed, and rawptr-on had slightly higher recorded VNNI kernel time than rawptr-off in this sample.

Next exact action: rerun this slice with a larger repeated sample and paired perf counters around the FFN-down VNNI route, then reject or retain the rawptr route based on kernel-cycle and end-to-end timing, not marker-only parity.
