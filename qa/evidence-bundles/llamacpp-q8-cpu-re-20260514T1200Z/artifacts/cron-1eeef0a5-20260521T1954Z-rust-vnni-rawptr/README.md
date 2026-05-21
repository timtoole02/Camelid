# Rust VNNI Raw-Pointer Decode Slice

## Scope

- Lane: Ubuntu/Linux x86_64 Q8 only.
- Slice: default-off Rust AVX512-VNNI raw-pointer inner loop for the existing FFN-down VNNI decode sidecar.
- Gate: `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on`, only reachable after `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on` has produced the VNNI sidecar.
- Non-goals: no Mac, Mixtral, public support, default-on, or broad throughput promotion.

## Source Archaeology Input

This slice follows the retained archaeology conclusion in:

- `/Users/timtoole/.openclaw/workspace/projects/Camelid/qa/evidence-bundles/llamacpp-q8-cpu-re-20260521T0229Z-source-archaeology/REPORT-20260521T110203Z.md`
- `/Users/timtoole/.openclaw/workspace/projects/Camelid/qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-d44c49a4-20260521T1426Z-source-archaeology/remote-build-and-symbols.txt`

Relevant retained finding: on the canonical Ubuntu host/model envelope, llama.cpp decode routes through the AMX-buffer-backed one-row VNNI kernel, not the generic row-dot fallback. This Camelid slice therefore targets the VNNI decode path and leaves the scalar/reference fallback intact.

## Implementation Notes

- Added `q8_0_vnni_decode_1x64_projection_rawptr_avx512` in Rust, cfg-limited to `linux + x86_64`.
- The fast loop keeps accumulators in AVX512 registers across blocks, uses raw pointers for tile/input/output traversal, and stores one 64-output group at a time.
- Existing scalar/AVX2/AVX512 tile helpers remain the fallback path.
- ExecutionPlan now manages the raw-pointer gate as default-off to avoid stale experimental env leakage.

## Local Gates

See:

- `local-gates.command.txt`
- `local-gates.result.txt`

Local host was macOS, so the Linux-only raw-pointer parity test is cfg-skipped locally. The existing FFN-down VNNI fallback/denial tests and execution-plan gate tests passed. The same patch was then applied on the canonical Ubuntu x86_64 host and `cargo test q8_ffn_down_vnni_decode -- --nocapture` passed 5 tests, including the raw-pointer parity test.

## Same-Host Status

Same-host Camelid vs llama.cpp benchmark was run on the canonical Ubuntu x86_64 host with:

- model: `/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf`
- llama.cpp: `/home/ubuntu/work/llama.cpp-archaeology-3e037f3/build/bin/llama-server`
- Camelid gate set: `CAMELID_PROFILE=experimental`, `CAMELID_X86_Q8_REPACK=on`, `CAMELID_X86_Q8_KERNEL=avx2`, `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on`, `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on`
- harness: `scripts/bench-llama3-same-host.mjs --max-tokens 8 --warmup 1 --repeats 1 --threads 16`

Recorded bounded result in `same-host-rawptr-experimental.json`:

- marker guard: passed
- Camelid TTFT: 503.62 ms
- llama.cpp TTFT: 167.12 ms
- Camelid backend generate: 444 ms
- `ffn_down_vnni_decode_taken`: 140
- `ffn_down_vnni_decode_kernel_us`: 97096

Interpretation: retain implementation and route/parity evidence only. This run does not justify a throughput/support/default-on promotion; llama.cpp remains faster on the bounded same-host TTFT/total elapsed snapshot.
