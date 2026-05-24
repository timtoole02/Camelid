# Planner passthrough knobs Ubuntu x86 Q8 benchmark

- Date: 2026-05-24
- Candidate: `fix(q8): manage planner scheduler passthrough knobs`
- Feature commit: `13e7f94`
- Result: retain as a bounded default-off follow-up candidate; do not widen this into a throughput, support, portability, or default-on claim

## What ran

- Targeted planner tests on the Ubuntu Linux x86_64 validator:
  - `cargo test --locked planner_env_apply -- --nocapture`
- Release builds on current `main` and the candidate head
- One-token parity for the retained FFN-down route mix and the planner-knob candidate
- Same-host benchmark with `warmup=1`, `repeats=3`, `max_tokens=16`, unique prompt, and marker guard enabled

## Common benchmark env

- `CAMELID_PROFILE=experimental`
- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`
- `CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL=on`
- `CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED=on`
- `CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2=on`
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER=on`
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING=on`
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUPS_PER_CHUNK=2`
- `CAMELID_Q8_SCHED_TELEMETRY=on`
- `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`

## Route evidence

- Both baseline and candidate kept the intended FFN-down route mix active on the validator:
  - `ffn_down_gemm4_prefill_reject_plan_off = 0`
  - `ffn_down_decode_consumer_taken = 168`
  - `x86_decode_consumer_group_chunking` stayed live on the decode tail rows
- The candidate did not widen route coverage or change the one-token result; it only changed planner passthrough handling for the owned chunk-size knobs.

## Measured result

- Baseline current `main`
  - Camelid TTFT `2948.81 ms`
  - Camelid total `2949.26 ms`
  - Camelid backend first content `2467.00 ms`
  - Camelid backend generate `2873.67 ms`
  - llama.cpp TTFT `304.58 ms`
  - llama.cpp total `492.51 ms`
- Candidate `13e7f94`
  - Camelid TTFT `2904.05 ms`
  - Camelid total `2904.47 ms`
  - Camelid backend first content `2443.00 ms`
  - Camelid backend generate `2830.33 ms`
  - llama.cpp TTFT `309.43 ms`
  - llama.cpp total `497.52 ms`

## Consistency and parity

- Candidate beat baseline on all three measured Camelid 16-token runs:
  - baseline total `2977.72 / 2943.58 / 2926.49 ms`
  - candidate total `2900.77 / 2911.80 / 2900.85 ms`
- Exact one-token output stayed `C` for both baseline and candidate
- One-token Camelid first-content timing improved from `5518.92 ms` to `5480.76 ms`

## Decision

- Keep the planner passthrough fix alive for follow-up because this bounded same-host run shows a small but consistent wall-clock win:
  - TTFT improved by about `1.52%`
  - total elapsed improved by about `1.52%`
  - backend first-content improved by about `0.97%`
  - backend generate improved by about `1.51%`
- This is still only bounded default-off evidence. Camelid remains materially slower than same-host llama.cpp, so there is no default-on, support, or throughput-promotion action here.
- Next exact action: rerun the same candidate at higher repeat count on the canonical Ubuntu lane and inspect whether the passthrough-fix win survives a broader `r7` sample without route drift.
