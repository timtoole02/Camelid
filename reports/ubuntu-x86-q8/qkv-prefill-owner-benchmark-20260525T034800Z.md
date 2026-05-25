# QKV prefill owner Ubuntu x86 Q8 benchmark

- Date: 2026-05-25
- Branch head: `431092f`
- Candidate: `fix(q8): honor qkv prefill owner gate`
- Result: reject the shared-quantized-input QKV prefill owner route for performance retention; keep the flag default-off and retain this run only as route-attribution evidence

## What changed

- `perf(q8): trace x86 qkv prefill route`
- `test: clear qkv prefill gate between plan cases`
- `fix(q8): honor qkv prefill owner gate`

## What ran

- Targeted Ubuntu Linux x86_64 tests:
  - `attention_qkv_prefill`
- Release build of Camelid on the candidate head
- Same-host streaming benchmark on the exact Llama 3.2 3B Instruct Q8_0 row with:
  - retained route-off baseline
  - route-on candidate via `CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER=on`
  - unique prompt marker guard enabled

## Common benchmark env

- `CAMELID_PROFILE=experimental`
- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`
- `CAMELID_Q8_SCHED_TELEMETRY=on`
- `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`

## Route evidence

- Baseline route-off run:
  - `attention_qkv.packed_rows4_matmul_prefill` calls: none recorded
  - `attention_qkv.packed_rows4_matmul_prefill.plan_off` denials mean: `168`
- Candidate route-on run:
  - `attention_qkv.packed_rows4_matmul_prefill` calls mean: `28`
  - `attention_qkv.packed_rows4_matmul_prefill` elapsed mean: `1371.327 ms`
  - `attention_qkv.packed_rows4_matmul_prefill.decode_or_empty_input` denials mean: `168`
- Interpretation:
  - The owner-gate fix made `CAMELID_X86_Q8_ATTENTION_QKV_PREFILL_CONSUMER=on` actually own the shared Q/K/V prefill route.
  - The route was definitely exercised in the candidate run, so the timing regression is attributable to the specialized path rather than route drift.

## Measured result

- Baseline route-off:
  - Camelid TTFT `3260.20 ms`
  - Camelid total `3260.56 ms`
  - llama.cpp TTFT `319.32 ms`
  - llama.cpp total `508.83 ms`
- Candidate route-on:
  - Camelid TTFT `4043.61 ms`
  - Camelid total `4043.92 ms`
  - llama.cpp TTFT `319.64 ms`
  - llama.cpp total `507.68 ms`

## Guardrails

- Targeted `attention_qkv_prefill` tests passed on Ubuntu Linux x86_64.
- The benchmark marker guard passed for both baseline and candidate runs.
- llama.cpp stayed effectively flat between the two runs, so the Camelid regression is not explained by major host drift during the measured pair.

## Decision

- Reject this route for performance retention:
  - Camelid TTFT regressed by about `24.03%`
  - Camelid total elapsed regressed by about `24.02%`
- Retain only the bounded implementation/evidence value:
  - the owner gate now behaves honestly
  - route telemetry proves exactly when the specialized QKV prefill path runs
- Do not promote this path to default-on, support, portability, or broad throughput claims.
