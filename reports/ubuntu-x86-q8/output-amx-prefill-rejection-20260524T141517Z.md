# Output AMX prefill Ubuntu x86 Q8 rejection

- Date: 2026-05-24
- Candidate: `perf(q8): gate output AMX prefill route`
- Feature commit: `e2ce7cd`
- Result: rejected for retention, default-on, and support-claim purposes on fresh Ubuntu Linux x86_64 same-host evidence

## What ran

- Release builds on current `main` and the candidate head
- Targeted candidate tests on the Ubuntu x86_64 validator:
  - `cargo test --locked --lib x86_q8_output_amx_prefill_matches_rows4_matmul_when_supported -- --nocapture`
  - `cargo test --locked --lib x86_q8_output_packed_rows4_matmul_is_plan_gated_and_prefill_limited -- --nocapture`
- One-token parity for the retained packed-rows4 baseline and the AMX-prefill candidate
- Same-host benchmark with `warmup=1`, `repeats=3`, `max_tokens=16`, unique prompt, and marker guard enabled

## Common benchmark env

- `CAMELID_PROFILE=experimental`
- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`
- `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL=on`
- `CAMELID_Q8_SCHED_TELEMETRY=on`
- `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`

## Candidate-only env

- `CAMELID_X86_Q8_AMX_REPACK=on`
- `CAMELID_X86_Q8_OUTPUT_AMX_PREFILL=on`

## Measured result

- Baseline retained packed-rows4 lane
  - Telemetry bucket `logits.q8_0_borrowed_packed_rows4`
  - Camelid TTFT `3283.36 ms`
  - Camelid total `3283.64 ms`
  - Camelid backend first content `2798.67 ms`
  - llama.cpp TTFT `377.79 ms`
  - llama.cpp total `570.25 ms`
- Candidate output AMX prefill lane
  - Focused Linux x86 AMX parity test passed on the validator
  - Telemetry still rolled the logits work into `logits.q8_0_borrowed_packed_rows4`
  - Camelid TTFT `4024.04 ms`
  - Camelid total `4024.35 ms`
  - Camelid backend first content `3523.67 ms`
  - llama.cpp TTFT `377.82 ms`
  - llama.cpp total `570.55 ms`

## Guardrails

- Baseline one-token output: Camelid `C`, llama.cpp `C`
- Candidate one-token output: Camelid `C`, llama.cpp `C`
- Marker guard passed for all measured 16-token runs

## Decision

- Enabling the AMX-prefill lane regressed Camelid TTFT by about `22.6%` versus the retained packed-rows4 baseline
- Backend first content regressed by about `25.9%`
- One-token parity stayed exact, but one-token Camelid TTFT still worsened from `5788.13 ms` to `7808.46 ms`
- Keep `CAMELID_X86_Q8_OUTPUT_AMX_PREFILL` default-off and reject the current slice until a fresh same-host rerun shows a wall-clock win and the route has a distinct telemetry label
