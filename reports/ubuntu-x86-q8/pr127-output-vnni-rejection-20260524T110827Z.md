# PR #127 Ubuntu x86 Q8 rejection

- Date: 2026-05-24
- PR: `#127 perf: add default-off q8 output vnni decode route`
- Head: `74d71c4`
- Result: rejected for merge, retention, default-on, and support-claim purposes on fresh Ubuntu Linux x86_64 same-host evidence

## What ran

- Release build on current `main`
- Targeted PR tests:
  - `cargo test --locked --lib q8_output_vnni_decode`
  - `cargo test --locked --lib q8_0_vnni_pack_allows_output_gate_to_request_sidecar`
- One-token parity for baseline and VNNI head
- Same-host benchmark with `warmup=1`, `repeats=3`, `max_tokens=16`, unique prompt, and marker guard enabled

## Common benchmark env

- `CAMELID_PROFILE=experimental`
- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`
- `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on`
- `CAMELID_Q8_SCHED_TELEMETRY=on`
- `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`

## Measured result

- Baseline `main` route `x86_output_decode_owner`
  - Camelid TTFT `3329.59 ms`
  - Camelid total `3329.98 ms`
  - Camelid backend first content `2831.33 ms`
  - llama.cpp TTFT `347.18 ms`
  - llama.cpp total `535.36 ms`
- PR route `CAMELID_X86_Q8_OUTPUT_VNNI_DECODE=on`
  - Route hit `x86_output_vnni_decode_consumer`
  - Camelid TTFT `4510.49 ms`
  - Camelid total `4510.80 ms`
  - Camelid backend first content `3984.33 ms`
  - llama.cpp TTFT `345.14 ms`
  - llama.cpp total `534.77 ms`
- PR rawptr route `CAMELID_X86_Q8_OUTPUT_VNNI_DECODE=on CAMELID_X86_Q8_OUTPUT_VNNI_DECODE_RAWPTR=on`
  - Route hit `x86_output_vnni_decode_rawptr_consumer`
  - Camelid TTFT `4429.63 ms`
  - Camelid total `4429.90 ms`
  - Camelid backend first content `3923.67 ms`
  - llama.cpp TTFT `343.58 ms`
  - llama.cpp total `533.56 ms`

## Guardrails

- Baseline one-token output: Camelid `C`, llama.cpp `C`
- VNNI one-token output: Camelid `C`, llama.cpp `C`
- Marker guard passed for all measured 16-token runs

## Decision

- Standard VNNI route regressed Camelid TTFT by about `35.5%` versus current `main`
- Rawptr variant still regressed Camelid TTFT by about `33.0%`
- Keep the logits VNNI route default-off and reject the current slice until the same-host exact-row lane returns to at or below the current `main` baseline
