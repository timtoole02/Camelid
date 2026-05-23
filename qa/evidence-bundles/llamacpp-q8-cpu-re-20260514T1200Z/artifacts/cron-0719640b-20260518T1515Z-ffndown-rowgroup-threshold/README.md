# Ubuntu x86 Q8 FFN-down GEMM4 row-group scheduler threshold

Target: `CAMELID_X86_Q8_REPACK=on` + `CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL=on` + `CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED=on`, first FFN-down.

Feedback loop / hypotheses:
1. Existing row-group scheduler parallelizes by input row-group. For shallow prefills (few input groups), it under-feeds Rayon versus the output-group scheduler and increases context-switch/scheduler overhead.
2. A default-off row-group scheduler should fail closed to the existing output-group GEMM4 path below a minimum input-group threshold without changing Q8 parity.
3. The threshold must remain tunable for future same-host model probes.

Slice:
- Added `CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS` (default 8) behind the already default-off row-group scheduler gate.
- Below threshold, row-group scheduler requests use the existing GEMM4 output-group kernel; at/above threshold, behavior is unchanged.
- Added a unit guard for the min-input-groups gate and an ignored manual x86 benchmark that compares baseline output-group, forced old row-group (`min=1`), and thresholded row-group (`min=8`).

Gates:
- Local macOS: `cargo test -q q8_ffn_down_gemm4` -> PASS (5 passed, 1 ignored).
- Local macOS: `cargo fmt --check` -> PASS.
- Canonical Ubuntu x86_64 host with Rust toolchain PATH normalized:
  - `cargo test -q x86_q8_ffn_down_gemm4_row_group_schedule_respects_min_input_groups` -> PASS.
  - `CAMELID_X86_Q8_SCHED_BENCH_ITERS=30 cargo test -q q8_ffn_down_gemm4_row_group_threshold_benchmark -- --ignored --nocapture` -> PASS.

Canonical host benchmark output:

```text
rows=16 input_groups=4 input_width=256 output_width=1024 iterations=30 baseline_us=251236 forced_row_group_us=560972 thresholded_us=256118
```

Interpretation:
- Forced old row-group scheduler was ~2.23x slower than output-group baseline for the shallow-prefill synthetic FFN-down surface (560,972µs vs 251,236µs over 30 iterations).
- Thresholded row-group preserves parity and stays near the existing output-group baseline (256,118µs, +1.9% vs baseline) instead of taking the slow row-group path.
- This is not a broad throughput/support claim and does not compare against llama.cpp; it is a scheduler/context-switch tracer bullet inside the existing default-off Q8 GEMM4 path.

Retain/reject:
- Retain as a safe default-off scheduler guard for FFN-down GEMM4 row-group scheduling.
- Not promoted to default-on; no support-contract/docs widening.

Next tracer bullet:
- Run the same threshold gate against model-backed same-host FFN-down timing with `CAMELID_X86_Q8_REPACK=on`, then test whether attention output/QKV need the same input-group threshold seam.
