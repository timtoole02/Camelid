# Runtime foundations default-decode A/B — 2026-07-28

Scope: regression check for the default-neutral project 2/4/5/6/7 slice.
This is a small same-host smoke, not a support-promotion benchmark.

Compared:

- before: saved pre-slice release binary
  `target/kv-bench-binaries/camelid-after.exe`
- after: `target/release/camelid.exe`

Configuration:

- `Llama-3.2-3B-Instruct-Q8_0.gguf`
- deterministic CPU, 8 threads
- five-token prompt, 65 generated tokens
- one warmup plus two measured iterations per process
- balanced order: before, after, after, before
- f32 head-major KV baseline
- speculative decode, CUDA resident decode, and K-quant owner disabled

Results:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| decode tok/s, mean of 4 | 7.537 | 7.757 | +2.9% |
| prefill ms, mean of 4 | 173.22 | 175.75 | +1.5% |
| peak working set, mean | 4,235,455,488 B | 4,243,330,048 B | +0.19% |

All four before and all four after iterations emitted the same 65 token IDs.
The small timing and working-set differences are within ordinary same-host run
noise; this smoke found no default decode throughput or output regression.
