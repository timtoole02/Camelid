# K-quant Metal microbenchmark

This standalone macOS tool compiles the live Metal shader from `src/metal.rs`,
checks multi-column outputs bit-for-bit against repeated single-column v2
dispatches, and measures production-shaped Q4_K/Q6_K GEMV kernels plus a
math-free weight-stream probe.

Build and run from the repository root:

```bash
cargo build --release --manifest-path qa/speculative/kbench/Cargo.toml
qa/speculative/kbench/target/release/kbench q4kmma --rows 14336 --nsb 16
qa/speculative/kbench/target/release/kbench q6kmma --rows 4096 --nsb 56
```

`--iters N` controls timing repetitions. `KBENCH_SKIP_CHECK=1` disables the
identity check for measurement-only experiments. `CAMELID_METAL_SOURCE` can
point at another `metal.rs`; otherwise the tool resolves this repository's
source relative to its manifest.

Useful Llama-3-8B Q4_K_M shapes:

| Projection | Command arguments |
| --- | --- |
| Q4 up/gate | `q4kmma --rows 14336 --nsb 16` |
| Q4 down | `q4kmma --rows 4096 --nsb 56` |
| Q6 down | `q6kmma --rows 4096 --nsb 56` |
| output head | `q4kmma --rows 128256 --nsb 16` |
| ragged guard | either MMA case with `--rows 130 --nsb 3` |

## Gate boundary

A green microbenchmark is necessary, not sufficient. It proves that the live
MMA kernel matches the live single-token v2 kernel for the generated input at
widths 2 through 16. It does not certify the entire resident decode path or
long-running shader behavior.

Every candidate that survives here must also pass the warmed 4137-token,
96-output `bench-speculative` command in `HANDOFF.md`, with both
`lossless=true` and `first_divergent_generated_token_index=-1`. Run it at least
twice. The 2026-08-29 direct-fragment experiment passed this micro gate but
failed the warmed model gate at output token 0 or 11; see
`../KQUANT_MMA_OVERNIGHT.md`.
