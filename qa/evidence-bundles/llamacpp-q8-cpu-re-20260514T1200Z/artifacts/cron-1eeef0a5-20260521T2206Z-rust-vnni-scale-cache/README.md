# cron-1eeef0a5 Rust VNNI scale cache

UTC: 2026-05-21T22:06Z

Source base: `origin/main` at `7904c25` (`test(frontend): harden 3B exact artifact closure`)

Branch: `rust-kernel-ubuntu-x86-q8-20260521T1506Z`

Commit: `6bf014c` (`perf(q8): cache VNNI decode scales`)

PR: https://github.com/timtoole02/Camelid/pull/62

## Slice

Implemented a default-off Rust kernel cleanup for the existing Ubuntu/Linux x86_64 Q8 FFN-down VNNI decode path:

- `Q8_0VnniTile16` now keeps both raw GGUF fp16 scale bits (`scale_f16`) and decoded f32 scale lanes (`scale_f32`).
- The scalar VNNI decode path and the Linux x86_64 AVX512 raw-pointer loop consume `scale_f32` directly instead of decoding fp16 scale bits inside the hot loop.
- Existing raw scale bits remain available for layout/parity checks and no new public support, Mac, Mixtral, or default-on claim is made.

## Validation

Local host:

```text
Darwin arm64 local control host, kernel details redacted for public evidence hygiene
rustc 1.95.0 (59807616e 2026-04-14), host aarch64-apple-darwin
cargo 1.95.0 (f2d3ce0bd 2026-03-21)
```

Commands:

```bash
cargo fmt --check
cargo check
cargo test q8_0_vnni --lib
cargo test q8_ffn_down_vnni --lib
cargo test q8_vnni_tile16 --lib
cargo test vnni --lib
cargo test --lib
cargo check --target x86_64-unknown-linux-gnu
```

Results:

- `cargo fmt --check`: passed.
- `cargo check`: passed.
- `cargo test q8_0_vnni --lib`: passed, 1 test.
- `cargo test q8_ffn_down_vnni --lib`: passed, 2 tests on this host; Linux-only VNNI tests were cfg-excluded.
- `cargo test q8_vnni_tile16 --lib`: passed with 0 tests on this host because the test is x86 cfg-gated.
- `cargo test vnni --lib`: passed, 3 tests on this host.
- `cargo test --lib`: passed, 293 passed, 1 ignored.
- `cargo check --target x86_64-unknown-linux-gnu`: blocked because the Homebrew Rust toolchain does not have `core/std` for `x86_64-unknown-linux-gnu` and no target install was available in this run.

## Benchmark Status

Same-host Camelid vs llama.cpp Ubuntu x86 Q8 benchmarking was not feasible in this cron shell because the available host is Darwin arm64, not Ubuntu/Linux x86_64. This slice is retained as a bounded implementation/parity cleanup only, with no throughput claim.
