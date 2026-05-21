# Rust VNNI Scale Cache Slice

Cron: `1eeef0a5-1081-4514-a08d-dbba561c7835`
Timestamp: `20260521T203839Z`
Base: `origin/main` at `7904c25`
Branch: `rust-kernel-ubuntu-x86-q8-20260521-2034`

## Scope

This is a bounded, default-off Rust implementation slice for the Ubuntu x86 Q8 lane. It keeps the existing `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE` and `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR` gates and does not add Mac, Mixtral, public support, or default-on behavior.

The VNNI sidecar now stores decoded f32 scale lanes next to the original fp16 scale bits. The existing scalar/reference and AVX2/AVX512 VNNI decode consumers read the cached f32 scales instead of decoding fp16 scale bits inside each hot decode dot.

## Validation

- `cargo fmt --check` passed.
- `cargo check` passed on the local Darwin arm64 runner.
- `cargo clippy --all-targets -- -D warnings` passed on the local Darwin arm64 runner.
- `cargo test q8_ffn_down_vnni_decode --lib` passed: 2 passed.
- `cargo test q8_0_vnni_pack --lib` passed: 1 passed.
- `cargo test --lib` passed: 293 passed, 1 ignored.

## Benchmark Status

Same-host Camelid vs llama.cpp Ubuntu x86 Q8 benchmarking was not feasible in this run because the executing host was `Darwin ... arm64`. No throughput, TTFT, RSS, support, or default-on promotion is claimed from this slice.

## Retain Decision

Retain as a default-off implementation cleanup only. Performance promotion is explicitly rejected for this run until the same exact slice has fresh Ubuntu x86_64 parity plus same-host Camelid vs llama.cpp timing evidence.
