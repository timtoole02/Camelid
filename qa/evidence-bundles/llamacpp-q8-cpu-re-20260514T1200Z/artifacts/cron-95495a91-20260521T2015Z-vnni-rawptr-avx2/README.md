# Ubuntu x86 Q8 VNNI rawptr AVX2 slice

Timestamp: 2026-05-21T20:21:01Z

Scope:
- Backend-only Rust implementation for the existing default-off `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR` gate.
- Extends the raw-pointer FFN-down VNNI decode loop from AVX512-VNNI-only to AVX2-capable Ubuntu x86_64 hosts.
- Keeps the existing scalar/rows4 fallback and the default-off VNNI decode/runtime gates.

Validation run in this worktree:
- `cargo fmt`
- `cargo check --all-targets --all-features`
- `cargo check --target x86_64-unknown-linux-gnu --lib` was attempted, but this host lacks the `x86_64-unknown-linux-gnu` Rust target (`can't find crate for core/std`).

Timing/parity status:
- Same-host Ubuntu x86_64 timing and parity were not feasible from this Darwin arm64 host.
- The PR includes a Linux x86_64 AVX2-specific parity test, `q8_ffn_down_vnni_decode_rawptr_avx2_matches_rows4_decode_baseline`, for canonical host execution.
- No throughput, support, or default-on claim is made from this local slice.
