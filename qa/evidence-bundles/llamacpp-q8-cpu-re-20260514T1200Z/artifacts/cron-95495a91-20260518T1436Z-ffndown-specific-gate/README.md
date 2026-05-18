# cron-95495a91 FFN-down specific packed-rows4 gate

Target: sharpen the Q8 FFN-down packed-rows4 prefill feedback loop by adding a projection-family-specific default-off gate while preserving the legacy evidence command gate.

Change: `CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL` now enables only the FFN-down packed-rows4 matmul plan bit; legacy `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL` remains accepted so existing evidence commands do not break. No public support claim changed.

Feedback loop:
- `cargo fmt --check`
- `cargo test --lib`
- exact Ubuntu SSH smoke with the requested command shape

Retain/reject: retain. This is Rust-native, default-off, family scoped, and covered by a focused unit test plus the full lib gate. It is not throughput/parity promotion evidence.
