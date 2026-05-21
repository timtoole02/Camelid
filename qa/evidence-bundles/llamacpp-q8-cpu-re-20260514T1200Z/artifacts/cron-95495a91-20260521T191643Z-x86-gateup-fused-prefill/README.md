# cron-95495a91 x86 Q8 gate/up fused prefill

UTC: 2026-05-21T19:16:43Z

Source head: `3776d22` (`origin/main`, includes PR #49)

Branch: `backend-ubuntu-x86-q8-95495a91`

Slice retained:

- `src/inference.rs`: x86 packed rows4 FFN gate/up prefill route now emits the activated output directly from paired Q8 projections. This removes the separate full-width `up` tensor allocation and the follow-on activation pass on that gated route. The existing fallback/reference path remains unchanged.
- `src/inference/tests.rs`: added parity coverage proving fused gate/up prefill output matches the separate pair-projection plus activation path.

Validation run on the local automation host:

- `cargo check` passed.
- `cargo test q8_packed_rows4 -- --nocapture` passed: 5 passed.
- `cargo test q8_ffn_gate_up -- --nocapture` passed: 4 passed.

Timing/parity status:

- Same-host Ubuntu x86 Q8 timing was not feasible in this cron shell: host is `Darwin ... RELEASE_ARM64_T8132 arm64`.
- `cargo check --target x86_64-unknown-linux-gnu` was attempted, but the Homebrew Rust toolchain has no installed `std/core` for `x86_64-unknown-linux-gnu` and no `rustup` in PATH to add it.
- Parity was validated by the new focused fused-vs-unfused Q8 rows4 unit test. No Mac, Mixtral, or support-surface promotion is claimed.
