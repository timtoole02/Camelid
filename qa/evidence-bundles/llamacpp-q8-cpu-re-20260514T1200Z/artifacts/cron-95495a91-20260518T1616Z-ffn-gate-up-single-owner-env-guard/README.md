# cron-95495a91 backend tracer: ffn gate-up single-owner env guard

## Target
Sharpen the Ubuntu/Linux x86_64 Q8 execution-plan feedback loop for active Q8 performance/parity lanes by preventing stale FFN gate-up single-owner env state from leaking into a retained/rejected run.

## Hypotheses
1. `CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER` is consumed by the Rust runtime but was not managed by the execution planner, so a stale shell env could accidentally activate it outside the current retained baseline envelope.
2. Adding it to managed planner keys and explicitly setting it `off` in the x86 experimental baseline keeps the slice default-off without widening support claims.
3. The existing execution-plan tests plus the gate-up owner default-off test are enough feedback loop for this control-plane guard; no throughput retain claim is made.

## Change
- `src/execution_plan.rs`
  - Added `CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER` to `MANAGED_ENV_KEYS`.
  - The x86 experimental AVX2 plan now explicitly writes it as `off` with the other default-off Q8 consumer experiments.
  - Tests now assert the env is cleared/applied and reset between cases; also filled missing GEMM4 env cleanup in the fixture helper.

## Gates
- Local: `cargo fmt --check && cargo test execution_plan --lib` — pass.
- Local: `cargo test q8_ffn_gate_up_single_owner_is_default_off_and_requires_runtime_storage --lib` — pass.
- Ubuntu x86_64 validation refresh on 2026-05-24:
  - `./scripts/with-rustup-cargo.sh +1.87.0 fmt --all -- --check` — pass.
  - `./scripts/with-rustup-cargo.sh +1.87.0 test --lib planner_env_apply_clears_stale_x86_q8_decode_consumer_flags -- --nocapture` — pass.
  - `./scripts/with-rustup-cargo.sh +1.87.0 test --lib ubuntu_experimental_validated_gates_select_rust_avx2_q8_path -- --nocapture` — pass.
  - `./scripts/with-rustup-cargo.sh +1.87.0 test --lib q8_ffn_gate_up_single_owner_is_default_off_and_requires_runtime_storage -- --nocapture` — pass.
  - `./scripts/with-rustup-cargo.sh +1.87.0 build --release --bin camelid` — pass.
  - Environment gate and toolchain summary: `uname -sm` returned `Linux x86_64`; the validated wrapper reported `cargo 1.87.0` and `rustc 1.87.0`; the run stayed on the shared external target lane only.
- Historical note: the host-default Rust installation previously observed on this lane was too old for the repo lockfile/MSRV, but the current validated path uses the repo's Rustup wrapper rather than the host-default cargo.

## Retain/reject
Retain as a default-off backend control-plane guard with fresh Ubuntu x86_64 validation. This is parity/performance-lane hygiene only: it prevents accidental gate leakage and does not claim throughput, parity-envelope expansion, frontend/API support, or public support widening.
