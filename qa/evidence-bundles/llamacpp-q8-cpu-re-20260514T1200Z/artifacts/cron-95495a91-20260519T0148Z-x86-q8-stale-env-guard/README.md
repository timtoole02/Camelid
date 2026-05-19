# cron-95495a91 backend tracer: x86 Q8 stale env guard

## Target
Sharpen the Ubuntu/Linux x86_64 Q8 execution-plan feedback loop for active Q8 performance/parity lanes by preventing stale default-off x86 Q8 experimental env state from leaking into retained/rejected runs.

## Hypotheses
1. Several Rust runtime Q8 experiment env knobs were consumed outside `ExecutionPlan` management, so a stale shell could accidentally alter an otherwise retained baseline run.
2. Managing those knobs and writing default-off values for boolean experiments preserves backend-owned packed runtime storage work while keeping the baseline fail-closed.
3. Execution-plan unit coverage is the right feedback loop for this control-plane slice; it is not a throughput, RSS, parity-envelope, API/frontend, or support-contract promotion.

## Change
- `src/execution_plan.rs`
  - Added the attention Q/K/V prefill consumer, packed-rows4 AVX2 dot/hoist, packed-rows4 serial decode, parallel input quantize, and packed-rows4 chunk tuning env keys to `MANAGED_ENV_KEYS`.
  - The x86 experimental AVX2 plan explicitly sets the new boolean experiment gates to `off`; chunk-size tuning is managed/cleared unless a future retained plan owns it.
  - Extended execution-plan tests to assert selected-plan defaults and stale-env clearing.
- `docs/performance/ubuntu-x86-q8.md`
  - Documented the control-plane guard without broadening Q8 performance or support claims.

## Evidence commands
- Ubuntu reachability smoke: operator-provided canonical SSH command shape succeeded (`stdout: camelid-ssh-ok`). Full private host/key path intentionally not committed to public evidence.
- `./scripts/with-rustup-cargo.sh test -q execution_plan::tests:: --lib`
- `./scripts/check-public-scrub.sh`
- `node scripts/audit-evidence-bundle-privacy.mjs --strict --root qa/evidence-bundles`
- `cargo fmt --check`
- `./scripts/with-rustup-cargo.sh clippy --all-targets --all-features -- -D warnings`
- `./scripts/with-rustup-cargo.sh test --all-targets --all-features`

## Gate results
Logs are under `logs/` with local checkout paths sanitized for public evidence.

- PASS: Ubuntu reachability smoke with the requested canonical SSH shape (`stdout: camelid-ssh-ok`).
- PASS: `./scripts/with-rustup-cargo.sh test -q execution_plan::tests:: --lib` — 13 execution-plan tests passed.
- PASS: `./scripts/check-public-scrub.sh`.
- PASS: `node scripts/audit-evidence-bundle-privacy.mjs --strict --root qa/evidence-bundles` — 0 findings.
- PASS: `cargo fmt --check`.
- PASS: `./scripts/with-rustup-cargo.sh clippy --all-targets --all-features -- -D warnings`.
- PASS: `./scripts/with-rustup-cargo.sh test --all-targets --all-features` — 408 tests passed, 1 ignored/manual benchmark.

## Repo state
- Base SHA before slice: `d6a4008b195c11d364644b45b35307c7eeb5598e`
- Changed files: `src/execution_plan.rs`, `docs/performance/ubuntu-x86-q8.md`, this evidence bundle.

## Retain/reject
Retain as a default-off backend control-plane hygiene slice. It prevents accidental experimental-gate leakage into Q8 performance/parity loops. It does not claim throughput/RSS improvement, parity-envelope expansion, frontend/API behavior, portability, default-on readiness, or support-contract widening.
