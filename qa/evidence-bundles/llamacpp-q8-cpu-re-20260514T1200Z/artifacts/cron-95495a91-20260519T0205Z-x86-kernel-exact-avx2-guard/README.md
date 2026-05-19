# cron-95495a91 backend tracer: x86 Q8 kernel exact-AVX2 guard

## Target
Sharpen the Ubuntu/Linux x86_64 Q8 execution-plan and runtime feedback loop by making `CAMELID_X86_Q8_KERNEL` an exact selector: only `avx2` opts into the x86 packed-kernel path, while stale boolean aliases such as `on`, `1`, or `true` fail closed.

## Hypotheses
1. The execution planner already required the exact `CAMELID_X86_Q8_KERNEL=avx2` shape, but the lower-level x86 kernel gate also accepted boolean aliases.
2. Sharing the exact-AVX2 predicate shape and adding unit coverage prevents ambiguous stale env values from changing Q8 performance/parity runs.
3. This is control-plane hygiene only: it does not claim throughput, RSS, parity-envelope expansion, frontend/API behavior, portability, default-on readiness, or support-contract widening.

## Change
- `src/inference.rs`
  - Narrowed the x86 AVX2 kernel gate so `CAMELID_X86_Q8_KERNEL` accepts only an explicit `avx2` value, case-insensitive with surrounding whitespace ignored.
  - Added a Rust unit test proving boolean aliases are rejected by the selector helper.
- `src/execution_plan.rs`
  - Routed planner AVX2 selection through the same exact-selector predicate shape.
  - Added an execution-plan test proving `CAMELID_PROFILE=experimental CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=on` still selects the safe Q8 path.
- `docs/CONFIGURATION.md`
  - Documented the exact selector and fail-closed boolean-alias behavior without broadening public support claims.

## Evidence commands
- `./scripts/check-public-scrub.sh`
- `node scripts/audit-evidence-bundle-privacy.mjs --strict --root qa/evidence-bundles`
- `node scripts/check-public-evidence-claims.mjs`
- `cargo fmt --check`
- `./scripts/with-rustup-cargo.sh clippy --all-targets --all-features -- -D warnings`
- `./scripts/with-rustup-cargo.sh test --all-targets --all-features`
- `./scripts/with-rustup-cargo.sh test -q x86_q8_kernel -- --nocapture`
- `./scripts/with-rustup-cargo.sh test -q x86_kernel_on_alias_does_not_select_avx2_plan -- --nocapture`

## Gate results
Logs are under `logs/` with local checkout paths sanitized for public evidence.

- PASS: `./scripts/check-public-scrub.sh`.
- PASS: `node scripts/audit-evidence-bundle-privacy.mjs --strict --root qa/evidence-bundles` — 0 findings.
- PASS: `node scripts/check-public-evidence-claims.mjs` — 95 manifests, 48 summaries.
- PASS: `cargo fmt --check`.
- PASS: `./scripts/with-rustup-cargo.sh clippy --all-targets --all-features -- -D warnings`.
- PASS: `./scripts/with-rustup-cargo.sh test --all-targets --all-features` — 408 tests passed, 1 ignored/manual benchmark.
- PASS: slice-specific `x86_q8_kernel` and `x86_kernel_on_alias_does_not_select_avx2_plan` unit-test filters.

## Repo state
- Base SHA before slice: `4003897ee8cf1fc8078808d34d949d5563c2ef12`
- Changed files: `src/inference.rs`, `src/execution_plan.rs`, `docs/CONFIGURATION.md`, this evidence bundle.

## Retain/reject
Retain as a default-off backend control-plane guard. It makes ambiguous x86 packed-kernel env values fail closed and keeps Q8 performance/parity lanes tied to the explicit retained evidence shape `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`.
