# Camelid v0.1 Release Gate

Date: 2026-05-31

Release checkout: branch-agnostic gate ledger; record the exact branch and SHA for the gate run that is used to cut rc1

Tag status: no tag created.

## Gate Summary

Current status: gate-refresh blockers cleared locally; not ready to tag because real comparator evidence is still missing.

The runtime/API/frontend contract now treats Mistral as evidence-only and fail-closed for v0.1. Lightweight code gates pass locally on this branch. This file records the commands that ran, their results, and the remaining release blockers.

## Required Lightweight Gates

Run these from the release-candidate checkout:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
node scripts/check-public-evidence-claims.mjs
bash scripts/check-public-scrub.sh
cd frontend && npm ci && npm run build && npm run smoke:model-state
```

## Gate Results

| Gate | Command | Status | Notes |
| --- | --- | --- | --- |
| Branch/SHA | `git status --short --branch && git rev-parse HEAD` | PASS | Confirmed a clean candidate checkout for that gate run; record the exact branch and SHA alongside the gate output that is used to cut rc1. |
| Rust format | `cargo fmt --all -- --check` | PASS | Source tree is formatted after applying the Mistral contract and clippy fixes. |
| Rust clippy | `CARGO_TERM_COLOR=never cargo clippy --all-targets --all-features -- -D warnings` | PASS | Clippy passed. Cargo emitted build-script hardlink warnings from the external target cache only. |
| Rust check | `CARGO_TERM_COLOR=never cargo check --all-targets --all-features` | PASS | Cargo check passed with external target-cache hardlink warnings only. |
| Rust tests | `CARGO_TERM_COLOR=never cargo test --all-targets --all-features --no-fail-fast` | PASS | Full suite passed: lib tests 310 passed / 1 ignored, main tests 12 passed, integration/example tests passed. Metal unit tests passed after test-only command-buffer reuse was disabled. |
| Release build | `CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Release binary built successfully. |
| Public evidence claims | `node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles` | PASS | Checked 96 manifest files and 49 summary files. |
| Public scrub | `bash scripts/check-public-scrub.sh` | PASS | No public scrub violations reported. |
| Frontend build/model-state smoke | `cd frontend && npm run build && npm run smoke:model-state` | PASS | Vite build passed and model-state smoke passed after removing Mistral from tracked full-support rows. |
| Benchmark harness self-test | `node tools/bench/test-v0.1-benchmark-harness.mjs` | PASS | Synthetic self-test passed; this is harness validation only, not real comparator evidence. |

## Comparator and Evidence Gates

| Gate | Status | Required before tag |
| --- | --- | --- |
| v0.1 evidence bundle | PARTIAL / BLOCKED | Dry-run bundle `qa/evidence-bundles/v0.1/dryrun-release-captain/` proves harness output shape only. Real Camelid/comparator benchmark entries are still required or must be explicitly deferred. No new real bundle was created in this gate-refresh slice. |
| llama.cpp baseline | BLOCKED | Run a pinned same-host baseline or explicitly defer with rationale. |
| MLX-LM baseline | PARTIAL | Memory comparison evidence exists; v0.1 speed baseline must be run or explicitly deferred. |
| Ollama baseline | BLOCKED | Run baseline or explicitly defer with rationale. |
| Support matrix | Out of scope for this lane | Owned by another lane; do not edit here. |
| Correctness matrix | Out of scope for this lane | Owned by another lane; do not edit here. |

## Current Blocking Failures

- Only a dry-run v0.1 evidence bundle exists; real benchmark evidence is missing.
- llama.cpp, Ollama, and MLX comparator baselines still need real runs or explicit release-captain deferrals.
- No rc tag is allowed yet.

## Tag Rule

Do not create `v0.1.0-rc1` or `v0.1.0` from this lane until:

- lightweight gates pass or have documented non-blocking failures
- comparator baseline status is resolved
- a fresh v0.1 evidence bundle exists or is explicitly deferred by the release captain
- release docs and README contain no unsupported performance, model-family, UI, or distributed claims
- release captain signs off
- Tim approves any final `v0.1.0` tag
