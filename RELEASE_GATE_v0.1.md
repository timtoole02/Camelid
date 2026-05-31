# Camelid v0.1 Release Gate

Date: 2026-05-31

Branch: `release/v0.1-evidence`

Release candidate SHA: `v0.1.0-rc1` tag target after the release-captain signoff refresh

Tag status: release-captain approved `v0.1.0-rc1`; final `v0.1.0` remains Tim-approval gated.

## Gate Summary

Current status: release-captain signoff validation passed locally; `v0.1.0-rc1` may be cut from the signed-off release branch.

The runtime/API/frontend contract treats Mistral as evidence-only and fail-closed for v0.1. A real llama.cpp CPU same-host comparator row now exists for Llama 3.2 3B Instruct Q8_0. Ollama, MLX, and llama.cpp Metal are explicitly deferred and must remain non-claims.

## Required Lightweight Gates

Run these from the release checkout:

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
| Branch/SHA | `git status --short --branch`; `git rev-parse HEAD`; `git ls-remote --heads origin release/v0.1-evidence` | PASS | Confirmed clean `release/v0.1-evidence`; branch matched `origin/release/v0.1-evidence` at `ab0cbdecaff373c501a2f1383342f71cee0f4f0d` before the signoff docs refresh. |
| Rust format | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" cargo fmt --all -- --check` | PASS | Source tree is formatted. |
| Rust clippy | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo clippy --all-targets --all-features -- -D warnings` | PASS | Clippy passed. |
| Rust check | `CARGO_TERM_COLOR=never cargo check --all-targets --all-features` | PASS | Earlier gate-refresh check passed; no Rust source changed in this evidence-publication update. |
| Rust tests | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo test --all-targets --all-features --no-fail-fast` | PASS | Full suite passed: lib tests 310 passed / 1 ignored, main tests 12 passed, integration/example tests passed, tokenizer tests passed, and Metal unit tests passed. |
| Release build | `CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Release binary built successfully. |
| External release build for comparator run | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Built from release worktree SHA `8026339531463ade269d7be7078da331ba3e4085`; local filesystem had insufficient free space for normal in-worktree build artifacts. |
| Public evidence claims | `node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles` | PASS | Checked 98 manifest files and 49 summary files after adding the v0.1 public bundle manifest. |
| Public scrub | `bash scripts/check-public-scrub.sh` | PASS | No public scrub violations reported. |
| Frontend build/model-state smoke | `cd frontend && npm run build && npm run smoke:model-state` | PASS | Vite build passed and model-state smoke passed. |
| Frontend dependency install | `cd frontend && npm ci --cache <external-npm-cache> --prefer-offline` | PASS | Installed 21 packages; npm audit found 0 vulnerabilities. |
| Benchmark harness self-test | `node tools/bench/test-v0.1-benchmark-harness.mjs` | PASS | Synthetic self-test passed; this is harness validation only. |
| Benchmark harness real run | `node tools/bench/v0.1-benchmark-harness.mjs --config target/v0.1-20260531T184150Z.config.json --timestamp 20260531T184150Z-real-local-raw --hash-models` | PASS | Clean-head source SHA `8026339531463ade269d7be7078da331ba3e4085`, marker guardrails passed, scrubbed bundle published at `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`. |
| Evidence privacy audit | `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles/v0.1/20260531T184150Z-real-local --strict` | PASS | Zero findings after macOS path-scrub update. |

## Comparator and Evidence Gates

| Gate | Status | Required before tag |
| --- | --- | --- |
| v0.1 evidence bundle | PASS | Real bundle `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`; dry-run bundle remains shape evidence only. |
| llama.cpp baseline | PARTIAL / ACCEPTED FOR CPU | Pinned CPU-only same-host baseline exists for `llama32_3b_instruct_q8_0`; Metal is deferred and must not be implied. |
| MLX-LM baseline | DEFERRED | `mlx_lm` is not installed in the default Python environment; historical MLX memory comparison remains context only. |
| Ollama baseline | DEFERRED | Installed `llama3.1:8b` is not an approved exact-row/quant-equivalent v0.1 comparator. |
| Support matrix | Out of scope for this lane | Owned by another lane; do not edit here. |
| Correctness matrix | Out of scope for this lane | Owned by another lane; do not edit here. |

## Current Blocking Failures

- No command failure is currently recorded for the real llama.cpp CPU bundle.
- The release captain accepts one llama.cpp CPU row plus explicit Metal/Ollama/MLX deferrals for `v0.1.0-rc1` because public docs make no comparator win or parity claims.
- Local validation passed after the signoff refresh; pushed branch/tag state still needs normal remote/CI observation.

## Tag Rule

Do not create final `v0.1.0` from this lane until Tim approves it. `v0.1.0-rc1` is approved only after:

- lightweight gates pass or have documented non-blocking failures
- comparator baseline status is resolved
- a fresh v0.1 evidence bundle exists or is explicitly deferred by the release captain
- release docs and README contain no unsupported performance, model-family, UI, or distributed claims
- release captain signs off
- the signed-off branch/tag are pushed for remote observation
