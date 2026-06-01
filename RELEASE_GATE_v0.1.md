# Camelid v0.1 Release Gate

Date: 2026-05-31 local / 2026-06-01 UTC

Branch: `release/v0.1-evidence`

Release QA repair SHA: `3001fa48e3d5fa41dbeca417dd511164a3bacc80`

Validated pre-doc-refresh branch head SHA: `5017ff28568dc1898fa490b4492848a1b3c022f0`

Existing `v0.1.0-rc1` tag target: `d9fb294f47e3ae80291f969499e2240c6cd640c3`

Tag status: the annotated `v0.1.0-rc1` tag is pushed and dereferences to `d9fb294f47e3ae80291f969499e2240c6cd640c3`. The release branch now contains post-rc1 QA repairs at `3001fa48e3d5fa41dbeca417dd511164a3bacc80` and status-only blocker documentation through `5017ff28568dc1898fa490b4492848a1b3c022f0`; the rc1 tag has not been moved. Final `v0.1.0` remains Tim-approval gated.

## Gate Summary

Current status: branch-head validation passed locally again at `5017ff28568dc1898fa490b4492848a1b3c022f0` after a post-rc1 frontend/README QA repair. The existing `v0.1.0-rc1` tag predates that repair, so Tim must decide whether to cut a new candidate, retag rc1, or keep rc1 as a known-pre-fix candidate.

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
| Remote rc1 branch/tag | `git ls-remote --heads origin release/v0.1-evidence`; `git ls-remote --tags origin 'v0.1.0-rc1^{}'`; GitHub connector workflow/status checks; public Actions API branch query | PASS / OBSERVED | At 2026-06-01 00:09 UTC, branch and dereferenced tag both pointed to `d9fb294f47e3ae80291f969499e2240c6cd640c3`; GitHub returned zero workflow runs and zero commit statuses for that commit. `.github/workflows/ci.yml` does not trigger on release-branch pushes. The public Actions API showed one older failed pull-request run on `069b4e205b1392a94af610d2450b76af8010851e`, not on the rc1 tag target. This status-only follow-up update does not retarget the rc1 tag. |
| Post-rc1 branch/tag observation | `git rev-parse HEAD`; `git ls-remote --heads origin release/v0.1-evidence`; `git ls-remote --tags origin 'v0.1.0-rc1^{}'`; GitHub connector workflow/status checks; public Actions API branch query | PASS / OBSERVED | At 2026-06-01 00:43 UTC, local branch and origin were at `4b6787c2d51788cd7839f0ac462d1b2767aa82c9`, while `v0.1.0-rc1^{}` remained at `d9fb294f47e3ae80291f969499e2240c6cd640c3`. GitHub returned zero workflow runs and zero commit statuses for both SHAs. The public Actions API still showed only older pull-request run `26719718160` at `069b4e205b1392a94af610d2450b76af8010851e`. |
| Branch-head refresh after hosted-CI blocker note | `git status --short --branch`; `git rev-parse HEAD`; `git ls-remote --heads origin release/v0.1-evidence`; `git ls-remote --tags origin 'v0.1.0-rc1^{}'`; `git describe --tags --exact-match HEAD`; GitHub connector workflow/status checks | PASS / OBSERVED | At 2026-06-01 01:17 UTC, local branch and origin were at `5017ff28568dc1898fa490b4492848a1b3c022f0`, while `v0.1.0-rc1^{}` remained at `d9fb294f47e3ae80291f969499e2240c6cd640c3`. `git describe --tags --exact-match HEAD` failed, confirming the validated pre-doc-refresh head is not tagged. GitHub returned zero workflow runs and zero commit statuses for both the branch head and rc1 tag target. |
| Post-rc1 frontend/README QA repair | `npm run smoke:integration`; `node scripts/test-readme-screenshot.mjs`; `npm run smoke:ui` | PASS AFTER FIX | Pre-fix failures reproduced for missing fresh-chat support-gated blocker copy, missing README screenshot contract, and competitor-branded source text. Repair commit `3001fa48e3d5fa41dbeca417dd511164a3bacc80` restores the exact-row readiness pills, README screenshot caption, current hero assertion, and neutral source naming. |
| Rust format | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" cargo fmt --all -- --check` | PASS | Source tree is formatted. |
| Rust clippy | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo clippy --all-targets --all-features -- -D warnings` | PASS | Clippy passed. |
| Rust check | `CARGO_TERM_COLOR=never cargo check --all-targets --all-features` | PASS | Earlier gate-refresh check passed; no Rust source changed in this evidence-publication update. |
| Rust tests | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo test --all-targets --all-features --no-fail-fast` | PASS | Full suite passed: lib tests 310 passed / 1 ignored, main tests 12 passed, integration/example tests passed, tokenizer tests passed, and Metal unit tests passed. |
| Release build | `CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Release binary built successfully. |
| Rust docs | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo doc --no-deps --all-features` | PASS | Rust docs generated successfully after the post-rc1 QA repair. |
| External release build for comparator run | `CARGO_TARGET_DIR="$EXTERNAL_VOLUME/Camelid/release-captain/v0.1-evidence/cargo-target" CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Built from release worktree SHA `8026339531463ade269d7be7078da331ba3e4085`; local filesystem had insufficient free space for normal in-worktree build artifacts. |
| Public evidence claims | `node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles` | PASS | Checked 97 manifest files and 49 summary files after the README screenshot restoration. |
| Public scrub | `bash scripts/check-public-scrub.sh` | PASS | No public scrub violations reported. |
| README screenshot guard | `node scripts/test-readme-screenshot.mjs` | PASS | README uses `docs/assets/camelid-readme-chat-surface-dark.png` with the approved dark collapsed-rail caption contract. |
| Frontend build/model-state smoke | `cd frontend && npm run build && npm run smoke:model-state` | PASS | Vite build passed and model-state smoke passed. |
| Frontend closure/integration/UI smokes | `cd frontend && npm run smoke:3b-closure && npm run smoke:integration && npm run smoke:streaming && npm run smoke:ui` | PASS | 3B closure, integration, streaming parser, and UI regression smokes passed after the repair. |
| Frontend dependency install | `cd frontend && npm ci --cache <external-npm-cache> --prefer-offline` | PASS | Installed 21 packages; npm audit found 0 vulnerabilities. |
| Validation script self-tests | `for test_script in scripts/test-*.mjs; do node "$test_script"; done` | PASS | All 20 root validation self-tests passed. |
| Benchmark harness self-test | `node tools/bench/test-v0.1-benchmark-harness.mjs` | PASS | Synthetic self-test passed; this is harness validation only. |
| Benchmark harness real run | `node tools/bench/v0.1-benchmark-harness.mjs --config target/v0.1-20260531T184150Z.config.json --timestamp 20260531T184150Z-real-local-raw --hash-models` | PASS | Clean-head source SHA `8026339531463ade269d7be7078da331ba3e4085`, marker guardrails passed, scrubbed bundle published at `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`. |
| Evidence privacy audit | `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles/v0.1/20260531T184150Z-real-local --strict` | PASS | Zero findings after macOS path-scrub update. |
| 2026-06-01 branch-head gate refresh | `cargo fmt --all -- --check`; `CARGO_TARGET_DIR='/Volumes/SSK Drive/cargo-target/Camelid-v0.1-evidence' CARGO_TERM_COLOR=never cargo clippy --all-targets --all-features -- -D warnings`; `CARGO_TARGET_DIR='/Volumes/SSK Drive/cargo-target/Camelid-v0.1-evidence' CARGO_TERM_COLOR=never cargo test --all-targets --all-features --no-fail-fast`; `CARGO_TARGET_DIR='/Volumes/SSK Drive/cargo-target/Camelid-v0.1-evidence' CARGO_TERM_COLOR=never cargo build --release --bin camelid`; `CARGO_TARGET_DIR='/Volumes/SSK Drive/cargo-target/Camelid-v0.1-evidence-doc' CARGO_TERM_COLOR=never cargo doc --no-deps --all-features`; frontend `npm ci`, build, and smokes; public/evidence guards | PASS | Refreshed at `5017ff28568dc1898fa490b4492848a1b3c022f0` from 2026-06-01 01:17-01:22 UTC. Full Rust tests passed with lib 310 passed / 1 ignored, main 12 passed, plus integration/example/tokenizer suites. Public evidence claims checked 97 manifests and 49 summaries. Evidence privacy audit had zero findings; bundle checksums passed. |

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
- Local validation passed after the signoff refresh; the pushed rc1 tag target is observed at `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Branch-head validation passed after the post-rc1 QA repair at `3001fa48e3d5fa41dbeca417dd511164a3bacc80`; the existing rc1 tag predates that repair and has not been moved.
- Branch-head validation passed again at `5017ff28568dc1898fa490b4492848a1b3c022f0`; the validated pre-doc-refresh head is not tagged.
- No GitHub workflow run or commit status was visible for the validated pre-doc-refresh head or rc1 commit at observation time; this is an observability gap, not a reported CI failure.
- Hosted CI is not automatically triggered by release-branch pushes because the workflow only runs on `push` to `main`, `pull_request`, and `workflow_dispatch`.
- Local hosted-CI dispatch was not available in this lane: `gh workflow run ci.yml --ref release/v0.1-evidence` failed with `zsh:1: command not found: gh`.

## Tag Rule

Do not create final `v0.1.0` from this lane until Tim approves it. `v0.1.0-rc1` is approved only after:

- lightweight gates pass or have documented non-blocking failures
- comparator baseline status is resolved
- a fresh v0.1 evidence bundle exists or is explicitly deferred by the release captain
- release docs and README contain no unsupported performance, model-family, UI, or distributed claims
- release captain signs off
- the signed-off branch/tag are pushed for remote observation
