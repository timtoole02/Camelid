# Camelid v0.1 Release Status

Last updated: 2026-06-01 01:00 PDT / 2026-06-01 08:00 UTC

Branch: `release/v0.1-evidence`

Current release branch head SHA before this status update: `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`

Current QA repair SHA: `3001fa48e3d5fa41dbeca417dd511164a3bacc80`

Validated pre-doc-refresh branch head SHA: `5017ff28568dc1898fa490b4492848a1b3c022f0`

Current `v0.1.0-rc1` tag target: `d9fb294f47e3ae80291f969499e2240c6cd640c3`

Release target: `v0.1.0-rc1`

Release posture: the release branch includes the post-rc1 QA repair at `3001fa48e3d5fa41dbeca417dd511164a3bacc80`, a validated pre-doc-refresh head at `5017ff28568dc1898fa490b4492848a1b3c022f0`, and later status-only blocker documentation through `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`. The existing `v0.1.0-rc1` tag still points to `d9fb294f47e3ae80291f969499e2240c6cd640c3` and has not been moved. A clean-head llama.cpp CPU comparator bundle exists for one exact row, with llama.cpp Metal, Ollama, and MLX explicitly deferred. Final `v0.1.0` remains Tim-approval gated, and a Tim decision is needed before treating the repaired release branch as the active release candidate.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Re-verified the clean release worktree on branch `release/v0.1-evidence`; the dirty primary checkout was not modified.
- Confirmed local branch head and `origin/release/v0.1-evidence` at `d81bf6f0edfa0372e4b53fa109f33d83e14f650b` before this status-only refresh.
- Fetched from origin with `--prune --tags`; `origin/release/v0.1-evidence` remained unchanged.
- Re-recorded the active release blocker with current branch, remote, tag, hosted-CI, and local-dispatch evidence.
- Preserved the previously passed full local branch-head gate at `a94eabb241f662ac328c02f6f5bb47dc33a9a14e`; later commits through `d81bf6f0edfa0372e4b53fa109f33d83e14f650b` are release documentation/status updates.
- Kept comparator deferrals unchanged: llama.cpp Metal, Ollama, and MLX remain non-claims.
- Did not move the existing annotated `v0.1.0-rc1` tag and did not create final `v0.1.0`.

Evidence:

- Status observation timestamp: 2026-06-01 01:00 PDT / 2026-06-01 08:00 UTC.
- `git status --short --branch` returned `## release/v0.1-evidence...origin/release/v0.1-evidence` with no file changes.
- Branch/remote observation: `release/v0.1-evidence` and `origin/release/v0.1-evidence` both resolved to `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`.
- Recent release-history observation: `d81bf6f chore(release): refresh rc1 blocker evidence`; `82331e3 chore(release): refresh v0.1 gate blocker status`; `a94eabb chore(release): refresh v0.1 local gate evidence`; `8840b01 chore(release): refresh release decision blocker`; `e1978da chore(release): refresh rc1 blocker observation`.
- Remote tag observation: `git ls-remote --tags origin 'v0.1.0-rc1^{}'` returned `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Local annotated tag object: `v0.1.0-rc1`, tagger date 2026-05-31 13:09:59 -0700, target `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- `git describe --tags --exact-match HEAD` failed with `fatal: no tag exactly matches 'd81bf6f0edfa0372e4b53fa109f33d83e14f650b'`, confirming the current release branch head is not tagged.
- Workflow trigger audit: `.github/workflows/ci.yml` runs on `push` to `main`, `pull_request`, and `workflow_dispatch`; release-branch pushes do not automatically start CI.
- GitHub connector observation: workflow runs and combined commit statuses were empty for both current branch head `d81bf6f0edfa0372e4b53fa109f33d83e14f650b` and rc1 tag target `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Local `gh` observation: `command -v gh` returned no path, so local workflow dispatch remains unavailable from this environment.
- Local QA refresh passed at `a94eabb241f662ac328c02f6f5bb47dc33a9a14e`: `cargo fmt --all -- --check`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo check --all-targets --all-features`; `cargo test --all-targets --all-features --no-fail-fast`; `cargo build --release --bin camelid`; `cargo doc --no-deps --all-features`; frontend `npm ci`, build, model-state smoke, 3B closure smoke, integration smoke, streaming smoke, and UI smoke; all 20 `scripts/test-*.mjs`; benchmark harness self-test; public evidence-claim check; public scrub guard; README screenshot guard; v0.1 privacy audit; evidence checksum check; and `git diff --check`.
- Full Rust tests passed with lib 310 passed / 1 ignored, main 12 passed, plus API vertical slice, distributed, GGUF metadata, inference session, model binding, tensor primitive/store, tokenizer, and example suites.
- Frontend `npm ci` installed 21 packages and reported 0 vulnerabilities; Vite build and all frontend smokes passed.
- Public evidence claims checked 97 manifests and 49 summary files; v0.1 privacy audit generated `2026-06-01T04:38:31.387Z` with zero findings; evidence checksum verification passed.
- This status-only edit was rechecked with `node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles`, `bash scripts/check-public-scrub.sh`, `node scripts/test-readme-screenshot.mjs`, and `git diff --check`.
- Commits after `5017ff28568dc1898fa490b4492848a1b3c022f0` through `d81bf6f0edfa0372e4b53fa109f33d83e14f650b` are release documentation/status updates to `RELEASE_GATE_v0.1.md`, `RELEASE_REPORT_v0.1.md`, and `RELEASE_STATUS.md`.
- Real bundle: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- Bundle source SHA: `8026339531463ade269d7be7078da331ba3e4085`; git status was clean at run time.
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`.
- llama.cpp source commit: `399739d5c5978351f39e3454bfbfbab4f369088f`; run mode was CPU-only via `-ngl 0`.
- Marker guardrails passed for both Camelid and llama.cpp measured runs.
- Privacy audit passed with zero findings for the scrubbed bundle.

Blocker/Risk:

- The existing `v0.1.0-rc1` tag target predates the frontend/README QA repair at `3001fa48e3d5fa41dbeca417dd511164a3bacc80` and the current status-only branch head at `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`. The tag was not moved; Tim needs to decide whether to cut a new candidate, retag, or keep rc1 as a known-pre-fix candidate.
- llama.cpp coverage is one CPU-only exact-row run, not a full table and not Metal evidence.
- Ollama is deferred because the only installed row observed here was `llama3.1:8b`, not an approved exact release comparator row.
- MLX is deferred because `mlx_lm` is not installed in the default Python environment.
- The source SHA in the benchmark bundle predates later docs/evidence-publication commits; no runtime code changes were made after that clean-head run.
- No GitHub workflow run or commit status was visible for either the current release branch head or the rc1 tag target at the current observation time; this is an observability gap, not a reported CI failure.
- Remote CI is not configured to run automatically on release-branch pushes; a manual `workflow_dispatch` or PR-based run is needed for hosted CI evidence on this branch.
- Hosted CI dispatch attempt after pushing the QA repair was blocked locally: `gh workflow run ci.yml --ref release/v0.1-evidence` failed with `zsh:1: command not found: gh`.
- No local gate is currently recorded failing at current branch head `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`; this update is documentation-only.

Next:

- Keep this refreshed release-status observation on `origin/release/v0.1-evidence`.
- Continue soak on the release branch head and monitor for any later remote CI/status signal.
- Request a manual GitHub Actions dispatch on `release/v0.1-evidence` from an environment with workflow-dispatch credentials/tooling if hosted CI evidence is required before any final approval.
- Keep comparator deferrals explicit in any downstream notes: llama.cpp Metal, Ollama, and MLX are non-claims for rc1.
- Do not create `v0.1.0` final without Tim approval.

Need Tim:

- Decide whether the repaired release branch, including validated pre-doc-refresh head `5017ff28568dc1898fa490b4492848a1b3c022f0`, current status-only head `d81bf6f0edfa0372e4b53fa109f33d83e14f650b`, and this status-only follow-up, should become a new release candidate, a retagged rc1, or remain only on the release branch. No final `v0.1.0` tag is authorized by this release-captain update.

## Current Checkout

- Primary repo checkout inspected: `<primary-checkout>`
- Primary checkout state at start: `main`, SHA `1b207f953ad8d40abcd833bf4d4677b22d44b334`, behind `origin/main` by 17 commits, with existing uncommitted work.
- Release worktree: `<release-worktree>`
- Release worktree state before the QA repair: clean branch `release/v0.1-evidence` at `4b6787c2d51788cd7839f0ac462d1b2767aa82c9`, matching origin; remote `v0.1.0-rc1` remained at `d9fb294f47e3ae80291f969499e2240c6cd640c3`
- Preservation rule: the dirty primary checkout is not modified by this release lane.

## Release Captain Update Format

Camelid v0.1 update:

Shipped:

Evidence:

Blocker/Risk:

Next:

Need Tim:

## v0.1 Blockers

- Benchmark harness must generate a complete evidence bundle under `qa/evidence-bundles/v0.1/<timestamp>/`. Status: complete for `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- llama.cpp baseline must be pinned, reproducible, and separated by backend mode. Status: CPU-only exact-row baseline complete; Metal deferred.
- Ollama baseline must exist or be explicitly deferred with release-captain rationale. Status: deferred in `OLLAMA_BASELINE_v0.1.md`.
- MLX baseline must exist or be explicitly deferred with release-captain rationale. Status: deferred in `MLX_BASELINE_v0.1.md`.
- Correctness and support matrices must cite exact-row evidence only.
- README and release docs must remove unsupported speed, model-family, UI, and distributed claims.
- Final QA gate must run on this release branch and record commands, machine, SHA, timestamps, pass/fail, and notes.
- `v0.1.0-rc1` was created and pushed only after gates passed. Final `v0.1.0` requires Tim approval.

## Evidence Bundle Contract

Every benchmark result must record:

- Camelid commit SHA
- Comparator commit or version
- Model name
- Model path
- Model SHA256 hash
- Quantization
- Prompt
- Context size
- Max generated tokens
- Thread count
- Batch settings
- Runtime flags
- Environment variables
- Hardware details
- OS version
- Raw command
- Raw output
- Timing data
- Memory data
- Pass/fail status

## Release Gate Checklist

- [x] Repo builds cleanly.
- [x] Tests pass or failures are documented as non-release blockers.
- [x] Benchmark harness runs from a clean checkout.
- [x] llama.cpp baseline exists.
- [x] Ollama baseline exists or is explicitly deferred with reason.
- [x] MLX baseline exists or is explicitly deferred with reason.
- [x] Correctness matrix exists.
- [x] Support matrix exists.
- [x] README is updated and does not overclaim.
- [x] Release notes exist.
- [x] Evidence bundle exists.
- [x] Public docs contain no unsupported performance claims.
- [x] Release Captain signs off for `v0.1.0-rc1`.
- [x] Annotated `v0.1.0-rc1` tag is pushed and dereferences to the signed-off release commit.

## Lane Ownership

- Release Captain: release scope, evidence standards, final checklist, tag decision.
- Benchmark Harness: repeatable matrix runner and evidence bundle schema.
- Correctness and Parity: exact support matrix and parity proof boundaries.
- Apple Silicon Performance: macOS arm64 evidence only, clearly separated by runtime/backend mode.
- llama.cpp Comparator: pinned llama.cpp build and benchmark baseline.
- Ollama Comparator: practical user-facing benchmark baseline.
- MLX Comparator: Apple Silicon MLX market-context baseline.
- Distributed Mac Mini: included only if stable; otherwise explicitly excluded.
- Documentation: public README and release documents.
- QA and Release Gate: final commands, pass/fail ledger, and tag readiness.
