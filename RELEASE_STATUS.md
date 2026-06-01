# Camelid v0.1 Release Status

Last updated: 2026-05-31

Branch: `release/v0.1-evidence`

Current QA repair SHA: `3001fa48e3d5fa41dbeca417dd511164a3bacc80`

Current `v0.1.0-rc1` tag target: `d9fb294f47e3ae80291f969499e2240c6cd640c3`

Release target: `v0.1.0-rc1`

Release posture: the release branch now contains a post-rc1 QA repair at `3001fa48e3d5fa41dbeca417dd511164a3bacc80`. The existing `v0.1.0-rc1` tag still points to `d9fb294f47e3ae80291f969499e2240c6cd640c3` and has not been moved. A clean-head llama.cpp CPU comparator bundle exists for one exact row, with llama.cpp Metal, Ollama, and MLX explicitly deferred. Final `v0.1.0` remains Tim-approval gated, and a Tim decision is needed before treating the post-rc1 branch head as the final release candidate.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Re-verified the clean release worktree on branch `release/v0.1-evidence`; the dirty primary checkout was not modified.
- Found real QA drift after `v0.1.0-rc1`: the frontend integration smoke no longer saw the runtime-ready/support-gated empty-state blocker, and the README screenshot guard failed because the v0.1 README rewrite had dropped the approved screenshot.
- Restored the fresh-chat exact-row readiness pills so runtime-ready-but-support-gated rows visibly show `Runtime ready, support gated` plus the exact artifact blocker.
- Restored the approved README chat screenshot with boundary-safe caption text.
- Updated the frontend integration smoke to assert the current sendable empty-state hero copy and removed competitor-branded component/comment source text caught by the UI smoke.
- Committed the QA repair as `3001fa48e3d5fa41dbeca417dd511164a3bacc80`.
- Kept comparator deferrals unchanged: llama.cpp Metal, Ollama, and MLX remain non-claims.
- Did not move the existing annotated `v0.1.0-rc1` tag and did not create final `v0.1.0`.

Evidence:

- Gate-refresh timestamp: 2026-06-01 00:43-00:50 UTC.
- Branch/remote observation before the QA repair: `release/v0.1-evidence` and `origin/release/v0.1-evidence` at `4b6787c2d51788cd7839f0ac462d1b2767aa82c9`.
- Remote tag observation: `git ls-remote --tags origin 'v0.1.0-rc1^{}'` returned `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Local annotated tag object: `v0.1.0-rc1`, tagger date 2026-05-31 13:09:59 -0700, target `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- GitHub connector observation for `d9fb294f47e3ae80291f969499e2240c6cd640c3` and `4b6787c2d51788cd7839f0ac462d1b2767aa82c9`: zero workflow runs and zero commit statuses returned.
- Workflow trigger audit: `.github/workflows/ci.yml` runs on `push` to `main`, `pull_request`, and `workflow_dispatch`; release-branch pushes do not automatically start CI.
- Public Actions API observation for branch `release/v0.1-evidence`: one older pull-request run, `26719718160`, failed at `069b4e205b1392a94af610d2450b76af8010851e`; no run was reported for the rc1 tag target or the post-rc1 status-note commits.
- Pre-fix local failures reproduced: `npm run smoke:integration` failed at `frontend/scripts/frontend-integration-smoke.mjs:159`, `node scripts/test-readme-screenshot.mjs` failed because README did not reference `docs/assets/camelid-readme-chat-surface-dark.png`, and `npm run smoke:ui` failed on competitor-branded source text in `ChatWorkspace.jsx`.
- Post-fix local QA passed: Rust fmt, clippy, full tests, release build, cargo doc, frontend `npm ci`, frontend build, frontend model-state smoke, frontend 3B closure smoke, frontend integration smoke, frontend streaming smoke, frontend UI smoke, public evidence-claim check, public scrub guard, README screenshot guard, all `scripts/test-*.mjs`, benchmark harness self-test, privacy audit, evidence checksum check, and diff whitespace check.
- Real bundle: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- Bundle source SHA: `8026339531463ade269d7be7078da331ba3e4085`; git status was clean at run time.
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`.
- llama.cpp source commit: `399739d5c5978351f39e3454bfbfbab4f369088f`; run mode was CPU-only via `-ngl 0`.
- Marker guardrails passed for both Camelid and llama.cpp measured runs.
- Privacy audit passed with zero findings for the scrubbed bundle.

Blocker/Risk:

- The existing `v0.1.0-rc1` tag target predates the frontend/README QA repair. The tag was not moved; Tim needs to decide whether to cut a new candidate, retag, or keep rc1 as a known-pre-fix candidate.
- llama.cpp coverage is one CPU-only exact-row run, not a full table and not Metal evidence.
- Ollama is deferred because the only installed row observed here was `llama3.1:8b`, not an approved exact release comparator row.
- MLX is deferred because `mlx_lm` is not installed in the default Python environment.
- The source SHA in the benchmark bundle predates later docs/evidence-publication commits; no runtime code changes were made after that clean-head run.
- No GitHub workflow run or commit status was visible for the rc1 commit at observation time; this is an observability gap, not a reported CI failure.
- Remote CI is not configured to run automatically on release-branch pushes; a manual `workflow_dispatch` or PR-based run is needed for hosted CI evidence on this branch.
- Hosted CI dispatch attempt after pushing the QA repair was blocked locally: `gh workflow run ci.yml --ref release/v0.1-evidence` failed with `zsh:1: command not found: gh`.
- No local gate is currently failing at the QA repair commit `3001fa48e3d5fa41dbeca417dd511164a3bacc80`; subsequent release-status edits are documentation-only.

Next:

- The post-rc1 QA repair and release-status commits are pushed to `origin/release/v0.1-evidence`.
- Continue soak on the release branch head and monitor for any later remote CI/status signal.
- Request a manual GitHub Actions dispatch on `release/v0.1-evidence` from an environment with workflow-dispatch credentials/tooling if hosted CI evidence is required before any final approval.
- Keep comparator deferrals explicit in any downstream notes: llama.cpp Metal, Ollama, and MLX are non-claims for rc1.
- Do not create `v0.1.0` final without Tim approval.

Need Tim:

- Decide whether the post-rc1 QA repair should become a new release candidate, a retagged rc1, or remain only on the release branch. No final `v0.1.0` tag is authorized by this release-captain update.

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
