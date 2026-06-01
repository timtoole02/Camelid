# Camelid v0.1 Release Status

Last updated: 2026-05-31

Branch: `release/v0.1-evidence`

Current release SHA: `d9fb294f47e3ae80291f969499e2240c6cd640c3`

Release target: `v0.1.0-rc1`

Release posture: evidence release candidate `v0.1.0-rc1` is cut and pushed from the signed-off release branch. A clean-head llama.cpp CPU comparator bundle exists for one exact row, with llama.cpp Metal, Ollama, and MLX explicitly deferred. Final `v0.1.0` remains Tim-approval gated.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Re-verified the clean release worktree on branch `release/v0.1-evidence`.
- Confirmed the dereferenced remote tag `v0.1.0-rc1^{}` points to `d9fb294f47e3ae80291f969499e2240c6cd640c3`; `origin/release/v0.1-evidence` matched that SHA before this status-only follow-up update.
- Accepted the documented comparator deferrals for `v0.1.0-rc1`: llama.cpp Metal, Ollama, and MLX remain non-claims.
- Re-ran the lightweight release gates using an external Cargo target directory because the main filesystem had less than 1 GiB free.
- Signed off the exact-row support matrix, correctness boundary, README, release notes, benchmark posture, and public evidence bundle as rc1-ready.
- Confirmed the annotated `v0.1.0-rc1` tag is present locally and on origin. Final `v0.1.0` still requires Tim approval.

Evidence:

- Gate run timestamp: 2026-05-31 20:05-20:08 UTC.
- Remote observation timestamp: 2026-06-01 00:09 UTC.
- Branch/remote observation before this status-only follow-up update: `release/v0.1-evidence` at `d9fb294f47e3ae80291f969499e2240c6cd640c3`, matching `origin/release/v0.1-evidence`.
- Remote tag observation: `git ls-remote --tags origin 'v0.1.0-rc1^{}'` returned `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Local annotated tag object: `v0.1.0-rc1`, tagger date 2026-05-31 13:09:59 -0700, target `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- GitHub connector observation for `d9fb294f47e3ae80291f969499e2240c6cd640c3`: zero workflow runs and zero commit statuses returned.
- Workflow trigger audit: `.github/workflows/ci.yml` runs on `push` to `main`, `pull_request`, and `workflow_dispatch`; release-branch pushes do not automatically start CI.
- Public Actions API observation for branch `release/v0.1-evidence`: one older pull-request run, `26719718160`, failed at `069b4e205b1392a94af610d2450b76af8010851e`; no run was reported for the rc1 tag target or the post-rc1 status-note commits.
- Real bundle: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- Bundle source SHA: `8026339531463ade269d7be7078da331ba3e4085`; git status was clean at run time.
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`.
- llama.cpp source commit: `399739d5c5978351f39e3454bfbfbab4f369088f`; run mode was CPU-only via `-ngl 0`.
- Marker guardrails passed for both Camelid and llama.cpp measured runs.
- Privacy audit passed with zero findings for the scrubbed bundle.
- Local QA passed after the signoff refresh: Rust fmt, clippy, full tests, release build, frontend `npm ci`, frontend build/model-state smoke, public evidence-claim check, public scrub guard, harness self-test, privacy audit, and diff whitespace check.

Blocker/Risk:

- llama.cpp coverage is one CPU-only exact-row run, not a full table and not Metal evidence.
- Ollama is deferred because the only installed row observed here was `llama3.1:8b`, not an approved exact release comparator row.
- MLX is deferred because `mlx_lm` is not installed in the default Python environment.
- The source SHA in the benchmark bundle predates later docs/evidence-publication commits; no runtime code changes were made after that clean-head run.
- The local machine's main filesystem had less than 1 GiB free during signoff, so Rust build artifacts were kept outside the repository.
- No GitHub workflow run or commit status was visible for the rc1 commit at observation time; this is an observability gap, not a reported CI failure.
- Remote CI is not configured to run automatically on release-branch pushes; a manual `workflow_dispatch` or PR-based run is needed for hosted CI evidence on this branch.
- No local gate is currently failing.

Next:

- Continue rc1 soak and monitor for any later remote CI/status signal on `d9fb294f47e3ae80291f969499e2240c6cd640c3`.
- Run or request a manual GitHub Actions dispatch on `release/v0.1-evidence` if hosted CI evidence is required before final approval.
- Keep comparator deferrals explicit in any downstream notes: llama.cpp Metal, Ollama, and MLX are non-claims for rc1.
- Do not create `v0.1.0` final without Tim approval.

Need Tim:

- Approve or reject any final `v0.1.0` tag after rc1 soak/CI review. No final tag is authorized by this release-captain signoff.

## Current Checkout

- Primary repo checkout inspected: `<primary-checkout>`
- Primary checkout state at start: `main`, SHA `1b207f953ad8d40abcd833bf4d4677b22d44b334`, behind `origin/main` by 17 commits, with existing uncommitted work.
- Release worktree: `<release-worktree>`
- Release worktree state before this status-only follow-up update: clean branch `release/v0.1-evidence` at `d9fb294f47e3ae80291f969499e2240c6cd640c3`, matching origin and remote `v0.1.0-rc1`
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
