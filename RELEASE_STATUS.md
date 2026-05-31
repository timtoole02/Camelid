# Camelid v0.1 Release Status

Last updated: 2026-05-31

Branch: `release/v0.1-evidence`

Current release SHA: `v0.1.0-rc1` tag target after the release-captain signoff refresh

Release target: `v0.1.0-rc1`

Release posture: evidence release candidate signed off for `v0.1.0-rc1`. A clean-head llama.cpp CPU comparator bundle exists for one exact row, with llama.cpp Metal, Ollama, and MLX explicitly deferred. Final `v0.1.0` remains Tim-approval gated.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Re-verified the clean release worktree on branch `release/v0.1-evidence`.
- Accepted the documented comparator deferrals for `v0.1.0-rc1`: llama.cpp Metal, Ollama, and MLX remain non-claims.
- Re-ran the lightweight release gates using an external Cargo target directory because the main filesystem had less than 1 GiB free.
- Signed off the exact-row support matrix, correctness boundary, README, release notes, benchmark posture, and public evidence bundle as rc1-ready.
- Approved cutting `v0.1.0-rc1` from the signed-off release branch only; final `v0.1.0` still requires Tim approval.

Evidence:

- Gate run timestamp: 2026-05-31 20:05-20:08 UTC.
- Branch/remote before signoff refresh: `release/v0.1-evidence` at `ab0cbdecaff373c501a2f1383342f71cee0f4f0d`, matching `origin/release/v0.1-evidence`.
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
- Remote CI still needs normal observation after the pushed signoff commit/tag; no local gate is currently failing.

Next:

- Commit and push the release branch.
- Create and push annotated tag `v0.1.0-rc1` from the signed-off release branch.
- Observe remote/CI state after the push.

Need Tim:

- Approve or reject any final `v0.1.0` tag after rc1 soak/CI review. No final tag is authorized by this release-captain signoff.

## Current Checkout

- Primary repo checkout inspected: `<primary-checkout>`
- Primary checkout state at start: `main`, SHA `1b207f953ad8d40abcd833bf4d4677b22d44b334`, behind `origin/main` by 17 commits, with existing uncommitted work.
- Release worktree: `<release-worktree>`
- Release worktree state at start: clean branch `release/v0.1-evidence` from `origin/main` at the release branch HEAD
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
- `v0.1.0-rc1` may be created only after gates pass. Final `v0.1.0` requires Tim approval.

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
