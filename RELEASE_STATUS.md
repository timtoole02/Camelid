# Camelid v0.1 Release Status

Last updated: 2026-05-31

Branch: `release/v0.1-evidence`

Current release SHA: release branch HEAD after this evidence-publication update

Release target: `v0.1.0-rc1`

Release posture: evidence release candidate in progress. A clean-head llama.cpp CPU comparator bundle now exists for one exact row, with Ollama and MLX explicitly deferred. No tag was created in this automation slice.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Built Camelid from the clean release worktree using an external Cargo target directory because the local filesystem had only about 246 MiB free after an attempted llama.cpp checkout.
- Built a pinned external llama.cpp comparator at source commit `399739d5c5978351f39e3454bfbfbab4f369088f`.
- Captured a real v0.1 same-host llama.cpp CPU comparator bundle for `llama32_3b_instruct_q8_0`.
- Published a scrubbed public bundle under `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- Tightened the public evidence publisher so macOS home paths and mounted model/build paths are scrubbed before publication.
- Explicitly deferred Ollama and MLX fresh baselines with release-captain rationale and no public win claims.

Evidence:

- Real bundle: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.
- Bundle source SHA: `8026339531463ade269d7be7078da331ba3e4085`; git status was clean at run time.
- Model SHA256: `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`.
- llama.cpp source commit: `399739d5c5978351f39e3454bfbfbab4f369088f`; run mode was CPU-only via `-ngl 0`.
- Marker guardrails passed for both Camelid and llama.cpp measured runs.
- Privacy audit passed with zero findings for the scrubbed bundle.
- Local QA passed after the evidence-publication update: Rust fmt, clippy, full tests, frontend build/model-state smoke, public evidence-claim check, public scrub guard, harness self-test, JS syntax check, privacy audit, and diff whitespace check.

Blocker/Risk:

- `v0.1.0-rc1` was not created. The remaining risk is release-captain acceptance of documented comparator deferrals plus final pushed-branch verification.
- llama.cpp coverage is one CPU-only exact-row run, not a full table and not Metal evidence.
- Ollama is deferred because the only installed row observed here was `llama3.1:8b`, not an approved exact release comparator row.
- MLX is deferred because `mlx_lm` is not installed in the default Python environment.
- The source SHA in the benchmark bundle predates later docs/evidence-publication commits; no runtime code changes were made after that clean-head run.

Next:

- Commit and push the release branch.
- Observe remote/CI state after push.
- Create `v0.1.0-rc1` only after the documented deferrals and final branch state are accepted.

Need Tim:

- Decide whether `v0.1.0-rc1` may proceed with one llama.cpp CPU exact-row baseline plus explicit Ollama/MLX/Metal deferrals. Final `v0.1.0` remains approval-gated.

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
- [ ] Release Captain signs off.

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
