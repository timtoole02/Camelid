# Camelid v0.1 Release Status

Last updated: 2026-05-31

Release checkout: branch-agnostic status ledger; record the exact branch and SHA when refreshing rc1 readiness

Release target: `v0.1.0-rc1`

Release posture: evidence release candidate in progress. Lightweight gates now pass locally, but no tag is allowed until real comparator evidence exists and the release captain signs off.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Tightened the runtime/API/frontend support contract so Mistral is evidence-only and fail-closed for v0.1.
- Removed Mistral from frontend tracked full-support rows and moved the API Mistral family posture out of `supported_model_families`.
- Cleared Rust format, clippy, full test suite, Metal tests, release build, frontend build/smoke, harness self-test, public evidence-claim check, and public scrub guard locally.
- Preserved the dirty primary checkout; all edits landed only in the release worktree.

Evidence:

- Full local gate evidence is recorded in `RELEASE_GATE_v0.1.md`.
- `cargo test --all-targets --all-features --no-fail-fast` passed, including Metal unit tests.
- `cd frontend && npm run build && npm run smoke:model-state` passed.
- `qa/evidence-bundles/v0.1/dryrun-release-captain/` still exists as skipped/dry-run harness evidence only.

Blocker/Risk:

- Real comparator evidence is still missing. The release branch has not yet produced actual llama.cpp, Ollama, or MLX v0.1 benchmark bundles.
- Existing historical bundles remain context only unless recaptured at the release branch SHA.
- No `v0.1.0-rc1` tag is allowed yet.

Next:

- Run the real comparator matrix against llama.cpp, Ollama, and MLX where available.
- Save real benchmark output under `qa/evidence-bundles/v0.1/<timestamp>/`.
- Update comparator docs and release report from actual results, including losses.

Need Tim:

- No decision needed yet. Final `v0.1.0` tag remains approval-gated; `v0.1.0-rc1` is allowed only if release gates pass.

## Current Checkout

- Record the exact checkout path, branch, SHA, and cleanliness for the release-candidate run that refreshes this file.
- Do not hard-code historical worktree names or stale branch labels into the public release ledger.
- Preservation rule: if a release lane uses a separate worktree, leave any dirty primary checkout untouched and state that fact explicitly for the run that produced the evidence.

## Release Captain Update Format

Camelid v0.1 update:

Shipped:

Evidence:

Blocker/Risk:

Next:

Need Tim:

## v0.1 Blockers

- Benchmark harness must generate a complete evidence bundle under `qa/evidence-bundles/v0.1/<timestamp>/`.
- llama.cpp baseline must be pinned, reproducible, and separated by backend mode.
- Ollama baseline must exist or be explicitly deferred with release-captain rationale.
- MLX baseline must exist or be explicitly deferred with release-captain rationale.
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

- [ ] Repo builds cleanly.
- [ ] Tests pass or failures are documented as non-release blockers.
- [ ] Benchmark harness runs from a clean checkout.
- [ ] llama.cpp baseline exists.
- [ ] Ollama baseline exists or is explicitly deferred with reason.
- [ ] MLX baseline exists or is explicitly deferred with reason.
- [ ] Correctness matrix exists.
- [ ] Support matrix exists.
- [ ] README is updated and does not overclaim.
- [ ] Release notes exist.
- [ ] Evidence bundle exists.
- [ ] Public docs contain no unsupported performance claims.
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
