# Camelid v0.1 Release Candidate Report

Current SHA: release branch HEAD after this gate-refresh commit

Branch: `release/v0.1-evidence`

Tag candidate: `v0.1.0-rc1`

Release status: not ready to tag. The release branch now has v0.1 docs, a benchmark harness, comparator baseline plans, a dry-run evidence bundle, and passing lightweight gates. Real comparator evidence is still missing.

Supported model rows:

- `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
- `Llama-3.2-1B-Instruct-Q8_0.gguf`
- `Llama-3.2-3B-Instruct-Q8_0.gguf`
- `Meta-Llama-3-8B-Instruct-Q8_0.gguf`

Correctness summary: `SUPPORT_MATRIX_v0.1.md` and `CORRECTNESS_v0.1.md` define the v0.1 boundary. Mistral is downgraded to evidence-only bring-up because the current API/WebUI support-surface evidence is fail-closed. Mixtral remains unsupported beyond bounded one-token backend MoE runtime evidence.

Benchmark summary: `tools/bench/v0.1-benchmark-harness.mjs` can emit the required bundle layout and passed its synthetic self-test. `qa/evidence-bundles/v0.1/dryrun-release-captain/` proves output shape only; it is not runtime benchmark evidence. No real v0.1 comparator bundle has been created yet.

Where Camelid wins: not claimed for v0.1 yet. Real comparator benchmark evidence is still missing.

Where Camelid loses: historical docs already record known losses against llama.cpp/MLX in scoped settings, but this release branch has not generated fresh v0.1 comparator results.

Known limitations:

- Broad model-family support is not claimed.
- Mistral support is not promoted in v0.1.
- Mixtral later-generation parity and continuation remain blocked.
- Production throughput, portability, arbitrary templates, and distributed inference are not v0.1 support claims.
- Real llama.cpp/Ollama/MLX comparator baselines are not complete.

Evidence bundle path: `qa/evidence-bundles/v0.1/dryrun-release-captain/` exists as dry-run harness evidence only.

Docs changed:

- `README.md`
- `RELEASE_STATUS.md`
- `RELEASE_REPORT_v0.1.md`
- `RELEASE_NOTES_v0.1.md`
- `BENCHMARKS_v0.1.md`
- `SUPPORT_MATRIX_v0.1.md`
- `CORRECTNESS_v0.1.md`
- `MARKET_POSITIONING_v0.1.md`
- `LLAMA_CPP_BASELINE_v0.1.md`
- `OLLAMA_BASELINE_v0.1.md`
- `MLX_BASELINE_v0.1.md`
- `DISTRIBUTED_MAC_v0.1.md`
- `RELEASE_GATE_v0.1.md`

Tests run: see `RELEASE_GATE_v0.1.md`. Local lightweight gates pass, including `cargo fmt --all -- --check`, clippy, cargo check, full Rust tests, release build, frontend build/model-state smoke, harness self-test, public evidence-claim check, and public scrub guard.

Remaining blockers:

- Real v0.1 comparator benchmark evidence has not been generated.
- Comparator baselines have not been finalized or explicitly release-captain-deferred.

Recommendation: do not tag `v0.1.0-rc1` until the release gate checklist in `RELEASE_STATUS.md` is complete.

## Release Captain Signoff

- [ ] Evidence bundle exists and is tied to the release branch SHA.
- [ ] Support matrix is exact-row only.
- [ ] Correctness claims cite evidence paths.
- [ ] Benchmark methodology is reproducible from a clean checkout.
- [ ] llama.cpp, Ollama, and MLX are each either benchmarked or explicitly deferred with reasons.
- [ ] CPU-only, Metal, MLX, and distributed evidence are separated and labeled.
- [ ] README contains no unsupported performance or model-family claims.
- [ ] Release notes explain wins, losses, and unsupported areas.
- [ ] QA gate records pass/fail for all required commands.
- [ ] Primary dirty checkout remains preserved or is explicitly reconciled later.
