# Camelid v0.1 Release Candidate Report

Current SHA: `v0.1.0-rc1` tag target after the release-captain signoff refresh

Branch: `release/v0.1-evidence`

Tag candidate: `v0.1.0-rc1`

Release status: release-captain signed off for `v0.1.0-rc1`. The release branch has v0.1 docs, a benchmark harness, a clean-head same-host llama.cpp CPU evidence bundle for one exact row, and release-captain deferrals for llama.cpp Metal, Ollama, and MLX. No final `v0.1.0` tag may be created without Tim approval.

Supported model rows:

- `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
- `Llama-3.2-1B-Instruct-Q8_0.gguf`
- `Llama-3.2-3B-Instruct-Q8_0.gguf`
- `Meta-Llama-3-8B-Instruct-Q8_0.gguf`

Correctness summary: `SUPPORT_MATRIX_v0.1.md` and `CORRECTNESS_v0.1.md` define the v0.1 boundary. Mistral is downgraded to evidence-only bring-up because the current API/WebUI support-surface evidence is fail-closed. Mixtral remains unsupported beyond bounded one-token backend MoE runtime evidence.

Benchmark summary: `tools/bench/v0.1-benchmark-harness.mjs` emitted a real local v0.1 bundle at `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`. The bundle records clean source SHA `8026339531463ade269d7be7078da331ba3e4085`, model SHA `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`, llama.cpp commit `399739d5c5978351f39e3454bfbfbab4f369088f`, and passing `CMLD-BENCH` marker guardrails for Camelid and llama.cpp. `qa/evidence-bundles/v0.1/dryrun-release-captain/` remains dry-run shape evidence only.

Where Camelid wins: no speed win is claimed for v0.1.

Where Camelid loses: the v0.1 same-host llama.cpp CPU row shows Camelid slower on bounded TTFT and total elapsed for Llama 3.2 3B Instruct Q8_0. Historical docs also record scoped MLX speed losses; those remain prior context only.

Known limitations:

- Broad model-family support is not claimed.
- Mistral support is not promoted in v0.1.
- Mixtral later-generation parity and continuation remain blocked.
- Production throughput, portability, arbitrary templates, and distributed inference are not v0.1 support claims.
- Full comparator coverage is not complete: llama.cpp has one CPU-only exact-row baseline, Metal is deferred, Ollama is deferred because no approved exact row is installed, and MLX is deferred because `mlx_lm` is not installed.

Evidence bundle path: `qa/evidence-bundles/v0.1/20260531T184150Z-real-local/`.

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

Remaining blockers and risks:

- llama.cpp Metal, Ollama, and MLX fresh baselines are deferred; public docs must keep those non-claims explicit.
- The clean-head evidence bundle source SHA predates later documentation/evidence-publication commits; no runtime code changes were made after the run.
- Remote CI still needs normal observation after the pushed signoff commit/tag.

Recommendation: create and push `v0.1.0-rc1` from the signed-off release branch. Do not create final `v0.1.0` without Tim approval.

## Release Captain Signoff

- [x] Evidence bundle exists and records clean source SHA.
- [x] Support matrix is exact-row only.
- [x] Correctness claims cite evidence paths.
- [x] Benchmark methodology is reproducible from a clean checkout.
- [x] llama.cpp, Ollama, and MLX are each either benchmarked or explicitly deferred with reasons.
- [x] CPU-only, Metal, MLX, and distributed evidence are separated and labeled.
- [x] README contains no unsupported performance or model-family claims.
- [x] Release notes explain wins, losses, and unsupported areas.
- [x] QA gate records pass/fail for all required commands after this publication update.
- [x] Primary dirty checkout remains preserved; this release lane used only the clean release worktree.
