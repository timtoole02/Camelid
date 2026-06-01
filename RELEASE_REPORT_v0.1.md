# Camelid v0.1 Release Candidate Report

Current QA repair SHA: `3001fa48e3d5fa41dbeca417dd511164a3bacc80`

Validated pre-doc-refresh branch head SHA: `5017ff28568dc1898fa490b4492848a1b3c022f0`

Current status-only branch head SHA before the latest report refresh: `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd`

Branch: `release/v0.1-evidence`

Tag: `v0.1.0-rc1` at `d9fb294f47e3ae80291f969499e2240c6cd640c3`

Release status: the annotated `v0.1.0-rc1` tag is pushed, but it predates a post-rc1 frontend/README QA repair now present on the release branch at `3001fa48e3d5fa41dbeca417dd511164a3bacc80`. The validated pre-doc-refresh branch head is `5017ff28568dc1898fa490b4492848a1b3c022f0`; later release-status commits through `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd` are documentation-only, and local branch-head gates passed again at `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd` on 2026-06-01 10:43-10:54 UTC. The rc1 tag has not been moved. The release branch has v0.1 docs, a benchmark harness, a clean-head same-host llama.cpp CPU evidence bundle for one exact row, and release-captain deferrals for llama.cpp Metal, Ollama, and MLX. No final `v0.1.0` tag may be created without Tim approval.

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

Post-rc1 QA repair:

- Restored fresh-chat exact-row readiness pills so runtime-ready-but-support-gated models expose the runtime/support/capability blocker text.
- Restored the approved README chat screenshot with a boundary-safe local-first readiness caption.
- Updated the frontend integration smoke for the current empty-state hero and cleaned competitor-branded source copy covered by the UI smoke.
- Post-fix branch-head QA passed locally; the existing rc1 tag still points to the pre-fix commit.

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

Tests run: see `RELEASE_GATE_v0.1.md`. Branch-head gates passed locally again at `5017ff28568dc1898fa490b4492848a1b3c022f0` on 2026-06-01 01:17-01:22 UTC, at `a94eabb241f662ac328c02f6f5bb47dc33a9a14e` on 2026-06-01 04:33-04:39 UTC, and at `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd` on 2026-06-01 10:43-10:54 UTC, including `cargo fmt --all -- --check`, clippy, `cargo check`, full Rust tests, release build, cargo doc, frontend `npm ci`, frontend build, frontend model-state/3B-closure/integration/streaming/UI smokes, root validation self-tests, harness self-test, README screenshot guard, public evidence-claim check, public scrub guard, evidence checksum check, and v0.1 privacy audit. Later commits through `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd` are release documentation/status updates only.

Remote release state: at the 2026-06-01 10:54 UTC observation, `origin/release/v0.1-evidence` was at `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd`, and the dereferenced remote tag `v0.1.0-rc1^{}` pointed to `d9fb294f47e3ae80291f969499e2240c6cd640c3`. `git describe --tags --exact-match HEAD` failed, confirming the current status-only branch head is not tagged. GitHub reported no workflow runs and no commit statuses for either the branch head or the rc1 tag target. The workflow file runs on `push` to `main`, `pull_request`, and `workflow_dispatch`, so release-branch pushes do not automatically start hosted CI. Later release-status commits are documentation-only and do not move the rc1 tag.

Remaining blockers and risks:

- llama.cpp Metal, Ollama, and MLX fresh baselines are deferred; public docs must keep those non-claims explicit.
- The existing `v0.1.0-rc1` tag predates the branch-head frontend/README QA repair; Tim needs to decide whether the repaired release branch, including validated pre-doc-refresh head `5017ff28568dc1898fa490b4492848a1b3c022f0` and later documentation-only status commits through `d6e99b74b8f40c4420fb20b6f96257a1ef01d5fd`, should become a new candidate, a retagged rc1, or remain only on the release branch.
- The clean-head evidence bundle source SHA predates later documentation/evidence-publication commits; no runtime code changes were made after the run.
- Remote CI/status observation returned no runs or statuses for the rc1 commit; this is an observability gap, not a reported CI failure.
- Hosted CI is not automatically triggered by release-branch pushes; the local dispatch attempt was blocked because `gh` is not installed, so use `workflow_dispatch` or PR-based CI from an environment with the required tooling/credentials if remote hosted evidence is required for final approval.

Recommendation: push and soak the repaired release branch head, then have Tim choose the candidate strategy before final approval. Run or request a manual hosted CI dispatch if remote CI evidence is required. Do not create final `v0.1.0` without Tim approval.

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
