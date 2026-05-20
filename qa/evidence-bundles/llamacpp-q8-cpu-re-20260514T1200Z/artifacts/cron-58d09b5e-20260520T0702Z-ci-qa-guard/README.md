# Camelid CI/QA guard evidence — cron 58d09b5e

CAMELID SLICE:
- Target: CI/QA guard lane for the current dirty Q8 attention-output GEMM4 prefill slice plus inherited repo dirty state, with strict fresh Ubuntu host validation.
- Domain terms used/updated: evidence bundle, tracer bullet, backend-owned packed runtime storage, same-host guard (not claimed; this is compile/unit QA only).
- Feedback loop: local macOS cargo fmt/test/clippy/full-test gates, then Ubuntu x86_64 cargo fmt/focused-test/clippy/full-test gates in a fresh `/tmp` rsync copy of the working tree.
- Files changed: no source edits by this QA run; this evidence bundle only.
- Gate/env:
  - Local: macOS arm64, repo HEAD `2ae221c8ac7c8f52e9cd26002c1de9dfc02685d3`, dirty worktree captured in `logs/local-git-state.log`.
  - Ubuntu: `ssh -o IdentitiesOnly=yes -i <operator-key> ubuntu@<validation-host>` succeeded; host `<validation-hostname>`, Linux `6.17.0-1013-aws`, x86_64, Rust/Cargo `1.95.0`, remote copy `<ubuntu-workdir>`.
- Baseline: current branch `docs-camelid-ubuntu-host-honesty-20260519T0218Z` at HEAD `2ae221c8ac7c8f52e9cd26002c1de9dfc02685d3` with pre-existing dirty source/evidence edits.
- Results:
  - Local `cargo fmt -- --check`: PASS.
  - Local `cargo test q8_attention_output_gemm4 -- --nocapture`: PASS, 3 passed.
  - Local `cargo clippy --all-targets -- -D warnings`: PASS.
  - Local `cargo test`: PASS, 445 passed, 0 failed, 1 ignored across lib/bin/integration/doc tests.
  - Ubuntu exact SSH command: PASS; login banner captured in `logs/ubuntu-exact-ssh.log`.
  - Ubuntu `cargo fmt -- --check`: PASS.
  - Ubuntu `cargo test q8_attention_output_gemm4 -- --nocapture`: PASS, 3 passed.
  - Ubuntu `cargo clippy --all-targets -- -D warnings`: PASS.
  - Ubuntu `cargo test`: PASS, 453 passed, 0 failed, 1 ignored across lib/bin/integration/doc tests.
- Retain/reject: retained as QA evidence only; no support-contract or performance claim promoted.
- Next tracer bullet: resolve/stage the inherited dirty source/evidence edits into a reproducible commit or split them before pushing.

Logs are under `logs/`. `SHA256SUMS` covers every file in this evidence bundle.
