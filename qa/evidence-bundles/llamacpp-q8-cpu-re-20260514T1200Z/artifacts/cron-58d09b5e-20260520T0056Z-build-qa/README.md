# cron-58d09b5e-20260520T0056Z-build-qa

CI/QA guard evidence for branch `buildqa/clippy-needless-range-loop-58d09b5e-20260518T2100Z` at `e59a1ea403b7ad07c7dd12bf42947ef640475706`.

## Gates

Local macOS arm64 worktree:

- `cargo fmt --all -- --check` — passed; empty log.
- `cargo clippy --all-targets --all-features -- -D warnings` — passed (`Finished dev profile`).
- `cargo test --all-targets --all-features` — passed; tail includes integration tokenizer `24 passed` and example unit `2 passed`.

Ubuntu x86_64 canonical host:

- Fresh host check used the operator-provided canonical Ubuntu SSH command and reached Ubuntu 24.04.4 LTS.
- First remote CI attempt failed before code checks because non-interactive SSH resolved `/usr/bin/cargo` 1.75.0, which cannot parse Cargo.lock v4. Exact clippy log: `lock file version 4 requires \`-Znext-lockfile-bump\``.
- Rerun exported `PATH="$HOME/.cargo/bin:$PATH"`, selecting rust/cargo 1.95.0 from `rust-toolchain.toml`.
- Ubuntu rerun `cargo fmt --all -- --check` — passed; empty log.
- Ubuntu rerun `cargo clippy --all-targets --all-features -- -D warnings` — passed (`Finished dev profile`).
- Ubuntu rerun `cargo test --all-targets --all-features` — passed; tail includes tensor tests `29 passed`, tokenizer `24 passed`, example unit `2 passed`.

## Files

- `commands.txt` — command ledger.
- `logs/` — local logs plus SSH stdout for Ubuntu attempts.
- `remote/` — imported redacted remote Ubuntu evidence bundle from `<remote-worktree>`.
- `SHA256SUMS` — checksums for bundle files.

## Retain/reject

Retain this CI/QA guard evidence. No code changes were required in this run; the only remote issue was PATH/toolchain selection, resolved in the rerun.
