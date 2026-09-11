# Contributor Quickstart

Last updated: 2026-05-24

This guide is the shortest safe path to getting productive in Camelid locally.

## Start here

1. Read [`README.md`](../README.md) for the current public support contract.
2. Read [`CONTRIBUTING.md`](../CONTRIBUTING.md) for contribution expectations.
3. Use this guide to get a local backend/frontend loop running.
4. Use [`docs/VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) to choose the smallest meaningful validation lane for your change.

Current public support is exact-row: TinyLlama Q8_0 is the supported gate; Llama 3.2 1B Instruct Q8_0 has verified bounded support through checked 512/1024/2048/4096/8192 context packs; Llama 3.2 3B Instruct Q8_0 remains supported exact-row smoke through the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder on its current canonical GGUF (the May 3B evidence is historical for the prior upload); and Llama 3 8B Instruct Q8_0 has verified bounded support through checked 512/1024/2048 context packs where row-specific PASS artifacts exist. The latest 1B 8192 bundle is `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`; the current-head 8B 1024/2048 bundle is `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`. Do not broaden any claim to model-native/larger context beyond checked packs, production throughput, portability, local experiments, arbitrary templates, or adjacent GGUFs.

## Prerequisites

Required local tools:

- Rust/Cargo via `rustup` (the exact version is pinned in
  [`rust-toolchain.toml`](../rust-toolchain.toml) and installed automatically on
  first `cargo` invocation in this repo)
- Node.js + npm for `frontend/`
- Git

Helpful but not required for every contribution:

- `llama-server` in `PATH` for parity comparisons
- local GGUF model files for supported-row smoke or parity work

If your host exposes an older distro `cargo`, source `$HOME/.cargo/env` first or use `scripts/with-rustup-cargo.sh ...` so the rustup-managed toolchain is used.

This matters more than it looks. `rust-toolchain.toml` is a *rustup* mechanism:
on a host with rustup it outranks `rustup default` and pins you to the same
toolchain CI uses, but on a host whose `cargo` comes from a distro package or
Homebrew with **no rustup installed**, the file is inert and you silently build
and lint with whatever version that package ships. That is a real way to get
`clippy` findings CI does not have — or to miss ones it does — so check
`rustc -vV` against the pinned channel before concluding that CI and your
machine disagree about the code. Every CI leg prints its own `rustc -vV` in the
"Show effective toolchain" step for exactly this comparison.

## Backend quickstart

This gets the backend running locally. By itself, it does **not** guarantee a working chat demo, because the repository does not bundle supported GGUF model files.

From the repo root:

```bash
git checkout main
git pull --ff-only
cargo build --release --bin camelid
target/release/camelid serve --addr 127.0.0.1:8181
```

If you prefer the rustup wrapper:

```bash
scripts/with-rustup-cargo.sh build --release --bin camelid
target/release/camelid serve --addr 127.0.0.1:8181
```

## Frontend quickstart

The frontend can run against your local backend, but chat readiness still depends on loading a supported local model file and meeting the current support contract.

In another shell:

```bash
cd frontend
npm ci
npm run dev
```

Default frontend URL:

```text
http://127.0.0.1:4175
```

Default backend API base:

```text
http://127.0.0.1:8181
```

## Basic local checks by change type

### Docs-only change

```bash
git diff --check
bash scripts/check-public-scrub.sh
```

### Backend or shared code change

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
bash scripts/check-public-scrub.sh
```

Windows note: cap test-build parallelism with `cargo test -j 4 ...`. The
default 16-way build spawns enough concurrent `link.exe` processes to fail
intermittently (`could not exec the linker`, or `LNK1104: cannot open file
'msvcrt.lib'`), especially under memory pressure. Also never run two cargo
commands concurrently in one checkout — the build lock deadlocks.

### Frontend change

```bash
cd frontend
npm ci
npm run build
```

If the frontend change affects live chat, model loading, or readiness gating, also run the frontend smoke flow described in [`frontend/README.md`](../frontend/README.md).

## What is intentionally not turnkey yet

These areas still require real manual setup and should not be described as one-command contributor onboarding:

- downloading or hosting large real model files
- setting up `llama-server` and reference-model parity hosts
- reproducing every remote Ubuntu validation-lane rerun locally
- maintainer-only SSH or remote validation workflows

## Public vs maintainer-only workflows

Public contributor docs should cover local development, public evidence, and exact support boundaries.

Maintainer-only workflows should stay out of public onboarding docs when they depend on private infrastructure, SSH access, local machine paths, or unpublished operational habits.

## Next docs to use

- [`docs/CONFIGURATION.md`](CONFIGURATION.md) — local config and env-var guidance
- [`docs/VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) — expected checks by change class
- [`frontend/README.md`](../frontend/README.md) — frontend-specific smoke and contract notes
