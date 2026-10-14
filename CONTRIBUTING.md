# Contributing to Camelid

Thanks for your interest in Camelid.

Camelid is intentionally evidence-gated. Optimize for correctness, explicit support boundaries, and reproducible validation over optimistic claims.

## First principles

- Keep the support contract honest.
- Do not broaden support claims without fresh evidence.
- TinyLlama 1.1B Chat Q8_0 remains the supported current gate.
- Llama 3.2 1B Instruct Q8_0 and Llama 3 8B Instruct Q8_0 have verified bounded support; Llama 3.2 3B Instruct Q8_0 remains supported as exact-row smoke.
- No neighboring Llama sizes, base variants, quantizations, longer contexts, or broad chat-template behavior inherit support.
- Full-support claims still require stronger longer-context, performance/portability, and broader behavior evidence on each exact row.
- Do not let UI copy, README language, API capability surfaces, or status docs drift out of sync.

## Before you start

1. Read `README.md` for the current public support contract.
2. Check `COMPATIBILITY.md` for the evidence-based support matrix.
3. Review `ROADMAP.md` and `docs/reference/STATUS.md` for current priorities and open work.
4. Read `docs/CONTRIBUTOR_QUICKSTART.md` for the shortest safe local setup path.
5. If your change affects claims, docs, or readiness wording, update every relevant source of truth in the same PR.

## Development setup

For the shortest contributor path, start with `docs/CONTRIBUTOR_QUICKSTART.md`.

For local toolchain, env-var, and path guidance, see `docs/CONFIGURATION.md`.

### Backend

```bash
git clone https://github.com/timtoole02/Camelid.git
cd Camelid
cargo build
cargo test --all-targets --all-features
```

Toolchain note: use the Rust/Cargo version pinned in `rust-toolchain.toml`; `Cargo.toml` declares Rust 1.89 as the minimum. On Ubuntu hosts that still expose an older distro `cargo` on `/usr/bin`, either source `$HOME/.cargo/env` first or run `scripts/with-rustup-cargo.sh build` / `scripts/with-rustup-cargo.sh test --all-targets --all-features` so the rustup-managed toolchain is used.

### Frontend

```bash
cd frontend
npm ci
npm run build
```

## Validation expectations

Run the smallest meaningful validation for your change. Use `docs/VALIDATION_MATRIX.md` to map change type to the expected lane.

For substantial backend changes, prefer the full standard gate:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
bash scripts/check-public-scrub.sh
```

For frontend changes:

```bash
cd frontend
npm ci
npm run build
```

If you change model support behavior, tokenizer behavior, generation semantics, or readiness
language, include concrete artifacts or references showing why the change is justified.

Keep public contributor docs focused on local/public workflows. Do not add private SSH details,
maintainer workstation paths, or unpublished operator procedures to public onboarding docs.

## Pull requests

Keep PRs focused and explain:

- what changed
- why it changed
- what evidence supports it
- what validation you ran
- whether any docs or compatibility rows changed

A strong PR description makes it obvious whether the change is:

- implementation groundwork only
- evidence-producing validation work
- a support-contract update justified by new evidence

## Commit style

Short, descriptive commits are preferred. A few examples:

- `docs: add CI badge and contributor guidance`
- `inference: guard tied output projection reuse`
- `qa: add llama3 tokenizer reference fixtures`

## Reporting bugs

When possible, include:

- OS / architecture
- exact command or API request
- model path and quantization
- expected behavior
- actual behavior
- logs, traces, or parity artifacts

## Code of conduct

Be respectful, clear, and evidence-driven. Strong technical disagreement is welcome; hand-wavy support claims are not.
