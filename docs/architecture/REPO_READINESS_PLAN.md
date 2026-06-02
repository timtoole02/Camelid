# Camelid Repo Readiness Plan

Last updated: 2026-05-05
Status: draft for maintainer review

## Why this exists

Camelid already has strong evidence discipline, but the contributor path is still harder than it should be. A new contributor currently has to piece together setup, validation expectations, model-path conventions, and frontend/backend coordination from several documents plus repo-local knowledge.

This draft focuses on repo-readiness work that is safe to promise now, without widening the product support contract.

## Current gaps observed in the repo

1. **Setup is not yet plug-and-play.**
   - Backend, frontend, model-loading, and parity flows are each documented, but they are spread across `README.md`, `CONTRIBUTING.md`, `frontend/README.md`, and QA notes.
   - Several commands assume local prerequisites (`llama-server`, model files, Rust 1.87+, Node/npm) without one short contributor-oriented checklist.

2. **Onboarding/docs need a cleaner entry path.**
   - The main `README.md` is thorough, but it is optimized for support/evidence precision more than first-run onboarding.
   - `CONTRIBUTING.md` gives principles and validation expectations, but not a concise “start here if you want to make a safe change” path.

3. **Configuration relies on ad-hoc environment and host knowledge.**
   - Some flows depend on local paths, `PATH` setup, rustup-vs-system cargo differences, or validation-lane/SSH context that experienced maintainers understand but new contributors do not.
   - The repo does not yet have one explicit configuration guide describing required local variables, optional overrides, and which assumptions are intentionally manual today.

4. **Testing and divergence validation need a clearer contributor lane.**
   - The repo has many useful scripts and strong QA artifacts, but it is not obvious which checks a contributor should run for docs-only, frontend-only, backend-only, or parity-affecting changes.
   - There is no single contributor-facing map from change type → expected validation command set → artifact expectations.

## Recommended improvement sequence

### 1) Add a short contributor quickstart

Goal: make a safe first local run possible without reading most of the repo.

Suggested output:
- a `docs/CONTRIBUTOR_QUICKSTART.md` or expanded `CONTRIBUTING.md` section covering:
  - required tool versions
  - backend build/start
  - frontend install/build/start
  - how to do a docs-only validation pass
  - how to do a basic code-change validation pass
  - what is intentionally *not* turnkey yet (model downloads, parity host setup, remote validation lane)

Recommendation: keep this focused on local development only, and link outward to the deeper evidence docs.

### 2) Split “product contract” from “first-run onboarding” in the README

Goal: preserve the current evidence-precise README while making the top of the repo easier to enter.

Suggested output:
- keep the current support language intact
- tighten the top-level “Start here” flow into two obvious tracks:
  - **Understand the support contract**
  - **Run Camelid locally / contribute safely**
- link the contributor quickstart near the top instead of expecting contributors to discover it later

Recommendation: do this as a small docs refactor, not a rewrite. The support boundary language is already valuable.

### 3) Add a dedicated configuration guide

Goal: reduce implicit maintainer knowledge around paths, env vars, and validation-host assumptions.

Suggested output:
- a `docs/CONFIGURATION.md` (or similar) covering:
  - Rust toolchain expectations and rustup fallback
  - Node/npm expectations for the frontend
  - API base overrides for the frontend
  - model path conventions used in examples
  - optional env vars and when they matter
  - explicit note that remote validation lane / SSH-based flows are maintainer workflows, not plug-and-play contributor requirements

Recommendation: document current reality honestly instead of trying to automate everything at once.

### 4) Publish a contributor validation matrix

Goal: make it obvious which validation is expected for each change class.

Suggested output:
- a `docs/VALIDATION_MATRIX.md` or a compact section in `CONTRIBUTING.md`
- suggested structure:
  - docs-only
  - frontend-only
  - backend-only non-inference changes
  - inference/tokenizer/runtime changes
  - support-contract or compatibility-row changes
  - QA/evidence-publication changes
- each row should state:
  - minimum local checks
  - when frontend smoke is expected
  - when parity artifacts are expected
  - when maintainer-only validation lane reruns are expected

Recommendation: this is likely the highest-leverage contributor improvement after quickstart.

### 5) Standardize command examples where easy

Goal: lower avoidable friction without changing behavior.

Suggested cleanup targets:
- prefer `npm ci` over `npm install` in reproducibility-sensitive docs unless a genuinely first-time workflow needs `install`
- keep backend build commands consistent (`cargo build --release --bin camelid` vs `cargo build`)
- mark commands that require exact supported local GGUF paths versus synthetic fixtures
- make docs-only validation commands consistent across `README.md` and `CONTRIBUTING.md`

## Proposed near-term deliverables

A practical first pass could be limited to:

1. contributor quickstart doc
2. configuration guide
3. validation matrix
4. small README / CONTRIBUTING cross-link cleanup

That would materially improve repo readiness without touching code, automation, or support claims.

## What should stay explicit/manual for now

To avoid overclaiming, the docs should continue to say that these areas are not fully turnkey yet:

- acquiring large real model files
- setting up `llama-server` for parity comparisons
- reproducing all remote/Ubuntu validation-lane runs
- broad support validation beyond the exact documented rows
- any workflow that depends on private/local infrastructure or SSH-only maintainer access

## Decisions that need maintainer approval

Before turning this draft into broader docs changes, Tim should decide:

1. whether new onboarding docs should live in a `docs/` directory or remain top-level
2. whether maintainer-only validation-lane/SSH guidance belongs in public docs, contributor docs, or not at all
3. whether the preferred first-run path should emphasize synthetic fixtures before real GGUFs
4. whether to standardize on `npm ci` everywhere reproducibility matters

## Suggested next action

Approve a small docs pass that adds the three contributor-facing guides above and then trims `README.md` / `CONTRIBUTING.md` to point at them cleanly.