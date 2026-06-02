# Camelid Documentation Index

Last updated: 2026-05-31

This index helps readers navigate the public Markdown set.

## Fast reader paths

- **Product/reviewer path:** start with `README.md`, then `COMPATIBILITY.md`, then the milestone snapshot in `STATUS.md`, then `BENCHMARKS.md`.
- **Evidence auditor path:** start with `PARITY.md`, then `qa/evidence-bundles/README.md`, then follow the row-specific manifests linked from `STATUS.md`.
- **Contributor path:** start with `docs/CONTRIBUTOR_QUICKSTART.md`, then use `docs/VALIDATION_MATRIX.md` to choose the smallest safe check lane.

## Public sources of truth

Read these first:

- [`README.md`](README.md) — product overview, milestone story, and current exact-row support table
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — authoritative support ledger and at-a-glance release contract
- [`STATUS.md`](STATUS.md) — current milestone/evidence snapshot and exact blockers
- [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) — public performance snapshot and benchmark-claim rules
- [`docs/WAR_ROOM_EVIDENCE_INDEX.md`](docs/WAR_ROOM_EVIDENCE_INDEX.md) — war-room claim-source order, evidence index, and public wording policy
- [`PARITY.md`](docs/benchmarks/PARITY.md) — exact-row parity proof map and audit trail
- [`ROADMAP.md`](ROADMAP.md) — phase-level plan of record

## Contributor and project policy

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and validation guidance
- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) — shortest safe local contributor path
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — current toolchain, env-var, and path guidance
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — expected checks by change class
- [`SECURITY.md`](SECURITY.md) — security reporting guidance
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — acknowledgements and license notices
- [`DECISIONS.md`](docs/architecture/DECISIONS.md) — design decision log

## QA and acceptance docs

- [`FULL_SUPPORT_BLOCKER_MATRIX.md`](FULL_SUPPORT_BLOCKER_MATRIX.md) — four-row full-support owner matrix with exact missing evidence by row
- [`QA_SMALL_MODEL_PARITY_MATRIX.md`](docs/release/QA_SMALL_MODEL_PARITY_MATRIX.md) — current small-model QA matrix
- [`QA_LLAMA32_3B_Q8_ACCEPTANCE.md`](docs/release/QA_LLAMA32_3B_Q8_ACCEPTANCE.md) — exact 3B acceptance checklist
- [`qa/evidence-bundles/README.md`](qa/evidence-bundles/README.md) — sanitized public evidence-bundle map, including the reopened-lane API/WebUI and bounded 8B broader/template/context summaries

## Architecture, recon, and planning notes

These documents are working notes, not support ledgers. When a note and a public source differ,
`COMPATIBILITY.md` and `STATUS.md` win.

- [`ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)
- [`FORGELOCAL_INTEGRATION.md`](docs/architecture/FORGELOCAL_INTEGRATION.md)
- [`INFERENCE_RECON.md`](docs/recon/INFERENCE_RECON.md)
- [`TENSOR_RECON.md`](docs/recon/TENSOR_RECON.md)
- [`TOKENIZER_RECON.md`](docs/recon/TOKENIZER_RECON.md)
- [`SAMPLING_API_RECON.md`](docs/recon/SAMPLING_API_RECON.md)
- [`SAFETENSORS_PLAN.md`](docs/architecture/SAFETENSORS_PLAN.md)
- [`ATTENTION_CHECKPOINTS.md`](docs/recon/ATTENTION_CHECKPOINTS.md)
- [`REPO_READINESS_PLAN.md`](docs/architecture/REPO_READINESS_PLAN.md) — draft repo-readiness improvement plan for contributor setup, configuration, and validation ergonomics

## Historical archives

- [`ROADMAP_ARCHIVE.md`](docs/archive/ROADMAP_ARCHIVE.md) — completed-phase history
- [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md) — detailed historical status log
