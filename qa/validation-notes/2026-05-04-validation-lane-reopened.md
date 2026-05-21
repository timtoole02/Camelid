# Validation note — historical Ubuntu validation lane operator context

Date: 2026-05-04

Superseded current-operator status: Tim paused the Ubuntu validation lane again on 2026-05-12. Use `qa/validation-notes/2026-05-12-local-only-validation-lane-paused.md` for current execution posture; this file remains historical evidence-lane context only.

Historical operator note: the Ubuntu validation lane was temporarily made available for Camelid promotion-grade exact-row runtime evidence on 2026-05-04. Do not publish private host addresses, SSH commands, key paths, or local operator-only details in the public repo.

Execution guardrails:

- Use clean public `main` checkouts for new validation runs.
- Preserve existing dirty remote worktrees; do not reset or overwrite them just to run current-head validation.
- Use `scripts/with-rustup-cargo.sh` or an equivalent rustup-managed toolchain on Ubuntu hosts whose distro `/usr/bin/cargo` is too old for the checked-in Rust floor.
- Generate full-support scaffolds with `node scripts/prepare-full-support-bundle.mjs ...`; when runtime validation has not been attempted for the current run, keep the default `evidence_needed` status so generated runtime commands preserve the missing-evidence boundary without implying host failure.
- Keep claims exact-row only. Host availability is not evidence; only passing artifacts can move docs, API, or frontend language.

Current promotion posture:

- TinyLlama remains the supported current gate.
- Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 remain supported exact-row smoke lanes only.
- Broader/full support still needs normalized current-head parity, API/WebUI, memory/perf, context, and durable-bundle evidence per exact row.
- The 8B broader three-prompt 50-token pack has a passing rerun at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`; the first 8B 512-context timeout has a passing rerun at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`; the bounded 8B compact chat-template-shapes pack has a passing rerun at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`; broader context/template coverage, performance/portability, and full-support normalization remain blockers.
