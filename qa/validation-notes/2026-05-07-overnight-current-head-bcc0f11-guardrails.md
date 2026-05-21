# Overnight current-head guardrails (bcc0f11)

Date: 2026-05-07 UTC

Scope: cron `Camelid overnight all-models fixer` safe guardrail slice on current `main`. This does not widen support, API capabilities, frontend readiness, context support, model-family claims, or 8B broader/full-support language.

Starting state:

- Local `main` was clean at `bcc0f11fc8533f286ab4e1fa207be02bf8789147` (`Polish Gemini-style chat landing`) and ahead of `origin/main` by one UI-only commit.
- Canonical Ubuntu validation-host check used the private operator SSH lane and found the stale/self-matching `run-8b-1024-diag-4c30c53` watcher, but no active `chat-parity-llama3`, `context-1024`, `context-2048`, or long 8B parity run on the tracked validation port. No duplicate 8B long-context run was launched.
- No additional long 8B remote validation was attempted for this local exact-support guardrail slice.

Current-head gates:

- `./scripts/with-rustup-cargo.sh fmt --all -- --check`
- `./scripts/with-rustup-cargo.sh test capabilities_report -- --nocapture` — 4 unit tests plus `capabilities_report_support_contract_and_planned_lanes` passed.
- `./scripts/check-evidence-bundle-checksums.sh`
- `node scripts/audit-evidence-bundle-privacy.mjs` — `finding_count: 0`.
- `node scripts/check-public-evidence-claims.mjs` — `51 manifest(s), 17 summary file(s)` passed.
- `node scripts/test-check-public-evidence-claims.mjs`
- `(cd frontend && npm run smoke:model-state)` — passed.

Local transcript artifact: `target/overnight-all-models-current-head-guardrails-20260507T100416Z.log`.

Boundary:

- TinyLlama remains the verified current gate.
- Llama 3.2 1B/3B remain exact-row bounded support through checked 2048 packs.
- Llama 3 8B remains exact bounded-pack support only for checked 512/1024/2048 packs; no broader/full 8B or Llama-family support is claimed.
- Mistral/Mixtral/Qwen/Gemma remain planned/candidate rows until row-specific PASS artifacts and synchronized docs/API/frontend alignment exist.
