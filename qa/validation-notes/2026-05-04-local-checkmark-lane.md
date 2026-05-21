# Validation note — local checkmark lane during operator-requested validation pause

Superseded operational note: the Ubuntu validation lane had a temporary operator-status change later on 2026-05-04, then Tim paused it again on 2026-05-12. Keep this file as historical context for the earlier local-only pause, but use `2026-05-12-local-only-validation-lane-paused.md` for current execution guidance.

Date: 2026-05-04
Repo lane: local/repo-safe only

## Operator constraint

Tim requested a local-only validation pause for the Ubuntu validation lane. Until Tim explicitly says the validation host/runtime lane is back:

- do not substitute local-only or substitute-remote runs for promotion-grade Ubuntu validation evidence
- do not run local Mac llama-server/reference-runtime workloads as a substitute
- treat promotion-grade 1B/3B/8B runtime reruns as not attempted during the operator pause
- keep 8B longer-context/performance validation unpromoted and uninferred

## Local work preserved on this lane

This pass only normalized repo-facing contract language and harness scaffolding:

- TinyLlama remains the supported current gate.
- Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B remain exact-row smoke-supported only.
- Historical at the time of this local-only pause: the 8B row kept its short-smoke/parity evidence, and the first 512-context current-head pack remained a documented blocker. This was later superseded by the passing rerun in `2026-05-04-8b-context-512-rerun.md`.
- `/api/capabilities`, frontend readiness copy, and docs should expose blocked template/context/perf tracks rather than implying broad/full Llama support.
- `scripts/prepare-full-support-bundle.mjs` is the safe scaffold generator for the next validation window; with the default operator-pause status, runtime command files preserve the original commands for review but do not run workloads.

## Resume rule

When Tim explicitly authorizes Ubuntu validation work, regenerate the full-support scaffold with an operator-approved validation-host status, run only on the approved validation host/runtime lane, and publish only scrubbed artifacts/manifests whose rows passed their exact tracks.
