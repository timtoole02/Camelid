# Validation note — local checkmark lane without remote Ubuntu validation

Superseded operational note: the approved Ubuntu validation lane reopened later on 2026-05-04, then Tim paused it again on 2026-05-12. Keep this file as historical context for the earlier local-only pause, but use `2026-05-12-local-only-validation-lane-paused.md` for current execution guidance.

Date: 2026-05-04
Repo lane: local/repo-safe only

## Operator constraint

This lane records local-only checks without fresh remote Ubuntu validation. Until Tim explicitly authorizes a remote validation pass:

- do not SSH into the validation host or any substitute remote validation host
- do not run local Mac llama-server/reference-runtime workloads as a substitute
- treat promotion-grade 1B/3B/8B runtime reruns as blocked by host shutdown
- keep 8B longer-context/performance validation blocked, not promoted or inferred

## Local work preserved on this lane

This pass only normalized repo-facing contract language and harness scaffolding:

- TinyLlama remains the supported current gate.
- Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B remain exact-row smoke-supported only.
- Historical at the time of this local-only pause: the 8B row kept its short-smoke/parity evidence, and the first 512-context current-head pack remained a documented blocker. This was later superseded by the passing rerun in `2026-05-04-8b-context-512-rerun.md`.
- `/api/capabilities`, frontend readiness copy, and docs should expose blocked template/context/perf tracks rather than implying broad/full Llama support.
- `scripts/prepare-full-support-bundle.mjs` is the safe scaffold generator for the next validation window; with the default `blocked_by_operator_shutdown` status, runtime command files preserve the original commands for review but exit blocked instead of running workloads.

## Resume rule

When Tim explicitly reopens the Ubuntu validation lane, regenerate the full-support scaffold with `--validation-host-status available`, run only on the approved validation host/runtime lane, and publish only scrubbed artifacts/manifests whose rows passed their exact tracks.
