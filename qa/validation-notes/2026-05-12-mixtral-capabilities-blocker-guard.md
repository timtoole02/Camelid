# 2026-05-12 — Mixtral capabilities blocker guard

Scope: local-only backend/API support-contract guardrail. No SSH, no validation host, and no new runtime promotion evidence.

Starting state:
- Repo head before edits: `7743b61` (`Align status support boundary wording`).
- Working tree already contained two untracked Llama 3 8B evidence bundle directories; this slice did not modify them.
- Operator posture: no fresh remote Ubuntu validation was attempted in this note; promotion-grade reruns require an explicitly authorized lane.

Evidence reviewed:
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/README.md`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/summary.json`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/summary.json`

Change:
- `/api/capabilities` now reports the Mixtral exact row's latest checked bucket/result as the Gate 9A later-generation blocker instead of leaving the latest result anchored on the earlier one-token probe.
- The Mixtral compatibility row still preserves bounded one-token MoE runtime evidence, but its latest result is now `blocked_later_generation_divergence` and cites the 2026-05-12 blocker reconciliation README.
- Backend unit and API vertical-slice guardrails now assert the blocker bucket, blocker result, blocker citation, generated-token-index-9 divergence, backend HTTP hang, and no broad Mixtral support claim.

Validation:
- `cargo fmt --all -- --check`
- `cargo test capabilities --all-targets`
- `scripts/check-public-scrub.sh`
- `git diff --check`

Claim boundary:
- This is guardrail validation and support-surface honesty only.
- It does not promote Mixtral, Mistral, neighboring rows, longer context, API/WebUI/RSS readiness, frontend support, throughput, portability, or arbitrary-template behavior.
- Llama support remains exact-row/bounded-pack only where row-specific PASS artifacts exist.
