# Frontend 3B Live Runtime Chat Gate

This bundle records a retained frontend slice for the Llama 3.2 3B Instruct Q8_0 WebUI support gate.

## Scope

- Keep WebUI chat readiness tied to live `/v1/health` state: `loaded_now=true`, `generation_ready=true`, and matching `active_model_id`.
- Keep the support contract tied to the exact `/api/capabilities` row, exact GGUF artifact identity, and Q8_0 evidence.
- Prevent stale browser-side model records from blocking an otherwise live exact-row runtime gate.
- Do not widen 3B support beyond the existing `supported_exact_row_smoke` contract.

## Validation

Commands passed on this retained tree:

- `git diff --check`
- `bash scripts/check-public-scrub.sh`
- `node scripts/check-public-evidence-claims.mjs`
- `node scripts/test-audit-evidence-bundle-privacy.mjs`
- `cd frontend && npm ci`
- `cd frontend && npm run build`
- `cd frontend && npm run smoke:3b-closure`
- `cd frontend && npm run smoke:model-state`
- `cd frontend && npm run smoke:streaming`
- `cd frontend && npm run smoke:ui`
- `cd frontend && npm run smoke:integration`
- `cd frontend && npm run smoke:tiny -- --api http://127.0.0.1:8181 --frontend http://127.0.0.1:4176`

The self-contained tiny smoke used local loopback services only. Remote Linux x86_64 validation was not attempted in this run; no fresh same-host 3B backend timing/parity claim is made.
