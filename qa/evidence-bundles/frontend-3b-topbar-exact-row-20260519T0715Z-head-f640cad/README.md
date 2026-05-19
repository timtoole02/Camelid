# Frontend 3B topbar exact-row support detail

## Target
Llama 3.2 3B Instruct Q8_0 end-to-end frontend/WebUI closure.

## Retained slice
The topbar support-contract detail now prefers the active exact supported row from the shared chat gate, then the selected exact supported row, before falling back to the first current-gate compatibility row. This keeps the live 3B surface from displaying a neighboring supported row when multiple supported rows are advertised.

## Gates
Run on this feature tree before commit:

- `npm ci`
- `npm run build`
- `npm run smoke:model-state`
- `npm run smoke:ui`
- `npm run smoke:streaming`
- `npm run smoke:integration`
- `npm run smoke:3b-closure`
- `scripts/check-public-scrub.sh`

## Result
Retained: all gates passed. Rust gates were not run because this slice touched only frontend source and frontend smoke scripts.
