# Frontend 3B acceptance quant gate

## Target
Harden the Llama 3.2 3B Instruct Q8_0 frontend acceptance card so a browser acceptance id does not hide the exact-row placeholder unless Q8_0 evidence is present.

## Feedback loop
- Local frontend gate: `npm run smoke:model-state && npm run smoke:ui && npm run smoke:streaming && npm run smoke:integration && npm run build`
- Canonical Ubuntu runtime probe attempted with the mandated SSH shape; backend API was unreachable on 127.0.0.1:8181, so remote live check is blocked and logged.

## Retain/reject
Retained locally: regression covers neighboring-quant browser acceptance records and the frontend gates are green. Remote API evidence is blocked by backend-down state, not by the frontend slice.

## Files
- frontend/src/views/ModelsView.jsx
- frontend/scripts/frontend-integration-smoke.mjs

## Notes
This slice does not widen support claims. It keeps the support contract exact-row scoped and leaves production-throughput/model-native context caveats intact.
