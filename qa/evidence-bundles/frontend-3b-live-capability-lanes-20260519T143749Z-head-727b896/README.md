# Frontend 3B live capability lanes

Retained slice: empty and non-empty 3B chat readiness keeps a row-scoped capability lane visible beside runtime and exact-row support. The lane shows Template/Jinja evidence and throughput promotion state from /api/capabilities without widening model-native context, production-throughput, portability, neighboring rows, or broad-family support.

Head before commit: 727b8969beb22d1c70fde5813375e4c50b20ff4b
Branch: frontend-3b-webui-closure-6e9017a8-20260519
Live backend: http://127.0.0.1:8181
Live frontend preview used for npm run smoke: http://127.0.0.1:4176

Gates captured:
- scripts/check-public-scrub.sh
- node scripts/audit-evidence-bundle-privacy.mjs --strict
- CAMELID_FRONTEND_URL=http://127.0.0.1:4176 npm run smoke
- npm run smoke:model-state
- npm run smoke:streaming
- npm run smoke:3b-closure
- npm run smoke:ui
- npm run smoke:integration
- npm run build
