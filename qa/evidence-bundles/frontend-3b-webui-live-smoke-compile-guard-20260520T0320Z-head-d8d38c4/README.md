# Frontend 3B WebUI live smoke compile guard — 2026-05-20T0320Z

Scope: Llama 3.2 3B Instruct Q8_0 frontend/WebUI closure support-contract gate.

Retained slice:
- Preserves the frontend 3B explicit `loaded_now` support-contract gate from the prior slice.
- Adds the narrow non-x86 fallback for the x86-only AVX2 Q8 kernel gate so the current macOS main-based tree can compile and start Camelid for the required live frontend smoke.
- No 3B support claim is widened: chat still unlocks only when `/v1/health.loaded_now=true`, `generation_ready=true`, active model identity matches, and the exact supported 3B Q8_0 row resolves from `/api/capabilities`.

Live backend note: canonical Ubuntu host was reachable by SSH, but its local Camelid API on `127.0.0.1:8181` was not listening during this slice. The live `npm run smoke` evidence below used a local exact-tree Camelid backend started with `cargo run -- serve --addr 127.0.0.1:8181` and no model loaded, proving the frontend/API surface starts and remains chat-blocked without runtime readiness.
