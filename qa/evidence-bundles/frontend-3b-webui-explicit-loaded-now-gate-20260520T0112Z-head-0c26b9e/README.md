# Frontend 3B explicit loaded_now gate — 2026-05-20T0112Z

Scope: Llama 3.2 3B Instruct Q8_0 frontend/WebUI closure only.

Retained slice:
- Dashboard runtime readiness now treats `/v1/health.loaded_now` as mandatory evidence instead of synthesizing loaded state from `active_model_id`.
- Merged runtime/model rows only expose `loaded_now=true` and `generation_ready=true` when the matched backend row explicitly reports `loaded_now=true`; `generation_ready` is also gated by `loaded_now`.
- 3B closure regression smoke now guards against reintroducing the `active_model_id` fallback and covers merged-row loaded_now/generation_ready source-of-truth copy.

Live backend note: the canonical Ubuntu validation host was reachable over SSH, but the local Camelid API on `127.0.0.1:8181` was not listening, so this slice claims no fresh live API/WebUI promotion evidence. The retained evidence is the current-head local CI-equivalent frontend/support-contract gate set in `logs/`.
