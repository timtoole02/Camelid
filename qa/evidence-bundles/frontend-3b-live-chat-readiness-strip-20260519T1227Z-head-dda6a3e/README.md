# Frontend 3B live-chat exact-row readiness strip

Target: Llama 3.2 3B Instruct Q8_0 frontend/WebUI closure.

Retained slice:
- Non-empty/live chat threads now keep the same runtime and support-contract readiness cards that the empty chat hero shows.
- The live surface names both required gates: loaded_now=true + generation_ready=true on the active runtime row, and an exact supported COMPATIBILITY.md / /api/capabilities row for the selected 3B Q8_0 GGUF.
- Regression coverage asserts the rendered live chat keeps those gates visible after messages exist, while existing exact-row gating and streaming-loader coverage remains green.

Support contract: this does not widen model support. Chat remains gated by loaded_now=true, generation_ready=true, active_model_id matching, and an exact supported compatibility row for the selected/active GGUF.

Evidence files:
- `git-state.txt`
- `dirty-diff-stat.txt`
- `frontend-gates.log`
- `public-scrub-privacy.log`
