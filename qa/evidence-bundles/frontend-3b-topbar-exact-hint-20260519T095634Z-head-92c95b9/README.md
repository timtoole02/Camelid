# Frontend 3B topbar exact-hint and 512-context coverage closure

Target: Llama 3.2 3B Instruct Q8_0 frontend/WebUI closure.

Retained slice:
- The topbar support-contract detail now prioritizes the active or selected exact-row hint label before falling back to the global current gate. That keeps exact-row blockers such as quant mismatch or quant missing visible instead of showing another supported row as the apparent support detail.
- The frontend model-state smoke fixture now preserves the 3B 512-context bounded-pack field and asserts the tracked exact-row 512 boundary alongside 1024/2048, so the 3B row cannot silently lose its first checked context pack in frontend regression coverage.

Support contract: this does not widen model support. Chat remains gated by loaded_now=true, generation_ready=true, active_model_id matching, and an exact supported compatibility row for the selected/active GGUF.

Evidence files:
- `git-state.txt`
- `dirty-diff-stat.txt`
- `frontend-gates.log`
- `public-scrub-privacy.log`
