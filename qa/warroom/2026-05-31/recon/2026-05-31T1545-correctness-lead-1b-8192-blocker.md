# Correctness Lead 1B 8192 Blocker

Timestamp: 2026-05-31 15:45 PDT / 2026-05-31T22:45Z

## Update

[15:45 PDT] Camelid War Room Update

Lane:
Correctness Lead / Four protected Q8_0 rows

Owner:
Camelid Correctness Lead

Current goal:
Protect TinyLlama Q8_0, Llama 3.2 1B Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0 before speed work. Run or locate prompt-token, 1/5/50-token parity, API completions/chat, capabilities/WebUI readiness, and RSS/context evidence only where exact model artifacts are available.

Commit/SHA:
Camelid `48ebaf6672b368776bf2761943c75d451ba4624f`; working tree was already dirty before this pass.

What changed:
Built the release binary and ran a fresh local correctness pass against the only protected exact GGUF available on this runner: Llama 3.2 1B Instruct Q8_0. No runtime/source files, support text, compatibility claims, frontend claims, or performance patches were changed.

Evidence produced:
- Fresh local artifact directory: `target/warroom/2026-05-31T1545-correctness-lead/`.
- Fresh 1B compact parity: `llama32-1b-q8-1tok.json`, `llama32-1b-q8-5tok.json`, and `llama32-1b-q8-50tok.json`.
- Fresh API captures: `api-capabilities.json`, `v1-models.json`, `v1-completions.json`, and `v1-chat-completions.json`.
- Fresh RSS/timing context: `unique-chat-rss.json`.
- Fresh context checks: `context-512/summary.json` and `context-8192/summary.json`.

Correctness status:
- Llama 3.2 1B Q8_0 compact parity: PASS. Prompt tokens, generated tokens, and generated text matched the reference for 1-token, 5-token, and bounded 50-token requests. The 50-token request stopped at EOS after 10 generated tokens with `Hello! How can I assist you today?`.
- API completions/chat: PASS. `/v1/completions` returned generated token `[11]` for raw `hello`; `/v1/chat/completions` returned generated token `[9906]` and the expected compact chat prompt-token sequence.
- Capabilities/WebUI readiness: PASS for the loaded exact 1B row. `/api/capabilities` reports `llama32_1b_instruct_q8_0` as `supported_exact_row_smoke`, and the frontend integration smoke passed.
- RSS/timing context: LOCAL ONLY. The unique-chat run completed two measured one-token requests with average wall `299.35 ms`, average backend generate `285 ms`, weight-cache hits true, prompt-cache hits false. Process RSS was unavailable because this run did not pass the backend PID to the sampler.
- Context 512: PASS. Prompt-token, generated-token, and generated-text parity all matched.
- Context 8192: BLOCKER. The reference completed the roughly-8192-token recall prompt, but Camelid `/v1/chat/completions` timed out after 120000 ms before producing backend diagnostics. This directly conflicts with the current 1B capability row claiming a validated 8192 context pack.

Located prior protected-row evidence:
- TinyLlama Q8_0: prior evidence exists, but no exact TinyLlama GGUF was available for a fresh rerun in this cron window.
- Llama 3.2 1B Q8_0: prior 512/1024/2048/4096/8192 evidence exists, but this fresh run regressed the 8192 lane to a timeout blocker.
- Llama 3.2 3B Q8_0: prior parity/API/WebUI/RSS/context evidence exists, but no exact 3B GGUF was available for a fresh rerun in this cron window.
- Llama 3 8B Q8_0: prior parity/API/WebUI/RSS/context evidence exists, but no exact 8B GGUF was available for a fresh rerun in this cron window.

Blockers:
1. Llama 3.2 1B Q8_0 8192-context parity is a correctness blocker: reference completed, backend timed out after 120000 ms, and no backend parity report was produced.
2. Current `/api/capabilities` still claims the 1B row has `bounded_context_8192_pack: validated_fifth_pack`; do not treat that claim as protected on current local evidence until the 8192 pack is rerun cleanly or the claim is narrowed.
3. Fresh four-row protection remains blocked by missing exact local GGUFs for TinyLlama Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0.
4. This is a local Darwin arm64 fallback lane, not canonical Ubuntu x86_64 validation.
5. Raw harness artifacts are local-only until scrubbed; several generated harness files contain absolute operator paths.

Next 30 minutes:
Keep speed work gated. Fix or narrow the 1B 8192-context support claim before performance work. If exact TinyLlama, 3B, or 8B artifacts become available, rerun their prompt-token, 1/5/50-token parity, API completions/chat, capabilities/WebUI readiness, and RSS/context checks before any speed work.

Need from Tim:
Provide approved exact GGUF access for TinyLlama Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0, or confirm this local 1B-only blocker pass is the expected fallback for this cron window.

## Clean-Room Boundary

- llama.cpp was used only as an executable reference target through `llama-server` and `llama-tokenize`.
- No llama.cpp implementation code was copied into Camelid.
- This note does not widen readiness, compatibility, support, context, production-throughput, portability, neighboring-row, or broad Llama-family claims.
