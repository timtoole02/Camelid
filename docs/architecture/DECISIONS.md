# Camelid Decisions

Naming transition: Camelid is the project/product name. Existing `camelid` names below are historical/current implementation names unless a later decision explicitly renames crates, binaries, or repository metadata.

## Decision log

### D0001 — Rust-native implementation

- Status: Accepted
- Date: 2026-04-27
- Decision: Implement Camelid in Rust rather than wrapping an existing native runtime as the primary runtime.
- Context: The project goal is a maintainable Rust-native inference backend, not just a thin wrapper around an existing runtime.
- Consequences:
  - More implementation work up front.
  - Better ownership of architecture, API shape, and future integration with ForgeLocal/Fathom.
  - Requires careful clean-room behavior: study public model formats and behavior without copying implementation code.

### D0002 — GGUF-first model format

- Status: Accepted
- Date: 2026-04-27
- Decision: Target GGUF first.
- Context: GGUF is a dominant local model container used by many local-model distributions.
- Consequences:
  - Correct parser behavior is foundational.
  - Metadata/tensor descriptor parsing must land before inference.
  - Test fixtures for valid and malformed files are required early.

### D0003 — CPU/reference implementation before acceleration

- Status: Accepted
- Date: 2026-04-27
- Decision: Build a correct CPU/reference path before SIMD, Metal, CUDA, or other acceleration.
- Context: Correctness and maintainable architecture are more important than early performance.
- Consequences:
  - Early inference will be slow.
  - Optimized backends can later be validated against reference behavior.

### D0004 — OpenAI-compatible `/v1` for ForgeLocal first integration

- Status: Accepted
- Date: 2026-04-27
- Decision: Expose a local OpenAI-compatible `/v1` API first, likely at `127.0.0.1:8181`.
- Context: ForgeLocal already supports external OpenAI-compatible providers, so this minimizes frontend changes.
- Consequences:
  - Camelid can integrate with ForgeLocal before a custom provider kind exists.
  - API must be honest: return `501` for unimplemented generation/streaming.

### D0005 — Typed errors

- Status: Accepted
- Date: 2026-04-27
- Decision: Use typed Rust errors and stable HTTP error envelopes.
- Context: Binary parsing, unsupported models, and inference failures need clear diagnostics.
- Consequences:
  - Better tests and safer API behavior.
  - Avoids stringly typed failures.

### D0006 — Phase-first delivery

- Status: Accepted
- Date: 2026-04-27
- Decision: Work in phases and keep every phase runnable/testable.
- Context: A full production-grade local inference runtime is too large for one pass.
- Consequences:
  - First vertical slice is metadata/API, not inference.
  - Unsupported functionality must be explicit rather than hidden.

### D0007 — Token-major final vocab projection for GGUF output weights

- Status: Accepted
- Date: 2026-04-28
- Decision: Ordinary Camelid generation interprets final vocab projection weights as token-major output rows by default, while descriptor-shaped projection remains available only for opt-in diagnostics.
- Context: The TinyLlama Q8_0 `hello` parity gate showed prompt/tokenizer parity but wrong first-token logits until `output.weight` was treated as token rows for the final vocab projection. The runtime fix is anchored in `src/inference.rs` by `output_projection_runtime` choosing `OutputProjectionLayout::TokenMajor` when dense diagnostics are not collected, with layout-specific handling in `output_projection_with_layout`.
- Consequences:
  - The Phase 7 TinyLlama Q8_0 first-token and five-prompt 50-token parity gate depends on preserving this default.
  - Diagnostic experiments may still force descriptor interpretation, but ordinary generation must not inherit that override accidentally.
  - Platform ports and future model-source work need tests/guardrails for this layout assumption before claiming parity.

## Decision template

```markdown
### DXXXX — Title

- Status: Proposed | Accepted | Rejected | Superseded
- Date: YYYY-MM-DD
- Decision:
- Context:
- Alternatives considered:
- Consequences:
- Follow-ups:
```
