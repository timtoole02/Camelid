# Camelid Roadmap Archive

This file keeps completed roadmap history concise so `ROADMAP.md` can stay focused on active and
upcoming work.

For detailed execution logs, artifact paths, validation outputs, and diagnostic notes, see
`STATUS.md`.

## Completed phase summary

### Phase 0 — Repo setup and discovery

Completed the initial planning and repo-orientation work, including the first architecture,
roadmap, status, and research documents.

### Phase 1 — Rust backend skeleton

Established the Rust project shape, CLI/server entrypoints, typed errors, tracing, and health
surfaces.

### Phase 2 — GGUF metadata lane

Implemented GGUF metadata and tensor-directory parsing with validation and structured inspection
output.

### Phase 3 — Initial tokenizer lane

Brought up the initial LLaMA/SPM tokenizer path with encode/decode support and explicit unsupported
behavior for out-of-scope tokenizer families.

### Phase 4 — Reference tensor loading lane

Implemented reference CPU tensor loading and the first required dequantization/runtime primitives for
supported early lanes.

### Phase 5–7 — Minimal inference and TinyLlama parity hardening

Built the narrow end-to-end generation path, OpenAI-compatible local API slice, TinyLlama Q8_0
parity investigation tooling, and the validated TinyLlama Q8_0 release gate.

## Archive rule

When a roadmap lane is no longer an active sequencing decision and its remaining value is historical
context, move the detail here or into `STATUS.md` instead of letting `ROADMAP.md` grow into an
execution log.
