# Camelid Naming Audit

This audit covers replacing user-facing `llama` naming with Camelid without breaking real model compatibility. It separates safe Camelid branding from technical identifiers that must remain unchanged.

## Safe Camelid naming

Use **Camelid** for product/project branding, user-facing capability descriptions, docs that describe the local inference path, and generic internal abstractions that are not tied to an external model format.

This pass updated safe wording in:

- `README.md`
- `ROADMAP.md`
- `ARCHITECTURE.md`
- `FORGELOCAL_INTEGRATION.md`
- `frontend/README.md`
- selected API capability/error strings in `src/api/mod.rs`

## Technical `llama` names to preserve

Do not rename these casually; they are compatibility or parity identifiers rather than branding:

- GGUF metadata values and keys, including `general.architecture = "llama"` and `llama.*` keys.
- GGUF tokenizer metadata such as `tokenizer.ggml.model = "llama"`.
- Tensor fixture writers and tests that encode GGUF-compatible metadata.
- Real model names, filenames, and model IDs such as TinyLlama and `tinyllama-q8`.
- External parity/runtime names such as `llama-server`, `llama-url`, and parity environment variables.
- Recon notes that quote external APIs or source symbols such as `llama_tokenize`.
- README/product provenance language should stay neutral and avoid naming external runtimes unless the exact technical identifier is required.

## Deferred/risky renames

Rust types and serialized API fields such as `LlamaModelConfig`, `LlamaTensorBinding`, `DenseLlamaDims`, `llama_config`, and `llama_tensors` still exist. They are currently tightly coupled to GGUF LLaMA-family metadata and test fixtures. Renaming them should be a separate compatibility-aware API/code refactor with migration notes, because it may affect frontend state, JSON consumers, tests, and ongoing parity work.
