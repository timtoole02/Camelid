# Camelid SafeTensors Plan

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Camelid's active runtime path remains GGUF/TinyLlama parity. SafeTensors support is a parallel architecture lane: design the seams now, avoid GGUF-only assumptions, and do not add loader/runtime churn until the current real-model correctness gate is stable.

## Current State

- `src/gguf/reader.rs` parses GGUF metadata and tensor descriptors directly from local files.
- `src/tensor/mod.rs` materializes supported GGUF tensor payloads into CPU `f32` tensors, preserving some source-type diagnostics.
- `src/model.rs` binds GGUF LLaMA metadata and tensor names into dense Camelid config/weight structures.
- `src/tokenizer/mod.rs` builds a minimal LLaMA/SPM tokenizer from GGUF `tokenizer.ggml.*` metadata and preserves `tokenizer.chat_template`.
- `src/api/mod.rs` currently accepts a local path through `/api/models/load` and treats loaded state as a GGUF file plus optional LLaMA config/tensors/tokenizer.

That shape is healthy for Phase 7, but SafeTensors would need sidecar config/tokenizer files and a model-source abstraction instead of directly assuming every loadable model is one self-contained GGUF file.


## 2026-04-28 Architecture Follow-Up

Inspection after the TinyLlama Q8_0 parity gate shows the right SafeTensors seam is still above the dense runtime, not inside the current GGUF reader or matmul path:

- `LoadedModel` in `src/api/mod.rs` is still GGUF-shaped (`GgufFile`, `LlamaModelConfig`, `LlamaTensorBinding`, tokenizer state). A SafeTensors slice should first replace that assumption with a source manifest/readiness summary while keeping the existing GGUF route behavior unchanged.
- `LlamaModelConfig::from_gguf` and `LlamaTensorBinding::bind` in `src/model.rs` are the useful extraction seams: add HF/config constructors and table-driven tensor-role binding next to them, rather than teaching GGUF descriptors about Hugging Face names.
- `CpuTensor` already records source dtype diagnostics and optional Q8_0 block metadata. SafeTensors materialization should initially decode F16/BF16/F32 into the same `CpuTensor` boundary; do not add quantized SafeTensors variants until a real supported source requires them.
- The active performance/correctness lane is GGUF/TinyLlama. Any SafeTensors code should start as manifest/config/descriptor tests and must not change `src/inference.rs` execution semantics until a descriptor-only fixture and tokenizer parity path are proven.

Recommended first API surface remains local-only: accept a directory path, detect `config.json`, `tokenizer.json`, `*.safetensors`/`model.safetensors.index.json`, and report `metadata_ready`, `tokenizer_ready`, `weights_ready`, and `generation_ready=false` until all required sidecars and dense LLaMA tensor roles validate.

## 2026-04-28 Late Architecture Refresh

This pass re-inspected the current loader/runtime seams while the GGUF/TinyLlama performance lane is active. The recommendation stays deliberately non-invasive: SafeTensors should enter Camelid as a model-source/readiness layer, not as a direct edit to `src/inference.rs` or the GGUF tensor math.

Concrete interface shape to add first:

```rust
pub enum ModelSourceKind {
    GgufFile,
    HuggingFaceDirectory,
}

pub struct ModelSourceReadiness {
    pub metadata_ready: bool,
    pub tokenizer_ready: bool,
    pub weights_ready: bool,
    pub generation_ready: bool,
    pub blockers: Vec<String>,
}

pub trait ModelSourceDescriptor {
    fn kind(&self) -> ModelSourceKind;
    fn id(&self) -> &str;
    fn architecture(&self) -> Option<&str>;
    fn tensor_names(&self) -> Vec<&str>;
    fn readiness(&self) -> ModelSourceReadiness;
}
```

Implementation guidance from current code inspection:

- Keep `/api/models/load` GGUF behavior untouched initially. Add directory detection only after a descriptor-only test fixture proves `config.json` + SafeTensors header parsing.
- Split `LoadedModel` conceptually into source manifest, dense config, tensor binding, tokenizer state, and readiness. Today those are all GGUF-shaped fields in `src/api/mod.rs`; a narrow manifest layer is the least disruptive migration.
- Add `LlamaModelConfig::from_hf_config` beside `from_gguf`, mapping `model_type=llama`, `hidden_size`, `num_hidden_layers`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`, `rms_norm_eps`, `rope_theta`, `vocab_size`, and `tie_word_embeddings`. Reject unsupported `architectures`, `rope_scaling`, sliding-window attention, MoE, or non-LLaMA models until tests exist.
- Add a table-driven HF tensor role mapper beside `LlamaTensorBinding::bind`; do not mutate GGUF descriptor names. Preserve source orientation and require fixture coverage before any transpose/reinterpretation.
- Decode initial SafeTensors dtypes `F32`, `F16`, and `BF16` into the existing `CpuTensor` f32 boundary. Do not add quantized SafeTensors, mmap lifetime plumbing, or runtime fast paths until descriptor/config/tokenizer readiness is stable.
- Gate generation on tokenizer parity. A parsed `tokenizer.json` should not set `generation_ready=true` until `tokenizers` adapter tests cover BOS/EOS, added/control tokens, chat template rendering, and round-trip decode for the target fixture.

Recommended crate posture remains:

- Core loader: `safetensors`, `memmap2`, `serde`, `serde_json`, `half`.
- Tokenizer adapter: `tokenizers` behind a feature or clearly isolated module.
- Optional future download/cache: `hf-hub`, but not in the first local-directory path because network access, gated repos, and license acceptance are approval-sensitive.

First safe milestones for the next implementation slice:

1. Add `ModelSourceManifest` / `ModelSourceReadiness` structs and tests without changing runtime behavior.
2. Add a tiny SafeTensors fixture generator and header/descriptor parser test for F32/F16/BF16 tensors.
3. Add local HF `config.json` parsing into `LlamaModelConfig`, with explicit unsupported errors.
4. Add tensor-role binding tests for dense LLaMA names including absent `lm_head.weight` + `tie_word_embeddings=true`.
5. Add a `tokenizers` adapter fixture and keep `generation_ready=false` until tokenizer parity passes.
6. Only then expose local-directory load as metadata/tokenizer/weights readiness, still leaving generation off until a small HF dense model proves one-token generation.

## 2026-04-29 Loader/Runtime Seam Check

The latest Llama 3 8B memory work strengthens the SafeTensors recommendation rather than changing it. Camelid now has a block-only Q8_0 GGUF path (`TensorStore::load_q8_0_blocks`) for future lazy/on-demand kernels, but SafeTensors should still arrive above that runtime boundary:

- First add source-manifest/readiness plumbing, not inference execution changes.
- Treat GGUF Q8_0 block retention as a pattern for source-native descriptor retention; do not force SafeTensors weights through an eager full-f32 materialization path just because `CpuTensor` currently does that for first support.
- Initial SafeTensors support should be descriptor/config/tokenizer readiness for local Hugging Face directories, with generation held at `generation_ready=false` until a tiny fixture proves dtype decode, tensor-role orientation, chat-template/tokenizer parity, and one-token dense execution.
- For 8B-class HF directories, use mmap-backed shard/header inspection and a materialization budget before any tensor decode, mirroring the current GGUF safety guard so Camelid never repeats the prior SIGKILL failure mode.
- Keep runtime fast-path work owned by the GGUF/Llama 3 performance lane. The SafeTensors lane should contribute only small interfaces and tests until the lazy/mmap/on-demand weight plan is stable.

## 2026-04-29 SafeTensors Architect Check

Current `main` is still GGUF-shaped at the API boundary: `LoadedModel` stores a path, `GgufFile`, optional `LlamaModelConfig`, optional `LlamaTensorBinding`, and tokenizer state. The loader/runtime modules now include stronger Llama 3 memory guardrails and `TensorStore::load_q8_0_blocks` / `Q8_0TensorBlocks` for GGUF Q8_0 lazy-execution groundwork, but no SafeTensors code should consume that as permission to enter the generation path yet.

Recommendation for the first implementation slice remains intentionally small:

- Add a source/readiness layer in front of `LoadedModel`, not inside `src/inference.rs`. The layer should represent `Gguf` and `HuggingFaceSafeTensors` with explicit `metadata_ready`, `tokenizer_ready`, `weights_ready`, and `generation_ready` flags.
- Keep `/api/models/load` GGUF behavior exactly compatible while adding local-directory detection only behind descriptor/config tests. A SafeTensors directory can report readiness without becoming chat-runnable.
- Map HF `config.json` into `LlamaModelConfig` beside `from_gguf`; reject unsupported `model_type`, `architectures`, `rope_scaling`, sliding-window attention, MoE, and missing KV/head fields with typed blockers.
- Parse mmap-backed SafeTensors headers/shards into source descriptors first. Materialize no large tensors by default, and apply the same budget mindset used by the current 8B GGUF materialization guard before any decode.
- Keep tensor-role mapping table-driven for dense LLaMA names and preserve orientation metadata. Do not transpose or reinterpret rows until a tiny SafeTensors fixture proves storage order against the existing Camelid matmul/output-projection conventions.
- Treat `tokenizer.json` + tokenizer sidecars as a separate readiness gate. `generation_ready` must stay false until BOS/EOS/control-token, chat-template, encode/decode, and one-token fixture parity are proven.

Near-term milestone I would queue after the active GGUF/Llama 3 lazy-Q8 lane: a docs/test-only `ModelSourceManifest` module plus a tiny local SafeTensors fixture that validates shard/header descriptors and HF config parsing, with no runtime generation changes.


## 2026-04-29 Current-Head Review

Fresh inspection on `main` at `bcaccfa` confirms the SafeTensors lane should stay interface/readiness-first while the GGUF Llama 3 lazy-Q8 execution path remains the active performance work. The current runtime evidence now includes serial `Q8_0TensorBlocks::dot_row_f32` / `dot_all_rows_f32` benchmarks and explicit memory-field reporting, but those are GGUF execution primitives, not a SafeTensors support claim.

Recommended next slice remains deliberately small and non-invasive:

- Add a `ModelSourceManifest` / `ModelSourceReadiness` layer ahead of the current GGUF-shaped `LoadedModel` in `src/api/mod.rs`; keep existing `/api/models/load` GGUF behavior byte-for-byte compatible until descriptor tests exist.
- Add local Hugging Face directory detection only for readiness reporting: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `*.safetensors`, and optional `model.safetensors.index.json`.
- Implement `LlamaModelConfig::from_hf_config` beside `from_gguf` and table-driven dense LLaMA tensor-role mapping beside `LlamaTensorBinding::bind`; do not teach GGUF descriptors Hugging Face names.
- Parse SafeTensors headers/shards via mmap-backed byte storage first. Defer tensor materialization and generation until dtype decode, tensor orientation, tokenizer/chat-template parity, and a tiny one-token fixture are proven.
- Keep `generation_ready=false` for Hugging Face SafeTensors directories until all config, tokenizer, required tensor roles, and one-token execution gates pass.

This keeps Camelid's product story honest: SafeTensors becomes a model-source abstraction and readiness lane first, while current technical identifiers and GGUF/TinyLlama/Llama 3 Q8 work continue under the existing `camelid` crate/binary during the transition.

## 2026-04-29 05:06 PT Current-Head Check

Fresh inspection on `main` at `f1cea2a` keeps the SafeTensors lane in architecture/readiness mode. The active Camelid runtime is still a GGUF-shaped loader: `/api/models/load` calls `read_metadata`, `LoadedModel` stores `GgufFile` plus optional `LlamaModelConfig`, `LlamaTensorBinding`, and tokenizer state, and `TensorStore` is still keyed by GGUF descriptors. The newer retained-Q8 helpers (`load_q8_0_blocks`, `dot_row_f32`, `dot_all_rows_f32`, and `dot_single_input_row_f32`) are useful examples for source-native, non-eager weight handling, but they remain GGUF Llama 3 lazy-linear groundwork and should not pull SafeTensors into the generation path yet.

Recommendation for the next SafeTensors slice:

- Add a small `ModelSourceManifest` / `ModelSourceReadiness` module ahead of `LoadedModel`, with variants for current GGUF files and local Hugging Face SafeTensors directories.
- Keep existing GGUF load/API behavior compatible; local-directory detection should initially report readiness only, not unlock chat.
- Parse HF `config.json` into a new `LlamaModelConfig::from_hf_config` path and produce typed blockers for unsupported `model_type`, `architectures`, `rope_scaling`, sliding-window attention, MoE, or incomplete head/KV metadata.
- Parse `model.safetensors.index.json` plus mmap-backed shard headers into tensor descriptors before any materialization. For 8B-class directories, apply the same materialization-budget mindset as the GGUF path before decoding weights.
- Keep dense LLaMA tensor-role mapping table-driven (`model.layers.*.self_attn.*`, `model.layers.*.mlp.*`, norms, embeddings, `lm_head`) and preserve orientation metadata until fixture tests prove Camelid row conventions.
- Treat `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` as a separate readiness gate. `generation_ready` stays false until tokenizer/chat-template parity and a tiny one-token fixture execution pass.

This is intentionally not a runtime support claim. It gives Camelid a clean model-source seam while Backend/Performance continue the current GGUF Llama 3 lazy-Q8 execution lane.

## 2026-05-01 Current-Head Check

Fresh inspection on `main` at `231d7e1` keeps the same recommendation, and the current code shape still supports it cleanly:

- `src/api/mod.rs` is still explicitly GGUF-shaped at load time: `LoadedModel` stores `path`, `GgufFile`, optional `LlamaModelConfig`, optional `LlamaTensorBinding`, and tokenizer state. That is the exact seam to split into `source manifest/readiness` plus `runtime-ready loaded model` later.
- `src/model.rs` still has the two most reusable extraction points: `LlamaModelConfig::from_gguf` for architecture/hparam lifting and `LlamaTensorBinding::bind` for dense LLaMA tensor-role binding/shape validation. A SafeTensors lane should add HF-side constructors/mappers beside these, not inside the GGUF reader.
- `src/tensor/mod.rs` still materializes into `CpuTensor` and already preserves source-type diagnostics plus optional Q8/file-backed metadata. That makes initial SafeTensors F32/F16/BF16 decode into `CpuTensor` a low-risk fit without touching active lazy-Q8 runtime work.
- `src/tokenizer/mod.rs` now supports the current GGUF LLaMA SPM plus GPT-2/BPE `llama-bpe` lane, but it is still GGUF-metadata-driven. Hugging Face `tokenizer.json` should remain a separate adapter/readiness lane instead of being forced through the GGUF tokenizer path.

Recommended next safe milestone is still docs/test-first: add a tiny `ModelSourceManifest` / `ModelSourceReadiness` module and fixture coverage for local HF directory detection plus `config.json` / SafeTensors header parsing, while keeping `generation_ready=false` and leaving `/api/models/load` GGUF behavior unchanged.

## 2026-05-01 Evening Current-Head Check

Fresh inspection on `main` at `c29fd71` keeps the SafeTensors lane firmly in docs/interface territory and does not justify runtime churn while the larger Llama GGUF evidence lane is still active:

- `src/api/mod.rs` still centers `LoadedModel` on a loaded `GgufFile` plus optional dense LLaMA config/binding and GGUF-derived tokenizer state. That remains the cleanest seam for a future split into `ModelSourceManifest`/readiness versus runtime-ready loaded weights.
- `src/model.rs` still exposes the two right extension points: `LlamaModelConfig::from_gguf` for architecture/config lifting and `LlamaTensorBinding::bind` for required dense LLaMA tensor-role validation. SafeTensors should add HF-side constructors and tensor-role mapping beside those functions rather than teaching the GGUF reader HF naming.
- `src/tensor/mod.rs` is still a good initial landing zone for SafeTensors dtype decode because `CpuTensor` already preserves source-type diagnostics plus optional retained/file-backed weight metadata. Initial SafeTensors support should stay limited to descriptor/header parsing and later F32/F16/BF16 decode into `CpuTensor`; quantized SafeTensors can wait until a real supported source demands them.
- `src/tokenizer/mod.rs` remains GGUF-metadata-driven even though it now covers both the current LLaMA/SPM and GPT-2/BPE `llama-bpe` lanes. Hugging Face `tokenizer.json` should still land as a separate adapter/readiness path, not as an overload of the current GGUF tokenizer constructor.

Recommended next safe milestone remains unchanged: add a tiny source/readiness layer and fixture coverage for local HF directory detection plus `config.json` / SafeTensors header parsing, while keeping `/api/models/load` GGUF-compatible and `generation_ready=false` for any SafeTensors directory.

## 2026-05-05 Current-Head Check

Fresh inspection on `main` at `96719418a03b` after the latest public evidence/docs updates and guarded metadata-template renderer follow-up does not change the SafeTensors architecture recommendation. Camelid's release ledger is now much stronger for exact GGUF Q8_0 rows, but the code seam is still the same:

- `src/api/mod.rs` still loads through `read_metadata`, stores `LoadedModel` as a GGUF path/file plus optional `LlamaModelConfig`, `LlamaTensorBinding`, and GGUF-derived tokenizer state, and reports capabilities from that runtime-ready GGUF shape. The guarded Llama 3 metadata-template renderer remains an opt-in GGUF metadata path, not a Hugging Face SafeTensors/source-manifest abstraction.
- `src/model.rs` still has the correct expansion seams: keep `LlamaModelConfig::from_gguf` and `LlamaTensorBinding::bind` intact, then add HF config parsing and table-driven HF tensor-role binding beside them.
- `src/tensor/mod.rs` now has mature GGUF retained/file-backed Q8_0 helpers and materialization-budget guardrails. Reuse that safety posture for SafeTensors descriptor/header inspection, but do not tie SafeTensors readiness to the GGUF Q8 execution fast path.
- `src/tokenizer/mod.rs` remains GGUF-metadata driven for SPM and `llama-bpe`; Hugging Face `tokenizer.json` still needs its own adapter/parity fixture before it can affect `generation_ready`.

The next SafeTensors implementation slice should therefore remain local, test-first, and non-generating: introduce `ModelSourceManifest` / `ModelSourceReadiness`, detect Hugging Face-style directories, parse `config.json` plus SafeTensors shard headers into descriptors, and keep `/api/models/load` behavior for existing GGUF files unchanged. `generation_ready` must stay false until tokenizer parity, tensor orientation, dtype decode, and a tiny one-token dense fixture all pass.

## 2026-05-05 09:20 PT Current-Head Check

Fresh inspection on `main` at `81a45c42aa6e` after the 1024/2048 context-capability guardrail work keeps the SafeTensors lane in the same architecture/readiness box. The new `/api/capabilities` tests now explicitly protect per-model context-pack boundaries: exact GGUF rows can advertise different 1024/2048 evidence states while unsupported or not-yet-promoted rows stay blocked or ready-to-run-only. That is useful for SafeTensors because it proves Camelid already has a public contract shape for honest readiness reporting without pretending every parsed source is generation-ready.

Architectural implications for the first SafeTensors slice:

- Treat `ModelSourceReadiness` as the same kind of evidence contract as the current capability rows: source detection, metadata, tokenizer assets, weights, and generation must be separate fields, not one broad support flag.
- Keep the current `LoadedModel` GGUF path unchanged. A Hugging Face SafeTensors directory should be able to produce a manifest/readiness summary while `/v1/completions` and chat paths continue to reject it until the runtime gates pass.
- Reuse the capability-boundary testing style for SafeTensors fixtures: one complete local-directory fixture, one missing-tokenizer fixture, one missing/invalid shard-index fixture, and one unsupported-config fixture. Tests should assert the exact blocker strings so future docs/API changes cannot silently promote support.
- Keep context-window claims out of the SafeTensors manifest until HF `config.json` parsing, RoPE fields, tokenizer/chat-template parity, and a bounded prompt-pack run all agree. `max_position_embeddings` alone is metadata, not validated context support.
- Preserve the no-network default. The local Ubuntu validation tree has active unrelated dirty work, so SafeTensors architecture work should avoid remote mutation and remain local/docs/test-first unless a dedicated implementation branch is opened.

Recommended next code seam is now slightly sharper: add `src/source.rs` (or `src/model_source.rs`) with owned manifest/readiness structs and descriptor-only tests, then have the existing GGUF file path populate that struct internally before any Hugging Face directory path is exposed through API responses.

## 2026-05-05 12:20 PT Implementation Slice

Local `main` head `fd894db8a560` now has the first SafeTensors source/readiness seam queued in code without changing the GGUF runtime path: `src/model_source.rs` introduces `ModelSourceManifest`, `ModelSourceReadiness`, and descriptor-only inspection for `.gguf` files plus local Hugging Face SafeTensors directories. The module is deliberately non-generating: SafeTensors directories can report separate metadata/tokenizer/weights readiness, but always carry `generation_disabled` until tokenizer parity, tensor orientation, dtype decode, and a one-token dense fixture pass.

The first fixture coverage matches the architecture guardrails from the 09:20 check: complete local HF directory, missing tokenizer, missing shard index, invalid shard index, and unsupported config all assert exact blocker codes. The implementation still performs no network access, no Hugging Face Hub downloads, no SafeTensors weight decode, and no `/api/models/load` behavior change.

Next code slice should stay local/test-first: parse richer HF `config.json` fields into typed Camelid-owned metadata, then add SafeTensors header/tensor-descriptor parsing for tiny fixture shards before any materialization or API exposure.

## Recommended Rust Crates / APIs

- `safetensors` (`0.7.0` current crates.io default as of 2026-04-28): use `safetensors::SafeTensors::deserialize` / tensor views for safe header parsing and per-tensor byte slices. Prefer read-only mmap-backed byte storage for large files; copy/decode into Camelid CPU tensors only at the runtime boundary.
- `memmap2` (`0.9.10` current crates.io default as of 2026-05-05): mmap local `.safetensors` shards and avoid eagerly reading multi-GB weights into intermediate buffers.
- `serde` / `serde_json`: parse Hugging Face sidecars (`config.json`, `tokenizer_config.json`, `generation_config.json`, `special_tokens_map.json`) into Camelid-owned structs.
- `tokenizers` (`0.23.1` current crates.io default as of 2026-04-28): use behind a feature flag or adapter for `tokenizer.json` parity instead of stretching the current GGUF-only SPM parser to cover BPE/Unigram/WordPiece variants.
- `hf-hub` (`1.0.0-rc.0` is the current crates.io default as of 2026-05-05): optional future download/cache layer. Keep it out of the core loader initially; gated model access and license acceptance are product/approval-sensitive, and an eventual adoption should pin deliberately rather than entering the first local-directory path by accident.
- `half` (`2.7.1` current crates.io default as of 2026-05-05): decode F16/BF16 tensors when SafeTensors dtype views are materialized into Camelid `f32` CPU tensors.

Avoid pulling in a full inference framework just to read SafeTensors. Camelid should own the config mapping, tensor binding, and runtime semantics.

## Model-Source Abstraction Shape

Introduce a narrow boundary before adding a SafeTensors loader:

```rust
pub enum ModelSourceKind {
    Gguf,
    HuggingFaceSafeTensors,
}

pub struct ModelSourceManifest {
    pub id: String,
    pub kind: ModelSourceKind,
    pub root: PathBuf,
    pub weight_files: Vec<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_config_path: Option<PathBuf>,
    pub generation_config_path: Option<PathBuf>,
}

pub trait ModelSource {
    fn manifest(&self) -> &ModelSourceManifest;
    fn architecture(&self) -> Option<&str>;
    fn tensor_names(&self) -> Vec<String>;
    fn tensor_descriptor(&self, name: &str) -> Option<TensorDescriptor>;
    fn tokenizer_assets(&self) -> TokenizerAssetRefs;
}
```

Then adapt existing GGUF code to implement the same source-level traits without changing dense runtime math. A later SafeTensors source can map Hugging Face tensor names into the existing `LlamaTensorBinding` roles.

Keep the conversion pipeline explicit:

1. Resolve a `ModelSourceManifest` from a local GGUF file or Hugging Face-style directory.
2. Parse architecture/config into Camelid-owned `LlamaModelConfig` or a future generic decoder config.
3. Bind source tensor names into Camelid model roles.
4. Materialize selected tensors into `CpuTensor` only when building runtime weights.
5. Report capability/readiness honestly if config, tokenizer, or required tensors are missing.

## Hugging Face Config / Tokenizer Gaps

SafeTensors is only weights. A runnable Hugging Face model also needs sidecars:

- `config.json`: architecture/model type, hidden size, layer count, intermediate size, attention/KV heads, RoPE settings/scaling, RMSNorm epsilon, vocab size, tie-word-embeddings, dtype expectations, sliding-window or attention variants.
- `model.safetensors.index.json`: required for sharded models; maps tensor names to shard files.
- `tokenizer.json`: source of truth for BPE/Unigram tokenization in most HF repos.
- `tokenizer_config.json`: chat template, BOS/EOS behavior, legacy tokenizer flags, cleanup behavior.
- `special_tokens_map.json`: role/control token IDs and added tokens.
- `generation_config.json`: default EOS/PAD/temperature/top-p semantics; useful for compatibility but should not silently override explicit API request settings.

Current Camelid tokenizer code is GGUF LLaMA/SPM-oriented. Do not pretend `tokenizer.json` works until a `tokenizers` adapter and parity tests exist.

## Tensor Name Mapping

Initial mapping should target dense LLaMA-family HF names and remain table-driven:

- embeddings: `model.embed_tokens.weight` -> `token_embd.weight`
- final norm: `model.norm.weight` -> `output_norm.weight`
- lm head: `lm_head.weight` -> `output.weight` or tied embeddings when absent and config allows tying
- per-layer attention: `model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
- per-layer MLP: `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`
- per-layer norms: `input_layernorm.weight`, `post_attention_layernorm.weight`

The mapper must preserve orientation metadata. Do not silently transpose until a tiny synthetic SafeTensors fixture proves descriptor shape, storage order, and Camelid matmul interpretation.

## Risks

- SafeTensors support can distract from the current Phase 7 GGUF parity blocker; keep this lane docs-first until real-model correctness is stable.
- HF model repos are not self-contained in a single file; missing or incompatible sidecars should produce typed readiness errors.
- Tokenizer parity is a larger risk than tensor parsing, especially chat templates and added/control tokens.
- RoPE variants (`rope_scaling`, Llama 3.x long context, YaRN-like fields), GQA, tied embeddings, and tensor orientation can silently break output quality.
- Gated model access and license acceptance on Hugging Face require explicit product approval; Camelid should first support local directories supplied by the user.
- Sharded weights demand strict tensor-to-file index validation before loading.

## First Implementation Milestones

1. Add a docs/test-only `ModelSourceManifest` sketch and keep GGUF behavior unchanged.
2. Add a tiny local SafeTensors fixture generator in tests with two or three known tensors and validate descriptor parsing only.
3. Parse local HF `config.json` into Camelid LLaMA config fields, returning typed errors for unsupported architectures/rope variants.
4. Add table-driven tensor-name binding for dense LLaMA-family SafeTensors fixtures, including tied-output behavior.
5. Add a `tokenizers`-backed tokenizer adapter behind a feature flag and parity tests for a tiny tokenizer fixture before exposing runtime readiness.
6. Only after GGUF Phase 7 correctness is stable, attempt a small local HF dense model load path that reports metadata/readiness first, then one-token generation.

## Non-Goals For Now

- No network downloads by default.
- No gated Hugging Face access automation.
- No broad architecture support beyond dense LLaMA-family mapping until the first path is proven.
- No replacement of the current GGUF runtime path.
