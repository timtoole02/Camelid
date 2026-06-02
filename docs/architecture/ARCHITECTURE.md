# Camelid Architecture

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Camelid is a Rust-native local inference backend for loading GGUF language models, tokenizing prompts, running autoregressive inference, sampling tokens, and serving local HTTP APIs. It should follow proven local-inference runtime patterns without copying external implementation code.

Naming transition: this architecture describes Camelid, but the current Rust crate, binary, repository metadata, and some proposed module/package names may still use `camelid`. Treat those as current implementation names, not the long-term product name.

## Goals

- Load and inspect GGUF model files.
- Provide tokenizer abstractions backed by GGUF metadata.
- Build CPU-first tensor loading/runtime primitives.
- Implement a narrow, correct Camelid inference path for supported GGUF decoder models before broad model support.
- Serve local clients through an OpenAI-compatible `/v1` API.
- Preserve a clean future path for provider/runtime integration without baking retired external-runtime assumptions into core APIs.

## Non-goals for early phases

- Full feature parity with every local inference runtime.
- Full model-family coverage.
- High-performance kernels before correctness.
- GPU/SIMD acceleration before reference CPU paths.
- Training or fine-tuning.
- Silent fallback when a model or feature is unsupported.

## Recommended project shape

Start as a single Rust crate if speed matters, but preserve these module boundaries. Split into workspace crates once the APIs stabilize.

```text
camelid/  # current repo/crate root during the Camelid transition
  Cargo.toml
  src/
    lib.rs
    error.rs
    config.rs
    gguf/
      mod.rs
      metadata.rs
      tensor.rs
      reader.rs
    tokenizer/
      mod.rs
    tensor/
      mod.rs
      cpu.rs
      quant.rs
    model/
      mod.rs
      llama.rs
    inference/
      mod.rs
      engine.rs
      kv_cache.rs
    sampling/
      mod.rs
    api/
      mod.rs
      openai.rs
      routes.rs
    bin/
      camelid.rs
      camelid-server.rs
  tests/
    gguf_metadata.rs
    gguf_malformed.rs
  fixtures/
    README.md
```

Long-term workspace split, pending a separate naming decision:

- Current/internal crate-prefix option: `camelid-*` (least disruptive during transition)
- Future Camelid-facing crate-prefix option: `camelid-*`

Do not rename crates or binaries as part of docs-only work; plan that separately so commands, package metadata, and tests move together.

## Core abstractions

### Errors

Use typed errors, likely via `thiserror`.

Principles:

- Binary parsing errors must include enough context to diagnose malformed files.
- Unsupported model features must return explicit unsupported errors.
- HTTP errors should map to stable JSON envelopes.

### GGUF

The GGUF reader is independent from inference. It should parse the file and expose metadata/tensor descriptors without knowing transformer semantics.

Responsibilities:

- Validate magic/version.
- Parse metadata key/value pairs.
- Parse tensor names, dimensions, dtype, and file offsets.
- Enforce alignment and bounds.
- Avoid eager tensor data copies.

Example public shape:

```rust
pub struct GgufFile {
    pub version: u32,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorDescriptor>,
}

pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64,
}
```

### SafeTensors / Hugging Face model sources

SafeTensors is a future model-source lane, not an active replacement for the GGUF Phase 7 path. The architecture target is to add a `ModelSource` boundary that can describe either a self-contained GGUF file or a Hugging Face-style local directory with `.safetensors` weights plus sidecar config/tokenizer files. Keep this docs/interface-first until GGUF real-model correctness is stable.

See [SAFETENSORS_PLAN.md](SAFETENSORS_PLAN.md) for the current crate/API recommendations, source-manifest shape, Hugging Face sidecar gaps, risks, and first implementation milestones.

### Tokenizer

Tokenizer support should be a separate lane. The first target is common LLaMA-style GGUF tokenizer metadata.

```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
}
```

Unsupported tokenizer types must fail clearly.

### Tensor runtime

Start CPU/reference-first. Keep acceleration pluggable later.

```rust
pub enum Device {
    Cpu,
    Metal { device_index: usize },
    External { provider: String },
}

pub trait TensorRuntime {
    type Tensor;
    fn load_tensor(&self, desc: &TensorDescriptor, bytes: &[u8]) -> Result<Self::Tensor>;
}
```

Initial tensor work should prioritize:

- dtype recognition
- byte-size validation
- quantization type recognition
- minimal dequantization for first supported model

### Model layer

The model layer converts generic GGUF metadata/tensors into typed model structures.

Responsibilities:

- Detect architecture from `general.architecture`.
- Parse model hparams.
- Validate required tensor names.
- Bind tensors to Camelid dense decoder structures while preserving GGUF architecture-specific metadata names.
- Return clear unsupported errors.

### Inference engine

The engine owns model sessions and generation orchestration.

Initial path:

1. Load model metadata.
2. Load tokenizer.
3. Load/validate tensors.
4. Prefill prompt.
5. Decode one token at a time.
6. Apply sampler.
7. Detokenize and stream/return output.

Do not build a generation loop until GGUF, tokenizer, tensor loading, and forward primitives are stable.

### Sampling

Sampling is independent from model execution.

Initial supported controls:

- deterministic greedy mode
- seed handling
- temperature
- top-k
- top-p
- OpenAI-style presence/frequency penalties
- logit bias
- OpenAI-style stop sequences

Tests must prove repeatability for fixed seeds and validation for unsupported or malformed sampling inputs.

### HTTP/API

Serve a local API suitable for local OpenAI-compatible clients and future provider/runtime alignment.

Minimum endpoints:

- `GET /health`
- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/models/:model`
- `POST /api/models/load`
- `POST /api/models/unload`
- `GET /api/models/current`
- `GET /api/models/metadata`
- `GET /api/capabilities`
- `POST /v1/chat/completions`
- `POST /v1/completions`

During early phases, generation endpoints must return typed unsupported/runtime errors rather than fake output; once a Camelid-supported dense GGUF model and tokenizer are loaded, the narrow autoregressive path may return non-streaming JSON or OpenAI-compatible SSE chunks.

## First vertical slice

The first implementation slice is intentionally narrow:

1. Rust binary starts.
2. Health endpoint responds.
3. API accepts a GGUF path.
4. GGUF metadata and tensor descriptors are parsed.
5. Model metadata is returned over HTTP.
6. Completion endpoints validate requests and, when a Camelid-supported dense GGUF model/tokenizer are loaded, return honest non-streaming or SSE-streamed output.

This gives local OpenAI-compatible clients something real to probe while preserving honesty about unsupported runtime states.

## Model binding lane

Camelid dense GGUF support now has a typed `model` module that extracts required architecture metadata (`llama.context_length`, embedding width, layer count, feed-forward width, attention heads, KV heads, RoPE settings, and RMSNorm epsilon) and binds the descriptor-level tensor set expected by supported dense checkpoints. Binding covers token embeddings, final norm/output tensors, and the per-layer attention/FFN tensors. If `output.weight` is absent, the binding treats logits as tied to `token_embd.weight`, matching common GGUF exports.

This lane has progressed beyond descriptor-only binding: loaded CPU f32 weights, KV-cache planning, RoPE, causal attention over prior/current KV positions, prompt-prefill next-token generation, public greedy/sampled autoregressive loops, and SSE streaming now exist. The next architectural step is to validate logits/output against exact model targets, then add only the consumer-driven sampling/API behavior that local OpenAI-compatible clients actually need without pretending broad runtime parity is ready.
