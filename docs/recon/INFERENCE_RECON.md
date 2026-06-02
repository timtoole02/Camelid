# Inference Engine Reverse Engineering

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Source: inference architecture recon focused on LLaMA-family model structure, GGUF naming conventions, KV-cache behavior, and decoder-only graph execution.

## Summary

Phase 5 should start with a narrow, explicit LLaMA-family dense decoder-only path. The goal is a correct CPU reference implementation before quantized/performance paths. camelid currently has GGUF metadata parsing and an HTTP skeleton; it needs model config extraction, tensor payload loading, tensor runtime primitives, KV cache, model binding, and inference session orchestration before generation can work.

## Initial supported model scope

Support initially:

- `general.architecture == "llama"`
- dense decoder-only LLaMA-family models
- causal attention
- RMSNorm
- RoPE with `rope.dimension_count == attention.key_length`
- separate Q/K/V tensors or fused `attn_qkv.weight`
- tied or explicit output head
- CPU only
- f32/f16/BF16 tensors converted to f32

Reject initially:

- MoE (`llama.expert_count > 0`)
- LLaMA 4
- sliding-window/SWA
- YaRN/longrope variants
- LoRA/adapters
- multimodal models
- embedding-only/non-causal variants
- quantized tensors unless dequantized eagerly to f32 in an earlier tensor phase

## Config extraction keys

Required:

```text
general.architecture
llama.context_length
llama.embedding_length
llama.block_count
llama.feed_forward_length
llama.attention.head_count
llama.attention.layer_norm_rms_epsilon
```

Optional/defaulted:

```text
general.name
llama.attention.head_count_kv       default = head_count
llama.attention.key_length          default = embedding_length / head_count
llama.attention.value_length        default = embedding_length / head_count
llama.rope.dimension_count          default = key_length
llama.rope.freq_base                default = 10000.0
llama.rope.scaling.type             reject unless absent/linear factor 1 initially
llama.rope.scaling.factor           default = 1.0; reject != 1.0 initially
llama.vocab_size                    or infer from token_embd/output tensor
```

Derived values:

```text
n_embd = embedding_length
n_layer = block_count
n_head = attention.head_count
n_head_kv = attention.head_count_kv.unwrap_or(n_head)
head_dim_k = attention.key_length.unwrap_or(n_embd / n_head)
head_dim_v = attention.value_length.unwrap_or(n_embd / n_head)
n_embd_k_gqa = head_dim_k * n_head_kv
n_embd_v_gqa = head_dim_v * n_head_kv
n_ff = feed_forward_length
n_rot = rope.dimension_count.unwrap_or(head_dim_k)
```

Validation:

- `n_embd % n_head == 0`
- `n_head % n_head_kv == 0`
- `n_rot == head_dim_k` for first target
- `head_dim_k == head_dim_v` for first target
- `attention.causal` absent or true
- reject recurrent/encoder/multimodal indicators

## Required dense LLaMA tensor names

Global tensors:

```text
token_embd.weight        shape [n_embd, n_vocab]
output_norm.weight       shape [n_embd]
output.weight            GGUF dimensions [n_embd, n_vocab], optional
```

`output.weight` is the final vocab projection. GGUF descriptor dimensions for TinyLlama-style untied output weights are `[n_embd, n_vocab]`, but the payload is interpreted as token-major rows for logits: each token row contains `n_embd` hidden components. Camelid must validate that the tensor has exactly `n_embd * n_vocab` values and route runtime logits through the token-major output-projection path; treating it as descriptor `[hidden, vocab]` silently points the final logits at the wrong vocab direction. If `output.weight` is absent, tie output projection to `token_embd.weight`, which is already token-major `[vocab, n_embd]` for embedding lookup/logit rows.

Portability guardrail: this is a model-file storage contract, not an Apple Silicon behavior. Windows, Linux, Intel, ARM, SIMD, mmap, and kernel-specific implementations must preserve the same token-row interpretation. For TinyLlama Q8_0, the `output.weight` guardrail expects GGUF dims `[2048, 32000]`, storage rows of `2048` values / `2176` bytes, logical token rows with `stride=1`, and descriptor-column samples only as contrast evidence with `stride=32000`. Run `scripts/check-output-projection-layout.mjs` against a `tensor-dump` artifact before trusting any platform-specific tensor-layout work.

Per layer `i`:

```text
blk.{i}.attn_norm.weight
blk.{i}.attn_q.weight
blk.{i}.attn_k.weight
blk.{i}.attn_v.weight
blk.{i}.attn_output.weight
blk.{i}.ffn_norm.weight
blk.{i}.ffn_gate.weight
blk.{i}.ffn_up.weight
blk.{i}.ffn_down.weight
```

Optional per layer:

```text
blk.{i}.attn_qkv.weight
blk.{i}.attn_q.bias
blk.{i}.attn_k.bias
blk.{i}.attn_v.bias
blk.{i}.attn_output.bias
blk.{i}.ffn_gate.bias
blk.{i}.ffn_up.bias
blk.{i}.ffn_down.bias
blk.{i}.rope_freqs.weight
```

Shape expectations:

```text
attn_q.weight       [n_embd, head_dim_k * n_head]
attn_k.weight       [n_embd, head_dim_k * n_head_kv]
attn_v.weight       [n_embd, head_dim_v * n_head_kv]
attn_output.weight  [head_dim_k * n_head, n_embd]
ffn_gate.weight     [n_embd, n_ff]
ffn_up.weight       [n_embd, n_ff]
ffn_down.weight     [n_ff, n_embd]
```

GGUF stores dimensions in GGML order; centralize shape matching and do not scatter transposition assumptions.

## Forward-pass execution order

For token IDs and positions:

1. token embedding lookup
2. for each layer:
   1. save residual
   2. RMSNorm with `attn_norm.weight`
   3. compute Q/K/V via projections or split fused QKV
   4. reshape Q to `[tokens, n_head, head_dim]`
   5. reshape K/V to `[tokens, n_head_kv, head_dim]`
   6. apply RoPE to Q/K
   7. write K/V to KV cache
   8. read cached K/V through current position
   9. grouped-query attention: map query heads to KV heads
   10. scores = `Q @ K^T * 1/sqrt(head_dim)`
   11. apply causal mask
   12. softmax
   13. context = weights @ V
   14. output projection
   15. residual add
   16. RMSNorm with `ffn_norm.weight`
   17. gated MLP: `SiLU(x @ ffn_gate) * (x @ ffn_up)` then `@ ffn_down`
   18. residual add
3. final RMSNorm with `output_norm.weight`
4. logits = final hidden dot token-major rows from `output.weight` or tied `token_embd.weight`
5. return logits for requested output positions

## Prefill vs decode

Common decoder-only runtimes use the same graph shape with different batch sizes.

For camelid first version:

```rust
struct InferenceSession { /* model, runtime, kv_cache, position */ }

impl InferenceSession {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>; // logits for last token
    fn decode_one(&mut self, token: u32) -> Result<Vec<f32>>;
    fn reset(&mut self);
}
```

Scope constraints:

- single sequence only
- `seq_id = 0`
- positions are contiguous `0..n`
- no cache compaction/defrag
- no batching across users

## KV cache

Minimal model:

```rust
struct KvCache<Tensor> {
    keys: Vec<Tensor>,   // one per layer, shape [n_ctx, n_head_kv, head_dim]
    values: Vec<Tensor>, // one per layer, shape [n_ctx, n_head_kv, head_dim]
    used: usize,
}
```

Required behavior:

- allocate at session creation
- bounds-check against context length
- write K/V for each token position
- read `0..=current_pos` for causal attention
- reset/clear session
- reject context overflow cleanly

## CPU f32 runtime primitives needed

Tensor/data:

- owned dense f32 tensor
- shape/stride metadata
- matmul
- vector/matrix add
- elementwise multiply
- SiLU
- RMSNorm
- stable softmax
- RoPE
- embedding lookup
- reshape/view helpers
- logits extraction

Numerics:

- f32 accumulators
- stable softmax with max subtraction
- RMSNorm: `x * weight / sqrt(mean(x^2) + eps)`

Tensor loading:

- load F32 directly
- load F16/BF16 and convert to f32
- initially reject quantized tensors, or only accept them after tensor phase implements eager dequantization

## Errors to add

Add explicit typed errors beyond `InferenceNotImplemented`:

```rust
UnsupportedArchitecture(String)
UnsupportedModelFeature(String)
MissingModelMetadata(String)
InvalidModelConfig(String)
MissingTensor(String)
InvalidTensorShape { name: String, expected: String, actual: Vec<u64> }
ContextLengthExceeded { requested: usize, max: usize }
Runtime(String)
```

## Phase 5 implementation tasks

1. Add `src/model/mod.rs` and `src/model/llama.rs`.
2. Add `LlamaConfig::from_gguf(&GgufFile)`.
3. Add model feature validation and explicit unsupported errors.
4. Add tensor name lookup helpers in GGUF/tensor index.
5. Add `src/tensor/` CPU f32 tensor type and primitives.
6. Add tensor loading for F32/F16/BF16 into f32.
7. Add model weight binding for dense LLaMA tensors.
8. Add `KvCache`.
9. Add `InferenceSession::prefill` and `decode_one`.
10. Add generation loop only after logits are correct.

## Tests

Config/model tests:

- valid minimal LLaMA config parses
- missing required metadata fails clearly
- MoE metadata rejects
- invalid head divisibility rejects
- rope scaling factor != 1 rejects initially
- missing required tensor fails clearly
- tied output accepted when `output.weight` is absent

Runtime primitive tests:

- RMSNorm against hand-computed values
- SiLU against hand-computed values
- softmax stable with large values
- RoPE shape/position smoke tests
- attention single-token causal behavior

Inference tests:

- tiny synthetic model with one layer and tiny vocab
- prefill updates KV cache and returns logits
- decode_one advances position
- context overflow returns typed error

## First inference milestone

The first real inference milestone is not full chat generation. It is:

1. Load tiny supported dense LLaMA-like fixture.
2. Bind config and tensors.
3. Run `prefill([token])`.
4. Return logits vector with correct vocab length.
5. Add deterministic test for the logits shape and a hand-computable tiny output.
