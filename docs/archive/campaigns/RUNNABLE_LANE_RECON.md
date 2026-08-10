# RUNNABLE_LANE_RECON.md — Phase 0 (Recon, no code)

Grounded in the actual `src/` tree at HEAD `e1408877`. Every claim cites `file:line`.
No build started. This doc satisfies **Gate 0**.

> Scope reminder (from spec): the runnable lane is a generic, f32-only, breadth-first
> path whose correctness becomes the promotion oracle for the supported lane. Phase 0
> is recon only — decide what to reuse vs. build, propose module layout, stop.

---

## 0. TL;DR reuse/build call

| Component | Verdict | One-line reason |
|---|---|---|
| GGUF parser | **REUSE + thin admission wrapper** | `src/gguf/reader.rs` already parses header/KV/tensor-index with strong structural validation; it lacks only a *covered-set admission* layer. |
| Dequant → f32 | **REUSE 6 of 7, build 1** | F32/F16/Q8_0/Q6_K/Q5_K/Q4_K already dequant to f32; only Q4_0 needs a runnable-lane wiring (eager decoder exists, lazy-wire path rejects it). |
| Tokenizer | **REUSE for SPM + BPE; add HF anchor** | `src/tokenizer/mod.rs` implements LlamaSpm + Gpt2Bpe, decoupled & tested — but anchored to **llama.cpp**, not **HF tokenizers** as the spec's Phase 3 demands. |
| Decoder block | **BUILD fresh f32 graph, reuse leaf ops** | The existing block (`forward_layer_timed`, `inference.rs:5255`) is a 540-line procedural function fused with Q8_0/GPU dispatch; reuse its leaf ops (RoPE, RMSNorm, GQA attn, SwiGLU, f32 matmul) but assemble a clean parametric f32 block. |

**Single biggest finding:** the codebase's parity oracle today is **llama.cpp**, but the
spec mandates **HF transformers / HF tokenizers** as the external anchor (Phases 3 & 5).
That reference harness **does not exist yet** and is the critical-path ask (see §7).

---

## 1. GGUF parsing — what exists & what it exposes

**Entry point:** `pub fn read_metadata(path) -> Result<GgufFile>` — `src/gguf/mod.rs:9`.
Public re-exports: `GgufFile`, `GgufMetadataValue`, `GgufTensorDescriptor`, `GgufTensorType` (`src/gguf/mod.rs:3`).

### Parses today (`src/gguf/reader.rs`)
- Magic `GGUF` + version (v2/v3 only) — `reader.rs:317-329`.
- Tensor & metadata counts — `reader.rs:331-337`.
- KV metadata, all scalar types + String + (non-nested) Array — enum `GgufMetadataValue` `reader.rs:19-33`, reader `reader.rs:456-500`; duplicate-key rejection `reader.rs:343-347`.
- `general.alignment` (default 32, power-of-two enforced) — `reader.rs:350-364`.
- Tensor index per tensor: name, dims (1–4), type id, relative+absolute offset, n_bytes — `reader.rs:366-442`; contiguity check `reader.rs:411-413`; data-beyond-EOF check `reader.rs:423-426`.

### Quant type enum (verified)
`GgufTensorType` `reader.rs:36-58`, `from_id()` `reader.rs:62-87`, `layout()` (block_size, type_size) `reader.rs:89-112`. Recognizes F32, F16, Q4_0/1, Q5_0/1, Q8_0/1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, IQ4NL, I8/16/32/64, F64, BF16, and `Unknown(i32)` for anything else. Unknown → `layout()` returns `None` → rejected at size calc (`reader.rs:503-506`).

### Metadata accessors exposed
`architecture()`, `model_name()`, `metadata_string/bool/u32/f32`, and typed array getters (`metadata_array_strings[_optional]`, `_f32_optional`, `_u32_optional`, `_bools_optional`, `_i32_optional`) — `reader.rs:137-301`. These cover every KV the parametric block + tokenizer need (head counts, rope base/scaling, tokenizer arrays).

### Validation that already exists
`error.rs:12-16` distinguishes `InvalidGguf` (structural) from `UnsupportedGguf` (recognized-but-rejected). A separate **architecture allowlist** lives in `model.rs:52-54` (`llama, mistral, qwen2, qwen3, smollm3, gemma3, gemma4, phi3, lfm2`) with explicit refusals for `gemma4-assistant` and anything containing `diffusion` (`model.rs:63-97`).

### Gap → admission gate (Phase 1 work)
The parser does **not** evaluate the spec's three-axis covered-set as a single gate. Quant rejection is *implicit* (only when `layout()` is `None`), and there is no machine-readable refusal naming `{axis, offending value, tensor}` per the spec's principle #2. The architecture allowlist exists but is not aligned to the v1 set (`{llama, qwen2, qwen3, gemma2, gemma3, phi3}` — note **gemma2** is in spec but not in `model.rs`'s list, and `mistral/smollm3/lfm2/gemma4` are in the code list but not the spec set). **Build:** a thin `runnable::admit(&GgufFile) -> Result<AdmissionOk, AdmissionReject>` wrapping the existing parser. Do **not** rewrite `reader.rs`.

---

## 2. Dequant → f32 — what exists

Two families coexist:
- **Eager block decoders** in `src/tensor/mod.rs` (materialize `Vec<f32>`).
- **Lazy wire dequant** in `src/tensor/wire_dequant.rs` (file-backed, on-demand), gated to formats with committed parity evidence.

### Coverage vs. the v1 quant set `{F32, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0}`

| Format | Eager (tensor/mod.rs) | Lazy wire | Bit-exact test | Verdict |
|---|---|---|---|---|
| F32 | passthrough | `wire_dequant.rs:158-162` | `wire_dequant.rs:373-381` | REUSE |
| F16 | `f16_bits_to_f32` `mod.rs:3564` | **not in lazy set** | (subnormal-correct conv) | REUSE eager |
| Q8_0 | `decode_q8_0_blocks` `mod.rs:3481-3510` (verified: f16 scale × 32×i8, 34-byte block) | `wire_dequant.rs:163-174` | `wire_dequant.rs:274-297` | REUSE (validated anchor) |
| Q6_K | `Q6KBlock::dequantize` `mod.rs:4120-4147` | `wire_dequant.rs:202-212` | `wire_dequant.rs:349-371` | REUSE |
| Q5_K_M | `Q5KBlock::dequantize` `mod.rs:4044-4074` | **not in lazy set** | (eager decode path) | REUSE eager |
| Q4_K_M | `Q4KBlock::dequantize` `mod.rs:3984-4005` | `wire_dequant.rs:190-200` | `wire_dequant.rs:322-347` | REUSE |
| **Q4_0** | `decode_q4_0_blocks` `mod.rs:4213` | **rejected** `wire_dequant.rs:47-51` | — | **BUILD wiring** |

Verified rejection text in `wire_dequant.rs:47-51`: lazy wire supports only `F32, Q8_0, Q5_0, Q4_K, Q6_K` ("the formats with committed dequant-parity evidence"). So F16, Q5_K, and Q4_0 must go through the **eager** path (which does support them) or get added to the lazy set.

### Existing test posture
All dequant tests are inline in `wire_dequant.rs:219-406`, comparing lazy reads against the eager block decoders (self-consistency), **not** against an external ggml reference dump. The spec's Gate 2 demands bit-exact vs. **ggml reference fixtures** under `tests/fixtures/dequant/` — that fixture set **does not exist yet** (no such dir). **Build:** the Python `gguf`/`llama-cpp` reference dumper + fixtures (Phase 2 ask).

---

## 3. Tokenizer — what exists

`src/tokenizer/mod.rs` (1276 lines), exported standalone (`lib.rs:27`), zero coupling to inference/model graph — proven by `tests/tokenizer.rs` exercising it on bare GGUF fixtures.

- **Algorithms:** `TokenizerModel { LlamaSpm, Gpt2Bpe }` (verified `mod.rs:10-13`). SPM = score-based merge + byte fallback; BPE = rank/merge via BinaryHeap, byte-level, two pre-tokenizer dialects `BpePreTokenizer { Llama3, Qwen2 }` `mod.rs:28-37`.
- **Driven from GGUF KV:** `Tokenizer::from_gguf` `mod.rs:212-393` reads `tokenizer.ggml.{model,pre,tokens,scores,token_type,merges}` + special-token ids + add_bos/eos flags + chat_template.
- **Encode + decode both present**, byte-fallback handled both directions (`mod.rs:404-466`, `813-830`, `1138-1155`).
- **Validation today:** parity vs **llama.cpp** (`tests/tokenizer.rs`, `tests/dg_tokenizer_parity.rs`) — 100% token-id match on Llama3/Mistral/Mixtral/DG fixtures.

### Gaps vs spec Phase 3
1. Reference is **llama.cpp**, not **HF `tokenizers`** — spec Gate 3 names HF explicitly. Either add an HF anchor or get the spec's reference relaxed to llama.cpp (decision/ask, §7).
2. `gemma2`/early-gemma SPM not wired (only `gemma4` routes to LlamaSpm, `mod.rs:221`). Spec covered-set includes `gemma2` → needs verification that its SPM is byte-identical to the existing LlamaSpm path.

---

## 4. Forward path & GPU coupling

- **Entry:** `forward_single_token()` `inference.rs:3060`; generation loop `generate_next_token()` `inference.rs:3831`; logits tail `forward_final_norm_and_logits()` `inference.rs:3181`.
- **Pure CPU f32 path EXISTS and is independent of GPU.** GPU is an *optional* dispatch: `try_resident_decode_forward()` returns `None` when ineligible, then the CPU layer loop runs unconditionally (`inference.rs:3641-3665`). CUDA is `#[cfg(feature="cuda")]` but the CPU loop is not gated. `CAMELID_DETERMINISTIC=1` pins CPU (`inference.rs:4137`). f32 GEMM: `matmul` / `matmul_rhs_transposed` `tensor/mod.rs:1283,1382`.
- **One parametric block** (`forward_layer_timed` `inference.rs:5255`, ~540 lines) handles llama/mistral/qwen2/qwen3/gemma3/phi3/smollm3; gemma4 has a separate MoE binding (`model.rs:1043`).
- **Leaf ops available to reuse:** RoPE `apply_rope` (`rope.rs`, supports none/linear/llama3-yarn, both pair styles `rope.rs:62-70,312-384`); per-head QK-norm `per_head_rms_norm` `tensor/mod.rs:1738`; GQA attention `causal_attention_context` `inference.rs:19058`; SwiGLU `gated_ffn_activation_with_plan` `inference.rs:8178` (falls through to f32); RMSNorm.
- **Switches already present:** GQA via head_count/head_count_kv ✅; QK-norm (qwen3) ✅ (`inference.rs:5390-5406`); RoPE styles/scaling ✅; SwiGLU ✅.
- **Switches MISSING (spec requires for full v1):** qwen2 attention **bias** ❌; gemma2 **logit soft-capping** ❌; gemma `(weight+1)` RMSNorm + `sqrt(d_model)` embedding scale ❌; phi3 **fused QKV** ❌; LayerNorm (vs RMSNorm) ❌. These are the Phase 6 build items, each justified by a real structural difference.

### Reuse/build call
The block is **fused with Q8_0/GPU fast-path dispatch** (`linear_for_role_runtime_with_plan` `inference.rs:6570`), so it is not cleanly a "f32-only parametric block." **Build** a fresh `runnable` decoder block that calls the existing *leaf* ops directly (RoPE, RMSNorm, GQA attn, SwiGLU, `matmul_rhs_transposed`) — this also keeps the runnable oracle free of the optimization dispatch it's meant to validate. Estimated ~250 lines stripped of dispatch.

---

## 5. Proposed module layout

New crate-internal module, isolated from the optimized lane:

```
src/runnable/
  mod.rs            // pub fn run(gguf, prompt, opts) -> RunnableReceipt; lane wiring
  admit.rs          // three-axis covered-set gate over GgufFile; structured AdmissionReject
  config.rs         // ArchConfig parsed from GGUF KV (heads, rope, norm kind, ffn, capping, bias)
  dequant.rs        // adapters over tensor/mod.rs decoders -> uniform f32 tensor provider
  block.rs          // parametric pre-norm f32 decoder block (reuses rope/attn/ffn leaf ops)
  graph.rs          // embeddings -> N blocks -> final norm -> logits, f32 only, greedy
  receipt.rs        // lane="runnable" receipt (reuse receipt::LaneIdentity, never copper)
tests/runnable/      // admission, dequant-vs-ggml, decoder determinism, HF parity fixtures
tests/fixtures/dequant/   // NEW: ggml reference block dumps (Phase 2)
tests/fixtures/runnable_parity/  // NEW: HF transformers logits/token fixtures (Phase 5)
```

Reuses without modification: `gguf::*`, `tensor::{decode_*, f16_bits_to_f32, matmul*}`, `tokenizer::*`, `inference::rope`, `receipt::LaneIdentity` (`receipt/mod.rs:98`). Touches `lib.rs` (one `pub mod runnable;`).

---

## 6. Working-environment note

This dev box is **Windows 11 + RTX 3060 Laptop**, not the Mac mini/USB setup in the spec's "working-environment guards" section. Models are local at `<home>\Camelid\models\` (TinyLlama-1.1B Q8_0, Llama-3.2-1B/3B Q8_0, Qwen3 0.6/1.7/4/8B Q8_0, DiffusionGemma Q4_K_M — gives F32/Q8_0 and a Q4_K_M sample on disk). The Mac/USB mount guards are **N/A here**; runnable is f32 and memory-heavy, so watch RAM on the larger models. No external-mount preflight needed on this box.

---

## 7. Asks (→ BACKEND_ASKS.md)

1. **HF reference harness does not exist.** Current parity oracle is llama.cpp (tokenizer + receipts). Spec Phases 3 & 5 mandate **HF `tokenizers`** and **HF `transformers`** as the external anchor. Need: decision to either (a) stand up an HF reference harness (Python, pinned versions, fixture dumps) or (b) amend the spec to accept llama.cpp as the anchor. **Critical path for oracle status.**
2. **No ggml dequant reference fixtures** under `tests/fixtures/dequant/` (Gate 2 requirement). Need the Python `gguf`/`llama-cpp` dumper + checked-in block fixtures.
3. **Tolerances undecided:** F16→f32 ULP bound (Gate 2) and logit max-abs-diff threshold (Gate 5, greedy token-sequence is the hard gate but logit diff is reported evidence).
4. **Covered-set vs code allowlist mismatch:** spec v1 archs = `{llama, qwen2, qwen3, gemma2, gemma3, phi3}`; `model.rs:52-54` allows `{llama, mistral, qwen2, qwen3, smollm3, gemma3, gemma4, phi3, lfm2}` and omits `gemma2`. Confirm the runnable covered-set is authoritative (admission gate will key off the spec set, not `model.rs`).

---

## Gate 0 — self-assessment

| Criterion | Status | Evidence |
|---|---|---|
| Recon grounded in real source (file:line) | ✅ | §1–§4, all anchors verified by direct read |
| No build started | ✅ | only `RUNNABLE_LANE_RECON.md` + `BACKEND_ASKS.md` written |
| Reuse/build decisions justified | ✅ | §0 table + per-component verdicts |

**STOP — awaiting human go for Phase 1.**
