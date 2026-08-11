# Qwen3-1.7B Q8_0 — Architecture Recon (Gate 0)

> [!NOTE]
> This is a recon/design note for the Qwen3 support lane, **not** the public
> support ledger. For current support truth use
> [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../reference/STATUS.md).
> Qwen3 is **not** a supported row until Gate 4 evidence lands.

Target row: **Qwen3-1.7B Instruct, Q8_0** — the smallest dense Qwen3 with the
fastest correctness loop. Neighboring rows (0.6B/4B/8B/14B/32B, base variants,
MoE A3B, other quants, thinking-mode) are explicitly out of scope and unsupported
until separately proven.

## Source artifacts

- GGUF: `Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` (HuggingFace, official Qwen).
  Local path: `/Volumes/Untitled/models/Qwen3-1.7B-Q8_0.gguf`
  - size 1,834,426,016 bytes; GGUF version 3; 310 tensors; 28 metadata keys;
    alignment 32; data_start_offset 5,951,136.
- Oracle: pinned llama.cpp `1 (5d56eff)` (built 2025-04-28, recognizes `qwen3`
  arch with `n_embd_head_k=128`), from the local pinned llama.cpp `build/bin/llama-server`
  (needs `DYLD_LIBRARY_PATH` set to that bin dir when the build tree was renamed,
  so @rpath resolves). Path kept out of the public tree per the repo scrub.

## The trap (why this is NOT already supported)

`qwen3` is already in the `general.architecture` allowlist (`src/model.rs`
`from_gguf`, `src/api/mod.rs` `model_family` → `llama-family`). But the dense
binder `LlamaTensorBinding::bind` never binds `attn_q_norm` / `attn_k_norm`.
Qwen3 (unlike Qwen2) applies a **per-head RMSNorm** to Q and K *after* the q/k
projections and *before* RoPE. Loading a Qwen3 GGUF through the current path runs
but silently **drops** the QK-norm weights → logits diverge from llama.cpp. This
is exactly the silent-correctness failure the evidence gates exist to catch.

## Full metadata (`camelid inspect`)

| key | value |
| --- | --- |
| general.architecture | `qwen3` |
| general.basename | `Qwen3` |
| general.file_type | 7 (MOSTLY_Q8_0) |
| general.finetune | `Instruct` |
| general.name | `Qwen3 1.7B Instruct` |
| general.quantization_version | 2 |
| general.size_label | `1.7B` |
| general.type | `model` |
| qwen3.attention.head_count | **16** |
| qwen3.attention.head_count_kv | **8** (GQA, 2:1) |
| qwen3.attention.key_length | **128** (= head_dim) |
| qwen3.attention.value_length | **128** |
| qwen3.attention.layer_norm_rms_epsilon | **1e-06** |
| qwen3.block_count | **28** |
| qwen3.context_length | 40960 |
| qwen3.embedding_length | **2048** |
| qwen3.feed_forward_length | **6144** |
| qwen3.rope.freq_base | **1000000.0** |
| tokenizer.ggml.model | `gpt2` (BPE) |
| tokenizer.ggml.pre | `qwen2` |
| tokenizer.ggml.add_bos_token | **false** |
| tokenizer.ggml.bos_token_id | 151643 `<|endoftext|>` |
| tokenizer.ggml.eos_token_id | 151645 `<|im_end|>` |
| tokenizer.ggml.padding_token_id | 151643 |
| tokenizer.ggml.tokens | array len 151936 |
| tokenizer.ggml.merges | array len 151387 |
| tokenizer.chat_template | ChatML (jinja, len 4100; `<|im_start|>`/`<|im_end|>`, supports `<think>` + `enable_thinking`) |

### Derived / verified shape facts

- **head_dim = key_length = value_length = 128.** For 1.7B this coincides with
  `embedding_length / head_count = 2048 / 16 = 128`, but the engine must read it
  from `qwen3.attention.key_length` because it does **not** hold for all Qwen3
  sizes (e.g. 0.6B is 1024/16=64 ≠ 128). Do not derive head_dim from embed/heads.
- Q projection out = `head_count * head_dim = 16 * 128 = 2048`.
- K/V projection out = `head_count_kv * head_dim = 8 * 128 = 1024`.
- **Tied embeddings: YES.** No `output.weight` tensor; `token_embd.weight`
  (Q8_0, [2048, 151936]) is reused for the output projection.

## QK-norm tensors (the whole point)

Confirmed present for every block (`blk.0` shown; all 28 blocks identical shape):

| tensor | dtype | dims |
| --- | --- | --- |
| `blk.0.attn_q_norm.weight` | **F32** | **[128]** (= head_dim) |
| `blk.0.attn_k_norm.weight` | **F32** | **[128]** (= head_dim) |

Per-block tensor inventory (Q8_0 weights, F32 norms):

| tensor | dtype | dims |
| --- | --- | --- |
| `blk.N.attn_norm.weight` | F32 | [2048] |
| `blk.N.attn_q.weight` | Q8_0 | [2048, 2048] |
| `blk.N.attn_q_norm.weight` | F32 | [128] |
| `blk.N.attn_k.weight` | Q8_0 | [2048, 1024] |
| `blk.N.attn_k_norm.weight` | F32 | [128] |
| `blk.N.attn_v.weight` | Q8_0 | [2048, 1024] |
| `blk.N.attn_output.weight` | Q8_0 | [2048, 2048] |
| `blk.N.ffn_norm.weight` | F32 | [2048] |
| `blk.N.ffn_gate.weight` | Q8_0 | [2048, 6144] |
| `blk.N.ffn_up.weight` | Q8_0 | [2048, 6144] |
| `blk.N.ffn_down.weight` | Q8_0 | [6144, 2048] |

Non-block: `token_embd.weight` (Q8_0 [2048, 151936]), `output_norm.weight`
(F32 [2048]). No `output.weight` (tied).

`n_embd_head_k=128`, `n_embd_k_gqa=1024` confirmed by llama.cpp's own load print.

### Forward-path order (to be matched in Gate 2)

Per the Qwen3 graph: `x → attn_norm → q_proj/k_proj/v_proj → reshape to heads →`
**`q_norm(per-head RMSNorm over head_dim) / k_norm(per-head RMSNorm over head_dim)`**
`→ RoPE → attention → attn_output`. The q_norm/k_norm RMSNorm is over the
head_dim axis (128), applied independently per head, using
`qwen3.attention.layer_norm_rms_epsilon = 1e-06`. This is to be confirmed against
the llama.cpp qwen3 build graph before claiming first-token parity.

## llama.cpp oracle — greedy reference (temperature 0, top_k 1)

Raw completion (no chat template), `cache_prompt:false`, deterministic. Captured
from the pinned `5d56eff` server. These token IDs are the parity bar for Gates 2–3.

Prompt A — `"The capital of France is"`
- prompt tokens (no special): `[785, 6722, 315, 9625, 374]` (5 tokens)
- 1-token: `[12095]` → `" Paris"`
- 5-token: `[12095, 13, 576, 6722, 315]` → `" Paris. The capital of"`

Prompt B — `"Q: What is 2+2? A:"`
- prompt tokens (no special): `[48, 25, 3555, 374, 220, 17, 10, 17, 30, 362, 25]` (11 tokens)
- 1-token: `[220]` → `" "`
- 5-token: `[220, 19, 13, 220, 17]` → `" 4. 2"`

Prompt C — `"Once upon a time"`
- prompt tokens (no special): `[12522, 5193, 264, 882]` (4 tokens)
- 1-token: `[11]` → `","`
- 5-token: `[11, 304, 264, 2613, 14126]` → `", in a small village"`

Raw JSON saved at `/tmp/qwen3_oracle.json` during recon (re-derivable from the
pinned server). The chat-template (ChatML, thinking-disabled) oracle for Gate 3
will be captured separately by `scripts/chat-parity-qwen3.mjs`.

## Gate 0 status: GREEN

All metadata keys recorded; `attn_q_norm`/`attn_k_norm` confirmed present at
[128] F32 for all 28 blocks; head_dim sourced explicitly from key_length; tied
embeddings confirmed; llama.cpp reference tokens captured for 3 fixed prompts at
1 and 5 tokens. Proceed to Gate 1 (config + binder QK-norm plumbing).
