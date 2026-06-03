# GPU-Resident Decode Forward Pass

> Goal: run the entire per-token decode forward on the Metal GPU with the hidden state
> resident in GPU buffers and **one command buffer per token**, so decode throughput beats
> the CPU path on Apple Silicon. **Achieved** — see Status and Results below.

## Why

The earlier `CAMELID_METAL_Q8` path was stuck at ~7 tok/s: the Q8 GEMV dispatches were fast,
but the non-matmul ops (norm, RoPE, attention, SiLU, residual) ran on the CPU, forcing a
command-buffer commit + wait after **every** matmul. The per-dispatch round trip dominated.
The fix is all-or-nothing: keep activations on the GPU across the whole token so there is
exactly **one** commit/wait per token, with weights and the KV cache resident across tokens.

## Building blocks (done — PRs #193–#196)

All parity-checked on-GPU vs the CPU reference:

| Op | Kernel | try_ fn |
| --- | --- | --- |
| Q8 GEMV | `q8_0_block_linear_row(_simd)` | `try_q8_0_block_linear_row` |
| Activation quantize | `quantize_q8_0_f32` | `try_quantize_q8_0_f32` |
| RMSNorm | `rms_norm_f32` | `try_rms_norm_f32` |
| RoPE | `rope_rotate_f32` | `try_rope_rotate_f32` |
| Attention (decode) | `attention_decode_f32` | `try_attention_decode_f32` |
| SiLU·mul | `silu_mul_f32` | `try_silu_mul_f32` |
| Residual add | `residual_add_f32` | `try_residual_add_f32` |

## Integration — done

### 1. Resident-buffer encode API — done (#198, #199, #200)
`encode_<op>` helpers bind GPU buffers and encode a dispatch into a caller-provided command
buffer (no commit, no readback). The attention and FFN op-chains are composed in
`encode_attention_block` / `encode_ffn_block`; `try_decode_layer_resident` fuses both for one
layer in a single command buffer.

### 2. All-layers forward in one command buffer — done (#201)
`try_decode_forward_resident` chains every layer's attention + FFN block back-to-back with the
hidden state ping-ponging between two GPU buffers, so a whole token costs one commit/wait.

### 3. Resident weights — done (#202)
Q8_0 weight blocks are resolved through `MetalLinearCache` keyed by `(pointer, len)`: uploaded
once on the first decode and reused on every subsequent token.

### 4. On-GPU KV cache — done (#203, #204, #206)
`attention_decode_f32` reads the cache with explicit strides (`position_stride`,
`kv_head_stride`, `kv_base_offset`), so it can address a per-layer slice directly. The
`ResidentDecodeState` session owns persistent per-layer K/V buffers (laid out
`[kv_head][max_positions][head_dim]`) and the reused hidden buffers; each token blits only its
new K/V slot in. The cache starts sized to the prompt + a chunk and **grows on demand** toward
the context-length cap via a GPU→GPU blit (`ensure_capacity`) — sizing it to the full (often
128K) context up front would allocate tens of GB and thrash unified memory.

### 5. Wire into the decode path + parity gate — done (#205)
`forward_single_token_timed_internal` routes through the session when
`CAMELID_METAL_RESIDENT_DECODE` is set and the model is eligible (dense Q8_0, not
distributed-sharded, default GQA/scale/gate-up/forward-RoPE). The session is seeded from the
CPU KV cache after the batched prefill, then appends decode tokens; the existing final norm +
output projection are reused. `rope::resident_decode_rope_tables` builds the per-position
cos/sin tables, reusing the CPU frequency math (incl. llama3 scaling). Default off; ineligible
configs fall back to the unchanged CPU layer loop.

### 6. Throughput — done (#206, #207)
- Resident matmuls use the SIMD-group Q8 GEMV (`q8_0_block_simd_pipeline`).
- `attention_decode_f32` runs one threadgroup (a 32-lane SIMD group) per head: lanes split
  positions for the score/softmax reductions (`simd_max`/`simd_sum`) and split output dims for
  the weighted-value sum, instead of one thread per head.

## Results

Llama-3.2-3B-Instruct-Q8_0, M4, greedy (temperature 0), **byte-for-byte identical token stream
vs the CPU path**:

| Context | Resident GPU | CPU path |
| --- | --- | --- |
| 32-token decode | **18.8 tok/s** | 7.1 tok/s |
| 600-token decode | **15.9 tok/s** | ~7 tok/s |

~2.2–2.65× the CPU path across short and long contexts.

## Lessons

- The bottleneck was never the Q8 matmul kernel. The two real costs were (1) sizing the KV
  cache to the full context length (tens of GB → unified-memory thrash, ~0.1 tok/s) and
  (2) the one-thread-per-head attention kernel (scaled badly with context length). Fixing both
  unlocked the win.
- Numerical parity across the configurable CPU paths (gate/up order, attention scale mode, GQA
  mapping, RoPE pairing/scaling) held by driving the GPU path from the same resolved config and
  gating eligibility on the defaults the kernels implement.

## Status

**Achieved and shipped**, gated behind `CAMELID_METAL_RESIDENT_DECODE` (default off).

## Possible follow-ups

- Fold RMSNorm/quantize/residual into fewer dispatches to cut per-token dispatch count.
- Tune threadgroup sizing (RMSNorm, attention) and try larger SIMD-group counts per head.
- Validate additional rows (TinyLlama 1.1B, Llama-3.2 1B, 13B Q8) and longer contexts.
- Distributed minis: each node runs the resident GPU decode for its layer range.
