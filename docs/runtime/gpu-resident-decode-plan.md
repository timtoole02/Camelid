# GPU-Resident Decode Forward Pass — Integration Plan

> Goal: run the entire per-token decode forward on the Metal GPU with the hidden state
> resident in GPU buffers and **one command buffer per token**, so decode throughput
> reaches GPU-class rates (target ≥ MLX/llama.cpp on Apple Silicon, ~28+ tok/s for
> Llama-3.2-3B Q8 on an M4). This is the last step of the GPU forward-pass effort.

## Why this is the only path to the win

Profiling (see `docs/benchmarks/camelid-vs-mlx.md`) showed the existing
`CAMELID_METAL_Q8` path is stuck at ~7 tok/s: the Q8 GEMV dispatches are fast, but the
non-matmul ops (norm, RoPE, attention, SiLU, residual) run on the CPU, forcing a
command-buffer commit + wait after **every** matmul. The per-dispatch round trip
dominates. The fix is all-or-nothing: keep activations on the GPU across the whole
token so there is exactly **one** commit/wait per token.

## Building blocks (done — PRs #193–#196)

All parity-checked on-GPU vs the CPU reference (13 metal tests):

| Op | Kernel | try_ fn |
| --- | --- | --- |
| Q8 GEMV | `q8_0_block_linear_row(_simd)` | `try_q8_0_block_linear_row` |
| Activation quantize | `quantize_q8_0_f32` | `try_quantize_q8_0_f32` |
| RMSNorm | `rms_norm_f32` | `try_rms_norm_f32` |
| RoPE | `rope_rotate_f32` | `try_rope_rotate_f32` |
| Attention (decode) | `attention_decode_f32` | `try_attention_decode_f32` |
| SiLU·mul | `silu_mul_f32` | `try_silu_mul_f32` |
| Residual add | `residual_add_f32` | `try_residual_add_f32` |

The standalone `try_*` fns each do CPU→GPU→CPU (one command buffer, full round trip).
They prove correctness; the integration needs **resident** variants that operate on
`&Buffer` and encode into a shared command buffer with no readback.

## Work remaining

### 1. Resident-buffer encode API
For each kernel, add an internal `encode_<op>(encoder, kernel, in_buf, out_buf, dims…)`
that binds GPU buffers and encodes a dispatch into a caller-provided command buffer —
no commit, no readback. Reuse the existing pipelines. Activations live in a small set
of reused `Buffer`s (hidden, norm, q/k/v, scores, gate/up/act, down) sized once per model.

### 2. KV cache on-GPU
Allocate per-layer K/V as Metal buffers (shared storage). The K/V projections write the
current token's slice directly into the cache buffer at `position`; `attention_decode_f32`
reads the cache buffer with the real per-position stride (generalize the kernel from the
contiguous test layout to the cache's `head_base_offset` + `position_stride`).

### 3. Per-token resident forward
Within one `start_inference_session()` command buffer, per layer:
`rms_norm → quantize → {q,k,v} matmul → rope(q,k) → write kv → attention → o matmul →
residual → rms_norm → quantize → {gate,up} matmul → silu_mul → quantize → down matmul →
residual`, then final `rms_norm → output matmul → logits`. Commit once; read only the
logits row back. Weights stay resident (already cached by pointer in `MetalLinearCache`).

### 4. Parity gate (before any speed claim)
Compare the GPU-resident decode token stream against the CPU path for the validated rows
(TinyLlama 1.1B, Llama-3.2 1B/3B Q8) at temperature 0 — identical greedy token ids, and
per-op max|Δ| ≈ 1e-3 on a captured trace. Gate behind `CAMELID_METAL_RESIDENT_DECODE`
(default off) until parity holds, exactly like the other Metal flags.

### 5. Measure
`bench-generate` decode tok/s + TTFT vs MLX-LM and llama.cpp on the same M4, same prompts,
temperature 0, warm. Report in `docs/benchmarks/camelid-vs-mlx.md`. Then consider the
distributed minis (each node runs the resident GPU decode for its layer range).

## Risks

- Numerical parity across the many configurable CPU paths (gate/up order, attention scale
  mode, GQA mapping, RoPE pairing/scaling). Mitigation: drive the GPU path from the same
  resolved plan/config the CPU path uses; gate strictly on parity.
- KV-cache stride/layout mismatches → silent wrong output. Mitigation: trace-compare K/V
  buffers against the CPU cache on the first tokens.
- Threadgroup sizing for RMSNorm (256) and attention (one thread/head) is correctness-first,
  not yet tuned; revisit after parity for throughput.

## Status

Kernels: **done & shipped.** Integration (steps 1–5): **next**, gated on parity, multi-step.
