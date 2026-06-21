# CAMELID_HOTSPOTS.md

Where Camelid actually spends time, by measurement — not guesses.

**Profiling methods used (no external CPU sampler was available on this Windows box, so attribution is triangulated):**
1. **Roofline / bandwidth analysis** — decode tok/s × model bytes = achieved GB/s vs measured peak (`qa/evidence-bundles/cpu-perf-characterization-20260620`, STREAM ~273 GB/s on GPU / ~51 GB/s CPU peak).
2. **Controlled A/B isolation** — flip one kernel/flag, measure the delta; a flag that barely moves a stage proves that stage is not compute-bound there (`PERF_RECEIPTS/same-host/cpu-prefill-matrix-*.json`, routing sweep).
3. **Camelid's own stage telemetry** — `LlamaForwardTimings` (embedding / per-layer attn+ffn / final_norm / logits / sample) via `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`.
4. **Nsight** for GPU (`prof1.nsys-rep`, and the campaign's ncu captures).
Recommended follow-up: a `samply`/WPA flamegraph of `forward_single_token_timed_internal` for exact line attribution.

---

## Hot #1 — the per-projection Q8 matvec (CPU decode)  — dominant

- **Where:** `src/inference.rs` decode path `forward_single_token_timed_internal` → `linear_for_role_runtime_with_plan` / `output_projection_runtime_with_plan` → the Q8 block-dot / packed-rows4 kernels in `src/tensor/mod.rs` (`quantize_q8_0_row`, `q8_0_packed_rows4_dot`).
- **% runtime:** ~all of decode. A 3B Q8 decode token streams ~3.2 GB of weights; at 5.97 tok/s that's ~19 GB/s achieved, **~33% of the ~51 GB/s CPU peak** (matches the 2026-06-20 characterization for 0.6B).
- **Why hot / expected?** Expected — decode is fundamentally a memory-bound matvec (each weight read once, used for one token). 
- **Proposed fix:** none cheap. SIMD doesn't help a memory-bound matvec (proven: VNNI +0.28–4.5%, packed-SIMD config B −8% here). The only kernel lever is a better-streaming GEMM (P1). **One free, bit-identical tuning win exists** (`SPEED_FIX_PLAN P0.5`): decode is *over-threaded* here — best at 1–2 threads (6.16 vs default 5.14, **+20%**, byte-identical), reachable now via `CAMELID_THREADS=2` (receipt `decode-thread-sweep-*.json`). Not a default change (over-fits this 2-channel-DDR4 box). llama proves higher bandwidth is reachable on this exact hardware (its decode = 8.8 tok/s ≈ 28 GB/s).
- **How to prove a fix:** decode tok/s on `cpu-prefill-matrix.mjs` + byte-identical greedy diff.

## Hot #2 — the batched prefill GEMM (CPU prefill)  — the throughput gap

- **Where:** `src/inference.rs` `forward_prefill_chunk_timed_fast` → `forward_prefill_layer_chunk_timed` → the same Q8 packed kernels, with `chunk_rows = n_prompt_tokens`.
- **% runtime:** ~all of prefill.
- **Why hot / expected?** Prefill is compute-bound (one weight read serves all chunk tokens). Camelid's batched path IS structurally correct (amortizes ~3.2× over decode, ≈ llama's ~3.5×) — but the inner kernel is **AVX2 bespoke, not register-tiled**, so per-kernel throughput is 0.61× llama's tinyBLAS.
- **Evidence it's the KERNEL, not routing/overhead:** prefill routing sweep (layer-major on/off, chunk 64/256/512/all) is flat within ~3% — so it's not chunk overhead; and the gated SIMD kernels REGRESS, so it's not "missing SIMD." It's tiling/scheduling quality.
- **Proposed fix:** P1 unified tiled GEMM owner (register-blocked AVX2, single input quantize, in-kernel chunk scheduler).
- **How to prove:** `cpu-prefill-matrix.mjs` prefill tok/s default vs owner-flag, byte-identical.

## Hot #3 — projection fragmentation / redundant input quantization (CPU)

- **Where:** per-role paths (`try_x86_q8_attention_qkv_decode_consumer_path`, `try_x86_q8_ffn_down_decode_consumer_path`, `output_projection_runtime_with_plan`) each derive shapes, dispatch, and quantize the activation independently. The QKV triplet already shares one input quantization (good); FFN gate/up and ffn_down/output may not fully.
- **% runtime:** secondary — the activation quantize is small vs the weight stream, but the per-projection dispatch/allocation adds up across ~7 projections × 28 layers per token.
- **Why hot / expected?** Partly avoidable. The mapping note (`x86-q8-llamacpp-mapping`) flags "decode consumers re-quantize the input row per projection cluster" as the structural cost vs llama's single `from_float` into `wdata`.
- **Proposed fix:** share one quantized-input object across all same-input projections; fold into the P1 owner. _(Specific verified candidates from the audit workflow are appended below.)_
- **How to prove:** stage telemetry `activation_quantize_pack_us` count drops; decode tok/s unchanged-or-up; byte-identical.

## Hot #4 — GPU decode attention (CUDA) — already addressed

- **Where:** `src/cuda_resident.rs` `q8_gemv` (matmuls) + attention decode (`attn_sk_partial` split-K).
- **% runtime / status:** `q8_gemv` ~76% DRAM BW ("already efficient — leave alone"). `attention_decode` WAS the hotspot (occupancy 4.4%, 0.07 waves/SM); split-K (committed `57977f6e`) fixed occupancy (→1.42 waves, ~42%). Residual is parity-locked (uncoalesced K/V).
- **Proposed fix:** none low-risk (see SPEED_FIX_PLAN P3). 

## NOT hotspots (ruled out by measurement)
- **Sampling / SSE streaming / tokenizer / template** — negligible vs the matvec; greedy argmax + Axum SSE. (The marker-harness "decode tok/s" undercounts because it counts SSE chunks, but engine timing via llama-server ground-truth and Camelid telemetry agree the matvec dominates.)
- **Model load / mmap** — Camelid's fast-load claim holds; first-token latency is prefill compute, not load.
- **Prompt-prefix cache** — a *feature* (repeat-turn win) and a benchmarking trap, not a perf cost.

---

## Audit workflow — result: ZERO confirmed wins (adversarially verified)

A multi-agent audit (`camelid-perf-deepdive-audit`, 8 agents) independently scanned `forward_single_token_timed_internal` and the Q8 projection paths for parity-safe micro-wins, then had separate skeptic agents try to refute each. **All 5 candidates were rejected with high confidence.** This is a real negative result — it proves the decode loop is allocation/clean and the matvec dominates.

| candidate (file) | skeptic verdict |
|---|---|
| Reuse gate/up buffers instead of per-token `vec![0.0;…]` (`inference.rs:8310`) | **Reject.** The buffer is *moved* into `CpuTensor::from_f32` (`:8461`) — can't reuse across iterations without an API change; and the fused/single-owner fastpaths (`:8289`,`:5692`) short-circuit it, so it's often not even hit. A 32 KB alloc is <1 µs vs hundreds of µs of matvec — sub-noise. |
| Drop `resident_logits.clone()` (`inference.rs:3818`) | **Reject — parity-unsafe.** Removing it makes `output_norm_state` hold logits `[1,vocab]` not the norm `[1,hidden]`; API-level dense diagnostics would read the wrong tensor (OOB). |
| Verify QKV triplet doesn't re-quantize (`:12509`) | Confirmed already-optimal; **no win** (validation only). |
| Verify FFN-down chain doesn't re-quantize (`:11112`) | Confirmed already-optimal; **no win**. The 2nd quantize is mandatory (different data). |
| Cache FFN-norm quantization (`:8370`) | Already cached via scope; **no win**, and quantize is ~20–30% of FFN time but is precision-sensitive (f16 rounding) so not safely removable. |

**Conclusion:** the decode hot loop already shares input quantization (QKV triplet, gate/up) and has no removable redundancy. The only CPU lever is the matvec kernel itself (Hot #2, P1). Receipt: `PERF_RECEIPTS/audit-workflow-result.json`.

## GPU — one deferred parity-safe win (medium effort)
The GPU audit confirmed the parity-lock, `q8_gemv` 76% BW, and batched-prefill-shipped. It surfaced ONE remaining parity-safe idea the campaign deferred: **multi-stream overlap of the independent Q/K/V and gate/up GEMVs** on separate CUDA streams to hide DRAM latency (`qa/perf/qwen3-cuda-resident-phase3-findings.md:131-139`) — est. 10–15% decode lift, parity-safe (concurrent independent ops don't change math). Medium–high effort (stream orchestration); see SPEED_FIX_PLAN P2.
