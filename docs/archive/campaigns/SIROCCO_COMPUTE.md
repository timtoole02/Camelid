# SIROCCO — Compute phase (prefill FLOPs / occupancy)

**Machine:** RTX 3060 Laptop (GA106, CC 8.6, 30 SM, ~3 MB L2), 6 GB, CUDA 12.9, **WDDM** (no Nsight — CUDA-event microbenches only) · Win11
**Status:** **M-C3 SHIPPED (byte-identical).** The tensor-core premise was **refuted**; the real binder is local-memory/occupancy.

> **Mandate.** After Phase P M1 (flash prefill, up to 11.5×), the two M2 rejections proved M1 cut the K/V traffic ~8× and moved prefill attention **off the DRAM wall onto compute**. This phase attacks that compute cost. A 5-scout design workflow **refuted tensor cores** (M1 attention runs at ~1% of the 3060's FP32 peak, so the FLOP ceiling addresses <0.2% of the wall) and pinpointed the binder as **local-memory/occupancy**.

---

## 1. The binder — local memory, confirmed at compile time (M-C0)

The M1 flash kernels take `k_tokens` as a **runtime** parameter. That blocks ptxas from unrolling the per-token loops, so the per-row accumulators (`dot[]`, `local_max[]`, `a[]`) are dynamically indexed → **local memory**. `ptxas -v` (`qa/evidence-bundles/sirocco-compute-mc3-constk-20260714/mc0-ptxas.txt`):

| kernel | stack frame | registers |
|---|---|---|
| `scores` (runtime `k_tokens`, shipped M1) | **128 bytes** (local mem) | 40 |
| `scores_k8` (compile-time `k==8`, `#pragma unroll`) | **0 bytes** (registers) | 48 |

Corroborated by M2b: bumping `FLASH_MAX_BQ` 16→32 regressed even k=8 — array size gates occupancy.

## 2. M-C3 — const-k=8 unroll (SHIPPED, byte-identical)

`flash_pref_scores_k8` + `flash_pref_partial_k8` are **byte-identical twins** of the flash kernels with the token loops compile-time-bounded (`FLASH_KT8=8`) + `#pragma unroll`, so ptxas promotes the accumulators to **registers** (0-byte frame). Same ops, same order → bit-identical output. `launch_attention_flash_prefill` dispatches them when `k_tokens==8` (the common prefill chunk); other sizes use the runtime kernels.

**Perf** (`mc3-ab.txt`, flash-on, M-C3 vs shipped M1, interleaved):

| ctx | M1 | M-C3 | speedup |
|---|---|---|---|
| 4072 | 13.8 s | 12.6 s | 1.09× |
| 6602 | 28.0 s | 25.2 s | 1.11× |
| 8802 | 47.1 s | 40.0 s | **1.18×** |

Grows with context. Composes with M1: vs the original baseline @8.8k, **133 s → M1 43 s → M-C3 40 s**.

**Correctness — byte-identical, no oracle needed.** M-C3 == M1 bit-exact (48/48 greedy tokens @ctx 8802); default path (flash off) unchanged (`splitk_spec_verify_bit_identical`, `prefill_then_decode_matches_sequential`, `verify_batch_matches_sequential` pass); the flash token-parity is inherited from M1 (identical output). Still opt-in (`CAMELID_FLASH_PREFILL`, since the *flash kernel itself* is token-parity vs `attention_batched`); M-C3 only makes that kernel faster, bit-for-bit.

## 3. What's refuted / remaining

- **Tensor cores / MMA — DEAD.** Roofline: M1 attention at ~1% of FP32 peak → the FLOP ceiling is <0.2% of the wall; the `m16n8` MMA tile also underfills at k_tokens=8. Not the lever.
- **Scores-I/O fusion — not the binder** (scores are a warp-broadcast, L1-resident, ~2% DRAM). Deprioritized.
- **M-C4 full-block AV — ❌ REJECTED (regression 0.81–0.89×).** The weighted-V loop uses only `head_dim`=64 of 256 block threads; distributing the `k_tokens*head_dim` output elements across all 256 (scalar accumulator, byte-identical) *looked* like free occupancy. But M-C3's per-dim assignment reads each `V[p][d]` **once** and reuses it across the `k_tokens` tokens; the full-block version gives each `(t,d)` its own thread, reading `V[p][d]` **`k_tokens`× more**. The AV phase was **V-reuse-sensitive, not occupancy-bound** — the lost reuse cost more than the 4× threads gained. Reverted (bit-exact confirmed first).
- **Remaining:** the weight-GEMM (~12% of the wall, `__dp4a` scalar) is bandwidth-bound and small. With tensor cores, scores-fusion, GQA-reuse, larger-chunk, and full-block-AV all refuted/rejected on evidence, **the prefill flash kernel (M1 + M-C3) is at its practical ceiling** for this GPU; the residual is the online-reduction structure itself.

_M-C0 ptxas + M-C3 A/B: `qa/evidence-bundles/sirocco-compute-mc3-constk-20260714/`. Design workflow (5 scouts + synthesis) refuted tensor cores and routed to the const-k unroll._
