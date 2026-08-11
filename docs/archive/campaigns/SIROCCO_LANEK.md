# SIROCCO — Lane K (kernel campaign)

**Machine:** RTX 3060 Laptop GPU (GA106, CC 8.6, 30 SM), 6 GB GDDR6, driver 576.83, CUDA 12.9, **WDDM** · Win11
**Target:** Llama-3.2-1B decode, Windows/CUDA · **Status:** three clean, correctness-preserving wins **shipped** (`uint4` decode-attn K-read [PR #440]; `SPLITK_MAX` uncap [PR #442]; **`uint4` prefill-attn K-read, +17–19% prefill, this PR**). The decode budget is exhausted; the newly-measured **prefill** regime (99% of long-context wall time) is the open frontier.

> **One-line result:** ten kernel experiments, **three shipped** (byte-identical `uint4` decode-attn K-read, +~10% long-ctx; token-identical `SPLITK_MAX` uncap, +~6% e2e @8k; **byte-identical `uint4` prefill-attn K-read, +17–19% prefill**), **seven rejected on measured evidence**. `main` only ever moved for a change that measurably earned it and cleared the bit-identical spec-verify gate.

> **⚠ REFRAME (2026-07-13): the campaign optimized the wrong 1%.** Direct measurement — Llama-3.2-1B Q8_0, ctx 8802: **prefill = 152.8 s (99.2%)**, decode-64 = 1.2 s. The whole Lane K decode campaign (#1–#9) tuned the ~1% decode tail of a long-context request. **Prefill is the other 99%**, and its attention (`attention_batched`) was still doing the scalar f16 K read that win #1 replaced everywhere on the decode path — hence experiment #10 below (a verbatim transplant, +17–19% prefill, byte-identical). See [SIROCCO Phase P bundle](../../../qa/evidence-bundles/sirocco-prefill-attn-kread-20260713/README.md).

---

## 1. The mandate — why Lane K

SIROCCO Phase 0 (the [roofline receipt](../../../qa/evidence-bundles/sirocco-phase0-roofline-20260712/README.md)) modeled the decode step as `t = B/BW_eff + C` and measured it. The verdict was decisive and **inverted the illustrative plan**, which had assumed an overhead-bound box (Lane A):

| quantity | measured | how |
|---|---|---|
| `BW_peak` (device triad) | **270.9 GB/s** | 94% of 288 nameplate |
| P1 decode (1B Q8_0, ctx≈0) | **126.4 tok/s**, 7.88 ms/tok | interleaved, monitored-boost |
| **`C` (host/loop overhead)** | **≈ 0** | `CAMELID_DECODE_TIME`: forward = step wall, sample/loop = 0 |
| **MBU@P1** | **0.615** | `B/(t·BW_peak)` |
| wire ratio (PC2) | 0.982 | API path not the bottleneck |

**`C ≈ 0` + MBU 0.615 ⇒ decode is kernel-bound, not overhead-bound.** The GEMV kernels are near the memory wall but leave ~35% on the floor; the host is not starving the GPU. That selected **Lane K (kernel campaign)** and ruled out Lane A/graphs (confirmed later: CUDA Graphs measured **−5%** at ctx≈0 — the empty-block fixed-shape attention tax exceeds any launch saving, and host cost is already ≈0).

---

## 2. Where the headroom actually is (regime map)

A follow-up decode-loop analysis (261 kernel launches/token; 148 non-GEMV moving only ~2.6 MB ≈ ~10 µs of real bandwidth) refined the picture. Headroom is **not** uniform:

| Regime | State | Why |
|---|---|---|
| **ctx≈0 Q8 (the 126 headline)** | 🪨 **nearly tapped out** | GEMVs at 84–101% of peak. Residual ~2.2 ms/token is ~130 tiny launches = per-kernel execution floors + the serial data-dependency chain (norm→QKV→rope→scatter→attn→O→FFN), **not** data movement or host launch (graphs already tried: −5%). Ceiling here is single digits, likely noise. |
| **long-ctx Q8 (4k–32k)** | 🌽 moderate, corner | Decode attention reads the f16 KV cache at ~10–20% of peak (PC3). +15–30% realistic at 2–8k; ~2× only at the 16–32k edge (atypical on a 6 GB/1B box). |
| **Q4 / K-quant models** | 🛢️ **fattest budget** | Q4_K_M decodes *slower* than Q8 despite half the bytes — the K-quant GEMVs are compute-bound. |

---

## 3. Scorecard

| # | Experiment | Regime | Outcome | Evidence |
|---|---|---|---|---|
| **1** | **`uint4` attention K-read** | long-ctx Q8 | ✅ **SHIPPED** — +~10% vs scalar, +5–10% vs the coalesced kernel, **byte-identical** | [PR #440](https://github.com/timtoole02/Camelid/pull/440) |
| 2 | Q8 GEMV `U`-unroll bump | ctx≈0 | ❌ refuted — GEMV already 84–102% of peak; U=4 optimal | microbench |
| 3 | CUDA Graphs default-on | ctx≈0 | ❌ −5% (empty-block attention tax; Lane A empty) | A/B |
| 4 | Coalesced attention default-on | long-ctx | ❌ breaks lossless spec-decode at ctx>512 | `splitk_spec_verify` self-skip |
| 5 | `uint4` weighted-V retile (`attention_decode`) | ctx≤512 | ❌ −4.7% (occupancy loss > load-width) | git-stash A/B, byte-identical |
| 6 | `uint4` split-K V (`attn_sk_partial`) | long-ctx | ❌ −18% (1/4-warp utilization) | git-stash A/B |
| 7 | Bit-exact dp4a `q6k_gemv` | Q4 models | ❌ regresses 0.34–0.68× (2-lane cap) | microbench, byte-identical |
| 8 | Token-parity (4-lane) dp4a `q6k` | Q4 models | ❌ fails token-parity (8/8 prompts diverge) | real-model verification |
| **9** | **`SPLITK_MAX` uncap 16→32** | long-ctx Q8 | ✅ **SHIPPED** — +~6% e2e measured @8k (V-read +19% micro @32k; larger at longer ctx by byte-share, not A/B'd); token-identical to oracle **and** bit-identical spec-verify | [bundle](../../../qa/evidence-bundles/sirocco-laneK-splitk-uncap-20260713/README.md) |
| **10** | **`uint4` K-read → `attention_batched` (PREFILL)** | long-ctx **prefill** | ✅ **SHIPPED** — **+17–19% prefill** (2k/4k/6.6k interleaved), byte-identical (6 parity gates + 48/48 e2e). Win #1's read, applied to the prefill/verify attention it never reached — which is 99% of long-ctx wall | [bundle](../../../qa/evidence-bundles/sirocco-prefill-attn-kread-20260713/README.md) |

> **#9 is the occupancy counterpart to the rejected #6.** #6 *vectorized* the split-K V read and lost by dropping threads to a quarter-warp; #9 keeps the kernel and adds split **blocks**, curing the same under-utilization from the other side. The V read at `attn_sk_partial` uses only `head_dim`=64 of 256 block threads, so it is occupancy- (not coalescing-) limited: 92→110 GB/s (+19%) going 16→32 splits. 32 is the knee — the K read is already optimal at 16 and regresses beyond it.

---

## 4. The one win — `uint4` attention K-read (shipped)

**Change:** widen the scalar `u16` f16 K-cache read in the decode attention to **`uint4` (128-bit = 8 keys/load)**, accumulated in the **same d-order**, in `attention_decode`, `attention_decode_sw`, and `attn_sk_scores`. 45-line diff.

**Why it's byte-identical:** the resident module compiles `--fmad=false`, so preserving textual operation order preserves the bits. The load widens *within a thread*; the thread/warp count and the reduction order are unchanged. Alignment is exact (head_dim=64 → every position row is 16-byte-aligned; a `(head_dim & 7)==0` guard falls back to scalar otherwise).

**Correctness:** `splitk_spec_verify_bit_identical` **passes** (ran, not skipped), asserting the decode path is bit-identical to the *unchanged* `attention_batched`/`attention_tree_batched` spec-verify kernels across pc 512→4096 (incl. `attention_decode` at G=4). Since the verify kernels are untouched, this proves the read is byte-identical to the old scalar read ⇒ **decode == spec-verify ⇒ lossless speculative decode preserved.**

**Measurement** (ctx~1542 split-K, interleaved):

| | tok/s |
|---|---|
| scalar (pre-change) | ~84–86 |
| coalesced kernel (token-parity, breaks lossless) | 90.2 / 88.0 / 83.5 |
| **`uint4` (byte-identical)** | **94.7 / 93.9 / 91.6** |

It **strictly dominates** the previously-shipped opt-in coalesced kernel — faster *and* it preserves the lossless-spec guarantee the coalesced kernel breaks. It is a no-op at ctx≈0 (split-K inactive), so zero headline regression.

---

## 5. The rejections — and what each one taught

**#4 Coalesced attention default-on — the near-miss that mattered most.** It passed the obvious gates (load, ctx≈0, +5–8% long-ctx, *greedy token-identity*). Then the stricter `splitk_spec_verify` test caught that it re-associates the split-K dot, so **decode ≠ spec-verify at the bit level → lossless speculative decode breaks at ctx>512.** The greedy-token check only compared decode-vs-decode; it never checked decode-vs-spec-verify (the actual lossless property). **A naive default-flip would have shipped a silent correctness regression *and* disabled the test that guards it.** → *A speedup that passes greedy token-identity can still break the bit-identical spec-verify contract.*

**#5, #6 V-read vectorization — a dead end in every form.** Widening the attention **V** read (weighted-V −4.7%; split-K `attn_sk_partial` −18%) always loses, because the V read is *already coalesced* and vectorizing it drops the active thread/warp count (8× fewer threads at `attention_decode`; a quarter-warp at `attn_sk_partial`). Split-K's grid-level parallelism does **not** rescue within-block warp under-utilization. → *Only vectorize a read where you can widen the load **within** a thread without shrinking the thread/warp count. That is exactly why the K-dot (#1) won and the V-reads lost: the K-dot keeps one-thread-per-position and only cuts loads-per-thread.*

**#7, #8 dp4a-ize the K-quant GEMV — the fattest budget, gated by precision.** Q4_K_M decodes *slower* than Q8 despite fewer bytes; root cause is **measured**: `q6k_gemv`/`q5k_gemv` have **0 `__dp4a`** (pure scalar integer MAC) while `q4k_gemv` is dp4a-tuned. A perf-prototype dp4a `q6k` hit **1.74× at the lm_head shape** (49% → 86% of peak). But:
- The **byte-identical** version regresses (0.34–0.68×). Q6_K carries **four different scales per stride-32 weight group**, so a parity-preserving dp4a is capped at **2-lane** (q4k gets 4-lane only because its stride-8 weights share one scale). The 1.74× came from a 4-lane collapse that **breaks the parity anchor**.
- Relaxing to **token-parity** doesn't rescue it either: Q6_K is used for the **per-layer** `attn_v` + `ffn_down` projections (not just the lm_head), so the f32-tail-association difference **compounds across 16 layers** and flips greedy tokens (**8/8 prompts diverged** on the real Q4_K_M model; the kernel is integer-correct — it's genuine precision loss, not a bug).

→ *"Token-parity" is fragile for **per-layer** kernel changes — any f32-association change compounds across layers and flips greedy tokens. Only a **final** (lm_head) projection could tolerate it, and even then only probabilistically (which weakens the greedy-determinism Runnable invariant — recommend against).*

---

## 6. Principles (reusable, hardware-agnostic)

1. **Fix the denominator before optimizing.** Phase 0's `C≈0` + MBU 0.615 measurement redirected the whole campaign from the (empty) host-overhead lane to the kernel lane. Running the wrong lane is worse than running no lane.
2. **A perf-prototype that isn't parity-preserving over-states the win.** The dp4a `q6k` "1.74×" evaporated (or went negative) once the byte-identical / token-parity versions were actually built. Build the correctness-preserving kernel *before* trusting the number.
3. **Vectorize *within* a thread, not by dropping threads — and if a read is thread-starved, add blocks instead.** Widening a per-element read that is already coalesced trades away the parallelism that hides latency (#5, #6). The *same* under-utilized `attn_sk_partial` V read that killed the vectorization attempt was cured from the other side by raising the split count (#9): more blocks → more resident warps → +19% on the exact read that lost 18% when vectorized. Diagnose whether a slow read is coalescing-bound or occupancy-bound *before* picking the lever.
4. **Greedy token-identity ≠ bit-identity.** For any change touching the spec-verify / split-K path, the device-side `splitk_spec_verify_bit_identical` gate is the real backstop; the runtime greedy probe is blind above ctx~23 / G=1.
5. **Prove correctness at the level the invariant lives.** Byte-exact-vs-oracle (Runnable lane) is stronger than token-parity; token-parity is stronger than "usually identical." Don't silently downgrade the contract.

---

## 7. Current state & what remains

- **In `main`:** (1) the `uint4` decode-attn K-read (PR #440) — byte-identical, lossless-spec-preserving, +~10% long-context decode; (2) the `SPLITK_MAX` uncap 16→32 (PR #442) — token-identical, +~6% e2e measured at ctx 8k; (3) the `uint4` **prefill**-attn K-read (this PR) — byte-identical, **+17–19% prefill**.
- **Exhausted (decode):** the clean, correctness-preserving Lane K *decode* gains on this GPU. ctx≈0 Q8 is at the wall; the attention **V-read via vectorization** is a proven dead end (but the **V-read via occupancy** — #9 — was the sleeper win that same analysis pointed at).
- **OPEN FRONTIER — prefill (SIROCCO Phase P):** prefill is **99% of long-context wall time** and was never on the decode roofline. Experiment #10 took the first, free, byte-identical bite (uint4 K-read, +17–19%). Remaining prefill levers (measured, not yet built): the prefill **V-read is also scalar**; the batch chunk is only `MAX_VERIFY_K`=8 tokens so weights are re-read ~n/8 times (a **larger prefill block** would amortize the linear term); and the O(n²) `attention_batched` is a **naive per-position kernel, not flash-tiled**. This is where the real long-context wall time lives.
- **Gated (not blocked by engineering):** the K-quant lm_head dp4a is a real ~1.74× kernel win, but capturing it requires **retiring the K-quant byte-exact-vs-oracle Runnable invariant** (or restricting to a probabilistic lm_head-only variant). That is a **precision-policy decision**, not a kernel tweak — and having built and measured both, the recommendation is that it is not worth trading greedy determinism for.

---

## 8. Reproduction

- **Phase 0 / G0:** [`qa/evidence-bundles/sirocco-phase0-roofline-20260712/`](../../../qa/evidence-bundles/sirocco-phase0-roofline-20260712/README.md) — `roofline-receipt.json`, `triad.cu`, all three sweeps.
- **Shipped win #1 (`uint4` K-read):** [`qa/evidence-bundles/sirocco-laneK-attn-kvec-uint4-20260712/`](../../../qa/evidence-bundles/sirocco-laneK-attn-kvec-uint4-20260712/README.md) — diff, `splitk_spec_verify` log, A/B, the design workflow.
- **Shipped win #2 (`SPLITK_MAX` uncap):** [`qa/evidence-bundles/sirocco-laneK-splitk-uncap-20260713/`](../../../qa/evidence-bundles/sirocco-laneK-splitk-uncap-20260713/README.md) — `micro-splitk.cu` n_splits sweep, the three correctness gates, the interleaved A/B (+5.96%).
- **Shipped win #3 (`uint4` prefill K-read, Phase P):** [`qa/evidence-bundles/sirocco-prefill-attn-kread-20260713/`](../../../qa/evidence-bundles/sirocco-prefill-attn-kread-20260713/README.md) — the prefill-vs-decode wall-time reframe, the interleaved prefill A/B (+17–19%), the six parity gates + 48/48 e2e.
- **Coalesced investigation:** [`qa/evidence-bundles/sirocco-laneK-coalesced-attn-20260712/`](../../../qa/evidence-bundles/sirocco-laneK-coalesced-attn-20260712/README.md).
- **Kernels:** `src/cuda_resident.rs` — `q8_gemv` @230, `q4k_gemv` @455, `q6k_gemv` @779, `attention_decode` @1371, `attn_sk_scores` @2089 (`attn_sk_partial` @2207). Correctness gate: `splitk_spec_verify_bit_identical` (`src/cuda_resident/tests.rs`), `--ignored`, requires a CUDA device (does **not** run in CI).

_All measurements: RTX 3060 Laptop, monitored-boost (hardware clock pin admin-gated on WDDM); mem clock stable at 6000 MHz — the bandwidth governor — so bandwidth-bound tok/s are robust; SM clock floats and governs `C≈0`. A/Bs are interleaved (thermal-robust deltas)._
