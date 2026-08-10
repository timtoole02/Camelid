# SIROCCO — Phase P (prefill)

**Machine:** RTX 3060 Laptop (GA106, CC 8.6, 30 SM, ~3 MB L2), 6 GB, CUDA 12.9, **WDDM** · Win11
**Status:** **M1 SHIPPED (opt-in).** M0a (traffic) GO, M0b (token-parity) PASSED, and **M1 — the tiled flash prefill kernel — is built, gated, and merged**: **2.2× @6.6k → 3.1× @8.8k → 11.5× @12.2k** prefill, token-identical to baseline (6/6), default path byte-exact, verify untouched. Opt-in `CAMELID_FLASH_PREFILL=1` (token-parity). See the [M1 bundle](../../../qa/evidence-bundles/sirocco-phaseP-m1-flash-20260713/README.md).

> **Mandate.** The Lane K decode campaign optimized the wrong ~1%. Direct measurement (Llama-3.2-1B Q8_0, ctx 8802): **prefill = 152.8 s (99.2 % of wall)**, decode-64 = 1.2 s. Prefill is **94–96 % O(n²) attention** and was never on the decode roofline. Phase P attacks it. The first, byte-identical bite already shipped ([PR #443](https://github.com/timtoole02/Camelid/pull/443), uint4 prefill K-read, +17–19 %); the rest of the wall needs **flash-style tiling**, which reassociates the softmax and is therefore **token-parity** — gated like the rejected #8, not byte-identical.

---

## 1. The bottleneck — a 24–32× K/V re-read (code-confirmed)

`attention_batched` launches `grid = (k_tokens × n_heads)` blocks, **one block per (query-token t, head)** (`cuda_resident.rs:1784-1789`, launch `:3636`). Each block re-streams the **entire** prefix K and V for its `kv_head`. So within one `MAX_VERIFY_K = 8`-token prefill chunk, each kv_head's K/V is read by `k_tokens(8) × GQA-repeats(4 for 1B / 3 for 3B) = 32× / 24×` independent blocks. With 8-token chunks the whole prompt's K/V is re-processed `n/8` times. At base≈8000, head_dim=128 (3B): current attention traffic ≈ **787 MB** vs a single-read minimum of ≈ **33 MB** = **24×**.

Two stacked reuse axes a flash kernel can collapse: the **query-token** axis (8 tokens can share a K/V tile) and the **GQA** axis (4/3 q-heads share one kv_head's K/V).

## 2. M0a — traffic reality (KILL gate: run, PASSED marginal)

`qa/evidence-bundles/sirocco-phaseP-m0a-20260713/` — a standalone microbench lifting `attention_batched`'s read pattern, timing k=8 vs k=1 at base=8000:

| model | t(k=1) | t(k=8) | ratio | k=8 naive-traffic BW |
|---|---|---|---|---|
| 1B (32/8, hd64) | 0.42 ms | 1.36 ms | 3.22 | 386 GB/s |
| 3B (24/8, hd128) | 0.51 ms | 1.90 ms | 3.70 | 414 GB/s |

**Read:** ratio 3.2–3.7 is between the strong-GO (≥5) and KILL (≤2) bars, but the **naive-traffic BW exceeds the 271 GB/s DRAM peak** — proof the kernel demands more read bandwidth than DRAM supplies, i.e. ~70 % of the re-reads hit DRAM (~30 % L2-absorbed by the small 3 MB L2). The re-read traffic is **real** (not KILL), but the realized win is **below the on-paper 24×** — plan for **~2–4× prefill**, not 5–6×. GO to M0b.

## 3. Design (chosen)

A **separate `launch_attention_flash` kernel, wired PREFILL-ONLY** — never on the verify path.
- **Never touch verify.** Thread a `flash_prefill` flag through `run_batched_layer_stack`; `verify_batch` passes `false`, so verify + `attention_tree_batched` keep `attention_batched` and its **byte-identity contract** (`splitk_spec_verify_bit_identical`, `verify_batch_matches_sequential`) intact.
- **Block = (kv_head, key-split)**, serving `Bq` query tokens × GQA-repeats q-heads — capturing **both** reuse axes. Grid = `n_kv_heads × n_key_splits` (flash-decoding-style key-split; 8 kv_heads alone underfill 30 SMs).
- Stream `Bc`-key f16 K/V **tiles through shared** via the #443-proven uint4 (128-bit) load. **Online-softmax** running `(m, l, acc[head_dim])` per query row **within** a split; **reuse the existing `attn_sk_combine`** (already token-parity-proven for decode) to merge per-row partials across key-splits — so the **only new reassociation is the online rescale within a split**.
- **Scores in GLOBAL** (attn_sk_scores-style), not shared → frees the O(prefix) shared-scores buffer that today caps context at ~11.2 k (a second win: raises the ctx ceiling).
- `Bc` sized to fit **48 KB static** (no opt-in): Bc=32 @hd128, Bc=64 @hd64. Start `Bq=8`, enlarge to 64 (M3).
- **Memory:** query-block size is bounded by the flash kernel's 48 KB static shared, **decoupled** from the 6 GB DRAM budget, from `MAX_VERIFY_K`, and from the batched-GEMM chunk (which must stay ≤32 — the down-proj `[warp][token][block]` scratch is an unguarded >48 KB cliff at k≈47). A vlogits-free prefill scratch makes Bq=64 cost ~9.6 MB (negligible).

## 4. Parity — TOKEN-PARITY (byte-identity is capacity-blocked)

Byte-identical reuse needs `Bq × repeats` full O(prefix) score rows in shared (`base×4` bytes each), busting the 99 KB opt-in max even at reuse-factor 2; and each query token's different `position_count` gives different split boundaries. So the traffic win **requires** O(head_dim) online-softmax state = reassociation = **token-parity**. Ship **opt-in / prefill-only** (`CAMELID_FLASH_PREFILL=1`), mirroring the existing `metal.rs` flash path.

**Required gate** (the #8 bar, stricter — prefill perturbs the persisted KV cache so error compounds per-layer **and** per-position): a multi-prompt long-context oracle — **8–16 diverse / near-tie / repetitive prompts, real model, 4k–8k ctx, ≥64 greedy tokens, 100 % token match vs the byte-exact serial path across all 16 layers.** Run it TWICE: in the M0b probe (before any kernel) and against the real tiled kernel (M2). `prefill_then_decode_matches_sequential` must also pass; verify's byte-identity gates must stay green at every milestone.

## 5. Milestones (biggest risk de-risked first, cheapest)

- **M0a — traffic reality.** ✅ run (§2): marginal GO, re-reads are real DRAM traffic.
- **M0b — parity reality.** ✅ **PASSED** ([bundle](../../../qa/evidence-bundles/sirocco-phaseP-m0b-20260713/README.md)). An env-gated `CAMELID_FLASH_PROBE` branch (compile-time `-DFLASH_PROBE`, early-return so probe-off is byte-identical) replaced the two-pass softmax in `attention_batched` with a **single-pass online running-max rescale** — flash's exact float reassociation, zero tiling. `bench-generate` uses `attention_batched` only in prefill (no drafter), so the probe isolates the prefill-attention reassociation. Oracle: **6/6 diverse prompts token-identical** (varied / technical / dialogue / near-tie-repetitive / code / numeric), ctx 6.5k–13.6k, 64/64 each → the online-softmax reassociation does not flip a greedy token across 16 layers. Token-parity holds.
- **M1 — flash kernel @Bq=8.** ✅ **SHIPPED** ([bundle](../../../qa/evidence-bundles/sirocco-phaseP-m1-flash-20260713/README.md)). Implemented as `flash_pref_scores` + `flash_pref_partial` = the split-K decode passes extended to the whole `k_tokens` query chunk (each key's K/V dequantized once, reused across all `k` query rows), grid `(n_heads, n_splits)`, buffers flat by `(t*n_heads+head)` so `attn_sk_combine` is reused unchanged, scores in global (removes the ~11.2k shared ceiling). Measured **2.2×→3.1×→11.5×** at ctx 6.6k/8.8k/12.2k; 6/6 token oracle; default byte-exact; verify untouched. (Chose the **query-reuse** axis via the existing split-K grid over the full flash-decoding key-split design — simpler, provably SM-filling, and it won decisively. GQA-axis reuse and larger `Bq` remain as M2/M3 multipliers.)
- **M2a — GQA-axis reuse.** ❌ **REJECTED (regression 0.69–0.92×).** Made the block serve all `repeats` q-heads of a kv_head (grid `n_kv_heads × n_splits`, one K/V read per kv_head). Token-correct (smoke matched) but **slower**, worsening with ctx: the `repeats*k_tokens`≤32 per-row accumulators spill to local memory (dynamic-indexed arrays → DRAM-backed, ~4× M1's local traffic), and the 4× smaller grid (32→8 head-blocks) cuts occupancy — while the GQA re-reads it removes were largely L2-absorbed already (consecutive heads co-schedule). The traffic saved < the spill + occupancy cost. A spill-free version needs a shared-K/V-tile rewrite (warp-per-row) for a saving that M0a suggests is mostly L2-resident — low value. Reverted.
- **M2b — enlarge Bq (bigger prefill chunk).** ❌ **REJECTED (marginal, +1–4% at best; K32 regresses).** Built an env-tunable prefill chunk (`CAMELID_PREFILL_K`, decoupled from `MAX_VERIFY_K`, scratch + `FLASH_MAX_BQ` sized to 32). Token-correct at 8/16/32. But the within-binary sweep gave only K16 = 1.01–1.04× over K8, and K32 = 0.92–0.95× (regression), and the `FLASH_MAX_BQ`=32 bump made even K8 slower than the shipped M1. **Root finding: M1 already cut the K/V traffic ~8×, moving the prefill attention from DRAM-bound to compute/local-mem-bound** — so cutting the chunk count (fewer prefix re-reads) barely helps, and the larger per-block work offsets it. Reverted.
- **⇒ The traffic-reduction lever is EXHAUSTED at M1.** Both M2 axes (GQA reuse, larger chunk) fail for the same reason: M1 is the sweet spot; prefill attention is now compute-bound. Further gains need **compute** optimization (tensor-core/MMA dot products, or a cheaper softmax/scores-I/O), or the weight-GEMM path — a different, larger effort, not a traffic tweak.
- **M3 — (optional) >48 KB dynamic shared** via `cudaFuncSetAttribute` for occupancy — only if M1/M2 profiling shows occupancy, not traffic, as the new limit. New infra; deprioritized.

## 6. Kill criteria

- **M0b:** the online-softmax reassociation flips **any** token on **any** prompt → token-parity unachievable, and byte-identity is capacity-blocked → **NO shippable flash variant, abandon Phase P flash.** (The flat-sum precheck is the immediate stop.)
- **M1/M2:** measured gated speedup < ~1.5× (re-read win eaten by online-rescale/tail/occupancy) → don't ship an opt-in kernel that carries token-parity maintenance risk; or a tiling-specific reassociation fails the oracle beyond M0b → fix or kill.
- **Overriding:** ship only measured, oracle-gated numbers; verify's byte-identity gates green at every milestone or revert.

_Design produced by a 6-scout + synthesis workflow (2026-07-13); all anchors code-verified. M0a evidence in the bundle above._
