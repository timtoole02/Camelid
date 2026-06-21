# Camelid GPU decode-speed campaign — Qwen3-4B Q8_0 vs llama.cpp

Branch `perf/decode-attention`. Host: RTX 3060 Laptop GPU, 6 GB, CUDA 12.9. Comparator:
llama.cpp @ acd79d603 (`llama-bench`). Model: Qwen3-4B Q8_0 (GPU-resident, greedy).
Tooling: Nsight Systems 2025.1.3, Nsight Compute 2025.2.1, a STREAM bandwidth microbench.

## Phase 0 — roofline baseline

Achievable DRAM bandwidth (empirical STREAM): **273.3 GB/s**. Decode at low context:

| engine | tok/s | achieved GB/s | roofline % |
|--------|------:|--------------:|-----------:|
| Camelid (baseline) | 39.65 | 169.7 | 62.1% |
| llama.cpp | 54.42 | 232.9 | 85.2% |

ratio 0.729 → recoverable → Phase 1. (llama at 85% of the measured 273 cross-validates it.)

## Phase 1 — bound classification (measured)

| kernel | ncu DRAM% | occupancy | waves/SM | verdict |
|--------|----------:|----------:|---------:|---------|
| q8_gemv (matmuls) | ~76% | 61% | 10.1 | already efficient — leave alone |
| attention_decode | **0.44%** | **4.4%** | **0.07** | occupancy-starved — the target |

attention_decode ran 1 block/head x 64 threads, single-thread softmax, O(context) cost.
The prior "CUDA graphs gave no benefit -> stuck" was explained, not inferred: the cost is
attention EXECUTION, not launch overhead.

### Depth is the real gap

llama.cpp decode is ~flat with context (flash-decode); Camelid's collapsed:

| context | Camelid baseline | llama.cpp | ratio |
|--------:|-----------------:|----------:|------:|
| ~70 | 39.65 | 54.65 | 0.73 |
| ~1881 | 17.1 | 51.14 | **0.33** |
| ~3000 | (6 GB OOM-thrash) | 49.61 | — |

The headline gap is a 3x deficit at realistic chat context, not the ~25% at tg128.

## Phase 2 — optimization

Parity gate (constraint #1): greedy `first_divergent == -1` vs the parity-green baseline
(== llama.cpp acd79d603), verified by token-id diff (and CPU `--deterministic` at depth).

| stage | change | low-ctx | depth ~1881 | parity | status |
|-------|--------|--------:|------------:|--------|--------|
| 1 | adaptive block_dim (bit-identical) | 41.4 | — | PASS | folded into 2 |
| 2 | weighted-V parallel split (token-parity) | 41.65 | 22.19 | PASS | committed 8d65b8b6 |
| 3 | warp-coalesced K/V (token-parity attempt) | ~40.9 | 26.81 | **FAIL** (flip @tok26) | **REVERTED** |
| 4 | **split-K flash-decode (token-parity)** | 41.63 | **26.23** | PASS | **committed 57977f6e** |

### Stage 3 negative result (coalescing) — kept as evidence

Full warp-coalescing (one warp per key position; consecutive K/V reads) measured +21% at
depth (26.81) with coherent output, but **broke parity**: greedy matched the baseline for
tokens 0-25 then flipped at token 26. 25 identical tokens before the flip ⇒ FP
**re-association near-tie flip, not a bug**. Coalescing the dot *requires* interleaving the
head_dim sum across warp lanes (re-association), which tips near-tie greedy tokens. The
score dot feeds softmax/exp, so it is more sensitive than Stage 2's weighted-V split.
Reverted per the non-negotiable losslessness rule.

### Stage 4 split-K flash-decode — the parity-safe fix (shipped)

grid = n_heads x n_splits (one block per (head, position-chunk)) so the 30 SMs fill even
with 32 heads. Used above 512 keys; below, the one-block-per-head launch stays. Three
passes: (1) sequential d-order dot (bit-identical) + chunk max; (2) exp with the EXACT
global max (bit-identical) + chunk exp-sum + chunk unnormalized weighted-V; (3) ordered
combine. TOKEN-PARITY: only the cross-chunk sum re-associates (same class as Stage 2,
which passed) plus a 1/s factored out; the dot and global max are exact. True bit-identity
is impossible for a split sequential reduction (re-association is unavoidable).

Key point: split-K reaches the speed of the reverted coalesced kernel (26 vs 26.8)
WITHOUT the token flip, because it fixes occupancy (the parity-safe half) while keeping
the exact dot. The remaining gap is the uncoalesced K/V bandwidth, which is parity-locked.

ncu confirms the mechanism (attn_sk_partial @ ctx 1881):

| metric | original attention_decode | split-K |
|--------|--------------------------:|--------:|
| grid (blocks) | 32 | 256 (= 32 heads x 8 splits) |
| waves / SM | 0.07 | 1.42 |
| achieved occupancy | 4.4% | ~42% |
| DRAM throughput | 0.44% | ~5% |

Occupancy is solved (0.07 -> 1.42 waves; SMs now filled). DRAM stays ~5% because the
reads are still uncoalesced — that is the parity-locked residual: coalescing would lift
it but re-associates the dot and flips near-tie tokens (Stage 3).

## Final benchmark (split-K, committed) vs llama.cpp

| context | Camelid baseline | Camelid split-K | llama.cpp | ratio (split-K) |
|--------:|-----------------:|----------------:|----------:|----------------:|
| ~7 | 39.65 | 41.63 | 54.65 | 0.76 |
| ~659 | — | 35.10 | ~53.4 | 0.66 |
| ~1881 | 17.1 | **26.23** | 51.14 | **0.51** (was 0.33) |

Parity verified: ctx 7 == baseline; ctx 659 (split-K, 3 splits) CUDA == CPU == llama.cpp.

## Honest conclusion

- **Depth (~0.33x -> 0.51x):** split-K roughly halves the deficit at realistic context,
  parity-safe. Cumulative low->shipped at ctx 1881: 17.1 -> 26.2 tok/s (+53%).
- **Low ctx / tg128 (~0.76x):** floor-bound by q8_gemv (already ~76% of bandwidth; the
  maintainers could not improve it via occupancy). 0.95x is not reachable here without a
  faster dequant-GEMV or a smaller quant (not apples-to-apples).
- **The binding limit is the parity gate, not the kernel.** Fully matching llama.cpp's flat
  depth curve needs coalesced K/V reads, which re-associate the dot and flip near-tie
  tokens (Stage 3). llama.cpp's flat curve is itself *enabled* by non-bit-exact flash
  attention. Split-K extracts the maximum occupancy win that strict token-parity allows.

### Out-of-decode-scope issues surfaced (flagged, not addressed)
- Prefill ~34x slower than llama (token-by-token, not the batched path by default).
- ctx ~4k decode OOM-thrashes on the 6 GB card.
- attention_batched (spec-verify) should mirror the decode reorder for spec-decode
  losslessness; greedy decode (default) is unaffected.

## Commits (branch perf/decode-attention)
- 8d65b8b6  Stage 2: parallelize attention_decode (weighted-V split)
- 57977f6e  Stage 4: split-K decode attention (fill SMs at depth)
