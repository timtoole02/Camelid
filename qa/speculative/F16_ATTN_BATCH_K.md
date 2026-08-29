# F16 split-K verifier row batch — 2026-08-29

## Scope and gate

Prototype base: `dedee151` (`perf(eagle3): retain dynamic frontier cursor`).

The verifier's existing F16 split-K attention encoded one partial kernel and one
merge kernel per row per layer. For `k=8` and 28 layers that is 448 dispatches.
This prototype moves verifier row into the Metal grid and emits 56 dispatches,
without sharing arithmetic state between rows. It is opt-in and verifier-only:

```text
CAMELID_METAL_ATTN_BATCH_K=1
```

Eligibility is deliberately narrow: F16 primary or F16 mirrors, split-K, every
row at `position_count >= 128`, and `2 <= k <= 8`. Every other shape takes the
unchanged row-wise path.

## Exactness

The batch metadata retains each row's own `position_count` and `n_splits`, so a
window crossing `pc=128 -> 129` keeps the row-wise two-split/three-split
partition. A packed offset table retains each branching tree row's exact draft
tail slots. The partial flash recurrence and merge loops are otherwise copied
in the same order.

Focused kernel A/B:

```text
cargo test --lib metal_attention_splitk_kv16_batch_matches_rowwise_linear_and_tree -- --nocapture
```

Result: PASS, bit-for-bit, for linear and branching-tree layouts through both
the staged and head-dim-128 direct kernels.

Full F16-primary verifier proof:

```text
CAMELID_METAL_KV_DTYPE=f16 \
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 \
CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_ATTN2=1 \
CAMELID_METAL_ATTN_BATCH_K=1 \
cargo test --release --lib metal_tree_verify_kv16_primary_path_runs \
  -- --ignored --nocapture
```

Result:

```text
LINEAR base=126 k=6 BIT-IDENTICAL
LINEAR base=510 k=6 BIT-IDENTICAL
BRANCH base=126 path=[0, 2, 5] compaction==linear-decode + class-A PASS
BRANCH base=510 path=[0, 2, 5] compaction==linear-decode + class-A PASS
PASS v2_encodes=6 splitk_encodes=30 batch_encodes=8 kv_format=F16
```

With `CAMELID_SPEC_VERIFY_TRACE=1`, the ordinary verify phase receipt now also
reports `attn_batch_layers=N`; `N=0` proves fallback and `N=layer_count` proves
the batch path owned every attention layer in that round.

## Production-shape micro

Local Apple M4, Llama-3.2-3B attention shape (`24q/8kv`, head dim 128), base
4,137, `k=8`, 28 layer repetitions, direct F16 split-K. Best of three GPU-busy
measurements includes both partial and merge kernels:

```text
cargo test --release --lib attention_splitk_kv16_batch_k8_depth4137_probe \
  -- --ignored --nocapture
```

| encode | best GPU busy | best CPU encode | relative GPU |
| --- | ---: | ---: | ---: |
| row-wise | 70.672 ms | 0.163 ms | 1.00x |
| row-dimensional | 41.752 ms | 0.054 ms | 1.693x |

Individual paired GPU ratios were 1.715x, 1.624x, and 1.621x. The first run
under concurrent GPU load was also positive (1.490x best: 139.457 -> 93.623
ms), so the verdict does not depend on the faster repeat. The probe asserts the
final output words are identical before timing. This is an attention-only lower
bound, not an end-to-end tok/s claim; full verifier benefit is capped by the
remaining K-quant projections and learned-head work.
