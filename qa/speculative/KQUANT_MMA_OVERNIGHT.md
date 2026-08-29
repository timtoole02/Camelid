# K-quant MMA follow-up — 2026-08-29, mini2

## Outcome

No kernel change was promoted. The committed `f416cab2` baseline was rebuilt
from a clean source export and reproduced at **13.50 plain / 30.50 speculative
tok/s**, 91.57% acceptance, and `lossless=true` on the 4137-token Llama-3-8B
Q4_K_M gate. The production branch was restored after every rejected candidate.

The useful result is a closed optimization path and a preserved benchmark:
direct `simdgroup_matrix::thread_elements()` access is fast in isolation, but
is not safe enough for this lossless lane on M4 under the full warmed workload.

## Direct-fragment experiment

The candidate removed each MMA result's `simdgroup_store` to threadgroup RAM,
barrier, and reload. It consumed the two lane-owned fragment elements directly
in the ordered f32 tail.

Representative micro results:

| Kernel/shape, k=8 | committed | direct fragment | delta |
| --- | ---: | ---: | ---: |
| Q4, rows=14336, n_sb=16 | 0.943 ms | 0.880 ms | -6.7% |
| Q6, rows=4096, n_sb=56 | 1.596 ms | 1.149 ms | -28.0% |

The synthetic harness reported bit identity for k=2..16 at up, down, output
head, and ragged shapes. The in-tree `metal_kquant_` release tests also passed.
Those gates were insufficient:

| Full 4k gate | spec tok/s | acceptance | first divergence | result |
| --- | ---: | ---: | ---: | --- |
| committed warm baseline | 30.50 | 91.57% | -1 | pass |
| Q4+Q6 direct, warm | 25.11 | 75.82% | 0 | reject |
| Q4+Q6 direct, stronger TG barriers, warm | 24.97 | 75.82% | 0 | reject |
| Q4-only direct, warm | 24.81 | 83.33% | 11 | reject |

The raw JSONL rows, in the same order as the baseline and three rejected warm
rows above, are preserved in
`receipts/m2_overnight_fragment_rejection.jsonl`.

A Q4+Q6 no-warm run happened to pass losslessness, while the warmed repeats did
not. That makes short or no-warm runs explicitly non-certifying for this path.

## Other rejected variants

- Direct weight decode into A fragments: Q4 k=8 regressed to 1.134 ms.
- Broadcast of Q4 scale/min headers: erased the fragment gain (~0.941 ms).
- Half-super-block Q6 staging: exact, but regressed k=8 to 1.656 ms and deep
  windows further.
- Paired Q6 packed loads: exact but measured 1.592-1.604 ms across repeats,
  indistinguishable from the 1.596 ms baseline.

## Rule for the next pass

Do not optimize away the matrix store boundary using an assumed per-lane
fragment mapping. Keep the full warmed model gate mandatory after micro and unit
identity. The remaining target is still K-quant MMA, but the safer next design
needs to retain an explicit, synchronized materialization boundary or establish
a model-level proof for any replacement.
