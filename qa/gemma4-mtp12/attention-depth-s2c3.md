# Exact V2 attention depth candidate

Explicit selector: `CAMELID_GEMMA4_DENSE_ATTN_ROWS_V2_VARIANT=2,3`, with V2 enabled. The default remains `0,0`.

Scores form 2 shares eight query pairs across four SIMD groups per threadgroup. Each lane still computes one score with the original scalar, sequential dot product. The HD256 tile occupies 8 KiB, and the HD512 tile occupies 16 KiB. The new wrapper initializes unused query slots on narrow verifier tails. Context form 3 instantiates the existing exact context template with two pairs for HD256; HD512 retains the established nest. Softmax, reciprocal, product and ascending-position value-fold expressions are unchanged.

On local Apple M4, W8, the final patched shader gave:

| Base position | V2 default attention ms | `2,3` attention ms | Saved ms |
| ---: | ---: | ---: | ---: |
| 128 | 4.902 | 3.517 | 1.385 |
| 512 | 17.223 | 12.353 | 4.870 |
| 529 | 17.677 | 12.620 | 5.057 |
| 600 | 20.382 | 14.361 | 6.022 |
| 721 | 24.060 | 17.184 | 6.876 |
| 768 | 25.841 | 18.348 | 7.493 |
| 1024 | 34.302 | 24.684 | 9.618 |

These are sums of separately timed full score/softmax/context command buffers for 40 sliding and 8 global layers, rotating 8 and 2 independent KV-cache pairs. They use the actual 12B geometries (16 query heads; sliding 8 KV heads × 256 dimensions, global 1 × 512), a 1024 sliding window and capacity 2048. Two warmups precede nine samples per form, with rotating form order. They are kernel measurements, not model throughput; full decoder cache interference and mini2 request performance still require qualification.

The production-library `metal_gemma4_dense_attention_rows_v2_variant_matrix` passes all 12 selectable forms against established pre-V2 row attention on 23 production geometries. Its expanded fixtures include base 529, 128/512/768, and exact sliding boundaries 1023/1024/1025, plus existing K1/K2/K4/K8/K16 and causal-poison cases. The separate patched-shader fixture covers widths 1/2/4/8/16 at 12 depths, including ragged counts and those boundaries, comparing raw scores, exponentials, denominator and output bits. All 480 width/depth/geometry/form cases passed.

Reproduce the production-library gate under the local GPU lock:

```sh
~/bin/cam-lock.sh env CARGO_TARGET_DIR=/Volumes/Untitled/cargo-targets/global \
  cargo test --lib metal_gemma4_dense_attention_rows_v2_variant_matrix -- --nocapture
```

Standalone source, harnesses, full sample logs and summaries are retained under `/Users/timtoole/Documents/Codex/2026-09-04/i-n/work/attention-depth/`. The final patched-source timing is `production-fixture-w8.log`; the production-library gate is `production-matrix.log`. No full model was run during this local kernel qualification.
