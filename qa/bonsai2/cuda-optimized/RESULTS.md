# CUDA optimization measurements

Median request tokens/second across three warmed repetitions, including uncached prefill. Same executable for the before/after control; 32 generated tokens, 2048 context slots, 8 threads, and a three-token Bonsai 2 2B drafter. See [README](README.md) for hardware and timing limitations.

| Model | Prompt | Before ordinary | After ordinary | Ordinary gain | After draft | Draft / ordinary | Port ordinary | Port draft |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B | count | 29.17 | 65.45 | 2.24x | 41.39 | 0.63x | 201.80 | 81.18 |
| 2B | code | 29.18 | 65.35 | 2.24x | 41.22 | 0.63x | 208.80 | 74.40 |
| 2B | explain | 29.89 | 67.50 | 2.26x | 42.89 | 0.64x | 198.08 | 58.34 |
| 4B | count | 13.35 | 29.90 | 2.24x | 27.38 | 0.92x | 116.89 | 71.09 |
| 4B | code | 13.33 | 30.10 | 2.26x | 27.52 | 0.91x | 115.25 | 62.64 |
| 4B | explain | 13.67 | 31.47 | 2.30x | 12.32 | 0.39x | 121.05 | 45.99 |
| 9B | count | 7.23 | 16.10 | 2.23x | 16.39 | 1.02x | 77.60 | 62.35 |
| 9B | code | 7.21 | 16.15 | 2.24x | 16.75 | 1.04x | 77.65 | 60.89 |
| 9B | explain | 7.39 | 16.72 | 2.26x | 10.03 | 0.60x | 78.84 | 39.26 |
| 27B | count | 2.14 | 3.45 | 1.61x | 5.22 | 1.52x | 1.61 | 1.09 |
| 27B | code | 2.08 | 3.82 | 1.83x | 5.23 | 1.37x | 1.60 | 1.02 |
| 27B | explain | 2.01 | 3.82 | 1.90x | 3.93 | 1.03x | 1.66 | 0.81 |

All ordinary and drafted native token sequences match the original kernel for these cases. Raw repetitions, acceptance counts, placement, and cross-engine parity are retained in the JSON receipts. These bounded checks do not establish identity for arbitrary prompts.
