# Transposed Q8 CUDA measurements

Request tokens/second, median of three warmed uncached 32-token requests. Each prompt/mode reuses one process after an eight-token warmup. Both Camelid policies use the same executable. Loading, NVRTC compilation and warmup are excluded; prefill and generation are included. See [README](README.md) for the hardware, entry-point differences, and 27B memory strategies.

| Model | Prompt | Float ordinary | Q8 ordinary | Q8 / float | Float draft | Q8 draft | Q8 draft / ordinary | Port ordinary | Port draft |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B | count | 64.77 | 141.28 | 2.18x | 41.16 | 80.41 | 0.57x | 194.66 | 79.22 |
| 2B | code | 64.73 | 141.51 | 2.19x | 41.11 | 78.90 | 0.56x | 177.83 | 65.25 |
| 2B | explain | 66.57 | 145.74 | 2.19x | 42.69 | 82.32 | 0.56x | 191.53 | 52.92 |
| 4B | count | 29.85 | 61.87 | 2.07x | 27.24 | 47.13 | 0.76x | 118.53 | 69.82 |
| 4B | code | 29.95 | 62.94 | 2.10x | 27.25 | 41.85 | 0.66x | 109.42 | 66.20 |
| 4B | explain | 31.10 | 66.49 | 2.14x | 12.19 | 22.41 | 0.34x | 117.98 | 45.65 |
| 9B | count | 16.05 | 39.72 | 2.47x | 16.54 | 39.64 | 1.00x | 68.57 | 42.46 |
| 9B | code | 16.07 | 39.72 | 2.47x | 16.60 | 39.34 | 0.99x | 64.72 | 36.09 |
| 9B | explain | 16.77 | 39.68 | 2.37x | 10.06 | 23.62 | 0.60x | 67.29 | 28.37 |
| 27B | count | 3.22 | 5.33 | 1.66x | 5.17 | 8.07 | 1.51x | 1.46 | 1.01 |
| 27B | code | 3.90 | 6.03 | 1.54x | 5.32 | 8.05 | 1.34x | 1.63 | 1.10 |
| 27B | explain | 3.98 | 5.10 | 1.28x | 3.71 | 4.55 | 0.89x | 1.66 | 0.80 |

Q8 changes activation arithmetic. Ordinary/draft equality is required within each Camelid policy; cross-policy and cross-engine differences are recorded separately in `optimization.json` and `comparison.json`.

## Fixed-continuation numerical check

Each model scores the same 96 continuation tokens under both arithmetic policies. This is a bounded regression check, not a general task-quality or corpus-perplexity evaluation.

| Model | Float mean NLL | Q8 mean NLL | Increase (nats/token) | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| 2B | 0.140591 | 0.140708 | 0.000118 | 100.00% |
| 4B | 0.124100 | 0.124834 | 0.000734 | 100.00% |
| 9B | 0.127993 | 0.128087 | 0.000093 | 100.00% |
| 27B | 0.138858 | 0.139552 | 0.000695 | 100.00% |
