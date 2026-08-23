# Exact-f32 MoE block-dot experiment: NO-GO

Date: 2026-08-23

Machine: 16 GiB Apple M4

Scope: production Tier-2 argument-buffer K<=8 GateUp and Down kernels

## Hypothesis

Replace only each exact Q4 x Q8 integer block dot with `float4` multiply/adds. The complete 32-term dot is bounded by 32 x 8 x 127 = 32,512, below 2^24, so integer products and their sums remain exactly representable in f32. Bindings, grids, masks, scale order, reductions, and output layout were unchanged.

## Correctness gate

The experimental kernels were compared directly with the shipping kernels for K=1 through K=8 using adversarial signed-int8 inputs. All complete payloads matched byte-for-byte:

- GateUp scales
- GateUp quantized activations
- Down f32 output using an independent extreme-valued activation

## Isolated A/B

The existing 30-layer, K=8, U=30 Metal microbenchmarks alternated baseline and experimental dispatches and reported median command-buffer GPU time across seven samples.

| Stage | Baseline | Exact-f32 | Speedup |
|---|---:|---:|---:|
| GateUp | 36.926 ms | 41.993 ms | 0.879x |
| Down | 16.424 ms | 17.730 ms | 0.926x |

The experiment made GateUp 5.067 ms slower and Down 1.306 ms slower in the representative 30-layer sweep.

## Decision

NO-GO. The default-off production selectors, experimental pipelines, and tests were removed after measurement. `src/metal/spec50_moe_argbuf.rs` was restored byte-for-byte to commit `0c130ee0`.
