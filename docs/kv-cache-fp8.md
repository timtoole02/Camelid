# FP8 KV cache (E4M3 / E5M2)

`--kv-quant fp8_e4m3` and `--kv-quant fp8_e5m2` store the KV cache as 8-bit
floating point with an f16 per-block scale, in the same 34-bytes-per-32-values
container as `q8_0`.

## What it is and is not

**It is not a memory win over `q8_0`.** Both formats are
`{ scale: u16, qs: [u8; 32] }` — 34 bytes per 32 values, byte-for-byte identical
to `BlockQ8_0`. `allocated_bytes()` and `kv_bytes_per_token()` return the same
numbers for all three. If you want a smaller cache than `q8_0`, the answer is
`q4_0` (18 bytes / 32 values), not FP8.

**It is a different error distribution at the same size.** `q8_0` spends all 8
bits on a uniform grid across `[-amax, amax]`. FP8 spends 4 (E4M3) or 5 (E5M2)
of them on an exponent, buying relative precision across a wide dynamic range at
the cost of mantissa bits. Which one wins depends entirely on how the values in
a block are distributed:

| block content | Q8_0 | FP8 E4M3 | FP8 E5M2 |
|---|---|---|---|
| smooth / Gaussian | **0.54%** | 2.43% | 4.80% |
| one 30x outlier per 32 | 1.26% | **0.49%** | 0.97% |
| one 100x outlier per 32 | 1.28% | **0.15%** | 0.30% |
| log-normal magnitudes | 0.82% | **0.63%** | 1.32% |
| student-t(2) heavy tail | **1.07%** | 1.32% | 2.62% |

Dot-product RMSE against an f32 reference, head_dim 128, 8000 trials per regime.
Lower is better; bold is the winner at equal bytes.

The mechanism is straightforward. A single large outlier sets `amax` for the
whole block, so `q8_0`'s uniform step becomes `amax/127` and every ordinary value
in that block collapses onto a handful of levels. FP8's exponent field keeps its
relative precision regardless of what else shares the block. On smoothly
distributed data there is no outlier to absorb and the extra mantissa bits win
instead.

`tensor::kv_quant::tests` pins both directions of this so neither claim can rot.

## What happened on a real model

The synthetic advantage did **not** transfer. Greedy 400-token generation,
Llama 3.2 3B Instruct Q8_0, identical prompt, every lane pinned to the CPU
(`scripts/kv-fp8-receipt.sh`):

| lane | agreement with the f16 reference |
|---|---|
| `q8_0` | **token-identical** for the whole generation |
| `q4_0` | diverges at byte 80 |
| `fp8_e4m3` | diverges at byte 80 |
| `fp8_e5m2` | diverges at byte 80 |

f16 and `q8_0` produce "...because it reduces the number of cache entries..."; all
three other lanes produce "...by reducing...". Both FP8 formats depart from the
reference exactly as early as `q4_0`, which uses **half** the memory.

Two things follow. First, the lane is demonstrably live — FP8 changes real output,
so the flag reaches the KV cache rather than being silently ignored (`q4_0` is
included as the control that proves the harness can detect a difference at all).
Second, and more important: **on this model FP8 buys nothing.** Llama 3.2 3B's KV
rows evidently look smooth rather than outlier-heavy, which is the regime the
table above says `q8_0` wins.

Divergence from a greedy reference measures *sensitivity*, not quality — an early
divergence does not by itself mean worse text. But it does mean FP8 perturbs the
cache more than `q8_0` does at identical cost, which is the opposite of a reason
to switch.

**So `q8_0` remains the right choice among the 8-bit options today.** FP8 is
justified only on a model whose KV is measurably outlier-heavy, and no such model
has been demonstrated here. Anyone proposing FP8 as a default should produce this
same receipt showing it beating `q8_0`, not a synthetic block.

## Limits worth knowing

**CPU lanes only.** The resident CUDA decoder honours `f16` and `q8_0`; every
other format — including the already-shipped `q4_0` — has its GPU-side KV
allocated and run as f16 (`src/cuda_resident.rs`, the `_ => kv_width * 2` arm).
The host-side cache is genuinely FP8-sized, so the host saving is real, but do
not expect a VRAM change.

This is the limitation worth understanding, because the GPU is where quantized
KV actually pays. There the baseline being displaced is f16 at 2 bytes per
element, not an already-8-bit format, so an 8-bit KV cache is a real ~50% VRAM
cut — which on a 6 GB card converts directly into usable context length. That
needs FP8 KV kernels written in `src/cuda_resident.rs` and is **deliberately out
of scope here**: it is tracked as a separate workstream with its own PR, so that
the CPU format work and the GPU kernel work can be reviewed and validated
independently.

**No SIMD path.** `q8_0` and `q4_0` have AVX2 `vec_dot` / `axpy` kernels; FP8 does
not. The scalar FP8 kernels are at parity with the *scalar* `q8_0` kernel, but on
x86 `q8_0` still reaches the vector path and FP8 does not, so FP8 decode is
slower in practice on this hardware.

Getting to that parity mattered. An FP8 byte has 256 possible values, so the row
kernels index `FP8_E4M3_LUT` / `FP8_E5M2_LUT` — 1 KiB tables built at compile
time from the decoder functions themselves, which keeps them from drifting.
Measured on a 128-dim KV row, best of 15 interleaved repetitions:

| decode strategy | ns / row | vs scalar Q8_0 |
|---|---|---|
| `(2.0f32).powi(exp - 7)` per element | 950.6 | 15.9x |
| branch-light bit assembly | 140.2 | 2.4x |
| compile-time lookup table | **58.7** | **0.99x** |

The `powi` call ran once per element per position per head per layer in the
attention inner loop, which is how a decode kernel ends up costing sixteen times
the entire dot product it feeds.

**Dynamic range floors.** The block scale is stored as f16 and clamped at f16's
smallest positive value (2^-24), so a block survives only while `amax / 2^-24`
still reaches the format's smallest subnormal: about **1.2e-10** for E4M3 and
**9e-13** for E5M2. Below that the block reads back as zeros. Before the clamp
those floors were 1.34e-5 and **1.71e-3** respectively — the latter well inside
the range of ordinary KV magnitudes, which silently deleted those positions from
attention. `fp8_dynamic_range_floors_are_where_the_formats_run_out` pins both.

**E4M3 is E4M3FN.** It has no infinity encoding: overflow saturates at ±448, and
`S.1111.111` is the sole NaN pattern. E5M2 keeps IEEE semantics (±Inf at
exponent 31, mantissa 0). `fp8_decoders_match_the_arithmetic_definition_for_all_256_codes`
pins every code against the arithmetic definition of each format.

## Reproducing the numbers

Synthetic regime comparison and the kernel microbenchmark are standalone; the
end-to-end token-identity receipt is:

```bash
cargo build --release --bin camelid
scripts/kv-fp8-receipt.sh /path/to/model.gguf 96
```

That runs one greedy generation per format against the same model and prompt,
pinned to the CPU so the lanes are comparable, and reports how far each
quantized lane's output drifts from the f16 reference.
