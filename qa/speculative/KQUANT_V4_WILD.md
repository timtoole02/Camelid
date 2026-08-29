# K-quant v4 combined-chain prototype — 2026-08-29

## Status

Experimental and opt-in only. Production routing requires:

```text
CAMELID_KQUANT_V4=1
```

The gate has precedence over v2/v3 for Q4_K and Q6_K decode/verify windows
with `n_tokens <= 8`. Wider prefill stays on the established path. The kbench
micro cases select `q4kv4` / `q6kv4` directly and therefore do not read the
production environment variable.

V4 uses one zero-padded 8-column matrix kernel for both `n_tokens=1` and
`n_tokens=2..8`. The preserved harness dispatches that same pipeline once per
column as its single-token oracle, then once for the full window. It asserts
every output word and records both dispatch counters; Q4 and Q6 pass for every
width 2 through 8.

## Arithmetic boundary

This is a new, uncertified arithmetic universe. Q4 stages exact integer matrix
operands in half. Q6 also uses half matrix operands for speed; `scale*q` can
exceed half's consecutive-integer range, so the largest dequantized Q6 values
round. Structural single/multi bit identity is proven, but model quality and a
full warmed losslessness gate are still required before any promotion.

## Exact Llama-3.2-3B shapes, local M4

Best-of-run times include activation staging. Raw row-major packed weights are
used; there is no tile mirror.

| Format / shape | v4 k=1 | v4 k=4 | v4 k=8 | v2 k=8 | k=8 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 up, 8192 x 3072 | 0.309 ms | 0.312 ms | 0.312 ms | 0.503 ms | -38.0% |
| Q4 down, 3072 x 8192 | 0.326 ms | 0.338 ms | 0.328 ms | 0.475 ms | -30.9% |
| Q6 down, 3072 x 8192 | 0.376 ms | 0.379 ms | 0.384 ms | 0.708 ms | -45.8% |
| Q6 head, 128256 x 3072 | 4.389 ms | 4.483 ms | 4.455 ms | 9.532 ms | -53.3% |

The decisive Q6 head streams 323 MB at 72.6 GB/s for k=8, below the requested
6 ms gate. Representative commands:

```text
cargo run --release -- q4kv4 --rows 8192 --nsb 12 --iters 10
cargo run --release -- q4kv4 --rows 3072 --nsb 32 --iters 10
cargo run --release -- q6kv4 --rows 3072 --nsb 32 --iters 10
cargo run --release -- q6kv4 --rows 128256 --nsb 12 --iters 5
```

An exact-f32 Q6 combined-chain precursor reached only 8.21 ms on the head. A
row-coalesced / skewed-threadgroup remap reached 8.38 ms and was reverted. The
half-operand Q6 lane is the change that clears the verifier gate.

## Ceiling, not a whole-model claim

The exact-shape GEMVs project to roughly 42-45 ms of K-quant work per complete
8-column verification window. With attention, normalization, and launch work,
an estimated 48-58 ms round implies a hard all-eight-accepted ceiling near
138-167 tok/s. Roughly four accepted tokens per round is needed to land around
69-83 tok/s. This is only a component projection; acceptance and whole-model
receipts decide the actual speed.
