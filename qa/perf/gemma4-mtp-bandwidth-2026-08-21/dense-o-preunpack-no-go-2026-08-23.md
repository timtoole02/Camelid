# Gemma 4 26B dense O pre-unpack: NO-GO

Date: 2026-08-23

Branch under test: `codex/gemma4-50tps-v2`

Scope: isolated Metal microbenchmark only; no server and no full-model run

## Question

Can the strict K=8 verifier accelerate the dense attention output projections
by replacing Q4_0 nibble unpacking in the kernel with a resident representation
containing one signed byte per weight plus the original f16 scale?

The candidate preserved the production exact-partition program selected by
`CAMELID_GEMMA4_DENSE_K8_GENERIC=1`: runtime K loop, row/lane assignment,
increasing-block accumulation, f32 operations, SIMD reduction, and stores were
unchanged. Only `(nibble - 8)` was materialized ahead of time as i8. Converting
that range `[-8, 7]` to f32 is exact, and the scale bits remained f16.

## Production seams audited

- Common Q4 tensor pages are assembled per layer in
  `src/gemma4_runtime.rs::prepare_ghost_common_metal`, then passed through
  `build_ghost_common_resident_layer`.
- `Gemma4ResidentLayer::from_wire_pages_owned_with_rope` creates the no-copy
  Metal views. `o_pages` becomes `o_w` alongside Q/K/V and shared-MLP weights.
- The chained verifier dispatches local fused QKV (or global Q and K separately)
  before attention. After attention it dispatches every layer's O projection via
  `encode_gemma4_q4_0_matmul_batch_k`.
- The strict gate forces K=8 through the runtime-width plain dense pipeline, so
  a candidate must compare against that pipeline, not the legacy static-K8 one.

Historical commit `07e8deb7` (`agent/spec50-unpack`) was not suitable for this
decision: it added additive scaffolding without production dispatch, and its
K=8 dense tests reached the legacy static-K8 wrapper by default rather than the
strict runtime-width partition oracle.

## Exact footprint

The 26B O family contains 25 local projections of shape
`2816 x 4096` and five global projections of shape `2816 x 8192`:

| Measure | Exact value |
|---|---:|
| Q4_0 blocks | 12,615,680 |
| Packed wire (18 B/block) | 227,082,240 B / 216.5625 MiB |
| i8 values + f16 scales (34 B/block) | 428,933,120 B / 409.0625 MiB |
| Net growth if packed GPU view were replaced | 201,850,880 B / 192.5 MiB |
| Extra allocation while both forms are retained | 428,933,120 B / 409.0625 MiB |

For context, applying the same 34-byte representation to every dense Q4 matrix
would allocate 1,748,285,440 B (about 1.748 GB decimal) while the packed views
remain live. The isolated O family was chosen to make the smallest meaningful
resident experiment.

## Validation and benchmark

The temporary probe was default-off and never connected to production. Before
removal it passed:

- host operand reconstruction for every packed nibble in a randomized tensor,
  including unchanged f16 scale bits;
- fail-closed size checks;
- exact footprint assertions;
- Metal raw-f32-bit comparison against the shipped runtime-width kernel for
  K=1 through K=8 at ragged and loop-carried shapes: zero mismatches.

The performance receipt used 30 distinct matrices with the exact 25-local /
5-global shape mix. Both representations remained resident, all 30 dispatches
were encoded into each measured sweep, K was 8, and two A/B orders were timed
after warm-up. This isolates GPU kernel time and avoids file I/O or server work.

| Variant | Aggregate median |
|---|---:|
| Packed runtime-width Q4_0 | 10.245 ms |
| Pre-unpacked i8 + f16 | 10.758 ms |
| Ratio | 0.952x |
| Delta | +0.513 ms (slower) |

The temporary test run completed with four active tests passing and the armed
ignored benchmark passing. `git diff --check` was clean. Swap I/O remained at
41 swap-ins / 440 swap-outs before and after the isolated run.

## Decision

NO-GO. The representation is bit-exact, but it nearly doubles projection
weight traffic, slows this real 30-layer family by about 5%, and consumes
409 MiB while the packed source remains available. Do not add a production
selector, persistent buffers, or loader integration for dense pre-unpack.
The next credible dense work must reduce arithmetic or scheduling cost without
expanding streamed weight bytes.
