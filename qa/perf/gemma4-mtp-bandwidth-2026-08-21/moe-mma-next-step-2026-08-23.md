# Gemma 4 K8 MoE: exact GateUp MMA NO-GO receipt

Date: 2026-08-23

Baseline audited: `0c130ee0` (`src/metal/spec50_moe_argbuf.rs`)

Status: NO-GO; experimental shader/PSO/selector/tests removed, production source restored byte-for-byte

## Decision

Both strict GateUp MMA layouts were bit-exact, but neither cleared the required
10 ms saving per K8/U30/30-layer pass. The faster staged layout saved 5.122 ms;
the follow-up direct-fragment layout saved only 3.436 ms. Per the promotion
contract, all MMA shader, PSO, flag, dispatch, parity-test, and benchmark-test
integration was removed. `src/metal/spec50_moe_argbuf.rs` is byte-identical to
HEAD blob `c6159d120b661069dabb69a47b7fdc82197c9013`.

The analysis below is retained as design evidence, not as an implementation
directive.

The next bounded kernel experiment should be a strict, default-off K8
`simdgroup_matrix` GateUp kernel. It must leave the Q4 wire records and argument
buffer bindings unchanged, use half matrices only for the *integer* 32-value
dot, and execute the existing floating-point scale/accumulate, GeGLU, max, and
quantization program afterward.

If that clears its isolated timing gate, the matching Down experiment should
use an on-chip `i16` term tile and then replay the current lane-local floating
reduction exactly. A direct expert-major Down accumulator is not admissible: it
was already both non-exact and slower.

The smaller exact-f32-dot experiment has now completed and is a NO-GO: it was
bit-exact, but GateUp regressed from 36.926 to 41.993 ms and Down from 16.424 to
17.730 ms in their isolated 30-layer sweeps. Its selectors and PSOs were
removed and the source was restored to `0c130ee0`. MMA is still distinct: it
uses the matrix pipe and radically changes instruction count instead of merely
moving the same vector dot from integer to scalar/vector f32 issue.

## Measured constraints

The current six-round profile attributes about 74.8 ms of each K8 GPU round to
the aggregate MoE-tagged region. Its logical expert-weight traffic is about
4.066 GB per round:

| Stage | Current logical weight traffic | Current production geometry |
|---|---:|---|
| GateUp | about 1.925 GB/round | `U * 22` groups, 32 threads; one lane/FF row; 88 Q4 blocks; active candidate mask |
| Down | 2.141 GB/round | 704 groups, `32 * K` threads; one SIMDgroup/candidate; four output rows; 176 route-block terms |

At the M4's roughly 120 GB/s memory ceiling that traffic alone is about 33.9
ms. Therefore scalar scheduling tweaks cannot produce a 47 ms whole-round
target. The MoE kernel must reduce instruction issue, and Down should also move
toward the expert-union traffic floor.

The local machine is a 10-GPU-core base M4 with 16 GB unified memory. Direct
Metal device queries report:

- SIMD width: 32 for all relevant compiled pipelines.
- Maximum threadgroup memory: 32,768 bytes.
- Device threadgroup width limit: 1,024 threads; every proposed PSO still has
  to validate its own `maxTotalThreadsPerThreadgroup`.
- Tier-2 argument buffers.
- Recommended working set: 12,713,115,648 bytes (11.84 GiB).
- Maximum single buffer: 9,534,832,640 bytes (8.88 GiB).

Those limits rule out a persistent pre-unpacked expert copy. Every proposed
tile below expands values only in the 32 KiB threadgroup store.

## What has already been falsified

Do not repeat these shapes under new names:

- Candidate-sliced GateUp: exact, but repeated warm weight reads did not beat
  the current K8 kernel materially.
- Four-candidate GateUp: exact, but the second weight pass was slower.
- Exact-f32 block dots: exact for K=1 through 8, but GateUp measured 0.879x and
  Down 0.926x; the implementation was removed after measurement.
- Eight-row Down and four-candidate-per-threadgroup Down: exact, no gain.
- Threadgroup-staged scalar weights: exact, but GateUp was 1.8x slower and Down
  1.1x slower from barriers, bank conflicts, and lost occupancy.
- Unique-expert/rank-indexed Down accumulation: 1.6x slower and up to 896 ULP
  wrong because it separated a product that the reference contracts into the
  lane-local accumulator.
- Exact dense activation tiling is useful context: SG=4/TB=0 measured 37.912 ms
  versus 38.196 ms (1.007x, noise), while staged variants measured 0.438x to
  0.964x. Staging alone is not the lever.

The MMA proposal differs in the essential way: its threadgroup expansion feeds
the matrix pipe and removes most scalar integer multiply/unpack issue. It is
not justified merely by fewer device reads.

## Repository idioms to reuse

The implementation should copy proven mechanics, not invent a second fragment
layout:

- `src/metal.rs`, `q8_0_block_wire_mm`: swizzle values into contiguous 8x8
  threadgroup blocks, use `simdgroup_half8x8` operands with
  `simdgroup_float8x8` accumulators, and avoid strided/transposed fragment
  loads. The existing note records a material regression for strided fragment
  loads.
- `src/metal.rs`, `steel_q8_mm`: use the repository's `frag_coord` mapping and
  `thread_elements()[0..2]` to consume two output cells per lane without a
  fragment store/reload. Its padded leading dimension is also the preferred
  fallback if the dense-block swizzle shows bank conflicts.
- `src/metal/spec50_dense.rs`: put
  `[[max_total_threads_per_threadgroup(...)]]` on each entry point and reject a
  compiled PSO whose SIMD width or maximum width does not match the dispatch.
- `src/metal/spec50_moe_argbuf.rs`: compile with fast math disabled. That is a
  parity requirement, not a tuning preference.

## A. GateUp MMA kernel

Proposed symbol:

`spec50_moe_argbuf_gateup_geglu_quant_batch_k8_mma8x8`

### Arithmetic contract

For each 32-value Q4/Q8 block, stage only the unscaled integer operands:

- Q4 code: `nibble - 8`, in `[-8, 7]`.
- Q8 activation: in `[-127, 127]`.
- Each product has magnitude at most 1,016 and is exactly representable in
  half.
- The complete block dot has magnitude at most 32,512 and is exactly
  representable in f32.

Thus four half-input/f32-accumulate 8x8 MMAs produce the exact same integral
f32 value as `float(isum)`, regardless of the MMA's integer-term association.
No weight or activation scale may enter the matrix multiply. After extracting
the two dot cells owned by a lane, the kernel must execute, in ascending `gb`
order, the exact existing statements:

```text
gate_acc += (dot_gate * w_scale_gate) * in_scale
up_acc   += (dot_up   * w_scale_up)   * in_scale
```

The GeGLU expression, `fabs`/max reduction, f16 scale round-trip, inverse,
`round`, clamp, and destination indices remain unchanged.

This proof applies to finite model scales. Tests must still include zero,
signed zero, smallest finite f16 scales, maximum finite f16 scales, and all
Q4/Q8 extrema. Any non-finite-scale case should be refused by model validation,
not normalized in this kernel.

### Grid and on-chip layout

- Grid: `U * 22` threadgroups, exactly the current logical grid.
- Width: 128 threads = four SIMDgroups.
- One threadgroup owns one `(unique expert, 32-row FF block)`.
- SIMDgroup `sg` owns rows `8*sg .. 8*sg+7` and the canonical eight candidate
  positions.
- Inactive candidate rows are staged as integer zero. This preserves canonical
  output indexing and avoids a second compaction table. The expected U=30/K=8
  routing has only 64 live expert/candidate pairs, so the first receipt must
  report the mask-popcount distribution; the matrix performs up to
  `8*U/64 = 3.75x` extra integer MACs at U=30.

Use disjoint per-SIMDgroup threadgroup regions so the 88-block hot loop needs
only SIMDgroup barriers:

| Region | Bytes |
|---|---:|
| Candidate Q8 tile, `4 * (8 x 32 x f16)` | 2,048 |
| Gate Q4 integer tile, `4 * (32 x 8 x f16)` | 2,048 |
| Up Q4 integer tile, `4 * (32 x 8 x f16)` | 2,048 |
| Per-row scales and final cross-SG max/scale scratch | less than 512 |
| Total | less than 6.5 KiB |

Each SIMDgroup loads one activation fragment and one Gate/Up weight fragment
for each of the four K-octets, then issues two MMAs. Persistent per-lane state
is four f32 accumulators (Gate/Up for two output rows), not the current sixteen.

At the end, the repository fragment-coordinate mapping gives one candidate and
two rows to each lane. Reduce the two local activations across the four lanes
that share a candidate within each SIMDgroup, write `4 x 8` maxima, perform one
uniform threadgroup barrier, and have SIMDgroup 0 derive the eight 32-row
scales. A second uniform barrier publishes those scales for quantization.
For finite nonnegative `fabs` results, max is association-independent; the raw
scale/quants test remains authoritative.

### Traffic and occupancy expectation

Global expert bytes remain at the current GateUp union floor; no record layout
or resident allocation changes. The win is from replacing scalar int4 issue,
removing repeated activation load instructions, and cutting live accumulator
state. A 128-thread group with less than 6.5 KiB of threadgroup memory permits
up to five groups by the M4 memory limit; registers or the execution core will
be the actual occupancy bound. The `U*22` grid supplies about 660 groups/layer
at U=30, enough to fill ten GPU cores.

The credible isolated target is 18-27 ms for 30 GateUp layer-equivalents. A
result less than 10 ms faster than the 36.926 ms shipping control is not enough
to justify the larger kernel.

## B. Down MMA with exact on-chip term replay

Proposed symbols:

- `spec50_moe_argbuf_down_union_batch_k8_mma_terms_rows8`
- optional occupancy control:
  `spec50_moe_argbuf_down_union_batch_k8_mma_terms_rows4`

A direct expert-major f32 accumulator is forbidden. Instead, let one
SIMDgroup build exact integer block dots into threadgroup memory and then replay
the current reduction.

### Rows8 geometry

- Grid: `2816 / 8 = 352` threadgroups.
- Width: 32 threads, one SIMDgroup.
- One group owns eight output rows and all eight candidates.
- Loop `u=0..U`, then `b=0..21` in the matrix phase.
- For each expert, find the unique route slot for every set bit in
  `work[u].candidate_mask`; duplicate slots or a missing live slot are an
  admission failure in the host-side test/fixture.
- Stage `8 x 32` candidate activations and `32 x 8` Q4 integer weights, issue
  four MMAs, and store the exact integral result as `i16` at
  `(candidate, local_row, route_slot*22+b)`.

The bound of 32,512 proves `i16` is lossless. The complete term tile is:

`8 candidates * 8 rows * 176 terms * 2 bytes = 22,528 bytes`.

Activation and weight fragments add about 1 KiB, keeping the kernel below 24
KiB and the M4's 32 KiB limit. The rows4 control uses 11,264 term bytes and may
allow two resident groups, but wastes half of each 8x8 output fragment.

After a SIMDgroup barrier, replay the production Down body candidate by
candidate. Lane `l` must visit `flat = l, l+32, ...` in the same order. For
each term it reloads the original half weight scale, reads the existing
activation scale and route weight, and executes the unchanged expression:

```text
term_scale = (weight_scale * act_scale) * route.weight
lane_total[row] += float(i16_dot) * term_scale
```

The final `simd_sum` is over the same 32 lane partials. This retains the FMA
site that the rejected rank-indexed design broke. Do not precompute/store an
f32 term; that introduces an extra rounding point and is not bit-exact.

`num_unique_experts` is not currently a Down argument. Add it as constant
buffer 7 for this experimental PSO; do not infer U from a 48-slot table or loop
over null work entries.

### Traffic and occupancy expectation

At U=30, the Q4-code read moves from 2.141 GB/round toward the 1.004 GB expert
union floor. Replaying the two-byte weight scales costs about 0.238 GB of
logical reads, so the proposed logical total is about 1.242 GB before cache-line
effects. There is no global term buffer and no persistent allocation.

The risk is explicit: a rows8 group consumes about 24 KiB, normally limiting
an M4 core to one resident group/one resident SIMDgroup, and the sparse route
masks make an 8x8 expert-major MMA do extra zero-row work. That is why rows4
must be a bounded control and why this design is not promoted on traffic math
alone. The credible target is 12-18 ms for 30 Down layer-equivalents; reject it
if it does not save at least 6 ms and contribute to a combined MoE saving of at
least 20 ms.

## Exact source and dispatch seams

All anchors below are symbol-based so the concurrent exact-f32 experiment can
move line numbers without invalidating the handoff.

In `src/metal/spec50_moe_argbuf.rs`:

1. Add the GateUp shader beside
   `spec50_moe_argbuf_gateup_geglu_quant_batch_k8`.
2. Add the Down shaders beside
   `spec50_moe_argbuf_down_union_batch_k`.
3. Add independent optional PSOs to `Spec50MoeArgbufKernels`; compile failures
   must leave existing production PSOs available.
4. Build them inside `spec50_moe_argbuf_kernels`, with fast math disabled, and
   reject GateUp unless SIMD width is 32 and max threads/TG is at least 128.
   Reject Down unless SIMD width is 32 and static threadgroup memory fits the
   queried 32 KiB limit.
5. Add `encode_argbuf_gateup_mma8x8` with the unchanged buffers 0 through 7,
   grid `U*22`, width 128.
6. Add `encode_argbuf_down_mma_terms` with the current buffers 0 through 6 plus
   `num_unique_experts` at 7, grid `2816/rows`, width 32.
7. Select only in `Gemma4MoeSlotArgTable::encode_chained_gateup_k8` and
   `encode_chained_down_k8`, behind one default-off K8-only environment flag.
   K other than 8 retains today's kernels. A missing PSO fails closed to the
   existing path.

In `src/metal.rs`:

8. At `encode_moe_topk_gateup_down`, no binding or memory-barrier change is
   needed for GateUp.
9. At `encode_moe_down`, pass the already-proven active union length as U to
   the experimental Down encoder. Do not use table capacity or hot-slot count.

Suggested flag while both kernels are experimental:

`CAMELID_GEMMA4_MOE_MMA_K8=1`

The receipt must print which of GateUp/Down actually obtained an eligible PSO;
partial silent activation is not acceptable.

## Bounded verification and benchmark plan

No server is needed for the first three gates.

1. **Compile/geometry gate**
   - Print SIMD width, PSO maximum threads, and static/dynamic threadgroup
     bytes for each new PSO.
   - Refuse any mismatch instead of shrinking a dispatch.
2. **Synthetic raw-bit gate**
   - Reuse the argument-buffer parity fixture.
   - Test K=1 through 8 even though production selects only K=8; pad matrix
     candidate rows with zero.
   - Include mask popcounts 1, 2, 4, and 8; the U=30/64-pair routing fixture;
     zero route weights; Q4 nibbles 0 and 15; Q8 -127 and 127; zero and finite
     f16-scale extrema.
   - Require byte equality for GateUp scales, GateUp quants, and Down f32 bits.
3. **Warm 30-layer synthetic microbenchmark**
   - Extend the existing ignored GateUp and Down microbenchmarks.
   - One warm-up per PSO, then nine alternating samples.
   - Headline is median `command_buffer_gpu_times_us`; retain every raw sample.
   - Benchmark current production, GateUp MMA, Down rows8 MMA, and Down rows4
     MMA independently. Keep the removed exact-f32 result as a frozen receipt;
     do not reintroduce that NO-GO PSO. Do not put GateUp and Down in the same
     command when deciding which won.
4. **One real mapped layer parity gate**
   - Reuse layer 26 of the local `.cghost` and the existing K=1..8 raw-bit
     oracle.
   - Warm only the active records; report pageins, decompressions, and swapins.
   - This gate must not create a second expert representation.
5. **Promotion gate**
   - Zero divergent bits.
   - GateUp saves at least 10 ms per 30-layer-equivalent pass versus the
     36.926 ms shipping control.
   - Down saves at least 6 ms and the admitted pair saves at least 20 ms.
   - No swapin, no new persistent expert bytes, no increase in the declared
     resource union.
   - Only after those conditions pass: exact release build and one guarded
     full-round A/B. Port 8181 remains untouched.

## Expected decision tree

- Exact-f32 dots did not recover issue time; that branch is closed.
- If GateUp MMA is exact and clears 10 ms, keep it default-off and proceed to
  the Down rows8/rows4 shootout.
- If GateUp fails but Down clears its gate, Down can be admitted independently.
- If both miss, do not weaken parity. The isolated dense O pre-unpack probe is
  also a NO-GO (10.245 -> 10.758 ms while adding 409 MiB). The next dense work
  must reduce arithmetic or scheduling cost without expanding streamed weight
  bytes; the measured 47 ms round target still requires dense-stage savings.

## GateUp experiment receipt (2026-08-23)

Only strict GateUp candidates were tested; Down remained design-only. Both
variants used a separate, default-off PSO during measurement and preserved
buffers 0..7, output indexing, the exact 32-value integer-dot boundary, and
the existing ascending per-block floating scale accumulation.

The direct synthetic gate used the sparse 30-expert route fixture at every
K=1..8, Q8 extrema, signed zero, subnormal and normal input scales, and the
fixture's full Q4 nibble patterns. Complete active GateUp scale and quant
payloads were byte-identical to the shipping K8 kernel:

```text
test metal::spec50_moe_argbuf::tests::gemma4_moe_mma_gateup_k1_to_k8_adversarial_raw_bit_parity ... ok
```

The first layout staged one 8x32 Q8 tile, two 32x32 unpacked Q4 tiles, and f16
scales in 5,792 bytes of threadgroup memory. Its scoped release benchmark used
one warm-up per PSO and nine interleaved, order-reversed K=8/U=30/30-layer A/B
samples:

```text
current median GPU: 34,568 us
MMA median GPU:     29,446 us
speedup:            1.1739x (17.39%)
median saving:       5,122 us per 30-layer GateUp pass
swapins:             0 in every current and MMA sample
```

Raw current samples were 34,537..34,667 us; raw staged-MMA samples were
29,419..29,507 us.

The bounded follow-up removed all Q8, Q4, and scale threadgroup staging and all
per-block threadgroup barriers. Each lane loaded its Q8 fragment values and
decoded its packed Q4 fragment values directly into matrix registers; only the
final cross-SIMDgroup max/quant exchange remained staged. It also passed the
same K=1..8 adversarial raw-bit gate. Because Metal GPU timestamps are
independent of the Rust host profile, it was measured with the already-built
scoped debug test using the same nine-sample interleaved A/B protocol:

```text
current median GPU: 34,560 us
direct median GPU:  31,124 us
speedup:            1.1104x (11.04%)
median saving:       3,436 us per 30-layer GateUp pass
swapins:             0 in every current and direct sample
```

Raw current samples were 34,542..34,616 us; raw direct-MMA samples were
31,073..31,172 us. Direct fragment loads were therefore 1.678 ms slower than
the staged winner, showing that the barriers were cheaper than duplicated
packed-nibble and scale loads on this M4.

Final decision: **NO-GO**. Even the 5.122 ms staged win is only 51% of the
required 10 ms saving, so it cannot be part of the 50 tok/s production plan.
No experimental code, selector, environment flag, or test remains in the
source. No commit was made, no server was run, and port 8181 was untouched.
