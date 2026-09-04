# EAGLE-3 target-authoritative N1 commit pipeline

Status: source-only architecture, frozen-tree host contract, standalone Metal falsifier, and a
default-off real-model selector shadow. The shadow changes no authoritative state. Every build,
test, Metal execution, and model measurement belongs on mini2.

## Architectural correction

The current N8 round is serialized as:

```text
draft forest -> target verify/wait -> host accept/capture gather -> EAGLE update/wait
```

The last stage costs about 2.024 ms/round on the pinned Pitch baseline. The first overlap design
moved authoritative edge K/V under the target tail, but then ran a complete authoritative EAGLE
cell for every possible endpoint. The measured N8 particle is about 3 ms, so that design makes the
post-target path larger than the stage it replaces.

The corrected design has only one complete post-target cell:

```text
 target layer-25 capture -> E1: FC all rows + edge K/V --------------------+
 target layers 25..27 -> head -> argmax -> integer accept kernel          |
                                           | selected leaf/token/path     |
                                           v                              v
                         E2: fixed-shape gathers -> one full EAGLE cell -> commit
```

The target model and EAGLE model are unchanged. There is still one target verification pass. The
target argmax remains the sole token authority.

## Why edge K/V is reusable

Speculative EAGLE K/V cannot be committed. An ephemeral draft row uses its parent draft cell's
`raw_hidden` as recurrent `g`, while an authoritative row uses `fc(target captures)`. Even an
accepted draft token therefore has the wrong recurrent input.

The exact reusable unit is an authoritative edge cell. For verifier edge `p -> c`:

```text
token input       = target embedding(tree.tokens[c])
recurrent g       = fc(target captures[p])
logical position  = stable_prefix + tree.depth[c] - 1
attention history = stable prefix + authoritative edges on root..p
```

Only this cell's K/V is needed. The next authoritative edge gets `g` from its own target capture,
not from this cell's attention/MLP output, so Q, attention, O, MLP, raw hidden, and head work can be
omitted for every non-root verifier row.

The final bonus at selected endpoint `r` is the only complete cell:

```text
token input       = target embedding(target_predictions[r])
recurrent g       = fc(target captures[r])
logical position  = stable_prefix + tree.depth[r]
attention history = stable prefix + selected authoritative edges + this terminal cell
```

Its token is unknowable before target argmax, but its endpoint can be selected on the GPU before
the CPU observes the target result.

## Exact dataflow

### 1. Freeze one tree contract

`Eagle3DraftForest::plan_authoritative_precompute` records every non-root edge's child token,
parent capture row, depth-derived position, and predecessor edge rows. Its embedded
`Eagle3DeviceAcceptancePlan` carries the exact token/parent/depth arrays consumed by device
selection. Resolution rejects a plan from any different tree.

`plan_device_acceptance` also validates:

- `1 <= nodes <= TREE_MAX_NODES`;
- root parent/depth are `-1/0`;
- every parent precedes its child and depth increments by one;
- every tree token is in the target vocabulary; and
- sibling token IDs are unique.

The dynamic lattice already rejects duplicate child tokens at expansion and reranking preserves
that property. The extra check is fail-closed admission for a future forest producer, not a new
acceptance rule.

The older optional terminal candidates remain a target-blind experiment. The N1 design supplies
`None` for every row and never depends on candidate coverage.

### 2. Split target verification at the last EAGLE tap

The required target layer inputs are `[2, 14, 25]`. Under the default-off gate, split
`ResidentDecodeState::verify_batch_inner` at the input of layer 25:

1. command buffer A copies the layer-25 input capture and signals `capture_ready`;
2. command buffer B, on the same target queue, runs unchanged layers 25--27 and the unchanged
   output head/argmax, then the integer acceptance kernel;
3. B signals `accept_done` immediately after all selector outputs are available; and
4. a pre-encoded target compactor T follows B on the target queue and consumes that selected path
   while E2 is free to start on the EAGLE queue.

Queue ordering preserves target arithmetic. The split must first prove predictions, captures, and
target K/V byte-identical to the unsplit route.

### 3. E1 overlaps FC and every known edge K/V

An EAGLE-private queue waits for `capture_ready`. Command buffer E1:

1. packs the three target capture buffers as `[N, 3H]`;
2. applies EAGLE `fc` once to produce `g[N,H]`;
3. pairs each non-root child embedding with `g[parent[child]]`;
4. runs the exact authoritative input/hidden norms and K/V projections; and
5. applies RoPE at `stable + depth[child] - 1` and scatters F16 K/V to private edge slots.

E1 ends before E2 on the same private queue, so E2 sees `g` and edge K/V without another event. It
races only the target tail, never target argmax or acceptance.

Use scratch slots disjoint from stable destinations:

```text
commit reserve       [stable, stable + N)
edge scratch         [stable + N, stable + 2N - 1)
terminal scratch     [stable + 2N - 1]
```

Fewer than `2N` safe positions fails closed before either command is submitted. Scratch slot
numbers never enter RoPE or causal-length arithmetic.

### 4. Reproduce `accept_longest_path` on device

The selector is a bounded integer kernel encoded immediately after production argmax. One thread
is sufficient for N8/N16. It starts at root row zero; at row `c` it reads `pred_buf[c]`, scans rows
`c+1..N` in increasing order for the first child whose parent is `c` and token equals that
prediction, and either continues at that child or terminates.

It writes fixed-capacity buffers:

```text
selected_leaf          last accepted verifier row r
emitted_count          matched edge count + one terminal bonus
terminal_token         raw pred_buf[r]
safe_terminal_token    terminal_token, or zero when out of vocabulary
terminal_depth         tree.depth[r]
terminal_valid         terminal_token < target vocabulary
path_rows[0..count]    root..r
emitted_ids[0..count]  target predictions followed during acceptance
```

This is semantically identical to `TokenTree::accept_longest_path`:

- There is no floating-point comparison or tie in acceptance. Production argmax has already
  resolved score ties and written integer IDs.
- Production siblings have unique token IDs, so a matching child is unique. The kernel still
  scans increasing rows and takes the first match, exactly like the host oracle, making even a
  synthetic duplicate-sibling tree deterministic.
- On a match, the emitted target prediction equals the child token. On the first miss, that same
  prediction is emitted as the guaranteed terminal bonus.
- Valid parent/depth structure gives `path_rows.len == emitted_count == terminal_depth + 1`.

The raw terminal ID is retained for parity. E2 binds `safe_terminal_token` to its embedding gather,
and the transaction is never made visible unless `terminal_valid == 1`; this prevents an
impossible all-invalid argmax sentinel from becoming an out-of-bounds embedding read.

The Metal falsifier in `src/metal.rs` compiles the same selector source as a standalone test
library. The first production staging slice is armed only by
`CAMELID_BENCH_EAGLE3_DEVICE_ACCEPT_SHADOW=1`. On a committing EAGLE tree verify with the exact
`[2, 14, 25]` capture contract, it lazily creates the selector pipeline, appends one thread to the
existing target command buffer after production argmax, and compares every returned field with
the host oracle. It does not split the target command buffer yet and does not feed emission,
target compaction, or EAGLE state.

Gate-off verification continues through the pre-existing wrappers with `None` for the acceptance
plan, so there is no selector pipeline lookup, allocation, binding, dispatch, or receipt readback.
Gate-on telemetry is emitted once per real tree round:

```text
[eagle3-device-accept-shadow] outcome=match ... requested_total=R encoded_total=R \
matched_total=R mismatched_total=0 fallback_total=0
```

The two accounting identities are `requested = encoded + fallback` and
`encoded = matched + mismatched`. Any mismatch is diagnostic only; the already-existing host path
still controls emitted tokens and `compact_tree_kv_path`.

### 5. Pre-encode one fixed-shape E2 cell

E2 is encoded and committed ahead of target completion on the EAGLE queue. It follows E1 and waits
for `accept_done`. Every dispatch dimension is fixed for one row; only buffer contents are selected
on device:

1. Existing quantized embedding gathers already take `device const uint* selected_id`; bind the
   selector's `safe_terminal_token` buffer directly.
2. Add a fixed-width F32 row gather from `g[N,H]`, indexed by `selected_leaf`, into `g_selected[H]`.
3. Gather one cos/sin row from prebound tables using `terminal_depth` (or teach a one-row RoPE
   wrapper to use a device position scalar).
4. Convert `path_rows[1..count]` to the corresponding E1 edge scratch slots, append the fixed
   terminal scratch slot, and write `position_count = stable + emitted_count`.
5. Run exactly one existing authoritative full cell: Q/K/V, RoPE, terminal K/V scatter, selected
   ancestry attention, O/residual/MLP, output norm, draft head, and top-k/LSE.

No CPU offset and no Metal indirect command buffer is required. Metal buffer bindings are fixed
when encoded, but their scalar contents are read when a kernel executes. Command-buffer order and
the shared event make selector writes visible to the gather and cell kernels.

Attention routing is the one non-obvious admission constraint. Current host code chooses v2,
split-K, prefetch, and split count from `position_count`. Before encoding E2, the host knows
`stable` and the possible interval `stable + 1 .. stable + max_depth + 1`. Admit this lane only if
every count in that interval chooses the same existing pipeline and split geometry. Allocate
scratch for the interval maximum, bind the device-written live count, and execute that one route. A
round crossing a 128-position or split-count boundary falls back to the serial update. This keeps
E2's arithmetic identical without waiting for the CPU or dispatching N endpoint cells.

### 6. Reuse the one selected path for both caches

Nothing reconstructs the accepted path on the host. The target and EAGLE committers consume the
same `path_rows` and `emitted_count` buffers.

Target verifier scratch may overlap its stable destination. Its compactor must therefore use a
fixed-size gather into temporary selected K/V followed by a scatter into the stable rows; each
kernel guards ranks `>= emitted_count`. EAGLE edge and terminal sources are disjoint from the
stable reserve, so its fixed-size scatter can copy directly:

```text
edge_scratch(path_rows[1]), ..., edge_scratch(path_rows[count-1]), terminal_scratch
    -> stable[0..count]
```

After E2 and both compactors complete, the CPU must eventually read emitted IDs to return them to
the caller. That wait is no longer needed to schedule E2. It validates `terminal_valid`, epoch,
count, leaf, and path bounds, then atomically installs the terminal `Eagle3MetalOutput`, advances
both cache lengths, and invalidates scratch.

A failure before visibility mutates no logical state and falls back to the established serial
update. A failure after a stable copy begins fails the request rather than risking observation of
a partial commit.

## Required invariants

1. **Target authority:** emitted IDs come only from production `pred_buf` under the existing
   greedy argmax and `accept_longest_path` rule.
2. **One tree epoch:** target verify, E1 edges, selection, E2, and compaction all carry the same
   immutable token/parent/depth signature and generation epoch.
3. **Exact edge pairing:** edge `p -> c` uses `(embedding[token(c)], captures[p])`; terminal row
   `r` uses `(embedding[pred[r]], captures[r])`.
4. **One complete tail:** E1 computes N-1 K/V-only edges; E2 computes exactly one full cell.
5. **Logical/physical separation:** target depth controls RoPE and causal length; scratch slots
   only control physical cache reads.
6. **Private ancestry:** E2 sees stable history, only selected edge slots, and its terminal slot.
7. **Route identity:** E2 uses the same attention implementation and partition as the serial N1
   oracle or declines the lane for that round.
8. **F16 identity:** scratch and committed K/V use the current post-RoPE F16 conversion.
9. **Atomic visibility:** cache lengths and terminal seed advance together only after every
   selected row is complete.
10. **Lifetime:** prediction, capture, selection, `g`, and scratch buffers outlive target B, E1,
    E2, and compaction.
11. **Cancellation safety:** reset, rollback, cancellation, or fallback invalidates the epoch.
12. **Default off:** without `CAMELID_BENCH_EAGLE3_TARGET_COMMIT_PIPELINE=1`, there is no target
    split, E1/E2, extra queue, or changed update path. Without the separate
    `CAMELID_BENCH_EAGLE3_DEVICE_ACCEPT_SHADOW=1` falsifier gate there is also no selector pipeline
    lookup, allocation, dispatch, or receipt readback. The staged E1 falsifier is independently
    gated by exact value `CAMELID_BENCH_EAGLE3_AUTHORITATIVE_E1_SHADOW=1`; without it the ordinary
    capture wrapper is called directly, so there is no target split, event, queue, E1 allocation,
    dispatch, or timing readback.
13. **Shadow isolation:** the device-acceptance shadow may add only the selector and parity receipt.
    It cannot arm E1/E2, select emitted tokens, or change either cache length.

## Implementation seams

- `src/eagle3_runtime.rs`
  - keep `Eagle3DeviceAcceptancePlan` and `Eagle3AuthoritativePrecomputePlan` as the frozen host
    contract;
  - add an opaque pending transaction to `Eagle3Drafter`;
  - install a completed transaction only under the new gate;
  - counters: rounds, selector parity, E1 edge rows, E2 full rows (must equal rounds), route-boundary
    fallbacks, target/EAGLE compact time, epoch failures.
- `src/inference/metal_resident.rs`
  - strict selector and E1 gates, real verifier receipts, and cumulative route counters;
  - current E1 slice: split the target immediately after the layer-25 input copy, signal a shared
    event, retain the three capture buffers until E1 completes, and leave target acceptance and
    compaction unchanged; the post-split target tail owns disjoint mutable GEMV scalars, so host
    encoding cannot race parameters still being read by the committed prefix;
  - encode selector and target two-stage path compactor after argmax;
  - leave ordinary tree verify untouched as gate-off oracle.
- `src/metal.rs`
  - lazily compile the proven selector source only for the shadow wrapper and bind the unchanged
    production `pred_buf`;
  - promote it into the permanent shader library only after real-model falsification;
  - fixed F32/rope/path-slot selected-row gathers;
  - current E1 slice: row-interleave the real target capture buffers, run one wide authoritative
    FC, pair every child embedding with its parent capture, run wide K/V projections, RoPE at
    `stable + depth - 1`, and scatter F16 into a private `[head][edge][dim]` cache;
  - one-row authoritative E2 twin using buffer-backed leaf/token/count/path metadata;
  - disjoint EAGLE compactor and transaction finisher.
- `src/main.rs` and `src/eagle3_serving.rs`
  - select the paired transaction path under the gate and record telemetry;
  - preserve the current serial call sequence as fallback and parity oracle.

## Latency budget

The N8 endpoint particle is closed: about 3 ms post-target is already slower than the measured
2.024 ms update. The N1 pipeline budget is:

```text
device accept + selected gathers        <= 0.12 ms median
one complete authoritative E2 cell      <= 0.88 ms median
selected-path cache compaction/install  <= 0.20 ms median
total post-target addition              <= 1.20 ms median
target slowdown from concurrent E1      <= 0.25 ms median
net saving versus current update        >= 0.70 ms/round
```

At roughly 4.7 emitted tokens/pass, 140 tok/s allows 33.57 ms per round. This change must remove a
serialized stage; concurrency by itself earns no budget credit because target and E1 share unified
memory bandwidth.

## Smallest falsification checkpoints

### Checkpoint A: selector semantics (implemented, mini2 only)

Run:

```text
cargo test --lib eagle3_device_acceptance -- --nocapture
```

On macOS this must run and pass these five tests:

```text
eagle3_runtime::tests::eagle3_device_acceptance_matches_target_oracle
inference::metal_resident::eagle3_device_acceptance_shadow_tests::eagle3_device_acceptance_shadow_gate_is_strict_and_default_off
inference::metal_resident::eagle3_device_acceptance_shadow_tests::eagle3_device_acceptance_shadow_compares_every_commit_field
metal::tests::eagle3_device_acceptance_plan_fails_closed_on_every_tree_invariant
metal::tests::metal_eagle3_device_acceptance_matches_host_oracle
```

The assertions cover root misses, every branch/depth, terminal bonuses, 256 deterministic N8
prediction vectors, an invalid UINT_MAX terminal made gather-safe, and a synthetic duplicate
sibling whose lower row must win. Any leaf, count, path, emitted ID, terminal ID, safe ID, validity,
or depth mismatch falsifies device acceptance. The production admission test separately proves
that malformed lengths, root metadata, parent/depth links, vocabulary IDs, and duplicate siblings
fail closed before allocation or dispatch.

### Checkpoint B: default-off real-model selector shadow (implemented, mini2 only)

Build one candidate, then run the protected N8/K4/X5 Pitch command once with the promoted
environment unchanged and once with this single addition:

```text
CAMELID_BENCH_EAGLE3_DEVICE_ACCEPT_SHADOW=1
```

The gate-on stderr must contain one `outcome=match` line per committing EAGLE tree verify, no
`outcome=mismatch`, and no `outcome=fallback`. On the final line, if `R` is `requested_total`, the
counters must satisfy:

```text
requested_total=R encoded_total=R matched_total=R mismatched_total=0 fallback_total=0
```

Every ordinary N8 line must say `rows=8`; a shortened final output-budget round may be narrower.
The control and shadow receipts must have identical target/EAGLE token arrays, `lossless=true`, and
the same first-divergence result. Any mismatch or cache/token difference blocks the target split.

### Checkpoint C1: split target plus state-inert E1 shadow (implemented, mini2 only)

Run the protected Pitch command with the promoted environment plus:

```text
CAMELID_BENCH_EAGLE3_AUTHORITATIVE_E1_SHADOW=1
```

Every serially updated real tree round must print `outcome=match`; `mismatched_total` and
`fallback_total` must remain zero. `prepared_edges` must be `rows - 1`, while `selected_edges`
must be the accepted path length minus the terminal bonus. `compared_f16_values` must equal
`selected_edges * 2 * kv_heads * head_dim`. The target token array and lossless receipt must be
byte-identical to gate-off. Compare gate-off/on `target_gpu_us` distributions to enforce the
0.25 ms median target-slowdown ceiling; `target_tail_interval`, `e1_interval`, and `overlap_us`
prove whether the second queue actually overlapped rather than merely moving the same work.
Treat `e1_encode_us` as host submission cost and `e1_post_target_wait_us`, `e1_readback_us`, and
`oracle_compare_us` as shadow-only diagnostic cost. They must be reported separately from E1 GPU
time and excluded when projecting the future device-resident production path.
If the existing terminal-head-skip optimization suppresses the last serial update, exactly that
round is instead accounted as `reason=serial_oracle_skipped`; it has no oracle and is not a K/V
parity claim.

This checkpoint deliberately does not expose E1 state to decoding. It waits for E1, runs the
unchanged serial authoritative update, and compares every selected edge's private F16 K/V bits
against the live serial rows. Non-selected edges have no mutation-free serial oracle and are not
claimed by this test. Any route, capture, tree, position, capacity, weight, command-buffer, or
pending-epoch failure produces a named fallback and leaves the serial path authoritative.

The model-free admission checks are selected together by:

```text
cargo test --release --lib eagle3_authoritative_e1 -- --nocapture --test-threads=1
```

They prove the gate accepts only trimmed exact `1`, the fixed capture contract, width/hidden/tree
and cache bounds, and sparse verifier-row to dense accepted-prefix edge mapping.

### Checkpoint C2: E2 and commit visibility (next)

Interleave control and candidate rounds under the mini2 lock:

1. E2 N1 versus the current serial terminal cell for K/V, raw hidden, draft IDs/logits, and LSE;
2. both compacted caches versus the current host-selected cache state; and
3. GPU start/end times for selector, E2, compactors, and the wall-time union.

Stop this route immediately if any exact-state mismatch survives a buffer/position/ancestry audit.
Also stop if median selector+gathers exceeds 0.12 ms, median E2 exceeds 0.88 ms, median total
post-target work exceeds 1.20 ms, or E1 adds more than 0.25 ms to target verification. Proceed to
an integrated default-off lane only if parity is exact and the median net saving is at least
0.70 ms/round. Promotion still requires ordinary Pitch lossless token equality and the full mini2
regression suite.
