# EAGLE-3 target-authoritative commit pipeline

Status: source-only architecture and host mapping contract.  It is default-inert.  Every build,
test, Metal execution, and model measurement belongs on mini2.

## Problem

The current N8 round is serialized as:

```
draft forest -> target verify/wait -> host accept and capture gather -> EAGLE update/wait
```

On the pinned Pitch baseline the last stage costs about 2.024 ms/round.  It also reconstructs a
path the target verifier has just traversed: target tree K/V is compacted to the accepted path,
then the same path's three capture rows are copied into new CPU tensors, interleaved, uploaded, FC
projected, and replayed through the private EAGLE head.

The tempting shortcut is invalid.  An ephemeral draft row uses its parent EAGLE cell's
`raw_hidden` as recurrent `g`; a stable authoritative row uses `fc(target captures)`.  Therefore
draft K/V is not commit K/V even when the draft token is accepted.

The exact reusable object is an authoritative **edge cell**.  For target-tree edge `p -> c`:

```
token input       = target embedding(tree.tokens[c])
recurrent g       = fc(target captures[p])
logical position  = stable_prefix + tree.depth[c] - 1
attention history = stable prefix + authoritative edge cells on root..c
```

If `c` is accepted, target prediction at `p` equals `tree.tokens[c]` by definition.  All inputs to
this cell except the captures are known before target verification, and the captures are ready at
the input of target layer 25, before layers 25--27 and the target output head run.

The final emitted bonus is different.  At accepted endpoint `r`, its exact cell is:

```
token input       = target embedding(target_predictions[r])
recurrent g       = fc(target captures[r])
logical position  = stable_prefix + tree.depth[r]
attention history = stable prefix + authoritative edge cells on root..r
```

Its token is unknown until the ordinary target argmax.  The pipeline below overlaps everything
else and evaluates all eight possible final endpoint cells in one multi-column EAGLE body stream
as soon as that argmax buffer is ready.

## Minimal exact pipeline

### 1. Freeze an immutable row plan

`Eagle3DraftForest::plan_authoritative_precompute` pins the host-side edge arithmetic.  One entry
exists for every non-root verifier row.  It records the child token, parent capture row, logical
position offset, and predecessor edge rows.  `resolve_authoritative_precompute` proves that an
accepted path maps to those entries without reinterpreting the target result.

The first implementation should use only `verified_edges`.  The optional virtual terminal token
is a later optimization and must never be required for exactness.

### 2. Split target verification at the last EAGLE tap

The three required target inputs are layers `[2, 14, 25]`.  In
`ResidentDecodeState::verify_batch_inner`, at the top of iteration `l == 25`:

1. encode the existing copy of `cur` into the layer-25 capture buffer;
2. end target command buffer A;
3. signal a `capture_ready` shared event;
4. commit A on the existing target queue;
5. continue layer 25 through target argmax in command buffer B on that same queue;
6. signal `target_done` from B after the unchanged production argmax.

The existing queue ordering preserves target arithmetic.  The new split is default-off and must
first prove that its predictions, captures, and target K/V are byte-identical to the one-buffer
route.

### 3. Precompute authoritative edge K/V on an EAGLE queue

Add a private `overlap_queue` and typed pending-commit epoch to `Eagle3MetalState`.  Command buffer
E1 waits for `capture_ready` and then:

1. packs the three target capture buffers into `[N, 3H]` on GPU;
2. applies EAGLE `fc` once for all N verifier rows, producing `g[N,H]`;
3. for every non-root child `c`, pairs the already-known child embedding with `g[parent[c]]`;
4. runs the same input/hidden RMS norms and exact Q4 K/V projections used by
   `encode_authoritative_kv_row_from_buffers`, multi-column where admitted;
5. applies RoPE at `stable + depth[c] - 1`;
6. scatters F16 K/V to one epoch-owned edge scratch slot.

Only K/V is needed for these intermediate cells.  Authoritative successor `g` comes from its own
target captures, not from the predecessor cell's output, so Q, attention, O, MLP, raw hidden, and
LM head are unnecessary for every verified edge.

Use scratch slots disjoint from the possible commit destination:

```
commit reserve       [stable, stable + N)
edge scratch         [stable + N, stable + 2N - 1)
endpoint scratch     [stable + 2N - 1, stable + 3N - 1)
```

The gap is intentional.  Selection can copy from scratch into the stable range with no in-place
alias hazard.  Fewer than `3N - 1` remaining cache positions fails closed to the current update.

### 4. Feed target argmax directly into one endpoint batch

Command buffer E2 is encoded and committed ahead of time on `overlap_queue`.  It waits for
`target_done`, so CPU encoding and scheduling are not on the post-target critical path.  It owns
the target prediction and capture buffers until completion.  For every verifier row `r` it:

1. gathers `target_predictions[r]` from the unchanged target `pred_buf`;
2. gathers/dequantizes that row from the target embedding wire table into a packed F32 embedding;
3. pairs it with `g[r]` from E1;
4. computes Q/K/V for all N rows with the particle batch's exact multi-column kernels;
5. ropes each row at `stable + depth[r]` and scatters K/V to its endpoint scratch slot;
6. runs branch-masked attention over the stable prefix, edge scratch slots for
   `tree.path_to(r)[1..]`, and its own endpoint slot;
7. runs O, residual, MLP, output norm, EAGLE LM head, and the existing GPU top-eight/LSE tail once
   for the N packed rows.

This is the particle batch with authoritative `g`, GPU-selected target embeddings, and static
endpoint ancestry.  The target prediction is not approximated and no candidate coverage
assumption exists.  N1 and N8 must both be compared bit-for-bit with the current serial
authoritative cell before throughput is considered.

### 5. Select, compact, and install

While E2 runs, the CPU reads the completed target result, applies the existing
`TokenTree::accept_longest_path`, and compacts target K/V exactly as today.  It does not gather
EAGLE captures or launch an update.

After E2 completes, accepted path `root, c1, ..., r` selects:

```
edge scratch(c1), ..., edge scratch(r), endpoint scratch(r)
```

There are exactly `emitted_tokens.len()` cells: one for every matched edge and one final target
bonus.  Copy their raw F16 K/V rows into `[stable, stable + emitted_len)`, install endpoint row
`r`'s `Eagle3MetalOutput` as `stable_seed`, set `filled = stable + emitted_len`, and invalidate the
scratch epoch.  A failure before the copy mutates no stable byte and falls back to the established
authoritative update.  A failure after stable copy starts must fail the request; silently replaying
would risk observing a partial commit.

The first implementation may use coherent-memory row copies after both commands finish.  A later
GPU compactor needs a two-dispatch temporary gather/scatter; copying path ranks in parallel in
place is racy because one rank's source can be another rank's destination.

## Required invariants

1. **Target authority:** emitted ids come only from the existing production `pred_buf` and
   `accept_longest_path`.
2. **Exact cell pairing:** matched edge `p -> c` uses `(embedding[token(c)], captures[p])`; terminal
   endpoint `r` uses `(embedding[pred[r]], captures[r])`.
3. **Logical/physical separation:** RoPE and sliding-window length use target-tree depth; scratch
   slot number never enters positional math.
4. **Private ancestry:** endpoint `r` sees only stable history, edge cells on its own path, and
   itself.
5. **F16 identity:** scratch and committed K/V use the same post-RoPE F16 scatter conversion as
   the current authoritative row.
6. **Atomic visibility:** `filled` and `stable_seed` move together only after a complete selected
   path is available.
7. **Lifetime:** prediction/capture/scratch buffers cannot return to the pool until target B, E1,
   and E2 have completed.
8. **Epoch safety:** reset, rollback, cancellation, or fallback invalidates every outstanding
   scratch handle.
9. **Default off:** without `CAMELID_BENCH_EAGLE3_TARGET_COMMIT_PIPELINE=1`, no target split,
   second queue, extra allocation, or changed update path exists.

## Code seams

- `src/eagle3_runtime.rs`
  - keep `Eagle3AuthoritativePrecomputePlan` as the normative row map;
  - add an opaque pending transaction to `Eagle3Drafter`;
  - replace `accept_authoritative_forest` only under the new gate;
  - receipt counters: rounds, edge rows, endpoint rows, post-target wait, compact time, fallbacks.
- `src/inference/metal_resident.rs`
  - add one paired tree-verify entry point that passes the target embedding wire and frozen row
    plan into Metal; ordinary tree verify stays untouched.
- `src/metal.rs`
  - split/gated target verify handle retaining `pred_buf` and capture buffers;
  - capture-pack/gather kernel;
  - E1 K/V-only edge encoder;
  - E2 authoritative endpoint twin of `forward_particle_batch`;
  - selected path F16 compactor and endpoint-output finisher.
- `src/main.rs` and `src/eagle3_serving.rs`
  - select the paired entry point under the gate and record telemetry; keep current call sequence as
    fallback and gate-off oracle.

## Latency budget

The 2.024 ms current update contains work in two classes:

- all-row FC and accepted-prefix K/V, which E1 moves under target layers 25--27 plus the target
  output head;
- one complete final EAGLE cell, which E2 retains after target argmax but executes as one N8
  weight stream with its command already scheduled.

The target tail after the layer-25 input is expected to provide several milliseconds of overlap,
but both models consume unified-memory bandwidth.  Concurrency is a hypothesis, not a credit in
the budget.  The architectural target is:

```
old post-target update       2.024 ms/round
new post-target E2 + commit <= 1.05 ms/round
target slowdown from E1     <= 0.25 ms/round
net saving                  >= 0.70 ms/round
```

At the parallel campaign targets (about 4.7 emitted/pass and a 24.0--24.6 ms target verifier),
0.7--1.0 ms of update removal is useful margin: a 140 tok/s round may spend at most 33.57 ms.
This pipeline is not sufficient by itself; it composes with the connected recurrent forest and
the target verifier pipeline.

## Smallest falsification checkpoint

Do not build the integrated commit first.  Add a default-off mini2-only timing/parity shadow with
the exact buffers and queue split, but discard every EAGLE scratch result:

1. target control: current one-command-buffer verify;
2. split control: target A/B only, proving identical predictions, captures, and K/V;
3. overlap shadow: target A/B plus E1 edge FC/K/V on the second queue;
4. endpoint shadow: E2 N8 authoritative endpoint batch after `target_done`, compared against the
   current accepted-path final cell for raw-hidden bits, draft ids/logits/LSE, and K/V bits.

Report target A busy time, target B busy time, E1/E2 busy time, union wall interval, event waits,
and command-buffer status.  Interleave controls and candidates under the mini2 lock.

Stop this route if either condition holds:

- any exact-state mismatch survives a buffer/position/ancestry bug audit; or
- median `(overlap target + E1 + E2 + compact) - target control` is at least 1.55 ms, leaving less
  than 0.47 ms net saving versus the existing 2.024 ms update.

Proceed to the integrated default-off lane if parity is exact and the added post-target critical
path is at most 1.30 ms, with no more than 0.25 ms target slowdown from E1 contention.  Promote
only after the ordinary Pitch lossless token equality and full mini2 regression suite pass.
