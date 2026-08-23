# GPU-addressable mapped-cold fallback without prediction — 2026-08-23

## Decision

**NO-GO for a production prototype.** With the public Metal resource model,
Camelid cannot simultaneously provide all three of these properties:

1. exact access to any of the 128 experts selected by the current layer's GPU
   router;
2. residency bounded to only that layer's actual routed union; and
3. no CPU/GPU phase boundary between routing and expert execution.

This is a resource-residency causality limit, not a missing shader pointer
trick. Before an indirect buffer resource can be read, Metal needs its
allocation resident. The current layer's exact set of allocations does not
exist until that layer's router has executed. If the CPU does not observe the
router result, it must make every possible cold allocation resident. If it
does observe the result, the current per-layer route barrier remains. A
candidate set plus exact retry can break the stalemate, but that is the
prediction/retry design and is intentionally outside this no-prediction
question.

No runtime code was changed and no memory-risking probe was run.

## Exact current seam

`GhostMetalResidentRuntime::execute_chained_round_all_layers` already places
the boundary at the narrowest safe point:

1. `src/metal.rs` encodes the current layer's attention, router, and expert
   input quantization.
2. The unpredicted path commits that command at the `cb_attn_router` creation.
3. `cb_attn_router.wait_until_completed()` makes the shared router-logit bytes
   valid for the CPU.
4. The slot filler derives the exact K-token union and canonical identity
   directory from those logits.
5. `record_work_address_plan` proves that each work item uses
   `expert_id * GEMMA4_Q4_EXPERT_SLOT_STRIDE`.
6. `Gemma4Q4ExpertSlotBinding::materialize_for_active_slots` creates no-copy
   `MTLBuffer` views only for that exact union.
7. `Gemma4MoeSlotArgTable::declare_active_slots` calls `use_resource` only for
   those records; the binding and mmap owner live until terminal command-buffer
   status.

The relevant code anchors are:

- router/quantize encoding: `src/metal.rs`, near
  `begin_gpu_stage!(GPU_STAGE_ROUTER)`;
- route barrier: `src/metal.rs`, `cb_attn_router.wait_until_completed()`;
- exact union: `src/metal.rs`, the first `filler(layer_idx, router_slice, None,
  ...)` call;
- sparse materialization: `src/metal.rs`,
  `slot_binding.materialize_for_active_slots(active_slots)`;
- residency declaration: `src/metal/spec50_moe_argbuf.rs`,
  `Gemma4MoeSlotArgTable::declare_active_slots`.

The route readback itself is only `K * 128 * sizeof(f32)` (4 KiB at K=8).
The visible `slot_wait_ms` is not a 4-KiB memcpy tax: it waits for queued GPU
work and any preceding mapped-expert page faults as well. Replacing the blocking
call with a callback or shared event can free a host thread, but it does not
remove the dependency or its wall-clock latency.

## Why the direct mapped-base kernel is unsafe

The obvious shader is arithmetically exact:

```metal
uint expert = work.expert_weight_offset / G4Q4_SLOT_STRIDE;
uint hot_slot = hot_directory[expert];
device const uchar *weights = hot_slot != 0xffffffffu
    ? hot_records[hot_slot]
    : mapped_layer_base + ulong(expert) * G4Q4_SLOT_STRIDE;
```

The problem is the resource behind `mapped_layer_base`, not the pointer math.
Binding that base makes one complete 128-record layer a possible resource for
the dispatch. Its exact size is:

| Geometry | Bytes | Binary size |
|---|---:|---:|
| one expert stride | 3,358,720 | 3.203125 MiB |
| 128-expert layer | 429,916,160 | 410 MiB |
| 30 layers | 12,897,484,800 | 12.011719 GiB |

Splitting the base into 128 record resources and storing their GPU addresses in
an argument buffer does not change the residency requirement. Metal does not
infer a smaller residency set from the pointer the router eventually selects;
every resource that may be accessed has to be declared before dispatch. GPU
encoding of the argument buffer and an indirect command buffer can choose a
pointer or dispatch size, but neither can make a previously nonresident
allocation legal to dereference.

The repository already contains the falsification evidence for retaining the
dense set. The first full-model mapped K=1 run held all 3,840 record views and
30 dense argument tables. Host wired memory rose from about 2.19 GiB to 9.05
GiB and the watchdog stopped at about 2.08 GiB reclaimable headroom. See
`macos-file-pager-implementation-handoff-2026-08-22.md`. The current
descriptor-only/sparse-transient implementation was introduced specifically to
remove that staircase. Reintroducing a layer-span buffer or a dense all-record
table would undo the pager fix.

A one-layer-at-a-time version bounds the *simultaneous* cold declaration, but
does not make the approach useful for the 50 tok/s lane. It would still prepare
410 MiB per layer, or 12.011719 GiB over a 30-layer round, instead of the
measured active-union scale (about 2.815 GiB at 30 experts/layer). It also needs
a completion callback or wait before submitting the next layer to keep that
residency bounded, so it is no longer a continuous 30-layer submit.

## Hot override does not solve the cold-residency set

Freezing the hot directory for one command makes the hot/cold selection exact,
but every non-hot expert remains a possible route. Declaring that complement
has these bounds:

| Hot records/layer | Anonymous hot tier, 30 layers | Possible cold complement, 30 layers |
|---:|---:|---:|
| 32 | 3,224,371,200 B / 3.002930 GiB | 9,673,113,600 B / 9.008789 GiB |
| 48 | 4,836,556,800 B / 4.504395 GiB | 8,060,928,000 B / 7.507324 GiB |

The split changes which resource supplies an expert; it does not reduce the
12.011719-GiB all-expert addressability requirement when the exact route is
unknown. Declaring all cold complements is especially poor for bandwidth: the
point of the current sparse table is that only actual cold misses enter the
Metal residency set.

## Metal 4 audit on this host

This host is macOS 26.5 on Apple M4 with Metal 4, and the installed 26.1 SDK
does expose placement-sparse buffers. They do not provide a file-paged escape
hatch:

- `newBufferWithLength:options:placementSparsePageSize:` creates an unbacked
  sparse GPU buffer.
- `MTL4CommandQueue::updateBufferMappings` maps ranges to tiles from an
  `MTLHeapTypePlacement` heap. It does not alias ranges of a read-only file
  mmap.
- The mapping-operation array and heap offsets are supplied by the CPU. There
  is no route-list-driven indirect sparse-map operation.
- Unbacked sparse-buffer reads are defined to return zero. They do not suspend
  a shader and invoke an application file-page fault handler.
- `MTLIOCommandBuffer::load` can load fixed file ranges into Metal buffers, but
  file offsets and destinations are CPU-encoded; MTLIO has no GPU-generated
  indirect load command.

Therefore a Metal 4 sparse canonical buffer could implement a bounded
*anonymous cache*. On a cold miss it would have to report the IDs, return to the
CPU, map/load those records, and retry the layer. That can remain exact, but it
retains the same per-layer phase boundary. Mapping every record up front again
requires the full physical model.

The currently pinned `metal = 0.31` crate also has no Metal 4 placement-sparse
bindings. Adopting them would require an Objective-C/FFI module and a second
MTL4 queue synchronized with the existing queue. That engineering cost does
not change the causality result above.

Authoritative platform references:

- Apple, [Tracking the resource residency of argument
  buffers](https://developer.apple.com/documentation/metal/tracking-the-resource-residency-of-argument-buffers)
  — indirect resources must be made resident before a pass; missing residency
  fails the command buffer.
- Apple, [MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset)
  — allocation membership and commits are CPU-managed.
- Apple, [Metal 4 sparse-buffer mapping](https://developer.apple.com/documentation/metal/mtl4commandqueue/updatemappings%28buffer%3Aheap%3Aoperations%3A%29)
  — placement-sparse storage aliases Metal heap tiles.
- Apple, [MTLBufferSparseTier1](https://developer.apple.com/documentation/metal/mtlbuffersparsetier/tier1)
  — unbacked reads return zero.
- Apple, [MTLIOCommandBuffer](https://developer.apple.com/documentation/metal/mtliocommandbuffer)
  — file loads are encoded commands into destination resources.

## Viable exact paths

### 1. Keep exact per-layer materialization

This remains the safe no-prediction architecture. Optimize what runs before
and around the barrier, but do not broaden the resource set. Candidate work is
queue overlap, eliminating unnecessary command-buffer ownership, and reducing
the expert kernels' GPU time. A shared-event callback may reduce host-thread
occupancy, but should not be presented as removal of `slot_wait` latency.

### 2. Bounded candidate table plus exact post-check/retry

Materialize 40–48 deterministic candidates per layer before the continuous
submit, keep the candidate bindings alive through completion, verify that every
actual routed union is covered, and retry unpredicted on any miss before token
commit. This preserves exact final output and bounded resources, but it is a
prediction/retry lane. It is the only current design that can plausibly obtain
all-layer continuous submission without declaring all 128 experts.

### 3. Metal 4 sparse cache with an explicit miss service

This is a future experiment only. It can replace transient `MTLBuffer` view
construction with one canonical sparse virtual buffer and a bounded anonymous
placement heap. The router/miss kernel would emit exact cold IDs; a CPU event
handler would map tiles, issue MTLIO loads, and resubmit the layer. It does not
remove the cold-miss barrier, does not use macOS clean file pages directly, and
must prove raw-bit parity plus bounded heap/residency telemetry before any
runtime integration.

## If an all-resource falsification probe is ever requested

Keep it out of the production runtime and require a child-process watchdog.
The probe should expose one layer only, declare all possible records, record
`MTLResidencySet.allocatedSize`, host wired bytes, reclaimable headroom,
swap-in/out deltas, `mincore` residency by expert, and command-buffer errors,
then exit. It must never construct all 30 dense tables or run alongside port
8181. The expected result is a resource-preparation/wired-memory regression,
not a promotable lane.

## Code-change disposition

No `src/` prototype is justified. The smallest direct-base patch would add a
`LayerMappedBase` binding, a full-span no-copy `MTLBuffer`, a hot-directory
buffer, and base-or-hot address selection in both arg-buffer GateUp and Down.
Those changes are mechanically straightforward but violate the bounded
residency invariant before they can reach a correctness benchmark. The safe
engineering decision is to leave the existing sparse exact path intact and
spend implementation effort on the bounded prediction/retry path or on kernel
time, not on a known all-resource fallback.
