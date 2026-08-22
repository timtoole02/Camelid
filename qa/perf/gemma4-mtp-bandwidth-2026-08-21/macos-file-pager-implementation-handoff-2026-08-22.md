# macOS File-Pager Implementation Handoff — 2026-08-22

## Status

The persistent-mapped-resource wired-memory staircase is fixed for the strict
K=1 and K=8 lanes. The production Metal runtime now stores mapped expert layers as
descriptor-only sources and creates sparse no-copy views for only the exact
routed union of one in-flight layer.

The 28–34 tok/s target is **not promoted**:

- After a reboot, the corrected K=8 48-token receipt passes every VM safety
  gate with exact token-ID/text parity against K=1. Pure decode is 11.5 tok/s
  (48 tokens in 4.190 s); whole-request generation is 17.076 s including
  prefill.
- The paired K=1 receipt also passes after K=8, with 15.389 s whole-request
  generation. It is a warm-state safety/parity receipt, not a cold-throughput
  comparison.
- K=8 still fails the requested compute gates: 11.5 tok/s is below 28 tok/s,
  and multiple exposed assistant chains exceed 35 ms.

Disk recovery is complete: `/System/Volumes/Data` had about 23 GiB available
and was 88% used for these runs, so the earlier disk-space refusal no longer
applies.

## What changed

### P1 — bounded writable slots

`Gemma4Q4ExpertSlots::new_record_granular_with_table_slots(logical, physical)` now allocates exactly `physical` Metal records. The logical tail is metadata only and has no `MTLBuffer`, writable pointer, table entry, or pin attempt.

For 30 layers at the exact 3,358,720-byte stride:

| Physical slots/layer | Anonymous base capacity |
|---:|---:|
| 48 | 4,836,556,800 B / 4.504395 GiB |
| 56 | 5,642,649,600 B / 5.255127 GiB |

`pin_working_set()` iterates only those allocated records. `mlock` success is still runtime/OS-dependent; there is no claimed universal “10 GiB macOS ceiling.”

Telemetry now distinguishes:

- logical slot metadata;
- actual anonymous slot count/capacity;
- GPU-addressable table width; and
- clean file-mapped address span.

### P2 — production clean file-backed experts

Mapped layers are persistent descriptors only: `Arc<GgufWireMmap>`, layer
offset, and validated span. They own no persistent record `MTLBuffer` objects
and no persistent 128-entry Metal argument table. Once routing is known, the
runtime builds a sparse `Gemma4MoeSlotArgTable` over only the distinct active
canonical records. The transient table retains `Arc<GgufWireMmap>`, so its
no-copy views cannot outlive their source bytes.

The new backing has these invariants:

- 128 canonical records per layer, `expert e -> slot e`;
- 12,897,484,800 B / 12.011719 GiB total mapped address span across 30 layers;
- 0 B anonymous expert payload capacity;
- 0 victim records;
- 0 overflow slots;
- 0 expert-page `mlock`;
- no writable slot API;
- only the routed active union has no-copy record views or `use_resource`
  declarations;
- original expert slot IDs remain the sparse table's 0…127 address indices;
- the binding, record views, argument table, and mmap remain owned until the
  terminal status of the command that uses them;
- the next layer's queue-order router barrier retires that ownership; and
- any construction/command failure is terminal before CPU fallback or token commit.

The existing HEAD K=1 and chained K=1…16 argument-buffer kernels are reused; no new MSL arithmetic was introduced.

Exact production opt-in:

```text
CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1
CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1
```

Required companion settings for a promotion run:

```text
CAMELID_GEMMA4_GHOST_METAL_SLOTS=1
CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1
CAMELID_GEMMA4_GHOST_METAL_COMMON=1
CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS=0
CAMELID_GEMMA4_CHAINED_PREDICT=0
CAMELID_GEMMA4_MTP_OUTER_PIPELINE=0
CAMELID_GEMMA4_MTP_PREFETCH=0
CAMELID_GEMMA4_OPTION_B=0
CAMELID_GEMMA4_VICTIM_CACHE=0
CAMELID_SPEC_DECODE=off
CAMELID_GEMMA4_SPEC_K1_LANE=chained
```

Do **not** set `CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER` in mapped mode. That knob selects the separate bounded writable-cache architecture; the runtime rejects the ambiguous combination.

Do **not** pass `--evict-page-cache`. That option deliberately removes the `.cghost` mmap and uses `F_NOCACHE` positioned reads, while mapped mode explicitly delegates clean-page eviction/reload to the macOS file pager. The runtime returns a typed error rather than silently remapping.

Use `--expert-cache-mib 0`. The promotion harness now requires that exact zero-byte host-cache profile so a retained host tier cannot obscure the memory result.

### Post-review fail-closed hardening

The mapped implementation and benchmark entry points received a second independent review. Four gaps were confirmed and fixed:

- The observed-load constructor and the generation harness now select the same mapped backing. The harness sets `CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1`, omits the physical-slot knob, uses a zero-byte host cache, and passes `evict_page_cache=false`.
- A committed asynchronous expert command now owns a cloned slot binding until completion. For mapped records, that binding retains the source mmap; callers cannot drop the runtime and unmap indirect record bytes while the GPU is in flight.
- Static identity directory entries are reported as mapped addressability, not anonymous occupancy. Live receipts show zero anonymous physical slots/occupied/touched bytes plus a separate mapped slot count and address span.
- CPU-fallback policy is latched from the admitted backing at construction rather than reread from mutable environment state. Both chained and scalar Metal expert paths terminate on a mapped/bounded refusal instead of silently executing host-cache experts.

### P3/P4 — evidence correction

- The assistant LM head is already mandatory Q4_0. Both proposal paths call the Q4_0 GEMV and there is no BF16 LM-head fallback.
- Prior steady K=8 assistant chains were about 58–61.5 ms (median 60.24 ms), so the directive's ≤35 ms gate currently **fails**. The 460 ms total was not evidence of a BF16 fallback.
- K=8 is one target anchor plus at most seven assistant drafts, not eight drafts.
- Outer lookahead is already default-off and exact opt-in only. Setting it to `0` keeps it off, but cannot guarantee a nonzero acceptance count.
- Prior draft acceptance was 39/54 = 72.2%, below the requested 85%, with two zero-accept rounds.
- `verify_fails` is victim-cache byte sampling and is unrelated to token parity.

## Verification completed

All commands used the scratch performance worktree and current source.

- `cargo check --lib`: pass.
- `cargo build --release`: pass.
- Integration harness compile: pass (`cargo test --test gemma4_mtp_assistant_experiment --no-run`).
- Physical-only bounded allocation/pinning test: pass.
- Exact mapped opt-in parser test: pass.
- Exact mapped benchmark-profile test: pass; zero host cache, no physical-prefix knob, outer pipeline/prefetch/prediction off, retained page cache.
- Static identity directory test: pass; arbitrary order/duplicates returned identity hits with zero loads/evictions.
- Production mapped-slot adapter test: pass; read-only, zero anonymous capacity, zero pin bytes, mmap retained after construction owners dropped.
- F_NOCACHE conflict test: pass; a `GhostFile` without its retained mapping is refused before mapped Metal construction.
- Real mapping/mincore bounds test: pass.
- Real layer 26 K=1…K=8 GateUp scales/quants and Down outputs: raw-bit exact against copied-slab kernels for every K.

Real layer 26, K=8, 30 active experts, active-only resource declarations:

| Case | Wall | Page-ins | Swap-ins |
|---|---:|---:|---:|
| mapped cold | 74.338 ms | 5,910 | 0 |
| mapped warm (first sample) | 4.126 ms | 0 | 0 |
| mapped warm samples | 4.004–5.977 ms | 0 | 0 |
| copied warm samples | 3.675–6.919 ms | 0 | 0 |

This proves correctness and warm reuse for one layer. It does not prove a 30-layer round latency; a cold K=8 union near 30 experts/layer addresses roughly 2.8 GiB of records, so the throughput target depends on page reuse and cannot be inferred from SSD headline bandwidth.

All 30 mapped tables also constructed successfully with the exact 12.011719 GiB clean address span and zero anonymous expert capacity. Watchdog receipt:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-all-layers-construction-2026-08-22-v3/`

The test child returned 0; the watchdog reported `watchdog_aborted=false`, no abort reasons, pressure level 1 throughout, unchanged swapout pages, and minimum reclaimable headroom 5,814,730,752 B. The wrapper returned 74 only because this unit test does not emit the probe-specific report file. The watchdog's direct-process footprint sampled Cargo rather than the spawned test executable, so that particular process-footprint number is not a promotion claim.

### Post-space-recovery full-model receipts

The first full K=1 run exposed a second defect in the original mapped design.
All 3,840 no-copy record views and all 30 dense argument tables were persistent.
The watchdog stopped the process when reclaimable headroom fell to
2,081,210,368 B. Host wired memory rose from about 2.19 GiB to 9.05 GiB while
the child footprint remained about 1.92 GiB. Receipt:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-promotion-48t-2026-08-22/k1/`

That failure caused the descriptor-only/sparse-transient implementation above.
Model-free sparse-ID, unbound-slot fail-closed, mmap lifetime, dense K=1…K=8
raw-bit parity, and real production `.cghost` adapter tests all pass.

The final strict K=1 receipt is:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-promotion-48t-2026-08-22-v3/k1/`

It records:

- 48/48 completion tokens and `finish_reason=length`;
- `child_returncode=0`, `watchdog_aborted=false`, and no abort reasons;
- pressure level 1 and unchanged swapout pages throughout;
- minimum reclaimable headroom 4,749,869,056 B;
- peak child physical footprint 1,880,452,448 B;
- peak host wired memory 4,219,797,504 B; and
- 18,190.124 ms whole-request generation time.

The watchdog also had a signal-mask defect: SIGINT/SIGTERM/SIGHUP were blocked
across `Popen` and restored only in the parent. Children inherited them blocked,
so a completed response could idle until a later host event invalidated the
receipt. The child now restores the original mask before `exec`, and the plain
HTTP server uses graceful Ctrl-C shutdown. The K=1 receipt proves exit code 0.

The paired K=8 attempt is a failed receipt:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-promotion-48t-2026-08-22-v3/k8/`

It was stopped during the first prefill pass when swapouts increased by 20
pages. At that sample pressure remained level 1, reclaimable headroom was
4,912,807,936 B, and child footprint was 1,847,504,152 B. No response existed
and the assistant had only been staged, not loaded. Do not attribute this abort
to assistant compute or report a K=8 throughput number.

### Fresh-reboot K=8/K=1 receipts

The fresh reboot was used for K=8 first so a preceding 12-GiB K=1 mmap walk
could not dirty its baseline. Receipt:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-promotion-48t-2026-08-22-v4/k8/`

It records:

- 48/48 completion tokens, `finish_reason=length`, and exact token-ID/text
  equality with the passed K=1 response;
- `child_returncode=0`, `watchdog_aborted=false`, and no abort reasons;
- pressure level 1 and exactly zero swapout-page growth throughout;
- clean-parent swap usage of zero bytes and zero pages;
- baseline reclaimable headroom 6,945,554,432 B;
- minimum reclaimable headroom 4,548,214,784 B;
- peak child physical footprint 2,021,470,208 B;
- Q4_0 assistant LM head, mapped experts, zero anonymous expert capacity, and
  outer lookahead disabled;
- 12.146 s prefill plus 4.190 s decode for 48 tokens (11.5 tok/s pure decode);
  and
- 17,075.697 ms whole-request generation time.

The v4 acceptance denominator needs care. There are 48 actual proposed drafts
and 39 accepted drafts, so all-round acceptance is 81.25% and fails the 85%
gate. The six full K=8 rounds are 36/42 = 85.71%, with no zero-accept full
round, but all six exposed assistant chains exceed 35 ms. Excluding the cold
first chain, their average is 54.95 ms. The footer's `accepted=39/65` is not an
acceptance rate: 65 sums configured `requested_k` values, including anchors and
unused tail budget, while 39 counts accepted drafts only.

K=1 was then run after K=8 to produce a same-v4 paired receipt without
contaminating the fresh K=8 baseline:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/mapped-promotion-48t-2026-08-22-v4/k1/`

It also completed 48/48 with exit code 0, pressure level 1, zero swapout growth,
minimum reclaimable headroom 4,853,006,336 B, peak child physical footprint
1,862,085,912 B, and exact K1/K8 token-ID/text parity. Its 15,389.324 ms
whole-request time reflects a warm post-K8 host state.

## Remaining performance work

The clean mapped pager foundation now has the required K1/K8 VM and parity
evidence. The remaining work is compute/residency optimization, not another
claim that macOS swap policy alone can reach the throughput target.

The following gates remain useful for performance promotion:

- 48/48 tokens in both runs;
- exact K1/K8 token-ID and text parity;
- log proves `lm_head=q4_0` and full rounds show `requested_k=8`, `proposed_k=7`, `verifier_k=8`;
- every outer-lookahead field false/zero;
- no zero-accept full round;
- `accepted_drafts / proposed_drafts >= 0.85` if retaining the directive's gate;
- each full assistant chain ≤35 ms if retaining that gate (known to fail current evidence);
- at least 28 tok/s pure MTP throughput;
- pressure level 1 throughout;
- zero swapout-page growth;
- at least 2 GiB reclaimable headroom;
- no watchdog abort; and
- peak physical footprint ≤7.5 GiB.

## macOS operating conclusion

The safe handoff is not “make macOS swap harder.” It is:

1. keep immutable weights in read-only file mappings so pressure can discard clean pages instead of writing anonymous copies to swap;
2. strictly bound any writable hot tier and pin only its real physical records;
3. avoid duplicate host caches and incompatible `F_NOCACHE`/mmap policies;
4. preserve enough disk headroom for the VM subsystem; and
5. promote only from fresh-process pressure, swap, parity, and throughput receipts.

## Neural Engine side quest

Core ML/ANE is a viable compute experiment, not a pager replacement. The recommended order is a small batched 30-router placement/compression smoke test, followed—only if actual ANE execution is proven—by a fixed single-dispatch K=7 assistant graph. Do not move the authoritative target or 12.011719-GiB file-paged expert store into Core ML.

Detailed design, memory boundaries, Apple platform constraints, and promotion gates:

`qa/perf/gemma4-mtp-bandwidth-2026-08-21/macos-coreml-ane-offload-handoff-2026-08-22.md`
