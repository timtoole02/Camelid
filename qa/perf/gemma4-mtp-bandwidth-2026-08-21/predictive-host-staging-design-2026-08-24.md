# Gemma 4 bounded predictive host staging design — 2026-08-24

Status: design audit only. No production source was changed for this report and
no model was run. This is the implementation boundary to use only if the
post-reboot, globally capped prediction receipt clears the go/no-go threshold.

## Recommendation

The smallest correctness-preserving experiment is a **round-owned immutable
host staging map**, started after the MTP assistant has returned the exact
candidate chunk and before the target verifier begins.

The stage may read predicted `(layer, expert)` records into ordinary owned host
memory. It may not plan or lease a Metal slot, mutate a slot directory, publish
`hot_slot_ids`, materialize an argument table, or alter the persistent
`GhostMoeExpertCache`. After the real target router produces a layer's exact
union, the existing `fill_hybrid_hot_slots` transaction remains authoritative:

1. acquire the existing exclusive hybrid refill lease;
2. plan slots from the exact routed union;
3. for each exact plan miss, copy a matching **ready** staged record if one is
   available;
4. otherwise immediately execute today's positioned read;
5. commit only successfully filled exact loads; and
6. publish the validated directory exactly once, as today.

A wrong prediction therefore costs only host bytes and I/O. It cannot become a
route, an address-table entry, or a GPU input.

Do not implement from raw per-layer predictor overlap. Admission should use the
probe's exact **global cap-64/cap-96, hot-filtered, ranked** recall and precision.
The existing predictor returns a canonical-ID-sorted set, which is adequate for
fixed-footprint coverage probes but throws away the confidence order needed by
a 64–96-record global budget. The probe-only ranked helper (maximum per-token
router probability, expert-ID tie break) is the appropriate candidate order to
promote into the experimental planner if its receipt wins.

## Why this is the compatible part of FreeToken

At pinned FreeToken commit `bd372b6`, prefill uses two full layer buffers. A
copy stream waits for the prior release event, fills the next buffer, records a
ready event, compute waits for ready, and compute records release before that
buffer can be reused. Decode is more conservative: it obtains the real top-k,
ensures/copies the missing experts, and only then runs expert GEMM.

- Prefill choreography:
  <https://github.com/FlashML-org/FreeToken/blob/bd372b630a028e3faa51f4ab0ef6a98c2f2de501/python/freetoken/moe/offload_cache.py#L533-L621>
- Decode ordering and prefill call site:
  <https://github.com/FlashML-org/FreeToken/blob/bd372b630a028e3faa51f4ab0ef6a98c2f2de501/python/freetoken/layers/moe.py#L264-L392>

Camelid should borrow the ready/release ownership invariant, not FreeToken's
full-layer policy. One Gemma 4 expert record is 3,345,408 bytes; 64 staged
records are 214,106,112 bytes and 96 are 321,159,168 bytes. Streaming every
expert of every layer would be orders of magnitude larger than the current
roughly 252 MiB target round.

## Current seams and invariants

The relevant code is all in `src/gemma4_runtime.rs` except for the already-safe
hybrid lease implementation in `src/metal.rs`.

- `Gemma4MtpAssistantMetal::propose_chain[_device_resident]` returns all draft
  tokens before target verification. In the target driver, `chunk` becomes
  final immediately before `verify_mtp_round_timed` (currently around lines
  13,920–13,930). This is the first point where the future target chunk is
  known without using target-router truth.
- `step_chunk_speculative_inner` embeds the candidate tokens, then takes the
  `metal_q4_experts` lock and calls
  `GhostMetalExpertRuntime::execute_chained_round_all_layers` (currently around
  lines 14,140–14,180).
- The chained Metal core waits for each exact GPU router, calls the host
  `slot_filler`, checks the returned exact union, materializes the mixed table,
  and only then commits expert work (`src/metal.rs`, currently around lines
  27,980–28,100).
- The hybrid filler obtains `begin_hybrid_refill`, invalidates old published
  identities before exposing writable bytes, fills complete records, commits
  successful directory loads, validates the bijection, and calls
  `publish_hot_slot_ids` once (`gemma4_runtime.rs`, currently around lines
  3,615–3,840).
- `Gemma4Q4HybridExpertSource::lease_state` in `metal.rs` already provides the
  GPU-side release invariant: positive values are live table leases, `-1` is
  one exclusive refill, and zero is idle. Predictive staging must not weaken or
  duplicate this mechanism.

The new stage is only a record source inside the existing exact fill. It is not
a second cache and not a second directory.

## Minimal data types

Names below are illustrative but intentionally close to a patchable Rust API.

```rust
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PredictiveRecordKey {
    layer: u8,
    expert: u8,
}

#[derive(Debug, PartialEq, Eq)]
struct PredictiveRoundIdentity {
    start_pos: usize,
    // Store the tiny K<=16 token list, not a fallible hash.
    tokens: Box<[u32]>,
}

enum PredictiveRecordState {
    Queued,
    Reading,
    Ready(Box<[u8]>),
    // Exact demand arrived before ready. The worker must skip or discard it.
    Bypassed,
    Consumed,
    Failed,
}

struct PredictiveRecordEntry {
    key: PredictiveRecordKey,
    state: std::sync::Mutex<PredictiveRecordState>,
}

struct PredictiveStageInner {
    identity: PredictiveRoundIdentity,
    record_bytes: usize,
    entries: std::collections::HashMap<
        PredictiveRecordKey,
        std::sync::Arc<PredictiveRecordEntry>,
    >,
    cancelled: std::sync::atomic::AtomicBool,
    counters: PredictiveStageCounters,
}

// Unique, non-Clone round owner. Worker jobs hold Arc<Inner>, never this handle.
struct PredictiveRoundStage {
    inner: std::sync::Arc<PredictiveStageInner>,
}
```

`Drop` for the unique `PredictiveRoundStage` sets `cancelled=true`. The worker
checks it between records. A currently executing positioned read may finish,
but it can only publish into its private entry and is discarded if the entry
was bypassed or the round was cancelled. The worker's `Arc<GhostFile>` keeps
the immutable file/index owner alive during an unload.

Add one cache-level active-stage permit (an `Arc<AtomicBool>` or equivalent
RAII permit) to `GhostMoeExpertCache`. A new round refuses predictive staging
while an older cancelled worker is still finishing. This preserves a hard
one-stage memory ceiling even if a positioned read stalls across a round
boundary. The worker releases the permit from a drop guard on success, error,
or unwind.

The stage should expose only these operations:

```rust
impl GhostMoeExpertCache {
    fn begin_predictive_round_stage(
        &self,
        identity: PredictiveRoundIdentity,
        candidates: Vec<PredictiveRecordKey>,
    ) -> Option<PredictiveRoundStage>;
}

impl PredictiveRoundStage {
    fn matches(&self, start_pos: usize, tokens: &[u32]) -> bool;

    // Never returns a borrowed staged slice. It either copies a fully published
    // exact record under the entry lock or leaves destination untouched.
    fn try_copy_ready_into(
        &self,
        layer: usize,
        expert: usize,
        destination: &mut [u8],
    ) -> PredictiveCopyOutcome;
}

enum PredictiveCopyOutcome {
    ReadyCopied,
    PendingBypassed,
    AbsentOrFailed,
}
```

`try_copy_ready_into` must use `Mutex::try_lock`, not `lock` and not a
condition variable. On `Ready`, it replaces the state with `Consumed`, copies
the exact-sized immutable box, then drops the box. On `Queued` or `Reading`, it
marks `Bypassed` and returns immediately so the exact fill performs its normal
pread. Lock contention or poisoning also falls back immediately. A worker that
finishes a bypassed read discards the bytes rather than republishing them.

This mutex unlock is the host ready publication barrier: a reader can observe
`Ready` only after `read_moe_expert_into` returned `Ok(())` for the full exact
record. No separate relaxed ready bit should be used.

## Candidate planner and bound

Keep candidate selection pure and separately testable:

```rust
fn predictive_stage_candidates(
    ranked_per_layer: &[Vec<usize>],
    resident_tables: &[[i16; 128]],
    max_records: usize,
) -> Vec<PredictiveRecordKey>;
```

Rules:

- accept only canonical expert IDs `0..128` and valid layers;
- remove duplicates within a layer;
- exclude experts that already have a hot anonymous override at stage launch;
- optionally exclude a `GhostMoeExpertCache::peek_resident` hit, because that
  immutable host record is already a zero-I/O source;
- merge by confidence depth across layers (`L0 rank0, L1 rank0, ... L29 rank0,
  L0 rank1, ...`) so an early layer cannot consume the global budget; and
- stop at the exact global record cap.

Round-robin depth also matches the execution schedule: each layer's first,
highest-confidence candidate is issued early, while later layers naturally
have more target-compute time in which to become ready.

Use an exact default-off opt-in, for example:

```text
CAMELID_GEMMA4_GHOST_METAL_PREDICTIVE_HOST_STAGE=1
CAMELID_GEMMA4_GHOST_METAL_PREDICTIVE_HOST_STAGE_RECORDS=96
```

Parse with a pure helper and latch the result at runtime construction. Clamp or
reject outside `1..=96`; do not silently turn malformed values on. Initial
admission should require the hybrid mapped lane, demand promotion and demand
promotion fill, an MTP verifier K in the measured range, and an asynchronous
read pool with at least three workers. Reject or disable simultaneous legacy
previous-union promotion-prefetch/rdadvise experiments so the A/B has one
source of speculative I/O.

## Worker scheduling

Do **not** enqueue 64–96 independent `spawn_fifo` jobs onto the existing Ghost
read pool. The exact fill later calls `pool.install` for authoritative demand
reads; dozens of equal-priority prediction jobs can occupy or queue ahead of
the work that correctness and latency actually need.

Use one `spawn_fifo` coordinator that owns the candidate sequence and performs
positioned reads serially, checking cancellation/bypass between records. On the
normal four-thread H2 pool this reserves at most one worker for prediction and
leaves three for authoritative per-layer reads. A one-worker sequential record
stream is still fast enough to cover much of the roughly 200–300 ms verifier
window; concurrency can be revisited only after receipts show the stage itself,
not prediction quality, is the limiter.

Each iteration is:

1. stop if the round is cancelled;
2. transition one exact entry `Queued -> Reading` under its mutex;
3. allocate one exact-sized zeroed box and call
   `GhostFile::read_moe_expert_into(layer, expert, bytes)`;
4. re-lock the entry;
5. publish `Ready(bytes)` only if it is still `Reading` and the round is live;
6. otherwise discard the box and preserve `Bypassed`/cancellation; and
7. record success/failure counters.

At most the configured count of ready/reading boxes exists. Consumed boxes are
freed immediately. The cache-level active permit prevents two round banks from
coexisting after a slow cancellation.

## Exact call-chain insertion

Prefer direct lifetime plumbing over a `pending_predictive_stage` field on
`GhostMetalExpertRuntime`. The direct form prevents stale stages by type and
avoids another `(expected_start_pos, receipt)` cleanup protocol.

1. In `generate_mtp_assistant_experiment_cancellable_with_prefill_seed`, leave
   the bootstrap call unchanged (`stage=None`). In a normal round, retain the
   existing abort check. After `chunk` is finalized and immediately before
   `verify_mtp_round_timed`, compute ranked advisory routes, snapshot the lane's
   hot tables, build the globally capped cold plan, and start one
   `PredictiveRoundStage`.
2. Add `Option<&PredictiveRoundStage>` to the private
   `verify_mtp_round_timed` call and to a private MTP verifier inner helper.
   Keep the public `step_chunk_speculative_mtp_experiment` API unchanged by
   calling that helper with `None`.
3. Add the same optional reference to `step_chunk_speculative_inner` and
   `GhostMetalExpertRuntime::execute_chained_round_all_layers`. Every generic
   speculative and prefill caller passes `None`.
4. Validate `stage.matches(start_pos, tokens)` before the target lane uses it.
   A mismatch is logged and treated as `None`; never attempt partial reuse.
5. Capture the validated stage reference beside `fill_counters` when building
   `slot_filler`.
6. Extend `fill_hybrid_hot_slots` with
   `stage: Option<&PredictiveRoundStage>`. Pass it only for `routed == true`,
   meaning after exact router logits produced `selected_experts`. Pass `None`
   for the old predicted-wave prefetch, HEAD promotion, prefill handoff, and
   terminal promotion.
7. In both the serial and parallel branches, after a destination is obtained
   from the existing exclusive `Gemma4Q4HybridRefillGuard`, try the exact stage
   key. A ready copy is treated like today's immutable host-cache copy. Pending,
   failed, absent, wrong-sized, poisoned, or contended entries go directly to
   today's `read_moe_expert_into` path.

The owner remains a local in the MTP generation round. Returning from target
verification—success, refusal, error, or cancellation—drops it and cancels
unused work. A refused chained prediction retry may reuse the same immutable
stage only if the exact `(start_pos, tokens)` identity still matches; passing
`None` on retry is the simpler initial rule and only sacrifices an advisory
optimization.

### Why not store it in the lane

A lane field is workable but inferior. It needs stale-position cleanup on
abort, embedding failure, retry, and the next request. It also creates borrow
pressure because the slot-filler closure already holds `&mut self.layers`.
Direct plumbing changes a few private signatures but gives the stage an exact
lexical lifetime and leaves the persistent runtime free of round state.

## Source precedence inside an exact fill

Recommended source order for a planned exact miss:

1. existing persistent host-cache hit (`peek_resident`), if any;
2. matching ready round stage;
3. positioned `.cghost` read into the leased hot slot.

The first two are byte-identical immutable copies. Checking the persistent host
cache first prevents counting a staged hit where no predictive I/O was needed;
the planner should normally have filtered such entries already. The stage API
must copy internally—never return `&[u8]` beyond the entry lock.

Directory behavior is unchanged. In particular, do not call `commit_load` for
a pending/failed stage entry until the direct fallback succeeds, and never call
`publish_hot_slot_ids` from a worker.

## Telemetry required for a truthful A/B

Keep predictive I/O separate from demand-fill I/O and from `slot_filler_ms`.
At minimum receipt:

- planned candidate records/bytes and exact cap;
- reads started, completed, failed, and cancelled-before-start;
- bytes actually read and summed read microseconds;
- ready exact hits and ready-copy microseconds;
- exact-selected entries bypassed while queued/reading;
- ready-but-unused predictions;
- active-stage refusal (previous worker still draining);
- per-layer ready hits; and
- exact stage identity `(start_pos, K)` without dumping token contents.

Do not add background read duration to `slot_filler_ms`: it overlaps target
work. Do not hide predictive bytes inside `demand_loads`/`nvme_bytes`, because
then a faster critical path can look like fewer reads when it actually issued
more. The round-end snapshot may race one final in-flight pread; have the single
worker emit a final keyed receipt when it exits, after cancellation, rather
than waiting in the verifier.

The performance gate is not merely token parity. A successful experiment must
show that staged hits were already ready before their exact layer fill, reduce
direct demand reads and `slot_filler_ms`, and do so without an equal increase in
GPU time or memory-pressure slowdown. Continue to require
`exact_match_expected: true` for every throughput number.

## Model-free tests

Add small-record test constructors so the state machine is tested without a
3.3 MiB production allocation or a model.

1. **Planner bound/order.** Invalid IDs and hot entries are removed, duplicates
   do not consume budget, layer-round-robin confidence order is exact, and both
   cap 64 and cap 96 are hard ceilings.
2. **Exact identity.** A stage matches only the same position and complete token
   slice. Same K with one different token and same tokens at another position
   both refuse.
3. **Full-read publication.** A loader writes a known marker, publishes only
   after success, and `try_copy_ready_into` returns exactly those bytes once.
4. **Destination integrity.** Absent, pending, failed, poisoned/contended, and
   wrong-sized lookups return fallback and leave the destination unchanged.
5. **Pending bypass is nonblocking.** A barrier-controlled loader holds an
   entry in `Reading`; the exact lookup returns through a channel before the
   loader is released, changes the state to `Bypassed`, and the later completed
   read never becomes ready.
6. **Cancellation.** Dropping the unique round owner prevents queued entries
   from starting and prevents the current read from publishing. The active
   stage permit is released after the worker exits.
7. **One-stage ceiling.** A blocked first worker makes a second stage refuse;
   after release/cancellation, a later stage is admitted.
8. **Error isolation.** One failed record does not cancel unrelated entries;
   exact lookup for the failure falls back.
9. **Source choice.** A pure helper proves persistent-host hit > ready-stage >
   direct-read precedence and exact accounting.
10. **Directory noninterference.** Completing and discarding a stage leaves a
    `GhostMetalSlotDirectory` snapshot bit-for-bit unchanged. Only a subsequent
    exact plan+successful copy+commit changes it.
11. **Parallel fill safety.** Multiple ready entries copied by the existing
    parallel fill target pairwise-distinct destinations and produce the same
    bytes/directory as the serial path.
12. **Hybrid lease regression.** Keep the existing Metal test proving a live
    mixed-table lease excludes refill and an abandoned refill invalidates all
    identities. The new stage must not change that test or need a lease itself.

After unit tests, the implementation still needs the frozen 48-token exact
oracle with staging on and off, followed by an interleaved throughput A/B. A
model run that lacks the new final predictive receipt is not evidence.

## Likely implementation and compilation risks

- **Rayon starvation:** many spawned prediction jobs can delay exact demand.
  Use one coordinator and require spare pool capacity.
- **Borrow checker conflicts:** the slot-filler closure already mutably borrows
  `self.layers`. Keep the stage as a local reference/`Arc`, not a field borrowed
  through `self` while constructing that closure.
- **`'static` worker capture:** `ThreadPool::spawn_fifo` cannot capture
  `&GhostMoeExpertCache`, `&GhostFile`, candidate slices, or the round handle.
  Capture owned candidate keys plus `Arc<GhostFile>` and `Arc<StageInner>` only.
- **Raw destination safety:** the parallel hybrid fill already converts
  pairwise-distinct leased destinations into raw pointers. Do not move a
  `&mut [u8]` into a background staging job. Staging reads only its own box; the
  existing fill worker performs the final copy while the refill lease is live.
- **Leaking a staged slice:** returning `&[u8]` from behind a mutex creates a
  lifetime/ownership trap. Expose `try_copy_ready_into` instead.
- **Double publication:** a `Ready` bit separate from the bytes can become
  visible early under weak ordering. Make the mutex-protected enum the sole
  state and publication barrier.
- **Round hash collision/staleness:** store the tiny token slice and compare it
  exactly; do not rely on `DefaultHasher` for authority.
- **Memory overlap at cancellation:** worker-held `Arc<Inner>` can retain every
  ready box after the round owner drops. The cache-level one-active-stage permit
  prevents a second bank until the worker observes cancellation and exits.
- **Panic cleanup:** the active permit needs an RAII reset guard inside the
  worker. Mutex poison must mean immediate direct fallback, never use uncertain
  bytes.
- **Accounting races:** round return can precede one final background read.
  Emit a worker-exit receipt rather than joining on the critical path.
- **Environment tests:** production `OnceLock` flags are hostile to tests that
  mutate environment variables. Test a pure parser/resolver and latch once at
  construction.
- **`cfg` drift:** keep the core planner/state under
  `cfg(any(target_os = "macos", test))` and Metal launch/fill wiring under
  `cfg(target_os = "macos")`; check a non-mac build if CI covers it.
- **Signature churn:** there are generic speculative, MTP, prefill, and recursive
  chained-round call sites. Preserve public APIs with private `*_inner(...,
  stage)` helpers and pass `None` everywhere outside the official MTP path.
- **Predictor ordering:** do not feed the existing canonical-ID-sorted route set
  into a globally capped stage. Promote the probe's deterministic ranked helper
  only after its cap-specific receipt is accepted.
- **False performance wins:** speculative reads can warm the OS page cache even
  when no stage entry is consumed. Receipt both total predictive bytes and
  ready exact hits, and use interleaved runs.

## Go/no-go

Use the already agreed lower bound: do not implement the staging fill unless
the post-reboot predictor shows roughly **30% or better residual-miss recall at
the actual global cap**, with enough precision that the 64–96 reads are not
mostly pollution. Prefer cap 96 only if it materially improves ready-hit count
over cap 64; otherwise the smaller 204 MiB bank is the safer first experiment.

Even a successful stage is a 25–45 ms-class lever on the measured path, not a
standalone route to 35 tok/s. Its value is that it attacks the genuinely idle
post-router fill while preserving the exact router/fill/publication barrier.
