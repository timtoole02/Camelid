# 48-hot / mapped-cold hybrid receipt ladder

This directory defines the fail-closed runner for the first real Gemma 4 hybrid
lane. It does not build binaries and it never advances to another rung by
itself. Each invocation runs exactly one of these stages:

1. `load-only`
2. `smoke-k8`
3. `smoke-k1`
4. `promotion-k8`
5. `promotion-k1`

K8 deliberately runs before K1 in both pairs. The K1 invocation writes the
pair's parity receipt only after its own lane passes. A missing, failed, or
modified predecessor leaves the next stage unspawned. Starting `promotion-k8`
also requires the operator to set
`CAMELID_HYBRID_PROMOTION_ACK=smoke-parity-reviewed`.

## Status at source base `a3e6ff293`

The runner is ready for review, not for a real model launch on that source
base. It refuses before loading a model unless matching integration contracts
are installed beside the frozen binaries:

- The committed load-only harness at that base still encodes the older bounded
  88/56 profile and file-cache eviction. The hybrid load-only harness must emit
  report schema v4 and prove retained mapping geometry before its contract can
  be issued.
- The HTTP response does not yet emit structured, every-completed-round,
  every-layer hybrid binding telemetry. Human log summaries are insufficient,
  so smoke and promotion stay blocked.
- The assistant currently has a separate hard-locked mapping. There is no
  consumed assistant pin switch in this profile. Its actual `mapped_bytes` and
  `locked_bytes` are preserved from `assistant_ledger`; they are not inferred
  from the pageable hot-expert policy.

Do not hand-author a contract to bypass either integration blocker. Produce it
as part of the build/harness qualification that adds the matching receipt.

## Frozen execution profile

The child receives a clean `env -i` environment. The defining hybrid controls
are:

```text
CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1
CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1
CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=48
CAMELID_GEMMA4_SLOT_PIN=0
```

`CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER` is absent. The runner
also refuses if that or either older slot-distribution alias is inherited,
even though the timed child is launched through `env -i`.

K1 additionally sets the consumed `CAMELID_GEMMA4_CHAINED_K1=1` selector.
Without it, ordinary non-speculative serving would use the HEAD step and could
not produce the same chained per-layer ledger required of K8.

`SLOT_PIN=0` labels the 1,440-slot anonymous hot tier as pageable. It does not
claim that the independent MTP assistant mapping is unpinned. The runner does
not set unused zero-looking controls such as `CAMELID_GEMMA4_MTP_PIN`,
`CAMELID_GEMMA4_PREFILL_SLOT_RESERVE`, `CAMELID_STARTUP_WARMUP`,
`CAMELID_GEMMA4_MTP_OUTER_PIPELINE`, `CAMELID_GEMMA4_MTP_PREFETCH`, or
`CAMELID_GEMMA4_OPTION_B`.

The startup receipt must contain exactly one of each:

- `HYBRID ACTIVE` with 30 layers, 128 canonical IDs/layer, 48 physical hot
  slots/layer, 4,836,556,800 hot bytes, 12,897,484,800 mapped-span bytes, zero
  overflow/victim, pin off, and prediction off.
- The exact clean file-pager geometry line.
- The demand-prewarm-skipped line.

## Fresh-baseline and live safety gates

Every stage gets a new watchdog process and a new child process group. Before
the child is spawned, the watchdog records a durable sample every 250 ms for at
least 60 seconds. Every baseline sample must have:

- current swapped pages exactly zero;
- flat swap-in and swapout counters;
- normal pressure level (`1`);
- at least 8 GiB reclaimable headroom (`strict free + inactive`);
- no more than 8 GiB host wired memory; and
- telemetry plus durable-write latency below 250 ms.

The shell separately requires at least 20 GiB available on the Data volume,
no more than 90% used, and a clear TCP port 8189.

After spawn, the watchdog samples the complete saved process group, not only
its leader. It aborts on any telemetry loss, swap movement, nonzero current
swap, pressure change, less than 2 GiB reclaimable headroom, more than 7.5 GiB
aggregate child physical footprint, more than 8 GiB host wired memory, or a
250 ms telemetry/durability overrun. TERM/KILL cleanup targets the saved group
even if its leader has already exited. A lane cannot pass until the group is
gone and port 8189 is independently observed clear.

The 4.5 GiB hot tier is pageable by design. If macOS compresses/swaps it, that
is a safety-rung failure and evidence for a separate explicitly pinned-hot
trial; it is not evidence that the hybrid address/binding arithmetic was
wrong. This ladder never silently relaxes the zero-swap gate.

## Receipt layout

The default root is:

```text
qa/perf/gemma4-mtp-bandwidth-2026-08-21/
  hybrid-hot48-mapped-cold-2026-08-22-v1/
```

It is overrideable with the task-specific
`CAMELID_HYBRID_RECEIPT_ROOT`. The root must already exist and hold the frozen
executables and integration contracts. Stage outputs are:

```text
01-load-only/
02-smoke-8t/k8/
02-smoke-8t/k1/
02-smoke-8t/parity.json
03-promotion-48t/k8/
03-promotion-48t/k1/
03-promotion-48t/parity.json
```

Every lane directory is new and mode 0700. It contains `intent.json`, a
human-readable `baseline.txt`, the watchdog JSONL, child log, child/report or
response JSON, `port-clear.json`, and finally `verdict.json`. Existing lane or
verdict paths are never reused. The intent freezes the source/binary/tooling
and predecessor hashes, boot identity, disk gate, input file stat identities,
and exact hybrid profile. Large model files are deliberately not re-hashed
after the fresh baseline because doing so would populate the file cache and
change the experiment.

## Integration contracts

The load-only contract is
`hybrid-load-only-schema-v1.json`:

```json
{
  "schema_version": 1,
  "load_binary_sha256": "<sha256>",
  "test_name": "gemma4_mtp_assistant_load_only_probe",
  "hybrid_hot_slots": 48,
  "assistant_residency_receipted": true,
  "evict_page_cache": false
}
```

Its v4 report must have zero tokenize/prefill/step/proposal/generation counters
and the exact final ledger: 3,840 canonical/mapped records, 1,440 anonymous hot
records, 4,836,556,800 hot bytes, 12,897,484,800 mapped span, 240 initially
bound active records, and zero overflow/victim/host cache/touch/prewarm.
The independent assistant ledger must report a nonzero hard lock with
`locked_bytes == mapped_bytes` and `resident_pages == total_pages`; this is
recorded separately from the pageable anonymous expert tier.

The server contract is `hybrid-telemetry-schema-v1.json`:

```json
{
  "schema_version": 1,
  "server_binary_sha256": "<sha256>",
  "response_field": "camelid.hybrid_telemetry",
  "coverage": "every_completed_measured_round_and_layer",
  "q4_assistant_head_fail_closed": true
}
```

The response telemetry itself uses schema v1 and must include:

- record payload 3,345,408 bytes and stride 3,358,720 bytes;
- live total/layer logical-addressable, anonymous-hot-capacity,
  file-mapped-addressable/span, overflow, victim, and host-cache ownership
  facts (hot locked/failed byte counts are not invented from the environment);
- every completed measured round with `success`, exact K fields, and 30 layer
  records;
- for every layer, `active_unique = hot_bound + mapped_bound` and
  `bound_records = active_unique`;
- zero selected-drop, missing-failclose, slot-capacity-overflow, and overflow
  records as round totals;
- an embedded `response.camelid.hybrid_telemetry` route interval labeled
  `measured_request_prefill_plus_generation` and aggregate deltas scoped to one
  `single_completed_measured_request`, with route lookups partitioned into hot
  hits and mapped-cold selections, zero direct reads/host-cache/victim activity,
  and separately consistent `chained_promotion_loads`/read bytes (mapped
  selections may legitimately exceed promotion loads); and
- structured performance metrics used by the promotion verdict.

A K8 union of 49–64 unique experts is a valid mapped-cold spill. It must retain
all records and remain `slot_capacity_overflow=0`; it is not an overflow or a
reason to drop experts. K8 must prove both nonzero hot-tier use and nonzero
mapped-cold selection.

## Manual sequence

After placing qualified frozen binaries/contracts in the receipt root, invoke
one command at a time and inspect the resulting verdict before continuing:

```sh
qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/run_stage.zsh load-only
qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/run_stage.zsh smoke-k8
qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/run_stage.zsh smoke-k1
CAMELID_HYBRID_PROMOTION_ACK=smoke-parity-reviewed \
  qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/run_stage.zsh promotion-k8
qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/run_stage.zsh promotion-k1
```

K1 and K8 use the same deterministic request fixture and must return identical
token IDs and decoded text. Both fixtures explicitly set
`"camelid_receipt": true`; ordinary API requests remain outside this diagnostic
path. The 48-token K8 promotion additionally requires at
least 28 decode tok/s, at least 0.85 accepted/proposed drafts, no zero-accept
full round, at most 35 ms exposed assistant time for every full round, and zero
outer-lookahead activity.

Every K1 round must be requested=1/proposed=0/verifier=1 with zero accepted
drafts and zero assistant CPU/GPU exposure. Its forwarded decode-round count
plus at most one terminal unforwarded token must equal the response completion
count.

Run the model-free checker during review with:

```sh
/usr/bin/python3 \
  qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner/tests/test_hybrid_runner.py
```
