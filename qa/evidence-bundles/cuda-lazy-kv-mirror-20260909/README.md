# Lazy CUDA KV mirror — A/B, 2026-09-09

Receipt for making the GPU→host KV mirror lazy after a CUDA-resident prefill.

## What this changes

The prefill used to copy the whole GPU KV cache back to the CPU cache on **every
request**, justified by a comment saying the copy was "a few MB of device->host
transfer, negligible next to the prefill compute it follows". Both halves had
stopped being true:

- It was never a few MB. The copy is `n_layers × n_kv × positions × head_dim × 2`
  elements — it scales with the **whole context**, not with the new tokens. For a
  3B (28 layers, 8 KV heads, head_dim 128) at 1589 positions that is ~182 MiB on
  the wire, plus ~13 MiB of transient host buffers allocated and freed per layer
  as `read_kv_layer` expands f16 to f32.
- It was negligible next to a *cold* prefill. Once prefix continuation (PR #737)
  cut prefill down to the newly appended tokens, there was no longer a large
  compute for it to be negligible next to.

Measured before the change, with the mirror timed separately for the first time:
it was **62–74% of a follow-up turn's reported prefill time**. It had been
invisible because the `GPU prefill N tokens in X ms` trace was printed *after* the
mirror using a timer started before it, so the mirror's cost was folded into the
prefill number.

The copy is now performed by `ensure_cpu_kv_materialized` at the moment a CPU
reader actually needs the history. On the common path — GPU prefill, GPU decode,
no fallback — nothing reads it and it is never paid.

## Result

Llama 3.2 3B Q8_0 fully resident, RTX 3060 Laptop, growing 5-turn conversation,
**one release binary** with the arm set by `CAMELID_CUDA_EAGER_KV_MIRROR`. ABBA
ordering (lazy, eager, eager, lazy) with a cool-to-55 °C gate between arms.

Request wall time in ms:

| turn | prompt tok | lazy-1 | lazy-2 | eager-1 | eager-2 |
|-----:|-----------:|-------:|-------:|--------:|--------:|
| 1 | 1437 | 9828 | 10000 | 10582 | 10642 |
| 2 | 1474 | 596 | 671 | 1139 | 1193 |
| 3 | 1512 | 586 | 602 | 1412 | 1194 |
| 4 | 1550 | 635 | 644 | 1248 | 2014 |
| 5 | 1590 | 577 | 578 | 1250 | 3247 |

Follow-up turns (8 samples per arm):

- lazy: median 599 ms, mean 611 ms
- eager: median 1249 ms, mean 1587 ms
- **median 2.09×, mean 2.60×**

The median is the headline; the gap between median and mean is not noise, it is a
property of the eager path (see the tail below).

**Turn 1 improves too**, unlike prefix continuation: 9914 ms vs 10612 ms paired,
~700 ms saved, which matches the measured mirror cost at 1436 positions. The very
first prefill pays the mirror as well, so removing it helps a cold request.

## The tail is the mirror, not noise

`eager-2` ran third of four and was the slowest arm, while `lazy-2` ran last and
was as fast as `lazy-1` — so this is not thermal drift. The trace shows the
mirror itself ballooning within that arm:

```
KV mirror to host: 1511 positions in  567 ms
KV mirror to host: 1549 positions in 1184 ms
KV mirror to host: 1589 positions in 2134 ms
KV mirror to host: 1436 positions in 1371 ms
KV mirror to host: 1511 positions in 1851 ms
```

Same position counts, 3–4× the time. The mirror allocates and frees ~13 MiB of
f32 staging per layer per request (28 layers), so it is allocator- and
page-fault-bound as much as PCIe-bound, and degrades under sustained load on a
box with ~4.8 GiB free RAM. Its cost is not just large, it is **unpredictable** —
which is a second reason not to pay it on every request.

## Mechanism and correctness

The lazy arm emits **zero** `KV mirror to host` lines across all five turns
(`trace.txt`) — the copy genuinely never runs, rather than merely running faster.

**Output is identical.** The same 5-turn conversation replayed with replies
recorded on both arms (`replies-lazy.txt` vs `replies-eager.txt`) is
byte-identical, and all four arms agree. Only *when* the copy happens changes;
the KV it produces is the same bytes either way.

**The regression test for this exact hazard passes with a real model.**
`tests/resident_kv_cpu_fallback.rs` self-skips without
`CAMELID_RESIDENT_KV_FALLBACK_GGUF` (it reports ok vacuously), so it was run with
one set:

```
pure-CPU              : [22463, 13, 578, 6864, 315, 18157, 374, 25048]
resident, no fallback : [22463, 13, 578, 6864, 315, 18157, 374, 25048]
resident, one fallback: [22463, 13, 578, 6864, 315, 18157, 374, 25048]
```

A CPU fallback mid-sequence still sees the real history, because
`ensure_cpu_kv_materialized` recovers it on demand. The speculative verify-chunk
sibling test passes the same way. Both genuinely ran (107 s, real tokens).

## Why lazy is safe

Every consumer of the CPU KV history already materializes on demand. PR #487
established that all three CPU forward readers call `ensure_cpu_kv_materialized`
(`forward_layer_range_from_hidden`, `forward_single_token_timed_internal`, and
the verify path — the last one's comment names the other two). This change adds
the fourth and last consumer: `rollback_to_position`, which gates on
`cpu_kv_authoritative()` and is reached by GPU speculative decode. Without that
addition it would have started returning `speculative_rollback_failed`.

## Tradeoffs

The lazy recovery's guard is `filled() == position`. A reseed is needed precisely
when `filled != position`. **Those two conditions are mutually exclusive**, so in
the exact case where the GPU needs re-seeding, the lazy recovery also declines.
Everything below follows from that.

### 1. Speculative rollback — closed structurally, not by argument

`rollback_to_position` requires `cpu_kv_authoritative()`, and a speculative
rollback can run with drafts written past `position`, which is the case the lazy
recovery cannot satisfy. That would turn a working opt-in feature into a
`speculative_rollback_failed` 503.

**This was NOT closed by reasoning about reachability.** Attempts to exercise GPU
speculative decode on this lane were all uninformative — identical replies prove
nothing if speculation never ran; `CAMELID_SPEC_VERIFY_TRACE` turned out to print
only from Metal paths, so it is structurally blind to CUDA; and a 400-token
maximally-repetitive generation produced no CUDA-visible signal either. There is
no per-request speculative field to observe.

So the mirror simply stays **eager whenever the request may speculate**
(`cpu_kv_mirror_eager`, default `true`, cleared by `prepare_generation` only when
`speculative.is_none()`). The default direction is the point: any caller this was
not audited against keeps the historical behaviour.

### 2. Reseed with a hollow history — REAL, and not fully closed

If the resident engine is rebuilt or evicted **between** this request's prefill
and a later decode step of the same request, the reseed and the recovery decline
for the same reason, and that forward attends over a zero-filled prefix. That is
**degraded output**, surfaced only as a one-shot stderr warning a user will not
see. The eager copy left a host copy that covered this.

It requires the engine to be displaced *during* a request, which the CUDA lane's
run-to-completion scheduling makes very hard to reach — but that scheduling was
not verified against a concurrent model load/unload, so this is mitigated, not
proven unreachable. `CAMELID_CUDA_EAGER_KV_MIRROR=1` restores the old behaviour
everywhere.

### 3. The first CPU fallback pays the copy inline

Previously the cost was spent at prefill, so a fallback was cheap. Now the first
fallback pays the full copy as a latency spike on that one token. Net still far
better, because fallbacks are rare — but it is moved, not removed.

### 4. The invariant's gate is a deletion-guard only

`check-cuda-prefill-parity-gate.mjs` now pins the count of
`ensure_cpu_kv_materialized` callers at 4 and the safe default at
`cpu_kv_mirror_eager: true`, and both mutations were negative-tested (removing a
call and flipping the default each fail the gate). But grep cannot know what
*reads* the history, so it cannot catch a NEW reader added without the call. That
reader would get degraded output rather than a failure. The gate says so in a
comment addressed to whoever adds the fifth one.

## Caveat

One host, one model, one prompt shape. The saving scales with context length,
since the removed copy is linear in positions — larger at 8k, smaller at 500
tokens. The ratio also depends on reply length, as the saving is a fixed
per-request cost: these turns emit ~8 tokens, so the whole-request ratio is close
to the prefill-phase ratio. A long generation amortizes it, exactly as with
[prefix continuation](../cuda-prefix-continuation-20260909/README.md).
