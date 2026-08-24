# The mapped-cold round cost is per-ROUND, not per-record — 2026-08-24

Machine: 16 GiB Apple M4. Branch `codex/gemma4-50tps-v2`. Model pair
`~/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.{hot.gguf,v3.cghost}` plus the
full-Q4 MTP assistant. Fixture: the frozen 48-token
`hybrid-hot48-runner/request-48.json`, compared token-for-token against
`expected-48-token-ids.json`. **Every row below is 48/48 exact.**

## Result

| config | round | decode tok/s | HTTP wall |
|---|---:|---:|---:|
| frozen 50 tok/s gate profile (baseline) | 9530 ms | 0.62 | 210.7 s |
| exact-union binding + demand promotion | 609 ms | 10.88 | 15.5 s |
| + 48 hot slots/layer | 451 ms | 14.51 | 12.5 s |
| + no redundant post-round fill | 354 ms | 19.14 | 11.0 s |
| + union-proportional hot profile (1786 slots, 5.59 GiB) | **294-306 ms** | **22.0** | 10.6 s |

128-token request on the same config: 291.8 ms/round, **23.86 tok/s**, alpha
6.67, first 48 token IDs identical to the fixture.

## The defect

`execute_chained_round_all_layers` encodes the unified decode command buffer
BEFORE the GPU router runs, so it cannot know the routed union and declares
`let active: Vec<usize> = (0..num_slots).collect()` — all 128 canonical records
— for every layer. Against a file-mapped hybrid source that binds 30 x 96 =
2,880 no-copy mmap records, ~9.65 GiB of file-backed buffers, every round.

The cost is flat in the number of records actually routed. Two measurements
prove it rather than the intuitive "cold records are slow" story:

- `DECODE_PROMOTION=1` (terminal promotion, shipped but disabled in the frozen
  profile) cut mapped-cold selections 94 -> 62 per round. Round: 9530 -> 9727 ms.
- `MAPPED_RDADVISE=1` on top: 9413 ms.

Meanwhile the same records read explicitly through the Ghost read pool move at
**2.1-6.5 GB/s** in the same session. Terminal promotion can never reach zero
mapped selections anyway, because ~10-25% of each round's union is newly routed.

## The fix (all opt-in, default off)

`CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION=1`

1. Excludes `record_demand && is_decode_hot && is_decode_round` from
   `unified_single_cb` for a file-mapped source, so the round takes the
   per-layer route -> fill -> materialize-exact-union schedule.
2. Promotes the round's exact routed union into the layer's anonymous hot
   records inside the slot filler, BEFORE the mixed table is materialized, so
   the command binds zero mapped records.
3. Disables the overlap pre-materialization for that source: without a
   predicted union it fell back to the same all-128 bind.
4. Skips the post-round filler pass, which the per-layer schedule makes
   redundant — it was a second full fill over the same unions (231 loads
   against 1 directory miss, evicting as many records as it loaded).

`CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_PROFILE_FREE=1` lifts the frozen
960-slot budget so `..._HYBRID_HOT_SLOTS_PER_LAYER` can be sized per layer.
Cap is memory: 2,400 records = 8.06 GiB.

Fail-open by construction: a refused refill (live table lease, or a union wider
than the layer's hot budget) simply retains the mapped fallback. Correctness
never depends on the switch.

## Sizing: the per-layer union is NOT uniform

Measured at 48 uniform hot slots/layer, K=8, with the per-fill trace
(`..._DEMAND_PROMOTION_TRACE=1`):

- per-layer routed-union medians run **25.5 to 39**, maxima **31 to 52**
- layers 0-1 and 25-29 are the wide ones; layers 3-20 the narrow ones
- at 48 uniform slots the wide layers thrash: layer 5 was observed evicting
  experts it reloaded on the very next round

An equal partition is therefore the wrong shape. The profile below is ~2x each
layer's measured median, clamped to [40, 96], totalling 1786 slots (5.59 GiB).
It beats uniform-64 (1920 slots, 6.01 GiB) — 306 ms vs 322 ms — on LESS memory:

```
74,78,57,55,52,53,56,54,60,58,52,53,56,59,54,63,52,51,58,60,53,59,57,55,59,70,73,69,72,64
```

This is the static approximation of a global expert pool. The dynamic version
(one pool, borrowed per layer per round) is the obvious follow-up.

## Negative results, with receipts

- **`SLOT_POLICY=lru`**: 724 ms/round vs 609 ms for the default LFU. Keep LFU.
- **Pre-router speculative prefetch from the previous round's routed union**
  (`..._DEMAND_PROMOTION_PREFETCH=1`, default off): 435.1 ms and 15.74 tok/s
  against 306.0 ms and 21.52 tok/s with it off. Once the hot tier is sized to
  the union, the previous round's union is already resident, so the speculative
  pass finds nothing to fetch (`wave_load` 0.2 ms) while still paying a
  plan/publish transaction and an LFU perturbation per layer. The residual
  reads are by construction the experts this round routes for the FIRST time;
  hiding those needs a predictor that sees the future token (the MTP assistant's
  drafts), not the past one.
- **`DECODE_PROMOTION` / `MAPPED_RDADVISE`**: see above, both inert against the
  real cost.

## Separately: the anonymous-slot lane is numerically broken on this branch

With `FILE_MAPPED_EXPERTS` unset, both slot geometries produce degenerate text
on the same binary and prompt:

| config | first tokens | text |
|---|---|---|
| `PHYSICAL_SLOTS_PER_LAYER=56`, `SLOTS_PER_LAYER=88` | 140, 9430, 2456, ... | `    pub struct CacheEntry < < < 1\`3\`3\`20 debric,` |
| `SLOTS_PER_LAYER=88` monolithic | — | ``    `pub` the entire struct definition again, as though it`` |

Both run fast (131 ms and 142 ms per round) and both are wrong. The hybrid lane
is the only correct one on this branch. Unfixed; not on the critical path for
throughput, but it means no throughput number measured on that lane is real.

## Round anatomy at 22 tok/s

```
round 294-306 ms = assistant 34 ms + verifier ~260 ms
  GPU 115-117 ms   (qkv_o ~37, shared ~20, gateup ~60)
  reads 252 MiB / ~100 ms at ~2.5 GB/s
  final_wait 2.4 ms, encode 1.0 ms   <- GPU is fully overlapped with host encode
  slot hit rate 1.000, misses 0/round
```

Reads and GPU are still serialized: `slot_wait` tracks `gpu_busy`, then the fill
runs with the GPU idle. That ~100 ms is the next target. The per-layer loads are
issued serially inside `fill_hybrid_hot_slots`; `bench-read-batch-depth.py` says
depth, not thread count, is the limiter (batch 7 = 2.21 GB/s, batch 28 = 4.01),
so issuing a round's loads at greater depth is worth ~1.7x on that term.
