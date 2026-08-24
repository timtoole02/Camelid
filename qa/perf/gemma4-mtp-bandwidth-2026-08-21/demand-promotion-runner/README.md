# Demand-promotion A/B runner

The harness behind `../mapped-cold-round-cost-2026-08-24.md`. Unlike
`../hybrid-hot48-runner/`, this one is **not** a fail-closed proof gate: it has
no watchdog, no baseline settle, and no provenance freeze. It is a fast A/B
loop — one run is ~10 s once the binary exists — for comparing execution
profiles against a fixed prompt and checking the emitted token IDs.

Use `../hybrid-hot48-runner/run_50tps_gate.zsh` when you need a receipt.
Note that gate's `env -i` list is frozen, so it does **not** pass any of the
switches below; it must be edited before it can see them.

## Run one

```sh
CAM_SESSION_PID=$$ /Users/timtoole/bin/cam-lock.sh \
  qa/perf/gemma4-mtp-bandwidth-2026-08-21/demand-promotion-runner/run_cfg.zsh \
  my-label \
  qa/perf/gemma4-mtp-bandwidth-2026-08-21/demand-promotion-runner/env/H2-proportional

python3 qa/perf/gemma4-mtp-bandwidth-2026-08-21/demand-promotion-runner/summarize.py \
  qa/perf/gemma4-mtp-bandwidth-2026-08-21/demand-promotion-runner/runs/my-label
```

Always wrap model runs and cargo builds in `cam-lock.sh`: this is a 16 GiB box
and one 26B run plus one rustc will exhaust it.

The runner refuses on a missing binary, a busy port 8189, or free memory under
55%. It writes `env.txt`, `health.json`, `server.log`, `response.json`,
`http-wall-seconds.txt` and pre/ready/post memory samples per run.

Overrides: `CAMELID_BENCH_BINARY`, `CAMELID_BENCH_REQUEST`, `CAMELID_BENCH_OUT`,
`CAMELID_BENCH_PORT`, `CAMELID_BENCH_CACHE_MIB`.

## What summarize.py reports

Median round wall, assistant/verifier split, accepted drafts and alpha, decode
tok/s, the per-round ledger (`final_wait`, `encode`, `gpu_busy`, disk), the GPU
stage split, slot hit rate, and — the important one —
`exact_match_expected` / `exact_prefix_len` against
`../hybrid-hot48-runner/expected-48-token-ids.json`. **A throughput number
without `exact_match_expected: true` is not a result.**

For requests longer than 48 tokens, `exact_match_expected` is false by
construction; check `exact_prefix_len == 48` instead.

## Profiles in env/

| file | what it is |
|---|---|
| `common` | shared base; the others append to it |
| `B-mapped32` | the frozen 50 tok/s gate profile. Correct, 0.62 tok/s |
| `H2-proportional` | **the winner.** Union-proportional hot profile, 22.3 tok/s |
| `H1-uniform64` | uniform 64 slots/layer. 20.5 tok/s on MORE memory than H2 |
| `G5-lru` | H2 + `SLOT_POLICY=lru`. Slower than the default LFU |
| `C4-record64-physical-lfu8` | corrected anonymous lane, fixed 64 physical slots/layer. Exact, but slower than H2 |
| `C5-record-proportional-lfu8` | corrected anonymous lane with H2's proportional capacities. Exact faster-starting research lane |
| `A-explicit56` | historical anonymous 56-slot profile; old measurements predate the ordering fix and are invalid |
| `C-mono88` | historical anonymous monolithic profile; old measurements predate the ordering fix and are invalid |

`request-48-plain.json` is the frozen gate fixture with `camelid_receipt`
removed — leaving it in makes the server 500 outside the hybrid lane, because
the receipt path fails closed when hybrid telemetry is unavailable.

## Re-deriving a hot profile for a different model or K

`CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION_TRACE=1` emits one
`[hybrid fill trace] layer=N selected=... hits=... loads=... evicted=...` line
per layer per fill. Take the median `selected` per layer over the decode rounds
(drop the wide prefill passes), multiply by ~2, clamp to `[40, 96]`, and pass it
as `..._HYBRID_HOT_SLOTS_PER_LAYER` with `..._HYBRID_HOT_PROFILE_FREE=1`. Keep
the total under the 2,400-record cap; 1786 was measured to fit comfortably
beside dense weights, the assistant and KV on 16 GiB.

## 2026-08-24 corrected anonymous-lane A/B

The anonymous overlap defect is fixed in `src/metal.rs`: record-backed commands
cannot commit until the router has been observed, required experts loaded, and
the table published. The directory also pre-pins all resident experts in the
current union and refreshes LFU/recency on all-hit rounds.

Two interleaved steady pairs on the frozen 48-token request all reported
`exact_match_expected: true`:

| profile | decode tok/s | model load | HTTP wall | misses/round | read MiB/round |
|---|---:|---:|---:|---:|---:|
| H2 pair 1 | 18.72 | 10.43 s | 11.019 s | 78.9 | 251.7 |
| C5 pair 1 | 18.41 | 7.58 s | 10.345 s | 136.5 | 435.5 |
| H2 pair 2 | 18.14 | 11.19 s | 11.154 s | 78.9 | 251.7 |
| C5 pair 2 | 16.46 | 7.73 s | 11.587 s | 136.5 | 435.5 |

H2 averages 18.43 tok/s versus C5's 17.44 (+5.4%) and remains the steady
decode winner. C5 loads ~3.16 s faster; load + first request is ~15% faster.
The machine was in a slower GPU state than the settled H2 range documented in
`../HANDOFF-2026-08-24.md`, so use the interleaved ratio rather than comparing
these absolute values with older warm runs.

After the source was committed as `3726b3f1`, a fresh release build produced
`root-final-h2-oracle`: exact 48/48 at 18.64 tok/s, 78.9 misses/round, and
251.7 MiB/round. This is the receipt for the exact committed binary.

The correctly bounded C4 run was exact at 16.31 tok/s and 138.5 misses/round.
Sorting pooled reads by file offset (H4) was also a no-go: two paired runs kept
the same 2,294 prompt reads / 7.32 GiB and changed prefill filler by only +0.4%.
The H4 source switch was removed.

The curated receipts are under `runs/`: `root-final-h2-oracle`, the four
`root-refresh-{h2,c5}-pair*` directories, the C4 run, and the two interleaved
H2/H4 pairs.
