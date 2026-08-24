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
| `A-explicit56` | anonymous-slot lane, 56 physical. **Emits garbage** |
| `C-mono88` | anonymous-slot lane, 88 monolithic. **Emits garbage** |

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
