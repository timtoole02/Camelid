# Gemma4 35 tok/s checkpoint — 2026-08-24

Machine: 16 GiB Apple M4. Branch `codex/gemma4-50tps-v2`. All measurements
below use the frozen 48-token request and reproduce all 48 expected token IDs.

## Draft-width sweep

The current full-Q4 MTP lane is already at its useful width frontier. The
candidate profiles differ only in speculative chunk/draft width; H2 is the
committed proportional-hot profile at K=8.

| profile | accepted alpha | rounds | mean round | GPU/round | fill/round | misses/round | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| K6 | 5.33 | 9 | 286.0 ms | 118.7 ms | 95.3 ms | 68.8 | 18.64 |
| K7 | 6.00 | 8 | 318.0 ms | 138.4 ms | 103.9 ms | 74.6 | **18.87** |
| H2 / K8 control | 6.00 | 8 | 333.9 ms | 143.6 ms | 111.9 ms | 78.9 | 17.97 |
| K9 | 6.86 | 7 | 382.4 ms | 159.5 ms | 114.3 ms | 83.3 | 17.93 |
| K10 | 6.86 | 7 | 403.4 ms | 172.2 ms | 125.8 ms | 89.4 | 17.00 |
| K16 | 8.00 | 6 | 560.2 ms | 243.5 ms | 175.9 ms | 125.7 | 14.28 |

K9 and K10 accept the same 41 draft tokens. K10 merely offers five more. K16
raises alpha only 33% over K8 while roughly doubling verifier cost. K7's small
lead over the same-session K8 control is below the known machine-state spread,
so it is not a promoted default without an interleaved post-reboot pair.

Receipts are under `demand-promotion-runner/runs/root-35-*-probe1/` and
`root-35-h2-control1/`; candidate env files are under
`demand-promotion-runner/env/K*-proportional`.

## What 35 requires

At alpha 6, 35 tok/s permits only 171 ms per round. Even deleting the full
roughly 100 ms exposed fill from a settled roughly 300 ms H2 round leaves a
roughly 200 ms / 30 tok/s ceiling. A fill-only change therefore cannot reach
35. The credible path needs both:

1. raise useful alpha to at least about 7 without K9/K10's proportional GPU and
   expert-union cost; and
2. hide or remove roughly 40-70 ms of verifier fill/compute.

The best next fill experiment is observation first: enable
`CAMELID_GEMMA4_GHOST_METAL_SPARSE_PREDICT_PROBE=1` and
`CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION_TRACE=1` on H2. If the predictor
has strong incremental precision, use its proposals only as advisory cache
fills across all 30 layers, then retain the exact router -> residual fill ->
table publication correctness barrier. Do not remove the existing
`predicted_ready = !record_demand` boundary.

## Machine-state warning and restart procedure

The current session remains in the known slow GPU state: K8 GPU time is about
143 ms versus the previously settled 107-116 ms, while read volume is unchanged
at 251.7 MiB/round. Four model runs did not clear it. Reboot before interpreting
small deltas, then:

1. run two disposable H2 warmups;
2. take an interleaved exact H2/K7/H2 pair;
3. run an exact H2 sparse-prediction probe plus demand-promotion trace;
4. implement predicted hybrid fill only if the observation receipt shows enough
   precision to beat its overfetch and eviction cost.

Every build and model run must remain wrapped in
`/Users/timtoole/bin/cam-lock.sh`.

## FreeToken-derived implementation boundary

FreeToken's safe prefill overlap uses two full, identity-indexed expert buffers
with ready/release ownership events. Its decode path still obtains the real
route before copying misses and executing GEMM. Copying its full-next-layer
policy is not viable here: one Gemma4 layer is roughly 429 MiB, or about
12.9 GiB of traffic across 30 layers, versus H2's current 252 MiB/round.

The smallest compatible experiment is instead a hard-capped, round-owned host
staging map driven by the existing observation-only sparse predictor. Before
target verification, issue only predicted, currently-cold `(layer, expert)`
records across all layers through the existing read pool. The exact router and
`fill_hybrid_hot_slots` remain authoritative: an exact hit copies a ready staged
record into its leased Metal hot slot; a pending, failed, or absent prediction
falls back immediately to today's positioned read. Never mutate a directory or
publish a slot table from prediction alone, and never wait for an unselected
prediction. A 64-96 record cap costs roughly 215-322 MiB.

Only implement this if the post-reboot probe shows at least roughly 30% residual
miss recall. At 40-60% recall the plausible saving is 25-45 ms, roughly a
24-26 tok/s lane by itself—not 35.
