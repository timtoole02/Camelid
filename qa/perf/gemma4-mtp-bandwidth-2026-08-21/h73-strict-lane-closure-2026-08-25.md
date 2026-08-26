# H73 — strict-lane levers closure and stack qualification, 2026-08-25

Continuation of `h69-final-result-2026-08-25.md` on the same 16 GiB M4 Mini2,
frozen 48-token fixture, strict 1,408-slot / `KV_INIT=192` profile. Source
commit for every run below: `2c3d980b` (branch tip `4d82a5cb` plus the H73
refresh opt-in), binary SHA-256
`5ffdfed1b75428c631b12324a859685a08a02f6fded12b6f7117a417b4acf754`
(`camelid v0.6.1-365-g2c3d980b`). Mini2 was rebooted before the first
sequence; every counted run reports `exact_match_expected: true` with a 48/48
prefix and zero current swap.

## H73a — chained terminal residency refresh: NO-GO

`CAMELID_GEMMA4_GHOST_METAL_CHAINED_TERMINAL_REFRESH=1` (commit `2c3d980b`,
default off, literal opt-in) re-admits the established terminal-barrier
promotion of each decode round's exact routed union on the strict lane, where
`HOT_COLD_OVERLAP_PUBLISH=0` otherwise freezes every persistent hot-slot
update. Hypothesis: recurring cold experts (H21 measured 54.5%
previous-cold→current-cold recall) stop being re-read from disk each round.

The mechanism worked as designed — four refresh receipts per request at
9.7–36.9 ms each (page-cache-speed promotion), 48/48 exact, zero swap — but
the ledger shows per-round `wave_load` did not fall: the strict lane's
residual cold set is dominated by first-touch experts, exactly as the earlier
prefetch-from-previous-union closure predicted ("the residual reads are by
construction the experts routed for the first time"). The interleaved
no-watchdog ABBA measured:

| arm | tok/s | mean |
|---|---|---:|
| C0 control | 30.22, 30.27 | 30.245 |
| R1 = C0 + refresh | 30.18, 30.15 | 30.165 (−0.26%) |
| S1 stack (below) | 32.69, 31.88 | 32.285 |
| R2 = S1 + refresh | 29.83, 28.55 | 29.19 (−9.6%) |

Flat on the control, a clear regression on the stack (promotion loads and
LFU evictions disturb the prompt-ranked residency). H73a is a **NO-GO**; the
default-off source and unit test remain as closure evidence.

## H73b — calibration-derived residency identity: NO-GO (offline)

Question: can a production-legitimate identity profile — ranked from OTHER
prompts' routing, not this fixture's decode future — recover part of the
~460 ms residency-misallocation gap H70 exposed? Three ~100-token calibration
requests (Python function edit, prose rewrite, YAML→JSON) ran under the H70
exact-residency-trace env; their decode route masks were aggregated into a
per-layer popularity ranking filling the same 1,408 capacities, and evaluated
against the fixture's four traced rounds with H70's linear
route-coverage model:

| profile | fixture decode coverage | projected residual wave |
|---|---:|---:|
| current (prompt/LFU content) | 3,016/4,246 (71.0%) | 661.5 ms |
| fixture-oracle (non-promotable) | 3,799/4,246 (89.5%) | 207.2 ms |
| calibration (3 prompts) | 3,084/4,246 (72.6%) | 675.8 ms (worse) |
| blend (calibration + current) | 3,235/4,246 (76.2%) | 574.8 ms |

Cross-prompt expert popularity does not transfer: calibration-only is worse
than the current content, and the blend's optimistic 87 ms projection is far
below the ~160 ms needed for 35 tok/s — before applying the historical
projection-to-measured haircut (H71 measured 48 ms of a much larger
projection). H73b is a **NO-GO** with no engine change; the study script and
receipts document the closure.

## H73c — strict-lane stack qualification

The demonstrated sub-threshold positives were combined into one checked-in
profile, `demand-promotion-runner/env/H73-strict-stack-kv192` = the H71
prompt-ranked handoff profile plus:

```
CAMELID_GEMMA4_MOE_MMA_K16=1
CAMELID_GEMMA4_MTP_BF16_PRODUCER_FUSION=1
CAMELID_GEMMA4_MTP_BF16_LATTICE_LOADS=1
CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_STAGE_CAP16=1
```

Every run verified the H71 receipt (`admitted=1 effective=1
source=current-prompt-exact-route-union selected_records=1408`). The
watchdog-qualified A/B/B/A/B sequence measured:

| run | lane | tok/s | exact | watchdog | peak child bytes | peak wired bytes |
|---|---|---:|---|---|---:|---:|
| A1 | C0 control | 31.83 | 48/48 | 0 violations | 6,927,809,464 | 7,878,787,072 |
| B1 | H73 stack | 32.02 | 48/48 | 0 violations | 6,954,925,008 | 7,961,313,280 |
| B2 | H73 stack | 31.15 | 48/48 | 0 violations | 6,995,442,664 | 7,919,288,320 |
| A2 | C0 control | 32.15 | 48/48 | 0 violations | 6,968,179,640 | 8,039,186,432 |
| B3 | H73 stack | 31.74 | 48/48 | 0 violations | 6,954,449,872 | 7,960,412,160 |

Controls averaged **31.99 tok/s**; the stack averaged **31.64 tok/s** (−1.1%,
inside run-to-run variance). The stack is therefore **not promoted**: in a
single interleaved qualified sequence its retained levers do not produce a
repeatable end-to-end win, consistent with H58's own ABBA (−1.25% while
saving 38.6 ms of GPU) and with H71's single-observation status. The earlier
apparent stack advantage in this session (32.29 vs 30.25 means) was
cross-sequence machine state: controls alone rose from 30.2 to 32.0 over the
evening at identical configuration and binary. Numbers from different
sequences must not be compared, only interleaved arms.

Qualified strict-lane state after H73: **~31.6–32.0 tok/s**, config-insensitive
across the retained levers, watchdog-qualified, exact, zero swap.

## Where 35 tok/s stands after H73

- The strict treaty lane (child < 7.5 GiB, wired < 8 GiB, zero swap,
  1,408 slots) now has its levers exhausted: H50–H66 were closed by the
  prior campaign, and H73a/H73b close cross-round retention and
  calibration residency. The qualified lane runs at ~31.6–32.0 tok/s and is
  config-insensitive across the retained levers; nothing known reaches 35
  under the treaty.
- The fixture-oracle residency ceiling (43.6 tok/s linear projection)
  remains structurally non-promotable (`throughput_promotion_allowed=false`).
- 35+ tok/s exists on this branch only on the legacy H40/H49 2,100-slot
  lane (35.02 mean promoted H40 receipts; 36.84 H49 historical peak), which
  the hardcoded harness caps now exclude in both supervision modes: the
  prefill live-record spike crosses the 7.5 GiB child ceiling. Reopening it
  is a supervision-contract decision, not a tuning step: it requires
  raising `MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES` in
  `demand-promotion-runner/run_cfg.zsh`, and the historical zero-swap
  H40/H41 receipts (post-run free memory 29–33%) are the evidence either
  way. The retained compute levers (H58/H60/H62) stack on that lane and
  none of them is 1,408-pinned, so its ceiling today is likely above the
  historical 36.84 — unmeasured pending that decision.
