# Mini2 exact-throughput checkpoint — 2026-08-24

This is the authoritative continuation of `HANDOFF-2026-08-24.md` for the
16 GiB M4 Mini2. It was reconstructed from source, exact run receipts, and the
deployed binary after the original Codex task became unreadable.

## Result

The frozen 48-token Gemma 4 fixture now runs at a repeatable **35.02 tok/s
mean / 35.08 median**, with a **35.28 tok/s peak**. All three promoted H40
receipts report `exact_match_expected: true` and `exact_prefix_len: 48`:

| receipt | verifier rounds | decode tok/s | misses/round | expert MiB/round |
|---|---:|---:|---:|---:|
| `mini2-h40-schedule14-13-14-direct-stage-hot2100-candidate1` | 4 | 34.69 | 145.0 | 462.6 |
| `mini2-h40-schedule14-13-14-direct-stage-hot2100-candidate2` | 4 | **35.28** | 145.0 | 462.6 |
| `mini2-h40-schedule14-13-14-direct-stage-hot2100-candidate3` | 4 | 35.08 | 145.0 | 462.6 |

Mini2 used zero swap in every H40 run. Post-run free memory was 29–33%, so
H40 is inside the measured safe envelope. H41's 2,200-slot profile also used
zero swap but did not improve throughput; the historical 2,400-slot receipt
used 366.69 MiB of swap and must not be retried as an ordinary tuning step.

The winning replay profile is
`demand-promotion-runner/env/H40-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100`.
The deployed Mini2 executable and the local release artifact both have SHA-256:

`5376959c68cc2f49974510ce585a268b4c65454cb24d1759211aa6da2df0374c`

The prior Mini2 executable is preserved as
`/Users/timtoole/bin/camelid-bench-pre-h36-direct-c2caa383d5fd`.

## What moved the result

1. The target-authoritative boundary correction is generated once, then the
   assistant resumes its suffix once. It no longer generates and discards a
   duplicate suffix.
2. A default-off verifier schedule override was added:
   `CAMELID_GEMMA4_MTP_WIDTH_SCHEDULE=14,13,14`. The 48-token tail truncates
   naturally, producing the exact four-pass schedule `14,13,14,7`.
3. The exact hot/cold single-Down path admits widened verifier K through 16.
4. A default-off direct-stage reader was added:
   `CAMELID_GEMMA4_GHOST_METAL_DIRECT_STAGE_READ=1`. With a zero host-cache
   budget and no previous-cold staging, each validated expert record is read
   directly into its final Metal stage slot. This removes the temporary
   allocation and second copy; H38 measured `copy=0.0ms` while remaining exact.
5. The already-tested 2,100-slot per-layer profile was paired with the new
   four-pass/direct-read lane. It reduced H38's 205.2 misses and 654.8 MiB per
   round to H40's 145.0 misses and 462.6 MiB without swap.

Every new behavior is opt-in and fails closed. The direct reader requires the
literal value `1`, a zero host expert-cache budget, and no previous-cold stage.
It preserves physical I/O sorting, original compact-slot order, exact record
length and payload validation, joins every parallel read before reporting a
failure, and publishes no cold table after a partial failure.

## Bounded A/B sequence

Every number below is from a receipt with `exact_match_expected: true` and a
48/48 exact prefix:

| profile | rounds | decode tok/s | conclusion |
|---|---:|---:|---|
| H29 fixed K10 + boundary + single-Down | 6 | 26.39 | no-go |
| H30 adaptive K10→K12 | 5 | 28.44 | useful controller |
| H30b + duplicate-suffix removal | 5 | 28.18, 29.45; matched control 29.49 | prior best |
| H31 specialized K12 GateUp | 5 | 29.12, 29.07 | flat/no-go |
| H32 fixed K12 | 5 | 28.76 | no-go |
| H33 fixed K15 | 4 | 29.31 | proves four passes; excess work |
| H34 fixed K14 | 4 | 29.68, 29.52 | prior four-pass best |
| H35 fixed K13 | 5 | 27.63 | semantic rejection cliff |
| H36 schedule 14,13,14,7 | 4 | 30.82 | zero-waste schedule win |
| H37 schedule 15,12,15,6 | 4 | 30.54 | exact, slower than H36 |
| H38 H36 + direct-stage reads | 4 | 32.54, 32.98 | direct-I/O win |
| H39 H37 + direct-stage reads | 4 | 32.10 | slower schedule |
| H40 H38 + 2,100 hot slots | 4 | **34.69, 35.28, 35.08** | promoted Mini2 profile |
| H41 2,200 hot slots | 4 | 34.77, 34.93 | more memory, no gain |
| H42 rebalanced 2,100 slots | 4 | 34.82 | fewer reads, no gain |
| H43 retained cold bank | 4 | 33.53, 33.42 | fewer reads, serialized blits, no-go |
| H44 retained bank + chunked reads | 4 | 34.33 | recovered read parallelism, no-go |
| H45 direct-to-bank, zero blits | 4 | 33.01 | removed copies, exposed no critical-path gain |

## Post-H40 ceiling

H40 candidate 2 took 1,360.46 ms for 48 tokens. Its serialized assistant work
was 226.90 ms and the target verifier was 1,133.00 ms. Inside the target,
measured GPU stages totaled 701.8 ms and direct-stage I/O totaled 512.3 ms;
81.1 ms overlapped.

- 40 tok/s requires 1,200 ms, so it needs another 160.46 ms removed. H43–H45
  closed retained-I/O microtuning as that source: avoiding 229 reads and then
  removing every replacement blit still failed to beat H40.
- 50 tok/s requires 960 ms, so it needs 400.46 ms removed. With assistant and
  GPU work unchanged, the stage-free arithmetic roof is only about 51.65
  tok/s, leaving 30.74 ms of total margin. Roughly 93% of exposed stage time
  must disappear.
- Perfect previous-round reuse cannot cover round zero and therefore tops out
  around 44.6 tok/s. H21's K8 recurrence projects only about 244–259 reusable
  H40 records, enough for roughly 40.7–42.4 tok/s under ideal scaling, not 50.

A default-off retained cold bank was implemented after the promoted H40
checkpoint: six records per layer (576.56 MiB), exact hot + retained + fresh
indexed tables, and two-phase identity publication after a queue-order terminal
barrier. H43 stayed exact and at zero swap but regressed to 33.53 and 33.42
tok/s. It avoided 229 of H40's 580 reads in the later comparison, but the
remaining small read batches fell from about 3.5 to 2.1 GiB/s and 278 serialized
stage-to-bank blits copied another 886.9 MiB on the Metal queue.

H44 adaptively split sparse fresh records across the eight-reader pool, capped
at four positioned reads per record. It remained exact at 34.33 tok/s with zero
swap and recovered about 38 ms versus H43 candidate 2, but did not beat H40.

H45 completed the direct-to-bank experiment. It replaced the monolithic bank
with six distinct record resources per layer, gave the committed hot command a
ready-only table, invalidated replacement identities before positioned reads,
and committed the full cold table only after every exact read joined. It stayed
48/48 exact and at zero swap, converted all 278 replacement blits to direct
fills, and removed about 1.73 GiB of duplicate unified-memory read/write
traffic. Throughput nevertheless fell to 33.01 tok/s. Its three later rounds
were essentially flat against H44; a 54 ms first-round regression accounted for
nearly the entire loss. The retained copies were therefore not on the exposed
critical path, and further bank/chunk/read tuning is closed.

Practical 50 tok/s on this exact 16 GiB lane needs both near-complete cold
availability including first-use records and additional target-compute margin,
for example compressed/global retention plus assistant-informed future-route
prefetch and an exact kernel improvement. Static capacity alone is closed:
2,200 did not win and 2,400 swapped.

## Live-hidden sequential staging (H46–H49)

H46 added an observation-only next-layer route predictor at the exact
post-attention barrier. It never changed I/O, slots, tables, routing, or model
output. Across the frozen four verifier rounds, cap-eight predicted 318 of the
later exact cold records (54.91% read-wall-weighted recall) while preserving
48/48 exact output and zero swap. The original scalar predictor cost about
101.4 ms, so it was useful as a quality proof but too expensive to promote.

H47 turned that signal into a bounded, one-worker record stage. Every staged
record stays in private owned bytes until exact `StageCold` requests the same
layer/expert identity; sealing never waits, and queued, late, failed, or absent
records fall through to the established eight-reader direct-to-stage path.
Three exact zero-swap runs measured 36.51, 34.69, and 35.41 tok/s (35.54 mean,
35.41 median). The best receipt had 261 ready hits and reduced exact demand
reads from 580 to 319, but the scalar predictor still cost 102.5 ms. This was a
real but modest gain over H40, not a stable promotion.

H48 replaced the per-row predictor matvecs with one Apple Accelerate SGEMM per
layer. The production-shape release microbenchmark measured 8.455 ms versus
77.892 ms for the scalar reference (9.21x faster), and focused shape,
non-finite, tie-order, ranking-parity, and literal-opt-in tests passed. On the
real Mini2 request, the first-run Accelerate warmup brought total predictor
time to 26.36 ms. Candidate quality stayed exactly 318 cap-eight truth hits,
but one serial reader delivered only 214 in time; the run remained 48/48 exact
at 36.22 tok/s with zero swap.

H49 enabled a second private reader without widening the cap-eight candidate
set. It remained exact with zero swap and reached **36.84 tok/s**. The stage
issued 544 of 612 candidates and delivered 261 useful ready hits, leaving 319
authoritative fallback reads. Four-round wall time was 1,302.99 ms:
220.34 ms assistant and 1,082.10 ms verifier. Inside the verifier, measured GPU
work totaled 678.4 ms and remaining demand waves totaled 346.4 ms. Therefore
40 tok/s now needs 102.99 ms removed. Perfect readiness of the current
cap-eight set can recover at most 57 additional true hits, so adding more
workers alone cannot close the gap.

## Post-H49 bounded experiments (H50–H54)

A same-state H49 control after H50 measured 35.81 tok/s. H50 added a capped
96-record previous-round source, remained 48/48 exact with zero swap, and fell
to 34.74 tok/s. H51 changed the verifier to three K16 passes and remained exact
at 34.08 tok/s. H53 then traced the decisive K16 rejection: the required target
token was rank 20 at draft index 10, with a 21.57-logit deficit. That closes a
simple wider-verifier or short-rescore shortcut for this fixture.

H54 split each sparse layer into exact hot, already-ready, and authoritative
demand prefixes. Demand reads started after the hot commit while ready records
ran GateUp, and a final command ran the demand GateUp plus the single Down/tail.
The run remained 48/48 exact at 36.25 tok/s with zero swap and no fallback or
error. It hid about 34 ms of record loading, but 82 additional GPU command
submissions raised slot wait by 26.4 ms and GPU time by 3.9 ms; verifier wall
improved only 2.73 ms versus the same-state H49 control. H54 is therefore an
exact architectural proof, not a throughput promotion.

The next bounded revision will keep the immediate hot GateUp command, preserve
the concurrent demand-read/ready-copy host staging, then fuse ready and demand
GateUp into one terminal command with the sole Down and tail. At the observed
layer mix that removes all 82 extra submissions and returns the sparse verifier
to the H49 command count. This cleanup alone is not projected to reach 40 tok/s;
it creates the lower-overhead exact base needed for the next compute change.

H50 tested the earlier complementary pre-assistant reuse hypothesis described
at the H49 checkpoint. Its exact regression closes that path at a 96-record
cap; it is not the next promotion candidate.

Profiles and executable for this checkpoint:

- `H46-live-hidden-sequential-probe`
- `H47-live-hidden-sequential-stage-cap8`
- `H48-live-hidden-sequential-fast-predict`
- `H49-live-hidden-sequential-fast-predict-dual-reader`
- `H50-live-hidden-dual-previous-cold96`
- `H51-live-hidden-fast-dual-k16x3`
- `H52-k16-7-step3-rejection-rank-probe`
- `H53-k16-draft10-rank-probe`
- `H54-three-wave-live-ready-gateup`
- Mini2 executable: `/Users/timtoole/bin/camelid-h48-fast-b5b770ef`
- SHA-256: `b5b770ef64f5ecb19d42eef6a489b9fad6bcd4897fdd5091dece3453f52a5f4c`
- H54 Mini2 executable: `/Users/timtoole/bin/camelid-h54-f7f96177a700`
- H54 SHA-256: `f7f96177a700332aa1486ed0100cc475bffa72d3881ee1d06872972c3e28194f`

## Validation and provenance

Focused gates passed under `cam-lock.sh` with `CARGO_BUILD_JOBS=2`:

- verifier schedule parser/sequence/tail: 3/3;
- existing adaptive-width tests: 3/3;
- direct-stage parser/admission/I/O-order tests: 3/3;
- retained-bank environment/chunk policy and direct target mapping: 7/7;
- retained planning, two-phase publication, legacy blits, record-resource
  isolation, and separate hot/full binding gates: 7/7;
- existing previous-cold-stage tests: 2/2;
- live sequential parser, probe accounting, Accelerate shape/ranking/parity,
  and production-shape latency tests: 10 passing plus one ignored benchmark;
- predictive record staging lifecycle, concurrency, cancellation, and global
  permit tests: 20/20;
- K9–K12 specialized/general GateUp raw-bit parity (H31 research lane);
- `cargo fmt --check`, `git diff --check`, and guarded release build.

Branch: `codex/gemma4-50tps-v2`. The Mini2 source, profiles, checkpoint notes,
and three promoted H40 receipts are published on that branch. Exploratory run
receipts remain local-only. The H40 executable was built from local HEAD
`17be2d9e53d6d2294633f96150d558492ff195f4` plus the checkpointed working-tree
changes recorded by this branch.

## Crash boundary

Do not run llama.cpp in this effort. The stock-runtime attempt exhausted Metal
memory and crashed the machine before producing a valid throughput result.
There is no llama.cpp number to recover, and none is needed for the Mini2 H40
result.
