# Demand-promotion A/B runner

The harness behind `../mapped-cold-round-cost-2026-08-24.md`. It began as a
fast exploratory loop, but the Mini2 50 tok/s campaign now runs it through the
same fail-closed watchdog boundary: strict child/wired ceilings, zero current
swap, no swap-in/out growth, isolated process-group accounting, frozen binary
and environment provenance, and mandatory 48/48 token parity. One run remains
roughly 10 seconds once the binary exists.

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
55%. The label must be one filename component, begin with an alphanumeric, and
contain only alphanumerics, `.`, `_`, or `-`; this is validated before the old
run directory is removed. It writes `env.txt`, `health.json`, `server.log`,
`response.json`, `http-wall-seconds.txt` and pre/ready/post memory samples per
run.

New `env.txt` receipts use `manifest_format=base64-v1` followed by one
`KEY@BASE64=...` line per environment or runner-metadata value. Decoding each
payload reconstructs its exact UTF-8 value, including embedded and trailing
newlines, without letting multiline content masquerade as another key. The
summary and predictor analyzers accept both this strict format and historical
plain `KEY=VALUE` receipts.

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
| `O1-oracle48-ceiling-k8` | **ceiling only:** assistant-free K8 verifier fed by the frozen 48-token answer |
| `O2-oracle48-ceiling-k16` | **ceiling only:** same oracle-seeded experiment at K16 |
| `O3-oracle48-overlap-no-publish-k8` | **ceiling only:** K8 oracle with H11 hot/cold overlap and no publication; reached 34.31/35.26 tok/s on clean Mini2 |
| `H11-hot-cold-overlap-h2-no-publish` | Historical corrected production baseline. Clean Mini2 measured ~25.08 tok/s before the MTP RoPE fix and 26.85/27.66 after it |
| `H16-hot-cold-overlap-h2-no-publish-prefill` | Prefill-overlap no-go: exact, but 19.55 tok/s and 2.27x decode misses/bytes |
| `H17-hot-cold-overlap-h2-no-publish-k1-bootstrap` | K1-bootstrap diagnostic; misses the same first answer-content token and does not improve throughput |
| `H18-hot-cold-overlap-h2-no-publish-bf16-mtp` | BF16-assistant no-go: same acceptance as full-Q4, slower at 24.77 tok/s |
| `H19-hot-cold-overlap-h2-no-publish-k10` | K10 no-go: six exact rounds, but widened verification falls to 23.07 tok/s |
| `H20-hot-cold-overlap-h2-no-publish-host-chain` | Host-chain parity check: same proposals as device chain, exact at 26.82 tok/s |
| `H21-hot-cold-overlap-h2-no-publish-recurrence-probe` | Observation only: previous cold union predicts current cold at 54.5% recall / 57.9% precision |
| `H22-hot-cold-overlap-h2-no-publish-step3-trace` | Diagnostic only: captures assistant draft-3 logits; target code-fence token was rank 2 with a 6.405-logit deficit |
| `H29-hot-cold-overlap-h2-no-publish-k10-boundary-single-down` | Exact K10 boundary/single-Down profile; 26.39 tok/s, no-go |
| `H30-hot-cold-overlap-h2-adaptive12-boundary-single-down` | Adaptive K10→K12 controller; five exact verifier passes |
| `H31-hot-cold-overlap-h2-adaptive12-k12-gateup` | K9–K12 specialized GateUp research lane; raw-bit exact, throughput flat/no-go |
| `H34-hot-cold-overlap-h2-fixed14-boundary-single-down` | Fixed K14; first exact four-pass improvement, 29.52–29.68 tok/s |
| `H36-hot-cold-overlap-h2-schedule14-13-14-single-down` | Exact zero-waste `14,13,14,7` schedule; 30.82 tok/s |
| `H38-hot-cold-overlap-h2-schedule14-13-14-direct-stage` | H36 plus direct-to-stage expert reads; 32.54–32.98 tok/s |
| `H40-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100` | **Promoted Mini2 profile:** 34.69, 35.28, 35.08 tok/s exact; zero swap |
| `H41-hot-cold-overlap-schedule14-13-14-direct-stage-hot2200` | 2,200-slot capacity check; exact but no faster, 34.77–34.93 tok/s |
| `H42-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100-rebalanced` | Same-memory rebalancing closure; exact at 34.82 tok/s, no-go |
| `H43-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100-retained-cold6` | Six retained cold records/layer; exact at 33.53 and 33.42 tok/s, zero swap, no-go |
| `H44-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100-retained-cold6-chunked-read` | H43 plus adaptive capped-four positioned-read splitting; exact at 34.33 tok/s, zero swap, no-go |
| `H45-hot-cold-overlap-schedule14-13-14-direct-stage-hot2100-retained-cold6-direct-bank` | H44 plus record-granular direct-to-bank fills and zero blits; exact at 33.01 tok/s, zero swap, no-go |
| `H46-live-hidden-sequential-probe` | Observation-only live next-layer predictor; exact, cap-eight truth hits 318, weighted recall 54.91% |
| `H47-live-hidden-sequential-stage-cap8` | Exact one-reader private cap-eight stage; 34.69–36.51 tok/s, modest gain, scalar predictor too costly |
| `H48-live-hidden-sequential-fast-predict` | H47 plus batched Accelerate SGEMM predictor; exact at 36.22 tok/s, one-reader readiness limited |
| `H49-live-hidden-sequential-fast-predict-dual-reader` | H48 plus a second private reader under the same cap; exact at 36.84 tok/s, zero swap; current research best |
| `H50-live-hidden-dual-previous-cold96` | H49 plus capped previous-round staging; exact at 34.74 tok/s, zero swap, no-go |
| `H51-live-hidden-fast-dual-k16x3` | Three K16 verifier passes; exact at 34.08 tok/s, zero swap, no-go |
| `H52-k16-7-step3-rejection-rank-probe` | Diagnostic width/rank profile retained for reproducibility; superseded by the narrower H53 rejection probe |
| `H53-k16-draft10-rank-probe` | Diagnostic K16 run; exact at 34.06 tok/s, but the decisive target token ranked 20th at draft index 10, closing the width-only shortcut |
| `H54-three-wave-live-ready-gateup` | Exact three-wave hot/ready/demand GateUp overlap; 36.25 tok/s, zero swap, no-go because 82 extra GPU submissions erased the hidden I/O gain |
| `H49-live-hidden-sequential-fast-predict-dual-reader-kv192-control` | Strict-memory H49 control: exact frozen 1,408-slot profile and KV192; ABBA mean 31.74 tok/s, zero swap |
| `H55-async-two-wave-collapse` | H54 submission recovery: exact two-command hot/terminal path; strict-memory ABBA mean 31.45 tok/s versus H49 31.74, no-go |
| `H56-mtp-assistant-router-probe` | Read-only assistant-hidden router projection; global-96 residual-cold recall 4.63% and 31.61 ms projected savings, so predictive host staging is closed |
| `H57-mtp-private-queue-warmup` | Target-free assistant warmup on the measured chain's private queue; exact strict-memory ABBA mean 31.34 tok/s versus H49 31.38, and the ~23 ms first-chain setup remained, no-go |
| `A-explicit56` | historical anonymous 56-slot profile; old measurements predate the ordering fix and are invalid |
| `C-mono88` | historical anonymous monolithic profile; old measurements predate the ordering fix and are invalid |

`request-48-plain.json` is the frozen gate fixture with `camelid_receipt`
removed — leaving it in makes the server 500 outside the hybrid lane, because
the receipt path fails closed when hybrid telemetry is unavailable.

## Oracle-seeded verifier ceiling

`O1` and `O2` are deliberately non-deployable upper-bound experiments. They
remove the learned MTP assistant and seed the zero-weight n-gram drafter with
the frozen rendered prompt plus its already-known 48-token answer. The target
still verifies every proposed token, so correctness remains target-authoritative,
but proposal accuracy is fixture oracle knowledge. Never report these numbers
as ordinary model throughput.

The raw seed contains literal newlines and hidden channel tokens. Profiles use
`CAMELID_GEMMA4_SPEC_SEED_TEXT@FILE=../oracle-48-seed.txt`; `run_cfg.zsh`
resolves `KEY@FILE=path` relative to the env file, requires `KEY` to be a valid
environment identifier, and passes the text as one environment value. The
loader uses an explicit sentinel so command substitution cannot discard any
trailing newline. Unix environment values cannot contain NUL.

Important provenance boundary: the existing `root-ceiling-only-o1-k8` and
`root-ceiling-only-o2-k16` receipts predate that preservation fix. Their loader
stripped the seed file's final newline, and the earlier tokenizer validation
described that effective value. They remain exact, clearly labeled evidence for
the historical dirty binary recorded in their `health.json`, but they are not a
byte-identical receipt for the hardened loader. The preserved trailing newline
may change tokenization; take a fresh tokenizer check and new model receipts
before quoting a result from the new loader.

Name every receipt with `ceiling-only`, for example
`root-ceiling-only-o1-k8`. Both profiles also set
`CAMELID_BENCH_CEILING_ONLY=oracle-seeded-ngram`, which makes `summarize.py`
emit `ceiling_only: true`, `draft_source: oracle-seeded-ngram`, and
`throughput_promotion_allowed: false`. A valid ceiling receipt must additionally
show `exact_match_expected: true`; read `decode_tok_s` and
`alpha_tokens_per_round` from the emitted `[spec round]` timings. `O1` measures
the current K8 target ceiling; `O2` asks whether fewer, wider K16 verifier passes
raise that ceiling despite their greater per-round cost.

## Re-deriving a hot profile for a different model or K

`CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION_TRACE=1` emits one
`[hybrid fill trace] layer=N selected=... hits=... loads=... evicted=...` line
per layer per fill. Take the median `selected` per layer over the decode rounds
(drop the wide prefill passes), multiply by ~2, clamp to `[40, 96]`, and pass it
as `..._HYBRID_HOT_SLOTS_PER_LAYER` with `..._HYBRID_HOT_PROFILE_FREE=1`. Keep
the total at or below the measured 2,100-record Mini2 envelope unless a new
memory gate proves otherwise. H40's 2,100 slots stayed at zero swap, 2,200 did
not improve throughput, and the historical 2,400-slot receipt used 366.69 MiB
of swap. The older 1,786-slot profile remains the comfortable baseline.

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
