# Speculative campaign — Phase 0

Phase 0 exists to make the speculative lane *reachable* and its numbers
*trustworthy*, before any drafter work is funded. It ships three guarded fixes
and a measurement pass whose job is to price — or kill — the phases after it.

Nothing here moves a ledger row, changes a lane default, or makes a support
claim. Every emitted token is still the target model's own greedy argmax.

## The spine (landed)

| Fix | What it does | Rollback |
| --- | --- | --- |
| A | `CAMELID_SPEC_GPU` auto-arms on a Metal host | `CAMELID_SPEC_GPU=0` |
| B | Streamed requests keep their speculation | `CAMELID_SPEC_STREAM=0` |
| C | Sampled (temperature>0) requests decline speculation | none — see below |

**A.** Speculative decoding without GPU verify switches
`CAMELID_METAL_RESIDENT_DECODE`/`_PREFILL` off *server-wide*, for every request
on the process. Selecting that plan merely because a variable was unset meant
`--spec-decode ngram` benchmarked the CPU repack plan on the exact host whose
fast lane was being measured, and announced it only through a `tracing::info!`
that a stock install (no `RUST_LOG`) never prints. `should_auto_arm_spec_gpu()`
is a pure, unit-tested decision: it fires only when the flag is unset, the
resident lane is armed, the run is not deterministic, and a Metal device
exists. An explicit `CAMELID_SPEC_GPU` wins in both directions, and both the
auto-arm and the CPU-plan demotion now print to stderr.

**B.** Streaming dropped speculation wholesale in the same commit that
introduced the feature, so the lane never fired for streaming clients — the
only kind agent traffic uses. There are **two** live streaming jobs, selected at
runtime (`CooperativeStreamDecodeJob::step` for continuous batching,
`run_stream_decode_job` when exclusive), so the round is lifted into
`run_speculative_round()` and shared by all three decode loops. Both SSE paths
already diff the whole decoded text against what they have streamed, so an
accepted run emits as one delta with stop-sequence truncation and usage counts
unchanged. The exclusive loop's counter is sized one-token-per-iteration and
gained the explicit `max_tokens` break the blocking loop already had.

**C.** A sampled round has no GPU verify lane — the batched verifier returns
argmax rows, not distributions — so it took a k-row CPU forward of the whole
model plus a resident→CPU KV mirror-back, against the resident step it was
trying to beat. Chat defaults are temperature>0, so the lane was a likely
*regression* for ordinary traffic. `speculation_admissible()` now requires
greedy sampling. The exact rejection sampler stays in the tree for a future GPU
distribution verifier.

Acceptance telemetry (`log_speculative_summary`) now fires on all three loops.
Streamed traffic previously emitted none, which is what let this go unmeasured.

## Losslessness

Unchanged and structural: every accepted token is the target's own argmax via
`accepted_draft_prefix`, so a drafter can only change how fast tokens arrive,
never which ones. Gates that must stay green:

```
cargo test --lib -- spec latch                     # incl. ngram_speculation_matches_vanilla_greedy_decode
cargo test --lib -- metal_spec_verify_bit_identical
cargo test --bins -- spec_gpu_auto_arms
```

## The api/mod.rs renderer seal

Any commit touching `src/api/mod.rs` must refresh five constants or the
`validation-scripts` CI job goes red. Order matters — the SHA-256 is of the
fixture that *contains* the SHA-1.

```bash
git hash-object -- src/api/mod.rs      # -> paste into all 3 blob-sha1 sites
#   scripts/hf-qualification-smollm3-chat-parity.mjs
#   scripts/test-hf-qualification-smollm3-chat-parity.mjs
#   qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json
node -e "const{createHash}=require('node:crypto'),fs=require('node:fs');const b=Buffer.from(fs.readFileSync('qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json').toString('utf8').replace(/\r\n/g,'\n'));console.log(b.length,createHash('sha256').update(b).digest('hex'))"
#   byte count must still be 5578; paste the hash into the 2 sha256 sites
```

Verify in under a second, no cargo required:

```bash
node scripts/test-hf-qualification-smollm3-chat-parity.mjs && \
node scripts/test-check-model-qualification-roster.mjs
```

## Measurements

Every heavy run is serialized by `~/bin/cam-lock.sh`. One model at a time — this
is a 16 GB machine.

| | What it settles |
| --- | --- |
| M1 | Plain 8B decode tok/s at 512 / 2k / 4k depth, plus a k=8 verify round cost. **Binds every absolute number the campaign quotes.** |
| M2 | Suffix/n-gram tree drafting on real agent transcripts at ≥4k depth, with prose as the control. |
| M3 | Acceptance for a pretrained EAGLE-3 head, measured on this machine. **Measured on CPU and resident.** |
| M4 | Synthetic-acceptance round-cost grid: fixed-α replay drafter driving the real verify. |

Two rules learned the hard way, both from the campaign's own red team:

- **Measure at depth.** Any headline must name the depth it was measured at.
  Acceptance in particular varies enormously with context length — 31% at 556
  tokens, 91% at 4137 — because a training-free drafter can only propose what
  the context already contains. (The round *cost* turned out to be flat in
  depth on this engine; see the measured section. The per-row prefix-KV re-read
  is real but is not the dominant term.)
- **Never import an acceptance rate.** Published figures conflate
  accepted/drafted per round with conditional per-position probability — nearly
  a 2× spread in projected speedup — and cross CUDA→Metal and temp-0→real-traffic
  boundaries on the way. Log both quantities and calibrate from those.

### Kill rules

- Tree/n-gram drafting: **< 1 accepted token per round** on genuine agent
  replay retires the training-free angle.
- Trained head: measured α **< 0.6**, or grid-interpolated end-to-end **< 1.8×**,
  descopes the training track to a converted pretrained head only.
- A weak third-party result is **no data**, not a kill. Kill authority belongs to
  the measurement taken through Camelid's own verify.

### Watch-outs

- A silent CPU-verify fallback invalidates a run: require `gpu_verify_rounds > 0`.
- Acceptance regressions are invisible to token-identity gates — both stay green
  while the lane quietly stops paying. Gate on an α floor as well.
- `bench-speculative` does not call `apply_serve_nocopy_default()`; pass
  `CAMELID_METAL_NOCOPY=1` explicitly to match `bench-generate`.
- No llama-arch 8B **K-quant** row is certified. Q4_K_M numbers for an 8B are
  measurements of an uncertified configuration and must say so.

## Measured, 2026-08-27 (Llama-3-8B, Apple M4 16 GB, release, greedy)

Raw records in `qa/speculative/receipts/`; the agentic prompts that produced them
are alongside in `receipts/prompts/`.

### M1 — plain decode baseline

| config | decode | GB/token | effective | % of ~120 GB/s wall |
| --- | --- | --- | --- | --- |
| Q8_0 | 11.9–12.0 tok/s | 8.54 | 102 GB/s | 85% |
| Q4_K_M | 11.2–11.5 tok/s | 4.92 | 56 GB/s | 47% |

Q4_K_M reads 43% fewer bytes and decodes **slower**. Prefill is worse still:
110 ms/prompt-token against Q8_0's 8.7, growing superlinearly (a 512-token
prefill did not finish in 5.5 minutes — profiled as genuinely computing, GPU at
100%, no restart events). The GGUF is clean (Q4_K/Q6_K/F32, all admitted), and
Q8_0 reproduced the published 12.1 tok/s, so this is the K-quant lane itself.
**Any Q4_K_M-target plan needs the K-quant kernels fixed first.**

Q8_0 decode is flat to 465 tokens (11.87) and still 8.38 tok/s at 4137.

### M2 — acceptance is not the bottleneck

On a 4137-token agentic transcript the stock n-gram drafter reached **90.8%
acceptance, 7.35 committed tokens per round** — the k=8 window is essentially
saturated, drafting costs ~10 ms per generation, and output is lossless. End to
end that bought only **1.11×**, because the round costs 694 ms.

### The round

A k-sweep at fixed depth fits `round = 13 ms fixed + 67.5 ms per verify row`,
against an 85 ms plain decode step. Round cost is flat in depth (553 ms at 304
tokens, 694 ms at 4137) and linear in k — so the per-row prefix-KV re-read is
**not** the first thing to fix, and a shared-prefix verify attention is not the
first move.

`CAMELID_SPEC_VERIFY_TRACE` now reports encode vs gpu_busy: **encode 2–5 ms,
gpu_busy 508–535 ms**. The round is not CPU-feed or dispatch-launch bound, which
retires buffer pooling, dispatch batching and concurrent encoders as first moves.

### Open

The verify GEMV's activation panel (~112 GB/round at k=8 vs 7.97 GB of weights)
predicted −50% from NR0 2→4; measured **−10%**. A single NR0=8 probe was no
better than NR0=2 once normalized to its host's decode step. Working hypothesis:
the kernel is **register-limited** (`yl[8][8]` is already 64 floats/thread), so
raising NR0 trades panel traffic for occupancy. Next step is to confirm that
directly — occupancy/spill for this pipeline — rather than sweep NR0 further.

## v2 + MMA lane on the 8B — measured 2026-08-27

Merged from `perf/metal-mc-gemv-spec-verify-session`. Arms selected only by env
off one binary; raw records in `receipts/v2.jsonl` and `receipts/v2depth.jsonl`.

Prefill and decode, Llama-3-8B, 556-token prompt, arms run back-to-back:

| arm | prefill ms/tok | decode tok/s |
| --- | --- | --- |
| Q4_K_M v1 (default) | 121.23 | 10.02 |
| Q4_K_M v2 | 49.35 | 12.68 |
| Q4_K_M v2+MMA | **14.98** | 12.55 |
| Q8_0 (control) | 4.82 | 10.77 |

Verify round, 304-token prompt: v1 1099 ms -> v2 380 ms -> **v2+MMA 178 ms**.

Three things this settles:

- **Both reasons Phase 0 disqualified Q4_K_M are addressed.** Decode is +26%
  over v1 and now beats Q8_0 by 17%; prefill is 8.1x faster than v1. The
  prefill fix comes specifically from the MMA kernel (14.98 vs 49.35 without
  it), because the multi-column lane serves batched prefill as well as verify.
- **Q8_0 still prefills 3.1x faster** (4.82 vs 14.98 ms/tok), so the quant
  choice is now a real trade, not a rout.
- **The end-to-end speculative multiplier is NOT demonstrated.** The decode A/B
  ran at 304 tokens where acceptance is only 28-36%, so speculation measured
  1.00x. The round is 6.2x cheaper; converting that into tokens needs the 4k
  prompt where acceptance is 91%. That sweep is written and unrun.

### The v1 K-quant speculative lane is not lossless

The v1 arm reported `lossless=false`, first divergence at generated token 58,
while both v2 arms were clean. Investigation (three read-only agents):

- **Routing confirmed.** With v2 off, multi-token dispatches go to
  `q4k_linear_tiled` while single-token decode uses `q4k_linear_simd`
  (src/metal.rs:14663-14671, 14829-14844), so a speculative verify and a plain
  decode run different kernels.
- **The kernels are algebraically identical** term for term — same buckets,
  same accumulation order, same f32 tail. The cause is not source-level.
- **It is compilation.** `LINEAR_ROW_SHADER`, holding both v1 kernels, is built
  with default `CompileOptions::new()` — fast math ON (src/metal.rs:8865,8869) —
  while `STRICT_Q8K_SHADER` (8873) and the whole v2 lane (14905) explicitly
  disable it. Under fast math the optimizer may contract and re-associate
  independently in two kernels with very different surrounding shape. The tree
  already names this hazard in the v2 library's own comment (14901-14904).
- **Nothing would have caught it.** No test compares tiled against simd; the
  only test that dispatches tiled compares to a CPU oracle under a
  length-scaled tolerance sized to absorb reordering, so it cannot detect a
  GPU-vs-GPU bit split. `metal_spec_verify_bit_identical` uses Q8_0 wire blocks
  only — the byte-exactness proof rests entirely on the Q8_0 lane.

Unproven: that the two COMPILED kernels actually differ on this M4. The
mechanism is compiler freedom, so it needs a GPU A/B. What is measured is that
flipping `CAMELID_KQUANT_V2` — which touches only these pipelines — is the
difference between `lossless=false` and `lossless=true`.

### Measurement hygiene

~4.5 h of continuous benchmarking downclocks this M4 (Q8_0 decode read 11.9
in the morning and 10.8 in the afternoon). Compare arms only within one
back-to-back set; treat absolutes across sessions as drifted.

## Result — 4.30x, measured 2026-08-27

Llama-3-8B Q4_K_M, 4137-token agentic prompt, one machine, one prompt; the last
row differs from the one above it by a single env flag on the same binary.
Records in `receipts/m2_*.jsonl`.

| | plain | speculative | verify round |
| --- | --- | --- | --- |
| default lane (v1 kernels) | 7.13 | 4.72 | 1528 ms |
| + K-quant v2 / MMA verify | 8.22 | 9.55 | 389 ms |
| + suffix drafter on the chain | 8.22 | 11.41 | 621 ms |
| **+ kv16 split-K attention** | **13.51** | **30.68** | **215 ms** |

**7.13 -> 30.68 tok/s = 4.30x.** Speculation went from a 0.66x *regression* to
2.27x. Plain decode alone went 1.89x, which lands for any K-quant workload at
depth whether it speculates or not.

### How it was found, in order

1. **The verify round, not the drafter.** The default lane had A=7.15 at 87.9%
   acceptance -- excellent drafting -- and still lost, because the round cost
   10.9 plain steps to commit 7.15 tokens. A perfect draft head would also have
   lost on that lane. Fixing the round came first for that reason.
2. **A width sweep that actually varied width.** An earlier sweep looked "flat
   in k" (+8% for 3x the width). It was an artifact: the n-gram drafter never
   filled the window, so every arm verified ~4 columns regardless of k. With a
   drafter that fills it, the round is ~70 ms per *actual* column, linear.
3. **Ablation over arithmetic.** A byte model said KV was 47% of round bytes, so
   shared-prefix attention looked like the lever. Ablating stages showed
   attention was 67% of the per-column cost while its KV bytes were only 8% of
   it -- the kernel was ~10x off its own bandwidth bound. The bytes were right
   and the conclusion was wrong.
4. **A routing gate, not a kernel.** Tracing the dispatch found split-K excluded
   kv16 primaries. The fix was deleting a condition.

### What this cost in wrong turns

Three confident models were falsified by measurement: that Q4_K_M would be the
fast target (it was slower until the K-quant lane was fixed), that the verify
GEMV's activation panel was the bottleneck (widening rows returned a fifth of
its prediction), and that shared-prefix attention was the remaining lever (worth
~6%). Each was caught by an A/B rather than by more reading. The habit worth
keeping: measure the thing, and check that the measurement varied what you think
it varied.

### Not done

- The v2 K-quant lane is ~1 ulp off v1, so it stays env-gated with no support
  claim. Promotion needs model-level parity receipts.
- The matching Llama-3.2-3B EAGLE3 benchmark is implemented. The committed raw
  gamma-3 row is **30.681 -> 41.536 tok/s (1.354x)** and lossless, but served
  chat is a regression (**0.851x** best). It is benchmark-only, not
  serving/native MTP. On the same provenance build at 4,137 tokens, EAGLE is
  `0.956x` while suffix is `1.995x`; both are lossless and fully resident. Broad
  learned-head acceleration needs bounded top-k/tree EAGLE.

## The v1 K-quant losslessness bug: found, root-caused, fixed

The default K-quant speculative lane emitted a **different token stream than its
own plain greedy decode**. Reproduced deterministically, same binary, one env
flag apart (`receipts/m2_repro_lossless.jsonl`):

| arm | lossless | divergence@ | runs |
| --- | --- | --- | --- |
| pre-change attention path (`CAMELID_METAL_ATTN_SPLITK=0`) | **false** | 58 | **5/5** |
| split-K attention (this branch) | **true** | — | **5/5** |

**The kv16 split-K admission fixed it.** Same one-condition edit as the 4.30x
speedup.

Two corrections worth keeping, because both cost real time:

- **It was never intermittent.** An early reading called it that, on the
  strength of one `false` at 304 tokens and one `true` at 4137. It reproduces
  5/5 at an identical divergence index; it only stopped reproducing because the
  binary being retested already contained the fix. One observation per arm is
  not evidence of intermittency.
- **The mechanism was NOT the GEMV.** The standing theory was that
  `q4k_linear_tiled` and `q4k_linear_simd` diverge under fast math, since their
  shared library compiles with it on while the v2 lane disables it. That
  analysis was detailed, well-evidenced, and about the wrong kernel: routing
  multi-token dispatches through the mc GEMV instead of tiled changed
  losslessness **not at all** (both true 4/4), while the attention path flips it
  every time (`receipts/m2_lossless_ab.jsonl`).

The mc-lane investigation still paid for itself: it is 1.4-2.0x cheaper per
verify round on the v1 lane (743 -> 539 ms at 256 tokens, 1063 -> 541 at 512),
using a kernel that already carries a passing bit-identity test. Defaulting
`CAMELID_KQUANT_MC_GEMV` on is justified on that alone.

## Session 2 findings — MTP justified, tree lane dead, 3.1 not promotable

### MTP is justified, on prose specifically
The suffix drafter reaches A=7.33 (91.6%) on agentic text and A=7.42 (91.7%) on
repo-documentation prose — so an early read said "prose does not collapse" and
the trained-head case was weak. That read was wrong: doc prose is repetitive in
exactly the way that flatters context-matching. Against the prompts this repo
ships flagged `spec_friendly: false` (`receipts/m2_hard.jsonl`):

| prompt | A | acceptance | multiplier |
| --- | --- | --- | --- |
| adversarial (60 unrelated words) | 1.00 | **0.0%** | **0.91x** |
| creative writing | 1.00 | **0.0%** | **0.87x** |
| normal chat | 1.36 | 6.0% | **0.80x** |

Zero acceptance, and speculation is a NET LOSS. A training-free drafter cannot
propose a token the context does not contain. That gap is what a trained head
fills, and it is the justification for the EAGLE-3 work — not raising an
acceptance number that is already 91% on agentic text.

Note the lane is a regression on this traffic even with the SpecLatch active.
Admission should probably learn from acceptance history, not just latch off
after a run of misses.

### The attention fix, re-attributed
After the kv16 split-K admission, ablation puts attention at **1.9 ms per verify
column, down from 52.8** — 67% of the round to 7%. Nothing else measurable:
rope, K/V scatter and argmax all show no share. The remaining ~25 ms/col is
inside the K-quant MMA GEMV itself, which is the next lever.
Width is now roughly neutral (k=7 30.61 tok/s, k=15 30.29) where wider used to
lose badly. Q8_0 on the fixed lane is far behind Q4_K_M (11.88 vs 30.61).

### The tree lane is a dead end here
Tree verify costs ~350 ms per verify column against the chain's 27 — 13x — and
it carries the highest acceptance measured anywhere (A=12.57 at k=15), so it
looked like the biggest remaining win. It is not, and the reason is not
attention: admitting kv16 primaries to the tree split-K path (the same edit that
gave the linear path its 2.9x) changed the cost **not at all** (352 vs 349
ms/col). Reverted — zero benefit, and a k=15 prose arm reported `lossless=false`
that could not be cleanly attributed. Whatever makes the tree path expensive is
in its own dispatch structure, unmeasured.

### Llama-3.1-8B is not promotable yet
The pinned parity probe still fails (`receipts/m2_probe.jsonl`), on the exact
artifact the fixture names (sha256 verified):

| prompt | oracle | camelid | match |
| --- | --- | --- | --- |
| Hello | `, I am a ` | `, I am interested in` | **no** |
| The capital of France is | ` a city of romance,` | same | yes |
| Once upon a time | `, in a small village` | same | yes |

Divergence at generated index 3, exactly where the 2026-08-11 fixture put it.
Better than recorded (that run tested one prompt, 0/1; this is 2/3) but still a
failure, so promotion needs a numerics root-cause, not receipts. Two obstacles
worth recording for whoever picks it up: the CPU reference lane needs
`CAMELID_LAZY_Q8_0_LINEAR=1` to match the fixture's own `no_repack` backend
(otherwise it refuses at the 6.44 GB materialization cap), and the health
endpoint is `/v1/health` with `generation_ready`, not `/api/health`.

## Session 3 findings — Llama-3.2 3B EAGLE3 measured

The matching benchmark is documented in
[`EAGLE3_LLAMA32_3B.md`](EAGLE3_LLAMA32_3B.md). Its winning raw prompt is 46
tokens, generation is 96 tokens, and gamma 3 ran 26 resident rounds and 0 CPU
rounds. The new served-chat sweep is also fully resident and lossless, but its
best arm is `30.649 -> 26.085 tok/s` (`0.851x`) at gamma 1. That distinction is
the current boundary: the recurrent learned head works; linear top-1 EAGLE is
not yet a general chat-speed path.
