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
| M3 | Acceptance for a pretrained EAGLE-3 head, measured on this machine. |
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
