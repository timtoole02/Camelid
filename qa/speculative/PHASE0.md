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

- **Measure at depth.** A speculative round re-reads the prefix KV once per
  verify row, so short-prompt multipliers do not survive to chat/agent context
  lengths. Any headline must name the depth it was measured at.
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
