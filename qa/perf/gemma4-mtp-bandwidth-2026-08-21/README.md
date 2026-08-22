# Gemma 4 26B-A4B — MTP throughput, 2026-08-21

Tooling and measurements from the session that took the MTP lane from 12.31 to
15.21 tok/s, bit-exact. Every step was verified by comparing `output_token_ids`
against the previous run; all three lanes stayed token-identical throughout.

## Headline measurements

| quantity | value | how |
|---|---|---|
| MTP throughput | 12.31 -> 15.21 tok/s | pilot receipts |
| MTP alpha | 6.40 -> 7.38 | `accepted_target_tokens_per_round` |
| acceptance probability | 1.000 | every proposed token accepted |
| round split | disk ~300 ms, GPU ~118 ms, slot_wait ~117 ms | `[metal chained ledger]` |
| expert traffic | 93.8 MB/emitted token (32 tok), 81.8 (96 tok) | lane-aware, decode rounds only |
| GPU stage split | gateup ~57%, qkv_o ~26%, shared ~17% | `[metal chained stages]` |
| decode miss floor | ~210 records/round, flat (-4% over 26 rounds) | does NOT decay |

## Tools

- `hybrid-hot48-runner/` — manual, fail-closed load-only → K8/K1 smoke →
  K8/K1 promotion ladder for the 48-hot/128-mapped hybrid profile. It blocks
  real runs until matching load-only v4 and every-round structured-telemetry
  integration contracts exist.
- `run-pilot-instrumented.zsh <run-dir>` — admission + pilot under the fail-closed
  memory watchdog, with the per-round ledger enabled. Takes a fresh run dir.
- `analyze-pilot.py <run-dir>` — summarizes a pilot report + watchdog: lanes,
  alpha, tok/s, kill reasons, routed-expert residency.
- `bench-expert-read.py` — raw .cghost read bandwidth at the engine's record size.
- `bench-read-batch-depth.py` — **the important one.** Replicates the engine's
  per-layer batch + barrier. Shows depth, not thread count, is the limiter:
  batch 7 gives 2.21 GB/s at 4 threads and 2.23 at 8; batch 28 gives 3.14 / 4.01.
- `bench-read-fd-sharing.py` — shared fd vs per-thread fd (worth ~10%).
- `gateup-coalescing-bench.rs.snippet` — in-crate Metal microbenchmark isolating
  lane-stride coalescing. Measured **1.20x**, not the hoped 2x, which is why the
  channel-tiled repack of the .cghost was NOT pursued.

## Results worth not re-deriving

Negative, with receipts:

- **Victim ring**: correct (`verify_fails=0`) but a net loss — 6-15 hits per 188
  misses against 28-33 ms of salvage. Per byte of RAM it is strictly worse than a
  larger slot table.
- **OS page cache**: no bandwidth change, more bytes, 12.31 -> 10.88 tok/s. A
  10.6 GiB working set cannot be cached beside 5.26 GiB of wired slots on 16 GB.
- **Contextual sparsity**: `down` is stored [hidden][ff], so intermediate-dim
  sparsity selects non-contiguous columns and saves zero bytes on that third.
- **More read threads**: 8 threads measured no gain and a slight loss in the
  engine. Depth is the limiter, not concurrency.
- **Channel-tiled repack**: only 1.20x from coalescing; not worth a 12.9 GB
  format change and new payload-identity hashes.

Two microbenchmarks misled this session and the real engine run caught both.
Prefer an engine A/B over a synthetic benchmark when the two disagree.

## Next lever

Cross-layer speculative expert prefetch. Justification is the depth curve above,
not latency hiding: prefetching ~1.45x the bytes at ~1.81x the rate is net
favourable, and the correctly-predicted records leave the critical path entirely.
The machinery exists (`predicted_ready` + verify-and-retry) and is disabled for
record-granular mode by a single `!record_demand` in src/metal.rs. Check first
that a mispredicted round retries rather than hitting the terminal
`bounded_record_fallback_forbidden`, and measure the real prefetch hit rate
(55% is `union_vs_prev`, a proxy).
