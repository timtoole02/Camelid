# Camelid Benchmarks

Last updated: 2026-05-14

This file is Camelid's public performance snapshot.

It is intentionally narrower than a marketing benchmark page:

- it reports only sanitized, reproducible numbers already backed by committed evidence bundles
- it separates **runtime measurements** from **support claims**
- it does **not** treat one good number as broad model-family proof

If a row or host is not listed here, Camelid is not claiming a benchmark result for it yet.

For cross-surface wording discipline, see [`docs/WAR_ROOM_EVIDENCE_INDEX.md`](../WAR_ROOM_EVIDENCE_INDEX.md). That policy keeps benchmark copy tied to scrubbed evidence and prevents timing probes from becoming support, API, or WebUI readiness claims.

## Reading rules

- **Exact row only.** Numbers apply only to the exact GGUF row named in the table.
- **Bounded workload only.** These are short, bounded validation or microbenchmark lanes, not production multi-user load tests.
- **Same-host comparison only.** A fair Camelid-vs-llama.cpp throughput claim requires the same prompt shape, same model file, same host, same thread settings, and the same token budget.
- **Parity first.** Camelid treats 1:1 token parity with llama.cpp as the prerequisite for performance claims, not a substitute for them.

## Benchmark snapshot: current committed numbers

### Ubuntu bounded unique-chat envelope

These results come from `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 2
- measured repeats: 4
- max tokens: 5
- prompt style: unique prompts, average prompt token count `22.5`
- weight cache: hot on measured runs
- prompt cache: no hits on measured runs

| Exact row | Avg wall ms | Avg generate ms | Approx ms / output token | Max backend RSS MiB | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama 3.2 1B Instruct Q8_0 | 7379.73 | 7065.25 | 1413.05 | 274.31 | Exact-row bounded unique-chat envelope only |
| Llama 3.2 3B Instruct Q8_0 | 19762.21 | 19449.25 | 3889.85 | 287.21 | Exact-row bounded unique-chat envelope only |

### Apple Silicon memory-first profile vs MLX-LM

These results come from `qa/evidence-bundles/apple-silicon-camelid-vs-mlx-memory-20260514T001835Z-head-775db673af32/summary.json`.

Method summary:

- scope: same-host Apple Silicon resident-memory profile
- Camelid mode: memory-first lazy GGUF Q8_0 (`CAMELID_LAZY_Q8_0_LINEAR=1`, retained Q8 blocks disabled)
- MLX mode: public `mlx-community` 4-bit MLX-LM weights, cache already warm for committed timing rows
- metric: observed process RSS sampled with `ps`; Camelid rows also include structured forward RSS and Q8 file-read counters
- boundary: this is a **memory** comparison, not a strict quant-equivalent speed claim; MLX is much faster in this short probe

| Model family | Camelid row/profile | Camelid observed RSS MiB | MLX-LM row/profile | MLX-LM observed RSS MiB | MLX / Camelid RSS |
| --- | --- | ---: | --- | ---: | ---: |
| Llama 3.2 1B Instruct | GGUF Q8_0, memory-first lazy | 257.72 | 4-bit MLX-LM | 1062.06 | 4.12x |
| Llama 3.2 3B Instruct | GGUF Q8_0, memory-first lazy | 328.92 | 4-bit MLX-LM | 2139.7 | 6.51x |

This is a useful Camelid result, but the claim must stay precise: Camelid's Rust GGUF path can run the exact Q8_0 rows with substantially lower resident memory in the memory-first profile. It does **not** claim Camelid is faster than MLX-LM on this workload.

### Apple Silicon retained-Q8 scheduler gate

These results come from `qa/evidence-bundles/apple-silicon-retained-q8-hybrid-default-off-20260514T020011Z-head-fbd37d1/summary.json`.

Method summary:

- scope: same-host Apple Silicon retained-Q8 short decode gate
- exact row: Llama 3.2 3B Instruct Q8_0
- endpoint: `/v1/chat/completions`
- max tokens: `8`
- weight cache: hot on measured rows
- prompt cache: unique prompts, no measured prompt-cache hits
- boundary: this is a narrow scheduler tuning gate, not a broad Metal benchmark

| Mode | Repeats | Avg generate ms | Avg layers ms | Avg FFN total ms | Observed peak backend RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous implicit hybrid default, 10% GPU suffix | 2 | 4867 | 4749.75 | 883.52 | 4458.73 |
| Explicit CPU-only sweep | 2 | 4808 | 4689.74 | 858.8 | 4255.06 |
| New default, hybrid off | 3 | 4819.33 | 4701.71 | 846.32 | 3817.66 |

Decision: retained-Q8 CPU+Metal hybrid remains available as an explicit experiment, but normal retained-Q8 serving defaults to the CPU path because the measured hybrid suffix scheduler did not win this gate.

### Ubuntu first-token direction probe

This result comes from `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 1
- measured repeats: 1
- max tokens: 1
- comparison: default serial file-backed Q8 path vs opt-in parallel output-row partitioning

| Exact row | Mode | Avg generate ms | Avg generate ms / prompt token | Max backend RSS MiB | Delta |
| --- | --- | ---: | ---: | ---: | --- |
| Llama 3.2 3B Instruct Q8_0 | serial Q8 path | 13960 | 775.56 | 283.57 | baseline |
| Llama 3.2 3B Instruct Q8_0 | opt-in parallel Q8 path | 12200 | 677.78 | 282.97 | `-12.61%` generate time |

This closes only the exact 3B **first-token direction** sub-box. It is not a broad production-throughput claim.

### Ubuntu compact smoke timing snapshot

These results come from `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`.

Method summary:

- compact smoke / API + WebUI oriented validation lane
- `hello` message, `max_tokens=1`
- same sanitized Ubuntu validation host

| Exact row | Model load ms | Chat completion ms | Backend generate ms | Max RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B Chat Q8_0 | 134 | 40558 | 40487 | 136588 |
| Llama 3.2 1B Instruct Q8_0 | 559 | 20973 | 20674 | 347316 |
| Llama 3.2 3B Instruct Q8_0 | 566 | 45138 | 44830 | 559572 |
| Llama 3 8B Instruct Q8_0 | 566 | 81086 | 80794 | 566980 |

These are useful for release-audit snapshots, not for broad “fastest local runtime” marketing.

## Llama.cpp comparison policy

Camelid already publishes bounded **parity** against llama.cpp for the exact supported rows, but the repo does **not yet** publish a fully normalized same-host throughput table against llama.cpp for those rows.

That means Camelid should say this plainly today:

- **Yes:** Camelid has exact-row 1:1 bounded parity with llama.cpp where cited.
- **Not yet:** Camelid has not published a repo-safe apples-to-apples throughput table versus llama.cpp on the same host for every headline row.

That missing table is an execution gap, not a copywriting gap.

The tracked harness for closing the first same-host 3B slice is:

```sh
CAMELID_BIN=target/release/camelid \
LLAMA3_LLAMA_SERVER=target/reference/llama.cpp/build/bin/llama-server \
node scripts/bench-llama3-same-host.mjs \
  --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-id llama32-3b-q8-throughput \
  --row-id llama32_3b_instruct_q8_0 \
  --max-tokens 16 --warmup 1 --repeats 3 --threads 8 \
  --out target/bench-llama32-3b-same-host.json
```

Use `--print-plan` with the same arguments to audit exact spawned commands, stdout keys, JSON schema, and metric bounds before starting servers. The harness reports bounded TTFT, elapsed-time, and streamed-chunk-derived decode estimates only; it does not promote production throughput, 1B, Mixtral, neighboring-row, portability, or broader-family support without separate row-specific evidence.

Latest readiness note: `qa/validation-notes/2026-05-14-throughput-host-readiness-recheck.md` records improved approved Ubuntu lane readiness after the full-root-storage condition, but no same-host benchmark has been run or published from that probe. Treat the apples-to-apples table as still missing until a scrubbed row-specific evidence bundle is committed.

## What should be added next

The next benchmark slice worth publishing is:

1. **Same-host Camelid vs llama.cpp on the exact 3B row**
   - same model SHA
   - same prompt pack
   - same token budget
   - same thread settings
   - report TTFT, decode tok/s, and ms/token
2. **Mac benchmark snapshot**
   - especially the exact Llama 3.2 3B Instruct Q8_0 row that is actively exercised in the UI
3. **8B bounded same-host comparison**
   - only after the same-host run is reproducible and repo-safe

## Claim boundary

This file is a benchmark snapshot, not a support matrix.

It does not widen any support claim beyond `COMPATIBILITY.md`, `STATUS.md`, and the cited evidence bundles.
