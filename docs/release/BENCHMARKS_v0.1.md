# Camelid v0.1 Benchmarks

Date: 2026-05-31

Branch: `release/v0.1-evidence`

Release candidate SHA: release branch HEAD; record exact SHA when cutting rc1

## Benchmark Posture

Camelid v0.1 publishes bounded benchmark evidence, not a broad speed claim. The committed benchmark story is:

- exact-row only
- short bounded workloads only
- parity and support evidence before performance language
- same-host comparison required before any comparator claim
- memory, timing, and runtime mode stated separately

If a number is not backed by a committed evidence bundle or a release-captain-approved v0.1 bundle, it should not be used in public copy.

## Committed Snapshot

The current public benchmark source of truth is [`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md). This v0.1 file restates the release-safe subset.

### Ubuntu Bounded Unique-Chat Envelope

Evidence: `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/summary.json`

Method:

- endpoint: `/v1/chat/completions`
- warmup: 2
- measured repeats: 4
- max tokens: 5
- prompt style: unique prompts
- prompt cache: no measured hits
- weight cache: hot on measured runs

| Exact row | Avg wall ms | Avg generate ms | Approx ms/output token | Max backend RSS MiB | Claim boundary |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama 3.2 1B Instruct Q8_0 | 7379.73 | 7065.25 | 1413.05 | 274.31 | Bounded unique-chat envelope only. |
| Llama 3.2 3B Instruct Q8_0 | 19762.21 | 19449.25 | 3889.85 | 287.21 | Bounded unique-chat envelope only. |

This does not claim production throughput, portability, broad Llama-family behavior, or support for neighboring rows.

### Apple Silicon Memory-First Profile vs MLX-LM

Evidence: `qa/evidence-bundles/apple-silicon-camelid-vs-mlx-memory-20260514T001835Z-head-775db673af32/summary.json`

Method:

- same-host Apple Silicon resident-memory profile
- Camelid mode: memory-first lazy GGUF Q8_0
- MLX mode: public `mlx-community` 4-bit MLX-LM weights
- metric: observed process RSS sampled with `ps`

| Model family | Camelid profile | Camelid RSS MiB | MLX-LM profile | MLX-LM RSS MiB | MLX/Camelid RSS |
| --- | --- | ---: | --- | ---: | ---: |
| Llama 3.2 1B Instruct | GGUF Q8_0, memory-first lazy | 257.72 | 4-bit MLX-LM | 1062.06 | 4.12x |
| Llama 3.2 3B Instruct | GGUF Q8_0, memory-first lazy | 328.92 | 4-bit MLX-LM | 2139.70 | 6.51x |

This is a resident-memory comparison only. It is not a quant-equivalent speed comparison, and the MLX rows were much faster in the short timing probe.

### Apple Silicon Retained-Q8 Scheduler Gate

Evidence: `qa/evidence-bundles/apple-silicon-retained-q8-hybrid-default-off-20260514T020011Z-head-fbd37d1/summary.json`

Method:

- same-host Apple Silicon retained-Q8 short decode gate
- exact row: Llama 3.2 3B Instruct Q8_0
- endpoint: `/v1/chat/completions`
- max tokens: 8
- prompt cache: unique prompts with no measured hits
- weight cache: hot on measured rows

| Mode | Repeats | Avg generate ms | Avg layers ms | Avg FFN total ms | Observed peak backend RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous implicit hybrid default, 10 percent GPU suffix | 2 | 4867.00 | 4749.75 | 883.52 | 4458.73 |
| Explicit CPU-only sweep | 2 | 4808.00 | 4689.74 | 858.80 | 4255.06 |
| New default, hybrid off | 3 | 4819.33 | 4701.71 | 846.32 | 3817.66 |

Decision recorded by the evidence: retained-Q8 CPU plus Metal hybrid remains opt-in because the measured hybrid suffix scheduler did not win this short 3B gate.

### Ubuntu Compact Smoke Timing Snapshot

Evidence: `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`

Method:

- compact API/WebUI validation lane
- message: `hello`
- max tokens: 1
- same sanitized Ubuntu validation host

| Exact row | Model load ms | Chat completion ms | Backend generate ms | Max RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B Chat Q8_0 | 134 | 40558 | 40487 | 136588 |
| Llama 3.2 1B Instruct Q8_0 | 559 | 20973 | 20674 | 347316 |
| Llama 3.2 3B Instruct Q8_0 | 566 | 45138 | 44830 | 559572 |
| Llama 3 8B Instruct Q8_0 | 566 | 81086 | 80794 | 566980 |

These are release-audit timing snapshots. They are not a broad runtime-speed claim.

## Comparator Baseline Status

`BENCHMARKS.md` states that Camelid does not yet publish a fully normalized same-host throughput table versus llama.cpp for every headline row. That remains true for v0.1.

Current comparator status:

- llama.cpp: bounded parity evidence exists for cited exact rows; a full same-host v0.1 throughput table is still required or must be explicitly deferred.
- MLX-LM: memory comparison evidence exists for two Apple Silicon rows; it is not a speed-win claim.
- Ollama: no v0.1 comparator result is recorded in this document.

## Next Benchmark Work

The next release-useful benchmark bundle should record:

- exact model file and SHA256
- Camelid SHA
- comparator version or commit
- host and OS summary after public scrubbing
- prompt pack, context, max tokens, temperature, and thread settings
- raw commands
- wall time, TTFT where available, decode estimate where available, and memory
- pass/fail status

Until that bundle exists, v0.1 should use the committed bounded snapshots above and avoid broader throughput positioning.
