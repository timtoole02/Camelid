# Camelid v0.1 Benchmark Summary

- Generated UTC: 2026-05-31T18:43:07.581Z
- Release version: v0.1.0-rc1
- Benchmark: v0.1 local same-host llama.cpp comparator
- Purpose: Capture a bounded same-host Camelid versus llama.cpp CPU comparator row for the exact Llama 3.2 3B Instruct Q8_0 model.
- Dry run: false

| Entry | Engine | Model | Status | Runs | Avg ms | Peak RSS KB |
|---|---|---|---:|---:|---:|---:|
| llamacpp-cpu-samehost-llama32-3b | llama.cpp | llama32-3b-instruct-q8-local | ok | 1 | 35658.57 | 109632 |

## Output Files

- `machine.json` captures host, Node, memory, CPU, and Git context.
- `model_manifest.json` records model metadata and optional local file stats/hash evidence.
- `commands.md` records the exact configured command for every run.
- `raw_logs/` stores per-run stdout, stderr, and command metadata.
- `results.json` and `results.csv` contain machine-readable timing, exit, output, and memory fields.
