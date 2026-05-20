# 2026-05-14 — Llama 3.2 3B throughput harness evidence-context guard

Scope: exact `llama32_3b_instruct_q8_0` production-throughput blocker path only. This is harness/validation hygiene; it does not promote production-throughput, portability, larger/model-native context, arbitrary templates, neighboring rows, 1B, 8B, Mixtral, or broad Llama-family support.

Current repo state at start of slice: local `main` was clean at `eb4df5407de198e54163b226eed935cd323e587b` and ahead of `origin/main` by the three local 3B support-surface commits (`150fdef`, `94df497`, `eb4df54`). Current 3B support evidence remained the exact-row API/WebUI current-head bundle `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`, the checked 512/1024/2048 bounded context packs, compact/broader parity packs, row-scoped metadata-Jinja/template-shape evidence, bounded unique-chat perf/RSS evidence, and the opt-in parallel Q8 first-token direction probe.

## Change

Hardened `scripts/bench-llama3-same-host.mjs` so future exact 3B same-host timing artifacts carry the missing evidence-context fields required before the production-throughput blocker can be responsibly cleared:

- repository head/branch/status in the machine-readable plan/report;
- host class without hostname/user/home disclosure;
- model, Camelid binary, and llama-server file evidence, including SHA256 in full run mode and explicit `not_computed_in_plan_mode` for dry plans;
- pre-start, before-measured, and after-measured resource snapshots with memory/load/storage fields plus Linux `/proc/meminfo` swap/available-memory fields when available;
- deterministic marker-presence guardrails for `CMLD-BENCH`, optionally enforced with `--require-marker` after writing the report.

## Local validation

- `node scripts/test-bench-llama3-same-host.mjs` — PASS
- `./scripts/with-rustup-cargo.sh test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture` — PASS
- `(cd frontend && npm run smoke:model-state)` — PASS
- Dry-run plan artifact generated at `target/cron-95495a91-20260514T1018Z-llama32-3b-throughput-harness-plan.json` for:

```bash
node scripts/bench-llama3-same-host.mjs \
  --print-plan \
  --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-id llama32-3b-q8-throughput \
  --row-id llama32_3b_instruct_q8_0 \
  --max-tokens 16 \
  --warmup 1 \
  --repeats 3 \
  --threads 8 \
  --out target/cron-95495a91-20260514T1018Z-llama32-3b-throughput-harness-plan.json
```

## Remaining blocker

The exact 3B production-throughput claim is not promotion-ready until a canonical Ubuntu same-host harness runs against the exact `Llama-3.2-3B-Instruct-Q8_0.gguf`, captures a scrubbed report/bundle with real GGUF and binary SHA256 values plus measured resource snapshots, and synchronizes `/api/capabilities`, frontend copy, README/COMPATIBILITY/STATUS only for the exact row and measured envelope.
