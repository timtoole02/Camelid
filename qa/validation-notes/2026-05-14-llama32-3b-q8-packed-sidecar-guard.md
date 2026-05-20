# 2026-05-14 — Llama 3.2 3B Q8 packed-sidecar load guard

Scope: exact `llama32_3b_instruct_q8_0` runtime/load behavior guard for the experimental Q8_0 packed rows4 dot kernels. This is code/regression coverage only; it does not promote production throughput, portability, larger/model-native context, arbitrary templates, neighboring rows, 1B, 8B, Mixtral, or broad Llama-family support.

Current repo state at start of slice: local `main` was clean at `0b9a1d010c998ad8077044386154f3d54ee64c15` and ahead of `origin/main` by the current 3B support-surface/runtime commits. Current 3B support evidence remains the canonical Ubuntu source-head API/WebUI support-gate bundle `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`, the checked 512/1024/2048 bounded context packs, compact/broader parity packs, row-scoped metadata-Jinja/template-shape evidence, bounded unique-chat perf/RSS evidence, and the opt-in parallel Q8 first-token direction probe.

## Change

Added `q8_packed_rows4_sidecars_stay_opt_in_per_layout` coverage in `src/tensor/mod.rs` to prove the new retained Q8_0 packed rows4 sidecars are not built during default tensor construction and are only built for the specific opted-in layout:

- no `CAMELID_Q8_0_PACKED_4X4_DOT` / `CAMELID_Q8_0_PACKED_4X8_DOT` env: no packed sidecar is materialized;
- `CAMELID_Q8_0_PACKED_4X4_DOT=on`: only the 4x4 packed sidecar is materialized;
- `CAMELID_Q8_0_PACKED_4X8_DOT=on`: only the 4x8 packed sidecar is materialized.

This keeps the 3B runtime/load lane fail-safe while the packed-dot kernels remain experimental: default 3B load behavior should not inherit extra sidecar memory or runtime shape changes from performance experiments.

## Local validation

- `cargo fmt --all -- --check` — PASS
- `./scripts/with-rustup-cargo.sh test q8_packed_rows4_sidecars_stay_opt_in_per_layout --lib -- --nocapture` — PASS
- `./scripts/with-rustup-cargo.sh test q8_0_packed --lib -- --nocapture` — PASS

Combined local test log: `target/cron-95495a91-20260514T1854Z-3b-q8-packed-sidecar-test.log`.

## Remaining blocker

The exact 3B production-throughput claim is not promotion-ready until a canonical Ubuntu same-host harness runs against the exact `Llama-3.2-3B-Instruct-Q8_0.gguf`, captures a scrubbed report/bundle with real GGUF and binary SHA256 values plus measured resource snapshots, and synchronizes `/api/capabilities`, frontend copy, README/COMPATIBILITY/STATUS only for the exact row and measured envelope. Any packed-sidecar throughput experiment must remain opt-in and must beat the default 3B evidence before it can be cited beyond experimental runtime work.
