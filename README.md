# Camelid

Camelid is a Rust-native local GGUF inference backend with an evidence-gated support contract. The v0.1 release candidate is intentionally narrow: it publishes exact model rows that have row-specific validation, and it keeps everything else behind explicit blockers.

Camelid is not presented as broad Llama, Mistral, Mixtral, distributed, production-throughput, or universal frontend support. If a model row, quantization, context window, comparator result, or UI readiness state is not backed by a cited artifact, it is outside the public v0.1 claim.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

The current WebUI is product-forward while still reflecting the local-first runtime contract: a dark, collapsed-rail chat surface with exact-row readiness gates instead of broad model support.

## Current Release Boundary

[`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) is the v0.1 release-candidate support contract. It is stricter than the broader repository docs where those docs conflict with current evidence. The short version is:

| Exact row | v0.1 public status | Checked boundary |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Verified support gate | Parity, API/WebUI, template-shape, bounded 512-context, and RSS/perf evidence. |
| Llama 3.2 1B Instruct Q8_0 | Verified bounded support | Load, completions, chat completions, WebUI, parity, row-scoped template evidence, unique-chat RSS/perf, and checked bounded context packs through 8192 where cited. |
| Llama 3.2 3B Instruct Q8_0 | Supported exact-row smoke | Canonical API/WebUI support-gate refresh, compact/broader parity, template evidence, bounded RSS/perf, and checked 512/1024/2048 context packs where cited. |
| Llama 3 8B Instruct Q8_0 | Verified bounded support | Compact/broader parity, API/WebUI, bounded memory evidence, and checked 512/1024/2048 context packs where cited. |
| Mistral-7B-Instruct-v0.3 Q8_0 | Active exact-row validation only | Load, tokenizer/template, one-token parity, broader 50-token parity, and checked 512/1024/2048/4096/8192 context evidence exist, but the API/WebUI support contract is still fail-closed for v0.1. |
| Mixtral-8x7B-Instruct-v0.1 Q8_0 | Active validation / partial backend runtime only | Bounded one-token backend MoE runtime evidence exists, but later-generation parity and continuation/API/WebUI readiness remain blocked. |
| Qwen2.5-7B-Instruct Q8_0 | Planned candidate only | No support claim. |
| gemma-2-9b-it Q8_0 | Planned candidate only | No support claim. |

Nothing adjacent inherits support across size, quantization, tokenizer, context, API surface, or frontend state.

Read-only compatibility discovery routes, including partial llama-server-style model discovery, are not support promotions. They must stay privacy-safe, list only currently loaded public state, and leave router-mode model management, native load/unload, and WebUI readiness locked to the evidence-backed compatibility contract.

## What v0.1 Is For

Camelid v0.1 is a release candidate for reviewers who care about reproducible local-inference evidence:

- exact-row compatibility language instead of family-wide claims
- committed parity, context, API/WebUI, RSS, and benchmark artifacts
- explicit unsupported states for partial or blocked rows
- small local gates that contributors can run before changing support-sensitive code

The benchmark story is similarly bounded. Camelid publishes memory and timing snapshots that are already committed in evidence bundles. It does not yet publish a complete apples-to-apples throughput table against llama.cpp for every headline row.

## Quickstart

Build the backend:

```bash
cargo build --release
```

Serve a local GGUF model:

```bash
./target/release/camelid serve \
  --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --threads 4
```

Check local capability reporting:

```bash
curl -s http://127.0.0.1:8181/api/capabilities
```

Run the frontend development server:

```bash
cd frontend
npm ci
npm run dev
```

The frontend is a local React/Vite chat surface. Its readiness state should be read literally: chat is enabled only when the loaded model row is recognized as supported by the current compatibility contract.

## Validation

For normal code changes, start with:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

For public evidence and release-doc changes, also run:

```bash
node scripts/check-public-evidence-claims.mjs
bash scripts/check-public-scrub.sh
cd frontend && npm ci && npm run build && npm run smoke:model-state
```

Real-model parity, comparator baselines, and full v0.1 evidence bundles require model files and comparator runtimes that are not vendored in this repository. Those runs must record the exact model path, hash, runtime versions, commands, timing, memory, and pass/fail status in a scrubbed evidence bundle.

## Documentation Map

- [`COMPATIBILITY.md`](COMPATIBILITY.md) - authoritative support contract
- [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) - v0.1 release-candidate support boundary
- [`CORRECTNESS_v0.1.md`](CORRECTNESS_v0.1.md) - v0.1 correctness boundary
- [`STATUS.md`](STATUS.md) - current evidence snapshot and blockers
- [`BENCHMARKS.md`](BENCHMARKS.md) - committed benchmark snapshot and claim rules
- [`docs/WAR_ROOM_EVIDENCE_INDEX.md`](docs/WAR_ROOM_EVIDENCE_INDEX.md) - claim-source order, evidence index, and public wording policy
- [`RELEASE_NOTES_v0.1.md`](RELEASE_NOTES_v0.1.md) - v0.1 release candidate notes
- [`BENCHMARKS_v0.1.md`](BENCHMARKS_v0.1.md) - v0.1 benchmark posture
- [`MARKET_POSITIONING_v0.1.md`](MARKET_POSITIONING_v0.1.md) - public positioning guardrails
- [`RELEASE_GATE_v0.1.md`](RELEASE_GATE_v0.1.md) - release gate commands and status
- [`ROADMAP.md`](ROADMAP.md) - planned engineering sequence
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - implementation architecture

## License and Reference Credits

Camelid is licensed under the [MIT License](LICENSE).

Camelid's tokenizer, reference compatibility layouts, and validation benchmarks are inspired by and checked against [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (Copyright (c) 2023-2026 The ggml authors, MIT License). Camelid maintains its own Rust-native codebase while crediting the reference work of the `ggml` ecosystem.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
