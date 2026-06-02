# Camelid v0.1 Release Notes

Date: 2026-05-31

Release checkout: branch-agnostic working notes; record the exact branch and SHA when cutting rc1

## Release Posture

Camelid v0.1 is an evidence-first release candidate. It is meant to show exactly what the repository can defend today, not what the runtime is expected to support later.

The release is bounded by exact model rows, committed validation artifacts, and fail-closed unsupported states. Neighboring model sizes, quantizations, template variants, larger contexts, and alternate runtimes do not inherit support.

No tag should be cut from this branch until the release gate in [`RELEASE_GATE_v0.1.md`](RELEASE_GATE_v0.1.md) is green and the release captain signs off.

## What Is Included

- Rust GGUF backend and OpenAI-style local API surfaces.
- React/Vite frontend that reflects the backend compatibility contract.
- Exact-row v0.1 support ledger in [`SUPPORT_MATRIX_v0.1.md`](../../SUPPORT_MATRIX_v0.1.md), with broader background in [`COMPATIBILITY.md`](../../COMPATIBILITY.md).
- Current support and blocker snapshot in [`STATUS.md`](../../STATUS.md).
- Committed benchmark snapshot in [`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md).
- Public evidence-bundle checks for support-sensitive JSON artifacts.
- Public scrub guard for private paths, host details, and legacy branding.

## Supported Public Claims

Camelid v0.1 may claim exact-row support only where [`SUPPORT_MATRIX_v0.1.md`](../../SUPPORT_MATRIX_v0.1.md) cites row-specific evidence:

- `TinyLlama 1.1B Chat Q8_0`: verified support gate.
- `Llama 3.2 1B Instruct Q8_0`: verified bounded support.
- `Llama 3.2 3B Instruct Q8_0`: supported exact-row smoke.
- `Llama 3 8B Instruct Q8_0`: verified bounded support.
- `Mistral-7B-Instruct-v0.3 Q8_0`: evidence-only bring-up; support is not promoted in v0.1 because the API/WebUI support contract is still fail-closed.

The release may also mention `Mixtral-8x7B-Instruct-v0.1 Q8_0` as active validation with bounded one-token backend MoE runtime evidence only. Later generation, continuation, API/WebUI/frontend readiness, long-context behavior, production throughput, and broad Mixtral support remain blocked.

`Qwen2.5-7B-Instruct Q8_0` and `gemma-2-9b-it Q8_0` are planned candidates only.

## What Is Not Included

Camelid v0.1 must not claim:

- broad Llama-family, Mistral-family, Mixtral-family, Qwen-family, or Gemma-family support
- support for neighboring model rows or unvalidated quantizations
- model-native or larger context support beyond checked packs
- arbitrary GGUF/Jinja template support beyond row-scoped evidence
- production throughput
- complete same-host throughput parity or superiority versus llama.cpp, MLX, or Ollama
- distributed inference release readiness
- default-on accelerated-kernel support outside documented gates

## Evidence Anchors

Primary public docs:

- [`COMPATIBILITY.md`](../../COMPATIBILITY.md)
- [`STATUS.md`](../../STATUS.md)
- [`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md)
- [`PARITY.md`](../benchmarks/PARITY.md)
- [`qa/evidence-bundles/README.md`](../../qa/evidence-bundles/README.md)

Representative evidence bundles cited by the public docs include:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-broader-50tok-ubuntu-20260509T000633Z-head-d330e97ae992/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-api-webui-rss-current-head-20260513T1935Z-head-9a296ea/manifest.json`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/manifest.json`

## Release Blockers

- A fresh v0.1 evidence bundle under `qa/evidence-bundles/v0.1/` is still required.
- Comparator baselines for llama.cpp, MLX, and Ollama must either be run or explicitly deferred by the release captain.
- Full lightweight gate results must be recorded in [`RELEASE_GATE_v0.1.md`](RELEASE_GATE_v0.1.md).
- No release tag has been created.
