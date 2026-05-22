# Camelid

[![CI][ci-badge]][ci-workflow]

**Camelid is trust-first local AI infrastructure for teams that need local models they can actually adopt.**

Most local-model stacks are easy to demo and hard to trust. Camelid is a Rust-native local inference backend for GGUF language models built for the point where local AI becomes real infrastructure: products, internal platforms, regulated environments, and customer-facing workflows.

Camelid does not treat “probably works” as “supported.” Support moves only when the evidence is green.

## Benchmark highlights

Camelid is working toward 1:1 behavioral parity with llama.cpp while keeping the implementation Rust-native. The first goal is verification, not speed theater: match the trusted local-inference baseline for exact rows, prove bit-perfect generated-token agreement where Camelid already agrees with it, and only then promote faster paths.

Current retained highlights, ordered by product importance:

| Priority | Workload / envelope | Path | Reference | Camelid | Status |
| --- | --- | --- | ---: | ---: | --- |
| Verification first | Exact-row generation parity for Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Q8_0 checked envelopes | Default supported path | llama.cpp token IDs and text | **Bit-perfect generated token IDs and text** | Promoted only where exact-row parity, API, WebUI, and evidence agree. |
| Headline win | Llama 3.2 3B Q8_0 guarded stream run | Confirmed experimental highlight | 5.60s total | **3.01s total** | **46.4% faster** with marker guards passing for both runtimes. |
| Current target | Llama 3.2 3B Q8_0 x86 route-map run | Default-off Rust Q8 optimization lane | **374.88ms total** | 413.42ms total | Camelid trails by **10.3%**; this is the current measured optimization target. |
| Direction probe | Llama 3.2 3B Q8_0 parallel Q8 first-token probe | Default-off parallel Q8 probe | Camelid baseline: 13.96s TTFT | **12.20s TTFT** | Rust-side Q8 work cut TTFT by **12.6%** in the retained direction probe. |

> **Benchmark boundary:** these are retained highlights, not a broad production-throughput claim. Optimized paths are evidence-gated and default-off. Camelid promotes code to the default path only after exact-row parity, repeatability, portability, and support-contract checks all agree.

Public evidence anchors include [`STATUS.md`](STATUS.md) and the [`Llama 3.2 3B parallel Q8 first-token manifest`](qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json). Same-host raw benchmark artifacts are retained in validation evidence and must be scrubbed before publication.

## Why organizations adopt Camelid

Organizations adopt Camelid because local AI needs more than raw inference. They need a product contract they can trust: what works, what does not, and whether the runtime, API, and UI all tell the same truth.

Camelid gives teams:

- **clear support contracts** instead of hand-wavy compatibility
- **OpenAI-compatible APIs** that fit existing tools and agent stacks
- **fail-closed readiness** so unsupported models do not masquerade as production-ready
- **evidence-backed support** tied directly to runtime reality
- **one consistent product contract** across backend, API, WebUI, docs, and validation

If you want local AI without guessing what is actually ready, Camelid is built for that gap.

## Why Camelid matters

The hard problem in local AI is no longer just getting a model to run. The hard problem is operational trust.

Most local inference stacks optimize for breadth first and clarity later. Camelid takes the opposite approach: make support explicit, make readiness visible, and make the product fail closed when the evidence is not there.

That is not just a docs preference. It is the product wedge.

## What teams can use today

Camelid already ships a serious local inference product surface:

- a Rust runtime for GGUF model loading and inference
- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI with fail-closed readiness gating
- parity and validation harnesses that verify behavior against llama.cpp before support language moves

Today, four exact Q8_0 rows are public and evidence-backed:

- **TinyLlama 1.1B Chat Q8_0** — verified support
- **Llama 3.2 1B Instruct Q8_0** — verified end-to-end support at checked 512/1024/2048/4096/8192 contexts
- **Llama 3.2 3B Instruct Q8_0** — supported exact-row smoke with canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65` plus checked 512/1024/2048 contexts
- **Llama 3 8B Instruct Q8_0** — verified support at checked 512/1024/2048 contexts

Mixtral has one-token backend MoE runtime evidence but is not yet promoted to API/WebUI/frontend readiness. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` is the active next exact-row bring-up lane.

## Why this can become a company

Camelid is not trying to be another thin local-model wrapper. The wedge is operational trust.

Teams adopting local AI for product, agent, or regulated workloads need more than raw inference. They need a runtime that tells the truth about what is actually ready, keeps the API and UI aligned with that truth, and expands support without breaking confidence. Camelid turns that requirement into product behavior.

That creates a real commercial path:

- **Enterprise local AI** that needs auditable support boundaries
- **Agent and developer platforms** that want an OpenAI-compatible local runtime without hand-wavy readiness claims
- **Regulated or privacy-sensitive deployments** where fail-closed behavior matters more than broad demo compatibility

> **Support boundary:** Camelid makes exact-row claims only. Current supported rows have row-scoped or bounded template/perf evidence where cited, but broad arbitrary/Jinja-template behavior, production throughput, wider model-native context, portability, neighboring rows, and broad-family behavior still move only when matching evidence is green.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

*Approved dark, collapsed-rail chat surface: product-forward while still reflecting the local-first runtime contract.*

## Current support boundary

Camelid claims llama.cpp parity only for exact GGUF rows and envelopes with published validation evidence: TinyLlama at the current validated gate; Llama 3.2 1B Q8_0 with verified end-to-end support at checked 512/1024/2048/4096/8192-context packs; Llama 3.2 3B Q8_0 as supported exact-row smoke with canonical Ubuntu API/WebUI refresh, compact/broader parity, and checked 512/1024/2048-context packs; and Llama 3 8B Q8_0 through exact-row smoke plus checked 512/1024/2048-context packs tied to cited source/runtime-head bundles.

`Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` has one-token backend MoE runtime evidence, but Gate 9A later-generation evidence diverges and a longer-continuation backend HTTP hang remains unresolved, so Mixtral API/WebUI/frontend readiness and broad Mixtral support are not claimed. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` remains an active exact-row bring-up lane, not a supported row.

## Current execution tracks

Camelid is advancing on three tracks:

- **Supported-row hardening:** preserve TinyLlama as the full current gate, keep Llama 3.2 3B support wording tied to its exact-row API/WebUI/parity/runtime evidence, and continue portability/broader-context/production-throughput work without blurring support scope.
- **Ubuntu x86 Q8 performance investigation:** default-off experimental acceleration work is exploring the measured Ubuntu x86 Q8 path through packed Q8 runtime storage, selected matrix-level experiments, and AVX2 packed kernels while keeping the safe fallback path intact. These paths remain under validation and are not public support, portability, or default-on acceleration claims.
- **Active next-model bring-up:** Mistral 7B Instruct is the lead exact-row bring-up lane; Qwen 2.5 7B Instruct, Gemma 2 9B Instruct, and Fortytwo Strand Rust Coder 14B remain planned exact-row candidates.

For deeper row-by-row promotion rules and blocker detail, see [`COMPATIBILITY.md`](COMPATIBILITY.md#locked-next-family-readiness-language), [`STATUS.md`](STATUS.md), and [`docs/performance/ubuntu-x86-q8.md`](docs/performance/ubuntu-x86-q8.md).

## Ubuntu x86 Q8 acceleration work

Camelid now includes default-off Ubuntu x86 Q8 acceleration experiments focused on packed Q8 runtime storage, AVX2 packed kernels, and selected matrix-level paths only where evidence identifies the exact default-off flag.

This work is evidence-gated. Optimized paths are not enabled by default yet. Each candidate is validated with parity checks, repeated timing runs, perf counters, and retained/rejected evidence notes.

Current focus areas:

- packed Q8 runtime storage
- explicit `CAMELID_X86_Q8_KERNEL=avx2` packed-kernel path
- evidence-gated matrix-level Q8 GEMM/MUL_MAT experiments
- FFN projection optimization
- attention projection optimization
- warm vs cold inference separation
- reducing wrapper/callback overhead in hot inference

The default/reference path remains available while accelerated paths continue to be validated. The current direction is production-directional runtime work, not a production-ready throughput claim.

| Area | Result | Status |
| --- | --- | --- |
| AVX2 packed-kernel path | Parity and bounded smoke evidence in the measured Ubuntu x86 lane; not a production-throughput claim | retained default-off |
| Packed Q8 runtime storage | Selected dense-tensor runtime-storage coverage in the bounded Ubuntu x86 lane | retained only where evidence-backed |
| Matrix-level Q8 work | Active validation around concrete default-off GEMM/MUL_MAT slices; no broad matrix-owner claim | default-off / experimental |
| Rejected experiments | Failed wall-clock, parity, or clean-host discipline are documented instead of hidden | rejected |

Camelid is also moving toward an appliance-style execution plan where validated runtime paths can be selected automatically while experimental acceleration remains opt-in.

## What makes it different

Camelid gives you:

- a Rust server with OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- GGUF metadata and tensor parsing, tokenizer binding, and typed unsupported-state errors
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI that unlocks chat only when runtime readiness and support-contract readiness agree
- parity and validation harnesses used to compare behavior against llama.cpp before support language moves

**Naming note.** Camelid is the product name. The repository, crate, binary, and diagnostics use `camelid`.

**Reference credit.** Camelid is original Rust code and keeps visible credit for the reference work behind tokenizer checks, compatibility baselines, and parity evidence. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for current acknowledgements and MIT-license notices, including llama.cpp / ggml.

## Support matrix

Camelid’s public support boundary is intentionally narrow and exact-row. Read each row literally; nothing adjacent inherits support.

| Exact lane | Public status | Green evidence today |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | **Verified support** | End-to-end generation, broader five-prompt/50-token parity, template-shape checks, 512-context coverage, and backend RSS/perf sampling. |
| Llama 3.2 1B Instruct Q8_0 | **Verified end-to-end support** | Load, completions, chat completions, WebUI validation, compact/broader parity, exact-row metadata-Jinja row-template checks, template-shape checks, unique-chat perf/RSS sampling, and checked 512/1024/2048/4096/8192-context packs. |
| Llama 3.2 3B Instruct Q8_0 | **Supported exact-row smoke** | Canonical Ubuntu API/WebUI support-gate refresh at source head `e9f926ed1a65` for load, completions, chat completions, frontend smoke, and `supported_exact_row_smoke`; compact/broader 50-token parity, five-prompt API smoke, metadata-Jinja row-template/template-shape evidence, bounded unique-chat perf/RSS, checked 512/1024/2048-context packs, and an opt-in parallel Q8 first-token direction probe. No production-throughput, portability, neighboring-row, larger-context, or broad-family claim. |
| Llama 3 8B Instruct Q8_0 | **Verified support** | Load, completions, chat completions, WebUI validation, compact parity, three-prompt 50-token parity, checked 512/1024/2048-context packs, compact chat-template-shapes pack, memory evidence, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path measurements. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | **In active validation; not supported yet** | Source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, checked 512/1024/2048 bring-up, checked 4096/8192 context validation, and fail-closed API/WebUI/RSS evidence are green; latest context bundle: `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | **Active validation; partial backend runtime only** | One-token backend MoE runtime evidence exists, with tokenizer/template and sparse MoE metadata proof. Blocker anchors: `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/`. |
| Qwen2.5-7B-Instruct-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. |
| gemma-2-9b-it-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. |
| Fortytwo-Network/Strand-Rust-Coder-14B-v1 | **Planned Rust-coder validation target** | Rust-specialized Qwen2.5-Coder-14B-derived model selected for future Camelid testing. Hugging Face lists Safetensors plus separate GGUF quantizations, including Q8_0; no Camelid load, tokenizer, GGUF, parity, API, WebUI, RSS, or throughput evidence is claimed yet. |

### Latest model checks

The maintainer matrix now includes four exact Q8_0 supported rows with checked row-specific evidence. Mistral remains an active validation row, and Mixtral remains partial backend runtime evidence only, not a supported row. These are exact-row checks, not universal model claims.

| Exact row | Latest checked bucket | Result | Output checked |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | direct chat validation | PASS | `Certainly! Here` |
| Llama 3.2 1B Instruct Q8_0 | 8192-context recall pack | PASS on cited source/runtime head `aaf9207d1669` | `CMLD-819` |
| Llama 3.2 3B Instruct Q8_0 | canonical Ubuntu API/WebUI gate plus 2048-context recall pack | PASS at source head `e9f926ed1a65` for API/WebUI; PASS for checked 2048 pack | `supported_exact_row_smoke`; `CMLD-204` |
| Llama 3 8B Instruct Q8_0 | 2048-context recall pack | PASS on cited source/runtime head `8e26be0a73c0` | `CMLD-204` |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active validation: Ubuntu bring-up plus exact tokenizer/template, 1-token, broader five-prompt/50-token parity, checked 4096/8192 context evidence, and fail-closed API/WebUI/RSS evidence | PASS for validation lane; still unsupported | Explicit contract promotion and synchronized support surfaces still required before any support claim |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | Active validation: one-token backend MoE runtime evidence plus Gate 9A/continuation blocker bundles | BLOCKED by later-generation divergence and backend HTTP hang | No Mixtral API/WebUI/frontend readiness, neighboring-row, long-context, arbitrary-template, production-throughput, portability, or broader/full claim |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- Support language currently means only the exact supported rows above; Mistral has no support claim yet and may only be discussed as the exact active validation row above. The Mixtral row may only be described as one-token backend MoE runtime evidence with later-generation/API/WebUI/frontend readiness not yet established.
- Checked context packs do **not** imply model-native or broader context support.
- Template and bounded perf/RSS evidence for current supported rows is exact-row scoped; it does **not** imply broad arbitrary/Jinja-template behavior, production throughput, neighboring GGUFs, portability, or broader context support.
- The next exact-row 8B, `Mistral-7B-Instruct-v0.3.Q8_0.gguf`, `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf`, and `Fortytwo-Network/Strand-Rust-Coder-14B-v1` gaps are broader context, portability, repeated durability evidence, and normalized support bundles; Mistral also still needs explicit contract promotion after fail-closed API/WebUI/RSS readiness, Mixtral must first fix later-generation divergence and the continuation hang before any API/WebUI/frontend support claim, and Strand first needs acquisition, GGUF/tokenizer/runtime mapping, and parity validation before any support claim.

Authoritative details live in [`COMPATIBILITY.md`](COMPATIBILITY.md). The current evidence snapshot lives in [`STATUS.md`](STATUS.md).

## Start here

- [`COMPATIBILITY.md`](COMPATIBILITY.md) — authoritative support matrix and promotion rules
- [`STATUS.md`](STATUS.md) — current evidence boundary, recent moves, and blockers
- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) — shortest safe local path
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — expected checks by change type
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution expectations and PR guidance
- [`DOCS.md`](DOCS.md) — full documentation map

## Quickstart

This quickstart verifies that Camelid builds cleanly, starts the backend, and returns a live API response. It is intentionally simple and honest: the repository does not bundle supported GGUF model files, so full local chat requires a supported GGUF already present on your machine.

### 1) Build and run the server

```bash
git checkout main
git pull --ff-only
cargo build --release --bin camelid
target/release/camelid serve --model /path/to/model.gguf
```

`serve --model` loads the model at startup and lets Camelid choose the safest validated execution plan for the current host. The default profile is `auto`; low-level environment variables remain developer overrides, not normal setup.

If you only want to bring up the API without a model:

```bash
target/release/camelid serve --addr 127.0.0.1:8181
```

Toolchain note: Camelid currently requires Rust/Cargo 1.87+. If your host exposes an older system `cargo`, use the rustup-managed toolchain described in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### 2) Verify the server is responding

From another shell:

```bash
curl -s http://127.0.0.1:8181/api/capabilities
```

Success looks like a live JSON capability response from Camelid. That confirms the backend is up and the product surface is reachable.

### 3) Before you expect local chat to work

You will need:

- a supported GGUF model file already present on your machine
- the model path passed with `--model`, or loaded later through the model API/UI
- any extra contributor setup described in [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)

For a first supported local run, TinyLlama is the clearest path. It is **not bundled** in this repository, but once you have a supported GGUF locally, Camelid is designed to make the readiness boundary explicit instead of guessing.

## Frontend

Camelid includes a built-in React/Vite frontend in [`frontend/`](frontend/) so the local runtime ships with a real product surface, not just a backend API.


The UI is designed to feel straightforward and approachable while still staying honest about model readiness. It keeps the main path simple — pick a local model, see whether Camelid reports it as ready, and start chatting when the runtime and support contract agree.

A few principles define the WebUI:

- **Chat-first and intuitive:** the interface emphasizes the primary action instead of burying it in operator-only controls.
- **Honest readiness signals:** chat only unlocks when the loaded model is runtime-ready and matched to an exact supported support-contract row.
- **One product story across surfaces:** the backend, API, UI, and docs are intended to agree instead of sending mixed signals about what is actually supported.

```bash
cd frontend
npm ci
npm run dev
```

By default, the UI talks to `http://127.0.0.1:8181` and only unlocks local chat when the loaded model is both runtime-ready and covered by the current support contract.

See [`frontend/README.md`](frontend/README.md) for frontend-specific details.

## How support moves

A row is promoted only when all of these agree for the exact lane being claimed:

1. runtime behavior
2. artifact-backed validation
3. documentation
4. API capability reporting
5. frontend readiness behavior

That is Camelid’s core discipline: **evidence first, broader claims later**.

## Contributing

If you want to contribute, start with the docs written for safe local iteration and contributor onboarding:

- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md)
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

Camelid intentionally separates the public support contract, the local contributor path, and maintainer-only evidence workflows. That keeps the project welcoming without leaking operator-only details or pretending every internal validation lane is part of normal onboarding.

## Validation

Use [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) to pick the smallest correct validation lane for your change.

Common baseline checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
bash scripts/check-public-scrub.sh
```

For docs-only changes, the minimum expected checks are:

```bash
git diff --check
bash scripts/check-public-scrub.sh
```

If your change affects the frontend, also run:

```bash
cd frontend
npm ci
npm run build
```

## Documentation map

- [`DOCS.md`](DOCS.md) — full documentation index
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — support ledger
- [`STATUS.md`](STATUS.md) — current evidence snapshot
- [`ROADMAP.md`](ROADMAP.md) — delivery plan of record
- [`FULL_SUPPORT_BLOCKER_MATRIX.md`](FULL_SUPPORT_BLOCKER_MATRIX.md) — row-by-row missing evidence for broader promotion
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architecture and module planning
- [`DECISIONS.md`](DECISIONS.md) — design decision log

## License and acknowledgements

Camelid is licensed under the [MIT License](LICENSE).

Camelid is inspired by and validated against [`llama.cpp`](https://github.com/ggml-org/llama.cpp), which is licensed under the MIT License:

> Copyright (c) 2023-2026 The ggml authors

The llama.cpp project and the broader GGUF ecosystem made the modern local-model path practical. Camelid keeps its runtime implementation Rust-native while intentionally crediting llama.cpp wherever reference comparisons, tokenizer fixtures, and parity gates rely on it.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
